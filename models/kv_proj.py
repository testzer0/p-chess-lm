import torch
import torch.nn as nn
from transformers import DynamicCache

from chesslm.models.base import (
    apply_lora,
    decoder_trainable_params,
    init_new_token_embeddings,
    load_decoder_state,
    save_decoder_state,
    unwrap_decoder,
)


def _build_mapping_dict(mode: str) -> dict[int, list[int]]:
    """Map decoder layer index → list of encoder layer indices to project from.

    channel_concat: every decoder layer receives all 16 encoder layers concatenated.
    interleaved:    encoder layer e is assigned to decoder layers in [e*36//16, (e+1)*36//16).
                    Yields a surjective partition — 12 encoder layers cover 2 decoder layers
                    each, 4 encoder layers (3, 7, 11, 15) cover 3.
    """
    if mode == "channel_concat":
        return {d: list(range(16)) for d in range(36)}
    elif mode == "interleaved":
        mapping = {}
        for e in range(16):
            for d in range(e * 36 // 16, (e + 1) * 36 // 16):
                mapping[d] = [e]
        return mapping
    else:
        raise ValueError(f"Unknown proj_mode: {mode!r}")


class KVProjChessLM(nn.Module):
    """
    Frozen LC0 encoder + trainable SmolLM3 3B decoder bridged via direct KV projection.

    Encoder hidden states are concatenated per square (per mapping_dict), projected to
    per-decoder-layer K and V tensors, and injected into the decoder via past_key_values.
    No new attention sublayers, no prefix tokens in the input sequence — the decoder's
    existing self-attention attends over 64 injected KV positions at every layer.

    Projection modes (--proj-mode):
      channel_concat (default): all 16 encoder layers concatenated → projected
        independently to each decoder layer's K/V. Input dim per layer: 16 * 1024 = 16384.
      interleaved: encoder layer i projected to a contiguous span of decoder layers
        proportional to i * 36/16. Input dim per decoder layer: 1024.

    Inputs:
      input_ids              : (B, S)             — tokenized decoder input
      encoder_hidden_states  : (B, 16, 64, 1024)  — pre-computed, canonicalized
      attention_mask         : (B, S)             — padding mask (optional)
      position_ids           : (B, S)             — 0-based TEXT positions
                                                    (model offsets by N_ENC_SQUARES
                                                    internally so text RoPE lives
                                                    at 64..63+S, treating the
                                                    injected KV cache as 64
                                                    prior context positions).
                                                    Default: arange(0, S).
    """

    N_ENC_LAYERS  = 16
    N_DEC_LAYERS  = 36
    ENCODER_DIM   = 1024
    DECODER_DIM   = 2048
    N_KV_HEADS    = 4
    HEAD_DIM      = 128
    KV_DIM        = N_KV_HEADS * HEAD_DIM   # 512
    N_ENC_SQUARES = 64

    def __init__(self, decoder: nn.Module, n_new_tokens: int = 0, proj_mode: str = "channel_concat", lora_rank: int = 0):
        super().__init__()

        self.lora_rank = lora_rank
        self.decoder   = apply_lora(decoder, lora_rank)
        self.proj_mode = proj_mode
        self.mapping_dict = _build_mapping_dict(proj_mode)

        # Verify all input dims are uniform within this mode so we can share one LayerNorm.
        in_dims = {len(self.mapping_dict[d]) * self.ENCODER_DIM for d in range(self.N_DEC_LAYERS)}
        assert len(in_dims) == 1, f"Non-uniform input dims across decoder layers: {in_dims}"
        in_dim = in_dims.pop()

        self.layer_norm = nn.LayerNorm(in_dim)
        self.W_K = nn.ModuleList([
            nn.Linear(len(self.mapping_dict[d]) * self.ENCODER_DIM, self.KV_DIM, bias=False)
            for d in range(self.N_DEC_LAYERS)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(len(self.mapping_dict[d]) * self.ENCODER_DIM, self.KV_DIM, bias=False)
            for d in range(self.N_DEC_LAYERS)
        ])
        for linear in self.W_V:
            nn.init.zeros_(linear.weight)

        self.n_new_tokens = n_new_tokens
        dec_param = next(self.decoder.parameters())
        self.new_embed, self.new_lm_head = init_new_token_embeddings(
            n_new_tokens, self.DECODER_DIM,
            device=dec_param.device, dtype=dec_param.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        # 1. Project encoder hidden states to per-decoder-layer K/V tensors.
        kv_list = []
        for d in range(self.N_DEC_LAYERS):
            enc = torch.cat(
                [encoder_hidden_states[:, e] for e in self.mapping_dict[d]], dim=-1
            ).to(self.layer_norm.weight.dtype)                       # (B, 64, in_dim)
            enc = self.layer_norm(enc)
            K = self.W_K[d](enc)                                     # (B, 64, KV_DIM)
            V = self.W_V[d](enc)
            K = K.view(B, self.N_ENC_SQUARES, self.N_KV_HEADS, self.HEAD_DIM).transpose(1, 2)
            V = V.view(B, self.N_ENC_SQUARES, self.N_KV_HEADS, self.HEAD_DIM).transpose(1, 2)
            kv_list.append((K, V))

        past_key_values = DynamicCache.from_legacy_cache(kv_list)

        # 2. Extend attention mask to cover the 64 injected KV positions.
        if attention_mask is not None:
            prefix_mask = torch.ones(B, self.N_ENC_SQUARES, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 3. Split embedding: route new tokens to trainable new_embed.
        base = self._base_decoder
        frozen_vocab = base.config.vocab_size
        if self.n_new_tokens > 0:
            clipped = input_ids.clamp(max=frozen_vocab - 1)
            h = base.model.embed_tokens(clipped)
            new_mask = input_ids >= frozen_vocab
            if new_mask.any():
                new_ids = (input_ids[new_mask] - frozen_vocab).clamp(min=0)
                h[new_mask] = self.new_embed(new_ids).to(h.dtype)
        else:
            h = base.model.embed_tokens(input_ids)

        # 4. Run decoder. Text token position_ids start at N_ENC_SQUARES so RoPE
        #    treats the injected KV positions as prior context.
        # If position_ids is supplied by the caller (0-based text positions), offset by N_ENC_SQUARES.
        if position_ids is None:
            position_ids = torch.arange(
                self.N_ENC_SQUARES, self.N_ENC_SQUARES + S, device=device,
            ).unsqueeze(0).expand(B, -1)
        else:
            position_ids = position_ids + self.N_ENC_SQUARES

        model_out = base.model(
            inputs_embeds=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=False,
        )

        h_out = model_out.last_hidden_state                          # (B, S, 2048)
        logits = base.lm_head(h_out)
        if self.n_new_tokens > 0:
            logits = torch.cat([logits, self.new_lm_head(h_out)], dim=-1)
        return logits

    @property
    def _base_decoder(self) -> nn.Module:
        return unwrap_decoder(self.decoder)

    @property
    def device(self) -> torch.device:
        return next(self.W_K.parameters()).device

    def trainable_parameters(self):
        return iter(
            list(self.layer_norm.parameters()) +
            list(self.W_K.parameters()) +
            list(self.W_V.parameters()) +
            decoder_trainable_params(self.decoder, self.lora_rank) +
            (list(self.new_embed.parameters()) if self.n_new_tokens > 0 else [])
        )

    def trainable_state_dict(self) -> dict:
        d = {
            "layer_norm": self.layer_norm.state_dict(),
            "W_K":        self.W_K.state_dict(),
            "W_V":        self.W_V.state_dict(),
        }
        if self.lora_rank >= 0:
            d["decoder"] = save_decoder_state(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            d["new_embed"] = self.new_embed.state_dict()
        return d

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        self.layer_norm.load_state_dict(state_dict["layer_norm"])
        self.W_K.load_state_dict(state_dict["W_K"])
        self.W_V.load_state_dict(state_dict["W_V"])
        if self.lora_rank >= 0 and "decoder" in state_dict:
            load_decoder_state(self.decoder, self.lora_rank, state_dict["decoder"])
        if self.n_new_tokens > 0 and "new_embed" in state_dict:
            self.new_embed.load_state_dict(state_dict["new_embed"])

    def param_groups(self, lr: float, decoder_lr: float | None = None) -> list[dict]:
        bridge = (
            list(self.layer_norm.parameters()) +
            list(self.W_K.parameters()) +
            list(self.W_V.parameters())
        )
        groups = [{"params": bridge, "lr": lr}]
        dec_params = decoder_trainable_params(self.decoder, self.lora_rank)
        if dec_params:
            # lora_rank=0 unfreezes the pretrained backbone — use lr*0.1 to avoid
            # destroying it. lora_rank>0 trains fresh adapters from scratch
            # (B=0 init) — full lr is appropriate.
            # decoder_lr overrides this when set explicitly (e.g. to decouple bridge/decoder LRs).
            dec_lr = decoder_lr if decoder_lr is not None else (lr if self.lora_rank > 0 else lr * 0.1)
            groups.append({"params": dec_params, "lr": dec_lr})
        if self.n_new_tokens > 0:
            groups.append({"params": list(self.new_embed.parameters()), "lr": lr * 0.1})
        return groups

    def get_diagnostics(self) -> dict[str, float]:
        return {}

    @classmethod
    def from_pretrained(
        cls,
        decoder_path: str,
        n_new_tokens: int = 0,
        proj_mode: str = "channel_concat",
        lora_rank: int = 0,
        device: torch.device | str = None,
        **hf_kwargs,
    ):
        from transformers import AutoModelForCausalLM
        if device is not None:
            hf_kwargs.setdefault("device_map", device)
        decoder = AutoModelForCausalLM.from_pretrained(decoder_path, **hf_kwargs)
        model = cls(decoder, n_new_tokens=n_new_tokens, proj_mode=proj_mode, lora_rank=lora_rank)
        decoder_dtype  = next(decoder.parameters()).dtype
        decoder_device = next(decoder.parameters()).device
        model.layer_norm.to(device=decoder_device, dtype=decoder_dtype)
        model.W_K.to(device=decoder_device, dtype=decoder_dtype)
        model.W_V.to(device=decoder_device, dtype=decoder_dtype)
        # new_embed / new_lm_head are already on decoder_device/dtype (allocated
        # there in __init__ via init_new_token_embeddings) — casting them here
        # would re-create the Parameter on each module and sever weight tying.
        return model
