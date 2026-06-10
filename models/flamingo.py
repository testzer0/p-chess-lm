import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

from chesslm.models.base import (
    ChessLMConfig,
    ChessLMPreTrainedModel,
    apply_lora,
    decoder_trainable_params,
    init_new_token_embeddings,
    load_decoder_state,
    save_decoder_state,
    unwrap_decoder,
)


# --- FFN helpers ---

class _SimpleFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, activation: str, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "relu2":
            x = F.relu(x).square()
        elif self.activation == "gelu":
            x = F.gelu(x)
        return self.drop(self.fc2(x))


class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up   = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def _make_ffn(d_model: int, activation: str, dropout: float) -> nn.Module:
    d_hidden = 2 * d_model
    if activation == "swiglu":
        return _SwiGLUFFN(d_model, d_hidden, dropout)
    assert activation in ("relu2", "gelu"), f"Unknown activation: {activation!r}"
    return _SimpleFFN(d_model, d_hidden, activation, dropout)


# --- Flamingo-style gated cross-attention sublayer ---

class DenseXAttn(nn.Module):
    """
    Flamingo-style gated cross-attention sublayer.
    Inserts between frozen decoder layers; only this module's parameters are trained.

    Args:
        encoder_dim:  hidden dim of the chess encoder (LC0 BT5: 1024)
        decoder_dim:  hidden dim of the LLM decoder   (SmolLM3 3B: 2048)
        n_heads:      number of Q heads               (default: 16, full MHA)
        n_kv_heads:   number of KV heads              (default: 16, equal to n_heads → full MHA,
                      no GQA constraint; set < n_heads to enable GQA)
        activation:   FFN activation — "relu2" (Flamingo default), "gelu", "swiglu"
        dropout:      applied inside attention and FFN
        alpha_init:   initial value of alpha (pre-tanh). 0.0 = original Flamingo (gate fully
                      closed at init, W_O must be non-zero to get alpha gradients).
                      atanh(0.5)≈0.549 = gate starts at 0.5 (avoids cold-start trap with frozen
                      decoder, but requires wo_zero_init=True to prevent residual corruption).
        wo_zero_init: if True, zero-initialize W_O so residual contribution is 0 at step 0
                      regardless of alpha. If False, use default init (Flamingo original).
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 16,
        activation: str = "relu2",
        dropout: float = 0.0,
        alpha_init: float = 0.0,
        wo_zero_init: bool = False,
    ):
        super().__init__()
        assert decoder_dim % n_heads == 0, "decoder_dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = decoder_dim // n_heads
        self.n_rep      = n_heads // n_kv_heads
        self.dropout    = dropout

        self.W_Q = nn.Linear(decoder_dim,                  n_heads    * self.head_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim,                  n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim,                  n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, decoder_dim,                     bias=False)
        if wo_zero_init:
            nn.init.zeros_(self.W_O.weight)

        self.alpha_attn = nn.Parameter(torch.tensor([alpha_init]))
        self.alpha_ffn  = nn.Parameter(torch.tensor([alpha_init]))

        self.norm_y   = nn.LayerNorm(decoder_dim)
        self.norm_x   = nn.LayerNorm(encoder_dim)
        self.norm_ffn = nn.LayerNorm(decoder_dim)

        self.ffn = _make_ffn(decoder_dim, activation, dropout)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        y : (B, S_dec, decoder_dim)  — decoder residual stream
        x : (B, S_enc, encoder_dim)  — encoder hidden states for this layer (S_enc = 64)
        returns: y updated with gated cross-attention and gated FFN, same shape as input y
        """
        B, S_dec, _ = y.shape
        S_enc = x.shape[1]

        q = self.W_Q(self.norm_y(y))
        x_n = self.norm_x(x)
        k = self.W_K(x_n)
        v = self.W_V(x_n)

        q = q.view(B, S_dec, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S_enc, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_enc, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S_dec, self.n_heads * self.head_dim)
        attn_out = self.W_O(attn_out)

        y = torch.tanh(self.alpha_attn).to(y.dtype) * attn_out + y
        y = torch.tanh(self.alpha_ffn).to(y.dtype) * self.ffn(self.norm_ffn(y)) + y

        return y


# --- FlamingoChessLM ---

class FlamingoChessLM(ChessLMPreTrainedModel):
    """
    Frozen SmolLM3 3B decoder bridged to a frozen LC0 chess encoder via 16
    trainable DenseXAttn sublayers (Flamingo-style gated cross-attention).

    Expects pre-computed, canonicalized encoder hidden states so that encoder
    inference can be batched and cached independently of the decoder.

    Architecture:
      - x-attn sublayer i is injected before decoder layer X_ATTN_POSITIONS[i]
      - x-attn sublayer i attends to encoder layer i (1-to-1 pairing, layers 0–15)
      - Trainable: DenseXAttn layers + new token embeddings always; plus LoRA
        adapters (lora_rank>0) or full decoder (lora_rank=0); decoder frozen (lora_rank<0)

    Inputs:
      input_ids              : (B, S)             — tokenized decoder input
      encoder_hidden_states  : (B, 16, 64, 1024)  — pre-computed, canonicalized
      attention_mask         : (B, S)             — padding mask (optional)
    """

    X_ATTN_POSITIONS = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]
    N_XATTN     = len(X_ATTN_POSITIONS)
    ENCODER_DIM = 1024
    DECODER_DIM = 2048

    def __init__(
        self,
        decoder: nn.Module,
        n_new_tokens: int = 0,
        lora_rank: int = -1,
        x_attn_kwargs: dict = None,
    ):
        super().__init__(ChessLMConfig())

        self.lora_rank = lora_rank
        self.decoder   = apply_lora(decoder, lora_rank)

        x_attn_kwargs = x_attn_kwargs or {}
        self.x_attn_layers = nn.ModuleList([
            DenseXAttn(
                encoder_dim=self.ENCODER_DIM,
                decoder_dim=self.DECODER_DIM,
                **x_attn_kwargs,
            )
            for _ in range(self.N_XATTN)
        ])

        self._xattn_at = {pos: i for i, pos in enumerate(self.X_ATTN_POSITIONS)}

        self.n_new_tokens = n_new_tokens
        dec_param = next(self.decoder.parameters())
        self.new_embed, self.new_lm_head = init_new_token_embeddings(
            n_new_tokens, self.DECODER_DIM,
            device=dec_param.device, dtype=dec_param.dtype,
        )

    @property
    def _base_decoder(self) -> nn.Module:
        return unwrap_decoder(self.decoder)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
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

        cache_position = torch.arange(S, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(B, -1)
        position_embeddings = base.model.rotary_emb(h, position_ids)

        causal_mask = create_causal_mask(
            config=base.config,
            input_embeds=h,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )

        for layer_idx, decoder_layer in enumerate(base.model.layers):
            if layer_idx in self._xattn_at:
                xattn_idx = self._xattn_at[layer_idx]
                enc = encoder_hidden_states[:, xattn_idx].to(h.dtype)
                h = self.x_attn_layers[xattn_idx](h, enc)

            h = decoder_layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                use_cache=False,
            )

        h = base.model.norm(h)
        if labels is not None:
            return self._loss_from_hidden(h, labels)
        logits = base.lm_head(h)
        if self.n_new_tokens > 0:
            logits = torch.cat([logits, self.new_lm_head(h)], dim=-1)
        return logits

    @property
    def device(self) -> torch.device:
        return next(self.x_attn_layers.parameters()).device

    def trainable_parameters(self):
        params = list(self.x_attn_layers.parameters())
        params += decoder_trainable_params(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            params += list(self.new_embed.parameters())
        return iter(params)

    def trainable_state_dict(self) -> dict:
        d = {"x_attn_layers": self.x_attn_layers.state_dict()}
        if self.lora_rank >= 0:
            d["decoder"] = save_decoder_state(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            d["new_embed"] = self.new_embed.state_dict()
        return d

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        self.x_attn_layers.load_state_dict(state_dict["x_attn_layers"])
        if self.lora_rank >= 0 and "decoder" in state_dict:
            load_decoder_state(self.decoder, self.lora_rank, state_dict["decoder"])
        if self.n_new_tokens > 0 and "new_embed" in state_dict:
            self.new_embed.load_state_dict(state_dict["new_embed"])

    def param_groups(self, lr: float, decoder_lr: float | None = None) -> list[dict]:
        groups = [{"params": list(self.x_attn_layers.parameters()), "lr": lr}]
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
        return {
            f"alpha_attn/layer_{i:02d}": torch.tanh(layer.alpha_attn.float()).item()
            for i, layer in enumerate(self.x_attn_layers)
        } | {
            f"alpha_ffn/layer_{i:02d}": torch.tanh(layer.alpha_ffn.float()).item()
            for i, layer in enumerate(self.x_attn_layers)
        }

    @classmethod
    def from_pretrained(
        cls,
        decoder_path: str,
        n_new_tokens: int = 0,
        lora_rank: int = -1,
        device: torch.device | str = None,
        x_attn_kwargs: dict = None,
        **hf_kwargs,
    ):
        from transformers import AutoModelForCausalLM
        if device is not None:
            hf_kwargs.setdefault("device_map", device)
        decoder = AutoModelForCausalLM.from_pretrained(decoder_path, **hf_kwargs)
        model = cls(decoder, n_new_tokens=n_new_tokens, lora_rank=lora_rank, x_attn_kwargs=x_attn_kwargs)
        decoder_dtype  = next(decoder.parameters()).dtype
        decoder_device = next(decoder.parameters()).device
        model.x_attn_layers.to(device=decoder_device, dtype=decoder_dtype)
        # new_embed / new_lm_head are already on decoder_device/dtype (allocated
        # there in __init__ via init_new_token_embeddings) — casting them here
        # would re-create the Parameter on each module and sever weight tying.
        return model
