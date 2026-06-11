from typing import Iterator, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class ChessLM(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def trainable_parameters(self) -> Iterator[nn.Parameter]: ...

    def trainable_state_dict(self) -> dict: ...

    def load_trainable_state_dict(self, state_dict: dict) -> None: ...

    def get_diagnostics(self) -> dict[str, float]: ...

    def param_groups(self, lr: float, decoder_lr: float | None = None) -> list[dict]: ...


# ---------------------------------------------------------------------------
# FSDP-ready base class + in-class chunked loss
#
# Shared base for the flamingo / llava archs. Subclassing
# PreTrainedModel is the minimal FSDP hook: exposes `_no_split_modules` for the
# auto-wrap policy and gives the model a `config`. HF weight-init is unused (the
# decoder is pretrained, bridges self-init), so `_init_weights` is a no-op.
# `_loss_from_hidden` streams the LM head in chunks so the full (B, S, V) logits
# are never materialized; `forward` calls it when `labels` is given.
# ---------------------------------------------------------------------------

class ChessLMConfig(PretrainedConfig):
    model_type = "chess_lm"


class ChessLMPreTrainedModel(PreTrainedModel):
    config_class = ChessLMConfig
    base_model_prefix = "chess_lm"
    supports_gradient_checkpointing = True
    # DenseXAttn exists only in the flamingo arch; harmless for the others.
    _no_split_modules = ["SmolLM3DecoderLayer", "DenseXAttn"]
    logit_chunk_size = 1024  # supervised tokens per LM-head chunk

    def _init_weights(self, module):
        return  # decoder is pretrained; bridges init in __init__

    def _loss_from_hidden(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Chunked next-token cross-entropy from final hidden states (B, S, D).

        Mean over supervised positions (labels != -100), matching a shifted
        cross_entropy(ignore_index=-100), without building the full logits.
        """
        chunk = chunk_size or self.logit_chunk_size
        base = self._base_decoder

        # Shift, then keep only supervised positions: run the LM head on exactly
        # the tokens that contribute to the loss.
        hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        labels = labels[:, 1:].reshape(-1)
        keep = labels != -100
        hidden = hidden[keep]
        labels = labels[keep]
        n = labels.numel()
        if n == 0:
            return hidden.sum() * 0.0  # degenerate batch: keep a grad-connected 0

        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, n, chunk):
            h_c = hidden[i:i + chunk]
            logits_c = base.lm_head(h_c)
            if self.n_new_tokens > 0:
                logits_c = torch.cat([logits_c, self.new_lm_head(h_c)], dim=-1)
            total = total + F.cross_entropy(logits_c.float(), labels[i:i + chunk], reduction="sum")
        return total / n


def init_new_token_embeddings(
    n_new_tokens: int,
    decoder_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Embedding | None, nn.Linear | None]:
    """
    Returns a (new_embed, new_lm_head) pair with tied weights, or (None, None).
    new_lm_head.weight is tied to new_embed.weight (matches SmolLM3's tie_word_embeddings=True).

    device/dtype: if provided, both modules are allocated directly on that
    device/dtype. Tying is established AFTER allocation so a subsequent .to()
    cannot sever it. Callers should avoid calling .to() on these modules after
    construction — that would re-create the Parameter on one module and break
    the tie.
    """
    if n_new_tokens == 0:
        return None, None
    factory = {"device": device, "dtype": dtype}
    new_embed = nn.Embedding(n_new_tokens, decoder_dim, **factory)
    new_lm_head = nn.Linear(decoder_dim, n_new_tokens, bias=False, **factory)
    new_lm_head.weight = new_embed.weight
    return new_embed, new_lm_head


# ---------------------------------------------------------------------------
# Shared LoRA / decoder helpers
#
# lora_rank semantics used across all three architectures:
#   < 0  →  decoder fully frozen (no grad, not saved in checkpoint)
#   = 0  →  decoder fully trainable
#   > 0  →  decoder backbone frozen + LoRA adapters on Q/K/V/O
# ---------------------------------------------------------------------------

def apply_lora(
    decoder: nn.Module,
    lora_rank: int,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA / freezing to decoder according to lora_rank semantics.

    lora_rank < 0: freeze all decoder parameters; return decoder unchanged.
    lora_rank = 0: leave decoder fully trainable; return unchanged.
    lora_rank > 0: wrap with PEFT LoRA (backbone frozen, adapters trainable).
    """
    if lora_rank < 0:
        for p in decoder.parameters():
            p.requires_grad_(False)
        return decoder
    if lora_rank == 0:
        return decoder
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=lora_rank,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(decoder, cfg)


def unwrap_decoder(decoder: nn.Module) -> nn.Module:
    """Return the underlying HF CausalLM, unwrapping PEFT if present."""
    if hasattr(decoder, "get_base_model"):
        return decoder.get_base_model()
    return decoder


def decoder_trainable_params(decoder: nn.Module, lora_rank: int) -> list[nn.Parameter]:
    """Return the decoder parameters that should be optimized.

    lora_rank < 0: [] (decoder is frozen)
    lora_rank = 0: all decoder parameters
    lora_rank > 0: only requires_grad=True params (the LoRA adapters)
    """
    if lora_rank < 0:
        return []
    if lora_rank > 0:
        return [p for p in decoder.parameters() if p.requires_grad]
    return list(decoder.parameters())


def save_decoder_state(decoder: nn.Module, lora_rank: int) -> dict:
    """Serialize trainable decoder state. Only call when lora_rank >= 0."""
    if lora_rank > 0:
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(decoder)
    return decoder.state_dict()


def load_decoder_state(decoder: nn.Module, lora_rank: int, state: dict) -> None:
    """Load decoder state produced by save_decoder_state."""
    if lora_rank > 0:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(decoder, state)
    else:
        decoder.load_state_dict(state)
