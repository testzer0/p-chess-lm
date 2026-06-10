from typing import Iterator, Protocol

import torch
import torch.nn as nn


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
