"""Shared attention-mask helpers for the Qwen3-based modules.

Этап-A fix A.1: passing ANY explicit `attention_mask` to a transformers Qwen3
decoder layer disables its built-in `is_causal` shortcut in SDPA, so attention
becomes fully bidirectional. The encoder / decoder / value / action /
opponent_action modules all supply a (padding-only) mask, which silently made
them non-causal. Causality is a load-bearing assumption (the modelling-phase
reconstruction target, and the latents MCTS extends context with), so it must
be encoded explicitly in the mask whenever a mask is supplied.
"""

import torch


def build_causal_padding_mask(padding_mask, seq_len, dtype, device):
    """Additive (B, 1, S, S) attention mask combining causality + key padding.

    A query at position i may attend to key j iff ``j <= i`` (causal) AND key j
    is a real (non-padded) token. Disallowed (i, j) get ``finfo(dtype).min``,
    allowed get 0.0.

    Args:
        padding_mask: (B, S) float tensor, 1.0 = real, 0.0 = padding. May be
            None, in which case a pure causal mask (no padding component) is
            returned with batch dim 1.
        seq_len: S — sequence length the causal triangle is built for.
        dtype: compute dtype (track AMP — pass the hidden-state dtype).
        device: target device.

    Returns:
        (1, 1, S, S) if padding_mask is None, else (B, 1, S, S).
    """
    min_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device),
        diagonal=1,
    )[None, None]  # (1, 1, S, S)
    if padding_mask is None:
        return causal
    pad = (1.0 - padding_mask[:, None, None, :].to(dtype)) * min_val  # (B, 1, 1, S)
    return causal + pad
