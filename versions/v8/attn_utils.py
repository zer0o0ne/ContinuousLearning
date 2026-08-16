"""Shared attention-mask helper.

Inherited from `versions/v7/agent/attn_utils.py` verbatim — it is
architecture-independent (a pure function over a padding mask), so it comes
across the same way `env/` and `utils.py` did.

`vendor/v7/attn_utils.py` holds the *frozen* copy the vendored v7 checkpoint
code imports. This one is v8's, and is the one v8's own modules use; the two are
identical today and are allowed to diverge, because the vendored snapshot must
keep behaving exactly as v7 did (CONCEPT.md §4.3).

Original note (v7, fix A.1): passing ANY explicit `attention_mask` to a
transformers Qwen3 decoder layer disables its built-in `is_causal` shortcut in
SDPA, so attention becomes fully bidirectional. Causality must therefore be
encoded explicitly in the mask whenever a mask is supplied. In v8 this is
load-bearing twice over: causal-within-hand is what stops a token from reading
the action it is being asked to predict (CONCEPT.md §5.2).
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
