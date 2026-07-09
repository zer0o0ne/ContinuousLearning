"""Stage A audit tests: architecture fixes.

A.1: Causal attention (all 5 modules use build_causal_padding_mask).
A.2: ModellingHead is autoregressive (PLAN_MODELLING_HEAD_REDESIGN.md §2):
     no cross-attention modules; the context self-attn stack is CAUSAL;
     action conditioning is a per-action-independent MLP.
A.3: head_dim = d_model // n_heads in all configs.
A.5.2: Sequence capping guard on max_seq_len.
A.5.3: Card clamp replaced by assert.
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.attn_utils import build_causal_padding_mask
from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.value.value import ValueHead
from agent.action.action import ActionHead
from agent.modelling.modelling import ModellingHead


# ── Shared helpers ───────────────────────────────────────────────────────────

D_MODEL = 64
N_HEADS = 4
N_KV_HEADS = 2
N_LAYERS = 1
D_FF = 128
MAX_SEQ = 256
N_ACTIONS = 10


def _build_encoder():
    return Encoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ)


def _build_decoder():
    return Decoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ)


def _build_value():
    return ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ)


def _build_action():
    return ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ)


def _build_modelling():
    return ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ)


# ── A.1: Causality ──────────────────────────────────────────────────────────

def test_build_causal_padding_mask_shape():
    """build_causal_padding_mask returns (B,1,S,S) with correct blocking."""
    B, S = 2, 5
    pad = torch.ones(B, S)
    pad[0, 3:] = 0.0
    mask = build_causal_padding_mask(pad, S, torch.float32, torch.device("cpu"))
    assert mask.shape == (B, 1, S, S)
    assert mask[0, 0, 0, 1].item() < -1e30, "Future token should be blocked"
    assert mask[0, 0, 1, 0].item() == 0.0, "Past token should be visible"
    assert mask[0, 0, 2, 4].item() < -1e30, "Padded future should be blocked"


def test_build_causal_padding_mask_no_padding():
    """With padding_mask=None, returns (1,1,S,S) pure causal."""
    mask = build_causal_padding_mask(None, 4, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() < -1e30


def _test_module_causality(module_fn, needs_mask=True):
    """Generic causality test: perturbing token t must NOT change output < t."""
    torch.manual_seed(42)
    mod = module_fn()
    mod.eval()
    B, S = 1, 6
    x = torch.randn(B, S, D_MODEL)
    mask = torch.ones(B, S) if needs_mask else None

    with torch.no_grad():
        out1 = mod(x, mask=mask)

    x2 = x.clone()
    x2[0, 4] += 10.0
    with torch.no_grad():
        out2 = mod(x2, mask=mask)

    diff_before = (out1[0, :4] - out2[0, :4]).abs().max().item()
    assert diff_before < 1e-5, (
        f"Perturbing token 4 changed output at positions 0-3 by {diff_before}")

    diff_at = (out1[0, 4] - out2[0, 4]).abs().max().item()
    assert diff_at > 1e-3, (
        f"Perturbing token 4 should change its own output; diff={diff_at}")


def test_encoder_causal():
    _test_module_causality(_build_encoder)


def test_decoder_causal():
    _test_module_causality(_build_decoder)


def test_value_head_causal():
    """ValueHead pools over the sequence, so we check the intermediate causal
    self-attention, not the scalar output."""
    torch.manual_seed(42)
    mod = _build_value()
    mod.eval()
    B, S = 1, 6
    x = torch.randn(B, S, D_MODEL)
    mask = torch.ones(B, S)

    # Hook into norm output (after self-attn layers, before pooling)
    hooks = []
    internals = {}

    def capture(name):
        def hook(module, input, output):
            internals[name] = output.clone()
        return hook

    hooks.append(mod.norm.register_forward_hook(capture("norm")))

    with torch.no_grad():
        mod(x, mask=mask)
    out1 = internals["norm"]

    x2 = x.clone()
    x2[0, 4] += 10.0
    with torch.no_grad():
        mod(x2, mask=mask)
    out2 = internals["norm"]

    for h in hooks:
        h.remove()

    diff_before = (out1[0, :4] - out2[0, :4]).abs().max().item()
    assert diff_before < 1e-5, f"ValueHead causal violation: {diff_before}"


def test_action_head_causal():
    """ActionHead uses self-attn → pool → Linear. Same causality check on norm."""
    torch.manual_seed(42)
    mod = _build_action()
    mod.eval()
    B, S = 1, 6
    x = torch.randn(B, S, D_MODEL)
    mask = torch.ones(B, S)

    internals = {}

    def capture(name):
        def hook(module, input, output):
            internals[name] = output.clone()
        return hook

    h = mod.norm.register_forward_hook(capture("norm"))
    with torch.no_grad():
        mod(x, mask=mask)
    out1 = internals["norm"]

    x2 = x.clone()
    x2[0, 4] += 10.0
    with torch.no_grad():
        mod(x2, mask=mask)
    out2 = internals["norm"]
    h.remove()

    diff_before = (out1[0, :4] - out2[0, :4]).abs().max().item()
    assert diff_before < 1e-5, f"ActionHead causal violation: {diff_before}"


# ── A.1: Source code check — all 5 modules import build_causal_padding_mask ──

def test_all_modules_use_causal_mask():
    """Encoder, Decoder, ValueHead, ActionHead, OpponentActionHead must all
    import and use build_causal_padding_mask."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [
        "agent/perception/encoder.py",
        "agent/perception/decoder.py",
        "agent/value/value.py",
        "agent/action/action.py",
        "agent/opponent_action/opponent_action.py",
    ]
    for f in files:
        path = os.path.join(base, f)
        with open(path) as fh:
            source = fh.read()
        assert "build_causal_padding_mask" in source, (
            f"{f} does not use build_causal_padding_mask")


# ── A.2: ModellingHead autoregressive — no cross-attn, causal self-attn ─────

def test_modelling_no_cross_attention():
    """The redesigned ModellingHead must have NO cross-attention path:
    Qwen3CrossAttention removed from the module, no cross_attns/cross_norms
    submodules, no cross-attention keys in the state_dict."""
    import agent.modelling.modelling as modelling_module
    assert not hasattr(modelling_module, "Qwen3CrossAttention"), (
        "Qwen3CrossAttention must be removed from agent.modelling.modelling")
    mod = _build_modelling()
    assert not hasattr(mod, "cross_attns"), "ModellingHead must not have cross_attns"
    assert not hasattr(mod, "cross_norms"), "ModellingHead must not have cross_norms"
    cross_keys = [k for k in mod.state_dict() if "cross" in k]
    assert cross_keys == [], (
        f"ModellingHead state_dict must have no cross-attention keys: {cross_keys}")


def test_modelling_forward_shape():
    """forward keeps the old head's (B, n_actions, d_model) output contract."""
    mod = _build_modelling()
    mod.eval()
    B = 2
    ctx = torch.randn(B, 5, D_MODEL)
    mask = torch.ones(B, 5)
    with torch.no_grad():
        out = mod(ctx, mask=mask)
    assert out.shape == (B, N_ACTIONS, D_MODEL)


def test_modelling_self_attn_causal():
    """The context self-attn stack must be CAUSAL: perturbing token t must NOT
    change the per-position states at positions < t (the head predicts the
    NEXT decision state; seeing the future would let it cheat)."""
    torch.manual_seed(42)
    mod = _build_modelling()
    mod.eval()
    B, S = 1, 6
    x = torch.randn(B, S, D_MODEL)
    mask = torch.ones(B, S)

    with torch.no_grad():
        s1 = mod._encode(x, mask=mask)

    x2 = x.clone()
    x2[0, 4] += 10.0
    with torch.no_grad():
        s2 = mod._encode(x2, mask=mask)

    diff_before = (s1[0, :4] - s2[0, :4]).abs().max().item()
    assert diff_before < 1e-5, (
        f"Perturbing token 4 changed modelling states at positions 0-3 by "
        f"{diff_before} — self-attn stack is not causal")

    diff_at = (s1[0, 4] - s2[0, 4]).abs().max().item()
    assert diff_at > 1e-3, (
        f"Perturbing token 4 should change its own state; diff={diff_at}")


def test_modelling_action_conditioning_independent():
    """Action conditioning is a per-action MLP: perturbing action 0's embedding
    must NOT change action N-1's output (old symmetric cross-talk removed),
    but must change action 0's own output."""
    torch.manual_seed(42)
    mod = _build_modelling()
    mod.eval()
    B = 1
    ctx = torch.randn(B, 5, D_MODEL)
    mask = torch.ones(B, 5)

    with torch.no_grad():
        out_base = mod(ctx, mask=mask)

    # Perturb action embedding 0
    orig_emb = mod.action_embeddings.weight.data[0].clone()
    mod.action_embeddings.weight.data[0] += 5.0
    with torch.no_grad():
        out_perturbed = mod(ctx, mask=mask)
    mod.action_embeddings.weight.data[0] = orig_emb

    diff_last = (out_base[0, -1] - out_perturbed[0, -1]).abs().max().item()
    assert diff_last == 0.0, (
        f"Perturbing action 0 must NOT affect action {N_ACTIONS-1} "
        f"(independent MLP conditioning); diff={diff_last}")

    diff_self = (out_base[0, 0] - out_perturbed[0, 0]).abs().max().item()
    assert diff_self > 1e-4, (
        f"Perturbing action 0 must change its own output; diff={diff_self}")


# ── A.3: head_dim = d_model // n_heads ──────────────────────────────────────

def test_head_dim_encoder():
    enc = _build_encoder()
    expected = D_MODEL // N_HEADS
    assert enc.config.head_dim == expected, (
        f"Encoder head_dim={enc.config.head_dim}, expected {expected}")


def test_head_dim_decoder():
    dec = _build_decoder()
    expected = D_MODEL // N_HEADS
    assert dec.config.head_dim == expected


def test_head_dim_value():
    val = _build_value()
    assert val.config.head_dim == D_MODEL // N_HEADS


def test_head_dim_action():
    act = _build_action()
    assert act.config.head_dim == D_MODEL // N_HEADS


def test_head_dim_modelling():
    mod = _build_modelling()
    assert mod.config.head_dim == D_MODEL // N_HEADS


def test_qproj_shape():
    """q_proj.weight.shape should be (d_model, d_model) when head_dim is correct."""
    enc = _build_encoder()
    q_proj = enc.layers[0].self_attn.q_proj
    assert q_proj.weight.shape == (D_MODEL, D_MODEL), (
        f"q_proj shape {q_proj.weight.shape}, expected ({D_MODEL}, {D_MODEL})")


# ── A.5.2: Sequence capping guard ───────────────────────────────────────────

def test_sequence_capping():
    """EventSequenceEmbedder should cap long sequences to max_events."""
    from agent.perception.perception import EventSequenceEmbedder
    emb = EventSequenceEmbedder(d_model=D_MODEL, n_actions=N_ACTIONS,
                                max_players=6, max_seq_len=70)
    cap = emb.max_events  # 70 // 7 = 10
    seq = [{"table": [0, 1, 2, 3, 4], "hand": [10, 11],
            "hero_pos": 0, "acting_pos": 1, "num_players": 2,
            "pot": 100.0, "stack": 500.0, "bets": [5.0, 10.0],
            "stacks": [500.0, 490.0], "action": [0.0] * N_ACTIONS}] * 20
    capped = emb._cap_sequences([seq])
    assert len(capped[0]) <= cap, (
        f"Sequence length {len(capped[0])} exceeds cap {cap}")


# ── A.5.3: Card assert instead of silent clamp ──────────────────────────────

def test_card_assert_on_invalid():
    """extract_event_tensors should raise on out-of-range card indices."""
    from agent.perception.perception import extract_event_tensors
    bad_event = {"table": [0, 1, 2, 3, 99], "hand": [10, 11],
                 "hero_pos": 0, "acting_pos": 1, "num_players": 2,
                 "pot": 100.0, "stack": 500.0, "bets": [5.0, 10.0],
                 "stacks": [500.0, 490.0], "action": [0.0] * N_ACTIONS}
    try:
        extract_event_tensors([[bad_event]], max_players=6)
        assert False, "Should have raised AssertionError on card index 99"
    except (AssertionError, IndexError):
        pass
