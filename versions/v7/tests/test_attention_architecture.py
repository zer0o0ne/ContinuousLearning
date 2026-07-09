"""Этап-A architecture regression tests.

A.1 — causality: in encoder / decoder / value / action / opponent_action,
      perturbing token t must NOT change the per-position representation at any
      position < t (tested with mask=ones AND with real trailing padding).
A.2 — ModellingHead autoregressive stack (PLAN_MODELLING_HEAD_REDESIGN.md §2):
      the context self-attn stack is CAUSAL (perturbing token t must not change
      per-position states < t, with mask=ones AND real trailing padding), and
      action conditioning is independent per action (perturbing action i's
      embedding must not change action j's output).
A.3 — head_dim = d_model // n_heads: every attention q_proj is (d_model, d_model)
      rather than the 128-default-inflated shape.

Run (from versions/v6):
    python -m tests.test_attention_architecture
"""

import torch

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.value.value import ValueHead
from agent.action.action import ActionHead
from agent.opponent_action.opponent_action import OpponentActionHead
from agent.modelling.modelling import ModellingHead

D_MODEL = 16
N_HEADS = 4
N_KV = 2
N_LAYERS = 2
D_FF = 32
MAX_SEQ = 64
N_ACTIONS = 14
B = 3
S = 12


def _build_all():
    torch.manual_seed(0)
    return {
        "encoder": Encoder(D_MODEL, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
        "decoder": Decoder(D_MODEL, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
        "value": ValueHead(D_MODEL, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
        "action": ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
        "opp_action": OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
        "modelling": ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV, N_LAYERS, D_FF, MAX_SEQ),
    }


def _causality_for(module, name, mask):
    """Hook the pre-pool RMSNorm to read per-position states; assert causality."""
    module.eval()
    cap = {}
    handle = module.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("o", o.detach().clone()))
    torch.manual_seed(1)
    x = torch.randn(B, S, D_MODEL)
    t = S // 2  # perturb a real token in the middle
    with torch.no_grad():
        module(x, mask=mask)
        states_a = cap["o"].clone()
        x2 = x.clone()
        x2[:, t] = x2[:, t] + 7.0
        module(x2, mask=mask)
        states_b = cap["o"].clone()
    handle.remove()
    assert torch.allclose(states_a[:, :t], states_b[:, :t], atol=1e-5), (
        f"{name}: positions < {t} changed when token {t} was perturbed — NOT causal")
    assert not torch.allclose(states_a[:, t], states_b[:, t], atol=1e-5), (
        f"{name}: perturbation at {t} had no effect — test is not exercising the model")


def test_causality_all_modules():
    mods = _build_all()
    causal_mods = ["encoder", "decoder", "value", "action", "opp_action"]
    mask_ones = torch.ones(B, S)
    # real trailing padding: last 4 positions padded
    mask_pad = torch.ones(B, S)
    mask_pad[:, S - 4:] = 0.0
    for name in causal_mods:
        _causality_for(mods[name], f"{name}[mask=ones]", mask_ones)
        _causality_for(mods[name], f"{name}[padding]", mask_pad)
    print(f"test_causality_all_modules: OK ({', '.join(causal_mods)}; ones + padding)")


def _modelling_causality(head, name, mask):
    """Perturb token t; assert per-position self-attn states < t are unchanged."""
    head.eval()
    torch.manual_seed(1)
    x = torch.randn(B, S, D_MODEL)
    t = S // 2  # perturb a real token in the middle
    with torch.no_grad():
        states_a = head._encode(x, mask=mask)
        x2 = x.clone()
        x2[:, t] = x2[:, t] + 7.0
        states_b = head._encode(x2, mask=mask)
    assert torch.allclose(states_a[:, :t], states_b[:, :t], atol=1e-5), (
        f"{name}: positions < {t} changed when token {t} was perturbed — NOT causal")
    assert not torch.allclose(states_a[:, t], states_b[:, t], atol=1e-5), (
        f"{name}: perturbation at {t} had no effect — test is not exercising the model")


def test_modelling_causality():
    head = _build_all()["modelling"]
    mask_ones = torch.ones(B, S)
    # real trailing padding: last 4 positions padded
    mask_pad = torch.ones(B, S)
    mask_pad[:, S - 4:] = 0.0
    _modelling_causality(head, "modelling[mask=ones]", mask_ones)
    _modelling_causality(head, "modelling[padding]", mask_pad)
    print("test_modelling_causality: OK (self-attn stack causal; ones + padding)")


def test_modelling_action_conditioning_independence():
    head = _build_all()["modelling"]
    head.eval()
    torch.manual_seed(2)
    context = torch.randn(1, S, D_MODEL)
    mask = torch.ones(1, S)
    a, b = 0, 13
    with torch.no_grad():
        base = head(context, mask=mask)
        w = head.action_embeddings.weight.data.clone()

        head.action_embeddings.weight.data[b] += 3.0
        out_b = head(context, mask=mask)
        head.action_embeddings.weight.data.copy_(w)
        eff_a_from_b = (out_b[0, a] - base[0, a]).abs().max().item()
        eff_b_from_b = (out_b[0, b] - base[0, b]).abs().max().item()

    assert eff_a_from_b == 0.0, (
        f"perturbing action {b} affected action {a} by {eff_a_from_b} — "
        f"action conditioning must be an independent per-action MLP")
    assert eff_b_from_b > 1e-4, (
        f"perturbing action {b} did not affect its own output — "
        f"conditioning path not exercised")
    print(f"test_modelling_action_conditioning_independence: OK "
          f"(a<-b={eff_a_from_b:.4f}, b<-b={eff_b_from_b:.4f})")


def test_head_dim_shapes():
    mods = _build_all()
    # Self-attention q_proj in the Qwen3 decoder layers.
    for name in ["encoder", "decoder", "value", "action", "opp_action"]:
        qw = mods[name].layers[0].self_attn.q_proj.weight
        assert tuple(qw.shape) == (D_MODEL, D_MODEL), f"{name}: q_proj {tuple(qw.shape)}"
    # ModellingHead: the causal self-attn stack — q_proj full, k_proj GQA-shaped.
    m = mods["modelling"]
    sqw = m.self_attn_layers[0].self_attn.q_proj.weight
    assert tuple(sqw.shape) == (D_MODEL, D_MODEL), f"modelling self q_proj {tuple(sqw.shape)}"
    head_dim = D_MODEL // N_HEADS
    skw = m.self_attn_layers[0].self_attn.k_proj.weight
    assert tuple(skw.shape) == (N_KV * head_dim, D_MODEL), (
        f"modelling self k_proj {tuple(skw.shape)} — expected GQA "
        f"({N_KV * head_dim}, {D_MODEL})")
    # Action-conditioning MLP: cat(s_t, e_a) → d_ff → d_model.
    assert tuple(m.mlp_in.weight.shape) == (D_FF, 2 * D_MODEL), (
        f"modelling mlp_in {tuple(m.mlp_in.weight.shape)}")
    assert tuple(m.mlp_out.weight.shape) == (D_MODEL, D_FF), (
        f"modelling mlp_out {tuple(m.mlp_out.weight.shape)}")
    print("test_head_dim_shapes: OK (all q_proj == (d_model, d_model))")


if __name__ == "__main__":
    test_head_dim_shapes()
    test_causality_all_modules()
    test_modelling_causality()
    test_modelling_action_conditioning_independence()
    print("\nALL ATTENTION-ARCHITECTURE TESTS PASSED")
