"""E2E tests for the compounding-error fixes (PLAN_COMPOUNDING_ERROR.md).

Covers, each mechanism with its flag ON and OFF (legacy):
  §1  street-boundary leaf in MCTS search: the child of a street-closing
      action is frozen as a cached value_head leaf (never expanded);
      legacy path still expands it.
  §2a chain_street_boundary in collect_training_data: chains stop before
      the first future decision on a different street.
  §2b build_lm_pairs(street_boundary=True): pairs whose target crosses a
      street boundary are dropped.
  §3  truncated BPTT in _mcts_forward: backward works, gradients differ
      from the stop-grad legacy, legacy stays bit-for-bit deterministic.
  §5  spectral clamp on the modelling conditioning MLP: σ(W_eff) bounded
      by max_sigma; exact identity below the cap; a legacy (clamp-less)
      state_dict loads under strict=False and reproduces outputs.

All tests are fully deterministic: fixed seeds (conftest reseeds before
every test), no probabilistic assertions, no order dependence.

Run from versions/v7/:
    python3 -m pytest tests/test_compounding_error_fixes.py -v
"""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from agent.agent import ASI
from agent.mcts.game_state import GameState
from agent.mcts.mcts import MCTS, MCTSNode
from agent.mcts.collect import ChainStep, collect_training_data
from agent.modelling.modelling import ModellingHead, build_lm_pairs
from agent.train_scenarios.mcts_predict.train import (
    _mcts_forward, _compute_loss, _scale_grad,
)

BIG_BLIND = 10
RAISE_SIZES = [[0.5, 1.0]] * 4
N_RAISE_BINS = len(RAISE_SIZES[0])
N_ACTIONS = N_RAISE_BINS + 3
MAX_PLAYERS = 2
D_MODEL = 32

_TINY_CONFIG = {
    "architecture": {
        "d_model": D_MODEL,
        "n_heads": 2,
        "n_kv_heads": 1,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 64,
        "max_players": MAX_PLAYERS,
        "modelling_dropout": 0.0,
        "memory": {
            "n_levels": 1,
            "max_cluster_size": 4,
            "max_cluster_size_after": 4,
            "beam_width": 2,
        },
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": [0.5, 1.0],
            "flop": [0.5, 1.0],
            "turn": [0.5, 1.0],
            "river": [0.5, 1.0],
        },
        "max_players": MAX_PLAYERS,
        "big_blind": BIG_BLIND,
        "max_stack": 200,
    },
    "solver": {"type": "v1"},
}


# ─── helpers ─────────────────────────────────────────────────────────────────

class _StubEvaluator:
    """Deterministic NN stub for MCTS: zero values, uniform priors,
    fixed action embeddings. Enough for tree-structure tests."""

    def __init__(self, d_model=16, n_actions=N_ACTIONS):
        self.n_actions = n_actions
        self.d_model = d_model
        g = torch.Generator().manual_seed(7)
        self._act_embs = torch.randn(1, n_actions, d_model, generator=g)

    def evaluate_root(self, event_sequences):
        ctx = torch.zeros(1, 3, self.d_model)
        mask = torch.ones(1, 3)
        act_logits = torch.zeros(1, self.n_actions)
        opp_logits = torch.zeros(1, self.n_actions)
        return ctx, mask, 0.0, act_logits, opp_logits, self._act_embs

    def evaluate_leaves(self, context, mask, needs_expansion):
        B = context.shape[0]
        values = torch.zeros(B)
        act_logits = torch.zeros(B, self.n_actions)
        opp_logits = torch.zeros(B, self.n_actions)
        act_embs = self._act_embs.expand(B, -1, -1)
        return values, act_logits, opp_logits, act_embs


def _preflop_check_closes_street_gs():
    """Heads-up preflop, hero to act, bets already matched: check (action 1)
    closes the street (turn 0 → 1); any raise keeps the street open."""
    return GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=[1, 0],
        credits=[90.0, 90.0], bets=[10.0, 10.0],
        pot=20.0, high_bet=10.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
    )


def _make_mcts(street_cap):
    cfg = {
        "n_simulations": 80,
        "c_puct": 1.5,
        "batch_size": 4,
        "virtual_loss": 0.5,
        "temperature": 1.0,
        "street_boundary_leaf": street_cap,
    }
    return MCTS(agent=None, device="cpu", mcts_config=cfg,
                evaluator=_StubEvaluator(), search_scale=1.0)


def _event(action_idx=None, pot=30.0, table=None):
    action = [0.0] * N_ACTIONS
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": list(table) if table is not None else [-1] * 5,
        "hand": [10, 11],
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 2,
        "pot": pot,
        "stack": 100.0,
        "bets": [5.0, 10.0],
        "stacks": [100.0, 95.0],
        "action": action,
    }


def _decision(turn, action_idx=1, player_pos=0):
    """Minimal decision dict for collect_training_data: a 2-child MCTS root
    with visits and a street snapshot."""
    root = MCTSNode(is_hero=True)
    for a, n in ((1, 3), (2, 1)):
        child = MCTSNode(action_idx=a, parent=root, is_hero=False)
        child.N = n
        root.children[a] = child
    root.N = 4
    root.W = 0.4
    root.Q = 0.1
    return {
        "player_pos": player_pos,
        "action_idx": action_idx,
        "mcts_root": root,
        "events_at_root": [_event(None)],
        "game_state_at_root": SimpleNamespace(turn=turn),
    }


def _make_agent(seed=0):
    torch.manual_seed(seed)
    agent = ASI(lambda m: None, config=_TINY_CONFIG)
    agent.set_device("cpu")
    agent.eval()
    return agent


def _chain_step(action_taken):
    return ChainStep(
        action_taken=action_taken,
        target_distribution=[1.0 / N_ACTIONS] * N_ACTIONS,
        is_hero=True,
        events_at_step=[],
        value_target=0.1,
    )


def _modelling_grads(agent):
    return [p.grad.clone() if p.grad is not None else None
            for p in agent.modelling_head.parameters()]


_CHAIN_ONLY = dict(value_weight=0.0, action_weight=0.0, chain_weight=1.0,
                   value_chain_weight=0.5, recon_weight=0.0,
                   terminal_value_weight=0.0, entropy_weight=0.0,
                   chain_depth_gamma=0.9)


def _run_chain_backward(bptt_depth, bptt_grad_scale=0.5, seed=0):
    agent = _make_agent(seed=seed)
    seqs = [[_event(None, pot=20.0), _event(1, pot=20.0),
             _event(None, pot=40.0)]]
    chains = [[_chain_step(1), _chain_step(2), _chain_step(0)]]
    out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0,
                        stop_grad_old_embs=True,
                        bptt_depth=bptt_depth,
                        bptt_grad_scale=bptt_grad_scale)
    vt = torch.zeros(1)
    at = torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS)
    loss, ldict = _compute_loss(out, vt, at, **_CHAIN_ONLY)
    loss.backward()
    return agent, loss.item(), ldict


# ─────────────────────────────────────────────────────────────────────────────
# §1 street-boundary leaf in MCTS search
# ─────────────────────────────────────────────────────────────────────────────

class TestStreetBoundaryLeaf:

    def test_street_leaf_frozen_and_cached(self):
        mcts = _make_mcts(street_cap=True)
        gs = _preflop_check_closes_street_gs()
        action = mcts.search([[_event(None)]], gs)

        root = mcts.last_root
        assert root is not None
        check_child = root.children[1]  # check closes the street
        assert check_child.N > 0, "check branch must be visited"
        assert check_child.is_street_leaf, \
            "child of a street-closing action must be a street leaf"
        assert check_child.children == {}, "street leaf must never expand"
        assert check_child._term_value is not None, \
            "street leaf caches ONE value_head estimate"
        assert not check_child.is_terminal, \
            "street leaf is NOT a terminal (equity override must skip it)"
        assert action in root.children

    def test_within_street_subtree_still_expands(self):
        mcts = _make_mcts(street_cap=True)
        gs = _preflop_check_closes_street_gs()
        mcts.search([[_event(None)]], gs)

        root = mcts.last_root
        # Action 3 = pot-size raise (the 0.5-pot bin is filtered by the NLHE
        # min-raise rule: 5 < BB); a raise keeps the street open.
        raise_child = root.children[3]
        assert raise_child.N > 0
        assert not raise_child.is_street_leaf
        assert raise_child.children, \
            "within-street opponent node must still be expanded"
        # The opponent's CALL of the raise closes the street → depth-2
        # street leaf; the opponent's re-raise keeps the tree growing.
        opp_call = raise_child.children[1]
        if opp_call.N > 0:
            assert opp_call.is_street_leaf
            assert opp_call.children == {}

    def test_legacy_flag_off_expands_across_street(self):
        mcts = _make_mcts(street_cap=False)
        gs = _preflop_check_closes_street_gs()
        mcts.search([[_event(None)]], gs)

        root = mcts.last_root
        check_child = root.children[1]
        assert check_child.N > 0
        assert not check_child.is_street_leaf
        assert check_child.children, \
            "legacy path must keep expanding across the street boundary"


# ─────────────────────────────────────────────────────────────────────────────
# §2a chain street cut in collect_training_data
# ─────────────────────────────────────────────────────────────────────────────

class TestChainStreetBoundary:

    def _hand_record(self):
        # decisions 0, 1 on preflop (turn 0); decision 2 on the flop (turn 1).
        return {"decisions": [_decision(0), _decision(0), _decision(1)]}

    def test_chain_cut_at_street(self):
        examples = collect_training_data(
            self._hand_record(), N_ACTIONS, max_chain_depth=5,
            chain_street_boundary=True)
        by_t = {t: ex for t, ex in examples}
        assert len(by_t[0].chain) == 1, \
            "chain of decision 0 must stop before the flop decision"
        assert len(by_t[1].chain) == 0, \
            "decision 1's only future decision is on the flop → empty chain"
        assert len(by_t[2].chain) == 0

    def test_chain_full_when_flag_off(self):
        examples = collect_training_data(
            self._hand_record(), N_ACTIONS, max_chain_depth=5,
            chain_street_boundary=False)
        by_t = {t: ex for t, ex in examples}
        assert len(by_t[0].chain) == 2
        assert len(by_t[1].chain) == 1

    def test_max_chain_depth_still_applies(self):
        record = {"decisions": [_decision(0), _decision(0), _decision(0),
                                _decision(0)]}
        examples = collect_training_data(
            record, N_ACTIONS, max_chain_depth=2,
            chain_street_boundary=True)
        by_t = {t: ex for t, ex in examples}
        assert len(by_t[0].chain) == 2, "hard depth cap composes (min of two)"


# ─────────────────────────────────────────────────────────────────────────────
# §2b LM-pair street filter
# ─────────────────────────────────────────────────────────────────────────────

class TestLMPairStreetFilter:

    def _seq(self):
        preflop = [-1] * 5
        flop = [1, 2, 3, -1, -1]
        # Decision 1 closes preflop: post-action snapshot (q=1) and the next
        # pre-decision snapshot (q=2) already show the flop.
        # Decision 2 (q=3) stays within the flop.
        return [
            _event(None, table=preflop),   # q=0 src of pair 1 (street 0)
            _event(1, table=flop),         # q=1 action
            _event(None, table=flop),      # q=2 tgt of pair 1 (street 1) / src of pair 2
            _event(2, table=flop),         # q=3 action
            _event(None, table=flop),      # q=4 tgt of pair 2 (street 1)
        ]

    def test_cross_street_pair_dropped(self):
        bi, src, act, tgt = build_lm_pairs([self._seq()],
                                           street_boundary=True)
        assert src.tolist() == [2], "only the within-flop pair survives"
        assert tgt.tolist() == [4]
        assert act.tolist() == [2]

    def test_legacy_keeps_all_pairs(self):
        bi, src, act, tgt = build_lm_pairs([self._seq()],
                                           street_boundary=False)
        assert src.tolist() == [0, 2]
        assert tgt.tolist() == [2, 4]

    def test_no_card_encodings_equivalent(self):
        """-1 (raw) and 52 (embedder token) must both count as no-card."""
        seq_52 = self._seq()
        for e in seq_52:
            e["table"] = [52 if c < 0 else c for c in e["table"]]
        bi, src, act, tgt = build_lm_pairs([seq_52], street_boundary=True)
        assert src.tolist() == [2]


# ─────────────────────────────────────────────────────────────────────────────
# §3 truncated BPTT
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncatedBPTT:

    def test_scale_grad_semantics(self):
        x = torch.ones(3, requires_grad=True)
        y = _scale_grad(x * 2.0, 0.5).sum()
        y.backward()
        assert torch.allclose(x.grad, torch.full((3,), 1.0)), \
            "forward ×2 with scale 0.5 → grad exactly 1.0 per element"

    def test_bptt_backward_finite_and_differs_from_stop_grad(self):
        agent_a, loss_a, _ = _run_chain_backward(bptt_depth=0)
        agent_b, loss_b, _ = _run_chain_backward(bptt_depth=2)

        # Forward values identical: BPTT changes only the gradient graph.
        assert abs(loss_a - loss_b) < 1e-9

        grads_a = _modelling_grads(agent_a)
        grads_b = _modelling_grads(agent_b)
        assert any(g is not None and torch.isfinite(g).all()
                   for g in grads_b)
        max_diff = 0.0
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            max_diff = max(max_diff, (ga - gb).abs().max().item())
        assert max_diff > 1e-9, \
            "BPTT must open gradient paths the stop-grad legacy blocks"

    def test_legacy_stop_grad_bit_for_bit_deterministic(self):
        agent_a, loss_a, _ = _run_chain_backward(bptt_depth=0)
        agent_b, loss_b, _ = _run_chain_backward(bptt_depth=0)
        assert loss_a == loss_b
        for ga, gb in zip(_modelling_grads(agent_a),
                          _modelling_grads(agent_b)):
            if ga is None:
                assert gb is None
            else:
                assert torch.equal(ga, gb)

    def test_grad_scale_damps(self):
        """Smaller bptt_grad_scale ⇒ the BPTT-only gradient contribution
        shrinks toward the stop-grad gradient."""
        agent_stop, _, _ = _run_chain_backward(bptt_depth=0)
        agent_hi, _, _ = _run_chain_backward(bptt_depth=2,
                                             bptt_grad_scale=1.0)
        agent_lo, _, _ = _run_chain_backward(bptt_depth=2,
                                             bptt_grad_scale=0.1)

        def _dist(a, b):
            s = 0.0
            for ga, gb in zip(_modelling_grads(a), _modelling_grads(b)):
                if ga is not None and gb is not None:
                    s += (ga - gb).abs().sum().item()
            return s

        assert _dist(agent_lo, agent_stop) < _dist(agent_hi, agent_stop)


# ─────────────────────────────────────────────────────────────────────────────
# §5 spectral clamp
# ─────────────────────────────────────────────────────────────────────────────

def _tiny_head(spectral_clamp=None, seed=3):
    torch.manual_seed(seed)
    return ModellingHead(
        d_model=16, n_actions=N_ACTIONS, n_heads=2, n_kv_heads=1,
        n_layers=1, d_ff=32, max_seq_len=32, dropout=0.0,
        spectral_clamp=spectral_clamp)


class TestSpectralClamp:

    def test_sigma_bounded_above_cap(self):
        head = _tiny_head({"enabled": True, "max_sigma": 1.0,
                           "n_power_iterations": 2})
        head.train()
        with torch.no_grad():
            head.mlp_in.weight.mul_(50.0)
        # Several forwards converge the power-iteration u buffer.
        for _ in range(10):
            w_eff = head._clamped_weight(head.mlp_in.weight, "sn_u_in")
        sigma = torch.linalg.matrix_norm(w_eff.detach().float(), ord=2)
        assert sigma.item() <= 1.0 + 0.05, \
            f"σ(W_eff)={sigma.item():.4f} must be clamped to ≈ max_sigma"

    def test_exact_identity_below_cap(self):
        head = _tiny_head({"enabled": True, "max_sigma": 1.0,
                           "n_power_iterations": 2})
        head.eval()
        with torch.no_grad():
            head.mlp_in.weight.mul_(0.01)
        w_eff = head._clamped_weight(head.mlp_in.weight, "sn_u_in")
        assert torch.equal(w_eff, head.mlp_in.weight), \
            "below the cap the clamp must be an exact identity"

    def test_legacy_state_dict_loads_and_matches(self):
        legacy = _tiny_head(spectral_clamp=None, seed=3)
        clamped = _tiny_head({"enabled": True, "max_sigma": 1e6,
                              "n_power_iterations": 1}, seed=4)
        missing, unexpected = clamped.load_state_dict(
            legacy.state_dict(), strict=False)
        assert not unexpected, "legacy checkpoint has no unknown keys"
        assert all("sn_u_" in k for k in missing), \
            "only the new power-iteration buffers may be missing"

        legacy.eval()
        clamped.eval()
        g = torch.Generator().manual_seed(11)
        ctx = torch.randn(2, 4, 16, generator=g)
        mask = torch.ones(2, 4)
        with torch.no_grad():
            out_a = legacy(ctx, mask=mask)
            out_b = clamped(ctx, mask=mask)
        assert torch.allclose(out_a, out_b, atol=1e-6), \
            "huge max_sigma ⇒ clamp inactive ⇒ legacy weights reproduce " \
            "legacy outputs"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
