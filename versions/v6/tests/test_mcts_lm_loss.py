"""E2E tests for the phase-6 modelling-head LM loss
(PLAN_MODELLING_HEAD_REDESIGN.md §6, wave 2b):

- Root-sequence LM loss: pairs built via `build_lm_pairs` on the root
  events, predicted via `modelling_head.forward_positions`, targets =
  detached root perception tokens at q+1.
- Chain LM supervision replaces the old rolled-context recon: at chain
  depth d the already-computed emb (h at the rolled context's last
  position) is supervised against the LAST true token of the real next
  decision's perception (`chain_perception_out[tf_idx, L_true-1]`,
  detached).
- Both pair sets share ONE `lm_loss` call (MSE + InfoNCE), gated by
  `recon_weight`; components logged as recon_mse / recon_infonce.
- Forced (target_valid=False) chain steps stay excluded from the chain KL
  (regression guard) but still receive LM supervision when they carry
  events_at_step (same guard set as the old recon: hero steps with
  events_at_step).

All tests are fully deterministic: fixed seeds, no probabilistic
assertions, no order dependence.

Run from versions/v6/:
    python3 -m pytest tests/test_mcts_lm_loss.py -v
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from agent.agent import ASI
from agent.mcts.collect import ChainStep
from agent.train_scenarios.mcts_predict.train import _mcts_forward, _compute_loss

N_ACTIONS = 5  # 2 raise sizes + fold + call + all-in
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
        "big_blind": 10,
        "max_stack": 200,
    },
    "solver": {"type": "v1"},
}

# Weights that isolate the LM (recon) loss: every other term zeroed.
_LM_ONLY = dict(value_weight=0.0, action_weight=0.0, chain_weight=0.0,
                value_chain_weight=0.0, terminal_value_weight=0.0,
                entropy_weight=0.0, recon_weight=1.0,
                infonce_weight=0.5, infonce_temperature=0.1)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _event(action_idx=None, pot=30.0, card_base=0):
    """One event dict. action_idx=None → pre-decision snapshot (zeros
    action vector); int → post-action snapshot (one-hot)."""
    action = [0.0] * N_ACTIONS
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": [card_base, card_base + 1, card_base + 2, 52, 52],
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


def _seq_two_decisions():
    """[snap, post(a=1), snap, post(a=2), snap] → 2 LM pairs
    (q=1: src 0 → tgt 2; q=3: src 2 → tgt 4)."""
    return [_event(None, pot=20.0), _event(1, pot=20.0),
            _event(None, pot=40.0), _event(2, pot=40.0),
            _event(None, pot=80.0)]


def _seq_one_decision():
    """[snap, post(a=1), snap] → exactly 1 LM pair."""
    return [_event(None, pot=20.0), _event(1, pot=20.0),
            _event(None, pot=40.0)]


def _seq_no_decision():
    """Single snapshot → 0 LM pairs."""
    return [_event(None, pot=20.0)]


def _chain_events(card_base=5):
    """events_at_step of a real next decision — ends at the pre-decision
    snapshot (its last true token IS the next decision-point token)."""
    return [_event(None, pot=20.0, card_base=card_base),
            _event(0, pot=20.0, card_base=card_base),
            _event(None, pot=60.0, card_base=card_base)]


def _chain_step(action_taken=2, is_hero=True, events=None, target_valid=True,
                target=None):
    if target is None:
        target = [1.0 / N_ACTIONS] * N_ACTIONS
    return ChainStep(
        action_taken=action_taken,
        target_distribution=list(target),
        is_hero=is_hero,
        events_at_step=list(events) if events else [],
        value_target=0.1,
        target_valid=target_valid,
    )


def _make_agent(seed=0, train_mode=False):
    torch.manual_seed(seed)
    agent = ASI(lambda m: None, config=_TINY_CONFIG)
    agent.set_device("cpu")
    agent.train() if train_mode else agent.eval()
    return agent


def _root_targets(B):
    return (torch.zeros(B),
            torch.full((B, N_ACTIONS), 1.0 / N_ACTIONS))


def _modelling_grad_sum(agent):
    total = 0.0
    for p in agent.modelling_head.parameters():
        if p.grad is not None:
            total += p.grad.abs().sum().item()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 1. Root-LM path: finite loss, nonzero modelling-head gradients even with
#    EMPTY chains (the dense collapse-independent signal of PLAN §6).
# ─────────────────────────────────────────────────────────────────────────────

class TestRootLMPath:

    def test_root_pairs_produce_grads_with_empty_chains(self):
        agent = _make_agent(seed=0)
        seqs = [_seq_two_decisions(), _seq_two_decisions()]
        chains = [[], []]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)

        assert out["lm_pred"].shape == (4, D_MODEL)  # 2 pairs per example
        assert out["lm_target"].shape == (4, D_MODEL)
        assert not out["lm_target"].requires_grad  # stop-grad targets

        vt, at = _root_targets(2)
        loss, ldict = _compute_loss(out, vt, at, **_LM_ONLY)
        assert torch.isfinite(loss).item()
        assert ldict["recon"] > 0.0

        loss.backward()
        assert _modelling_grad_sum(agent) > 0.0, (
            "root-sequence LM loss must give the modelling head gradient "
            "even when every chain is empty")

    def test_root_targets_are_next_decision_tokens(self):
        """lm_target rows equal perception_out[bi, q+1] (PLAN §3: target =
        the NEXT decision-point token)."""
        agent = _make_agent(seed=1)
        seqs = [_seq_two_decisions()]
        out = _mcts_forward(agent, seqs, [[]], "cpu", p_tf=0.0)

        with torch.no_grad():
            pe, _, _ = agent.perception.forward_batch(
                seqs, device="cpu", skip_memory=True)
        assert torch.allclose(out["lm_target"][0], pe[0, 2], atol=1e-6)
        assert torch.allclose(out["lm_target"][1], pe[0, 4], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chain-LM path: emb at the rolled context's last position supervised
#    against the LAST true token of the real next decision's perception.
# ─────────────────────────────────────────────────────────────────────────────

class TestChainLMPath:

    def test_chain_pairs_produce_grads_without_root_pairs(self):
        """Root sequences carry NO decisions (0 root pairs) — the chain LM
        pairs alone must drive modelling-head gradients."""
        agent = _make_agent(seed=2)
        seqs = [_seq_no_decision(), _seq_no_decision()]
        chains = [[_chain_step(action_taken=2, events=_chain_events(5))],
                  [_chain_step(action_taken=3, events=_chain_events(15))]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)

        assert out["lm_pred"].shape == (2, D_MODEL)  # chain pairs only
        assert out["lm_pred"].requires_grad, (
            "the CURRENT step's emb must stay attached so the head gets "
            "gradient (stop_grad_old_embs only detaches the next-depth ctx)")

        vt, at = _root_targets(2)
        loss, ldict = _compute_loss(out, vt, at, **_LM_ONLY)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert _modelling_grad_sum(agent) > 0.0, (
            "chain LM pairs must give the modelling head gradient")

    def test_chain_target_is_last_true_token(self):
        """Chain LM target == the last TRUE token of the real next
        decision's perception (per-token, NOT the masked mean pool)."""
        agent = _make_agent(seed=3)
        events = _chain_events(5)
        seqs = [_seq_no_decision()]
        chains = [[_chain_step(action_taken=1, events=events)]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)

        with torch.no_grad():
            pe, _, pm = agent.perception.forward_batch(
                [events], device="cpu", skip_memory=True)
        L_true = int(pm[0].sum().item())
        expected = pe[0, L_true - 1]
        assert out["lm_target"].shape == (1, D_MODEL)
        assert torch.allclose(out["lm_target"][0], expected, atol=1e-6)
        # Regression vs the OLD recon: target must NOT be the mean pool.
        pooled = (pe[0, :L_true]).mean(dim=0)
        assert not torch.allclose(out["lm_target"][0], pooled, atol=1e-4)

    def test_opp_and_eventless_steps_get_no_lm_pair(self):
        """Old guard semantics preserved: only hero steps with a real
        events_at_step are LM-supervised (D.3)."""
        agent = _make_agent(seed=4)
        seqs = [_seq_no_decision()]
        chains = [[
            _chain_step(action_taken=1, is_hero=False,
                        events=_chain_events(5)),      # opp → no pair
            _chain_step(action_taken=2, is_hero=True,
                        events=None),                  # no events → no pair
            _chain_step(action_taken=3, is_hero=True,
                        events=_chain_events(15)),     # hero+events → pair
        ]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)
        assert out["lm_pred"].shape == (1, D_MODEL)

    def test_teacher_forced_prediction_contexts_still_work(self):
        """p_tf=1.0 (random() < 1.0 always holds → deterministic TF): the
        chain perception feeds BOTH the TF prediction contexts and the LM
        targets; loss stays finite and the modelling head still trains."""
        agent = _make_agent(seed=5)
        seqs = [_seq_one_decision()]
        chains = [[_chain_step(action_taken=2, events=_chain_events(5))]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=1.0)

        assert out["lm_pred"].shape == (2, D_MODEL)  # 1 root + 1 chain
        vt, at = _root_targets(1)
        loss, _ = _compute_loss(out, vt, at, **_LM_ONLY)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert _modelling_grad_sum(agent) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss composition: recon_weight gate, MSE + InfoNCE components,
#    shared root+chain InfoNCE batch.
# ─────────────────────────────────────────────────────────────────────────────

class TestLMLossComposition:

    def _forward(self, seed=6):
        agent = _make_agent(seed=seed)
        seqs = [_seq_two_decisions(), _seq_one_decision()]
        chains = [[_chain_step(action_taken=2, events=_chain_events(5))], []]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)
        return agent, out

    def test_recon_weight_zero_disables_lm_gradient(self):
        agent, out = self._forward()
        vt, at = _root_targets(2)
        w = dict(_LM_ONLY)
        w["recon_weight"] = 0.0
        loss, ldict = _compute_loss(out, vt, at, **w)
        # Components still REPORTED (unweighted) for logging...
        assert ldict["recon"] > 0.0
        # ...but the total carries no LM term and gives the head no grad.
        assert loss.item() == pytest.approx(0.0, abs=1e-8)
        loss.backward()
        assert _modelling_grad_sum(agent) == 0.0

    def test_total_shifts_by_weighted_recon(self):
        _, out = self._forward()
        vt, at = _root_targets(2)
        w0 = dict(_LM_ONLY); w0["recon_weight"] = 0.0
        w1 = dict(_LM_ONLY); w1["recon_weight"] = 0.5
        loss0, ld0 = _compute_loss(out, vt, at, **w0)
        loss1, ld1 = _compute_loss(out, vt, at, **w1)
        assert ld0["recon"] == pytest.approx(ld1["recon"], rel=1e-7)
        assert loss1.item() - loss0.item() == pytest.approx(
            0.5 * ld1["recon"], rel=1e-5)

    def test_infonce_component_present_with_two_or_more_pairs(self):
        _, out = self._forward()
        assert out["lm_pred"].shape[0] >= 2
        vt, at = _root_targets(2)
        _, ldict = _compute_loss(out, vt, at, **_LM_ONLY)
        assert ldict["recon_infonce"] > 0.0
        assert ldict["recon_mse"] > 0.0
        assert ldict["recon"] == pytest.approx(
            ldict["recon_mse"] + 0.5 * ldict["recon_infonce"], rel=1e-5)

    def test_infonce_skipped_with_single_pair(self):
        agent = _make_agent(seed=7)
        seqs = [_seq_one_decision()]
        out = _mcts_forward(agent, seqs, [[]], "cpu", p_tf=0.0)
        assert out["lm_pred"].shape[0] == 1
        vt, at = _root_targets(1)
        _, ldict = _compute_loss(out, vt, at, **_LM_ONLY)
        assert ldict["recon_infonce"] == 0.0
        assert ldict["recon"] == pytest.approx(ldict["recon_mse"], rel=1e-7)

    def test_root_and_chain_share_one_infonce_batch(self):
        """One root pair + one chain pair: neither alone reaches M ≥ 2, but
        pooled they do → InfoNCE fires (PLAN §6: both pair sets share one
        L_lm)."""
        agent = _make_agent(seed=8)
        seqs = [_seq_one_decision()]  # 1 root pair
        chains = [[_chain_step(action_taken=2, events=_chain_events(5))]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)
        assert out["lm_pred"].shape[0] == 2
        vt, at = _root_targets(1)
        _, ldict = _compute_loss(out, vt, at, **_LM_ONLY)
        assert ldict["recon_infonce"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge cases: zero-pair batches, forced steps, gradient checkpointing.
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_pair_batch_trains(self):
        """No decisions anywhere → (0, D) LM tensors, zero recon, finite
        total, backward works and the OTHER heads still get gradients."""
        agent = _make_agent(seed=9)
        seqs = [_seq_no_decision(), _seq_no_decision()]
        out = _mcts_forward(agent, seqs, [[], []], "cpu", p_tf=0.0)
        assert out["lm_pred"].shape == (0, D_MODEL)
        assert out["lm_target"].shape == (0, D_MODEL)

        vt, at = _root_targets(2)
        loss, ldict = _compute_loss(
            out, vt, at, value_weight=1.0, action_weight=1.0,
            chain_weight=1.0, value_chain_weight=0.5, recon_weight=0.5,
            infonce_weight=0.5, infonce_temperature=0.1)
        assert torch.isfinite(loss).item()
        assert ldict["recon"] == 0.0
        assert ldict["recon_mse"] == 0.0
        assert ldict["recon_infonce"] == 0.0
        loss.backward()
        value_grads = sum(p.grad.abs().sum().item()
                          for p in agent.value_head.parameters()
                          if p.grad is not None)
        assert value_grads > 0.0

    def test_forced_steps_excluded_from_chain_kl_but_lm_supervised(self):
        """Regression guard (fix landed 2026-07-02): target_valid=False
        steps contribute nothing to the chain KL — yet they still take an
        action that extends ctx, so they DO get an LM pair when they carry
        events_at_step (old recon guard did not check target_valid)."""
        agent = _make_agent(seed=10)
        seqs = [_seq_one_decision()]
        chains = [[
            _chain_step(action_taken=1, events=_chain_events(5),
                        target_valid=True,
                        target=[0.7, 0.1, 0.1, 0.1, 0.0]),
            _chain_step(action_taken=2, events=_chain_events(15),
                        target_valid=False),  # forced: uniform placeholder
        ]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)

        assert out["chain_action_valid"][0] == [True, False]
        # 1 root pair + 2 chain pairs (forced step included).
        assert out["lm_pred"].shape[0] == 3

        vt, at = _root_targets(1)
        weights = dict(_LM_ONLY)
        weights["chain_weight"] = 1.0
        _, ldict_a = _compute_loss(out, vt, at, **weights)
        # Garbage target on the INVALID step must not change the chain KL.
        out["chain_action_targets"][0][1] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0])
        _, ldict_b = _compute_loss(out, vt, at, **weights)
        assert ldict_a["chain"] == pytest.approx(ldict_b["chain"], rel=1e-7)
        assert ldict_a["total"] == pytest.approx(ldict_b["total"], rel=1e-7)

    def test_gradient_checkpointing_path(self):
        """gradient_checkpointing on (train mode): forward through both LM
        paths, finite loss, nonzero modelling-head grads."""
        agent = _make_agent(seed=11, train_mode=True)
        agent.set_gradient_checkpointing(True)
        seqs = [_seq_two_decisions()]
        chains = [[_chain_step(action_taken=2, events=_chain_events(5))]]
        out = _mcts_forward(agent, seqs, chains, "cpu", p_tf=0.0)
        assert out["lm_pred"].shape[0] == 3  # 2 root + 1 chain

        vt, at = _root_targets(1)
        loss, _ = _compute_loss(out, vt, at, **_LM_ONLY)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert _modelling_grad_sum(agent) > 0.0
        agent.set_gradient_checkpointing(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
