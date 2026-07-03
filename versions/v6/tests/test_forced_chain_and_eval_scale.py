"""E2E tests for two MCTS self-play fixes (2026-07-02):

1. Forced decisions (single legal action → `mcts.search` returns
   `last_root=None` with no fallback_action_distribution) must NOT poison
   chain targets: the chain step is kept (its action still extends the
   modelling-chain context; its realized value target is valid) but is
   marked `target_valid=False` and EXCLUDED from the chain KL loss —
   previously it was trained toward a uniform distribution over ALL
   n_actions. Past-snapshot opponents' real fallback distributions keep
   working exactly as before.
   Source: agent/mcts/collect.py (chain assembly),
           agent/train_scenarios/mcts_predict/train.py (_mcts_forward,
           _compute_loss).

2. Agent-vs-agent evaluation constructs MCTS with the agent's checkpoint
   `mcts_value_scale` as `search_scale` (fallback: big blind) — both the
   sequential path (`_build_agent_bundle`) and the threaded server path
   (`_EvalInferenceServer.create_mcts`).
   Source: evaluation/evaluate.py.

All tests are fully deterministic: fixed inputs, no probabilistic
assertions, no order dependence.

Run from versions/v6/:
    python -m pytest tests/test_forced_chain_and_eval_scale.py -v
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from agent.mcts.collect import collect_training_data
from agent.mcts.mcts import MCTSNode
from agent.train_scenarios.mcts_predict.train import _compute_loss

N_ACTIONS = 5


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_root(visit_counts, q=0.5):
    """MCTSNode root with children whose N are `visit_counts[action_idx]`."""
    root = MCTSNode(is_hero=True)
    root.Q = q
    for a, n in visit_counts.items():
        child = MCTSNode(action_idx=a, parent=root)
        child.N = n
        root.children[a] = child
    return root


def _decision(player_pos, action_idx, mcts_root, fallback=None):
    return {
        "player_pos": player_pos,
        "action_idx": action_idx,
        "mcts_root": mcts_root,
        "fallback_action_distribution": fallback,
        "events_at_root": [{"num_players": 2}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1a. collect_training_data: forced decisions → target_valid=False,
#     past-snapshot fallbacks → target_valid=True with the real distribution.
# ─────────────────────────────────────────────────────────────────────────────

class TestForcedChainStepCollection:

    def _build_examples(self):
        past_fallback = [0.0, 0.6, 0.3, 0.1, 0.0]
        decisions = [
            # t=0: hero (pos 0), real tree
            _decision(0, 1, _make_root({1: 3, 2: 1}, q=0.4)),
            # t=1: pos 1, FORCED single-legal-action decision by an ACTIVE
            # agent: no tree, no fallback distribution.
            _decision(1, 2, None, fallback=None),
            # t=2: pos 1, past-snapshot opponent: no tree, REAL fallback.
            _decision(1, 0, None, fallback=list(past_fallback)),
            # t=3: hero again (pos 0), real tree
            _decision(0, 3, _make_root({0: 2, 3: 2}, q=-0.1)),
        ]
        hand_record = {"decisions": decisions}
        examples = collect_training_data(hand_record, N_ACTIONS)
        return examples, past_fallback

    def test_forced_step_marked_invalid(self):
        examples, _ = self._build_examples()
        # First example is the tree at t=0; its chain[0] targets t=1 (forced).
        t0, ex0 = examples[0]
        assert t0 == 0
        assert ex0.chain[0].target_valid is False
        # Placeholder target stays well-formed (uniform) so batching works.
        assert ex0.chain[0].target_distribution == pytest.approx(
            [1.0 / N_ACTIONS] * N_ACTIONS)
        # The forced step still records the context-extending action.
        assert ex0.chain[0].action_taken == 1  # action at t=0 advances t0→t1

    def test_past_snapshot_fallback_still_valid(self):
        examples, past_fallback = self._build_examples()
        _, ex0 = examples[0]
        # chain[1] targets t=2 — the past-snapshot decision.
        assert ex0.chain[1].target_valid is True
        assert ex0.chain[1].target_distribution == pytest.approx(past_fallback)

    def test_tree_backed_steps_valid(self):
        examples, _ = self._build_examples()
        _, ex0 = examples[0]
        # chain[2] targets t=3 — a real MCTS tree.
        assert ex0.chain[2].target_valid is True
        # Visit distribution of the t=3 tree: N={0:2, 3:2} → 0.5/0.5.
        expected = [0.5, 0.0, 0.0, 0.5, 0.0]
        assert ex0.chain[2].target_distribution == pytest.approx(expected)

    def test_no_examples_for_forced_or_past_decisions(self):
        examples, _ = self._build_examples()
        # Only t=0 and t=3 produced examples (t=1 forced, t=2 past).
        assert [t for t, _ in examples] == [0, 3]


# ─────────────────────────────────────────────────────────────────────────────
# 1b. _compute_loss: invalid chain steps contribute zero to the chain KL
#     while other steps contribute normally (original-depth gamma weights).
# ─────────────────────────────────────────────────────────────────────────────

def _forward_out(preds, targets, valid):
    """Minimal forward_out dict for _compute_loss with one example."""
    return {
        "value_preds": torch.zeros(1, 1),
        "action_preds": torch.zeros(1, N_ACTIONS),
        "chain_action_preds": [list(preds)],
        "chain_action_targets": [list(targets)],
        "chain_action_valid": [list(valid)],
        "chain_is_hero": [[True] * len(preds)],
        "chain_value_preds": [[]],
        "chain_value_targets": [[]],
        # No LM pairs (modelling-head redesign): _compute_loss falls back
        # to a zero recon loss when "lm_pred" is absent.
        "terminal_value_preds": [[]],
        "terminal_value_targets": [[]],
    }


def _kl(pred, target):
    return F.kl_div(F.log_softmax(pred, dim=-1), target,
                    reduction="sum").item()


class TestForcedChainStepLossMasking:
    GAMMA = 0.5

    def _preds_targets(self):
        p0 = torch.tensor([1.0, 0.0, -1.0, 0.5, 0.2], requires_grad=True)
        p1 = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3], requires_grad=True)
        p2 = torch.tensor([-0.5, 1.5, 0.0, 0.0, -1.0], requires_grad=True)
        t0 = torch.tensor([0.7, 0.1, 0.1, 0.1, 0.0])
        t1 = torch.tensor([0.2] * 5)  # uniform placeholder of a forced step
        t2 = torch.tensor([0.0, 0.9, 0.1, 0.0, 0.0])
        return [p0, p1, p2], [t0, t1, t2]

    def test_invalid_step_contributes_zero(self):
        """Chain loss with step 1 invalid == weighted mean over steps 0 and 2
        with their ORIGINAL depth weights gamma^0 and gamma^2."""
        preds, targets = self._preds_targets()
        fo = _forward_out(preds, targets, [True, False, True])
        _, ldict = _compute_loss(
            fo, torch.zeros(1), torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS),
            chain_depth_gamma=self.GAMMA)
        w0, w2 = self.GAMMA ** 0, self.GAMMA ** 2
        expected = (w0 * _kl(preds[0], targets[0])
                    + w2 * _kl(preds[2], targets[2])) / (w0 + w2)
        assert ldict["chain"] == pytest.approx(expected, rel=1e-5)

    def test_garbage_target_of_invalid_step_is_ignored(self):
        """Changing the invalid step's target must not change the loss."""
        preds, targets = self._preds_targets()
        fo_a = _forward_out(preds, targets, [True, False, True])
        garbage = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        fo_b = _forward_out(preds, [targets[0], garbage, targets[2]],
                            [True, False, True])
        root_v = torch.zeros(1)
        root_a = torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS)
        _, ldict_a = _compute_loss(fo_a, root_v, root_a,
                                   chain_depth_gamma=self.GAMMA)
        _, ldict_b = _compute_loss(fo_b, root_v, root_a,
                                   chain_depth_gamma=self.GAMMA)
        assert ldict_a["chain"] == pytest.approx(ldict_b["chain"], rel=1e-7)
        assert ldict_a["total"] == pytest.approx(ldict_b["total"], rel=1e-7)

    def test_invalid_step_receives_no_gradient(self):
        """Backward through the total loss: the invalid step's prediction
        gets no gradient; valid steps' predictions do."""
        preds, targets = self._preds_targets()
        fo = _forward_out(preds, targets, [True, False, True])
        total, _ = _compute_loss(
            fo, torch.zeros(1), torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS),
            chain_depth_gamma=self.GAMMA)
        total.backward()
        p0, p1, p2 = preds
        assert p1.grad is None or torch.all(p1.grad == 0), (
            "invalid chain step must contribute zero gradient")
        assert p0.grad is not None and torch.any(p0.grad != 0)
        assert p2.grad is not None and torch.any(p2.grad != 0)

    def test_all_valid_matches_legacy_weighting(self):
        """With every step valid, the loss equals the legacy per-example
        gamma-weighted mean over all steps."""
        preds, targets = self._preds_targets()
        fo = _forward_out(preds, targets, [True, True, True])
        _, ldict = _compute_loss(
            fo, torch.zeros(1), torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS),
            chain_depth_gamma=self.GAMMA)
        weights = [self.GAMMA ** i for i in range(3)]
        expected = sum(w * _kl(p, t) for w, p, t
                       in zip(weights, preds, targets)) / sum(weights)
        assert ldict["chain"] == pytest.approx(expected, rel=1e-5)

    def test_missing_valid_key_treats_all_steps_valid(self):
        """Legacy forward_out without 'chain_action_valid' behaves as before."""
        preds, targets = self._preds_targets()
        fo = _forward_out(preds, targets, [True, True, True])
        del fo["chain_action_valid"]
        _, ldict = _compute_loss(
            fo, torch.zeros(1), torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS),
            chain_depth_gamma=self.GAMMA)
        weights = [self.GAMMA ** i for i in range(3)]
        expected = sum(w * _kl(p, t) for w, p, t
                       in zip(weights, preds, targets)) / sum(weights)
        assert ldict["chain"] == pytest.approx(expected, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluation wires search_scale from checkpoint norm_stats into MCTS.
# ─────────────────────────────────────────────────────────────────────────────

MAX_PLAYERS = 2

_TINY_CONFIG = {
    "architecture": {
        "d_model": 32,
        "n_heads": 2,
        "n_kv_heads": 1,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 56,
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

_IDENTITY_NORM = {
    "pot_mean": 0.0, "pot_std": 1.0,
    "stack_mean": 0.0, "stack_std": 1.0,
    "bets_mean": 0.0, "bets_std": 1.0,
    "blind_mean": 0.0, "blind_std": 1.0,
}


def _save_ckpt(tmpdir, norm_stats):
    from agent.agent import ASI
    asi = ASI(lambda m: None, config=_TINY_CONFIG)
    path = os.path.join(tmpdir, "best.pt")
    torch.save({
        "model_state_dict": asi.state_dict(),
        "norm_stats": norm_stats,
        "temperature": 0.7,
    }, path)
    return path


class TestEvalMCTSSearchScale:

    def test_sequential_bundle_uses_checkpoint_scale(self):
        """_build_agent_bundle(use_mcts=True) passes the checkpoint's
        mcts_value_scale into MCTS.search_scale."""
        from evaluation.evaluate import _build_agent_bundle
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = dict(_IDENTITY_NORM)
            ns["mcts_value_scale"] = 123.0
            path = _save_ckpt(tmpdir, ns)
            bundle = _build_agent_bundle(
                "a", path, _TINY_CONFIG, "cpu", lambda m: None,
                fallback_temperature=0.5, use_mcts=True,
                mcts_cfg={"n_simulations": 4})
            assert bundle["mcts"] is not None
            assert bundle["mcts"].search_scale == pytest.approx(123.0)

    def test_sequential_bundle_falls_back_to_big_blind(self):
        """Checkpoint without mcts_value_scale → search_scale == big blind."""
        from evaluation.evaluate import _build_agent_bundle
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_ckpt(tmpdir, dict(_IDENTITY_NORM))
            bundle = _build_agent_bundle(
                "a", path, _TINY_CONFIG, "cpu", lambda m: None,
                fallback_temperature=0.5, use_mcts=True,
                mcts_cfg={"n_simulations": 4})
            assert bundle["mcts"].search_scale == pytest.approx(
                float(_TINY_CONFIG["game"]["big_blind"]))

    def test_threaded_create_mcts_passes_search_scale(self):
        """_EvalInferenceServer.create_mcts forwards search_scale into the
        RemoteEvaluator-backed MCTS (agent=None path)."""
        from evaluation.evaluate import _EvalInferenceServer
        srv = _EvalInferenceServer(
            mcts_agents=[], config=_TINY_CONFIG, device="cpu",
            log=lambda m: None, n_tables=1)
        # No server process needed: create_mcts only allocates a thread slot
        # and builds the (side-effect-free) RemoteEvaluator + MCTS objects.
        srv._all_resp_qs = [None]
        srv._req_q = None
        mcts = srv.create_mcts("a", {"n_simulations": 4}, N_ACTIONS,
                               search_scale=77.0)
        assert mcts.search_scale == pytest.approx(77.0)
