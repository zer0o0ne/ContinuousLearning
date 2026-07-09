"""
E2E tests for the phase-4 modelling trainer with the LM next-decision-state
loss (PLAN_MODELLING_HEAD_REDESIGN.md §5).

Covers, through the REAL train_modelling entry point on a tiny ASI:
  1. Full run: LM pairs from a realistic scenario batch produce finite losses;
     ONLY modelling_head parameters change (perception/value/action frozen);
     mse/infonce components appear in the logs; best.pt is written; all
     params are unfrozen on exit.
  2. recon_weight=0 disables the LM term entirely (_reconstruction_loss is
     never called).
  3. M=0 batches (sequences with no valid LM pairs) train without error.
  4. Direct check on real perception output: the LM loss backward puts
     non-zero grads on modelling_head only; perception/value/action get none.

All tests are deterministic: fixed seeds, exact assertions, CPU.

Run from versions/v6/:
    python3 -m pytest tests/test_modelling_lm_train_e2e.py -v
"""

import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import ASI
import agent.train_scenarios.modelling_predict.train as mod_train
from agent.train_scenarios.modelling_predict.train import (
    train_modelling,
    _reconstruction_loss,
)

# ── Tiny ASI config (fast on CPU) ────────────────────────────────────────────

D_MODEL = 32
N_ACTIONS = 5   # fold + call + 2 raise bins + all-in
MAX_PLAYERS = 2

_TINY_CONFIG = {
    "architecture": {
        "d_model": D_MODEL,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 84,  # 84 // 7 = 12 max events
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
        "max_stack": 100,
    },
}

_BASE_TRAIN_CFG = {
    "lr": 1e-3,
    "batch_size": 4,
    "epochs": 2,
    "val_split": 0.34,
    "log_every": 1,
    "val_every": None,
    "interrupt_after_fails": None,
    "recon_weight": 0.8,
    "infonce_weight": 0.5,
    "infonce_temperature": 0.1,
}


class _CaptureLog:
    """Callable logger capturing every message."""

    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(str(msg))

    def contains(self, substring):
        return any(substring in m for m in self.messages)


def _make_event(hand, action_idx=None, pot=20.0):
    """Event dict with all fields the real EventSequenceEmbedder + the
    normalization pipeline require."""
    action = [0.0] * N_ACTIONS
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": [0, 1, 2, -1, -1],
        "hand": list(hand),
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 2,
        "pot": pot,
        "stack": 100.0,
        "big_blind": 10.0,
        "small_blind": 5.0,
        "bets": [5.0, 10.0],
        "stacks": [100.0, 95.0],
        "action": action,
    }


def _make_scenario(hand, action_indices, hand_id, ev=5.0):
    """Scenario whose event stream follows the PLAN §3 collect.py convention:
    [initial(None), post_0(one-hot), pre_1(None), post_1(one-hot), pre_2(None), ...].

    action_indices=None builds an action-free stream (no valid LM pairs).
    """
    events = [_make_event(hand, action_idx=None)]
    if action_indices is None:
        events += [_make_event(hand, action_idx=None) for _ in range(4)]
    else:
        for a in action_indices:
            events.append(_make_event(hand, action_idx=a))
            events.append(_make_event(hand, action_idx=None))
    return {
        "events": events,
        "ev_target": ev,
        "action_evs": [float(ev + i) for i in range(N_ACTIONS)],
        "action_probs": [1.0 / N_ACTIONS] * N_ACTIONS,
        "pot": 20.0,
        "facing_bet": 10.0,
        "num_players": 2,
        "hand_id": hand_id,
    }


def _make_scenarios(n_hands=6, per_hand=3, with_actions=True):
    scens = []
    for h in range(n_hands):
        hand = [10 + 2 * h, 11 + 2 * h]  # distinct hole cards per hand
        for k in range(per_hand):
            if with_actions:
                actions = [(h + k) % N_ACTIONS, (h + 2 * k + 1) % N_ACTIONS]
            else:
                actions = None
            scens.append(_make_scenario(hand, actions, hand_id=h,
                                        ev=5.0 + h + 0.5 * k))
    return scens


def _make_agent(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def _log(msg):
        pass

    agent = ASI(_log, config=_TINY_CONFIG)
    agent.set_device("cpu")
    return agent


def _named_param_snapshot(agent):
    return {name: p.detach().clone() for name, p in agent.named_parameters()}


def _changed_param_names(agent, snapshot):
    return [name for name, p in agent.named_parameters()
            if not torch.equal(p.detach(), snapshot[name])]


# ── 1. Full trainer run ──────────────────────────────────────────────────────

class TestFullTrainerRun(unittest.TestCase):

    def test_lm_loss_trains_modelling_head_only_and_logs_components(self):
        agent = _make_agent(seed=0)
        scenarios = _make_scenarios(n_hands=6, with_actions=True)
        snapshot = _named_param_snapshot(agent)
        logger = _CaptureLog()

        with tempfile.TemporaryDirectory() as run_dir:
            torch.manual_seed(123)
            random.seed(123)
            np.random.seed(123)
            history, out_dir = train_modelling(
                agent, dict(_BASE_TRAIN_CFG), "cpu", logger,
                scenarios_override=scenarios, run_dir=run_dir)

            # Trainer ran and produced finite step losses
            self.assertGreater(len(history["step_loss"]), 0)
            for _, step_loss in history["step_loss"]:
                self.assertTrue(np.isfinite(step_loss),
                                f"non-finite step loss: {step_loss}")
            for _, val_loss in history["val_loss"]:
                self.assertTrue(np.isfinite(val_loss),
                                f"non-finite val loss: {val_loss}")

            # best.pt written
            self.assertTrue(os.path.exists(os.path.join(run_dir, "best.pt")))

        # ONLY modelling_head parameters changed
        changed = _changed_param_names(agent, snapshot)
        self.assertGreater(len(changed), 0, "training must update parameters")
        for name in changed:
            self.assertTrue(
                name.startswith("modelling_head"),
                f"non-modelling parameter changed in phase 4: {name}")
        self.assertTrue(any(name.startswith("modelling_head") for name in changed))

        # LM components logged alongside the train loss
        self.assertTrue(logger.contains("LM mse:"),
                        "training logs must include the LM mse component")
        self.assertTrue(logger.contains("LM infonce:"),
                        "training logs must include the LM infonce component")
        self.assertTrue(logger.contains("infonce_weight=0.5"),
                        "startup log must state the configured infonce_weight")

        # Unfreeze on exit
        for name, p in agent.named_parameters():
            self.assertTrue(p.requires_grad,
                            f"{name} must be requires_grad=True after training")


# ── 2. recon_weight=0 disables the LM term ───────────────────────────────────

class TestReconWeightZero(unittest.TestCase):

    def test_recon_weight_zero_never_calls_lm_loss(self):
        agent = _make_agent(seed=1)
        scenarios = _make_scenarios(n_hands=6, with_actions=True)
        logger = _CaptureLog()

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError(
                "_reconstruction_loss must not be called when recon_weight=0")

        cfg = dict(_BASE_TRAIN_CFG)
        cfg["recon_weight"] = 0.0

        original = mod_train._reconstruction_loss
        mod_train._reconstruction_loss = _must_not_be_called
        try:
            with tempfile.TemporaryDirectory() as run_dir:
                torch.manual_seed(123)
                random.seed(123)
                np.random.seed(123)
                history, _ = train_modelling(
                    agent, cfg, "cpu", logger,
                    scenarios_override=scenarios, run_dir=run_dir)
        finally:
            mod_train._reconstruction_loss = original

        self.assertGreater(len(history["step_loss"]), 0)
        for _, step_loss in history["step_loss"]:
            self.assertTrue(np.isfinite(step_loss))
        # No LM component logging when the term is disabled
        self.assertFalse(logger.contains("LM mse:"))
        self.assertFalse(logger.contains("LM infonce:"))


# ── 3. M=0 batches (no valid LM pairs) ───────────────────────────────────────

class TestNoPairsBatches(unittest.TestCase):

    def test_training_with_actionless_sequences_runs_without_error(self):
        """Sequences whose events carry no one-hot action produce zero LM
        pairs (M=0) in EVERY batch — training must still complete with the
        zero-grad fallback."""
        agent = _make_agent(seed=2)
        scenarios = _make_scenarios(n_hands=6, with_actions=False)
        logger = _CaptureLog()

        with tempfile.TemporaryDirectory() as run_dir:
            torch.manual_seed(123)
            random.seed(123)
            np.random.seed(123)
            history, _ = train_modelling(
                agent, dict(_BASE_TRAIN_CFG), "cpu", logger,
                scenarios_override=scenarios, run_dir=run_dir)

        self.assertGreater(len(history["step_loss"]), 0)
        for _, step_loss in history["step_loss"]:
            self.assertTrue(np.isfinite(step_loss),
                            f"non-finite step loss with M=0 batches: {step_loss}")
        # M=0 → both components are exactly 0 in the running averages
        self.assertTrue(logger.contains("LM mse: 0.000000"))
        self.assertTrue(logger.contains("LM infonce: 0.000000"))


# ── 4. Direct gradient routing on real perception output ────────────────────

class TestGradientRouting(unittest.TestCase):

    def test_lm_loss_grads_only_on_modelling_head(self):
        agent = _make_agent(seed=3)
        agent.train()

        # Phase-4 freeze pattern
        for p in agent.perception.parameters():
            p.requires_grad = False
        for p in agent.value_head.parameters():
            p.requires_grad = False
        for p in agent.action_head.parameters():
            p.requires_grad = False
        for p in agent.modelling_head.parameters():
            p.requires_grad = True

        scenarios = _make_scenarios(n_hands=2, with_actions=True)
        event_sequences = [s["events"] for s in scenarios]

        with torch.no_grad():
            p_out, _, mask = agent.perception.forward_batch(
                event_sequences, device="cpu", skip_memory=True)
        p_out = p_out.detach()

        loss, components = _reconstruction_loss(
            agent, p_out, mask, event_sequences,
            infonce_weight=0.5, infonce_temperature=0.1)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(np.isfinite(components["mse"]))
        self.assertTrue(np.isfinite(components["infonce"]))
        loss.backward()

        has_modelling_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in agent.modelling_head.parameters()
        )
        self.assertTrue(has_modelling_grad,
                        "LM loss must put non-zero grads on modelling_head")

        for name, p in agent.named_parameters():
            if name.startswith(("perception", "value_head", "action_head",
                                "opponent_action_head")):
                self.assertIsNone(
                    p.grad,
                    f"{name} must receive NO grad from the LM loss")


if __name__ == "__main__":
    unittest.main()
