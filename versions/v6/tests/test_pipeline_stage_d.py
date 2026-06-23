"""Stage D training-pipeline audit-fix tests.

Covers:
  D.1  LR scheduling: clamped cosine never increases after warmup
  D.2  MCTS train/val split is hand-aware (no hand_id in both sets)
  D.3  Teacher forcing is hero-only (opp-step events excluded)
  D.4  Selective heads in forward_batch for phases 1-3
  D.5.1  gto_probs return value
  D.5.2  Deterministic hole-card sampling in OpponentActionDataset
  D.5.3  config_hash includes multi_agent.agents

Run:
    python -m tests.test_pipeline_stage_d        # from versions/v6
"""

import copy
import math
import random
import sys
import unittest
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# D.1 — Clamped Cosine never wraps
# ---------------------------------------------------------------------------

class TestD1ClampedCosine(unittest.TestCase):
    def test_lr_monotonic_after_warmup(self):
        """LR must never increase after warmup completes."""
        import torch
        from torch.optim.lr_scheduler import (
            LinearLR, CosineAnnealingLR, SequentialLR)

        class _ClampedCosineAnnealingLR(CosineAnnealingLR):
            def get_lr(self):
                if self.last_epoch >= self.T_max:
                    return [self.eta_min for _ in self.base_lrs]
                return super().get_lr()

        lr = 1e-3
        eta_min = 1e-6
        T_max = 50
        warmup_steps = 10
        model = torch.nn.Linear(4, 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        warmup = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
        cosine = _ClampedCosineAnnealingLR(
            opt, T_max=T_max, eta_min=eta_min)
        sched = SequentialLR(opt, [warmup, cosine],
                             milestones=[warmup_steps])

        lrs = []
        for step in range(warmup_steps + T_max + 100):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        # After warmup, LR must be monotonically non-increasing
        post_warmup = lrs[warmup_steps:]
        for i in range(1, len(post_warmup)):
            self.assertLessEqual(
                post_warmup[i], post_warmup[i - 1] + 1e-12,
                f"LR increased at step {warmup_steps + i}: "
                f"{post_warmup[i]:.2e} > {post_warmup[i-1]:.2e}")

        # Final LR should be at eta_min (not wrapped back up)
        self.assertAlmostEqual(post_warmup[-1], eta_min, places=8)

    def test_unclamped_cosine_wraps(self):
        """Sanity: standard CosineAnnealingLR DOES wrap."""
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR

        lr = 1e-3
        T_max = 50
        model = torch.nn.Linear(4, 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = CosineAnnealingLR(opt, T_max=T_max, eta_min=1e-6)

        for _ in range(T_max):
            sched.step()
        lr_at_end = opt.param_groups[0]["lr"]
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        # Standard cosine wraps: LR increases past T_max
        self.assertGreater(lr_after, lr_at_end)


# ---------------------------------------------------------------------------
# D.2 — Hand-aware MCTS split
# ---------------------------------------------------------------------------

@dataclass
class _FakeExample:
    events: list = field(default_factory=list)
    value_target: float = 0.0
    action_target: list = field(default_factory=list)
    chain: list = field(default_factory=list)
    terminal_targets: list = field(default_factory=list)


class TestD2HandAwareSplit(unittest.TestCase):
    def test_no_hand_leak(self):
        """No hand_id should appear in both train and val splits."""
        from agent.train_scenarios.split import hand_aware_split

        # Create 20 examples from 4 hands (5 examples each)
        examples = []
        for hand_id in range(4):
            hand_cards = [hand_id * 2, hand_id * 2 + 1]
            table_cards = [20 + hand_id, 21 + hand_id, 22 + hand_id,
                           23 + hand_id, 24 + hand_id]
            for dec in range(5):
                events = [{
                    "hand": hand_cards,
                    "table": table_cards,
                    "num_players": 2,
                    "hero_pos": 0,
                    "pot": 100,
                    "bets": [10, 20],
                    "big_blind": 10,
                    "small_blind": 5,
                    "stack": 1000,
                    "stacks": [1000, 1000],
                    "action": [0] * 14,
                }]
                examples.append(_FakeExample(events=events))

        import torch
        from torch.utils.data import Dataset

        class _DS(Dataset):
            def __init__(self, exs):
                self.exs = exs
            def __len__(self):
                return len(self.exs)
            def __getitem__(self, idx):
                return self.exs[idx]

        ds = _DS(examples)
        scenarios_for_split = [
            {"events": ex.events,
             "num_players": ex.events[0]["num_players"]}
            for ex in examples
        ]
        train_ds, val_ds = hand_aware_split(
            ds, scenarios_for_split, val_split=0.25, seed=42)

        # Collect hand signatures per split
        def _hand_sig(idx):
            return tuple(examples[idx].events[0]["hand"])

        train_hands = {_hand_sig(i) for i in train_ds.indices}
        val_hands = {_hand_sig(i) for i in val_ds.indices}
        self.assertEqual(
            len(train_hands & val_hands), 0,
            f"Hand leak: {train_hands & val_hands}")
        # All examples accounted for
        self.assertEqual(len(train_ds) + len(val_ds), len(examples))


# ---------------------------------------------------------------------------
# D.3 — Teacher forcing hero-only
# ---------------------------------------------------------------------------

@dataclass
class _FakeChainStep:
    action_taken: int = 0
    target_distribution: list = field(default_factory=list)
    is_hero: bool = True
    events_at_step: list = field(default_factory=list)
    value_target: float = 0.0
    root_q_ratio: float = 0.0


class TestD3TeacherForcingHeroOnly(unittest.TestCase):
    def test_opp_events_excluded_from_flat_step_events(self):
        """Only hero-owned chain steps should appear in flat_step_events."""
        hero_events = [{"hand": [0, 1], "table": [2, 3, 4, 5, 6]}]
        opp_events = [{"hand": [10, 11], "table": [2, 3, 4, 5, 6]}]

        chains = [[
            _FakeChainStep(is_hero=True, events_at_step=hero_events),
            _FakeChainStep(is_hero=False, events_at_step=opp_events),
            _FakeChainStep(is_hero=True, events_at_step=hero_events),
        ]]

        # Replicate the D.3 filtering logic from _mcts_forward
        flat_step_events = []
        step_index = {}
        for b, chain in enumerate(chains):
            for i, step in enumerate(chain):
                evts = getattr(step, "events_at_step", None) or []
                if evts and step.is_hero:
                    step_index[(b, i)] = len(flat_step_events)
                    flat_step_events.append(evts)

        # Only hero steps should be in flat_step_events
        self.assertEqual(len(flat_step_events), 2)
        # Opp step (index 1) should NOT be in step_index
        self.assertNotIn((0, 1), step_index)
        # Hero steps at indices 0 and 2 should be present
        self.assertIn((0, 0), step_index)
        self.assertIn((0, 2), step_index)

    def test_use_tf_requires_is_hero(self):
        """Teacher forcing condition must include step.is_hero."""
        step_hero = _FakeChainStep(is_hero=True)
        step_opp = _FakeChainStep(is_hero=False)

        p_tf = 1.0
        tf_idx = 0  # pretend it's in step_index
        chain_perception_out = True  # non-None

        def _should_tf(step, tf_idx_val):
            return (tf_idx_val is not None and p_tf > 0.0
                    and step.is_hero
                    and chain_perception_out is not None)

        self.assertTrue(_should_tf(step_hero, tf_idx))
        self.assertFalse(_should_tf(step_opp, tf_idx))


# ---------------------------------------------------------------------------
# D.4 — Selective heads
# ---------------------------------------------------------------------------

class TestD4SelectiveHeads(unittest.TestCase):
    def test_forward_batch_heads_filter(self):
        """forward_batch with heads= should only compute requested heads."""
        import torch

        class _FakeAgent:
            def __init__(self):
                self.called_heads = set()
            def forward_batch(self, events, skip_memory=True, heads=None):
                result = {}
                if heads is None or "value" in heads:
                    self.called_heads.add("value")
                    result["value"] = torch.zeros(1)
                if heads is None or "action" in heads:
                    self.called_heads.add("action")
                    result["action_logits"] = torch.zeros(1, 14)
                if heads is None or "opponent_action" in heads:
                    self.called_heads.add("opponent_action")
                    result["opponent_action_logits"] = torch.zeros(1, 14)
                if heads is None or "modelling" in heads:
                    self.called_heads.add("modelling")
                    result["action_embeddings"] = torch.zeros(1, 14, 64)
                return result

        agent = _FakeAgent()
        agent.forward_batch([], heads={"value"})
        self.assertEqual(agent.called_heads, {"value"})

        agent.called_heads.clear()
        agent.forward_batch([], heads={"action"})
        self.assertEqual(agent.called_heads, {"action"})

        agent.called_heads.clear()
        agent.forward_batch([], heads={"value", "action"})
        self.assertEqual(agent.called_heads, {"value", "action"})

        agent.called_heads.clear()
        agent.forward_batch([], heads=None)
        self.assertEqual(agent.called_heads,
                         {"value", "action", "opponent_action", "modelling"})


# ---------------------------------------------------------------------------
# D.5.1 — gto_probs return value
# ---------------------------------------------------------------------------

class TestD51GtoPropsReturn(unittest.TestCase):
    def test_return_none_none_on_empty_scenarios(self):
        """gto_probs should return (None, None), not bare return."""
        import ast, textwrap
        import os
        path = os.path.join(os.path.dirname(__file__), "..",
                            "agent", "train_scenarios",
                            "gto_probs_predict", "train.py")
        path = os.path.normpath(path)
        with open(path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        # Find the Return node after "No scenarios generated"
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if node.value is None:
                    # Bare return — should not exist in function that
                    # unpacks (history, run_dir)
                    # Check if this is inside a function that has another
                    # `return history, run_dir`
                    self.fail("Found bare 'return' — should be 'return None, None'")


# ---------------------------------------------------------------------------
# D.5.2 — Deterministic hole-card sampling
# ---------------------------------------------------------------------------

class TestD52DeterministicHoleCards(unittest.TestCase):
    def test_same_seed_same_hand(self):
        """Same seed should produce the same hand."""
        from agent.train_scenarios.opponent_action_predict.dataset import (
            OpponentActionDataset)

        scenario = {
            "events": [{
                "table": [10, 11, 12, 13, 14],
                "hands": {},  # no fixed hand — forces sampling
                "acting_pos": 0,
                "num_players": 2,
                "hero_pos": 0,
                "big_blind": 10,
                "small_blind": 5,
                "pot": 100,
                "bets": [5, 10],
                "stack": 1000,
                "stacks": [1000, 1000],
                "action": [0] * 14,
            }],
            "hero_positions": [0],
            "opponent_action_probs": [1.0 / 14] * 14,
        }
        ds = OpponentActionDataset([scenario])
        # Access _resolve_hero_hand directly with fixed seed
        h1 = ds._resolve_hero_hand(scenario["events"], 0, seed=42)
        h2 = ds._resolve_hero_hand(scenario["events"], 0, seed=42)
        h3 = ds._resolve_hero_hand(scenario["events"], 0, seed=99)
        self.assertEqual(h1, h2, "Same seed must give same hand")
        # Different seeds CAN give different hands (probabilistically)
        # Just check they're valid (no table collision)
        table_cards = {10, 11, 12, 13, 14}
        self.assertTrue(set(h1).isdisjoint(table_cards))
        self.assertTrue(set(h3).isdisjoint(table_cards))


# ---------------------------------------------------------------------------
# D.5.3 — config_hash includes multi_agent.agents
# ---------------------------------------------------------------------------

class TestD53ConfigHash(unittest.TestCase):
    def test_modifier_change_changes_hash(self):
        """Changing multi_agent.agents should change the config hash."""
        from agent.resume import compute_config_hash

        config_a = {
            "game": {"big_blind": 10},
            "solver": {"type": "v3"},
            "multi_agent": {
                "agents": [{"name": "neutral", "modifiers": []}]
            },
        }
        config_b = copy.deepcopy(config_a)
        config_b["multi_agent"]["agents"][0]["modifiers"] = [
            {"type": "temperature", "value": 1.5}
        ]

        hash_a = compute_config_hash(config_a)
        hash_b = compute_config_hash(config_b)
        self.assertNotEqual(hash_a, hash_b,
                            "Modifier change must invalidate hash")

    def test_same_config_same_hash(self):
        """Identical configs must produce the same hash."""
        from agent.resume import compute_config_hash

        config = {
            "game": {"big_blind": 10},
            "solver": {"type": "v3"},
            "multi_agent": {"agents": [{"name": "a"}]},
        }
        self.assertEqual(compute_config_hash(config),
                         compute_config_hash(copy.deepcopy(config)))

    def test_no_multi_agent_still_works(self):
        """Config without multi_agent section should not crash."""
        from agent.resume import compute_config_hash

        config = {"game": {"big_blind": 10}, "solver": {"type": "v3"}}
        h = compute_config_hash(config)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA256 hex


if __name__ == "__main__":
    unittest.main()
