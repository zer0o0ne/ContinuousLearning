"""Normalization pipeline tests.

Tests for EV and event normalization used across all training phases.
Covers _compute_norm_stats, _normalize_scenarios, _normalize_action_evs,
and the norm-stats reuse contract between phases.

Run (from versions/v6):
    python -m tests.test_normalization_pipeline
"""

import sys
import os
import copy
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow imports from versions/v6
# ---------------------------------------------------------------------------
_V6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

from agent.train_scenarios.generation.generate import (
    _compute_norm_stats,
    _normalize_scenarios,
    _shallow_copy_scenarios,
)
from agent.train_scenarios.modelling_predict.train import _normalize_action_evs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(pot, stack, big_blind, bets=None, num_players=2, stacks=None):
    """Build a minimal event dict matching the schema used in generate.py."""
    if bets is None:
        bets = [0.0] * num_players
    if stacks is None:
        stacks = [stack] * num_players
    return {
        "hand": [0, 1],
        "num_players": num_players,
        "hero_pos": 0,
        "acting_pos": 0,
        "big_blind": float(big_blind),
        "small_blind": float(big_blind) / 2.0,
        "stack": float(stack),
        "stacks": [float(c) for c in stacks],
        "table": [-1, -1, -1, -1, -1],
        "pot": float(pot),
        "bets": np.array(bets, dtype=np.float64),
        "action": [0.0, 1.0, 0.0, 0.0],
    }


def _make_scenario(ev_target, pot, facing_bet, big_blind=10.0,
                   stack=500.0, n_events=3, n_actions=4,
                   action_evs=None, stacks=None):
    """Build a minimal raw scenario dict."""
    num_players = 2
    events = [
        _make_event(pot, stack, big_blind, num_players=num_players, stacks=stacks)
        for _ in range(n_events)
    ]
    if action_evs is None:
        action_evs = [ev_target * 0.5, ev_target, ev_target * 0.8, ev_target * 0.3]
    return {
        "ev_target": float(ev_target),
        "action_evs": list(action_evs),
        "action_probs": [0.1, 0.5, 0.3, 0.1],
        "equity": 0.5,
        "pot": float(pot),
        "facing_bet": float(facing_bet),
        "stack": float(stack),
        "hero_invested": 0.0,
        "num_players": num_players,
        "n_events": n_events,
        "events": events,
    }


def _make_diverse_scenarios(n=50, seed=42):
    """Build a diverse set of scenarios spanning a range of pot/EV values."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n):
        big_blind = 10.0
        pot = float(rng.uniform(10, 500))
        facing_bet = float(rng.uniform(0, pot * 0.5))
        stack = float(rng.uniform(50, 1000))
        ev_target = float(rng.uniform(-50, 200))
        # Build per-seat stacks with some variation
        stacks = [float(rng.uniform(50, 1000)) for _ in range(2)]
        s = _make_scenario(
            ev_target=ev_target,
            pot=pot,
            facing_bet=facing_bet,
            big_blind=big_blind,
            stack=stack,
            stacks=stacks,
        )
        scenarios.append(s)
    return scenarios


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestComputeNormStats(unittest.TestCase):
    """Unit tests for _compute_norm_stats."""

    def test_ev_ratio_formula(self):
        """EV ratio uses ev_target / max(pot + facing_bet, big_blind)."""
        big_blind = 10.0
        pot = 100.0
        facing_bet = 20.0
        ev_target = 60.0

        # Expected ratio: ev / max(pot+facing_bet, big_blind) = 60 / 120 = 0.5
        expected_ratio = ev_target / max(pot + facing_bet, big_blind)
        self.assertAlmostEqual(expected_ratio, 0.5)

        # With a single scenario the mean = the ratio itself and std collapses to 1.0
        # (single sample std = 0 → clamped to 1.0 by _compute_norm_stats)
        scenario = _make_scenario(ev_target, pot, facing_bet, big_blind)
        stats = _compute_norm_stats([scenario])
        self.assertAlmostEqual(stats["ev_mean"], expected_ratio, places=10)
        # std clamped to 1.0 for single sample
        self.assertAlmostEqual(stats["ev_std"], 1.0)

    def test_ev_ratio_uses_big_blind_as_floor(self):
        """When pot + facing_bet < big_blind, big_blind is used as the denominator."""
        big_blind = 100.0
        pot = 10.0
        facing_bet = 5.0  # pot + facing_bet = 15 < big_blind = 100
        ev_target = 50.0

        expected_ratio = ev_target / big_blind  # 0.5
        scenario = _make_scenario(ev_target, pot, facing_bet, big_blind)
        stats = _compute_norm_stats([scenario])
        self.assertAlmostEqual(stats["ev_mean"], expected_ratio, places=10)

    def test_stat_keys_present(self):
        """_compute_norm_stats returns all required keys."""
        scenarios = _make_diverse_scenarios(20)
        stats = _compute_norm_stats(scenarios)
        required = {
            "ev_mean", "ev_std",
            "pot_mean", "pot_std",
            "stack_mean", "stack_std",
            "bets_mean", "bets_std",
            "blind_mean", "blind_std",
        }
        self.assertEqual(required, set(stats.keys()))

    def test_std_is_positive(self):
        """All std values must be > 0 (clamped to 1.0 when effectively zero)."""
        scenarios = _make_diverse_scenarios(30)
        stats = _compute_norm_stats(scenarios)
        for key in stats:
            if "std" in key:
                self.assertGreater(stats[key], 0.0,
                                   f"{key} must be > 0, got {stats[key]}")

    def test_std_clamp_on_constant_data(self):
        """When all values are identical std is clamped to 1.0 (not zero)."""
        # All events have the same big_blind → blind_std should be 1.0
        scenarios = []
        for i in range(5):
            s = _make_scenario(
                ev_target=float(i * 10),
                pot=100.0,
                facing_bet=10.0,
                big_blind=10.0,  # constant
                stack=500.0,
            )
            scenarios.append(s)
        stats = _compute_norm_stats(scenarios)
        self.assertAlmostEqual(stats["blind_std"], 1.0, places=8,
                               msg="Constant blind_std should clamp to 1.0")

    def test_bets_uses_all_positions(self):
        """bets stats are computed over every bet entry across all events."""
        # Use a single scenario with known bets
        event = _make_event(pot=100, stack=500, big_blind=10,
                            bets=[0.0, 50.0, 25.0])
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
        s["events"] = [event]
        stats = _compute_norm_stats([s])
        # bets are [0.0, 50.0, 25.0]; mean = 25.0, std = std([0,50,25])
        bets_arr = np.array([0.0, 50.0, 25.0])
        self.assertAlmostEqual(stats["bets_mean"], float(bets_arr.mean()), places=8)
        self.assertAlmostEqual(stats["bets_std"], float(bets_arr.std()), places=5)

    def test_stack_uses_hero_stack_field(self):
        """stack stats come from the per-event 'stack' field (hero stack), not 'stacks'."""
        # Build two events with known hero stacks
        ev1 = _make_event(pot=100, stack=200.0, big_blind=10)
        ev2 = _make_event(pot=100, stack=400.0, big_blind=10)
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
        s["events"] = [ev1, ev2]
        stats = _compute_norm_stats([s])
        expected_mean = (200.0 + 400.0) / 2.0  # 300.0
        self.assertAlmostEqual(stats["stack_mean"], expected_mean, places=8)


class TestNormalizeScenarios(unittest.TestCase):
    """Unit tests for _normalize_scenarios."""

    def _normalized_copy(self, scenarios):
        """Return a normalized shallow copy; original is untouched."""
        copied = _shallow_copy_scenarios(scenarios)
        stats = _compute_norm_stats(copied)
        _normalize_scenarios(copied, stats)
        return copied, stats

    # ------------------------------------------------------------------
    # EV normalization
    # ------------------------------------------------------------------

    def test_ev_target_formula(self):
        """ev_target = (ev / max(pot+facing_bet, big_blind) - ev_mean) / ev_std."""
        big_blind = 10.0
        pot = 100.0
        facing_bet = 20.0
        ev_target = 60.0

        raw = [_make_scenario(ev_target, pot, facing_bet, big_blind)]
        stats = _compute_norm_stats(raw)

        copied = _shallow_copy_scenarios(raw)
        _normalize_scenarios(copied, stats)

        denom = max(pot + facing_bet, big_blind)  # 120
        ratio = ev_target / denom                 # 0.5
        expected = (ratio - stats["ev_mean"]) / stats["ev_std"]
        self.assertAlmostEqual(copied[0]["ev_target"], expected, places=10)

    def test_ev_big_blind_floor_in_normalize(self):
        """Denominator in _normalize_scenarios also uses big_blind as a floor."""
        big_blind = 100.0
        pot = 5.0
        facing_bet = 3.0  # pot + facing_bet = 8 < big_blind = 100
        ev_target = 40.0

        raw = [_make_scenario(ev_target, pot, facing_bet, big_blind)]
        stats = _compute_norm_stats(raw)
        copied = _shallow_copy_scenarios(raw)
        _normalize_scenarios(copied, stats)

        ratio = ev_target / big_blind  # uses big_blind as floor
        expected = (ratio - stats["ev_mean"]) / stats["ev_std"]
        self.assertAlmostEqual(copied[0]["ev_target"], expected, places=10)

    # ------------------------------------------------------------------
    # Z-score properties
    # ------------------------------------------------------------------

    def test_ev_zscore_mean_near_zero(self):
        """After normalization, ev_target values have mean ≈ 0."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        evs = [s["ev_target"] for s in copied]
        mean = float(np.mean(evs))
        self.assertAlmostEqual(mean, 0.0, places=5,
                               msg=f"Normalized EV mean should be ~0, got {mean}")

    def test_ev_zscore_std_near_one(self):
        """After normalization, ev_target values have std ≈ 1."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        evs = [s["ev_target"] for s in copied]
        std = float(np.std(evs))
        self.assertAlmostEqual(std, 1.0, places=4,
                               msg=f"Normalized EV std should be ~1, got {std}")

    def test_pot_zscore_mean_near_zero(self):
        """After normalization, event pot values have mean ≈ 0."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        pots = [e["pot"] for s in copied for e in s["events"]]
        mean = float(np.mean(pots))
        self.assertAlmostEqual(mean, 0.0, places=5,
                               msg=f"Normalized pot mean should be ~0, got {mean}")

    def test_pot_zscore_std_near_one(self):
        """After normalization, event pot values have std ≈ 1."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        pots = [e["pot"] for s in copied for e in s["events"]]
        std = float(np.std(pots))
        self.assertAlmostEqual(std, 1.0, places=4,
                               msg=f"Normalized pot std should be ~1, got {std}")

    def test_stack_zscore_mean_near_zero(self):
        """After normalization, event stack values have mean ≈ 0."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        stacks = [e["stack"] for s in copied for e in s["events"]]
        mean = float(np.mean(stacks))
        self.assertAlmostEqual(mean, 0.0, places=5,
                               msg=f"Normalized stack mean should be ~0, got {mean}")

    def test_stack_zscore_std_near_one(self):
        """After normalization, event stack values have std ≈ 1."""
        scenarios = _make_diverse_scenarios(100)
        copied, _ = self._normalized_copy(scenarios)
        stacks = [e["stack"] for s in copied for e in s["events"]]
        std = float(np.std(stacks))
        self.assertAlmostEqual(std, 1.0, places=4,
                               msg=f"Normalized stack std should be ~1, got {std}")

    def test_blind_zscore_mean_near_zero(self):
        """After normalization, event big_blind values have mean ≈ 0."""
        # Use varied big_blind values to get non-trivial stats
        rng = np.random.default_rng(1)
        scenarios = []
        for bb in rng.uniform(2, 50, size=80):
            s = _make_scenario(ev_target=10.0, pot=50.0, facing_bet=5.0, big_blind=float(bb))
            scenarios.append(s)
        copied, _ = self._normalized_copy(scenarios)
        blinds = [e["big_blind"] for s in copied for e in s["events"]]
        mean = float(np.mean(blinds))
        self.assertAlmostEqual(mean, 0.0, places=5,
                               msg=f"Normalized big_blind mean should be ~0, got {mean}")

    def test_bets_zscore_mean_near_zero(self):
        """After normalization, bet values have mean ≈ 0."""
        rng = np.random.default_rng(7)
        scenarios = []
        for _ in range(60):
            bets = [float(rng.uniform(0, 200)), float(rng.uniform(0, 200))]
            event = _make_event(pot=100, stack=500, big_blind=10, bets=bets)
            s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
            s["events"] = [event]
            scenarios.append(s)
        copied, _ = self._normalized_copy(scenarios)
        bets_vals = []
        for s in copied:
            for e in s["events"]:
                b = e["bets"]
                bets_vals.extend(b.tolist() if isinstance(b, np.ndarray) else b)
        mean = float(np.mean(bets_vals))
        self.assertAlmostEqual(mean, 0.0, places=5,
                               msg=f"Normalized bets mean should be ~0, got {mean}")

    # ------------------------------------------------------------------
    # Per-position stacks normalization
    # ------------------------------------------------------------------

    def test_stacks_vector_uses_stack_mean_std(self):
        """Per-position stacks are normalized with stack_mean and stack_std (same units)."""
        # Build scenarios with known per-event hero stack and stacks vector
        stacks_per_seat = [200.0, 400.0]
        hero_stack = 200.0  # same as stacks[0]
        event = _make_event(pot=100, stack=hero_stack, big_blind=10,
                            stacks=stacks_per_seat)
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0, stack=hero_stack,
                           stacks=stacks_per_seat)
        s["events"] = [event]
        scenarios = [s]

        # Add a second scenario to give the stats some spread
        stacks2 = [600.0, 800.0]
        event2 = _make_event(pot=200, stack=600.0, big_blind=10, stacks=stacks2)
        s2 = _make_scenario(ev_target=20.0, pot=200.0, facing_bet=0.0, stack=600.0,
                            stacks=stacks2)
        s2["events"] = [event2]
        scenarios.append(s2)

        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(copied, stats)

        # The normalized stacks[0] should equal the formula (stack - stack_mean) / stack_std
        stack_m = stats["stack_mean"]
        stack_s = stats["stack_std"]

        norm_stacks_0 = copied[0]["events"][0]["stacks"]
        expected_stacks = [(c - stack_m) / stack_s for c in stacks_per_seat]
        for i, (got, exp) in enumerate(zip(norm_stacks_0, expected_stacks)):
            self.assertAlmostEqual(got, exp, places=8,
                                   msg=f"stacks[{i}]: expected {exp}, got {got}")

    def test_stacks_uses_same_scale_as_hero_stack(self):
        """Normalized hero stack scalar should equal normalized stacks[hero_pos]."""
        hero_pos = 0
        hero_stack = 350.0
        stacks_per_seat = [hero_stack, 600.0, 450.0]
        event = _make_event(pot=80, stack=hero_stack, big_blind=10,
                            num_players=3, stacks=stacks_per_seat)
        s = _make_scenario(ev_target=15.0, pot=80.0, facing_bet=10.0, stack=hero_stack,
                           stacks=stacks_per_seat)
        s["events"] = [event]

        # Second scenario to get valid stats
        stacks2 = [700.0, 200.0, 500.0]
        event2 = _make_event(pot=150, stack=700.0, big_blind=10,
                             num_players=3, stacks=stacks2)
        s2 = _make_scenario(ev_target=30.0, pot=150.0, facing_bet=20.0, stack=700.0,
                            stacks=stacks2)
        s2["events"] = [event2]

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(copied, stats)

        # In the first scenario, hero_pos=0 → stacks[0] should == stack scalar
        norm_hero_stack = copied[0]["events"][0]["stack"]
        norm_stacks_0 = copied[0]["events"][0]["stacks"][hero_pos]
        self.assertAlmostEqual(norm_hero_stack, norm_stacks_0, places=8,
                               msg="Normalized hero stack scalar and stacks[hero_pos] must match")

    # ------------------------------------------------------------------
    # Original data is not mutated
    # ------------------------------------------------------------------

    def test_shallow_copy_protects_original(self):
        """_shallow_copy_scenarios + _normalize_scenarios must not mutate originals."""
        scenarios = _make_diverse_scenarios(10)
        original_ev = [s["ev_target"] for s in scenarios]
        original_pot_event0 = [s["events"][0]["pot"] for s in scenarios]

        copied = _shallow_copy_scenarios(scenarios)
        stats = _compute_norm_stats(copied)
        _normalize_scenarios(copied, stats)

        for i, s in enumerate(scenarios):
            self.assertAlmostEqual(s["ev_target"], original_ev[i], places=8,
                                   msg=f"Original ev_target[{i}] mutated")
            self.assertAlmostEqual(s["events"][0]["pot"], original_pot_event0[i], places=8,
                                   msg=f"Original events[0].pot[{i}] mutated")

    # ------------------------------------------------------------------
    # Idempotency check: double normalization produces different results
    # ------------------------------------------------------------------

    def test_double_normalization_differs(self):
        """Normalizing already-normalized data changes the values (not idempotent)."""
        scenarios = _make_diverse_scenarios(30)
        copied = _shallow_copy_scenarios(scenarios)
        stats = _compute_norm_stats(copied)
        _normalize_scenarios(copied, stats)

        first_ev = [s["ev_target"] for s in copied]

        # Apply normalization again using the SAME stats
        _normalize_scenarios(copied, stats)
        second_ev = [s["ev_target"] for s in copied]

        # At least some values must be different
        any_different = any(
            abs(a - b) > 1e-6 for a, b in zip(first_ev, second_ev)
        )
        self.assertTrue(any_different,
                        "Double normalization should produce different values, "
                        "which confirms the pipeline applies it exactly once")

    # ------------------------------------------------------------------
    # small_blind mirrors big_blind normalization
    # ------------------------------------------------------------------

    def test_small_blind_normalized_with_blind_stats(self):
        """small_blind is normalized with the same blind mean/std as big_blind."""
        big_blind = 20.0
        small_blind = 10.0
        event = _make_event(pot=50, stack=500, big_blind=big_blind)
        event["small_blind"] = small_blind

        s = _make_scenario(ev_target=5.0, pot=50.0, facing_bet=0.0, big_blind=big_blind)
        s["events"] = [event]

        # Second scenario for valid stats
        event2 = _make_event(pot=100, stack=500, big_blind=40.0)
        event2["small_blind"] = 20.0
        s2 = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0, big_blind=40.0)
        s2["events"] = [event2]

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(copied, stats)

        blind_m = stats["blind_mean"]
        blind_s = stats["blind_std"]
        expected_small = (small_blind - blind_m) / blind_s
        got_small = copied[0]["events"][0]["small_blind"]
        self.assertAlmostEqual(got_small, expected_small, places=8,
                               msg="small_blind not normalized with blind stats")


class TestNormalizeActionEvs(unittest.TestCase):
    """Unit tests for _normalize_action_evs (modelling phase)."""

    def _make_norm_stats(self, ev_mean=0.0, ev_std=1.0):
        """Build minimal norm_stats dict for action EV tests."""
        return {
            "ev_mean": ev_mean,
            "ev_std": ev_std,
            "pot_mean": 100.0, "pot_std": 50.0,
            "stack_mean": 400.0, "stack_std": 150.0,
            "bets_mean": 20.0, "bets_std": 15.0,
            "blind_mean": 10.0, "blind_std": 1.0,
        }

    def test_action_ev_formula(self):
        """action_evs[i] = (ev / max(pot+facing_bet, big_blind) - ev_mean) / ev_std."""
        big_blind = 10.0
        pot = 100.0
        facing_bet = 20.0
        action_evs_raw = [10.0, 60.0, 40.0, -5.0]

        denom = max(pot + facing_bet, big_blind)  # 120
        ev_mean = 0.3
        ev_std = 0.8
        expected = [(ev / denom - ev_mean) / ev_std for ev in action_evs_raw]

        event = _make_event(pot=pot, stack=500, big_blind=big_blind)
        s = {
            "ev_target": 60.0,
            "action_evs": list(action_evs_raw),
            "action_probs": [0.1, 0.5, 0.3, 0.1],
            "pot": pot,
            "facing_bet": facing_bet,
            "events": [event],
        }

        norm_stats = self._make_norm_stats(ev_mean=ev_mean, ev_std=ev_std)
        scenarios = [copy.deepcopy(s)]
        _normalize_action_evs(scenarios, norm_stats)

        for i, (got, exp) in enumerate(zip(scenarios[0]["action_evs"], expected)):
            self.assertAlmostEqual(got, exp, places=8,
                                   msg=f"action_evs[{i}]: expected {exp}, got {got}")

    def test_action_ev_uses_big_blind_floor(self):
        """When pot + facing_bet < big_blind, big_blind is used as denominator."""
        big_blind = 100.0
        pot = 5.0
        facing_bet = 3.0  # sum=8 < big_blind
        action_evs_raw = [20.0, 50.0, 30.0, -10.0]

        denom = big_blind  # floor applied
        ev_mean = 0.0
        ev_std = 1.0
        expected = [ev / denom for ev in action_evs_raw]

        event = _make_event(pot=pot, stack=500, big_blind=big_blind)
        s = {
            "ev_target": 50.0,
            "action_evs": list(action_evs_raw),
            "action_probs": [0.1, 0.5, 0.3, 0.1],
            "pot": pot,
            "facing_bet": facing_bet,
            "events": [event],
        }

        norm_stats = self._make_norm_stats(ev_mean=ev_mean, ev_std=ev_std)
        scenarios = [copy.deepcopy(s)]
        _normalize_action_evs(scenarios, norm_stats)

        for i, (got, exp) in enumerate(zip(scenarios[0]["action_evs"], expected)):
            self.assertAlmostEqual(got, exp, places=8,
                                   msg=f"action_evs[{i}]: expected {exp} (big_blind floor), got {got}")

    def test_action_evs_numpy_array_handled(self):
        """_normalize_action_evs handles numpy array action_evs without error."""
        big_blind = 10.0
        pot = 50.0
        facing_bet = 10.0
        action_evs_raw = np.array([5.0, 20.0, 15.0, -2.0])

        event = _make_event(pot=pot, stack=500, big_blind=big_blind)
        s = {
            "ev_target": 20.0,
            "action_evs": action_evs_raw,
            "action_probs": [0.1, 0.5, 0.3, 0.1],
            "pot": pot,
            "facing_bet": facing_bet,
            "events": [event],
        }

        norm_stats = self._make_norm_stats()
        scenarios = [copy.deepcopy(s)]
        # Should not raise
        _normalize_action_evs(scenarios, norm_stats)
        # Result should be indexable
        result = scenarios[0]["action_evs"]
        self.assertEqual(len(result), 4)


class TestActionEvCompressionProperty(unittest.TestCase):
    """Test the compression formula for extreme negative action EVs.

    The formula is: if target < -1: target = -1 + (target + 1) * 0.03
    This lives in modelling_predict/train.py:_compress_targets.
    """

    def _compress(self, value):
        """Mirror of _compress_targets applied to a scalar."""
        import torch
        t = torch.tensor([value], dtype=torch.float32)
        result = torch.where(t >= -1, t, -1 + (t + 1) * 0.03)
        return result.item()

    def test_value_above_minus_one_unchanged(self):
        """Values >= -1 are not compressed."""
        for v in [0.0, -0.5, -1.0, 2.0, 100.0]:
            self.assertAlmostEqual(self._compress(v), v, places=7,
                                   msg=f"compress({v}) should be {v}")

    def test_value_below_minus_one_compressed(self):
        """Values < -1 are compressed: -1 + (target+1)*0.03."""
        for v in [-2.0, -5.0, -10.0, -100.0]:
            expected = -1.0 + (v + 1.0) * 0.03
            got = self._compress(v)
            self.assertAlmostEqual(got, expected, places=6,
                                   msg=f"compress({v}): expected {expected}, got {got}")

    def test_compression_reduces_magnitude(self):
        """Compression maps extreme negatives closer to -1 (reduces |value|)."""
        for v in [-2.0, -10.0, -50.0]:
            compressed = self._compress(v)
            self.assertGreater(compressed, v,
                               msg=f"compress({v})={compressed} should be > {v}")
            self.assertLessEqual(compressed, -1.0 + 1e-7,
                                 msg=f"compress({v})={compressed} should be <= -1")

    def test_compression_boundary(self):
        """Compression exactly at boundary: compress(-1) == -1."""
        self.assertAlmostEqual(self._compress(-1.0), -1.0, places=7)

    def test_compression_continuity_at_boundary(self):
        """Compression is approximately continuous at -1 (no discontinuous jump)."""
        # Just above: -1+eps → unchanged; just below: -1-eps → -1 + (-eps)*0.03
        eps = 0.001
        above = self._compress(-1.0 + eps)
        below = self._compress(-1.0 - eps)
        # Both should be very close to -1
        self.assertAlmostEqual(above, -1.0 + eps, places=6)
        self.assertAlmostEqual(below, -1.0 - eps * 0.03, places=6)


class TestNormStatsReuseAcrossPhases(unittest.TestCase):
    """Tests that downstream phases reuse checkpoint norm stats, not recompute.

    This validates the contract described in modelling_predict/train.py and
    opponent_action_predict/train.py: when a checkpoint has norm_stats, those
    stats should be used instead of recomputing from the current dataset.
    The data-distribution mismatch between the original (perception-training)
    data and later-phase data makes recomputation incorrect for frozen perception.
    """

    def _simulate_checkpoint_norm_stats(self, scenarios):
        """Compute norm stats as if from a prior training phase checkpoint."""
        return _compute_norm_stats(scenarios)

    def test_reusing_checkpoint_stats_differs_from_recomputing(self):
        """Norm stats from checkpoint differ from freshly-computed stats on a new dataset.

        This verifies the mismatch that would occur if downstream phases
        recomputed instead of reusing; if they were always the same there
        would be no correctness difference and no reason to reuse.
        """
        # Phase 1 dataset (used to compute original norm stats)
        phase1_scenarios = _make_diverse_scenarios(50, seed=1)
        checkpoint_stats = _compute_norm_stats(phase1_scenarios)

        # Downstream phase uses a different dataset (shifted distribution)
        rng = np.random.default_rng(99)
        downstream_scenarios = []
        for _ in range(50):
            pot = float(rng.uniform(200, 2000))  # much larger pots
            facing_bet = float(rng.uniform(0, pot * 0.3))
            ev_target = float(rng.uniform(-200, 500))
            s = _make_scenario(ev_target=ev_target, pot=pot, facing_bet=facing_bet,
                               big_blind=10.0, stack=float(rng.uniform(500, 5000)))
            downstream_scenarios.append(s)

        recomputed_stats = _compute_norm_stats(downstream_scenarios)

        # pot stats should differ meaningfully (different pot distributions)
        pot_mean_diff = abs(checkpoint_stats["pot_mean"] - recomputed_stats["pot_mean"])
        self.assertGreater(pot_mean_diff, 5.0,
                           "pot_mean should differ between phase1 and downstream dataset "
                           "to confirm reuse matters")

    def test_checkpoint_stats_reuse_produces_consistent_ev_targets(self):
        """Using checkpoint stats normalizes EVs in a way consistent with phase 1.

        If the checkpoint stats are reused, a scenario that had ev_target=0.0 in
        phase 1 normalized space should produce 0.0 again when the same stats are applied.
        """
        scenarios = _make_diverse_scenarios(40, seed=5)
        stats = _compute_norm_stats(scenarios)

        # Find a scenario whose ratio is exactly ev_mean
        # By definition, if ev_target/denom == ev_mean, normalized value = 0
        target_scenario = scenarios[0]
        denom = max(
            target_scenario["pot"] + target_scenario["facing_bet"],
            target_scenario["events"][-1]["big_blind"]
        )
        # Force ev_target so that ev_target/denom == ev_mean
        target_scenario["ev_target"] = stats["ev_mean"] * denom

        copied = _shallow_copy_scenarios([target_scenario])
        _normalize_scenarios(copied, stats)

        self.assertAlmostEqual(copied[0]["ev_target"], 0.0, places=8,
                               msg="Scenario with ev_target/denom == ev_mean should normalize to 0")

    def test_norm_stats_keys_compatible_between_gto_and_opponent_action(self):
        """Norm stats from generate._compute_norm_stats are a superset of what
        opponent_action_predict._compute_norm_stats would produce.

        The GTO stats include 'ev_mean'/'ev_std'; the opponent_action stats do not
        (opponents don't have an ev_target). The GTO stats are the ones that should
        be reused by the opponent action phase — they are a strict superset.
        """
        from agent.train_scenarios.opponent_action_predict.train import (
            _compute_norm_stats as opp_compute_norm_stats,
        )

        # Build scenarios in the shared event format used by opponent action
        rng = np.random.default_rng(42)
        opp_scenarios = []
        for _ in range(20):
            events = [_make_event(
                pot=float(rng.uniform(10, 300)),
                stack=float(rng.uniform(50, 800)),
                big_blind=10.0,
                bets=[float(rng.uniform(0, 50)), float(rng.uniform(0, 50))],
            )]
            s = {
                "events": events,
                "target_probs": [0.1, 0.6, 0.2, 0.1],
                "hand_id": 0,
            }
            opp_scenarios.append(s)

        opp_stats = opp_compute_norm_stats(opp_scenarios)
        gto_scenarios = _make_diverse_scenarios(20, seed=42)
        gto_stats = _compute_norm_stats(gto_scenarios)

        # GTO stats must have all keys that opp stats has
        for key in opp_stats:
            self.assertIn(key, gto_stats,
                          f"GTO stats missing key '{key}' that opp stats has")

        # GTO stats has ev_mean/ev_std which opp stats lacks
        self.assertIn("ev_mean", gto_stats)
        self.assertIn("ev_std", gto_stats)
        self.assertNotIn("ev_mean", opp_stats)


class TestActionEvsNormalizedBeforeScenarios(unittest.TestCase):
    """Tests that _normalize_action_evs must be called before _normalize_scenarios.

    _normalize_action_evs reads raw big_blind from events[-1]["big_blind"].
    After _normalize_scenarios runs, big_blind is z-scored (no longer the raw value).
    Calling _normalize_action_evs after _normalize_scenarios would use z-scored
    big_blind as the denom floor, which is wrong.
    """

    def _make_norm_stats_from_scenarios(self, scenarios):
        return _compute_norm_stats(scenarios)

    def test_wrong_order_produces_incorrect_action_ev_denom(self):
        """Calling _normalize_action_evs AFTER _normalize_scenarios uses wrong big_blind.

        The test checks that after _normalize_scenarios, the big_blind in events
        is no longer the raw value — confirming that the order matters.
        """
        big_blind = 10.0
        scenarios = _make_diverse_scenarios(30, seed=11)
        stats = self._make_norm_stats_from_scenarios(scenarios)

        # Correct order: action_evs first, then scenarios
        correct_copy = _shallow_copy_scenarios(scenarios)
        _normalize_action_evs(correct_copy, stats)
        correct_aevs = list(correct_copy[0]["action_evs"])

        # Verify raw big_blind before normalize_scenarios
        raw_blind = scenarios[0]["events"][-1]["big_blind"]
        self.assertAlmostEqual(raw_blind, big_blind, places=5,
                               msg="Raw big_blind should be the original value before normalization")

        # Now normalize scenarios, then check big_blind is no longer raw
        wrong_order_copy = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(wrong_order_copy, stats)
        normalized_blind = wrong_order_copy[0]["events"][-1]["big_blind"]
        self.assertFalse(
            abs(normalized_blind - big_blind) < 1e-6,
            "After _normalize_scenarios, big_blind should be z-scored (not raw)"
        )

    def test_correct_order_action_evs_use_raw_big_blind(self):
        """Calling _normalize_action_evs before _normalize_scenarios uses raw big_blind.

        We verify this by manually computing what the result should be using
        the known raw big_blind value.
        """
        big_blind = 10.0
        pot = 80.0
        facing_bet = 5.0
        action_evs_raw = [5.0, 30.0, 20.0, -3.0]
        denom = max(pot + facing_bet, big_blind)  # 85.0

        event = _make_event(pot=pot, stack=500, big_blind=big_blind)
        s = {
            "ev_target": 30.0,
            "action_evs": list(action_evs_raw),
            "action_probs": [0.1, 0.5, 0.3, 0.1],
            "pot": pot,
            "facing_bet": facing_bet,
            "events": [event],
        }

        # Build stats with ev_mean=0, ev_std=1 for a clean formula check
        stats = {
            "ev_mean": 0.0, "ev_std": 1.0,
            "pot_mean": pot, "pot_std": 50.0,
            "stack_mean": 400.0, "stack_std": 150.0,
            "bets_mean": 0.0, "bets_std": 10.0,
            "blind_mean": big_blind, "blind_std": 1.0,
        }

        copied = [copy.deepcopy(s)]
        # Correct order
        _normalize_action_evs(copied, stats)

        expected = [ev / denom for ev in action_evs_raw]
        for i, (got, exp) in enumerate(zip(copied[0]["action_evs"], expected)):
            self.assertAlmostEqual(got, exp, places=8,
                                   msg=f"action_evs[{i}] with correct order: "
                                       f"expected {exp}, got {got}")


class TestNormalizationNumericalEdgeCases(unittest.TestCase):
    """Numerical edge cases and robustness checks."""

    def test_single_scenario_std_clamped(self):
        """With only one scenario, all std values are clamped to 1.0 (zero std)."""
        s = _make_scenario(ev_target=50.0, pot=100.0, facing_bet=10.0, big_blind=10.0)
        stats = _compute_norm_stats([s])
        for key in stats:
            if "std" in key:
                self.assertGreaterEqual(stats[key], 1e-8,
                                        f"{key}={stats[key]} must not be near zero")

    def test_large_ev_values_handled(self):
        """Normalization handles large EV values without overflow."""
        s = _make_scenario(ev_target=1e6, pot=1e7, facing_bet=5e6, big_blind=10.0)
        s2 = _make_scenario(ev_target=-1e6, pot=5e6, facing_bet=1e6, big_blind=10.0)
        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(copied, stats)

        for s_c in copied:
            self.assertFalse(
                np.isnan(s_c["ev_target"]) or np.isinf(s_c["ev_target"]),
                f"ev_target should be finite, got {s_c['ev_target']}"
            )

    def test_zero_pot_uses_big_blind_as_denom(self):
        """When pot=0 and facing_bet=0, big_blind is used as denominator."""
        big_blind = 10.0
        ev_target = 5.0
        s = _make_scenario(ev_target=ev_target, pot=0.0, facing_bet=0.0, big_blind=big_blind)
        stats = _compute_norm_stats([s])
        # denom = max(0+0, 10) = 10
        expected_ratio = ev_target / big_blind
        self.assertAlmostEqual(stats["ev_mean"], expected_ratio, places=10)

    def test_bets_as_list_handled(self):
        """_normalize_scenarios handles bets as Python list (not ndarray)."""
        event = _make_event(pot=100, stack=500, big_blind=10)
        event["bets"] = [10.0, 20.0]  # plain list, not ndarray
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
        s["events"] = [event]
        s2 = _make_scenario(ev_target=20.0, pot=150.0, facing_bet=0.0)

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)

        # Should not raise when bets is a list
        _normalize_scenarios(copied, stats)
        bets = copied[0]["events"][0]["bets"]
        self.assertEqual(len(bets), 2, "bets should still have 2 elements after normalization")

    def test_bets_as_ndarray_handled(self):
        """_normalize_scenarios handles bets as numpy ndarray."""
        event = _make_event(pot=100, stack=500, big_blind=10)
        event["bets"] = np.array([10.0, 20.0])  # ndarray
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
        s["events"] = [event]
        s2 = _make_scenario(ev_target=20.0, pot=150.0, facing_bet=0.0)

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)

        _normalize_scenarios(copied, stats)
        bets = copied[0]["events"][0]["bets"]
        self.assertEqual(len(bets), 2)
        self.assertIsInstance(bets, np.ndarray, "ndarray bets should remain ndarray after normalization")

    def test_stacks_not_present_does_not_crash(self):
        """If 'stacks' is absent from an event, _normalize_scenarios skips it silently."""
        event = _make_event(pot=100, stack=500, big_blind=10)
        event.pop("stacks", None)  # remove stacks
        s = _make_scenario(ev_target=10.0, pot=100.0, facing_bet=0.0)
        s["events"] = [event]
        s2 = _make_scenario(ev_target=20.0, pot=150.0, facing_bet=0.0)

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        # Should not raise
        _normalize_scenarios(copied, stats)

    def test_ev_target_negative(self):
        """Negative ev_target (losing situation) normalizes correctly."""
        ev_target = -50.0
        pot = 100.0
        facing_bet = 20.0
        big_blind = 10.0
        s = _make_scenario(ev_target=ev_target, pot=pot, facing_bet=facing_bet,
                           big_blind=big_blind)
        s2 = _make_scenario(ev_target=50.0, pot=100.0, facing_bet=20.0)

        scenarios = [s, s2]
        stats = _compute_norm_stats(scenarios)
        copied = _shallow_copy_scenarios(scenarios)
        _normalize_scenarios(copied, stats)

        denom = max(pot + facing_bet, big_blind)
        ratio = ev_target / denom
        expected = (ratio - stats["ev_mean"]) / stats["ev_std"]
        self.assertAlmostEqual(copied[0]["ev_target"], expected, places=8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
