"""Observer-blocker target tests (Audit B.7.2) + per-combo data plumbing.

B.7.2: the opponent-action target must exclude combos blocked by the OBSERVER's
own cards. Scenarios now persist per-combo distributions + weights + combo card
pairs; the dataset loader re-averages per observer.

  - re-averaging over ALL combos reproduces the stored un-blocked average;
  - blocking an observer's card drops the combos containing it;
  - if every combo is blocked, falls back to the stored average.

Also checks generation persists the per-combo fields and the loader emits valid
targets (the B.7.1 likelihood-floor belief update is exercised by generation).

Run (from versions/v6):
    python -m tests.test_opponent_blockers
"""

import torch

from agent.train_scenarios.opponent_action_predict.dataset import OpponentActionDataset


def _toy_scenario():
    return {
        "events": [{  # minimal shared event (only fields the loader touches)
            "hands": {1: [0, 4]}, "num_players": 2, "acting_pos": 0,
            "big_blind": 10.0, "small_blind": 5.0, "stacks": [200.0, 200.0],
            "table": [-1, -1, -1, -1, -1], "pot": 30.0,
            "bets": [5.0, 10.0], "action": [0.0, 1.0, 0.0],
        }],
        "hero_positions": [1],
        "opponent_action_probs": [0.48, 0.44, 0.08],
        "forward_combos": [[0, 1], [2, 3], [0, 4]],
        "per_combo_probs": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.5, 0.5, 0.0]],
        "forward_weights": [0.5, 0.3, 0.2],
    }


def test_observer_target_reaverage():
    ds = OpponentActionDataset([_toy_scenario()], norm_stats=None)
    sc = ds.scenarios[0]

    # No blockers → reproduces the stored un-blocked average exactly.
    t0 = ds._observer_target(sc, [6, 7])
    assert torch.allclose(t0, torch.tensor([0.48, 0.44, 0.08]), atol=1e-5), t0

    # Block card 0 → combos [0,1] and [0,4] drop; only [2,3] survives.
    t1 = ds._observer_target(sc, [0, 9])
    assert torch.allclose(t1, torch.tensor([0.1, 0.8, 0.1]), atol=1e-5), t1
    assert not torch.allclose(t1, t0), "blocker had no effect"

    # Block cards 0 and 2 → every combo blocked → fall back to stored average.
    t2 = ds._observer_target(sc, [0, 2])
    assert torch.allclose(t2, torch.tensor([0.48, 0.44, 0.08]), atol=1e-5), t2

    # Legacy scenario without per-combo data → stored average.
    legacy = {"opponent_action_probs": [0.2, 0.3, 0.5]}
    t3 = ds._observer_target(legacy, [0, 1])
    assert torch.allclose(t3, torch.tensor([0.2, 0.3, 0.5]), atol=1e-6)
    print("test_observer_target_reaverage: OK")


def test_generation_persists_per_combo_and_valid_targets():
    """Generation plumbing: per-combo fields present, consistent, and the full
    re-average (all combos) reproduces the stored un-blocked average.

    (The stub agent returns uniform per-combo distributions, so blocking can't
    shift a uniform mean — the blocker SHIFT itself is proven against
    non-uniform data in test_observer_target_reaverage.)
    """
    from tests.test_opponent_data_collisions import _generate
    scenarios = _generate(n_hands=120, seed=9)
    assert len(scenarios) > 30
    n_checked = 0
    for s in scenarios:
        assert "forward_combos" in s and "per_combo_probs" in s and "forward_weights" in s
        K = len(s["forward_combos"])
        assert len(s["per_combo_probs"]) == K and len(s["forward_weights"]) == K
        assert K >= 1
        # Full re-average (keep every combo) must reproduce the stored average,
        # i.e. the persisted per-combo data is consistent with opponent_action_probs.
        pcp = torch.tensor(s["per_combo_probs"], dtype=torch.float32)
        w = torch.tensor(s["forward_weights"], dtype=torch.float32)
        w = w / w.sum().clamp(min=1e-9)
        full = (pcp * w.unsqueeze(-1)).sum(dim=0)
        full = full / full.sum().clamp(min=1e-9)
        shared = torch.tensor(s["opponent_action_probs"], dtype=torch.float32)
        assert torch.allclose(full, shared, atol=1e-4), (
            f"per-combo data inconsistent with stored average: {full} vs {shared}")
        n_checked += 1

    ds = OpponentActionDataset(scenarios, norm_stats=None)
    n_targets = 0
    for i in range(min(len(ds), 300)):
        events, target, _aux = ds[i]
        assert abs(float(target.sum()) - 1.0) < 1e-4, f"target must sum to 1: {target.sum()}"
        assert torch.all(target >= -1e-6) and not torch.any(torch.isnan(target))
        n_targets += 1
    assert n_checked > 30 and n_targets > 30
    print(f"test_generation_persists_per_combo_and_valid_targets: OK "
          f"(scenarios={n_checked}, targets={n_targets})")


if __name__ == "__main__":
    test_observer_target_reaverage()
    test_generation_persists_per_combo_and_valid_targets()
    print("\nALL B.7 OBSERVER-BLOCKER TESTS PASSED")
