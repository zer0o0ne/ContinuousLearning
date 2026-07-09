"""Opponent-data card-collision tests (Audit B.4).

Opponent-data hands are SAMPLED from belief ranges while the board is the
fixed deck[:5], revealed incrementally. Without care this yields physically
impossible hands:
  - a fixed hand sharing a card with another player's fixed hand (B.4.1);
  - a hand fixed on an early street colliding with a later board card (B.4.2);
  - an observer kept in hero_positions while their fixed hand is on the board
    (B.4.3) → an impossible per-observer training example.

This drives the real generator with a stub agent (uniform logits — no
checkpoints needed) over many hands and asserts:
  1. fixed hands in every scenario are pairwise disjoint;
  2. every surviving observer's fixed hand is disjoint from the revealed board;
  3. every emitted dataset sample's hero hand is disjoint from the board
     (covers the no-fixed-hand observer path, which samples a random hand).

Run (from versions/v6):
    python -m tests.test_opponent_data_collisions
"""

import random

import numpy as np
import torch

from agent.train_scenarios.generation.generate_opponent import generate_opponent_hand
from agent.train_scenarios.opponent_action_predict.dataset import OpponentActionDataset

N_ACTIONS = 6  # 3 raise bins + fold/call/allin
_RS = [0.5, 1.0, 2.0]


class _StubAgent:
    """Returns uniform action logits — exercises generation without a model."""

    def forward_batch(self, batch_events, skip_memory=True, heads=None):
        return {"action_logits": torch.zeros(len(batch_events), N_ACTIONS)}


def _dummy_norm_stats():
    return {
        "pot_mean": 0.0, "pot_std": 1.0,
        "stack_mean": 0.0, "stack_std": 1.0,
        "bets_mean": 0.0, "bets_std": 1.0,
        "blind_mean": 0.0, "blind_std": 1.0,
    }


def _config():
    return {
        "big_blind": 10, "max_stack": 300, "min_stack": 120, "max_players": 5,
        "max_batch_combos": 256,
        "raise_sizes": {s: list(_RS) for s in ("preflop", "flop", "turn", "river")},
        "bayes": {"enabled": True, "tau_belief": 2.0, "ess_truncation_mass": 0.9},
    }


def _revealed_board(event):
    return {int(c) for c in event["table"] if int(c) >= 0}


def _generate(n_hands, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agents_list = [{
        "agent": _StubAgent(), "name": "stub",
        "norm_stats": _dummy_norm_stats(), "temperature": 1.0,
    }]
    amp_config = (False, "cpu", torch.float32)
    cfg = _config()
    all_scenarios = []
    for _ in range(n_hands):
        res = generate_opponent_hand(cfg, agents_list, "cpu", amp_config)
        if res:
            all_scenarios.extend(res)
    return all_scenarios


def test_no_card_collisions_in_scenarios():
    scenarios = _generate(n_hands=400, seed=7)
    assert len(scenarios) > 50, f"too few scenarios generated: {len(scenarios)}"

    n_pair_checks = 0
    n_observer_checks = 0
    multi_street = 0
    for s in scenarios:
        last = s["events"][-1]
        board = _revealed_board(last)
        if len(board) >= 3:
            multi_street += 1
        hands = last["hands"]  # {pos: [c1, c2]}

        # (1) fixed hands pairwise disjoint
        items = list(hands.items())
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ca, cb = set(items[a][1]), set(items[b][1])
                assert ca.isdisjoint(cb), (
                    f"fixed hands collide: pos {items[a][0]}={ca} pos {items[b][0]}={cb}"
                )
                n_pair_checks += 1

        # (2) surviving observers' fixed hands disjoint from the revealed board
        for hero_pos in s["hero_positions"]:
            if hero_pos in hands:
                hcards = set(hands[hero_pos])
                assert hcards.isdisjoint(board), (
                    f"observer {hero_pos} hand {hcards} collides with board {board}"
                )
                n_observer_checks += 1

    assert multi_street > 10, f"test mostly preflop ({multi_street} multi-street) — weak"
    assert n_observer_checks > 20, f"observer board-checks vacuous ({n_observer_checks})"
    print(f"test_no_card_collisions_in_scenarios: OK "
          f"(scenarios={len(scenarios)}, multi_street={multi_street}, "
          f"pair_checks={n_pair_checks}, observer_checks={n_observer_checks})")


def test_dataset_samples_board_consistent():
    """Every emitted (scenario, hero) sample's hero hand is board-disjoint."""
    scenarios = _generate(n_hands=300, seed=11)
    random.seed(123)  # deterministic random-hand sampling in _to_standard
    ds = OpponentActionDataset(scenarios, norm_stats=None)
    assert len(ds) > 50
    n_checked = 0
    for i in range(len(ds)):
        events, _target, _aux = ds[i]
        hero_hand = set(events[0]["hand"])
        board = _revealed_board(events[-1])
        assert hero_hand.isdisjoint(board), (
            f"sample {i}: hero hand {hero_hand} collides with board {board}"
        )
        assert len(hero_hand) == 2 and len(set(events[0]["hand"])) == 2
        n_checked += 1
    print(f"test_dataset_samples_board_consistent: OK ({n_checked} samples)")


if __name__ == "__main__":
    test_no_card_collisions_in_scenarios()
    test_dataset_samples_board_consistent()
    print("\nALL B.4 OPPONENT-DATA COLLISION TESTS PASSED")
