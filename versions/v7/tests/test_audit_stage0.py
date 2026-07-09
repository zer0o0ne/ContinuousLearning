"""Stage 0 audit tests: engine chip conservation + evaluation.

Covers checklist items 0.1 (showdown preserves chips) and 0.2 (no leak correction).
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.table import Table


def _make_table(num_players=2, start_credits=1000, big_blind=10, small_blind=5):
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    return Table(num_players, raise_sizes, start_credits=start_credits,
                 big_blind=big_blind, small_blind=small_blind)


def _action_tensor(idx, n_actions=5):
    t = torch.zeros(n_actions)
    t[idx] = 1.0
    return t


def _play_hand_with_actions(table, action_sequence):
    """Play a hand with predetermined actions, return credits after."""
    table.start_table()
    original_total = sum(table.start_credits)
    for action_idx in action_sequence:
        end, _, _, _ = table.step(_action_tensor(action_idx, n_actions=table.n_raise_bins + 3))
        if end:
            break
    return original_total, table.credits.copy()


# ── 0.1: Chip conservation ──────────────────────────────────────────────────

def test_chip_conservation_fold():
    """Fold-terminated hand: total chips must be conserved."""
    table = _make_table()
    original_total, credits_after = _play_hand_with_actions(table, [0])
    assert abs(sum(credits_after) - original_total) < 1e-6, (
        f"Chip leak: {sum(credits_after)} != {original_total}")


def test_chip_conservation_call_call_through_streets():
    """Check-down through all streets: chips conserved at showdown."""
    table = _make_table()
    table.start_table()
    original_total = sum(table.start_credits)
    end = False
    safety = 0
    while not end and safety < 100:
        end, _, _, _ = table.step(_action_tensor(1, n_actions=table.n_raise_bins + 3))
        safety += 1
    assert abs(sum(table.credits) - original_total) < 1e-6, (
        f"Chip leak: before={original_total}, after={sum(table.credits)}")


def test_chip_conservation_allin_showdown():
    """All-in on preflop: total chips conserved after showdown."""
    table = _make_table()
    table.start_table()
    original_total = sum(table.start_credits)
    n_act = table.n_raise_bins + 3
    allin_idx = n_act - 1
    end = False
    safety = 0
    while not end and safety < 100:
        end, _, _, _ = table.step(_action_tensor(allin_idx, n_actions=n_act))
        safety += 1
    assert abs(sum(table.credits) - original_total) < 1e-6, (
        f"Chip leak: before={original_total}, after={sum(table.credits)}")


def test_chip_conservation_winner_gets_pot():
    """Fold on first action: non-folder should gain the blind pot."""
    table = _make_table(start_credits=1000, big_blind=10, small_blind=5)
    table.start_table()
    n_act = table.n_raise_bins + 3
    # HU: active_player = 0 (SB). Fold → BB (player 1) wins pot.
    end, _, _, _ = table.step(_action_tensor(0, n_actions=n_act))
    assert end, "Hand should end after HU fold"
    assert table.credits[1] > 1000 - 10, "BB should get the pot"
    assert abs(sum(table.credits) - 2000.0) < 1e-6


def test_chip_conservation_random_hands():
    """Run 50 random hands, each must conserve chips."""
    table = _make_table(num_players=3, start_credits=500)
    n_act = table.n_raise_bins + 3
    np.random.seed(42)
    original_total = sum(table.start_credits)
    for _ in range(50):
        table.start_table()
        end = False
        safety = 0
        while not end and safety < 200:
            idx = np.random.choice([0, 1, n_act - 1], p=[0.2, 0.5, 0.3])
            end, _, _, _ = table.step(_action_tensor(idx, n_actions=n_act))
            safety += 1
        assert abs(sum(table.credits) - original_total) < 1e-6, (
            f"Chip leak on random hand: {sum(table.credits)} != {original_total}")


# ── 0.1: cumulative_bets exists and tracks correctly ─────────────────────────

def test_cumulative_bets_exists():
    """Table must have cumulative_bets that accumulates across streets."""
    table = _make_table()
    table.start_table()
    assert hasattr(table, "cumulative_bets"), "Table missing cumulative_bets"
    assert table.cumulative_bets[0] > 0 or table.cumulative_bets[1] > 0, (
        "cumulative_bets should include blinds after start_table")


def test_cumulative_bets_includes_blinds():
    """cumulative_bets at start should equal blind postings."""
    table = _make_table(big_blind=10, small_blind=5)
    table.start_table()
    assert abs(table.cumulative_bets[0] - 5.0) < 1e-6
    assert abs(table.cumulative_bets[1] - 10.0) < 1e-6


def test_cumulative_bets_accumulates_across_streets():
    """cumulative_bets must not reset on street change (unlike self.bets)."""
    table = _make_table()
    table.start_table()
    n_act = table.n_raise_bins + 3
    initial_cum = table.cumulative_bets.copy()
    end = False
    street_changes = 0
    prev_turn = table.turn
    safety = 0
    while not end and safety < 100:
        end, _, _, _ = table.step(_action_tensor(1, n_actions=n_act))
        if table.turn != prev_turn:
            street_changes += 1
            prev_turn = table.turn
        safety += 1
    if street_changes > 0:
        assert np.all(table.cumulative_bets >= initial_cum), (
            "cumulative_bets should never decrease")


# ── 0.2: No equal-split leak correction in evaluate.py ───────────────────────

def test_no_leak_correction_in_evaluate():
    """evaluate.py must not contain the equal-split 'leak correction' pattern."""
    eval_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "evaluation/evaluate.py")
    with open(eval_path) as f:
        source = f.read()
    assert "leak / num_players" not in source and "leak/num_players" not in source, (
        "Found equal-split leak correction in evaluate.py — should be removed per 0.2")
