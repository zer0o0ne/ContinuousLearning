"""Test that _make_solver_table_stub produces a table stub compatible with
_compute_all_action_evs.

Regression test for: `start_credits` was a scalar float instead of a
per-player array, causing `TypeError: 'float' object is not subscriptable`
when `_compute_all_action_evs` did `table.start_credits[player_pos]`.

Run (from versions/v6):
    python -m pytest tests/test_slumbot_solver_stub.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

import numpy as np

from evaluation.slumbot_eval import _make_solver_table_stub


def _make_state():
    return {
        "pot": 300,
        "bets": [150, 150],
        "credits": [9850, 9850],
        "players_state": [1, 1],
        "high_bet": 150,
        "turn": 1,
        "active_pos": 0,
        "last_bet_size": 0,
    }


RAISE_SIZES = [
    [0.5, 1.0, 1.5],
    [0.33, 0.5, 0.75],
    [0.33, 0.5, 0.75],
    [0.33, 0.5, 0.75],
]
BIG_BLIND = 10.0
SMALL_BLIND = 5.0
CHIP_SCALE = 1.0


def test_start_credits_is_per_player_array():
    """start_credits must be indexable per player, not a scalar float."""
    state = _make_state()
    stub = _make_solver_table_stub(
        hero_user_pos=0,
        hole_cards_int=[0, 1],
        board_ints=[10, 11, 12, -1, -1],
        state=state,
        raise_sizes=RAISE_SIZES,
        big_blind_internal=BIG_BLIND,
        small_blind_internal=SMALL_BLIND,
        chip_scale=CHIP_SCALE,
        num_players=2,
    )
    assert hasattr(stub, "start_credits")
    assert stub.start_credits[0] == stub.start_credits[1]
    assert float(stub.start_credits[0]) > 0
    assert len(stub.start_credits) == 2


def test_stub_hero_invested_both_positions():
    """hero_invested = start_credits[pos] - credits[pos] must work for pos 0 and 1."""
    state = _make_state()
    for hero_pos in [0, 1]:
        stub = _make_solver_table_stub(
            hero_user_pos=hero_pos,
            hole_cards_int=[0, 1],
            board_ints=[10, 11, 12, -1, -1],
            state=state,
            raise_sizes=RAISE_SIZES,
            big_blind_internal=BIG_BLIND,
            small_blind_internal=SMALL_BLIND,
            chip_scale=CHIP_SCALE,
            num_players=2,
        )
        hero_invested = stub.start_credits[hero_pos] - stub.credits[hero_pos]
        assert isinstance(float(hero_invested), float)
        assert hero_invested >= 0


def test_stub_all_attributes_indexable():
    """All per-player attributes on the stub must be subscriptable."""
    state = _make_state()
    stub = _make_solver_table_stub(
        hero_user_pos=0,
        hole_cards_int=[0, 1],
        board_ints=[10, 11, 12, -1, -1],
        state=state,
        raise_sizes=RAISE_SIZES,
        big_blind_internal=BIG_BLIND,
        small_blind_internal=SMALL_BLIND,
        chip_scale=CHIP_SCALE,
        num_players=2,
    )
    for attr in ("start_credits", "credits", "bets", "players_state"):
        val = getattr(stub, attr)
        _ = val[0]
        _ = val[1]


def test_stub_raise_sizes_indexable():
    """raise_sizes[street] must return a list of raise fractions."""
    state = _make_state()
    stub = _make_solver_table_stub(
        hero_user_pos=0,
        hole_cards_int=[0, 1],
        board_ints=[10, 11, 12, -1, -1],
        state=state,
        raise_sizes=RAISE_SIZES,
        big_blind_internal=BIG_BLIND,
        small_blind_internal=SMALL_BLIND,
        chip_scale=CHIP_SCALE,
        num_players=2,
    )
    street_raises = stub.raise_sizes[stub.turn]
    assert len(street_raises) == 3
    assert stub.n_raise_bins == 3
