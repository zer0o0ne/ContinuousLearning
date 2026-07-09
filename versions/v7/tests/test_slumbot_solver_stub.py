"""Test that _make_solver_table_stub produces a table stub compatible with
GameState.from_table and _compute_all_action_evs.

Regression tests for missing/mistyped attributes on the SimpleNamespace stub:
  - start_credits was a scalar float (not per-player array)
  - active_player was missing entirely
  - several_all_in was missing entirely

Run (from versions/v6):
    python -m pytest tests/test_slumbot_solver_stub.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

import numpy as np

from evaluation.slumbot_eval import _make_solver_table_stub
from agent.mcts.game_state import GameState


def _make_state(active_pos=0, players_state=None):
    return {
        "pot": 300,
        "bets": [150, 150],
        "credits": [9850, 9850],
        "players_state": players_state or [1, 1],
        "high_bet": 150,
        "turn": 1,
        "active_pos": active_pos,
        "last_bet_size": 0,
    }


def _make_stub(state=None, hero_user_pos=0):
    if state is None:
        state = _make_state()
    return _make_solver_table_stub(
        hero_user_pos=hero_user_pos,
        hole_cards_int=[0, 1],
        board_ints=[10, 11, 12, -1, -1],
        state=state,
        raise_sizes=RAISE_SIZES,
        big_blind_internal=BIG_BLIND,
        small_blind_internal=SMALL_BLIND,
        chip_scale=CHIP_SCALE,
        num_players=2,
    )


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
    stub = _make_stub()
    assert hasattr(stub, "start_credits")
    assert stub.start_credits[0] == stub.start_credits[1]
    assert float(stub.start_credits[0]) > 0
    assert len(stub.start_credits) == 2


def test_stub_hero_invested_both_positions():
    """hero_invested = start_credits[pos] - credits[pos] must work for pos 0 and 1."""
    for hero_pos in [0, 1]:
        stub = _make_stub(hero_user_pos=hero_pos)
        hero_invested = stub.start_credits[hero_pos] - stub.credits[hero_pos]
        assert isinstance(float(hero_invested), float)
        assert hero_invested >= 0


def test_stub_all_attributes_indexable():
    """All per-player attributes on the stub must be subscriptable."""
    stub = _make_stub()
    for attr in ("start_credits", "credits", "bets", "players_state"):
        val = getattr(stub, attr)
        _ = val[0]
        _ = val[1]


def test_stub_raise_sizes_indexable():
    """raise_sizes[street] must return a list of raise fractions."""
    stub = _make_stub()
    street_raises = stub.raise_sizes[stub.turn]
    assert len(street_raises) == 3
    assert stub.n_raise_bins == 3


def test_active_player_present_and_correct():
    """active_player must be present and mapped from Slumbot→user frame."""
    # Slumbot active_pos=0 (BB) → user pos 1
    stub = _make_stub(state=_make_state(active_pos=0))
    assert hasattr(stub, "active_player")
    assert stub.active_player == 1

    # Slumbot active_pos=1 (SB) → user pos 0
    stub = _make_stub(state=_make_state(active_pos=1))
    assert stub.active_player == 0


def test_several_all_in_present():
    """several_all_in must be present as a bool."""
    stub = _make_stub(state=_make_state(players_state=[1, 1]))
    assert hasattr(stub, "several_all_in")
    assert stub.several_all_in is False

    stub = _make_stub(state=_make_state(players_state=[2, 2]))
    assert stub.several_all_in is True


def test_last_raise_size_present():
    """last_raise_size must be present as a float."""
    stub = _make_stub()
    assert hasattr(stub, "last_raise_size")
    assert isinstance(stub.last_raise_size, float)
    assert stub.last_raise_size > 0


def test_game_state_from_table_succeeds():
    """GameState.from_table must not raise on the stub."""
    for hero_pos in [0, 1]:
        stub = _make_stub(hero_user_pos=hero_pos)
        gs = GameState.from_table(stub, hero_pos)
        assert gs.num_players == 2
        assert gs.hero_pos == hero_pos
        assert gs.active_player == stub.active_player
        assert gs.several_all_in == stub.several_all_in


def test_game_state_legal_mask_from_stub():
    """get_legal_action_mask on a GameState built from the stub must return
    a valid mask of the correct length."""
    stub = _make_stub()
    n_actions = stub.n_raise_bins + 3
    gs = GameState.from_table(stub, 0)
    mask = gs.get_legal_action_mask(n_actions)
    assert len(mask) == n_actions
    assert mask[1] == 1  # call/check is always legal
