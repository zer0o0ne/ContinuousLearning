"""Tests for _action_idx_to_incr rounding to nearest legal Slumbot bet
and _build_game_state big_blind scaling fix.

Run (from versions/v6):
    python -m pytest tests/test_slumbot_round_to_legal.py -v
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.slumbot_eval import (
    _action_idx_to_incr,
    _build_game_state,
    SLUMBOT_BIG_BLIND,
)

RAISE_SIZES_FLOP = [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0]
RAISE_SIZES_PREFLOP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
N_RAISE_BINS = 11


def _counters():
    return defaultdict(int)


def _flop_state_after_preflop_raise(raise_total=300):
    """Flop state after both players put raise_total preflop."""
    return {
        "pot": raise_total * 2,
        "bets": [0, 0],
        "credits": [20000 - raise_total, 20000 - raise_total],
        "high_bet": 0,
        "last_bet_size": raise_total - SLUMBOT_BIG_BLIND,
        "turn": 1,
        "active_pos": 1,
        "players_state": [1, 1],
    }


# ============================================================================
# _action_idx_to_incr: round-to-nearest tests
# ============================================================================

class TestRoundToNearestLegal:
    """Small bets below min-legal should round to call/check when closer to
    call than to min-legal, not always bump up."""

    def test_small_bet_rounds_to_check(self):
        """0.10x pot on flop (bet=60) << min_legal=200 → check."""
        state = _flop_state_after_preflop_raise(300)
        c = _counters()
        result = _action_idx_to_incr(state, 2, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 0, c)
        assert result == "k", f"Expected check, got {result}"
        assert c["raise_rounded_to_call"] == 1

    def test_quarter_pot_rounds_to_min_raise(self):
        """0.25x pot on flop (bet=150), dist_to_call=150 > dist_to_min=50 → min raise."""
        state = _flop_state_after_preflop_raise(300)
        c = _counters()
        result = _action_idx_to_incr(state, 3, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 0, c)
        assert result == "b200", f"Expected b200, got {result}"
        assert c["raise_bumped"] == 1

    def test_third_pot_rounds_to_min_raise(self):
        """0.33x pot on flop (bet=198), dist_to_call=198, dist_to_min=2 → min raise 200."""
        state = _flop_state_after_preflop_raise(300)
        c = _counters()
        result = _action_idx_to_incr(state, 4, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 0, c)
        assert result == "b200", f"Expected b200, got {result}"
        assert c["raise_bumped"] == 1

    def test_half_pot_no_clamp(self):
        """0.5x pot (bet=300) >= min_legal=200 → passes through unchanged."""
        state = _flop_state_after_preflop_raise(300)
        c = _counters()
        result = _action_idx_to_incr(state, 6, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 0, c)
        assert result == "b300"
        assert c["raise_bumped"] == 0
        assert c["raise_rounded_to_call"] == 0

    def test_round_to_call_when_facing_bet(self):
        """When facing a bet, small raise rounds to call ('c'), not check ('k')."""
        state = {
            "pot": 800,
            "bets": [200, 0],
            "credits": [19800, 20000],
            "high_bet": 200,
            "last_bet_size": 200,
            "turn": 1,
            "active_pos": 1,
            "players_state": [0, 1],
        }
        c = _counters()
        result = _action_idx_to_incr(state, 2, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 1, c)
        # 0.10x: call=200, eff_pot=800, added=round(200+0.1*800)=280
        # new_total=280, min_legal=200+max(100,200)=400
        # dist_to_call=280-200=80, dist_to_min=400-280=120 → call
        assert result == "c", f"Expected call, got {result}"
        assert c["raise_rounded_to_call"] == 1

    def test_round_to_min_when_closer(self):
        """When bet is closer to min_legal than call, bump up to min_legal."""
        state = {
            "pot": 800,
            "bets": [200, 0],
            "credits": [19800, 20000],
            "high_bet": 200,
            "last_bet_size": 200,
            "turn": 1,
            "active_pos": 1,
            "players_state": [0, 1],
        }
        c = _counters()
        # 0.25x: call=200, eff_pot=800, added=round(200+0.25*800)=400
        # new_total=400, min_legal=400 → exactly at min_legal, no clamp
        result = _action_idx_to_incr(state, 3, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 1, c)
        assert result == "b400", f"Expected b400, got {result}"
        assert c["raise_bumped"] == 0
        assert c["raise_rounded_to_call"] == 0

    def test_fold_check_allin_unchanged(self):
        """Fold, call/check, and all-in actions are not affected."""
        state = _flop_state_after_preflop_raise(300)
        c = _counters()
        # Fold with no facing bet → check
        assert _action_idx_to_incr(state, 0, RAISE_SIZES_FLOP,
                                   N_RAISE_BINS, 0, c) == "k"
        # Call/check
        assert _action_idx_to_incr(state, 1, RAISE_SIZES_FLOP,
                                   N_RAISE_BINS, 0, c) == "k"
        # All-in
        result = _action_idx_to_incr(state, N_RAISE_BINS + 2, RAISE_SIZES_FLOP,
                                     N_RAISE_BINS, 0, _counters())
        assert result.startswith("b")

    def test_preflop_small_raise_rounds_to_call(self):
        """Preflop: 0.5x (bet=150) with min_legal=200 → call."""
        state = {
            "pot": 150,
            "bets": [100, 50],
            "credits": [19900, 19950],
            "high_bet": 100,
            "last_bet_size": 50,
            "turn": 0,
            "active_pos": 1,
            "players_state": [1, 1],
        }
        c = _counters()
        result = _action_idx_to_incr(state, 2, RAISE_SIZES_PREFLOP,
                                     N_RAISE_BINS, 1, c)
        # 0.5x: call=50, eff_pot=100, added=round(50+0.5*100)=100
        # new_total=50+100=150, min_legal=100+max(100,50)=200
        # dist_to_call=150-100=50, dist_to_min=200-150=50 → tie → call
        assert result == "c", f"Expected call, got {result}"
        assert c["raise_rounded_to_call"] == 1


# ============================================================================
# _build_game_state: big_blind scaling fix
# ============================================================================

RAISE_SIZES_DICT = {
    0: RAISE_SIZES_PREFLOP,
    1: RAISE_SIZES_FLOP,
    2: RAISE_SIZES_FLOP,
    3: RAISE_SIZES_FLOP,
}


class TestBuildGameStateBigBlind:
    """_build_game_state must pass big_blind in internal units (not re-scaled)."""

    def test_big_blind_is_internal_units(self):
        """big_blind=10 internal should remain 10, not become 1."""
        state = {
            "pot": 600,
            "bets": [0, 0],
            "credits": [19700, 19700],
            "players_state": [1, 1],
            "high_bet": 0,
            "turn": 1,
            "active_pos": 0,
        }
        chip_scale = 10.0
        big_blind_internal = 10.0
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES_DICT,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=chip_scale,
                               big_blind_internal=big_blind_internal)
        assert gs.big_blind == 10.0, f"Expected 10.0, got {gs.big_blind}"
        assert gs.last_raise_size == 10.0, f"Expected 10.0, got {gs.last_raise_size}"

    def test_big_blind_not_scaled_by_inv(self):
        """Regression: big_blind must NOT be multiplied by inv_scale."""
        state = {
            "pot": 150,
            "bets": [100, 50],
            "credits": [19900, 19950],
            "players_state": [1, 1],
            "high_bet": 100,
            "turn": 0,
            "active_pos": 1,
        }
        for chip_scale in [10.0, 5.0, 1.0]:
            big_blind_internal = 10.0
            gs = _build_game_state(state, hero_user_pos=0,
                                   raise_sizes=RAISE_SIZES_DICT,
                                   n_raise_bins=N_RAISE_BINS,
                                   chip_scale=chip_scale,
                                   big_blind_internal=big_blind_internal)
            assert gs.big_blind == 10.0, (
                f"chip_scale={chip_scale}: expected bb=10.0, got {gs.big_blind}")

    def test_legal_mask_filters_small_raises_with_correct_bb(self):
        """With correct big_blind=10, small flop raises should be filtered
        when last_raise_size (from big_blind) is 10."""
        state = {
            "pot": 600,
            "bets": [0, 0],
            "credits": [19700, 19700],
            "players_state": [1, 1],
            "high_bet": 0,
            "turn": 1,
            "active_pos": 0,
        }
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES_DICT,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=10.0,
                               big_blind_internal=10.0)
        n_actions = N_RAISE_BINS + 3
        legal = gs.get_legal_actions()
        # 0.10x pot = 0.1*60 = 6 internal (last_raise=10) → filtered
        assert 2 not in legal, "0.10x pot raise should be filtered (below min-raise)"
