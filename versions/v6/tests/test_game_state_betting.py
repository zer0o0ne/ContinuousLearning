"""
Tests for GameState betting logic and street transitions.

Run from versions/v6/:
    python -m unittest tests.test_game_state_betting -v
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcts.game_state import GameState

# ── shared fixtures ──────────────────────────────────────────────────────────

BIG_BLIND = 10
SMALL_BLIND = 5
# 4 raise bins so the arithmetic stays simple; reused for all streets
RAISE_SIZES = [[0.5, 1.0, 1.5, 2.0]] * 4
N_RAISE_BINS = len(RAISE_SIZES[0])
N_ACTIONS = N_RAISE_BINS + 3  # fold(0) call(1) r0..r3(2..5) allin(6)
TOL = 1e-9


def _make_gs(
    players_state=None,
    credits=None,
    bets=None,
    pot=30.0,
    high_bet=10.0,
    turn=0,
    active_player=0,
    hero_pos=0,
    num_players=2,
    big_blind=BIG_BLIND,
    last_raise_size=None,
    last_full_raise_level=None,
    is_terminal=False,
    several_all_in=False,
):
    """Construct a two-player GameState with sensible defaults."""
    if players_state is None:
        players_state = [1, 0]
    if credits is None:
        credits = [90.0, 90.0]
    if bets is None:
        bets = [10.0, 0.0]
    return GameState(
        num_players=num_players,
        hero_pos=hero_pos,
        active_player=active_player,
        players_state=list(players_state),
        credits=list(credits),
        bets=list(bets),
        pot=pot,
        high_bet=high_bet,
        turn=turn,
        raise_sizes=RAISE_SIZES,
        n_raise_bins=N_RAISE_BINS,
        is_terminal=is_terminal,
        several_all_in=several_all_in,
        big_blind=big_blind,
        last_raise_size=last_raise_size,
        last_full_raise_level=last_full_raise_level,
    )


# ── 1. Fold action ────────────────────────────────────────────────────────────

class TestFoldAction(unittest.TestCase):

    def test_fold_sets_state_minus_one(self):
        """Fold sets the acting player's state to -1 (folded)."""
        gs = _make_gs(players_state=[1, 0])
        pot_before = gs.pot
        gs.step(0)  # fold
        self.assertEqual(gs.players_state[0], -1)

    def test_fold_pot_unchanged(self):
        """Folding does not change the pot."""
        gs = _make_gs(players_state=[1, 0], pot=30.0)
        gs.step(0)
        # pot may have changed only if next_turn causes a street change (not here)
        # With player 1 still active, no street change; pot stays 30
        self.assertAlmostEqual(gs.pot, 30.0, places=9)

    def test_fold_one_active_terminal(self):
        """When only one active player remains after a fold, game is terminal."""
        # player 0 folds; player 1 is the only remaining active
        gs = _make_gs(
            players_state=[1, 0],
            credits=[90.0, 90.0],
            bets=[10.0, 10.0],
            pot=20.0, high_bet=10.0,
        )
        gs.step(0)
        self.assertTrue(gs.is_terminal)

    def test_fold_non_terminal_when_multiple_active(self):
        """Folding with 3+ players is only terminal if exactly 1 remains."""
        gs = GameState(
            num_players=3, hero_pos=0, active_player=0,
            players_state=[1, 0, 0],
            credits=[80.0, 90.0, 90.0],
            bets=[20.0, 10.0, 10.0],
            pot=40.0, high_bet=20.0, turn=0,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
        )
        gs.step(0)  # player 0 folds; players 1 and 2 still active
        self.assertFalse(gs.is_terminal)
        self.assertEqual(gs.players_state[0], -1)


# ── 2. Call action ────────────────────────────────────────────────────────────

class TestCallAction(unittest.TestCase):

    def test_call_bet_equals_deficit(self):
        """Call bet = high_bet - bets[pos] when player has enough credits.

        After calling, player 0's bet matches high_bet. We verify the pot
        increases correctly (bets are reset on street transition, so we check
        the pot delta rather than the transient bets value).
        """
        # pos=0 has bet 0; high_bet=10; deficit=10; player 1 is done (state=0)
        gs = _make_gs(
            players_state=[1, 0],
            credits=[100.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        pot_before = gs.pot
        gs.step(1)  # call — bet = min(10 - 0, 100) = 10; pot += 10
        # After the call both players have matched the bet, so _next_turn
        # advances to the next street and resets bets to 0. We verify via pot.
        self.assertAlmostEqual(gs.pot, pot_before + 10.0, places=9)

    def test_call_increases_pot(self):
        """Pot increases by the call amount."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[100.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        gs.step(1)
        # call amount = 10; new pot = 15 + 10 = 25
        self.assertAlmostEqual(gs.pot, 25.0, places=9)

    def test_call_credits_decrease(self):
        """Credits decrease by the call amount."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[100.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        gs.step(1)
        self.assertAlmostEqual(gs.credits[0], 90.0, places=9)

    def test_call_allin_when_credits_exhausted(self):
        """If call commits all remaining credits, state becomes 2 (all-in)."""
        # Player has exactly the call amount
        gs = _make_gs(
            players_state=[1, 0],
            credits=[10.0, 90.0],  # pos 0 has exactly 10 = call amount
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        gs.step(1)
        self.assertEqual(gs.players_state[0], 2)
        self.assertAlmostEqual(gs.credits[0], 0.0, places=9)

    def test_call_capped_at_credits(self):
        """Call is capped at available credits (partial call / short-stack).

        Player 0 has 7 credits but faces a deficit of 10. The call commits
        only 7 (all of them), making the player all-in. We verify via credits
        becoming 0 and the pot increasing by exactly 7.
        """
        gs = _make_gs(
            players_state=[1, 0],
            credits=[7.0, 90.0],   # less than the deficit of 10
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        pot_before = gs.pot
        gs.step(1)
        # credits[0] committed all 7
        self.assertAlmostEqual(gs.credits[0], 0.0, places=9)
        # pot increased by exactly 7
        self.assertAlmostEqual(gs.pot, pot_before + 7.0, places=9)


# ── 3. Raise action ───────────────────────────────────────────────────────────

class TestRaiseAction(unittest.TestCase):

    def _raise_amount(self, gs, action_idx):
        """Compute expected bet for a raise action."""
        pos = gs.active_player
        raise_pct = RAISE_SIZES[gs.turn][action_idx - 2]
        effective_pot = gs.pot - gs.bets[pos]
        call_amount = gs.high_bet - gs.bets[pos]
        bet = min(call_amount + raise_pct * effective_pot, gs.credits[pos])
        return bet, call_amount

    def test_raise_increases_pot(self):
        """Pot increases by the raise amount."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 200.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        pot_before = gs.pot
        action = 2  # first raise bin (0.5×pot)
        bet, _ = self._raise_amount(gs, action)
        gs.step(action)
        self.assertAlmostEqual(gs.pot, pot_before + bet, places=9)

    def test_raise_updates_high_bet(self):
        """high_bet becomes the new bets[pos] after a raise."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 200.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        action = 2
        bet, _ = self._raise_amount(gs, action)
        gs.step(action)
        self.assertAlmostEqual(gs.high_bet, bet, places=9)

    def test_raise_reopens_action(self):
        """After a raise, the opponent (state=0) transitions to state=1 (must act)."""
        # Player 1 has already matched the bet (state=0)
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 200.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        gs.step(2)  # player 0 raises
        # _next_turn sets player 1 to 1 because bets[1] < high_bet
        self.assertEqual(gs.players_state[1], 1)

    def test_raise_state_not_allin_when_credits_remain(self):
        """Player state stays 0 (active) when raise doesn't exhaust credits."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 200.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        gs.step(2)
        self.assertEqual(gs.players_state[0], 0)
        self.assertGreater(gs.credits[0], 0.0)

    def test_raise_allin_when_credits_exhausted(self):
        """If raise amount would exceed credits, player goes all-in (state=2)."""
        # Give just enough credits so the raise hits the cap
        gs = _make_gs(
            players_state=[1, 0],
            credits=[5.0, 200.0],  # tiny stack
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        gs.step(2)  # raise that exceeds credits → goes to min(raise, credits)=5
        self.assertEqual(gs.players_state[0], 2)
        self.assertAlmostEqual(gs.credits[0], 0.0, places=9)


# ── 4. All-in action ─────────────────────────────────────────────────────────

class TestAllInAction(unittest.TestCase):

    def test_allin_commits_all_credits(self):
        """All-in action bets all remaining credits."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[80.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        credits_before = gs.credits[0]
        gs.step(N_RAISE_BINS + 2)  # all-in action index
        self.assertAlmostEqual(gs.credits[0], 0.0, places=9)

    def test_allin_state_becomes_2(self):
        """All-in action sets player state to 2."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[80.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        gs.step(N_RAISE_BINS + 2)
        self.assertEqual(gs.players_state[0], 2)

    def test_allin_pot_increases(self):
        """All-in adds credits to the pot."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[80.0, 90.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        credits_before = gs.credits[0]
        pot_before = gs.pot
        gs.step(N_RAISE_BINS + 2)
        self.assertAlmostEqual(gs.pot, pot_before + credits_before, places=9)

    def test_allin_updates_high_bet_when_larger(self):
        """All-in updates high_bet when the bet exceeds current high_bet."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 90.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        gs.step(N_RAISE_BINS + 2)
        self.assertAlmostEqual(gs.high_bet, gs.bets[0], places=9)


# ── 5. Street transition ─────────────────────────────────────────────────────

class TestStreetTransition(unittest.TestCase):

    def _two_player_closed_street(self, turn=0):
        """Both players have matched bets; next step should advance street."""
        # Both players in state=0 (waiting). After calling the street action
        # completes and _next_turn triggers a new street.
        gs = GameState(
            num_players=2, hero_pos=0, active_player=0,
            players_state=[1, 0],
            credits=[90.0, 100.0],
            bets=[10.0, 20.0],
            pot=35.0, high_bet=20.0, turn=turn,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
        )
        return gs

    def test_street_turn_increments(self):
        """turn increments when betting action is complete."""
        gs = self._two_player_closed_street(turn=0)
        old_turn = gs.turn
        gs.step(1)  # call closes the street
        self.assertEqual(gs.turn, old_turn + 1)

    def test_street_bets_reset(self):
        """bets reset to 0 on a new street."""
        gs = self._two_player_closed_street(turn=0)
        gs.step(1)
        self.assertTrue(all(abs(b) < TOL for b in gs.bets))

    def test_street_high_bet_reset(self):
        """high_bet resets to 0 on a new street."""
        gs = self._two_player_closed_street(turn=0)
        gs.step(1)
        self.assertAlmostEqual(gs.high_bet, 0.0, places=9)

    def test_street_last_raise_size_resets(self):
        """last_raise_size resets to big_blind on a new street (C.7.5)."""
        gs = self._two_player_closed_street(turn=0)
        gs.last_raise_size = 50.0  # artificially inflate
        gs.step(1)
        self.assertAlmostEqual(gs.last_raise_size, BIG_BLIND, places=9)

    def test_river_complete_is_terminal(self):
        """After river betting completes, game is terminal."""
        gs = self._two_player_closed_street(turn=3)
        gs.step(1)
        self.assertTrue(gs.is_terminal)


# ── 6. Terminal detection ────────────────────────────────────────────────────

class TestTerminalDetection(unittest.TestCase):

    def test_one_active_player_terminal(self):
        """Only one non-folded player → terminal."""
        gs = _make_gs(
            players_state=[1, -1],
            credits=[90.0, 0.0],
            bets=[10.0, 10.0],
            pot=20.0, high_bet=10.0,
            active_player=0,
        )
        gs.step(1)  # check/call; only player 0 left
        self.assertTrue(gs.is_terminal)

    def test_all_active_allin_runout_terminal(self):
        """All active players are all-in → terminal after the final call."""
        # Player 0 is already all-in (state=2). Player 1 is waiting to act
        # (state=1) and will call, exhausting their credits → both all-in.
        gs = GameState(
            num_players=2, hero_pos=0, active_player=1,  # player 1 is acting
            players_state=[2, 1],  # pos 0 all-in, pos 1 waiting to call
            credits=[0.0, 50.0],
            bets=[100.0, 50.0],
            pot=155.0, high_bet=100.0, turn=1,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
        )
        # Player 1 calls (going all-in), leaving no non-all-in players
        gs.step(1)
        self.assertTrue(gs.is_terminal)

    def test_not_terminal_with_two_active(self):
        """With two active players and pending action, not terminal."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[90.0, 90.0],
            bets=[10.0, 10.0],
            pot=25.0, high_bet=10.0,
        )
        self.assertFalse(gs.is_terminal)


# ── 7. Legal action masking ───────────────────────────────────────────────────

class TestLegalActionMasking(unittest.TestCase):

    def test_fold_illegal_when_facing_no_bet(self):
        """Fold (action 0) is illegal when facing_bet=0 (checking is free)."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[100.0, 100.0],
            bets=[0.0, 0.0],
            pot=0.0, high_bet=0.0,
            active_player=0,
        )
        legal = gs.get_legal_actions()
        self.assertNotIn(0, legal)

    def test_fold_legal_when_facing_bet(self):
        """Fold (action 0) is legal when there is a bet to face."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[90.0, 100.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        legal = gs.get_legal_actions()
        self.assertIn(0, legal)

    def test_call_always_legal(self):
        """Call/check (action 1) is always in legal actions."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[100.0, 100.0],
            bets=[0.0, 0.0],
            pot=0.0, high_bet=0.0,
            active_player=0,
        )
        legal = gs.get_legal_actions()
        self.assertIn(1, legal)

    def test_no_raises_when_all_opponents_allin(self):
        """No raise or all-in action when all live opponents are already all-in (C.5)."""
        gs = _make_gs(
            players_state=[1, 2],          # hero waiting, opp all-in
            credits=[500.0, 0.0],
            bets=[0.0, 100.0],
            pot=100.0, high_bet=100.0,
            active_player=0,
        )
        legal = gs.get_legal_actions()
        for a in legal:
            self.assertLessEqual(a, 1, f"raise/allin action {a} should be illegal")

    def test_capped_raise_bin_masked(self):
        """Raise bins whose computed bet >= credits are excluded (collapses to all-in)."""
        # Give very small credits so only the explicit all-in remains; raised fractions
        # would exceed credits and therefore must be masked.
        gs = GameState(
            num_players=2, hero_pos=0, active_player=0,
            players_state=[1, 0],
            credits=[15.0, 200.0],  # small stack
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0, turn=0,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
        )
        legal = gs.get_legal_actions()
        # All raise bins (2..N_RAISE_BINS+1) should be absent; only allin (N_RAISE_BINS+2)
        for a in range(2, N_RAISE_BINS + 2):
            if a in legal:
                # Verify the bet would not exceed credits
                raise_pct = RAISE_SIZES[0][a - 2]
                effective_pot = gs.pot - gs.bets[0]
                call_amount = gs.high_bet - gs.bets[0]
                bet = call_amount + raise_pct * effective_pot
                self.assertLess(bet, gs.credits[0],
                                f"Raise bin {a} should be masked because bet={bet} >= credits={gs.credits[0]}")

    def test_min_raise_filter_masks_small_increments(self):
        """Raise increments below last_raise_size are masked (C.7.5)."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[500.0, 500.0],
            bets=[0.0, 10.0],
            pot=20.0, high_bet=10.0,
            active_player=0,
        )
        gs.last_raise_size = 100.0  # very high: all raise bins will be filtered
        legal = gs.get_legal_actions()
        for a in range(2, N_RAISE_BINS + 2):
            self.assertNotIn(a, legal, f"Raise bin {a} should be filtered by min-raise")


# ── 8. Min-raise tracking (C.7.5) ────────────────────────────────────────────

class TestMinRaiseTracking(unittest.TestCase):

    def test_full_raise_updates_last_raise_size(self):
        """A full raise (increment >= last_raise_size) updates last_raise_size."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[200.0, 200.0],
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0,
            active_player=0,
        )
        old_lrs = gs.last_raise_size
        action = 3  # raise index 3 → raise_pct=1.5 → increment = 1.5 * 20 = 30 > big_blind(10)
        gs.step(action)
        self.assertNotEqual(gs.last_raise_size, old_lrs,
                            "last_raise_size should update after full raise")

    def test_short_allin_below_last_raise_does_not_update(self):
        """Short all-in that raises by less than last_raise_size does NOT update last_raise_size."""
        # last_raise_size = 50 (large); short all-in raises by only 5
        gs = GameState(
            num_players=2, hero_pos=0, active_player=0,
            players_state=[1, 0],
            credits=[5.0, 200.0],  # tiny stack → all-in increment < last_raise_size
            bets=[0.0, 0.0],
            pot=20.0, high_bet=0.0, turn=0,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
            last_raise_size=50.0,
        )
        old_lrs = gs.last_raise_size
        gs.step(N_RAISE_BINS + 2)  # all-in with credits=5
        self.assertAlmostEqual(gs.last_raise_size, old_lrs, places=9,
                               msg="Short all-in should not update last_raise_size")

    def test_short_allin_does_not_reopen(self):
        """After a short all-in, a player who matched _last_full_raise_level cannot re-raise (C.7.5)."""
        # Setup: player 1 raised to 30 (full raise). Player 0 called (bets[0]=30).
        # Now a short all-in from player 1 pushed high_bet to 35.
        # _last_full_raise_level = 30; bets[0]=30 >= _last_full_raise_level;
        # high_bet=35 > _last_full_raise_level → player 0 is NOT reopened.
        gs = GameState(
            num_players=2, hero_pos=0, active_player=0,
            players_state=[1, 2],
            credits=[200.0, 0.0],
            bets=[30.0, 35.0],
            pot=70.0, high_bet=35.0, turn=1,
            raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
            big_blind=BIG_BLIND,
            last_raise_size=30.0,
            last_full_raise_level=30.0,
        )
        legal = gs.get_legal_actions()
        # No raises allowed; player 0 can only call or fold
        for a in legal:
            self.assertLessEqual(a, 1,
                                 f"Action {a} should be illegal: not reopened after short all-in")

    def test_last_raise_size_resets_on_new_street(self):
        """last_raise_size resets to big_blind at start of each new street."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[90.0, 100.0],
            bets=[10.0, 20.0],
            pot=35.0, high_bet=20.0,
            turn=0,
            active_player=0,
        )
        gs.last_raise_size = 99.0
        gs.step(1)  # call → street closes → new street
        self.assertAlmostEqual(gs.last_raise_size, BIG_BLIND, places=9)


# ── 9. get_legal_action_mask ─────────────────────────────────────────────────

class TestGetLegalActionMask(unittest.TestCase):

    def test_mask_shape(self):
        """get_legal_action_mask returns a list of length n_actions."""
        gs = _make_gs()
        mask = gs.get_legal_action_mask(N_ACTIONS)
        self.assertEqual(len(mask), N_ACTIONS)

    def test_mask_is_bool_list(self):
        """Each element is a bool."""
        gs = _make_gs()
        mask = gs.get_legal_action_mask(N_ACTIONS)
        for v in mask:
            self.assertIsInstance(v, bool)

    def test_mask_consistent_with_get_legal_actions(self):
        """Mask True positions match exactly the legal action set."""
        gs = _make_gs(
            players_state=[1, 0],
            credits=[90.0, 100.0],
            bets=[0.0, 10.0],
            pot=15.0, high_bet=10.0,
            active_player=0,
        )
        legal = set(gs.get_legal_actions())
        mask = gs.get_legal_action_mask(N_ACTIONS)
        for i, v in enumerate(mask):
            if i in legal:
                self.assertTrue(v, f"action {i} in legal but mask[{i}]=False")
            else:
                self.assertFalse(v, f"action {i} not in legal but mask[{i}]=True")

    def test_mask_all_false_beyond_n_actions(self):
        """Asking for n_actions+5 extra slots returns False for those extras."""
        gs = _make_gs()
        mask = gs.get_legal_action_mask(N_ACTIONS + 5)
        self.assertEqual(len(mask), N_ACTIONS + 5)
        for i in range(N_ACTIONS, N_ACTIONS + 5):
            self.assertFalse(mask[i])

    def test_mask_can_be_converted_to_tensor(self):
        """get_legal_action_mask output can be converted directly to a bool tensor."""
        gs = _make_gs()
        mask = gs.get_legal_action_mask(N_ACTIONS)
        t = torch.tensor(mask, dtype=torch.bool)
        self.assertEqual(t.shape, (N_ACTIONS,))
        self.assertEqual(t.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
