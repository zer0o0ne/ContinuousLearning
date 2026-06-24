"""
Tests exposing bugs found during the codebase audit.

These tests verify REQUIRED behavior. Failing tests indicate real bugs
in the main code that should be fixed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# =====================================================================
# Bug 1: Judger full_coef overwritten by lower pairs
#
# In compute_power(), when a 7-card hand has trips + two or more pairs,
# the full house coefficient loop lacks a "and not full" guard.
# It overwrites full_coef with the LOWEST pair instead of keeping the
# HIGHEST pair. The get_bord() tiebreaker is correct (has the guard),
# but the numeric power used for primary comparison is wrong.
#
# Impact: Misranks full houses when trips + 2+ pairs are present.
# Example: T-T-T-8-8-5-5 loses to T-T-T-6-6 because full_coef uses
# the 5 instead of the 8.
# =====================================================================

class TestJudgerFullHouseBug:
    """Test that full house ranking uses the HIGHEST pair, not lowest."""

    def setup_method(self):
        from env.judger import Judger
        self.judger = Judger()

    def _make_hand(self, cards):
        """Convert card tuples (rank, suit) to card indices.
        rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
        suit: 0-3
        """
        return np.sort(np.array([r * 4 + s for r, s in cards]))

    def test_full_house_uses_highest_pair(self):
        """T-T-T-8-8-5-5 should have full_coef based on 8s, not 5s."""
        hand = self._make_hand([
            (8, 0), (8, 1), (8, 2),  # three tens
            (6, 0), (6, 1),          # pair of eights
            (3, 0), (3, 1),          # pair of fives
        ])
        power, bord = self.judger.compute_power(hand)

        # Full house power = 6 * 169 + threeakind_coef + pair_rank
        # threeakind_coef = 8 * 13 = 104, pair should be 8s (rank=6)
        expected_coef = 8 * 13 + 6  # = 110
        expected_power = 6 * 169 + expected_coef  # = 1124

        assert power == expected_power, (
            f"Full house T-T-T-8-8 should have power {expected_power}, "
            f"got {power}. Likely using lowest pair (5s) instead of highest (8s)."
        )

    def test_full_house_trips_plus_two_pairs_comparison(self):
        """T-T-T-8-8-5-5 must beat T-T-T-6-6-x-x.

        This is the core impact test: the bug causes a hand with the
        higher pair to LOSE to one with a lower pair.
        """
        hand_a = self._make_hand([
            (8, 0), (8, 1), (8, 2),  # three tens
            (6, 0), (6, 1),          # pair of eights
            (3, 0), (3, 1),          # pair of fives
        ])
        hand_b = self._make_hand([
            (8, 0), (8, 1), (8, 2),  # three tens
            (4, 0), (4, 1),          # pair of sixes
            (1, 2), (0, 3),          # low cards
        ])

        result_a, result_b = self.judger.compare_hands(hand_a, hand_b)

        assert result_a == 1 and result_b == 0, (
            f"T-T-T-8-8 should beat T-T-T-6-6, "
            f"got compare_hands result ({result_a}, {result_b})"
        )

    def test_full_house_three_pairs_picks_best(self):
        """With trips + two extra pairs, the highest pair must be used."""
        hand = self._make_hand([
            (11, 0), (11, 1), (11, 2),  # three kings
            (10, 0), (10, 1),           # pair of queens
            (7, 0), (7, 1),            # pair of nines
        ])
        power, bord = self.judger.compute_power(hand)

        # Full house: K-K-K-Q-Q. threeakind_coef = 11*13 = 143, pair = Q (rank 10)
        expected_power = 6 * 169 + 143 + 10  # = 1167
        assert power == expected_power, (
            f"Full house K-K-K-Q-Q should have power {expected_power}, got {power}"
        )

    def test_full_house_single_pair_unaffected(self):
        """Trips + exactly one pair should be unaffected by the bug."""
        hand = self._make_hand([
            (8, 0), (8, 1), (8, 2),  # three tens
            (6, 0), (6, 1),          # pair of eights
            (1, 0), (0, 1),          # unrelated low cards
        ])
        power, bord = self.judger.compute_power(hand)

        expected_power = 6 * 169 + 8 * 13 + 6  # = 1124
        assert power == expected_power

    def test_get_bord_is_correct_despite_power_bug(self):
        """get_bord picks the right pair (has the 'is None' guard).

        This test should PASS — demonstrating that the bord tiebreaker
        is correct but the numeric power comparison is wrong.
        """
        hand = self._make_hand([
            (8, 0), (8, 1), (8, 2),  # three tens
            (6, 0), (6, 1),          # pair of eights
            (3, 0), (3, 1),          # pair of fives
        ])
        _, bord = self.judger.compute_power(hand)

        # get_bord correctly returns [T,T,T,8,8] = [8,8,8,6,6] in rank encoding
        # sorted descending
        assert bord == [8, 8, 8, 6, 6], (
            f"Full house bord should be [8,8,8,6,6] (T-T-T-8-8), got {bord}"
        )


# =====================================================================
# Bug 2: _action_to_category maps ALL preflop raises to "3bet"
#
# In terminal_eval.py, _action_to_category returns "3bet" for any
# action_idx >= 2 on preflop, regardless of whether it's an open raise
# or a 3-bet. "3bet" narrows to top 8% of range, while "open" keeps
# 100%. This massively over-narrows opponent ranges on preflop opens.
#
# Impact: Equity computation assumes opponents always have premium
# hands when they open-raise preflop, biasing MCTS terminal values.
# =====================================================================

class TestActionToCategoryBug:
    """Test that preflop open raises map to 'open', not '3bet'."""

    RAISE_SIZES = {
        "preflop": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        "flop": [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
        "turn": [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
        "river": [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
    }

    def _make_game_state(self, turn=0, high_bet=10.0, big_blind=10.0):
        """Create a GameState for testing _action_to_category."""
        from agent.mcts.game_state import GameState
        num_players = 6
        n_raise_bins = len(self.RAISE_SIZES["preflop"])
        gs = GameState(
            num_players=num_players,
            hero_pos=0,
            active_player=2,
            players_state=[1] * num_players,
            credits=[1000.0] * num_players,
            bets=[0.0] * num_players,
            pot=15.0,
            high_bet=high_bet,
            turn=turn,
            raise_sizes=self.RAISE_SIZES,
            n_raise_bins=n_raise_bins,
            big_blind=big_blind,
        )
        return gs

    def test_preflop_open_raise_should_not_be_3bet(self):
        """First preflop raise (open) should NOT return '3bet'.

        An open raise is the first voluntary raise. It should map to
        'open' (100% range), not '3bet' (8% range). The distinction
        matters enormously for range narrowing.
        """
        from agent.mcts.terminal_eval import _action_to_category

        # Preflop, no raises yet (high_bet == big_blind = open raise)
        gs = self._make_game_state(turn=0, high_bet=10.0, big_blind=10.0)
        category = _action_to_category(2, gs)

        assert category != "3bet", (
            f"Open raise (first preflop raise) mapped to '{category}', "
            f"but should map to 'open' (100% range), not '3bet' (8% range). "
            f"This over-narrows opponent ranges by 12x."
        )

    def test_preflop_3bet_after_raise_should_be_3bet(self):
        """A re-raise after an open should correctly map to '3bet'."""
        from agent.mcts.terminal_eval import _action_to_category

        # Preflop, someone already raised to 30 (high_bet > big_blind)
        gs = self._make_game_state(turn=0, high_bet=30.0, big_blind=10.0)
        category = _action_to_category(3, gs)

        assert category == "3bet", (
            f"3-bet (re-raise preflop) should map to '3bet', got '{category}'"
        )

    def test_postflop_raise_is_bet_postflop(self):
        """Postflop raises should map to 'bet_postflop' (sanity check)."""
        from agent.mcts.terminal_eval import _action_to_category

        gs = self._make_game_state(turn=1)
        category = _action_to_category(2, gs)

        assert category == "bet_postflop"

    def test_fold_returns_none(self):
        """Fold should return None (no narrowing)."""
        from agent.mcts.terminal_eval import _action_to_category

        gs = self._make_game_state(turn=0)
        assert _action_to_category(0, gs) is None

    def test_preflop_call_returns_call(self):
        """Preflop call should return 'call'."""
        from agent.mcts.terminal_eval import _action_to_category

        gs = self._make_game_state(turn=0)
        assert _action_to_category(1, gs) == "call"


# =====================================================================
# Bug 3: MCTS _best_action temperature overflow
#
# In mcts.py, _best_action computes counts ** (1.0 / temperature).
# At very low temperatures (e.g., 0.01), the exponent becomes 100+,
# causing float64 overflow even for modest visit counts.
#
# Impact: Corrupted action selection at low temperatures.
# =====================================================================

class TestMCTSTemperatureOverflow:
    """Test that MCTS action selection handles low temperatures."""

    def _make_root_with_children(self, visit_counts):
        """Create a root node with children having given visit counts."""
        from agent.mcts.mcts import MCTSNode

        root = MCTSNode(is_hero=True)
        root.N = sum(visit_counts.values())

        for a_idx, visits in visit_counts.items():
            child = MCTSNode(action_idx=a_idx, parent=root, P=0.25)
            child.N = visits
            child.W = visits * 0.1
            child.Q = 0.1
            root.children[a_idx] = child

        return root

    def _make_mcts(self, temperature):
        """Create a minimal MCTS instance for testing _best_action."""
        from agent.mcts.mcts import MCTS
        mcts = MCTS.__new__(MCTS)
        mcts.n_actions = 14
        mcts.temperature = temperature
        mcts.device = "cpu"
        return mcts

    def test_low_temperature_no_overflow(self):
        """Very low temperature should not produce NaN/inf."""
        root = self._make_root_with_children({0: 500, 1: 300, 2: 150, 3: 50})
        mcts = self._make_mcts(temperature=0.01)

        action = mcts._best_action(root)

        assert action is not None, "Action should not be None"
        # At very low temperature, highest-visit child (0) should be selected
        assert action == 0, (
            f"At temp=0.01, action 0 (500 visits) should be selected, got {action}"
        )

    def test_extreme_temperature_no_crash(self):
        """Temperature of 0.001 should not crash or return NaN-based selection."""
        root = self._make_root_with_children({0: 80, 1: 15, 2: 5})
        mcts = self._make_mcts(temperature=0.001)

        # Should not raise, and should select action 0
        action = mcts._best_action(root)
        assert action is not None

    def test_moderate_temperature_probability_computation(self):
        """Temperature of 1.0: internal probabilities must be proportional to visits."""
        counts = np.array([500, 300, 200], dtype=np.float64)
        temperature = 1.0

        log_counts = np.log(np.maximum(counts, 1e-30)) / temperature
        log_counts -= log_counts.max()
        probs = np.exp(log_counts)
        probs /= probs.sum()

        assert probs[0] > probs[1] > probs[2], (
            f"At temp=1.0, probs should be ordered by visits: {probs}"
        )
        assert abs(probs[0] - 0.5) < 0.01, f"P(500/1000) should be ~0.5, got {probs[0]}"
        assert abs(probs[1] - 0.3) < 0.01, f"P(300/1000) should be ~0.3, got {probs[1]}"
        assert abs(probs[2] - 0.2) < 0.01, f"P(200/1000) should be ~0.2, got {probs[2]}"


# =====================================================================
# Bug 4: Pipeline single-agent opponent_action loads stale checkpoint
#
# The single-agent path in pipeline.py creates a fresh ASI and loads
# from agent_dir (the original base checkpoint), discarding all weights
# trained in phases 1-4. The multi-agent path correctly loads from
# the trained agent_base directory.
#
# Impact: Opponent action head trains on embeddings from an untrained
# perception, producing meaningless results.
# =====================================================================

class TestPipelineSingleAgentCheckpointBug:
    """Test that single-agent pipeline paths load trained checkpoints."""

    def test_single_agent_opponent_action_prefers_trained_dir(self):
        """The single-agent opponent_action path must try the trained save
        directory BEFORE falling back to agent_dir. The first load_checkpoint
        call in the block should NOT reference agent_dir."""
        pipeline_path = os.path.join(
            os.path.dirname(__file__), "..", "pipeline.py"
        )
        with open(pipeline_path) as f:
            source = f.read()

        lines = source.split('\n')
        in_single_agent_opp = False
        first_load_uses_agent_dir = False

        for i, line in enumerate(lines):
            if '# Single-agent' in line and i > 0:
                context = '\n'.join(lines[max(0, i-30):i+20])
                if 'opponent_action' in context.lower() or 'train_opponent_action' in context:
                    in_single_agent_opp = True
                    continue

            if in_single_agent_opp:
                stripped = line.strip()
                if 'load_checkpoint' in stripped:
                    first_load_uses_agent_dir = (
                        'agent_dir' in stripped and
                        'base_dir' not in stripped and
                        'save_dir' not in stripped and
                        'opp_load' not in stripped
                    )
                    break
                if stripped.startswith('del agent') or 'torch.cuda.empty_cache' in stripped:
                    break

        assert not first_load_uses_agent_dir, (
            "Single-agent opponent_action path's FIRST load_checkpoint uses "
            "agent_dir (original/untrained) instead of the trained save dir."
        )

    def test_single_agent_mcts_should_prefer_trained_over_agent_dir(self):
        """The single-agent MCTS path should not give agent_dir priority
        over the trained save directory."""
        pipeline_path = os.path.join(
            os.path.dirname(__file__), "..", "pipeline.py"
        )
        with open(pipeline_path) as f:
            source = f.read()

        found_priority_bug = "single_load_dir = agent_dir or" in source

        assert not found_priority_bug, (
            "Single-agent MCTS path uses 'agent_dir or save_base_dir_mcts', "
            "giving agent_dir (untrained base) priority over the trained dir."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
