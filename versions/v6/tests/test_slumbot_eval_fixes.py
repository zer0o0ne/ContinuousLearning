"""E2E tests for the 2026-07 slumbot_eval.py audit fixes.

Covered fixes (evaluation/slumbot_eval.py):
  1. Per-event board masking: each event's `table` is masked to that event's
     street (training convention `_get_table_display_from_turn`), instead of
     stamping the CURRENT board into every past event.
  2. B.5.6 preflop raise classification: open vs 3bet decided by the number
     of raises already made this street (all-in first raise = "open"),
     mirroring generate.py:896-911.
  3. Min-raise propagation: `_build_game_state` passes `last_raise_size`
     (scaled `state["last_bet_size"]`, floored at big blind) into GameState;
     `_make_solver_table_stub` carries `_last_full_raise_level` so
     GameState.from_table does not spuriously set short_allin_restricted.
  4. Opening snapshot pattern: replayed sequences start with TWO action=None
     snapshots (initial + first pre-decision), like generate.py:723-731 +
     813-822 and evaluate.py:565-572 + 798-806.
  5. search_scale: eval-time MCTS constructions receive
     norm_stats["mcts_value_scale"] (big-blind fallback), like collect.py:52.

All tests are deterministic: fixed seeds (conftest reseeds all RNGs), fixed
action strings, no probabilistic assertions.

Run (from versions/v6):
    python -m pytest tests/test_slumbot_eval_fixes.py -v
"""

import numpy as np
import torch

from agent.mcts.game_state import GameState
from env.table import Table
from evaluation.evaluate import (
    _get_table_display_from_turn,
    _rebuild_events,
)
from evaluation.slumbot_eval import (
    _action_idx_to_history_act_type,
    _build_events,
    _build_game_state,
    _load_one_agent,
    _make_snapshot,
    _make_solver_table_stub,
    _replay_action_string,
)

# Same shape as production config / existing slumbot tests.
RAISE_SIZES_PREFLOP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
RAISE_SIZES_FLOP = [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0]
RAISE_SIZES = {
    0: RAISE_SIZES_PREFLOP,
    1: RAISE_SIZES_FLOP,
    2: RAISE_SIZES_FLOP,
    3: RAISE_SIZES_FLOP,
}
N_RAISE_BINS = 11
N_ACTIONS = N_RAISE_BINS + 3

BIG_BLIND_INTERNAL = 10.0
SMALL_BLIND_INTERNAL = 5.0
CHIP_SCALE = 10.0  # Slumbot BB 100 / internal BB 10


def _replay(action_str, hole=(20, 21), board=(0, 4, 8, 12, 16), client_pos=1,
            hero_action_indices=None):
    return _replay_action_string(
        action_str, list(hole), list(board), client_pos,
        RAISE_SIZES, N_RAISE_BINS, N_ACTIONS,
        hero_action_indices if hero_action_indices is not None else [],
    )


# ============================================================================
# Fix 1 + Fix 4 e2e: Slumbot-replayed event sequence must equal the
# training-format sequence (evaluate.py._rebuild_events) for the same hand.
# ============================================================================

class TestEventSequenceMatchesTrainingFormat:
    """Play the identical hand through env.table.Table (training side) and
    through the Slumbot replay harness; the produced event sequences must be
    field-by-field identical (up to chip-scaling float error)."""

    # Hand: preflop SB limp, BB check / flop BB check, SB bets 0.5 pot
    # (exactly the 0.5 raise bin -> action_idx 6), BB calls / turn BB checks,
    # hero (SB, user pos 0) is now to act. Non-terminal, three streets seen.
    ACTION_STR = "ck/kb100c/k"
    # (user_pos, action_idx) in play order, mirrored on the Table side.
    ACTIONS = [(0, 1), (1, 1), (1, 1), (0, 6), (1, 1), (1, 1)]

    def _training_side(self):
        """generate.py/evaluate.py-style snapshots for the scripted hand."""
        np.random.seed(123)  # fixes table.deck
        table = Table(
            num_players=2,
            raise_sizes=[RAISE_SIZES_PREFLOP, RAISE_SIZES_FLOP,
                         RAISE_SIZES_FLOP, RAISE_SIZES_FLOP],
            start_credits=2000,
            big_blind=int(BIG_BLIND_INTERNAL),
            small_blind=int(SMALL_BLIND_INTERNAL),
        )
        table.start_table()

        def snap(action):
            return {
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": table.active_player,
                "action": action,
            }

        # Initial snapshot (generate.py:723-731)
        snapshots = [snap(None)]
        for expected_pos, action_idx in self.ACTIONS:
            assert table.active_player == expected_pos, (
                "test-script bug: unexpected actor")
            # Pre-decision snapshot (generate.py:813-822)
            snapshots.append(snap(None))
            action = torch.zeros(N_ACTIONS, dtype=torch.float32)
            action[action_idx] = 1.0
            table.step(action)
            # Post-action snapshot (generate.py:953-960)
            snapshots.append(snap(action))
        # Hero (user pos 0) decision snapshot for the pending decision
        assert table.active_player == 0
        snapshots.append(snap(None))

        events = _rebuild_events(
            snapshots, table.deck, hero_pos=0, num_players=2,
            big_blind=BIG_BLIND_INTERNAL, small_blind=SMALL_BLIND_INTERNAL,
            n_actions=N_ACTIONS, up_to=len(snapshots) - 1,
        )
        return events, table.deck

    def _slumbot_side(self, deck):
        """Replay the same hand from the Slumbot action string (chips x10)."""
        hole = [int(c) for c in deck[5:7]]        # hero = user pos 0 = SB
        board = [int(c) for c in deck[:5]]
        client_pos = 1                            # Slumbot frame: 1 = SB
        state, snapshots, _, _ = _replay(
            self.ACTION_STR, hole=hole, board=board, client_pos=client_pos)
        assert not state["is_terminal"]
        assert state["active_pos"] == client_pos  # hero to act
        # _play_one_hand appends the hero pre-decision snapshot
        snapshots = list(snapshots)
        snapshots.append(_make_snapshot(state, N_ACTIONS, None, client_pos))
        events = _build_events(
            snapshots, hole, board, hero_user_pos=0, client_pos=client_pos,
            num_players=2, big_blind_internal=BIG_BLIND_INTERNAL,
            small_blind_internal=SMALL_BLIND_INTERNAL,
            chip_scale=CHIP_SCALE, n_actions=N_ACTIONS,
        )
        return events

    def test_full_sequence_equal(self):
        train_events, deck = self._training_side()
        slumbot_events = self._slumbot_side(deck)

        # initial + 6 x (pre, post) + hero pre-decision = 14
        assert len(train_events) == 14
        assert len(slumbot_events) == len(train_events)

        for i, (se, te) in enumerate(zip(slumbot_events, train_events)):
            ctx = f"event {i}"
            assert [int(x) for x in se["hand"]] == \
                   [int(x) for x in te["hand"]], ctx
            assert se["num_players"] == te["num_players"], ctx
            assert se["hero_pos"] == te["hero_pos"], ctx
            assert se["acting_pos"] == te["acting_pos"], ctx
            assert se["big_blind"] == te["big_blind"], ctx
            assert se["small_blind"] == te["small_blind"], ctx
            # Fix 1: per-event board masked to the event's street
            assert [int(x) for x in se["table"]] == \
                   [int(x) for x in te["table"]], ctx
            assert abs(se["pot"] - te["pot"]) < 1e-4, ctx
            assert abs(se["stack"] - te["stack"]) < 1e-3, ctx
            np.testing.assert_allclose(
                np.asarray(se["bets"], dtype=np.float64),
                np.asarray(te["bets"], dtype=np.float64),
                atol=1e-3, err_msg=ctx)
            np.testing.assert_allclose(
                np.asarray(se["stacks"], dtype=np.float64),
                np.asarray(te["stacks"], dtype=np.float64),
                atol=1e-3, err_msg=ctx)
            assert torch.equal(se["action"], te["action"]), ctx

    def test_opens_with_two_action_none_events(self):
        """Fix 4: sequence starts with TWO action=None snapshots, like
        training (initial + first pre-decision)."""
        train_events, deck = self._training_side()
        slumbot_events = self._slumbot_side(deck)
        zero = torch.zeros(N_ACTIONS, dtype=torch.float32)
        for events in (train_events, slumbot_events):
            assert torch.equal(events[0]["action"], zero)
            assert torch.equal(events[1]["action"], zero)
            # third event is the first post-action snap (SB limp = call)
            assert float(events[2]["action"][1]) == 1.0


# ============================================================================
# Fix 1: board masking per event street (direct assertions)
# ============================================================================

class TestBoardMaskingPerStreet:
    BOARD = [0, 4, 8, 12, 16]

    def _events_and_snaps(self, action_str):
        state, snapshots, _, _ = _replay(action_str, board=self.BOARD)
        events = _build_events(
            snapshots, [20, 21], self.BOARD, hero_user_pos=0, client_pos=1,
            num_players=2, big_blind_internal=BIG_BLIND_INTERNAL,
            small_blind_internal=SMALL_BLIND_INTERNAL,
            chip_scale=CHIP_SCALE, n_actions=N_ACTIONS,
        )
        return events, snapshots

    def test_table_masked_to_snapshot_street(self):
        # Full hand to showdown: check/call only.
        events, snapshots = self._events_and_snaps("ck/kk/kk/kk")
        assert len(events) == len(snapshots)
        for i, (event, snap) in enumerate(zip(events, snapshots)):
            expected = _get_table_display_from_turn(self.BOARD, snap["turn"])
            assert [int(x) for x in event["table"]] == \
                   [int(x) for x in expected], f"event {i} (turn {snap['turn']})"

    def test_no_board_leak_into_preflop_events(self):
        """Regression: preflop events must show [-1]*5 even though the hand
        reached the river and the full board is known at build time."""
        events, snapshots = self._events_and_snaps("ck/kk/kk/kk")
        preflop = [e for e, s in zip(events, snapshots) if s["turn"] == 0]
        river = [e for e, s in zip(events, snapshots) if s["turn"] == 3]
        assert len(preflop) >= 2 and len(river) >= 1
        for event in preflop:
            assert [int(x) for x in event["table"]] == [-1] * 5
        for event in river:
            assert [int(x) for x in event["table"]] == self.BOARD

    def test_flop_and_turn_partial_masks(self):
        events, snapshots = self._events_and_snaps("ck/kk/k")
        flop = [e for e, s in zip(events, snapshots) if s["turn"] == 1]
        turn = [e for e, s in zip(events, snapshots) if s["turn"] == 2]
        assert flop and turn
        for event in flop:
            assert [int(x) for x in event["table"]] == [0, 4, 8, -1, -1]
        for event in turn:
            assert [int(x) for x in event["table"]] == [0, 4, 8, 12, -1]


# ============================================================================
# Fix 2: B.5.6 preflop open/3bet classification by street_raise_count
# ============================================================================

class TestPreflopRaiseClassification:
    def test_open_then_3bet_then_call(self):
        # SB (user 0) raises to 300 = open; BB (user 1) reraises to 900 =
        # 3bet; SB calls.
        _, _, _, history = _replay("b300b900c")
        assert history == [(0, "open"), (1, "3bet"), (0, "call")]

    def test_4bet_is_3bet_class(self):
        _, _, _, history = _replay("b300b900b2700c")
        assert history == [(0, "open"), (1, "3bet"), (0, "3bet"), (1, "call")]

    def test_allin_first_raise_is_open(self):
        """Regression: an all-in FIRST raise was labeled '3bet' by the old
        all-in-vs-sized rule; generate.py (B.5.6) classifies it 'open'."""
        _, _, _, history = _replay("b20000")
        assert history == [(0, "open")]

    def test_allin_over_raise_is_3bet(self):
        _, _, _, history = _replay("b300b20000")
        assert history == [(0, "open"), (1, "3bet")]

    def test_sized_raise_over_limp_is_open(self):
        # SB limps, BB raises: still the FIRST raise this street -> open.
        _, _, _, history = _replay("cb300")
        assert history == [(0, "call"), (1, "open")]

    def test_street_counter_resets_postflop(self):
        """Raise count resets on street change; postflop raises (first or
        re-raise) are all 'bet_postflop', and calls 'call_postflop'."""
        _, _, _, history = _replay("b300c/b300b900c")
        assert history == [
            (0, "open"), (1, "call"),
            (1, "bet_postflop"), (0, "bet_postflop"), (1, "call_postflop"),
        ]

    def test_classifier_mirrors_generate_rules(self):
        """Direct classifier checks against generate.py:896-911."""
        # fold -> None
        assert _action_idx_to_history_act_type(0, N_RAISE_BINS, 0, 0) is None
        # call: street-dependent
        assert _action_idx_to_history_act_type(1, N_RAISE_BINS, 0, 5) == "call"
        assert _action_idx_to_history_act_type(
            1, N_RAISE_BINS, 2, 0) == "call_postflop"
        # preflop: raise count decides, for sized AND all-in raises
        for raise_idx in (2, N_RAISE_BINS + 2):
            assert _action_idx_to_history_act_type(
                raise_idx, N_RAISE_BINS, 0, 0) == "open"
            assert _action_idx_to_history_act_type(
                raise_idx, N_RAISE_BINS, 0, 1) == "3bet"
            assert _action_idx_to_history_act_type(
                raise_idx, N_RAISE_BINS, 0, 3) == "3bet"
            # postflop: always bet_postflop
            assert _action_idx_to_history_act_type(
                raise_idx, N_RAISE_BINS, 1, 0) == "bet_postflop"
            assert _action_idx_to_history_act_type(
                raise_idx, N_RAISE_BINS, 1, 2) == "bet_postflop"


# ============================================================================
# Fix 3: min-raise propagation into GameState + stub _last_full_raise_level
# ============================================================================

class TestMinRaisePropagation:
    def test_last_raise_size_scaled_from_state(self):
        """Facing a 600-chip (Slumbot) pot bet, GameState.last_raise_size
        must be 60 internal chips — not the big-blind default."""
        state = {
            "pot": 1200, "bets": [600, 0], "credits": [19100, 19700],
            "players_state": [0, 1], "high_bet": 600, "last_bet_size": 600,
            "turn": 1, "active_pos": 1,
        }
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=CHIP_SCALE,
                               big_blind_internal=BIG_BLIND_INTERNAL)
        assert gs.last_raise_size == 60.0

    def test_legal_mask_excludes_sub_min_raises(self):
        """With last_raise_size propagated, sub-min-raise bins are illegal.
        Internal: call=60, effective_pot=120 -> increments 12/30/39.6/48 for
        the 0.1/0.25/0.33/0.4 bins (< 60 -> illegal); 0.5 -> 60 (legal)."""
        state = {
            "pot": 1200, "bets": [600, 0], "credits": [19100, 19700],
            "players_state": [0, 1], "high_bet": 600, "last_bet_size": 600,
            "turn": 1, "active_pos": 1,
        }
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=CHIP_SCALE,
                               big_blind_internal=BIG_BLIND_INTERNAL)
        legal = gs.get_legal_actions()
        for sub_min_idx in (2, 3, 4, 5):     # 0.1, 0.25, 0.33, 0.4
            assert sub_min_idx not in legal, f"bin {sub_min_idx} admitted"
        assert 6 in legal                    # 0.5 pot = exactly min-raise
        assert 0 in legal and 1 in legal     # fold/call facing a bet
        mask = gs.get_legal_action_mask(N_ACTIONS)
        assert mask[2] is False and mask[6] is True

    def test_preflop_floor_at_big_blind(self):
        """Preflop initial last_bet_size is BB-SB=50 Slumbot chips (5
        internal); Table semantics floor last_raise_size at the big blind."""
        state = {
            "pot": 150, "bets": [100, 50], "credits": [19900, 19950],
            "players_state": [1, 1], "high_bet": 100, "last_bet_size": 50,
            "turn": 0, "active_pos": 1,
        }
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=CHIP_SCALE,
                               big_blind_internal=BIG_BLIND_INTERNAL)
        assert gs.last_raise_size == BIG_BLIND_INTERNAL

    def test_missing_last_bet_size_falls_back_to_big_blind(self):
        state = {
            "pot": 600, "bets": [0, 0], "credits": [19700, 19700],
            "players_state": [1, 1], "high_bet": 0,
            "turn": 1, "active_pos": 0,
        }
        gs = _build_game_state(state, hero_user_pos=0,
                               raise_sizes=RAISE_SIZES,
                               n_raise_bins=N_RAISE_BINS,
                               chip_scale=CHIP_SCALE,
                               big_blind_internal=BIG_BLIND_INTERNAL)
        assert gs.last_raise_size == BIG_BLIND_INTERNAL


class TestSolverStubLastFullRaiseLevel:
    def _stub(self, state, hero_user_pos=0):
        return _make_solver_table_stub(
            hero_user_pos=hero_user_pos,
            hole_cards_int=[20, 21],
            board_ints=[0, 4, 8, -1, -1],
            state=state,
            raise_sizes=RAISE_SIZES,
            big_blind_internal=BIG_BLIND_INTERNAL,
            small_blind_internal=SMALL_BLIND_INTERNAL,
            chip_scale=CHIP_SCALE,
            num_players=2,
        )

    def test_level_follows_table_semantics(self):
        """Table semantics: preflop pre-raise level = BB; street change resets
        to 0.0; a full raise moves it to the new high_bet. All three cases
        collapse to `high_bet` scaled."""
        # Preflop, blinds only: high_bet=100 -> level 10 = BB internal
        preflop = {
            "pot": 150, "bets": [100, 50], "credits": [19900, 19950],
            "players_state": [1, 1], "high_bet": 100, "last_bet_size": 50,
            "turn": 0, "active_pos": 1,
        }
        assert self._stub(preflop)._last_full_raise_level == BIG_BLIND_INTERNAL
        # Flop, no bet yet: high_bet=0 -> level 0.0
        flop_open = {
            "pot": 600, "bets": [0, 0], "credits": [19700, 19700],
            "players_state": [1, 1], "high_bet": 0, "last_bet_size": 0,
            "turn": 1, "active_pos": 0,
        }
        assert self._stub(flop_open)._last_full_raise_level == 0.0
        # Flop after a raise to 600: level = 60
        flop_raised = {
            "pot": 1400, "bets": [600, 200], "credits": [19100, 19500],
            "players_state": [1, 1], "high_bet": 600, "last_bet_size": 400,
            "turn": 1, "active_pos": 1,
        }
        assert self._stub(flop_raised)._last_full_raise_level == 60.0

    def test_no_spurious_short_allin_restriction(self):
        """Regression: hero bet 200 on the flop and got raised to 600. With
        the stub lacking _last_full_raise_level, GameState.from_table
        defaulted it to big_blind -> short_allin_restricted stripped ALL
        raises from the legal mask. Hero was fully reopened and must be
        allowed to re-raise."""
        state = {
            "pot": 1400, "bets": [600, 200], "credits": [19100, 19500],
            "players_state": [1, 1], "high_bet": 600, "last_bet_size": 400,
            "turn": 1, "active_pos": 1,   # hero = Slumbot SB = user 0
        }
        stub = self._stub(state)
        gs = GameState.from_table(stub, 0)
        legal = gs.get_legal_actions()
        assert N_RAISE_BINS + 2 in legal, (
            "all-in stripped: short_allin_restricted misfired")
        assert any(2 <= a < N_RAISE_BINS + 2 for a in legal), (
            "all sized raise bins stripped: short_allin_restricted misfired")


# ============================================================================
# Fix 4: opening snapshot pattern (direct assertions on the replay)
# ============================================================================

class TestOpeningSnapshotPattern:
    def test_two_action_none_snapshots_before_first_action(self):
        _, snapshots, _, _ = _replay("ck")
        # initial + (pre, post) x 2 tokens
        assert len(snapshots) == 5
        assert snapshots[0]["action"] is None
        assert snapshots[1]["action"] is None   # first pre-decision snap
        assert snapshots[2]["action"] is not None
        assert float(snapshots[2]["action"][1]) == 1.0
        assert snapshots[3]["action"] is None
        assert snapshots[4]["action"] is not None

    def test_first_two_snapshots_identical_state(self):
        """The initial snap and the first pre-decision snap describe the same
        pre-action state (like generate.py's initial + first decision snap)."""
        _, snapshots, _, _ = _replay("ck")
        s0, s1 = snapshots[0], snapshots[1]
        assert s0["pot"] == s1["pot"]
        assert s0["turn"] == s1["turn"]
        assert s0["active_pos"] == s1["active_pos"]
        np.testing.assert_array_equal(s0["bets"], s1["bets"])
        assert s0["credits"] == s1["credits"]

    def test_empty_action_string_keeps_single_initial_snap(self):
        """With no actions replayed, only the initial snap exists; the hand
        loop then appends the hero pre-decision snap -> two action=None."""
        state, snapshots, _, _ = _replay("")
        assert len(snapshots) == 1
        assert snapshots[0]["action"] is None
        snapshots.append(_make_snapshot(state, N_ACTIONS, None, 1))
        assert snapshots[1]["action"] is None
        assert snapshots[0]["active_pos"] == snapshots[1]["active_pos"]

    def test_pre_post_pairing_all_tokens(self):
        """Every action contributes exactly one (pre None, post one-hot)
        pair after the initial snap — full-hand structural check."""
        _, snapshots, _, _ = _replay("ck/kk/kk/kk")
        assert len(snapshots) == 1 + 2 * 8
        assert snapshots[0]["action"] is None
        for k in range(8):
            assert snapshots[1 + 2 * k]["action"] is None, f"pre snap {k}"
            assert snapshots[2 + 2 * k]["action"] is not None, f"post snap {k}"


# ============================================================================
# Fix 5: search_scale wired into eval-time MCTS
# ============================================================================

_TINY_CFG = {
    "architecture": {
        "d_model": 32, "n_heads": 2, "n_kv_heads": 1,
        "n_encoder_layers": 1, "n_decoder_layers": 1,
        "n_value_layers": 1, "n_action_layers": 1,
        "n_opponent_action_layers": 1, "n_modelling_layers": 1,
        "d_ff": 64, "max_seq_len": 256, "max_players": 9,
        "modelling_dropout": 0.0,
        "memory": {"n_levels": 1, "max_cluster_size": 4,
                   "max_cluster_size_after": 4, "beam_width": 2},
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": RAISE_SIZES_PREFLOP, "flop": RAISE_SIZES_FLOP,
            "turn": RAISE_SIZES_FLOP, "river": RAISE_SIZES_FLOP,
        },
        "max_players": 9, "big_blind": 10, "max_stack": 1000,
    },
    "solver": {"type": "v1"},
    "mcts": {"n_simulations": 4},
}

_BASE_NORM_STATS = {
    "pot_mean": 0.0, "pot_std": 1.0,
    "stack_mean": 0.0, "stack_std": 1.0,
    "bets_mean": 0.0, "bets_std": 1.0,
    "blind_mean": 0.0, "blind_std": 1.0,
}


def _save_tiny_checkpoint(dir_path, norm_stats):
    from agent.agent import ASI
    asi = ASI(lambda m: None, config=_TINY_CFG)
    ckpt_path = dir_path / "best.pt"
    torch.save({
        "model_state_dict": asi.state_dict(),
        "norm_stats": norm_stats,
        "temperature": 0.4,
    }, str(ckpt_path))
    return str(dir_path)


class TestMctsSearchScale:
    def test_search_scale_from_checkpoint_norm_stats(self, tmp_path):
        """_load_one_agent must pass norm_stats['mcts_value_scale'] into the
        MCTS construction (collect.py:52 derivation)."""
        agent_dir = tmp_path / "agent_a"
        agent_dir.mkdir()
        norm_stats = dict(_BASE_NORM_STATS, mcts_value_scale=3.75)
        path = _save_tiny_checkpoint(agent_dir, norm_stats)

        bundle = _load_one_agent(
            {"name": "tiny", "path": path, "use_mcts": True},
            _TINY_CFG, "cpu", project_root=str(tmp_path), version="v6",
            fallback_temperature=0.5, log=lambda m: None,
        )
        assert bundle is not None and bundle["mcts"] is not None
        assert bundle["mcts"].search_scale == 3.75

    def test_search_scale_falls_back_to_internal_big_blind(self, tmp_path):
        """Without mcts_value_scale in the checkpoint, search_scale falls back
        to the internal big blind (game.big_blind = 10), like collect.py."""
        agent_dir = tmp_path / "agent_b"
        agent_dir.mkdir()
        path = _save_tiny_checkpoint(agent_dir, dict(_BASE_NORM_STATS))

        bundle = _load_one_agent(
            {"name": "tiny_fb", "path": path, "use_mcts": True},
            _TINY_CFG, "cpu", project_root=str(tmp_path), version="v6",
            fallback_temperature=0.5, log=lambda m: None,
        )
        assert bundle is not None and bundle["mcts"] is not None
        assert bundle["mcts"].search_scale == 10.0
