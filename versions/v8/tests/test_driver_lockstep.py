"""The lock-step driver (CONCEPT.md §3, §15).

The property that matters: **N hands in lock-step produce exactly the same
trajectories as N sequential hands under the same seeds.** Getting this wrong is
silent — the oracle would keep producing plausible EVs that are biased by
whatever the batching changed.

The pool used here is degenerate strategies only, whose policies are exact
numpy, so "the same" can be asserted bit for bit rather than within a tolerance.
"""

import numpy as np
import pytest

from env.legal import legal_action_mask
from env.table import Table
from tests.g1_fixtures import (
    BIG_BLIND, N_ACTIONS, RAISE_SIZES, make_pool, make_specs, play,
)


def _trajectory(record):
    return [(d["snap_idx"], d["acting_pos"], d["member"], d["action_idx"])
            for d in record.decisions]


def test_lockstep_matches_sequential():
    pool = make_pool()
    specs = make_specs(seed=1, n_hands=40, n_members=len(pool))

    full = play(pool, specs, batch_size=len(specs))
    sequential = play(pool, specs, batch_size=1)
    chunked = play(pool, specs, batch_size=7)

    for a, b, c in zip(full, sequential, chunked):
        assert _trajectory(a) == _trajectory(b) == _trajectory(c)
        assert np.array_equal(a.deck, b.deck) and np.array_equal(a.deck, c.deck)
        assert np.array_equal(a.rewards, b.rewards)
        assert np.array_equal(a.rewards, c.rewards)
        assert len(a.snapshots) == len(b.snapshots) == len(c.snapshots)


def test_chips_are_conserved_over_every_hand():
    """Chips in = chips out. The engine guarantees it; the driver must not
    break it by mishandling the all-in runout."""
    pool = make_pool()
    records = play(pool, make_specs(seed=2, n_hands=60, n_members=len(pool)))
    for r in records:
        assert abs(float(r.rewards.sum())) < 1e-9, (
            f"{r.num_players}-handed hand leaked {r.rewards.sum()} chips")


def test_every_sampled_action_was_legal():
    pool = make_pool()
    records = play(pool, make_specs(seed=3, n_hands=60, n_members=len(pool)))
    seen_any = False
    for r in records:
        for d in r.decisions:
            seen_any = True
            assert d["legal_mask"][d["action_idx"]], (
                "driver sampled an action outside the legal mask")
            assert d["legal_mask"][1], "call/check is always playable"
    assert seen_any


def test_snapshot_convention_matches_v7():
    """Initial snap, then a (pre-decision, post-action) pair per decision."""
    pool = make_pool()
    records = play(pool, make_specs(seed=4, n_hands=30, n_members=len(pool)))
    for r in records:
        assert r.snapshots[0]["action"] is None
        assert len(r.snapshots) == 1 + 2 * len(r.decisions)
        for i, d in enumerate(r.decisions):
            pre = r.snapshots[d["snap_idx"]]
            post = r.snapshots[d["snap_idx"] + 1]
            assert d["snap_idx"] == 1 + 2 * i
            assert pre["action"] is None
            assert pre["active_pos"] == d["acting_pos"], (
                "a pre-decision snapshot names the player on turn")
            assert post["action"] is not None
            assert int(np.argmax(post["action"])) == d["action_idx"]


def test_a_hand_never_exceeds_the_max_actions_cap():
    pool = make_pool()
    records = play(pool, make_specs(seed=5, n_hands=40, n_members=len(pool)))
    for r in records:
        assert len(r.decisions) <= 6 * r.num_players + 8


def test_heads_up_and_nine_handed_both_run():
    """Nothing in the driver may assume a table size (CLAUDE.md §1)."""
    pool = make_pool()
    for n in range(2, 10):
        records = play(pool, make_specs(seed=6 + n, n_hands=6,
                                        n_members=len(pool), num_players=n))
        assert all(r.num_players == n for r in records)
        assert all(abs(float(r.rewards.sum())) < 1e-9 for r in records)


def test_short_and_deep_stacks_both_run():
    pool = make_pool()
    for stack_bb in (10, 25, 100, 300):
        records = play(pool, make_specs(seed=100 + stack_bb, n_hands=8,
                                        n_members=len(pool), stack_bb=stack_bb))
        assert all(abs(float(r.rewards.sum())) < 1e-9 for r in records)


# --------------------------------------------------------------- legality rule


def _table_at(num_players, credits, bets, pot, high_bet, turn, active,
              players_state, last_raise_size=None, last_full=None):
    t = Table(num_players=num_players, raise_sizes=RAISE_SIZES,
              start_credits=[float(c) for c in credits],
              big_blind=BIG_BLIND, small_blind=BIG_BLIND / 2)
    t.credits = [float(c) for c in credits]
    t.bets = np.asarray(bets, dtype=float)
    t.pot = float(pot)
    t.high_bet = float(high_bet)
    t.turn = int(turn)
    t.active_player = int(active)
    t.players_state = np.asarray(players_state, dtype=float)
    t.last_raise_size = float(BIG_BLIND if last_raise_size is None
                              else last_raise_size)
    t._last_full_raise_level = float(0.0 if last_full is None else last_full)
    t.several_all_in = False
    return t


def test_fold_is_dropped_when_checking_is_free():
    t = _table_at(3, [1000, 1000, 1000], [0, 0, 0], 30, 0, 1, 0, [1, 1, 1])
    mask = legal_action_mask(t, N_ACTIONS)
    assert not mask[0], "folding for free is strictly dominated by checking"
    assert mask[1]


def test_fold_is_available_when_facing_a_bet():
    t = _table_at(3, [1000, 900, 1000], [0, 100, 0], 130, 100, 2, 0, [1, 1, 1])
    assert legal_action_mask(t, N_ACTIONS)[0]


def test_no_raise_when_every_other_live_player_is_all_in():
    t = _table_at(3, [1000, 0, 0], [0, 500, 500], 1030, 500, 2, 0, [1, 2, 2])
    mask = legal_action_mask(t, N_ACTIONS)
    assert mask[0] and mask[1]
    assert not mask[2:].any(), (
        "raising into players who are all-in can only return uncalled chips")


def test_short_all_in_does_not_reopen_the_betting():
    """C.7.5: a player who already matched the last full raise level may only
    call or fold after a short all-in pushed the high bet above it."""
    t = _table_at(2, [1000, 0], [200, 260], 460, 260, 1, 0, [1, 2],
                  last_raise_size=100, last_full=200)
    mask = legal_action_mask(t, N_ACTIONS)
    assert mask[0] and mask[1]
    assert not mask[2:].any()


def test_a_stack_shorter_than_the_call_can_only_call_or_fold():
    t = _table_at(2, [30, 900], [0, 100], 130, 100, 0, 0, [1, 1])
    mask = legal_action_mask(t, N_ACTIONS)
    assert mask[0] and mask[1]
    assert not mask[2:].any()


def test_the_all_in_action_is_offered_when_no_sized_raise_fits():
    """A stack too small for any sized raise still has the shove."""
    t = _table_at(2, [120, 900], [0, 100], 200, 100, 0, 0, [1, 1])
    mask = legal_action_mask(t, N_ACTIONS)
    assert mask[N_ACTIONS - 1]
    assert not mask[2:N_ACTIONS - 1].any()


@pytest.mark.parametrize("num_players", [2, 5, 9])
def test_call_is_always_playable(num_players):
    credits = [1000.0] * num_players
    bets = [0.0] * num_players
    t = _table_at(num_players, credits, bets, 15, 0, 0, 0, [1] * num_players)
    assert legal_action_mask(t, N_ACTIONS)[1]
