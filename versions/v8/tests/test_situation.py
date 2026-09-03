"""Reading a decision into scalars, and the ranges that follow
(`PLAN_PROCEDURAL_POOL.md` §P2).

Every hand below is *scripted*: the driver's forced-prefix plumbing plays an
exact line, so the numbers asserted are the ones arithmetic gives and not
whatever a policy happened to do. Table sizes and stacks are chosen to make the
quantity under test unambiguous — a 6-max hand for position, heads-up for the
blinds, ten big blinds for an all-in.

The one property with no numbers in it is the last test: a `Situation` built on
the finished record equals the one built on the record truncated at that
decision. That is §9's future leak wearing a different hat, and the posterior
asks about decisions from the middle of finished hands all day.
"""

import numpy as np
import pytest

from env.driver import HandSpec, LockstepDriver
from env.showdown import label_showdowns
from pool.situation import (PreflopAction, Situation, moves,
                            opening_position, opening_range,
                            opponent_ranges, position_fraction, situation,
                            street_order)
from pool.strength import COMBO_CLASS, preflop_equity_table, preflop_rank_pct
from tests.g1_fixtures import (BIG_BLIND, N_ACTIONS, RAISE_SIZES, SMALL_BLIND,
                               contexts_from)
from pool.degenerate import AlwaysCall
from pool.style import StyleParams

FOLD, CALL, HALF_POT, POT, DOUBLE_POT, ALL_IN = 0, 1, 2, 3, 4, 5

#: UTG limps, seat 3 raises, the cutoff re-raises, the button and the small
#: blind fold, the big blind calls, UTG folds, seat 3 calls.
LIMP_RAISE_RERAISE = [CALL, POT, POT, FOLD, FOLD, CALL, FOLD, CALL]


def deck_with(board):
    """A 52-card permutation whose first cards are `board`, rest in order."""
    board = [int(c) for c in board]
    assert len(set(board)) == len(board), f"duplicate cards in {board}"
    return np.array(board + [c for c in range(52) if c not in board],
                    dtype=np.int64)


def scripted(actions, num_players=6, stack_bb=100, seed=4242, deck=None):
    """Play one hand along an exact line, then let it finish by calling.

    `deck` pins the cards, so a test that is about a board — a wet flop, a
    blank river — names it instead of hunting for a seed that deals it.
    """
    spec = HandSpec(
        num_players=num_players,
        start_credits=[float(stack_bb * BIG_BLIND)] * num_players,
        seat_members=[0] * num_players, seed=seed,
        big_blind=BIG_BLIND, small_blind=SMALL_BLIND, raise_sizes=RAISE_SIZES,
        forced_actions=list(actions), deck=deck)
    pool = [AlwaysCall("call", N_ACTIONS, StyleParams.identity())]
    record = LockstepDriver(pool, N_ACTIONS).run([spec])[0]
    label_showdowns([record])
    return record


def at(record, k):
    """The `Situation` of decision `k` of a played hand."""
    return situation(contexts_from([record])[k])


def at_seat(record, street, seat, nth=0):
    """The `Situation` of a seat's `nth` decision on one street."""
    hits = [c for c in contexts_from([record])
            if c.turn == street and c.acting_pos == seat]
    return situation(hits[nth])


#: UTG opens, one caller and the big blind, then a checked-to c-bet. Preflop the
#: smallest legal raise is a pot-sized one: half the pot is under a big blind.
#: seats: 0 SB, 1 BB, 2 UTG, 3, 4 CO, 5 button. Preflop acts 2,3,4,5,0,1.
SIX_MAX_LINE = [POT, FOLD, CALL, FOLD, FOLD, CALL,          # preflop
                CALL, HALF_POT, CALL, CALL]                  # flop


# ---------------------------------------------------------------------------
# Seat order and position
# ---------------------------------------------------------------------------

def test_position_and_acting_order_follow_the_engine():
    assert street_order(0, 6) == [2, 3, 4, 5, 0, 1]
    assert street_order(1, 6) == [0, 1, 2, 3, 4, 5]
    assert street_order(0, 2) == [0, 1]          # heads-up the button acts first
    assert street_order(1, 2) == [1, 0]

    assert position_fraction(0, 6) == 0.0
    assert position_fraction(1, 6) == pytest.approx(0.2)
    assert position_fraction(5, 6) == 1.0
    assert position_fraction(0, 2) == 1.0        # heads-up the SB is the button
    assert position_fraction(1, 2) == 0.0


# ---------------------------------------------------------------------------
# The situation
# ---------------------------------------------------------------------------

def test_a_six_max_hand_reads_the_way_the_table_looks():
    record = scripted(SIX_MAX_LINE)

    flop_bb = at(record, 6)                       # BB, first to act on the flop
    assert flop_bb.street == 1 and flop_bb.hero == 1
    assert flop_bb.live == (1, 2, 4) and flop_bb.n_live == 3
    assert flop_bb.pf_aggressor == 2 and not flop_bb.hero_is_pf_aggressor
    assert flop_bb.n_raises_pre == 1 and flop_bb.n_limpers == 0
    assert flop_bb.pos_frac == pytest.approx(0.2)
    assert flop_bb.n_behind == 2                  # UTG and the cutoff
    assert flop_bb.facing == 0.0 and not flop_bb.checked_to_hero
    assert flop_bb.n_bets_street == 0
    assert flop_bb.opp_pf_action == {
        0: PreflopAction.FOLD, 1: PreflopAction.CALL, 2: PreflopAction.RAISE,
        3: PreflopAction.FOLD, 4: PreflopAction.CALL, 5: PreflopAction.FOLD}

    flop_utg = at(record, 7)                      # the c-bet, checked to
    assert flop_utg.hero == 2 and flop_utg.checked_to_hero
    assert flop_utg.hero_is_pf_aggressor and flop_utg.last_aggressor == 2
    assert flop_utg.hero_barrels == 0

    flop_co = at(record, 8)                       # the cutoff, closing the action
    assert flop_co.hero == 4 and flop_co.n_behind == 0
    # A half-pot bet: the pot a snapshot carries already contains it.
    assert flop_co.facing == pytest.approx(0.5)
    assert flop_co.pot_odds == pytest.approx(0.25)
    assert flop_co.n_bets_street == 1 and not flop_co.checked_to_hero


def test_barrelling_is_counted_from_the_previous_streets():
    record = scripted(SIX_MAX_LINE + [CALL, HALF_POT])   # turn: BB checks, UTG bets
    turn_utg = at_seat(record, 2, 2)
    assert turn_utg.hero_bet_prev_street and turn_utg.hero_barrels == 1
    assert turn_utg.last_aggressor == 2

    river_utg = at_seat(record, 3, 2)
    assert river_utg.hero_barrels == 2
    assert river_utg.last_aggressor == 2

    # The flop c-bet itself follows a preflop raise, which is not a barrel.
    assert at_seat(record, 1, 2).hero_barrels == 0


def test_a_heads_up_three_bet_pot():
    record = scripted([POT, POT, CALL], num_players=2)

    facing_the_3bet = at(record, 2)               # the SB, facing the re-raise
    assert facing_the_3bet.hero == 0 and facing_the_3bet.is_sb
    assert facing_the_3bet.pos_frac == 1.0
    assert facing_the_3bet.n_raises_pre == 2
    assert facing_the_3bet.pf_aggressor == 1
    assert facing_the_3bet.opp_pf_action[1] == PreflopAction.RERAISE

    flop = at(record, 3)                          # BB is first to act postflop
    assert flop.hero == 1 and flop.is_bb and flop.n_behind == 1
    # 5 + 10 blinds, raised to 20, re-raised to 40, called: an 80-chip pot
    # with 960 behind on both sides.
    assert flop.pot_bb == pytest.approx(8.0)
    assert flop.eff_stack_bb == pytest.approx(96.0)
    assert flop.spr == pytest.approx(12.0)


def test_facing_an_all_in():
    record = scripted([ALL_IN], num_players=2, stack_bb=10)
    sit = at(record, 1)
    assert sit.hero == 1 and sit.facing_allin
    # 100-chip stacks: the shove is 100, the pot 110, the call 90.
    assert sit.pot_odds == pytest.approx(90.0 / 200.0)
    assert sit.facing == pytest.approx(90.0 / 20.0)
    assert sit.n_behind == 0


def test_every_preflop_line_is_recognised():
    # UTG limps, seat 3 raises, the cutoff re-raises, then the rest.
    record = scripted(LIMP_RAISE_RERAISE)

    facing = at(record, 3)                        # the button, facing the 3-bet
    assert facing.opp_pf_action[2] == PreflopAction.LIMP
    assert facing.opp_pf_action[3] == PreflopAction.RAISE
    assert facing.opp_pf_action[4] == PreflopAction.RERAISE
    assert facing.opp_pf_action[0] == PreflopAction.UNOPENED
    assert facing.opp_pf_action[1] == PreflopAction.UNOPENED
    assert facing.n_limpers == 1 and facing.n_raises_pre == 2

    later = at(record, 6)                         # UTG, facing the 3-bet
    assert later.opp_pf_action[5] == PreflopAction.FOLD
    assert later.opp_pf_action[1] == PreflopAction.CALL
    assert later.live == (1, 2, 3, 4)


def test_the_situation_is_a_function_of_the_prefix():
    """A finished record and one truncated at the decision read the same."""
    from dataclasses import replace

    record = scripted(SIX_MAX_LINE)
    contexts = contexts_from([record])
    for ctx in contexts:
        cut = replace(
            record, snapshots=record.snapshots[:ctx.snap_idx + 1],
            decisions=[d for d in record.decisions
                       if int(d["snap_idx"]) < int(ctx.snap_idx)],
            showdown=[], showdown_strength={}, showdown_class={})
        short = contexts_from([cut])
        rebuilt = type(ctx)(cut, ctx.snap_idx, ctx.acting_pos, ctx.legal_mask,
                            ctx.turn)
        assert situation(rebuilt) == situation(ctx)
        assert len(short) == len(cut.decisions)


def test_moves_stop_where_they_are_told_to():
    record = scripted(SIX_MAX_LINE)
    ctx = contexts_from([record])[7]
    assert len(moves(record)) == len(record.decisions)
    assert len(moves(record, upto_snap=ctx.snap_idx)) == 7
    assert all(m.street == 0 for m in moves(record, upto_snap=ctx.snap_idx)[:6])


# ---------------------------------------------------------------------------
# Preflop-implied ranges
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rank_pct(tmp_path_factory):
    path = tmp_path_factory.mktemp("tables") / "preflop_equity.npy"
    table = preflop_equity_table(str(path), seed=0, n_deals=100_000)
    return lambda n_opps: preflop_rank_pct(table, n_opps)


def _aces(weights):
    from env.showdown import hand_class_169
    aa = hand_class_169(12 * 4, 12 * 4 + 1)
    return weights[np.flatnonzero(COMBO_CLASS == aa)]


def test_an_opening_range_widens_with_position():
    assert opening_range(2, 6) < opening_range(5, 6)
    assert opening_range(5, 6) == pytest.approx(0.45)
    assert opening_range(0, 2) == pytest.approx(0.75)

    # Preflop the small blind acts second to last, so it opens a button's
    # range — even though postflop it is the earliest seat there is.
    assert opening_position(0, 6) == 1.0 and position_fraction(0, 6) == 0.0
    assert opening_range(0, 6) == opening_range(5, 6)
    assert opening_range(0, 9) == opening_range(8, 9)


def test_ranges_follow_the_preflop_line(rank_pct):
    early = scripted([POT, FOLD, FOLD, FOLD, FOLD, CALL])   # UTG opens
    late = scripted([FOLD, FOLD, FOLD, POT, FOLD, CALL])    # button opens

    flop_early = opponent_ranges(at(early, 6), rank_pct)
    flop_late = opponent_ranges(at(late, 6), rank_pct)
    assert flop_late[5].sum() > flop_early[2].sum()

    # The big blind is hero on both flops, and every live opponent has a range.
    assert set(flop_early) == {2} and set(flop_late) == {5}


def test_a_three_bet_range_is_polar_and_a_call_range_is_not(rank_pct):
    record = scripted(LIMP_RAISE_RERAISE)
    ranges = opponent_ranges(at(record, 5), rank_pct)   # the big blind's view

    three_bet = ranges[4]
    assert set(np.unique(three_bet)) == {0.0, 0.5, 1.0}
    strong = np.flatnonzero(three_bet == 1.0)
    bluff = np.flatnonzero(three_bet == 0.5)
    assert len(strong) and len(bluff)
    assert not set(strong) & set(bluff)
    assert (_aces(three_bet) == 1.0).all()

    limped = ranges[2]
    assert set(np.unique(limped)) == {0.0, 1.0}
    assert limped.sum() > three_bet.sum()

    caller = opponent_ranges(at(record, 6), rank_pct)[1]
    assert (_aces(caller) == 0.0).all()           # no premium in a call range
    assert caller.sum() > 0


def test_an_unopened_seat_is_the_whole_range(rank_pct):
    record = scripted([POT, CALL, FOLD, FOLD, FOLD, CALL])
    ranges = opponent_ranges(at(record, 4), rank_pct)   # SB, facing an open
    assert (ranges[1] == 1.0).all()               # the big blind, yet to act
    assert ranges[2].sum() < len(ranges[2])


def test_ranges_refuse_a_percentile_of_the_wrong_shape():
    record = scripted(SIX_MAX_LINE)
    with pytest.raises(AssertionError):
        opponent_ranges(at(record, 6), lambda n: np.zeros(169))
