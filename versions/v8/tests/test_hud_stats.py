"""The stat line a member is read by (`PLAN_PROCEDURAL_POOL.md` §P2, ⚠6).

Every hand is scripted to an exact line, and every assertion is a
`(numerator, denominator)` pair small enough to count by hand from the line
above it. That is the only way this file is worth anything: a stat module that
is checked against itself, or against a ratio computed the same way it was, will
happily agree with its own mistake.

The lines deliberately include the three spots the §P4 additions are told apart
by — a check-raise, an unopened button steal, and an overbet — because those are
the stats with no second source in the tree.
"""

from pool.stats import hud_stats, steal_seats
from tests.test_situation import (ALL_IN, CALL, DOUBLE_POT, FOLD, HALF_POT,
                                  POT, scripted)


def only(seat):
    return lambda _record, s: s == seat


#: seats: 0 SB, 1 BB, 2 UTG, 3, 4 cutoff, 5 button. Preflop acts 2,3,4,5,0,1.
#: UTG opens, the cutoff 3-bets, everybody else folds and UTG folds too.
OPEN_AND_THREE_BET = [POT, FOLD, POT, FOLD, FOLD, FOLD, FOLD]

#: UTG opens, the cutoff and the big blind call. Flop: the big blind checks, UTG
#: c-bets, the cutoff calls, the big blind folds. Turn: UTG barrels, the cutoff
#: calls. River: both check it down.
CBET_AND_BARREL = [POT, FOLD, CALL, FOLD, FOLD, CALL,
                   CALL, HALF_POT, CALL, FOLD,
                   HALF_POT, CALL]

#: The same pot, but the big blind check-raises the c-bet to twice the pot.
CHECK_RAISE = [POT, FOLD, CALL, FOLD, FOLD, CALL,
               CALL, HALF_POT, FOLD, DOUBLE_POT]


# ---------------------------------------------------------------------------
# Preflop
# ---------------------------------------------------------------------------

def test_the_preflop_line_of_an_opener_and_a_three_bettor():
    record = scripted(OPEN_AND_THREE_BET)

    utg = hud_stats([record], only(2))
    assert utg["vpip"] == (1, 1) and utg["pfr"] == (1, 1)
    assert utg["limp"] == (0, 1)
    assert utg["threebet"] == (0, 0)          # it opened; it never faced one raise
    assert utg["fold_to_threebet"] == (1, 1)

    cutoff = hud_stats([record], only(4))
    assert cutoff["threebet"] == (1, 1)
    assert cutoff["fold_to_threebet"] == (0, 0)
    assert cutoff["vpip"] == (1, 1) and cutoff["pfr"] == (1, 1)

    button = hud_stats([record], only(5))
    assert button["vpip"] == (0, 1) and button["pfr"] == (0, 1)
    assert button["threebet"] == (0, 0)       # it folded facing *two* raises


def test_a_limp_is_not_a_raise_and_is_not_a_fold():
    record = scripted([CALL, FOLD, FOLD, FOLD, FOLD, CALL])
    utg = hud_stats([record], only(2))
    assert utg["vpip"] == (1, 1) and utg["pfr"] == (0, 1)
    assert utg["limp"] == (1, 1)

    big_blind = hud_stats([record], only(1))
    assert big_blind["vpip"] == (0, 1)        # checking the option is free
    assert big_blind["limp"] == (0, 1)


def test_a_steal_is_an_unopened_pot_from_the_three_late_seats():
    assert steal_seats(6) == {0, 4, 5}
    assert steal_seats(3) == {0, 2}           # no cutoff at three seats
    assert steal_seats(2) == {0}              # heads-up the SB is the button

    stolen = scripted([FOLD, FOLD, FOLD, POT, FOLD, FOLD])
    assert hud_stats([stolen], only(5))["steal"] == (1, 1)

    limped_in = scripted([FOLD, FOLD, FOLD, CALL, FOLD, CALL])
    assert hud_stats([limped_in], only(5))["steal"] == (0, 1)

    # A limper ahead of the button ends the opportunity rather than making one.
    entered = scripted([CALL, FOLD, FOLD, POT, FOLD, CALL])
    assert hud_stats([entered], only(5))["steal"] == (0, 0)
    assert hud_stats([entered], only(2))["steal"] == (0, 0)   # UTG never steals


def test_the_small_blind_steals_heads_up():
    record = scripted([POT, FOLD], num_players=2)
    assert hud_stats([record], only(0))["steal"] == (1, 1)
    assert hud_stats([record], only(1))["steal"] == (0, 0)


# ---------------------------------------------------------------------------
# Postflop
# ---------------------------------------------------------------------------

def test_a_c_bet_two_barrels_and_the_hands_that_faced_them():
    record = scripted(CBET_AND_BARREL)

    utg = hud_stats([record], only(2))
    assert utg["cbet_flop"] == (1, 1)
    assert utg["barrel_turn"] == (1, 1)
    assert utg["barrel_river"] == (0, 1)      # it gave up on the river
    assert utg["af"] == (2, 0)                # two bets, no calls
    assert utg["agg_pct"] == (2, 3)           # …and one check
    assert utg["fold_to_cbet"] == (0, 0)      # it made the c-bet
    assert utg["check_raise"] == (0, 0)

    cutoff = hud_stats([record], only(4))
    assert cutoff["fold_to_cbet"] == (0, 1)
    assert cutoff["cbet_flop"] == (0, 0)
    assert cutoff["af"] == (0, 2)
    assert cutoff["agg_pct"] == (0, 3)

    big_blind = hud_stats([record], only(1))
    assert big_blind["fold_to_cbet"] == (1, 1)
    assert big_blind["check_raise"] == (0, 1)     # it checked, then folded


def test_a_check_raise_and_an_overbet():
    record = scripted(CHECK_RAISE)

    big_blind = hud_stats([record], only(1))
    assert big_blind["check_raise"] == (1, 1)
    # The pot was 120 when the check-raise went in and the raise put in 280.
    assert big_blind["overbet_pct"] == (1, 1)
    assert big_blind["af"] == (1, 0)

    utg = hud_stats([record], only(2))
    assert utg["cbet_flop"] == (1, 1)
    assert utg["overbet_pct"] == (0, 1)       # a half-pot bet is not an overbet
    assert utg["check_raise"] == (0, 0)


def test_showdown_stats_count_who_got_there_and_who_won():
    record = scripted(CBET_AND_BARREL)
    assert set(record.showdown) == {2, 4}

    for seat in (2, 4):
        stats = hud_stats([record], only(seat))
        assert stats["wtsd"] == (1, 1)
        assert stats["wsd"] == (int(record.rewards[seat] > 0), 1)

    folded = hud_stats([record], only(1))
    assert folded["wtsd"] == (0, 1)           # it saw the flop and folded on it
    assert folded["wsd"] == (0, 0)

    never_saw_it = hud_stats([record], only(3))
    assert never_saw_it["wtsd"] == (0, 0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_counts_add_up_over_hands_and_over_seats():
    records = [scripted(OPEN_AND_THREE_BET), scripted(CBET_AND_BARREL)]

    per_seat = [hud_stats(records, only(s)) for s in range(6)]
    everyone = hud_stats(records)
    for name in everyone:
        assert everyone[name] == (sum(s[name][0] for s in per_seat),
                                  sum(s[name][1] for s in per_seat))

    # Twelve seats were dealt in over the two hands, and every one of them had
    # a preflop decision.
    assert everyone["vpip"][1] == 12
    assert everyone["pfr"] == (3, 12)


def test_a_seat_that_never_acts_is_not_counted():
    record = scripted([ALL_IN], num_players=2, stack_bb=10)
    assert hud_stats([record], only(0))["vpip"] == (1, 1)
    assert hud_stats([record], lambda _r, s: s == 7) == {
        name: (0, 0) for name in hud_stats([record])}


def test_no_hand_is_needed_to_report_nothing():
    empty = hud_stats([])
    assert set(empty) and all(v == (0, 0) for v in empty.values())
