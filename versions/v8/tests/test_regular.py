"""The rule cascade (`PLAN_PROCEDURAL_POOL.md` §P3).

Everything here goes through `policy()` — the same entry the driver and the
posterior use — and never through an internal. The situation helpers are
imported only to build the *reference* weights a frequency is measured against:
"this archetype defends half its range" is a statement about a range, so the
range has to be named to check it.

Two kinds of assertion, and the difference matters:

* **identities**, which hold by construction and are checked to a few
  thousandths — the defence frequency is `defend_factor/(1 + bet/pot)` of hero's
  own range mass, the river bluff share is the size's own indifference ratio,
  a batch equals the concatenation of single calls;
* **orderings**, which are what an archetype *is* — a button opens more combos
  than a first seat, a dry board is c-bet more than a wet one, six players are
  played more tightly than one.

A frequency that is neither is not asserted at all: §P3's "the c-bet mass is
`cbet_dry` ± 0.03", for instance, cannot hold, because value hands bet at
`1 − slowplay` whatever the c-bet frequency is, so the mass is always above it.
What is pinned instead is the part that does equal the knob — the bluffs.
"""

import numpy as np
import pytest

from env.driver import DecisionContext
from pool.regular import (RegularMember, RegularParams, call_range,
                          push_range)
from pool.situation import combo_percentile, preflop_range, situation
from pool.strength import (ALL_COMBOS, COMBO_CLASS, StrengthCache,
                           preflop_equity_table)
from pool.style import StyleParams
from env.showdown import hand_class_169
from tests.g1_fixtures import (N_ACTIONS, RAISE_SIZES, contexts_from,
                               make_specs, play)
from tests.test_situation import (ALL_IN, CALL, DOUBLE_POT, FOLD, HALF_POT,
                                  POT, deck_with, scripted)
from tests.test_strength import card, cards

#: heads-up: the small blind opens, the big blind calls, the big blind checks.
HU_TO_FLOP = [POT, CALL, CALL]
#: …then a c-bet, a call, a check, a barrel, a call, and a check to the river.
HU_TO_RIVER = HU_TO_FLOP + [HALF_POT, CALL, CALL, HALF_POT, CALL, CALL]


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    path = tmp_path_factory.mktemp("tables") / "preflop_equity.npy"
    table = preflop_equity_table(str(path), seed=0, n_deals=100_000)
    return table, StrengthCache(512)


def build(parts, **knobs):
    table, cache = parts
    return RegularMember("reg", N_ACTIONS, RegularParams(**knobs), cache,
                         table, RAISE_SIZES)


def every_combo(member, ctx):
    """`(1326, n_actions)` — what this member does with every possible hand."""
    batch = [DecisionContext(ctx.record, ctx.snap_idx, ctx.acting_pos,
                             ctx.legal_mask, ctx.turn,
                             hole_override=list(ALL_COMBOS[i]))
             for i in range(len(ALL_COMBOS))]
    return member.policy(batch)


def own_range(member, sit):
    """The preflop range this member credits *itself* with in this spot."""
    pct = combo_percentile(member._rank_pct, sit.n_players)
    return preflop_range(sit.opp_pf_action[sit.hero], sit.hero, sit.n_players,
                         pct)


def mass(weights, value):
    return float((weights * value).sum() / weights.sum())


def klass(text):
    a, b = text.split()
    return np.flatnonzero(COMBO_CLASS == hand_class_169(cards(a)[0],
                                                        cards(b)[0]))


def bets(p):
    return p[:, 2:].sum(axis=1)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_a_batch_is_the_concatenation_of_single_calls(parts):
    member = build(parts)
    record = scripted(HU_TO_FLOP, num_players=2)
    ctx = contexts_from([record])[3]

    batched = every_combo(member, ctx)
    for i in range(0, len(ALL_COMBOS), 27):
        one = member.policy([DecisionContext(
            ctx.record, ctx.snap_idx, ctx.acting_pos, ctx.legal_mask, ctx.turn,
            hole_override=list(ALL_COMBOS[i]))])
        assert np.array_equal(one[0], batched[i])


def test_it_plays_legal_hands_at_every_table_size_and_depth(parts):
    member = build(parts)
    decisions = 0
    for n_players in (2, 3, 5, 7, 9):
        for stack_bb in (10, 25, 100, 300):
            specs = make_specs(seed=n_players * 1000 + stack_bb, n_hands=6,
                               n_members=1, num_players=n_players,
                               stack_bb=stack_bb)
            for record in play([member], specs):
                assert abs(float(record.rewards.sum())) < 1e-9
                for d in record.decisions:
                    assert d["legal_mask"][d["action_idx"]]
                    decisions += 1
    assert decisions > 300


def test_every_row_is_a_distribution_over_legal_actions(parts):
    member = build(parts)
    specs = make_specs(seed=7, n_hands=12, n_members=1, num_players=6)
    contexts = contexts_from(play([member], specs))
    p = member.policy(contexts)
    legal = np.stack([np.asarray(c.legal_mask, bool) for c in contexts])
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-12
    assert not p[~legal].any()


def test_two_members_with_the_same_numbers_play_the_same(parts):
    one, two = build(parts), build(parts)
    specs = make_specs(seed=11, n_hands=40, n_members=1, num_players=4)
    contexts = contexts_from(play([one], specs))
    assert len(contexts) > 150
    assert np.array_equal(one.policy(contexts), two.policy(contexts))


def test_the_style_layer_still_applies(parts):
    plain = build(parts)
    biased = RegularMember(
        "folder", N_ACTIONS, RegularParams(), plain.cache, plain.preflop_table,
        RAISE_SIZES,
        style=StyleParams(uncond=np.array([3.0, 0, 0, 0, 0]),
                          position=np.zeros(5), street=np.zeros((4, 5)),
                          temperature=1.0, uniform_mix=0.0))
    record = scripted([POT, FOLD, FOLD, FOLD, FOLD, CALL])
    ctx = contexts_from([record])[5]              # the big blind, facing a raise
    assert every_combo(biased, ctx)[:, 0].mean() > every_combo(plain, ctx)[:, 0].mean()


# ---------------------------------------------------------------------------
# Preflop
# ---------------------------------------------------------------------------

def test_the_best_hand_always_raises_and_the_worst_always_folds(parts):
    member = build(parts)
    record = scripted([POT, FOLD, CALL, FOLD, FOLD, CALL])
    p = every_combo(member, contexts_from([record])[0])          # UTG, unopened
    assert bets(p)[klass("As Ah")].min() > 0.99
    assert p[klass("7s 2h"), 0].min() > 0.99


def test_an_opening_range_widens_from_the_first_seat_to_the_button(parts):
    member = build(parts)
    early = scripted([POT, FOLD, FOLD, FOLD, FOLD, CALL])
    late = scripted([FOLD, FOLD, FOLD, POT, FOLD, CALL])
    utg = bets(every_combo(member, contexts_from([early])[0])).mean()
    button = bets(every_combo(member, contexts_from([late])[3])).mean()
    assert 0.10 < utg < button < 0.60

    # Heads-up the small blind is the button and opens most of the deck.
    hu = scripted([POT, CALL], num_players=2)
    entered = 1.0 - every_combo(member, contexts_from([hu])[0])[:, 0]
    assert entered.mean() > 0.70


def test_a_three_bet_range_is_polar_and_a_four_bet_range_is_not(parts):
    member = build(parts, threebet_value=0.06, call_open=0.15,
                   threebet_bluff=0.04)
    record = scripted([POT, CALL, FOLD, FOLD, FOLD, FOLD])
    raising = bets(every_combo(member, contexts_from([record])[1])) > 0.5

    pct = combo_percentile(member._rank_pct, 6)
    assert raising[pct <= 0.06].all()                     # the value band
    assert not raising[(pct > 0.06) & (pct <= 0.15)].any()   # the calling gap
    assert raising[(pct > 0.15) & (pct <= 0.19)].all()    # the bluff band
    assert not raising[pct > 0.19].any()


def test_a_short_stack_shoves_and_a_deep_one_raises(parts):
    member = build(parts, push_fold_bb=12.0)
    for stack_bb, floor in ((10, 0.05), (5, 0.05)):
        record = scripted([POT, CALL], num_players=2, stack_bb=stack_bb)
        sit = situation(contexts_from([record])[0])
        p = every_combo(member, contexts_from([record])[0])
        shoved = p[:, -1].mean()
        assert abs(shoved - push_range(sit.eff_stack_bb, sit.n_behind)) < 0.05
        assert p[:, 2:-1].sum() < floor                   # no raise bin is used

    deep = scripted([POT, CALL], num_players=2, stack_bb=100)
    assert every_combo(member, contexts_from([deep])[0])[:, 2:-1].sum() > 1.0

    short = scripted([POT, CALL], num_players=2, stack_bb=5)
    ten = scripted([POT, CALL], num_players=2, stack_bb=10)
    assert (every_combo(member, contexts_from([short])[0])[:, -1].mean()
            > every_combo(member, contexts_from([ten])[0])[:, -1].mean())


def test_a_short_stack_calls_a_shove_with_its_calling_range(parts):
    member = build(parts, push_fold_bb=12.0)
    record = scripted([ALL_IN], num_players=2, stack_bb=10)
    ctx = contexts_from([record])[1]
    sit = situation(ctx)
    # A shove leaves nothing behind, so the effective stack is zero and the
    # calling range is the widest the table has — and still tighter than the
    # jamming range at that depth.
    called = 1.0 - every_combo(member, ctx)[:, 0]
    assert sit.eff_stack_bb == 0.0
    assert abs(called.mean() - call_range(sit.eff_stack_bb)) < 0.03
    assert called.mean() < push_range(sit.eff_stack_bb, sit.n_behind)


def test_a_full_table_is_played_more_tightly_than_a_short_one(parts):
    member = build(parts, multiway_tighten=0.2)
    six = scripted([CALL, CALL, CALL, POT, FOLD, CALL], num_players=6)
    heads_up = scripted([POT, CALL], num_players=2)
    # Four seats already in the pot against one: the same seat enters less.
    crowded = bets(every_combo(member, contexts_from([six])[3])).mean()
    assert crowded < bets(every_combo(member, contexts_from([heads_up])[0])).mean()


# ---------------------------------------------------------------------------
# Postflop
# ---------------------------------------------------------------------------

def _flop_spot(parts, board, **knobs):
    member = build(parts, **knobs)
    record = scripted(HU_TO_FLOP, num_players=2,
                      deck=deck_with(cards(board) + cards("2h 3h")))
    ctx = contexts_from([record])[3]              # the small blind, checked to
    return member, ctx, situation(ctx)


def test_a_dry_board_is_c_bet_more_than_a_wet_one(parts):
    dry, dry_ctx, _ = _flop_spot(parts, "Ks 7h 2d", slowplay=0.0)
    wet, wet_ctx, _ = _flop_spot(parts, "9s 8s 7d", slowplay=0.0)
    assert bets(every_combo(dry, dry_ctx)).mean() > bets(
        every_combo(wet, wet_ctx)).mean()


def test_the_bluffs_bet_at_the_c_bet_frequency(parts):
    """The knob is a *bluffing* frequency: value bets at `1 − slowplay` always.

    So the number to check it against is the air, which bets at exactly
    `f · bluff_ratio` — and `f` is the texture's own frequency, moved by the
    range advantage the board gives, which is why two archetypes differing only
    in `bluff_ratio` differ by exactly that factor on the same board.
    """
    board = "Ks 7h 2d"
    one, ctx, sit = _flop_spot(parts, board, bluff_ratio=1.0, slowplay=0.0)
    half, _, _ = _flop_spot(parts, board, bluff_ratio=0.5, slowplay=0.0)
    none, _, _ = _flop_spot(parts, board, bluff_ratio=0.0, slowplay=0.0)

    table = one.cache.get([c for c in ctx.board if c >= 0])
    air = (table.hs < 0.55) & ~(table.flush_draw | table.oesd) & table.live
    a1 = bets(every_combo(one, ctx))[air].mean()
    a2 = bets(every_combo(half, ctx))[air].mean()
    # A never-bluffing archetype leaves only the logit floor behind.
    assert bets(every_combo(none, ctx))[air].max() < 1e-7
    assert abs(a2 - 0.5 * a1) < 0.02
    assert 0.3 < a1 < 1.0


def test_multiway_shrinks_the_c_bet(parts):
    member = build(parts, multiway_tighten=0.2, slowplay=0.0)
    board = cards("Ks 7h 2d")
    heads_up = scripted(HU_TO_FLOP, num_players=2,
                        deck=deck_with(board + cards("2h 3h")))
    six = scripted([POT, CALL, CALL, CALL, CALL, CALL, CALL, CALL],
                   num_players=6, deck=deck_with(board + cards("2h 3h")))
    hu_ctx = contexts_from([heads_up])[3]
    six_ctx = [c for c in contexts_from([six])
               if c.turn == 1 and c.acting_pos == 2][0]
    assert situation(six_ctx).n_live > 3
    assert bets(every_combo(member, six_ctx)).mean() < bets(
        every_combo(member, hu_ctx)).mean()


def test_the_river_bluffs_are_balanced_against_the_size(parts):
    """`bluff_ratio = 1` means the size's own indifference ratio, exactly."""
    deck = deck_with(cards("9s 8s 2d 5h Kc") + cards("2h 3h"))
    shares = {}
    for ratio in (0.0, 1.0, 2.0):
        member = build(parts, bluff_ratio=ratio, slowplay=0.0,
                       barrel_river=1.0, size_river=0.75)
        record = scripted(HU_TO_RIVER, num_players=2, deck=deck)
        ctx = contexts_from([record])[9]
        sit = situation(ctx)
        assert sit.street == 3 and sit.hero == 0

        board = tuple(c for c in ctx.board if c >= 0)
        table, turn = member.cache.get(board), member.cache.get(board[:4])
        q = table.hs ** max(1, sit.n_live - 1)
        value = q >= member.params.value_hs
        missed = (q < 0.55) & (turn.flush_draw | turn.oesd | turn.gutshot)

        w = own_range(member, sit) * bets(every_combo(member, ctx))
        shares[ratio] = float((w * missed).sum()
                              / max(float((w * (value | missed)).sum()), 1e-9))

    s = 0.75
    assert shares[0.0] < 1e-6
    assert abs(shares[1.0] - s / (1.0 + s)) < 0.03
    # Twice the ratio cannot always buy twice the bluffs: a board only holds so
    # many missed draws, and every one of them already bets at ratio 2.
    assert shares[2.0] > shares[1.0]


def test_a_second_barrel_is_a_separate_knob(parts):
    """`barrel_turn` decides how much of the air keeps firing on the turn."""
    deck = deck_with(cards("Ks 7h 2d 5c Jh") + cards("2h 3h"))
    air_bet = {}
    for barrel in (0.0, 1.0):
        member = build(parts, barrel_turn=barrel, slowplay=0.0)
        record = scripted(HU_TO_FLOP + [HALF_POT, CALL, CALL], num_players=2,
                          deck=deck)
        ctx = contexts_from([record])[6]          # the turn, checked to the SB
        sit = situation(ctx)
        assert sit.street == 2 and sit.hero == 0 and sit.hero_barrels == 1

        table = member.cache.get([c for c in ctx.board if c >= 0])
        air = table.live & (table.hs < 0.55) & ~(table.flush_draw | table.oesd)
        air_bet[barrel] = bets(every_combo(member, ctx))[air].mean()
    assert air_bet[0.0] < 1e-7 < air_bet[1.0]


def test_betting_into_the_aggressor_is_its_own_frequency(parts):
    """A donk bet is rarer than a stab, and `donk = 0` means never."""
    deck = deck_with(cards("Ks 7h 2d") + cards("2h 3h"))
    record = scripted(HU_TO_FLOP[:2], num_players=2, deck=deck)
    ctx = contexts_from([record])[2]              # the big blind, first to act
    sit = situation(ctx)
    assert sit.last_aggressor == 0 and sit.hero == 1

    never = bets(every_combo(build(parts, donk=0.0), ctx))
    always = bets(every_combo(build(parts, donk=1.0), ctx))
    assert never.max() < 1e-7 < always.mean()


def test_the_defence_frequency_is_the_minimum_defence_frequency(parts):
    """The top `defend_factor · 1/(1 + s)` of hero's own range, and no more."""
    deck = deck_with(cards("Ks 7h 2d 5c Jh") + cards("2h 3h"))
    for size, bin_idx in ((1.0, POT), (0.5, HALF_POT)):
        for factor in (0.55, 1.0, 1.35):
            member = build(parts, defend_factor=factor)
            record = scripted(HU_TO_FLOP + [bin_idx], num_players=2, deck=deck)
            ctx = contexts_from([record])[4]      # the big blind, facing a bet
            sit = situation(ctx)
            assert abs(sit.facing - size) < 1e-9

            continued = 1.0 - every_combo(member, ctx)[:, 0]
            want = min(1.0, factor / (1.0 + size))
            assert abs(mass(own_range(member, sit), continued) - want) < 0.03


def test_a_draw_getting_the_right_price_never_folds(parts):
    """The human floor, on the draw's own odds rather than on its equity."""
    member = build(parts, defend_factor=0.2)      # would otherwise fold nearly all
    deck = deck_with(cards("9s 8s 2d") + cards("2h 3h"))
    record = scripted(HU_TO_FLOP + [HALF_POT], num_players=2, deck=deck)
    ctx = contexts_from([record])[4]
    sit = situation(ctx)

    table = member.cache.get([c for c in ctx.board if c >= 0])
    priced = table.live & (table.p_improve >= sit.pot_odds)
    assert priced.any()
    assert every_combo(member, ctx)[priced, 0].max() < 1e-9


def test_facing_an_all_in_is_a_price_and_nothing_else(parts):
    """No frequency, no range: the hand either beats the price or it does not.

    And the price is read through the same generous potential term the rest of
    the cascade uses — outs times four — so a bare flush draw calls a shove a
    real player would fold. That is the heuristic §P1 chose showing up where it
    is worst, and it is left as it is: the archetypes are meant to be
    recognisable, and calling too wide with a draw is a recognisable leak.
    """
    deck = deck_with(cards("9s 8s 2d") + cards("2h 3h"))
    for stack_bb in (40, 120):
        member = build(parts, defend_factor=1.0)
        record = scripted(HU_TO_FLOP + [ALL_IN], num_players=2,
                          stack_bb=stack_bb, deck=deck)
        ctx = contexts_from([record])[4]
        sit = situation(ctx)
        assert sit.facing_allin

        table = member.cache.get([c for c in ctx.board if c >= 0])
        p = every_combo(member, ctx)
        assert p[:, 2:].sum() < 1e-6              # only call or fold is legal
        called = p[:, 1] > 0.5
        assert np.array_equal(called, table.ehs(1) >= sit.pot_odds)

    # A bigger shove into the same pot is a worse price, so fewer hands call.
    def calling_mass(stack_bb):
        member = build(parts)
        record = scripted(HU_TO_FLOP + [ALL_IN], num_players=2,
                          stack_bb=stack_bb, deck=deck)
        ctx = contexts_from([record])[4]
        return every_combo(member, ctx)[:, 1].mean()

    assert calling_mass(120) < calling_mass(40)


def test_a_low_spr_turns_value_into_a_shove(parts):
    deck = deck_with(cards("Ks 7h 2d") + cards("2h 3h"))
    # A 3-bet pot 20 big blinds deep: one pot-sized bet is the whole stack.
    record = scripted([DOUBLE_POT, DOUBLE_POT, CALL], num_players=2,
                      stack_bb=20, deck=deck)
    ctx = contexts_from([record])[3]
    sit = situation(ctx)
    assert sit.spr <= 1.5 and sit.last_aggressor == sit.hero

    member = build(parts, allin_spr=1.5, slowplay=0.0)
    never = build(parts, allin_spr=0.0, slowplay=0.0)
    table = member.cache.get([c for c in ctx.board if c >= 0])
    value = table.live & (table.hs >= member.params.value_hs)
    assert value.any()
    assert every_combo(member, ctx)[value, -1].mean() > 0.9
    assert every_combo(never, ctx)[value, -1].mean() < 0.1


def test_the_posterior_batch_is_cheap_enough(parts):
    """§P3's acceptance: a 1 326-row query on a cached board, well under 20 ms."""
    import time

    member = build(parts)
    record = scripted(HU_TO_FLOP, num_players=2)
    ctx = contexts_from([record])[3]
    every_combo(member, ctx)                      # build and cache the board
    start = time.time()
    every_combo(member, ctx)
    assert time.time() - start < 0.1
