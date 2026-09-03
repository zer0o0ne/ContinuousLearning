"""Ten archetypes, and whether they are ten different players
(`PLAN_PROCEDURAL_POOL.md` §P4).

The presets themselves are checked cheaply — a draw at zero spread is the preset,
a draw at full spread stays inside every domain, a zero stays zero. The claim
that matters costs more: seat all ten at one table, rotate them through every
chair, and read their stat lines. §0.3 makes that the whole point of the
exercise, so it is asserted here rather than left to a gate nobody runs.

**Only the orderings that are resolvable at this sample size are asserted.**
Every archetype gets ~1 350 hands, and a stat with sixty opportunities behind it
has a standard error of six points, so an ordering with a four-point gap is not
evidence of anything. Those are reported by the gate and not pinned here:
`check_raise` (trapper above tag by one standard error), `wtsd` (loose-passive
above nit by 1.4 of them) and `steal` (a nine-handed table reaches the cutoff
with the pot unopened about twenty times in fifteen hundred hands).
"""

import numpy as np
import pytest

from dataclasses import fields

from env.driver import LockstepDriver
from env.showdown import label_showdowns
from gates.pool_realism import make_archetypes, mixed_hands, ratio
from pool.archetypes import (ADDITIVE, ARCHETYPES, DEFAULT_DOMAIN, DOMAIN,
                             draw_params)
from pool.regular import RegularParams
from pool.stats import hud_stats
from pool.strength import StrengthCache, preflop_equity_table
import json
import os

#: The archetypes are claims about the raise grid the pool will actually be
#: built on — a size of 1.25 pot is not a different player from one of 0.75 pot
#: on a grid whose only bins are a half, a whole and a double pot. So this file
#: reads the run's own grid rather than the three-bin fixture the other tests
#: use, and if the grid changes these are the assertions that should be re-read.
_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(_CONFIG) as _fh:
    GAME = json.load(_fh)["game"]
N_ACTIONS = int(GAME["n_actions"])
BIG_BLIND = float(GAME["big_blind"])
SMALL_BLIND = float(GAME["small_blind"])
RAISE_SIZES = [GAME["raise_sizes"][s]
               for s in ("preflop", "flop", "turn", "river")]
NAMES = list(ARCHETYPES)


# ---------------------------------------------------------------------------
# The presets and the jitter
# ---------------------------------------------------------------------------

def test_there_are_ten_distinct_presets():
    assert len(ARCHETYPES) == 10
    seen = {name: tuple(getattr(p, f.name) for f in fields(RegularParams))
            for name, p in ARCHETYPES.items()}
    assert len(set(seen.values())) == 10


def test_zero_spread_is_the_preset_itself():
    rng = np.random.default_rng(0)
    for name, preset in ARCHETYPES.items():
        assert draw_params(name, rng, spread=0.0) == preset


def test_a_full_spread_draw_stays_inside_every_domain():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for name in NAMES:
            drawn = draw_params(name, rng, spread=1.0)
            for f in fields(RegularParams):
                lo, hi = DOMAIN.get(f.name, DEFAULT_DOMAIN)
                assert lo <= getattr(drawn, f.name) <= hi, (name, f.name)


def test_a_knob_that_is_zero_stays_zero():
    """An archetype that never bluffs is not jittered into bluffing."""
    rng = np.random.default_rng(2)
    for _ in range(30):
        nit = draw_params("nit", rng, spread=1.0)
        assert nit.threebet_bluff == 0.0
        assert nit.raise_bluff == 0.0
        assert nit.donk == 0.0
        assert nit.overbet == 0.0


def test_sizes_are_shifted_and_rates_are_scaled():
    """A pot fraction of 0.33 and one of 1.5 want the same absolute spread."""
    rng = np.random.default_rng(3)
    draws = [draw_params("tag", rng, spread=1.0) for _ in range(400)]
    base = ARCHETYPES["tag"]

    for name in ("size_dry", "size_river", "push_fold_bb"):
        spread = np.std([getattr(d, name) for d in draws])
        assert abs(spread - ADDITIVE[name]) < 0.4 * ADDITIVE[name]

    ratios = np.array([d.cbet_dry / base.cbet_dry for d in draws])
    assert abs(np.std(np.log(ratios)) - 0.15) < 0.05


def test_the_draw_is_reproducible():
    one = [draw_params("maniac", np.random.default_rng(4), spread=1.0)
           for _ in range(3)]
    two = [draw_params("maniac", np.random.default_rng(4), spread=1.0)
           for _ in range(3)]
    assert one == two
    assert one[0] != ARCHETYPES["maniac"]


def test_an_unknown_archetype_is_refused():
    with pytest.raises(AssertionError):
        draw_params("shark", np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Are they ten different players?
# ---------------------------------------------------------------------------

HEADLINE = ("vpip", "pfr", "limp", "threebet", "fold_to_threebet",
            "cbet_flop", "fold_to_cbet", "overbet_pct", "af", "wtsd")


@pytest.fixture(scope="module")
def records_and_pool(tmp_path_factory):
    """One 9-max session of 1 500 hands with all ten archetypes at the table.

    All ten sit at the same table and are dealt a fresh seating every hand, so
    each is read over the same hands, against the same field, from every chair
    — which is what makes the orderings below a comparison and not ten separate
    experiments. It is the gate's own seating, so this covers that too.
    """
    path = tmp_path_factory.mktemp("tables") / "preflop_equity.npy"
    table = preflop_equity_table(str(path), seed=0, n_deals=200_000)
    pool = make_archetypes(NAMES, StrengthCache(4096), table, GAME, seed=1)

    specs = mixed_hands(len(NAMES), n_players=9, hands=1500,
                        stack_range=(10, 300), big_blind=BIG_BLIND,
                        small_blind=SMALL_BLIND, raise_sizes=RAISE_SIZES,
                        seed=5)
    records = LockstepDriver(pool, N_ACTIONS).run(specs, batch_size=64)
    label_showdowns(records)
    return records, pool


@pytest.fixture(scope="module")
def records_and_pool_parts(records_and_pool):
    return records_and_pool[1]


@pytest.fixture(scope="module")
def stat_lines(records_and_pool):
    records = records_and_pool[0]
    return {name: hud_stats(records,
                            (lambda k: lambda r, s: r.spec.seat_members[s] == k)(i))
            for i, name in enumerate(NAMES)}


def freq(stat_lines, name, stat):
    return ratio(stat_lines[name][stat])


def test_every_archetype_played_a_comparable_number_of_hands(stat_lines):
    dealt = [stat_lines[name]["vpip"][1] for name in NAMES]
    # Nine of ten seats are dealt each hand and the seating is drawn, so the
    # counts are binomial around 1350 rather than exactly equal.
    assert min(dealt) > 1200
    assert max(dealt) - min(dealt) < 120


def test_how_many_pots_they_enter_orders_them(stat_lines):
    vpip = [freq(stat_lines, n, "vpip")
            for n in ("nit", "tag", "bluffer", "loose_passive", "maniac")]
    assert vpip == sorted(vpip)
    assert vpip[0] < 0.12 and vpip[-1] > 0.45


def test_how_they_enter_orders_them(stat_lines):
    pfr = [freq(stat_lines, n, "pfr")
           for n in ("loose_passive", "tag", "bluffer", "maniac")]
    assert pfr == sorted(pfr)
    # The passive ones enter by calling; the aggressive ones never limp at all.
    assert freq(stat_lines, "loose_passive", "limp") > 0.1
    assert freq(stat_lines, "tag", "limp") == 0.0
    assert freq(stat_lines, "maniac", "limp") == 0.0


def test_aggression_orders_them(stat_lines):
    af = [freq(stat_lines, n, "af")
          for n in ("loose_passive", "tag", "maniac")]
    assert af == sorted(af)
    # Ratios, not levels: what an aggression factor is worth is how many times
    # the next player up bets for every call.
    assert af[1] > 1.4 * af[0] and af[2] > 3.0 * af[1]


def test_the_four_added_archetypes_each_break_their_own_regularity(stat_lines):
    # A trapper checks where a TAG bets, and a bluffer bets more than either.
    cbet = [freq(stat_lines, n, "cbet_flop") for n in ("trapper", "tag",
                                                       "bluffer")]
    assert cbet == sorted(cbet)
    assert cbet[1] - cbet[0] > 0.15 and cbet[2] - cbet[1] > 0.15

    # A weak-tight enters as many pots as a loose-passive and then folds
    # instead of calling — the same preflop point, the opposite answer to a bet.
    assert (freq(stat_lines, "weak_tight", "fold_to_cbet")
            > freq(stat_lines, "loose_passive", "fold_to_cbet") + 0.1)

    # The polar reg is the only one that overbets by design.
    assert (freq(stat_lines, "polar_reg", "overbet_pct")
            > freq(stat_lines, "tag", "overbet_pct") + 0.1)
    assert (freq(stat_lines, "polar_reg", "overbet_pct")
            > freq(stat_lines, "trapper", "overbet_pct") + 0.1)

    # The stealer's regularity is the *shape* of the position curve, and the
    # stat that isolates it — steal attempts — has a dozen opportunities behind
    # it at a nine-handed table. `test_the_position_curve_is_the_stealers_own`
    # measures the same thing with three hundred.


@pytest.fixture(scope="module")
def six_max(records_and_pool_parts):
    """A shorter six-handed session, for the stats a full ring starves.

    A steal is an attempt on a pot nobody has entered, and at nine seats the
    cutoff sees one about a dozen times in fifteen hundred hands — a dozen is
    not a frequency. Six seats give it five times as many.
    """
    pool = records_and_pool_parts
    specs = mixed_hands(len(NAMES), n_players=6, hands=1000,
                        stack_range=(10, 300), big_blind=BIG_BLIND,
                        small_blind=SMALL_BLIND, raise_sizes=RAISE_SIZES,
                        seed=17)
    records = LockstepDriver(pool, N_ACTIONS).run(specs, batch_size=64)
    label_showdowns(records)
    return {name: hud_stats(records,
                            (lambda k: lambda r, s: r.spec.seat_members[s] == k)(i))
            for i, name in enumerate(NAMES)}


def test_the_position_curve_is_the_stealers_own(six_max):
    """Every other archetype bends its range by the same shape with position.

    The stealer opens a fifteenth of the deck from the first seat and most of
    it from the button, and the stat that isolates that is how often it takes a
    shot at a pot nobody has entered.
    """
    attempts = {name: six_max[name]["steal"] for name in NAMES}
    assert min(den for _num, den in attempts.values()) > 40

    steal = {name: ratio(counts) for name, counts in attempts.items()}
    assert steal["stealer"] > steal["tag"] + 0.15
    assert steal["stealer"] > steal["polar_reg"] + 0.15
    assert steal["nit"] < steal["weak_tight"] < steal["tag"]
    # The maniac opens everything from everywhere, which is not a curve.
    assert steal["maniac"] > steal["stealer"]


def test_no_two_archetypes_have_the_same_stat_line(stat_lines):
    """The diversity claim, as one assertion: every pair is told apart."""
    lines = {name: np.array([freq(stat_lines, name, s) for s in HEADLINE])
             for name in NAMES}
    assert all(np.isfinite(v).all() for v in lines.values())
    worst = min(
        float(np.abs(lines[a] - lines[b]).max())
        for i, a in enumerate(NAMES) for b in NAMES[i + 1:])
    assert worst > 0.05
