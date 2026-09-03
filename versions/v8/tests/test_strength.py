"""The per-board strength table the procedural pool reads
(`PLAN_PROCEDURAL_POOL.md` §P1).

Everything here is deterministic: the only randomness in `pool/strength.py` is
the preflop equity integral, which is seeded, and the boards below are either
hand-written or drawn from a seeded generator.

Two families of check, and they are testing different things:

* **exactness** — the river percentile against `env/showdown.py`, the
  card-removal denominator against an explicit loop, `range_equity` against
  `hs`. These have no tolerance beyond floating point, and they are what makes
  the table trustworthy as an input to a betting rule.
* **recognisability** — hand classes, draws, outs and texture on named boards.
  These pin *interpretations*: "top pair, good kicker" and "wet board" are
  phrases a human uses, and a member that reads them differently from a human
  is not the member the plan asked for.
"""

import numpy as np
import pytest

from env.showdown import hand_class_169, strength_percentiles
from pool.strength import (ALL_COMBOS, COMBO_CLASS, DISJOINT, N_COMBOS,
                           BoardStrength, HandClass, StrengthCache,
                           combo_index, preflop_equity_table, preflop_rank_pct)

RANKS = {r: i for i, r in enumerate("23456789TJQKA")}
SUITS = {s: i for i, s in enumerate("shdc")}


def card(text):
    """`"As"` → the card id the engine uses (rank × 4 + suit)."""
    return RANKS[text[0]] * 4 + SUITS[text[1]]


def cards(text):
    return [card(t) for t in text.split()]


def combo(text):
    a, b = text.split()
    return combo_index(card(a), card(b))


# ---------------------------------------------------------------------------
# The combo grid
# ---------------------------------------------------------------------------

def test_combo_grid_is_a_bijection():
    assert ALL_COMBOS.shape == (N_COMBOS, 2)
    assert (ALL_COMBOS[:, 0] < ALL_COMBOS[:, 1]).all()
    for i, (c0, c1) in enumerate(ALL_COMBOS):
        assert combo_index(c0, c1) == i
        assert combo_index(c1, c0) == i


def test_disjoint_matches_an_explicit_check():
    rng = np.random.default_rng(0)
    for i in rng.integers(0, N_COMBOS, 40):
        a, b = ALL_COMBOS[i]
        shares = ((ALL_COMBOS[:, 0] == a) | (ALL_COMBOS[:, 1] == a)
                  | (ALL_COMBOS[:, 0] == b) | (ALL_COMBOS[:, 1] == b))
        assert np.array_equal(DISJOINT[i], ~shares)
    assert not DISJOINT[np.arange(N_COMBOS), np.arange(N_COMBOS)].any()


def test_combo_class_agrees_with_the_showdown_labels():
    for i in (0, 5, 700, N_COMBOS - 1):
        a, b = ALL_COMBOS[i]
        assert COMBO_CLASS[i] == hand_class_169(a, b)


# ---------------------------------------------------------------------------
# The percentile
# ---------------------------------------------------------------------------

def test_river_percentile_equals_the_showdown_label():
    """The number `env/showdown.py` computes one hand at a time, for all 1326.

    The tolerance is 1e-6 rather than 1e-12 because the reference does its
    arithmetic in torch's default dtype — `0.5 * ties` on an int64 tensor is a
    float32 — so it is the reference that carries the rounding, not this table.
    """
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(10):
        deck = rng.permutation(52)
        board, holes = deck[:5], deck[5:15].reshape(5, 2)
        table = BoardStrength(board)
        ref = strength_percentiles(board, holes)
        got = np.array([table.hs[combo_index(*h)] for h in holes])
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst < 1e-6


def test_card_removal_is_exact_and_moves_the_number():
    """Blocking is the point of the exact percentile, so measure both halves.

    The hand holds the ace of the board's flush suit, which removes every combo
    that could have made the nut flush — the naive "rank among all live combos"
    and the card-removal percentile therefore disagree, and the card-removal one
    is the number an explicit loop gives.
    """
    table = BoardStrength(cards("Ks 9s 4s"))
    live = np.flatnonzero(table.live)
    score = table.score[live]

    naive = ((score[:, None] > score[None, :]).sum(1)
             + 0.5 * (score[:, None] == score[None, :]).sum(1)) / len(live)
    gap = np.abs(naive - table.hs[live])
    assert gap.max() > 0.02                    # §P1's "up to ~4 %"

    h = int(live[int(gap.argmax())])
    keep = [o for o in live if DISJOINT[h, o]]
    wins = sum(1 for o in keep if table.score[o] < table.score[h])
    ties = sum(1 for o in keep if table.score[o] == table.score[h])
    assert table.hs[h] == pytest.approx((wins + 0.5 * ties) / len(keep),
                                        abs=1e-12)


def test_blocked_combos_are_inert():
    table = BoardStrength(cards("Ks 9s 4s"))
    dead = combo("Ks 2d")
    assert not table.live[dead]
    assert table.score[dead] == 0
    assert table.hs[dead] == 0.0
    assert table.outs[dead] == 0.0
    assert table.hand_class[dead] == HandClass.AIR
    assert table.live.sum() == 1326 - (3 * 49 + 3)


def test_range_equity_is_the_weighted_percentile():
    table = BoardStrength(cards("Ks 7h 2d"))

    uniform = table.range_equity(table.live.astype(np.float64))
    assert np.abs(uniform - table.hs)[table.live].max() < 1e-12

    nuts = int(np.argmax(table.hs))
    weights = np.zeros(N_COMBOS)
    weights[nuts] = 1.0
    got = table.range_equity(weights)
    # Nothing beats the nuts, and the nuts itself has no opponent left once the
    # range is the one combo it is holding — a blocked-out range is 0.5.
    beaten = np.flatnonzero(DISJOINT[nuts] & table.live)
    assert not got[beaten].any()
    assert got[nuts] == 0.5
    assert (got[~DISJOINT[nuts]] == 0.5).all()


def test_range_equity_equals_the_masked_reduction():
    """The fast form is the definition, reordered — so check it against it.

    `range_equity` is inclusion–exclusion over the two cards a combo holds; the
    reduction below is the formula as written, three `1326 × 1326` passes. They
    are the same sums in a different order, so the gap is floating point.
    """
    rng = np.random.default_rng(19)
    for board in ("Ks 9s 4s", "As 8h 4d 2c", "9s 8s 7d Th 2c"):
        table = BoardStrength(cards(board))
        mask = DISJOINT & table.live[None, :]
        score = table.score
        for _ in range(3):
            w = rng.random(N_COMBOS) * (rng.random(N_COMBOS) < 0.4)
            masked = w * table.live
            num = (mask & (score[:, None] > score[None, :])) @ masked
            num = num + 0.5 * ((mask & (score[:, None] == score[None, :]))
                               @ masked)
            den = mask @ masked
            want = np.full(N_COMBOS, 0.5)
            np.divide(num, den, out=want, where=den > 0)
            assert np.abs(table.range_equity(w) - want).max() < 1e-12


def test_range_equity_refuses_a_malformed_range():
    table = BoardStrength(cards("Ks 7h 2d"))
    with pytest.raises(AssertionError):
        table.range_equity(np.zeros(52))
    with pytest.raises(AssertionError):
        table.range_equity(-np.ones(N_COMBOS))


# ---------------------------------------------------------------------------
# What a human would call the hand
# ---------------------------------------------------------------------------

def test_hand_classes_on_named_boards():
    ace = BoardStrength(cards("As 7h 2d"))
    assert ace.hand_class[combo("Ac Kd")] == HandClass.TOP_PAIR_GOOD
    assert ace.hand_class[combo("Ac 3d")] == HandClass.TOP_PAIR_WEAK

    jack = BoardStrength(cards("Js 7h 2d"))
    assert jack.hand_class[combo("Qc Qd")] == HandClass.OVERPAIR
    assert jack.hand_class[combo("8c 8d")] == HandClass.UNDERPAIR
    assert jack.hand_class[combo("7c 9d")] == HandClass.MIDDLE_PAIR
    assert jack.hand_class[combo("2c 9d")] == HandClass.WEAK_PAIR
    assert jack.hand_class[combo("Ac Kd")] == HandClass.ACE_HIGH
    assert jack.hand_class[combo("9c 8d")] == HandClass.AIR
    assert jack.hand_class[combo("Jc 7d")] == HandClass.TWO_PAIR
    assert jack.hand_class[combo("Jc Jd")] == HandClass.TRIPS_SET

    paired = BoardStrength(cards("7s 7h 2d"))
    assert paired.hand_class[combo("7c 9d")] == HandClass.TRIPS_SET
    assert paired.hand_class[combo("Qc Qd")] == HandClass.OVERPAIR
    assert paired.hand_class[combo("5c 5d")] == HandClass.TWO_PAIR
    assert paired.hand_class[combo("Ac Kd")] == HandClass.ACE_HIGH

    big = BoardStrength(cards("9s 8s 7s Th 2d"))
    assert big.hand_class[combo("Jc Qd")] == HandClass.STRAIGHT
    assert big.hand_class[combo("As 2s")] == HandClass.FLUSH
    assert big.hand_class[combo("9c 9d")] == HandClass.TRIPS_SET

    full = BoardStrength(cards("9s 9h 7s Th 2d"))
    assert full.hand_class[combo("7c 7d")] == HandClass.FULL_PLUS
    assert full.hand_class[combo("9c 2h")] == HandClass.FULL_PLUS


def test_draw_flags_on_named_hands():
    flop = BoardStrength(cards("9s 8d 2c"))
    jt = combo("Js Ts")
    assert flop.oesd[jt] and flop.backdoor_flush[jt]
    assert not flop.flush_draw[jt] and not flop.gutshot[jt]

    two_tone = BoardStrength(cards("Qs 7s 3d"))
    k2 = combo("Ks 2s")
    assert two_tone.flush_draw[k2] and not two_tone.oesd[k2]

    wheel = BoardStrength(cards("3d 4h Ks"))
    a5 = combo("Ac 5c")
    assert wheel.gutshot[a5] and not wheel.oesd[a5]

    made = BoardStrength(cards("9s 8d 7c"))
    assert not made.oesd[combo("Jh Td")] and not made.gutshot[combo("Jh Td")]
    assert made.hand_class[combo("Jh Td")] == HandClass.STRAIGHT


def test_no_draw_survives_the_river():
    rng = np.random.default_rng(3)
    for _ in range(3):
        table = BoardStrength(rng.permutation(52)[:5])
        assert not table.flush_draw.any()
        assert not table.oesd.any()
        assert not table.gutshot.any()
        assert not table.backdoor_flush.any()
        assert not table.p_improve.any()


def test_outs_and_ehs_are_the_rule_of_four_and_two():
    flop = BoardStrength(cards("9s 8d 2c"))
    jt = combo("Js Ts")                    # 8 straight outs, backdoor, 2 over
    assert flop.outs[jt] == pytest.approx(8 + 1 + 3.0)
    assert flop.p_improve[jt] == pytest.approx(12.0 * 0.04)

    two_tone = BoardStrength(cards("Qs 7s 3d"))
    k2 = combo("Ks 2s")                    # 9 flush outs, one overcard
    assert two_tone.outs[k2] == pytest.approx(9 + 1.5)

    combined = BoardStrength(cards("9s 8s 2c"))
    jt_s = combo("Js Ts")                  # flush draw *and* open-ender
    assert combined.flush_draw[jt_s] and combined.oesd[jt_s]
    assert combined.outs[jt_s] == pytest.approx(9 + 2 + 3.0)

    turn = BoardStrength(cards("9s 8d 2c 3h"))
    assert turn.p_improve[jt] == pytest.approx(turn.outs[jt] * 0.02)
    assert not turn.backdoor_flush.any()

    hs = flop.hs[jt]
    assert flop.ehs(1)[jt] == pytest.approx(hs + (1 - hs) * flop.p_improve[jt])
    assert flop.ehs(3)[jt] == pytest.approx(
        hs ** 3 + (1 - hs ** 3) * flop.p_improve[jt])
    assert (flop.ehs(3) <= flop.ehs(1) + 1e-12).all()

    river = BoardStrength(cards("9s 8d 2c 3h Kd"))
    assert np.abs(river.ehs(1) - river.hs).max() == 0.0


def test_outs_are_capped():
    table = BoardStrength(cards("9s 8s 2c"))
    assert table.outs.max() <= 15.0


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

def test_texture_buckets_on_named_boards():
    dry = BoardStrength(cards("Ks 7h 2d"))
    assert (dry.texture, dry.paired, dry.monotone, dry.two_tone) == (
        "dry", False, False, False)

    wet = BoardStrength(cards("9s 8s 7d"))
    assert wet.texture == "wet" and wet.two_tone

    mid = BoardStrength(cards("As 8s 4d"))
    assert mid.texture == "mid" and mid.two_tone and not mid.monotone

    mono = BoardStrength(cards("Qs Js Ts"))
    assert mono.texture == "wet" and mono.monotone

    paired = BoardStrength(cards("7s 7h 2d"))
    assert paired.paired and paired.high_rank == RANKS["7"]

    assert 0.0 <= dry.wetness <= wet.wetness <= 1.0


def test_a_board_must_be_a_board():
    with pytest.raises(AssertionError):
        BoardStrength(cards("Ks 7h"))
    with pytest.raises(AssertionError):
        BoardStrength([44, 21, -1])
    with pytest.raises(AssertionError):
        BoardStrength([44, 44, 21])


# ---------------------------------------------------------------------------
# Preflop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preflop(tmp_path_factory):
    """One 100 000-deal integral, shared by the tests below (~1 s)."""
    path = tmp_path_factory.mktemp("tables") / "preflop_equity.npy"
    return preflop_equity_table(str(path), seed=0, n_deals=100_000)


def cls(text):
    a, b = text.split()
    return hand_class_169(card(a), card(b))


def test_preflop_equity_is_a_pot_share(preflop):
    """The exact invariant: over *combos*, everyone's share is 1/(n+1).

    Per class the estimate is noisy — at this sample size a class carries a few
    hundred deals — but the combo-weighted mean averages that away, and it is
    the one number the estimator cannot get wrong by construction.
    """
    assert preflop.shape == (169, 8)
    assert ((preflop >= 0) & (preflop <= 1)).all()

    weight = np.bincount(COMBO_CLASS, minlength=169) / 1326.0
    pooled = preflop.T @ weight
    ideal = np.array([1.0 / (n + 1) for n in range(1, 9)])
    assert np.abs(pooled - ideal).max() < 0.01
    assert (np.diff(pooled) < 0).all()


def test_preflop_equity_orders_the_hand_groups(preflop):
    pairs = [cls(f"{r}s {r}h") for r in RANKS]
    suited = [cls(f"{a}s {b}s") for a in RANKS for b in RANKS
              if RANKS[a] > RANKS[b]]
    offsuit = [cls(f"{a}s {b}h") for a in RANKS for b in RANKS
               if RANKS[a] > RANKS[b]]

    assert (preflop[pairs, 0].mean() > preflop[suited, 0].mean()
            > preflop[offsuit, 0].mean())
    assert 0.80 < preflop[cls("As Ah"), 0] < 0.90
    assert preflop[cls("7s 2h"), 0] < 0.45
    aces = preflop[cls("As Ah")]
    assert aces[0] > aces[3] > aces[7]


def test_preflop_table_is_deterministic_and_cached(tmp_path):
    one = preflop_equity_table(str(tmp_path / "a.npy"), seed=1, n_deals=5_000)
    two = preflop_equity_table(str(tmp_path / "b.npy"), seed=1, n_deals=5_000)
    assert np.array_equal(one, two)

    other = preflop_equity_table(str(tmp_path / "c.npy"), seed=2, n_deals=5_000)
    assert not np.array_equal(one, other)

    # Written on first use, read afterwards: the second call cannot see the
    # different sample size because it never runs the integral.
    again = preflop_equity_table(str(tmp_path / "a.npy"), seed=1, n_deals=999)
    assert np.array_equal(one, again)


def test_rank_pct_is_a_combo_percentile():
    """A hand-built ordering, so the percentile is checked exactly.

    Class `c` is given the `c`-th best equity, so the ordering is known and the
    cumulative combo mass can be written down.
    """
    table = np.tile((169 - np.arange(169))[:, None] / 169.0, (1, 8))
    pct = preflop_rank_pct(table, 1)

    counts = np.bincount(COMBO_CLASS, minlength=169)
    expected = np.cumsum(counts) / 1326.0
    assert np.abs(pct - expected[COMBO_CLASS]).max() < 1e-12
    assert pct.max() == pytest.approx(1.0)
    assert (pct > 0).all()

    top = np.flatnonzero(COMBO_CLASS == 0)
    assert pct[top[0]] == pytest.approx(counts[0] / 1326.0)


def test_rank_pct_reads_the_requested_table_column(preflop):
    best = int(np.argmax(preflop[:, 0]))
    pct = preflop_rank_pct(preflop, 1)
    mass = np.bincount(COMBO_CLASS, minlength=169)[best] / 1326.0
    assert pct[np.flatnonzero(COMBO_CLASS == best)[0]] == pytest.approx(mass)
    assert pct[np.flatnonzero(COMBO_CLASS == cls("As Ah"))[0]] < 0.02
    assert pct[np.flatnonzero(COMBO_CLASS == cls("7s 2h"))[0]] > 0.8

    # A different number of opponents is a different ordering, hence a
    # different percentile for at least some combos.
    assert not np.array_equal(pct, preflop_rank_pct(preflop, 8))

    with pytest.raises(AssertionError):
        preflop_rank_pct(preflop, 0)
    with pytest.raises(AssertionError):
        preflop_rank_pct(preflop, 9)
    with pytest.raises(AssertionError):
        preflop_rank_pct(preflop[:, :4], 1)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

def test_cache_builds_once_per_board():
    cache = StrengthCache(max_boards=4)
    board = cards("Ks 7h 2d")
    first = cache.get(board)
    assert cache.get(board) is first
    assert cache.get(list(reversed(board))) is first
    assert len(cache) == 1

    turn = cache.get(cards("Ks 7h 2d 9c"))
    assert turn is not first and len(cache) == 2


def test_cache_evicts_the_oldest_board():
    cache = StrengthCache(max_boards=2)
    boards = [cards("Ks 7h 2d"), cards("As 8s 4d"), cards("9s 8s 7d")]
    tables = [cache.get(b) for b in boards]
    assert len(cache) == 2
    assert cache.get(boards[2]) is tables[2]
    assert cache.get(boards[1]) is tables[1]
    assert cache.get(boards[0]) is not tables[0]


def test_a_flop_table_is_cheap_enough_to_build_per_board():
    """§P1's acceptance: ~30 ms on the dev box, asserted loosely so it is not
    a timing test in disguise."""
    import time

    rng = np.random.default_rng(5)
    boards = [rng.permutation(52)[:3] for _ in range(5)]
    BoardStrength(boards[0])                       # warm the import-time work
    start = time.time()
    for board in boards:
        BoardStrength(board)
    assert (time.time() - start) / len(boards) < 0.2
