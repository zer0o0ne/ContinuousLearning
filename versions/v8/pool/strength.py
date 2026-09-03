"""What a hand is worth on a board, for all 1 326 holdings at once
(`PLAN_PROCEDURAL_POOL.md` §P1).

The procedural pool members of §P3 read a board and decide from it. Two facts
about how they are *asked* decide the shape of everything here, and neither is
negotiable:

1. **A member answers for ~1 225 holdings at once.** `oracle/posterior.py` asks
   every member "what would you have done holding *this*" for every combo
   consistent with the board, at every opponent decision of a labelled hand. A
   rule written for one hand and looped over combos is the wrong shape. So every
   quantity here is a `(1326,)` array indexed by `ALL_COMBOS`, and there is no
   per-hand Python anywhere below.
2. **Anything expensive is paid once per board.** One `BoardStrength` is one
   evaluator call over the combos that are still live, and it then serves every
   member, every decision and every posterior query on that board.
   `StrengthCache` is what makes "once per board" true across members.

**The percentile is exact, with card removal.** `hs[h]` counts only the combos
an opponent could actually hold — not on the board and not sharing a card with
`h` — because the hands that block many strong holdings are exactly the hands a
regular's rules care about. `env/showdown.py::strength_percentiles` computes the
same number on the river one hand at a time; this is that number for all 1 326
at once, and a test pins them equal.

**`hs` and `range_equity` are one formula.** Equity against a weighted range is

    Σ_o w_o · disjoint(h,o) · ([s_o < s_h] + ½[s_o = s_h])  /  Σ_o w_o · disjoint(h,o)

and `hs` is its uniform case, `w = live` — one implementation, so the percentile
a rule thresholds on and the equity it compares ranges with can never drift
apart. Written as the masked reduction it reads as, that is three `1326 × 1326`
passes and ~21 ms; it is computed instead by inclusion–exclusion on the two
cards a combo holds, exactly, in `O(1326 · 52)` and 0.43 ms, because the cascade
needs several of them per decision.

**The potential term is a table heuristic and is meant to be.** `outs` × 4 on
the flop and × 2 on the turn is what a human regular does at the table; it is
systematically generous to combo draws and to dominated draws. Sampling runouts
instead is deferred (§0.1) — the control variate already pays 16 runouts per
street per sample, and doubling that for the pool's benefit is a measured
label-cost decision, not a free improvement.

Nothing here is random except the preflop equity table, which is a Monte-Carlo
integral over deals, seeded, computed once and cached on disk **outside**
`versions/` (`CLAUDE.md` §2).
"""

import os
import tempfile
from enum import IntEnum

import numpy as np
import torch

from env.showdown import hand_class_169
from gto_utils.gpu_solver import evaluate_hands
from utils import progress

N_CARDS = 52
N_COMBOS = 1326
N_HAND_CLASSES = 169
#: `evaluate_hands` packs the hand category into the top digits of its score.
SCORE_BASE = 371293                                   # 13**5
#: Rank index of a ten — the "good kicker" floor of §P1.
RANK_TEN = 8
#: Where the preflop integral lives once it has been paid for. Outside
#: `versions/`, per `CLAUDE.md` §2, and named in one place so the pool
#: builder, the realism gate and the Slumbot runner cannot disagree.
DEFAULT_TABLE_PATH = "../../data/v8/tables/preflop_equity_v1.npy"


class HandClass(IntEnum):
    """How a regular *describes* its holding, weakest first.

    Descriptive only: the cascade thresholds on `hs`, and mentions a class where
    a human rule would ("needs top pair, good kicker"). The ordering is the one
    a player would give, which is why `TWO_PAIR` sits above `OVERPAIR` and a set
    above two pair.
    """

    AIR = 0
    ACE_HIGH = 1
    UNDERPAIR = 2
    WEAK_PAIR = 3
    MIDDLE_PAIR = 4
    TOP_PAIR_WEAK = 5
    TOP_PAIR_GOOD = 6
    OVERPAIR = 7
    TWO_PAIR = 8
    TRIPS_SET = 9
    STRAIGHT = 10
    FLUSH = 11
    FULL_PLUS = 12


def _all_combos():
    i, j = np.triu_indices(N_CARDS, k=1)
    return np.stack([i, j], axis=1).astype(np.int64)


#: (1326, 2) every two-card holding, `c0 < c1`, lexicographic. Every per-combo
#: array in the procedural pool is indexed by this order.
ALL_COMBOS = _all_combos()

#: (52, 52) combo index of a card pair, −1 on the diagonal. `combo_index` reads
#: it; it is also what makes `COMBO_CLASS` a lookup rather than a loop.
_COMBO_INDEX = np.full((N_CARDS, N_CARDS), -1, dtype=np.int64)
_COMBO_INDEX[ALL_COMBOS[:, 0], ALL_COMBOS[:, 1]] = np.arange(N_COMBOS)
_COMBO_INDEX[ALL_COMBOS[:, 1], ALL_COMBOS[:, 0]] = np.arange(N_COMBOS)

#: (1326, 52) which cards a combo holds.
_HOLDS = np.zeros((N_COMBOS, N_CARDS), dtype=bool)
_HOLDS[np.arange(N_COMBOS), ALL_COMBOS[:, 0]] = True
_HOLDS[np.arange(N_COMBOS), ALL_COMBOS[:, 1]] = True

#: (52, 51) the combos holding each card.
_COMBOS_WITH = np.stack([np.flatnonzero(_HOLDS[:, c]) for c in range(N_CARDS)])


def _disjoint():
    """(1326, 1326) True where two combos share no card. 1.7 MB, built once."""
    out = np.ones((N_COMBOS, N_COMBOS), dtype=bool)
    for c in range(N_CARDS):
        idx = _COMBOS_WITH[c]
        out[np.ix_(idx, idx)] = False
    return out


DISJOINT = _disjoint()

#: (52, 52) the 169-way class of a card pair. Built from `hand_class_169` so the
#: pool and the showdown labels can never disagree about what a class is.
_CLASS_TABLE = np.zeros((N_CARDS, N_CARDS), dtype=np.int64)
for _a in range(N_CARDS):
    for _b in range(N_CARDS):
        if _a != _b:
            _CLASS_TABLE[_a, _b] = hand_class_169(_a, _b)

#: (1326,) the 169-way class of every combo.
COMBO_CLASS = _CLASS_TABLE[ALL_COMBOS[:, 0], ALL_COMBOS[:, 1]]

#: (1326,) how many combos share each combo's 169-class: 6 for a pair, 4 suited,
#: 12 offsuit. `preflop_rank_pct` is a percentile over *combos*, so it weights
#: classes by this.
_CLASS_COMBOS = np.bincount(COMBO_CLASS, minlength=N_HAND_CLASSES)


def _straight_windows():
    """(10, 13) the five-rank windows a straight can occupy, A-5 through T-A."""
    out = np.zeros((10, 13), dtype=bool)
    for w in range(9):
        out[w, w:w + 5] = True
    out[9, [12, 0, 1, 2, 3]] = True                   # the wheel
    return out


_WINDOWS = _straight_windows()


def combo_index(c0, c1):
    """The row of `ALL_COMBOS` holding these two cards. Order-independent."""
    idx = int(_COMBO_INDEX[int(c0), int(c1)])
    assert idx >= 0, f"{c0} and {c1} are not two distinct cards"
    return idx


# ---------------------------------------------------------------------------
# Preflop: equity of a 169-class against n random opponents
# ---------------------------------------------------------------------------

def preflop_equity_table(path, seed=0, n_deals=2_000_000, chunk_rows=200_000):
    """(169, 8) share of the pot a class wins against `n = 1..8` random hands.

    Monte Carlo over whole deals: nine holdings and a board dealt from one
    shuffle, all nine seats scored, and seat 0's share of the pot accumulated
    against the first `n` opponents with ties split. One deal is one sample for
    every `n`, which is what makes the column-to-column comparison ("equity
    falls with more opponents") free of sampling noise between columns.

    The alternative — the repo's 169 *ordering* in `gto_utils/gto_helper.py` —
    is not usable here twice over: it needs `eval7`, which is not the project's
    evaluator and has no `aarch64` wheel guarantee (`CLAUDE.md` §3), and an
    ordering is not an equity against *n*, which is what a multiway range needs.

    Deterministic in `seed`. Written to `path` on first use and loaded after, so
    the 20 s integral is paid once per machine; `path` belongs under
    `/data/v8/tables/`, never inside `versions/` (`CLAUDE.md` §2).
    """
    if os.path.exists(path):
        return np.load(path)

    rng = np.random.default_rng(seed)
    total = np.zeros((N_HAND_CLASSES, 8), dtype=np.float64)
    count = np.zeros(N_HAND_CLASSES, dtype=np.float64)
    per_chunk = max(1, int(chunk_rows) // 9)
    done = 0
    bar = progress(total=int(n_deals), desc="preflop equity", unit="deal")
    while done < int(n_deals):
        n = min(per_chunk, int(n_deals) - done)
        deals = rng.permuted(np.tile(np.arange(N_CARDS, dtype=np.int64),
                                     (n, 1)), axis=1)[:, :23]
        board, holes = deals[:, :5], deals[:, 5:].reshape(n, 9, 2)
        rows = np.concatenate([np.repeat(board, 9, axis=0),
                               holes.reshape(n * 9, 2)], axis=1)
        scores = evaluate_hands(torch.from_numpy(rows)).numpy().reshape(n, 9)
        cls = _CLASS_TABLE[holes[:, 0, 0], holes[:, 0, 1]]
        for k in range(1, 9):
            sub = scores[:, :k + 1]
            best = sub.max(axis=1)
            share = np.where(scores[:, 0] == best,
                             1.0 / (sub == best[:, None]).sum(axis=1), 0.0)
            total[:, k - 1] += np.bincount(cls, weights=share,
                                           minlength=N_HAND_CLASSES)
        count += np.bincount(cls, minlength=N_HAND_CLASSES)
        done += n
        bar.update(n)
    bar.close()

    # A class no deal ever dealt is a coin flip: at the sample sizes this table
    # is built with it cannot happen, and at a test's sample size a fabricated
    # 0.0 would be a worse answer than an honest one.
    table = np.full((N_HAND_CLASSES, 8), 0.5, dtype=np.float64)
    np.divide(total, count[:, None], out=table, where=count[:, None] > 0)
    # Written through a temporary file and renamed, because the evaluation
    # runner builds one hero per worker process and all of them reach for this
    # table at once: a half-written `.npy` would be read as one.
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=directory, suffix=".npy")
    os.close(handle)
    np.save(staged, table, allow_pickle=False)
    os.replace(staged if staged.endswith(".npy") else staged + ".npy", path)
    return table


def preflop_rank_pct(table, n_opps):
    """(1326,) where each combo sits in the combo-weighted preflop ordering.

    0 is the best hand's end of the ordering: a combo's value is the share of
    *combos* at least as good as it, so `pct ≤ 0.20` is "the top 20 %" in the
    sense published ranges mean — pairs counting six combos, suited four,
    offsuit twelve, rather than every class counting one.

    The ordering is by equity against exactly `n_opps` opponents, so a range
    written as a fraction tightens by itself as the table fills up.
    """
    table = np.asarray(table, dtype=np.float64)
    assert table.shape == (N_HAND_CLASSES, 8), (
        f"a preflop table is (169, 8) — equity vs 1..8 opponents; got "
        f"{table.shape}")
    n = int(n_opps)
    assert 1 <= n <= 8, f"{n} opponents is outside the 2–9 player range"

    order = np.argsort(-table[:, n - 1], kind="stable")
    mass = np.cumsum(_CLASS_COMBOS[order]) / float(N_COMBOS)
    pct_of_class = np.empty(N_HAND_CLASSES, dtype=np.float64)
    pct_of_class[order] = mass
    return pct_of_class[COMBO_CLASS]


# ---------------------------------------------------------------------------
# Postflop: one table per board
# ---------------------------------------------------------------------------

class BoardStrength:
    """Every holding's standing on one board — 3, 4 or 5 cards.

    Built in one evaluator call and a handful of `(1326, 1326)` boolean
    reductions; ~25 ms on the dev box, and then read by everything on that
    board.
    """

    def __init__(self, board):
        board = np.asarray(board, dtype=np.int64).reshape(-1)
        assert board.shape[0] in (3, 4, 5), (
            f"a board is 3, 4 or 5 visible cards; got {board.tolist()}")
        assert (board >= 0).all(), (
            f"a board carries no undealt cards; got {board.tolist()}")
        assert len(set(board.tolist())) == len(board), (
            f"a board carries no duplicate cards; got {board.tolist()}")
        self.board = tuple(int(c) for c in board)
        self.n_board = len(self.board)

        self.live = ~_HOLDS[:, board].any(axis=1)
        self._score()
        self._draws(board)
        self._classes(board)
        self._texture()

    # -- made strength ------------------------------------------------------

    def _score(self):
        idx = np.flatnonzero(self.live)
        rows = np.concatenate(
            [np.repeat(np.asarray(self.board, dtype=np.int64)[None, :],
                       len(idx), axis=0), ALL_COMBOS[idx]], axis=1)
        self.score = np.zeros(N_COMBOS, dtype=np.int64)
        self.score[idx] = evaluate_hands(torch.from_numpy(rows)).numpy()

        # Score order, and where each combo sits in it. `range_equity` reads
        # both; they are three int arrays, so a cached table stays ~40 KB.
        self._order = np.argsort(self.score, kind="stable")
        ordered = self.score[self._order]
        self._lo = np.searchsorted(ordered, self.score, side="left")
        self._hi = np.searchsorted(ordered, self.score, side="right")

        self.hs = self.range_equity(self.live.astype(np.float64))
        self.hs[~self.live] = 0.0

    def _weighted(self, w):
        """`(win + ½·tie, denominator)` of one weighted range, per combo.

        The definition is a masked reduction over pairs — for each combo `h`,
        the weight of every combo `o` that shares no card with it and loses to
        it. Written that way it is three `1326 × 1326` passes, ~21 ms, and the
        cascade needs several per decision against a 20 ms budget for a whole
        1 225-row batch.

        It is instead computed exactly in `O(1326 · 52)` by inclusion–exclusion
        on the two cards `h` holds: the combos sharing a card with `h` are those
        holding `c0`, plus those holding `c1`, minus `h` itself, which is the
        only combo holding both. So a cumulative weight in score order — one
        total and one per card — answers every row at once. The arithmetic is
        the same sums in a different order, and a test pins it against the
        masked reduction to 1e-12.
        """
        ws = w[self._order]
        cum = np.concatenate([[0.0], np.cumsum(ws)])                # (1327,)
        per_card = np.concatenate(
            [np.zeros((1, N_CARDS)),
             np.cumsum(_HOLDS[self._order] * ws[:, None], axis=0)])  # (1327, 52)

        lo, hi = self._lo, self._hi
        c0, c1 = ALL_COMBOS[:, 0], ALL_COMBOS[:, 1]
        beaten = cum[lo] - per_card[lo, c0] - per_card[lo, c1]
        tied = ((cum[hi] - cum[lo])
                - (per_card[hi, c0] - per_card[lo, c0])
                - (per_card[hi, c1] - per_card[lo, c1]) + w)
        den = cum[-1] - per_card[-1, c0] - per_card[-1, c1] + w
        return beaten + 0.5 * tied, den

    def range_equity(self, weights):
        """(1326,) equity against a weighted range, on this board, right now.

        `weights` is a non-negative `(1326,)` vector; combos the board blocks
        are dropped from it, because nobody can hold them. A hand whose range is
        entirely blocked — every combo of it shares a card with the hand — has
        no opponent to be measured against and scores 0.5.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.shape == (N_COMBOS,), (
            f"a range is one weight per combo, {N_COMBOS} of them; got "
            f"{w.shape}")
        assert (w >= 0).all(), "range weights are non-negative"

        num, den = self._weighted(w * self.live)
        out = np.full(N_COMBOS, 0.5, dtype=np.float64)
        np.divide(num, den, out=out, where=den > 1e-12)
        return out

    # -- draws --------------------------------------------------------------

    def _draws(self, board):
        river = self.n_board == 5
        flop = self.n_board == 3
        ranks, suits = ALL_COMBOS // 4, ALL_COMBOS % 4
        b_ranks, b_suits = board // 4, board % 4

        present = np.zeros((N_COMBOS, 13), dtype=bool)
        present[:, b_ranks] = True
        present[np.arange(N_COMBOS), ranks[:, 0]] = True
        present[np.arange(N_COMBOS), ranks[:, 1]] = True

        suit_count = np.bincount(b_suits, minlength=4)[None, :].repeat(
            N_COMBOS, axis=0)
        suit_count += (suits[:, 0][:, None] == np.arange(4)[None, :])
        suit_count += (suits[:, 1][:, None] == np.arange(4)[None, :])
        in_hole = ((suits[:, 0][:, None] == np.arange(4)[None, :])
                   | (suits[:, 1][:, None] == np.arange(4)[None, :]))
        self.flush_draw = (not river) & ((suit_count == 4) & in_hole).any(1)
        self.backdoor_flush = flop & ((suit_count == 3) & in_hole).any(1)

        n_present = present.astype(np.int8) @ _WINDOWS.T.astype(np.int8)
        in_window = (_WINDOWS[:, ranks[:, 0]] | _WINDOWS[:, ranks[:, 1]]).T
        made_straight = (n_present == 5).any(axis=1)
        draw_win = (n_present == 4) & in_window
        completes = np.zeros((N_COMBOS, 13), dtype=bool)
        for w in range(len(_WINDOWS)):
            completes |= (_WINDOWS[w][None, :] & ~present) & draw_win[:, [w]]
        n_complete = completes.sum(axis=1)
        # Four cards of every rank that completes a window are still in the
        # deck, so two completing ranks is the eight-out draw and one is the
        # four-out one.
        self.oesd = (not river) & (n_complete >= 2) & ~made_straight
        self.gutshot = (not river) & (n_complete == 1) & ~made_straight

        # Overcards are what a hand has *instead* of a pair, so a hand that
        # already paired something has none of them.
        top_board = int(b_ranks.max())
        no_pair = (self.score // SCORE_BASE) == 0
        self.overcards = np.where(
            no_pair, (ranks > top_board).sum(axis=1), 0).astype(np.int8)

        second_draw = (self.flush_draw.astype(np.int8)
                       + self.oesd.astype(np.int8)
                       + self.gutshot.astype(np.int8)) >= 2
        best = np.maximum.reduce([9.0 * self.flush_draw, 8.0 * self.oesd,
                                  4.0 * self.gutshot])
        self.outs = np.minimum(
            best + 2.0 * second_draw + 1.5 * self.overcards
            + 1.0 * self.backdoor_flush, 15.0)

        for name in ("flush_draw", "backdoor_flush", "oesd", "gutshot"):
            setattr(self, name, getattr(self, name) & self.live)
        self.overcards = np.where(self.live, self.overcards, 0).astype(np.int8)
        self.outs = np.where(self.live, self.outs, 0.0)

    @property
    def p_improve(self):
        """(1326,) the rule of four and two: outs × 4 %, × 2 %, then nothing."""
        per_out = {3: 0.04, 4: 0.02, 5: 0.0}[self.n_board]
        return self.outs * per_out

    def ehs(self, n_opps):
        """(1326,) hand strength with the cards still to come folded in.

        `hs^n` is holding the best hand against `n` independent opponents now;
        the rest of the probability improves at the table-talk rate. This is
        Poki's EHS′ — potential without negative potential, which is the right
        asymmetry for a member deciding whether to *put money in*.
        """
        n = int(n_opps)
        assert n >= 1, f"{n} opponents is not a hand anyone is playing"
        ahead = self.hs ** n
        return ahead + (1.0 - ahead) * self.p_improve

    # -- descriptive class --------------------------------------------------

    def _classes(self, board):
        ranks = ALL_COMBOS // 4
        hi, lo = ranks.max(axis=1), ranks.min(axis=1)
        b_ranks = board // 4
        b_count = np.bincount(b_ranks, minlength=13)
        distinct = np.flatnonzero(b_count)[::-1]                  # descending
        order = np.full(13, 99, dtype=np.int64)
        order[distinct] = np.arange(len(distinct))
        top, second = int(distinct[0]), int(distinct[1]) if len(distinct) > 1 else -1

        cat = self.score // SCORE_BASE
        pocket = ranks[:, 0] == ranks[:, 1]
        m_hi, m_lo = b_count[hi] >= 1, b_count[lo] >= 1
        paired_rank = np.where(m_hi, hi, np.where(m_lo, lo, -1))
        kicker = np.where(m_hi, lo, hi)
        idx = np.where(paired_rank >= 0, order[paired_rank], 99)
        good = (kicker >= RANK_TEN) | (kicker >= second)
        one_pair = cat == 1

        self.hand_class = np.select(
            [cat >= 6,
             cat == 5,
             cat == 4,
             pocket & (hi > top),
             cat == 3,
             cat == 2,
             one_pair & pocket,
             one_pair & (idx == 0) & good,
             one_pair & (idx == 0) & ~good,
             one_pair & (idx == 1),
             one_pair & (idx >= 2) & (idx < 99),
             hi == 12],
            [HandClass.FULL_PLUS,
             HandClass.FLUSH,
             HandClass.STRAIGHT,
             HandClass.OVERPAIR,
             HandClass.TRIPS_SET,
             HandClass.TWO_PAIR,
             HandClass.UNDERPAIR,
             HandClass.TOP_PAIR_GOOD,
             HandClass.TOP_PAIR_WEAK,
             HandClass.MIDDLE_PAIR,
             HandClass.WEAK_PAIR,
             HandClass.ACE_HIGH],
            HandClass.AIR).astype(np.int8)
        self.hand_class = np.where(self.live, self.hand_class,
                                   HandClass.AIR).astype(np.int8)

    # -- board scalars ------------------------------------------------------

    def _texture(self):
        board = np.asarray(self.board, dtype=np.int64)
        b_count = np.bincount(board // 4, minlength=13)
        suit_count = np.bincount(board % 4, minlength=4)
        self.paired = bool((b_count >= 2).any())
        self.monotone = bool(suit_count.max()
                             >= (3 if self.n_board == 3 else 4))
        # "A flush draw is live on this board" — the flop notion of two-tone,
        # read on later streets the way the cascade uses it.
        self.two_tone = bool(not self.monotone and suit_count.max() >= 2)
        self.high_rank = int((board // 4).max())
        drawy = (self.flush_draw | self.oesd
                 | (self.hand_class >= HandClass.STRAIGHT))
        self.wetness = float(drawy[self.live].mean())

    @property
    def texture(self):
        """`dry`, `wet` or `mid` — the bucket the cascade's c-bet rules read."""
        if self.wetness > 0.25 or self.monotone:
            return "wet"
        if self.wetness < 0.12 and not self.two_tone:
            return "dry"
        return "mid"

    def __repr__(self):
        return (f"BoardStrength({self.board}, {self.texture}, "
                f"wetness={self.wetness:.3f})")


class StrengthCache:
    """One table per board, shared by every member in the process.

    Keyed on the board as a *set* of cards, so two hands that saw the same flop
    in a different order share the work. Eviction is by insertion order, which
    for a corpus played hand after hand is the same as by recency at the only
    scale that matters: a board is asked about many times within one hand and
    then never again.
    """

    def __init__(self, max_boards=2048):
        self.max_boards = int(max_boards)
        assert self.max_boards >= 1, "a cache holds at least one board"
        self._tables = {}

    def get(self, board):
        key = tuple(sorted(int(c) for c in np.asarray(board).reshape(-1)))
        table = self._tables.get(key)
        if table is None:
            table = BoardStrength(board)
            self._tables[key] = table
            while len(self._tables) > self.max_boards:
                self._tables.pop(next(iter(self._tables)))
        return table

    def __len__(self):
        return len(self._tables)
