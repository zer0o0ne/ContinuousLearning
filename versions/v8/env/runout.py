"""What the hand is worth if the betting stops here and the board runs out.

The control variate the rollouts are averaged through (`CONCEPT.md` §7.3 —
Monte-Carlo noise is the one reducible error in a label, and §13 measures it at
0.5–1.4 of the stack per sample, because nearly every rollout is a stack-off).

``b(s)`` is every seat's share of the **matched** pot, weighted by how often it
holds the best hand over the boards that can still come, minus what it put in.
Three properties are all that is used, and each buys one thing:

* **It is an expectation over the cards**, so dealing one and re-averaging gives
  it back. The correction at a chance node is therefore exactly
  ``b(after) − b(before)`` — the card's luck and nothing else — with no
  expectation left to evaluate.
* **It ignores chips beyond the call**, which the settlement returns anyway. So
  ``b`` after a raise, an all-in and a call is one number, the correction at a
  decision node is two numbers rather than one per raise size, and it is
  identically zero where folding is not legal.
* **It is cheap**: one pass over the ranking matrix, no pot logic, so its cost
  does not grow with the table.

**Integrating the cards out is not a second mechanism.** Once a rollout has no
decisions left — everybody still in is all-in — the corrections for the streets
still to come telescope into ``b(final board) − b(that state)``, and what the
rollout reports is ``R − that``. Where the pot has no side pots, ``b`` on a
complete board *is* the settlement, the two ``R``s cancel, and the rollout
reports the exact average over every runout that could have happened. Where
there are side pots ``b`` prices a short all-in as if it could win the whole
matched pot, so the cancellation is partial and so is the reduction — never the
correctness.

**Correctness does not rest on any of that.** Each correction has zero mean over
the draw it corrects, whatever ``b`` is worth, so an inaccurate baseline removes
less noise and can never move what a rollout estimates. That is why `samples` —
how many board completions ``b`` averages over, exhaustive when there are no more
than that many and a uniform draw otherwise — is a cost knob and not a
correctness one: with `M` of them the runout's variance comes out reduced by
about `1 − 1/M`, and the rest is diminishing returns.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

from gto_utils.gpu_solver import evaluate_hands

N_CARDS = 52
#: Board cards a street has already shown, indexed by `Table.turn`.
BOARD_AT_TURN = (0, 3, 4, 5)


@dataclass
class RunoutConfig:
    """How finely the board is integrated over.

    16 is where the curve flattens on the dev box: the runout's variance comes
    out reduced by about `1 − 1/samples`, so 8 buys 88% and 64 buys 98%, while
    the cost is linear in it and roughly doubles between 16 and 64.
    """
    samples: int = 16


def board_completions(known, dead, n_needed, rng, cfg):
    """`(M, n_needed)` ways the board can finish, given what is already dealt.

    Args:
        known: the board cards already shown.
        dead: cards that cannot come — every seat's hole cards. Conditioning on
            all of them (folded seats included) is exact: the rollout deck deals
            the runout and the folded seats' cards out of the same remainder, so
            either order gives the same joint.
        n_needed: how many cards the board still wants.

    Exhaustive when there are no more than `cfg.samples` completions, a uniform
    draw of `cfg.samples` otherwise.
    """
    if n_needed <= 0:
        return np.zeros((1, 0), dtype=np.int64)
    used = np.zeros(N_CARDS, dtype=bool)
    used[np.asarray(known, dtype=np.int64)] = True
    used[np.asarray(dead, dtype=np.int64)] = True
    pool = np.flatnonzero(used == False)  # noqa: E712 — ~used on a bool array
    assert len(pool) >= n_needed, (
        f"{len(pool)} cards left cannot finish a board wanting {n_needed}")
    if math.comb(len(pool), n_needed) <= int(cfg.samples):
        return np.asarray(list(itertools.combinations(pool, n_needed)),
                          dtype=np.int64)
    order = np.argsort(rng.random((int(cfg.samples), len(pool))), axis=1)
    return pool[order[:, :n_needed]]


def score_rows(known, holes, comps):
    """`(M * N, 7)` — one seven-card hand per (completed board, seat)."""
    m, n = len(comps), len(holes)
    known = np.asarray(known, dtype=np.int64).reshape(1, -1)
    boards = np.concatenate([np.repeat(known, m, axis=0), comps], axis=1)
    return np.concatenate([np.repeat(boards, n, axis=0),
                           np.tile(np.asarray(holes, dtype=np.int64), (m, 1))],
                          axis=1)


def seat_scores(known, holes, comps):
    """`(M, N)` — every seat's hand strength on every completed board."""
    rows = score_rows(known, holes, comps)
    return evaluate_hands(torch.from_numpy(rows)).numpy().reshape(len(comps),
                                                                  len(holes))


def matched(bets):
    """Contributions with the part nobody can cover taken back out.

    Chips beyond what any single other seat put in are returned by the
    settlement, so they are not at stake and the baseline must not price them.
    Capping here is what makes a raise, an all-in and a call worth the same, and
    that in turn is what lets a decision's correction be two numbers rather than
    one per raise size.
    """
    bets = np.asarray(bets, dtype=np.float64)
    if len(bets) < 2:
        return bets.copy()
    order = np.argsort(bets)
    highest, second = bets[order[-1]], bets[order[-2]]
    cap = np.full(len(bets), highest)
    cap[order[-1]] = second
    return np.minimum(bets, cap)


def equity_baseline(scores, live, bets):
    """`(N,)` a cheap stand-in for `freeze_rewards`, for the corrections only.

    Every seat's share of the matched pot, weighted by how often it holds the
    best hand over the boards still possible, minus what it put in. It ignores
    side pots — a short all-in is priced as if it could win the whole matched
    pot — so it is *not* a settlement and is never used as one.

    That is sound because a control variate's correction has zero mean whatever
    baseline it is built from: an inaccurate baseline removes less noise and can
    never move the estimate. What it must be is (a) a function of the state and
    (b) an expectation over the cards, so that dealing one and re-averaging
    gives it back — both hold here, and neither needs the pot logic.

    It costs one pass over the ranking matrix instead of one pot settlement per
    distinct ranking, which at a full ring is the difference between a few tens
    of microseconds and a few tens of milliseconds.
    """
    live = np.asarray(live, dtype=bool)
    caps = matched(bets)
    out = -caps.copy()
    idx = np.flatnonzero(live)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[idx[0]] += float(caps.sum())
        return out
    sub = scores[:, idx]
    best = sub == sub.max(axis=1, keepdims=True)
    share = (best / best.sum(axis=1, keepdims=True)).mean(axis=0)
    out[idx] += share * float(caps.sum())
    return out


class HandRunout:
    """`V` for one hand in flight, with the board rankings cached per street.

    The completions and the rankings depend on the street and on the cards, not
    on the betting, so one street's work serves every decision taken on it and
    every hypothetical the control variate needs.

    The draw comes from a generator of its own, seeded from the hand's seed, so
    switching the estimator on does not move a single card or action of the hand
    it is measuring — and does not depend on which other hands shared the batch
    (`env/driver.py`, §15).
    """

    def __init__(self, deck, num_players, cfg, seed):
        self.deck = np.asarray(deck, dtype=np.int64)
        self.num_players = int(num_players)
        self.cfg = cfg
        self.rng = np.random.default_rng([int(seed) % (2 ** 63), 0x5EED])
        self._scores = {}

    def _holes(self):
        return self.deck[5:5 + 2 * self.num_players].reshape(-1, 2)

    def _job(self, turn):
        """`(known board, holes, completions)` for one street of this hand."""
        k = BOARD_AT_TURN[int(turn)]
        holes = self._holes()
        return (self.deck[:k], holes,
                board_completions(self.deck[:k], holes.reshape(-1), 5 - k,
                                  self.rng, self.cfg))

    def scores(self, turn):
        cached = self._scores.get(int(turn))
        if cached is None:
            known, holes, comps = self._job(turn)
            cached = seat_scores(known, holes, comps)
            self._scores[int(turn)] = cached
        return cached

    def baseline(self, turn, live, bets):
        """`(N,)` the cheap baseline the corrections are built from."""
        live = np.asarray(live, dtype=bool)
        if int(live.sum()) <= 1:
            return equity_baseline(None, live, bets)
        return equity_baseline(self.scores(turn), live, bets)


def prime(hands, chunk_rows=200_000):
    """Rank every street of every hand in one batch, ahead of any of them.

    A hand needs its ranking matrix the moment it takes a decision, and asking
    for it then means one evaluator call per hand per street — a few hundred
    rows a call, which is the shape `env/driver.py` exists to avoid. Every board
    a hand can reach is already fixed by its deck, so all four streets can be
    ranked before the hand is played and all the hands of a wave together: the
    oracle starts its whole label in one wave, so this is one batched call for
    the label rather than thousands of tiny ones.

    The waste is the streets a hand never reaches. It is bounded by four
    rankings a hand and it is rows, not calls — which is the term that is cheap.
    """
    jobs = [(hand, turn) + hand._job(turn)
            for hand in hands for turn in range(4)
            if int(turn) not in hand._scores]
    batch, rows = [], 0
    for job in jobs:
        block = score_rows(job[2], job[3], job[4])
        if rows and rows + len(block) > chunk_rows:
            _flush(batch)
            batch, rows = [], 0
        batch.append((job, block))
        rows += len(block)
    _flush(batch)


def _flush(batch):
    if not batch:
        return
    scored = evaluate_hands(
        torch.from_numpy(np.concatenate([b for _job, b in batch], axis=0)))
    scored = scored.numpy()
    at = 0
    for (hand, turn, _known, holes, comps), block in batch:
        hand._scores[int(turn)] = scored[at:at + len(block)].reshape(
            len(comps), len(holes))
        at += len(block)
    assert at == len(scored), f"{at} of {len(scored)} ranked rows were claimed"
