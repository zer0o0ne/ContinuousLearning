"""Card/fold control variates with independent auxiliary board samples.

For fixed live seats and matched contributions, the exact equity baseline B
is a martingale as public cards are revealed. A uniform M-board estimate
b(s, U) satisfies E_U[b(s, U)] = B(s), so the card correction
b(after, U) - b(before, U) has zero expectation over game and auxiliary draws.
U must be independent of the game trajectory and refreshed per outer sample.
Seeding only by visible cards freezes an MC error and violates this identity.

The fold correction is explicitly centred by the policy's fold probability;
it has zero conditional expectation even for an approximate baseline. With
all players all-in and no side pots, corrected rewards equal the sampled
baseline at the all-in state (exact only when completions are enumerated).
Side pots affect variance reduction, not the zero-mean correction argument.
"""

import hashlib
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
    """Number of auxiliary boards; enumerate if the support fits this budget."""
    samples: int = 16

    def __post_init__(self):
        if int(self.samples) != self.samples or self.samples <= 0:
            raise ValueError("runout samples must be a positive integer")


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

    Card corrections require an expectation over future cards for fixed bets
    and live seats. That property holds after averaging independent auxiliary
    draws. It does not require this heuristic to equal the side-pot settlement.
    Fold corrections are centred explicitly by the actual action probability.

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
    """Baseline with rankings cached by deck, auxiliary seed and sample count.

    Supply an independent runout_seed per outer sample; alternatives of that
    sample share it. The default seed supports deterministic direct callers.
    The driver supplies a separate stream even for ordinary hands. Future
    cards may identify cache entries but never seed the auxiliary sampler.
    Policies do not observe this seed or the baseline.
    """

    def __init__(self, deck, num_players, cfg, scores=None, runout_seed=0):
        self.deck = np.asarray(deck, dtype=np.int64)
        self.num_players = int(num_players)
        self.cfg = cfg
        self.runout_seed = int(runout_seed)
        self.key = (self.deck.tobytes(), self.num_players, self.runout_seed,
                    int(cfg.samples))
        self._scores = {} if scores is None else scores

    def _rng(self, known, holes):
        """Visible cards plus an independent auxiliary draw for this sample."""
        digest = hashlib.blake2b(
            np.asarray(known, dtype=np.int64).tobytes() + holes.tobytes()
            + bytes([self.num_players]), digest_size=8).digest()
        return np.random.default_rng([
            self.runout_seed, int.from_bytes(digest, "big"), 0x5EED])

    def _holes(self):
        return self.deck[5:5 + 2 * self.num_players].reshape(-1, 2)

    def _job(self, turn):
        """`(known board, holes, completions)` for one street of this hand."""
        k = BOARD_AT_TURN[int(turn)]
        holes = self._holes()
        return (self.deck[:k], holes,
                board_completions(self.deck[:k], holes.reshape(-1), 5 - k,
                                  self._rng(self.deck[:k], holes), self.cfg))

    def cached(self, turn):
        """The ranking matrix of one street, or None if nobody has ranked it."""
        return self._scores.get((self.key, int(turn)))

    def scores(self, turn):
        cached = self.cached(turn)
        if cached is None:
            known, holes, comps = self._job(turn)
            cached = seat_scores(known, holes, comps)
            self._scores[(self.key, int(turn))] = cached
        return cached

    def baseline(self, turn, live, bets):
        """`(N,)` the cheap baseline the corrections are built from."""
        live = np.asarray(live, dtype=bool)
        if int(live.sum()) <= 1:
            return equity_baseline(None, live, bets)
        return equity_baseline(self.scores(turn), live, bets)


def prime(demands, chunk_rows=200_000):
    """Rank the streets a wave of hands is about to ask for, in one batch.

    A hand needs its ranking matrix the moment it takes a decision, and asking
    for it then means one evaluator call per hand per street — a few hundred
    rows a call, which is the shape `env/driver.py` exists to avoid. Ranking a
    whole wave together makes it one call for the label instead of thousands of
    tiny ones.

    Args:
        demands: `(HandRunout, turn)` pairs — what the hands of this round can
            ask for before the next one. The driver knows that, this does not:
            ranking every street of every hand ahead of time pays for streets a
            hand folds before reaching, and for a rollout that replays a forced
            prefix it pays for streets that were already visible when the
            labelled decision was taken and can never be queried.
        chunk_rows: how many seven-card rows go to the evaluator at once.

    Deduplicated by `(deck, auxiliary seed, sample count, street)`, so all
    actions of one outer sample rank their boards once between them.

    It is an optimisation and nothing else: a street nobody primed is ranked on
    demand by `HandRunout.scores`, one call at a time but with the same numbers.
    """
    jobs, seen = [], set()
    for hand, turn in demands:
        k = (hand.key, int(turn))
        if k in seen or hand.cached(turn) is not None:
            continue
        seen.add(k)
        jobs.append((hand, int(turn)) + hand._job(turn))
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
        hand._scores[(hand.key, int(turn))] = scored[at:at + len(block)].reshape(
            len(comps), len(holes))
        at += len(block)
    assert at == len(scored), f"{at} of {len(scored)} ranked rows were claimed"
