"""Soft opponent ranges over the whole hand (CONCEPT.md §7.2, owner decision
2026-09-02).

`oracle/posterior.py` answers "what does seat *i* hold, given what they did up
to decision *k*" for **one** prefix, exactly. This module answers the same
question at **every** token of a hand, for **every** live opponent at once, and
it is what produces the target of the range head (§5.7).

It is the same estimator — the reach-weighted posterior

    w(combo) ∝ prior(combo) · Π_t  max(floor, P_i(a_t | combo, history_t))

— run as a filter instead of as a batch computation: the weights are carried
forward, each new action of that opponent multiplies them, and each new board
card removes the combos it blocks. With `prune = 0` that is *bit-identical* to
calling `opponent_posterior` at every prefix, and `tests/test_ranges.py` pins
exactly that. One definition of "opponent range" in the tree, two consumers.

**`prune` is the one approximation, and it buys forwards.** A combo whose
weight is below `prune` times the **largest** weight in that range is dropped
from the support permanently, so every later street asks the member about fewer
combos. Relative to the maximum and not absolute, for a reason that is not a
preference: a uniform prior over `C` combos gives every one of them `1 / C`, so
any absolute threshold near that scale empties the whole range at the first
token and any threshold below it never bites at all — the knob would do nothing
or everything depending on the size of the support. Against the maximum,
`prune = 1e-3` reads as "a thousand times less likely than the best combo" at
every street and every table size, and the mode always survives.

This is a cost knob, not a second definition: the mass it discards is reported
(`RangeStats.dropped`) so a run can show that the threshold threw away a tail
and not a mode. At `prune = 0` nothing is dropped and nothing is approximated.

**The order of the two operations in a token is the whole no-future-leak rule.**
The belief emitted at token *t* conditions on actions **strictly before** *t*,
and on the board **visible at** *t*. So a token first grows its dead set, then
emits, and only then multiplies in its own action. A player who folds at token
*t* still holds cards while that decision is being taken, so their range is
emitted at *t* and dropped from *t + 1* on — dropping them for the whole hand
would condition the target on the future.
"""

import itertools

import numpy as np

from oracle.posterior import _context_of, combo_universe

N_CARDS = 52
N_COMBOS = 1326          # C(52, 2)

# The canonical order every 1326-vector in v8 is indexed by: ascending within a
# combo, lexicographic between them — `combo_universe([])`, tabulated. Every
# other order in this file is a subset of it, never a re-sort.
COMBO_CARDS = np.asarray(list(itertools.combinations(range(N_CARDS), 2)),
                         dtype=np.int64)
assert len(COMBO_CARDS) == N_COMBOS

_COMBO_INDEX = np.full((N_CARDS, N_CARDS), -1, dtype=np.int64)
_COMBO_INDEX[COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]] = np.arange(N_COMBOS)
_COMBO_INDEX[COMBO_CARDS[:, 1], COMBO_CARDS[:, 0]] = np.arange(N_COMBOS)


def combo_index(combos):
    """(C,) canonical indices of a (C, 2) array of card pairs."""
    combos = np.asarray(combos, dtype=np.int64).reshape(-1, 2)
    idx = _COMBO_INDEX[combos[:, 0], combos[:, 1]]
    assert (idx >= 0).all(), "a combo must be two distinct cards in 0..51"
    return idx


class RangeStats:
    """What one hand's tracking cost, for the same reason `LabelStats` exists."""

    __slots__ = ("forwards", "dropped", "emitted", "collapsed")

    def __init__(self):
        self.forwards = 0      # policy rows actually evaluated
        self.dropped = 0.0     # total posterior mass thrown away by `prune`
        self.emitted = 0       # (token, seat) targets produced
        self.collapsed = 0     # supports a board card wiped out entirely

    def merge(self, other):
        self.forwards += other.forwards
        self.dropped += other.dropped
        self.emitted += other.emitted
        self.collapsed += other.collapsed
        return self


class RangeTracker:
    """The Bayes filter of one observer's beliefs about every live opponent.

    Constructed at the top of a hand and stepped through it in decision order.
    `emit` is the belief *before* the current token's action, `apply` folds that
    action in. The two are separate calls because the caller decides which
    tokens it wants a target at (`emit_at` in `hand_ranges`) while every token's
    action still has to be conditioned on.
    """

    def __init__(self, record, observer_pos, pool, n_actions, floor=1e-6,
                 prune=0.0):
        n = record.num_players
        assert 0 <= observer_pos < n, (
            f"observer seat {observer_pos} is not at this {n}-handed table")
        self.record = record
        self.observer_pos = int(observer_pos)
        self.pool = pool
        self.n_actions = int(n_actions)
        self.floor = float(floor)
        self.prune = float(prune)
        self.stats = RangeStats()

        dead = record.hole_cards(observer_pos)
        combos = combo_universe(dead)
        prior = np.full(len(combos), 1.0 / len(combos), dtype=np.float64)
        # One independent marginal per opponent (§7.3): the joint is not
        # represented here and is not what the head is asked to predict.
        self.live = {seat: (combos.copy(), prior.copy())
                     for seat in range(n) if seat != self.observer_pos}
        self._dead_board = set()

    # ------------------------------------------------------------- the filter

    def advance(self, turn):
        """Card removal for the board visible on `turn`. Idempotent.

        Called at **every** token, emitted or not: asking a member about a combo
        the board already holds is a wasted forward and a nonsense question, and
        the answer would be discarded at the next removal anyway.

        **A pruned support can be wiped out by a board card** — the mode of a
        narrow range can be exactly the card that comes off. That is the one
        place `prune` can do more than shorten a tail, so it does not assert: the
        seat falls back to the uniform prior over what the observer can still
        rule out, and `RangeStats.collapsed` counts it. With `prune = 0` the
        universe always survives and the counter stays at zero.
        """
        board = [int(c) for c in self.record.deck[:(0, 3, 4, 5)[int(turn)]]]
        fresh = [c for c in board if c not in self._dead_board]
        if not fresh:
            return
        self._dead_board.update(fresh)
        gone = np.asarray(fresh, dtype=np.int64)
        prior = None
        for seat, (combos, weights) in list(self.live.items()):
            keep = ~np.isin(combos, gone).any(axis=1)
            if not keep.any():
                if prior is None:
                    prior = combo_universe(
                        sorted(self._dead_board)
                        + self.record.hole_cards(self.observer_pos))
                self.stats.collapsed += 1
                self.live[seat] = (
                    prior.copy(),
                    np.full(len(prior), 1.0 / len(prior), dtype=np.float64))
                continue
            combos, weights = combos[keep], weights[keep]
            self.live[seat] = (combos, weights / weights.sum())

    def emit(self, seats=None):
        """`{seat: (combo_idx int16, weight float32)}` before this token's action.

        `advance` must have been called for this token's street first, so the
        dead set is the board as the observer sees it *at that token* — one
        street ahead of what a posterior conditioned through the previous
        decision would have used.
        """
        out = {}
        for seat, (combos, weights) in self.live.items():
            if seats is not None and seat not in seats:
                continue
            out[seat] = (combo_index(combos).astype(np.int16),
                         weights.astype(np.float32))
            self.stats.emitted += 1
        return out

    def apply(self, t):
        """Condition on decision `t`, then prune. Drops a seat that folded."""
        dec = self.record.decisions[t]
        pos = int(dec["acting_pos"])
        if pos in self.live:
            combos, weights = self.live[pos]
            contexts = [_context_of(self.record, dec, hole_override=combo)
                        for combo in combos]
            probs = np.asarray(self.pool[dec["member"]].policy(contexts),
                               dtype=np.float64)
            C = len(combos)
            assert probs.shape == (C, self.n_actions), (
                f"pool member {dec['member']} returned {probs.shape}, expected "
                f"{(C, self.n_actions)}")
            self.stats.forwards += C
            weights = weights * np.maximum(
                probs[:, int(dec["action_idx"])], self.floor)
            total = weights.sum()
            # A positive floor makes this unreachable; if it ever is reached the
            # honest answer is the previous belief, not a vector of NaN.
            weights = (weights / total if total > 0.0
                       else np.full(C, 1.0 / C, dtype=np.float64))

            if self.prune > 0.0:
                keep = weights >= self.prune * weights.max()
                assert keep.any(), (
                    "the mode of a range is always at least `prune` times "
                    "itself, so a support can never come out empty here")
                self.stats.dropped += float(weights[~keep].sum())
                combos, weights = combos[keep], weights[keep]
                weights = weights / weights.sum()
            self.live[pos] = (combos, weights)

        # The fold is folded in *after* the belief at this token was emitted.
        if int(dec["action_idx"]) == 0:
            self.live.pop(pos, None)
        return self.stats


class HandRangeCache:
    """One `RangeTracker` carried across the decisions of a hand.

    Labels are produced in decision order, and a label wants the belief at its
    own decision only. Rebuilding the filter per label would re-ask every
    earlier action of every opponent — quadratic in the hand, and the whole
    reason `PosteriorCache` exists next to `opponent_posterior`. This is the
    same trick for the same reason: keep the tracker, advance it, emit.

    Out-of-order use (a decision earlier than the cursor, a different record or
    observer) rebuilds rather than guessing, so the answer never depends on
    which order the caller happened to ask in.
    """

    def __init__(self, floor=1e-6, prune=0.0):
        self.floor = float(floor)
        self.prune = float(prune)
        self._key = None
        self._tracker = None
        self._cursor = 0

    def clear(self):
        self._key = None
        self._tracker = None
        self._cursor = 0

    def target(self, record, observer_pos, pool, n_actions, decision_idx):
        """`({seat: (combo_idx, weight)}, forwards)` before `decision_idx`."""
        key = (id(record), int(observer_pos), id(pool), int(n_actions))
        if key != self._key or decision_idx < self._cursor:
            self._key = key
            self._tracker = RangeTracker(record, observer_pos, pool, n_actions,
                                         self.floor, self.prune)
            self._cursor = 0
        before = self._tracker.stats.forwards
        while self._cursor < int(decision_idx):
            dec = record.decisions[self._cursor]
            self._tracker.advance(record.snapshots[dec["snap_idx"]]["turn"])
            self._tracker.apply(self._cursor)
            self._cursor += 1
        turn = record.snapshots[
            record.decisions[int(decision_idx)]["snap_idx"]]["turn"]
        self._tracker.advance(turn)
        return (self._tracker.emit(),
                self._tracker.stats.forwards - before)


def hand_ranges(record, observer_pos, pool, n_actions, floor=1e-6, prune=0.0,
                emit_at=None, pending_turn=None):
    """`{(token, seat): (combo_idx, weight)}` over one hand, plus `RangeStats`.

    Token indices are the ones `nets.features.hand_tokens` produces: decision
    *t* is token *t*, and a pending decision is token `len(record.decisions)`.

    Args:
        emit_at: token indices to produce a target at. `None` means every
            decision token — what the embedding corpus wants. A label wants the
            pending token alone, and passing that set skips the *storage* of the
            rest while still conditioning on every action, which is the part
            that costs forwards.
        pending_turn: the street of a pending decision. Given, one more target
            is emitted at token `len(record.decisions)`.
    """
    tracker = RangeTracker(record, observer_pos, pool, n_actions, floor, prune)
    out = {}
    last = None if emit_at is None else max(emit_at, default=-1)
    for t, dec in enumerate(record.decisions):
        if last is not None and t > last:
            break                       # nothing left to emit; stop paying
        tracker.advance(record.snapshots[dec["snap_idx"]]["turn"])
        if emit_at is None or t in emit_at:
            for seat, target in tracker.emit().items():
                out[(t, seat)] = target
        tracker.apply(t)
    if pending_turn is not None:
        t = len(record.decisions)
        tracker.advance(pending_turn)
        if emit_at is None or t in emit_at:
            for seat, target in tracker.emit().items():
                out[(t, seat)] = target
    return out, tracker.stats


def label_ranges(sessions, pool, n_actions, floor=1e-6, prune=0.0, desc=None):
    """§5.7 targets over a whole corpus, in the shape `Session.tokens` reads.

    The same place in the pipeline as `env.showdown.label_showdowns`, and for
    the same reason: it is a property of the played hands and of the members
    that played them, so it is computed **once** over the corpus and never
    inside a training loop. One `Session.ranges` per session, one entry per
    hand, keyed by (token, seat).

    The observer is slot 0's seat in that hand — hero, the only view a corpus
    hand is ever tokenised from (`env/session.py`).

    Returns the merged `RangeStats`. `dropped` is the mass `prune` threw away
    and `collapsed` the number of supports a board card wiped out; a run that
    reports either as a large fraction has a threshold set past a tail and into
    a mode, which is the one way this knob can change the target rather than
    only its cost.
    """
    from utils import progress

    total = sum(len(s.records) for s in sessions)
    stats = RangeStats()
    bar = progress(total=total, desc=desc or "ranges", unit="hand")
    for session in sessions:
        session.ranges = []
        for h, record in enumerate(session.records):
            ranges, one = hand_ranges(
                record, observer_pos=session.seat_of_slot(0, h), pool=pool,
                n_actions=n_actions, floor=floor, prune=prune)
            session.ranges.append(ranges)
            stats.merge(one)
            bar.update(1)
    bar.close()
    return stats


def label_range_target(cache, session, hand_idx, decision_idx, pool, n_actions):
    """The §5.7 target of one labelled decision, keyed as `hand_tokens` wants.

    A label's tokens are the hand truncated at `decision_idx` with one pending
    token last, so that token's index *is* `decision_idx` and the belief it
    carries is the one held before that decision — the same prefix the oracle
    conditions its posteriors on. `None` when the head is off.

    The cost is one pass of that hand's opponent likelihoods, which is what the
    oracle's own posterior already costs, so a label pays it roughly twice
    (~10 % of a heads-up label, measured against G3's 1225 posterior rows in
    ~11k). Sharing one computation between the two would mean threading the
    oracle's internal ranges out through `action_values`, whose three-tuple every
    caller and a dozen tests read; that is the trade, and it is written down
    rather than assumed.
    """
    if cache is None:
        return None
    record = session.records[hand_idx]
    observer_pos = session.seat_of_slot(0, hand_idx)
    targets, _rows = cache.target(record, observer_pos, pool, n_actions,
                                  decision_idx)
    return {(int(decision_idx), seat): value for seat, value in targets.items()}
