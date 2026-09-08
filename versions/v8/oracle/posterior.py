"""Reach-weighted opponent ranges (CONCEPT.md §7.2).

"The oracle knows the opponents' strategies, therefore it knows their ranges" is
right only if the range is the *posterior implied by what they did*, not the set
of hands that could still be in front of them:

    w(combo) ∝ prior(combo) · Π_t  max(floor, P_i(a_t | combo, history_t))

over every action `a_t` opponent *i* took in this hand, under their own policy.
This is the same soft-Bayes belief v7 computed in
`generation/generate_opponent.py`, including its two fixes — a likelihood floor,
and card removal taken **relative to the observer**.

Three properties are what the tests pin, and each is a way of getting it wrong:

* **The observer's information, and no more.** The dead cards are the board as
  visible at the conditioning decision plus the observer's own two cards. The
  opponent's real holding is *in* the universe — it has to be, it is what the
  posterior is over — and nothing downstream of here may look at it.
* **A prefix function.** Conditioning stops at `through_decision`, so the
  posterior at a decision is identical whether or not the hand was played on.
  That is CONCEPT.md §9's no-future-leak rule, inside the oracle.
* **One question per decision.** A member is asked once per decision it made,
  with all `C` combos in a single batch — an opponent with five decisions costs
  five batched forwards of ~1200 rows, not 5·1200 forwards of one.

A member is asked about a hypothetical holding through `DecisionContext`'s
`hole_override`: the same situation, other cards, so the observation the member
answers is built by exactly the code that built the real one.

**Reach factors (§7.3).** This module returns one seat's normalised likelihood
factor, including its fold if any. It is not that seat's full marginal after
conditioning on every other player's actions. For fixed policies and frozen
hand-start memory, the joint is proportional to the product of these factors
times the indicator of disjoint cards. The oracle samples that joint by
rejection, including folded seats; it need not enumerate the full joint.

**`max_combos` (§7.3) — self-normalised importance sampling over the prior.**
v7's `gpu_solver_v5._prepare_range` draws the subsample *from the posterior*
(`rng.choice(..., p=w)`) and renormalises what it kept. Two things are wrong with
that here, and both are fixed:

* it is **biased**. Renormalising the kept weights is the right ratio estimator
  only if the subset was drawn independently of them; drawing the subset ∝ `w`
  and then renormalising counts the mode twice — once in the selection, once in
  the weight — and the kept weights are never divided by their inclusion
  probabilities to undo it;
* it happens **after** the likelihood pass, so it caps the range that is returned
  and saves not one policy call.

So the cap is applied to the **prior**, before any member is asked anything:
`max_combos` combos are drawn uniformly without replacement, each is carried at
`prior / inclusion probability`, and the likelihood pass then runs over that
subsample alone. Simple random sampling without replacement has an exact
inclusion probability of `m / C` for every combo, so with a uniform prior the
correction is a constant — it is written out anyway, because a constant that
cancels is easier to check than a constant that was assumed. Normalising at the
end makes the whole thing the standard self-normalised importance-sampling ratio
estimator of the posterior, consistent in `m` and with no selection bias.

The proposal is the prior rather than the posterior for a reason that is not a
preference: a posterior-weighted proposal cannot be drawn before the likelihoods
that define it exist, and its without-replacement inclusion probabilities have no
closed form to divide out. The price is variance — a sharply concentrated range
needs a larger `m` than v5's mode-seeking draw would — and that is the trade the
cap now makes: cost falls with `m`, noise rises with it, bias stays at zero.
"""

import itertools
import warnings
from dataclasses import dataclass

import numpy as np

from env.driver import DecisionContext

N_CARDS = 52


def combo_universe(dead_cards):
    """All 2-card combos of the cards not in `dead_cards`. (C, 2) int64.

    Ascending within a combo and lexicographic between them, so the order is a
    function of the dead set alone — every weight vector in this module is
    indexed by it.
    """
    dead = set(int(c) for c in dead_cards if int(c) >= 0)
    live = [c for c in range(N_CARDS) if c not in dead]
    combos = list(itertools.combinations(live, 2))
    return np.asarray(combos, dtype=np.int64).reshape(len(combos), 2)


def _context_of(record, dec, hole_override=None):
    """The `DecisionContext` a recorded decision was taken under."""
    snap = record.snapshots[dec["snap_idx"]]
    return DecisionContext(record, dec["snap_idx"], int(dec["acting_pos"]),
                           dec["legal_mask"], int(snap["turn"]),
                           hole_override=hole_override)


def _visible_board(record, through_decision):
    """Board cards face up at `through_decision`; empty before the first one."""
    if through_decision < 0:
        return []
    ctx = _context_of(record, record.decisions[through_decision])
    return [c for c in ctx.board if c >= 0]


def _validate_request(record, opp_pos, observer_pos, through_decision):
    """Shared argument checks for fresh and incrementally cached ranges."""
    n = record.num_players
    assert 0 <= opp_pos < n and 0 <= observer_pos < n, (
        f"seats {opp_pos} and {observer_pos} must both be at this {n}-handed table")
    assert opp_pos != observer_pos, (
        "the observer knows its own cards — there is no posterior to compute")
    n_dec = len(record.decisions)
    assert -1 <= through_decision < n_dec, (
        f"through_decision {through_decision} is not a decision index of a hand "
        f"with {n_dec} decisions (-1 is the empty prefix)")


@dataclass
class _CachedPosterior:
    """One exact-prefix posterior, kept only while its hand is current."""

    combos: np.ndarray
    weights: np.ndarray
    through_decision: int
    dead: frozenset


class PosteriorCache:
    """Incrementally extend opponent posteriors through one finished hand.

    Labels are produced in decision order.  At hero's next decision, an
    opponent's range differs from the previous one only by newly visible board
    cards and by the opponent actions taken since that prefix.  Re-evaluating
    every earlier action over every surviving combo is therefore redundant:
    Bayes' rule lets us restrict the previous weights by card removal and
    multiply only the new likelihoods.

    The cache intentionally holds one hand.  Label generation is ordered by
    hand, so retaining older records would grow memory without creating hits.
    It is also intentionally disabled when `max_combos` is set: those ranges
    use a fresh, per-label random prior subsample, and reusing one would change
    both the estimator and the label RNG stream.  Production's exact-posterior
    path (`max_combos=None`) has neither issue.

    `posterior` returns `(combos, weights, rows)`, where `rows` counts policy
    rows actually evaluated by this call.  This keeps `LabelStats.forwards`
    honest after a cache hit.
    """

    def __init__(self):
        self._record = None
        self._entries = {}

    def clear(self):
        self._record = None
        self._entries.clear()

    def posterior(self, record, opp_pos, observer_pos, pool, n_actions,
                  through_decision, floor=1e-6, max_combos=None, rng=None):
        # A positive floor guarantees that a valid cached posterior never went
        # through the zero-likelihood fallback.  With floor <= 0, preserve the
        # standalone function's exact fallback and RNG semantics instead.
        cacheable = max_combos is None and float(floor) > 0.0
        if record is not self._record:
            self._record = record
            self._entries.clear()

        if not cacheable:
            combos, weights = opponent_posterior(
                record, opp_pos, observer_pos, pool, n_actions,
                through_decision, floor=floor, max_combos=max_combos, rng=rng)
            acted = sum(
                1 for dec in record.decisions[:through_decision + 1]
                if int(dec["acting_pos"]) == int(opp_pos))
            return combos, weights, len(combos) * acted

        _validate_request(record, opp_pos, observer_pos, through_decision)
        key = (int(opp_pos), int(observer_pos), id(pool), int(n_actions),
               float(floor))
        entry = self._entries.get(key)
        new_dead = frozenset(
            _visible_board(record, through_decision)
            + record.hole_cards(observer_pos))

        # Prefixes normally advance monotonically.  Any out-of-order use, or a
        # dead-card set that is not an extension of the cached observer view,
        # simply takes the original path rather than guessing.
        reusable = (entry is not None
                    and through_decision >= entry.through_decision
                    and entry.dead.issubset(new_dead))
        if not reusable:
            combos, weights = opponent_posterior(
                record, opp_pos, observer_pos, pool, n_actions,
                through_decision, floor=floor, max_combos=max_combos, rng=rng)
            acted = sum(
                1 for dec in record.decisions[:through_decision + 1]
                if int(dec["acting_pos"]) == int(opp_pos))
            rows = len(combos) * acted
            if np.isfinite(weights).all():
                self._entries[key] = _CachedPosterior(
                    combos=combos.copy(), weights=weights.copy(),
                    through_decision=int(through_decision), dead=new_dead)
            return combos, weights, rows

        combos = entry.combos
        weights = entry.weights
        appeared = new_dead.difference(entry.dead)
        if appeared:
            # `combo_universe` is lexicographic.  Removing newly dead cards
            # from its previous result is exactly the new universe, in exactly
            # the same order, without rebuilding all combinations in Python.
            dead_now = np.fromiter(appeared, dtype=np.int64)
            keep = ~np.isin(combos, dead_now).any(axis=1)
            combos = combos[keep]
            weights = weights[keep]
        else:
            # Never hand the cache's owned weights to code that might mutate
            # the returned array in place.
            weights = weights.copy()

        C = len(combos)
        assert C > 0, "no combo survives card removal"
        rows = 0
        for t in range(entry.through_decision + 1, through_decision + 1):
            dec = record.decisions[t]
            if int(dec["acting_pos"]) != int(opp_pos):
                continue
            contexts = [_context_of(record, dec, hole_override=combo)
                        for combo in combos]
            probs = np.asarray(pool[dec["member"]].policy(contexts),
                               dtype=np.float64)
            assert probs.shape == (C, n_actions), (
                f"pool member {dec['member']} returned {probs.shape}, expected "
                f"{(C, n_actions)}")
            weights *= np.maximum(probs[:, int(dec["action_idx"])], floor)
            rows += C

        total = weights.sum()
        if total > 0.0:
            weights /= total
        else:  # Defensive: positive `floor` should make this unreachable.
            warnings.warn(
                f"every combo has zero likelihood for seat {opp_pos} through "
                f"decision {through_decision}; falling back to the prior",
                RuntimeWarning)
            weights = np.full(C, 1.0 / C, dtype=np.float64)
            self._entries.pop(key, None)
            return combos, weights, rows

        if np.isfinite(weights).all():
            self._entries[key] = _CachedPosterior(
                combos=combos.copy(), weights=weights.copy(),
                through_decision=int(through_decision), dead=new_dead)
        else:
            self._entries.pop(key, None)
        return combos, weights, rows


def opponent_posterior(record, opp_pos, observer_pos, pool, n_actions,
                       through_decision, floor=1e-6, max_combos=None, rng=None):
    """Reach-weighted posterior over `opp_pos`'s holding, from `observer_pos`'s view.

    Returns (combos, weights) — (C, 2) int64 and (C,) float64 summing to 1.
    `through_decision` is an index into `record.decisions`; only decisions at or
    before it are conditioned on, so the posterior is a prefix function. `-1` is
    the empty prefix: nobody has acted, so the answer is the prior.

    Args:
        record: the `env.driver.HandRecord` being reasoned about.
        opp_pos: the seat whose holding is unknown.
        observer_pos: the seat whose information defines the posterior.
        pool: the members `record.decisions[t]["member"]` indexes.
        n_actions: size of the action set — checks what a member returns.
        floor: likelihood floor, so one unexpected action cannot annihilate a
            combo the member merely finds unlikely.
        max_combos: cap on how many combos the members are asked about — the
            range is subsampled from the prior first, so this is a cost knob
            and not only a shorter answer. See the module docstring.
        rng: `np.random.Generator`, required only when the cap actually bites.
    """
    _validate_request(record, opp_pos, observer_pos, through_decision)

    dead = _visible_board(record, through_decision) + record.hole_cards(observer_pos)
    combos = combo_universe(dead)
    C = len(combos)
    assert C > 0, "no combo survives card removal"
    weights = np.full(C, 1.0 / C, dtype=np.float64)

    # The cap is spent on the prior, so it buys policy calls and not just a
    # shorter answer: everything below asks the members about `m` combos.
    if max_combos is not None and C > int(max_combos):
        assert rng is not None, (
            "subsampling a range needs an explicit np.random.Generator")
        m = int(max_combos)
        keep = np.sort(rng.choice(C, size=m, replace=False))
        inclusion = m / C          # exact for sampling without replacement
        combos = combos[keep]
        weights = weights[keep] / inclusion
        C = m

    for t in range(through_decision + 1):
        dec = record.decisions[t]
        if int(dec["acting_pos"]) != opp_pos:
            continue
        contexts = [_context_of(record, dec, hole_override=combo)
                    for combo in combos]
        probs = np.asarray(pool[dec["member"]].policy(contexts), dtype=np.float64)
        assert probs.shape == (C, n_actions), (
            f"pool member {dec['member']} returned {probs.shape}, expected "
            f"{(C, n_actions)}")
        weights = weights * np.maximum(probs[:, int(dec["action_idx"])], floor)

    total = weights.sum()
    if total > 0.0:
        weights = weights / total
    else:
        # The member assigns zero probability to what it did. That is a bug in
        # the member or in the record, not in the data — say so and fall back to
        # the prior rather than returning NaN.
        warnings.warn(
            f"every combo has zero likelihood for seat {opp_pos} through "
            f"decision {through_decision}; falling back to the prior",
            RuntimeWarning)
        weights = np.full(C, 1.0 / C, dtype=np.float64)

    return combos, weights
