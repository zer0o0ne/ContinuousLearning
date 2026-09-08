"""The BR oracle, variant A — full rollout (CONCEPT.md §7.1).

At one hero decision, for every legal action *a*: apply *a*, play the hand out
against the pool, and average hero's chip delta. The average is over a joint
assignment of hole cards to the opponents drawn from their reach-weighted
posteriors (§7.2), so `Q(s, a)` is the EV of *a* against the pool's actual
strategies, up to the approximations §7.3 declares.

Three things make this affordable, and all three are compositions of pieces that
already exist rather than new machinery:

* **A rollout is a `HandSpec`.** `deck` pins the cards, `forced_actions` replays
  the prefix that led to the decision and then plays *a*. The prefix costs no
  policy call at all (`env/driver.py`), so the only forwards paid for are the
  ones after hero acted.
* **One lock-step batch per label.** Every (action, sample) pair is an
  independent hand, so a single hero decision is one `driver.run` of up to
  `|A| × samples_per_action` hands — the batch shape the GB10 wants
  (`CLAUDE.md` §3), not `|A| × S` batches of one.
* **Common random numbers.** The joint sample is drawn *once* and every legal
  action is rolled out on the same assignments, so the differences between the
  `Q`s — which is what a target is built from — are not swamped by the card
  variance the actions share.

**The card-leak rule.** The opponents' cards are fixed *in the deck*, and
nowhere else. Hero's member is queried through `DecisionContext`, whose
`hole_cards` reads hero's own seat; there is no path from an opponent's sampled
holding into hero's observation. `tests/test_oracle.py` asserts it by recording
every holding hero's member is ever handed, not by reading this file.

**Why the runout is dealt, not reused.** Only the board hero can *see* at the
decision is fixed; the streets still to come are dealt afresh for every sample,
out of whatever the assignment left in the deck. §7.1 defines `Q` as the EV over
everything hero does not know, and the turn and the river are part of that. An
earlier version pinned the whole five-card board of the hand the decision came
out of, which makes the label an estimate of `E[chips | this exact river]` — a
quantity whose error `samples_per_action` cannot reduce at all, because every
sample shares the one runout. It is drawn *after* the opponents' combos, so the
decomposition is `p(hands | visible board) · p(runout | hands)`: the posterior
conditions on what hero saw, the runout comes out of what is left, and both are
exact. The draw is per sample and not per action, so common random numbers still
hold — the actions of one sample are compared on one board.

**Fold needs no special case.** Hero's chip delta after folding is minus what
hero has already put in, whatever the opponents hold, so the rollout returns the
closed form with zero variance. Special-casing it would be a second path to the
same number.
"""

import hashlib
import time
from dataclasses import dataclass, replace

import numpy as np

from env.runout import RunoutConfig
from oracle.posterior import (PosteriorCache, _visible_board,
                              opponent_posterior)
from oracle.ranges import HandRangeCache

FOLD = 0
N_CARDS = 52
RANGE_MODEL = "all_seats_reach_v2"


@dataclass
class OracleConfig:
    """The knobs of one label. Every one of them trades cost against noise."""
    samples_per_action: int = 256
    max_combos: int = None
    likelihood_floor: float = 1e-6
    batch_hands: int = 2048
    max_collision_retries: int = 32
    labels_per_batch: int = 1   # how many labels share one `driver.run` (§7.1)
    # Variance reduction (`env/runout.py`): subtract the luck of the cards and
    # of the opponents' fold/no-fold draws from every rollout, and integrate the
    # board out entirely once a rollout has no decisions left. Unbiased — it
    # changes the noise on a label, never what the label estimates.
    control_variate: bool = True
    runout_samples: int = 16    # boards averaged over; exhaustive when it fits
    # §5.7 — the belief target that travels with a label. `range_target` off
    # means no target is produced at all, which is the head's ablation; the
    # threshold is the support cut of `oracle/ranges.py` and buys forwards.
    range_target: bool = False
    range_prune: float = 0.0

    def runout_config(self):
        """What `LockstepDriver` needs to produce the reduced value, or None."""
        if not self.control_variate:
            return None
        return RunoutConfig(samples=int(self.runout_samples))

    def range_cache(self):
        """A `HandRangeCache` for the §5.7 target, or `None` with the head off."""
        if not self.range_target:
            return None
        return HandRangeCache(floor=self.likelihood_floor,
                              prune=float(self.range_prune))


@dataclass
class LabelStats:
    """What one label cost. G3 (S4, CONCEPT.md §14) reads exactly this."""
    forwards: int          # policy rows evaluated: posterior + rollouts
    seconds: float         # wall clock of the whole call
    collision_rate: float  # fraction of joint draws rejected (§7.3)
    n_rollouts: int        # hands actually played


def _rollout_seed(*parts):
    """A seed that depends on the label and on nothing else.

    The hand's cards and its action draws come from this, so a label has to be
    reproducible independently of which other rollouts shared its batch (§15).
    A hash rather than a linear combination, so consecutive samples do not get
    consecutive seeds.
    """
    key = ",".join(str(int(p)) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _rollout_deck(record, hero_pos, opp_seats, opp_cards, n_visible, rng):
    """The 52-card deck of one rollout.

    The board hero can see, hero's real cards, the sampled cards at the
    opponents' seats (including folded ones), the streets still to come drawn
    from what is left, and the undealt stub shuffled into the remaining slots.
    All hole cards are sampled jointly from reach factors before the runout;
    `env/runout.py` then conditions its board completions on that same complete
    assignment. Folded players stay folded in the forced action prefix, but
    their cards still block both live holdings and future community cards.
    """
    deck = np.full(N_CARDS, -1, dtype=np.int64)
    deck[:n_visible] = [int(c) for c in record.deck[:n_visible]]
    deck[5 + 2 * hero_pos: 7 + 2 * hero_pos] = record.hole_cards(hero_pos)
    for seat, cards in zip(opp_seats, opp_cards):
        deck[5 + 2 * seat: 7 + 2 * seat] = cards

    used = np.zeros(N_CARDS, dtype=bool)
    used[deck[deck >= 0]] = True
    free = np.flatnonzero(~used)
    n_hidden = 5 - n_visible
    if n_hidden:
        drawn = rng.choice(free, size=n_hidden, replace=False)
        deck[n_visible:5] = drawn
        free = free[~np.isin(free, drawn)]
    deck[np.flatnonzero(deck < 0)] = rng.permutation(free)
    return deck


def _sample_joint(posteriors, dead_mask, n_samples, max_retries, rng):
    """Sample the collision-free product of opponents' reach factors (§7.3).

    With fixed policies, the public action likelihood factorises by seat.
    Independent proposals from those factors followed by **rejection** sample
    the joint exactly: a draw in which two opponents share a card, or in
    which a card is already on the *visible* board or in hero's hand, is thrown
    away and redrawn. A sample that has not survived `max_retries` redraws is dropped —
    it reduces the divisor of the average, it is not an outcome of zero.

    Returns `(cards, n_attempts, n_rejected)` with `cards` of shape
    `(kept, n_opponents, 2)`. The retry cap reduces sample count, not accuracy
    of the accepted distribution. Floors and combo subsampling still change
    the reach factors; rejection does not undo those approximations.
    """
    packs = [(combos, np.cumsum(weights)) for combos, weights in posteriors]
    n_opp = len(packs)

    kept = []
    pending = int(n_samples)
    attempts = 0
    for _ in range(int(max_retries) + 1):
        if pending == 0:
            break
        draw = np.empty((pending, n_opp, 2), dtype=np.int64)
        for j, (combos, cum) in enumerate(packs):
            # side="right" on the cumulative weights skips zero-weight combos:
            # a combo the posterior has ruled out can never be drawn.
            pick = np.searchsorted(cum, rng.random(pending), side="right")
            draw[:, j] = combos[np.minimum(pick, len(combos) - 1)]
        attempts += pending

        flat = draw.reshape(pending, 2 * n_opp)
        ok = ~dead_mask[flat].any(axis=1)
        if n_opp:
            ordered = np.sort(flat, axis=1)
            ok &= (np.diff(ordered, axis=1) != 0).all(axis=1)
        kept.append(draw[ok])
        pending = int((~ok).sum())

    cards = (np.concatenate(kept, axis=0) if kept
             else np.empty((0, n_opp, 2), dtype=np.int64))
    return cards, attempts, attempts - len(cards)


def _label_plan(record, decision_idx, driver, pool, hero_member_idx, cfg, rng,
                posterior_cache=None):
    """Build one label's rollouts without playing them.

    Returns a `_LabelPlan`; `action_values` plays it and `_label_q` reduces it
    to `(q, legal, stats)` — `q` is `(n_actions,)` float64 with `nan` at illegal
    actions, `legal` the recorded mask, `stats` a `LabelStats`.

    `nan` rather than zero at an illegal action is deliberate: S6 masks them,
    and a masking bug then fails loudly instead of quietly averaging in a value
    that was never computed. The same `nan` is what a label reports when every
    joint draw collided and nothing could be rolled out.

    Args:
        record: the `env.driver.HandRecord` the decision belongs to.
        decision_idx: index into `record.decisions`. Nothing after it is read,
            so the label does not depend on how the hand went on (§9).
        driver: a `LockstepDriver` over `pool` — the rollouts run through it.
        pool: the members the record's `member` indices refer to.
        hero_member_idx: the member seated in hero's seat *for the rollouts*.
            At iteration 0 a v7 member, from iteration 1 the agent (§7.1).
        cfg: an `OracleConfig`.
        rng: `np.random.Generator` for the combo subsample and the joint draw.
        posterior_cache: optional `PosteriorCache` shared by consecutive labels
            of the same hand.  It changes only how repeated prefix likelihoods
            are computed, not the posterior or rollout samples.
    """
    started = time.perf_counter()
    n_dec = len(record.decisions)
    assert 0 <= decision_idx < n_dec, (
        f"decision {decision_idx} is not a decision of a hand with {n_dec}")

    n_actions = driver.n_actions
    dec = record.decisions[decision_idx]
    hero_pos = int(dec["acting_pos"])
    legal = np.asarray(dec["legal_mask"], dtype=bool).copy()
    legal_idx = np.flatnonzero(legal).tolist()
    q = np.full(n_actions, np.nan, dtype=np.float64)

    # A fold is evidence about two cards still missing from the deck. Include
    # every opponent's full action likelihood (the fold itself included), so
    # conditioning changes live ranges and the runout through card removal.
    opp_seats = [p for p in range(record.num_players) if p != hero_pos]

    forwards = 0
    posteriors = []
    for seat in opp_seats:
        if posterior_cache is None:
            combos, weights = opponent_posterior(
                record, seat, hero_pos, pool, n_actions,
                through_decision=decision_idx - 1,
                floor=cfg.likelihood_floor,
                max_combos=cfg.max_combos, rng=rng)
            acted = sum(1 for d in record.decisions[:decision_idx]
                        if int(d["acting_pos"]) == seat)
            forwards += len(combos) * acted
        else:
            combos, weights, rows = posterior_cache.posterior(
                record, seat, hero_pos, pool, n_actions,
                through_decision=decision_idx - 1,
                floor=cfg.likelihood_floor,
                max_combos=cfg.max_combos, rng=rng)
            forwards += rows
        posteriors.append((combos, weights))

    # Hero's information and no more: the board hero can see at this decision
    # plus hero's own cards. A card of a street still to come is not dead — it
    # is in the deck, and an opponent holding it is exactly a runout that
    # cannot happen, which is what dealing the runout after the combos says.
    visible = _visible_board(record, decision_idx)
    dead_mask = np.zeros(N_CARDS, dtype=bool)
    dead_mask[[int(c) for c in visible]] = True
    dead_mask[record.hole_cards(hero_pos)] = True

    samples, attempts, rejected = _sample_joint(
        posteriors, dead_mask, cfg.samples_per_action,
        cfg.max_collision_retries, rng)

    prefix = [int(d["action_idx"]) for d in record.decisions[:decision_idx]]
    seat_members = list(record.spec.seat_members)
    seat_members[hero_pos] = int(hero_member_idx)

    specs = []
    for s in range(len(samples)):
        deck = _rollout_deck(record, hero_pos, opp_seats, samples[s],
                             len(visible), rng)
        for a in legal_idx:
            specs.append(replace(
                record.spec, seat_members=seat_members, deck=deck,
                forced_actions=prefix + [a],
                seed=_rollout_seed(record.spec.seed, decision_idx, a, s)))

    return _LabelPlan(specs=specs, q=q, legal=legal, legal_idx=legal_idx,
                      n_samples=len(samples), hero_pos=hero_pos,
                      prefix_len=len(prefix),
                      big_blind=float(record.spec.big_blind),
                      forwards=forwards, attempts=attempts, rejected=rejected,
                      prepared=time.perf_counter() - started)


@dataclass
class _LabelPlan:
    """One label's rollouts, built but not yet played.

    Splitting the label here is what lets several of them share one
    `driver.run`: everything above depends on the labelled decision alone, and
    everything below is arithmetic over that decision's own hands. Nothing
    crosses between labels — the specs carry their own seats, decks and seeds.
    """
    specs: list
    q: np.ndarray
    legal: np.ndarray
    legal_idx: list
    n_samples: int
    hero_pos: int
    prefix_len: int
    big_blind: float
    forwards: int
    attempts: int
    rejected: int
    prepared: float


def hero_values(played, hero_pos):
    """Hero's value in each rollout, in chips.

    The variance-reduced one where the driver was configured to produce it
    (`env/runout.py`), the raw chip delta otherwise. Both are unbiased for the
    same expectation, so only the noise on the average differs — which is why
    everything that averages rollouts, the oracle and G3's split-half alike,
    reads them through here rather than off `rewards`.
    """
    return np.asarray(
        [(r.rewards if r.baseline_rewards is None
          else r.baseline_rewards)[hero_pos] for r in played],
        dtype=np.float64)


def _label_q(plan, played, seconds):
    """`(q, legal, stats)` from the hands `plan.specs` turned into."""
    q, legal_idx = plan.q, plan.legal_idx
    forwards = plan.forwards
    forwards += sum(len(r.decisions) - plan.prefix_len - 1 for r in played)

    if played:
        rewards = hero_values(played, plan.hero_pos)
        q[legal_idx] = (rewards.reshape(plan.n_samples, len(legal_idx)).mean(axis=0)
                        / plan.big_blind)

    stats = LabelStats(
        forwards=forwards,
        seconds=seconds,
        collision_rate=(plan.rejected / plan.attempts) if plan.attempts else 0.0,
        n_rollouts=len(played))
    return q, plan.legal, stats


def action_values(record, decision_idx, driver, pool, hero_member_idx, cfg, rng,
                  posterior_cache=None):
    """Q for every legal action at one decision. See `_label_plan` for the args."""
    plan = _label_plan(record, decision_idx, driver, pool, hero_member_idx,
                       cfg, rng, posterior_cache=posterior_cache)
    started = time.perf_counter()
    played = driver.run(plan.specs, batch_size=cfg.batch_hands)
    return _label_q(plan, played, plan.prepared + time.perf_counter() - started)


def action_values_batch(requests, driver, pool, cfg, posterior_cache=None):
    """Label several decisions through **one** `driver.run` (§7.1).

    `requests` is a list of `(record, decision_idx, hero_member_idx, rng)`, and
    the result is one `(q, legal, stats)` per request, in the same order.
    `posterior_cache`, when supplied, carries exact prefix ranges across batch
    boundaries; otherwise a cache local to this batch is used.

    **This changes what a label costs, not what it is.** Every rollout hand
    keeps its own deck and its own `_rollout_seed`, and the driver samples each
    hand from a generator seeded by that alone (`env/driver.py`), so which other
    hands shared the batch cannot move an action. What it does move is the
    *width* of the policy batches: a single label's rollouts drain away as its
    hands finish and the lock-step group shrinks with them, while `k` labels
    queued together keep the driver refilled to `batch_hands` throughout. The
    network sees fewer, wider batches for exactly the same work.

    The one thing that is not bit-identical is floating point: a wider batch
    reduces in a different order, which on GPU can move a logit in its last bits
    and, through the inverse-CDF draw, an occasional action. That is the same
    class of difference as changing `batch_hands`, it is unbiased — the rollout
    policy, the sampled ranges and the seeds are untouched — and it is why
    `labels_per_batch` is a config knob and not a silent default.

    `seconds` is the label's own preparation plus its share of the shared run,
    split by how many hands it contributed; the total over a batch is the batch's
    wall clock.
    """
    # A local cache benefits labels of the same hand inside this batch.  The
    # generation pipeline passes a longer-lived instance as well, so a hand
    # split across label chunks still reuses its previous prefixes.
    if posterior_cache is None:
        posterior_cache = PosteriorCache()
    plans = [_label_plan(record, decision_idx, driver, pool, hero_member_idx,
                         cfg, rng, posterior_cache=posterior_cache)
             for record, decision_idx, hero_member_idx, rng in requests]
    specs = [spec for plan in plans for spec in plan.specs]

    started = time.perf_counter()
    played = driver.run(specs, batch_size=cfg.batch_hands)
    elapsed = time.perf_counter() - started

    out, offset = [], 0
    for plan in plans:
        n = len(plan.specs)
        share = elapsed * (n / len(specs)) if specs else 0.0
        out.append(_label_q(plan, played[offset:offset + n],
                            plan.prepared + share))
        offset += n
    assert offset == len(specs), (
        f"{offset} of {len(specs)} rollout hands were claimed by a label")
    return out
