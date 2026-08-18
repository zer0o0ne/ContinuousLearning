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

**Why the board is the real one.** Hero's action does not change the runout, and
the posterior was computed with the visible board dead. Re-dealing the board
would make `Q` an average over runouts the posterior has already conditioned
away — a different quantity, and the wrong one. The price is that a sampled
combo can collide with a board card that was not yet visible when the posterior
was computed; those draws are rejected below and counted, because that is the
size of an approximation and it belongs in the log rather than in a comment.

**Fold needs no special case.** Hero's chip delta after folding is minus what
hero has already put in, whatever the opponents hold, so the rollout returns the
closed form with zero variance. Special-casing it would be a second path to the
same number.
"""

import hashlib
import time
from dataclasses import dataclass, replace

import numpy as np

from oracle.posterior import opponent_posterior

FOLD = 0
N_CARDS = 52


@dataclass
class OracleConfig:
    """The knobs of one label. Every one of them trades cost against noise."""
    samples_per_action: int = 256
    max_combos: int = None
    likelihood_floor: float = 1e-6
    batch_hands: int = 2048
    max_collision_retries: int = 32


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


def _folded_before(record, decision_idx):
    """Seats that have given up their cards by `decision_idx`."""
    return {int(d["acting_pos"]) for d in record.decisions[:decision_idx]
            if int(d["action_idx"]) == FOLD}


def _rollout_deck(record, hero_pos, opp_seats, opp_cards):
    """The 52-card deck of one rollout.

    The real board, hero's real cards, the sampled cards at the opponents'
    seats, and every card nobody was dealt filled into the remaining slots in
    ascending order — arbitrary, but a function of the assignment alone, so two
    identical assignments deal identical hands.
    """
    deck = np.full(N_CARDS, -1, dtype=np.int64)
    deck[:5] = [int(c) for c in record.deck[:5]]
    deck[5 + 2 * hero_pos: 7 + 2 * hero_pos] = record.hole_cards(hero_pos)
    for seat, cards in zip(opp_seats, opp_cards):
        deck[5 + 2 * seat: 7 + 2 * seat] = cards

    used = np.zeros(N_CARDS, dtype=bool)
    used[deck[deck >= 0]] = True
    empty = np.flatnonzero(deck < 0)
    deck[empty] = np.flatnonzero(~used)
    return deck


def _sample_joint(posteriors, dead_mask, n_samples, max_retries, rng):
    """Draw each opponent's combo from its own marginal (§7.3).

    The exact joint over eight opponents is combinatorially impossible, so the
    marginals are treated as independent and the card-removal correction is
    made by **rejection**: a draw in which two opponents share a card, or in
    which a card is already on the board or in hero's hand, is thrown away and
    redrawn. A sample that has not survived `max_retries` redraws is dropped —
    it reduces the divisor of the average, it is not an outcome of zero.

    Returns `(cards, n_attempts, n_rejected)` with `cards` of shape
    `(kept, n_opponents, 2)`. Rejection is the approximation, so its rate is
    returned rather than hidden.
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


def action_values(record, decision_idx, driver, pool, hero_member_idx, cfg, rng):
    """Q for every legal action at `record.decisions[decision_idx]`, in BB.

    Returns (q, legal, stats): q is (n_actions,) float64 with `nan` at illegal
    actions, legal is the recorded mask, stats is a `LabelStats`.

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

    # A seat that has folded holds nothing that can change a showdown, so it is
    # not worth a posterior; its slots get filler cards.
    folded = _folded_before(record, decision_idx)
    opp_seats = [p for p in range(record.num_players)
                 if p != hero_pos and p not in folded]

    forwards = 0
    posteriors = []
    for seat in opp_seats:
        combos, weights = opponent_posterior(
            record, seat, hero_pos, pool, n_actions,
            through_decision=decision_idx - 1, floor=cfg.likelihood_floor,
            max_combos=cfg.max_combos, rng=rng)
        acted = sum(1 for d in record.decisions[:decision_idx]
                    if int(d["acting_pos"]) == seat)
        forwards += len(combos) * acted
        posteriors.append((combos, weights))

    # The full board, not the visible one: these cards are in the rollout deck.
    dead_mask = np.zeros(N_CARDS, dtype=bool)
    dead_mask[[int(c) for c in record.deck[:5]]] = True
    dead_mask[record.hole_cards(hero_pos)] = True

    samples, attempts, rejected = _sample_joint(
        posteriors, dead_mask, cfg.samples_per_action,
        cfg.max_collision_retries, rng)

    prefix = [int(d["action_idx"]) for d in record.decisions[:decision_idx]]
    seat_members = list(record.spec.seat_members)
    seat_members[hero_pos] = int(hero_member_idx)

    specs = []
    for s in range(len(samples)):
        deck = _rollout_deck(record, hero_pos, opp_seats, samples[s])
        for a in legal_idx:
            specs.append(replace(
                record.spec, seat_members=seat_members, deck=deck,
                forced_actions=prefix + [a],
                seed=_rollout_seed(record.spec.seed, decision_idx, a, s)))

    played = driver.run(specs, batch_size=cfg.batch_hands)
    forwards += sum(len(r.decisions) - len(prefix) - 1 for r in played)

    if played:
        rewards = np.asarray([r.rewards[hero_pos] for r in played],
                             dtype=np.float64)
        q[legal_idx] = (rewards.reshape(len(samples), len(legal_idx)).mean(axis=0)
                        / float(record.spec.big_blind))

    stats = LabelStats(
        forwards=forwards,
        seconds=time.perf_counter() - started,
        collision_rate=(rejected / attempts) if attempts else 0.0,
        n_rollouts=len(played))
    return q, legal, stats
