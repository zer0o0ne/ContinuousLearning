"""Sampling opponents out of the pool (CONCEPT.md §4.4, `PLAN_PIPELINE.md` S8).

Uniform sampling over a growing pool spends almost all of its compute on
opponents the agent already crushes, and it is also the configuration in which
fictitious play is known to be weakest (§11.1). Three mechanisms answer that,
all of them config-driven, and every draw of a seat goes through all three:

* **PFSP** (AlphaStar) — member *i* is drawn with probability ∝ ``f(hardness of
  i)``, ``f(x) = x ** pfsp_exponent``. Hard opponents dominate the batch; the
  member hero beats by the most has a weight of exactly zero and is reached only
  through the uniform floor.
* **Embedding-space dedup** (§11.3) — the pool is clustered by its own opponent
  vectors and a *cluster* is drawn before a member inside it. 500 near-duplicate
  checkpoints of one lineage then share one cluster's worth of probability
  instead of taking 500 full shares. The cluster's weight is the **mean** of its
  members' PFSP weights, not the sum: a sum would restore exactly the
  proportional-to-size behaviour the clustering exists to remove.
* **Uniform floor** — a fixed fraction of draws ignores both and is uniform over
  the whole pool, "so old styles are never fully evicted" (§4.4). This is also
  the only way a member with a zero PFSP weight is ever seen again, which is why
  it is not optional.

Nothing is ever deleted; §4.4's answer to "how do we filter the pool
retroactively" is that things are down-weighted.

**From BB/100 to a number PFSP can use.** §4.4 transplants AlphaStar's formula,
which is written in terms of a *loss rate*, and that quantity does not exist
here: StarCraft results are binary, poker results are money. Something has to
bridge them, and the bridge is the one place in this file where a bad choice is
expensive rather than merely inelegant.

The bridge is: accumulate hero's **total** BB and total hands against each
member, take the mean BB/100, and min-max it over the pool —

    hardness_i = (m_max − m_i) / (m_max − m_min),   m = mean BB/100

so the member hero loses the most to has hardness 1 and the one hero beats the
most has 0. Members with no results yet have hardness 1 ("sampled immediately").
No new config knob: the scale comes out of the pool itself, which is also the
only scale that stays meaningful as the agent gets stronger.

The obvious cheaper reading — count sessions whose BB/100 was negative — was
tried and rejected, and the reason is worth keeping because it is not obvious.
A session's *sign* is very nearly a coin flip: at σ ≈ 6 BB/hand a 200-hand
session has SE ≈ 42 BB/100, so a member hero genuinely beats by 10 BB/100 still
produces a losing session 41 % of the time and one that beats hero by 10 produces
one 59 % of the time. Every rate bunches around 0.5, PFSP flattens towards
uniform, and §4.4 stops doing anything at all. Thresholding at zero once per
session throws away information that averaging would have kept, and it throws it
away irreversibly.

Min-max is sensitive to one extreme member stretching the denominator and
compressing everybody else; if that ever shows up, the replacement is a rank
instead of a min-max, which costs no config either.

**The accumulation forgets, because the quantity it estimates is not stationary.**
A procedural pool member never changes, but hero does — every iteration replaces
it — so "hero's BB/100 against member *i*" is a property of a *pair*, and a
lifetime mean over thirty iterations is a mean over thirty different heroes. The
consequence is the one that matters operationally: a member the early agents
crushed keeps a positive mean built on tens of thousands of hands, and the ~330
hands per iteration the floor delivers would take on the order of thirty
iterations to drag it back across zero — the length of a whole run. The loop
would not notice that it had stopped beating it. The same smearing also destroys
the `PLAN_PIPELINE.md` R2 diagnostic, which reads fictitious-play cycling off
exactly these scores.

`end_iteration` therefore scales both accumulators by `result_decay` at every
iteration boundary, giving an effective window of roughly
`hands_per_iteration / (1 - result_decay)` hands. Two properties make this the
right shape rather than a patch:

* the mean of a member nobody sampled this iteration is **unchanged** —
  numerator and denominator decay together. No new information, no new estimate.
* what does shrink is its *confidence*, the effective hand count, which is
  precisely what lets the next session it does play flip the estimate quickly.

`result_decay = 1.0` is the lifetime accumulation, kept reachable because it is
the honest way to turn the mechanism off.

Optional `evaluation_weight` blends in a separate, decayed estimate from the
current agent's training/warm evaluation. Each iteration contributes one mean
per opponent, over independent sessions, regardless of physical hand count.
Evaluation never adds pseudo-hands to the collection accumulators. Missing
evaluation coverage gradually loses influence as its accumulated mass decays.

**The floor is an independent coin flip per draw.** A deterministic schedule
("every fifth draw") gives the same mean with less variance, and was the first
implementation, but it aligns with table structure: at a fixed table size and a
floor of 0.25 the floor lands on *the same two slots of every table*, forever,
and the slot index is carried on the token. Uniform table sizes (§4.4) make the
phase drift so it would not bite today, but it is a trap laid for the first
fixed-size diagnostic anybody runs.

Seats within one table are drawn **independently**, so the same member may sit
in two seats. That is deliberate: two seats playing the same style is a table
configuration like any other, and rejecting it would quietly bias the draw away
from small clusters — the opposite of what the dedup is for.
"""

import numpy as np


def _kmeans_labels(vectors, n_clusters, n_iter=50):
    """Cluster `vectors` into at most `n_clusters` groups, deterministically.

    Plain Lloyd's algorithm with a farthest-point (Gonzalez) initialisation.
    Deterministic is the requirement, not optimal: the initialisation consumes
    no randomness, ties go to the lowest index through `argmin`/`argmax`, and an
    empty cluster keeps its previous centre rather than being re-seeded. A
    sampler whose clustering moved between two runs of the same seed would make
    every downstream artefact irreproducible.
    """
    v = np.asarray(vectors, dtype=np.float64)
    n = v.shape[0]
    if n_clusters >= n:
        return np.arange(n, dtype=np.int64)
    if n_clusters <= 1:
        return np.zeros(n, dtype=np.int64)

    # Farthest-point init: start from the point furthest from the pool's mean,
    # then repeatedly add the point furthest from everything chosen so far.
    d0 = ((v - v.mean(axis=0)) ** 2).sum(axis=1)
    chosen = [int(np.argmax(d0))]
    dist = ((v - v[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < n_clusters:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, ((v - v[nxt]) ** 2).sum(axis=1))
    centres = v[chosen].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d = ((v[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1).astype(np.int64)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(n_clusters):
            members = v[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    return labels


class PoolSampler:
    """PFSP + embedding-space dedup + a uniform floor (§4.4).

    Args:
        n_members: size of the pool this sampler draws from. §8 grows the pool by
            one member per iteration; the growth path is to build a sampler with
            the larger `n_members` and `load_state_dict` the smaller one's state
            into it (see `load_state_dict`).
        cfg: the `pool_sampling` config section (§8.1) — `pfsp_exponent`,
            `floor_fraction`, `n_clusters`, `result_decay`, `evaluation_weight`.
        rng: `np.random.Generator`. Owned by the sampler: its state is part of
            `state_dict`, so a restart resumes the same stream.
    """

    def __init__(self, n_members, cfg, rng):
        assert int(n_members) > 0, "an empty pool has nothing to sample"
        self.n_members = int(n_members)
        self.pfsp_exponent = float(cfg["pfsp_exponent"])
        self.floor_fraction = float(cfg["floor_fraction"])
        self.n_clusters = int(cfg["n_clusters"])
        self.result_decay = float(cfg["result_decay"])
        self.evaluation_weight = float(cfg.get("evaluation_weight", 0.0))
        if not 0 <= self.evaluation_weight <= 1:
            raise ValueError("pool_sampling.evaluation_weight must be in [0, 1]")
        assert 0.0 <= self.floor_fraction <= 1.0, (
            f"floor_fraction is a fraction of draws, got {self.floor_fraction}")
        assert 0.0 < self.result_decay <= 1.0, (
            f"result_decay scales the accumulated results once per iteration, "
            f"got {self.result_decay}")
        self.rng = rng
        # Float rather than integer hands: `end_iteration` scales them, so the
        # count is an *effective* number of hands, not a tally.
        self._hands = np.zeros(self.n_members, dtype=np.float64)
        self._bb = np.zeros(self.n_members, dtype=np.float64)
        self._evaluation_sum = np.zeros(self.n_members, dtype=np.float64)
        self._evaluation_mass = np.zeros(self.n_members, dtype=np.float64)
        self._last_evaluation_iteration = -1
        # Until `set_vectors` is called there is no embedding space to cluster
        # in, so every member is its own cluster and the dedup step is a no-op.
        self._cluster_of = np.arange(self.n_members, dtype=np.int64)

    # ----------------------------------------------------------------- results

    def update(self, member_idx, hero_bb, n_hands):
        """Record hero's result against `member_idx` (§4.4 PFSP input).

        `hero_bb` is hero's net result in big blinds over `n_hands` hands played
        against that member, **not** a rate: the two are accumulated separately
        so that sessions of different lengths carry their proper weight and the
        mean converges instead of being re-thresholded per session. Negative is
        a loss for hero, hence weight for that member.
        """
        i = int(member_idx)
        assert 0 <= i < self.n_members, (
            f"member {i} is outside a pool of {self.n_members}")
        assert float(n_hands) > 0, "a result over zero hands says nothing"
        self._hands[i] += float(n_hands)
        self._bb[i] += float(hero_bb)

    def end_iteration(self):
        """Age the accumulated results by `result_decay` (§8's iteration boundary).

        Called once per iteration of the outer loop, by the loop — the sampler
        has no notion of an iteration of its own. See the module docstring for
        why forgetting is required rather than optional.
        """
        self._hands *= self.result_decay
        self._bb *= self.result_decay
        self._evaluation_sum *= self.result_decay
        self._evaluation_mass *= self.result_decay

    def update_evaluation(self, iteration, by_member):
        """Add this generation's session means once, after end_iteration().

        `units` counts independent sessions, not hands or cyclic seatings. It
        establishes that an estimate exists; it does not scale that generation's
        contribution. More evaluation improves the estimate's precision without
        increasing its mixture weight. Empty observations still mark the
        generation consumed, so a rerun cannot later insert it twice.
        """
        if not self.evaluation_weight or iteration <= self._last_evaluation_iteration:
            return False
        values = []
        for member, result in by_member.items():
            i, units, mean = int(member), result["units"], result["bb_per_100"]
            if (not 0 <= i < self.n_members or int(units) != units or units <= 0
                    or mean is None or not np.isfinite(mean)):
                raise ValueError("Evaluation feedback needs a finite mean over valid sessions per member")
            values.append((i, float(mean)))
        for i, mean in values:
            self._evaluation_sum[i] += mean
            self._evaluation_mass[i] += 1.0
        self._last_evaluation_iteration = int(iteration)
        return True

    def collection_mean_bb_per_100(self):
        """Hero's BB/100 from label collection only; NaN when unplayed."""
        out = np.full(self.n_members, np.nan, dtype=np.float64)
        played = self._hands > 0
        out[played] = 100.0 * self._bb[played] / self._hands[played]
        return out

    def evaluation_mean_bb_per_100(self):
        out = np.full(self.n_members, np.nan, dtype=np.float64)
        np.divide(self._evaluation_sum, self._evaluation_mass, out=out,
                  where=self._evaluation_mass > 0)
        return out

    def evaluation_mix(self):
        """Effective evaluation share, fading if this opponent is not revisited."""
        share = self.evaluation_weight * np.minimum(1.0, self._evaluation_mass)
        # With no collection evidence, evaluation is the only known result.
        share[(self._hands == 0) & (share > 0)] = 1.0
        return share

    def mean_bb_per_100(self):
        """The collection/evaluation mixture PFSP ranks; NaN when unplayed."""
        out = self.collection_mean_bb_per_100()
        share = self.evaluation_mix()
        seen = share > 0
        prior = np.where(np.isfinite(out), out, 0.0)
        out[seen] = ((1 - share[seen]) * prior[seen]
                     + share[seen] * self.evaluation_mean_bb_per_100()[seen])
        return out

    def hardness(self):
        """§4.4's "loss rate" slot, in [0, 1]; see the module docstring.

        1.0 is the member hero loses the most to, 0.0 the one hero beats the
        most, and a member with no results yet is 1.0 — "members never played get
        the maximum weight, so a newly appended agent is sampled immediately".
        A pool in which every played member has the same mean carries no signal
        to discriminate on, and is flat at 1.0 rather than at some arbitrary
        point inside the range.
        """
        out = np.ones(self.n_members, dtype=np.float64)
        means = self.mean_bb_per_100()
        played = np.isfinite(means)
        if not played.any():
            return out
        lo, hi = float(means[played].min()), float(means[played].max())
        if hi > lo:
            out[played] = (hi - means[played]) / (hi - lo)
        return out

    def weights(self):
        """PFSP weight per member — `f(x) = x ** pfsp_exponent`.

        At least one member always has weight 1 (the hardest, or every member of
        a pool with no results), so the weights never sum to zero and `_draw_pfsp`
        never has to fall back to uniform at the cluster level.
        """
        return self.hardness() ** self.pfsp_exponent

    # ---------------------------------------------------------------- clusters

    def set_vectors(self, vectors):
        """Re-cluster the pool from its trained embedding table (§5.4, §11.3).

        `vectors` is `(n_members, d_emb)` — the embedding network's own table,
        which is what makes this deduplication *in the space the agent actually
        conditions on* rather than in some proxy of it.
        """
        v = np.asarray(vectors, dtype=np.float64)
        assert v.ndim == 2 and v.shape[0] == self.n_members, (
            f"expected ({self.n_members}, d_emb) vectors, got {v.shape}")
        self._cluster_of = _kmeans_labels(v, self.n_clusters)

    # ---------------------------------------------------------------- sampling

    def probabilities(self):
        """Exact marginal seat probabilities, including dedup and uniform floor.

        P(cluster) * P(member | cluster) reduces to w_i / cluster_size_i,
        normalised across all members. Reading it does not advance the RNG.
        """
        counts = np.bincount(self._cluster_of)
        weighted = self.weights() / counts[self._cluster_of]
        return (self.floor_fraction / self.n_members
                + (1 - self.floor_fraction) * weighted / weighted.sum())

    def distribution(self):
        """JSON-safe snapshot of the mixture used to collect training labels."""
        def finite(values):
            return [float(x) if np.isfinite(x) else None for x in values]

        return {"probabilities": self.probabilities().tolist(),
                "pfsp_weights": self.weights().tolist(),
                "cluster_of": self._cluster_of.tolist(),
                "hero_bb_per_100": finite(self.mean_bb_per_100()),
                "collection_bb_per_100": finite(self.collection_mean_bb_per_100()),
                "evaluation_bb_per_100": finite(self.evaluation_mean_bb_per_100()),
                "evaluation_mix": self.evaluation_mix().tolist()}

    def _draw_pfsp(self):
        """One member: a cluster by mean weight, then a member inside it."""
        w = self.weights()
        labels = self._cluster_of
        n_c = int(labels.max()) + 1
        totals = np.bincount(labels, weights=w, minlength=n_c)
        counts = np.bincount(labels, minlength=n_c).astype(np.float64)
        means = np.zeros(n_c, dtype=np.float64)
        np.divide(totals, counts, out=means, where=counts > 0)
        c = int(self.rng.choice(n_c, p=means / means.sum()))
        # A cluster is drawn by its mean weight, so a cluster of members hero
        # beats the most — mean zero — is never drawn, and the weights inside
        # the cluster that *was* drawn therefore never sum to zero either.
        idx = np.flatnonzero(labels == c)
        wc = w[idx]
        return int(self.rng.choice(idx, p=wc / wc.sum()))

    def sample_table(self, n_opponents):
        """`n_opponents` member indices, one per non-hero seat (§4.4).

        Hero is not drawn from the pool — it occupies slot 0 of every session
        (`env/session.py`) — so this is asked for `num_players - 1` seats.
        Seats are drawn independently; see the module docstring.
        """
        k = int(n_opponents)
        assert k >= 0, f"a table cannot have {k} opponents"
        out = []
        for _ in range(k):
            if self.rng.random() < self.floor_fraction:
                out.append(int(self.rng.integers(self.n_members)))
            else:
                out.append(self._draw_pfsp())
        return out

    # ------------------------------------------------------------------- state

    def state_dict(self):
        """Everything a restart needs, in plain JSON-able types (§8, resume)."""
        return {
            "n_members": self.n_members,
            "hands": self._hands.tolist(),
            "bb": self._bb.tolist(),
            "cluster_of": self._cluster_of.tolist(),
            "rng_state": self.rng.bit_generator.state,
            "evaluation": {"sum_bb_per_100": self._evaluation_sum.tolist(),
                           "mass": self._evaluation_mass.tolist(),
                           "last_iteration": self._last_evaluation_iteration},
        }

    def load_state_dict(self, state):
        """Restore a saved sampler, possibly into a pool that has since grown.

        §8 appends the trained agent to the pool at the end of every iteration,
        so the sampler that resumes an experiment is one member larger than the
        one that saved it. A stored state may therefore cover a **prefix** of
        this sampler's members; the rest arrive unplayed (hardness 1, "sampled
        immediately") and each in a cluster of its own until the next
        `set_vectors`. A stored state larger than this pool is an error, not a
        truncation.
        """
        n_old = int(state["n_members"])
        assert n_old <= self.n_members, (
            f"saved sampler has {n_old} members, this pool has "
            f"{self.n_members} — the pool only ever grows (§8)")
        self._hands = np.zeros(self.n_members, dtype=np.float64)
        self._bb = np.zeros(self.n_members, dtype=np.float64)
        self._hands[:n_old] = np.asarray(state["hands"], dtype=np.float64)
        self._bb[:n_old] = np.asarray(state["bb"], dtype=np.float64)
        evaluation = state.get("evaluation", {})
        self._evaluation_sum = np.zeros(self.n_members, dtype=np.float64)
        self._evaluation_mass = np.zeros(self.n_members, dtype=np.float64)
        self._evaluation_sum[:n_old] = evaluation.get("sum_bb_per_100", [0.0] * n_old)
        self._evaluation_mass[:n_old] = evaluation.get("mass", [0.0] * n_old)
        self._last_evaluation_iteration = int(evaluation.get("last_iteration", -1))
        clusters = np.asarray(state["cluster_of"], dtype=np.int64)
        self._cluster_of = np.arange(self.n_members, dtype=np.int64)
        self._cluster_of[:n_old] = clusters
        if n_old < self.n_members:
            self._cluster_of[n_old:] = (
                int(clusters.max()) + 1 + np.arange(self.n_members - n_old))
        self.rng.bit_generator.state = state["rng_state"]
