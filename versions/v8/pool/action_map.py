"""Transport between two `[fold, call, raise_0 … raise_{bins-1}, all-in]` grids.

A vendored v7 checkpoint is welded to the raise grid it was trained on, from
both ends: `perception.action_proj` is a `Linear(n_actions, d_model)` reading a
one-hot of the *action just taken*, and `action_head` emits one logit per action
of that same layout. So a v8 run whose `game.raise_sizes` differs from the
checkpoint's cannot hand its own action indices to a v7 member, and cannot read
that member's logits as v8 actions — index `raise_k` means a different fraction
of the pot on each side.

This module is the translation, and it is the *only* place that knows the two
grids are different. Both directions are **nearest bin by raise fraction**,
per street (`env/legal.py` reads `raise_sizes[turn][i]` as a fraction of the
effective pot on every street, preflop included, so the two grids' numbers are
directly comparable):

``dst → src`` (`src_onehot`)
    rewrites the history a v7 member reads. The approximation is bounded: the
    chips that actually went in are carried exactly by `pot`, `bets` and
    `stacks` in the same event, and the one-hot is a redundant categorical
    channel beside them.

``src → dst`` (`dst_logprobs`)
    moves the member's played distribution onto the v8 action set. The whole
    mass of a v7 bin lands on the single nearest v8 bin, so the member keeps
    betting the size it meant to; v8 bins that are nobody's nearest receive no
    mass from v7 members at all (owner decision 2026-08-20). That is a
    deliberate hole and not a defect: the agent's targets are the oracle's Q
    over every legal v8 action (`oracle/rollout.py`), not an imitation of the
    pool, so a bin no v7 member plays is still trained.

`fold`, `call` and `all-in` are positional and map to themselves.
"""

import numpy as np

# A dst bin that no src bin maps to gets this probability rather than zero.
# `pool.style.StyleParams.apply` masks illegal actions with `-inf` and then
# subtracts the row max: a row in which *every* legal action carried `-inf`
# would come out `nan` instead of loudly empty. Floored, such a row still ends
# in the driver's "zero mass on every legal action" assertion. e^-60 is
# negligible under the whole temperature range of §8.1 (`T ∈ [0.5, 2]`).
EMPTY_BIN_PROB = 1e-26

N_STREETS = 4


def nearest_bin(fracs_from, fracs_to):
    """Index in `fracs_to` of the nearest fraction to each of `fracs_from`."""
    a = np.asarray(fracs_from, dtype=np.float64)[:, None]
    b = np.asarray(fracs_to, dtype=np.float64)[None, :]
    return np.abs(a - b).argmin(axis=1)


class RaiseGridMap:
    """Nearest-bin transport from a `src` raise grid onto a `dst` one.

    `src_sizes` and `dst_sizes` are both in `env.session.raise_sizes_from`'s
    shape: four lists (one per street) of raise fractions.
    """

    def __init__(self, src_sizes, dst_sizes):
        self.src_bins = _bin_count(src_sizes, "src")
        self.dst_bins = _bin_count(dst_sizes, "dst")
        self.n_src = self.src_bins + 3
        self.n_dst = self.dst_bins + 3
        self.is_identity = (
            self.src_bins == self.dst_bins
            and all(np.allclose(s, d) for s, d in zip(src_sizes, dst_sizes)))

        # dst action -> src action, and the src one-hot to hand to v7 for it.
        self.to_src = np.zeros((N_STREETS, self.n_dst), dtype=np.int64)
        self.onehots = []
        # dst raise bin -> the src raise bins whose nearest it is.
        self.groups = []

        for street in range(N_STREETS):
            src, dst = src_sizes[street], dst_sizes[street]
            self.to_src[street, 0] = 0
            self.to_src[street, 1] = 1
            self.to_src[street, self.n_dst - 1] = self.n_src - 1
            self.to_src[street, 2:self.n_dst - 1] = nearest_bin(dst, src) + 2

            onehot = np.zeros((self.n_dst, self.n_src), dtype=np.float32)
            onehot[np.arange(self.n_dst), self.to_src[street]] = 1.0
            self.onehots.append([row.tolist() for row in onehot])

            forward = nearest_bin(src, dst)
            groups = [[] for _ in range(self.n_dst)]
            groups[0] = [0]
            groups[1] = [1]
            groups[self.n_dst - 1] = [self.n_src - 1]
            for i, j in enumerate(forward):
                groups[int(j) + 2].append(i + 2)
            self.groups.append([np.asarray(g, dtype=np.int64) for g in groups])

    def src_onehot(self, street, dst_idx):
        """The `n_src` one-hot a v7 event carries for a v8 action index."""
        return self.onehots[int(street)][int(dst_idx)]

    def dst_logprobs(self, logits, streets):
        """(B, n_dst) log-probabilities from (B, n_src) src logits.

        `streets` is one street index per row: the transport is per street, and
        a batch of decisions spans several of them.

        Normalising here and returning log-probabilities rather than raw logits
        changes nothing downstream — `StyleParams.apply` softmaxes what it is
        given, and `softmax(log softmax(z) / T) == softmax(z / T)`.
        """
        logits = np.asarray(logits, dtype=np.float64)
        assert logits.shape[1] == self.n_src, (
            f"src logits are {logits.shape[1]}-wide, this map transports "
            f"{self.n_src}")
        streets = np.asarray(streets, dtype=np.int64)
        assert streets.shape == (logits.shape[0],), (
            f"one street per row: {streets.shape} streets for "
            f"{logits.shape[0]} rows")

        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)

        out = np.zeros((logits.shape[0], self.n_dst), dtype=np.float64)
        for street in np.unique(streets):
            rows = streets == street
            for j, src in enumerate(self.groups[int(street)]):
                if len(src):
                    out[rows, j] = p[np.ix_(rows, src)].sum(axis=1)
        return np.log(np.maximum(out, EMPTY_BIN_PROB))


def _bin_count(sizes, which):
    assert len(sizes) == N_STREETS, (
        f"{which} grid has {len(sizes)} streets, expected {N_STREETS}")
    counts = {len(s) for s in sizes}
    assert len(counts) == 1, (
        f"{which} grid has a different number of raise bins per street "
        f"({[len(s) for s in sizes]}); the action layout is one size for the "
        f"whole hand (`env/table.py` reads `len(raise_sizes[0])`)")
    return counts.pop()
