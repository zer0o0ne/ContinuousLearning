"""Live style modifiers on a pool member's output (CONCEPT.md §4.2, OI-5).

    p = (1 − λ) · softmax( (logits + b(s)) / T )  +  λ · uniform_over_legal

`b(s) ∈ R^{n_actions}` is broadcast from per-category style scalars to the
actions of each category. The categorisation is v7's — `resolve_actions` from
`vendor/v7/modifiers.py`, reused verbatim as §4.2 requires — giving the five
blocks fold / call / small raise / big raise / all-in.

**32 scalars per member**, matching §4.2's table:

===================  =====  =========================================
block                size   gated on
===================  =====  =========================================
unconditional            5   —
position                 5   acting-position bucket
street                4×5   street
temperature, λ           2   —
===================  =====  =========================================

*Reading of the position block.* §4.2 gives it size 5 while gating it on an
"acting position bucket", which only adds up if the bucket is binary: the
5-vector applies in the late bucket and is zero in the early one. The bucket is
``acting_pos >= ceil(num_players / 2)`` — seat 0 is the small blind and the
highest seat is the button, so this is "second half of the table", and it is
defined for every table size from 2 to 9. Recorded here because it is an
interpretation, not something §4.2 states.

Everything is free: no extra forward, no extra parameters, no extra network —
the bias is added before the softmax the member already computes. One random
draw is one style, so the opponent space is continuous and effectively infinite
(§4.1, §11.3).

Deliberately **not** here: v7's `equity <` / `equity >` conditions. They need an
equity evaluation at every decision inside the innermost rollout loop, which
§4.2 rules out as the single most expensive item in the oracle.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np

from vendor.v7.modifiers import resolve_actions

N_CATEGORIES = 5
N_STREETS = 4
N_STYLE_SCALARS = N_CATEGORIES * (2 + N_STREETS) + 2  # 32


def action_categories(n_actions):
    """The five action categories, as index lists — v7's partition.

    `resolve_actions("big_raises", ·)` includes all-in, but the five blocks have
    to be disjoint for a per-category bias to be well defined, so all-in is
    taken out of the big-raise block. That reproduces exactly the `cat_members`
    partition of v7's `build_style_vector`.
    """
    allin = resolve_actions("allin", n_actions)
    big = [a for a in resolve_actions("big_raises", n_actions) if a not in allin]
    return [
        resolve_actions("fold", n_actions),
        resolve_actions("call", n_actions),
        resolve_actions("small_raises", n_actions),
        big,
        allin,
    ]


def category_matrix(n_actions):
    """(5, n_actions) 0/1 broadcast matrix from category scalars to actions."""
    m = np.zeros((N_CATEGORIES, n_actions), dtype=np.float64)
    for c, members in enumerate(action_categories(n_actions)):
        for a in members:
            m[c, a] = 1.0
    return m


def position_bucket(acting_pos, num_players):
    """0 = early half of the table, 1 = late half. See module docstring."""
    return 1 if acting_pos >= ceil(num_players / 2) else 0


@dataclass
class StyleParams:
    """One member's frozen style draw — 32 scalars."""

    uncond: np.ndarray        # (5,)
    position: np.ndarray      # (5,) — applied only in the late bucket
    street: np.ndarray        # (4, 5)
    temperature: float
    uniform_mix: float

    @staticmethod
    def identity():
        return StyleParams(
            uncond=np.zeros(N_CATEGORIES),
            position=np.zeros(N_CATEGORIES),
            street=np.zeros((N_STREETS, N_CATEGORIES)),
            temperature=1.0,
            uniform_mix=0.0,
        )

    @property
    def is_identity(self):
        return (not self.uncond.any() and not self.position.any()
                and not self.street.any()
                and self.temperature == 1.0 and self.uniform_mix == 0.0)

    def to_list(self):
        """Flat 32-vector — the form a config carries as an explicit style."""
        return (list(self.uncond) + list(self.position)
                + list(self.street.reshape(-1))
                + [self.temperature, self.uniform_mix])

    @staticmethod
    def from_list(values):
        v = np.asarray(values, dtype=np.float64)
        assert v.shape == (N_STYLE_SCALARS,), (
            f"a style vector is {N_STYLE_SCALARS} scalars, got {v.shape}")
        return StyleParams(
            uncond=v[:5].copy(),
            position=v[5:10].copy(),
            street=v[10:30].reshape(N_STREETS, N_CATEGORIES).copy(),
            temperature=float(v[30]),
            uniform_mix=float(v[31]),
        )

    def category_bias(self, contexts):
        """(B, 5) per-category bias for each pending decision."""
        b = np.tile(self.uncond, (len(contexts), 1))
        for i, ctx in enumerate(contexts):
            if position_bucket(ctx.acting_pos, ctx.num_players):
                b[i] += self.position
            b[i] += self.street[ctx.turn]
        return b

    def apply(self, logits, legal, contexts):
        """(B, n_actions) played distribution. `legal` is a boolean mask."""
        n_actions = logits.shape[1]
        bias = self.category_bias(contexts) @ category_matrix(n_actions)

        z = (logits + bias) / self.temperature
        z = np.where(legal, z, -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)

        if self.uniform_mix > 0.0:
            counts = legal.sum(axis=1, keepdims=True)
            uniform = legal.astype(np.float64) / np.maximum(counts, 1)
            p = (1.0 - self.uniform_mix) * p + self.uniform_mix * uniform
        return p


def sample_style(rng, cfg):
    """Draw one style from the `style` config section (§8.1).

    Config keys — all optional, all with the scale/range semantics named in
    §8.1 ("per-block scale for the bias vector, temperature and λ ranges"):

    ``uncond_scale``, ``position_scale``, ``street_scale``
        standard deviations of the zero-mean Gaussian each bias block is drawn
        from, in logit units;
    ``log_temperature_range``
        ``T = exp(U(lo, hi))`` — symmetric in log space, so sharpening and
        flattening are equally likely;
    ``uniform_mix_range``
        ``λ ~ U(lo, hi)``.
    """
    lo_t, hi_t = cfg.get("log_temperature_range", [0.0, 0.0])
    lo_l, hi_l = cfg.get("uniform_mix_range", [0.0, 0.0])
    return StyleParams(
        uncond=rng.normal(0.0, cfg.get("uncond_scale", 0.0), N_CATEGORIES),
        position=rng.normal(0.0, cfg.get("position_scale", 0.0), N_CATEGORIES),
        street=rng.normal(0.0, cfg.get("street_scale", 0.0),
                          (N_STREETS, N_CATEGORIES)),
        temperature=float(np.exp(rng.uniform(lo_t, hi_t))),
        uniform_mix=float(rng.uniform(lo_l, hi_l)),
    )
