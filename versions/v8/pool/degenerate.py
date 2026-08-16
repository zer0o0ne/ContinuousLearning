"""Degenerate pool members (CONCEPT.md §4.1).

always-fold, always-call, always-min-raise, maniac, nit. No network, no cost.
Their job is to pin the corners of the style space that iterated agents never
visit on their own, so the embedding space has something to be continuous
*between* (§11.3).

They emit logits, not one-hots, on a finite scale (`LOGIT_SCALE`). That is
deliberate: a style draw (§4.2) has to remain able to move them, and an infinite
logit would make temperature and the uniform mix inert.

The nit's hand-strength test is a **hole-card lookup**, not an equity
evaluation — §4.2 drops equity conditions as far too expensive inside rollouts,
and names a preflop rank-class bucket as the cheap surrogate. It therefore reads
the same two cards on every street; that is a crude nit, and an intentionally
crude one.
"""

import numpy as np

from pool.base import PoolMember

LOGIT_SCALE = 4.0


def _empty(contexts, n_actions):
    return np.zeros((len(contexts), n_actions), dtype=np.float64)


class AlwaysFold(PoolMember):
    """Folds whenever folding is playable; checks when it is free."""

    def logits(self, contexts):
        out = _empty(contexts, self.n_actions)
        out[:, 0] = LOGIT_SCALE
        return out


class AlwaysCall(PoolMember):
    """Calls or checks, always."""

    def logits(self, contexts):
        out = _empty(contexts, self.n_actions)
        out[:, 1] = LOGIT_SCALE
        return out


class AlwaysMinRaise(PoolMember):
    """Takes the smallest playable sized raise; calls when none is playable."""

    def logits(self, contexts):
        out = _empty(contexts, self.n_actions)
        for i, ctx in enumerate(contexts):
            sized = [a for a in np.flatnonzero(ctx.legal_mask)
                     if 2 <= a < self.n_actions - 1]
            out[i, sized[0] if sized else 1] = LOGIT_SCALE
        return out


class Maniac(PoolMember):
    """All-in biased: shove > other raises > call > fold.

    Call outranks fold so that a maniac reduced to call-or-fold — every other
    live player already all-in, where `env.legal` drops all raises — still plays
    like a maniac instead of folding half the time.
    """

    def logits(self, contexts):
        out = _empty(contexts, self.n_actions)
        out[:, 1] = LOGIT_SCALE / 4.0
        out[:, 2:] = LOGIT_SCALE / 2.0
        out[:, self.n_actions - 1] = LOGIT_SCALE
        return out


class Nit(PoolMember):
    """Folds without a strong hand; calls with one.

    Strong = a pocket pair of tens or better, or two cards jack or better.
    Ranks are `card // 4`, so 0 = deuce and 12 = ace.
    """

    def logits(self, contexts):
        out = _empty(contexts, self.n_actions)
        for i, ctx in enumerate(contexts):
            r = sorted(c // 4 for c in ctx.hole_cards)
            strong = (r[0] == r[1] and r[1] >= 8) or r[0] >= 9
            out[i, 1 if strong else 0] = LOGIT_SCALE
        return out


DEGENERATE_STRATEGIES = {
    "always_fold": AlwaysFold,
    "always_call": AlwaysCall,
    "always_min_raise": AlwaysMinRaise,
    "maniac": Maniac,
    "nit": Nit,
}
