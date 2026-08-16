"""The opponent-pool member interface (CONCEPT.md §4).

A pool member is anything that answers

    P(action | observation) → distribution over the discrete action set

for an arbitrary legal situation, in batch. Members expose **logits**; turning
logits into a played distribution — style modifier, temperature, uniform mix,
legality — happens once, here, so that every member (v7 network, degenerate
strategy, later a v8 agent) goes through exactly the same last mile.

Observation parity (CONCEPT.md §9) is a property of `DecisionContext`: it hands
out the *acting player's* hole cards and the board as of the current street, and
there is no accessor on it for anybody else's cards.
"""

import copy

import numpy as np

from pool.style import StyleParams


class PoolMember:
    """Base class. Subclasses implement `logits`."""

    def __init__(self, name, n_actions, style=None):
        self.name = name
        self.n_actions = n_actions
        self.style = style if style is not None else StyleParams.identity()

    def logits(self, contexts):
        """(B, n_actions) float array of unnormalised action scores."""
        raise NotImplementedError

    def policy(self, contexts):
        """(B, n_actions) distribution, zero on illegal actions."""
        logits = np.asarray(self.logits(contexts), dtype=np.float64)
        legal = np.stack([np.asarray(c.legal_mask, dtype=bool) for c in contexts])
        return self.style.apply(logits, legal, contexts)

    def with_style(self, name, style):
        """A sibling member sharing this one's base policy under a new style.

        "How do we expand the opponent space cheaply": not more checkpoints,
        more draws (§4.2). The base — a loaded v7 network, or nothing at all for
        a degenerate strategy — is shared by reference, never copied.
        """
        sibling = copy.copy(self)
        sibling.name = name
        sibling.style = style
        return sibling

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"
