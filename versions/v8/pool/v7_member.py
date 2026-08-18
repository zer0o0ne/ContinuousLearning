"""A vendored v7 checkpoint as a pool member (CONCEPT.md §4.1, §4.3).

The member is: vendored perception + action head, the v7 event builder,
``skip_opponent_emb=True``, ``heads={"action"}``. Nothing else of v7 runs — no
MCTS, no opponent embedding, no value or modelling head.

Events are built from the **acting player's** seat (`hero_pos = acting_pos`),
which is both what v7 expects and what observation parity requires: a member
deciding at seat p sees p's own hole cards and nothing else's.
"""

import numpy as np
import torch

from pool.base import PoolMember
from vendor.v7.events import build_v7_events


def _deck_seen_by(ctx):
    """The hand's deck with the acting seat's hole cards as the context says.

    `build_v7_events` reads the acting player's cards out of the deck, so a
    member asked "what would you have done holding *this*" (`hole_override`,
    §7.2) has to be handed a deck that says *this*. Without the swap the member
    answers about the real hand, every combo gets the same likelihood, and the
    posterior of §7.2 silently stays at the prior for every network member in
    the pool.

    The record is never touched: the swap lands in a copy. Only the board and
    the acting seat's two cards are read downstream, so a combo that collides
    with some other seat's real cards is harmless here.
    """
    if ctx.hole_override is None:
        return ctx.record.deck
    deck = np.array(ctx.record.deck, copy=True)
    lo = 5 + 2 * int(ctx.acting_pos)
    deck[lo:lo + 2] = [int(c) for c in ctx.hole_override]
    return deck


class V7NetworkMember(PoolMember):
    """Wraps a loaded `vendor.v7.agent.V7Agent`.

    Several members can share one `V7Agent` — style draws differ, weights do
    not, and that is exactly the "more draws, not more checkpoints" argument of
    §4.2. `agent` is therefore held by reference and never mutated here.
    """

    def __init__(self, name, n_actions, agent, style=None):
        super().__init__(name, n_actions, style)
        self.agent = agent
        assert agent.n_actions == n_actions, (
            f"v7 checkpoint has {agent.n_actions} actions, pool uses "
            f"{n_actions}. The action layouts must match — see CONCEPT.md §6.1.")

    def logits(self, contexts):
        event_sequences = []
        for ctx in contexts:
            rec = ctx.record
            event_sequences.append(build_v7_events(
                rec.snapshots, _deck_seen_by(ctx), ctx.acting_pos,
                rec.num_players, rec.spec.big_blind, rec.spec.small_blind,
                self.n_actions, up_to=ctx.snap_idx,
            ))
        with torch.no_grad():
            out = self.agent.action_logits(event_sequences)
        return out.float().cpu().numpy().astype(np.float64)
