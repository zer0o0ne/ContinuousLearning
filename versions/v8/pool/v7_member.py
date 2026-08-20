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
from utils import get_amp_config
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

    def __init__(self, name, n_actions, agent, style=None, action_map=None):
        super().__init__(name, n_actions, style)
        self.agent = agent
        self.action_map = action_map
        if action_map is None:
            assert agent.n_actions == n_actions, (
                f"v7 checkpoint has {agent.n_actions} actions, pool uses "
                f"{n_actions}. Either the action layouts match — see "
                f"CONCEPT.md §6.1 — or the member is given a `RaiseGridMap` "
                f"between them.")
        else:
            assert (action_map.n_src, action_map.n_dst) == (agent.n_actions,
                                                            n_actions), (
                f"the grid map transports {action_map.n_src} → "
                f"{action_map.n_dst} actions, but the v7 checkpoint has "
                f"{agent.n_actions} and the pool uses {n_actions}")
        # G3 measured the label's wall clock: 85% of it is this forward and
        # under 6% is everything Python does to prepare it, so precision is the
        # only lever left on the pool's side. `get_amp_config` is v8's existing
        # answer to "what does this device want" and picks bf16 on CUDA, which
        # is the regime v7 itself was trained and evaluated under
        # (`evaluation/slumbot_eval.py`); on CPU it disables autocast, so the
        # dev box keeps running the fp32 path the test battery pins.
        # Resolved here because `build_pool` calls `set_device` before it
        # constructs a member, and never moves one afterwards.
        (self.amp_enabled, self.amp_device_type,
         self.amp_dtype, _scaler) = get_amp_config(agent.device_)

    def _snapshots(self, record, cache):
        """`record.snapshots` with every action one-hot in the v7 layout.

        Without a map this is the record's own list and nothing is copied. With
        one, each action-bearing snapshot is shallow-copied with its one-hot
        rewritten for that snapshot's street; `bets` and the rest are shared,
        and the record itself is never touched.

        `cache` memoises per record for the duration of one `logits` call, for
        the same reason `build_pool` memoises checkpoints: the oracle asks one
        member about many hole-card combos of the *same* record (§7.2), and the
        translation depends on the record alone. The record is kept in the
        cache alongside its translation so its `id` cannot be recycled while
        the entry is live.
        """
        if self.action_map is None:
            return record.snapshots
        hit = cache.get(id(record))
        if hit is not None:
            return hit[1]
        out = []
        for snap in record.snapshots:
            action = snap["action"]
            if action is None:
                out.append(snap)
            else:
                out.append(dict(snap, action=self.action_map.src_onehot(
                    snap["turn"], int(np.argmax(action)))))
        cache[id(record)] = (record, out)
        return out

    def logits(self, contexts):
        cache = {}
        event_sequences = []
        for ctx in contexts:
            rec = ctx.record
            event_sequences.append(build_v7_events(
                self._snapshots(rec, cache), _deck_seen_by(ctx), ctx.acting_pos,
                rec.num_players, rec.spec.big_blind, rec.spec.small_blind,
                self.agent.n_actions, up_to=ctx.snap_idx,
            ))
        with torch.no_grad(), torch.autocast(
                device_type=self.amp_device_type, dtype=self.amp_dtype,
                enabled=self.amp_enabled):
            out = self.agent.action_logits(event_sequences)
        out = out.float().cpu().numpy().astype(np.float64)
        if self.action_map is None:
            return out
        return self.action_map.dst_logprobs(
            out, [int(ctx.turn) for ctx in contexts])
