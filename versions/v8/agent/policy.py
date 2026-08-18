"""The agent wearing the pool-member interface (CONCEPT.md §6.1, §7.1, §4).

Everything that needs an action from a player asks a `PoolMember`: the lock-step
driver during self-play and inside the oracle's rollouts, and — through the
adapter of §12 — the Slumbot evaluation. Wrapping the agent in that same
interface is what keeps "hero is the agent" from becoming a special case in
three different call sites, each free to build the observation slightly
differently. There is one observation builder (`nets.features.hand_tokens`) and
one last mile from logits to a played distribution (`PoolMember.policy`), and
the agent goes through both.

**One member is one seat at one table.** `slot_of_seat` and `observer_pos` are
fixed at construction because the observation depends on both: the slot decides
which fitted vector each seat is conditioned on, and the seat decides whose hole
cards the tokens may show. A session that rotates the button rotates the agent's
seat too, and that is a new member — cheap, since the network and the vectors
are held by reference.

The three assertions in `logits` are all the same invariant, which is §9's
observation parity, and they exist because breaking it is silent: the loss keeps
falling while the network reads cards or actions it will not have at deployment.
"""

import numpy as np
import torch

from nets.features import collate, hand_tokens
from pool.base import PoolMember


class AgentPoolMember(PoolMember):
    """Entity 3 as a pool member. One batched `AgentNet` forward per call."""

    def __init__(self, net, embeddings, slot_of_seat, max_players, n_actions,
                 observer_pos, device):
        super().__init__("agent", n_actions)
        self.net = net
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32,
                                          device=device)
        assert self.embeddings.shape == (max_players, net.d_emb), (
            f"embeddings must be ({max_players}, {net.d_emb}) — one vector per "
            f"slot, hero's own among them (§5.3); got "
            f"{tuple(self.embeddings.shape)}")
        self.slot_of_seat = list(slot_of_seat)
        self.max_players = max_players
        self.observer_pos = int(observer_pos)
        self.device = device

    def logits(self, contexts):
        hands = []
        for ctx in contexts:
            assert int(ctx.acting_pos) == self.observer_pos, (
                f"this member observes from seat {self.observer_pos} but was "
                f"asked to act at seat {ctx.acting_pos}; build one member per "
                f"seat rather than reusing one across the table (§9)")
            assert ctx.hole_override is None, (
                "the agent has no answer for a hypothetical holding: its "
                "observation comes from the record's own cards, so an override "
                "would be silently ignored and the answer would be about the "
                "real hand (§7.2)")
            assert ctx.snap_idx == len(ctx.record.snapshots) - 1, (
                "the pending decision is not this record's latest snapshot — "
                "the tokens would carry decisions taken after the one being "
                "asked about, which is the future leak §9 forbids")
            hands.append(hand_tokens(
                ctx.record, observer_pos=int(ctx.acting_pos),
                slot_of_seat=self.slot_of_seat, max_players=self.max_players,
                n_actions=self.n_actions, pending=ctx))

        batch = collate(hands, device=self.device)
        emb = self.embeddings[batch["slot"]]
        with torch.no_grad():
            out = self.net(batch, emb)
        return out.float().cpu().numpy().astype(np.float64)
