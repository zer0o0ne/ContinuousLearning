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

**Two members, because there are two positions.** `AgentPoolMember` is the agent
as *hero* — one seat, one table, conditioned on the vectors §5.5 fitted for that
table. `FrozenAgentMember` is a past agent as an *opponent*, which §8 creates at
the end of every iteration: any seat, any table, `e = 0` (D12), and answering the
`hole_override` question the §7.2 posterior asks of every member it conditions
on. They share the network class and the observation builder and nothing else,
because the two positions have genuinely different contracts.

A past agent plays at `e = 0` by default and may instead be handed the table its
own §5.5 fit produced (`PLAN_AMORTISED_POOL.md`); which of the two happens is a
decision of the phase that seats it, and this file only says what a member with
a table does with it.
"""

import copy
from dataclasses import replace

import numpy as np
import torch

from nets.features import collate, hand_tokens
from pool.base import PoolMember
from pool.v7_member import _deck_seen_by


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
        # §5.7 — the range head is asked about seats the acting player is not,
        # so it reads every seat's vector, not only hero's own.
        seat_emb = self.embeddings[batch["seat_slot"]]
        with torch.no_grad():
            out = self.net(batch, emb, seat_emb)
        return out.float().cpu().numpy().astype(np.float64)


class FrozenAgentMember(PoolMember):
    """A trained agent, frozen, seated as an *opponent* (§8's last line, D12).

    §8 appends the agent to the pool at the end of every iteration, and from
    then on it is asked the two questions every pool member is asked: *what do
    you do here* (the driver, in play and inside the oracle's rollouts) and
    *what would you have done holding this* (`hole_override`, the §7.2
    posterior). `AgentPoolMember` answers neither in that position, and refuses
    both loudly: it is built for one seat of one table and only ever asked about
    the moment it is acting, because that is all hero is ever asked.

    **The vectors are zero by default** (D12 option (a), the plan's
    recommendation, taken 2026-08-19 when S9 was built). A past agent then plays
    its *unconditional* policy — the one §6.2's embedding dropout trains
    explicitly — so it is a fixed policy like every other member of the pool: no
    fit nested inside a fit, no recursion, and no answer needed to "what did
    agent *k−3* believe about its tablemates".

    Two consequences follow from `e = 0` and both are what make this a member
    rather than a second hero:

    * `slot_of_seat` cannot change an answer — every slot reads the same zero
      vector — so **one member serves every seat of every table**, which is what
      `PoolMember` requires and `AgentPoolMember` cannot give.
    * the style layer applies to it exactly as to a v7 checkpoint (§4.2), so the
      `style.agent_variants` members an agent contributes are `with_style`
      siblings sharing one network by reference (D11).

    **Or it is handed a table** — `(embeddings, own_slot)`, the `K = 0` output
    of the amortised head over the hands this member's *own* slot observed
    (`PLAN_AMORTISED_POOL.md`, option (c)). It then conditions on its tablemates
    exactly as hero does, and neither consequence above is lost:

    * one member still serves every seat, because a member that knows its own
      slot can *derive* the rotation from the seat it is asked to act at: slot
      `own_slot` sits at seat `acting_pos` in hand `h` iff
      `h ≡ own_slot − acting_pos (mod n)`, and that fixes every seat's slot
      (`_slot_of_seat`). One member per (session, slot, block), not per seat.
    * `with_vectors` is `with_style`'s sibling pattern with the table swapped
      instead of the style, so the network is still shared by reference.

    Nothing here decides *which* table is right for a block; that is the block
    discipline of the phase that seats the member (`train/generate.py`). What
    the member does decide is **whose network** may compute that table:
    `embed_net` is its own generation's, frozen, and a member with none is never
    conditioned at all.

    **The observation is rebuilt for the moment being asked about.** A member
    answering a posterior query is handed a *finished* record and a decision
    from the middle of it, so the tokens must stop there: `hand_tokens`
    tokenises the whole record, which for that query would carry every action
    taken after the one being asked about and the showdown as well. The view
    below truncates the record to the asked-about snapshot and swaps in the
    hypothetical holding, which is exactly what `pool/v7_member.py` does for the
    same two reasons — hence the shared `_deck_seen_by`. On the hot path (the
    driver, acting now, real cards) no copy is made at all.
    """

    def __init__(self, name, n_actions, net, max_players, device, style=None,
                 embeddings=None, own_slot=None, embed_net=None):
        super().__init__(name, n_actions, style)
        self.net = net
        # The embedding network of *this member's own generation*, frozen: the
        # one whose vectors its policy was trained to read. `None` means nobody
        # may condition it. Never the loop's current network — that one is still
        # training, its coordinates drift, and a frozen policy reading them
        # would be reading a description in a basis it has never seen. It would
        # also stop this member from being a fixed algorithm, which is what the
        # pool is for: the same history has to produce the same action at every
        # iteration, or hero's accumulated results per member are results
        # against a moving target.
        self.embed_net = embed_net
        self.max_players = int(max_players)
        self.device = device
        # Identity, and arbitrary: with `e = 0` the slot only selects which zero
        # vector is read. With a table the rotation is derived per context
        # instead, because it depends on the seat being asked about.
        self.slot_of_seat = list(range(self.max_players))
        self._seat_at(embeddings, own_slot)

    def _seat_at(self, embeddings, own_slot):
        """Adopt `(max_players, d_emb)` vectors read from slot `own_slot`.

        `None, None` is the `e = 0` member. The two travel together: a table
        with no slot to read it from cannot be indexed, and a slot with no table
        would be a rotation nothing uses.
        """
        assert (embeddings is None) == (own_slot is None), (
            "embeddings and own_slot are one thing: a table is read from the "
            "slot the member occupies, so neither is meaningful alone (§5.5)")
        if embeddings is None:
            self.embeddings = None
            self.own_slot = None
            return
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32,
                                          device=self.device)
        assert self.embeddings.shape == (self.max_players, self.net.d_emb), (
            f"embeddings must be ({self.max_players}, {self.net.d_emb}) — one "
            f"vector per slot of the table this member sits at, its own among "
            f"them (§5.3); got {tuple(self.embeddings.shape)}")
        self.own_slot = int(own_slot)
        assert 0 <= self.own_slot < self.max_players, (
            f"slot {own_slot} is not a slot of a {self.max_players}-slot table")

    def with_vectors(self, name, embeddings, own_slot):
        """A sibling of this member conditioned on one table, at one slot.

        `with_style`'s pattern (`pool/base.py`): the network and the style are
        shared by reference and only the table changes, so a refresh costs a
        `copy.copy` and not a second network.
        """
        sibling = copy.copy(self)
        sibling.name = name
        sibling._seat_at(embeddings, own_slot)
        return sibling

    def _slot_of_seat(self, ctx):
        """Seat → slot for the hand `ctx` is a decision of.

        Identity at `e = 0`, where it cannot change an answer. With a table it
        is the rotation that puts `own_slot` at the acting seat — the same
        derivation `Session.slot_of_seat` makes from the hand index, reached
        from the seat instead, which is all a member is handed.
        """
        if self.embeddings is None:
            return self.slot_of_seat
        n = int(ctx.record.num_players)
        hand_idx = (self.own_slot - int(ctx.acting_pos)) % n
        return [(seat + hand_idx) % n for seat in range(n)]

    def _observation(self, ctx):
        record = ctx.record
        at_latest = int(ctx.snap_idx) == len(record.snapshots) - 1
        if ctx.hole_override is not None or not at_latest or record.showdown:
            record = replace(
                record,
                deck=_deck_seen_by(ctx),
                snapshots=record.snapshots[:int(ctx.snap_idx) + 1],
                decisions=[d for d in record.decisions
                           if int(d["snap_idx"]) < int(ctx.snap_idx)],
                showdown=[], showdown_strength={}, showdown_class={})
        return hand_tokens(
            record, observer_pos=int(ctx.acting_pos),
            slot_of_seat=self._slot_of_seat(ctx), max_players=self.max_players,
            n_actions=self.n_actions, pending=ctx)

    def logits(self, contexts):
        batch = collate([self._observation(ctx) for ctx in contexts],
                        device=self.device)
        if self.embeddings is None:
            emb = torch.zeros(*batch["mask"].shape, self.net.d_emb,
                              device=self.device)
            seat_emb = torch.zeros(*batch["seat_slot"].shape, self.net.d_emb,
                                   device=self.device)
        else:
            # The same two gathers hero makes (§5.3, §5.7): the acting seat's
            # own vector for the decision token, every seat's for the range
            # head.
            emb = self.embeddings[batch["slot"]]
            seat_emb = self.embeddings[batch["seat_slot"]]
        with torch.no_grad():
            out = self.net(batch, emb, seat_emb)
        return out.float().cpu().numpy().astype(np.float64)
