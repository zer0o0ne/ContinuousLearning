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
"""

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
        with torch.no_grad():
            out = self.net(batch, emb)
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

    **The vectors are zero** (D12 option (a), the plan's recommendation, taken
    2026-08-19 when S9 was built). A past agent plays its *unconditional*
    policy — the one §6.2's embedding dropout trains explicitly — so it is a
    fixed policy like every other member of the pool: no fit nested inside a
    fit, no recursion, and no answer needed to "what did agent *k−3* believe
    about its tablemates". The alternative, giving every past agent its own §5.5
    fit over the table it is sitting at, has no cheap form and no measurement
    asking for it yet.

    Two consequences follow from `e = 0` and both are what make this a member
    rather than a second hero:

    * `slot_of_seat` cannot change an answer — every slot reads the same zero
      vector — so **one member serves every seat of every table**, which is what
      `PoolMember` requires and `AgentPoolMember` cannot give.
    * the style layer applies to it exactly as to a v7 checkpoint (§4.2), so the
      `style.agent_variants` members an agent contributes are `with_style`
      siblings sharing one network by reference (D11).

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

    def __init__(self, name, n_actions, net, max_players, device, style=None):
        super().__init__(name, n_actions, style)
        self.net = net
        self.max_players = int(max_players)
        self.device = device
        # Identity, and arbitrary: with `e = 0` the slot only selects which zero
        # vector is read.
        self.slot_of_seat = list(range(self.max_players))

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
            slot_of_seat=self.slot_of_seat, max_players=self.max_players,
            n_actions=self.n_actions, pending=ctx)

    def logits(self, contexts):
        batch = collate([self._observation(ctx) for ctx in contexts],
                        device=self.device)
        emb = torch.zeros(*batch["mask"].shape, self.net.d_emb,
                          device=self.device)
        with torch.no_grad():
            out = self.net(batch, emb)
        return out.float().cpu().numpy().astype(np.float64)
