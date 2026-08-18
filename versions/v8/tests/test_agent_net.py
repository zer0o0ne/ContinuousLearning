"""The agent network and the agent as a pool member (CONCEPT.md §6.1, §9).

The trunk extraction (plan D3) is asserted by the *rest* of the battery: the
computation moved into `nets/trunk.py` unchanged, so `test_embedding_net_masking.py`
and `test_inference_fit.py` had to keep passing with no edits, and they do.

What is new here is the agent, and the properties worth pinning are the ones
that fail silently:

* the logits are read at each hand's **own** last real token, so a padded batch
  does not answer about a token that does not exist;
* the observation the agent acts on obeys §9 parity — the same invariant
  `test_observation_parity.py` pins for the embedding network, now through the
  agent's own call path;
* with `e = 0` the network cannot tell which player is which, which is the
  property embedding dropout (§6.2) exists to create and which should hold
  structurally before any training happens.
"""

import dataclasses

import numpy as np
import pytest
import torch

from agent.policy import AgentPoolMember
from env.driver import HandSpec, LockstepDriver
from nets.agent_net import AgentNet
from nets.features import (
    TOKEN_DECISION, UNKNOWN_CARD, collate, hand_tokens,
)
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG, RAISE_SIZES, SMALL_BLIND,
    make_pool, make_specs, play,
)


def _net(seed=0):
    torch.manual_seed(seed)
    return AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()


def _zero_embeddings():
    return np.zeros((MAX_PLAYERS, NET_CFG["d_emb"]), dtype=np.float32)


class _RecordingAgent(AgentPoolMember):
    """The agent, keeping what it observed and what it played.

    A pool member is asked for actions from inside the driver, so the only way
    to see the observation it acted on is to keep it. The tokens are the object
    under test in the parity case, and re-tokenising in the test would be
    testing a copy of the code rather than the call path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []
        self.played = []

    def logits(self, contexts):
        out = super().logits(contexts)
        for ctx in contexts:
            self.seen.append((ctx.record, len(ctx.record.decisions),
                              hand_tokens(
                ctx.record, observer_pos=int(ctx.acting_pos),
                slot_of_seat=self.slot_of_seat,
                max_players=self.max_players, n_actions=self.n_actions,
                pending=ctx)))
        return out

    def policy(self, contexts):
        probs = super().policy(contexts)
        for row, ctx in zip(probs, contexts):
            self.played.append((np.asarray(row, dtype=np.float64),
                                np.asarray(ctx.legal_mask, dtype=bool)))
        return probs


def _agent_at_seat_zero(num_players, stack_bb, n_hands, seed):
    """Play `n_hands` with the agent at seat 0 and degenerates elsewhere."""
    opponents = make_pool(seed=seed)
    pool = [_RecordingAgent(_net(seed), _zero_embeddings(),
                            slot_of_seat=list(range(MAX_PLAYERS)),
                            max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                            observer_pos=0, device="cpu")] + opponents
    specs = []
    for h in range(n_hands):
        specs.append(HandSpec(
            num_players=num_players,
            start_credits=[float(stack_bb * BIG_BLIND)] * num_players,
            seat_members=[0] + [1 + (s + h) % len(opponents)
                                for s in range(num_players - 1)],
            seed=310_000 + seed * 100 + h,
            big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
            raise_sizes=RAISE_SIZES, meta={"hand": h},
        ))
    records = LockstepDriver(pool, N_ACTIONS).run(specs)
    return pool[0], records


# ------------------------------------------------ 2: shape and the last token


def _ragged_batch(seed=0):
    records = play(make_pool(), make_specs(seed=seed, n_hands=12, n_members=9))
    hands = [hand_tokens(r, 0, list(range(MAX_PLAYERS)), MAX_PLAYERS, N_ACTIONS)
             for r in records if r.num_players >= 2]
    hands = [h for h in hands if len(h) >= 3]
    assert len(hands) >= 4
    lengths = [len(h) for h in hands]
    assert len(set(lengths)) > 1, "the batch has to be ragged to test anything"
    return collate(hands), lengths


def test_the_logits_have_one_row_per_hand():
    net = _net()
    batch, lengths = _ragged_batch()
    with torch.no_grad():
        out = net(batch, torch.zeros(batch["mask"].shape + (net.d_emb,)))
    assert out.shape == (len(lengths), N_ACTIONS)
    assert torch.isfinite(out).all()


def test_padding_cannot_reach_the_logits():
    """Rewrite every padded position; the answer must not move at all."""
    net = _net()
    batch, lengths = _ragged_batch()
    emb = torch.zeros(batch["mask"].shape + (net.d_emb,))
    with torch.no_grad():
        base = net(batch, emb)

    pad = batch["mask"] == 0
    assert bool(pad.any()), "no padded position in this batch"
    poisoned = {k: v.clone() for k, v in batch.items()}
    poisoned["cards"][pad] = 0
    poisoned["scalars"][pad] = 7.5
    poisoned["seat_stacks"][pad] = 3.0
    poisoned["prev_action"][pad] = 1.0
    poisoned["acting_pos"][pad] = MAX_PLAYERS - 1
    with torch.no_grad():
        after = net(poisoned, emb)
    assert torch.equal(base, after), (
        "a padded position moved the logits — the read is at a fixed index, "
        "not at each row's own last real token")


def test_each_row_answers_from_its_own_last_real_token():
    """Attention is causal, so only a read *at* the last token sees a change
    made there. Every row is perturbed at its own length, and every row moves."""
    net = _net()
    batch, lengths = _ragged_batch()
    emb = torch.zeros(batch["mask"].shape + (net.d_emb,))
    with torch.no_grad():
        base = net(batch, emb)

    for row, t in enumerate(lengths):
        perturbed = {k: v.clone() for k, v in batch.items()}
        perturbed["scalars"][row, t - 1] += 5.0
        with torch.no_grad():
            after = net(perturbed, emb)
        assert not torch.equal(base[row], after[row]), (
            f"row {row} ignored a change at its last real token {t - 1}")
        others = [r for r in range(len(lengths)) if r != row]
        assert torch.equal(base[others], after[others]), (
            "a change in one hand moved another hand's logits")


# -------------------------------------------------------- 3: parity, via §9


def test_the_agent_only_ever_observes_what_its_seat_could_know():
    member, _records = _agent_at_seat_zero(num_players=4, stack_bb=100,
                                           n_hands=8, seed=1)
    assert member.seen, "the agent never acted"
    checked_own = checked_masked = 0
    for record, n_dec, tok in member.seen:
        own = record.hole_cards(0)
        # One token per decision taken *by then*, plus the pending one. The
        # record keeps growing after the agent answered; the observation does
        # not, which is the whole point.
        assert len(tok) == n_dec + 1
        for t in range(len(tok)):
            assert tok.token_type[t] == TOKEN_DECISION, (
                "a hand with a pending decision has not reached showdown, so "
                "it carries no showdown token")
            slots = tok.cards[t, 5:].tolist()
            if int(tok.acting_pos[t]) == 0:
                assert slots == own
                checked_own += 1
            else:
                assert slots == [UNKNOWN_CARD, UNKNOWN_CARD], (
                    f"seat {int(tok.acting_pos[t])}'s cards leaked into the "
                    f"agent's observation")
                checked_masked += 1

        # The board is never ahead of the street its token was taken on —
        # including the pending token, whose street is the one being decided.
        snap_of = [int(d["snap_idx"]) for d in record.decisions[:n_dec]] + \
            [int(record.decisions[n_dec]["snap_idx"]) if n_dec < len(
                record.decisions) else len(record.snapshots) - 1]
        for t, snap_idx in enumerate(snap_of):
            turn = int(record.snapshots[snap_idx]["turn"])
            revealed = int((tok.cards[t, :5] != UNKNOWN_CARD).sum())
            assert revealed == {0: 0, 1: 3, 2: 4, 3: 5}[turn]
            assert tok.cards[t, :revealed].tolist() == \
                [int(c) for c in record.deck[:revealed]]

        # The pending token is the question, not the answer.
        assert tok.action[-1] == -1
    assert checked_own > 0 and checked_masked > 0


def test_a_pending_token_never_carries_the_action_it_is_asking_about():
    member, _records = _agent_at_seat_zero(num_players=3, stack_bb=100,
                                           n_hands=6, seed=2)
    for _record, _n_dec, tok in member.seen:
        assert tok.action[-1] == -1
        assert bool(tok.legal[-1].any()), "the pending token carries no mask"
        if len(tok) == 1:
            assert not tok.prev_action[-1].any()


# ------------------------------- 4: a valid distribution at every table shape


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("stack_bb", [10, 300])
def test_the_agent_plays_a_valid_distribution_over_legal_actions(num_players,
                                                                 stack_bb):
    member, records = _agent_at_seat_zero(num_players, stack_bb, n_hands=4,
                                          seed=3 + num_players)
    assert member.played, "the agent never acted"
    for probs, legal in member.played:
        assert probs.shape == (N_ACTIONS,)
        assert np.isfinite(probs).all()
        assert probs[~legal].tolist() == [0.0] * int((~legal).sum())
        assert probs.sum() == pytest.approx(1.0)
        assert (probs >= 0.0).all()
    for record in records:
        assert abs(float(record.rewards.sum())) < 1e-9
        for dec in record.decisions:
            assert dec["legal_mask"][dec["action_idx"]]


# --------------------------------------------------------- 5: e = 0 is blind


def _relabelled(record, perm):
    """The same hand with the seats' pool members permuted."""
    spec = dataclasses.replace(
        record.spec,
        seat_members=[record.spec.seat_members[perm[s]]
                      for s in range(record.num_players)])
    return dataclasses.replace(record, spec=spec)


def test_with_a_zero_embedding_the_network_cannot_tell_the_players_apart():
    net = _net()
    records = play(make_pool(), make_specs(seed=5, n_hands=6, n_members=9,
                                           num_players=4))
    record = max(records, key=lambda r: len(r.decisions))
    perm = [1, 2, 3, 0]

    def logits(rec, slots, vectors):
        tok = hand_tokens(rec, 0, slots, MAX_PLAYERS, N_ACTIONS)
        batch = collate([tok])
        with torch.no_grad():
            return net(batch, vectors[batch["slot"]]), tok

    zeros = torch.zeros(MAX_PLAYERS, net.d_emb)
    straight = list(range(MAX_PLAYERS))
    permuted = [perm[s] if s < len(perm) else s for s in range(MAX_PLAYERS)]

    base, tok_a = logits(record, straight, zeros)
    swapped, tok_b = logits(_relabelled(record, perm), permuted, zeros)
    assert not np.array_equal(tok_a.member, tok_b.member) or \
        not np.array_equal(tok_a.slot, tok_b.slot), \
        "the relabelling changed nothing; the test is vacuous"
    assert torch.equal(base, swapped), (
        "with e = 0 the network still told the players apart — something "
        "other than the embedding carries player identity")

    torch.manual_seed(1)
    vectors = torch.randn(MAX_PLAYERS, net.d_emb)
    with_vec, _ = logits(record, straight, vectors)
    moved, _ = logits(_relabelled(record, perm), permuted, vectors)
    assert not torch.equal(with_vec, moved), (
        "a non-zero embedding did not change the answer — the vector is not "
        "reaching the network at all")


# ------------------------------------------------------------ 6: determinism


def test_two_identical_calls_give_identical_logits():
    member, _records = _agent_at_seat_zero(num_players=4, stack_bb=100,
                                           n_hands=3, seed=7)
    again, _records2 = _agent_at_seat_zero(num_players=4, stack_bb=100,
                                           n_hands=3, seed=7)
    assert len(member.played) == len(again.played)
    for (a, _l1), (b, _l2) in zip(member.played, again.played):
        assert np.array_equal(a, b)


def test_asking_the_agent_for_an_action_does_not_move_its_weights():
    net = _net()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    opponents = make_pool()
    pool = [AgentPoolMember(net, _zero_embeddings(),
                            slot_of_seat=list(range(MAX_PLAYERS)),
                            max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                            observer_pos=0, device="cpu")] + opponents
    specs = make_specs(seed=9, n_hands=4, n_members=len(pool), num_players=3)
    for spec in specs:
        spec.seat_members = [0] + [1, 2][:spec.num_players - 1]
    LockstepDriver(pool, N_ACTIONS).run(specs)
    after = net.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"parameter {k} moved"


# ---------------------------------------------- the parity guards themselves


def test_the_agent_refuses_a_question_about_another_seat():
    from tests.g1_fixtures import contexts_from
    records = play(make_pool(), make_specs(seed=11, n_hands=2, n_members=9,
                                           num_players=3))
    member = AgentPoolMember(_net(), _zero_embeddings(),
                             slot_of_seat=list(range(MAX_PLAYERS)),
                             max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                             observer_pos=0, device="cpu")
    others = [c for c in contexts_from(records) if int(c.acting_pos) != 0]
    assert others
    with pytest.raises(AssertionError, match="observes from seat"):
        member.logits(others[:1])


def test_the_agent_refuses_a_decision_that_is_not_the_latest_snapshot():
    """A finished record carries decisions taken after the one being asked
    about; answering from it would read the future."""
    from tests.g1_fixtures import contexts_from
    records = play(make_pool(), make_specs(seed=12, n_hands=4, n_members=9,
                                           num_players=3))
    member = AgentPoolMember(_net(), _zero_embeddings(),
                             slot_of_seat=list(range(MAX_PLAYERS)),
                             max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                             observer_pos=0, device="cpu")
    stale = [c for c in contexts_from(records)
             if int(c.acting_pos) == 0
             and c.snap_idx != len(c.record.snapshots) - 1]
    assert stale
    with pytest.raises(AssertionError, match="latest snapshot"):
        member.logits(stale[:1])
