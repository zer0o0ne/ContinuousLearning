"""A past agent that conditions on its tablemates (PLAN_AMORTISED_POOL.md, P1).

`FrozenAgentMember` is the agent seated as an *opponent*, and until now it
played at `e = 0` (D12 option (a)): every slot read the same zero vector, so the
seat it acted from could not change its answer. This file is about the other
option — the member is handed the `K = 0` table its own slot's view produced —
and about the two things that go wrong silently if it is handed the wrong one:

* **the rotation.** A member is told which *slot* it occupies and is asked to
  act at a *seat*; the two are related by the hand index, which nobody hands it.
  Deriving it wrong reads a tablemate's vector for another tablemate, and the
  policy is then conditioned on a stranger — with no error anywhere.
* **the observer.** The table is fitted from the hands *this* member saw, and a
  tokenisation from hero's seat would show it hero's hole cards. That is §9's
  observation parity broken from the pool's side, and it is exactly as silent.

Everything below runs through played hands and through the member's public
answers, not through a re-implementation of the tokeniser (`CLAUDE.md` §4).
"""

import numpy as np
import pytest
import torch

from agent.policy import FrozenAgentMember
from env.driver import DecisionContext, LockstepDriver
from env.session import Session
from env.showdown import label_showdowns
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet, fit_embeddings
from nets.features import TOKEN_DECISION, UNKNOWN_CARD, collate, hand_tokens
from pool.style import sample_style
from train.generate import _pad_vectors, amortised_vectors
from tests.g1_fixtures import (MAX_PLAYERS, N_ACTIONS, NET_CFG, STYLE_CFG,
                               contexts_from, make_pool, session_specs)

NUM_PLAYERS = 4
N_HANDS = 8
D_EMB = NET_CFG["d_emb"]


def _session(num_players=NUM_PLAYERS, n_hands=N_HANDS, seed=0):
    """One played, showdown-labelled session with a rotating button."""
    pool = make_pool(seed=seed)
    members = list(range(num_players))
    specs = session_specs(members, num_players, n_hands, seed=seed)
    records = LockstepDriver(pool, N_ACTIONS).run(specs)
    label_showdowns(records)
    return pool, Session(idx=0, num_players=num_players, stack_bb=100,
                         members=members, specs=specs, records=list(records))


def _agent_net(seed=0):
    torch.manual_seed(seed)
    return AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()


def _embed_net(n_members, seed=0):
    torch.manual_seed(seed)
    return OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS,
                                n_members=n_members).eval()


def _member(net, embeddings=None, own_slot=None, style=None, name="agent0"):
    return FrozenAgentMember(name, N_ACTIONS, net, MAX_PLAYERS, "cpu",
                             style=style, embeddings=embeddings,
                             own_slot=own_slot)


def _table(seed=0):
    """A distinct non-zero vector per slot."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(MAX_PLAYERS, D_EMB)).astype(np.float32)


def _acting_contexts(session, seat):
    """Every decision of the session taken at `seat`, as a context."""
    out = []
    for record in session.records:
        for dec in record.decisions:
            if int(dec["acting_pos"]) != seat:
                continue
            snap = record.snapshots[dec["snap_idx"]]
            out.append(DecisionContext(record, dec["snap_idx"],
                                       dec["acting_pos"], dec["legal_mask"],
                                       int(snap["turn"])))
    return out


# ------------------------------------------------ the member with no table


def test_a_member_with_no_table_is_the_member_that_was_there_before():
    """The default is `e = 0` and it is bit-identical to a zero table.

    Both halves matter: the `None` path is what every existing caller gets, and
    the zero table is the cold start of block 0 (§5.5), which must not be a
    different policy from the one D12 settled on.
    """
    _pool, session = _session()
    net = _agent_net()
    contexts = contexts_from(session.records)

    plain = _member(net)
    assert plain.embeddings is None and plain.own_slot is None
    cold = _member(net, np.zeros((MAX_PLAYERS, D_EMB), dtype=np.float32), 2)

    assert np.array_equal(plain.logits(contexts), cold.logits(contexts)), (
        "a zero table changed the answer, so something other than the "
        "embedding carries player identity")


def test_a_table_without_a_slot_to_read_it_from_is_refused():
    net = _agent_net()
    with pytest.raises(AssertionError, match="one thing"):
        _member(net, _table(), None)
    with pytest.raises(AssertionError, match="one thing"):
        _member(net, None, 1)
    with pytest.raises(AssertionError, match="one vector per slot"):
        _member(net, _table()[:3], 1)
    with pytest.raises(AssertionError, match="not a slot"):
        _member(net, _table(), MAX_PLAYERS)


# --------------------------------------------------------- the sibling


def test_with_vectors_swaps_the_table_and_shares_everything_else():
    """`with_style`'s pattern: the network is shared, the base is untouched."""
    _pool, session = _session()
    net = _agent_net()
    style = sample_style(np.random.default_rng(3), STYLE_CFG)
    base = _member(net, style=style)
    contexts = contexts_from(session.records)
    before = base.logits(contexts)

    sibling = base.with_vectors("agent0@s1", _table(1), 1)
    assert sibling.net is base.net and sibling.style is base.style
    assert sibling.name == "agent0@s1" and base.name == "agent0"
    assert base.embeddings is None and base.own_slot is None, (
        "the sibling wrote its table back into the member it was copied from")
    assert np.array_equal(base.logits(contexts), before)
    assert not np.array_equal(sibling.logits(contexts), before), (
        "the table changed nothing — it never reached the forward")

    again = sibling.with_vectors("agent0@s2", _table(2), 2)
    assert again.own_slot == 2 and sibling.own_slot == 1
    assert not np.array_equal(again.logits(contexts),
                              sibling.logits(contexts))


# ------------------------------------------------------------ the rotation


def test_the_rotation_a_member_derives_is_the_sessions_own():
    """Slot `j` at seat `s` fixes the hand index, and with it every seat's slot.

    Over every table size 2–9 and every hand of the session: the member is told
    its slot and the seat it is acting at, and the slots it labels the table
    with must be the ones the session's rotation gives that hand.
    """
    net = _agent_net()
    checked = 0
    for num_players in range(2, 10):
        _pool, session = _session(num_players=num_players,
                                  n_hands=2 * num_players, seed=num_players)
        for slot in range(num_players):
            member = _member(net, _table(slot), slot)
            for h, record in enumerate(session.records):
                seat = session.seat_of_slot(slot, h)
                expected = session.slot_of_seat(h)
                for ctx in _acting_contexts(session, seat):
                    if ctx.record is not record:
                        continue
                    tokens = member._observation(ctx)
                    assert [expected[p] for p in tokens.acting_pos] == \
                        list(tokens.slot), (
                        f"a {num_players}-handed table, slot {slot} at seat "
                        f"{seat} in hand {h}: the member labelled the seats "
                        f"with a rotation the session does not have")
                    checked += 1
    assert checked > 100, f"only {checked} decisions exercised the rotation"


def test_the_same_tablemates_under_a_relabelling_are_the_same_table():
    """The answer depends on the table only through *whose* vector each seat
    reads — never through the slot numbers themselves.

    Two members occupying different slots, each handed the table its own
    labelling implies, sit at the same table with the same tablemates. They
    tokenise it differently (the slot ids differ) and must answer identically.
    """
    _pool, session = _session()
    net = _agent_net()
    n = session.num_players
    table = _table(7)

    for seat in range(n):
        contexts = _acting_contexts(session, seat)
        assert contexts, f"seat {seat} never acted; the test would be vacuous"
        answers, slots = [], []
        for slot in range(n):
            # Seat x reads slot (x + h) % n, and h = (slot − seat) % n, so a
            # member at slot k reads row (x + k − seat) % n for seat x: shifting
            # the table by (slot − k) keeps every seat's vector where it was.
            shifted = table.copy()
            shifted[:n] = np.roll(table[:n], shift=slot, axis=0)
            member = _member(net, shifted, slot)
            answers.append(member.logits(contexts))
            slots.append([list(member._observation(c).slot) for c in contexts])
        assert any(slots[k] != slots[0] for k in range(1, n)), (
            "every slot produced the same tokenisation; the test is vacuous")
        for k in range(1, n):
            assert np.allclose(answers[k], answers[0], atol=1e-6), (
                f"seat {seat} answered differently when the same tablemates "
                f"were labelled from slot {k} instead of slot 0")


def test_only_the_rows_of_the_slots_at_the_table_are_ever_read():
    """A 9-slot table seats 4 players; the other five rows say nothing."""
    _pool, session = _session()
    net = _agent_net()
    n = session.num_players
    table = _table(11)
    contexts = contexts_from(session.records)
    base = _member(net, table, 1).logits(contexts)

    unused = table.copy()
    unused[n:] = _table(12)[n:]
    assert np.array_equal(_member(net, unused, 1).logits(contexts), base), (
        "a row for a slot nobody occupies changed the answer")

    for slot in range(n):
        moved = table.copy()
        moved[slot] = moved[slot] + 1.0
        assert not np.array_equal(
            _member(net, moved, 1).logits(contexts), base), (
            f"moving slot {slot}'s vector changed nothing, so that tablemate "
            f"is not being conditioned on")


# ------------------------------- the two contracts a past agent already had


def test_a_conditioned_member_still_answers_the_posteriors_question():
    """§7.2 asks "holding *this*, what then?" — a table must not deafen it."""
    _pool, session = _session()
    net = _agent_net()
    member = _member(net, _table(4), 2)
    record = max(session.records, key=lambda r: len(r.decisions))
    dec = record.decisions[0]
    seat = int(dec["acting_pos"])

    def ctx_with(cards):
        return DecisionContext(record, dec["snap_idx"], seat,
                               dec["legal_mask"], 0, hole_override=cards)

    logits = member.logits([ctx_with([0, 1]), ctx_with([50, 51]),
                            ctx_with([0, 1])])
    assert np.allclose(logits[0], logits[2]), "the same holding, twice"
    assert not np.allclose(logits[0], logits[1]), (
        "two different holdings produced the same answer — the override was "
        "not read")


def test_a_conditioned_member_observes_the_moment_it_is_asked_about():
    """§9, for a conditioned member handed a decision from the middle."""
    _pool, session = _session()
    net = _agent_net()
    member = _member(net, _table(5), 3)
    record = next(r for r in session.records if r.showdown
                  and len(r.decisions) > 2)

    for t, dec in enumerate(record.decisions):
        seat = int(dec["acting_pos"])
        ctx = DecisionContext(record, dec["snap_idx"], seat, dec["legal_mask"],
                              0, hole_override=[0, 1])
        tokens = member._observation(ctx)

        assert (tokens.token_type == TOKEN_DECISION).all(), (
            "a hand still in progress has no showdown token")
        assert len(tokens) == t + 1, "the observation ran past the decision"
        assert int(tokens.action[-1]) == -1, (
            "the pending decision has no action")
        assert list(tokens.action[:-1]) == [
            int(d["action_idx"]) for d in record.decisions[:t]]
        assert list(tokens.cards[-1, 5:]) == [0, 1]
        for u in range(t):
            if int(tokens.acting_pos[u]) != seat:
                assert list(tokens.cards[u, 5:]) == [UNKNOWN_CARD] * 2


# --------------------------------------------------- tokens from any slot


def test_slot_zeros_tokens_are_what_they_have_always_been():
    """The default call is the call `Session.tokens` always made."""
    _pool, session = _session()
    was = [hand_tokens(record, observer_pos=session.seat_of_slot(0, h),
                       slot_of_seat=session.slot_of_seat(h),
                       max_players=MAX_PLAYERS, n_actions=N_ACTIONS)
           for h, record in enumerate(session.records)]
    now = session.tokens(MAX_PLAYERS, N_ACTIONS)

    assert len(now) == len(was)
    for a, b in zip(was, now):
        for field in ("cards", "acting_pos", "slot", "seat_slot", "action",
                      "token_type", "own_hole", "scalars", "seat_stacks"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_the_tokens_of_a_slot_show_that_slots_cards_and_nobody_elses():
    """The reason a past agent may not reuse hero's tokenisation (§9)."""
    _pool, session = _session()
    seen_own = seen_masked = 0
    for slot in range(session.num_players):
        tokens = session.tokens(MAX_PLAYERS, N_ACTIONS, observer_slot=slot)
        for h, tok in enumerate(tokens):
            observer = session.seat_of_slot(slot, h)
            own = session.records[h].hole_cards(observer)
            for t in range(len(tok)):
                cards = tok.cards[t, 5:].tolist()
                if int(tok.acting_pos[t]) == observer:
                    assert cards == own
                    seen_own += 1
                else:
                    assert cards == [UNKNOWN_CARD, UNKNOWN_CARD], (
                        f"seat {tok.acting_pos[t]}'s cards leaked into the "
                        f"view of slot {slot}")
                    seen_masked += 1
            assert np.array_equal(tok.own_hole,
                                  np.tile(np.asarray(own), (len(tok), 1)))
    assert seen_own > 0 and seen_masked > 0


def test_a_window_is_the_last_hands_and_nothing_earlier():
    _pool, session = _session()
    whole = session.tokens(MAX_PLAYERS, N_ACTIONS, observer_slot=1)
    for window in (0, 1, 3, N_HANDS, N_HANDS + 5):
        got = session.tokens(MAX_PLAYERS, N_ACTIONS, observer_slot=1,
                             window=window)
        expected = whole[len(whole) - min(window, len(whole)):]
        assert len(got) == len(expected)
        for a, b in zip(expected, got):
            assert np.array_equal(a.action, b.action)
            assert np.array_equal(a.cards, b.cards)


def test_the_range_targets_belong_to_slot_zero_alone():
    """§5.7's keys name hero's live opponents, so no other view may take them.

    Handing them to another observer is not a silent mismatch — `hand_tokens`
    refuses a key naming a seat that observer has no live opponent at, its own
    seat among them — so this is the assertion that would fire.
    """
    from oracle.ranges import label_ranges

    pool, session = _session(num_players=3, n_hands=4, seed=2)
    label_ranges([session], pool, N_ACTIONS)
    assert any(session.ranges), "the fixture produced no range targets"

    hero = session.tokens(MAX_PLAYERS, N_ACTIONS, observer_slot=0)
    assert any(t.ranges is not None and len(t.ranges.token) for t in hero)
    for slot in (1, 2):
        other = session.tokens(MAX_PLAYERS, N_ACTIONS, observer_slot=slot)
        assert all(t.ranges is None for t in other)


# ------------------------------------------------------ the K = 0 vectors


def test_amortised_vectors_are_the_head_with_no_gradient_step():
    """`K = 0` is `fit_embeddings(steps=0)`, over this slot's own view."""
    pool, session = _session()
    embed_net = _embed_net(len(pool))

    for slot in range(session.num_players):
        got = amortised_vectors(embed_net, session, slot, MAX_PLAYERS,
                                N_ACTIONS, None, "cpu")
        observed = [t for t in session.tokens(MAX_PLAYERS, N_ACTIONS,
                                              observer_slot=slot) if len(t)]
        batch = collate(observed, device="cpu")
        expected = fit_embeddings(embed_net, batch, session.num_players,
                                  steps=0, lr=0.02, reg=0.01)
        assert got.shape == (MAX_PLAYERS, D_EMB)
        assert np.allclose(got, _pad_vectors(expected.cpu().numpy(),
                                             MAX_PLAYERS, D_EMB))
        assert not got[session.num_players:].any(), (
            "a slot the table does not have got a non-zero vector")


def test_the_observer_axis_is_not_decoration():
    """Slot 1's vectors are fitted from slot 1's hands, not from hero's."""
    pool, session = _session()
    embed_net = _embed_net(len(pool))
    hero = amortised_vectors(embed_net, session, 0, MAX_PLAYERS, N_ACTIONS,
                             None, "cpu")
    other = amortised_vectors(embed_net, session, 1, MAX_PLAYERS, N_ACTIONS,
                              None, "cpu")
    assert not np.allclose(hero, other), (
        "two observers with different hole cards produced the same table")


def test_a_slot_that_has_observed_nothing_starts_cold():
    """Block 0's value, and the value of a session with no hands yet (§5.5)."""
    pool, session = _session()
    embed_net = _embed_net(len(pool))
    empty = Session(idx=0, num_players=session.num_players, stack_bb=100,
                    members=session.members, specs=session.specs, records=[])
    got = amortised_vectors(embed_net, empty, 0, MAX_PLAYERS, N_ACTIONS, None,
                            "cpu")
    zero = np.zeros((MAX_PLAYERS, D_EMB), dtype=np.float32)
    assert np.array_equal(got, zero)
    # A window that reaches back over no played hand is the same cold start.
    assert np.array_equal(
        amortised_vectors(embed_net, session, 0, MAX_PLAYERS, N_ACTIONS, 0,
                          "cpu"), zero)


def test_the_window_is_the_history_the_fit_may_look_at():
    pool, session = _session()
    embed_net = _embed_net(len(pool))
    whole = amortised_vectors(embed_net, session, 0, MAX_PLAYERS, N_ACTIONS,
                              None, "cpu")
    assert np.allclose(whole, amortised_vectors(
        embed_net, session, 0, MAX_PLAYERS, N_ACTIONS, N_HANDS, "cpu"))
    assert not np.allclose(whole, amortised_vectors(
        embed_net, session, 0, MAX_PLAYERS, N_ACTIONS, 2, "cpu")), (
        "a two-hand window gave the same vectors as the whole session")
