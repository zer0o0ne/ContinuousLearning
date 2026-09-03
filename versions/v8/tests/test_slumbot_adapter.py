"""The Slumbot adapter (CONCEPT.md §12, §10, `PLAN_PIPELINE.md` S10).

No network access: every case runs off canned action strings, which is also the
only way these properties can be pinned deterministically (`CLAUDE.md` §4).

What is worth testing here is the seam, because both halves of it are already
tested elsewhere. The protocol layer is v7's and unchanged; the observation
builder, the legality rule and the last mile from logits to a played action are
v8's and covered by `test_observation_parity.py`, `test_driver_lockstep.py` and
`test_agent_net.py`. What is new is the join, and it has exactly three ways to
be silently wrong:

* **the frame** — Slumbot numbers seats the other way round, so a flip in the
  wrong place puts hero in the opponent's chair and nothing complains;
* **the chips** — hero's own bets go through an abstraction (`action_idx_to_incr`)
  and Slumbot's do not, so a replay through the engine would show the agent the
  *bin's* pot rather than the table's;
* **parity** — the record is built here rather than played by the engine, and a
  built record is exactly where a card or an action from the future can appear.
"""

import numpy as np
import pytest
import torch

import evaluation.protocol as protocol
from agent.policy import AgentPoolMember
from env.legal import legal_action_mask
from evaluation.protocol import (
    SLUMBOT_BIG_BLIND, SLUMBOT_STACK_SIZE, action_idx_to_incr, board_to_ints,
    card_to_int, clamp_counters, effective_action_idx, replay_action_string,
    token_to_action_idx,
)
from evaluation.v8_adapter import (
    AgentMemberFactory, MemberFactory, N_SEATS, SlumbotAgent, _flip,
    _table_view, check_table_is_in_range, slumbot_record,
)
from env.session import raise_sizes_from
from nets.agent_net import AgentNet
from nets.features import TOKEN_DECISION, UNKNOWN_CARD, hand_tokens
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG, RAISE_SIZES, SMALL_BLIND,
)

GAME = {
    "n_actions": N_ACTIONS,
    "max_players": MAX_PLAYERS,
    "big_blind": BIG_BLIND,
    "small_blind": SMALL_BLIND,
    "players_range": [2, 9],
    "stack_bb_range": [10, 300],
    "raise_sizes": {"preflop": RAISE_SIZES[0], "flop": RAISE_SIZES[1],
                    "turn": RAISE_SIZES[2], "river": RAISE_SIZES[3]},
}
N_RAISE_BINS = N_ACTIONS - 3
SCALE = BIG_BLIND / SLUMBOT_BIG_BLIND

HOLE = ["Ac", "Kd"]
BOARD5 = ["7h", "2s", "Ts", "9c", "3d"]

# (action string, whose turn it is in Slumbot's frame, how much board is out).
# Between them they cover every street, both seats, a limp, a raise, a
# three-bet, a check-through and an all-in.
CANNED = [
    ("", 1, 0),
    ("b200", 0, 0),
    ("c", 0, 0),
    ("b200b600", 1, 0),
    ("b250c/", 0, 3),
    ("b250c/b300", 1, 3),
    ("b250c/kk/", 0, 4),
    ("b250c/kk/b400c/", 0, 5),
    ("b250c/kk/kk/b500", 1, 5),
    ("b20000", 0, 0),
]


def _hand(action_str, client_pos, n_board):
    return dict(action_str=action_str, client_pos=client_pos,
                hole_cards=[card_to_int(c) for c in HOLE],
                board=board_to_ints(BOARD5[:n_board]))


def _record(action_str, client_pos, n_board, hero_action_indices=()):
    h = _hand(action_str, client_pos, n_board)
    return slumbot_record(h["action_str"], h["client_pos"], h["hole_cards"],
                          h["board"], GAME,
                          hero_action_indices=hero_action_indices)


def _tokens(record, ctx):
    return hand_tokens(record, observer_pos=int(ctx.acting_pos),
                       slot_of_seat=[0 if seat == int(ctx.acting_pos) else 1
                                     for seat in range(N_SEATS)],
                       max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                       pending=ctx)


# ------------------------------------------------------- 1: the legality rule


def test_the_mask_hero_acts_under_is_env_legals_and_not_a_second_copy():
    for action_str, client_pos, n_board in CANNED:
        record, ctx, state = _record(action_str, client_pos, n_board)
        table = _table_view(state, GAME, SCALE)
        assert np.array_equal(np.asarray(ctx.legal_mask, dtype=bool),
                              legal_action_mask(table, N_ACTIONS)), action_str
        # The mask reaches the token the agent reads, unchanged.
        assert np.array_equal(_tokens(record, ctx).legal[-1],
                              np.asarray(ctx.legal_mask, dtype=bool))
        assert ctx.legal_mask.any()


def test_the_mask_says_what_the_situation_says():
    """Two situations that differ only in whether hero faces a bet."""
    # Preflop, hero is the SB and owes the big blind: folding is a real choice.
    _record_sb, ctx_sb, _s = _record("", 1, 0)
    assert bool(ctx_sb.legal_mask[0]), "facing the blind, fold must be legal"
    assert bool(ctx_sb.legal_mask[1])

    # Flop, hero is first to act into an unopened pot: checking dominates
    # folding, and `env/legal.py` drops the dominated branch.
    _record_bb, ctx_bb, _s = _record("b250c/", 0, 3)
    assert not bool(ctx_bb.legal_mask[0]), (
        "nothing to call, so folding is dominated and must not be offered")
    assert bool(ctx_bb.legal_mask[1])
    assert bool(ctx_bb.legal_mask[N_ACTIONS - 1]), "an all-in is always there"


def test_facing_an_all_in_leaves_no_raise():
    _r, ctx, _s = _record("b20000", 0, 0)
    assert bool(ctx.legal_mask[0]) and bool(ctx.legal_mask[1])
    assert not ctx.legal_mask[2:].any(), (
        "the opponent is all-in — a raise could only come back uncalled")


# ---------------------------------------------- 2: the chips are the table's


def test_the_observation_carries_slumbots_chips_and_not_the_abstractions():
    """A bet that lands on no bin of ours still reads as itself.

    `b250` is not `0.5 / 1.0 / 2.0` of anything on this grid, so an adapter that
    replayed the hand through the engine from action *indices* would show the
    agent the nearest bin's pot instead of the real one. The pot after
    `b250c` is 500 chips, which is 5 BB, and that is what the token must say.
    """
    record, ctx, state = _record("b250c/", 0, 3)
    tokens = _tokens(record, ctx)
    stack_bb, pot_bb, to_call_bb = (float(x) for x in tokens.scalars[-1])
    assert pot_bb == pytest.approx(500 / SLUMBOT_BIG_BLIND, abs=1e-9)
    assert to_call_bb == pytest.approx(0.0, abs=1e-9)
    assert stack_bb == pytest.approx(
        (SLUMBOT_STACK_SIZE - 250) / SLUMBOT_BIG_BLIND, abs=1e-9)
    assert float(state["pot"]) == 500.0


def test_every_canned_state_reads_its_own_pot_and_stacks():
    for action_str, client_pos, n_board in CANNED:
        record, ctx, state = _record(action_str, client_pos, n_board)
        tokens = _tokens(record, ctx)
        stack_bb, pot_bb, to_call_bb = (float(x) for x in tokens.scalars[-1])
        hero = int(client_pos)
        assert pot_bb == pytest.approx(
            state["pot"] / SLUMBOT_BIG_BLIND, abs=1e-9), action_str
        assert stack_bb == pytest.approx(
            state["credits"][hero] / SLUMBOT_BIG_BLIND, abs=1e-9), action_str
        owed = state["high_bet"] - state["bets"][hero]
        assert to_call_bb == pytest.approx(
            max(0.0, owed) / SLUMBOT_BIG_BLIND, abs=1e-9), action_str


def test_the_two_seat_frames_are_mirrors_and_hero_sits_where_it_should():
    for action_str, client_pos, n_board in CANNED:
        record, ctx, _state = _record(action_str, client_pos, n_board)
        assert int(ctx.acting_pos) == _flip(client_pos)
        assert record.hole_cards(int(ctx.acting_pos)) == [
            card_to_int(c) for c in HOLE]
        # Preflop the small blind acts first and sits at seat 0 in v8's engine;
        # postflop, heads-up, seat 1 does. Both hold for every recorded seat.
        for dec in record.decisions:
            assert 0 <= int(dec["acting_pos"]) < N_SEATS
    # The very first decision of a hand is the small blind's, which is seat 0.
    record, _ctx, _s = _record("b250c/kk/", 0, 4)
    assert int(record.decisions[0]["acting_pos"]) == 0
    # ...and the first decision of the flop is the big blind's, which is seat 1.
    flop = [d for d in record.decisions
            if record.snapshots[d["snap_idx"]]["turn"] == 1]
    assert int(flop[0]["acting_pos"]) == 1


# ----------------------------------------------------------- 3: §9 parity


def test_observation_parity_on_the_replayed_tokens():
    for action_str, client_pos, n_board in CANNED:
        record, ctx, _state = _record(action_str, client_pos, n_board)
        tokens = _tokens(record, ctx)
        hero = int(ctx.acting_pos)

        assert (tokens.token_type == TOKEN_DECISION).all(), (
            "a hand in progress has no showdown token")
        assert len(tokens) == len(record.decisions) + 1
        assert int(tokens.action[-1]) == -1, "the pending decision has no action"
        assert list(tokens.action[:-1]) == [
            int(d["action_idx"]) for d in record.decisions]

        for t in range(len(tokens)):
            own = int(tokens.acting_pos[t]) == hero
            expected = ([card_to_int(c) for c in HOLE] if own
                        else [UNKNOWN_CARD, UNKNOWN_CARD])
            assert list(tokens.cards[t, 5:]) == expected, (action_str, t)

            street = int(record.snapshots[
                record.decisions[t]["snap_idx"]]["turn"]) if t < len(
                    record.decisions) else int(ctx.turn)
            visible = {0: 0, 1: 3, 2: 4, 3: 5}[street]
            board = list(tokens.cards[t, :5])
            assert board[:visible] == [card_to_int(c)
                                       for c in BOARD5[:visible]], action_str
            assert board[visible:] == [UNKNOWN_CARD] * (5 - visible), (
                f"{action_str}: the board is ahead of the street")


def test_a_prefix_observation_does_not_depend_on_what_happened_later():
    """The tokens of the first k decisions are the same however the hand went."""
    short, ctx_short, _s = _record("b250", 0, 0)
    long, _ctx_long, _s2 = _record("b250c/kk/b400", 1, 4)
    a = _tokens(short, ctx_short)
    b = hand_tokens(long, observer_pos=int(ctx_short.acting_pos),
                    slot_of_seat=[0 if s == int(ctx_short.acting_pos) else 1
                                  for s in range(N_SEATS)],
                    max_players=MAX_PLAYERS, n_actions=N_ACTIONS)
    assert np.array_equal(a.cards[:1], b.cards[:1])
    assert np.array_equal(a.scalars[:1], b.scalars[:1])
    assert np.array_equal(a.legal[:1], b.legal[:1])


# --------------------------------------------- 4: the wire round trip


def test_every_legal_action_round_trips_through_the_wire_on_every_street():
    """index → `incr` → index, with no clamp firing on a legal action.

    A clamp is `action_idx_to_incr` rescuing an action our own legality rule
    should never have offered, so on a legal index it must not fire at all — and
    if it ever does, the two rules have drifted and the agent is playing
    something other than what it chose.
    """
    streets_seen = set()
    for action_str, client_pos, n_board in CANNED:
        _record_, ctx, state = _record(action_str, client_pos, n_board)
        street = int(state["turn"])
        streets_seen.add(street)
        sizes = RAISE_SIZES[street]
        for idx in np.flatnonzero(np.asarray(ctx.legal_mask, dtype=bool)):
            counters = clamp_counters()
            incr = action_idx_to_incr(state, int(idx), sizes, N_RAISE_BINS,
                                      hero_slumbot_pos=int(client_pos),
                                      clamp_counters=counters)
            assert sum(counters.values()) == 0, (
                f"{action_str}: action {idx} was clamped though it is legal "
                f"({counters})")
            assert effective_action_idx(state, incr, sizes,
                                        N_RAISE_BINS) == int(idx), (
                f"{action_str}: action {idx} came back as {incr!r}")
    assert streets_seen == {0, 1, 2, 3}, streets_seen


def test_a_clamp_reports_itself_and_the_effective_index_is_what_was_played():
    """Folding with nothing to call becomes a check, and says so."""
    _r, _ctx, state = _record("b250c/", 0, 3)   # flop, nothing to call
    counters = clamp_counters()
    incr = action_idx_to_incr(state, 0, RAISE_SIZES[1], N_RAISE_BINS,
                              hero_slumbot_pos=0, clamp_counters=counters)
    assert incr == "k"
    assert counters["fold_to_check"] == 1
    assert effective_action_idx(state, incr, RAISE_SIZES[1], N_RAISE_BINS) == 1


def test_heros_own_clamped_action_is_what_the_replay_reads_back():
    """A clamped raise leaves as a call, and the record must say `call`."""
    # Hero (client_pos 0, the BB) checks the flop; the wire token is `k`, whose
    # recovered index is 1, and that is what the next replay is handed.
    record, _ctx, _s = _record("b250c/kk/", 0, 4, hero_action_indices=[1])
    flop = [d for d in record.decisions
            if record.snapshots[d["snap_idx"]]["turn"] == 1]
    assert [int(d["action_idx"]) for d in flop] == [1, 1]


def test_the_replay_prefers_what_hero_played_over_what_the_token_says():
    """`hero_action_indices` overrides only hero's seat, and only in order."""
    _state, steps = replay_action_string(
        "b250c/b400c/", RAISE_SIZES, N_RAISE_BINS, hero_pos=1,
        hero_action_indices=[N_ACTIONS - 1, 4])
    hero_steps = [s for s in steps if s["acting_pos"] == 1]
    opp_steps = [s for s in steps if s["acting_pos"] == 0]
    assert [s["action_idx"] for s in hero_steps] == [N_ACTIONS - 1, 4]
    assert [s["action_idx"] for s in opp_steps] == [
        token_to_action_idx(s["state"], s["token"],
                            RAISE_SIZES[s["state"]["turn"]], N_RAISE_BINS)
        for s in opp_steps]


# ------------------------------------------------- 5: the result accounting


def test_bb_per_100_and_its_standard_error_are_the_textbook_ones():
    chips = [100.0, -50.0, 200.0]
    bb = np.asarray(chips) / SLUMBOT_BIG_BLIND

    assert protocol.bb_per_100(sum(chips), len(chips)) == pytest.approx(
        bb.sum() / (len(chips) / 100.0), abs=1e-12)
    assert protocol.bb_per_100(sum(chips), len(chips)) == pytest.approx(
        250.0 / 3.0, abs=1e-12)

    expected_se = float(bb.std(ddof=1) / np.sqrt(len(bb)) * 100.0)
    assert protocol.stderr_bb_per_100(chips) == pytest.approx(
        expected_se, abs=1e-12)

    # Welford's online form must agree with the batch one to the last bit that
    # matters — a million-hand run keeps no list.
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in bb:
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    assert protocol.stderr_bb_per_100_online(n, m2) == pytest.approx(
        expected_se, abs=1e-12)


def test_the_edge_cases_of_the_accounting_are_zero_and_not_a_crash():
    assert protocol.bb_per_100(0.0, 0) == 0.0
    assert protocol.stderr_bb_per_100([]) == 0.0
    assert protocol.stderr_bb_per_100([100.0]) == 0.0
    assert protocol.stderr_bb_per_100_online(1, 0.0) == 0.0
    assert protocol.bb_per_100(-500.0, 100) == pytest.approx(-5.0, abs=1e-12)


# ------------------------------------------- 6: the table is checked, not clamped


def test_a_table_outside_the_training_ranges_is_refused():
    assert check_table_is_in_range(GAME) == (2, 200.0)

    with pytest.raises(AssertionError, match="players_range"):
        check_table_is_in_range({**GAME, "players_range": [3, 9]})
    with pytest.raises(AssertionError, match="stack_bb_range"):
        check_table_is_in_range({**GAME, "stack_bb_range": [10, 100]})
    with pytest.raises(AssertionError, match="stack_bb_range"):
        SlumbotAgent(AgentMemberFactory(_net(), GAME, "cpu"),
                     {**GAME, "stack_bb_range": [10, 100]}, "cpu")


# ------------------------------------------------------- 7: the whole path


def _net(seed=0):
    torch.manual_seed(seed)
    return AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()


def _agent_hero(seed=0):
    """The default hero: entity 3, reached through its own member factory."""
    return AgentMemberFactory(_net(seed), GAME, "cpu")


def test_the_agent_answers_every_canned_state_with_a_legal_wire_token():
    agent = SlumbotAgent(_agent_hero(), GAME, "cpu", rng=np.random.default_rng(1))
    for action_str, client_pos, n_board in CANNED:
        h = _hand(action_str, client_pos, n_board)
        probs, _record, ctx, _state = agent.policy(
            h["action_str"], h["client_pos"], h["hole_cards"], h["board"])
        legal = np.asarray(ctx.legal_mask, dtype=bool)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        assert not probs[~legal].any(), "mass on an illegal action"

        counters = clamp_counters()
        incr, effective, chosen = agent.act(
            h["action_str"], h["client_pos"], h["hole_cards"], h["board"],
            counters=counters)
        assert legal[chosen] and legal[effective]
        assert sum(counters.values()) == 0, (action_str, counters)
        assert incr == "f" or incr in ("c", "k") or incr.startswith("b")


def test_the_agent_is_deterministic_under_its_own_generator():
    a = SlumbotAgent(_agent_hero(), GAME, "cpu", rng=np.random.default_rng(7))
    b = SlumbotAgent(_agent_hero(), GAME, "cpu", rng=np.random.default_rng(7))
    h = _hand("b250c/", 0, 3)
    for _ in range(5):
        assert a.act(h["action_str"], h["client_pos"], h["hole_cards"],
                     h["board"]) == b.act(
            h["action_str"], h["client_pos"], h["hole_cards"], h["board"])


def test_cold_is_the_zero_table_and_warm_replaces_it():
    net = _net()
    agent = SlumbotAgent(AgentMemberFactory(net, GAME, "cpu"), GAME, "cpu")
    assert not agent.embeddings.any(), "the cold start is the zero vector (§5.5)"

    h = _hand("b250c/", 0, 3)
    cold = agent.policy(h["action_str"], h["client_pos"], h["hole_cards"],
                        h["board"])[0]
    rng = np.random.default_rng(3)
    agent.set_embeddings(rng.normal(size=(MAX_PLAYERS, net.d_emb)))
    warm = agent.policy(h["action_str"], h["client_pos"], h["hole_cards"],
                        h["board"])[0]
    assert not np.allclose(cold, warm), (
        "the fitted vector did not reach the policy — §12's warm run would be "
        "measuring the cold one")
    with pytest.raises(AssertionError, match="max_players"):
        agent.set_embeddings(np.zeros((2, net.d_emb)))


def _regular_hero(tmp_path):
    """A procedural §P4 archetype as the hero, on the fixture raise grid."""
    from pool.archetypes import draw_params
    from pool.regular import RegularMember
    from pool.strength import StrengthCache, preflop_equity_table

    table = preflop_equity_table(str(tmp_path / "preflop.npy"), seed=0,
                                 n_deals=20_000)
    member = RegularMember("tag", N_ACTIONS, draw_params(
        "tag", np.random.default_rng(0), 0.0), StrengthCache(64), table,
        raise_sizes_from(GAME))
    return MemberFactory(member)


def test_a_procedural_member_can_be_the_hero(tmp_path):
    """§P5: the same wire, the same legality, a different player.

    Nothing about the adapter knows which kind of member is answering — which
    is the point of the change, and why this test is the same assertions as the
    agent's own with the hero swapped.
    """
    agent = SlumbotAgent(_regular_hero(tmp_path), GAME, "cpu",
                         rng=np.random.default_rng(2))
    assert agent.embeddings is None

    for action_str, client_pos, n_board in CANNED:
        h = _hand(action_str, client_pos, n_board)
        probs, _record, ctx, _state = agent.policy(
            h["action_str"], h["client_pos"], h["hole_cards"], h["board"])
        legal = np.asarray(ctx.legal_mask, dtype=bool)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        assert not probs[~legal].any()

        counters = clamp_counters()
        incr, effective, chosen = agent.act(
            h["action_str"], h["client_pos"], h["hole_cards"], h["board"],
            counters=counters)
        assert legal[chosen] and legal[effective]
        assert incr == "f" or incr in ("c", "k") or incr.startswith("b")


def test_a_procedural_hero_has_no_vector_to_warm_up(tmp_path):
    agent = SlumbotAgent(_regular_hero(tmp_path), GAME, "cpu")
    with pytest.raises(AssertionError, match="no opponent vector"):
        agent.set_embeddings(np.zeros((MAX_PLAYERS, 8)))


def test_the_agent_reaches_the_network_through_the_ordinary_pool_member(
        monkeypatch):
    """§6.1, §9: one observation builder and one last mile, not a Slumbot copy."""
    seen = []
    real = AgentPoolMember.logits

    def spy(self, contexts):
        seen.append((self.observer_pos, tuple(self.slot_of_seat)))
        return real(self, contexts)

    monkeypatch.setattr(AgentPoolMember, "logits", spy)
    agent = SlumbotAgent(_agent_hero(), GAME, "cpu")
    h = _hand("b250c/", 0, 3)
    agent.act(h["action_str"], h["client_pos"], h["hole_cards"], h["board"])
    assert seen == [(_flip(0), (1, 0))], seen
