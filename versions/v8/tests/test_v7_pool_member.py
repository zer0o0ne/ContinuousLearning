"""Vendored v7 checkpoints as pool members (CONCEPT.md §4.1, §4.3, §16 OI-6).

What can be tested on the dev box is the plumbing, and the plumbing is where
this breaks: the vendored perception + action stack has to import and run under
the installed `transformers`, the v7 event format has to be built correctly from
v8's snapshots, and a v7 member has to obey the same observation-parity and
legality contracts as every other member.

What **cannot** be tested here is a real checkpoint: `data/v7/` does not exist on
the dev box and there is no GPU (`CLAUDE.md` §3). The network below is randomly
initialised, so nothing here says anything about how a trained v7 network plays
— only that it is wired up correctly. Loading real weights stays a hypothesis
until it is run on the Spark.
"""

import numpy as np
import pytest
import torch

from pool.v7_member import V7NetworkMember
from tests.g1_fixtures import (
    BIG_BLIND, N_ACTIONS, RAISE_SIZES, SMALL_BLIND, make_pool, make_specs, play,
)
from vendor.v7.agent import V7Agent, n_actions_from_config
from vendor.v7.events import build_v7_events

V7_CONFIG = {
    "architecture": {
        "d_model": 32, "n_heads": 4, "n_kv_heads": 2, "d_ff": 64,
        "n_encoder_layers": 1, "n_decoder_layers": 1, "n_action_layers": 1,
        "max_seq_len": 512, "max_players": 9,
        "memory": {"n_levels": 1, "max_cluster_size": 8,
                   "max_cluster_size_after": 4, "beam_width": 2},
    },
    "game": {"raise_sizes": {"preflop": RAISE_SIZES[0], "flop": RAISE_SIZES[1],
                             "turn": RAISE_SIZES[2], "river": RAISE_SIZES[3]}},
}


def _member():
    agent = V7Agent(V7_CONFIG, log=lambda _m: None).eval()
    return V7NetworkMember("v7", n_actions_from_config(V7_CONFIG), agent)


def test_the_action_layout_matches_v8s():
    assert n_actions_from_config(V7_CONFIG) == N_ACTIONS


def test_a_v7_member_plays_legal_hands_at_every_table_size():
    pool = [_member()] + make_pool()
    for num_players in (2, 5, 9):
        specs = make_specs(seed=41 + num_players, n_hands=3,
                           n_members=len(pool), num_players=num_players)
        for spec in specs:                      # seat the v7 member everywhere
            spec.seat_members = [0] * num_players
        for record in play(pool, specs):
            assert abs(float(record.rewards.sum())) < 1e-9
            for d in record.decisions:
                assert d["legal_mask"][d["action_idx"]]


def test_a_v7_member_returns_a_distribution_over_legal_actions():
    from tests.g1_fixtures import contexts_from
    pool = [_member()]
    specs = make_specs(seed=42, n_hands=3, n_members=1, num_players=3)
    contexts = contexts_from(play(pool, specs))
    assert contexts
    p = pool[0].policy(contexts)
    legal = np.stack([c.legal_mask for c in contexts])
    assert p.shape == (len(contexts), N_ACTIONS)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[~legal], 0.0)


def test_a_style_draw_changes_a_v7_members_play_without_touching_its_weights():
    from tests.g1_fixtures import STYLE_CFG, contexts_from
    from pool.style import sample_style
    base = _member()
    sibling = base.with_style("v7#1", sample_style(np.random.default_rng(0),
                                                   STYLE_CFG))
    assert sibling.agent is base.agent, "the network must be shared, not copied"

    contexts = contexts_from(play([base], make_specs(seed=43, n_hands=2,
                                                     n_members=1,
                                                     num_players=3)))
    assert np.allclose(base.logits(contexts), sibling.logits(contexts))
    assert not np.allclose(base.policy(contexts), sibling.policy(contexts))


# ------------------------------------------------------- the v7 event format


def _one_record():
    return play([_member()], make_specs(seed=44, n_hands=1, n_members=1,
                                        num_players=4))[0]


def test_v7_events_are_built_from_the_acting_players_seat_only():
    record = _one_record()
    for dec in record.decisions:
        pos = dec["acting_pos"]
        events = build_v7_events(
            record.snapshots, record.deck, pos, record.num_players,
            BIG_BLIND, SMALL_BLIND, N_ACTIONS, up_to=dec["snap_idx"])
        own = record.hole_cards(pos)
        others = {c for p in range(record.num_players) if p != pos
                  for c in record.hole_cards(p)}
        for ev in events:
            assert ev["hand"] == own
            assert ev["hero_pos"] == pos
            assert not (set(ev["hand"]) & others)
            assert not (set(c for c in ev["table"] if c >= 0) & others)


def test_v7_events_mask_the_board_to_each_events_own_street():
    record = _one_record()
    events = build_v7_events(
        record.snapshots, record.deck, 0, record.num_players, BIG_BLIND,
        SMALL_BLIND, N_ACTIONS, up_to=len(record.snapshots) - 1)
    for ev, snap in zip(events, record.snapshots):
        revealed = [c for c in ev["table"] if c >= 0]
        assert len(revealed) == {0: 0, 1: 3, 2: 4, 3: 5}[int(snap["turn"])]
        assert revealed == [int(c) for c in record.deck[:len(revealed)]]


def test_v7_events_stop_at_the_decision_they_were_built_for():
    record = _one_record()
    for dec in record.decisions:
        events = build_v7_events(
            record.snapshots, record.deck, dec["acting_pos"],
            record.num_players, BIG_BLIND, SMALL_BLIND, N_ACTIONS,
            up_to=dec["snap_idx"])
        assert len(events) == dec["snap_idx"] + 1
        assert not any(events[-1]["action"]), (
            "the last event of a decision sequence is the pre-decision "
            "snapshot and carries no action")


def test_v7_events_carry_the_per_seat_stack_vector():
    """B.6.2 — v7 found a per-seat effective-stack signal necessary, and v8
    covers 10–300 BB across 2–9 seats (§5.1, OI-1)."""
    record = _one_record()
    events = build_v7_events(
        record.snapshots, record.deck, 1, record.num_players, BIG_BLIND,
        SMALL_BLIND, N_ACTIONS, up_to=3)
    for ev, snap in zip(events, record.snapshots):
        assert ev["stacks"] == [float(c) for c in snap["credits"]]
        assert ev["stack"] == pytest.approx(float(snap["credits"][1]))
        assert ev["num_players"] == record.num_players


def test_a_missing_checkpoint_is_an_error_not_a_random_network():
    """A randomly initialised "v7 member" is not the strategy the pool is meant
    to hold, so it must not be created silently."""
    agent = V7Agent(V7_CONFIG, log=lambda _m: None)
    with pytest.raises(FileNotFoundError):
        agent.load_checkpoint("../../data/v7/definitely-not-here/best.pt")


# ------------------------------------ D2: `hole_override` (CONCEPT.md §7.2)


def test_a_v7_member_answers_the_override_not_the_real_hand():
    """The posterior asks "what would you have done holding *this*". A member
    that reads its cards out of the deck has to be handed the hypothetical
    deck, or every combo gets the same answer and §7.2's posterior never
    leaves the prior."""
    from env.driver import DecisionContext
    torch.manual_seed(0)
    member = _member()
    record = _one_record()
    dec = record.decisions[0]
    pos, snap_idx = int(dec["acting_pos"]), int(dec["snap_idx"])
    turn = int(record.snapshots[snap_idx]["turn"])
    real = record.hole_cards(pos)

    def ask(hole):
        ctx = DecisionContext(record, snap_idx, pos, dec["legal_mask"], turn,
                              hole_override=hole)
        return member.logits([ctx])[0]

    plain = member.logits([DecisionContext(record, snap_idx, pos,
                                           dec["legal_mask"], turn)])[0]
    assert np.allclose(ask(real), plain), (
        "an override naming the real cards must be the real answer")

    aces, trash = ask([48, 49]), ask([0, 5])
    assert not np.allclose(aces, trash), (
        "the member gave the same answer for aces and for deuce-trey — the "
        "override never reached it")
    # The record is untouched: the swap lands in a copy of the deck.
    assert record.hole_cards(pos) == real


def test_the_posterior_leaves_the_prior_for_a_v7_opponent():
    """End to end: a network opponent's posterior has to depend on what it did.
    Before the override reached `build_v7_events` this test's weights were flat
    to machine precision, and nothing else in the battery could see it."""
    from oracle.posterior import opponent_posterior
    torch.manual_seed(0)
    pool = [_member()]
    specs = make_specs(seed=45, n_hands=4, n_members=1, num_players=3)
    for spec in specs:
        spec.seat_members = [0] * spec.num_players
    for record in play(pool, specs):
        for t, dec in enumerate(record.decisions):
            opp = int(dec["acting_pos"])
            hero = (opp + 1) % record.num_players
            _combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                            through_decision=t)
            assert abs(float(w.sum()) - 1.0) < 1e-9
            if float(w.max() - w.min()) > 1e-9:
                return
    raise AssertionError("no decision moved the posterior off the prior")
