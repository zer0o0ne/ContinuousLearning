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

from pool.action_map import RaiseGridMap
from pool.v7_member import V7NetworkMember
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, RAISE_SIZES, SMALL_BLIND, STYLE_CFG,
    make_pool, make_specs, play,
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


# ------------------------------------- a checkpoint whose raise grid differs

# The pool's grid: finer than `RAISE_SIZES` ([0.5, 1.0, 2.0]) and starting
# below it, so the transport has to handle every case at once — several v7 bins
# nearest to one pool bin, pool bins that are nobody's nearest, and a pool bin
# under the smallest size the checkpoint knows.
POOL_RAISE_SIZES = [[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]] * 4
POOL_N_ACTIONS = len(POOL_RAISE_SIZES[0]) + 3
POOL_GAME = {"n_actions": POOL_N_ACTIONS, "max_players": MAX_PLAYERS,
             "big_blind": BIG_BLIND, "small_blind": SMALL_BLIND,
             "players_range": [2, 9], "stack_bb_range": [10, 300],
             "raise_sizes": {"preflop": POOL_RAISE_SIZES[0],
                             "flop": POOL_RAISE_SIZES[1],
                             "turn": POOL_RAISE_SIZES[2],
                             "river": POOL_RAISE_SIZES[3]}}

FOLD, CALL = 0, 1
V7_ALLIN = N_ACTIONS - 1
POOL_ALLIN = POOL_N_ACTIONS - 1


def _map(src=None, dst=None):
    return RaiseGridMap(src if src else RAISE_SIZES,
                        dst if dst else POOL_RAISE_SIZES)


def _mapped_member():
    agent = V7Agent(V7_CONFIG, log=lambda _m: None).eval()
    return V7NetworkMember("v7", POOL_N_ACTIONS, agent, action_map=_map())


def test_a_map_between_two_identical_grids_is_the_identity():
    """The gate configs run the checkpoint's own grid, and on that path the
    member must behave exactly as it did before any of this existed."""
    assert _map(dst=RAISE_SIZES).is_identity
    assert not _map().is_identity


def test_the_positional_actions_map_to_themselves_on_every_street():
    m = _map()
    for street in range(4):
        assert m.to_src[street][FOLD] == FOLD
        assert m.to_src[street][CALL] == CALL
        assert m.to_src[street][POOL_ALLIN] == V7_ALLIN
        assert list(m.groups[street][FOLD]) == [FOLD]
        assert list(m.groups[street][CALL]) == [CALL]
        assert list(m.groups[street][POOL_ALLIN]) == [V7_ALLIN]


def test_a_pool_action_reaches_v7_as_the_nearest_size_it_knows():
    """The history a v7 member reads is written in the pool's action set, and
    the one-hot it is handed has to name the size v7 has a bin for."""
    m = _map()
    # pool bins   0.25  0.5  0.75  1.0  1.5  2.0  3.0
    # nearest v7  0.5   0.5  0.5*  1.0  1.0* 2.0  2.0     (* tie → smaller)
    expected = [0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    for j, frac in enumerate(expected):
        onehot = m.src_onehot(0, j + 2)
        assert sum(onehot) == 1.0 and len(onehot) == N_ACTIONS
        assert RAISE_SIZES[0][onehot.index(1.0) - 2] == frac


def test_the_transport_moves_every_v7_bin_onto_the_nearest_pool_bin():
    """§ the owner's decision: whole mass to the single nearest size, so a v7
    member keeps betting what it meant to bet."""
    m = _map()
    logits = np.zeros((1, N_ACTIONS))
    logits[0, 2] = 10.0                       # v7's 0.5x, almost all the mass
    p = np.exp(m.dst_logprobs(logits, [0]))[0]
    assert POOL_RAISE_SIZES[0][int(p[2:-1].argmax())] == 0.5


def test_the_transport_conserves_probability_mass():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(16, N_ACTIONS)) * 3.0
    streets = rng.integers(0, 4, size=16)
    p = np.exp(_map().dst_logprobs(logits, streets))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_mass_of_two_v7_bins_that_collide_into_one_pool_bin_is_summed():
    """The other direction: a pool grid coarser than the checkpoint's."""
    coarse = [[1.0]] * 4                                  # one bin, 4 actions
    m = RaiseGridMap(RAISE_SIZES, coarse)
    logits = np.log([[1.0, 1.0, 2.0, 3.0, 4.0, 1.0]])     # 6 v7 actions
    p = np.exp(m.dst_logprobs(logits, [0]))[0]
    assert p.shape == (4,)
    assert p[2] == pytest.approx(9.0 / 12.0)              # 0.5x + 1x + 2x
    assert p[CALL] == pytest.approx(1.0 / 12.0)
    assert p[3] == pytest.approx(1.0 / 12.0)              # all-in


def test_a_pool_bin_no_v7_bin_is_nearest_to_gets_no_mass():
    """The deliberate hole (owner decision): the checkpoint has no opinion
    about a size it was never given, so it never plays it. Floored rather than
    zeroed so masking can never produce a `nan` row."""
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(8, N_ACTIONS)) * 3.0
    p = np.exp(_map().dst_logprobs(logits, [0] * 8))
    unreachable = [j for j in range(2, POOL_N_ACTIONS - 1)
                   if not len(_map().groups[0][j])]
    assert unreachable, "the fixture must have a pool bin nobody maps onto"
    assert np.all(p[:, unreachable] < 1e-20)
    assert np.all(np.isfinite(np.log(p[:, unreachable])))


def test_a_v7_member_on_a_different_grid_plays_legal_hands():
    """The whole point: the checkpoint drives a table whose raise grid it was
    never trained on, and every action it takes is a legal action of *that*
    table."""
    member = _mapped_member()
    pool = [member]
    for num_players in (2, 5, 9):
        specs = make_specs(seed=61 + num_players, n_hands=3, n_members=1,
                           num_players=num_players,
                           raise_sizes=POOL_RAISE_SIZES)
        for record in play(pool, specs, n_actions=POOL_N_ACTIONS):
            assert abs(float(record.rewards.sum())) < 1e-9
            for d in record.decisions:
                assert len(d["legal_mask"]) == POOL_N_ACTIONS
                assert d["legal_mask"][d["action_idx"]]


def test_a_mapped_member_returns_a_distribution_over_the_pools_action_set():
    from tests.g1_fixtures import contexts_from
    member = _mapped_member()
    specs = make_specs(seed=62, n_hands=3, n_members=1, num_players=3,
                       raise_sizes=POOL_RAISE_SIZES)
    contexts = contexts_from(play([member], specs, n_actions=POOL_N_ACTIONS))
    assert contexts
    p = member.policy(contexts)
    legal = np.stack([c.legal_mask for c in contexts])
    assert p.shape == (len(contexts), POOL_N_ACTIONS)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[~legal], 0.0)


def test_the_history_a_mapped_member_reads_is_rewritten_but_the_record_is_not():
    """The translation lands in a copy: the record keeps the pool's own action
    indices, which is what every other consumer of it reads."""
    from tests.g1_fixtures import contexts_from
    member = _mapped_member()
    records = play([member], make_specs(seed=63, n_hands=2, n_members=1,
                                        num_players=4,
                                        raise_sizes=POOL_RAISE_SIZES),
                   n_actions=POOL_N_ACTIONS)
    contexts = contexts_from(records)
    member.logits(contexts)                        # runs the translation

    cache = {}
    for record in records:
        translated = member._snapshots(record, cache)
        for snap, was in zip(translated, record.snapshots):
            assert len(was["action"] or []) in (0, POOL_N_ACTIONS)
            if was["action"] is None:
                assert snap["action"] is None
            else:
                assert len(snap["action"]) == N_ACTIONS
                assert sum(snap["action"]) == 1.0
                assert len(was["action"]) == POOL_N_ACTIONS


def test_the_pool_builds_a_mapped_member_from_a_checkpoint(tmp_path):
    """End to end through `build_pool`: a checkpoint on one grid, a config on
    another, and a pool that plays."""
    import json

    from pool.build import build_pool

    ckpt = tmp_path / "best.pt"
    arch = tmp_path / "config.json"
    torch.save({"model_state_dict":
                V7Agent(V7_CONFIG, log=lambda _m: None).state_dict()}, ckpt)
    arch.write_text(json.dumps(V7_CONFIG))

    config = {
        "game": POOL_GAME,
        "style": STYLE_CFG,
        "bootstrap": [{"kind": "v7", "checkpoint": str(ckpt),
                       "arch_config": str(arch), "n_variants": 2,
                       "label": "v7_other_grid"}],
    }
    members, descriptors = build_pool(config, np.random.default_rng(0),
                                      log=lambda _m: None)
    assert len(members) == 2
    assert all(d["base"] == "v7_other_grid" for d in descriptors)
    for member in members:
        assert member.n_actions == POOL_N_ACTIONS
        assert member.action_map is not None
        assert member.agent is members[0].agent, "one load, shared weights"

    specs = make_specs(seed=64, n_hands=2, n_members=2, num_players=3,
                       raise_sizes=POOL_RAISE_SIZES)
    for record in play(members, specs, n_actions=POOL_N_ACTIONS):
        for d in record.decisions:
            assert d["legal_mask"][d["action_idx"]]


def test_the_pool_refuses_a_checkpoint_whose_grid_it_cannot_align(tmp_path):
    import json

    from pool.build import build_pool

    ckpt = tmp_path / "best.pt"
    arch = tmp_path / "config.json"
    torch.save({"model_state_dict":
                V7Agent(V7_CONFIG, log=lambda _m: None).state_dict()}, ckpt)
    # A config that says how many actions the network has but not what they
    # mean: there is nothing to align the grids by.
    blind = {"architecture": V7_CONFIG["architecture"],
             "game": {"table_bins": len(RAISE_SIZES[0])}}
    arch.write_text(json.dumps(blind))

    config = {"game": POOL_GAME, "style": STYLE_CFG,
              "bootstrap": [{"kind": "v7", "checkpoint": str(ckpt),
                             "arch_config": str(arch), "label": "blind"}]}
    with pytest.raises(AssertionError, match="no `game.raise_sizes`"):
        build_pool(config, np.random.default_rng(0), log=lambda _m: None)
