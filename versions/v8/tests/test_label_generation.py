"""Label generation end to end (CONCEPT.md §8, `PLAN_PIPELINE.md` S7).

Toy scale — four sessions, six hands, two rollout samples per action — so
nothing here says whether a label is any *good*. What it pins is the ordering
that makes a label honest, and every case below is one that fails silently if it
is wrong:

* the embedding attached to a label is the one hero held when it acted, fitted
  from the hands before the refresh and from nothing after it (§5.5, §9);
* the stored token prefix is hero's own observation, with no card and no
  decision from the future of the labelled moment;
* the same seed produces the same bytes, so a training set is reproducible;
* the session machinery is the one G1 used, not a copy of it (D4).
"""

import json

import numpy as np
import pytest
import torch

import env.session
import gates.g1
import train.generate
from agent.policy import AgentPoolMember
from env.driver import LockstepDriver
from env.session import build_sessions
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from nets.features import TOKEN_DECISION, UNKNOWN_CARD
from pool.base import PoolMember
from train.generate import generate_labels, load_shard
from train.targets import policy_target
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG, RAISE_SIZES, SMALL_BLIND,
    make_pool,
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


def _cfg(seed=3, n_sessions=4, hands=6, R=3, samples=2, max_combos=4):
    return {
        "seed": seed,
        "n_sessions": n_sessions,
        "hands_per_session": hands,
        "driver_batch_size": 16,
        "labels_per_shard": 8,
        "game": GAME,
        "embedding_net": {
            "K": 2, "fit_lr": 0.1, "fit_reg": 0.01, "R": R,
            "amortised_weight": 1.0, "showdown_strength_weight": 0.3,
            "showdown_class_weight": 0.1,
        },
        "oracle": {"samples_per_action": samples, "max_combos": max_combos,
                   "batch_hands": 256},
    }


class _UniformSampler:
    """The §4.4 sampler S8 will replace: uniform, without replacement."""

    def __init__(self, n_members, seed):
        self.n_members = n_members
        self.rng = np.random.default_rng(seed)

    def sample_table(self, k):
        return [int(m) for m in
                self.rng.choice(self.n_members, size=k, replace=False)]


class _JamFrom(PoolMember):
    """Hero, playing its own policy until hand `from_hand` and then jamming.

    The intervention test 3 needs: an identical prefix of hands followed by a
    completely different tail, inside one `generate_labels` call.
    """

    def __init__(self, inner, from_hand):
        super().__init__("jam-from", inner.n_actions)
        self.inner = inner
        self.from_hand = from_hand

    def logits(self, contexts):
        return self.inner.logits(contexts)

    def policy(self, contexts):
        probs = np.asarray(self.inner.policy(contexts), dtype=np.float64)
        for row, ctx in enumerate(contexts):
            if int(ctx.record.spec.meta["hand"]) >= self.from_hand:
                legal = np.flatnonzero(np.asarray(ctx.legal_mask, dtype=bool))
                probs[row] = 0.0
                probs[row, legal[-1]] = 1.0
        return probs


def _networks(n_members, seed=0):
    torch.manual_seed(seed)
    embed_net = OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS,
                                     n_members).eval()
    torch.manual_seed(seed + 1)
    agent_net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()
    return embed_net, agent_net


def _hero_factory(agent_net, wrap=None):
    def make(observer_pos, slot_of_seat, embeddings):
        member = AgentPoolMember(agent_net, embeddings, slot_of_seat,
                                 MAX_PLAYERS, N_ACTIONS, observer_pos, "cpu")
        return member if wrap is None else wrap(member)
    return make


def _run(tmp_path, cfg=None, wrap=None, seed=0, hero=None):
    """One `generate_labels` call. Returns (manifest, labels)."""
    cfg = cfg or _cfg()
    pool = make_pool(seed=seed)
    embed_net, agent_net = _networks(len(pool), seed=seed)
    driver = LockstepDriver(pool, N_ACTIONS)
    sampler = _UniformSampler(len(pool), seed=cfg["seed"])
    factory = hero or _hero_factory(agent_net, wrap=wrap)
    manifest = generate_labels(driver, pool, sampler, embed_net, factory, cfg,
                               str(tmp_path), log=lambda _m: None)
    labels = [lab for path in manifest["shards"] for lab in load_shard(path)]
    return manifest, labels


# ------------------------------------------------------- 1: the move itself


def test_the_session_machinery_is_shared_and_not_copied():
    """D4: G1 and label generation must run the *same* sessions.

    A behavioural test cannot see the difference between a shared function and
    an identical copy — and the whole risk D4 addresses is the copy drifting
    later. This is the guard `CLAUDE.md` §4 allows as the exception.
    """
    for name in ("Session", "build_sessions", "play", "raise_sizes_from"):
        assert getattr(gates.g1, name) is getattr(env.session, name)


# --------------------------------------------------------- 2: the toy run


def test_a_toy_run_produces_usable_labels(tmp_path):
    manifest, labels = _run(tmp_path)

    assert manifest["n_sessions"] == 4
    assert manifest["n_hands"] == 24
    assert manifest["n_labels"] == len(labels) > 0
    assert manifest["stats"].n_rollouts > 0
    assert 0.0 <= manifest["stats"].collision_rate <= 1.0
    assert len(manifest["shards"]) == max(1, -(-len(labels) // 8))

    for lab in labels:
        legal = np.asarray(lab["legal"], dtype=bool)
        assert legal.any()
        # The mask is the environment's, carried on the token the agent read.
        assert np.array_equal(legal, lab["tokens"].legal[-1])
        assert np.isfinite(lab["q"][legal]).all()
        assert np.isnan(lab["q"][~legal]).all()
        target = policy_target(lab["q"], legal, lab["pot_bb"],
                               lab["facing_bet_bb"], temperature=0.5)
        assert target.sum() == pytest.approx(1.0, abs=1e-12)
        assert not target[~legal].any()
        assert lab["embeddings"].shape == (MAX_PLAYERS, NET_CFG["d_emb"])
        assert 2 <= lab["num_players"] <= 9
        assert 10 <= lab["stack_bb"] <= 300


def test_every_label_is_a_decision_hero_actually_took(tmp_path):
    """The labelled seat is hero's slot, and every hero decision is labelled."""
    cfg = _cfg()
    manifest, labels = _run(tmp_path, cfg=cfg)

    rng = np.random.default_rng(cfg["seed"])
    sessions = build_sessions(rng, list(range(11)), GAME, cfg["n_sessions"],
                              cfg["hands_per_session"],
                              seed_base=cfg["seed"] * 1_000_000, tag="labels")
    for lab in labels:
        s = sessions[lab["session"]]
        assert lab["num_players"] == s.num_players
        assert lab["stack_bb"] == s.stack_bb
        assert lab["hero_seat"] == s.seat_of_slot(0, lab["hand"])
        # slot 0 is the observer, in G1 and here alike
        assert int(lab["tokens"].slot[-1]) == 0

    seen = {(lab["session"], lab["hand"], lab["decision"]) for lab in labels}
    assert len(seen) == len(labels), "a decision was labelled twice"


def test_the_iteration_zero_hero_is_an_ordinary_pool_member(tmp_path):
    """§7.1: at iteration 0 a pool member sits in hero's seat, not the agent."""
    pool = make_pool(seed=0)

    def make(_observer_pos, _slot_of_seat, _embeddings):
        return pool[0]

    manifest, labels = _run(tmp_path, hero=make)
    assert manifest["n_labels"] == len(labels) > 0
    assert all(np.isfinite(lab["q"][lab["legal"]]).all() for lab in labels)


# ------------------------------------- 3: no future leak into the embedding


def test_the_embedding_of_a_block_ignores_every_later_hand(tmp_path):
    """§5.5, §9: the vectors are fitted from the hands before the refresh.

    Both runs play hands 0–2 identically and hands 3–5 completely differently
    (hero jams from hand 3). The vectors in force in block 1 were fitted over
    block 0 only, so they must come out bit-identical anyway.
    """
    cfg = _cfg(R=3, hands=6)
    _m_a, labels_a = _run(tmp_path / "a", cfg=cfg)
    _m_b, labels_b = _run(tmp_path / "b", cfg=cfg,
                          wrap=lambda inner: _JamFrom(inner, from_hand=3))

    def vectors_by_block(labels):
        out = {}
        for lab in labels:
            key = (lab["session"], lab["hand"] // 3)
            if key in out:
                assert np.array_equal(out[key], lab["embeddings"])
            out[key] = lab["embeddings"]
        return out

    va, vb = vectors_by_block(labels_a), vectors_by_block(labels_b)
    block0 = [k for k in va if k[1] == 0 and k in vb]
    block1 = [k for k in va if k[1] == 1 and k in vb]
    assert block1, "no labelled hero decision in the second block"

    for key in block0 + block1:
        assert np.array_equal(va[key], vb[key]), (
            f"the vectors of block {key[1]} moved when later hands changed")
    # Block 1's vectors are a real fit, not the zero cold start — otherwise the
    # comparison above would hold for a reason that has nothing to do with §5.5.
    assert any(np.abs(va[k]).max() > 0 for k in block1)
    assert all(not np.abs(va[k]).any() for k in block0), "block 0 is cold"

    # ... and the intervention did change the hands it was supposed to change.
    tail_a = {(l["session"], l["hand"], l["decision"]) for l in labels_a
              if l["hand"] >= 3}
    tail_b = {(l["session"], l["hand"], l["decision"]) for l in labels_b
              if l["hand"] >= 3}
    assert tail_a != tail_b


def test_the_first_block_is_the_cold_start(tmp_path):
    """§5.5: no history, so the vector is zero — and hero acted on that."""
    _manifest, labels = _run(tmp_path)
    cold = [lab for lab in labels if lab["hand"] < 3]
    assert cold
    assert all(not lab["embeddings"].any() for lab in cold)


# ------------------------------------------- 4: parity of the stored prefix


def test_the_stored_prefix_stops_at_the_labelled_decision(tmp_path):
    """§9: no decision after the one being labelled is in the observation."""
    _manifest, labels = _run(tmp_path)
    for lab in labels:
        tok = lab["tokens"]
        assert len(tok) == lab["decision"] + 1
        assert (tok.token_type == TOKEN_DECISION).all(), "no showdown token"
        assert int(tok.action[-1]) == -1, "the pending decision has no action"
        assert (tok.action[:-1] >= 0).all()
        assert np.array_equal(tok.decision_idx, np.arange(len(tok)))


def test_the_stored_prefix_shows_only_heros_cards(tmp_path):
    """§9: the observer's own hole cards, every other seat masked."""
    cfg = _cfg()
    _manifest, labels = _run(tmp_path, cfg=cfg)
    rng = np.random.default_rng(cfg["seed"])
    sessions = build_sessions(rng, list(range(11)), GAME, cfg["n_sessions"],
                              cfg["hands_per_session"],
                              seed_base=cfg["seed"] * 1_000_000, tag="labels")

    checked = 0
    for lab in labels:
        tok, s = lab["tokens"], sessions[lab["session"]]
        assert (tok.num_players == s.num_players).all()
        own = tok.acting_pos == lab["hero_seat"]
        assert own[-1], "the labelled decision is hero's own"
        # Cards appear on hero's own tokens and nowhere else — one pair, the
        # same pair every time, and the unknown token everywhere else.
        assert (tok.cards[~own, 5:] == UNKNOWN_CARD).all()
        hole = tok.cards[own, 5:]
        assert (hole != UNKNOWN_CARD).all()
        assert (hole == hole[0]).all()
        assert len(set(int(c) for c in hole[0])) == 2
        # Hero's own two cards can never turn up among the board slots.
        assert not np.isin(tok.cards[:, :5], hole[0]).any()
        checked += 1
    assert checked > 0


def test_the_board_in_the_prefix_is_never_ahead_of_the_street(tmp_path):
    _manifest, labels = _run(tmp_path)
    for lab in labels:
        board = lab["tokens"].cards[:, :5]
        known = (board != UNKNOWN_CARD).sum(axis=1)
        assert set(known.tolist()) <= {0, 3, 4, 5}
        # The street never goes backwards inside a hand.
        assert (np.diff(known) >= 0).all()


# ------------------------------------------------------- 5: reproducibility


def test_the_same_seed_writes_byte_identical_shards(tmp_path):
    a, _labels_a = _run(tmp_path / "a")
    b, _labels_b = _run(tmp_path / "b")
    assert [p.split("/")[-1] for p in a["shards"]] == \
           [p.split("/")[-1] for p in b["shards"]]
    assert a["n_labels"] == b["n_labels"] > 0
    for pa, pb in zip(a["shards"], b["shards"]):
        assert open(pa, "rb").read() == open(pb, "rb").read()


def test_a_different_seed_writes_different_labels(tmp_path):
    a, labels_a = _run(tmp_path / "a", cfg=_cfg(seed=3))
    b, labels_b = _run(tmp_path / "b", cfg=_cfg(seed=4))
    assert a["n_labels"] > 0 and b["n_labels"] > 0
    assert ([lab["stack_bb"] for lab in labels_a]
            != [lab["stack_bb"] for lab in labels_b])


def test_a_shard_round_trips(tmp_path):
    _manifest, labels = _run(tmp_path)
    reloaded = load_shard(str(tmp_path / "shard_0000.npz"))
    assert len(reloaded) == min(8, len(labels))
    for got, want in zip(reloaded, labels):
        assert np.array_equal(got["q"], want["q"], equal_nan=True)
        assert np.array_equal(got["legal"], want["legal"])
        assert np.array_equal(got["tokens"].cards, want["tokens"].cards)
        assert np.array_equal(got["tokens"].scalars, want["tokens"].scalars)
        assert np.array_equal(got["embeddings"], want["embeddings"])
        assert got["pot_bb"] == want["pot_bb"]


# --------------------------------------------------------- 6: the sampling


def test_the_table_configuration_is_uniform_over_the_whole_space():
    """`CLAUDE.md` §1: 2–9 players and 10–300 BB, nothing weighted.

    The exact multiset of a fixed seed, not a statistical property — a test that
    accepts "close enough to uniform" accepts a distribution that is quietly
    not.
    """
    sessions = build_sessions(np.random.default_rng(7), list(range(11)), GAME,
                              n_sessions=400, hands_per_session=1,
                              seed_base=0, tag="t")
    sizes = [s.num_players for s in sessions]
    stacks = [s.stack_bb for s in sessions]
    counts = {p: sizes.count(p) for p in range(2, 10)}
    assert counts == {2: 49, 3: 49, 4: 64, 5: 51, 6: 38, 7: 45, 8: 56,
                      9: 48}
    assert sum(counts.values()) == 400
    assert min(stacks) == 10 and max(stacks) == 299
    assert len(set(stacks)) == 216


def test_hero_is_slot_zero_and_the_sampler_supplies_the_opponents(tmp_path):
    """Hero is not drawn from the pool, so the sampler is asked for the rest."""
    asked = []

    class _Spy(_UniformSampler):
        def sample_table(self, k):
            asked.append(k)
            return super().sample_table(k)

    cfg = _cfg()
    pool = make_pool(seed=0)
    embed_net, agent_net = _networks(len(pool))
    manifest = generate_labels(
        LockstepDriver(pool, N_ACTIONS), pool, _Spy(len(pool), cfg["seed"]),
        embed_net, _hero_factory(agent_net), cfg, str(tmp_path),
        log=lambda _m: None)

    rng = np.random.default_rng(cfg["seed"])
    sessions = build_sessions(rng, list(range(len(pool))), GAME,
                              cfg["n_sessions"], cfg["hands_per_session"],
                              seed_base=cfg["seed"] * 1_000_000, tag="labels")
    assert asked == [s.num_players - 1 for s in sessions]
    assert manifest["n_labels"] > 0


# ------------------------------------------- 7: resuming a crashed phase


def test_labelling_resumes_at_the_shard_boundary_and_loses_only_a_shard(
        tmp_path, monkeypatch):
    """The phase is budgeted in days (§13); a crash must not cost all of it.

    The interrupted run is stopped in the middle of its second shard. What comes
    back is the *same* label set, shard for shard and byte for byte — which is
    the only version of this property worth having, because a resume that merely
    produced *a* label set would hide a splice between two different ones.
    """
    cfg = _cfg(n_sessions=4, hands=6)
    whole, labels = _run(tmp_path / "whole", cfg=cfg)
    assert len(whole["shards"]) >= 3, "the toy run must span several shards"

    calls = {"n": 0}
    real = train.generate.action_values_batch

    def crash_after(requests, *args, **kwargs):
        calls["n"] += len(requests)
        if calls["n"] > 12:
            raise RuntimeError("the box went away")
        return real(requests, *args, **kwargs)

    monkeypatch.setattr(train.generate, "action_values_batch", crash_after)
    with pytest.raises(RuntimeError, match="the box went away"):
        _run(tmp_path / "part", cfg=cfg)
    monkeypatch.setattr(train.generate, "action_values_batch", real)

    part_dir = tmp_path / "part"
    survived = sorted(p.name for p in part_dir.glob("shard_*.npz"))
    assert survived, "not even one shard survived the crash"
    assert len(survived) < len(whole["shards"]), "nothing was left to resume"
    progress = json.loads((part_dir / "progress.json").read_text())
    assert progress["todo_done"] > 0
    assert len(progress["shards"]) == len(survived)

    resumed, resumed_labels = _run(part_dir, cfg=cfg)
    assert resumed["n_labels"] == whole["n_labels"]
    assert resumed["n_dropped"] == whole["n_dropped"]
    assert len(resumed["shards"]) == len(whole["shards"])
    for a, b in zip(sorted(whole["shards"]), sorted(resumed["shards"])):
        assert open(a, "rb").read() == open(b, "rb").read(), (a, b)
    for a, b in zip(labels, resumed_labels):
        assert np.array_equal(np.nan_to_num(a["q"], nan=-1.0),
                              np.nan_to_num(b["q"], nan=-1.0))
        assert (a["session"], a["hand"], a["decision"]) == (
            b["session"], b["hand"], b["decision"])


def test_a_progress_file_from_different_sessions_is_refused(tmp_path):
    """Resuming into a directory whose labels came from another run is a splice."""
    _manifest, _labels = _run(tmp_path, cfg=_cfg(n_sessions=4, hands=6))
    with pytest.raises(AssertionError, match="not the same sessions"):
        _run(tmp_path, cfg=_cfg(n_sessions=3, hands=6))
