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
import os

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
            # The seed layout lays the phases' ranges end to end, so the labels
            # phase's own base depends on how many hands the corpus takes
            # (`env/session.hand_seed_bases`). A fixture that named only one
            # phase would describe half an iteration.
            "corpus_sessions": 3, "corpus_hands_per_session": 5,
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


def _run(tmp_path, cfg=None, wrap=None, seed=0, hero=None, pool=None,
         sampler=None):
    """One `generate_labels` call. Returns (manifest, labels)."""
    cfg = cfg or _cfg()
    pool = make_pool(seed=seed) if pool is None else pool
    embed_net, agent_net = _networks(len(pool), seed=seed)
    driver = LockstepDriver(pool, N_ACTIONS)
    sampler = sampler or _UniformSampler(len(pool), seed=cfg["seed"])
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

    # A shard is flushed once it is full *and* the hand it is in is finished,
    # so it holds at least `labels_per_shard` labels rather than exactly that
    # many, and no hand has its labels split across two of them. That is what
    # makes `progress.json`'s hand count a place a later call can start from:
    # the hands before it are labelled whole, in shards that are complete.
    shards = [load_shard(path) for path in manifest["shards"]]
    assert sum(len(sh) for sh in shards) == len(labels)
    assert all(len(sh) >= 8 for sh in shards[:-1])
    hands = [{(lab["session"], lab["hand"]) for lab in sh} for sh in shards]
    for i, earlier in enumerate(hands):
        for later in hands[i + 1:]:
            assert earlier.isdisjoint(later), "a hand was split across shards"

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
    assert 8 <= len(reloaded) <= len(labels)
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


def _played_hands(monkeypatch):
    """Every hand the phase deals from here on, as `(session, hand)`.

    The list fills as the phase plays; a resumed call is supposed to leave the
    hands it has already labelled out of it entirely.
    """
    seen = []
    real = train.generate.play

    def spy(driver, sessions, batch_size, log, tag, bar=True):
        seen.extend((int(s.idx), int(spec.meta["hand"]))
                    for s in sessions for spec in s.specs)
        return real(driver, sessions, batch_size, log, tag, bar=bar)

    monkeypatch.setattr(train.generate, "play", spy)
    return seen


def test_labelling_resumes_at_the_hand_boundary_and_loses_only_a_shard(
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
    assert 0 < progress["hands_done"] < progress["n_hands"]
    assert len(progress["shards"]) == len(survived)

    resumed, resumed_labels = _run(part_dir, cfg=cfg)
    assert resumed["n_labels"] == whole["n_labels"]
    assert resumed["n_dropped"] == whole["n_dropped"]
    assert resumed["n_hands"] == whole["n_hands"]
    assert resumed["results"] == whole["results"]
    assert len(resumed["shards"]) == len(whole["shards"])
    for a, b in zip(sorted(whole["shards"]), sorted(resumed["shards"])):
        assert open(a, "rb").read() == open(b, "rb").read(), (a, b)
    for a, b in zip(labels, resumed_labels):
        assert np.array_equal(np.nan_to_num(a["q"], nan=-1.0),
                              np.nan_to_num(b["q"], nan=-1.0))
        assert (a["session"], a["hand"], a["decision"]) == (
            b["session"], b["hand"], b["decision"])


def test_a_resumed_call_plays_the_unlabelled_hands_and_no_others(
        tmp_path, monkeypatch):
    """The point of keeping the corpus: a labelled hand is never dealt again.

    Not a saving — a correctness property. Re-playing a labelled hand means
    re-deriving it from the network and the fits, which is not bit-reproducible
    across processes on a GPU, and one flipped action makes the shards on disk
    the labels of a corpus that no longer exists.
    """
    cfg = _cfg(n_sessions=4, hands=6)
    out_dir = tmp_path / "part"
    _crashed_run(out_dir, monkeypatch, cfg)
    done = json.loads((out_dir / "progress.json").read_text())
    frontier = int(done["hands_done"])
    assert 0 < frontier < done["n_hands"], "the crash left nothing to resume"

    played = _played_hands(monkeypatch)
    _run(out_dir, cfg=cfg)

    assert len(played) == len(set(played)), "a hand was dealt twice"
    assert sorted(played) == sorted(
        (i, h) for i in range(4) for h in range(6) if i * 6 + h >= frontier)


def test_a_finished_directory_is_handed_back_without_playing_anything(
        tmp_path, monkeypatch):
    """The degenerate resume: every hand is labelled, so there is nothing to do.

    It has to come back as the same manifest all the same — the results hero
    owes §4.4's sampler and the hand count are read from the call that played
    the corpus, because this call has no records to compute them from.
    """
    cfg = _cfg(n_sessions=4, hands=6)
    out_dir = tmp_path / "labels"
    whole, labels = _run(out_dir, cfg=cfg)

    played = _played_hands(monkeypatch)
    again, again_labels = _run(out_dir, cfg=cfg)

    assert played == []
    assert again["shards"] == whole["shards"]
    assert again["n_labels"] == whole["n_labels"]
    assert again["n_dropped"] == whole["n_dropped"]
    assert again["n_hands"] == whole["n_hands"]
    assert again["results"] == whole["results"]
    for a, b in zip(labels, again_labels):
        assert (a["session"], a["hand"], a["decision"]) == (
            b["session"], b["hand"], b["decision"])
        assert np.array_equal(np.nan_to_num(a["q"], nan=-1.0),
                              np.nan_to_num(b["q"], nan=-1.0))


def test_a_resume_holds_when_the_corpus_no_longer_replays_the_same_way(
        tmp_path, monkeypatch):
    """The failure this design exists for, forced.

    Hero plays the second call *differently* — the stand-in for a GPU that does
    not reduce in the same order twice, or a fit that lands a hair off where it
    landed before. Every hand from the frontier on therefore comes out unlike
    the one the crashed call played. Nothing about the labels already on disk
    may move: they are a prefix of the finished set, byte for byte, no decision
    is labelled twice, and the hands stay in order across the join.
    """
    cfg = _cfg(n_sessions=4, hands=6)
    out_dir = tmp_path / "part"
    _crashed_run(out_dir, monkeypatch, cfg)
    before = [lab for path in json.loads(
        (out_dir / "progress.json").read_text())["shards"]
        for lab in load_shard(path)]
    assert before, "the crash left nothing to resume"

    # Jamming from the first hand: not one replayed hand is the hand the
    # crashed call played.
    manifest, labels = _run(out_dir, cfg=cfg,
                            wrap=lambda member: _JamFrom(member, 0))

    keys = [(lab["session"], lab["hand"], lab["decision"]) for lab in labels]
    assert len(keys) == len(set(keys)), "a decision was labelled twice"
    assert keys == sorted(keys), "the labels are out of order across the join"
    assert manifest["n_labels"] == len(labels) == len(keys)
    for a, b in zip(before, labels):
        assert (a["session"], a["hand"], a["decision"]) == (
            b["session"], b["hand"], b["decision"])
        assert np.array_equal(np.nan_to_num(a["q"], nan=-1.0),
                              np.nan_to_num(b["q"], nan=-1.0))
        assert np.array_equal(a["tokens"].cards, b["tokens"].cards)

    tail = labels[len(before):]
    assert tail, "the resumed call labelled nothing"
    assert min(lab["session"] * 6 + lab["hand"] for lab in tail) >= max(
        lab["session"] * 6 + lab["hand"] for lab in before), (
        "a hand was labelled on both sides of the join")


def _assert_relabelled_from_scratch(out_dir, cfg, fresh_dir, stale_before):
    """The phase set the directory aside and produced a from-scratch label set.

    Two halves, and both matter: the old directory is still there under
    `.stale` (its shards cost hours and are not this code's to delete), and what
    the phase wrote is bit-for-bit what a run into an empty directory writes —
    nothing of the old set was spliced into it.
    """
    aside = out_dir.parent / (out_dir.name + ".stale")
    assert json.loads((aside / "progress.json").read_text()) == stale_before

    manifest = json.loads((out_dir / "progress.json").read_text())
    assert manifest["hands_done"] == manifest["n_hands"]
    labels = [lab for path in sorted(manifest["shards"])
              for lab in load_shard(path)]
    fresh, fresh_labels = _run(fresh_dir, cfg=cfg)
    assert manifest["n_labels"] == fresh["n_labels"]
    assert len(manifest["shards"]) == len(fresh["shards"])
    for a, b in zip(labels, fresh_labels):
        assert (a["session"], a["hand"], a["decision"]) == (
            b["session"], b["hand"], b["decision"])
        assert np.array_equal(np.nan_to_num(a["q"], nan=-1.0),
                              np.nan_to_num(b["q"], nan=-1.0))


def test_a_directory_from_different_sessions_is_set_aside_and_relabelled(
        tmp_path):
    """Resuming into a directory whose labels came from another run would be a
    splice. The sessions are named in `play.json`, so it is caught before a hand
    is played: the old directory moves aside and this call labels its own."""
    out_dir = tmp_path / "labels"
    _run(out_dir, cfg=_cfg(n_sessions=4, hands=6))
    stale = json.loads((out_dir / "progress.json").read_text())

    cfg = _cfg(n_sessions=3, hands=6)
    _run(out_dir, cfg=cfg)
    _assert_relabelled_from_scratch(out_dir, cfg, tmp_path / "fresh", stale)
def test_a_resumed_run_does_not_count_the_skipped_labels_as_this_runs_work(
        tmp_path, monkeypatch):
    """The bar's rate is `(n - initial) / elapsed` (`utils.progress`).

    A resumed phase that jumped the bar forward with `bar.update(start)` instead
    counted labels an *earlier* run computed as having taken this run zero
    seconds, and reported a rate — and an ETA — inflated by exactly that ratio.
    Seen in the wild: 4.5 s/label displayed as 1.03 s/label. Behaviour cannot
    catch this, so the construction is what is pinned.
    """
    cfg = _cfg(n_sessions=4, hands=6)
    calls = {"n": 0}
    real = train.generate.action_values_batch

    def crash_after(requests, *args, **kwargs):
        calls["n"] += len(requests)
        if calls["n"] > 12:
            raise RuntimeError("the box went away")
        return real(requests, *args, **kwargs)

    monkeypatch.setattr(train.generate, "action_values_batch", crash_after)
    with pytest.raises(RuntimeError):
        _run(tmp_path, cfg=cfg)
    monkeypatch.setattr(train.generate, "action_values_batch", real)

    done = json.loads(
        (tmp_path / "progress.json").read_text())["decisions_done"]
    assert done > 0, "the crash must leave something to resume from"

    seen = {}
    real_progress = train.generate.progress

    def spy(*args, **kwargs):
        if kwargs.get("desc") == "label":
            seen.update(kwargs)
        return real_progress(*args, **kwargs)

    monkeypatch.setattr(train.generate, "progress", spy)
    _run(tmp_path, cfg=cfg)

    assert seen.get("initial") == done, (
        f"the resumed bar starts at {seen.get('initial')} of its total but "
        f"{done} labels were already on disk")


# ------------------------------------------------- the hand-seed layout (§15)


def test_the_two_phases_of_an_iteration_never_deal_the_same_hand():
    """The property the layout exists for. Two phases sharing a `HandSpec.seed`
    share the deck *and* every action draw — they are the same hand — which
    would correlate the embedding corpus with the labelled set invisibly."""
    from env.session import build_sessions, hand_seed_bases, phase_hands

    cfg = _cfg(n_sessions=7, hands=11)
    cfg["embedding_net"]["corpus_sessions"] = 5
    cfg["embedding_net"]["corpus_hands_per_session"] = 13
    bases, span = hand_seed_bases(3, phase_hands(cfg))

    seeds = {}
    for phase, n_sessions, hands in (("labels", 7, 11), ("corpus", 5, 13)):
        sessions = build_sessions(np.random.default_rng(0), list(range(9)),
                                  GAME, n_sessions, hands,
                                  seed_base=bases[phase], tag=phase)
        seeds[phase] = {spec.seed for s in sessions for spec in s.specs}
        assert len(seeds[phase]) == n_sessions * hands, "a phase repeated a seed"

    assert not (seeds["labels"] & seeds["corpus"]), "the phases share hands"
    assert span == 7 * 11 + 5 * 13, "the span is what the phases actually ask"


def test_consecutive_iterations_never_deal_the_same_hand():
    from env.session import hand_seed_bases, phase_hands

    cfg = _cfg(n_sessions=4, hands=6)
    hands = phase_hands(cfg)
    seen = set()
    for k in range(5):
        bases, span = hand_seed_bases(k, hands)
        for phase, n in hands.items():
            block = set(range(bases[phase], bases[phase] + n))
            assert not (block & seen), f"iteration {k} reuses {phase} seeds"
            seen |= block
    assert len(seen) == 5 * sum(hands.values())


def test_a_corpus_bigger_than_the_old_fixed_reservation_is_fine_now():
    """The cap this replaced refused any phase over 500 000 hands, which is a
    limit on the experiment imposed by its bookkeeping."""
    from env.session import assert_seeds_stay_distinct, hand_seed_bases

    hands = {"labels": 40_000, "corpus": 2_000_000}
    bases, span = hand_seed_bases(29, hands)
    assert span == 2_040_000
    assert bases["corpus"] == 29 * span + 40_000
    assert_seeds_stay_distinct(span, 30)


def test_a_layout_too_wide_for_the_deck_draw_is_refused():
    """`np.random.seed` takes 32 bits, so past that two hands get one deck."""
    from env.session import assert_seeds_stay_distinct

    with pytest.raises(AssertionError, match="dealt identical cards"):
        assert_seeds_stay_distinct(2 ** 30, 30)


def _crashed_run(tmp_path, monkeypatch, cfg, after=12, **kwargs):
    """A half-finished labels directory: shards on disk, `progress.json` written."""
    calls = {"n": 0}
    real = train.generate.action_values_batch

    def crash_after(requests, *args, **kwargs_):
        calls["n"] += len(requests)
        if calls["n"] > after:
            raise RuntimeError("the box went away")
        return real(requests, *args, **kwargs_)

    monkeypatch.setattr(train.generate, "action_values_batch", crash_after)
    with pytest.raises(RuntimeError):
        _run(tmp_path, cfg=cfg, **kwargs)
    monkeypatch.setattr(train.generate, "action_values_batch", real)
    return tmp_path / "progress.json"


def test_resuming_after_the_seed_layout_moved_starts_over(tmp_path, monkeypatch):
    """The phases' ranges are laid end to end, so changing how many hands the
    *corpus* takes moves where the labelled hands start. The recorded base is
    what catches it; the hands themselves would just quietly be other hands.
    Caught, it is the same outcome as any other stale directory: aside, and
    label again."""
    cfg = _cfg(n_sessions=4, hands=6)
    out_dir = tmp_path / "labels"
    path = _crashed_run(out_dir, monkeypatch, cfg)
    stale = json.loads(path.read_text())

    play_path = out_dir / "play.json"
    played = json.loads(play_path.read_text())
    played["seed_base"] = int(played["seed_base"]) + 1
    play_path.write_text(json.dumps(played))

    _run(out_dir, cfg=cfg)
    _assert_relabelled_from_scratch(out_dir, cfg, tmp_path / "fresh", stale)


def test_a_directory_whose_corpus_was_not_kept_starts_over(tmp_path,
                                                           monkeypatch):
    """The directories already on disk when the corpus started being kept, and
    any directory whose `play.json` is gone. Their hands cannot be replayed and
    their fits cannot be recovered, so their shards cannot be resumed into —
    and nothing in them says which hands they are."""
    cfg = _cfg(n_sessions=4, hands=6)
    out_dir = tmp_path / "labels"
    path = _crashed_run(out_dir, monkeypatch, cfg)
    stale = json.loads(path.read_text())

    os.remove(out_dir / "play.json")

    _run(out_dir, cfg=cfg)
    _assert_relabelled_from_scratch(out_dir, cfg, tmp_path / "fresh", stale)


def test_a_second_stale_directory_does_not_overwrite_the_first(tmp_path,
                                                               monkeypatch):
    """Set-aside names are derived and not timestamped, so two failures in the
    same place must not land on the same name — the first one's labels would be
    gone, which is the thing setting them aside was for."""
    out_dir = tmp_path / "labels"
    _run(out_dir, cfg=_cfg(n_sessions=4, hands=6))
    _run(out_dir, cfg=_cfg(n_sessions=3, hands=6))
    _run(out_dir, cfg=_cfg(n_sessions=2, hands=6))

    first = json.loads((tmp_path / "labels.stale" / "progress.json").read_text())
    second = json.loads(
        (tmp_path / "labels.stale.1" / "progress.json").read_text())
    assert first["n_hands"] != second["n_hands"]


# --------------- 8: a past agent in the pool that reads its tablemates


"""`pool_agent_vectors` (PLAN_AMORTISED_POOL.md P3).

With the switch off a past agent seated as an opponent plays at `e = 0`, which
is what every test above runs under. With it on it reads the `K = 0` vectors of
its own tablemates, on the same block boundaries as hero's fit, through the
embedding network of **its own generation** — never the one the loop is still
training, whose coordinates drift under it.

Four things must hold and each of them is silent if it does not:

* off is off — the same bytes, to the shard;
* the table a past agent acts under in block *b* is computed from the hands of
  blocks `0 … b−1`, from *its* seat, and never from a later hand;
* it is that member's own network that computes it;
* everything the resume machinery guarantees for hero's tables it guarantees for
  these too, because they decide actions the same way.
"""


def _vintage_net(n_members, seed):
    """An embedding network standing in for one past generation's."""
    torch.manual_seed(seed)
    return OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS,
                                n_members=n_members).eval()


def _past_agent(embed_net, seed=9, name="agent0"):
    from agent.policy import FrozenAgentMember
    from pool.style import StyleParams
    torch.manual_seed(seed)
    net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()
    return FrozenAgentMember(name, N_ACTIONS, net, MAX_PLAYERS, "cpu",
                             StyleParams.identity(), embed_net=embed_net)


def _pool_with_agent(embed_net, seed=0):
    """The toy pool plus one past agent, which is the last member."""
    pool = make_pool(seed=seed)
    return pool + [_past_agent(embed_net)]


class _AgentAlwaysSampler(_UniformSampler):
    """Seats the past agent at every table.

    The uniform sampler would put it at one toy table in four, and a test that
    silently exercises nothing is worse than no test.
    """

    def __init__(self, n_members, agent_idx, seed):
        super().__init__(n_members, seed)
        self.agent_idx = int(agent_idx)

    def sample_table(self, k):
        others = [m for m in super().sample_table(self.n_members - 1)
                  if m != self.agent_idx][:k - 1]
        return [self.agent_idx] + others


def _conditioned(cfg, window=None):
    cfg["embedding_net"]["pool_agent_vectors"] = "amortised"
    cfg["embedding_net"]["pool_agent_window"] = window
    return cfg


def _run_conditioned(tmp_path, cfg, pool, **kwargs):
    sampler = _AgentAlwaysSampler(len(pool), len(pool) - 1, seed=cfg["seed"])
    return _run(tmp_path, cfg=cfg, pool=pool, sampler=sampler, **kwargs)


def _stored_vectors(out_dir):
    with np.load(os.path.join(str(out_dir), "vectors.npz")) as z:
        return z["vectors"]


def _stored_play(out_dir):
    with open(os.path.join(str(out_dir), "play.json")) as fh:
        return json.load(fh)


def _shard_bytes(manifest):
    return [open(p, "rb").read() for p in manifest["shards"]]


def _capture_sessions(monkeypatch):
    """The session objects the phase played, with their records."""
    held = []
    real = train.generate._play_sessions

    def spy(driver, play_pool, sessions, *args, **kwargs):
        held.append(sessions)
        return real(driver, play_pool, sessions, *args, **kwargs)

    monkeypatch.setattr(train.generate, "_play_sessions", spy)
    return held


def test_with_the_switch_off_the_new_keys_change_nothing(tmp_path):
    """`"zero"` is D12 option (a) and is the same run, byte for byte.

    Including the window, which is read only when the switch is on: a value
    there must not reach a single hand.
    """
    pool = _pool_with_agent(_vintage_net(16, seed=21))
    absent = _cfg()
    zeroed = _cfg()
    zeroed["embedding_net"]["pool_agent_vectors"] = "zero"
    zeroed["embedding_net"]["pool_agent_window"] = 2

    m_a, _l = _run_conditioned(tmp_path / "a", absent, pool)
    m_b, _l = _run_conditioned(tmp_path / "b", zeroed, pool)

    assert _shard_bytes(m_a) == _shard_bytes(m_b), (
        "turning the switch off is not the run that had no switch")
    # One observer's tables and not nine: the default writes what it always did.
    assert _stored_vectors(tmp_path / "a").shape[2] == 1


def test_a_past_agents_table_is_the_reading_of_its_own_seat_and_network(
        tmp_path, monkeypatch):
    """The exact vectors, recomputed from the hands the phase actually played.

    `K = 0` over the session as *this slot* saw it, through the network this
    member carries, over the blocks before the one it acts in.
    """
    from env.session import Session
    from train.generate import amortised_vectors

    vintage = _vintage_net(16, seed=21)
    pool = _pool_with_agent(vintage)
    agent_idx = len(pool) - 1
    cfg = _conditioned(_cfg(R=3, hands=6))
    held = _capture_sessions(monkeypatch)
    _m, _l = _run_conditioned(tmp_path, cfg, pool)

    sessions = held[0]
    vectors = _stored_vectors(tmp_path)
    assert vectors.shape[2] == MAX_PLAYERS
    checked = 0
    for i, s in enumerate(sessions):
        slots = [j for j in range(1, s.num_players)
                 if int(s.members[j]) == agent_idx]
        assert slots, "the past agent was not seated at this table"
        for b in range(vectors.shape[1]):
            before = Session(idx=s.idx, num_players=s.num_players,
                             stack_bb=s.stack_bb, members=s.members,
                             specs=s.specs[:b * 3],
                             records=s.records[:b * 3])
            for slot in slots:
                expected = amortised_vectors(vintage, before, slot, MAX_PLAYERS,
                                             N_ACTIONS, None, "cpu")
                assert np.allclose(vectors[i][b][slot], expected, atol=1e-6), (
                    f"session {i} block {b} slot {slot}")
                checked += 1
            if b:
                assert np.abs(vectors[i][b][slots[0]]).max() > 0, (
                    "the reading is zero, so the test compares nothing")
    assert checked > 4
    # Block 0 is the cold start for every observer, hero included.
    assert not vectors[:, 0].any()
    # Nobody else's row is ever filled: a degenerate member has no vectors to
    # read and hero's own table is row 0.
    for i, s in enumerate(sessions):
        for slot in range(1, s.num_players):
            if int(s.members[slot]) != agent_idx:
                assert not vectors[i][:, slot].any()


def test_the_table_of_a_block_ignores_every_later_hand_for_a_past_agent(
        tmp_path):
    """§5.5 for the pool's own agents: hero jams from hand 3 and the tables of
    blocks 0 and 1 must come out bit-identical anyway."""
    pool = _pool_with_agent(_vintage_net(16, seed=21))
    cfg = _conditioned(_cfg(R=3, hands=6))

    _m_a, labels_a = _run_conditioned(tmp_path / "a", cfg, pool)
    _m_b, labels_b = _run_conditioned(
        tmp_path / "b", cfg, pool,
        wrap=lambda inner: _JamFrom(inner, from_hand=3))

    va, vb = _stored_vectors(tmp_path / "a"), _stored_vectors(tmp_path / "b")
    assert np.array_equal(va[:, 0], vb[:, 0]), "block 0 is not the cold start"
    assert np.array_equal(va[:, 1], vb[:, 1]), (
        "a table in force during block 1 moved when block 1's hands changed")
    assert np.abs(va[:, 1]).max() > 0
    tail_a = {(l["session"], l["hand"], l["decision"]) for l in labels_a
              if l["hand"] >= 3}
    tail_b = {(l["session"], l["hand"], l["decision"]) for l in labels_b
              if l["hand"] >= 3}
    assert tail_a != tail_b, "the intervention changed nothing"


def test_a_past_agent_reads_its_own_generation_and_not_the_loops_network(
        tmp_path):
    """Two runs differing only in the network the past agent carries.

    If the phase used the network the loop is training — the one hero fits with
    — both runs would be identical, and a frozen policy would be reading a
    description written in a basis that drifts under it.
    """
    cfg = _conditioned(_cfg(R=3, hands=6))
    m_a, _l = _run_conditioned(tmp_path / "a", cfg,
                               _pool_with_agent(_vintage_net(16, seed=21)))
    m_b, _l = _run_conditioned(tmp_path / "b", cfg,
                               _pool_with_agent(_vintage_net(16, seed=22)))

    va, vb = _stored_vectors(tmp_path / "a"), _stored_vectors(tmp_path / "b")
    assert np.array_equal(va[:, :, 0], vb[:, :, 0]), (
        "hero's own fit moved, so the two runs differ by more than the "
        "generation the past agent carries")
    assert not np.allclose(va[:, 1:], vb[:, 1:]), (
        "the past agent's tables are the same under two different networks")
    assert _shard_bytes(m_a) != _shard_bytes(m_b), (
        "the tables reached no decision, so nothing was conditioned")


def test_a_past_agent_with_no_generation_is_never_conditioned(tmp_path):
    """A member that carries no network of its own plays at `e = 0`, switch or
    no switch — there is nothing it could correctly be given."""
    from agent.policy import FrozenAgentMember
    from pool.style import StyleParams

    torch.manual_seed(9)
    net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()
    blind = FrozenAgentMember("agent0", N_ACTIONS, net, MAX_PLAYERS, "cpu",
                              StyleParams.identity())
    pool = make_pool(seed=0) + [blind]
    on = _conditioned(_cfg(R=3, hands=6))
    off = _cfg(R=3, hands=6)

    m_on, _l = _run_conditioned(tmp_path / "on", on, pool)
    m_off, _l = _run_conditioned(tmp_path / "off", off, pool)
    assert _shard_bytes(m_on) == _shard_bytes(m_off)
    assert _stored_vectors(tmp_path / "on").shape[2] == 1


def test_the_window_is_the_history_a_past_agent_may_look_at(tmp_path):
    """A one-hand window and a whole-session window are different readings.

    Refreshing every hand, the two settings agree at block 1 — one played hand
    is the whole history there — and part company at block 2, where the capped
    reading drops the older hand. That is the cost lever the corpus phase needs,
    and it has to actually cap something.
    """
    pool = _pool_with_agent(_vintage_net(16, seed=21))
    whole, short = (_conditioned(_cfg(R=1, hands=4)),
                    _conditioned(_cfg(R=1, hands=4), window=1))

    _m_a, _l = _run_conditioned(tmp_path / "a", whole, pool)
    _m_b, _l = _run_conditioned(tmp_path / "b", short, pool)
    va, vb = _stored_vectors(tmp_path / "a"), _stored_vectors(tmp_path / "b")
    assert np.allclose(va[:, 1], vb[:, 1]), (
        "one hand of history read two different ways")
    assert not np.allclose(va[:, 2], vb[:, 2]), (
        "the window capped nothing")


def test_a_directory_labelled_under_the_other_setting_is_set_aside(
        tmp_path, monkeypatch):
    """Two settings play different hands, so their labels cannot be spliced."""
    pool = _pool_with_agent(_vintage_net(16, seed=21))
    out_dir = tmp_path / "labels"
    cfg = _conditioned(_cfg(n_sessions=4, hands=6))
    sampler = _AgentAlwaysSampler(len(pool), len(pool) - 1, seed=cfg["seed"])
    _crashed_run(out_dir, monkeypatch, cfg, pool=pool, sampler=sampler)

    off = _cfg(n_sessions=4, hands=6)
    _run(out_dir, cfg=off, pool=pool,
         sampler=_AgentAlwaysSampler(len(pool), len(pool) - 1, seed=off["seed"]))
    assert (tmp_path / "labels.stale").exists(), (
        "a directory played with the pool conditioned was resumed into with it "
        "off")


def test_a_conditioned_run_resumes_into_the_run_it_would_have_been(
        tmp_path, monkeypatch):
    """The tables a past agent acted under are on disk and are read back.

    Refitting them on resume would be a different fit — the head of the session
    is not replayed — so they are persisted exactly as hero's are, and the
    resumed run must produce the same labels the uninterrupted one did.
    """
    pool = _pool_with_agent(_vintage_net(16, seed=21))
    cfg = _conditioned(_cfg(n_sessions=4, hands=6))
    whole, _l = _run_conditioned(tmp_path / "whole", cfg, pool)

    out_dir = tmp_path / "resumed"
    _crashed_run(out_dir, monkeypatch, cfg, pool=pool,
                 sampler=_AgentAlwaysSampler(len(pool), len(pool) - 1,
                                             seed=cfg["seed"]))
    dealt = _played_hands(monkeypatch)
    resumed, _l = _run_conditioned(out_dir, cfg, pool)

    assert not (tmp_path / "resumed.stale").exists(), "the resume was refused"
    assert _shard_bytes(whole) == _shard_bytes(resumed)
    assert dealt, "the resumed call played nothing at all"
    assert len(dealt) < 4 * 6, "the resumed call replayed the whole corpus"
