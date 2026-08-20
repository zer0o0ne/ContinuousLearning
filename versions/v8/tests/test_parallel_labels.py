"""Labelling across processes (`oracle/parallel.py`).

Toy scale, two workers, CPU only — the battery has 30 minutes for everything
(`CLAUDE.md` §4) and none of the properties here need size. What they do need is
to be the properties that fail *silently* if the parallel path is wrong:

* the labels are the labels one process computes, in the same order — a shard
  whose rows moved is a different held-out split, and nothing downstream would
  say so;
* the barrier terminates: every worker's last label comes back before the
  phase reports itself finished;
* a batch merged from several workers is the batch the model would have been
  given had those rows arrived together, so batching across processes is not
  quietly a different computation;
* a member kind the mirror cannot rebuild is refused loudly rather than
  silently seated as something else.

What is *not* tested here is throughput, which is the whole point of the
module and is a property of the Spark (`CLAUDE.md` §3).
"""

import numpy as np
import pytest
import torch

from agent.policy import AgentPoolMember
from env.driver import LockstepDriver
from nets.agent_net import AgentNet
from nets.features import collate
from oracle.parallel import merge_tokens, merge_v7, mirror_spec
from pool.base import PoolMember
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import StyleParams
from pool.v7_member import V7NetworkMember
from tests.g1_fixtures import MAX_PLAYERS, N_ACTIONS, NET_CFG, make_pool
from tests.test_label_generation import _cfg, _networks, _run
from tests.test_v7_pool_member import V7_CONFIG
from train.generate import load_shard
from vendor.v7.agent import V7Agent, n_actions_from_config
from vendor.v7.events import build_v7_events
from vendor.v7.perception.perception import extract_event_tensors


def _parallel_cfg(n_workers, **kwargs):
    cfg = _cfg(**kwargs)
    cfg["oracle"]["n_workers"] = n_workers
    return cfg


class _Scripted(PoolMember):
    """A member with no network at all: the same logits every time.

    The equality test below has to compare bytes, and a networked member cannot
    give that guarantee across a change of batch composition — a matmul is free
    to reduce in a different order for a different batch size, one logit moves
    in its last bits, an inverse-CDF draw lands the other side of a boundary and
    the rollout goes somewhere else entirely. That is a property of floating
    point, not of this module, so the byte-for-byte test uses arithmetic that
    has no such freedom and the networks are checked separately.
    """

    def __init__(self, name, n_actions, style=None):
        super().__init__(name, n_actions, style)
        self.table = np.linspace(-1.0, 1.0, n_actions)

    def logits(self, contexts):
        out = np.tile(self.table, (len(contexts), 1))
        out[:, 1] += np.array([0.3 * (c.acting_pos % 3) for c in contexts])
        return out


def _scripted_hero():
    """Hero as a fixed, network-free member — see `_Scripted` for why.

    A degenerate strategy rather than `_Scripted` itself because hero has to be
    a member the mirror can rebuild inside a worker; the arithmetic is numpy
    either way, which is the part that makes the comparison exact.
    """
    member = DEGENERATE_STRATEGIES["nit"]("hero", N_ACTIONS,
                                          StyleParams.identity())
    return lambda observer_pos, slot_of_seat, embeddings: member


# --------------------------------------------------------------- equivalence


def _labels(tmp_path, n_workers, seed=0):
    cfg = _parallel_cfg(n_workers)
    _manifest, labels = _run(tmp_path, cfg=cfg, seed=seed,
                             hero=_scripted_hero())
    return labels


def test_two_workers_produce_the_labels_one_process_produces(tmp_path_factory):
    """The property the whole module rests on: who computes a label, and in
    what order, cannot change what it is."""
    one = _labels(tmp_path_factory.mktemp("seq"), 0)
    two = _labels(tmp_path_factory.mktemp("par"), 2)

    assert len(two) == len(one) > 0
    for a, b in zip(one, two):
        for key in ("session", "hand", "decision", "num_players", "stack_bb",
                    "hero_seat"):
            assert a[key] == b[key], f"{key} moved between the two paths"
        assert np.array_equal(a["legal"], b["legal"])
        assert np.array_equal(np.nan_to_num(a["q"], nan=-999.0),
                              np.nan_to_num(b["q"], nan=-999.0))
        assert np.array_equal(a["tokens"].cards, b["tokens"].cards)
        assert np.allclose(a["embeddings"], b["embeddings"])


def test_the_shards_a_parallel_run_writes_are_the_same_shards(tmp_path_factory):
    """Not just the labels — their partition into files, because `labels_per_shard`
    and the order together decide which rows `split_heldout` withholds."""
    out_one = tmp_path_factory.mktemp("seq2")
    out_two = tmp_path_factory.mktemp("par2")
    m1, _ = _run(out_one, cfg=_parallel_cfg(0), hero=_scripted_hero())
    m2, _ = _run(out_two, cfg=_parallel_cfg(3), hero=_scripted_hero())

    assert m1["n_labels"] == m2["n_labels"]
    assert m1["n_dropped"] == m2["n_dropped"]
    assert [len(load_shard(p)) for p in m1["shards"]] == \
           [len(load_shard(p)) for p in m2["shards"]]


def test_more_workers_than_sessions_still_labels_everything(tmp_path_factory):
    """A worker with an empty slice must still reach the barrier's exit — it is
    the case where a `while` over live workers is easiest to get wrong."""
    cfg = _parallel_cfg(6, n_sessions=2, hands=4)
    _m, labels = _run(tmp_path_factory.mktemp("many"), cfg=cfg,
                      hero=_scripted_hero())
    _m2, expected = _run(tmp_path_factory.mktemp("many_seq"),
                         cfg=_parallel_cfg(0, n_sessions=2, hands=4),
                         hero=_scripted_hero())
    assert len(labels) == len(expected) > 0


# ------------------------------------------------- the networks, through IPC


def test_a_v7_member_and_the_agent_are_labelled_through_the_server(
        tmp_path_factory):
    """The path that actually uses the proxies: a v7 checkpoint in the pool and
    the agent in hero's seat, every forward taken in the parent process."""
    v7 = V7Agent(V7_CONFIG, log=lambda _m: None).eval()
    pool = [V7NetworkMember("v7", n_actions_from_config(V7_CONFIG), v7)]
    pool += make_pool()

    torch.manual_seed(5)
    agent_net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()

    def hero(observer_pos, slot_of_seat, embeddings):
        return AgentPoolMember(agent_net, embeddings, slot_of_seat,
                               MAX_PLAYERS, N_ACTIONS, observer_pos, "cpu")

    out = tmp_path_factory.mktemp("nets")
    cfg = _parallel_cfg(2, n_sessions=2, hands=4)
    manifest, labels = _run_with_pool(out, cfg, pool, hero)

    assert labels, "nothing was labelled"
    assert manifest["n_labels"] == len(labels)
    for lab in labels:
        assert np.isfinite(lab["q"][lab["legal"]]).all()


def _run_with_pool(out, cfg, pool, hero):
    """`_run` with a caller-supplied pool — `tests.test_label_generation`'s
    helper builds its own, and this test needs a v7 member in it."""
    from train.generate import generate_labels
    from tests.test_label_generation import _UniformSampler

    embed_net, _agent = _networks(len(pool), seed=0)
    driver = LockstepDriver(pool, N_ACTIONS)
    sampler = _UniformSampler(len(pool), seed=cfg["seed"])
    manifest = generate_labels(driver, pool, sampler, embed_net, hero, cfg,
                               str(out), log=lambda _m: None)
    return manifest, [lab for p in manifest["shards"] for lab in load_shard(p)]


# ------------------------------------------------------------ batch merging


def _v7_payloads(n_seqs):
    """Two v7 payloads out of one played record, as two workers would send."""
    member = V7NetworkMember("v7", n_actions_from_config(V7_CONFIG),
                             V7Agent(V7_CONFIG, log=lambda _m: None).eval())
    from tests.g1_fixtures import BIG_BLIND, SMALL_BLIND, make_specs, play
    record = play([member], make_specs(seed=77, n_hands=1, n_members=1,
                                       num_players=4))[0]
    seqs = [build_v7_events(record.snapshots, record.deck,
                            dec["acting_pos"], record.num_players, BIG_BLIND,
                            SMALL_BLIND, N_ACTIONS, up_to=dec["snap_idx"])
            for dec in record.decisions[:n_seqs]]
    assert len(seqs) >= 2, "the fixture hand must have at least two decisions"
    return member.agent, seqs


def test_a_merged_v7_batch_is_the_batch_the_model_would_have_been_given():
    agent, seqs = _v7_payloads(4)
    cut = len(seqs) // 2
    parts = [extract_event_tensors(seqs[:cut], agent.max_players),
             extract_event_tensors(seqs[cut:], agent.max_players)]

    merged = merge_v7(parts)
    together = extract_event_tensors(seqs, agent.max_players)
    for key in ("card_ids", "hero_pos", "acting_pos", "num_players", "scalars",
                "bets", "stacks", "actions", "batch_idx", "event_idx"):
        assert torch.equal(merged[key], together[key]), key
    assert merged["seq_lengths"] == together["seq_lengths"]
    assert merged["B"] == together["B"]
    assert merged["max_events"] == together["max_events"]


def test_a_merged_v7_batch_gives_each_row_its_own_answer():
    """The rows must come back in the order they went in — a scatter bug here
    hands one worker another worker's policy and nothing ever complains."""
    agent, seqs = _v7_payloads(4)
    one_at_a_time = torch.cat([agent.action_logits([s]) for s in seqs], dim=0)
    merged = merge_v7([extract_event_tensors([s], agent.max_players)
                       for s in seqs])
    out, _enc, mask = agent.perception.forward_batch(
        None, device="cpu", skip_memory=True, skip_opponent_emb=True,
        precomputed=merged)
    batched = agent.action_head(out, mask=mask)
    assert torch.allclose(batched, one_at_a_time, atol=1e-4)


def test_merged_token_batches_pad_the_way_collate_pads():
    from tests.g1_fixtures import contexts_from, make_specs, play
    from nets.features import hand_tokens

    pool = make_pool()
    records = play(pool, make_specs(seed=81, n_hands=4, n_members=len(pool),
                                    num_players=3))
    hands = []
    for record in records:
        hands.append(hand_tokens(record, observer_pos=0,
                                 slot_of_seat=[0, 1, 2],
                                 max_players=MAX_PLAYERS,
                                 n_actions=N_ACTIONS))
    hands = [h for h in hands if len(h) > 0]
    assert len(hands) >= 3
    assert len({len(h) for h in hands}) > 1, (
        "the fixture must mix hand lengths — padding is what is under test")

    d_emb = 4
    parts = []
    for group in ([hands[0]], hands[1:]):
        batch = collate(group)
        emb = torch.zeros(*batch["mask"].shape, d_emb)
        parts.append((batch, emb))
    merged, emb = merge_tokens(parts)
    together = collate(hands)

    for key, value in together.items():
        assert torch.equal(merged[key], value), key
    assert emb.shape == (len(hands), together["mask"].shape[1], d_emb)


# ---------------------------------------------------------------- the mirror


def test_a_member_kind_the_mirror_cannot_rebuild_is_refused():
    with pytest.raises(TypeError, match="no mirror"):
        mirror_spec(_Scripted("scripted", N_ACTIONS), lambda net: "net0")


def test_a_finished_phase_resumes_into_a_no_op(tmp_path):
    """Resume is per-phase and the workers are handed only the positions after
    it. The boundary case — nothing left to do — must not hang on a barrier
    waiting for work that was never handed out."""
    cfg = _parallel_cfg(2, n_sessions=2, hands=4)
    first, labels = _run(tmp_path, cfg=cfg, hero=_scripted_hero())
    again, again_labels = _run(tmp_path, cfg=cfg, hero=_scripted_hero())

    assert again["n_labels"] == first["n_labels"] == len(labels)
    assert again["shards"] == first["shards"]
    assert len(again_labels) == len(labels)
