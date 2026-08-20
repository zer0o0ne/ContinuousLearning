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
from oracle.parallel import V7Runner, mirror_spec
from oracle.transport import Slab, token_fields, v7_fields
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
    the agent in hero's seat, every forward taken in the parent process — and
    the labels are still the labels one process computes, to the bit."""
    v7 = V7Agent(V7_CONFIG, log=lambda _m: None).eval()
    pool = [V7NetworkMember("v7", n_actions_from_config(V7_CONFIG), v7)]
    pool += make_pool()

    torch.manual_seed(5)
    agent_net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()

    def hero(observer_pos, slot_of_seat, embeddings):
        return AgentPoolMember(agent_net, embeddings, slot_of_seat,
                               MAX_PLAYERS, N_ACTIONS, observer_pos, "cpu")

    seq = _run_with_pool(tmp_path_factory.mktemp("nets_seq"),
                         _parallel_cfg(0, n_sessions=2, hands=4), pool, hero)[1]
    par = _run_with_pool(tmp_path_factory.mktemp("nets_par"),
                         _parallel_cfg(2, n_sessions=2, hands=4), pool, hero)[1]

    assert seq, "nothing was labelled"
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert (a["session"], a["hand"], a["decision"]) == \
               (b["session"], b["hand"], b["decision"])
        assert np.array_equal(a["legal"], b["legal"])
        # Bit for bit: every forward the workers asked for was taken over
        # exactly the rows one process would have handed the model, so no
        # reduction anywhere happened in a different order.
        assert np.array_equal(np.nan_to_num(a["q"], nan=-999.0),
                              np.nan_to_num(b["q"], nan=-999.0))


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


# ------------------------------------------------------------- the transport


def _v7_seqs(n_seqs):
    """A few real v7 event sequences out of one played hand."""
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


def test_a_v7_payload_survives_the_slab_unchanged():
    agent, seqs = _v7_seqs(4)
    payload = extract_event_tensors(seqs, agent.max_players)
    fields = v7_fields(agent.max_players, agent.n_actions)
    slab = Slab(64, [fields], agent.n_actions)
    cells = int(payload["card_ids"].shape[0])

    slab.pack(fields, payload, cells)
    out = slab.unpack(fields, cells, (cells,))
    for name, _cols, _dtype in fields:
        assert torch.equal(out[name], payload[name]), name


def test_a_forward_through_the_slab_is_the_forward_without_it():
    """The determinism claim in one assertion: the transport moves bytes and
    changes no number, so a worker's forward is the forward one process takes."""
    agent, seqs = _v7_seqs(4)
    direct = agent.action_logits(seqs)

    payload = extract_event_tensors(seqs, agent.max_players)
    fields = v7_fields(agent.max_players, agent.n_actions)
    slab = Slab(64, [fields], agent.n_actions)
    cells = int(payload["card_ids"].shape[0])
    slab.pack(fields, payload, cells)
    runner = V7Runner(agent, (False, "cpu", torch.float32))
    slab.meta[:int(payload["B"])].copy_(
        torch.as_tensor(payload["seq_lengths"], dtype=torch.int64))
    through = runner.run(slab, cells, (int(payload["B"]),),
                         int(payload["max_events"]))
    assert torch.equal(through, direct)


def test_a_token_batch_survives_the_slab_unchanged():
    from nets.features import hand_tokens
    from tests.g1_fixtures import make_specs, play

    pool = make_pool()
    records = play(pool, make_specs(seed=81, n_hands=4, n_members=len(pool),
                                    num_players=3))
    hands = [hand_tokens(r, observer_pos=0, slot_of_seat=[0, 1, 2],
                         max_players=MAX_PLAYERS, n_actions=N_ACTIONS)
             for r in records]
    hands = [h for h in hands if len(h) > 0]
    assert len({len(h) for h in hands}) > 1, "the fixture must mix hand lengths"

    batch = collate(hands)
    d_emb = 4
    emb = torch.arange(batch["mask"].numel() * d_emb, dtype=torch.float32)
    emb = emb.reshape(*batch["mask"].shape, d_emb)
    fields = token_fields(MAX_PLAYERS, N_ACTIONS, d_emb)
    b, t = batch["mask"].shape
    slab = Slab(max(b, 8), [fields], N_ACTIONS)

    tensors = dict(batch)
    tensors["emb"] = emb
    slab.pack(fields, tensors, b * t)
    out = slab.unpack(fields, b * t, (b, t))
    for name, _cols, _dtype in fields:
        assert torch.equal(out[name], tensors[name]), name
    # The three derived masks are rebuilt, not carried.
    assert set(batch) - set(out) == {"decision_mask", "showdown_mask",
                                     "strength_mask"}


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


# --------------------------------------------------- labels_per_batch (§7.1)


def _batched_cfg(labels_per_batch, n_workers=0, **kwargs):
    cfg = _parallel_cfg(n_workers, **kwargs)
    cfg["oracle"]["labels_per_batch"] = labels_per_batch
    return cfg


def test_labelling_several_decisions_in_one_run_changes_no_label(tmp_path_factory):
    """`labels_per_batch` is a batching knob and nothing else.

    Every rollout hand keeps its own deck and its own `_rollout_seed`, and the
    driver draws each hand's actions from a generator seeded by that alone, so
    queueing another label's hands alongside cannot move an action. With a
    network-free pool the arithmetic has no reordering freedom either, so the
    two runs agree byte for byte — which is the version of "no bias" that can
    actually be asserted.
    """
    one = _run(tmp_path_factory.mktemp("k1"), cfg=_batched_cfg(1),
               hero=_scripted_hero())[1]
    many = _run(tmp_path_factory.mktemp("k8"), cfg=_batched_cfg(8),
                hero=_scripted_hero())[1]

    assert len(many) == len(one) > 0
    for a, b in zip(one, many):
        assert (a["session"], a["hand"], a["decision"]) == \
               (b["session"], b["hand"], b["decision"])
        assert np.array_equal(a["legal"], b["legal"])
        assert np.array_equal(np.nan_to_num(a["q"], nan=-999.0),
                              np.nan_to_num(b["q"], nan=-999.0))


def test_batching_does_not_cross_a_block_boundary():
    """Hero is reseated between blocks (§5.5) and a chunk's rollouts are all
    built before any of them is played — so a chunk that straddled a boundary
    would build half its labels against the wrong seated member."""
    from train.generate import label_chunks

    R = 3
    todo = [(pos, 0, h, d) for pos, (h, d) in
            enumerate([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])]
    todo += [(pos + 6, 1, h, 0) for pos, h in enumerate([0, 1, 2])]

    chunks = list(label_chunks(todo, R, 8))
    for chunk in chunks:
        blocks = {(i, h // R) for _pos, i, h, _d in chunk}
        assert len(blocks) == 1, f"chunk spans {blocks}"
    assert [p for chunk in chunks for p, _i, _h, _d in chunk] == \
           [p for p, _i, _h, _d in todo], "every label, still in order"


def test_batching_and_workers_compose(tmp_path_factory):
    one = _run(tmp_path_factory.mktemp("plain"), cfg=_batched_cfg(1),
               hero=_scripted_hero())[1]
    both = _run(tmp_path_factory.mktemp("both"),
                cfg=_batched_cfg(4, n_workers=2), hero=_scripted_hero())[1]
    assert len(both) == len(one) > 0
    for a, b in zip(one, both):
        assert np.array_equal(np.nan_to_num(a["q"], nan=-999.0),
                              np.nan_to_num(b["q"], nan=-999.0))
