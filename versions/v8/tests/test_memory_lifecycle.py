"""Memory ownership contracts, exercised on real CPU forwards and phases.

Weak references check liveness, not allocator-dependent RSS. CUDA peaks must
be measured on the Spark; CPU tests pin which objects may survive each phase.
"""

import gc
import weakref

import numpy as np
import pytest
import torch

import pipeline
import train.agent_train as agent_train
import train.embed_train as embed_train
import train.generate as generate
from env.runout import HandRunout, RunoutConfig, prime, seat_scores
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from nets.features import collate, hand_tokens
from tests.g1_fixtures import (MAX_PLAYERS, N_ACTIONS, make_pool, make_specs, play)
from tests.test_pipeline import _run, toy_config
from tests.test_range_head import RANGE_CFG, _batch


@pytest.mark.parametrize("players", [2, 6, 9])
def test_configured_range_width_preserves_checkpoint_outputs_and_gradients(players):
    pool = make_pool()
    records = play(pool, make_specs(seed=5, n_hands=4, n_members=len(pool),
                                    num_players=players))
    batch = collate([hand_tokens(r, observer_pos=0, slot_of_seat=list(range(players)),
                                max_players=MAX_PLAYERS, n_actions=N_ACTIONS)
                     for r in records if r.decisions])
    full = AgentNet(RANGE_CFG, N_ACTIONS, MAX_PLAYERS).eval()
    bounded = AgentNet(RANGE_CFG, N_ACTIONS, MAX_PLAYERS,
                       table_max_players=players).eval()
    bounded.load_state_dict(full.state_dict(), strict=True)
    shapes = []
    hook = bounded.encoder.range_head.blocks[0].register_forward_pre_hook(
        lambda _m, args: shapes.append(tuple(args[0].shape)))
    outputs, grads = [], []
    for net in (full, bounded):
        emb = torch.zeros(*batch["mask"].shape, net.d_emb, requires_grad=True)
        seats = torch.zeros(*batch["seat_slot"].shape, net.d_emb, requires_grad=True)
        policy, ranges = net.logits_and_range(batch, emb, seats)
        loss = policy.square().mean() + ranges.masked_fill(~torch.isfinite(ranges), 0).square().mean()
        loss.backward()
        outputs.append((policy.detach(), ranges.detach()))
        grads.append((emb.grad, seats.grad, {n: p.grad for n, p in net.named_parameters()}))
    hook.remove()
    assert shapes == [(batch["mask"].shape[0], batch["mask"].shape[1] * players,
                       RANGE_CFG["d_range"])]
    for a, b in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    for a, b in zip(grads[0][:2], grads[1][:2]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-5)
    for name, a in grads[0][2].items():
        b = grads[1][2][name]
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-5, msg=name)


def test_inference_drops_range_graph_but_keeps_input_gradients():
    batch = _batch(n_hands=3, ranges=False)
    net = AgentNet(RANGE_CFG, N_ACTIONS, MAX_PLAYERS, table_max_players=3)
    watched = []
    hook = net.encoder.range_head.blocks[-1].register_forward_hook(
        lambda _m, _args, out: watched.append(out.requires_grad))
    saved, outputs, grads = [], [], []
    for return_range in (True, False):
        emb = torch.zeros(*batch["mask"].shape, net.d_emb, requires_grad=True)
        seats = torch.zeros(*batch["seat_slot"].shape, net.d_emb, requires_grad=True)
        storage = []

        def pack(tensor):
            storage.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            policy, ranges = net.logits_and_range(batch, emb, seats, return_range=return_range)
            if not return_range:
                assert ranges is None
            grads.append(torch.autograd.grad(policy.square().mean(), emb)[0])
        outputs.append(policy.detach())
        saved.append(sum(storage))
    hook.remove()
    assert watched == [True, False]
    assert saved[1] < saved[0]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


def test_range_limit_refuses_a_larger_table():
    batch = _batch(n_hands=2, ranges=False)
    net = OpponentEmbeddingNet(RANGE_CFG, N_ACTIONS, MAX_PLAYERS, 20,
                                table_max_players=2)
    with pytest.raises(ValueError, match="player limit"):
        net(batch, net.member_emb(batch), net.member_seat_emb(batch))


def test_training_batches_gradients_and_shard_storage_die_before_next_phase(tmp_path, monkeypatch):
    cfg = toy_config(n_iterations=3)
    cfg["embedding_net"].update(RANGE_CFG, max_iterations=3, range_weight=0.5)
    cfg["agent_train"]["range_weight"] = 0.5
    shard_refs, record_refs, batch_refs, trained = [], [], [], []
    real_load, real_gap = pipeline.load_shard, pipeline.oracle_gap
    real_train_emb, real_train_agent = pipeline.train_embedding_net, pipeline.train_agent

    def load(path, **kwargs):
        labels = real_load(path, **kwargs)
        shard_refs.extend(weakref.ref(lab["tokens"].cards.base) for lab in labels)
        return labels

    def gap(net, labels, *args, **kwargs):
        gc.collect()
        assert shard_refs and all(r() is None for r in shard_refs)
        assert all(lab["tokens"].ranges is None for lab in labels)
        assert all(p.grad is None for p in net.parameters())
        return real_gap(net, labels, *args, **kwargs)

    def training_emb(net, sessions, *args, **kwargs):
        record_refs.extend(weakref.ref(r) for s in sessions for r in s.records)
        result = real_train_emb(net, sessions, *args, **kwargs)
        assert not sessions
        assert all(p.grad is None for p in net.parameters())
        trained.append("embedding")
        return result

    def training_agent(net, *args, **kwargs):
        result = real_train_agent(net, *args, **kwargs)
        assert all(p.grad is None for p in net.parameters())
        trained.append("agent")
        return result

    def training_collate(*args, **kwargs):
        assert all(r() is None for r in record_refs), "raw corpus survived tokenisation"
        assert all(r() is None for r in batch_refs), "previous training batch survived"
        batch = collate(*args, **kwargs)
        batch_refs[:] = [weakref.ref(t) for t in batch.values()]
        return batch

    monkeypatch.setattr(pipeline, "load_shard", load)
    monkeypatch.setattr(pipeline, "oracle_gap", gap)
    monkeypatch.setattr(pipeline, "train_embedding_net", training_emb)
    monkeypatch.setattr(pipeline, "train_agent", training_agent)
    monkeypatch.setattr(embed_train, "collate", training_collate)
    monkeypatch.setattr(agent_train, "collate", training_collate)
    _run(tmp_path, cfg)
    assert trained == ["embedding", "agent"] * 3
    assert all(r() is None for r in batch_refs + record_refs + shard_refs)


def test_label_records_are_released_as_soon_as_their_last_decision_is_consumed(tmp_path, monkeypatch):
    from tests.test_label_generation import _cfg, _run as labels_run
    previous = {}
    real_requests = generate._label_requests

    def requests(chunk, sessions, *args):
        active = {(i, h) for _, i, h, _ in chunk}
        assert all(ref() is None for key, ref in previous.items() if key not in active)
        result = real_requests(chunk, sessions, *args)
        for (_, i, h, _), (record, *_rest) in zip(chunk, result):
            previous[(i, h)] = weakref.ref(record)
        return result

    monkeypatch.setattr(generate, "_label_requests", requests)
    cfg = _cfg(n_sessions=2, hands=5)
    cfg["embedding_net"]["range_enabled"] = True
    labels_run(tmp_path, cfg=cfg)
    assert len(previous) > 1 and all(r() is None for r in previous.values())


def test_runout_cache_entries_do_not_pin_a_whole_ranking_batch(monkeypatch):
    import env.runout as runout
    refs = []
    real = runout.evaluate_hands

    def rank(cards):
        result = real(cards)
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(runout, "evaluate_hands", rank)
    rng = np.random.default_rng(41)
    hands = [HandRunout(rng.permutation(52), 2, RunoutConfig(samples=4)) for _ in range(3)]
    prime([(h, 1) for h in hands])
    assert refs and all(r() is None for r in refs)
    for h in hands:
        known, holes, comps = h._job(1)
        np.testing.assert_array_equal(h.cached(1), seat_scores(known, holes, comps))


def test_shards_share_embedding_views_and_metric_load_skips_range_payload(tmp_path):
    from tests.test_label_generation import _cfg, _run as labels_run
    cfg = _cfg(n_sessions=2, hands=6)
    cfg["embedding_net"]["range_enabled"] = True
    manifest, _ = labels_run(tmp_path, cfg=cfg)
    repeated = 0
    for path in manifest["shards"]:
        labels = generate.load_shard(path)
        metric_labels = generate.load_shard(path, include_ranges=False)
        with np.load(path) as z:
            indices = z["emb_index"]
            assert len(z["emb_tables"]) == len(set(indices))
            for i, index in enumerate(indices):
                first = int(np.flatnonzero(indices == index)[0])
                if first < i:
                    repeated += 1
                    assert labels[first]["embeddings"] is labels[i]["embeddings"]
        for label, metric in zip(labels, metric_labels):
            assert label["tokens"].ranges is not None
            assert metric["tokens"].ranges is None
            np.testing.assert_array_equal(label["tokens"].cards, metric["tokens"].cards)
            np.testing.assert_array_equal(label["q"], metric["q"])
    assert repeated > 0


def test_resuming_completed_metrics_does_not_load_training_shards(tmp_path, monkeypatch):
    cfg = toy_config(n_iterations=1)
    _, exp_dir = _run(tmp_path, cfg)
    from pathlib import Path
    Path(exp_dir, "iter_0000", "state.json").unlink()

    def forbidden(_path):
        pytest.fail("completed training/metrics must not load the label corpus")

    monkeypatch.setattr(pipeline, "load_shard", forbidden)
    result = pipeline.run(cfg, lambda _: None, exp_dir)
    assert len(result) == 1


def test_phase_cleanup_reclaims_cycles_without_logging(capsys):
    from utils import release_memory

    class Cycle:
        pass

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        value = Cycle()
        value.self = value
        ref = weakref.ref(value)
        del value
        assert ref() is not None
        release_memory("cpu")
        assert ref() is None
        captured = capsys.readouterr()
        assert not captured.out and not captured.err
    finally:
        if was_enabled:
            gc.enable()
