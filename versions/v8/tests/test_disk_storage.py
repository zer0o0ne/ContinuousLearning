"""Real disk roundtrips, identical optimisation, and crash-safe deletion."""
import copy
import pickle
import subprocess
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

import pipeline
from env.driver import LockstepDriver
from env.session import build_sessions, play
from nets.features import TOKEN_FIELDS
from train.storage import DiskStore, DatasetView
from tests.g1_fixtures import make_pool, N_ACTIONS, MAX_PLAYERS
from tests.test_pipeline import GAME, toy_config, _run
from tests.test_label_generation import _cfg, _run as label_run
from tests.test_memory_lifecycle import RANGE_CFG


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_disk_labels_match_ram_bytes_and_posterior_cache_work(tmp_path, monkeypatch, chunk_size):
    import train.generate as generate
    cfg = _cfg(n_sessions=2, hands=7)
    cfg["game"] = dict(cfg["game"], players_range=[2, 2])
    cfg["embedding_net"]["range_enabled"] = True
    cfg["oracle"].update(max_combos=None, labels_per_batch=chunk_size)
    disk, _ = label_run(tmp_path / "disk", cfg=cfg)
    monkeypatch.setattr(generate, "DiskStore", lambda _: nullcontext(None))
    ram, _ = label_run(tmp_path / "ram", cfg=cfg)
    assert [Path(p).read_bytes() for p in disk["shards"]] == [Path(p).read_bytes() for p in ram["shards"]]
    assert disk["stats"].forwards == ram["stats"].forwards


def test_disk_sequence_bounds_live_payloads_and_spawn_pickle(tmp_path):
    refs = []
    with DiskStore(tmp_path) as store:
        rows = store.sequence("data")
        for i in range(80):
            value = np.full(64_000, i, dtype=np.float32)
            refs.append(weakref.ref(value))
            rows.append(value)
        del value
        assert all(r() is None for r in refs)
        assert Path(store.path).stat().st_size > 20_000_000
        # Access one item before spawn: the cache must not travel either.
        assert rows[17][0] == 17
        blob = pickle.dumps(rows)
        assert len(blob) < 2000
        restored = pickle.loads(blob)
        np.testing.assert_array_equal(restored[17], rows[17])
        restored.data.store.close()
        view = DatasetView(rows, np.array([79, 2, 15]))
        assert [int(x[0]) for x in view] == [79, 2, 15]
    assert not list(tmp_path.glob(".working-data-*"))


def test_exception_and_abandoned_process_cleanup_preserve_live_stores(tmp_path):
    with pytest.raises(RuntimeError):
        with DiskStore(tmp_path):
            raise RuntimeError("interrupted")
    assert not list(tmp_path.glob(".working-data-*"))
    script = ("import os, sys; from train.storage import DiskStore; "
              "s=DiskStore(sys.argv[1]); s.sequence('x').append(b'payload'); os._exit(0)")
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    abandoned = next(tmp_path.glob(".working-data-*"))
    with DiskStore(tmp_path) as live:
        assert not abandoned.exists()
        live.sequence("x").append(42)
        with DiskStore(tmp_path):
            assert Path(live.path).exists()
    assert not list(tmp_path.glob(".working-data-*"))


def test_play_to_disk_preserves_actions_cards_targets_and_batches(tmp_path):
    pool = make_pool()
    def sessions(store=None):
        return build_sessions(np.random.default_rng(19), list(range(len(pool))),
                              GAME, 4, 7, 919, "corpus", store=store)
    ram = sessions()
    calls = []
    real = pool[0].policy
    pool[0].policy = lambda contexts: (calls.append(len(contexts)), real(contexts))[1]
    play(LockstepDriver(pool, N_ACTIONS), ram, 5, lambda _: None, "ram", bar=False)
    first_calls = list(calls)
    calls.clear()
    with DiskStore(tmp_path) as store:
        disk = sessions(store)
        play(LockstepDriver(pool, N_ACTIONS), disk, 5, lambda _: None, "disk",
             bar=False, store=store)
        assert calls == first_calls
        for a, b in zip(ram, disk):
            for ra, rb in zip(a.records, b.records):
                np.testing.assert_array_equal(ra.rewards, rb.rewards)
                assert ra.hand_strength == rb.hand_strength
                assert [d["action_idx"] for d in ra.decisions] == [d["action_idx"] for d in rb.decisions]
            for ta, tb in zip(a.tokens(MAX_PLAYERS, N_ACTIONS), b.tokens(MAX_PLAYERS, N_ACTIONS)):
                for field in TOKEN_FIELDS:
                    np.testing.assert_array_equal(getattr(ta, field), getattr(tb, field))


def test_agent_disk_training_is_bit_exact_to_ram_with_range_targets(tmp_path):
    cfg = _cfg(n_sessions=2, hands=5)
    cfg["embedding_net"].update(RANGE_CFG)
    # Fixture networks use their own small dimensions; targets still exercise
    # sparse beliefs through the persisted label format.
    cfg["embedding_net"]["range_enabled"] = True
    manifest, labels = label_run(tmp_path / "labels", cfg=cfg)
    config = toy_config()
    config["embedding_net"].update(RANGE_CFG)
    train = dict(config["agent_train"], range_weight=0.5, temperature=0.5,
                 steps=3, first_iteration_steps=3)
    a = pipeline.AgentNet(config["embedding_net"], N_ACTIONS, MAX_PLAYERS)
    b = copy.deepcopy(a)
    indices = np.arange(len(labels))[::-1].copy()
    picked = [labels[i] for i in indices]
    target = pipeline.build_targets(picked, "soft_q", 0.5, "pot_plus_bet")
    first = pipeline.train_agent(a, [l["tokens"] for l in picked], target,
                                 [l["embeddings"] for l in picked], train, "cpu",
                                 lambda _: None, 73)
    second = pipeline.train_shards(b, manifest, indices, train, 0.5,
                                    "pot_plus_bet", "cpu", lambda _: None,
                                    73, 0, str(tmp_path))
    assert first == second
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    assert not list(tmp_path.glob(".working-data-*"))


def test_pipeline_cleanup_after_metrics_and_resume_after_cleanup(tmp_path, monkeypatch):
    cfg = toy_config(n_iterations=1)
    real = pipeline.cleanup_iteration_data
    called = []
    def cleanup(directory, log):
        assert Path(directory, "agent.pt").is_file()
        assert Path(directory, "metrics.json").is_file()
        real(directory, log)
        called.append(directory)
        if len(called) == 1:
            raise RuntimeError("crash after removing labels, before state commit")
    monkeypatch.setattr(pipeline, "cleanup_iteration_data", cleanup)
    with pytest.raises(RuntimeError, match="after removing"):
        _run(tmp_path, cfg)
    exp = tmp_path / "exp" / "toy"
    iteration = exp / "iter_0000"
    assert not (iteration / "labels").exists()
    assert not (iteration / "state.json").exists()
    monkeypatch.setattr(pipeline, "cleanup_iteration_data", real)
    def forbidden(*_, **__):
        pytest.fail("resume with checkpoint and metrics must not read removed data")
    monkeypatch.setattr(pipeline, "load_shard", forbidden)
    monkeypatch.setattr(pipeline, "generate_labels", forbidden)
    pipeline.run(cfg, lambda _: None, str(exp))
    assert (iteration / "state.json").is_file()
    assert not list(exp.rglob(".working-data-*"))


def test_embedding_disk_training_is_bit_exact_to_ram(tmp_path):
    from oracle.ranges import label_ranges
    from train.embed_train import train_embedding_net
    pool = make_pool()
    sessions = build_sessions(np.random.default_rng(9), list(range(len(pool))),
                              GAME, 2, 5, 121, "corpus")
    play(LockstepDriver(pool, N_ACTIONS), sessions, 4, lambda _: None, "corpus", bar=False)
    label_ranges(sessions, pool, N_ACTIONS, show_progress=False)
    cfg = dict(toy_config()["embedding_net"], **RANGE_CFG)
    a = pipeline.OpponentEmbeddingNet(cfg, N_ACTIONS, MAX_PLAYERS, len(pool))
    b = copy.deepcopy(a)
    first = train_embedding_net(a, copy.deepcopy(sessions), cfg, GAME, "cpu", lambda _: None, 72)
    with DiskStore(tmp_path) as store:
        second = train_embedding_net(b, sessions, cfg, GAME, "cpu", lambda _: None,
                                     72, store=store, release_sessions=True)
    assert not sessions
    assert first == second
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
