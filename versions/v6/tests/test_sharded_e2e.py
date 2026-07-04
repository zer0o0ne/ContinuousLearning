"""End-to-end tests for shard-based dataset storage and training.

Each test simulates a realistic usage scenario (generate → train → resume)
with small data, verifying observable outcomes — not internal function logic.
"""

import json
import os
import random
import tempfile

import numpy as np
import torch
import pytest

MAX_PLAYERS = 6
N_ACTIONS = 5  # 2 raise sizes + fold + call + all-in

_TINY_CONFIG = {
    "architecture": {
        "d_model": 32,
        "n_heads": 2,
        "n_kv_heads": 1,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 56,
        "max_players": MAX_PLAYERS,
        "modelling_dropout": 0.0,
        "memory": {
            "n_levels": 1,
            "max_cluster_size": 4,
            "max_cluster_size_after": 4,
            "beam_width": 2,
        },
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": [0.5, 1.0],
            "flop": [0.5, 1.0],
            "turn": [0.5, 1.0],
            "river": [0.5, 1.0],
        },
        "max_players": MAX_PLAYERS,
        "big_blind": 10,
        "max_stack": 200,
    },
    "solver": {"type": "v1"},
}


def _make_event(hand_cards=(0, 1), pot=100.0, stack=190.0, big_blind=10.0,
                action_vec=None):
    """Create a minimal event dict matching the production format."""
    if action_vec is None:
        action_vec = [0.0] * N_ACTIONS
    return {
        "table": [-1, -1, -1, -1, -1],
        "hand": list(hand_cards),
        "hands": {},
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 2,
        "pot": pot,
        "stack": stack,
        "bets": np.array([0.0] * MAX_PLAYERS, dtype=np.float32),
        "stacks": [stack, stack] + [0.0] * (MAX_PLAYERS - 2),
        "action": action_vec,
        "big_blind": big_blind,
        "small_blind": big_blind / 2,
    }


def _make_scenario(hand_id, n_events=3, pot=100.0, rng=None):
    """Create a minimal GTO scenario dict matching the production format."""
    if rng is None:
        rng = random.Random(hand_id)
    events = []
    for t in range(n_events):
        action = [0.0] * N_ACTIONS
        if t > 0:
            action[rng.randint(0, N_ACTIONS - 1)] = 1.0
        card_a = rng.randint(0, 51)
        card_b = (card_a + 1) % 52
        events.append(_make_event(
            hand_cards=(card_a, card_b),
            pot=pot + t * 20,
            stack=200.0 - t * 10,
            action_vec=action,
        ))
    evs = [rng.gauss(0, 50) for _ in range(N_ACTIONS)]
    probs_raw = [max(0.01, rng.random()) for _ in range(N_ACTIONS)]
    total = sum(probs_raw)
    probs = [p / total for p in probs_raw]
    return {
        "hand_id": hand_id,
        "events": events,
        "n_events": n_events,
        "ev_target": max(evs),
        "action_evs": evs,
        "action_probs": probs,
        "equity": rng.random(),
        "pot": pot,
        "facing_bet": rng.random() * 20,
    }


def _write_shards(tmpdir, scenarios, shard_size=10, subdir="dataset_shards"):
    """Write scenarios as numbered shards + meta.json. Returns save_dir."""
    shard_dir = os.path.join(tmpdir, subdir)
    os.makedirs(shard_dir, exist_ok=True)
    shard_counts = []
    for i in range(0, len(scenarios), shard_size):
        chunk = scenarios[i:i + shard_size]
        path = os.path.join(shard_dir, f"shard_{i // shard_size:06d}.pt")
        torch.save(chunk, path)
        shard_counts.append(len(chunk))
    meta = {
        "version": 1,
        "storage": "sharded",
        "shard_counts": shard_counts,
        "n_shards": len(shard_counts),
        "done": True,
        "target": len(scenarios),
        "completed_attempts": len(scenarios),
    }
    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return tmpdir


# ─── Test 1: Shard-based generate → load → iterate ─────────────────────────

class TestShardedGenerateAndLoad:
    """Scenario: generate 30 scenarios in shards of 10, then load them
    through ShardedScenarios and verify all items are accessible."""

    def test_generate_shards_then_load_all_items(self):
        from agent.train_scenarios.sharded import ShardedScenarios

        rng = random.Random(42)
        scenarios = [_make_scenario(i, rng=random.Random(i)) for i in range(30)]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = _write_shards(tmpdir, scenarios, shard_size=10)

            shards = ShardedScenarios(save_dir)
            assert len(shards) == 30
            assert shards.n_shards == 3

            for i in range(30):
                s = shards[i]
                assert s["hand_id"] == i
                assert len(s["events"]) == 3

    def test_shard_files_exist_no_monolithic(self):
        """After shard-based generation, there must be shard files
        and NO monolithic dataset.pt."""
        scenarios = [_make_scenario(i) for i in range(15)]

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=5)

            shard_dir = os.path.join(tmpdir, "dataset_shards")
            shard_files = sorted(os.listdir(shard_dir))
            assert len(shard_files) == 3
            assert shard_files[0] == "shard_000000.pt"
            assert not os.path.exists(os.path.join(tmpdir, "dataset.pt"))


# ─── Test 2: ShardedGTODataset returns correct items per phase ──────────────

class TestShardedDatasetPhases:
    """Scenario: create ShardedGTODataset with different phase arguments,
    verify each returns the right format."""

    def test_ev_phase_returns_events_and_scalar(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )

        scenarios = [_make_scenario(i) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(shards)
            ds = ShardedGTODataset(shards, norm_stats, phase="ev")
            events, target = ds[0]
            assert isinstance(events, list)
            assert target.shape == ()

    def test_probs_phase_returns_distribution(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )

        scenarios = [_make_scenario(i) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(shards)
            ds = ShardedGTODataset(shards, norm_stats, phase="probs")
            events, probs = ds[0]
            assert probs.shape == (N_ACTIONS,)
            assert abs(probs.sum().item() - 1.0) < 0.01

    def test_combined_phase_returns_three_items(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )

        scenarios = [_make_scenario(i) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(shards)
            ds = ShardedGTODataset(shards, norm_stats, phase="combined")
            result = ds[0]
            assert len(result) == 3
            events, ev, probs = result
            assert ev.shape == ()
            assert probs.shape == (N_ACTIONS,)

    def test_modelling_phase_returns_action_evs(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )

        scenarios = [_make_scenario(i) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(shards)
            ds = ShardedGTODataset(shards, norm_stats, phase="modelling")
            events, aevs = ds[0]
            assert aevs.shape == (N_ACTIONS,)


# ─── Test 3: ShardBatchSampler keeps batches within shards ──────────────────

class TestShardBatchSamplerKeepsBatchesLocal:
    """Scenario: with 3 shards, every batch produced by the sampler must
    contain indices from exactly one shard. This prevents LRU cache thrashing."""

    def test_all_batches_are_shard_local(self):
        from agent.train_scenarios.sharded import ShardedScenarios, ShardBatchSampler

        scenarios = [_make_scenario(i) for i in range(30)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)

            sampler = ShardBatchSampler(
                shard_sizes=shards.shard_sizes,
                shard_offsets=shards.shard_offsets,
                batch_size=4,
            )

            for batch in sampler:
                shard_ids = set()
                for idx in batch:
                    if idx < 10:
                        shard_ids.add(0)
                    elif idx < 20:
                        shard_ids.add(1)
                    else:
                        shard_ids.add(2)
                assert len(shard_ids) == 1, \
                    f"Batch {batch} spans shards {shard_ids}"


# ─── Test 4: Full train/val split + DataLoader cycle ───────────────────────

class TestFullTrainingCycle:
    """Scenario: create sharded data → split → create DataLoaders → iterate
    one full epoch. Verifies the pipeline works end-to-end without OOM or
    data corruption."""

    def test_sharded_dataloader_yields_all_train_samples(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, ShardBatchSampler,
            scan_shard_metadata, compute_norm_stats_from_shards,
            shard_aware_split,
        )
        from torch.utils.data import DataLoader

        n_scenarios = 40
        scenarios = [_make_scenario(i) for i in range(n_scenarios)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            hand_ids, n_events = scan_shard_metadata(shards)
            norm_stats = compute_norm_stats_from_shards(shards)
            train_idx, val_idx = shard_aware_split(hand_ids, val_split=0.2)

            assert len(train_idx) + len(val_idx) == n_scenarios

            train_ds = ShardedGTODataset(shards, norm_stats, phase="ev",
                                         indices=train_idx)
            sampler = ShardBatchSampler(
                shard_sizes=shards.shard_sizes,
                shard_offsets=shards.shard_offsets,
                batch_size=8,
                indices=train_idx,
            )

            def collate(batch):
                events = [b[0] for b in batch]
                targets = torch.stack([b[1] for b in batch])
                return events, targets

            loader = DataLoader(train_ds, batch_sampler=sampler,
                                collate_fn=collate, num_workers=0)

            seen = 0
            for events, targets in loader:
                assert targets.dim() == 1
                seen += targets.shape[0]
            assert seen == len(train_idx)


# ─── Test 5: Legacy monolithic → shards conversion ─────────────────────────

class TestLegacyConversion:
    """Scenario: existing monolithic dataset.pt gets auto-converted to shards
    when ShardedScenarios is initialized. After conversion, the monolithic
    file must be removed and data must be accessible through shards."""

    def test_monolithic_to_shards_conversion(self):
        from agent.train_scenarios.generation.generate import (
            _convert_monolithic_to_shards,
        )
        from agent.train_scenarios.sharded import ShardedScenarios

        scenarios = [_make_scenario(i) for i in range(25)]
        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = os.path.join(tmpdir, "dataset.pt")
            torch.save(scenarios, mono_path)
            with open(os.path.join(tmpdir, "meta.json"), "w") as f:
                json.dump({"version": 1, "done": True, "target": 25}, f)

            _convert_monolithic_to_shards(tmpdir, mono_path,
                                          log=lambda m: None, shard_size=10)

            assert not os.path.exists(mono_path), "dataset.pt should be deleted"
            shard_dir = os.path.join(tmpdir, "dataset_shards")
            assert os.path.isdir(shard_dir)
            assert len(os.listdir(shard_dir)) == 3  # 25 items / 10 per shard

            shards = ShardedScenarios(tmpdir)
            assert len(shards) == 25
            for i in range(25):
                assert shards[i]["hand_id"] == i


# ─── Test 6: Norm stats streaming vs in-memory are equivalent ──────────────

class TestNormStatsConsistency:
    """Scenario: compute norm stats via streaming shards and via the legacy
    in-memory path. Results must be numerically identical (same data, same
    formula)."""

    def test_streaming_matches_inmemory(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, compute_norm_stats_from_shards,
        )
        from agent.train_scenarios.generation.generate import _compute_norm_stats

        scenarios = [_make_scenario(i, rng=random.Random(i + 100))
                     for i in range(50)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=15)
            shards = ShardedScenarios(tmpdir)
            streamed = compute_norm_stats_from_shards(shards)

        inmem = _compute_norm_stats(scenarios)

        for key in inmem:
            assert abs(streamed[key] - inmem[key]) < 1e-4, \
                f"{key}: streamed={streamed[key]:.6f} vs inmem={inmem[key]:.6f}"


# ─── Test 7: Modifiers applied lazily produce same result as upfront ────────

class TestLazyModifiers:
    """Scenario: apply modifiers lazily (via ShardedGTODataset) and upfront
    (via apply_modifiers). The ev_target and action_probs must match."""

    def test_lazy_matches_upfront(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )
        from agent.train_scenarios.modifiers import apply_modifiers

        scenarios = [_make_scenario(i, rng=random.Random(i + 200))
                     for i in range(20)]

        modifiers = [
            {"type": "action_bias", "actions": "call",
             "factor": 0.3},
        ]
        big_blind = 10.0
        temperature = 1.0

        upfront = apply_modifiers(scenarios, modifiers, N_ACTIONS,
                                  big_blind, temperature)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(
                shards, modifiers=modifiers,
                mod_params=(N_ACTIONS, big_blind, temperature))
            ds = ShardedGTODataset(shards, norm_stats, phase="ev",
                                   modifiers=modifiers,
                                   mod_params=(N_ACTIONS, big_blind, temperature))

            for i in range(20):
                _, lazy_ev = ds[i]
                upfront_denom = max(
                    upfront[i]["pot"] + upfront[i].get("facing_bet", 0),
                    upfront[i]["events"][-1]["big_blind"])
                upfront_ev_normalized = (
                    upfront[i]["ev_target"] / upfront_denom
                    - norm_stats["ev_mean"]
                ) / norm_stats["ev_std"]
                assert abs(lazy_ev.item() - upfront_ev_normalized) < 1e-4, \
                    f"Item {i}: lazy={lazy_ev.item():.6f} vs upfront={upfront_ev_normalized:.6f}"


# ─── Test 8: ShardedScenarios LRU cache evicts old shards ──────────────────

class TestLRUCacheEviction:
    """Scenario: access items from different shards in sequence. Only the
    most recently accessed shard should be in memory (LRU-1 cache)."""

    def test_only_one_shard_in_memory(self):
        from agent.train_scenarios.sharded import ShardedScenarios

        scenarios = [_make_scenario(i) for i in range(30)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)

            _ = shards[5]   # loads shard 0
            path_0 = shards._cache_path
            assert shards._cache_data is not None

            _ = shards[15]  # loads shard 1
            path_1 = shards._cache_path
            assert path_0 != path_1  # different shard loaded

            _ = shards[25]  # loads shard 2
            path_2 = shards._cache_path
            assert path_1 != path_2
            # shard 0 and 1 are no longer in cache
            assert shards._cache_path == path_2

    def test_clear_cache_frees_memory(self):
        from agent.train_scenarios.sharded import ShardedScenarios

        scenarios = [_make_scenario(i) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            _ = shards[0]
            assert shards._cache_data is not None
            shards.clear_cache()
            assert shards._cache_data is None
            assert shards._cache_path is None


# ─── Test 9: Hand-aware split has no leaks ──────────────────────────────────

class TestHandAwareSplit:
    """Scenario: scenarios with shared hand_ids must all go to the same
    split (train or val). No hand should appear in both."""

    def test_no_hand_leaks_between_splits(self):
        from agent.train_scenarios.sharded import (
            ShardedScenarios, scan_shard_metadata, shard_aware_split,
        )

        scenarios = []
        for h in range(20):
            n_events_per_hand = random.Random(h).randint(1, 3)
            for _ in range(n_events_per_hand):
                scenarios.append(_make_scenario(len(scenarios), rng=random.Random(len(scenarios))))
                scenarios[-1]["hand_id"] = h

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=10)
            shards = ShardedScenarios(tmpdir)
            hand_ids, _ = scan_shard_metadata(shards)
            train_idx, val_idx = shard_aware_split(hand_ids, val_split=0.2)

            train_hands = {hand_ids[i] for i in train_idx}
            val_hands = {hand_ids[i] for i in val_idx}
            assert not train_hands & val_hands, \
                f"Hands appear in both splits: {train_hands & val_hands}"


# ─── Test 10: Full forward pass with sharded data ──────────────────────────

class TestShardedForwardPass:
    """Scenario: create a tiny agent, load sharded data, run a forward pass
    through perception + value head. Verifies the full data→model pipeline."""

    def test_agent_forward_with_sharded_data(self):
        from agent.agent import ASI
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, compute_norm_stats_from_shards,
        )

        scenarios = [_make_scenario(i) for i in range(10)]
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        agent.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=5)
            shards = ShardedScenarios(tmpdir)
            norm_stats = compute_norm_stats_from_shards(shards)
            ds = ShardedGTODataset(shards, norm_stats, phase="ev")

            batch_events = [ds[i][0] for i in range(4)]
            with torch.no_grad():
                p_out, _, mask = agent.perception.forward_batch(
                    batch_events, device="cpu", skip_memory=True)
                values = agent.value_head(p_out, mask=mask)

            assert values.shape == (4, 1)
            assert torch.isfinite(values).all()


# ─── Memory-bounded generation: incremental shards, no full-RAM dataset ────

class TestGenerationMemoryBoundedShards:
    """Scenario: real tiny generation run — incremental shards land on disk
    as generation progresses (the in-RAM buffer is flushed per shard), the
    final dataset loads back complete, and an already-present dataset
    short-circuits without touching the shard files."""

    @staticmethod
    def _gen_cfg(n_scenarios):
        return {
            "raise_sizes": {
                "preflop": [0.5, 1.0],
                "flop": [0.5, 1.0],
                "turn": [0.5, 1.0],
                "river": [0.5, 1.0],
            },
            "max_players": MAX_PLAYERS,
            "big_blind": 10,
            "max_stack": 200,
            "solver": "v1",
            "mc_iterations": 20,
            "n_scenarios": n_scenarios,
            "n_workers": 1,
            "save_every_hands": 1,
            "device": "cpu",
        }

    def test_generation_writes_incremental_shards(self):
        from agent.train_scenarios.generation.generate import (
            generate_dataset, load_dataset,
        )
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = generate_dataset(self._gen_cfg(3), tmpdir, log=None)
            assert out == tmpdir

            shard_dir = os.path.join(tmpdir, "dataset_shards")
            shard_files = sorted(os.listdir(shard_dir))
            # save_every_hands=1 → one shard per successful hand
            assert len(shard_files) >= 2
            # no monolithic dataset.pt
            assert not os.path.exists(os.path.join(tmpdir, "dataset.pt"))

            scen = load_dataset(tmpdir)
            assert scen is not None and len(scen) > 0
            assert all("events" in s and "ev_target" in s for s in scen)
            # shards together contain exactly the loaded dataset
            counts = [
                len(torch.load(os.path.join(shard_dir, f), weights_only=False))
                for f in shard_files
            ]
            assert sum(counts) == len(scen)

    def test_existing_dataset_short_circuits_without_touching_shards(self):
        from agent.train_scenarios.generation.generate import generate_dataset

        scenarios = [_make_scenario(i) for i in range(12)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_shards(tmpdir, scenarios, shard_size=6)
            shard_dir = os.path.join(tmpdir, "dataset_shards")
            before = {
                f: os.path.getmtime(os.path.join(shard_dir, f))
                for f in os.listdir(shard_dir)
            }

            # n_scenarios=100 would take long if it actually regenerated —
            # the presence check must return immediately without loading.
            out = generate_dataset(self._gen_cfg(100), tmpdir, log=None)
            assert out == tmpdir

            after = {
                f: os.path.getmtime(os.path.join(shard_dir, f))
                for f in os.listdir(shard_dir)
            }
            assert after == before


# ─── Disk-backed perception cache ───────────────────────────────────────────

class TestDiskShardedCache:
    """Scenario: precompute per-sample tensors into a disk-backed cache, run
    one training epoch through ShardBatchSampler + DataLoader, verify every
    subset item is served exactly once with intact content, batches never
    cross cache shards, and cleanup removes the files."""

    @staticmethod
    def _build_cache(tmpdir, n_items=25, items_per_shard=10):
        from agent.train_scenarios.sharded import DiskShardedCache

        items = []
        for i in range(n_items):
            L = 2 + (i % 4)
            items.append((torch.full((L, 8), float(i)),
                          torch.ones(L),
                          torch.tensor(float(i))))
        cache = DiskShardedCache(os.path.join(tmpdir, "pcache"),
                                 items_per_shard=items_per_shard)
        for it in items:
            cache.append(it)
        cache.finalize()
        return cache, items

    def test_epoch_serves_each_subset_item_once_and_intact(self):
        from agent.train_scenarios.sharded import (
            CachedSubsetDataset, ShardBatchSampler,
        )
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            cache, items = self._build_cache(tmpdir)
            assert len(cache) == 25
            assert cache.shard_sizes == [10, 10, 5]
            assert sorted(os.listdir(cache.cache_dir)) == [
                "cache_000000.pt", "cache_000001.pt", "cache_000002.pt",
            ]

            subset = [i for i in range(25) if i % 5 != 0]
            ds = CachedSubsetDataset(cache, subset)
            n_events = [it[0].shape[0] for it in items]
            sampler = ShardBatchSampler(
                cache.shard_sizes, cache.shard_offsets, batch_size=4,
                indices=subset, n_events=n_events)

            random.seed(42)
            loader = DataLoader(ds, batch_sampler=sampler,
                                collate_fn=lambda b: b, num_workers=0)
            seen = []
            for batch in loader:
                shard_ids = set()
                for p, m, t in batch:
                    gi = int(t.item())
                    seen.append(gi)
                    assert torch.equal(p, items[gi][0])
                    assert torch.equal(m, items[gi][1])
                    shard_ids.add(gi // 10)
                # a batch must never span two cache shards (LRU-1 safety)
                assert len(shard_ids) == 1
            assert sorted(seen) == subset

    def test_cleanup_removes_cache_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache, _ = self._build_cache(tmpdir)
            cache_dir = cache.cache_dir
            assert os.listdir(cache_dir)
            cache.cleanup()
            assert not os.path.exists(cache_dir)


class _LogCapture:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))

    def contains(self, needle):
        return any(needle in l for l in self.lines)


class TestGtoProbsTrainsOnDiskCache:
    """Scenario: full train_gto_probs run over a sharded dataset directory.
    The perception cache is spilled to disk shards during training and the
    cache directory is removed after training completes."""

    def test_train_gto_probs_sharded(self):
        from agent.agent import ASI
        from agent.train_scenarios.gto_probs_predict.train import train_gto_probs

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        scenarios = [_make_scenario(i) for i in range(30)]
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        train_cfg = {"lr": 1e-3, "batch_size": 8, "epochs": 1,
                     "val_split": 0.2, "log_every": 1000}

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            _write_shards(data_dir, scenarios, shard_size=10)
            run_dir = os.path.join(tmpdir, "run")

            logger = _LogCapture()
            history, out_dir = train_gto_probs(
                agent, train_cfg, "cpu", logger,
                scenarios_dir=data_dir, run_dir=run_dir)

            assert logger.contains("perception_cache"), \
                "training must go through the disk-backed perception cache"
            assert history is not None
            assert len(history["epoch_train_loss"]) == 1
            assert os.path.exists(os.path.join(run_dir, "best.pt"))
            # disk cache cleaned up after training
            assert not os.path.exists(os.path.join(run_dir, "perception_cache"))


class TestModellingTrainsOnDiskCache:
    """Scenario: full train_modelling run over a sharded dataset directory —
    the 4-tuple perception cache (p_out, mask, lm_pair, target) goes through
    disk shards and is removed after training completes."""

    def test_train_modelling_sharded(self):
        from agent.agent import ASI
        from agent.train_scenarios.modelling_predict.train import train_modelling

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        scenarios = [_make_scenario(i) for i in range(30)]
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        train_cfg = {"lr": 1e-3, "batch_size": 8, "epochs": 1,
                     "val_split": 0.2, "log_every": 1000,
                     "recon_weight": 0.5, "infonce_weight": 0.5,
                     "infonce_temperature": 0.1}

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            _write_shards(data_dir, scenarios, shard_size=10)
            run_dir = os.path.join(tmpdir, "run")

            logger = _LogCapture()
            history, out_dir = train_modelling(
                agent, train_cfg, "cpu", logger,
                scenarios_dir=data_dir, run_dir=run_dir)

            assert logger.contains("perception_cache"), \
                "training must go through the disk-backed perception cache"
            assert history is not None
            assert len(history["epoch_train_loss"]) == 1
            assert os.path.exists(os.path.join(run_dir, "best.pt"))
            assert not os.path.exists(os.path.join(run_dir, "perception_cache"))


class TestOpponentActionTrainsOnDiskCache:
    """Scenario: full train_opponent_action run over a sharded opponent
    dataset (shards/ subdir, expanded per-observer samples) — the perception
    cache goes through disk shards and is removed after training completes."""

    @staticmethod
    def _make_opp_scenario(hand_id, rng):
        n_events = 2 + hand_id % 3
        events = []
        for t in range(n_events):
            action = [0.0] * N_ACTIONS
            if t > 0:
                action[rng.randint(0, N_ACTIONS - 1)] = 1.0
            events.append({
                "hands": {0: [8, 9], 1: [0, 4]},
                "num_players": 2,
                "acting_pos": t % 2,
                "hero_pos": 0,
                "big_blind": 10.0,
                "small_blind": 5.0,
                "stacks": [200.0, 200.0] + [0.0] * (MAX_PLAYERS - 2),
                "table": [-1, -1, -1, -1, -1],
                "pot": 30.0 + 10 * t,
                "stack": 200.0 - 10 * t,
                "bets": [5.0, 10.0] + [0.0] * (MAX_PLAYERS - 2),
                "action": action,
            })
        probs_raw = [max(0.01, rng.random()) for _ in range(N_ACTIONS)]
        total = sum(probs_raw)
        return {
            "hand_id": hand_id,
            "events": events,
            "hero_positions": [0, 1],
            "opponent_action_probs": [p / total for p in probs_raw],
        }

    def test_train_opponent_action_sharded(self):
        from agent.agent import ASI
        from agent.train_scenarios.opponent_action_predict.train import (
            train_opponent_action,
        )

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        rng = random.Random(7)
        scenarios = [self._make_opp_scenario(i, rng) for i in range(20)]
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        train_cfg = {"lr": 1e-3, "batch_size": 8, "epochs": 1,
                     "val_split": 0.2, "log_every": 1000}

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            _write_shards(data_dir, scenarios, shard_size=8, subdir="shards")
            run_dir = os.path.join(tmpdir, "run")

            logger = _LogCapture()
            history, out_dir = train_opponent_action(
                agent, train_cfg, "cpu", logger,
                scenarios_dir=data_dir, run_dir=run_dir)

            assert logger.contains("perception_cache"), \
                "training must go through the disk-backed perception cache"
            assert history is not None
            assert len(history["epoch_train_loss"]) == 1
            assert os.path.exists(os.path.join(run_dir, "best.pt"))
            assert not os.path.exists(os.path.join(run_dir, "perception_cache"))
