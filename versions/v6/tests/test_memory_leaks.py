"""Tests for memory leaks and inefficient data storage in the pipeline.

Covers:
  - Scenario fields use memory-efficient types (numpy, not Python lists)
  - _load_agents does not double-load checkpoint files
  - Pipeline frees base_scenarios before opponent phase
  - Train functions release internal datasets after return
  - Events use shallow copy, not deepcopy

Run from versions/v6/:
    python -m pytest tests/test_memory_leaks.py -v
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

MAX_PLAYERS = 2

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

N_ACTIONS = 5  # 2 raise sizes + fold + call + all-in


def _make_agent():
    from agent.agent import ASI
    log = lambda msg: None
    agent = ASI(log, config=_TINY_CONFIG)
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# 1. Scenario storage efficiency: per_combo_probs / forward_weights must be
#    numpy arrays, not Python lists. Python float objects use ~28 bytes each
#    vs 4 bytes in numpy float32 — 7x overhead on unified memory.
# ---------------------------------------------------------------------------

class TestScenarioStorageEfficiency(unittest.TestCase):
    """Scenario dicts produced by generate_opponent_hand must store
    per_combo_probs and forward_weights as numpy arrays, not Python lists."""

    @classmethod
    def setUpClass(cls):
        """Generate scenarios once, reuse across tests."""
        import random
        from agent.train_scenarios.generation.generate_opponent import (
            generate_opponent_hand,
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        agent = _make_agent()
        game_cfg = _TINY_CONFIG["game"]
        config = {
            **game_cfg,
            "max_batch_combos": 32,
            "bayes": {"enabled": True, "tau_belief": 2.0,
                      "ess_truncation_mass": 0.995},
            "min_stack": 100,
        }

        agents_list = [{
            "agent": agent,
            "norm_stats": {
                "pot_mean": 0.0, "pot_std": 1.0,
                "stack_mean": 0.0, "stack_std": 1.0,
                "bets_mean": 0.0, "bets_std": 1.0,
                "blind_mean": 0.0, "blind_std": 1.0,
            },
            "name": "test_agent",
            "temperature": 1.0,
        }]

        amp_config = (False, "cpu", torch.float32)
        cls._scenarios = None
        for seed in range(42, 142):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = generate_opponent_hand(config, agents_list, "cpu",
                                            amp_config)
            if result is not None and len(result) >= 2:
                cls._scenarios = result
                break

    def test_per_combo_probs_is_numpy(self):
        """per_combo_probs must be a numpy array, not a Python list."""
        if self._scenarios is None:
            self.skipTest("Could not generate scenarios")
        for i, s in enumerate(self._scenarios):
            self.assertIsInstance(
                s["per_combo_probs"], np.ndarray,
                f"scenario[{i}].per_combo_probs is {type(s['per_combo_probs']).__name__}, "
                f"expected numpy.ndarray (Python lists waste ~7x memory)")

    def test_forward_weights_is_numpy(self):
        """forward_weights must be a numpy array, not a Python list."""
        if self._scenarios is None:
            self.skipTest("Could not generate scenarios")
        for i, s in enumerate(self._scenarios):
            self.assertIsInstance(
                s["forward_weights"], np.ndarray,
                f"scenario[{i}].forward_weights is {type(s['forward_weights']).__name__}, "
                f"expected numpy.ndarray")

    def test_forward_combos_is_numpy(self):
        """forward_combos must be a numpy int32 array, not Python list of lists.
        Python list of lists wastes ~160 KB per scenario vs ~10 KB as numpy."""
        if self._scenarios is None:
            self.skipTest("Could not generate scenarios")
        for i, s in enumerate(self._scenarios):
            self.assertIsInstance(
                s["forward_combos"], np.ndarray,
                f"scenario[{i}].forward_combos is {type(s['forward_combos']).__name__}, "
                f"expected numpy.ndarray")


# ---------------------------------------------------------------------------
# 2. _load_agents must not double-load checkpoints. The agent already stores
#    _checkpoint_norm_stats after load_checkpoint; loading the full .pt file
#    again just to extract norm_stats + temperature wastes memory.
# ---------------------------------------------------------------------------

class TestLoadAgentsNoDoublePt(unittest.TestCase):
    """_load_agents must call torch.load at most once per agent, not twice."""

    def test_single_torch_load_per_agent(self):
        from agent.agent import ASI

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_subdir = os.path.join(tmpdir, "agent_a", "gto_ev_predict",
                                        "20260101_000000")
            os.makedirs(agent_subdir)

            log = lambda msg: None
            dummy_agent = ASI(log, config=_TINY_CONFIG)
            ckpt_path = os.path.join(agent_subdir, "best.pt")
            torch.save({
                "model_state_dict": dummy_agent.state_dict(),
                "norm_stats": {"pot_mean": 1.0, "pot_std": 2.0,
                               "stack_mean": 3.0, "stack_std": 4.0,
                               "bets_mean": 0.0, "bets_std": 1.0,
                               "blind_mean": 0.0, "blind_std": 1.0},
                "temperature": 0.8,
            }, ckpt_path)

            from agent.train_scenarios.generation.generate_opponent import (
                _load_agents,
            )

            original_torch_load = torch.load
            load_calls = []

            def tracking_load(*args, **kwargs):
                load_calls.append(args[0] if args else kwargs.get("f"))
                return original_torch_load(*args, **kwargs)

            with mock.patch("torch.load", side_effect=tracking_load):
                with mock.patch("agent.agent.torch.load",
                                side_effect=tracking_load):
                    agents = _load_agents(tmpdir, _TINY_CONFIG, "cpu", log,
                                          fallback_temperature=1.0)

            self.assertEqual(len(agents), 1, "Expected 1 agent loaded")
            self.assertLessEqual(
                len(load_calls), 1,
                f"torch.load called {len(load_calls)} times for 1 agent "
                f"(paths: {load_calls}). Expected at most 1 — the second "
                f"load holds the full checkpoint in memory unnecessarily.")
            self.assertIsNotNone(agents[0]["norm_stats"])
            self.assertIsNotNone(agents[0]["temperature"])


# ---------------------------------------------------------------------------
# 3. Events in scenarios must use shallow copy (or equivalent), not
#    copy.deepcopy. Deep-copying numpy arrays inside events wastes memory.
# ---------------------------------------------------------------------------

class TestEventsShallowCopy(unittest.TestCase):
    """generate_opponent_hand must NOT use copy.deepcopy on shared_events.
    Shallow dict copy ({**e} for e) is sufficient and avoids duplicating
    every numpy array inside events."""

    def test_no_deepcopy_import_used(self):
        """Verify that generate_opponent.py does not call copy.deepcopy
        on shared_events (the only place deepcopy was historically used)."""
        import ast

        path = os.path.join(
            _PKG_ROOT, "agent", "train_scenarios", "generation",
            "generate_opponent.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "generate_opponent_hand":
                continue

            for child in ast.walk(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "deepcopy"):
                    self.fail(
                        "copy.deepcopy is still used in generate_opponent_hand. "
                        "Use [{**e} for e in shared_events] instead — deepcopy "
                        "duplicates every numpy array, wasting ~2-4 GB on "
                        "5000 hands."
                    )
            break
        else:
            self.fail("Could not find generate_opponent_hand function")


# ---------------------------------------------------------------------------
# 4. Pipeline must free base_scenarios BEFORE starting opponent data
#    generation, not after it.
# ---------------------------------------------------------------------------

class TestPipelineNoInMemoryDataset(unittest.TestCase):
    """Pipeline must not hold a full in-memory dataset (base_scenarios).
    With sharded storage, only scenarios_dir (a string path) is passed."""

    def test_no_base_scenarios_in_pipeline(self):
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        self.assertNotIn("base_scenarios", source,
            "pipeline.py still references 'base_scenarios' — the full GTO "
            "dataset should never be loaded into memory. Use scenarios_dir.")


# ---------------------------------------------------------------------------
# 5. Pipeline must explicitly `del modified` after each agent in the
#    multi-agent loop (alongside `del agent`).
# ---------------------------------------------------------------------------

class TestPipelineNoModifiedList(unittest.TestCase):
    """With sharded storage, modifiers are applied lazily per-scenario in
    ShardedGTODataset.__getitem__. Pipeline must not create an in-memory
    'modified' list."""

    def test_no_modified_scenarios_in_pipeline(self):
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        self.assertNotIn("apply_modifiers(", source,
            "pipeline.py still calls apply_modifiers() — modifiers should be "
            "applied lazily per-scenario in ShardedGTODataset, not upfront.")


# ---------------------------------------------------------------------------
# 6. OpponentActionDataset must not hold a direct alias to the caller's
#    scenarios list, so the caller can free the original without breaking
#    the dataset.
# ---------------------------------------------------------------------------

class TestOpponentTrainReleasesScenarios(unittest.TestCase):
    """OpponentActionDataset.scenarios must not be the exact same object
    as the input list."""

    def test_dataset_does_not_alias_input_list(self):
        from agent.train_scenarios.opponent_action_predict.dataset import (
            OpponentActionDataset,
        )

        fake_scenarios = [
            {
                "events": [{"pot": 10, "bets": np.array([5, 5]),
                            "table": [-1, -1, -1, -1, -1],
                            "hand": [0, 1], "hands": {},
                            "acting_pos": 0, "hero_pos": 1,
                            "num_players": 2, "stacks": [100, 100],
                            "action": [1.0, 0.0, 0.0, 0.0, 0.0]}],
                "opponent_action_probs": [0.5, 0.3, 0.1, 0.05, 0.05],
                "hero_positions": [1],
                "acting_pos": 0,
                "num_players": 2,
                "forward_combos": [[2, 3]],
                "per_combo_probs": np.array([[0.5, 0.3, 0.1, 0.05, 0.05]]),
                "forward_weights": np.array([1.0]),
                "pot": 10,
                "facing_bet": 0,
                "n_events": 1,
            },
        ]

        dataset = OpponentActionDataset(fake_scenarios)
        self.assertIsNot(
            dataset.scenarios, fake_scenarios,
            "OpponentActionDataset.scenarios is the same object as the input "
            "list. This prevents the caller from freeing the original. "
            "The dataset should take ownership via a shallow copy.")


# ---------------------------------------------------------------------------
# 7. opp_scenarios must be freed before MCTS phase.
# ---------------------------------------------------------------------------

class TestPipelineNoInMemoryOppScenarios(unittest.TestCase):
    """With sharded storage, pipeline passes opp_scenarios_dir (string path),
    not an in-memory opp_scenarios list."""

    def test_no_opp_scenarios_list_in_pipeline(self):
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        import re
        bare = re.findall(r'\bopp_scenarios\b(?!_dir)', source)
        self.assertEqual(len(bare), 0,
            f"pipeline.py has {len(bare)} reference(s) to 'opp_scenarios' "
            f"(not 'opp_scenarios_dir'). Opponent data must use sharded "
            f"paths, not in-memory lists.")


# ---------------------------------------------------------------------------
# 8. _release_memory must be called after each major cleanup in pipeline.
# ---------------------------------------------------------------------------

class TestPipelineCallsReleaseMemory(unittest.TestCase):
    """Pipeline must call _release_memory at key cleanup points to force
    Python's allocator to return freed pages to the OS (malloc_trim)."""

    def test_release_memory_function_exists(self):
        import importlib
        pipeline = importlib.import_module("pipeline")
        self.assertTrue(
            hasattr(pipeline, "_release_memory"),
            "_release_memory function not found in pipeline.py")

    def test_release_memory_called_after_del_agent(self):
        """_release_memory must be called after `del agent` in the
        multi-agent training loop."""
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if "del agent" in line and i > 100:
                window = "".join(lines[i:i + 5])
                if "_release_memory" in window:
                    found = True
                    break

        self.assertTrue(found,
                        "_release_memory not called near `del agent`")


# ---------------------------------------------------------------------------
# 9. _run_or_resume_phase must not hold ckpt dict after extracting fields.
# ---------------------------------------------------------------------------

class TestResumePhaseFreesCheckpoint(unittest.TestCase):
    """_run_or_resume_phase must `del ckpt` after extracting resume_state
    so the model_state_dict (~200 MB) is freed during training."""

    def test_ckpt_deleted_in_resume_path(self):
        import ast
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_run_or_resume_phase":
                continue

            found_del_ckpt = False
            for child in ast.walk(node):
                if isinstance(child, ast.Delete):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "ckpt":
                            found_del_ckpt = True
                        elif isinstance(target, ast.Subscript):
                            val = child.targets[0]
                            if (isinstance(val, ast.Subscript)
                                    and isinstance(val.value, ast.Name)
                                    and val.value.id == "ckpt"):
                                found_del_ckpt = True

            self.assertTrue(
                found_del_ckpt,
                "`del ckpt` not found in _run_or_resume_phase. The full "
                "checkpoint dict (model_state_dict ~200 MB) stays alive "
                "through the entire training run via resume_state's "
                "reference to the parent dict.")
            break
        else:
            self.fail("Could not find _run_or_resume_phase function")


# ---------------------------------------------------------------------------
# 10. ASI.load_checkpoint must store _checkpoint_temperature.
# ---------------------------------------------------------------------------

class TestAgentStoresCheckpointTemperature(unittest.TestCase):
    """ASI.load_checkpoint must extract temperature from the checkpoint
    so _load_agents doesn't need a second torch.load."""

    def test_checkpoint_temperature_stored(self):
        from agent.agent import ASI

        with tempfile.TemporaryDirectory() as tmpdir:
            log = lambda msg: None
            agent = ASI(log, config=_TINY_CONFIG)

            ckpt_path = os.path.join(tmpdir, "best.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "norm_stats": {"pot_mean": 0.0, "pot_std": 1.0,
                               "stack_mean": 0.0, "stack_std": 1.0,
                               "bets_mean": 0.0, "bets_std": 1.0,
                               "blind_mean": 0.0, "blind_std": 1.0},
                "temperature": 0.42,
            }, ckpt_path)

            agent2 = ASI(log, config=_TINY_CONFIG)
            self.assertIsNone(agent2._checkpoint_temperature)

            agent2.load_checkpoint(ckpt_path)
            self.assertEqual(agent2._checkpoint_temperature, 0.42,
                             "load_checkpoint must store temperature in "
                             "_checkpoint_temperature")

    def test_checkpoint_temperature_none_when_absent(self):
        from agent.agent import ASI

        with tempfile.TemporaryDirectory() as tmpdir:
            log = lambda msg: None
            agent = ASI(log, config=_TINY_CONFIG)

            ckpt_path = os.path.join(tmpdir, "best.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
            }, ckpt_path)

            agent2 = ASI(log, config=_TINY_CONFIG)
            agent2.load_checkpoint(ckpt_path)
            self.assertIsNone(agent2._checkpoint_temperature)


# ---------------------------------------------------------------------------
# 11. OpponentActionDataset._observer_target works with numpy forward_combos.
# ---------------------------------------------------------------------------

class TestDatasetWorksWithNumpyCombos(unittest.TestCase):
    """The observer_target blocker logic must handle numpy int32 arrays
    for forward_combos (not just Python lists)."""

    def test_numpy_forward_combos_blocker_works(self):
        from agent.train_scenarios.opponent_action_predict.dataset import (
            OpponentActionDataset,
        )

        combos = np.array([[10, 20], [30, 40], [10, 50]], dtype=np.int32)
        probs = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.2, 0.4, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.3, 0.3, 0.2],
        ], dtype=np.float32)
        weights = np.array([0.4, 0.4, 0.2], dtype=np.float32)

        scenario = {
            "events": [{"pot": 10, "bets": np.array([5, 5]),
                        "table": [-1, -1, -1, -1, -1],
                        "hand": [0, 1], "hands": {},
                        "acting_pos": 0, "hero_pos": 1,
                        "num_players": 2, "stacks": [100, 100],
                        "action": [1.0, 0.0, 0.0, 0.0, 0.0]}],
            "opponent_action_probs": [0.3, 0.3, 0.2, 0.1, 0.1],
            "hero_positions": [1],
            "acting_pos": 0,
            "num_players": 2,
            "forward_combos": combos,
            "per_combo_probs": probs,
            "forward_weights": weights,
            "pot": 10,
            "facing_bet": 0,
            "n_events": 1,
        }

        dataset = OpponentActionDataset([scenario])
        target = dataset._observer_target(scenario, hero_hand=[10, 11])

        self.assertEqual(target.shape, (5,))
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)
        # Hero holds card 10 → combos 0 and 2 are blocked (contain card 10)
        # Only combo 1 ([30, 40]) survives → target should be combo 1's probs
        expected = torch.tensor([0.2, 0.4, 0.2, 0.1, 0.1])
        for i in range(5):
            self.assertAlmostEqual(
                float(target[i]), float(expected[i]), places=4,
                msg=f"target[{i}] mismatch with numpy forward_combos")


# ---------------------------------------------------------------------------
# 12. Pipeline skips GTO dataset load when all training phases are done.
# ---------------------------------------------------------------------------

class TestPipelineSkipsDatasetWhenDone(unittest.TestCase):
    """In resume mode, if every enabled training phase is status=done for
    every agent, the pipeline must NOT load the (10+ GB) GTO dataset."""

    def test_needs_training_false_when_all_done(self):
        """Verify the AST pattern: resume + needs_training → check state →
        set needs_training = False."""
        import ast
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "main":
                continue

            found_all_done_check = False
            found_needs_training_false = False
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == "all_done":
                    found_all_done_check = True
                if (isinstance(child, ast.Assign)
                        and len(child.targets) == 1
                        and isinstance(child.targets[0], ast.Name)
                        and child.targets[0].id == "needs_training"
                        and isinstance(child.value, ast.Constant)
                        and child.value.value is False):
                    found_needs_training_false = True

            self.assertTrue(
                found_all_done_check,
                "main() doesn't check `all_done` for training phases in "
                "resume mode. Without this, the GTO dataset (~10 GB) is "
                "loaded even when all phases are complete.")
            self.assertTrue(
                found_needs_training_false,
                "main() doesn't set `needs_training = False` when all "
                "phases are done.")
            break
        else:
            self.fail("Could not find main() in pipeline.py")


# ---------------------------------------------------------------------------
# 13. Opponent data uses shard-based storage, not monolithic torch.save.
# ---------------------------------------------------------------------------

class TestOpponentDataShardedStorage(unittest.TestCase):
    """generate_opponent_dataset must use shard-based storage instead of
    re-serialising the entire scenario list on every periodic save."""

    def test_shard_helpers_exist(self):
        from agent.train_scenarios.generation.generate_opponent import (
            _opp_shard_dir, _save_opp_shard, _list_opp_shard_paths,
            load_opponent_shards,
        )
        self.assertTrue(callable(_opp_shard_dir))
        self.assertTrue(callable(_save_opp_shard))
        self.assertTrue(callable(_list_opp_shard_paths))
        self.assertTrue(callable(load_opponent_shards))

    def test_save_and_load_shards_roundtrip(self):
        from agent.train_scenarios.generation.generate_opponent import (
            _save_opp_shard, load_opponent_shards,
        )
        from agent.train_scenarios.generation.generate import _write_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            shard0 = [{"x": 1}, {"x": 2}]
            shard1 = [{"x": 3}]
            _save_opp_shard(shard0, tmpdir, 0)
            _save_opp_shard(shard1, tmpdir, 1)
            _write_meta(tmpdir, {"storage": "sharded", "done": True})

            loaded = load_opponent_shards(tmpdir)
            self.assertEqual(len(loaded), 3)
            self.assertEqual([s["x"] for s in loaded], [1, 2, 3])

    def test_load_shards_backward_compat_with_legacy_dataset_pt(self):
        """If only dataset.pt exists (no shards), load_opponent_shards
        must still work (backward compat with pre-shard runs)."""
        from agent.train_scenarios.generation.generate_opponent import (
            load_opponent_shards,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_data = [{"x": 10}, {"x": 20}]
            torch.save(legacy_data, os.path.join(tmpdir, "dataset.pt"))

            loaded = load_opponent_shards(tmpdir)
            self.assertEqual(len(loaded), 2)
            self.assertEqual([s["x"] for s in loaded], [10, 20])


# ---------------------------------------------------------------------------
# 14. Opponent data resume does NOT torch.load prior scenarios.
# ---------------------------------------------------------------------------

class TestOpponentResumeZeroMemory(unittest.TestCase):
    """On resume, generate_opponent_dataset must NOT load the entire
    prior dataset into memory. It should read only meta.json."""

    def test_sequential_resume_does_not_load_prior_data(self):
        """AST check: the sequential resume path must NOT have
        `torch.load(dataset_path)` — it should read meta.json only."""
        import ast

        path = os.path.join(
            _PKG_ROOT, "agent", "train_scenarios", "generation",
            "generate_opponent.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "generate_opponent_dataset":
                continue

            # Look for torch.load calls that load dataset_path in the
            # resume/prior_scenarios section. The ONLY torch.load should
            # be via load_opponent_shards at the end (return value).
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                is_torch_load = (
                    (isinstance(func, ast.Attribute) and func.attr == "load"
                     and isinstance(func.value, ast.Name)
                     and func.value.id == "torch")
                )
                if not is_torch_load:
                    continue
                if child.args:
                    arg0 = child.args[0]
                    if isinstance(arg0, ast.Name) and arg0.id == "dataset_path":
                        self.fail(
                            "generate_opponent_dataset still calls "
                            "torch.load(dataset_path) in its body. The "
                            "resume path must NOT load the prior dataset — "
                            "only meta.json should be read (O(1) memory). "
                            "Prior data stays on disk as shards.")
            break
        else:
            self.fail("Could not find generate_opponent_dataset")

    def test_no_prior_scenarios_variable(self):
        """The old `prior_scenarios` list variable should be gone —
        it loaded the entire dataset into memory on resume."""
        import ast

        path = os.path.join(
            _PKG_ROOT, "agent", "train_scenarios", "generation",
            "generate_opponent.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "generate_opponent_dataset":
                continue

            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == "prior_scenarios":
                    self.fail(
                        "`prior_scenarios` variable still exists in "
                        "generate_opponent_dataset. This means the old "
                        "torch.load(dataset_path) path is still present.")
            break
        else:
            self.fail("Could not find generate_opponent_dataset")


# ---------------------------------------------------------------------------
# 15. Pipeline skips opponent data load when all opp training done.
# ---------------------------------------------------------------------------

class TestPipelineSkipsOppDataWhenDone(unittest.TestCase):
    """In resume mode, if all opponent_action_predict phases are done,
    the pipeline must NOT call load_opponent_shards (multi-GB load)."""

    def test_all_opp_train_done_check_exists(self):
        """Pipeline must check if all opponent training is done before
        loading the (multi-GB) opponent dataset for training."""
        import ast
        pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "main":
                continue

            found = False
            for child in ast.walk(node):
                if (isinstance(child, ast.Name)
                        and child.id == "all_opp_train_done"):
                    found = True
                    break
            self.assertTrue(
                found,
                "main() does not check `all_opp_train_done`. Without this, "
                "the opponent dataset (multi-GB) is loaded even when all "
                "opponent_action_predict phases are done.")
            break
        else:
            self.fail("Could not find main() in pipeline.py")


# ---------------------------------------------------------------------------
# 16. generate_opponent_dataset returns save_dir, not loaded list.
# ---------------------------------------------------------------------------

class TestGenerateOpponentReturnsDir(unittest.TestCase):
    """generate_opponent_dataset must return save_dir (str), not
    the loaded scenario list. The pipeline loads data only when
    training actually needs it."""

    def test_return_is_not_list(self):
        """AST check: final return must NOT be a variable named 'scenarios'."""
        import ast

        path = os.path.join(
            _PKG_ROOT, "agent", "train_scenarios", "generation",
            "generate_opponent.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "generate_opponent_dataset":
                continue

            returns = [n for n in ast.walk(node)
                       if isinstance(n, ast.Return)]
            for ret in returns:
                if (ret.value and isinstance(ret.value, ast.Name)
                        and ret.value.id == "scenarios"):
                    self.fail(
                        "generate_opponent_dataset returns `scenarios` "
                        "(the loaded list). It should return `save_dir` "
                        "(str) so the caller can decide when/whether to "
                        "load the data.")
            break
        else:
            self.fail("Could not find generate_opponent_dataset")


# ---------------------------------------------------------------------------
# 17. Periodic save uses shards (O(buffer) per save, not O(total)).
# ---------------------------------------------------------------------------

class TestPeriodicSaveUsesShards(unittest.TestCase):
    """Periodic saves during opponent generation must write only the new
    buffer to a shard file, not re-serialise the entire growing list."""

    def test_no_atomic_torch_save_of_full_list(self):
        """generate_opponent_dataset must NOT call atomic_torch_save with
        the full scenarios list. Only _save_opp_shard (or _flush_buffer)
        with a small buffer should be used."""
        import ast

        path = os.path.join(
            _PKG_ROOT, "agent", "train_scenarios", "generation",
            "generate_opponent.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "generate_opponent_dataset":
                continue

            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                is_save = (
                    isinstance(func, ast.Name) and func.id == "atomic_torch_save"
                )
                if is_save and child.args:
                    arg0 = child.args[0]
                    if isinstance(arg0, ast.Name) and arg0.id == "scenarios":
                        self.fail(
                            "generate_opponent_dataset calls "
                            "atomic_torch_save(scenarios, ...) — this "
                            "re-serialises the ENTIRE growing list on each "
                            "periodic save (O(n²) I/O). Use shard-based "
                            "saves instead.")
            break
        else:
            self.fail("Could not find generate_opponent_dataset")


if __name__ == "__main__":
    unittest.main()
