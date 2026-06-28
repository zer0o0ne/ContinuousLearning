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

class TestPipelineFreesBaseBeforeOpponent(unittest.TestCase):
    """base_scenarios must be set to None before the opponent data phase
    so the GTO dataset isn't held in memory during opponent generation."""

    def test_base_scenarios_freed_before_opponent_data(self):
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

            none_lines = []
            opponent_data_line = None

            for child in ast.walk(node):
                if (isinstance(child, ast.Assign)
                        and len(child.targets) == 1
                        and isinstance(child.targets[0], ast.Name)
                        and child.targets[0].id == "base_scenarios"
                        and isinstance(child.value, ast.Constant)
                        and child.value.value is None):
                    none_lines.append(child.lineno)

                if isinstance(child, ast.If):
                    src_segment = ast.dump(child.test)
                    if "run_opponent_data" in src_segment:
                        opponent_data_line = child.lineno

            self.assertIsNotNone(opponent_data_line,
                                 "Could not find `run_opponent_data` check")

            # Filter: only None-assignments BETWEEN the initial declaration
            # and the opponent data block count. The initial `base_scenarios
            # = None` on ~line 391 is just declaration, not cleanup.
            cleanup_lines = [
                ln for ln in none_lines
                if ln > 400 and ln < opponent_data_line
            ]

            self.assertTrue(
                len(cleanup_lines) > 0,
                f"base_scenarios is only freed at line(s) {none_lines}, "
                f"but opponent_data starts at line {opponent_data_line}. "
                f"Must free base_scenarios BEFORE opponent data generation "
                f"to avoid holding ~1 GB of GTO data during opponent phase.")
            break
        else:
            self.fail("Could not find main() in pipeline.py")


# ---------------------------------------------------------------------------
# 5. Pipeline must explicitly `del modified` after each agent in the
#    multi-agent loop (alongside `del agent`).
# ---------------------------------------------------------------------------

class TestPipelineFreesModifiedPerAgent(unittest.TestCase):
    """The multi-agent loop must free `modified` alongside `agent`."""

    def test_modified_freed_in_agent_loop(self):
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

            found_del_modified = False
            for child in ast.walk(node):
                if isinstance(child, ast.Delete):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "modified":
                            found_del_modified = True

            self.assertTrue(
                found_del_modified,
                "`del modified` not found in main(). The modified "
                "scenarios list (~same size as base_scenarios) must be "
                "explicitly freed after each agent finishes training.")
            break
        else:
            self.fail("Could not find main() in pipeline.py")


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

class TestPipelineFreesOppScenariosBeforeMCTS(unittest.TestCase):
    """opp_scenarios must be set to None before the MCTS block."""

    def test_opp_scenarios_freed_before_mcts(self):
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

            opp_free_line = None
            mcts_line = None

            for child in ast.walk(node):
                if (isinstance(child, ast.Assign)
                        and len(child.targets) == 1
                        and isinstance(child.targets[0], ast.Name)
                        and child.targets[0].id == "opp_scenarios"
                        and isinstance(child.value, ast.Constant)
                        and child.value.value is None):
                    if opp_free_line is None or child.lineno > opp_free_line:
                        opp_free_line = child.lineno

                if isinstance(child, ast.If):
                    src_segment = ast.dump(child.test)
                    if "run_mcts_train" in src_segment:
                        mcts_line = child.lineno

            self.assertIsNotNone(opp_free_line,
                                 "Could not find `opp_scenarios = None`")
            self.assertIsNotNone(mcts_line,
                                 "Could not find run_mcts_train block")

            self.assertLess(
                opp_free_line, mcts_line,
                f"opp_scenarios freed at line {opp_free_line} but MCTS starts "
                f"at line {mcts_line}.")
            break
        else:
            self.fail("Could not find main() in pipeline.py")


if __name__ == "__main__":
    unittest.main()
