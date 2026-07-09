"""Tests for the checkpoint loading priority system and checkpoint_io phase gating.

Covers:
  - _find_best_checkpoint priority order and latest-timestamp selection
  - make_checkpoint phase validation (VALID_PHASES)
  - restore_optim_sched phase gating (match / mismatch / strict)
  - trainable_signature gating for optimizer restore
  - Legacy checkpoint handling (no phase key)
  - legacy_path_phase_hint acceptance path
  - _snapshot_sched_config / _restore_sched_config counter preservation
  - _SCHED_COUNTER_KEYS membership

Run from the repo root (versions/v6/):
    python -m pytest tests/test_checkpoint_chain.py -v
or:
    python tests/test_checkpoint_chain.py
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# ---------------------------------------------------------------------------
# Make sure the package root (versions/v6/) is importable regardless of how
# the test runner sets sys.path.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # versions/v6/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from agent.agent import ASI
from agent.train_scenarios._checkpoint_io import (
    VALID_PHASES,
    _SCHED_COUNTER_KEYS,
    _restore_sched_config,
    _snapshot_sched_config,
    make_checkpoint,
    param_signature,
    restore_optim_sched,
    trainable_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> nn.Module:
    """Two-layer MLP — tiny enough that tests run fast."""
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def _adam(model: nn.Module) -> optim.Adam:
    return optim.Adam(model.parameters(), lr=1e-3)


def _cosine(opt: optim.Adam, T_max: int = 100, eta_min: float = 1e-6) -> CosineAnnealingLR:
    return CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)


def _sequential(opt: optim.Adam, T_max: int = 100) -> SequentialLR:
    warmup = LinearLR(opt, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(opt, T_max=T_max, eta_min=1e-6)
    return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[10])


def _write_dummy_checkpoint(path: str, content: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(content, path)


def _make_real_checkpoint(model, phase, optimizer=None, scheduler=None):
    """Build a valid checkpoint dict via make_checkpoint."""
    opt = optimizer or _adam(model)
    sched = scheduler or _cosine(opt)
    return make_checkpoint(
        phase=phase,
        model=model,
        optimizer=opt,
        scheduler=sched,
        norm_stats={"test": 1.0},
        val_loss=0.5,
    )


# ---------------------------------------------------------------------------
# 1.  _find_best_checkpoint — priority order
# ---------------------------------------------------------------------------

class TestFindBestCheckpointPriorityOrder(unittest.TestCase):
    """Higher-stage scenarios shadow lower-stage ones."""

    PRIORITY_ORDER = [
        "mcts_predict",
        "opponent_action_predict",
        "modelling_predict",
        "gto_predict",
        "gto_probs_predict",
        "gto_ev_predict",
    ]

    def _make_agent_dir(self, tmp: str, present_scenarios: list[str]) -> str:
        for scenario in present_scenarios:
            ckpt_path = os.path.join(tmp, scenario, "20240101_000000", "best.pt")
            _write_dummy_checkpoint(ckpt_path, {"model_state_dict": {}})
        return tmp

    def test_mcts_wins_over_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_agent_dir(tmp, self.PRIORITY_ORDER)
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("mcts_predict", result)

    def test_opponent_action_wins_without_mcts(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = [s for s in self.PRIORITY_ORDER if s != "mcts_predict"]
            self._make_agent_dir(tmp, scenarios)
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("opponent_action_predict", result)

    def test_modelling_wins_without_higher_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = ["modelling_predict", "gto_predict", "gto_probs_predict", "gto_ev_predict"]
            self._make_agent_dir(tmp, scenarios)
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("modelling_predict", result)

    def test_gto_predict_wins_without_modelling_and_above(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = ["gto_predict", "gto_probs_predict", "gto_ev_predict"]
            self._make_agent_dir(tmp, scenarios)
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("gto_predict", result)

    def test_gto_probs_wins_without_gto_predict_and_above(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = ["gto_probs_predict", "gto_ev_predict"]
            self._make_agent_dir(tmp, scenarios)
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("gto_probs_predict", result)

    def test_gto_ev_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_agent_dir(tmp, ["gto_ev_predict"])
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("gto_ev_predict", result)

    def test_empty_agent_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ASI._find_best_checkpoint(tmp)
            self.assertIsNone(result)

    def test_scenario_dir_with_no_best_pt_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            # mcts_predict dir exists but has no best.pt
            empty_ts = os.path.join(tmp, "mcts_predict", "20240101_000000")
            os.makedirs(empty_ts)
            # gto_ev_predict has a real checkpoint
            gto_path = os.path.join(tmp, "gto_ev_predict", "20240101_000000", "best.pt")
            _write_dummy_checkpoint(gto_path, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("gto_ev_predict", result)

    def test_each_phase_in_priority_order(self):
        """Verify each scenario beats all phases below it."""
        for i, winner in enumerate(self.PRIORITY_ORDER):
            losers = self.PRIORITY_ORDER[i + 1 :]
            if not losers:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                for scenario in [winner] + losers:
                    ckpt_path = os.path.join(tmp, scenario, "20240101_000000", "best.pt")
                    _write_dummy_checkpoint(ckpt_path, {"model_state_dict": {}})
                result = ASI._find_best_checkpoint(tmp)
                self.assertIn(
                    winner,
                    result,
                    msg=f"{winner!r} should beat losers {losers}; got {result!r}",
                )


# ---------------------------------------------------------------------------
# 2.  _find_best_checkpoint — latest timestamp selection
# ---------------------------------------------------------------------------

class TestFindBestCheckpointLatestTimestamp(unittest.TestCase):

    def test_latest_timestamp_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ts in ("20230101_000000", "20240601_120000", "20231231_235959"):
                p = os.path.join(tmp, "gto_ev_predict", ts, "best.pt")
                _write_dummy_checkpoint(p, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("20240601_120000", result)

    def test_single_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = "20240101_090000"
            p = os.path.join(tmp, "mcts_predict", ts, "best.pt")
            _write_dummy_checkpoint(p, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn(ts, result)

    def test_higher_phase_with_older_timestamp_beats_lower_phase_with_newer(self):
        """Phase priority is checked before timestamp."""
        with tempfile.TemporaryDirectory() as tmp:
            # mcts_predict with old timestamp
            p1 = os.path.join(tmp, "mcts_predict", "20220101_000000", "best.pt")
            _write_dummy_checkpoint(p1, {"model_state_dict": {}})
            # gto_ev_predict with very new timestamp
            p2 = os.path.join(tmp, "gto_ev_predict", "20991231_235959", "best.pt")
            _write_dummy_checkpoint(p2, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("mcts_predict", result)

    def test_multiple_timestamps_in_same_phase_reverse_sorted(self):
        """Reverse-sorted directory listing: lexicographically largest wins."""
        with tempfile.TemporaryDirectory() as tmp:
            timestamps = [
                "20240101_000000",
                "20240102_000000",
                "20240103_000000",
            ]
            for ts in timestamps:
                p = os.path.join(tmp, "gto_predict", ts, "best.pt")
                _write_dummy_checkpoint(p, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("20240103_000000", result)
            self.assertNotIn("20240101_000000", result.replace("20240103_000000", ""))

    def test_subdirectory_without_best_pt_is_skipped_in_same_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            # newest dir has no best.pt
            empty_dir = os.path.join(tmp, "gto_ev_predict", "20991231_000000")
            os.makedirs(empty_dir)
            # older dir has best.pt
            p = os.path.join(tmp, "gto_ev_predict", "20230101_000000", "best.pt")
            _write_dummy_checkpoint(p, {"model_state_dict": {}})
            result = ASI._find_best_checkpoint(tmp)
            self.assertIn("20230101_000000", result)


# ---------------------------------------------------------------------------
# 3.  make_checkpoint — phase validation
# ---------------------------------------------------------------------------

class TestMakeCheckpointPhaseValidation(unittest.TestCase):

    def setUp(self):
        self.model = _tiny_model()
        self.opt = _adam(self.model)
        self.sched = _cosine(self.opt)

    def _call(self, phase):
        return make_checkpoint(
            phase=phase,
            model=self.model,
            optimizer=self.opt,
            scheduler=self.sched,
            norm_stats={},
            val_loss=0.1,
        )

    def test_all_valid_phases_accepted(self):
        for phase in VALID_PHASES:
            with self.subTest(phase=phase):
                ckpt = self._call(phase)
                self.assertEqual(ckpt["phase"], phase)

    def test_invalid_phase_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call("unknown_phase")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call("")

    def test_partial_phase_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call("mcts")

    def test_phase_with_typo_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call("gto_ev_predict_typo")

    def test_checkpoint_contains_required_keys(self):
        ckpt = self._call("gto_ev_predict")
        for key in ("phase", "param_signature", "trainable_signature",
                    "model_state_dict", "optimizer_state_dict",
                    "scheduler_state_dict", "norm_stats", "val_loss"):
            with self.subTest(key=key):
                self.assertIn(key, ckpt)

    def test_extra_keys_stored_without_collision(self):
        ckpt = make_checkpoint(
            phase="gto_ev_predict",
            model=self.model,
            optimizer=self.opt,
            scheduler=self.sched,
            norm_stats={},
            val_loss=0.1,
            extra={"step": 42, "epoch": 3},
        )
        self.assertEqual(ckpt["step"], 42)
        self.assertEqual(ckpt["epoch"], 3)

    def test_extra_keys_colliding_with_reserved_raises(self):
        with self.assertRaises(ValueError):
            make_checkpoint(
                phase="gto_ev_predict",
                model=self.model,
                optimizer=self.opt,
                scheduler=self.sched,
                norm_stats={},
                val_loss=0.1,
                extra={"phase": "collision"},
            )

    def test_valid_phases_tuple_contains_all_expected(self):
        expected = {
            "gto_ev_predict",
            "gto_probs_predict",
            "gto_predict",
            "modelling_predict",
            "opponent_action_predict",
            "mcts_predict",
        }
        self.assertEqual(set(VALID_PHASES), expected)


# ---------------------------------------------------------------------------
# 4.  restore_optim_sched — phase gating
# ---------------------------------------------------------------------------

class TestRestoreOptimSchedPhaseGating(unittest.TestCase):

    def setUp(self):
        self.model = _tiny_model()
        self.opt = _adam(self.model)
        self.sched = _cosine(self.opt)

    def _make_ckpt(self, phase):
        return _make_real_checkpoint(self.model, phase, self.opt, self.sched)

    def _call(self, ckpt, expected_phase, strict=False, legacy_hint=None):
        return restore_optim_sched(
            optimizer=self.opt,
            scheduler=self.sched,
            ckpt=ckpt,
            expected_phase=expected_phase,
            model=self.model,
            strict=strict,
            legacy_path_phase_hint=legacy_hint,
        )

    def test_matching_phase_restores_opt_and_sched(self):
        ckpt = self._make_ckpt("gto_ev_predict")
        restored_opt, restored_sched, _ = self._call(ckpt, "gto_ev_predict")
        self.assertTrue(restored_opt)
        self.assertTrue(restored_sched)

    def test_phase_mismatch_skips_restore(self):
        ckpt = self._make_ckpt("gto_ev_predict")
        restored_opt, restored_sched, reason = self._call(ckpt, "mcts_predict")
        self.assertFalse(restored_opt)
        self.assertFalse(restored_sched)
        self.assertIn("mismatch", reason)

    def test_phase_mismatch_strict_raises(self):
        ckpt = self._make_ckpt("gto_ev_predict")
        with self.assertRaises(RuntimeError):
            self._call(ckpt, "mcts_predict", strict=True)

    def test_all_valid_phases_match_themselves(self):
        for phase in VALID_PHASES:
            with self.subTest(phase=phase):
                model = _tiny_model()
                opt = _adam(model)
                sched = _cosine(opt)
                ckpt = _make_real_checkpoint(model, phase, opt, sched)
                r_opt, r_sched, _ = restore_optim_sched(
                    optimizer=opt,
                    scheduler=sched,
                    ckpt=ckpt,
                    expected_phase=phase,
                    model=model,
                    strict=False,
                )
                self.assertTrue(r_opt, f"optimizer not restored for phase {phase}")
                self.assertTrue(r_sched, f"scheduler not restored for phase {phase}")

    def test_phase_in_reason_string_on_mismatch(self):
        ckpt = self._make_ckpt("gto_probs_predict")
        _, _, reason = self._call(ckpt, "modelling_predict")
        self.assertIn("gto_probs_predict", reason)
        self.assertIn("modelling_predict", reason)

    def test_cross_phase_mismatch_all_pairs(self):
        """Every off-diagonal pair skips restore (non-strict)."""
        for src_phase in VALID_PHASES:
            for dst_phase in VALID_PHASES:
                if src_phase == dst_phase:
                    continue
                model = _tiny_model()
                opt = _adam(model)
                sched = _cosine(opt)
                ckpt = _make_real_checkpoint(model, src_phase, opt, sched)
                r_opt, r_sched, _ = restore_optim_sched(
                    optimizer=opt,
                    scheduler=sched,
                    ckpt=ckpt,
                    expected_phase=dst_phase,
                    model=model,
                    strict=False,
                )
                self.assertFalse(
                    r_opt,
                    msg=f"opt should be skipped when src={src_phase!r} dst={dst_phase!r}",
                )


# ---------------------------------------------------------------------------
# 5.  restore_optim_sched — trainable_signature gating
# ---------------------------------------------------------------------------

class TestRestoreOptimSchedSignatureGating(unittest.TestCase):

    def test_signature_match_restores_optimizer(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = _make_real_checkpoint(model, "gto_ev_predict", opt, sched)
        # Re-create same arch — signatures will match
        restored_opt, _, _ = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
        )
        self.assertTrue(restored_opt)

    def test_signature_mismatch_skips_optimizer(self):
        model_a = _tiny_model()
        opt_a = _adam(model_a)
        sched_a = _cosine(opt_a)
        ckpt = _make_real_checkpoint(model_a, "gto_ev_predict", opt_a, sched_a)

        # Different architecture
        model_b = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
        opt_b = _adam(model_b)
        sched_b = _cosine(opt_b)

        restored_opt, _, reason = restore_optim_sched(
            optimizer=opt_b,
            scheduler=sched_b,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model_b,
            strict=False,
        )
        self.assertFalse(restored_opt)
        self.assertIn("mismatch", reason)

    def test_signature_mismatch_strict_raises(self):
        model_a = _tiny_model()
        opt_a = _adam(model_a)
        sched_a = _cosine(opt_a)
        ckpt = _make_real_checkpoint(model_a, "gto_ev_predict", opt_a, sched_a)

        model_b = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
        opt_b = _adam(model_b)
        sched_b = _cosine(opt_b)

        with self.assertRaises(RuntimeError):
            restore_optim_sched(
                optimizer=opt_b,
                scheduler=sched_b,
                ckpt=ckpt,
                expected_phase="gto_ev_predict",
                model=model_b,
                strict=True,
            )

    def test_frozen_params_change_trainable_signature(self):
        """Freezing layers changes trainable_signature, blocking optimizer restore."""
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = _make_real_checkpoint(model, "modelling_predict", opt, sched)

        # Freeze first layer — trainable_signature now differs
        model_frozen = _tiny_model()
        for param in model_frozen[0].parameters():
            param.requires_grad = False
        opt_frozen = optim.Adam(
            filter(lambda p: p.requires_grad, model_frozen.parameters()), lr=1e-3
        )
        sched_frozen = _cosine(opt_frozen)

        restored_opt, _, _ = restore_optim_sched(
            optimizer=opt_frozen,
            scheduler=sched_frozen,
            ckpt=ckpt,
            expected_phase="modelling_predict",
            model=model_frozen,
            strict=False,
        )
        self.assertFalse(restored_opt)

    def test_trainable_signature_stable_across_identical_models(self):
        model1 = _tiny_model()
        model2 = _tiny_model()
        self.assertEqual(trainable_signature(model1), trainable_signature(model2))

    def test_param_signature_includes_all_params(self):
        model = _tiny_model()
        sig = param_signature(model)
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 32)  # md5 hex digest length

    def test_trainable_signature_differs_from_param_signature_when_frozen(self):
        model = _tiny_model()
        # Freeze one layer
        for p in model[0].parameters():
            p.requires_grad = False
        p_sig = param_signature(model)
        t_sig = trainable_signature(model)
        # param sig covers all params (frozen + trainable), trainable only covers unfrozen
        self.assertNotEqual(p_sig, t_sig)

    def test_ckpt_missing_trainable_signature_with_phase_tag_skips_opt(self):
        """A checkpoint that has a phase tag but no trainable_signature is
        handled defensively — optimizer not restored."""
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = _make_real_checkpoint(model, "gto_ev_predict", opt, sched)
        del ckpt["trainable_signature"]  # simulate partial-write / old format

        restored_opt, _, _ = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
        )
        self.assertFalse(restored_opt)

    def test_ckpt_missing_trainable_signature_strict_raises(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = _make_real_checkpoint(model, "gto_ev_predict", opt, sched)
        del ckpt["trainable_signature"]

        with self.assertRaises(RuntimeError):
            restore_optim_sched(
                optimizer=opt,
                scheduler=sched,
                ckpt=ckpt,
                expected_phase="gto_ev_predict",
                model=model,
                strict=True,
            )


# ---------------------------------------------------------------------------
# 6.  Legacy checkpoint handling (no phase key)
# ---------------------------------------------------------------------------

class TestLegacyCheckpointHandling(unittest.TestCase):

    def _legacy_ckpt(self, model):
        """Build a checkpoint dict WITHOUT a phase key."""
        opt = _adam(model)
        sched = _cosine(opt)
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "norm_stats": {},
            "val_loss": 0.5,
        }

    def test_no_phase_key_skips_optim_sched(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = self._legacy_ckpt(model)

        restored_opt, restored_sched, reason = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
        )
        self.assertFalse(restored_opt)
        self.assertFalse(restored_sched)
        self.assertIn("legacy", reason.lower())

    def test_no_phase_key_strict_raises(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = self._legacy_ckpt(model)

        with self.assertRaises(RuntimeError):
            restore_optim_sched(
                optimizer=opt,
                scheduler=sched,
                ckpt=ckpt,
                expected_phase="gto_ev_predict",
                model=model,
                strict=True,
            )

    def test_legacy_hint_matching_expected_phase_restores(self):
        """legacy_path_phase_hint matching expected_phase → accept the checkpoint."""
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = self._legacy_ckpt(model)

        restored_opt, restored_sched, _ = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
            legacy_path_phase_hint="gto_ev_predict",
        )
        # Optimizer restore depends on trainable_signature availability;
        # legacy ckpt has no trainable_signature → falls into legacy path hint branch
        # which loads the optimizer directly.
        self.assertTrue(restored_opt)
        self.assertTrue(restored_sched)

    def test_legacy_hint_wrong_phase_still_skips(self):
        """legacy_path_phase_hint that doesn't match expected_phase → still skip."""
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = self._legacy_ckpt(model)

        restored_opt, restored_sched, reason = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="mcts_predict",
            model=model,
            strict=False,
            legacy_path_phase_hint="gto_ev_predict",  # hint != expected
        )
        self.assertFalse(restored_opt)
        self.assertFalse(restored_sched)
        self.assertIn("legacy", reason.lower())

    def test_legacy_hint_none_with_no_phase_key_skips(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt)
        ckpt = self._legacy_ckpt(model)

        restored_opt, restored_sched, _ = restore_optim_sched(
            optimizer=opt,
            scheduler=sched,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
            legacy_path_phase_hint=None,
        )
        self.assertFalse(restored_opt)
        self.assertFalse(restored_sched)


# ---------------------------------------------------------------------------
# 7.  _snapshot_sched_config / _restore_sched_config
# ---------------------------------------------------------------------------

class TestSnapshotRestoreSchedConfig(unittest.TestCase):

    def test_snapshot_excludes_optimizer(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt, T_max=50, eta_min=1e-7)
        snap = _snapshot_sched_config(sched)
        self.assertNotIn("optimizer", snap)

    def test_snapshot_captures_t_max(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = CosineAnnealingLR(opt, T_max=77, eta_min=5e-7)
        snap = _snapshot_sched_config(sched)
        self.assertEqual(snap["T_max"], 77)
        self.assertAlmostEqual(snap["eta_min"], 5e-7)

    def test_snapshot_excludes_counter_keys(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _cosine(opt, T_max=100)
        # Advance the scheduler so counters are non-trivial
        for _ in range(5):
            sched.step()
        snap = _snapshot_sched_config(sched)
        for key in _SCHED_COUNTER_KEYS:
            self.assertNotIn(key, snap, msg=f"Counter key {key!r} should not be in snapshot")

    def test_restore_preserves_counters_from_load_state_dict(self):
        """After load_state_dict, _restore_sched_config re-imposes fresh T_max
        while counters (last_epoch, _step_count) from the checkpoint survive."""
        model = _tiny_model()
        opt_old = _adam(model)
        sched_old = CosineAnnealingLR(opt_old, T_max=200, eta_min=1e-6)
        # Advance old scheduler
        for _ in range(20):
            sched_old.step()
        old_state = sched_old.state_dict()
        saved_last_epoch = old_state["last_epoch"]

        # New scheduler with different T_max
        opt_new = _adam(model)
        sched_new = CosineAnnealingLR(opt_new, T_max=500, eta_min=2e-6)
        snap = _snapshot_sched_config(sched_new)  # captures T_max=500, eta_min=2e-6

        sched_new.load_state_dict(old_state)  # overwrites T_max with 200
        # After blind load, T_max would be 200; restore re-imposes 500
        _restore_sched_config(sched_new, snap)

        self.assertEqual(sched_new.T_max, 500)
        self.assertAlmostEqual(sched_new.eta_min, 2e-6)
        # Counter from the old checkpoint is preserved
        self.assertEqual(sched_new.last_epoch, saved_last_epoch)

    def test_restore_returns_diffs_when_config_changed(self):
        model = _tiny_model()
        opt_a = _adam(model)
        sched_a = CosineAnnealingLR(opt_a, T_max=100, eta_min=1e-6)
        old_state = sched_a.state_dict()

        opt_b = _adam(model)
        sched_b = CosineAnnealingLR(opt_b, T_max=999, eta_min=9e-9)
        snap = _snapshot_sched_config(sched_b)

        sched_b.load_state_dict(old_state)
        diffs = _restore_sched_config(sched_b, snap)
        # T_max changed 100→999, eta_min changed — should report diffs
        self.assertTrue(len(diffs) > 0)

    def test_restore_returns_empty_diffs_when_config_unchanged(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        state = sched.state_dict()

        # Same config
        opt2 = _adam(model)
        sched2 = CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
        snap = _snapshot_sched_config(sched2)
        sched2.load_state_dict(state)
        diffs = _restore_sched_config(sched2, snap)
        self.assertEqual(diffs, [])

    def test_snapshot_sequential_lr_captures_nested_schedulers(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = _sequential(opt, T_max=100)
        snap = _snapshot_sched_config(sched)
        self.assertIn("_schedulers", snap)
        self.assertEqual(len(snap["_schedulers"]), 2)
        # Inner cosine should have T_max captured
        cosine_snap = snap["_schedulers"][1]
        self.assertEqual(cosine_snap["T_max"], 100)

    def test_restore_sequential_lr_re_imposes_nested_config(self):
        model = _tiny_model()
        opt_old = _adam(model)
        sched_old = _sequential(opt_old, T_max=50)
        for _ in range(5):
            sched_old.step()
        old_state = sched_old.state_dict()

        opt_new = _adam(model)
        sched_new = _sequential(opt_new, T_max=400)
        snap = _snapshot_sched_config(sched_new)  # nested T_max=400

        sched_new.load_state_dict(old_state)  # overwrites nested T_max with 50
        _restore_sched_config(sched_new, snap)

        # Inner cosine (index 1) should have T_max=400
        inner_cosine = sched_new._schedulers[1]
        self.assertEqual(inner_cosine.T_max, 400)


# ---------------------------------------------------------------------------
# 8.  _SCHED_COUNTER_KEYS constant
# ---------------------------------------------------------------------------

class TestSchedCounterKeys(unittest.TestCase):

    def test_counter_keys_are_frozenset(self):
        self.assertIsInstance(_SCHED_COUNTER_KEYS, frozenset)

    def test_contains_last_epoch(self):
        self.assertIn("last_epoch", _SCHED_COUNTER_KEYS)

    def test_contains_step_count(self):
        self.assertIn("_step_count", _SCHED_COUNTER_KEYS)

    def test_contains_last_lr(self):
        self.assertIn("_last_lr", _SCHED_COUNTER_KEYS)

    def test_exactly_three_keys(self):
        self.assertEqual(len(_SCHED_COUNTER_KEYS), 3)

    def test_does_not_contain_config_attrs(self):
        for key in ("T_max", "eta_min", "base_lrs", "milestones", "total_iters",
                    "optimizer", "_schedulers"):
            self.assertNotIn(key, _SCHED_COUNTER_KEYS, msg=f"{key!r} should NOT be a counter")


# ---------------------------------------------------------------------------
# 9.  Scheduler counter keys preserved through restore_optim_sched flow
# ---------------------------------------------------------------------------

class TestRestoreOptimSchedSchedulerCounters(unittest.TestCase):
    """Integration: restore_optim_sched preserves counters but re-imposes config."""

    def test_last_epoch_preserved_after_restore(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
        for _ in range(30):
            sched.step()
        saved_last_epoch = sched.last_epoch

        ckpt = make_checkpoint(
            phase="gto_ev_predict",
            model=model,
            optimizer=opt,
            scheduler=sched,
            norm_stats={},
            val_loss=0.1,
        )

        # New identical scheduler to restore into
        opt2 = _adam(model)
        sched2 = CosineAnnealingLR(opt2, T_max=200, eta_min=1e-6)

        restore_optim_sched(
            optimizer=opt2,
            scheduler=sched2,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
        )

        self.assertEqual(sched2.last_epoch, saved_last_epoch)

    def test_t_max_from_fresh_scheduler_wins_over_checkpoint(self):
        model = _tiny_model()
        opt_old = _adam(model)
        sched_old = CosineAnnealingLR(opt_old, T_max=50, eta_min=1e-6)
        for _ in range(10):
            sched_old.step()

        ckpt = make_checkpoint(
            phase="gto_ev_predict",
            model=model,
            optimizer=opt_old,
            scheduler=sched_old,
            norm_stats={},
            val_loss=0.1,
        )

        # New scheduler with different T_max
        opt_new = _adam(model)
        sched_new = CosineAnnealingLR(opt_new, T_max=9999, eta_min=1e-6)

        restore_optim_sched(
            optimizer=opt_new,
            scheduler=sched_new,
            ckpt=ckpt,
            expected_phase="gto_ev_predict",
            model=model,
            strict=False,
        )

        # Fresh T_max must win
        self.assertEqual(sched_new.T_max, 9999)
        # Counter from checkpoint
        self.assertEqual(sched_new.last_epoch, 10)

    def test_sched_not_restored_when_phase_mismatches(self):
        model = _tiny_model()
        opt = _adam(model)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        ckpt = make_checkpoint(
            phase="gto_ev_predict",
            model=model,
            optimizer=opt,
            scheduler=sched,
            norm_stats={},
            val_loss=0.1,
        )

        opt2 = _adam(model)
        sched2 = CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
        initial_last_epoch = sched2.last_epoch

        _, restored_sched, _ = restore_optim_sched(
            optimizer=opt2,
            scheduler=sched2,
            ckpt=ckpt,
            expected_phase="mcts_predict",  # different phase
            model=model,
            strict=False,
        )

        self.assertFalse(restored_sched)
        self.assertEqual(sched2.last_epoch, initial_last_epoch)


# ---------------------------------------------------------------------------
# 10.  make_checkpoint round-trip via file system
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip(unittest.TestCase):
    """Save to disk, reload, pass to restore_optim_sched — full cycle."""

    def test_save_and_reload_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            opt = _adam(model)
            sched = _cosine(opt, T_max=50)
            # Advance a bit
            for _ in range(5):
                opt.zero_grad()
                loss = model(torch.randn(4, 4)).sum()
                loss.backward()
                opt.step()
                sched.step()

            ckpt = make_checkpoint(
                phase="gto_predict",
                model=model,
                optimizer=opt,
                scheduler=sched,
                norm_stats={"mean": 0.5, "std": 1.0},
                val_loss=0.42,
            )
            path = os.path.join(tmp, "best.pt")
            torch.save(ckpt, path)

            loaded = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(loaded["phase"], "gto_predict")
            self.assertAlmostEqual(loaded["val_loss"], 0.42)

            # Restore into fresh model/opt/sched
            model2 = _tiny_model()
            model2.load_state_dict(loaded["model_state_dict"], strict=True)
            opt2 = _adam(model2)
            sched2 = _cosine(opt2, T_max=50)

            r_opt, r_sched, _ = restore_optim_sched(
                optimizer=opt2,
                scheduler=sched2,
                ckpt=loaded,
                expected_phase="gto_predict",
                model=model2,
                strict=False,
            )
            self.assertTrue(r_opt)
            self.assertTrue(r_sched)
            self.assertEqual(sched2.last_epoch, 5)

    def test_find_best_checkpoint_with_real_file(self):
        """_find_best_checkpoint returns a path that actually exists."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            ckpt = {"model_state_dict": model.state_dict()}
            ts = "20240601_120000"
            path = os.path.join(tmp, "mcts_predict", ts, "best.pt")
            _write_dummy_checkpoint(path, ckpt)

            result = ASI._find_best_checkpoint(tmp)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))
            self.assertIn("mcts_predict", result)
            self.assertIn(ts, result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
