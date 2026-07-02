"""
Tests for MCTS value target finalization and normalization math.

Tests the mathematical behavior of:
  - _finalize_value_targets(per_agent_examples, agents_list, search_scales,
                            alpha, clip_val, big_blind, log)
  - _robust_scale(chips, fallback)

Source: versions/v6/agent/mcts/collect.py
Run from: /home/dev/ContinuousLearning/versions/v6/
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.mcts.collect import (
    _finalize_value_targets,
    _robust_scale,
    MCTSTrainingExample,
    ChainStep,
)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _null_log(*args, **kwargs):
    pass


def _make_example(
    value_target=0.0,
    root_q_ratio=float("nan"),
    chain=None,
    terminal_targets=None,
):
    ex = MCTSTrainingExample()
    ex.value_target = value_target
    ex.root_q_ratio = root_q_ratio
    ex.action_target = [0.5, 0.5, 0.0]
    ex.events = []
    ex.chain = chain if chain is not None else []
    ex.terminal_targets = terminal_targets if terminal_targets is not None else []
    return ex


def _make_chain_step(
    value_target=0.0,
    root_q_ratio=float("nan"),
    is_hero=True,
):
    step = ChainStep(
        action_taken=0,
        target_distribution=[0.5, 0.5, 0.0],
        is_hero=is_hero,
        events_at_step=[],
        value_target=value_target,
        root_q_ratio=root_q_ratio,
    )
    return step


def _run_finalize(
    examples,
    search_scale,
    alpha,
    clip_val,
    big_blind=10.0,
    norm_stats=None,
    agent_name="agent_a",
    ema=0.0,
):
    """Thin wrapper: pack into the dict-of-lists structure _finalize_value_targets
    expects, run it, and return the (possibly mutated) examples list.

    `ema=0.0` (default) keeps an existing `mcts_value_scale` unchanged —
    the fixed-scale setting the math tests below rely on. Pass `ema>0`
    to exercise the per-cycle EMA scale update."""
    ns = dict(norm_stats) if norm_stats else {}
    agents_list = [{"name": agent_name, "norm_stats": ns}]
    per_agent_examples = {agent_name: examples}
    search_scales = {agent_name: search_scale}
    _finalize_value_targets(
        per_agent_examples=per_agent_examples,
        agents_list=agents_list,
        search_scales=search_scales,
        alpha=alpha,
        clip_val=clip_val,
        big_blind=float(big_blind),
        log=_null_log,
        ema=ema,
    )
    return examples, ns


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rescale identity: when search_scale == new_scale, rescale_q = 1.0
#    and the Q values pass through the scale conversion unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class TestRescaleIdentity:
    """When search_scale == new_scale the Q rescaling is a no-op (×1)."""

    def test_q_value_unchanged_under_identity_rescale(self):
        """q_in_new = root_q_ratio * (search_scale / new_scale)
        == root_q_ratio when search_scale == new_scale."""
        q = 0.7
        realized = 0.0  # pure TD: alpha=1, realized does not contribute
        ex = _make_example(value_target=realized, root_q_ratio=q)
        # Supply norm_stats with mcts_value_scale already set so we don't
        # re-bootstrap; the scale matches search_scale exactly.
        scale = 50.0
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex],
            search_scale=scale,
            alpha=1.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        # Pure TD (alpha=1): value_target = q_in_new = q * (scale/scale) = q
        assert abs(examples[0].value_target - q) < 1e-6, (
            f"Expected {q}, got {examples[0].value_target}"
        )

    def test_rescale_ratio_is_one_when_scales_equal(self):
        """Any root_q_ratio should come out unmodified (before clip) when scales match."""
        for q in [-2.0, 0.0, 1.5, 3.0]:
            ex = _make_example(value_target=0.0, root_q_ratio=q)
            scale = 100.0
            ns = {"mcts_value_scale": scale}
            examples, _ = _run_finalize(
                [ex],
                search_scale=scale,
                alpha=1.0,
                clip_val=100.0,  # wide clip so we see raw value
                norm_stats=ns,
            )
            assert abs(examples[0].value_target - q) < 1e-6, (
                f"q={q}: expected identity, got {examples[0].value_target}"
            )

    def test_rescale_ratio_scaled_when_scales_differ(self):
        """When search_scale != new_scale the Q is rescaled proportionally."""
        q = 1.0
        search_scale = 50.0
        new_scale = 100.0  # new_scale supplied via norm_stats
        expected_q_in_new = q * (search_scale / new_scale)  # == 0.5

        ex = _make_example(value_target=0.0, root_q_ratio=q)
        ns = {"mcts_value_scale": new_scale}
        examples, _ = _run_finalize(
            [ex],
            search_scale=search_scale,
            alpha=1.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        assert abs(examples[0].value_target - expected_q_in_new) < 1e-6, (
            f"Expected {expected_q_in_new}, got {examples[0].value_target}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pure MC (alpha=0): value_target = realized_chips / new_scale (clamped)
# ─────────────────────────────────────────────────────────────────────────────

class TestPureMC:
    """alpha=0 → pure realized, Q ignored."""

    def test_pure_mc_uses_realized_not_q(self):
        realized = 80.0   # raw chip delta
        q = 999.0          # should be completely ignored
        scale = 100.0
        ns = {"mcts_value_scale": scale}
        ex = _make_example(value_target=realized, root_q_ratio=q)
        examples, _ = _run_finalize(
            [ex],
            search_scale=scale,
            alpha=0.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        expected = max(-10.0, min(10.0, realized / scale))
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_pure_mc_q_nan_same_result(self):
        """NaN Q with alpha=0 must produce the same result as a finite Q with alpha=0."""
        realized = 60.0
        scale = 100.0
        ns = {"mcts_value_scale": scale}

        ex_nan = _make_example(value_target=realized, root_q_ratio=float("nan"))
        ex_fin = _make_example(value_target=realized, root_q_ratio=99.0)

        ex_nan_list, _ = _run_finalize(
            [ex_nan], search_scale=scale, alpha=0.0, clip_val=10.0, norm_stats={"mcts_value_scale": scale}
        )
        ex_fin_list, _ = _run_finalize(
            [ex_fin], search_scale=scale, alpha=0.0, clip_val=10.0, norm_stats={"mcts_value_scale": scale}
        )
        assert abs(ex_nan_list[0].value_target - ex_fin_list[0].value_target) < 1e-6

    def test_pure_mc_formula(self):
        """value_target = clamp(realized / new_scale, ±clip)."""
        for realized, scale, clip in [
            (50.0, 100.0, 5.0),
            (-300.0, 100.0, 5.0),
            (10.0, 10.0, 5.0),
        ]:
            ns = {"mcts_value_scale": scale}
            ex = _make_example(value_target=realized, root_q_ratio=float("nan"))
            examples, _ = _run_finalize(
                [ex],
                search_scale=scale,
                alpha=0.0,
                clip_val=clip,
                norm_stats=ns,
            )
            expected = max(-clip, min(clip, realized / scale))
            assert abs(examples[0].value_target - expected) < 1e-6, (
                f"realized={realized}, scale={scale}, clip={clip}: "
                f"expected {expected}, got {examples[0].value_target}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pure TD (alpha=1): value_target = q_in_new (clamped). No realized.
# ─────────────────────────────────────────────────────────────────────────────

class TestPureTD:
    """alpha=1 → pure Q, realized ignored (unless Q is NaN)."""

    def test_pure_td_uses_q_not_realized(self):
        q = 0.8
        realized = 999.0  # should be completely ignored
        scale = 100.0
        ns = {"mcts_value_scale": scale}
        ex = _make_example(value_target=realized, root_q_ratio=q)
        examples, _ = _run_finalize(
            [ex],
            search_scale=scale,
            alpha=1.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        # q_in_new = q * (scale/scale) = q (identity rescale)
        expected = max(-10.0, min(10.0, q))
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_pure_td_with_rescale(self):
        q = 1.0
        search_scale = 50.0
        new_scale = 100.0
        ns = {"mcts_value_scale": new_scale}
        ex = _make_example(value_target=999.0, root_q_ratio=q)
        examples, _ = _run_finalize(
            [ex],
            search_scale=search_scale,
            alpha=1.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        expected = q * (search_scale / new_scale)  # == 0.5
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_pure_td_nan_falls_back_to_realized(self):
        """When Q is NaN, even alpha=1 must fall back to pure MC."""
        realized = 50.0
        scale = 100.0
        ns = {"mcts_value_scale": scale}
        ex = _make_example(value_target=realized, root_q_ratio=float("nan"))
        examples, _ = _run_finalize(
            [ex],
            search_scale=scale,
            alpha=1.0,
            clip_val=10.0,
            norm_stats=ns,
        )
        expected = max(-10.0, min(10.0, realized / scale))
        assert abs(examples[0].value_target - expected) < 1e-6, (
            f"NaN Q + alpha=1 should fall back to MC; "
            f"expected {expected}, got {examples[0].value_target}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid blend (alpha=0.5):
#    value_target = 0.5 * q_in_new + 0.5 * realized_in_new, clamped
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridBlend:
    """alpha=0.5 → equal weight on Q and realized."""

    def test_hybrid_blend_formula(self):
        realized = 60.0
        q = 0.4
        scale = 100.0
        ns = {"mcts_value_scale": scale}
        clip = 5.0
        ex = _make_example(value_target=realized, root_q_ratio=q)
        examples, _ = _run_finalize(
            [ex],
            search_scale=scale,
            alpha=0.5,
            clip_val=clip,
            norm_stats=ns,
        )
        realized_in_new = realized / scale     # 0.6
        q_in_new = q * (scale / scale)        # 0.4 (identity rescale)
        expected_blend = 0.5 * q_in_new + 0.5 * realized_in_new  # 0.5
        expected = max(-clip, min(clip, expected_blend))
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_hybrid_blend_with_scale_mismatch(self):
        realized = 100.0
        q = 2.0             # in search_scale units
        search_scale = 50.0
        new_scale = 100.0
        clip = 10.0
        ns = {"mcts_value_scale": new_scale}
        ex = _make_example(value_target=realized, root_q_ratio=q)
        examples, _ = _run_finalize(
            [ex],
            search_scale=search_scale,
            alpha=0.5,
            clip_val=clip,
            norm_stats=ns,
        )
        q_in_new = q * (search_scale / new_scale)  # 2.0 * 0.5 = 1.0
        realized_in_new = realized / new_scale      # 1.0
        expected_blend = 0.5 * q_in_new + 0.5 * realized_in_new  # 1.0
        expected = max(-clip, min(clip, expected_blend))
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_arbitrary_alpha_interpolates(self):
        """value_target = alpha*q_in_new + (1-alpha)*realized_in_new."""
        realized = 200.0
        q = -1.0
        scale = 100.0
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ns = {"mcts_value_scale": scale}
            ex = _make_example(value_target=realized, root_q_ratio=q)
            examples, _ = _run_finalize(
                [ex],
                search_scale=scale,
                alpha=alpha,
                clip_val=100.0,  # no clipping — check raw blend
                norm_stats=ns,
            )
            realized_in_new = realized / scale      # 2.0
            q_in_new = q                            # identity rescale → -1.0
            expected = alpha * q_in_new + (1.0 - alpha) * realized_in_new
            assert abs(examples[0].value_target - expected) < 1e-6, (
                f"alpha={alpha}: expected {expected}, got {examples[0].value_target}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clip bounds: value_target always in [-clip, +clip]
# ─────────────────────────────────────────────────────────────────────────────

class TestClipBounds:
    """Output must always be in [-clip_val, +clip_val]."""

    def test_positive_overflow_clipped(self):
        scale = 10.0
        clip = 2.0
        ex = _make_example(value_target=500.0, root_q_ratio=float("nan"))
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        assert examples[0].value_target <= clip

    def test_negative_overflow_clipped(self):
        scale = 10.0
        clip = 2.0
        ex = _make_example(value_target=-500.0, root_q_ratio=float("nan"))
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        assert examples[0].value_target >= -clip

    def test_all_examples_within_clip(self):
        clip = 3.0
        scale = 50.0
        ns = {"mcts_value_scale": scale}
        raw_values = [-1000.0, -200.0, -10.0, 0.0, 10.0, 200.0, 1000.0]
        for alpha in [0.0, 0.5, 1.0]:
            examples_in = [_make_example(value_target=v, root_q_ratio=float("nan"))
                           for v in raw_values]
            examples, _ = _run_finalize(
                examples_in,
                search_scale=scale,
                alpha=alpha,
                clip_val=clip,
                norm_stats={"mcts_value_scale": scale},
            )
            for ex in examples:
                assert -clip <= ex.value_target <= clip, (
                    f"alpha={alpha}: out-of-clip value {ex.value_target}"
                )

    def test_chain_steps_within_clip(self):
        clip = 2.0
        scale = 50.0
        steps = [
            _make_chain_step(value_target=v, root_q_ratio=float("nan"), is_hero=True)
            for v in [-500.0, 0.0, 500.0]
        ]
        ex = _make_example(value_target=0.0, root_q_ratio=float("nan"), chain=steps)
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        for step in examples[0].chain:
            assert -clip <= step.value_target <= clip, (
                f"Chain step out of clip: {step.value_target}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. NaN root_q_ratio falls back to pure MC regardless of alpha
# ─────────────────────────────────────────────────────────────────────────────

class TestNaNQFallback:
    """NaN Q must always produce the same result as alpha=0."""

    def _mc_target(self, realized, scale, clip):
        return max(-clip, min(clip, realized / scale))

    def test_nan_q_alpha_half_equals_pure_mc(self):
        realized = 40.0
        scale = 100.0
        clip = 5.0
        ns_half = {"mcts_value_scale": scale}
        ex_half = _make_example(value_target=realized, root_q_ratio=float("nan"))
        examples_half, _ = _run_finalize(
            [ex_half], search_scale=scale, alpha=0.5, clip_val=clip, norm_stats=ns_half
        )
        expected = self._mc_target(realized, scale, clip)
        assert abs(examples_half[0].value_target - expected) < 1e-6

    def test_nan_q_alpha_one_equals_pure_mc(self):
        realized = -70.0
        scale = 100.0
        clip = 5.0
        ns = {"mcts_value_scale": scale}
        ex = _make_example(value_target=realized, root_q_ratio=float("nan"))
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=1.0, clip_val=clip, norm_stats=ns
        )
        expected = self._mc_target(realized, scale, clip)
        assert abs(examples[0].value_target - expected) < 1e-6

    def test_nan_q_all_alphas_produce_same_mc_result(self):
        """For NaN Q, alpha is irrelevant — always pure MC."""
        realized = 30.0
        scale = 100.0
        clip = 5.0
        mc_expected = self._mc_target(realized, scale, clip)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ns = {"mcts_value_scale": scale}
            ex = _make_example(value_target=realized, root_q_ratio=float("nan"))
            examples, _ = _run_finalize(
                [ex], search_scale=scale, alpha=alpha, clip_val=clip, norm_stats=ns
            )
            assert abs(examples[0].value_target - mc_expected) < 1e-6, (
                f"alpha={alpha}: NaN Q should always give MC result {mc_expected}, "
                f"got {examples[0].value_target}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Chain step value targets
# ─────────────────────────────────────────────────────────────────────────────

class TestChainStepTargets:
    """Chain steps follow the same hybrid formula with per-step is_hero / NaN rules."""

    def test_hero_chain_step_td_blend(self):
        """Hero step with finite Q: blended like root."""
        realized = 60.0
        q = 0.5
        scale = 100.0
        clip = 10.0
        step = _make_chain_step(value_target=realized, root_q_ratio=q, is_hero=True)
        ex = _make_example(value_target=0.0, root_q_ratio=0.0, chain=[step])
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.5, clip_val=clip, norm_stats=ns
        )
        realized_in_new = realized / scale   # 0.6
        q_in_new = q * (scale / scale)      # 0.5
        expected = max(-clip, min(clip, 0.5 * q_in_new + 0.5 * realized_in_new))
        assert abs(examples[0].chain[0].value_target - expected) < 1e-6

    def test_opponent_chain_step_pure_mc(self):
        """Opponent step (is_hero=False): always pure MC regardless of alpha/Q."""
        realized = 80.0
        q = 999.0  # must be ignored
        scale = 100.0
        clip = 10.0
        step = _make_chain_step(value_target=realized, root_q_ratio=q, is_hero=False)
        ex = _make_example(value_target=0.0, root_q_ratio=0.0, chain=[step])
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.5, clip_val=clip, norm_stats=ns
        )
        # Opponent chain step: step_q_in_new = step_realized / new_scale
        # blend = alpha * (realized/scale) + (1-alpha) * (realized/scale) = realized/scale
        expected = max(-clip, min(clip, realized / scale))
        assert abs(examples[0].chain[0].value_target - expected) < 1e-6, (
            f"Opponent chain step should be pure MC; "
            f"expected {expected}, got {examples[0].chain[0].value_target}"
        )

    def test_hero_nan_q_chain_step_pure_mc(self):
        """Hero step with NaN Q: falls back to pure MC."""
        realized = 40.0
        scale = 100.0
        clip = 5.0
        step = _make_chain_step(value_target=realized, root_q_ratio=float("nan"), is_hero=True)
        ex = _make_example(value_target=0.0, root_q_ratio=0.0, chain=[step])
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.5, clip_val=clip, norm_stats=ns
        )
        expected = max(-clip, min(clip, realized / scale))
        assert abs(examples[0].chain[0].value_target - expected) < 1e-6

    def test_chain_pure_mc_when_is_hero_false_all_alphas(self):
        """For opp steps, alpha is always irrelevant (same as pure MC)."""
        realized = 50.0
        scale = 100.0
        clip = 5.0
        mc_expected = max(-clip, min(clip, realized / scale))
        for alpha in [0.0, 0.5, 1.0]:
            step = _make_chain_step(value_target=realized, root_q_ratio=0.8, is_hero=False)
            ex = _make_example(value_target=0.0, root_q_ratio=0.0, chain=[step])
            ns = {"mcts_value_scale": scale}
            examples, _ = _run_finalize(
                [ex], search_scale=scale, alpha=alpha, clip_val=clip, norm_stats=ns
            )
            assert abs(examples[0].chain[0].value_target - mc_expected) < 1e-6, (
                f"alpha={alpha}: opp chain step should be MC {mc_expected}, "
                f"got {examples[0].chain[0].value_target}"
            )

    def test_multiple_chain_steps_independent(self):
        """Each chain step is processed independently."""
        scale = 100.0
        clip = 10.0
        steps = [
            _make_chain_step(value_target=30.0, root_q_ratio=0.2, is_hero=True),
            _make_chain_step(value_target=70.0, root_q_ratio=float("nan"), is_hero=True),
            _make_chain_step(value_target=50.0, root_q_ratio=0.9, is_hero=False),
        ]
        ex = _make_example(value_target=0.0, root_q_ratio=0.0, chain=steps)
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.5, clip_val=clip, norm_stats=ns
        )
        # Step 0: hero + finite Q → blend
        s0 = examples[0].chain[0]
        expected0 = max(-clip, min(clip,
            0.5 * 0.2 * (scale / scale) + 0.5 * (30.0 / scale)))
        assert abs(s0.value_target - expected0) < 1e-6

        # Step 1: hero + NaN Q → pure MC
        s1 = examples[0].chain[1]
        expected1 = max(-clip, min(clip, 70.0 / scale))
        assert abs(s1.value_target - expected1) < 1e-6

        # Step 2: opp → pure MC (q ignored)
        s2 = examples[0].chain[2]
        expected2 = max(-clip, min(clip, 50.0 / scale))
        assert abs(s2.value_target - expected2) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 8. _robust_scale: MAD → IQR → std → fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustScale:
    """_robust_scale picks the first valid estimator in MAD → IQR → std → fallback."""

    def test_empty_returns_fallback(self):
        assert _robust_scale([], fallback=10.0) == pytest.approx(10.0)
        assert _robust_scale(np.array([]), fallback=5.0) == pytest.approx(5.0)

    def test_gaussian_mad_approx_std(self):
        """For a Gaussian sample, MAD×1.4826 ≈ σ (within 10% for N≥1000)."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 50, 2000)
        scale = _robust_scale(data, fallback=1.0)
        assert 40.0 < scale < 60.0, f"Expected ≈50, got {scale}"

    def test_constant_array_falls_back_beyond_mad(self):
        """All-constant → MAD=0 → IQR=0 → std=0 → fallback."""
        data = np.ones(100) * 5.0
        result = _robust_scale(data, fallback=7.0)
        assert result == pytest.approx(7.0), (
            f"All-constant should fall back to 7.0, got {result}"
        )

    def test_two_value_array_mad_works(self):
        """[0, 200] → MAD = 100, scale = 1.4826 * 100 = 148.26."""
        data = np.array([0.0, 200.0])
        scale = _robust_scale(data, fallback=1.0)
        assert scale == pytest.approx(1.4826 * 100.0, rel=1e-4)

    def test_mad_preferred_over_iqr(self):
        """When MAD is valid, IQR is NOT used. Changing the tails should not affect
        the result once MAD >= 1e-8."""
        base = np.array([0.0, 1.0, 2.0, 3.0, 4.0] * 100, dtype=float)
        scale = _robust_scale(base, fallback=1.0)
        # MAD of [0..4 repeated] — should be finite and > 0
        med = np.median(base)
        mad = np.median(np.abs(base - med))
        expected = 1.4826 * mad
        assert scale == pytest.approx(expected, rel=1e-6)

    def test_iqr_fallback_when_mad_zero(self):
        """Data with MAD=0 but non-zero IQR should use IQR/1.349.

        Dataset: 60 copies of 5.0, 20 copies of 0.0, 20 copies of 10.0.
        Median = 5.0; MAD = median(|x - 5|) = 0.0 (>50% are exactly 5.0).
        Q25 = 5.0, Q75 = 5.0 for this distribution → IQR = 0 as well.
        So the fallback chain reaches std, which is non-zero (~3.16).
        We verify the returned value matches whichever estimator is first non-zero.
        """
        data = np.array([5.0] * 60 + [0.0] * 20 + [10.0] * 20, dtype=float)
        # Verify MAD=0 (test precondition)
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        assert mad < 1e-8, f"Test setup error: MAD should be 0, got {mad}"

        scale = _robust_scale(data, fallback=1.0)

        # Determine expected: IQR → std → fallback (first >= 1e-8)
        q75, q25 = np.percentile(data, [75, 25])
        iqr_scale = (q75 - q25) / 1.349
        if iqr_scale >= 1e-8:
            assert scale == pytest.approx(iqr_scale, rel=1e-6)
        else:
            std_c = float(data.std())
            if std_c >= 1e-8:
                assert scale == pytest.approx(std_c, rel=1e-6)
            else:
                assert scale == pytest.approx(1.0)

    def test_positive_result(self):
        """_robust_scale should never return a negative number."""
        rng = np.random.RandomState(123)
        for _ in range(20):
            data = rng.uniform(-1000, 1000, rng.randint(1, 200))
            result = _robust_scale(data, fallback=10.0)
            assert result > 0, f"Got non-positive scale {result}"

    def test_fallback_value_is_positive(self):
        """When every estimator fails, the exact fallback value is returned."""
        result = _robust_scale(np.array([3.0]), fallback=42.0)
        # Single-element array: MAD=0, IQR=0, std=0 → fallback
        assert result == pytest.approx(42.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. New scale is stored in norm_stats after finalization
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleStoredInNormStats:
    """norm_stats["mcts_value_scale"] is updated after _finalize_value_targets."""

    def test_bootstrap_sets_scale_key(self):
        """When 'mcts_value_scale' is absent, it is set after finalization."""
        ex = _make_example(value_target=50.0, root_q_ratio=float("nan"))
        ns = {}
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": [ex]}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": 10.0},
            alpha=0.5,
            clip_val=5.0,
            big_blind=10.0,
            log=_null_log,
        )
        assert "mcts_value_scale" in ns, "mcts_value_scale must be added to norm_stats"
        assert ns["mcts_value_scale"] > 0

    def test_existing_scale_reused(self):
        """With ema=0 (default), an existing 'mcts_value_scale' is not
        overwritten (fixed-scale behaviour)."""
        existing_scale = 77.0
        ns = {"mcts_value_scale": existing_scale}
        ex = _make_example(value_target=50.0, root_q_ratio=float("nan"))
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": [ex]}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": existing_scale},
            alpha=0.0,
            clip_val=10.0,
            big_blind=10.0,
            log=_null_log,
        )
        assert ns["mcts_value_scale"] == pytest.approx(existing_scale)

    def test_ema_moves_scale_toward_cycle_data(self):
        """With ema>0 an existing scale is EMA-blended toward the cycle's
        robust scale (replaces the removed one-shot forced rebootstrap: the
        scale still tracks fresh chip-delta data, just gradually)."""
        old_scale = 999.0
        ema = 0.1
        ns = {"mcts_value_scale": old_scale}
        # Chip deltas: median 0, MAD 30 → cycle_scale = 1.4826 * 30
        chips = [50.0, -50.0, 30.0, -30.0, 20.0, -20.0]
        examples = [_make_example(value_target=float(v), root_q_ratio=float("nan"))
                    for v in chips]
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": examples}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": old_scale},
            alpha=0.0,
            clip_val=100.0,
            big_blind=10.0,
            log=_null_log,
            ema=ema,
        )
        cycle_scale = 1.4826 * 30.0
        expected = (1.0 - ema) * old_scale + ema * cycle_scale
        assert ns["mcts_value_scale"] == pytest.approx(expected)
        # Bookkeeping reflects THIS cycle's sample count and chip range.
        assert ns["mcts_value_scale_n_samples"] == len(chips)
        assert ns["mcts_value_chip_min"] == pytest.approx(-50.0)
        assert ns["mcts_value_chip_max"] == pytest.approx(50.0)

    def test_scale_is_positive_after_bootstrap(self):
        examples = [_make_example(value_target=float(v), root_q_ratio=float("nan"))
                    for v in [100.0, -100.0, 50.0, -50.0]]
        ns = {}
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": examples}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": 10.0},
            alpha=0.5,
            clip_val=5.0,
            big_blind=10.0,
            log=_null_log,
        )
        assert ns["mcts_value_scale"] > 0

    def test_empty_examples_do_not_crash_or_set_scale(self):
        """Agent with no examples: norm_stats unchanged, no error."""
        ns = {}
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": []}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": 10.0},
            alpha=0.5,
            clip_val=5.0,
            big_blind=10.0,
            log=_null_log,
        )
        # No crash; mcts_value_scale should NOT be set (no data to bootstrap from)
        assert "mcts_value_scale" not in ns


# ─────────────────────────────────────────────────────────────────────────────
# 10. Terminal targets rescaling
# ─────────────────────────────────────────────────────────────────────────────

class TestTerminalTargetsRescaling:
    """terminal_targets are rescaled by rescale_q = search_scale / new_scale and clipped."""

    def test_terminal_rescaled_by_rescale_q(self):
        """Each terminal Q is multiplied by search_scale/new_scale then clipped."""
        search_scale = 50.0
        new_scale = 100.0
        clip = 5.0
        q_search_1 = 2.0   # in search_scale units → q_new = 2 * 0.5 = 1.0
        q_search_2 = -4.0  # → q_new = -4 * 0.5 = -2.0
        terminals = [
            ([0, 1], q_search_1),
            ([1, 2, 0], q_search_2),
        ]
        ex = _make_example(
            value_target=0.0,
            root_q_ratio=float("nan"),
            terminal_targets=terminals,
        )
        ns = {"mcts_value_scale": new_scale}
        examples, _ = _run_finalize(
            [ex], search_scale=search_scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        result = examples[0].terminal_targets
        assert len(result) == 2
        expected_1 = max(-clip, min(clip, q_search_1 * (search_scale / new_scale)))
        expected_2 = max(-clip, min(clip, q_search_2 * (search_scale / new_scale)))
        assert abs(result[0][1] - expected_1) < 1e-6, (
            f"terminal 0: expected {expected_1}, got {result[0][1]}"
        )
        assert abs(result[1][1] - expected_2) < 1e-6, (
            f"terminal 1: expected {expected_2}, got {result[1][1]}"
        )

    def test_terminal_action_paths_preserved(self):
        """Action paths inside terminal_targets must not be mutated."""
        path1 = [0, 2, 1]
        path2 = [1]
        terminals = [(path1, 1.0), (path2, -0.5)]
        ex = _make_example(
            value_target=0.0,
            root_q_ratio=float("nan"),
            terminal_targets=terminals,
        )
        scale = 100.0
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.0, clip_val=10.0, norm_stats=ns
        )
        result = examples[0].terminal_targets
        assert result[0][0] == path1
        assert result[1][0] == path2

    def test_terminal_clipped_to_clip_val(self):
        """Terminal Q exceeding clip is clamped."""
        search_scale = 100.0
        new_scale = 100.0
        clip = 2.0
        large_q = 50.0  # way beyond clip after rescale
        terminals = [([0], large_q), ([1], -large_q)]
        ex = _make_example(
            value_target=0.0,
            root_q_ratio=float("nan"),
            terminal_targets=terminals,
        )
        ns = {"mcts_value_scale": new_scale}
        examples, _ = _run_finalize(
            [ex], search_scale=search_scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        result = examples[0].terminal_targets
        assert result[0][1] == pytest.approx(clip)
        assert result[1][1] == pytest.approx(-clip)

    def test_terminal_identity_rescale(self):
        """When search_scale == new_scale, terminal Q passes through unchanged (before clip)."""
        scale = 100.0
        clip = 10.0
        q = 3.5
        terminals = [([0, 1], q)]
        ex = _make_example(
            value_target=0.0,
            root_q_ratio=float("nan"),
            terminal_targets=terminals,
        )
        ns = {"mcts_value_scale": scale}
        examples, _ = _run_finalize(
            [ex], search_scale=scale, alpha=0.0, clip_val=clip, norm_stats=ns
        )
        assert abs(examples[0].terminal_targets[0][1] - q) < 1e-6

    def test_no_terminal_targets_no_crash(self):
        """Example with empty terminal_targets must not crash."""
        ex = _make_example(
            value_target=50.0,
            root_q_ratio=float("nan"),
            terminal_targets=[],
        )
        ns = {"mcts_value_scale": 100.0}
        examples, _ = _run_finalize(
            [ex], search_scale=100.0, alpha=0.5, clip_val=5.0, norm_stats=ns
        )
        assert examples[0].terminal_targets == []


# ─────────────────────────────────────────────────────────────────────────────
# 11. Multi-agent: each agent's own scale is applied independently
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAgentIndependence:
    """Each agent bootstraps and applies its own mcts_value_scale independently."""

    def test_two_agents_different_scales(self):
        scale_a = 50.0
        scale_b = 200.0
        realized_a = 30.0
        realized_b = 80.0
        clip = 10.0

        ex_a = _make_example(value_target=realized_a, root_q_ratio=float("nan"))
        ex_b = _make_example(value_target=realized_b, root_q_ratio=float("nan"))

        ns_a = {"mcts_value_scale": scale_a}
        ns_b = {"mcts_value_scale": scale_b}
        agents_list = [
            {"name": "agent_a", "norm_stats": ns_a},
            {"name": "agent_b", "norm_stats": ns_b},
        ]
        per_agent_examples = {
            "agent_a": [ex_a],
            "agent_b": [ex_b],
        }
        search_scales = {"agent_a": scale_a, "agent_b": scale_b}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=agents_list,
            search_scales=search_scales,
            alpha=0.0,
            clip_val=clip,
            big_blind=10.0,
            log=_null_log,
        )
        expected_a = max(-clip, min(clip, realized_a / scale_a))
        expected_b = max(-clip, min(clip, realized_b / scale_b))
        assert abs(ex_a.value_target - expected_a) < 1e-6
        assert abs(ex_b.value_target - expected_b) < 1e-6

    def test_agent_with_no_examples_does_not_affect_other(self):
        scale = 100.0
        realized = 50.0
        clip = 5.0
        ex = _make_example(value_target=realized, root_q_ratio=float("nan"))
        ns_with = {"mcts_value_scale": scale}
        ns_empty = {"mcts_value_scale": scale}
        agents_list = [
            {"name": "with_data", "norm_stats": ns_with},
            {"name": "empty_agent", "norm_stats": ns_empty},
        ]
        per_agent_examples = {
            "with_data": [ex],
            "empty_agent": [],
        }
        search_scales = {"with_data": scale, "empty_agent": scale}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=agents_list,
            search_scales=search_scales,
            alpha=0.0,
            clip_val=clip,
            big_blind=10.0,
            log=_null_log,
        )
        expected = max(-clip, min(clip, realized / scale))
        assert abs(ex.value_target - expected) < 1e-6
        # empty_agent's norm_stats should be untouched (no new bootstrapping)
        assert ns_empty["mcts_value_scale"] == pytest.approx(scale)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Bootstrap uses chip deltas from examples when scale is absent
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapScale:
    """When mcts_value_scale is missing, it is bootstrapped from the chip deltas."""

    def test_bootstrap_reflects_data_spread(self):
        """Bootstrap scale should be on the same order of magnitude as chip spread."""
        chips = [100.0, -100.0, 80.0, -80.0, 60.0, -60.0, 40.0, -40.0]
        examples = [_make_example(value_target=c, root_q_ratio=float("nan"))
                    for c in chips]
        ns = {}
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": examples}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": 10.0},
            alpha=0.0,
            clip_val=100.0,
            big_blind=10.0,
            log=_null_log,
        )
        new_scale = ns["mcts_value_scale"]
        # Roughly: MAD of the chip distribution is ~80, scale ≈ 1.4826*80 ≈ 119
        # Regardless of exact value, it must be in a sane range
        assert 10.0 < new_scale < 500.0, (
            f"Bootstrapped scale {new_scale} seems unreasonable for chip spread ≈100"
        )

    def test_bootstrap_uses_root_value_targets_not_q(self):
        """Bootstrap is computed from ex.value_target (chip deltas), not root_q_ratio."""
        # All chip deltas are 100, Q ratios are giant → scale must be from chips
        chips = [100.0] * 10 + [-100.0] * 10
        examples = [_make_example(value_target=c, root_q_ratio=9999.0)
                    for c in chips]
        ns = {}
        agent_info = {"name": "a", "norm_stats": ns}
        per_agent_examples = {"a": examples}
        _finalize_value_targets(
            per_agent_examples=per_agent_examples,
            agents_list=[agent_info],
            search_scales={"a": 10.0},
            alpha=0.0,
            clip_val=100.0,
            big_blind=10.0,
            log=_null_log,
        )
        # MAD of [100]*10 + [-100]*10 = MAD(|x - 0|) where med=0 → MAD=100
        # robust scale = 1.4826*100 ≈ 148.26
        expected = 1.4826 * 100.0
        assert abs(ns["mcts_value_scale"] - expected) < 1.0, (
            f"Expected robust scale ≈{expected}, got {ns['mcts_value_scale']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 13. In-place mutation: value_target field is actually overwritten
# ─────────────────────────────────────────────────────────────────────────────

class TestInPlaceMutation:
    """_finalize_value_targets must modify example objects in place."""

    def test_value_target_overwritten(self):
        original = 12345.0
        ex = _make_example(value_target=original, root_q_ratio=float("nan"))
        ns = {"mcts_value_scale": 100.0}
        _run_finalize([ex], search_scale=100.0, alpha=0.0, clip_val=5.0, norm_stats=ns)
        assert ex.value_target != original, "value_target must be overwritten in place"

    def test_chain_step_value_target_overwritten(self):
        original = 9999.0
        step = _make_chain_step(value_target=original, root_q_ratio=float("nan"), is_hero=True)
        ex = _make_example(value_target=0.0, root_q_ratio=float("nan"), chain=[step])
        ns = {"mcts_value_scale": 100.0}
        _run_finalize([ex], search_scale=100.0, alpha=0.0, clip_val=5.0, norm_stats=ns)
        assert step.value_target != original, "chain step value_target must be overwritten"

    def test_terminal_targets_list_replaced(self):
        terminals = [([0], 100.0)]
        ex = _make_example(value_target=0.0, root_q_ratio=float("nan"),
                           terminal_targets=terminals)
        ns = {"mcts_value_scale": 100.0}
        _run_finalize([ex], search_scale=100.0, alpha=0.0, clip_val=1.0, norm_stats=ns)
        # After rescale + clip, the Q should be 1.0 (clipped from 100/100*1 = 1.0... exactly clip)
        result_q = ex.terminal_targets[0][1]
        assert abs(result_q) <= 1.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 14. Per-cycle EMA scale smoothing (mcts_train.value_scale_ema)
#     e2e over two simulated cycles: cycle 0 full-bootstraps, cycle 1
#     EMA-updates; Q rescaling bridges the collection-time snapshot axis
#     onto the updated axis, and is the identity when the scale is stable.
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAScaleTrajectory:
    """Deterministic two-cycle trajectory of mcts_value_scale under EMA."""

    # Chip sets with exactly known robust scales (median 0 → MAD is the
    # median of |chips|).
    CYCLE0_CHIPS = [100.0, -100.0, 80.0, -80.0, 60.0, -60.0]   # MAD 80
    CYCLE1_CHIPS = [50.0, -50.0, 40.0, -40.0, 30.0, -30.0]     # MAD 40
    SCALE0 = 1.4826 * 80.0    # cycle-0 full bootstrap
    CYCLE1_SCALE = 1.4826 * 40.0
    EMA = 0.1

    def _cycle(self, chips, ns, search_scale, alpha=0.0, q=float("nan")):
        """Simulate one collection cycle's finalize pass. Returns examples."""
        examples = [_make_example(value_target=float(c), root_q_ratio=q)
                    for c in chips]
        agent_info = {"name": "a", "norm_stats": ns}
        _finalize_value_targets(
            per_agent_examples={"a": examples},
            agents_list=[agent_info],
            search_scales={"a": search_scale},
            alpha=alpha,
            clip_val=100.0,
            big_blind=10.0,
            log=_null_log,
            ema=self.EMA,
        )
        return examples

    def test_two_cycle_ema_trajectory(self):
        """Cycle 0: full bootstrap from the cycle's chips (no EMA — no prior
        scale). Cycle 1: EMA blend of the previous scale and the cycle's
        robust scale."""
        ns = {}
        # Cycle 0 — no existing scale; search ran on the BB fallback (10).
        self._cycle(self.CYCLE0_CHIPS, ns, search_scale=10.0)
        assert ns["mcts_value_scale"] == pytest.approx(self.SCALE0)
        assert ns["mcts_value_scale_n_samples"] == len(self.CYCLE0_CHIPS)

        # Cycle 1 — search snapshot is the cycle-0 scale; EMA update.
        self._cycle(self.CYCLE1_CHIPS, ns, search_scale=ns["mcts_value_scale"])
        expected1 = (1.0 - self.EMA) * self.SCALE0 + self.EMA * self.CYCLE1_SCALE
        assert ns["mcts_value_scale"] == pytest.approx(expected1)
        assert ns["mcts_value_scale_n_samples"] == len(self.CYCLE1_CHIPS)

    def test_q_rescaled_from_snapshot_axis_onto_ema_axis(self):
        """Cycle 1's root.Q (stored in the collection-start snapshot scale)
        must be multiplied by search_scale/new_scale — the snapshot scale
        over the EMA-updated scale."""
        ns = {"mcts_value_scale": self.SCALE0}
        q = 0.7
        examples = self._cycle(self.CYCLE1_CHIPS, ns,
                               search_scale=self.SCALE0, alpha=1.0, q=q)
        new_scale = (1.0 - self.EMA) * self.SCALE0 + self.EMA * self.CYCLE1_SCALE
        expected_q = q * (self.SCALE0 / new_scale)
        for ex in examples:
            assert ex.value_target == pytest.approx(expected_q)

    def test_q_rescale_is_identity_when_scale_stable(self):
        """When the cycle's robust scale equals the existing scale, the EMA
        update is a fixed point (new == old) and rescale_q == 1: root.Q
        passes through unchanged (alpha=1)."""
        ns = {"mcts_value_scale": self.SCALE0}
        q = 0.7
        # CYCLE0_CHIPS reproduce exactly the existing scale → stable.
        examples = self._cycle(self.CYCLE0_CHIPS, ns,
                               search_scale=self.SCALE0, alpha=1.0, q=q)
        assert ns["mcts_value_scale"] == pytest.approx(self.SCALE0)
        for ex in examples:
            assert ex.value_target == pytest.approx(q)
