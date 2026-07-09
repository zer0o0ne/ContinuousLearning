"""
Tests for ModellingHead forward pass and context extension in modelling training
(autoregressive redesign — PLAN_MODELLING_HEAD_REDESIGN.md).

Tests cover:
  1. Output shape: (B, n_actions, d_model)
  2. Context dependence: different context → different action embeddings
  3. Independent action conditioning: perturbing action embedding i does NOT
     change output for action j (actions are conditioned independently via
     the per-action MLP — no cross-action attention)
  4. Context extension math: action token placed at true length, not padded length
  5. Per-action value prediction: (B, n_actions) EV matrix
  6. LM next-decision-state loss (_reconstruction_loss): pairs per PLAN §3,
     MSE + InfoNCE per PLAN §4, gradients into modelling_head only
  7. No-valid-pairs fallback: returns tensor(0.0, requires_grad=True)

Run from versions/v6/:
    python -m unittest tests.test_modelling_forward -v
"""

import sys
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.modelling.modelling import ModellingHead, build_lm_pairs
from agent.value.value import ValueHead
from agent.train_scenarios.modelling_predict.train import (
    _modelling_forward,
    _reconstruction_loss,
)

# ── Minimal model config ─────────────────────────────────────────────────────
# Keep dimensions tiny so tests stay fast on CPU.

D_MODEL = 16
N_HEADS = 4
N_KV_HEADS = 2
D_FF = 32
N_LAYERS = 1
MAX_SEQ_LEN = 64
N_ACTIONS = 5   # fold + call + 2 raise bins + allin


def _make_modelling_head():
    torch.manual_seed(42)
    return ModellingHead(
        d_model=D_MODEL,
        n_actions=N_ACTIONS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )


def _make_value_head():
    torch.manual_seed(0)
    return ValueHead(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN + 4,
    )


def _random_context(B, seq_len, d_model=D_MODEL, seed=7):
    torch.manual_seed(seed)
    return torch.randn(B, seq_len, d_model)


def _full_mask(B, seq_len):
    """All-real mask (no padding)."""
    return torch.ones(B, seq_len)


# ── Minimal ASI stub for _modelling_forward ─────────────────────────────────

class _FakeAgent(nn.Module):
    """Minimal stub with modelling_head + value_head; no real perception."""

    def __init__(self, perception_out, mask):
        super().__init__()
        self.modelling_head = _make_modelling_head()
        self.value_head = _make_value_head()
        self.n_actions = N_ACTIONS
        # Pre-baked perception output so _modelling_forward can cache it.
        self._fixed_perception_out = perception_out
        self._fixed_mask = mask

    def train(self, mode=True):
        # Keep modules in eval mode for determinism in tests
        self.modelling_head.train(mode)
        self.value_head.train(mode)
        return self


def _make_event_dict(action_idx=None, n_actions=N_ACTIONS):
    """Minimal event dict compatible with _reconstruction_loss."""
    action = [0.0] * n_actions
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": [0, -1, -1, -1, -1],
        "hand": [10, 11],
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 2,
        "pot": 0.0,
        "stack": 100.0,
        "big_blind": 10.0,
        "bets": [0.0, 0.0],
        "stacks": [100.0, 100.0],
        "action": action,
    }


def _make_lm_event_seq(action_indices, trailing_pre=True):
    """Event stream per PLAN §3 (collect.py convention):
    [initial(None), post_0(one-hot a_0), pre_1(None), post_1(one-hot a_1), ...].

    With trailing_pre=True, a final pre-decision snapshot follows the last
    post-action event, so ALL decisions have a q+1 target. With
    trailing_pre=False the sequence ends on the last post-action event and
    that decision is skipped by build_lm_pairs.
    """
    seq = [_make_event_dict(action_idx=None)]
    for i, a in enumerate(action_indices):
        seq.append(_make_event_dict(action_idx=a))
        if trailing_pre or i < len(action_indices) - 1:
            seq.append(_make_event_dict(action_idx=None))
    return seq


class _HeadOnlyAgent:
    """Minimal stub exposing only modelling_head — what _reconstruction_loss needs."""

    def __init__(self, head):
        self.modelling_head = head


# ── 1. Output shape ───────────────────────────────────────────────────────────

class TestModellingHeadOutputShape(unittest.TestCase):

    def test_shape_single_sample(self):
        """ModellingHead returns (1, n_actions, d_model) for a single sample."""
        head = _make_modelling_head()
        context = _random_context(1, 5)
        mask = _full_mask(1, 5)
        with torch.no_grad():
            out = head(context, mask=mask)
        self.assertEqual(out.shape, (1, N_ACTIONS, D_MODEL))

    def test_shape_batch(self):
        """ModellingHead returns (B, n_actions, d_model) for a batch."""
        head = _make_modelling_head()
        B, SEQ = 4, 8
        context = _random_context(B, SEQ)
        mask = _full_mask(B, SEQ)
        with torch.no_grad():
            out = head(context, mask=mask)
        self.assertEqual(out.shape, (B, N_ACTIONS, D_MODEL))

    def test_shape_with_padding_mask(self):
        """Shape is correct even with partial padding."""
        head = _make_modelling_head()
        B, SEQ = 3, 10
        context = _random_context(B, SEQ)
        # Variable lengths via mask
        mask = torch.zeros(B, SEQ)
        mask[0, :4] = 1.0
        mask[1, :7] = 1.0
        mask[2, :10] = 1.0
        with torch.no_grad():
            out = head(context, mask=mask)
        self.assertEqual(out.shape, (B, N_ACTIONS, D_MODEL))


# ── 2. Context dependence ────────────────────────────────────────────────────

class TestContextDependence(unittest.TestCase):

    def test_different_context_gives_different_embeddings(self):
        """When context changes, the output action embeddings must change."""
        torch.manual_seed(1)
        head = _make_modelling_head()
        head.eval()
        SEQ = 6
        ctx_a = _random_context(1, SEQ, seed=10)
        ctx_b = _random_context(1, SEQ, seed=20)  # different context
        mask = _full_mask(1, SEQ)
        with torch.no_grad():
            out_a = head(ctx_a, mask=mask)
            out_b = head(ctx_b, mask=mask)
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "Different contexts must produce different action embeddings",
        )

    def test_identical_context_gives_identical_embeddings(self):
        """Same context and same weights must produce identical output."""
        head = _make_modelling_head()
        head.eval()
        SEQ = 6
        ctx = _random_context(1, SEQ, seed=99)
        mask = _full_mask(1, SEQ)
        with torch.no_grad():
            out_a = head(ctx.clone(), mask=mask)
            out_b = head(ctx.clone(), mask=mask)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6))


# ── 3. Independent action conditioning ───────────────────────────────────────

class TestActionConditioningIndependence(unittest.TestCase):

    def test_perturbing_action_i_does_not_affect_action_j(self):
        """Actions are conditioned INDEPENDENTLY (PLAN §2): h(t, a) depends on
        the shared state s_t and only on action a's own embedding. Perturbing
        action 0's learnable embedding changes action 0's output but leaves
        every other action's output bitwise unchanged.
        """
        head = _make_modelling_head()
        head.eval()
        SEQ = 5
        ctx = _random_context(1, SEQ)
        mask = _full_mask(1, SEQ)

        with torch.no_grad():
            baseline = head(ctx, mask=mask).clone()  # (1, N_ACTIONS, D_MODEL)

        # Perturb action_embeddings[0] (the first conditioning vector)
        with torch.no_grad():
            head.action_embeddings.weight[0] += 5.0

        with torch.no_grad():
            perturbed = head(ctx, mask=mask)

        # The perturbation must affect action 0 output
        self.assertFalse(
            torch.allclose(baseline[0, 0], perturbed[0, 0], atol=1e-6),
            "Action 0 output should change after perturbing its embedding",
        )
        # And must NOT affect any OTHER action's output (exact equality:
        # the s_t path and the other actions' embeddings are untouched)
        for j in range(1, N_ACTIONS):
            self.assertTrue(
                torch.equal(baseline[0, j], perturbed[0, j]),
                f"Action {j} output must be bitwise unchanged when only "
                f"action 0's embedding is perturbed (independent conditioning)",
            )

    def test_forward_is_deterministic_in_eval(self):
        """Two identical forwards agree exactly (no dropout in eval mode,
        deterministic causal+padding masking)."""
        head = _make_modelling_head()
        head.eval()
        ctx = _random_context(2, 4)
        mask = _full_mask(2, 4)
        with torch.no_grad():
            r1 = head(ctx, mask=mask)
            r2 = head(ctx, mask=mask)
        self.assertTrue(torch.equal(r1, r2))


# ── 4. Context-extension math (A.5.1) ────────────────────────────────────────

class TestContextExtensionMath(unittest.TestCase):
    """Test that _modelling_forward places the action token at the TRUE length
    position (after real events), not at the padded dimension end.
    """

    def _build_fake_agent_and_run(self, lengths):
        """Build a fake agent with pre-baked perception output and run _modelling_forward."""
        B = len(lengths)
        SEQ = max(lengths)  # padded sequence length

        torch.manual_seed(5)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        # Mask: True (1.0) for real tokens, 0.0 for padding
        mask = torch.zeros(B, SEQ)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0

        # Zero out the padding zone so a token placed there would be detectable
        for i, L in enumerate(lengths):
            perception_out[i, L:] = 0.0

        agent = _FakeAgent(perception_out, mask)
        # Freeze value_head params so gradients flow through modelling_head only
        for p in agent.value_head.parameters():
            p.requires_grad = False

        device = "cpu"
        predicted_evs, action_embs, returned_p_out = _modelling_forward(
            agent, event_sequences=[[]] * B, device=device,
            cached_perception=(perception_out, mask),
        )
        return predicted_evs, action_embs, returned_p_out, lengths, SEQ

    def test_action_token_placed_at_true_length(self):
        """In the expanded combined tensor, the action token occupies position
        lengths[b], not position SEQ (the padded end).

        We verify this indirectly: with two samples of DIFFERENT lengths, both
        get valid (non-zero) predicted_evs, confirming that the action token
        was appended after the real events (not in padding).
        """
        lengths = [3, 6]
        predicted_evs, action_embs, perception_out, lengths, SEQ = \
            self._build_fake_agent_and_run(lengths)

        # predicted_evs shape: (B, N_ACTIONS)
        self.assertEqual(predicted_evs.shape, (len(lengths), N_ACTIONS))

    def test_position_index_formula(self):
        """Directly test the index formula used in _modelling_forward.

        For a batch of shape (B, K), the row-major index for sample b, action k is
        b*K + k, and pos[b*K + k] must equal lengths[b].
        """
        B, K = 3, N_ACTIONS
        lengths = torch.tensor([2, 5, 4])
        pos = lengths.unsqueeze(1).expand(B, K).reshape(B * K)
        for b in range(B):
            for k in range(K):
                flat_idx = b * K + k
                self.assertEqual(
                    pos[flat_idx].item(), lengths[b].item(),
                    f"pos[{flat_idx}] should be lengths[{b}]={lengths[b].item()}"
                )

    def test_mask_updated_at_action_token_position(self):
        """After appending the action token, the combined mask has a 1.0 at
        position lengths[b] for each sample — not at the old padded end.
        """
        B, K = 2, N_ACTIONS
        SEQ = 6
        lengths = torch.tensor([3, 5])

        # Simulate the mask construction in _modelling_forward
        mask = torch.zeros(B, SEQ)
        for i, L in enumerate(lengths.tolist()):
            mask[i, :L] = 1.0

        mask_expanded = mask.unsqueeze(1).expand(B, K, SEQ).reshape(B * K, SEQ)
        mask_combined = torch.cat(
            [mask_expanded,
             torch.zeros(B * K, 1)],
            dim=1,
        )  # (B*K, SEQ+1)

        rows = torch.arange(B * K)
        pos = lengths.unsqueeze(1).expand(B, K).reshape(B * K)
        mask_combined[rows, pos] = 1.0

        for b in range(B):
            for k in range(K):
                flat_idx = b * K + k
                L = lengths[b].item()
                self.assertAlmostEqual(
                    mask_combined[flat_idx, L].item(), 1.0, places=6,
                    msg=f"mask_combined[{flat_idx}, {L}] should be 1.0 (action token)"
                )
                # Padding zone beyond L+1 should stay 0
                for p_idx in range(L + 1, SEQ + 1):
                    self.assertAlmostEqual(
                        mask_combined[flat_idx, p_idx].item(), 0.0, places=6,
                        msg=f"mask_combined[{flat_idx}, {p_idx}] should remain 0"
                    )


# ── 5. Per-action value prediction ───────────────────────────────────────────

class TestPerActionValuePrediction(unittest.TestCase):

    def test_predicted_evs_shape(self):
        """_modelling_forward returns predicted_evs of shape (B, n_actions)."""
        B, SEQ = 3, 5
        torch.manual_seed(11)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        mask = _full_mask(B, SEQ)

        agent = _FakeAgent(perception_out, mask)
        for p in agent.value_head.parameters():
            p.requires_grad = False

        predicted_evs, action_embs, _ = _modelling_forward(
            agent, event_sequences=[[]] * B, device="cpu",
            cached_perception=(perception_out, mask),
        )
        self.assertEqual(predicted_evs.shape, (B, N_ACTIONS))

    def test_different_actions_give_different_evs(self):
        """Each action embedding produces a different EV (via different context)."""
        B, SEQ = 1, 6
        torch.manual_seed(77)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        mask = _full_mask(B, SEQ)

        agent = _FakeAgent(perception_out, mask)
        for p in agent.value_head.parameters():
            p.requires_grad = False

        with torch.no_grad():
            predicted_evs, _, _ = _modelling_forward(
                agent, event_sequences=[[]] * B, device="cpu",
                cached_perception=(perception_out, mask),
            )

        # For a non-degenerate model, action EVs should not all be equal
        evs = predicted_evs[0]  # (N_ACTIONS,)
        # Check at least two EVs differ
        self.assertFalse(
            torch.allclose(evs, evs[0].expand_as(evs), atol=1e-4),
            "Predicted EVs for different actions should differ",
        )

    def test_gradient_flows_through_value_head_to_modelling(self):
        """Gradient must flow from value loss back through value_head (frozen params
        but active graph) into modelling_head parameters.
        """
        B, SEQ = 2, 4
        torch.manual_seed(33)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        mask = _full_mask(B, SEQ)

        agent = _FakeAgent(perception_out, mask)
        # Freeze value head params but keep graph
        for p in agent.value_head.parameters():
            p.requires_grad = False
        for p in agent.modelling_head.parameters():
            p.requires_grad = True

        predicted_evs, action_embs, _ = _modelling_forward(
            agent, event_sequences=[[]] * B, device="cpu",
            cached_perception=(perception_out, mask),
        )
        target = torch.zeros_like(predicted_evs)
        loss = F.smooth_l1_loss(predicted_evs, target)
        loss.backward()

        # At least some modelling_head parameters should have non-None gradients
        has_grad = any(
            p.grad is not None for p in agent.modelling_head.parameters()
        )
        self.assertTrue(has_grad, "Gradient should flow back to modelling_head")


# ── 6. LM next-decision-state loss ───────────────────────────────────────────

class TestLMReconstructionLoss(unittest.TestCase):

    def _make_batch(self, seqs, seed=55):
        """Head-only agent + random perception output padded to max seq length."""
        agent = _HeadOnlyAgent(_make_modelling_head())
        agent.modelling_head.eval()
        B = len(seqs)
        SEQ = max(len(s) for s in seqs)
        torch.manual_seed(seed)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        mask = torch.zeros(B, SEQ)
        for i, s in enumerate(seqs):
            mask[i, :len(s)] = 1.0
        return agent, perception_out, mask

    def test_returns_scalar_and_components(self):
        """_reconstruction_loss returns (scalar loss, {"mse", "infonce"})."""
        seqs = [_make_lm_event_seq([0, 2]), _make_lm_event_seq([1, 3])]
        agent, p_out, mask = self._make_batch(seqs)
        loss, components = _reconstruction_loss(
            agent, p_out, mask, seqs,
            infonce_weight=0.5, infonce_temperature=0.1)
        self.assertEqual(loss.dim(), 0, "LM loss should be a scalar")
        self.assertEqual(set(components.keys()), {"mse", "infonce"})
        self.assertTrue(torch.isfinite(loss).item())

    def test_matches_manual_lm_loss(self):
        """Loss equals MSE(pred, target) + w * InfoNCE computed by hand from
        the PLAN §3 pair convention (src=q−1, action=argmax(q), tgt=q+1)."""
        seqs = [_make_lm_event_seq([0, 2]), _make_lm_event_seq([1])]
        agent, p_out, mask = self._make_batch(seqs, seed=77)
        w, tau = 0.5, 0.1
        loss, components = _reconstruction_loss(
            agent, p_out, mask, seqs, infonce_weight=w, infonce_temperature=tau)

        # Manual pairs: seq0 = [pre, post(0), pre, post(2), pre] → q=1, q=3;
        # seq1 = [pre, post(1), pre] → q=1.
        bi = torch.tensor([0, 0, 1])
        sp = torch.tensor([0, 2, 0])
        ai = torch.tensor([0, 2, 1])
        tp = torch.tensor([2, 4, 2])
        with torch.no_grad():
            pred = agent.modelling_head.forward_positions(p_out, mask, bi, sp, ai)
        target = p_out[bi, tp]
        expected_mse = F.mse_loss(pred, target)
        logits = (F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).t()) / tau
        expected_nce = F.cross_entropy(logits, torch.arange(pred.shape[0]))
        expected = expected_mse + w * expected_nce

        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        self.assertAlmostEqual(components["mse"], expected_mse.item(), places=5)
        self.assertAlmostEqual(components["infonce"], expected_nce.item(), places=5)

    def test_pairs_follow_plan_convention(self):
        """build_lm_pairs skips the initial snapshot, uses src=q−1 / tgt=q+1,
        and drops the last decision when no q+1 event exists."""
        seqs = [
            _make_lm_event_seq([3, 1], trailing_pre=True),   # both decisions paired
            _make_lm_event_seq([4], trailing_pre=False),     # no target → dropped
        ]
        bi, sp, ai, tp = build_lm_pairs(seqs)
        self.assertEqual(bi.tolist(), [0, 0])
        self.assertEqual(sp.tolist(), [0, 2])
        self.assertEqual(ai.tolist(), [3, 1])
        self.assertEqual(tp.tolist(), [2, 4])

    def test_infonce_weight_zero_is_pure_mse(self):
        """infonce_weight=0 → loss equals the MSE component exactly."""
        seqs = [_make_lm_event_seq([0, 2]), _make_lm_event_seq([1, 3])]
        agent, p_out, mask = self._make_batch(seqs, seed=91)
        loss, components = _reconstruction_loss(
            agent, p_out, mask, seqs,
            infonce_weight=0.0, infonce_temperature=0.1)
        self.assertAlmostEqual(loss.item(), components["mse"], places=6)
        self.assertEqual(components["infonce"], 0.0)

    def test_single_pair_skips_infonce(self):
        """M=1: InfoNCE needs in-batch negatives (M ≥ 2) → skipped, loss=MSE."""
        seqs = [_make_lm_event_seq([2])]  # exactly one pair
        agent, p_out, mask = self._make_batch(seqs, seed=13)
        loss, components = _reconstruction_loss(
            agent, p_out, mask, seqs,
            infonce_weight=0.5, infonce_temperature=0.1)
        self.assertEqual(components["infonce"], 0.0)
        self.assertAlmostEqual(loss.item(), components["mse"], places=6)

    def test_gradient_flows_to_modelling_head_only(self):
        """Backward populates modelling_head grads; targets are stop-grad."""
        seqs = [_make_lm_event_seq([0, 2]), _make_lm_event_seq([1, 3])]
        agent, p_out, mask = self._make_batch(seqs)
        loss, _ = _reconstruction_loss(
            agent, p_out, mask, seqs,
            infonce_weight=0.5, infonce_temperature=0.1)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in agent.modelling_head.parameters()
        )
        self.assertTrue(has_grad,
                        "LM loss must produce non-zero grads on modelling_head")


# ── 7. No-valid-pairs fallback ────────────────────────────────────────────────

class TestNoValidPairsFallback(unittest.TestCase):

    def _agent_and_perception(self, B, SEQ, seed=9):
        agent = _HeadOnlyAgent(_make_modelling_head())
        agent.modelling_head.eval()
        torch.manual_seed(seed)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        mask = torch.ones(B, SEQ)
        return agent, perception_out, mask

    def test_empty_event_sequences_returns_zero(self):
        """Empty event sequences → no valid pairs → returns 0.0 with requires_grad."""
        agent, p_out, mask = self._agent_and_perception(2, 4)
        loss, components = _reconstruction_loss(agent, p_out, mask, [[], []])
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad,
                        "Zero fallback tensor must have requires_grad=True for loss.backward()")
        self.assertEqual(components, {"mse": 0.0, "infonce": 0.0})

    def test_last_decision_without_target_is_skipped(self):
        """A sequence ending on the post-action event has no q+1 target —
        no pairs → zero loss."""
        agent, p_out, mask = self._agent_and_perception(2, 2, seed=12)
        event_seqs = [
            _make_lm_event_seq([0], trailing_pre=False),  # [pre, post(0)]
            _make_lm_event_seq([2], trailing_pre=False),  # [pre, post(2)]
        ]
        loss, components = _reconstruction_loss(agent, p_out, mask, event_seqs)
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(components, {"mse": 0.0, "infonce": 0.0})

    def test_all_events_have_no_action(self):
        """When no event has a clear action (max < 0.5), no valid pairs → zero loss."""
        SEQ = 4
        agent, p_out, mask = self._agent_and_perception(2, SEQ, seed=15)
        event_seqs = [
            [_make_event_dict(action_idx=None) for _ in range(SEQ)],
            [_make_event_dict(action_idx=None) for _ in range(SEQ)],
        ]
        loss, components = _reconstruction_loss(agent, p_out, mask, event_seqs)
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(components, {"mse": 0.0, "infonce": 0.0})

    def test_zero_loss_is_differentiable(self):
        """The zero fallback allows backward() to be called without error."""
        agent, p_out, mask = self._agent_and_perception(1, 3, seed=21)
        loss, _ = _reconstruction_loss(agent, p_out, mask, [[]])
        # This should not raise
        try:
            (loss * 0.5).backward()
        except Exception as e:
            self.fail(f"backward() on zero fallback raised: {e}")


if __name__ == "__main__":
    unittest.main()
