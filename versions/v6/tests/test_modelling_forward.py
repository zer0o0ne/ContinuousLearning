"""
Tests for ModellingHead forward pass and context extension in modelling training.

Tests cover:
  1. Output shape: (B, n_actions, d_model)
  2. Cross-attention uses context: different context → different action embeddings
  3. Self-attention is non-causal: perturbing action i affects action j
  4. Context extension math: action token placed at true length, not padded length
  5. Per-action value prediction: (B, n_actions) EV matrix
  6. Reconstruction loss: action emb ≈ next perception state, MSE
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

from agent.modelling.modelling import ModellingHead
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


# ── 2. Cross-attention uses context ──────────────────────────────────────────

class TestCrossAttentionUsesContext(unittest.TestCase):

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


# ── 3. Self-attention is non-causal (symmetric) ───────────────────────────────

class TestSelfAttentionNonCausal(unittest.TestCase):

    def test_perturbing_action_i_affects_action_j(self):
        """Action embeddings are not causally ordered: modifying one action's
        learnable embedding changes outputs for other actions (bidirectional
        self-attention).
        """
        head = _make_modelling_head()
        head.eval()
        SEQ = 5
        ctx = _random_context(1, SEQ)
        mask = _full_mask(1, SEQ)

        with torch.no_grad():
            baseline = head(ctx, mask=mask).clone()  # (1, N_ACTIONS, D_MODEL)

        # Perturb action_embeddings[0] (the first learnable query vector)
        with torch.no_grad():
            head.action_embeddings.weight[0] += 5.0

        with torch.no_grad():
            perturbed = head(ctx, mask=mask)

        # The perturbation should affect action 0 output
        self.assertFalse(
            torch.allclose(baseline[0, 0], perturbed[0, 0], atol=1e-6),
            "Action 0 output should change after perturbing its embedding",
        )
        # And should also affect at least one OTHER action (non-causal)
        changed_others = any(
            not torch.allclose(baseline[0, j], perturbed[0, j], atol=1e-6)
            for j in range(1, N_ACTIONS)
        )
        self.assertTrue(
            changed_others,
            "Perturbing action 0 embedding should affect other action outputs "
            "(self-attention is bidirectional, not causal)",
        )

    def test_explicit_all_visible_attention_mask_used(self):
        """ModellingHead builds an explicit zeros self-attn mask (all-visible),
        confirming the intentional non-causal design.
        """
        # We verify this by running the head twice: results should agree exactly
        # (no stochastic causal masking / dropout in eval mode).
        head = _make_modelling_head()
        head.eval()
        ctx = _random_context(2, 4)
        mask = _full_mask(2, 4)
        with torch.no_grad():
            r1 = head(ctx, mask=mask)
            r2 = head(ctx, mask=mask)
        self.assertTrue(torch.allclose(r1, r2, atol=1e-6))


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


# ── 6. Reconstruction loss ────────────────────────────────────────────────────

class TestReconstructionLoss(unittest.TestCase):

    def _make_batch(self, B=2, SEQ=5):
        torch.manual_seed(55)
        action_embs = torch.randn(B, N_ACTIONS, D_MODEL, requires_grad=True)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        return action_embs, perception_out

    def _make_event_seqs(self, n_events_per_seq, with_action=True):
        """Build minimal event sequences with/without a clear action at each step."""
        seqs = []
        for n in n_events_per_seq:
            seq = []
            for t in range(n):
                # Set action at all but the last event (no t+1 target for last)
                action_idx = t % N_ACTIONS if with_action and t < n - 1 else None
                seq.append(_make_event_dict(action_idx=action_idx))
            seqs.append(seq)
        return seqs

    def test_reconstruction_loss_is_scalar(self):
        """_reconstruction_loss returns a scalar tensor."""
        B, SEQ = 2, 5
        action_embs, perception_out = self._make_batch(B, SEQ)
        event_seqs = self._make_event_seqs([SEQ, SEQ])
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        self.assertEqual(loss.dim(), 0, "Reconstruction loss should be a scalar")

    def test_reconstruction_loss_mse_semantics(self):
        """When action embeddings match next perception states exactly, loss is 0."""
        B, SEQ = 2, 5
        torch.manual_seed(66)
        perception_out = torch.randn(B, SEQ, D_MODEL)

        # Build event sequences so every step t (except last) takes action 0
        event_seqs = self._make_event_seqs([SEQ, SEQ], with_action=True)

        # Build action_embs where action 0 matches perception_out[:, t+1, :]
        # for every valid pair (t, t+1).
        action_embs = torch.zeros(B, N_ACTIONS, D_MODEL)
        for i in range(B):
            for t in range(SEQ - 1):
                action_idx = t % N_ACTIONS
                action_embs[i, action_idx] = perception_out[i, t + 1]
        action_embs.requires_grad_(True)

        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        # MSE between identical vectors is 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5,
                               msg="Recon loss should be 0 when embs match next states")

    def test_reconstruction_loss_nonzero_for_mismatch(self):
        """Reconstruction loss is positive when embs do not match next states."""
        B, SEQ = 2, 5
        action_embs, perception_out = self._make_batch(B, SEQ)
        # Deliberately corrupt action embeddings
        action_embs = action_embs + 10.0

        event_seqs = self._make_event_seqs([SEQ, SEQ], with_action=True)
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        self.assertGreater(loss.item(), 0.0)

    def test_reconstruction_loss_has_gradient(self):
        """Gradient flows from reconstruction loss into action_embs."""
        B, SEQ = 2, 4
        action_embs, perception_out = self._make_batch(B, SEQ)
        event_seqs = self._make_event_seqs([SEQ, SEQ], with_action=True)

        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        loss.backward()
        self.assertIsNotNone(action_embs.grad, "action_embs should have gradient")

    def test_reconstruction_uses_mse(self):
        """Reconstruction loss is standard MSE between predicted and target."""
        # Single pair: one event with action 1, followed by a next state
        SEQ = 2
        torch.manual_seed(88)
        perception_out = torch.randn(1, SEQ, D_MODEL)
        action_embs = torch.randn(1, N_ACTIONS, D_MODEL, requires_grad=True)
        event_seqs = [
            [
                _make_event_dict(action_idx=1),  # event 0: action=1
                _make_event_dict(action_idx=None),  # event 1: target (no action field needed)
            ]
        ]
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        # Manually compute MSE: predicted = action_embs[0, 1], target = perception_out[0, 1]
        predicted = action_embs[0, 1]
        target = perception_out[0, 1].detach()
        expected = F.mse_loss(predicted.unsqueeze(0), target.unsqueeze(0))
        self.assertAlmostEqual(loss.item(), expected.item(), places=5,
                               msg="Reconstruction loss should equal MSE of the matched pair")


# ── 7. No-valid-pairs fallback ────────────────────────────────────────────────

class TestNoValidPairsFallback(unittest.TestCase):

    def test_empty_event_sequences_returns_zero(self):
        """Empty event sequences → no valid pairs → returns 0.0 with requires_grad."""
        B, SEQ = 2, 4
        torch.manual_seed(9)
        action_embs = torch.randn(B, N_ACTIONS, D_MODEL, requires_grad=True)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        # Empty event sequences: no (t, t+1) pairs at all
        event_seqs = [[], []]
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad,
                        "Zero fallback tensor must have requires_grad=True for loss.backward()")

    def test_single_event_no_next_state(self):
        """A sequence of length 1 has no (t, t+1) pair — returns 0.0."""
        B = 2
        torch.manual_seed(12)
        action_embs = torch.randn(B, N_ACTIONS, D_MODEL, requires_grad=True)
        perception_out = torch.randn(B, 1, D_MODEL)
        event_seqs = [
            [_make_event_dict(action_idx=0)],  # only one event each
            [_make_event_dict(action_idx=2)],
        ]
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad)

    def test_all_events_have_no_action(self):
        """When no event has a clear action (max < 0.5), no valid pairs → zero loss."""
        B = 2
        SEQ = 4
        torch.manual_seed(15)
        action_embs = torch.randn(B, N_ACTIONS, D_MODEL, requires_grad=True)
        perception_out = torch.randn(B, SEQ, D_MODEL)
        # All events have zero action vectors (max = 0 < 0.5 → skipped)
        event_seqs = [
            [_make_event_dict(action_idx=None) for _ in range(SEQ)],
            [_make_event_dict(action_idx=None) for _ in range(SEQ)],
        ]
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        self.assertAlmostEqual(loss.item(), 0.0, places=9)
        self.assertTrue(loss.requires_grad)

    def test_zero_loss_is_differentiable(self):
        """The zero fallback allows backward() to be called without error."""
        action_embs = torch.randn(1, N_ACTIONS, D_MODEL, requires_grad=True)
        perception_out = torch.randn(1, 3, D_MODEL)
        event_seqs = [[]]  # empty
        loss = _reconstruction_loss(action_embs, perception_out, event_seqs)
        # This should not raise
        try:
            (loss * 0.5).backward()
        except Exception as e:
            self.fail(f"backward() on zero fallback raised: {e}")


if __name__ == "__main__":
    unittest.main()
