"""
E2E tests for the redesigned autoregressive ModellingHead
(PLAN_MODELLING_HEAD_REDESIGN.md §2).

Covers:
  1. Shapes: forward → (B, n_actions, D) on padded variable-length batches;
     forward_positions → (M, D).
  2. Causality: perturbing context tokens AFTER position t does not change
     forward_positions output at t; perturbing tokens ≤ t does.
  3. Last-position consistency: forward(context, mask) equals
     forward_positions at positions = lengths−1 for all actions.
  4. Action differentiation: different action indices at the same position
     give different outputs.
  5. Padding invariance: appending pad tokens (mask 0) does not change
     outputs at real positions.
  6. agent.forward_batch integration ("action_embeddings" key + shape) and
     old-checkpoint compat (strict=False load of a state_dict with OLD
     cross-attention modelling keys).

All tests are deterministic: fixed seeds, eval mode (dropout off), CPU.

Run from versions/v6/:
    python3 -m pytest tests/test_modelling_head_autoregressive.py -v
"""

import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.modelling.modelling import ModellingHead
from agent.agent import ASI

# ── Tiny dims (fast on CPU) ──────────────────────────────────────────────────

D_MODEL = 32
N_HEADS = 4
N_KV_HEADS = 2
D_FF = 64
N_LAYERS = 2
MAX_SEQ_LEN = 64
N_ACTIONS = 5   # fold + call + 2 raise bins + all-in

MAX_PLAYERS = 2

_TINY_CONFIG = {
    "architecture": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": D_FF,
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
        "max_stack": 100,
    },
}


def _make_head(seed=42, dropout=0.0, n_layers=N_LAYERS):
    torch.manual_seed(seed)
    head = ModellingHead(
        d_model=D_MODEL,
        n_actions=N_ACTIONS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        n_layers=n_layers,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=dropout,
    )
    head.eval()
    return head


def _padded_batch(lengths, seed=7):
    """Random context padded to max(lengths) + left-aligned 1/0 mask."""
    torch.manual_seed(seed)
    B = len(lengths)
    SEQ = max(lengths)
    context = torch.randn(B, SEQ, D_MODEL)
    mask = torch.zeros(B, SEQ)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1.0
    return context, mask


def _make_event(action_idx=None, n_actions=N_ACTIONS):
    action = [0.0] * n_actions
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": [0, -1, -1, -1, -1],
        "hand": [10, 11],
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 2,
        "pot": 15.0,
        "stack": 100.0,
        "big_blind": 10.0,
        "bets": [5.0, 10.0],
        "stacks": [100.0, 95.0],
        "action": action,
    }


# ── 1. Shapes ────────────────────────────────────────────────────────────────

class TestShapes(unittest.TestCase):

    def test_forward_shape_padded_variable_lengths(self):
        head = _make_head()
        context, mask = _padded_batch([3, 7, 5])
        with torch.no_grad():
            out = head(context, mask=mask)
        self.assertEqual(out.shape, (3, N_ACTIONS, D_MODEL))
        self.assertEqual(out.dtype, context.dtype)

    def test_forward_shape_no_mask(self):
        head = _make_head()
        context, _ = _padded_batch([6, 6])
        with torch.no_grad():
            out = head(context, mask=None)
        self.assertEqual(out.shape, (2, N_ACTIONS, D_MODEL))

    def test_forward_positions_shape(self):
        head = _make_head()
        context, mask = _padded_batch([4, 8])
        batch_idx = torch.tensor([0, 0, 1, 1, 1])
        positions = torch.tensor([0, 2, 1, 4, 6])
        actions = torch.tensor([0, 3, 1, 2, 4])
        with torch.no_grad():
            out = head.forward_positions(context, mask, batch_idx, positions, actions)
        self.assertEqual(out.shape, (5, D_MODEL))
        self.assertEqual(out.dtype, context.dtype)


# ── 2. Causality ─────────────────────────────────────────────────────────────

class TestCausality(unittest.TestCase):

    def test_perturbing_future_tokens_does_not_change_output_at_t(self):
        """s_t depends only on positions ≤ t: perturbing tokens strictly after
        t leaves forward_positions output at t exactly unchanged (masked
        attention weights are exactly zero)."""
        head = _make_head(seed=1)
        L = 8
        t = 3
        context, mask = _padded_batch([L], seed=11)
        batch_idx = torch.tensor([0] * N_ACTIONS)
        positions = torch.tensor([t] * N_ACTIONS)
        actions = torch.arange(N_ACTIONS)

        with torch.no_grad():
            base = head.forward_positions(context, mask, batch_idx, positions, actions)

        perturbed = context.clone()
        perturbed[0, t + 1:] += 100.0  # only tokens AFTER t
        with torch.no_grad():
            after = head.forward_positions(perturbed, mask, batch_idx, positions, actions)

        self.assertTrue(torch.equal(base, after),
                        "Output at position t must be exactly independent of "
                        "tokens after t (causal self-attention)")

    def test_perturbing_past_tokens_changes_output_at_t(self):
        head = _make_head(seed=1)
        L = 8
        t = 3
        context, mask = _padded_batch([L], seed=11)
        batch_idx = torch.tensor([0])
        positions = torch.tensor([t])
        actions = torch.tensor([2])

        with torch.no_grad():
            base = head.forward_positions(context, mask, batch_idx, positions, actions)

        perturbed = context.clone()
        perturbed[0, 0] += 100.0  # token 0 ≤ t
        with torch.no_grad():
            after = head.forward_positions(perturbed, mask, batch_idx, positions, actions)

        self.assertFalse(torch.allclose(base, after),
                         "Output at position t must depend on tokens ≤ t")

    def test_perturbing_token_t_itself_changes_output_at_t(self):
        head = _make_head(seed=1)
        L = 6
        t = 4
        context, mask = _padded_batch([L], seed=13)
        batch_idx = torch.tensor([0])
        positions = torch.tensor([t])
        actions = torch.tensor([0])

        with torch.no_grad():
            base = head.forward_positions(context, mask, batch_idx, positions, actions)
        perturbed = context.clone()
        perturbed[0, t] += 100.0
        with torch.no_grad():
            after = head.forward_positions(perturbed, mask, batch_idx, positions, actions)
        self.assertFalse(torch.allclose(base, after))


# ── 3. Last-position consistency ─────────────────────────────────────────────

class TestLastPositionConsistency(unittest.TestCase):

    def test_forward_equals_forward_positions_at_last_true_position(self):
        head = _make_head(seed=3)
        lengths = [3, 7, 5]
        context, mask = _padded_batch(lengths, seed=17)

        with torch.no_grad():
            out_fwd = head(context, mask=mask)  # (B, A, D)

        B = len(lengths)
        batch_idx = torch.arange(B).repeat_interleave(N_ACTIONS)
        positions = (torch.tensor(lengths) - 1).repeat_interleave(N_ACTIONS)
        actions = torch.arange(N_ACTIONS).repeat(B)
        with torch.no_grad():
            out_pos = head.forward_positions(context, mask, batch_idx, positions, actions)

        self.assertTrue(
            torch.allclose(out_fwd.reshape(B * N_ACTIONS, D_MODEL), out_pos,
                           atol=1e-6),
            "forward at lengths−1 must equal forward_positions at the same "
            "(position, action) pairs",
        )

    def test_forward_no_mask_equals_full_mask(self):
        head = _make_head(seed=3)
        context, mask = _padded_batch([6, 6], seed=19)  # full lengths
        with torch.no_grad():
            out_none = head(context, mask=None)
            out_full = head(context, mask=mask)
        self.assertTrue(torch.allclose(out_none, out_full, atol=1e-6))


# ── 4. Action differentiation ────────────────────────────────────────────────

class TestActionDifferentiation(unittest.TestCase):

    def test_different_actions_same_position_differ(self):
        head = _make_head(seed=5)
        context, mask = _padded_batch([6], seed=23)
        t = 4
        batch_idx = torch.tensor([0, 0])
        positions = torch.tensor([t, t])
        with torch.no_grad():
            out = head.forward_positions(
                context, mask, batch_idx, positions, torch.tensor([0, 3]))
        self.assertFalse(
            torch.allclose(out[0], out[1]),
            "Different action indices at the same position must give "
            "different outputs (randomly initialized action embeddings)",
        )

    def test_forward_rows_differ_across_actions(self):
        head = _make_head(seed=5)
        context, mask = _padded_batch([5], seed=29)
        with torch.no_grad():
            out = head(context, mask=mask)[0]  # (A, D)
        for a in range(1, N_ACTIONS):
            self.assertFalse(
                torch.allclose(out[0], out[a]),
                f"Action 0 and action {a} embeddings must differ",
            )


# ── 5. Padding invariance ────────────────────────────────────────────────────

class TestPaddingInvariance(unittest.TestCase):

    def test_extra_pad_tokens_do_not_change_real_positions(self):
        head = _make_head(seed=9)
        L = 5
        context, mask = _padded_batch([L], seed=31)

        # Same real tokens + 3 extra pad tokens (mask 0, garbage values)
        torch.manual_seed(99)
        pad = torch.randn(1, 3, D_MODEL) * 50.0
        context_padded = torch.cat([context, pad], dim=1)
        mask_padded = torch.cat([mask, torch.zeros(1, 3)], dim=1)

        batch_idx = torch.tensor([0] * L)
        positions = torch.arange(L)
        actions = torch.tensor([i % N_ACTIONS for i in range(L)])

        with torch.no_grad():
            base = head.forward_positions(context, mask, batch_idx, positions, actions)
            padded = head.forward_positions(
                context_padded, mask_padded, batch_idx, positions, actions)

        self.assertTrue(torch.allclose(base, padded, atol=1e-6),
                        "Pad tokens (mask 0) must not affect real positions")

    def test_forward_last_position_ignores_trailing_padding(self):
        head = _make_head(seed=9)
        L = 5
        context, mask = _padded_batch([L], seed=31)
        torch.manual_seed(101)
        pad = torch.randn(1, 4, D_MODEL) * 50.0
        context_padded = torch.cat([context, pad], dim=1)
        mask_padded = torch.cat([mask, torch.zeros(1, 4)], dim=1)

        with torch.no_grad():
            base = head(context, mask=mask)
            padded = head(context_padded, mask=mask_padded)
        self.assertTrue(torch.allclose(base, padded, atol=1e-6),
                        "forward must anchor at the last TRUE position, "
                        "unaffected by trailing padding")


# ── 6. agent.forward_batch integration + checkpoint compat ──────────────────

class TestAgentIntegration(unittest.TestCase):

    def _make_event_seqs(self):
        return [
            [_make_event(None), _make_event(1), _make_event(None)],
            [_make_event(None), _make_event(3)],
        ]

    def test_forward_batch_modelling_head_only(self):
        torch.manual_seed(0)
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        agent.eval()
        with torch.no_grad():
            result = agent.forward_batch(self._make_event_seqs(),
                                         heads={"modelling"})
        self.assertEqual(set(result.keys()), {"action_embeddings"})
        self.assertEqual(result["action_embeddings"].shape,
                         (2, N_ACTIONS, D_MODEL))

    def test_forward_batch_all_heads_still_work(self):
        torch.manual_seed(0)
        agent = ASI(lambda m: None, config=_TINY_CONFIG)
        agent.eval()
        with torch.no_grad():
            result = agent.forward_batch(self._make_event_seqs())
        self.assertEqual(
            set(result.keys()),
            {"action_logits", "opponent_action_logits", "value",
             "action_embeddings"},
        )
        self.assertEqual(result["action_embeddings"].shape,
                         (2, N_ACTIONS, D_MODEL))

    def test_old_checkpoint_loads_with_strict_false(self):
        """A state_dict with OLD modelling_head keys (cross_attns, cross_norms)
        and no new MLP keys loads with strict=False: all other modules are
        restored, the new head's MLP stays freshly initialized."""
        torch.manual_seed(1)
        source = ASI(lambda m: None, config=_TINY_CONFIG)
        old_sd = {k: v.clone() for k, v in source.state_dict().items()}

        # Simulate an OLD checkpoint: drop the new action-MLP keys, add
        # synthetic cross-attention keys as the old head would have saved.
        for k in list(old_sd.keys()):
            if k.startswith("modelling_head.mlp_in") or \
               k.startswith("modelling_head.mlp_out"):
                del old_sd[k]
        old_sd["modelling_head.cross_norms.0.weight"] = torch.ones(D_MODEL)
        old_sd["modelling_head.cross_attns.0.q_proj.weight"] = \
            torch.zeros(D_MODEL, D_MODEL)
        old_sd["modelling_head.cross_attns.0.k_proj.weight"] = \
            torch.zeros(D_MODEL // 2, D_MODEL)
        old_sd["modelling_head.cross_attns.0.v_proj.weight"] = \
            torch.zeros(D_MODEL // 2, D_MODEL)
        old_sd["modelling_head.cross_attns.0.o_proj.weight"] = \
            torch.zeros(D_MODEL, D_MODEL)

        torch.manual_seed(2)
        target = ASI(lambda m: None, config=_TINY_CONFIG)
        fresh_mlp_in = target.modelling_head.mlp_in.weight.detach().clone()

        missing, unexpected = target.load_state_dict(old_sd, strict=False)

        # New MLP keys are missing (stay randomly initialized) …
        self.assertIn("modelling_head.mlp_in.weight", missing)
        self.assertIn("modelling_head.mlp_out.weight", missing)
        # … old cross-attention keys are ignored …
        self.assertIn("modelling_head.cross_attns.0.q_proj.weight", unexpected)
        self.assertIn("modelling_head.cross_norms.0.weight", unexpected)
        # … and every other module is fully restored from the checkpoint.
        self.assertTrue(torch.equal(
            target.value_head.output.weight
            if hasattr(target.value_head, "output")
            else next(target.value_head.parameters()),
            source.value_head.output.weight
            if hasattr(source.value_head, "output")
            else next(source.value_head.parameters()),
        ))
        for name, p in target.perception.named_parameters():
            self.assertTrue(
                torch.equal(p, old_sd["perception." + name]),
                f"perception.{name} should be restored from the checkpoint",
            )
        # The fresh head initialization is untouched by the old keys.
        self.assertTrue(torch.equal(
            target.modelling_head.mlp_in.weight, fresh_mlp_in))

        # The loaded agent still produces action embeddings.
        target.eval()
        with torch.no_grad():
            result = target.forward_batch(self._make_event_seqs(),
                                          heads={"modelling"})
        self.assertEqual(result["action_embeddings"].shape,
                         (2, N_ACTIONS, D_MODEL))


if __name__ == "__main__":
    unittest.main()
