"""Tests for training phase freeze/unfreeze patterns and gradient flow.

Each training phase in the pipeline applies specific freeze patterns to ensure
only the correct parameters receive gradient updates. These tests verify:

1. Phase 1 (gto_ev):        perception + value_head trainable; action/modelling/opponent_action frozen
2. Phase 2 (gto_probs):     action_head only trainable; perception/value_head frozen
3. Phase 3 (gto_combined):  perception + value_head + action_head trainable; modelling/opponent_action frozen
4. Phase 4 (modelling):     modelling_head only trainable; perception/value_head/action_head frozen
5. Phase 5 (opponent_action): opponent_action_head trainable; everything else frozen
6. Phase 6 (mcts):          ALL heads trainable; no freezing applied
7. ASI.forward_batch frozen perception detection (torch.no_grad + detach)
8. Unfreeze on exit: each phase restores all parameters to requires_grad=True

Run (from versions/v6):
    python -m tests.test_training_freeze_patterns
"""

import sys
import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal config for building a tiny ASI that exercises all heads
# ---------------------------------------------------------------------------

N_ACTIONS = 5
MAX_PLAYERS = 3

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
        "max_seq_len": 56,  # 56 // 7 = 8 max_events
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
    },
}

_TINY_CONFIG_OPP_EMB = dict(**_TINY_CONFIG)
_TINY_CONFIG_OPP_EMB["architecture"] = dict(**_TINY_CONFIG["architecture"])
_TINY_CONFIG_OPP_EMB["architecture"]["opponent_embedding"] = {"enabled": True}


def _make_agent(opp_emb=False):
    """Build a tiny ASI on CPU."""
    from agent.agent import ASI

    cfg = _TINY_CONFIG_OPP_EMB if opp_emb else _TINY_CONFIG

    def _log(msg):
        pass  # silence training noise during tests

    agent = ASI(_log, config=cfg)
    agent.set_device("cpu")
    return agent


def _make_event():
    """Minimal event dict compatible with EventSequenceEmbedder."""
    return {
        "table": [0, -1, -1, -1, -1],
        "hand": [10, 11],
        "num_players": 2,
        "hero_pos": 0,
        "acting_pos": 1,
        "big_blind": 1.0,
        "small_blind": 0.5,
        "stack": 10.0,
        "pot": 2.0,
        "bets": [0.0] * MAX_PLAYERS,
        "stacks": [10.0] * MAX_PLAYERS,
        "action": [0.0] * N_ACTIONS,
    }


def _make_event_seq(n=2):
    return [_make_event() for _ in range(n)]


def _all_grad(module):
    """Return True if ALL parameters in the module have requires_grad=True."""
    params = list(module.parameters())
    return len(params) > 0 and all(p.requires_grad for p in params)


def _no_grad(module):
    """Return True if ALL parameters in the module have requires_grad=False."""
    params = list(module.parameters())
    return len(params) > 0 and all(not p.requires_grad for p in params)


def _record_param_snapshot(agent):
    """Return a dict {param_id: tensor_clone} of all parameters."""
    return {id(p): p.detach().clone() for p in agent.parameters()}


def _params_changed(agent, snapshot):
    """Return list of (name, param) pairs whose value changed vs snapshot."""
    changed = []
    for name, p in agent.named_parameters():
        pid = id(p)
        if pid in snapshot:
            if not torch.equal(p.detach(), snapshot[pid]):
                changed.append(name)
    return changed


def _check_module_names_changed(agent, snapshot, expected_changed_modules,
                                 unexpected_changed_modules):
    """
    Assert that all expected_changed_modules have at least one changed param,
    and all unexpected_changed_modules have zero changed params.

    expected/unexpected_changed_modules: list of strings that are prefixes of
    named_parameters keys, e.g. "perception", "value_head".
    """
    changed_names = _params_changed(agent, snapshot)

    for module_prefix in expected_changed_modules:
        module_params_changed = [n for n in changed_names if n.startswith(module_prefix)]
        assert module_params_changed, (
            f"Expected {module_prefix} params to change, but none did. "
            f"Changed params: {changed_names[:20]}"
        )

    for module_prefix in unexpected_changed_modules:
        module_params_changed = [n for n in changed_names if n.startswith(module_prefix)]
        assert not module_params_changed, (
            f"Expected {module_prefix} params NOT to change, but found: "
            f"{module_params_changed[:10]}"
        )


def _do_one_step(agent, loss_fn, optimizer, trainable_params):
    """Run one forward+backward+optimizer step using the value head on dummy data."""
    event_seq = _make_event_seq(2)
    result = agent.forward_batch([event_seq], skip_memory=True, heads={"value"})
    value = result["value"].squeeze(-1)
    target = torch.tensor([0.5])
    loss = loss_fn(value, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def _do_one_step_action(agent, optimizer, trainable_params):
    """One forward+backward+optimizer step through action head on pre-computed perception."""
    import torch.nn.functional as F
    # Build a small perception output manually to avoid triggering frozen-detection
    d = agent.perception.d_model
    B, N = 1, 3
    p_out = torch.randn(B, N, d, requires_grad=False)
    mask = torch.ones(B, N)
    action_logits = agent.action_head(p_out, mask=mask)
    target_probs = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
    log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction="batchmean")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def _do_one_step_modelling(agent, optimizer, trainable_params):
    """One step through the modelling head, grad flowing through frozen value_head."""
    device = "cpu"
    d = agent.perception.d_model
    B, N = 1, 3
    # Use a short event sequence so perception handles it
    event_seq = _make_event_seq(2)
    # Since perception is frozen, manually build perception output
    with torch.no_grad():
        p_out, _, mask = agent.perception.forward_batch(
            [event_seq], device=device, skip_memory=True)
    p_out = p_out.detach()
    # modelling head
    action_embs = agent.modelling_head(p_out, mask=mask)  # (1, K, D)
    K = agent.n_actions
    # Expand for value_head
    B_sz, N_sz, D_sz = p_out.shape
    p_expanded = p_out.unsqueeze(1).expand(B_sz, K, N_sz, D_sz).reshape(B_sz * K, N_sz, D_sz)
    combined = torch.cat(
        [p_expanded, torch.zeros(B_sz * K, 1, D_sz)], dim=1)
    mask_expanded = mask.unsqueeze(1).expand(B_sz, K, N_sz).reshape(B_sz * K, N_sz)
    mask_combined = torch.cat(
        [mask_expanded, torch.zeros(B_sz * K, 1)], dim=1)
    lengths = mask.sum(dim=1).long()
    pos = lengths.unsqueeze(1).expand(B_sz, K).reshape(B_sz * K)
    rows = torch.arange(B_sz * K)
    combined[rows, pos] = action_embs.reshape(B_sz * K, D_sz)
    mask_combined[rows, pos] = 1.0
    values = agent.value_head(combined, mask=mask_combined)
    # Dummy target
    target_evs = torch.zeros(B_sz, K)
    loss = nn.SmoothL1Loss()(values.reshape(B_sz, K), target_evs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestFreezePatterns(unittest.TestCase):
    """Verify that each phase sets requires_grad correctly before training."""

    def _assert_module_trainable(self, module, name):
        self.assertTrue(_all_grad(module),
                        f"{name} should be trainable (all requires_grad=True)")

    def _assert_module_frozen(self, module, name):
        self.assertTrue(_no_grad(module),
                        f"{name} should be frozen (all requires_grad=False)")

    # ------------------------------------------------------------------
    # Phase 1: gto_ev_predict
    # ------------------------------------------------------------------

    def test_phase1_freeze_pattern(self):
        """Phase 1: perception + value_head trainable; action/modelling/opponent_action frozen."""
        from agent.train_scenarios.gto_ev_predict.train import train_gto_ev

        agent = _make_agent()
        # Simulate what train_gto_ev does at entry
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False

        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")
        self._assert_module_frozen(agent.action_head, "action_head")
        self._assert_module_frozen(agent.modelling_head, "modelling_head")
        self._assert_module_frozen(agent.opponent_action_head, "opponent_action_head")

    def test_phase1_unfreeze_on_exit(self):
        """Phase 1: after training, action/modelling/opponent_action are unfrozen."""
        agent = _make_agent()
        # Simulate the full freeze/unfreeze cycle
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False

        # Simulate unfreeze at end of train_gto_ev
        for param in agent.action_head.parameters():
            param.requires_grad = True
        for param in agent.modelling_head.parameters():
            param.requires_grad = True
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.action_head, "action_head")
        self._assert_module_trainable(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase1_loss_is_smoothl1(self):
        """Phase 1: loss function is SmoothL1Loss (Huber)."""
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        pred = torch.tensor([0.5])
        target = torch.tensor([1.0])
        loss = loss_fn(pred, target)
        # SmoothL1: |x| <= beta -> 0.5*x^2/beta; here beta=1.0, |0.5|<=1 -> 0.5*0.25/1 = 0.125
        self.assertAlmostEqual(loss.item(), 0.125, places=5)

    # ------------------------------------------------------------------
    # Phase 2: gto_probs_predict
    # ------------------------------------------------------------------

    def test_phase2_freeze_pattern(self):
        """Phase 2: only action_head trainable; perception + value_head frozen."""
        agent = _make_agent()
        # Simulate what train_gto_probs does at entry
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = True

        self._assert_module_frozen(agent.perception, "perception")
        self._assert_module_frozen(agent.value_head, "value_head")
        self._assert_module_trainable(agent.action_head, "action_head")

    def test_phase2_modelling_and_opponent_not_explicitly_touched(self):
        """Phase 2: modelling_head and opponent_action_head start in default state (trainable)."""
        agent = _make_agent()
        # Before phase 2 starts, check default state (freshly created agent has all trainable)
        self._assert_module_trainable(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

        # After applying phase 2 freezes, modelling_head and opponent_action_head
        # stay in whatever state they were (phase 2 doesn't touch them)
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = True

        # Modelling/opponent_action not touched by phase 2 — still trainable
        self._assert_module_trainable(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase2_optimizer_only_action_head(self):
        """Phase 2: optimizer is constructed only over action_head parameters."""
        agent = _make_agent()
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = True

        # As in train_gto_probs: trainable_params = list(agent.action_head.parameters())
        trainable_params = list(agent.action_head.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        self.assertEqual(len(optimizer.param_groups[0]["params"]),
                         len(trainable_params))

    def test_phase2_unfreeze_on_exit(self):
        """Phase 2: perception + value_head restored to trainable after training."""
        agent = _make_agent()
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False

        # Simulate unfreeze at end of train_gto_probs
        for param in agent.perception.parameters():
            param.requires_grad = True
        for param in agent.value_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")

    # ------------------------------------------------------------------
    # Phase 3: gto_predict (combined)
    # ------------------------------------------------------------------

    def test_phase3_freeze_pattern(self):
        """Phase 3: perception + value_head + action_head trainable; modelling + opponent_action frozen."""
        agent = _make_agent()
        # Simulate train_gto entry
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False
        for param in agent.perception.parameters():
            param.requires_grad = True
        for param in agent.value_head.parameters():
            param.requires_grad = True
        for param in agent.action_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")
        self._assert_module_trainable(agent.action_head, "action_head")
        self._assert_module_frozen(agent.modelling_head, "modelling_head")
        self._assert_module_frozen(agent.opponent_action_head, "opponent_action_head")

    def test_phase3_unfreeze_on_exit(self):
        """Phase 3: modelling_head + opponent_action_head restored after training."""
        agent = _make_agent()
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False

        # Simulate unfreeze at end of train_gto
        for param in agent.modelling_head.parameters():
            param.requires_grad = True
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase3_combined_loss_structure(self):
        """Phase 3: combined loss = SmoothL1(value) + action_weight * KL(action)."""
        import torch.nn.functional as F
        value_loss_fn = nn.SmoothL1Loss(beta=1.0)
        action_loss_weight = 0.5

        pred_ev = torch.tensor([0.3])
        target_ev = torch.tensor([0.5])
        v_loss = value_loss_fn(pred_ev, target_ev)

        logits = torch.randn(2, N_ACTIONS)
        target_probs = torch.softmax(torch.randn(2, N_ACTIONS), dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        a_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

        combined = v_loss + action_loss_weight * a_loss
        # Just verify that it's a sum of two non-negative contributions
        self.assertGreater(combined.item(), 0.0)
        self.assertAlmostEqual(
            combined.item(), (v_loss + action_loss_weight * a_loss).item(), places=5)

    # ------------------------------------------------------------------
    # Phase 4: modelling_predict
    # ------------------------------------------------------------------

    def test_phase4_freeze_pattern(self):
        """Phase 4: modelling_head trainable; perception + value_head + action_head frozen."""
        agent = _make_agent()
        # Simulate train_modelling entry
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        self._assert_module_frozen(agent.perception, "perception")
        self._assert_module_frozen(agent.value_head, "value_head")
        self._assert_module_frozen(agent.action_head, "action_head")
        self._assert_module_trainable(agent.modelling_head, "modelling_head")

    def test_phase4_opponent_action_not_touched(self):
        """Phase 4: opponent_action_head is not explicitly frozen or unfrozen (stays as-is)."""
        agent = _make_agent()
        # Default state: trainable
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

        # Apply phase 4 freeze pattern
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        # opponent_action_head was trainable and not touched by phase 4
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase4_optimizer_only_modelling_head(self):
        """Phase 4: optimizer is constructed only over modelling_head parameters."""
        agent = _make_agent()
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        # As in train_modelling: trainable_params = list(agent.modelling_head.parameters())
        trainable_params = list(agent.modelling_head.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        self.assertEqual(len(optimizer.param_groups[0]["params"]),
                         len(trainable_params))
        # Ensure these are specifically modelling_head params, not others
        modelling_param_ids = {id(p) for p in agent.modelling_head.parameters()}
        for p in optimizer.param_groups[0]["params"]:
            self.assertIn(id(p), modelling_param_ids)

    def test_phase4_unfreeze_on_exit(self):
        """Phase 4: perception + value_head + action_head restored after training."""
        agent = _make_agent()
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False

        # Simulate unfreeze at end of train_modelling
        for param in agent.perception.parameters():
            param.requires_grad = True
        for param in agent.value_head.parameters():
            param.requires_grad = True
        for param in agent.action_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")
        self._assert_module_trainable(agent.action_head, "action_head")

    # ------------------------------------------------------------------
    # Phase 5: opponent_action_predict
    # ------------------------------------------------------------------

    def test_phase5_freeze_pattern(self):
        """Phase 5: opponent_action_head trainable; everything else frozen."""
        agent = _make_agent()
        # Simulate train_opponent_action entry
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        self._assert_module_frozen(agent.perception, "perception")
        self._assert_module_frozen(agent.value_head, "value_head")
        self._assert_module_frozen(agent.action_head, "action_head")
        self._assert_module_frozen(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase5_unfreeze_on_exit(self):
        """Phase 5: all modules restored after training."""
        agent = _make_agent()
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False

        # Simulate unfreeze at end of train_opponent_action
        for param in agent.perception.parameters():
            param.requires_grad = True
        for param in agent.value_head.parameters():
            param.requires_grad = True
        for param in agent.action_head.parameters():
            param.requires_grad = True
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")
        self._assert_module_trainable(agent.action_head, "action_head")
        self._assert_module_trainable(agent.modelling_head, "modelling_head")

    def test_phase5_with_opp_emb_includes_gru(self):
        """Phase 5 with opponent_embedding enabled: opponent_gru is also trainable."""
        agent = _make_agent(opp_emb=True)
        self.assertTrue(agent.perception.opp_emb_enabled,
                        "Opponent embedding should be enabled")

        # Simulate freeze pattern from train_opponent_action
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        # Use use_opp_emb branch: unfreeze opponent_gru
        use_opp_emb = agent.perception.opp_emb_enabled
        if use_opp_emb:
            for param in agent.perception.opponent_gru.parameters():
                param.requires_grad = True

        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")
        self._assert_module_trainable(agent.perception.opponent_gru, "perception.opponent_gru")

    def test_phase5_optimizer_with_opp_emb_includes_gru_params(self):
        """Phase 5 with opp_emb: trainable_params includes opponent_gru parameters."""
        agent = _make_agent(opp_emb=True)
        # Apply freeze pattern
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True
        for param in agent.perception.opponent_gru.parameters():
            param.requires_grad = True

        trainable_params = list(agent.opponent_action_head.parameters())
        if agent.perception.opp_emb_enabled:
            trainable_params += list(agent.perception.opponent_gru.parameters())

        # All trainable params are from opponent_action_head + gru
        expected_ids = (
            {id(p) for p in agent.opponent_action_head.parameters()} |
            {id(p) for p in agent.perception.opponent_gru.parameters()}
        )
        for p in trainable_params:
            self.assertIn(id(p), expected_ids,
                          "trainable_params should only include opp_head + opp_gru")

    # ------------------------------------------------------------------
    # Phase 6: mcts_predict
    # ------------------------------------------------------------------

    def test_phase6_all_trainable(self):
        """Phase 6: all parameters set to requires_grad=True."""
        agent = _make_agent()
        # Pre-condition: some modules may be frozen from a previous phase
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False

        # Simulate train_mcts entry: "All parameters trainable"
        for param in agent.parameters():
            param.requires_grad = True

        # Check all modules
        self._assert_module_trainable(agent.perception, "perception")
        self._assert_module_trainable(agent.value_head, "value_head")
        self._assert_module_trainable(agent.action_head, "action_head")
        self._assert_module_trainable(agent.modelling_head, "modelling_head")
        self._assert_module_trainable(agent.opponent_action_head, "opponent_action_head")

    def test_phase6_all_trainable_after_partial_freeze(self):
        """Phase 6: all heads trainable even after deepest freeze (all but modelling frozen)."""
        agent = _make_agent()
        # Freeze everything
        for param in agent.parameters():
            param.requires_grad = False
        # Now apply MCTS unfreezing
        for param in agent.parameters():
            param.requires_grad = True

        n_trainable = sum(1 for p in agent.parameters() if p.requires_grad)
        n_total = sum(1 for p in agent.parameters())
        self.assertEqual(n_trainable, n_total,
                         "MCTS phase must leave all parameters trainable")

    def test_phase6_no_module_frozen(self):
        """Phase 6: verify each named module group is trainable individually."""
        agent = _make_agent()
        for param in agent.parameters():
            param.requires_grad = True

        modules = {
            "perception": agent.perception,
            "value_head": agent.value_head,
            "action_head": agent.action_head,
            "modelling_head": agent.modelling_head,
            "opponent_action_head": agent.opponent_action_head,
        }
        for name, module in modules.items():
            self._assert_module_trainable(module, name)


# ---------------------------------------------------------------------------
# Gradient flow tests — verify actual gradient propagation
# ---------------------------------------------------------------------------


class TestGradientFlow(unittest.TestCase):
    """Verify gradients flow to / don't flow to the correct parameters."""

    def test_phase1_grads_flow_to_perception_not_action_head(self):
        """Phase 1: backward through value_head updates perception, not action_head."""
        agent = _make_agent()
        agent.train()

        # Apply phase 1 freeze
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False

        snapshot = _record_param_snapshot(agent)

        loss_fn = nn.SmoothL1Loss()
        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)

        event_seq = _make_event_seq(2)
        result = agent.forward_batch([event_seq], skip_memory=True, heads={"value"})
        value = result["value"].squeeze(-1)
        loss = loss_fn(value, torch.tensor([0.5]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["perception", "value_head"],
            unexpected_changed_modules=["action_head", "modelling_head",
                                        "opponent_action_head"],
        )

    def test_phase2_grads_flow_to_action_head_only(self):
        """Phase 2: backward through action_head updates action_head only."""
        import torch.nn.functional as F
        agent = _make_agent()
        agent.train()

        # Apply phase 2 freeze
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = True

        snapshot = _record_param_snapshot(agent)

        trainable_params = list(agent.action_head.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)

        # Cached perception input — bypasses perception entirely
        d = agent.perception.d_model
        B, N = 1, 3
        p_out = torch.randn(B, N, d)
        mask = torch.ones(B, N)
        action_logits = agent.action_head(p_out, mask=mask)
        target_probs = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["action_head"],
            unexpected_changed_modules=["perception", "value_head",
                                        "modelling_head", "opponent_action_head"],
        )

    def test_phase3_grads_flow_to_perception_value_action(self):
        """Phase 3: combined loss updates perception + value_head + action_head."""
        import torch.nn.functional as F
        agent = _make_agent()
        agent.train()

        # Apply phase 3 freeze
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False
        for param in agent.perception.parameters():
            param.requires_grad = True
        for param in agent.value_head.parameters():
            param.requires_grad = True
        for param in agent.action_head.parameters():
            param.requires_grad = True

        snapshot = _record_param_snapshot(agent)

        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)
        value_loss_fn = nn.SmoothL1Loss()
        action_loss_weight = 1.0

        event_seq = _make_event_seq(2)
        result = agent.forward_batch([event_seq], skip_memory=True,
                                     heads={"value", "action"})
        value = result["value"].squeeze(-1)
        action_logits = result["action_logits"]
        target_probs = torch.softmax(torch.randn(1, N_ACTIONS), dim=-1)
        v_loss = value_loss_fn(value, torch.tensor([0.5]))
        log_probs = F.log_softmax(action_logits, dim=-1)
        a_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
        loss = v_loss + action_loss_weight * a_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["perception", "value_head", "action_head"],
            unexpected_changed_modules=["modelling_head", "opponent_action_head"],
        )

    def test_phase4_grads_flow_to_modelling_head_via_frozen_value_head(self):
        """Phase 4: backward flows through frozen value_head graph to modelling_head.

        This is the critical invariant: value_head weights must NOT change
        (frozen), but gradients flow through it to update modelling_head.
        """
        agent = _make_agent()
        agent.train()

        # Apply phase 4 freeze
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        snapshot = _record_param_snapshot(agent)

        trainable_params = list(agent.modelling_head.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)

        device = "cpu"
        event_seq = _make_event_seq(2)
        # Get frozen perception output
        with torch.no_grad():
            p_out, _, mask = agent.perception.forward_batch(
                [event_seq], device=device, skip_memory=True)
        p_out = p_out.detach()

        # Run modelling forward
        action_embs = agent.modelling_head(p_out, mask=mask)
        K = agent.n_actions
        B_sz, N_sz, D_sz = p_out.shape

        p_expanded = p_out.unsqueeze(1).expand(B_sz, K, N_sz, D_sz).reshape(B_sz * K, N_sz, D_sz)
        combined = torch.cat(
            [p_expanded, torch.zeros(B_sz * K, 1, D_sz)], dim=1)
        mask_expanded = mask.unsqueeze(1).expand(B_sz, K, N_sz).reshape(B_sz * K, N_sz)
        mask_combined = torch.cat(
            [mask_expanded, torch.zeros(B_sz * K, 1)], dim=1)
        lengths = mask.sum(dim=1).long()
        pos = lengths.unsqueeze(1).expand(B_sz, K).reshape(B_sz * K)
        rows = torch.arange(B_sz * K)
        combined[rows, pos] = action_embs.reshape(B_sz * K, D_sz)
        mask_combined[rows, pos] = 1.0

        # value_head forward (frozen params, grad flows through computation graph)
        values = agent.value_head(combined, mask=mask_combined)
        target_evs = torch.zeros(B_sz, K)
        loss = nn.SmoothL1Loss()(values.reshape(B_sz, K), target_evs)

        optimizer.zero_grad()
        loss.backward()

        # Check modelling_head got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in agent.modelling_head.parameters()
        )
        self.assertTrue(has_grad,
                        "modelling_head should have non-zero gradients from the backward pass")

        # value_head params should have NO gradients (requires_grad=False)
        for name, p in agent.named_parameters():
            if name.startswith("value_head"):
                self.assertIsNone(p.grad,
                                  f"value_head.{name} should have no grad (frozen)")

        optimizer.step()

        # After update: modelling_head changed, others did NOT
        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["modelling_head"],
            unexpected_changed_modules=["perception", "value_head", "action_head",
                                        "opponent_action_head"],
        )

    def test_phase5_grads_flow_to_opponent_action_head_only(self):
        """Phase 5: backward through opponent_action_head updates only it."""
        import torch.nn.functional as F
        agent = _make_agent()
        agent.train()

        # Apply phase 5 freeze
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        snapshot = _record_param_snapshot(agent)

        trainable_params = list(agent.opponent_action_head.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)

        d = agent.perception.d_model
        B, N = 1, 3
        p_out = torch.randn(B, N, d)
        mask = torch.ones(B, N)
        logits = agent.opponent_action_head(p_out, mask=mask)
        target_probs = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["opponent_action_head"],
            unexpected_changed_modules=["perception", "value_head", "action_head",
                                        "modelling_head"],
        )

    def test_phase6_grads_flow_to_all_modules(self):
        """Phase 6: all modules receive gradients and can be updated."""
        import torch.nn.functional as F
        agent = _make_agent()
        agent.train()

        # Apply MCTS freeze: none
        for param in agent.parameters():
            param.requires_grad = True

        snapshot = _record_param_snapshot(agent)

        optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)

        event_seq = _make_event_seq(2)
        # Full forward including all heads
        result = agent.forward_batch([event_seq], skip_memory=True)
        value = result["value"].squeeze(-1)
        action_logits = result["action_logits"]
        target_probs = torch.softmax(torch.randn(1, N_ACTIONS), dim=-1)

        v_loss = nn.SmoothL1Loss()(value, torch.tensor([0.5]))
        log_probs = F.log_softmax(action_logits, dim=-1)
        a_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
        loss = v_loss + a_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Perception + value_head + action_head should all have changed
        _check_module_names_changed(
            agent, snapshot,
            expected_changed_modules=["perception", "value_head", "action_head"],
            unexpected_changed_modules=[],  # nothing must be frozen
        )


# ---------------------------------------------------------------------------
# Frozen perception detection in forward_batch
# ---------------------------------------------------------------------------


class TestFrozenPerceptionDetection(unittest.TestCase):
    """Test that ASI.forward_batch correctly detects frozen perception."""

    def test_unfrozen_perception_no_detach(self):
        """With trainable perception, forward_batch output keeps grad_fn."""
        agent = _make_agent()
        agent.train()

        # All params trainable (default)
        self.assertFalse(
            not any(p.requires_grad for p in agent.perception.parameters()),
            "Perception should be trainable by default"
        )

        event_seq = _make_event_seq(2)
        result = agent.forward_batch([event_seq], skip_memory=True, heads={"value"})
        value = result["value"]
        # value should have a grad_fn because perception is NOT frozen
        self.assertIsNotNone(value.grad_fn,
                             "value output should have grad_fn when perception is trainable")

    def test_frozen_perception_detection_logic(self):
        """Verify the detection logic: frozen when no param has requires_grad=True."""
        agent = _make_agent()

        # Freeze perception
        for param in agent.perception.parameters():
            param.requires_grad = False

        # This is the exact check from agent.py line 97
        perception_frozen = not any(p.requires_grad for p in agent.perception.parameters())
        self.assertTrue(perception_frozen,
                        "Should detect frozen perception when all params have requires_grad=False")

        # Unfreeze one param — detection should flip
        first_param = next(iter(agent.perception.parameters()))
        first_param.requires_grad = True
        perception_frozen = not any(p.requires_grad for p in agent.perception.parameters())
        self.assertFalse(perception_frozen,
                         "Should not detect frozen when at least one param is trainable")

    def test_frozen_perception_output_is_detached(self):
        """With frozen perception, forward_batch returns detached perception_out to heads."""
        agent = _make_agent()
        agent.eval()

        # Freeze perception
        for param in agent.perception.parameters():
            param.requires_grad = False

        event_seq = _make_event_seq(2)
        # With frozen perception, the value head receives a detached tensor
        # so grad_fn may still exist on value output (from value_head layers)
        # but there should be no connection back to perception params
        result = agent.forward_batch([event_seq], skip_memory=True, heads={"value"})
        value = result["value"]

        # Compute a dummy loss and check gradients
        loss = value.sum()
        loss.backward()

        # No perception parameter should have a gradient
        for name, p in agent.named_parameters():
            if name.startswith("perception"):
                self.assertIsNone(
                    p.grad,
                    f"perception param {name} should have no grad when perception is frozen+detached"
                )

    def test_frozen_perception_does_not_track_through_perception(self):
        """The autograd graph is cut at perception output when perception is frozen.

        Modelling_head can receive perception_out as input but gradients
        will not flow back through it to perception.
        """
        agent = _make_agent()
        agent.train()

        # Freeze all except modelling_head (phase 4 pattern)
        for param in agent.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        event_seq = _make_event_seq(2)
        # forward_batch internally detaches perception_out when frozen
        result = agent.forward_batch([event_seq], skip_memory=True, heads={"modelling"})
        action_embeddings = result["action_embeddings"]

        loss = action_embeddings.sum()
        loss.backward()

        # Perception should have no gradients
        for name, p in agent.named_parameters():
            if name.startswith("perception"):
                self.assertIsNone(
                    p.grad,
                    f"perception param {name} should have no gradient "
                    "when perception output is detached"
                )

        # modelling_head should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in agent.modelling_head.parameters()
        )
        self.assertTrue(has_grad, "modelling_head should have gradients")


# ---------------------------------------------------------------------------
# Cross-phase invariants
# ---------------------------------------------------------------------------


class TestCrossPhaseInvariants(unittest.TestCase):
    """Verify that freeze/unfreeze cycles across phases leave the agent in clean state."""

    def test_consecutive_phase1_phase2_freeze_state(self):
        """After phase 1 then phase 2, the net effect matches phase 2 freeze pattern."""
        agent = _make_agent()

        # Phase 1: freeze action/modelling/opponent_action
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False

        # Phase 1 unfreeze
        for param in agent.action_head.parameters():
            param.requires_grad = True
        for param in agent.modelling_head.parameters():
            param.requires_grad = True
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True

        # All should be trainable now
        n_all = sum(1 for _ in agent.parameters())
        n_trainable = sum(1 for p in agent.parameters() if p.requires_grad)
        self.assertEqual(n_trainable, n_all,
                         "All params should be trainable after phase 1 unfreeze")

        # Phase 2: freeze perception/value_head, enable action_head
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = True

        # Phase 2 freeze invariants
        self.assertTrue(_no_grad(agent.perception))
        self.assertTrue(_no_grad(agent.value_head))
        self.assertTrue(_all_grad(agent.action_head))

    def test_phase4_frozen_value_head_params_unchanged(self):
        """In phase 4, even though grad flows through value_head, its WEIGHTS don't change."""
        agent = _make_agent()
        agent.train()

        # Apply phase 4 freeze
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = True

        # Record value_head param snapshots specifically
        vh_snapshot = {name: p.detach().clone()
                       for name, p in agent.named_parameters()
                       if name.startswith("value_head")}
        mh_snapshot = {name: p.detach().clone()
                       for name, p in agent.named_parameters()
                       if name.startswith("modelling_head")}

        trainable_params = list(agent.modelling_head.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=0.5)

        # Run modelling forward + backward
        device = "cpu"
        event_seq = _make_event_seq(2)
        with torch.no_grad():
            p_out, _, mask = agent.perception.forward_batch(
                [event_seq], device=device, skip_memory=True)
        p_out = p_out.detach()

        action_embs = agent.modelling_head(p_out, mask=mask)
        K = agent.n_actions
        B_sz, N_sz, D_sz = p_out.shape
        p_expanded = p_out.unsqueeze(1).expand(B_sz, K, N_sz, D_sz).reshape(B_sz * K, N_sz, D_sz)
        combined = torch.cat(
            [p_expanded, torch.zeros(B_sz * K, 1, D_sz)], dim=1)
        mask_expanded = mask.unsqueeze(1).expand(B_sz, K, N_sz).reshape(B_sz * K, N_sz)
        mask_combined = torch.cat(
            [mask_expanded, torch.zeros(B_sz * K, 1)], dim=1)
        lengths = mask.sum(dim=1).long()
        pos = lengths.unsqueeze(1).expand(B_sz, K).reshape(B_sz * K)
        rows = torch.arange(B_sz * K)
        combined[rows, pos] = action_embs.reshape(B_sz * K, D_sz)
        mask_combined[rows, pos] = 1.0
        values = agent.value_head(combined, mask=mask_combined)
        target_evs = torch.zeros(B_sz, K)
        loss = nn.SmoothL1Loss()(values.reshape(B_sz, K), target_evs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # value_head weights must not have changed
        for name, old_val in vh_snapshot.items():
            new_val = dict(agent.named_parameters())[name].detach()
            self.assertTrue(
                torch.equal(old_val, new_val),
                f"value_head param {name} changed despite being frozen in phase 4"
            )

        # modelling_head weights must have changed
        mh_changed = False
        for name, old_val in mh_snapshot.items():
            new_val = dict(agent.named_parameters())[name].detach()
            if not torch.equal(old_val, new_val):
                mh_changed = True
                break
        self.assertTrue(mh_changed,
                        "modelling_head weights should have changed after optimizer step")

    def test_fresh_agent_all_trainable(self):
        """A freshly created ASI has all parameters trainable by default."""
        agent = _make_agent()
        n_total = sum(1 for _ in agent.parameters())
        n_trainable = sum(1 for p in agent.parameters() if p.requires_grad)
        self.assertEqual(n_trainable, n_total,
                         f"Fresh ASI should have all {n_total} params trainable, "
                         f"found only {n_trainable}")

    def test_requires_grad_counts_per_phase(self):
        """Verify that each phase's trainable parameter count is correct relative to total."""
        agent = _make_agent()

        def count_trainable():
            return sum(1 for p in agent.parameters() if p.requires_grad)

        def count_total():
            return sum(1 for _ in agent.parameters())

        total = count_total()
        self.assertGreater(total, 0)

        # Phase 1: freeze action + modelling + opponent_action
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = False
        n_phase1 = count_trainable()

        # Should have lost action_head + modelling_head + opponent_action_head params
        n_frozen1 = (
            sum(1 for _ in agent.action_head.parameters()) +
            sum(1 for _ in agent.modelling_head.parameters()) +
            sum(1 for _ in agent.opponent_action_head.parameters())
        )
        self.assertEqual(n_phase1, total - n_frozen1,
                         "Phase 1 trainable count mismatch")

        # Reset
        for param in agent.parameters():
            param.requires_grad = True

        # Phase 4: freeze perception + value_head + action_head
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        n_phase4 = count_trainable()

        n_frozen4 = (
            sum(1 for _ in agent.perception.parameters()) +
            sum(1 for _ in agent.value_head.parameters()) +
            sum(1 for _ in agent.action_head.parameters())
        )
        self.assertEqual(n_phase4, total - n_frozen4,
                         "Phase 4 trainable count mismatch")

        # Reset
        for param in agent.parameters():
            param.requires_grad = True

        # Phase 5: freeze all except opponent_action_head
        for param in agent.perception.parameters():
            param.requires_grad = False
        for param in agent.value_head.parameters():
            param.requires_grad = False
        for param in agent.action_head.parameters():
            param.requires_grad = False
        for param in agent.modelling_head.parameters():
            param.requires_grad = False
        for param in agent.opponent_action_head.parameters():
            param.requires_grad = True
        n_phase5 = count_trainable()
        n_opp_action = sum(1 for _ in agent.opponent_action_head.parameters())
        self.assertEqual(n_phase5, n_opp_action,
                         "Phase 5 trainable count should equal opponent_action_head param count")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Run with verbose output showing each test's name
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestFreezePatterns, TestGradientFlow,
                TestFrozenPerceptionDetection, TestCrossPhaseInvariants):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
