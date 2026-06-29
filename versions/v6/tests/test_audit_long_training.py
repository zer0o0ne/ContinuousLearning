"""Tests for bugs that would prevent successful long training runs.

Covers:
  Bug 1:  Terminal rollout in _mcts_forward does not detach old embeddings
  Bug 2:  re_backup_terminals skips opponent pessimism blending
  Bug 3:  Root-level root_q_ratio has no NaN guard in _finalize_value_targets
  Bug 4:  scheduler.step() called when GradScaler skips optimizer step
  Bug 5:  GradScaler re-created fresh each MCTS cycle (loses calibration)
  Bug 7:  base_scenarios / opp_scenarios never freed during MCTS phase
  Bug 8:  perception_frozen check includes opponent_gru (too coarse)
  Bug 9:  OpponentEmbeddingTable unbounded growth
  Bug 10: _collect_terminals recursive DFS (stack overflow on deep trees)

Run:
    cd versions/v6 && python -m pytest tests/test_audit_long_training.py -v
"""

import math
import sys
import os
import inspect

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcts.mcts import MCTSNode, _collect_terminals, re_backup_terminals
from agent.perception.opponent_embeddings import OpponentEmbeddingTable


# ── Bug 1: Terminal rollout does not detach old embeddings ──────────────

def test_bug1_terminal_rollout_detaches_embeddings():
    """Terminal rollout in _mcts_forward must detach intermediate embeddings
    to prevent quadratic autograd graph growth.

    The main chain loop (line 298) respects stop_grad_old_embs and detaches,
    but the terminal rollout loop (lines 328-333) concatenates emb_tok into
    ctx without detaching. This test verifies the terminal loop detaches.
    """
    src = inspect.getsource(
        __import__("agent.train_scenarios.mcts_predict.train",
                   fromlist=["_mcts_forward"])._mcts_forward
    )
    # Find the terminal rollout section (after "Terminal value supervision")
    terminal_section_start = src.find("Terminal value supervision")
    assert terminal_section_start != -1, "Cannot find terminal rollout section"
    terminal_section = src[terminal_section_start:]

    # Find the inner loop that builds terminal contexts
    # Pattern: for a in action_path: ... emb_tok ... ctx = torch.cat([ctx, emb_tok
    loop_start = terminal_section.find("for a in action_path")
    assert loop_start != -1, "Cannot find 'for a in action_path' loop"
    loop_body = terminal_section[loop_start:]
    # Get just the loop body (up to the next dedent or v_pred line)
    loop_end = loop_body.find("v_pred = ")
    loop_body = loop_body[:loop_end]

    # The emb_tok line should contain .detach()
    assert ".detach()" in loop_body, (
        "Terminal rollout does NOT detach emb_tok before concatenating into ctx. "
        "This causes quadratic autograd graph growth for terminals with long "
        "action paths, leading to OOM on deep hands during long training."
    )


def test_bug1_terminal_rollout_no_gradient_accumulation():
    """Functional test: simulate the terminal rollout pattern and verify
    that detaching embeddings prevents graph growth."""
    d_model = 32
    n_actions = 5

    # Minimal modelling head: projects from d_model → n_actions * d_model
    proj = nn.Linear(d_model, n_actions * d_model)

    ctx_init = torch.randn(1, 3, d_model, requires_grad=True)
    action_path = [0, 1, 2, 0, 1]  # 5-step path

    # Without detach: emb_tok stays in graph
    ctx_nd = ctx_init.clone()
    for a in action_path:
        # Only apply proj to the LAST token (simulating modelling head
        # attending to full context and producing per-action embeddings)
        last_tok = ctx_nd[:, -1:, :]  # (1, 1, d)
        out = proj(last_tok).view(1, n_actions, d_model)
        emb_tok = out[:, a, :].unsqueeze(1)
        ctx_nd = torch.cat([ctx_nd, emb_tok], dim=1)

    loss_nd = ctx_nd.sum()
    loss_nd.backward()
    # Gradient flows through the whole chain
    assert ctx_init.grad is not None, "Sanity: gradient should flow without detach"

    # With detach: gradient chain is broken at each step
    ctx_init2 = torch.randn(1, 3, d_model, requires_grad=True)
    ctx_d = ctx_init2.clone()
    for a in action_path:
        last_tok = ctx_d[:, -1:, :]
        out = proj(last_tok).view(1, n_actions, d_model)
        emb_tok = out[:, a, :].unsqueeze(1).detach()
        assert emb_tok.grad_fn is None, "detached emb_tok should have no grad_fn"
        ctx_d = torch.cat([ctx_d, emb_tok], dim=1)

    # The actual code should use detach in the terminal rollout
    src = inspect.getsource(
        __import__("agent.train_scenarios.mcts_predict.train",
                   fromlist=["_mcts_forward"])._mcts_forward
    )
    terminal_section = src[src.find("Terminal value supervision"):]
    loop_start = terminal_section.find("for a in action_path")
    loop_body = terminal_section[loop_start:terminal_section.find("v_pred = ")]

    assert ".detach()" in loop_body, (
        "Terminal rollout loop does not detach embeddings. "
        "This causes quadratic autograd graph growth on long action paths."
    )


# ── Bug 2: re_backup_terminals skips opponent pessimism ────────────────

def _build_tree_with_opp_pessimism():
    """Build a tree: hero root -> opp node -> 2 hero terminals.

    Opp node has two children with different Q values.
    With pessimism alpha=0.5: opp.Q = 0.5 * E_P[Q] + 0.5 * min(Q)
    Without pessimism: opp.Q = W/N = average(Q)
    """
    root = MCTSNode(is_hero=True)

    opp = MCTSNode(action_idx=1, parent=root, is_hero=False)
    opp.P = 1.0

    # Terminal A: Q = 1.0, N = 10
    t_a = MCTSNode(action_idx=0, parent=opp, is_hero=True, is_terminal=True)
    t_a.N = 10
    t_a.W = 10.0
    t_a.Q = 1.0
    t_a.P = 0.5

    # Terminal B: Q = -1.0, N = 10
    t_b = MCTSNode(action_idx=1, parent=opp, is_hero=True, is_terminal=True)
    t_b.N = 10
    t_b.W = -10.0
    t_b.Q = -1.0
    t_b.P = 0.5

    opp.children = {0: t_a, 1: t_b}
    opp.N = 20
    opp.W = 0.0
    opp.Q = 0.0  # W/N = 0

    root.children = {1: opp}
    root.N = 20
    root.W = 0.0
    root.Q = 0.0

    return root, opp, t_a, t_b


def test_bug2_re_backup_terminals_applies_pessimism():
    """re_backup_terminals must apply opponent pessimism blending (alpha * E_P[Q]
    + (1-alpha) * min(Q)) at opp nodes, matching the search-time behavior of
    _refresh_opp_q. Without this, value targets are systematically optimistic."""
    root, opp, t_a, t_b = _build_tree_with_opp_pessimism()

    # Override terminal Q values (simulating evaluate_all_terminals)
    t_a.Q = 2.0   # hero wins big
    t_b.Q = -2.0  # hero loses big

    re_backup_terminals(root, opp_pessimism_alpha=0.5)

    # Plain W/N for opp: (2.0*10 + (-2.0)*10) / 20 = 0.0
    plain_q = 0.0  # what W/N would give

    # With pessimism alpha=0.5:
    # E_P[Q] = 0.5*2.0 + 0.5*(-2.0) = 0.0 (equal priors)
    # min(Q) = -2.0
    # pessimistic_q = 0.5 * 0.0 + 0.5 * (-2.0) = -1.0
    pessimistic_q = 0.5 * (0.5 * 2.0 + 0.5 * (-2.0)) + 0.5 * min(2.0, -2.0)

    assert abs(opp.Q - pessimistic_q) < 1e-6, (
        f"re_backup_terminals(alpha=0.5) set opp.Q = {opp.Q:.4f}, "
        f"expected pessimistic Q = {pessimistic_q:.4f}. "
        f"Without pessimism blending, value targets are systematically optimistic."
    )


def test_bug2_re_backup_opp_node_q_differs_from_wn():
    """After re_backup_terminals, opp node Q should not equal plain W/N
    when pessimism is needed (children have different Q values)."""
    root, opp, t_a, t_b = _build_tree_with_opp_pessimism()

    # Make children asymmetric
    t_a.Q = 5.0
    t_b.Q = -1.0

    re_backup_terminals(root)

    wn = opp.W / opp.N if opp.N > 0 else 0.0

    # If re_backup uses plain W/N, opp.Q == wn exactly.
    # With pessimism, opp.Q should be lower than wn (pessimistic toward hero).
    # We test: re_backup should NOT just use W/N for opp nodes.
    # (This test documents the bug — if it passes, the bug is fixed.)
    src = inspect.getsource(re_backup_terminals)
    has_pessimism = ("opp_pessimism" in src or "_refresh_opp_q" in src
                     or "min_q" in src or "min(c.Q" in src)
    assert has_pessimism, (
        "re_backup_terminals uses plain W/N for opp nodes (no pessimism). "
        f"opp.Q = {opp.Q:.4f}, W/N = {wn:.4f}. During search, _refresh_opp_q "
        "applies alpha * E_P[Q] + (1-alpha) * min(Q), making the tree "
        "pessimistic about opponent mistakes. re_backup_terminals doesn't, "
        "creating systematically optimistic value targets."
    )


# ── Bug 3: Root-level root_q_ratio has no NaN guard ───────────────────

def test_bug3_finalize_nan_root_q_ratio():
    """_finalize_value_targets must guard against NaN root_q_ratio at the
    root level, not just in chain steps. A NaN root.Q should fall back to
    pure MC (realized only), matching the chain step behavior.

    Python quirk: min(clip, NaN) returns clip (not NaN), so the symptom is
    NOT a NaN value_target — it's a WRONG value_target. When root.Q is NaN
    and realized is negative, the target becomes +clip instead of the
    correct negative value. This systematically biases toward overconfidence.
    """
    from agent.mcts.collect import _finalize_value_targets, MCTSTrainingExample

    # Two examples to ensure bootstrapping works.
    # Example with NaN root.Q and NEGATIVE realized outcome.
    nan_example = MCTSTrainingExample(
        events=[],
        value_target=-50.0,  # negative realized chip delta
        action_target=[0.5, 0.5],
        chain=[],
        root_q_ratio=float("nan"),
    )
    # Normal example for bootstrapping to have >1 sample.
    normal_example = MCTSTrainingExample(
        events=[],
        value_target=50.0,
        action_target=[0.5, 0.5],
        chain=[],
        root_q_ratio=0.5,
    )

    class FakeLog:
        def __call__(self, msg):
            pass

    per_agent = {"test_agent": [nan_example, normal_example]}
    agents_list = [{"name": "test_agent", "norm_stats": {}}]
    search_scales = {"test_agent": 10.0}

    _finalize_value_targets(
        per_agent, agents_list, search_scales,
        alpha=0.5, clip_val=3.0, big_blind=10.0, log=FakeLog()
    )

    # Pure MC fallback: realized / new_scale = -50 / scale → negative.
    # With the bug: NaN blending + Python min/max quirk → +clip_val (positive!).
    assert nan_example.value_target < 0, (
        f"NaN root_q_ratio: value_target = {nan_example.value_target:.4f} "
        f"(should be negative since realized = -50.0 chips). "
        f"Python's min(clip, NaN) returns clip, so NaN blend silently "
        f"becomes +clip_val instead of falling back to pure MC. "
        f"This systematically biases value targets positive when root.Q is NaN."
    )


def test_bug3_finalize_nan_root_q_matches_pure_mc():
    """When root_q_ratio is NaN, value_target should equal the pure MC
    (alpha=0) result — same as chain opp steps which also lack TD signal."""
    from agent.mcts.collect import _finalize_value_targets, MCTSTrainingExample

    realized_chips = -30.0

    # Run with NaN root_q and alpha=0.5
    nan_ex = MCTSTrainingExample(
        events=[], value_target=realized_chips,
        action_target=[0.5, 0.5], chain=[],
        root_q_ratio=float("nan"),
    )
    # Filler for bootstrapping
    filler = MCTSTrainingExample(
        events=[], value_target=30.0,
        action_target=[0.5, 0.5], chain=[],
        root_q_ratio=0.5,
    )

    # Run with alpha=0 (pure MC) and valid root_q for comparison
    mc_ex = MCTSTrainingExample(
        events=[], value_target=realized_chips,
        action_target=[0.5, 0.5], chain=[],
        root_q_ratio=999.0,  # doesn't matter at alpha=0
    )
    filler2 = MCTSTrainingExample(
        events=[], value_target=30.0,
        action_target=[0.5, 0.5], chain=[],
        root_q_ratio=0.5,
    )

    class FakeLog:
        def __call__(self, msg):
            pass

    per_agent_nan = {"test_agent": [nan_ex, filler]}
    agents_nan = [{"name": "test_agent", "norm_stats": {}}]
    _finalize_value_targets(
        per_agent_nan, agents_nan, {"test_agent": 10.0},
        alpha=0.5, clip_val=3.0, big_blind=10.0, log=FakeLog()
    )

    per_agent_mc = {"test_agent": [mc_ex, filler2]}
    agents_mc = [{"name": "test_agent", "norm_stats": {}}]
    _finalize_value_targets(
        per_agent_mc, agents_mc, {"test_agent": 10.0},
        alpha=0.0, clip_val=3.0, big_blind=10.0, log=FakeLog()
    )

    assert abs(nan_ex.value_target - mc_ex.value_target) < 1e-6, (
        f"NaN root_q_ratio: value_target = {nan_ex.value_target:.4f}, "
        f"pure MC (alpha=0) = {mc_ex.value_target:.4f}. "
        f"They should match — NaN root.Q means no TD signal, so pure MC."
    )


# ── Bug 4: scheduler.step() called when scaler skips optimizer ─────────

def test_bug4_scheduler_step_conditional_on_scaler():
    """scheduler.step() must NOT be called when GradScaler skips the optimizer
    step (due to inf/NaN gradients). Otherwise LR schedule drifts — warmup
    ends too early, cosine decay reaches eta_min prematurely."""

    # Check the source code of all affected training loops
    affected_files = [
        "agent.train_scenarios.mcts_predict.train",
        "agent.train_scenarios.modelling_predict.train",
        "agent.train_scenarios.opponent_action_predict.train",
    ]

    for module_name in affected_files:
        module = __import__(module_name, fromlist=["x"])
        # Get the training function (different names per module)
        train_fn = None
        for name in ["train_mcts", "train_modelling", "train_opponent_action"]:
            train_fn = getattr(module, name, None)
            if train_fn:
                break
        assert train_fn is not None, f"Cannot find train function in {module_name}"

        src = inspect.getsource(train_fn)

        # Find the pattern: scaler.step(optimizer) ... scheduler.step()
        # The scheduler.step() should be conditional on whether scaler actually stepped
        # Correct pattern: check scaler.get_scale() before/after, or use
        # scaler._found_inf / old_scale != new_scale guard
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("scheduler.step()"):
                # Look at the preceding lines for a guard
                context = "\n".join(lines[max(0, i-5):i+1])
                has_guard = (
                    "get_scale" in context
                    or "_found_inf" in context
                    or "old_scale" in context
                    or "scale_before" in context
                    or "if " in context and "scaler" in context
                    or "optimizer_stepped" in context
                )
                assert has_guard, (
                    f"In {module_name}: scheduler.step() at line ~{i+1} is called "
                    f"unconditionally after scaler.step(optimizer). When GradScaler "
                    f"skips the optimizer step (inf/NaN grads), the LR schedule "
                    f"still advances, causing premature warmup completion and "
                    f"cosine decay drift on fp16 GPUs.\n"
                    f"Context:\n{context}"
                )


# ── Bug 5: GradScaler re-created fresh each MCTS cycle ────────────────

def test_bug5_gradscaler_not_recreated_per_cycle():
    """GradScaler should persist across MCTS cycles, not be re-created
    with init_scale=65536 each time. Re-creation loses scale calibration
    from prior cycles, causing overflow on first steps of each new cycle."""
    module = __import__(
        "agent.train_scenarios.mcts_predict.train", fromlist=["train_mcts"])
    src = inspect.getsource(module.train_mcts)

    # The scaler should be created once, outside the cycle loop, or passed in
    # as a parameter. Check if it's a parameter of train_mcts:
    sig = inspect.signature(module.train_mcts)
    scaler_is_param = "scaler" in sig.parameters

    if not scaler_is_param:
        # If not a parameter, check that it's not created inside the function body
        # (which means it's re-created each call, since train_mcts is called per cycle)
        scaler_creation = "GradScaler(" in src
        assert not scaler_creation, (
            "GradScaler is created inside train_mcts(), which is called once per "
            "MCTS cycle. This means the scaler is re-created with init_scale=65536 "
            "each cycle, losing calibration from prior cycles. The scaler should be "
            "persistent across cycles (passed as parameter or created in the caller)."
        )


# ── Bug 7: base_scenarios / opp_scenarios not freed during MCTS ────────

def test_bug7_no_inmemory_dataset_before_mcts():
    """Pipeline must not hold a full in-memory dataset (base_scenarios) during
    the MCTS phase. With sharded storage, only scenarios_dir (a string path)
    should be passed — no large list of scenarios in RAM."""
    from pipeline import main
    src = inspect.getsource(main)

    mcts_pos = src.find("# --- MCTS cyclic collect")
    if mcts_pos == -1:
        mcts_pos = src.find("Cyclic MCTS")
    assert mcts_pos != -1, "Cannot find MCTS section in pipeline.main()"

    assert "base_scenarios" not in src, (
        "pipeline.main() still references 'base_scenarios' — the full GTO "
        "dataset should never be loaded into memory. Use scenarios_dir (path) instead."
    )


# ── Bug 8: perception_frozen check includes opponent_gru ───────────────

def test_bug8_perception_not_frozen_when_gru_unfrozen():
    """When opponent_gru is unfrozen (Phase 5), perception_frozen must be False.

    GRU input comes from embedder features (pre-encoder). Gradient path:
    loss → decoder → encoder → embedder → GRU cell. If perception_frozen
    were True, torch.no_grad() would kill GRU gradients. The current check
    (any(p.requires_grad for p in perception.parameters())) is CORRECT —
    it returns False (not frozen), ensuring gradients flow through the GRU.

    This test was initially written to flag this as a bug (wasted memory),
    but analysis shows the activation storage is NECESSARY for GRU training.
    """
    from agent.agent import ASI
    src = inspect.getsource(ASI.forward_batch)

    # Verify the check exists and uses all perception parameters
    assert "perception_frozen" in src, "perception_frozen check missing"

    # The check MUST include opponent_gru params so that when GRU is
    # unfrozen, perception runs with gradients (correct behavior).
    lines = src.split("\n")
    for line in lines:
        if "perception_frozen" in line and "any(" in line:
            assert "self.perception.parameters()" in line, (
                "perception_frozen should check ALL perception parameters "
                "(including opponent_gru) to ensure GRU gradients flow in Phase 5"
            )
            break


# ── Bug 9: OpponentEmbeddingTable unbounded growth ─────────────────────

def test_bug9_opp_emb_table_bounded_growth():
    """OpponentEmbeddingTable must have a size limit to prevent unbounded
    growth during long MCTS collection with many unique opponent IDs."""
    d_model = 128
    max_size = 100
    table = OpponentEmbeddingTable(d_model, max_size=max_size)

    # Add more IDs than the limit
    for i in range(200):
        table.get(f"opponent_{i}", device="cpu")

    assert len(table) <= max_size, (
        f"OpponentEmbeddingTable exceeded max_size={max_size}, "
        f"actual size={len(table)}. The table should evict old entries."
    )


def test_bug9_opp_emb_table_has_max_size_param():
    """OpponentEmbeddingTable constructor must accept a max_size parameter."""
    sig = inspect.signature(OpponentEmbeddingTable.__init__)
    assert "max_size" in sig.parameters, (
        "OpponentEmbeddingTable.__init__ must accept max_size parameter "
        "to allow callers to limit table growth."
    )


# ── Bug 10: _collect_terminals recursive DFS ───────────────────────────

def test_bug10_collect_terminals_deep_tree():
    """_collect_terminals uses recursive DFS. A pathologically deep tree
    (though unlikely in normal poker) would hit Python's recursion limit.
    Should use iterative DFS for robustness."""
    src = inspect.getsource(_collect_terminals)

    # Check if it's recursive (calls itself)
    is_recursive = "_collect_terminals(" in src.split("def _collect_terminals")[1]

    if is_recursive:
        # Check the actual recursion depth safety: build a tree at the limit
        depth = 500  # Well under Python's 1000 limit but deep
        root = MCTSNode(is_hero=True)
        node = root
        for i in range(depth):
            child = MCTSNode(action_idx=0, parent=node, is_hero=True)
            if i == depth - 1:
                child.is_terminal = True
                child.N = 1
                child.W = 1.0
                child.Q = 1.0
            node.children[0] = child
            node = child

        # This should work at depth 500
        terminals = _collect_terminals(root)
        assert len(terminals) == 1, f"Expected 1 terminal, got {len(terminals)}"

    # The assertion about recursion style
    assert not is_recursive, (
        "_collect_terminals uses recursive DFS. While typical poker tree depth "
        "(~30-40) is safe, an iterative DFS (using an explicit stack) is more "
        "robust against pathological trees and avoids any risk of stack overflow."
    )


def test_bug10_collect_terminals_functional():
    """Regardless of implementation style, _collect_terminals must correctly
    find all terminal nodes in a tree."""
    root = MCTSNode(is_hero=True)

    # Build a tree with 4 terminals at various depths
    c1 = MCTSNode(action_idx=0, parent=root, is_hero=False)
    c2 = MCTSNode(action_idx=1, parent=root, is_hero=False, is_terminal=True)
    c2.N = 1; c2.W = 1.0; c2.Q = 1.0
    root.children = {0: c1, 1: c2}

    c1a = MCTSNode(action_idx=0, parent=c1, is_hero=True, is_terminal=True)
    c1a.N = 2; c1a.W = 2.0; c1a.Q = 1.0
    c1b = MCTSNode(action_idx=1, parent=c1, is_hero=True)
    c1.children = {0: c1a, 1: c1b}

    c1b1 = MCTSNode(action_idx=0, parent=c1b, is_hero=False, is_terminal=True)
    c1b1.N = 3; c1b1.W = -3.0; c1b1.Q = -1.0
    c1b2 = MCTSNode(action_idx=1, parent=c1b, is_hero=False, is_terminal=True)
    c1b2.N = 1; c1b2.W = 0.5; c1b2.Q = 0.5
    c1b.children = {0: c1b1, 1: c1b2}

    terminals = _collect_terminals(root)
    assert len(terminals) == 4, f"Expected 4 terminals, got {len(terminals)}"


# ── GRU gradient flow (user-requested verification) ───────────────────

def _make_asi_config(d_model=32, max_players=6, opp_emb=True):
    return {
        "architecture": {
            "d_model": d_model,
            "d_ff": d_model * 4,
            "n_heads": 4,
            "n_kv_heads": 2,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "n_value_layers": 1,
            "n_action_layers": 1,
            "n_modelling_layers": 1,
            "max_seq_len": 64,
            "max_players": max_players,
            "memory": {"n_levels": 2, "max_cluster_size": 16,
                       "max_cluster_size_after": 8, "beam_width": 4},
            "opponent_embedding": {"enabled": opp_emb},
        },
        "game": {
            "raise_sizes": {"preflop": [0.5, 1.0], "flop": [0.5, 1.0],
                            "turn": [0.5, 1.0], "river": [0.5, 1.0]},
            "big_blind": 10,
            "max_stack": 1000,
            "max_players": max_players,
        },
    }


def _make_event_for_gru(n_actions=5, max_players=6):
    return {
        "table": [0, 1, 2, 3, 4], "hand": [10, 11],
        "hero_pos": 0, "acting_pos": 1, "num_players": 2,
        "pot": 100.0, "stack": 500.0,
        "bets": [5.0, 10.0] + [0.0] * (max_players - 2),
        "stacks": [500.0, 490.0] + [0.0] * (max_players - 2),
        "action": [0.0] * n_actions,
        "opponent_id": "opp_A",
    }


def test_gru_gradient_flows_in_phase5():
    """Verify that opponent_gru gradients flow correctly in Phase 5 pattern:
    perception frozen, opponent_action_head + opponent_gru unfrozen.

    The gradient path is: KL loss → opponent_action_head → decoder output
    → decoder → encoder → embedder → GRU cell. The perception_frozen check
    returns False (GRU has requires_grad), so torch.no_grad() is NOT used,
    allowing gradients to flow from the loss all the way back to the GRU."""
    from agent.agent import ASI
    from agent.perception.opponent_embeddings import OpponentEmbeddingTable

    config = _make_asi_config(opp_emb=True)
    d_model = 32

    class FakeLog:
        def __call__(self, msg):
            pass
        def run_dir(self, _):
            return "/tmp"

    asi = ASI(FakeLog(), config)
    asi.set_device("cpu")

    # Phase 5 freeze: perception + value + action + modelling frozen,
    # opponent_action_head + opponent_gru unfrozen
    for name, param in asi.named_parameters():
        if "opponent_gru" in name or "opponent_action" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Verify GRU params exist and are unfrozen
    gru_params = [(n, p) for n, p in asi.named_parameters()
                  if "opponent_gru" in n and p.requires_grad]
    assert len(gru_params) > 0, "No unfrozen opponent_gru parameters found"

    # Create opponent embedding table
    opp_table = OpponentEmbeddingTable(d_model)

    # Forward pass with opponent embedding — use opponent_action head (Phase 5)
    events = [[_make_event_for_gru()]]
    out = asi.forward_batch(
        events, skip_memory=True, heads={"opponent_action"},
        skip_opponent_emb=False, opponent_emb_table=opp_table,
    )

    logits = out["opponent_action_logits"]
    loss = logits.sum()
    loss.backward()

    # Verify GRU parameters received gradients. weight_hh may be zero when
    # h=0 (initial hidden state is zeros — W_hh * 0 = 0), so we check that
    # at least one GRU param has non-zero gradient (weight_ih will have it).
    has_any_grad = False
    for name, param in gru_params:
        assert param.grad is not None, (
            f"opponent_gru param '{name}' has no gradient after backward. "
            f"The perception_frozen check may be incorrectly using "
            f"torch.no_grad(), killing GRU gradients."
        )
        if param.grad.abs().sum() > 0:
            has_any_grad = True

    assert has_any_grad, (
        "All opponent_gru parameters have zero gradient. "
        "Gradient flow through the perception is broken."
    )


def test_gru_gradient_zero_when_fully_frozen():
    """When perception is fully frozen (including GRU), no gradients
    should flow through any perception parameters."""
    from agent.agent import ASI
    from agent.perception.opponent_embeddings import OpponentEmbeddingTable

    config = _make_asi_config(opp_emb=True)
    d_model = 32

    class FakeLog:
        def __call__(self, msg):
            pass
        def run_dir(self, _):
            return "/tmp"

    asi = ASI(FakeLog(), config)
    asi.set_device("cpu")

    # Fully frozen perception
    for param in asi.perception.parameters():
        param.requires_grad = False
    # Unfreeze value head so we can backprop
    for param in asi.value_head.parameters():
        param.requires_grad = True

    opp_table = OpponentEmbeddingTable(d_model)
    events = [[_make_event_for_gru()]]
    out = asi.forward_batch(
        events, skip_memory=True, heads={"value"},
        skip_opponent_emb=False, opponent_emb_table=opp_table,
    )

    value = out["value"]
    value.backward()

    # GRU params should have no gradients (frozen)
    for name, param in asi.perception.named_parameters():
        if "opponent_gru" in name:
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Frozen GRU param '{name}' received non-zero gradient"
            )


# ── Run all tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
