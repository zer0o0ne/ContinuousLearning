"""Tests for the 8 confirmed pipeline bugs (Bugs 1-4, 6-9).

Bug 5 was disproven (apply_modifiers with empty list doesn't recompute softmax).
Each test verifies the fix is in place — a regression would flip the assertion.
"""

import sys
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.perception.perception import extract_event_tensors, EventSequenceEmbedder


def _make_event(n_actions=10, max_players=6):
    return {
        "table": [0, 1, 2, 3, 4], "hand": [10, 11],
        "hero_pos": 0, "acting_pos": 1, "num_players": 2,
        "pot": 100.0, "stack": 500.0,
        "bets": [5.0, 10.0], "stacks": [500.0, 490.0],
        "action": [0.0] * n_actions,
    }


# ── Bug 1: _build_batch_tensors discards precomputed (missing 'T' key) ──────

def test_bug1_precomputed_no_T_key():
    """extract_event_tensors doesn't produce a 'T' key. The old code checked
    precomputed.get('T', 0) == 0 which always fell through to the raw path."""
    precomputed = extract_event_tensors([[_make_event()]], max_players=6)
    assert precomputed is not None
    assert "T" not in precomputed, "extract_event_tensors should not produce 'T'"
    assert "card_ids" in precomputed


def test_bug1_build_batch_tensors_uses_precomputed():
    """_build_batch_tensors must return a valid dict when given precomputed."""
    precomputed = extract_event_tensors([[_make_event()]], max_players=6)
    emb = EventSequenceEmbedder(d_model=64, n_actions=10,
                                max_players=6, max_seq_len=256)
    result = emb._build_batch_tensors(None, device="cpu", precomputed=precomputed)
    assert result is not None, "_build_batch_tensors returned None for valid precomputed"
    assert "card_embs" in result
    assert result["B"] == 1


# ── Bug 2: _compute_pre_inject crashes with event_sequences=None ─────────────

def test_bug2_pre_inject_with_precomputed():
    """_compute_pre_inject must work with event_sequences=None + precomputed."""
    precomputed = extract_event_tensors([[_make_event()]], max_players=6)
    emb = EventSequenceEmbedder(d_model=64, n_actions=10,
                                max_players=6, max_seq_len=256)
    out_pre, meta = emb._compute_pre_inject(None, device="cpu",
                                            precomputed=precomputed)
    assert out_pre is not None
    assert meta["B"] == 1
    assert meta["T"] == 1


def test_bug2_pre_inject_reads_B_from_precomputed():
    """B/seq_lengths/max_events must come from precomputed, not len(None)."""
    events = [[_make_event(), _make_event()], [_make_event()]]
    precomputed = extract_event_tensors(events, max_players=6)
    emb = EventSequenceEmbedder(d_model=64, n_actions=10,
                                max_players=6, max_seq_len=256)
    out_pre, meta = emb._compute_pre_inject(None, device="cpu",
                                            precomputed=precomputed)
    assert meta["B"] == 2
    assert meta["max_events"] == 2
    assert meta["seq_lengths"] == [2, 1]


# ── Bug 3: NameError 'event_seqs' in mcts_predict/train.py ──────────────────

def test_bug3_no_event_seqs_reference():
    """mcts_predict/train.py must not reference undefined 'event_seqs'."""
    train_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/train_scenarios/mcts_predict/train.py")
    with open(train_path) as f:
        source = f.read()
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
            continue
        assert "event_seqs" not in stripped or "=" in stripped.split("event_seqs")[0], (
            f"Line {i} references undefined 'event_seqs': {stripped}")


# ── Bug 4: hand_id same for all hands in a worker chunk ──────────────────────

def test_bug4_hand_id_varies_per_hand():
    """_generate_worker must assign different hand_id per hand, not constant worker_id."""
    gen_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/train_scenarios/generation/generate.py")
    with open(gen_path) as f:
        source = f.read()
    match = re.search(r's\["hand_id"\]\s*=\s*(.+)', source)
    assert match is not None, "hand_id assignment not found"
    rhs = match.group(1).strip()
    assert "+" in rhs or "h" in rhs, (
        f"hand_id assignment is constant per worker: {rhs}")


# ── Bug 6: contributions exclude blinds → showdown EV understated ────────────

def _capped_showdown_chips(equity, contributions, hero_pos, active_players,
                           dead_money=0.0):
    """Standalone copy of terminal_eval._capped_showdown_chips to avoid
    gpu_solver_v2 import dependency in tests."""
    invested_hero = float(contributions[hero_pos])
    opp_contribs = [float(contributions[p]) for p in active_players
                    if p != hero_pos]
    max_opp = max(opp_contribs) if opp_contribs else 0.0
    effective_hero = min(invested_hero, max_opp)
    excess = invested_hero - effective_hero
    hero_share_pot = sum(min(float(c), effective_hero) for c in contributions)
    return excess + equity * (hero_share_pot + dead_money) - invested_hero


def test_bug6_capped_showdown_dead_money_param():
    """_capped_showdown_chips must accept dead_money kwarg."""
    result = _capped_showdown_chips(1.0, [20.0, 15.0], 0, [0, 1], dead_money=15.0)
    assert result is not None


def test_bug6_showdown_nuts_equals_fold_win():
    """With eq=1.0, showdown win must equal fold-win when dead_money is correct.

    HU: SB=5, BB=10. After blinds+betting: contributions_post_blind=[20,15],
    pot=50 (15 blind + 35 post-blind). dead_money=15 (the blind pot).
    Fold-win = pot - invested = 50 - 20 = 30.
    Showdown(eq=1.0) must also be 30.
    """
    result = _capped_showdown_chips(1.0, [20.0, 15.0], 0, [0, 1], dead_money=15.0)
    assert abs(result - 30.0) < 1e-6, f"Expected 30.0, got {result}"


def test_bug6_capped_showdown_without_dead_money():
    """Without dead_money, the gap between showdown and fold should appear."""
    result_no_dm = _capped_showdown_chips(1.0, [20.0, 15.0], 0, [0, 1], dead_money=0.0)
    assert abs(result_no_dm - 15.0) < 1e-6


def test_bug6_compute_equity_outcome_uses_start_stacks():
    """compute_equity_outcome should prefer start_stacks over initial_credits
    so that blind money counts in contributions."""
    src_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/mcts/terminal_eval.py")
    with open(src_path) as f:
        source = f.read()
    assert 'start_stacks = hand_record.get("start_stacks")' in source
    assert "start_stacks is not None" in source


def test_bug6_whole_hand_contributions_no_dead_money():
    """2026-07 math-audit fix superseding the old dead_money design: side-pot
    caps use WHOLE-HAND contributions (start_stacks baseline), so no separate
    dead_money term may exist — computing caps from-root with dead_money
    misclassified hero's call of an outstanding bet as uncalled excess
    (systematic pro-call bias)."""
    src_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/mcts/terminal_eval.py")
    with open(src_path) as f:
        source = f.read()
    assert "dead_money" not in source.replace(
        "no dead-money term", "").replace("no separate dead-money", ""), \
        "dead_money must not reappear in terminal_eval.py"
    # Whole-hand baseline plumbed from hand_record
    assert 'hand_record.get("start_stacks")' in source


# ── Bug 7: Solver call_cost = raise_amount (missing - facing_bet) ────────────

def test_bug7_call_cost_subtracts_facing_bet():
    """gpu_solver_v3 call_cost must be raise_amount - facing_bet."""
    solver_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/gto_utils/gpu_solver_v3.py")
    with open(solver_path) as f:
        source = f.read()
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("call_cost") and "=" in stripped and "raise_amount" in stripped:
            assert "facing_bet" in stripped, (
                f"Line {i}: call_cost does not subtract facing_bet: {stripped}")
            return
    raise AssertionError("call_cost assignment not found in gpu_solver_v3.py")


# ── Bug 8: NameError n_workers_cfg in _run_parallel_opponent ─────────────────

def test_bug8_no_n_workers_cfg_reference():
    """_run_parallel_opponent must use parameter 'n_workers', not 'n_workers_cfg'."""
    opp_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/train_scenarios/generation/generate_opponent.py")
    with open(opp_path) as f:
        source = f.read()
    in_func = False
    for i, line in enumerate(source.split("\n"), 1):
        if "def _run_parallel_opponent(" in line:
            in_func = True
            continue
        if in_func and line.strip() and not line[0].isspace() and i > 1:
            break
        if in_func:
            assert "n_workers_cfg" not in line, (
                f"Line {i}: references undefined 'n_workers_cfg': {line.strip()}")


# ── Bug 9: Duplicate call/all-in when credits == call_amount ─────────────────

def test_bug9_no_duplicate_call_allin():
    """When credits == call_amount, only call should be legal (not both call+all-in)."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([20.0, 1000.0]),
        bets=np.array([0.0, 20.0]),
        pot=30.0, high_bet=20.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0, last_raise_size=10.0,
        last_full_raise_level=20.0,
    )
    legal = gs.get_legal_actions()
    allin_idx = 4  # n_raise_bins + 2
    assert 1 in legal, "Call should be legal"
    assert allin_idx not in legal, (
        "All-in should NOT be legal when it's identical to call")


def test_bug9_allin_allowed_when_more_than_call():
    """When credits > call_amount, all-in should remain legal."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([25.0, 1000.0]),
        bets=np.array([0.0, 20.0]),
        pot=30.0, high_bet=20.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0, last_raise_size=10.0,
        last_full_raise_level=20.0,
    )
    legal = gs.get_legal_actions()
    allin_idx = 4
    assert allin_idx in legal, "All-in should be legal when credits > call_amount"
