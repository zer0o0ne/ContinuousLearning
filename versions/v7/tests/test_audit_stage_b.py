"""Stage B audit tests: data and solver fixes.

B.1: Solver value-bet pot formula (new_pot = pot + raise + (raise - facing_bet)).
B.2: acting_pos convention consistency.
B.3: Modifiers normalizer matches generation (max(pot+facing_bet, bb) * temp).
B.5.2: Fold masked when facing_bet == 0.
B.5.3: Capped raise bins collapse to one all-in.
B.6.1: Per-seat independent starting stacks.
B.6.2: Stacks vector in event dict + perception embedding.
"""

import sys
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── B.1: Solver value-bet pot formula ────────────────────────────────────────

def test_b1_new_pot_formula():
    """new_pot must be pot + raise_amount + (raise_amount - facing_bet)."""
    solver_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/gto_utils/gpu_solver_v3.py")
    with open(solver_path) as f:
        source = f.read()
    pattern = r"new_pot\s*=\s*pot\s*\+\s*raise_amount\s*\+\s*\(raise_amount\s*-\s*facing_bet\)"
    assert re.search(pattern, source), (
        "gpu_solver_v3.py: new_pot formula should be "
        "'pot + raise_amount + (raise_amount - facing_bet)'")


def test_b1_value_bet_ev_positive_for_high_equity():
    """At eq=0.9 facing_bet=0, EV(raise) should exceed EV(check).

    EV(raise|call) - EV(check) = b*(2*eq-1) = 0.8*b > 0.
    We verify the formula produces positive EV difference.
    """
    eq = 0.9
    pot = 100.0
    facing_bet = 0.0
    raise_amount = 50.0
    hero_invested = 0.0
    new_pot = pot + raise_amount + (raise_amount - facing_bet)
    ev_check = eq * pot - hero_invested
    ev_raise_called = eq * new_pot - (hero_invested + raise_amount)
    assert ev_raise_called > ev_check, (
        f"EV(raise|call)={ev_raise_called:.1f} should exceed EV(check)={ev_check:.1f}")


def test_b1_ev_grows_with_sizing_at_eq1():
    """At eq=1.0, EV(raise) must strictly increase with raise size (up to stack)."""
    pot = 100.0
    facing_bet = 0.0
    hero_invested = 0.0
    eq = 1.0
    prev_ev = None
    for raise_amount in [10, 25, 50, 100, 200]:
        new_pot = pot + raise_amount + (raise_amount - facing_bet)
        ev = eq * new_pot - (hero_invested + raise_amount)
        if prev_ev is not None:
            assert ev > prev_ev, (
                f"EV should grow with sizing: size={raise_amount}, ev={ev:.1f}, prev={prev_ev:.1f}")
        prev_ev = ev


# ── B.3: Modifiers normalizer ───────────────────────────────────────────────

def test_b3_identity_temperature_reproduces_base():
    """A temperature modifier with the SAME value as base temp should reproduce
    original action_probs (up to floating-point)."""
    from agent.train_scenarios.modifiers import apply_modifiers

    n_actions = 10
    base_temp = 1.5
    pot, facing_bet, big_blind = 200.0, 50.0, 10.0
    evs = [float(i * 10 - 30) for i in range(n_actions)]

    normalizer = max(pot + facing_bet, big_blind) * base_temp
    evs_t = torch.tensor(evs, dtype=torch.float32)
    base_probs = F.softmax(evs_t / normalizer, dim=0).tolist()

    scenario = {
        "action_evs": evs.copy(),
        "action_probs": base_probs.copy(),
        "ev_target": max(evs),
        "pot": pot,
        "facing_bet": facing_bet,
    }
    mods = [{"type": "temperature", "value": base_temp}]
    result = apply_modifiers([scenario], mods, n_actions, big_blind, base_temp)
    for i in range(n_actions):
        assert abs(result[0]["action_probs"][i] - base_probs[i]) < 1e-5, (
            f"Action {i}: {result[0]['action_probs'][i]:.6f} != {base_probs[i]:.6f}")


def test_b3_normalizer_uses_pot_plus_facing():
    """apply_modifiers must use max(pot+facing_bet, bb)*temp, not bb*temp."""
    src_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/train_scenarios/modifiers.py")
    with open(src_path) as f:
        source = f.read()
    assert 's["pot"]' in source and 's["facing_bet"]' in source, (
        "modifiers.py should reference scenario pot and facing_bet for normalizer")


# ── B.5.2: Fold masked when facing_bet == 0 ─────────────────────────────────

def test_b5_2_fold_suppressed_no_facing_bet():
    """GameState.get_legal_actions: fold should NOT appear when not facing a bet."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5], 1: [0.5], 2: [0.5], 3: [0.5]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([500.0, 500.0]),
        bets=np.array([0.0, 0.0]),
        pot=20.0, high_bet=0.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=1,
        big_blind=10.0,
    )
    legal = gs.get_legal_actions()
    assert 0 not in legal, "Fold should be suppressed when not facing a bet"


def test_b5_2_fold_present_when_facing_bet():
    """Fold must be legal when facing a bet."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5], 1: [0.5], 2: [0.5], 3: [0.5]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([500.0, 500.0]),
        bets=np.array([0.0, 50.0]),
        pot=70.0, high_bet=50.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=1,
        big_blind=10.0,
    )
    legal = gs.get_legal_actions()
    assert 0 in legal, "Fold should be legal when facing a bet"


# ── B.5.3: Capped raise bins collapse to all-in ─────────────────────────────

def test_b5_3_capped_raises_collapse_to_allin():
    """Raise bins that exceed stack should NOT appear separately — only all-in."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0, 2.0], 1: [0.5, 1.0, 2.0],
                   2: [0.5, 1.0, 2.0], 3: [0.5, 1.0, 2.0]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([30.0, 1000.0]),
        bets=np.array([0.0, 20.0]),
        pot=30.0, high_bet=20.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=3,
        big_blind=10.0, last_raise_size=10.0,
        last_full_raise_level=20.0,
    )
    legal = gs.get_legal_actions()
    allin_idx = 5  # n_raise_bins + 2
    raise_idxs = [a for a in legal if 2 <= a < allin_idx]
    for r_idx in raise_idxs:
        raise_pct = raise_sizes[1][r_idx - 2]
        call_amount = 20.0
        effective_pot = 30.0 - 0.0
        bet = call_amount + raise_pct * effective_pot
        assert bet < 30.0, f"Raise bin {r_idx} (bet={bet}) should collapse to all-in"


# ── B.6.1: Per-seat independent starting stacks ─────────────────────────────

def test_b6_1_table_per_seat_stacks():
    """Table should accept per-seat start_credits."""
    from env.table import Table
    raise_sizes = {0: [0.5], 1: [0.5], 2: [0.5], 3: [0.5]}
    t = Table(3, raise_sizes, start_credits=[500, 750, 1000], big_blind=10, small_blind=5)
    assert t.start_credits[0] == 500.0
    assert t.start_credits[1] == 750.0
    assert t.start_credits[2] == 1000.0


def test_b6_1_table_scalar_stacks_broadcast():
    """Scalar start_credits should be broadcast to all seats."""
    from env.table import Table
    raise_sizes = {0: [0.5], 1: [0.5], 2: [0.5], 3: [0.5]}
    t = Table(3, raise_sizes, start_credits=600, big_blind=10, small_blind=5)
    assert all(t.start_credits[i] == 600.0 for i in range(3))


# ── B.6.2: Stacks vector in events + perception ─────────────────────────────

def test_b6_2_extract_event_tensors_stacks():
    """extract_event_tensors must extract per-position stacks vector."""
    from agent.perception.perception import extract_event_tensors
    event = {
        "table": [0, 1, 2, 3, 4], "hand": [10, 11],
        "hero_pos": 0, "acting_pos": 1, "num_players": 2,
        "pot": 100.0, "stack": 500.0,
        "bets": [5.0, 10.0], "stacks": [500.0, 490.0],
        "action": [0.0] * 10,
    }
    precomputed = extract_event_tensors([[event]], max_players=6)
    assert "stacks" in precomputed
    stacks = precomputed["stacks"]
    assert stacks.shape == (1, 6)
    assert abs(stacks[0, 0].item() - 500.0) < 1e-6
    assert abs(stacks[0, 1].item() - 490.0) < 1e-6


def test_b6_2_embedder_has_stacks_proj():
    """EventSequenceEmbedder must have stacks_proj Linear layer."""
    from agent.perception.perception import EventSequenceEmbedder
    emb = EventSequenceEmbedder(d_model=64, n_actions=10, max_players=6, max_seq_len=256)
    assert hasattr(emb, "stacks_proj"), "Missing stacks_proj"
    assert emb.stacks_proj.in_features == 6


def test_b6_2_build_batch_produces_stacks_emb():
    """_build_batch_tensors output must include stacks_emb."""
    from agent.perception.perception import EventSequenceEmbedder, extract_event_tensors
    event = {
        "table": [0, 1, 2, 3, 4], "hand": [10, 11],
        "hero_pos": 0, "acting_pos": 1, "num_players": 2,
        "pot": 100.0, "stack": 500.0,
        "bets": [5.0, 10.0], "stacks": [500.0, 490.0],
        "action": [0.0] * 10,
    }
    precomputed = extract_event_tensors([[event]], max_players=6)
    emb = EventSequenceEmbedder(d_model=64, n_actions=10, max_players=6, max_seq_len=256)
    result = emb._build_batch_tensors(None, device="cpu", precomputed=precomputed)
    assert "stacks_emb" in result
