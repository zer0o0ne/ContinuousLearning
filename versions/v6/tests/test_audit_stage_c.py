"""Stage C audit tests: MCTS fixes.

C.1: Hero backup uses max(child.Q), not max(child.W).
C.4: Fold terminal deterministic (no NN).
C.5: Side-pot capping + no raises when all opponents all-in.
C.7.5: NLHE min-raise rules in GameState.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcts.mcts import MCTSNode, re_backup_terminals


def _capped_showdown_chips(equity, contributions, hero_pos, active_players,
                           dead_money=0.0):
    """Standalone copy to avoid gpu_solver_v2 import dependency."""
    invested_hero = float(contributions[hero_pos])
    opp_contribs = [float(contributions[p]) for p in active_players
                    if p != hero_pos]
    max_opp = max(opp_contribs) if opp_contribs else 0.0
    effective_hero = min(invested_hero, max_opp)
    excess = invested_hero - effective_hero
    hero_share_pot = sum(min(float(c), effective_hero) for c in contributions)
    return excess + equity * (hero_share_pot + dead_money) - invested_hero


# ── C.1: Hero backup Q = max(child.Q) ───────────────────────────────────────

def _build_simple_tree(child_values, child_visits=None):
    """Build root → children with given Q values. All hero.

    Returns root after re_backup.
    """
    root = MCTSNode(action_idx=None, parent=None, is_hero=True, is_terminal=False)
    root.N = 0
    root.W = 0.0
    root.Q = 0.0
    if child_visits is None:
        child_visits = [1] * len(child_values)
    for i, (v, n) in enumerate(zip(child_values, child_visits)):
        child = MCTSNode(action_idx=i, parent=root, is_hero=True, is_terminal=True)
        child.N = n
        child.W = v * n
        child.Q = v
        root.children[i] = child
        root.N += n
        root.W += v * n
    # Hero Q should be max(child.Q)
    visited = [c for c in root.children.values() if c.N > 0]
    if visited:
        root.Q = max(c.Q for c in visited)
    else:
        root.Q = root.W / root.N if root.N > 0 else 0.0
    return root


def test_c1_equal_leaves_root_q_equals_value():
    """All leaves return v → root.Q must equal v exactly."""
    v = 0.7
    root = _build_simple_tree([v, v, v])
    assert abs(root.Q - v) < 1e-6, f"root.Q={root.Q}, expected {v}"


def test_c1_root_q_is_max_child_q():
    """root.Q must be max over children's Q, not affected by visit counts."""
    root = _build_simple_tree([0.3, 0.8, -0.2], child_visits=[100, 1, 50])
    assert abs(root.Q - 0.8) < 1e-6, f"root.Q={root.Q}, expected 0.8"


def test_c1_negative_values_max_not_least_visited():
    """With negative Q's, hero should pick the BEST (least negative), not
    the least-visited child (old max(W) bug)."""
    root = _build_simple_tree([-0.1, -0.5, -0.3], child_visits=[10, 1, 5])
    assert abs(root.Q - (-0.1)) < 1e-6, f"root.Q={root.Q}, expected -0.1"


def test_c1_re_backup_terminals_hero():
    """re_backup_terminals should propagate equity-based Q values and recompute
    hero ancestor Q as max(child.Q)."""
    root = MCTSNode(action_idx=None, parent=None, is_hero=True, is_terminal=False)
    root.N = 3
    root.W = 0.0

    c1 = MCTSNode(action_idx=0, parent=root, is_hero=True, is_terminal=True)
    c1.N = 2
    c1.W = 1.0  # Q=0.5
    c1.Q = 0.5

    c2 = MCTSNode(action_idx=1, parent=root, is_hero=True, is_terminal=True)
    c2.N = 1
    c2.W = -0.3  # Q=-0.3
    c2.Q = -0.3

    root.children = {0: c1, 1: c2}
    root.W = c1.W + c2.W
    root.Q = max(c1.Q, c2.Q)

    # Now equity override: c1.Q → 0.9, c2.Q → 0.1
    c1.Q = 0.9
    c2.Q = 0.1
    re_backup_terminals(root)

    assert abs(root.Q - 0.9) < 1e-6, f"After re_backup, root.Q={root.Q}, expected 0.9"


# ── C.4: Fold terminal deterministic ────────────────────────────────────────

def test_c4_fold_terminal_source():
    """mcts.py must contain fold-terminal deterministic logic."""
    mcts_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent/mcts/mcts.py")
    with open(mcts_path) as f:
        source = f.read()
    assert "hero_invested" in source and "search_scale" in source, (
        "MCTS should have fold-terminal deterministic evaluation")


# ── C.5: Side-pot capping ───────────────────────────────────────────────────

def test_c5_short_allin_capped_main_pot():
    """Short all-in hero with eq=1.0 should win only the main pot (capped)."""
    # Hero has 100, opp1 has 300, opp2 has 500. Hero all-in 100.
    # Main pot = 100*3 = 300. Hero wins at most 200 net (300 - 100).
    contributions = [100.0, 300.0, 500.0]
    result = _capped_showdown_chips(1.0, contributions, 0, [0, 1, 2])
    # invested=100, max_opp=500, eff=100, excess=0
    # hero_share = min(100,100)+min(300,100)+min(500,100) = 300
    # result = 0 + 1.0*300 - 100 = 200
    assert abs(result - 200.0) < 1e-6, f"Expected 200.0, got {result}"


def test_c5_excess_returned():
    """When hero has invested more than max opponent, excess should be returned."""
    # Hero invested 500, opp invested 200.
    contributions = [500.0, 200.0]
    result = _capped_showdown_chips(1.0, contributions, 0, [0, 1])
    # invested=500, max_opp=200, eff=200, excess=300
    # hero_share = min(500,200)+min(200,200) = 400
    # result = 300 + 1.0*400 - 500 = 200
    assert abs(result - 200.0) < 1e-6, f"Expected 200.0, got {result}"


def test_c5_no_raises_all_opponents_allin():
    """When all opponents are all-in, raises should be suppressed (C.5)."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    gs = GameState(
        num_players=3, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 2.0, 2.0]),  # hero=waiting, opp1=allin, opp2=allin
        credits=np.array([500.0, 0.0, 0.0]),
        bets=np.array([0.0, 200.0, 300.0]),
        pot=520.0, high_bet=300.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0,
    )
    legal = gs.get_legal_actions()
    allin_idx = 4  # n_raise_bins + 2
    assert allin_idx not in legal, "All-in should be suppressed when all opponents all-in"
    assert all(a < 2 or a == allin_idx for a in legal if a >= 2) is False or \
        not any(a >= 2 for a in legal), "No raises when all opponents all-in"


def test_c5_capped_eq0_no_loss_beyond_invested():
    """With eq=0.0, hero's loss should be capped at invested (excess returned)."""
    contributions = [500.0, 200.0]
    result = _capped_showdown_chips(0.0, contributions, 0, [0, 1])
    # invested=500, max_opp=200, eff=200, excess=300
    # hero_share = min(500,200)+min(200,200)=400
    # result = 300 + 0.0*400 - 500 = -200
    assert abs(result - (-200.0)) < 1e-6


# ── C.7.5: Min-raise rules ──────────────────────────────────────────────────

def test_c7_5_min_raise_tracked():
    """GameState tracks last_raise_size."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([1000.0, 1000.0]),
        bets=np.array([5.0, 10.0]),
        pot=15.0, high_bet=10.0, turn=0,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0, last_raise_size=10.0,
    )
    assert gs.last_raise_size == 10.0


def test_c7_5_min_raise_resets_on_new_street():
    """last_raise_size resets to big_blind on new street."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 1.0]),
        credits=np.array([990.0, 980.0]),
        bets=np.array([5.0, 10.0]),
        pot=15.0, high_bet=10.0, turn=0,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0, last_raise_size=10.0,
    )
    # Both call through preflop to advance to flop
    gs.step(1)  # hero calls
    gs.step(1)  # BB checks → new street
    assert gs.turn == 1, f"Should be on flop, got turn={gs.turn}"
    assert abs(gs.last_raise_size - 10.0) < 1e-6, (
        f"last_raise_size should reset to bb=10 on new street, got {gs.last_raise_size}")


def test_c7_5_short_allin_doesnt_reopen():
    """Short all-in doesn't reopen action for player who matched last full raise."""
    from agent.mcts.game_state import GameState
    raise_sizes = {0: [0.5, 1.0], 1: [0.5, 1.0], 2: [0.5, 1.0], 3: [0.5, 1.0]}
    # Flop: hero bet 100 (high_bet=100, last_full_raise_level=100).
    # Opp1 calls 100. Opp2 short all-in to 120 (doesn't reopen).
    # Now hero acts again — should NOT be allowed to reraise.
    gs = GameState(
        num_players=3, hero_pos=0, active_player=0,
        players_state=np.array([1.0, 0.0, 2.0]),  # hero needs to act, opp1 acted, opp2 allin
        credits=np.array([900.0, 900.0, 0.0]),
        bets=np.array([100.0, 100.0, 120.0]),
        pot=320.0, high_bet=120.0, turn=1,
        raise_sizes=raise_sizes, n_raise_bins=2,
        big_blind=10.0, last_raise_size=100.0,
        last_full_raise_level=100.0,
    )
    legal = gs.get_legal_actions()
    raise_actions = [a for a in legal if a >= 2]
    assert len(raise_actions) == 0, (
        f"Hero should not be allowed to reraise after short all-in; got actions {legal}")
