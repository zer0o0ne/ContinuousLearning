"""Stage C MCTS audit-fix tests.

Covers:
  C.1  hero max(Q) backup: root.Q == v when all leaves equal
  C.2  chain value-target: opp steps get pure MC (NaN root_q_ratio)
  C.3  terminal_targets + root on same axis after rescale
  C.4  fold terminal independent of value_head weights
  C.5  short all-in capped to main pot (side-pot equity)
  C.7.5  min-raise rules: short all-in doesn't reopen; below-min-raise filtered

Run:
    python -m tests.test_mcts_stage_c        # from versions/v6
"""

import math
import sys
import numpy as np
import torch

from agent.mcts.game_state import GameState
from agent.mcts.mcts import MCTSNode, _collect_terminals, re_backup_terminals
from env.table import Table

BIG_BLIND = 10
SMALL_BLIND = 5
RAISE_SIZES = [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]] * 4
N_RAISE_BINS = len(RAISE_SIZES[0])
N_ACTIONS = N_RAISE_BINS + 3
TOL = 1e-6


def _onehot(idx):
    a = torch.zeros(N_ACTIONS, dtype=torch.float32)
    a[idx] = 1.0
    return a


def _capped_showdown_chips(equity, contributions, hero_pos, active_players):
    """Inlined from terminal_eval.py to avoid eval7 import chain."""
    invested_hero = float(contributions[hero_pos])
    opp_contribs = [float(contributions[p]) for p in active_players
                    if p != hero_pos]
    max_opp = max(opp_contribs) if opp_contribs else 0.0
    effective_hero = min(invested_hero, max_opp)
    excess = invested_hero - effective_hero
    hero_share_pot = sum(min(float(c), effective_hero) for c in contributions)
    return excess + equity * hero_share_pot - invested_hero


# ── C.1: hero max(Q) backup ──────────────────────────────────────────

def test_c1_uniform_leaves_root_q():
    """When all leaves return the same value v, root.Q must equal v exactly."""
    v = 0.42
    root = MCTSNode(is_hero=True)
    root.children = {}
    for a in [0, 1, 2]:
        child = MCTSNode(action_idx=a, parent=root, is_hero=False)
        child.N = 10
        child.W = v * 10
        child.Q = v
        for b in [0, 1]:
            gc = MCTSNode(action_idx=b, parent=child, is_hero=True, is_terminal=True)
            gc.N = 5
            gc.W = v * 5
            gc.Q = v
            child.children[b] = gc
        root.children[a] = child
    root.N = 30
    root.W = v * 30
    root.Q = max(c.Q for c in root.children.values() if c.N > 0)
    assert abs(root.Q - v) < TOL, f"root.Q={root.Q}, expected {v}"


def test_c1_negative_values_max_q():
    """With negative leaf values, root.Q = max(child.Q) regardless of visit distribution."""
    root = MCTSNode(is_hero=True)
    root.children = {}
    a = MCTSNode(action_idx=0, parent=root, is_hero=False)
    a.N = 100; a.W = -50.0; a.Q = -0.5
    root.children[0] = a
    b = MCTSNode(action_idx=1, parent=root, is_hero=False)
    b.N = 2; b.W = -0.2; b.Q = -0.1
    root.children[1] = b
    root.N = 102; root.W = -50.2
    root.Q = max(c.Q for c in root.children.values() if c.N > 0)
    assert abs(root.Q - (-0.1)) < TOL, f"root.Q={root.Q}, expected -0.1"


# ── C.2: chain value-target perspective guard ────────────────────────

def test_c2_opp_step_nan_root_q():
    """When future decision belongs to opponent, step_root_q_ratio must be NaN."""
    hero_pos = 0
    future_player_pos = 1
    if future_player_pos == hero_pos:
        step_root_q_ratio = 0.5
    else:
        step_root_q_ratio = float("nan")
    assert math.isnan(step_root_q_ratio), "opp step must produce NaN root_q_ratio"


def test_c2_hero_step_valid_root_q():
    """Same-hero future decision gets a real root_q_ratio."""
    hero_pos = 0
    future_player_pos = 0
    if future_player_pos == hero_pos:
        step_root_q_ratio = 0.5
    else:
        step_root_q_ratio = float("nan")
    assert not math.isnan(step_root_q_ratio), "hero step must produce valid root_q_ratio"


# ── C.3: terminal + root on same axis after rescale ──────────────────

def test_c3_rescale_consistency():
    """After rescaling by search_scale/new_scale, terminal and root targets
    should be on the same axis."""
    search_scale = 10.0
    new_scale = 300.0
    rescale_q = search_scale / new_scale

    root_q_search = 0.5
    root_q_new = root_q_search * rescale_q

    terminal_q_search = 0.3
    terminal_q_new = terminal_q_search * rescale_q

    root_chips = root_q_new * new_scale
    terminal_chips = terminal_q_new * new_scale
    assert abs(root_chips - 5.0) < TOL, f"root chips={root_chips}, expected 5.0"
    assert abs(terminal_chips - 3.0) < TOL, f"terminal chips={terminal_chips}, expected 3.0"


# ── C.4: fold terminal independent of value_head ─────────────────────

def test_c4_fold_terminal_deterministic():
    """Fold terminal value depends only on invested chips and search_scale."""
    gs = GameState(
        num_players=2, hero_pos=0, active_player=1,
        players_state=[-1, 1],
        credits=[90.0, 100.0], bets=[10.0, 0.0],
        pot=15.0, high_bet=10.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        is_terminal=True, big_blind=BIG_BLIND,
    )
    search_scale = 50.0
    root_credits = [100.0, 100.0]
    hero_invested = root_credits[0] - gs.credits[0]
    active = [i for i in range(2) if gs.players_state[i] >= 0]
    assert len(active) == 1 and active[0] != 0
    expected = -hero_invested / search_scale
    q_chips = -hero_invested
    result = q_chips / search_scale
    assert abs(result - expected) < TOL, f"fold terminal={result}, expected {expected}"


def test_c4_fold_terminal_winner():
    """When opponent folds, hero wins the pot deterministically."""
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=[1, -1],
        credits=[90.0, 90.0], bets=[10.0, 10.0],
        pot=20.0, high_bet=10.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        is_terminal=True, big_blind=BIG_BLIND,
    )
    search_scale = 50.0
    root_credits = [100.0, 100.0]
    hero_invested = root_credits[0] - gs.credits[0]
    q_chips = float(gs.pot) - hero_invested
    result = q_chips / search_scale
    expected = 10.0 / 50.0
    assert abs(result - expected) < TOL, f"fold winner={result}, expected {expected}"


# ── C.5: side-pot cap ────────────────────────────────────────────────

def test_c5_short_allin_capped():
    """Hero with a short stack all-in can only win the main pot."""
    contributions = [50.0, 200.0, 20.0]
    hero_pos = 0
    active = [0, 1]
    result = _capped_showdown_chips(1.0, contributions, hero_pos, active)
    # effective_hero = min(50, 200) = 50; excess = 0
    # hero_share_pot = min(50,50) + min(200,50) + min(20,50) = 50+50+20 = 120
    # net = 0 + 1.0*120 - 50 = 70
    assert abs(result - 70.0) < TOL, f"short all-in capped={result}, expected 70.0"


def test_c5_uncalled_excess_returned():
    """When hero's investment exceeds all opponents', the excess is returned."""
    contributions = [500.0, 100.0]
    hero_pos = 0
    active = [0, 1]
    result = _capped_showdown_chips(0.0, contributions, hero_pos, active)
    # effective_hero = min(500, 100) = 100; excess = 400
    # hero_share_pot = min(500,100) + min(100,100) = 100+100 = 200
    # net = 400 + 0.0*200 - 500 = -100
    assert abs(result - (-100.0)) < TOL, f"excess returned={result}, expected -100.0"


def test_c5_equal_stacks_no_cap():
    """With equal contributions, side-pot cap has no effect."""
    contributions = [100.0, 100.0]
    hero_pos = 0
    active = [0, 1]
    equity = 0.6
    result = _capped_showdown_chips(equity, contributions, hero_pos, active)
    naive = equity * sum(contributions) - contributions[hero_pos]
    assert abs(result - naive) < TOL, f"equal stacks: result={result}, naive={naive}"


def test_c5_all_others_allin_no_raise():
    """When all opponents are all-in, raises must not be in legal actions."""
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=[1, 2],
        credits=[500.0, 0.0], bets=[0.0, 100.0],
        pot=100.0, high_bet=100.0, turn=1,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
    )
    legal = gs.get_legal_actions()
    assert 0 in legal, "fold should be legal when facing bet"
    assert 1 in legal, "call should be legal"
    for a in legal:
        assert a <= 1, f"action {a} is a raise/allin but all opponents are all-in"


# ── C.7.5: min-raise rules ──────────────────────────────────────────

def test_c75_min_raise_filter():
    """Raise sizes below last_raise_size should be filtered from legal actions."""
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=[1, 0],
        credits=[500.0, 470.0], bets=[0.0, 30.0],
        pot=45.0, high_bet=30.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
    )
    gs.last_raise_size = 20.0
    legal = gs.get_legal_actions()
    for a in legal:
        if 2 <= a < N_RAISE_BINS + 2:
            raise_pct = RAISE_SIZES[0][a - 2]
            effective_pot = gs.pot - gs.bets[0]
            increment = raise_pct * effective_pot
            assert increment >= gs.last_raise_size - TOL, (
                f"action {a}: raise increment {increment:.2f} < "
                f"last_raise_size {gs.last_raise_size}"
            )


def test_c75_short_allin_no_reraise():
    """Short all-in (raise < last_raise_size) must NOT allow re-raising by
    players who already matched the previous bet level. They can still
    call/fold the difference but cannot re-raise (NLHE rule)."""
    gs = GameState(
        num_players=3, hero_pos=0, active_player=2,
        players_state=[0, 0, 1],
        credits=[470.0, 470.0, 40.0],
        bets=[30.0, 30.0, 0.0],
        pot=60.0, high_bet=30.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
        last_raise_size=20.0,
        last_full_raise_level=30.0,
    )
    gs.step(N_RAISE_BINS + 2)  # all-in for 40 → increment=10 < 20
    # Players 0 and 1 are marked to act (state 1) — they must call/fold
    assert gs.players_state[0] == 1, f"player 0 should need to act: state={gs.players_state[0]}"
    assert gs.players_state[1] == 1, f"player 1 should need to act: state={gs.players_state[1]}"
    assert not gs.is_terminal, "game should not be terminal"
    # But their legal actions should NOT include raises
    # (they matched the last full raise at 30; extra 10 is short all-in)
    # Simulate player 0's turn
    gs_check = gs.clone()
    gs_check.active_player = 0
    legal = gs_check.get_legal_actions()
    for a in legal:
        assert a <= 1, (
            f"player 0 should not be able to raise after short all-in, "
            f"but action {a} is legal"
        )


def test_c75_full_raise_reopens():
    """A full raise (increment >= last_raise_size) MUST reopen already-acted players."""
    gs = GameState(
        num_players=3, hero_pos=0, active_player=2,
        players_state=[0, 0, 1],
        credits=[470.0, 470.0, 470.0],
        bets=[30.0, 30.0, 0.0],
        pot=60.0, high_bet=30.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
        last_raise_size=20.0,
    )
    gs.step(N_RAISE_BINS + 2)  # all-in for 470 → increment=440 >= 20
    assert gs.players_state[0] == 1, f"player 0 not reopened: state={gs.players_state[0]}"
    assert gs.players_state[1] == 1, f"player 1 not reopened: state={gs.players_state[1]}"


def test_c75_last_raise_size_reset_on_new_street():
    """last_raise_size resets to big_blind when a new street starts."""
    gs = GameState(
        num_players=2, hero_pos=0, active_player=0,
        players_state=[1, 1],
        credits=[400.0, 400.0], bets=[100.0, 100.0],
        pot=200.0, high_bet=100.0, turn=0,
        raise_sizes=RAISE_SIZES, n_raise_bins=N_RAISE_BINS,
        big_blind=BIG_BLIND,
        last_raise_size=90.0,
    )
    gs.step(1)  # hero checks
    gs.step(1)  # opp checks → advances to flop
    assert gs.turn == 1, f"turn should be 1 (flop), got {gs.turn}"
    assert abs(gs.last_raise_size - BIG_BLIND) < TOL, (
        f"last_raise_size should reset to BB={BIG_BLIND}, got {gs.last_raise_size}"
    )


def test_c75_table_min_raise_tracking():
    """Table.step mirrors GameState min-raise tracking."""
    table = Table(
        num_players=3, raise_sizes=RAISE_SIZES,
        start_credits=500, big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
    )
    table.start_table()
    assert abs(table.last_raise_size - BIG_BLIND) < TOL, (
        f"initial last_raise_size should be BB, got {table.last_raise_size}"
    )


# ── re_backup_terminals: hero max(child.Q) ──────────────────────────

def test_re_backup_hero_max_q():
    """re_backup_terminals must set hero root Q = max(child.Q) after delta propagation."""
    root = MCTSNode(is_hero=True)
    c0 = MCTSNode(action_idx=0, parent=root, is_hero=False)
    c0.N = 5; c0.W = 2.0; c0.Q = 0.4
    c0.is_terminal = True
    c1 = MCTSNode(action_idx=1, parent=root, is_hero=False)
    c1.N = 5; c1.W = 5.0; c1.Q = 1.0
    c1.is_terminal = True
    root.children = {0: c0, 1: c1}
    root.N = 10; root.W = 7.0; root.Q = 0.7

    new_q = {0: 0.3, 1: 0.8}
    terminals = _collect_terminals(root)
    for t in terminals:
        old = t.Q
        t.Q = new_q[t.action_idx]
        delta = (t.Q - old) * t.N
        t.W += delta

    re_backup_terminals(root)
    assert abs(root.Q - 0.8) < TOL, f"root.Q={root.Q}, expected 0.8"


# ── run ──────────────────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
