"""Solver / generation target-distortion tests (Audit B.5).

Covers the cleanly unit-testable and checklist-critical items:
  B.5.2  fold masked when checking is free (facing_bet == 0);
  B.5.3  raise bins that collapse to the all-in are masked (no duplicate mass);
  B.5.4  multiway raise EV uses Π p_fold (win pot only if ALL opponents fold;
         reduces exactly to the heads-up model for a single opponent);
  B.5.5  street discount only reduces a POSITIVE call EV (no improving losers);
  B.5.9  preflop equity-realization treats the BB (seat 1) as in position.

B.5.1 (sizing denominator), B.5.6 (3bet classification), B.5.7 (geometric
factor on positive EV) and B.5.8 (combo-weighted response probs) are exercised
by the real-generation runs in the other test modules; their correctness is
asserted by inspection of the one-line formula changes.

Run (from versions/v6):
    python -m tests.test_solver_targets
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

import random
import numpy as np
import torch

import gpu_solver_v3
from gpu_solver_v3 import _compute_ev_v3_from_state, _get_eqr, EQR_TABLE
from env.table import Table
from agent.mcts.game_state import GameState
from agent.train_scenarios.generation.generate import (
    _compute_all_action_evs, generate_scenario,
)

TOL = 1e-4


# --------------------------------------------------------------------------- #
# Shared solver-state builder (no MC: opp_eq supplied directly).
# --------------------------------------------------------------------------- #

def _state(raw_equity, opp_eq_by_opp, primary_idx=0, street=3, eqr_enabled=False):
    """opp_eq_by_opp: list of per-opponent opp-equity tensors (or None)."""
    K = 10
    n_opp = len(opp_eq_by_opp)
    dummy = torch.zeros((K, 2), dtype=torch.long)
    opp_eq_cpu = opp_eq_by_opp[primary_idx] if n_opp > 0 else None
    return {
        "hero_cards": torch.tensor([0, 1], dtype=torch.long),
        "board_cards": torch.tensor([], dtype=torch.long),
        "opp_combos": [dummy] * n_opp,
        "combo_weights": None,
        "eqr_raw": 1.0, "eqr_enabled": eqr_enabled, "raw_equity": raw_equity,
        "reraise_threshold": 0.99, "street": street, "hero_position": 0,
        "opponent_positions": None, "n_iters": 100, "device": "cpu",
        "primary_idx": primary_idx,
        "opp_eq_cpu": opp_eq_cpu,
        "opp_eq_by_opp": opp_eq_by_opp,
        "primary_combos": dummy if n_opp > 0 else None,
        "threshold_smoothing": None, "dynamic_reraise": False,
        "polarized_reraise": None,
        "reraise_mask": None, "p_reraise_per_combo": None, "eq_vs_reraisers": None,
    }


# --------------------------------------------------------------------------- #
# B.5.4 — multiway Π p_fold
# --------------------------------------------------------------------------- #

def _raise_ev(state, pot=100.0):
    _, _, raise_ev, _ = _compute_ev_v3_from_state(
        state, pot=pot, facing_bet=0.0, stack=1e9, hero_invested=0.0, raise_frac=1.0)
    return raise_ev


def test_multiway_fold_product():
    """Win pot uncontested only if ALL opponents fold; HU reduces exactly."""
    fold = lambda: torch.full((10,), 0.2)   # opp_eq below fold_threshold → folds
    call = lambda: torch.full((10,), 0.6)   # opp_eq above fold_threshold → calls
    # raw_equity=0.5 → showdown_ev = 0.5*(300-100)+0.5*(-100)=50; uncontested=100.

    # Heads-up: primary folds → win pot uncontested.
    hu = _raise_ev(_state(0.5, [fold()]))
    assert abs(hu - 100.0) < 1e-2, f"HU primary-fold should win pot=100, got {hu}"

    # 2-way, both fold → still uncontested (Π p_fold ≈ 1).
    both_fold = _raise_ev(_state(0.5, [fold(), fold()]))
    assert abs(both_fold - 100.0) < 1e-2, f"all fold → 100, got {both_fold}"

    # 2-way, primary folds but the OTHER calls → showdown (Π p_fold ≈ 0).
    other_calls = _raise_ev(_state(0.5, [fold(), call()]))
    assert abs(other_calls - 50.0) < 1e-2, (
        f"one caller → showdown_ev=50, got {other_calls}"
    )
    # The whole point: a single opponent folding no longer wins the whole pot.
    assert other_calls < both_fold - 1.0
    print(f"test_multiway_fold_product: OK (HU={hu:.1f}, all-fold={both_fold:.1f}, "
          f"one-caller={other_calls:.1f})")


# --------------------------------------------------------------------------- #
# B.5.5 — street discount sign-asymmetry
# --------------------------------------------------------------------------- #

def test_street_discount_only_helps_positive():
    # Negative call EV must NOT be improved by the discount.
    st = _state(0.1, [], street=0)  # no opponents → opp_eq_cpu None; only call_ev matters
    _, call_ev, _, _ = _compute_ev_v3_from_state(
        st, pot=100.0, facing_bet=50.0, stack=1000.0, hero_invested=0.0, raise_frac=1.0)
    # call_ev = 0.1*100 + 0.9*(-50) = -35; discount 0.92 would lift it to -32.2.
    assert abs(call_ev - (-35.0)) < TOL, f"negative call_ev must be undiscounted, got {call_ev}"

    # Positive call EV IS discounted (street 0 → 0.92).
    st2 = _state(0.9, [], street=0)
    _, call_ev2, _, _ = _compute_ev_v3_from_state(
        st2, pot=100.0, facing_bet=0.0, stack=1000.0, hero_invested=0.0, raise_frac=1.0)
    # call_ev = 0.9*100 = 90 → *0.92 = 82.8
    assert abs(call_ev2 - 82.8) < 1e-2, f"positive call_ev should be discounted, got {call_ev2}"
    print(f"test_street_discount_only_helps_positive: OK (neg={call_ev:.1f}, pos={call_ev2:.1f})")


# --------------------------------------------------------------------------- #
# B.5.9 — preflop EQR: BB (seat 1) is in position
# --------------------------------------------------------------------------- #

def test_eqr_preflop_bb_in_position():
    active = [1, 3, 5]
    assert _get_eqr(1, 0, 6, active) == EQR_TABLE[(True, 0)], "BB (seat 1) is IP preflop"
    assert _get_eqr(5, 0, 6, active) == EQR_TABLE[(False, 0)], "BTN is NOT IP preflop"
    assert _get_eqr(3, 0, 6, active) == EQR_TABLE[(False, 0)]
    # Postflop: highest active seat acts last → IP.
    assert _get_eqr(5, 1, 6, active) == EQR_TABLE[(True, 1)]
    assert _get_eqr(1, 1, 6, active) == EQR_TABLE[(False, 1)]
    print("test_eqr_preflop_bb_in_position: OK")


# --------------------------------------------------------------------------- #
# B.5.3 — capped raise bins masked (matches GameState inference mask)
# --------------------------------------------------------------------------- #

def _short_stack_table():
    rs = [[0.5, 1.0, 2.0]] * 4
    n_actions = len(rs[0]) + 3  # 6
    t = Table(num_players=2, raise_sizes=rs, start_credits=20, big_blind=10, small_blind=5)
    t.start_table()
    t.deck = np.arange(52)  # deterministic
    return t, n_actions


def test_capped_bins_masked():
    t, n_actions = _short_stack_table()
    pos = t.active_player  # SB acts first preflop in HU
    mask = GameState.from_table(t, pos).get_legal_action_mask(n_actions)
    # SB: facing_bet=5, effective_pot=10, stack=15. Raise bins:
    #   0.5 -> bet 10 (< 15, legal); 1.0 -> bet 15 (>= 15, capped); 2.0 -> 25 (capped).
    assert mask[1] is True, "call must be legal"
    assert mask[n_actions - 1] is True, "all-in must be legal"
    assert not all(mask[2:n_actions - 1]), "at least one raise bin must be masked (capped)"
    n_legal_raises = sum(1 for b in range(2, n_actions - 1) if mask[b])
    assert n_legal_raises < (n_actions - 3), "capped bins not collapsed"

    # Wiring: _compute_all_action_evs returns this exact mask in meta.
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    evs, meta = _compute_all_action_evs(
        t, pos, [], n_actions, solver_name="v3", device="cpu", mc_iters=60,
        eqr_enabled=True, combo_response_iters=4, reraise_threshold=0.72,
        weighted_sampling=False,
        threshold_smoothing={"enabled": True, "beta_fold": 0.07, "beta_reraise": 0.07},
    )
    assert meta["legal_mask"] == mask, "generation mask must equal the GameState mask"
    print(f"test_capped_bins_masked: OK (mask={mask}, legal_raises={n_legal_raises})")


# --------------------------------------------------------------------------- #
# B.5.2 — fold masked in the saved policy target when checking is free
# --------------------------------------------------------------------------- #

def test_fold_masked_when_free():
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    cfg = {
        "mc_iterations": 150, "big_blind": 10, "max_stack": 300, "max_players": 4,
        "gto_temperature": 0.2, "solver": "v3",
        "raise_sizes": {s: [0.5, 1.0, 2.0] for s in ("preflop", "flop", "turn", "river")},
        "eqr_enabled": True, "combo_response_iters": 8, "reraise_threshold": 0.72,
        "weighted_sampling": False,
        "threshold_smoothing": {"enabled": True, "beta_fold": 0.07, "beta_reraise": 0.07},
    }
    n_free = 0
    n_scen = 0
    for _ in range(12):
        res = generate_scenario(cfg, device="cpu")
        if not res:
            continue
        for s in res:
            if s.get("scenario_type") == "modelling":
                continue
            n_scen += 1
            probs = s["action_probs"]
            assert abs(sum(probs) - 1.0) < 1e-4, f"probs must sum to 1: {sum(probs)}"
            assert all(p >= -1e-9 for p in probs), "no negative probs"
            assert all(p == p for p in probs), "no NaN probs"
            if abs(s["facing_bet"]) < 1e-9:
                n_free += 1
                assert probs[0] < 1e-6, (
                    f"fold must have ~0 prob when checking is free, got {probs[0]}"
                )
    assert n_scen > 30, f"too few scenarios ({n_scen})"
    assert n_free > 3, f"no free-check decisions exercised ({n_free})"
    print(f"test_fold_masked_when_free: OK (scenarios={n_scen}, free-check decisions={n_free})")


if __name__ == "__main__":
    test_multiway_fold_product()
    test_street_discount_only_helps_positive()
    test_eqr_preflop_bb_in_position()
    test_capped_bins_masked()
    test_fold_masked_when_free()
    print("\nALL B.5 SOLVER-TARGET TESTS PASSED")
