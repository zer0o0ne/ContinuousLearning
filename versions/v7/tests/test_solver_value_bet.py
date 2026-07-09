"""Value-bet EV tests for the v3 GPU solver (Audit B.1).

Verifies the fix to `new_pot` in `_compute_ev_v3_from_state`:

    new_pot = pot + raise_amount + (raise_amount - facing_bet)

`pot` already contains the opponent's facing_bet (a called win pays
`pot - hero_invested`). The old `pot + facing_bet + raise_amount` double-counted
facing_bet and dropped the opponent's call of the raise, so a value bet could
never beat a check: EV(raise|call) - EV(check) collapsed to -(1-eq)*b < 0.
Fixed, the showdown EV reduces to the textbook identity

    raise_ev(facing_bet=0) = pot*eq + raise_amount*(2*eq - 1)

so EV(raise) - EV(check) = raise_amount*(2*eq - 1) > 0 for eq > 0.5, and the
value bet's EV strictly increases with sizing when eq > 0.5.

Equity is pinned deterministically via the analytical per-combo path (E.3.1):
eq_vs_callers = 1 - opp_eq_cpu. We set opp_eq_cpu so that opponents sit in the
pure-call zone (above fold_threshold, below reraise_threshold) and hero's
effective equity > 0.5.

Run (from versions/v6):
    python -m tests.test_solver_value_bet
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

import torch
import gpu_solver_v3


TOL = 1e-4
N_COMBOS = 10


def _build_pure_call_state(opp_eq):
    """State where all opp combos have the same equity → pure call.

    opp_eq must be above the fold_threshold for the raise_frac used
    (fold_th for frac=1.0 is ~0.39) and below reraise_threshold (0.99).
    E.3.1 analytical equity: eq_vs_callers = 1 - opp_eq.
    """
    dummy_combos = torch.zeros((N_COMBOS, 2), dtype=torch.long)
    raw_equity = 1.0 - opp_eq
    return {
        "hero_cards": torch.tensor([0, 1], dtype=torch.long),
        "board_cards": torch.tensor([], dtype=torch.long),
        "opp_combos": [dummy_combos],
        "combo_weights": None,
        "eqr_raw": 1.0,
        "eqr_enabled": False,
        "raw_equity": raw_equity,
        "reraise_threshold": 0.99,
        "street": 3,            # river → street_discount == 1.0
        "hero_position": 0,
        "opponent_positions": None,
        "n_iters": 100,
        "device": "cpu",
        "primary_idx": 0,
        "opp_eq_cpu": torch.full((N_COMBOS,), opp_eq),
        "opp_eq_by_opp": [torch.full((N_COMBOS,), opp_eq)],
        "primary_combos": dummy_combos,
        "threshold_smoothing": None,
        "dynamic_reraise": False,
        "polarized_reraise": None,
        "reraise_mask": None,
        "p_reraise_per_combo": None,
        "eq_vs_reraisers": None,
    }


def _raise_ev(opp_eq, pot, raise_frac, stack=1e9):
    """raise_ev from a pure-call state, facing_bet=0."""
    state = _build_pure_call_state(opp_eq)
    fold_ev, call_ev, raise_ev, best_ev = gpu_solver_v3._compute_ev_v3_from_state(
        state, pot=pot, facing_bet=0.0, stack=stack, hero_invested=0.0,
        raise_frac=raise_frac,
    )
    return fold_ev, call_ev, raise_ev


def test_value_bet_beats_check_high_equity():
    """eq=0.6 (opp_eq=0.4), facing_bet=0 → EV(bet) > EV(check)."""
    pot = 100.0
    opp_eq = 0.4   # → eq_vs_callers = 0.6, above fold_th ~0.39 → pure call
    eq = 1.0 - opp_eq
    _, call_ev, raise_ev = _raise_ev(opp_eq, pot, raise_frac=1.0)
    assert abs(call_ev - eq * pot) < TOL, f"call_ev={call_ev} expected {eq*pot}"
    assert raise_ev > call_ev + TOL, (
        f"value bet must beat check: raise_ev={raise_ev} call_ev={call_ev}"
    )
    # Closed form (pure call, facing_bet=0):
    #   raise_ev = pot*eq + raise_amount*(2*eq - 1)
    raise_amount = 1.0 * pot
    expected = pot * eq + raise_amount * (2 * eq - 1)
    assert abs(raise_ev - expected) < TOL, f"raise_ev={raise_ev} expected {expected}"
    print(f"test_value_bet_beats_check_high_equity: OK (call={call_ev:.3f}, raise={raise_ev:.3f})")


def test_value_bet_monotone_in_sizing():
    """eq=0.6 → EV(bet) strictly increases with sizing (pure-call zone)."""
    pot = 100.0
    opp_eq = 0.4
    eq = 1.0 - opp_eq
    evs = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        _, _, rev = _raise_ev(opp_eq, pot, raise_frac=frac)
        evs.append(rev)
        raise_amount = frac * pot
        expected = pot * eq + raise_amount * (2 * eq - 1)
        assert abs(rev - expected) < TOL, f"frac={frac}: raise_ev={rev} expected {expected}"
    for a, b in zip(evs, evs[1:]):
        assert b > a + TOL, f"raise_ev must strictly increase with sizing: {evs}"
    print(f"test_value_bet_monotone_in_sizing: OK (evs={[round(e,2) for e in evs]})")


def test_value_bet_monotone_partial_equity():
    """eq=0.55 (>0.5) → EV(bet) still strictly increases with sizing."""
    pot = 100.0
    opp_eq = 0.45
    evs = [_raise_ev(opp_eq, pot, raise_frac=f)[2] for f in (0.25, 0.5, 0.75)]
    for a, b in zip(evs, evs[1:]):
        assert b > a + TOL, f"eq>0.5 bet EV must increase with sizing: {evs}"
    print(f"test_value_bet_monotone_partial_equity: OK (evs={[round(e,2) for e in evs]})")


if __name__ == "__main__":
    test_value_bet_beats_check_high_equity()
    test_value_bet_monotone_in_sizing()
    test_value_bet_monotone_partial_equity()
    print("\nALL B.1 VALUE-BET TESTS PASSED")
