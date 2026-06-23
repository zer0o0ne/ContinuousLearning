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

To pin the arithmetic deterministically (no MC noise) we drive the opponent
response into a pure-call regime via a hand-built state and monkeypatch the one
remaining MC call (`gpu_equity_v3`, used for equity-vs-callers) to a fixed
value. The fold/call/reraise split is exercised by real code; only the random
board/combo sampling is stubbed.

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


def _build_pure_call_state(eq_vs_callers, raw_equity):
    """State that classifies every opp combo as a pure call (no fold/reraise).

    opp_eq_cpu = 0.6 sits above the (raise-size-dependent) fold threshold, which
    caps at 0.5**0.85 ≈ 0.554, and below the static reraise_threshold (0.99), so
    p_call == 1 for any raise_frac. eqr disabled → eff equities are raw values.
    """
    dummy_combos = torch.zeros((N_COMBOS, 2), dtype=torch.long)
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
        "opponent_positions": None,  # skip postflop position tilt
        "n_iters": 100,
        "device": "cpu",
        "primary_idx": 0,
        "opp_eq_cpu": torch.full((N_COMBOS,), 0.6),
        "primary_combos": dummy_combos,
        "threshold_smoothing": None,  # legacy hard-mask path
        "dynamic_reraise": False,
        "polarized_reraise": None,
        # placeholders consumed only by external readers
        "reraise_mask": None,
        "p_reraise_per_combo": None,
        "eq_vs_reraisers": None,
    }


def _raise_ev(eq, pot, raise_frac, stack=1e9):
    """raise_ev from a pure-call state with eq_vs_callers == eq, facing_bet=0."""
    state = _build_pure_call_state(eq_vs_callers=eq, raw_equity=eq)
    orig = gpu_solver_v3.gpu_equity_v3
    gpu_solver_v3.gpu_equity_v3 = lambda *a, **k: eq
    try:
        fold_ev, call_ev, raise_ev, best_ev = gpu_solver_v3._compute_ev_v3_from_state(
            state, pot=pot, facing_bet=0.0, stack=stack, hero_invested=0.0,
            raise_frac=raise_frac,
        )
    finally:
        gpu_solver_v3.gpu_equity_v3 = orig
    return fold_ev, call_ev, raise_ev


def test_value_bet_beats_check_high_equity():
    """eq=0.9, facing_bet=0 → EV(bet) > EV(check) (a value bet must pay)."""
    pot = 100.0
    eq = 0.9
    _, call_ev, raise_ev = _raise_ev(eq, pot, raise_frac=1.0)
    # check == calling 0 chips: eq*pot
    assert abs(call_ev - eq * pot) < TOL, f"call_ev={call_ev} expected {eq*pot}"
    assert raise_ev > call_ev + TOL, (
        f"value bet must beat check: raise_ev={raise_ev} call_ev={call_ev}"
    )
    # Closed form: raise_ev = pot*eq + raise_amount*(2*eq-1), raise_amount=raise_frac*pot=100
    expected = pot * eq + 100.0 * (2 * eq - 1)
    assert abs(raise_ev - expected) < TOL, f"raise_ev={raise_ev} expected {expected}"
    print(f"test_value_bet_beats_check_high_equity: OK (call={call_ev:.3f}, raise={raise_ev:.3f})")


def test_value_bet_monotone_in_sizing():
    """eq=1.0 → EV(bet) strictly increases with sizing (up to stack)."""
    pot = 100.0
    eq = 1.0
    evs = []
    for frac in (0.5, 1.0, 2.0, 4.0):
        _, _, raise_ev = _raise_ev(eq, pot, raise_frac=frac)
        evs.append(raise_ev)
        # closed form at eq=1: pot + raise_amount
        expected = pot + frac * pot
        assert abs(raise_ev - expected) < TOL, f"frac={frac}: raise_ev={raise_ev} expected {expected}"
    for a, b in zip(evs, evs[1:]):
        assert b > a + TOL, f"raise_ev must strictly increase with sizing: {evs}"
    print(f"test_value_bet_monotone_in_sizing: OK (evs={[round(e,2) for e in evs]})")


def test_value_bet_monotone_partial_equity():
    """eq=0.7 (>0.5) → EV(bet) still strictly increases with sizing."""
    pot = 100.0
    eq = 0.7
    evs = [ _raise_ev(eq, pot, raise_frac=f)[2] for f in (0.5, 1.0, 2.0) ]
    for a, b in zip(evs, evs[1:]):
        assert b > a + TOL, f"eq>0.5 bet EV must increase with sizing: {evs}"
    print(f"test_value_bet_monotone_partial_equity: OK (evs={[round(e,2) for e in evs]})")


if __name__ == "__main__":
    test_value_bet_beats_check_high_equity()
    test_value_bet_monotone_in_sizing()
    test_value_bet_monotone_partial_equity()
    print("\nALL B.1 VALUE-BET TESTS PASSED")
