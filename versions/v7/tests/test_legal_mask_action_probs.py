"""Legal-mask propagation into modifier-recomputed action_probs.

Data generation masks illegal/dominated actions before the tempered policy
softmax (generate.py:997-999): fold is dropped when checking is free, and
capped raise bins that collapse into the all-in are dropped. The saved
scenario dict now persists that mask as `legal_mask` (generate.py), and the
training-time modifier recomputes in modifiers.apply_modifiers and
sharded.apply_modifier_single must apply the SAME mask — otherwise every
modified agent (all have at least a temperature modifier) trains on targets
where fold regains probability when checking is free and capped raise bins
duplicate the all-in mass.

Old-format scenarios without `legal_mask` must still process (unmasked) with
a single warning per run.

Run (from versions/v6):
    python -m pytest tests/test_legal_mask_action_probs.py -v
"""

import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import agent.train_scenarios.modifiers as modifiers_mod
from agent.train_scenarios.modifiers import apply_modifiers
from agent.train_scenarios.sharded import apply_modifier_single

N_ACTIONS = 6  # fold, call, 3 raise bins, all-in
BIG_BLIND = 10.0
TEMP = 0.2
TOL = 1e-6


def _make_scenario(action_evs, pot, facing_bet, legal_mask, temperature=TEMP):
    """Build a scenario dict exactly as generate.py saves it (masked softmax)."""
    evs_t = torch.tensor(action_evs, dtype=torch.float32)
    normalizer = max(pot + facing_bet, BIG_BLIND) * temperature
    mask_t = torch.tensor(legal_mask, dtype=torch.bool)
    masked_evs = evs_t.masked_fill(~mask_t, float("-inf"))
    base_probs = F.softmax(masked_evs / normalizer, dim=0).tolist()
    return {
        "events": [{"hero_pos": 3}, {"hero_pos": 3}],
        "ev_target": float(evs_t.max().item()),
        "action_probs": base_probs,
        "action_evs": list(action_evs),
        "legal_mask": list(legal_mask),
        "equity": 0.5,
        "pot": float(pot),
        "facing_bet": float(facing_bet),
        "stack": 300.0,
        "hero_invested": 10.0,
        "num_players": 4,
        "n_events": 2,
    }, base_probs


def _expected_probs(action_evs, pot, facing_bet, legal_mask, temp):
    evs_t = torch.tensor(action_evs, dtype=torch.float32)
    normalizer = max(pot + facing_bet, BIG_BLIND) * temp
    if legal_mask is not None:
        mask_t = torch.tensor(legal_mask, dtype=torch.bool)
        evs_t = evs_t.masked_fill(~mask_t, float("-inf"))
    return F.softmax(evs_t / normalizer, dim=0).tolist()


def _reset_warn_flag():
    modifiers_mod._LEGAL_MASK_WARNED = False


# ---------------------------------------------------------------------------
# Check-is-free spot: fold illegal
# ---------------------------------------------------------------------------

def test_check_free_fold_stays_zero_with_temperature_modifier():
    """facing_bet == 0 → fold masked; temperature modifier must keep fold at 0."""
    _reset_warn_flag()
    action_evs = [0.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    legal_mask = [False, True, True, True, True, True]
    scenario, _ = _make_scenario(action_evs, pot=100.0, facing_bet=0.0,
                                 legal_mask=legal_mask)

    new_temp = 0.5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = apply_modifiers(
            [scenario], [{"type": "temperature", "value": new_temp}],
            n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
        )
    assert len(caught) == 0, "masked scenario must not trigger the legacy warning"

    probs = out[0]["action_probs"]
    assert probs[0] == 0.0, f"fold regained probability when checking is free: {probs}"
    assert abs(sum(probs) - 1.0) < TOL
    legal_sum = sum(p for p, m in zip(probs, legal_mask) if m)
    assert abs(legal_sum - 1.0) < TOL, "probability mass must sit on legal actions only"

    expected = _expected_probs(action_evs, 100.0, 0.0, legal_mask, new_temp)
    for a, b in zip(probs, expected):
        assert abs(a - b) < TOL, f"{probs} vs {expected}"

    # Sanity (non-vacuous): the unmasked (buggy) recompute would give fold mass.
    buggy = _expected_probs(action_evs, 100.0, 0.0, None, new_temp)
    assert buggy[0] > 1e-4, "test is vacuous — fold had ~no mass even unmasked"
    print("test_check_free_fold_stays_zero_with_temperature_modifier: OK")


# ---------------------------------------------------------------------------
# Capped-raise spot: dominated raise bins collapse into all-in
# ---------------------------------------------------------------------------

def test_capped_raise_bins_do_not_duplicate_allin_mass():
    """Raise bins 3,4 capped to all-in (masked); bias modifier must not revive them."""
    _reset_warn_flag()
    # Capped bins carry the all-in EV (as generation computes them).
    allin_ev = 12.0
    action_evs = [-10.0, 3.0, 8.0, allin_ev, allin_ev, allin_ev]
    legal_mask = [True, True, True, False, False, True]
    scenario, _ = _make_scenario(action_evs, pot=80.0, facing_bet=40.0,
                                 legal_mask=legal_mask)

    mods = [{"type": "temperature", "value": TEMP},
            {"type": "action_bias", "actions": "allin", "factor": 0.5}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = apply_modifiers([scenario], mods, n_actions=N_ACTIONS,
                              big_blind=BIG_BLIND, temperature=TEMP)
    assert len(caught) == 0

    probs = out[0]["action_probs"]
    assert probs[3] == 0.0 and probs[4] == 0.0, (
        f"capped raise bins regained probability: {probs}")
    assert abs(sum(probs) - 1.0) < TOL

    # Expected: bias applied to all-in only, then masked softmax.
    evs = list(action_evs)
    evs[5] = evs[5] + abs(evs[5]) * 0.5
    expected = _expected_probs(evs, 80.0, 40.0, legal_mask, TEMP)
    for a, b in zip(probs, expected):
        assert abs(a - b) < TOL, f"{probs} vs {expected}"

    # Sanity (non-vacuous): unmasked recompute would triple the all-in-EV mass
    # across bins 3, 4, 5.
    buggy = _expected_probs(evs, 80.0, 40.0, None, TEMP)
    assert buggy[3] > 1e-4 and buggy[4] > 1e-4, "test is vacuous — capped bins had ~no mass"
    print("test_capped_raise_bins_do_not_duplicate_allin_mass: OK")


def test_identity_temperature_reproduces_masked_base_probs():
    """Temperature modifier == base temp must reproduce generation's saved probs."""
    _reset_warn_flag()
    action_evs = [-5.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    legal_mask = [True, True, True, True, False, True]
    scenario, base_probs = _make_scenario(action_evs, pot=100.0, facing_bet=20.0,
                                          legal_mask=legal_mask)

    out = apply_modifiers(
        [scenario], [{"type": "temperature", "value": TEMP}],
        n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
    )
    for a, b in zip(out[0]["action_probs"], base_probs):
        assert abs(a - b) < TOL, f"{out[0]['action_probs']} vs {base_probs}"
    print("test_identity_temperature_reproduces_masked_base_probs: OK")


# ---------------------------------------------------------------------------
# sharded.apply_modifier_single parity
# ---------------------------------------------------------------------------

def test_apply_modifier_single_matches_apply_modifiers():
    """Sharded per-item path must produce identical masked targets."""
    _reset_warn_flag()
    action_evs = [0.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    legal_mask = [False, True, True, False, False, True]
    scenario, _ = _make_scenario(action_evs, pot=60.0, facing_bet=0.0,
                                 legal_mask=legal_mask)

    mods = [{"type": "temperature", "value": 0.4},
            {"type": "action_bias", "actions": "raises", "factor": -0.3},
            {"type": "conditional_bias", "actions": "call",
             "condition": "equity > 0.4", "factor": 0.2}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = apply_modifiers([scenario], mods, n_actions=N_ACTIONS,
                                big_blind=BIG_BLIND, temperature=TEMP)[0]
        single = apply_modifier_single(scenario, mods, N_ACTIONS, BIG_BLIND, TEMP)
    assert len(caught) == 0

    assert single["action_probs"] == batch["action_probs"]
    assert single["action_evs"] == batch["action_evs"]
    assert single["ev_target"] == batch["ev_target"]
    assert single["action_probs"][0] == 0.0
    assert single["action_probs"][3] == 0.0 and single["action_probs"][4] == 0.0
    assert abs(sum(single["action_probs"]) - 1.0) < TOL
    print("test_apply_modifier_single_matches_apply_modifiers: OK")


# ---------------------------------------------------------------------------
# Backward compatibility: old dataset without legal_mask
# ---------------------------------------------------------------------------

def test_old_format_scenario_warns_once_and_processes_unmasked():
    """No legal_mask → unmasked recompute + exactly ONE warning per run."""
    _reset_warn_flag()
    action_evs = [-5.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    old_scenarios = []
    for _ in range(3):
        s, _ = _make_scenario(action_evs, pot=100.0, facing_bet=20.0,
                              legal_mask=[True] * N_ACTIONS)
        del s["legal_mask"]
        old_scenarios.append(s)

    new_temp = 0.5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = apply_modifiers(
            old_scenarios, [{"type": "temperature", "value": new_temp}],
            n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
        )
    mask_warnings = [w for w in caught if "legal_mask" in str(w.message)]
    assert len(mask_warnings) == 1, (
        f"expected exactly 1 warning for 3 old-format scenarios, got {len(mask_warnings)}")

    expected = _expected_probs(action_evs, 100.0, 20.0, None, new_temp)
    for s in out:
        for a, b in zip(s["action_probs"], expected):
            assert abs(a - b) < TOL
    print("test_old_format_scenario_warns_once_and_processes_unmasked: OK")


def test_old_format_single_warns_once_across_calls():
    """apply_modifier_single: one warning across many per-item calls."""
    _reset_warn_flag()
    action_evs = [-5.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    s, _ = _make_scenario(action_evs, pot=100.0, facing_bet=20.0,
                          legal_mask=[True] * N_ACTIONS)
    del s["legal_mask"]

    mods = [{"type": "temperature", "value": 0.5}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outs = [apply_modifier_single(s, mods, N_ACTIONS, BIG_BLIND, TEMP)
                for _ in range(4)]
    mask_warnings = [w for w in caught if "legal_mask" in str(w.message)]
    assert len(mask_warnings) == 1, (
        f"expected exactly 1 warning across 4 calls, got {len(mask_warnings)}")

    expected = _expected_probs(action_evs, 100.0, 20.0, None, 0.5)
    for o in outs:
        for a, b in zip(o["action_probs"], expected):
            assert abs(a - b) < TOL
    print("test_old_format_single_warns_once_across_calls: OK")


# ---------------------------------------------------------------------------
# Originals untouched (copy semantics preserved)
# ---------------------------------------------------------------------------

def test_original_scenarios_not_mutated():
    """apply_modifiers copy semantics preserved with the legal_mask recompute."""
    _reset_warn_flag()
    action_evs = [0.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    legal_mask = [False, True, True, True, True, True]
    scenario, base_probs = _make_scenario(action_evs, pot=100.0, facing_bet=0.0,
                                          legal_mask=legal_mask)
    before_evs = list(scenario["action_evs"])

    out = apply_modifiers(
        [scenario], [{"type": "action_bias", "actions": "allin", "factor": 1.0}],
        n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
    )
    assert scenario["action_evs"] == before_evs
    assert scenario["action_probs"] == base_probs
    assert scenario["legal_mask"] == legal_mask
    assert out[0]["action_evs"] != before_evs  # modifier actually applied
    print("test_original_scenarios_not_mutated: OK")


# ---------------------------------------------------------------------------
# E2E: generation persists legal_mask consistently with saved action_probs
# ---------------------------------------------------------------------------

def test_generation_persists_legal_mask():
    """generate_scenario saves a legal_mask consistent with its action_probs."""
    from agent.train_scenarios.generation.generate import generate_scenario
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    cfg = {
        "mc_iterations": 120, "big_blind": 10, "max_stack": 400, "max_players": 4,
        "gto_temperature": 0.2, "solver": "v3",
        "raise_sizes": {s: [0.5, 1.0, 2.0] for s in ("preflop", "flop", "turn", "river")},
        "eqr_enabled": True, "combo_response_iters": 6, "reraise_threshold": 0.72,
        "weighted_sampling": False,
        "threshold_smoothing": {"enabled": True, "beta_fold": 0.07, "beta_reraise": 0.07},
    }
    n_actions = len(cfg["raise_sizes"]["preflop"]) + 3
    n_scen = 0
    saw_check_free = False
    for _ in range(10):
        res = generate_scenario(cfg, device="cpu")
        if not res:
            continue
        for s in res:
            if s.get("scenario_type") == "modelling":
                continue
            n_scen += 1
            lm = s.get("legal_mask")
            assert lm is not None, "GTO scenario missing persisted legal_mask"
            assert len(lm) == n_actions
            assert all(isinstance(m, bool) for m in lm)
            # Saved action_probs were computed with this mask: illegal → 0.
            for p, m in zip(s["action_probs"], lm):
                if not m:
                    assert p == 0.0, f"illegal action has prob {p} (mask {lm})"
            # Fold must be masked exactly when checking is free.
            assert lm[0] == (s["facing_bet"] > 0), (
                f"fold legality {lm[0]} inconsistent with facing_bet {s['facing_bet']}")
            if s["facing_bet"] == 0:
                saw_check_free = True
    assert n_scen > 10, f"too few scenarios generated ({n_scen}) — seed drifted?"
    assert saw_check_free, "no check-is-free decision in fixed-seed stream"
    print(f"test_generation_persists_legal_mask: OK (scenarios={n_scen})")


if __name__ == "__main__":
    test_check_free_fold_stays_zero_with_temperature_modifier()
    test_capped_raise_bins_do_not_duplicate_allin_mass()
    test_identity_temperature_reproduces_masked_base_probs()
    test_apply_modifier_single_matches_apply_modifiers()
    test_old_format_scenario_warns_once_and_processes_unmasked()
    test_old_format_single_warns_once_across_calls()
    test_original_scenarios_not_mutated()
    test_generation_persists_legal_mask()
    print("\nALL LEGAL-MASK ACTION-PROBS TESTS PASSED")
