"""Modifier softmax-normalizer tests (Audit B.3).

apply_modifiers recomputes action_probs after biasing action_evs. It must use
the SAME per-scenario normalizer as data generation —
`max(pot + facing_bet, big_blind) * temperature` (generate.py:834/913) — not
`big_blind * temperature`. Otherwise an identity-temperature modifier reshapes
the base targets onto a different (much sharper) scale.

Run (from versions/v6):
    python -m tests.test_modifiers_normalizer
"""

import torch
import torch.nn.functional as F

from agent.train_scenarios.modifiers import apply_modifiers

N_ACTIONS = 6
BIG_BLIND = 10.0
TEMP = 0.2
TOL = 1e-6


def _make_scenario(action_evs, pot, facing_bet):
    evs_t = torch.tensor(action_evs, dtype=torch.float32)
    normalizer = max(pot + facing_bet, BIG_BLIND) * TEMP
    base_probs = F.softmax(evs_t / normalizer, dim=0).tolist()
    return {
        "action_evs": list(action_evs),
        "action_probs": base_probs,
        "ev_target": float(evs_t.max().item()),
        "pot": float(pot),
        "facing_bet": float(facing_bet),
        "equity": 0.5,
        "events": [{"hero_pos": 3}],
        "num_players": 4,
    }, base_probs


def test_identity_temperature_reproduces_base_probs():
    """A temperature modifier == base temp reproduces base action_probs (fp)."""
    action_evs = [-5.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    # pot + facing_bet (120) >> big_blind (10): the two normalizers diverge,
    # so this case distinguishes the fix (24) from the bug (2).
    scenario, base_probs = _make_scenario(action_evs, pot=100.0, facing_bet=20.0)

    out = apply_modifiers(
        [scenario], [{"type": "temperature", "value": TEMP}],
        n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
    )
    got = out[0]["action_probs"]
    assert len(got) == len(base_probs)
    for a, b in zip(got, base_probs):
        assert abs(a - b) < TOL, f"probs diverge: {got} vs {base_probs}"
    assert abs(out[0]["ev_target"] - max(action_evs)) < TOL

    # Sanity: the OLD normalizer (big_blind * temp) would NOT reproduce them.
    buggy = F.softmax(torch.tensor(action_evs) / (BIG_BLIND * TEMP), dim=0).tolist()
    assert any(abs(a - b) > 1e-3 for a, b in zip(buggy, base_probs)), (
        "test is vacuous — buggy normalizer happens to match"
    )
    print("test_identity_temperature_reproduces_base_probs: OK")


def test_normalizer_consistent_with_bias():
    """A bias modifier reshapes probs but on the correct per-scenario scale."""
    action_evs = [-5.0, 3.0, 10.0, 2.0, -1.0, 8.0]
    scenario, _ = _make_scenario(action_evs, pot=100.0, facing_bet=20.0)

    out = apply_modifiers(
        [scenario], [{"type": "action_bias", "actions": "raises", "factor": 0.5}],
        n_actions=N_ACTIONS, big_blind=BIG_BLIND, temperature=TEMP,
    )
    # Recompute expected: bias raises (indices 2..n-2) by |ev|*0.5, then softmax
    # on the per-scenario normalizer.
    evs = list(action_evs)
    for idx in range(2, N_ACTIONS - 1):
        evs[idx] = evs[idx] + abs(evs[idx]) * 0.5
    normalizer = max(100.0 + 20.0, BIG_BLIND) * TEMP
    expected = F.softmax(torch.tensor(evs) / normalizer, dim=0).tolist()
    for a, b in zip(out[0]["action_probs"], expected):
        assert abs(a - b) < TOL, f"{out[0]['action_probs']} vs {expected}"
    print("test_normalizer_consistent_with_bias: OK")


if __name__ == "__main__":
    test_identity_temperature_reproduces_base_probs()
    test_normalizer_consistent_with_bias()
    print("\nALL B.3 MODIFIER-NORMALIZER TESTS PASSED")
