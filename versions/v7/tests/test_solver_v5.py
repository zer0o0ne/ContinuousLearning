"""E2E tests for solver v5 (chance-sampled vector CFR, gpu_solver_v5).

Usage-scenario tests, fully deterministic (fixed seeds, no probabilistic
assertions):

1. A polarized river toy spot whose GTO EVs are known analytically —
   verifies the solver converges to the equilibrium value (the property v5
   exists for: no early-showdown bias, opponent responds with a strategy,
   not a threshold heuristic).
2. Same-seed reproducibility of a full solve.
3. Dataset generation end-to-end with solver "v5" — scenarios carry the
   same schema as v3-generated ones (action_evs / action_probs /
   legal_mask / equity) with finite values and a normalized policy.
4. The Slumbot eval path: _compute_all_action_evs on the eval table stub
   with solver_name="v5" returns playable EVs.

Run (from versions/v7):
    python -m pytest tests/test_solver_v5.py -v
"""

import os
import sys
import random
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

from gpu_solver_v5 import solve_spot  # noqa: E402
from agent.train_scenarios.generation.generate import (  # noqa: E402
    generate_scenario,
    _compute_all_action_evs,
)


def _card(rank, suit):
    return rank * 4 + suit


# K 9 5 2 6 rainbow-ish river board — AA is the effective nuts vs QQ.
_RIVER_BOARD = torch.tensor([
    _card(11, 0), _card(7, 1), _card(3, 2), _card(0, 3), _card(4, 0),
])


def _solve_river_toy(hero_cards, seed=7):
    """Polarized toy: hero range {AA (nuts), 32 (air)} vs QQ bluffcatcher.

    Pot 100, stack 100 (one pot-sized bet), hero acts first on the river.
    Analytic GTO: EV(bet pot | nuts) = P + P*B/(P+B) = 150,
    EV(check | nuts) = P = 100, EV(check | air) = 0.
    """
    return solve_spot(
        hero_cards=hero_cards,
        board_cards=_RIVER_BOARD,
        opponent_range_hand_types=[["QQ"]],
        pot=100.0, facing_bet=0.0, stack=100.0, hero_invested=0.0,
        street_raises=[1.0], effective_pot=100.0, n_actions=4,
        street=3, hero_position=0, n_players=2,
        action_history=[], opponent_positions=[1],
        big_blind=10.0,
        v5_params={"iterations": 400, "batch_runouts": 1,
                   "future_bet_sizes": [1.0], "raise_cap": 1},
        seed=seed,
        hero_range_hand_types=["AA", "32s", "32o"],
    )


def test_river_polarized_equilibrium():
    hero_nuts = torch.tensor([_card(12, 0), _card(12, 1)])  # AA
    evs, equity = _solve_river_toy(hero_nuts)

    assert evs is not None
    # fold EV is exact
    assert float(evs[0]) == 0.0
    # nuts always win at showdown on this board
    assert equity > 0.99
    # checking back realizes exactly the pot (opponent checks behind /
    # never value-bets a bluffcatcher into a polarized range)
    assert abs(float(evs[1]) - 100.0) < 3.0
    # pot-sized value bet converges to the analytic equilibrium EV of 150
    # (opponent calls with frequency P/(P+B) = 1/2)
    assert abs(float(evs[2]) - 150.0) < 10.0
    # betting the nuts strictly beats checking them
    assert float(evs[2]) > float(evs[1]) + 20.0

    hero_air = torch.tensor([_card(1, 0), _card(0, 1)])  # 3-2 offsuit
    evs_air, equity_air = _solve_river_toy(hero_air)
    assert equity_air < 0.01
    # air checking loses the pot: EV 0 (nothing more invested)
    assert abs(float(evs_air[1])) < 1.0
    # at equilibrium a bluff is close to indifferent — it must never look
    # clearly profitable, and can be mildly negative at finite iterations
    assert float(evs_air[2]) < 5.0
    assert float(evs_air[2]) > -25.0


def _solve_river_short_stack(hero_cards, bet_action_only=False, seed=11):
    """HU river vs a SHORT stack: pot 100, hero stack 200, opp stack 50.

    Any hero bet >= 50 is effectively a 50 bet (opp calls all-in for less,
    the excess is an uncalled-bet refund side layer). Analytic GTO with
    effective bet B=50: opp calls with q = 2/3 (bluff indifference),
    EV(bet | nuts) = P + q*B = 133.3 — identical for the 100-bet and the
    200-jam.
    """
    return solve_spot(
        hero_cards=hero_cards,
        board_cards=_RIVER_BOARD,
        opponent_range_hand_types=[["QQ"]],
        pot=100.0, facing_bet=0.0, stack=200.0, hero_invested=0.0,
        street_raises=[1.0], effective_pot=100.0, n_actions=4,
        street=3, hero_position=0, n_players=2,
        action_history=[], opponent_positions=[1],
        big_blind=10.0,
        v5_params={"iterations": 400, "batch_runouts": 1,
                   "future_bet_sizes": [1.0], "raise_cap": 1},
        seed=seed,
        hero_range_hand_types=["AA", "32s", "32o"],
        opponent_stacks=[50.0], opponent_invested=[0.0],
    )


def test_side_pot_refund_heads_up():
    """Uncalled-bet refund: betting 100 and jamming 200 into a 50-stack
    must both play as an effective 50 bet."""
    hero_nuts = torch.tensor([_card(12, 0), _card(12, 1)])  # AA
    evs, _ = _solve_river_short_stack(hero_nuts)
    ev_bet100, ev_jam = float(evs[2]), float(evs[3])
    # analytic equilibrium: 100 + (2/3)*50 = 133.3
    assert abs(ev_bet100 - 133.3) < 10.0
    assert abs(ev_jam - 133.3) < 10.0
    # the two sizings are the same effective action — EVs must match
    assert abs(ev_bet100 - ev_jam) < 5.0

    hero_air = torch.tensor([_card(1, 0), _card(0, 1)])  # 3-2 offsuit
    evs_air, _ = _solve_river_short_stack(hero_air)
    # a bluff risks only the 50 the opponent can call, not the full jam:
    # at equilibrium it is ~indifferent to checking (EV 0). Without the
    # refund layer the 200-jam would burn ~-100 or worse here.
    assert float(evs_air[3]) > -25.0
    assert float(evs_air[3]) < 10.0
    assert abs(float(evs_air[2]) - float(evs_air[3])) < 8.0


def test_side_pot_multiway_short_stack():
    """3-way river with a 40-chip short stack: side-pot solve is finite,
    deterministic, and the nut hand's jam keeps at least the pot (refund
    guarantees the uncalled excess is never lost)."""
    hero_nuts = torch.tensor([_card(12, 0), _card(12, 1)])  # AA (nuts)
    kwargs = dict(
        hero_cards=hero_nuts,
        board_cards=_RIVER_BOARD,
        opponent_range_hand_types=[["QQ"], ["JJ"]],
        pot=90.0, facing_bet=0.0, stack=300.0, hero_invested=30.0,
        street_raises=[1.0], effective_pot=90.0, n_actions=4,
        street=3, hero_position=0, n_players=3,
        action_history=[], opponent_positions=[1, 2],
        big_blind=10.0,
        v5_params={"iterations": 200, "batch_runouts": 1,
                   "future_bet_sizes": [1.0], "raise_cap": 1},
        seed=13,
        hero_range_hand_types=["AA", "32s", "32o"],
        opponent_stacks=[40.0, 300.0], opponent_invested=[30.0, 30.0],
    )
    evs, equity = solve_spot(**kwargs)
    evs2, equity2 = solve_spot(**kwargs)
    assert torch.equal(evs, evs2) and equity == equity2
    assert bool(torch.isfinite(evs).all())
    assert equity > 0.99
    # nuts jam: worst case everyone folds -> hero wins the 90 pot and the
    # whole jam is refunded; calls only add chips
    assert float(evs[3]) > 80.0
    # and it cannot exceed pot + everything the opponents can still pay
    assert float(evs[3]) <= 90.0 + 40.0 + 300.0 + 1.0


def test_solve_deterministic_same_seed():
    hero = torch.tensor([_card(12, 0), _card(11, 0)])  # AKs
    board = torch.tensor([_card(10, 1), _card(5, 2), _card(0, 3)])
    kwargs = dict(
        hero_cards=hero, board_cards=board,
        opponent_range_hand_types=[["AA", "KK", "QQ", "JJ", "AKs", "AQs"]],
        pot=60.0, facing_bet=20.0, stack=400.0, hero_invested=20.0,
        street_raises=[0.5, 1.0, 2.0], effective_pot=60.0, n_actions=6,
        street=1, hero_position=5, n_players=6,
        action_history=[(2, "open")], opponent_positions=[2],
        big_blind=10.0,
        v5_params={"iterations": 24, "batch_runouts": 2, "max_combos": 60},
        seed=123,
    )
    evs_a, eq_a = solve_spot(**kwargs)
    evs_b, eq_b = solve_spot(**kwargs)
    assert torch.equal(evs_a, evs_b)
    assert eq_a == eq_b
    assert bool(torch.isfinite(evs_a).all())
    # fold EV convention matches v3: -hero_invested
    assert float(evs_a[0]) == -20.0


_GEN_CONFIG = {
    "big_blind": 10,
    "max_stack": 300,
    "max_players": 2,
    "gto_temperature": 0.2,
    "raise_sizes": {
        "preflop": [0.5, 1.0, 2.0],
        "flop": [0.33, 0.75, 1.5],
        "turn": [0.33, 0.75, 1.5],
        "river": [0.33, 0.75, 1.5],
    },
    "solver": "v5",
    "v5": {
        "iterations": 12,
        "batch_runouts": 2,
        "max_combos": 50,
        "max_tree_nodes": 2500,
    },
}


def test_generation_e2e_v5():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    n_actions = len(_GEN_CONFIG["raise_sizes"]["preflop"]) + 3
    collected = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            scens = generate_scenario(_GEN_CONFIG, device="cpu")
            if scens:
                collected.extend(scens)
    assert collected, "v5 generation produced no scenarios in 4 hands"

    for s in collected:
        assert len(s["action_evs"]) == n_actions
        assert len(s["action_probs"]) == n_actions
        assert len(s["legal_mask"]) == n_actions
        assert all(np.isfinite(v) for v in s["action_evs"])
        assert abs(sum(s["action_probs"]) - 1.0) < 1e-4
        # illegal actions carry zero policy mass
        for prob, legal in zip(s["action_probs"], s["legal_mask"]):
            if not legal:
                assert prob == 0.0
        assert 0.0 <= s["equity"] <= 1.0
        # ev_target is the max action EV (generate.py invariant)
        assert abs(s["ev_target"] - max(s["action_evs"])) < 1e-4
        assert len(s["events"]) >= 2


def test_slumbot_stub_v5():
    from evaluation.slumbot_eval import _make_solver_table_stub

    random.seed(42)
    np.random.seed(42)

    raise_sizes = [
        [0.5, 1.0, 2.0],
        [0.33, 0.75, 1.5],
        [0.33, 0.75, 1.5],
        [0.33, 0.75, 1.5],
    ]
    state = {
        "pot": 300,
        "bets": [150, 150],
        "credits": [9850, 9850],
        "players_state": [1, 1],
        "high_bet": 150,
        "turn": 1,
        "active_pos": 0,
        "last_bet_size": 0,
    }
    stub = _make_solver_table_stub(
        hero_user_pos=1, hole_cards_int=[_card(12, 0), _card(12, 1)],
        board_ints=[_card(10, 1), _card(5, 2), _card(0, 3), -1, -1],
        state=state, raise_sizes=raise_sizes,
        big_blind_internal=10.0, small_blind_internal=5.0,
        chip_scale=10.0, num_players=2,
    )
    evs, meta = _compute_all_action_evs(
        stub, 1, [(0, "open")], 6,
        solver_name="v5", device="cpu",
        v5_params={"iterations": 12, "batch_runouts": 2, "max_combos": 50,
                   "max_tree_nodes": 2500},
    )
    assert evs is not None and meta is not None
    assert bool(torch.isfinite(evs).all())
    assert len(meta["legal_mask"]) == 6
    assert 0.0 <= float(meta["equity"]) <= 1.0
