"""acting_pos convention tests (Audit B.2).

generate.py previously wrote the ACTOR into the post-action snapshot's
`acting_pos`, while collect.py / evaluate.py / generate_opponent.py write the
NEXT player to act. That made training events disagree with inference events.
The fix makes generate.py's post-action snapshot use `table.active_player`.

Two checks:

1. test_acting_pos_invariant_real_generation — runs the real solver-driven
   generator. In any rebuilt event list every post-action event (one-hot
   action) is immediately followed by a decision event, and after the fix both
   read `table.active_player` at the same moment, so their `acting_pos` MUST be
   equal. Under the bug the post-action event carries the actor, which differs
   from the next player whenever the table has > 1 active player → violations.

2. test_rebuild_events_cross_path_parity — drives a real Table through a forced
   action sequence, builds snapshots in the production (post-fix) format, and
   asserts generate._rebuild_events and evaluate._rebuild_events produce
   byte-identical event dicts (modulo the action list-vs-tensor representation,
   which the dataset loader normalizes).

Run (from versions/v6):
    python -m tests.test_acting_pos_convention
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils"))

import random
import numpy as np
import torch

from env.table import Table
from agent.train_scenarios.generation.generate import (
    generate_scenario,
    _rebuild_events as rebuild_gen,
)
from evaluation.evaluate import _rebuild_events as rebuild_eval

BIG_BLIND = 10
SMALL_BLIND = 5
RAISE_SIZES = [[0.5, 1.0, 2.0]] * 4
N_ACTIONS = len(RAISE_SIZES[0]) + 3  # 6


def _action_sum(act):
    return sum(float(x) for x in act)


def test_acting_pos_invariant_real_generation():
    """Real generator: post-action events share acting_pos with the next event."""
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    cfg = {
        "mc_iterations": 150, "big_blind": BIG_BLIND, "max_stack": 300,
        "max_players": 4, "gto_temperature": 0.2, "solver": "v3",
        "raise_sizes": {s: [0.5, 1.0, 2.0] for s in ("preflop", "flop", "turn", "river")},
        "eqr_enabled": True, "combo_response_iters": 8, "reraise_threshold": 0.72,
        "weighted_sampling": False,
        "threshold_smoothing": {"enabled": True, "beta_fold": 0.07, "beta_reraise": 0.07},
    }
    n_post = 0
    n_viol = 0
    n_scen = 0
    for _ in range(10):
        res = generate_scenario(cfg, device="cpu")
        if not res:
            continue
        for s in res:
            if s.get("scenario_type") == "modelling":
                continue
            evs = s["events"]
            n_scen += 1
            for i in range(len(evs) - 1):
                if abs(_action_sum(evs[i]["action"]) - 1.0) < 1e-6:  # post-action event
                    n_post += 1
                    if evs[i]["acting_pos"] != evs[i + 1]["acting_pos"]:
                        n_viol += 1
    assert n_scen > 0, "generator produced no scenarios"
    assert n_post > 0, "no post-action events exercised — test is vacuous"
    assert n_viol == 0, f"{n_viol}/{n_post} post-action events use the wrong acting_pos"
    print(f"test_acting_pos_invariant_real_generation: OK "
          f"(scenarios={n_scen}, post_events={n_post}, violations=0)")


def _normalize_action(act):
    if isinstance(act, torch.Tensor):
        return [float(x) for x in act.tolist()]
    return [float(x) for x in act]


def _forced_call_snapshots(num_players, seed):
    """Drive a real Table with all-calls and record production-format snapshots."""
    rng = np.random.RandomState(seed)
    table = Table(
        num_players=num_players, raise_sizes=RAISE_SIZES,
        start_credits=300, big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
    )
    table.credits = [300.0] * num_players
    table.start_table()
    deck = np.arange(52)
    rng.shuffle(deck)
    table.deck = deck

    snapshots = [{
        "pot": table.pot, "bets": np.copy(table.bets), "credits": list(table.credits),
        "turn": table.turn, "active_pos": table.active_player, "action": None,
    }]
    decisions = []  # (snap_idx, actor)
    for _ in range(4 * num_players):
        active_pos = table.active_player
        if table.players_state[active_pos] != 1:
            break
        snapshots.append({
            "pot": table.pot, "bets": np.copy(table.bets), "credits": list(table.credits),
            "turn": table.turn, "active_pos": active_pos, "action": None,
        })
        decisions.append((len(snapshots) - 1, active_pos))
        action_vec = torch.zeros(N_ACTIONS, dtype=torch.float32)
        action_vec[1] = 1.0  # call/check
        end, several_all_in, _state, _bet = table.step(action_vec)
        snapshots.append({
            "pot": table.pot, "bets": np.copy(table.bets), "credits": list(table.credits),
            "turn": table.turn, "active_pos": table.active_player, "action": action_vec,
        })
        if end or several_all_in:
            break
    return table, snapshots, decisions


def test_rebuild_events_cross_path_parity():
    """generate._rebuild_events == evaluate._rebuild_events on identical snapshots."""
    checked = 0
    for num_players in (2, 3, 4):
        table, snapshots, decisions = _forced_call_snapshots(num_players, seed=num_players)
        assert len(decisions) >= 2, f"{num_players}p: too few decisions"
        for snap_idx, actor in decisions:
            evs_gen = rebuild_gen(snapshots, table.deck, actor, num_players,
                                  BIG_BLIND, SMALL_BLIND, N_ACTIONS, up_to=snap_idx)
            evs_eval = rebuild_eval(snapshots, table.deck, actor, num_players,
                                    BIG_BLIND, SMALL_BLIND, N_ACTIONS, up_to=snap_idx)
            assert len(evs_gen) == len(evs_eval) == snap_idx + 1
            for g, e in zip(evs_gen, evs_eval):
                assert g["hand"] == e["hand"]
                assert g["num_players"] == e["num_players"]
                assert g["hero_pos"] == e["hero_pos"]
                assert g["acting_pos"] == e["acting_pos"], (
                    f"{num_players}p snap {snap_idx}: acting_pos {g['acting_pos']} != {e['acting_pos']}"
                )
                assert g["big_blind"] == e["big_blind"]
                assert g["small_blind"] == e["small_blind"]
                assert g["stack"] == e["stack"]
                assert g["table"] == e["table"], f"table cards differ: {g['table']} vs {e['table']}"
                assert abs(g["pot"] - e["pot"]) < 1e-9
                assert np.array_equal(np.asarray(g["bets"]), np.asarray(e["bets"]))
                assert _normalize_action(g["action"]) == _normalize_action(e["action"])
            checked += 1
    assert checked > 0
    print(f"test_rebuild_events_cross_path_parity: OK ({checked} decision points across 2/3/4-handed)")


if __name__ == "__main__":
    test_acting_pos_invariant_real_generation()
    test_rebuild_events_cross_path_parity()
    print("\nALL B.2 ACTING_POS TESTS PASSED")
