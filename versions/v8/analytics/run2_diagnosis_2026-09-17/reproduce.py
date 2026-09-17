"""Read-only diagnosis of run2: parse logs and reproduce the oracle bias.

Run with the repository venv. No training weights or production code are changed.
The example uses the production 17-action grid and 200 BB stacks. The opponent
is deliberately diagnostic, not claimed to occur in the recorded training run.
"""

import json
import math
from pathlib import Path
import re
import statistics
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gto_utils"))

import numpy as np
import eval7
import torch

from env.driver import HandSpec, LockstepDriver
from env.runout import HandRunout, RunoutConfig, prime
from oracle.posterior import opponent_posterior
from oracle.rollout import OracleConfig, action_values
from pool.base import PoolMember
from train.targets import decision_temperature, normalised_q, soft_q_loss


CARDS = [eval7.Card("23456789TJQKA"[i // 4] + "cdhs"[i % 4]) for i in range(52)]


def independent_payoff(deck, amount=200.):
    board = [CARDS[int(c)] for c in deck[:5]]
    left = eval7.evaluate(board + [CARDS[int(c)] for c in deck[5:7]])
    right = eval7.evaluate(board + [CARDS[int(c)] for c in deck[7:9]])
    return amount * ((left > right) - (left < right))


class Caller(PoolMember):
    def logits(self, contexts):
        result = np.zeros((len(contexts), self.n_actions))
        result[:, 1] = 1e9
        return result


class DiagnosticOpponent(PoolMember):
    def logits(self, contexts):
        result = np.zeros((len(contexts), self.n_actions))
        for i, ctx in enumerate(contexts):
            if ctx.turn == 2:
                action = 16 if set(ctx.hole_cards) == {30, 51} else 1
                result[i, action] = 1e9
        return result


def every_river(deck):
    for index in [4] + list(range(9, 52)):
        variant = deck.copy()
        variant[4], variant[index] = variant[index], variant[4]
        yield variant


def reproduce_oracle():
    game = json.loads((ROOT / "config.json").read_text())["game"]
    grid = [game["raise_sizes"][s] for s in ("preflop", "flop", "turn", "river")]
    assert game["n_actions"] == 17
    pool = [Caller("hero", 17), DiagnosticOpponent("diagnostic", 17)]
    spec = HandSpec(
        num_players=2, start_credits=[2000., 2000.], seat_members=[0, 1],
        seed=4005, big_blind=10., small_blind=5., raise_sizes=grid,
        deck=np.random.RandomState(4005).permutation(52),
        forced_actions=[15, 6, 1, 11, 7, 1, 16, 1],
    )
    plain = LockstepDriver(pool, 17)
    record = plain.run([spec])[0]
    snapshot = record.snapshots[record.decisions[7]["snap_idx"]]
    river_specs = [replace(spec, deck=d) for d in every_river(record.deck)]
    raw = plain.run(river_specs)
    cv = LockstepDriver(pool, 17, runout=RunoutConfig(samples=16)).run(river_specs)
    exact_call = float(np.mean([r.rewards[0] for r in raw]) / 10.)
    frozen_cv = np.asarray([r.baseline_rewards[0] / 10. for r in cv])
    fold = -(2000. - snapshot["credits"][0]) / 10.
    assert len(river_specs) == 44
    assert np.allclose(frozen_cv, -175.)
    assert math.isclose(exact_call, -127.27272727272728)
    assert math.isclose(fold, -154.35)
    assert exact_call > fold > frozen_cv[0]
    independent = [independent_payoff(s.deck) for s in river_specs]
    np.testing.assert_allclose(independent, [r.rewards[0] / 10. for r in raw])

    # Same defect without an all-in: check the turn through, then check river.
    check_spec = replace(spec, seat_members=[0, 0],
                         forced_actions=[15, 6, 1, 11, 7, 1, 1, 1])
    checked_specs = [replace(check_spec, deck=d) for d in every_river(record.deck)]
    checked_raw = plain.run(checked_specs)
    checked_cv = LockstepDriver(pool, 17, runout=RunoutConfig(samples=16)).run(checked_specs)
    check_exact = float(np.mean([r.rewards[0] for r in checked_raw]) / 10.)
    check_estimated = float(np.mean([r.baseline_rewards[0] for r in checked_cv]) / 10.)
    assert not math.isclose(check_exact, check_estimated)

    # Directly compare the actual policy loss to the true return on this state.
    q_bad = np.full(17, np.nan)
    q_bad[:2] = [fold, -175.]
    legal = np.zeros(17, dtype=bool)
    legal[:2] = True
    q_norm = normalised_q(q_bad, legal, snapshot["pot"] / 10., snapshot["bets"].max() / 10.)
    temperature = decision_temperature({"initial_ev_loss_bb": 1.}, legal,
                                      snapshot["pot"] / 10., snapshot["bets"].max() / 10.,
                                      iteration=7)
    policy_checks = []
    for fold_p in (.5, .999):
        probs = np.zeros(17)
        probs[:2] = [fold_p, 1 - fold_p]
        logits = np.zeros(17)
        logits[:2] = np.log(probs[:2])
        loss = soft_q_loss(torch.tensor(logits)[None], torch.tensor(q_norm)[None],
                           torch.tensor(legal)[None], temperature)
        policy_checks.append({"fold_probability": fold_p, "soft_q_loss": float(loss),
                              "true_ev_bb": fold_p * fold + (1 - fold_p) * exact_call})
    assert policy_checks[1]["soft_q_loss"] < policy_checks[0]["soft_q_loss"]
    assert policy_checks[1]["true_ev_bb"] < policy_checks[0]["true_ev_bb"]

    checks = []
    for floor in (0., 1e-6):
        for runouts in (16, 64):
            for samples in (128, 1024):
                ocfg = OracleConfig(
                    samples_per_action=samples, likelihood_floor=floor,
                    control_variate=True, runout_samples=runouts,
                )
                q, legal, _ = action_values(
                    record, 7,
                    LockstepDriver(pool, 17, runout=ocfg.runout_config()),
                    pool, 0, ocfg, np.random.default_rng(7),
                )
                assert np.flatnonzero(legal).tolist() == [0, 1]
                expected_call = -175. if runouts == 16 else exact_call
                np.testing.assert_allclose(q[legal], [fold, expected_call], atol=1e-9)
                checks.append({"likelihood_floor": floor, "runout_samples": runouts,
                               "samples_per_action": samples,
                               "q_fold_bb": float(q[0]), "q_call_bb": float(q[1])})

    combos, weights = opponent_posterior(
        record, 1, 0, pool, 17, through_decision=6, floor=1e-6)
    mass = float(weights[np.all(combos == [30, 51], axis=1)][0])
    raw_bounds = [mass * exact_call + (1 - mass) * z for z in (-200., 200.)]
    cv_bounds = [mass * -175. + (1 - mass) * z for z in (-200., 200.)]
    assert raw_bounds[0] > fold > cv_bounds[1]
    return {
        "board_ids": record.deck[:4].tolist(),
        "hero_hole_ids": record.deck[5:7].tolist(),
        "opponent_hole_ids": record.deck[7:9].tolist(),
        "forced_actions": spec.forced_actions,
        "stack_bb": 200., "pot_bb": snapshot["pot"] / 10.,
        "call_bb": snapshot["bets"].max() / 10.,
        "rivers_enumerated": len(river_specs), "exact_q_call_bb": exact_call,
        "q_fold_bb": fold, "cv_q_call_bb": float(frozen_cv[0]),
        "true_call_advantage_bb": exact_call - fold,
        "oracle_call_advantage_bb": float(frozen_cv[0]) - fold,
        "checks": checks, "floored_posterior_main_combo_mass": mass,
        "true_call_bounds_with_floor_bb": raw_bounds,
        "asymptotic_cv_call_bounds_with_floor_bb": cv_bounds,
        "independent_eval7_wins_ties_losses": [independent.count(x) for x in (200., 0., -200.)],
        "non_allin_transition": {"exact_ev_bb": check_exact, "cv_ev_bb": check_estimated},
        "loss_versus_true_ev": policy_checks,
    }


def synthetic_turn_stress(n=1000):
    """Synthetic fixed-holding turn states, NOT the training distribution."""
    rng = np.random.default_rng(20260917)
    decks = [rng.permutation(52) for _ in range(n)]
    for deck in decks:
        deck[7:9] = np.sort(deck[7:9])  # The oracle's opponent-combo convention.
    runouts = [HandRunout(deck, 2, RunoutConfig(samples=16)) for deck in decks]
    prime([(runout, 2) for runout in runouts])
    estimated = np.array([r.baseline(2, [True, True], [2000., 2000.])[0] / 10.
                          for r in runouts])
    exact = np.array([np.mean([independent_payoff(d) for d in every_river(deck)])
                      for deck in decks])
    error = estimated - exact
    flips = {}
    for call_bb in (20., 50., 100., 150., 199.):
        fold_q = -(200. - call_bb)
        # Only strict preference reversals; ties are excluded.
        flips[str(call_bb)] = int(np.sum((exact - fold_q) * (estimated - fold_q) < 0))
    return {"distribution": "1000 uniform fixed-holding HU turn matchups; NOT run2 labels",
            "seed": 20260917, "n": n, "runout_samples": 16,
            "mean_signed_error_bb": float(error.mean()),
            "mean_absolute_error_bb": float(np.abs(error).mean()),
            "rmse_bb": float(np.sqrt(np.mean(error ** 2))),
            "max_absolute_error_bb": float(np.abs(error).max()),
            "strict_fold_call_reversals_by_call_bb": flips}


def parse_logs(directory):
    rows = {}
    for path in sorted(directory.glob("*.txt")):
        iteration = None
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r"===== iteration (\d+)", line)
            if match:
                iteration = int(match[1])
                rows.setdefault(iteration, {"iteration": iteration})
            if line.startswith("[agent] step "):
                match = re.search(r"step (\d+)/(\d+) kl=([\d.]+).*range_kl=([\d.]+)", line)
                if match:
                    rows[iteration].setdefault("training", []).append(
                        [int(match[1]), float(match[3]), float(match[4])])
            for mode in ("cold", "warm"):
                if line.startswith(f"[loop] {mode} oracle gap"):
                    rows[iteration][mode] = {
                        key: float(value) for key, value in
                        re.findall(r"(\w+)=([+-]?[\d.]+)", line)
                    }
                    rows[iteration][mode + "_source"] = f"{path}:{lineno}"
    completed = []
    for iteration, row in sorted(rows.items()):
        if "cold" not in row:
            continue
        training = row.pop("training")
        row["train_policy_last5_mean"] = statistics.mean(x[1] for x in training[-5:])
        row["train_range_last5_mean"] = statistics.mean(x[2] for x in training[-5:])
        row["target_weighted_geomean_policy_prob_upper"] = math.exp(-row["cold"]["kl"])
        completed.append(row)
    return completed


if __name__ == "__main__":
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/logs_poker"
    result = {"oracle_counterexample": reproduce_oracle(),
              "synthetic_turn_stress": synthetic_turn_stress(),
              "log_metrics": parse_logs(logs)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
