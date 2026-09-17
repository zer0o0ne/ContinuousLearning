"""Paired CPU cost probe; production modules remain unchanged.

Measures complete oracle calls on synthetic HU check-through records using
the production 17-action grid. No learned checkpoint or GPU is exercised.
The fresh prototype includes extra dataclass replacements and HandRunout
construction that an integrated fix can avoid.
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
import platform
import statistics
import time

from fresh_cv_probe import (Caller, FreshDriver, FreshRunout, HandRunout,
                            LockstepDriver, OracleConfig, _label_plan,
                            _label_q, fixture)
import numpy as np
import torch
from utils import progress


def prepare(pool, record, decision, samples, seed, fresh, driver):
    children = np.random.SeedSequence(seed).spawn(2)
    cfg = OracleConfig(samples_per_action=samples, likelihood_floor=0.,
                       control_variate=True, runout_samples=16,
                       batch_hands=2048)
    plan = _label_plan(record, decision, driver, pool, 0, cfg,
                       np.random.default_rng(children[0]))
    if fresh:
        aux = np.random.default_rng(children[1])
        seeds = aux.integers(0, 2**63, size=plan.n_samples)
        width = len(plan.legal_idx)
        plan.specs = [replace(s, meta={**s.meta, "cv_seed": int(seeds[i // width])})
                      for i, s in enumerate(plan.specs)]
    return plan


def measure(pool, record, decision, samples, seed, fresh):
    cfg = OracleConfig(runout_samples=16)
    started = time.perf_counter()
    driver = (FreshDriver if fresh else LockstepDriver)(
        pool, 17, runout=cfg.runout_config())
    plan = prepare(pool, record, decision, samples, seed, fresh, driver)
    prepared = time.perf_counter()
    played = driver.run(plan.specs, batch_size=2048)
    finished = time.perf_counter()
    _, _, stats = _label_q(plan, played, finished - started)
    return {"seconds": time.perf_counter() - started,
            "prepare_seconds": prepared - started,
            "play_seconds": finished - prepared,
            "rollouts": len(played), "forwards": stats.forwards,
            "legal_actions": len(plan.legal_idx)}, played, plan


class CountOld(HandRunout):
    counts = None

    def __init__(self, deck, num_players, cfg, auxiliary_seed, scores=None):
        super().__init__(deck, num_players, cfg, scores=scores)

    def _job(self, turn):
        job = super()._job(turn)
        self.counts["jobs"] += 1
        self.counts["ranking_rows"] += len(job[1]) * len(job[2])
        return job


class CountFresh(FreshRunout):
    counts = None

    def _job(self, turn):
        job = super()._job(turn)
        self.counts["jobs"] += 1
        self.counts["ranking_rows"] += len(job[1]) * len(job[2])
        return job


def work_counts(pool, record, decision, samples, seed, fresh):
    cls = CountFresh if fresh else CountOld
    cls.counts = Counter()
    driver = FreshDriver(pool, 17, runout=OracleConfig().runout_config())
    driver.runout_class = cls
    # Old instrumented class accepts and ignores the auxiliary seed; this
    # permits both counters through the same subclass seam without patching.
    plan = prepare(pool, record, decision, samples, seed, True, driver)
    played = driver.run(plan.specs, batch_size=2048)
    return dict(cls.counts), played, plan


def check_trajectories(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a.deck, b.deck)
        np.testing.assert_array_equal(a.rewards, b.rewards)
        assert [(x["acting_pos"], x["action_idx"]) for x in a.decisions] == [
            (x["acting_pos"], x["action_idx"]) for x in b.decisions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, base, _ = fixture()
    pool = [Caller("hero", 17), Caller("opponent", 17)]
    base = replace(base, forced_actions=None)
    record = LockstepDriver(pool, 17).run([base])[0]
    decisions = [next(i for i, d in enumerate(record.decisions)
                      if record.snapshots[d["snap_idx"]]["turn"] == turn)
                 for turn in range(4)]
    result = {"platform": platform.platform(), "torch": torch.__version__,
              "torch_threads": 1, "S": args.samples, "M": 16,
              "repetitions": args.repetitions,
              "scope": "synthetic HU Caller policies, complete labels, CPU only",
              "cases": []}
    bar = progress(total=4 * (2 * args.repetitions + 4),
                   desc="CPU CV cost probe", unit="label")
    for name, decision in zip(("preflop", "flop", "turn", "river"), decisions):
        for fresh in (False, True):
            measure(pool, record, decision, args.samples, 100, fresh)
            bar.update(1)
        counts, traces, plans = [], [], []
        for fresh in (False, True):
            c, trace, plan = work_counts(pool, record, decision,
                                         args.samples, 777, fresh)
            counts.append(c)
            traces.append(trace)
            plans.append(plan)
            bar.update(1)
        check_trajectories(*traces)
        assert counts[0] == counts[1]
        assert len({s.deck.tobytes() for s in plans[0].specs}) == args.samples
        pairs = []
        for rep in range(args.repetitions):
            pair = {}
            for fresh in ((False, True) if rep % 2 == 0 else (True, False)):
                measurement, _, _ = measure(pool, record, decision,
                                             args.samples, 1000 + rep, fresh)
                pair["fresh" if fresh else "old"] = measurement
                bar.update(1)
            assert pair["old"]["forwards"] == pair["fresh"]["forwards"]
            assert pair["old"]["rollouts"] == pair["fresh"]["rollouts"]
            pairs.append(pair)
        old = statistics.median(p["old"]["seconds"] for p in pairs)
        fresh = statistics.median(p["fresh"]["seconds"] for p in pairs)
        result["cases"].append({"street": name, "decision": decision,
            "old_median_seconds": old, "fresh_median_seconds": fresh,
            "ratio_of_medians": fresh / old,
            "median_paired_ratio": statistics.median(
                p["fresh"]["seconds"] / p["old"]["seconds"] for p in pairs),
            "counts_old": counts[0], "counts_fresh": counts[1],
            "identical_trajectories": True, "unique_decks": args.samples,
            "pairs": pairs})
    bar.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
