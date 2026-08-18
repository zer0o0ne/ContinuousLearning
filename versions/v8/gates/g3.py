"""G3 — what does one oracle label actually cost? (`CONCEPT.md` §14, §13.)

§13 puts a *design figure* on variant A: ~10⁵ policy forwards and ~3 s per
label, from which 10⁵ labels per iteration is affordable and 10⁶ is not. Every
number in that paragraph is a hypothesis about hardware this repository cannot
run (`CLAUDE.md` §3). This gate replaces the hypothesis with a measurement, and
it is deliberately the last thing built before the pipeline is designed around
the answer (`PLAN_PIPELINE.md` S4).

**What it measures.** A realistic pool plays a few hundred hands; a fixed set of
hero decisions out of those hands is then labelled by `oracle/rollout.py` under
a sweep over the four axes that plausibly move the cost:

    samples_per_action × max_combos × table size × stack depth

Per cell: wall clock per label, policy forwards per label split into the
posterior's share and the rollouts' share, the rollout depth actually seen, the
collision rate of the joint draw, and the labels-per-hour that follows.

**Why the split-half column.** The sweep above says what a label *costs*; it
says nothing about how many samples a label *needs*. Both halves of that are
required to choose `oracle.samples_per_action`, and the second half is free:
the rollouts are already played, so splitting each label's samples in two and
comparing the halves estimates the standard error of `q` at no extra cost.
`E[(q_A − q_B)²] = 4·Var(q)` when the halves are equal and independent, so the
reported `se_q` is `sqrt(mean((q_A − q_B)²)) / 2`, in BB, pooled over every
label and every legal action of the cell. (This column is an addition to what
§14 asks for; it is in because the owner asked for it.)

**What it does not do.** It does not tune anything and it does not try to make
the number better (`PLAN_PIPELINE.md` S4, non-goals). It measures, prints the
table, and stops — the decision that follows, variant A as it stands versus
variant C of §7.4, is the owner's and is taken by reading this table.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from env.driver import HandSpec, LockstepDriver
from gates.g1 import raise_sizes_from
from oracle.rollout import OracleConfig, action_values
from pool.build import build_pool
from utils import Logger, progress, resolve_device


class MeasuringDriver(LockstepDriver):
    """A driver that remembers the hands of its last `run`.

    `LabelStats` reports totals, which is all the pipeline needs. Two of G3's
    columns cannot be recovered from a total: the rollout depth (forwards per
    rollout, and only the *rollout* forwards — not the posterior's) and the
    per-sample rewards the split-half standard error is built from. Both are
    sitting in the records the label just played, so the gate reads them here
    instead of widening the oracle's return value for one experiment.
    """

    last_records = None

    def run(self, specs, batch_size=None, desc=None):
        records = super().run(specs, batch_size=batch_size, desc=desc)
        self.last_records = records
        return records


def build_hands(rng, member_ids, game, num_players, stack_bb, n_hands,
                seed_base):
    """`n_hands` hands at a **fixed** table size and stack depth.

    G3 is the one place where the two axes are pinned rather than sampled: the
    question is how the cost varies along them, which needs cells, not a
    uniform draw. Nothing here is training data, so `CLAUDE.md` §1's sampling
    rule is not weakened by it.
    """
    assert num_players <= len(member_ids), (
        f"a {num_players}-handed table needs {num_players} distinct pool "
        f"members, this pool has {len(member_ids)}")
    bb, sb = game["big_blind"], game["small_blind"]
    raise_sizes = raise_sizes_from(game)
    specs = []
    for h in range(n_hands):
        members = [int(m) for m in
                   rng.choice(member_ids, size=num_players, replace=False)]
        specs.append(HandSpec(
            num_players=num_players,
            start_credits=[float(stack_bb * bb)] * num_players,
            seat_members=members,
            seed=seed_base + h,
            big_blind=bb, small_blind=sb, raise_sizes=raise_sizes,
            meta={"players": num_players, "stack_bb": stack_bb, "hand": h},
        ))
    return specs


def choose_decisions(rng, records, n_wanted):
    """A fixed set of hero decisions, shared by every cell of the sweep.

    Shared on purpose: the cells differ only in `samples_per_action` and
    `max_combos`, so labelling the *same* decisions makes the columns
    comparable and makes the forwards count monotone in the sample budget for a
    reason other than luck.
    """
    pairs = [(h, d) for h, record in enumerate(records)
             for d in range(len(record.decisions))]
    if not pairs:
        return []
    take = min(int(n_wanted), len(pairs))
    picked = rng.choice(len(pairs), size=take, replace=False)
    return [pairs[int(i)] for i in sorted(picked)]


def measure_label(record, decision_idx, driver, pool, cfg, rng):
    """Label one hero decision and report what it cost."""
    driver.last_records = None
    hero_pos = int(record.decisions[decision_idx]["acting_pos"])
    # The member already in hero's seat plays hero in the rollouts. At
    # iteration 0 of §8 that is exactly what happens — the oracle improves on a
    # pool member — so the cost measured here is the cost of the real thing.
    hero_member = int(record.spec.seat_members[hero_pos])

    q, legal, stats = action_values(record, decision_idx, driver, pool,
                                    hero_member, cfg, rng)

    played = driver.last_records or []
    # Identical to the oracle's own count: `forced_actions` is prefix + [a], so
    # the decisions past it are the ones a policy was actually asked about.
    rollout_forwards = sum(len(r.decisions) - len(r.spec.forced_actions)
                           for r in played)
    n_legal = int(legal.sum())
    n_samples = len(played) // n_legal if n_legal else 0

    row = {
        "players": int(record.num_players),
        "stack_bb": int(round(record.spec.start_credits[hero_pos]
                              / record.spec.big_blind)),
        "n_legal": n_legal,
        "n_samples": n_samples,
        "seconds": float(stats.seconds),
        "forwards": int(stats.forwards),
        "rollout_forwards": int(rollout_forwards),
        "posterior_forwards": int(stats.forwards - rollout_forwards),
        "n_rollouts": int(stats.n_rollouts),
        "collision_rate": float(stats.collision_rate),
        "depth": (rollout_forwards / stats.n_rollouts) if stats.n_rollouts
                 else float("nan"),
        "half_gap_sq": [],
    }

    if n_samples >= 2:
        bb = float(record.spec.big_blind)
        rewards = np.asarray([r.rewards[hero_pos] for r in played],
                             dtype=np.float64).reshape(n_samples, n_legal) / bb
        half = n_samples // 2
        gap = rewards[:half].mean(axis=0) - rewards[half:2 * half].mean(axis=0)
        row["half_gap_sq"] = [float(g * g) for g in gap]
    return row


def _mean(values):
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def aggregate_cell(cell, rows, iteration_labels):
    """One row of the printed table, from the labels of one cell."""
    seconds = _mean([r["seconds"] for r in rows])
    gaps = [g for r in rows for g in r["half_gap_sq"]]
    out = dict(cell)
    out.update({
        "n_labels": len(rows),
        "seconds_per_label": seconds,
        "forwards_per_label": _mean([r["forwards"] for r in rows]),
        "posterior_forwards": _mean([r["posterior_forwards"] for r in rows]),
        "rollout_forwards": _mean([r["rollout_forwards"] for r in rows]),
        "rollouts_per_label": _mean([r["n_rollouts"] for r in rows]),
        "legal_actions": _mean([r["n_legal"] for r in rows]),
        "depth": _mean([r["depth"] for r in rows]),
        "collision_rate": _mean([r["collision_rate"] for r in rows]),
        "empty_labels": sum(1 for r in rows if r["n_rollouts"] == 0),
        "labels_per_hour": (3600.0 / seconds) if seconds > 0 else float("nan"),
        "se_q": (math.sqrt(float(np.mean(gaps))) / 2.0) if gaps
                else float("nan"),
    })
    out["iteration_hours"] = (iteration_labels / out["labels_per_hour"]
                              if out["labels_per_hour"] > 0 else float("nan"))
    return out


def headline(cells, iteration_labels):
    """The §13 sentence, one line per sample budget.

    Aggregated over table size and stack depth at the *exact* posterior
    (`max_combos = None`), because that is the configuration §7.2 describes and
    the one the pipeline runs unless this table says it cannot afford it.
    """
    out = []
    budgets = sorted({c["samples_per_action"] for c in cells})
    for s in budgets:
        rows = [c for c in cells
                if c["samples_per_action"] == s and c["max_combos"] is None]
        if not rows:
            continue
        seconds = _mean([r["seconds_per_label"] for r in rows])
        per_hour = 3600.0 / seconds if seconds > 0 else float("nan")
        out.append({
            "samples_per_action": s,
            "seconds_per_label": seconds,
            "forwards_per_label": _mean([r["forwards_per_label"]
                                         for r in rows]),
            "labels_per_hour": per_hour,
            "iteration_labels": iteration_labels,
            "iteration_hours": (iteration_labels / per_hour
                                if per_hour > 0 else float("nan")),
            "se_q": _mean([r["se_q"] for r in rows]),
        })
    return out


def format_report(report, log):
    log("")
    log("G3 — cost of one oracle label (variant A, CONCEPT.md §7.1)")
    log(f"{'smp':>4} {'combos':>7} {'plr':>4} {'stack':>6} {'s/label':>9} "
        f"{'forwards':>10} {'post':>8} {'roll':>8} {'depth':>7} "
        f"{'coll%':>7} {'lab/h':>9} {'SE(q)BB':>9}")
    for c in report["cells"]:
        combos = "all" if c["max_combos"] is None else str(c["max_combos"])
        log(f"{c['samples_per_action']:>4} {combos:>7} {c['players']:>4} "
            f"{c['stack_bb']:>6} {c['seconds_per_label']:>9.3f} "
            f"{c['forwards_per_label']:>10.0f} {c['posterior_forwards']:>8.0f} "
            f"{c['rollout_forwards']:>8.0f} {c['depth']:>7.2f} "
            f"{100 * c['collision_rate']:>7.1f} {c['labels_per_hour']:>9.0f} "
            f"{c['se_q']:>9.3f}")

    log("")
    log("Headline (exact posterior, averaged over table size and stack depth)")
    for h in report["headline"]:
        log(f"  samples_per_action={h['samples_per_action']:<4} "
            f"{h['seconds_per_label']:.2f} s/label, "
            f"{h['forwards_per_label']:.0f} forwards/label "
            f"→ {h['labels_per_hour']:.0f} labels/hour; an iteration of "
            f"{h['iteration_labels']} labels costs "
            f"{h['iteration_hours']:.1f} h; SE(q) ≈ {h['se_q']:.3f} BB")

    n = report["n_labels"]
    log("")
    log(f"labels planned {n['planned']}, done {n['done']}, "
        f"skipped {n['skipped']} (too few decisions in the played hands)")
    log(f"hands played to source the decisions: {report['n_hands_played']}")


def run(config, log, out_dir):
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")
    rng = np.random.default_rng(config["seed"])
    game = config["game"]
    sweep = config["sweep"]
    oracle_cfg = config.get("oracle", {})
    iteration_labels = int(config.get("iteration_labels", 100_000))

    pool, descriptors = build_pool(config, rng, device=device, log=log)
    log(f"pool: {len(pool)} members")
    driver = MeasuringDriver(pool, int(game["n_actions"]))
    member_ids = list(range(len(pool)))

    # ---- the hands the labelled decisions come out of, one batch, one bar
    groups = [(int(p), int(s)) for p in sweep["table_sizes"]
              for s in sweep["stack_bb"]]
    hands_per_group = int(sweep["hands_per_table"])
    specs, spans = [], []
    for gi, (players, stack_bb) in enumerate(groups):
        group_specs = build_hands(rng, member_ids, game, players, stack_bb,
                                  hands_per_group,
                                  seed_base=int(config["seed"]) * 1_000_000
                                            + gi * 10_000)
        spans.append((len(specs), len(specs) + len(group_specs)))
        specs += group_specs
    t0 = time.perf_counter()
    played = driver.run(specs, batch_size=int(sweep.get("driver_batch_size",
                                                        512)),
                        desc="play")
    log(f"played {len(played)} hands in {time.perf_counter() - t0:.1f}s")

    # ---- the sweep
    budgets = list(sweep["samples_per_action"])
    combo_caps = list(sweep["max_combos"])
    labels_per_cell = int(sweep["labels_per_cell"])
    planned = len(groups) * len(budgets) * len(combo_caps) * labels_per_cell
    bar = progress(total=planned, desc="label", unit="label")

    cells, rows_all, done, skipped = [], [], 0, 0
    t0 = time.perf_counter()
    for gi, (players, stack_bb) in enumerate(groups):
        lo, hi = spans[gi]
        records = played[lo:hi]
        chosen = choose_decisions(np.random.default_rng([config["seed"], gi]),
                                  records, labels_per_cell)
        if len(chosen) < labels_per_cell:
            log(f"[{players}p/{stack_bb}bb] only {len(chosen)} decisions "
                f"available, wanted {labels_per_cell}")
        for si, samples in enumerate(budgets):
            for ci, max_combos in enumerate(combo_caps):
                cfg = OracleConfig(
                    samples_per_action=int(samples),
                    max_combos=None if max_combos is None else int(max_combos),
                    likelihood_floor=float(oracle_cfg.get("likelihood_floor",
                                                          1e-6)),
                    batch_hands=int(oracle_cfg.get("batch_hands", 2048)),
                    max_collision_retries=int(
                        oracle_cfg.get("max_collision_retries", 32)),
                )
                rows = []
                for li, (h, d) in enumerate(chosen):
                    label_rng = np.random.default_rng(
                        [config["seed"], gi, si, ci, li])
                    rows.append(measure_label(records[h], d, driver, pool,
                                              cfg, label_rng))
                    bar.update(1)
                    done += 1
                # A skipped label still owes the bar its units (`CLAUDE.md` §5)
                missing = labels_per_cell - len(chosen)
                if missing > 0:
                    bar.update(missing)
                    skipped += missing
                cell = {"samples_per_action": int(samples),
                        "max_combos": (None if max_combos is None
                                       else int(max_combos)),
                        "players": players, "stack_bb": stack_bb}
                cells.append(aggregate_cell(cell, rows, iteration_labels))
                rows_all += [{**cell, **r} for r in rows]
    bar.close()
    elapsed = time.perf_counter() - t0
    log(f"labelled {done} decisions over {len(cells)} cells in {elapsed:.1f}s")

    report = {
        "cells": cells,
        "headline": headline(cells, iteration_labels),
        "n_labels": {"planned": planned, "done": done, "skipped": skipped},
        "n_hands_played": len(played),
        "labelling_seconds": elapsed,
        "device": str(device),
    }
    format_report(report, log)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "g3_report.json")
    with open(path, "w") as fh:
        json.dump({"config": config, "pool": descriptors,
                   "report": report, "rows": rows_all},
                  fh, indent=1, default=float)
    log(f"wrote {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="CONCEPT.md §14 gate G3")
    parser.add_argument("--config", default="config_g3.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    base_dir = config.get("out_dir", "../../data/v8")
    log = Logger(base_dir)
    out_dir = log.run_dir("g3")
    try:
        run(config, log, out_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
