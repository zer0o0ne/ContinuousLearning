"""Are the archetypes ten different players, and are they better than the
degenerate five? (`PLAN_PROCEDURAL_POOL.md` §P4, and §0.3's owner decision)

Those are the only two questions asked of the procedural pool, and both are
answered in self-play, with no external opponent:

**Diverse.** Every archetype is seated in a mixed field of the other nine, at
each table size, over the whole stack range, and read by its stat line — the
sixteen frequencies a player would be read by. Ten archetypes that produce ten
distinguishable stat lines *is* the claim. The report prints them side by side
so a human can see the shape rather than a pass/fail flag, and splits every stat
by short and deep stacks, because a push/fold regime that is silently wrong
shows up nowhere else.

**Better than the degenerate five.** Every archetype is seated against
always-fold / always-call / always-min-raise / maniac / nit and measured in
BB/100, with its standard error. A board-reading regular that cannot beat a
member which never looks at its cards is not worth its cost.

There are no acceptance bands and nothing here passes or fails. That is the
owner's decision of 2026-09-03: with two dozen hand-set knobs there is no
realistic path from a measurement back into a fitted pool, so the machinery that
would have guarded against one is not built.

Run::

    cd versions/v8 && python3 -m gates.pool_realism --config config_pr.json
"""

import argparse
import json
import os
import time

import numpy as np

from env.driver import HandSpec, LockstepDriver
from env.showdown import label_showdowns
from pool.archetypes import ARCHETYPES, draw_params
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.regular import RegularMember
from pool.stats import hud_stats
from pool.strength import (DEFAULT_TABLE_PATH, StrengthCache,
                           preflop_equity_table)
from pool.style import StyleParams
from utils import Logger, progress

#: Table sizes the report is cut by, and what to call them.
SIZES = {"hu": 2, "6max": 6, "9max": 9}
#: Stacks at or below this are the short bucket — the push/fold regime.
SHORT_BB = 25.0


def make_archetypes(names, cache, table, game, seed, spread=0.0):
    """One member per archetype, jittered by `spread` (0 = the preset)."""
    rng = np.random.default_rng(seed)
    n_actions = int(game["n_actions"])
    raise_sizes = [game["raise_sizes"][s]
                   for s in ("preflop", "flop", "turn", "river")]
    return [RegularMember(name, n_actions, draw_params(name, rng, spread),
                          cache, table, raise_sizes)
            for name in names]


def make_degenerates(game):
    n_actions = int(game["n_actions"])
    return [cls(name, n_actions, StyleParams.identity())
            for name, cls in DEGENERATE_STRATEGIES.items()]


def seat_hands(hero_idx, field_idx, n_players, hands, stack_range, big_blind,
               small_blind, raise_sizes, seed):
    """`hands` specs with the hero rotating through every seat.

    Rotating matters: a seat that never leaves the small blind has a positional
    stat line and not an archetype's. The field fills the other seats
    round-robin, and the stack depth is drawn per hand over the whole range, so
    the profile is over the distribution rather than at one depth.
    """
    rng = np.random.default_rng(seed)
    lo, hi = float(stack_range[0]), float(stack_range[1])
    specs = []
    for h in range(int(hands)):
        hero_seat = h % n_players
        stack_bb = float(rng.uniform(lo, hi))
        members, k = [], 0
        for seat in range(n_players):
            if seat == hero_seat:
                members.append(hero_idx)
            else:
                members.append(field_idx[k % len(field_idx)])
                k += 1
        specs.append(HandSpec(
            num_players=n_players,
            start_credits=[stack_bb * big_blind] * n_players,
            seat_members=members, seed=int(rng.integers(1, 2 ** 31)),
            big_blind=big_blind, small_blind=small_blind,
            raise_sizes=raise_sizes,
            meta={"hero_seat": hero_seat, "stack_bb": stack_bb}))
    return specs


def mixed_hands(n_members, n_players, hands, stack_range, big_blind,
                small_blind, raise_sizes, seed):
    """`hands` specs seating *every* archetype, cycling through the chairs.

    One hand is a data point for every seat at it, so reading all ten stat
    lines off one table costs what reading one costs — and every archetype is
    then measured against the same field over the same hands, which is what
    makes the numbers comparable rather than ten separate experiments.

    **The seating is drawn, not rotated.** Marching the archetypes round the
    table by a fixed step looks like it visits every chair and does not: with
    ten members at a two-handed table the step is two, so the even-indexed
    archetypes never leave the button and the odd ones never leave the big
    blind, and every stat line comes back positional. That is not a subtle bias
    — it made a maniac read tighter than a TAG, because one was always defending
    and the other always opening. A fresh permutation per hand gives every
    archetype every seat and every opponent.
    """
    rng = np.random.default_rng(seed)
    lo, hi = float(stack_range[0]), float(stack_range[1])
    specs = []
    for _h in range(int(hands)):
        stack_bb = float(rng.uniform(lo, hi))
        specs.append(HandSpec(
            num_players=n_players,
            start_credits=[stack_bb * big_blind] * n_players,
            seat_members=[int(m) for m in rng.permutation(
                np.tile(np.arange(n_members),
                        -(-n_players // n_members)))[:n_players]],
            seed=int(rng.integers(1, 2 ** 31)),
            big_blind=big_blind, small_blind=small_blind,
            raise_sizes=raise_sizes, meta={"stack_bb": stack_bb}))
    return specs


#: Hands played between two updates of the progress bar. A cell is thousands of
#: hands and takes minutes; a bar that only moves when a cell ends is
#: indistinguishable from a hung run for most of the job, which is the one
#: thing `CLAUDE.md` §5 asks a bar not to be.
CHUNK = 250


def play_cell(pool, specs, n_actions, batch_size, bar):
    driver = LockstepDriver(pool, n_actions)
    records = []
    for start in range(0, len(specs), CHUNK):
        block = driver.run(specs[start:start + CHUNK], batch_size=batch_size)
        label_showdowns(block)
        records.extend(block)
        bar.update(len(block))
    return records


def _hero(record, seat):
    return seat == record.spec.meta["hero_seat"]


def _member(index):
    return lambda record, seat: record.spec.seat_members[seat] == index


def profile(records, seat_filter=_hero):
    """One player's stat line over these hands, whole and by stack depth."""
    short = [r for r in records if r.spec.meta["stack_bb"] <= SHORT_BB]
    deep = [r for r in records if r.spec.meta["stack_bb"] > SHORT_BB]
    return {"all": hud_stats(records, seat_filter),
            "short": hud_stats(short, seat_filter),
            "deep": hud_stats(deep, seat_filter)}


def winrate(records, big_blind):
    """BB/100 of the hero seat over these hands, and its standard error."""
    per_hand = np.array([float(r.rewards[r.spec.meta["hero_seat"]]) / big_blind
                         for r in records], dtype=np.float64)
    n = len(per_hand)
    se = (float(per_hand.std(ddof=1) / np.sqrt(n) * 100.0) if n > 1 else 0.0)
    return {"hands": n, "bb_per_100": float(per_hand.mean() * 100.0) if n else 0.0,
            "se": se}


def _load(path, settings, log):
    """Whatever a previous run of these settings already computed.

    A cell is thousands of hands and the whole job is twenty-odd minutes; a run
    that is interrupted — or a box that is needed for something else — should
    not throw that away. Settings that differ start over rather than splicing
    two experiments together.
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        stored = json.load(fh)
    if stored.get("settings") != settings:
        log("an earlier report is here for other settings; starting over")
        return None
    report = {"profile": stored.get("profile", {}),
              "strength": stored.get("strength", {}),
              "settings": settings}
    for name in settings["archetypes"]:
        report["profile"].setdefault(name, {})
        report["strength"].setdefault(name, {})
    return report


def _save(path, report, config):
    with open(path, "w") as fh:
        json.dump({**report, "config": config}, fh, indent=2)


def _already_done(report, names, sizes, mixed_per_size, hands):
    """Hands a resumed run does not have to play again."""
    done = 0
    for label in sizes:
        if all(label in report["profile"][name] for name in names):
            done += mixed_per_size[label]
        done += hands * sum(1 for name in names
                            if label in report["strength"][name])
    return done


def run(config, log, out_dir):
    section = config.get("pool_realism", {})
    game = config["game"]
    names = list(section.get("archetypes", ARCHETYPES))
    sizes = {k: SIZES[k] for k in section.get("sizes", list(SIZES))}
    hands = int(section.get("hands", 2000))
    spread = float(section.get("spread", 0.0))
    seed = int(section.get("seed", 20260903))
    batch_size = int(section.get("batch_size", 64))
    stack_range = section.get("stack_bb_range", [10, 300])
    n_actions = int(game["n_actions"])
    big_blind = float(game["big_blind"])
    small_blind = float(game["small_blind"])
    raise_sizes = [game["raise_sizes"][s]
                   for s in ("preflop", "flop", "turn", "river")]

    table = preflop_equity_table(section.get("preflop_table", DEFAULT_TABLE_PATH))
    cache = StrengthCache(int(section.get("max_boards", 4096)))
    members = make_archetypes(names, cache, table, game, seed, spread)
    degenerates = make_degenerates(game)
    pool = members + degenerates
    degenerate_idx = list(range(len(members), len(pool)))

    log(f"pool realism: {len(names)} archetypes × {len(sizes)} table sizes, "
        f"~{hands} hands each, stacks {stack_range[0]}–{stack_range[1]} BB, "
        f"spread {spread}")

    # One mixed table per size answers "are they diverse" for all ten at once;
    # "are they better than the degenerate five" is one table per archetype,
    # because that comparison is what the table *is*.
    mixed_per_size = {label: -(-hands * len(names) // n)
                      for label, n in sizes.items()}
    settings = {"hands": hands, "spread": spread, "seed": seed,
                "stack_bb_range": list(stack_range), "archetypes": names,
                "sizes": list(sizes)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pool_realism.json")
    report = _load(path, settings, log) or {
        "profile": {name: {} for name in names},
        "strength": {name: {} for name in names}, "settings": settings}

    # Written after every cell and resumed from, so an interrupted run keeps
    # what it paid for and a reader can watch the table fill in.
    done = _already_done(report, names, sizes, mixed_per_size, hands)
    if done:
        log(f"resuming: {done} hands are already on disk")
    started = time.perf_counter()
    bar = progress(total=sum(mixed_per_size.values())
                   + len(names) * len(sizes) * hands,
                   initial=done, desc="realism hands", unit="hand")

    for label, n_players in sizes.items():
        if not all(label in report["profile"][name] for name in names):
            table_hands = mixed_per_size[label]
            log(f"[{label}] {table_hands} mixed hands — every archetype at "
                f"one table")
            mixed = play_cell(pool, mixed_hands(
                len(members), n_players, table_hands, stack_range, big_blind,
                small_blind, raise_sizes, seed=seed + 101 * n_players),
                n_actions, batch_size, bar)
            for i, name in enumerate(names):
                report["profile"][name][label] = profile(mixed, _member(i))
            _save(path, report, config)

        for i, name in enumerate(names):
            if label in report["strength"][name]:
                continue
            versus = play_cell(pool, seat_hands(
                i, degenerate_idx, n_players=n_players, hands=hands,
                stack_range=stack_range, big_blind=big_blind,
                small_blind=small_blind, raise_sizes=raise_sizes,
                seed=seed + 13 * i + 307 * n_players),
                n_actions, batch_size, bar)
            report["strength"][name][label] = winrate(versus, big_blind)
            _save(path, report, config)
        # Printed as each size finishes rather than all at the end: a job this
        # long should be readable while it runs.
        format_size(report, label, log)
    bar.close()
    report["wall_seconds"] = time.perf_counter() - started

    log("")
    log(f"wall clock {report['wall_seconds']:.0f} s")
    _save(path, report, config)
    log(f"wrote {path}")
    return report


def ratio(counts):
    """A stat as a frequency, or `nan` where it never had the chance."""
    num, den = counts
    return float(num) / float(den) if den else float("nan")


#: The stats printed side by side. The rest are in the JSON.
HEADLINE = ("vpip", "pfr", "threebet", "cbet_flop", "fold_to_cbet",
            "check_raise", "steal", "overbet_pct", "af", "wtsd")


def format_size(report, label, log):
    """One table size's two tables, printed as soon as that size is done."""
    names = report["settings"]["archetypes"]
    log("")
    log(f"— stat profile, {label} " + "-" * 40)
    log("archetype           " + "".join(f"{s:>13s}" for s in HEADLINE))
    for name in names:
        row = report["profile"][name][label]["all"]
        log(f"{name:<20s}"
            + "".join(f"{ratio(row[s]):>13.3f}" for s in HEADLINE))

    log(f"— against the degenerate five, {label} " + "-" * 24)
    for name in names:
        w = report["strength"][name][label]
        log(f"{name:<20s}{w['bb_per_100']:>10.1f} ± {w['se']:.1f} BB/100 "
            f"over {w['hands']} hands")


def format_report(report, log):
    for label in report["settings"]["sizes"]:
        format_size(report, label, log)
    log("")
    log(f"wall clock {report['wall_seconds']:.0f} s")


def main():
    parser = argparse.ArgumentParser(
        description="PLAN_PROCEDURAL_POOL.md P4 — are the archetypes diverse, "
                    "and better than the degenerate strategies?")
    parser.add_argument("--config", default="config_pr.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    base_dir = config.get("out_dir", "../../data/v8")
    log = Logger(base_dir)
    # Named and not timestamped, so a re-launch resumes the same job instead of
    # starting a second copy of it beside the first.
    out_dir = os.path.join(base_dir, "pool_realism",
                           config.get("pool_realism", {}).get("run", "run0"))
    try:
        run(config, log, out_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
