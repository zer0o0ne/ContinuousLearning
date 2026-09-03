"""Is a past agent stronger reading its tablemates? (PLAN_AMORTISED_POOL.md P2)

`PLAN_PIPELINE.md` D12 settled that a past agent, seated in the pool as an
*opponent*, plays at `e = 0`: it reads a zero vector for every tablemate and so
plays the unconditional policy §6.2's embedding dropout trains. P1 gave the
member the option of reading the `K = 0` output of the amortised head over the
hands it has seen from **its own** seat instead. This gate is the measurement
that decides whether that option is worth wiring into the phases that seat it.

**The question is narrow.** Not "is the agent good" — *is the same network
stronger at the same table when it conditions on its tablemates than when it
does not*. Nothing else here is a result about the agent, which is why the
report carries a stamp saying so and lives nowhere near `evaluation/`.

**Why not measure it anywhere else.** Slumbot cannot answer it: benchmark
numbers must not feed back into training (`CLAUDE.md` §1), and a past agent
never sits at that table anyway. The labels phase can answer it, but an
iteration of it is measured in days (`CONCEPT.md` §13) and the answer would
arrive after the plumbing it was supposed to justify. A paired self-play
comparison over identical hands answers it in minutes.

**Paired, and that is the whole design.** Both conditions play the *same*
sessions: same seeds, same tables, same opponents, same seating, same button
rotation. Only what the agent reads for its tablemates differs. The hands
themselves diverge as soon as the two conditions choose different actions —
that is the effect being measured — but the *situations* they are dealt into
are identical, so the difference is taken per session and its standard error is
over sessions. A single 400-hand session of NLHE has a standard error of tens of
BB/100; the paired difference is what makes a small effect legible at all.

**The decision rule is pre-registered** in `PLAN_AMORTISED_POOL.md` §P2 and is
printed beside the number rather than encoded here: ≥ 2 SE above zero with no
table-size or stack bucket more than 2 SE below → build P3; within ±2 SE → the
conditioning is not worth its plumbing and D12 stays at option (a); clearly
negative → a result to report about the agent's `e ≠ 0` input distribution.

Run::

    cd versions/v8 && python3 -m gates.pool_conditioning --config config_pc.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from agent.policy import FrozenAgentMember
from env.driver import LockstepDriver
from env.session import Session, build_sessions, play
from gates.g1 import _stack_bucket, _stats
from nets.embedding_net import OpponentEmbeddingNet
from pipeline import frozen_agent_net
from pool.build import build_pool
from pool.style import StyleParams
from train.generate import amortised_vectors
from utils import Logger, progress, resolve_device

TAG = "pc"
HERO_SLOT = 0
STAMP = "POOL MEMBER CONDITIONING — not an agent result"
CONDITIONS = ("zero", "amortised")

# The architecture keys both networks are built from. A checkpoint trained under
# a different value of any of them does not load into the network this gate
# builds, and one trained on a different action set or raise grid would play a
# different game against this pool — so they are compared, not assumed.
ARCH_KEYS = ("d_model", "d_emb", "n_heads", "n_kv_heads", "n_layers", "d_ff",
             "d_card", "d_index", "max_decisions", "range_enabled",
             "range_layer", "n_range_blocks")
GAME_KEYS = ("n_actions", "max_players", "big_blind", "small_blind",
             "raise_sizes")


# ----------------------------------------------------------------- checkpoints


def load_checkpoints(config, game, device, log):
    """The agent and embedding networks of one pipeline run, frozen.

    Both files carry the config they were trained under, so the mismatch that
    would otherwise be silent — a checkpoint from a run with a different raise
    grid, or a different trunk — is an assertion here rather than a number
    nobody can interpret later.
    """
    pc = config["pool_conditioning"]
    agent_ckpt = torch.load(pc["agent_checkpoint"], map_location=device,
                            weights_only=False)
    embed_ckpt = torch.load(pc["embedding_checkpoint"], map_location=device,
                            weights_only=False)
    for what, ckpt in (("agent", agent_ckpt), ("embedding", embed_ckpt)):
        trained = ckpt.get("config", {})
        for key in GAME_KEYS:
            assert trained["game"][key] == game[key], (
                f"the {what} checkpoint was trained with game.{key} = "
                f"{trained['game'][key]!r} and this gate is configured for "
                f"{game[key]!r}; it would be playing a different game")
        for key in ARCH_KEYS:
            assert (trained["embedding_net"].get(key)
                    == config["embedding_net"].get(key)), (
                f"the {what} checkpoint was trained with embedding_net.{key} = "
                f"{trained['embedding_net'].get(key)!r}, this gate is "
                f"configured for {config['embedding_net'].get(key)!r}")
        log(f"[{TAG}] {what} checkpoint from iteration "
            f"{ckpt.get('iteration')} of experiment "
            f"{trained.get('experiment')!r}")

    agent_net = frozen_agent_net(agent_ckpt["model_state_dict"], config, game,
                                 device)
    # The member table's height is the checkpoint's, not this gate's business:
    # nothing here reads a trained row (§5.5's fit is the point), but the state
    # dict does not load into a network of a different height.
    state = embed_ckpt["model_state_dict"]
    embed_net = OpponentEmbeddingNet(
        config["embedding_net"], int(game["n_actions"]),
        int(game["max_players"]),
        n_members=int(state["embeddings.weight"].shape[0])).to(device)
    embed_net.load_state_dict(state)
    embed_net.eval()
    for p in embed_net.parameters():
        p.requires_grad_(False)
    return agent_net, embed_net


# -------------------------------------------------------------------- sessions


def build_tables(config, game, pool_size, n_sessions, hands_per_session, seed):
    """The tables both conditions play, with the agent at slot 0.

    `build_sessions` draws a member for every slot including slot 0 and that
    draw is dropped, exactly as `train/generate.py` drops it: the table
    *configuration* is what is being reused, and keeping the draw leaves the
    uniform 2–9 × 10–300 BB stream identical to G1's (D4). Slot 0 is the agent
    and is not a pool index at all, so `members[0]` is `-1`.
    """
    rng = np.random.default_rng(seed)
    sessions = build_sessions(
        rng, list(range(pool_size)), game, n_sessions, hands_per_session,
        seed_base=int(config["pool_conditioning"]["seed_base"]), tag=TAG)
    for s in sessions:
        s.members = [-1] + [int(m) for m in s.members[1:]]
    return sessions


def seat_agent(sessions, agent_idx):
    """Point every hand's slot-0 seat at that session's reserved pool entry."""
    for s, idx in zip(sessions, agent_idx):
        for h, spec in enumerate(s.specs):
            sos = s.slot_of_seat(h)
            spec.seat_members = [
                idx if sos[seat] == HERO_SLOT else s.members[sos[seat]]
                for seat in range(s.num_players)]


# ----------------------------------------------------------------- the two runs


def play_condition(condition, driver, play_pool, sessions, agent_idx, base,
                   embed_net, config, game, device, log, bar):
    """Play every session in blocks of `R`, reseating the agent between blocks.

    The block discipline is hero's (`ARCHITECTURE.md` §2.10): the vectors the
    agent acts under in block *b* are computed over the hands of blocks
    `0 … b−1` and over nothing else, and block 0 is the cold start. Under
    `"zero"` no table is ever installed, which is D12 option (a) as it stands
    today; under `"amortised"` the table is the `K = 0` output of the amortised
    head over the agent's own view of the hands it has played so far, at most
    `pool_agent_window` of them.

    Block 0 seats the same member in both conditions on purpose: a zero table
    and no table are bit-identical (`tests/test_frozen_agent_vectors.py`), so
    the cold start is one code path rather than two that have to be kept in
    step.
    """
    emb_cfg = config["embedding_net"]
    R = int(emb_cfg["R"])
    window = emb_cfg.get("pool_agent_window")
    max_players, n_actions = int(game["max_players"]), int(game["n_actions"])
    hands = len(sessions[0].specs)
    n_blocks = (hands + R - 1) // R
    refresh = {"refreshes": 0, "window_hands": 0, "refresh_seconds": 0.0}

    for s in sessions:
        s.records = []
    for b in range(n_blocks):
        lo, hi = b * R, min((b + 1) * R, hands)
        blocks = []
        for s, idx in zip(sessions, agent_idx):
            member = base
            if condition == "amortised" and b:
                t0 = time.perf_counter()
                vectors = amortised_vectors(embed_net, s, HERO_SLOT,
                                            max_players, n_actions, window,
                                            device)
                refresh["refresh_seconds"] += time.perf_counter() - t0
                refresh["refreshes"] += 1
                refresh["window_hands"] += (len(s.records) if window is None
                                            else min(len(s.records),
                                                     int(window)))
                member = base.with_vectors(f"{base.name}@s{s.idx}b{b}",
                                           vectors, HERO_SLOT)
            play_pool[idx] = member
            blocks.append(Session(idx=s.idx, num_players=s.num_players,
                                  stack_bb=s.stack_bb, members=s.members,
                                  specs=s.specs[lo:hi]))
        play(driver, blocks, int(config["driver_batch_size"]), log,
             f"{TAG}:{condition}:block{b}", bar=False)
        for s, blk in zip(sessions, blocks):
            s.records.extend(blk.records)
        bar.update(sum(len(blk.specs) for blk in blocks))
    return refresh


def session_results(sessions):
    """The agent's BB/100 in each session, and the table it played it at.

    A hand's result is the agent's own chip delta over the big blind — the same
    accounting `train/generate.py::_results_by_member` does for the sampler, and
    the only accounting there is.
    """
    out = []
    for s in sessions:
        bb = 0.0
        for h, record in enumerate(s.records):
            seat = s.seat_of_slot(HERO_SLOT, h)
            bb += float(record.rewards[seat]) / float(record.spec.big_blind)
        out.append({"session": int(s.idx), "num_players": int(s.num_players),
                    "stack_bb": int(s.stack_bb), "hands": len(s.records),
                    "bb": bb,
                    "bb_per_100": 100.0 * bb / len(s.records)
                    if s.records else float("nan")})
    return out


# ------------------------------------------------------------------ the report


def _paired(rows):
    """Per-condition means and the paired difference, all with their SE."""
    out = {c: _stats([r[f"bb_per_100_{c}"] for r in rows]) for c in CONDITIONS}
    out["difference"] = _stats([r["difference"] for r in rows])
    return out


def aggregate(rows, timings):
    """Everything P2 asks to be reported, from the same per-session rows."""
    report = {"stamp": STAMP,
              "n_sessions": len(rows),
              "hands_per_condition": sum(r["hands"] for r in rows),
              "overall": _paired(rows),
              "by_table_size": {}, "by_stack_depth": {},
              "timings": timings}
    for size in sorted({r["num_players"] for r in rows}):
        group = [r for r in rows if r["num_players"] == size]
        report["by_table_size"][str(size)] = _paired(group)
    for bucket in sorted({_stack_bucket(r["stack_bb"]) for r in rows}):
        group = [r for r in rows if _stack_bucket(r["stack_bb"]) == bucket]
        report["by_stack_depth"][bucket] = _paired(group)
    return report


def format_report(report, log):
    def line(name, cell):
        return (f"  {name:<14} n={cell['difference']['n']:<4} "
                f"zero {cell['zero']['mean']:+8.2f} ± {cell['zero']['se']:5.2f}"
                f"   amortised {cell['amortised']['mean']:+8.2f} ± "
                f"{cell['amortised']['se']:5.2f}"
                f"   difference {cell['difference']['mean']:+8.2f} ± "
                f"{cell['difference']['se']:5.2f}")

    log("")
    log(f"=== {STAMP} ===")
    log(f"the same checkpoint at slot 0, {report['n_sessions']} paired "
        f"sessions, {report['hands_per_condition']} hands per condition; "
        f"BB/100, mean ± SE over sessions")
    log(line("overall", report["overall"]))
    log("  by table size:")
    for size, cell in report["by_table_size"].items():
        log(line(f"    {size}-handed", cell))
    log("  by stack depth:")
    for bucket, cell in report["by_stack_depth"].items():
        log(line(f"    {bucket} BB", cell))

    diff = report["overall"]["difference"]
    se = diff["se"]
    ratio = (diff["mean"] / se) if se and not math.isnan(se) and se > 0 \
        else float("nan")
    log(f"  paired difference is {ratio:+.2f} SE from zero")
    log("  pre-registered rule (PLAN_AMORTISED_POOL.md §P2): ≥ +2 SE overall "
        "and no bucket below −2 SE → build P3; within ±2 SE → the conditioning "
        "is not worth its plumbing and D12 stays at option (a); clearly "
        "negative → a result to report.")
    t = report["timings"]
    log(f"  timings on this box: hand_tokens {t['hand_tokens_us']:.0f} µs per "
        f"call; amortised_vectors {t['amortised_us_per_window_hand']:.0f} µs "
        f"per hand of window over {t['refreshes']} refreshes "
        f"({t['refresh_seconds']:.1f} s total, play "
        f"{t['play_seconds']:.1f} s)")
    log(f"=== {STAMP} ===")
    log("")


def write_report(path, report):
    """Refuse to write a report that could be read as an agent result.

    The stamp and the standard errors are the two things that stop this number
    from being quoted as "the agent's BB/100": the first says what it is, the
    second says how much of it is noise (`CONCEPT.md` §12 — never one without
    the other).
    """
    assert report.get("stamp") == STAMP, (
        f"a pool-conditioning report must carry the stamp {STAMP!r}, or it "
        f"will be read as an agent result")
    for name, cell in [("overall", report["overall"])] + \
            list(report["by_table_size"].items()) + \
            list(report["by_stack_depth"].items()):
        for condition, stats in cell.items():
            assert "se" in stats, (
                f"{name}/{condition} carries no standard error; a BB/100 "
                f"without one says nothing (§12)")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    return path


# ----------------------------------------------------------------------- run


def run(config, log, out_dir):
    game = config["game"]
    pc = config["pool_conditioning"]
    seed = int(config.get("seed", 0))
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")

    rng = np.random.default_rng(seed)
    pool, descriptors = build_pool(config, rng, device=device, log=log)
    log(f"[{TAG}] pool: {len(pool)} members "
        f"({len({d['base'] for d in descriptors})} bases), the agent's "
        f"opponents at every table")

    agent_net, embed_net = load_checkpoints(config, game, device, log)
    base = FrozenAgentMember(pc.get("agent_name", "agent"),
                             int(game["n_actions"]), agent_net,
                             int(game["max_players"]), device,
                             StyleParams.identity())

    n_sessions = int(pc["n_sessions"])
    hands_per_session = int(pc["hands_per_session"])
    sessions = build_tables(config, game, len(pool), n_sessions,
                            hands_per_session, seed)
    # One reserved entry per session, not per seat: a member that knows its own
    # slot derives the rotation from the seat it is asked to act at (P1), so one
    # member serves every seat of its table.
    play_pool = list(pool) + [None] * len(sessions)
    agent_idx = [len(pool) + i for i in range(len(sessions))]
    seat_agent(sessions, agent_idx)
    driver = LockstepDriver(play_pool, int(game["n_actions"]))

    window = config["embedding_net"].get("pool_agent_window")
    log(f"[{TAG}] {n_sessions} sessions × {hands_per_session} hands, refreshed "
        f"every {config['embedding_net']['R']} hands over a "
        f"{'whole-session' if window is None else str(window) + '-hand'} "
        f"window, in both conditions")

    # One bar over both conditions — the job is the pair, not either half.
    bar = progress(total=2 * n_sessions * hands_per_session, desc="pc:play",
                   unit="hand")
    results, timings = {}, {"play_seconds": 0.0, "refreshes": 0,
                            "refresh_seconds": 0.0, "window_hands": 0}
    for condition in CONDITIONS:
        t0 = time.perf_counter()
        refresh = play_condition(condition, driver, play_pool, sessions,
                                 agent_idx, base, embed_net, config, game,
                                 device, log, bar)
        timings["play_seconds"] += time.perf_counter() - t0
        for key, value in refresh.items():
            timings[key] += value
        results[condition] = session_results(sessions)
    bar.close()

    timings["amortised_us_per_window_hand"] = (
        1e6 * timings["refresh_seconds"] / timings["window_hands"]
        if timings["window_hands"] else float("nan"))
    timings["hand_tokens_us"] = _tokenisation_cost(sessions[0], game)

    rows = []
    for zero, amortised in zip(results["zero"], results["amortised"]):
        assert zero["session"] == amortised["session"]
        rows.append({
            "session": zero["session"], "num_players": zero["num_players"],
            "stack_bb": zero["stack_bb"], "hands": zero["hands"],
            "bb_per_100_zero": zero["bb_per_100"],
            "bb_per_100_amortised": amortised["bb_per_100"],
            "difference": amortised["bb_per_100"] - zero["bb_per_100"]})

    report = aggregate(rows, timings)
    format_report(report, log)

    os.makedirs(out_dir, exist_ok=True)
    payload = {**report, "config": config, "pool": descriptors, "rows": rows}
    path = write_report(os.path.join(out_dir, "pool_conditioning.json"),
                        payload)
    log(f"wrote {path}")
    return report


def _tokenisation_cost(session, game):
    """µs per `hand_tokens` call on this box — the number §1.1 needs.

    Measured on the tokenisation the refresh actually performs, from slot 0's
    view over the whole session, and divided by the hands in it.
    """
    if not session.records:
        return float("nan")
    t0 = time.perf_counter()
    tokens = session.tokens(int(game["max_players"]), int(game["n_actions"]),
                            observer_slot=HERO_SLOT)
    return 1e6 * (time.perf_counter() - t0) / len(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="PLAN_AMORTISED_POOL.md P2 — pool member conditioning")
    parser.add_argument("--config", default="config_pc.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    base_dir = config.get("out_dir", "../../data/v8")
    log = Logger(base_dir)
    out_dir = log.run_dir("pool_conditioning")
    try:
        run(config, log, out_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
