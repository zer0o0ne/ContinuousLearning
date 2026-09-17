"""Played-policy evaluation, independent of oracle labels, for 2–9 seats.

Each independent unit draws a table and plays every cyclic seating for both
candidates on common decks/action uniforms/auxiliary CV streams. Warm units
are complete sessions; every seating and candidate owns its own history.
The point estimate weights units equally, not individual correlated hands.
Current and fixed bootstrap pools are separate, explicitly uniform benchmarks.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np
import torch

from agent.policy import FrozenAgentMember, logits_for_members
from env.driver import HandSpec, LockstepDriver
from env.runout import RunoutConfig
from env.session import Session, raise_sizes_from
from env.showdown import label_showdowns
from evaluation.identity import atomic_json, check_identity, model_digest
from nets.embedding_net import fit_embeddings, loss_weights
from nets.features import collate
from oracle.rollout import Q_ESTIMATOR
from pool.base import PoolMember
from train.generate import _pad_vectors, amortised_vectors
from utils import progress

VERSION = "paired_pool_sessions_v1"


@dataclass
class Candidate:
    member: PoolMember
    embedding_config: dict


@dataclass
class EvaluationSession(Session):
    rotation: int = 0

    def seat_of_slot(self, slot, hand_idx):
        return (slot - hand_idx - self.rotation) % self.num_players

    def slot_of_seat(self, hand_idx):
        return [(seat + hand_idx + self.rotation) % self.num_players
                for seat in range(self.num_players)]


@dataclass
class Lane:
    session: EvaluationSession
    candidate: int
    tables: dict = field(default_factory=dict)


class RoutedMember(PoolMember):
    """Batch a base policy across tables while retaining each table's vectors."""

    def __init__(self, base, lanes):
        super().__init__(base.name, base.n_actions, base.style)
        self.base, self.lanes = base, lanes
        self.seated = {}

    def policy(self, contexts):
        if not isinstance(self.base, FrozenAgentMember):
            return self.base.policy(contexts)
        return super().policy(contexts)

    def logits(self, contexts):
        seated = []
        for ctx in contexts:
            lane = self.lanes[ctx.record.spec.meta["eval_lane"]]
            slot = lane.session.slot_of_seat(ctx.record.spec.meta["hand"])[ctx.acting_pos]
            table = lane.tables.get(slot)
            key = (lane.session.idx, slot)
            if key not in self.seated:
                self.seated[key] = self.base.with_vectors(
                    self.base.name, table, slot if table is not None else None)
            seated.append(self.seated[key])
        return logits_for_members(seated, contexts)


def sample_summary(values, confidence, target_halfwidth):
    """Normal-approximation CI over independent blocks or complete sessions."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean()) if n else None
    sd = float(values.std(ddof=1)) if n >= 2 else None
    se = 100 * sd / math.sqrt(n) if sd is not None else None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    half = z * se if se is not None else None
    return {"units": n, "bb_per_100": None if mean is None else 100 * mean,
            "sd_bb_per_unit": sd, "stderr_bb_per_100": se,
            "ci_bb_per_100": None if half is None else [100 * mean - half, 100 * mean + half],
            "halfwidth_bb_per_100": half,
            "estimated_units_for_target": None if sd is None else max(
                2, math.ceil((z * 100 * sd / target_halfwidth)**2))}


def summarise(rows, confidence, target_halfwidth):
    out = {"units": len(rows), "hands": sum(r["hands"] for r in rows),
           "seconds": sum(r["seconds"] for r in rows)}
    for kind in ("raw", "cv"):
        if not rows or rows[0][kind] is None:
            out[kind] = None
            continue
        x = np.asarray([r[kind] for r in rows], dtype=float)
        out[kind] = {"new": sample_summary(x[:, 0], confidence, target_halfwidth),
                     "old": sample_summary(x[:, 1], confidence, target_halfwidth),
                     "delta": sample_summary(x[:, 0] - x[:, 1], confidence,
                                             target_halfwidth)}
    return out


def configuration(config):
    cfg = config.get("pool_evaluation", {})
    if not cfg.get("enabled", False):
        return None
    defaults = {"run": "pilot_v1", "benchmarks": ["current", "anchor"],
                "cold_blocks": 1000, "warm_sessions": 10,
                "warm_hands_per_session": 100, "blocks_per_batch": 16,
                "batch_hands": 2048, "control_variate": True,
                "runout_samples": 16, "fit_window": 500,
                "confidence": 0.95, "target_halfwidth_bb100": 10.0,
                "seed": 1729}
    unknown = set(cfg) - set(defaults) - {"enabled"}
    if unknown:
        raise ValueError(f"Unknown pool_evaluation keys: {sorted(unknown)}")
    out = {**defaults, **cfg}
    if not out["benchmarks"] or len(set(out["benchmarks"])) != len(out["benchmarks"]):
        raise ValueError("pool_evaluation.benchmarks must be nonempty and unique")
    if set(out["benchmarks"]) - {"current", "anchor"}:
        raise ValueError("pool_evaluation.benchmarks accepts current and anchor")
    if not out["run"] or Path(out["run"]).name != out["run"] or out["run"] in (".", ".."):
        raise ValueError("pool_evaluation.run must be a directory name")
    for key in ("cold_blocks", "warm_sessions"):
        if int(out[key]) != out[key] or out[key] < 0:
            raise ValueError(f"pool_evaluation.{key} must be a nonnegative integer")
    if not (out["cold_blocks"] or out["warm_sessions"]):
        raise ValueError("pool_evaluation needs cold blocks or warm sessions")
    for key in ("warm_hands_per_session", "blocks_per_batch", "batch_hands",
                "runout_samples", "fit_window"):
        if int(out[key]) != out[key] or out[key] <= 0:
            raise ValueError(f"pool_evaluation.{key} must be a positive integer")
    if (not 0 < out["confidence"] < 1 or not np.isfinite(out["target_halfwidth_bb100"])
            or out["target_halfwidth_bb100"] <= 0):
        raise ValueError("Invalid pool_evaluation confidence or target halfwidth")
    if int(out["seed"]) != out["seed"] or out["seed"] < 0:
        raise ValueError("pool_evaluation.seed must be a nonnegative integer")
    lo, hi = config["game"]["players_range"]
    if not 2 <= lo <= hi <= min(9, config["game"]["max_players"]):
        raise ValueError("pool evaluation supports 2–9 players within max_players")
    return out


def _identity(pool, descriptors, candidates, game, emb_cfg, cfg, iteration, anchor_size):
    cache = {}

    def network(net):
        if net is None:
            return None
        if id(net) not in cache:
            cache[id(net)] = model_digest(net)
        return cache[id(net)]

    def member(m):
        out = {"name": m.name, "type": type(m).__name__,
                "style": m.style.to_list(),
                "policy": network(getattr(m, "net", getattr(m, "agent", None))),
                "embedding": network(getattr(m, "embed_net", None))}
        params = getattr(m, "params", None)
        if params is not None:
            out["params"] = asdict(params) if is_dataclass(params) else dict(params)
        if hasattr(m, "preflop_table"):
            out["preflop_table"] = hashlib.sha256(m.preflop_table.tobytes()).hexdigest()
        if hasattr(m, "raise_sizes"):
            out["raise_sizes"] = m.raise_sizes
        action_map = getattr(m, "action_map", None)
        if action_map is not None:
            out["action_map"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in vars(action_map).items()}
        return out

    # JSON round-trip normalises tuples and numpy scalar values before an
    # identity is compared with its persisted representation.
    return json.loads(json.dumps({"version": VERSION, "q_estimator": Q_ESTIMATOR, "iteration": int(iteration),
            "game": game, "anchor_size": anchor_size,
            "sampling": "uniform tables/stacks, opponents without replacement",
            "settings": {k: v for k, v in cfg.items()
                         if k not in ("blocks_per_batch", "batch_hands")},
            "embedding_config": emb_cfg,
            "pool": [{"descriptor": d, **member(m)} for m, d in zip(pool, descriptors)],
            "candidates": [{**member(c.member), "embedding_config": c.embedding_config}
                           for c in candidates]}, default=lambda x:
                           x.tolist() if isinstance(x, np.ndarray) else x.item()))


def _rng(cfg, iteration, benchmark, mode, block):
    name = hashlib.sha256(f"{cfg['run']}:{benchmark}:{mode}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(
        [int(cfg["seed"]), int(iteration), int(block), int.from_bytes(name[:8], "big")]))


def _plan(cfg, game, iteration, benchmark, mode, count, member_ids):
    units = []
    hands = 1 if mode == "cold" else int(cfg["warm_hands_per_session"])
    for block in range(count):
        rng = _rng(cfg, iteration, benchmark, mode, block)
        n = int(rng.integers(game["players_range"][0], game["players_range"][1] + 1))
        stack = int(rng.integers(game["stack_bb_range"][0], game["stack_bb_range"][1] + 1))
        opponents = [int(x) for x in rng.choice(member_ids, n - 1, replace=False)]
        # Three separate streams; no seed is derived from realised game cards.
        streams = np.random.SeedSequence(rng.integers(0, 2**32, 4).tolist()).spawn(3)
        cards, actions, auxiliary = [np.random.default_rng(s) for s in streams]
        units.append({"block": block, "players": n, "stack_bb": stack,
                      "opponents": opponents, "length": hands,
                      "decks": [cards.permutation(52) for _ in range(hands)],
                      "actions": actions.integers(0, 2**63, size=hands),
                      "auxiliary": auxiliary.integers(0, 2**63, size=hands)})
    return units


def _fit(lane, pool, candidates, config, cfg, hand):
    session = lane.session
    game, common = config["game"], config["embedding_net"]
    candidate = candidates[lane.candidate]
    max_players, n_actions = int(game["max_players"]), int(game["n_actions"])
    for slot, member_idx in enumerate(session.members):
        member = candidate.member if slot == 0 else pool[member_idx]
        if not isinstance(member, FrozenAgentMember):
            continue
        net = member.embed_net
        emb_cfg = candidate.embedding_config if slot == 0 else common
        if hand == 0 or hand % int(emb_cfg["R"]):
            continue
        if slot and common.get("pool_agent_vectors", "zero") != "amortised":
            continue
        if net is None:
            if slot == 0:
                raise ValueError("Warm evaluation of an agent needs its generation's embedding network")
            continue
        device = next(net.parameters()).device
        if slot:
            lane.tables[slot] = amortised_vectors(
                net, session, slot, max_players, n_actions,
                common.get("pool_agent_window"), device)
        else:
            tokens = [t for t in session.tokens(max_players, n_actions,
                                                window=int(cfg["fit_window"])) if len(t)]
            if tokens:
                batch = collate(tokens, device=device)
                vectors = fit_embeddings(
                    net, batch, session.num_players, steps=int(emb_cfg["K"]),
                    lr=emb_cfg["fit_lr"], reg=emb_cfg["fit_reg"],
                    weights=loss_weights(emb_cfg))
                lane.tables[0] = _pad_vectors(vectors.cpu().numpy(), max_players, net.d_emb)


def _play_units(units, pool, candidates, config, cfg, mode, bar):
    game = config["game"]
    lanes = []
    for unit in units:
        for candidate in range(2):
            for rotation in range(unit["players"]):
                session = EvaluationSession(
                    idx=len(lanes), num_players=unit["players"], stack_bb=unit["stack_bb"],
                    members=[-1] + unit["opponents"], rotation=rotation)
                lane = Lane(session, candidate)
                base = candidates[candidate].member
                if isinstance(base, FrozenAgentMember):
                    lane.tables[0] = np.zeros((game["max_players"], base.net.d_emb), np.float32)
                lanes.append(lane)
                for h in range(unit["length"]):
                    slots = session.slot_of_seat(h)
                    session.specs.append(HandSpec(
                        num_players=session.num_players,
                        start_credits=[float(session.stack_bb * game["big_blind"])] * session.num_players,
                        seat_members=[len(pool) + candidate if slot == 0
                                      else session.members[slot] for slot in slots],
                        seed=int(unit["actions"][h]), big_blind=game["big_blind"],
                        small_blind=game["small_blind"], raise_sizes=raise_sizes_from(game),
                        deck=unit["decks"][h], runout_seed=int(unit["auxiliary"][h]),
                        meta={"eval_lane": session.idx, "hand": h}))
    bases = list(pool) + [c.member for c in candidates]
    driver = LockstepDriver([RoutedMember(m, lanes) for m in bases], game["n_actions"],
                            runout=RunoutConfig(cfg["runout_samples"])
                            if cfg["control_variate"] else None)
    length = units[0]["length"]
    at = 0
    while at < length:
        if mode == "warm":
            if at:
                bar.set_postfix_str(f"fit: hand {at}", refresh=True)
                for lane in lanes:
                    _fit(lane, pool, candidates, config, cfg, at)
                for router in driver.pool:
                    router.seated.clear()
            periods = [int(c.embedding_config["R"]) for c in candidates]
            periods.append(int(config["embedding_net"]["R"]))
            end = min([length] + [(at // r + 1) * r for r in periods])
        else:
            end = length
        specs = [s for lane in lanes for s in lane.session.specs[at:end]]
        records = driver.run(specs, batch_size=int(cfg["batch_hands"]))
        if any(r.truncated for r in records):
            raise RuntimeError("Pool evaluation hit the action cap; no truncated result was recorded")
        if mode == "warm":
            label_showdowns(records)
        for i, lane in enumerate(lanes):
            lane.session.records.extend(records[i * (end-at):(i+1) * (end-at)])
        bar.update(len(records))
        bar.set_postfix_str("", refresh=False)
        at = end
    out, offset = [], 0
    for unit in units:
        n = unit["players"]
        values = {"raw": [], "cv": []}
        for candidate in range(2):
            candidate_lanes = lanes[offset + candidate*n:offset + (candidate+1)*n]
            for key, attr in (("raw", "rewards"), ("cv", "baseline_rewards")):
                if key == "cv" and not cfg["control_variate"]:
                    continue
                values[key].append(float(np.mean([
                    getattr(record, attr)[lane.session.seat_of_slot(0, h)] / record.spec.big_blind
                    for lane in candidate_lanes for h, record in enumerate(lane.session.records)])))
        out.append({"block": unit["block"], "players": n, "stack_bb": unit["stack_bb"],
                    "opponents": unit["opponents"], "hands": 2*n*unit["length"],
                    "raw": values["raw"], "cv": values["cv"] or None})
        offset += 2*n
    return out


def evaluate(pool, descriptors, candidates, config, out_dir, iteration, anchor_size, log=print):
    started = time.perf_counter()
    cfg = configuration(config)
    if cfg is None:
        return None
    if len(candidates) != 2 or len(pool) != len(descriptors):
        raise ValueError("Pool evaluation needs two candidates and one descriptor per member")
    if cfg["warm_sessions"]:
        for candidate in candidates:
            if isinstance(candidate.member, FrozenAgentMember) and candidate.member.embed_net is None:
                raise ValueError("Warm evaluation requires each candidate's embedding generation")
    max_opponents = int(config["game"]["players_range"][1]) - 1
    if len(pool) < max_opponents or ("anchor" in cfg["benchmarks"] and anchor_size < max_opponents):
        raise ValueError("Evaluation pool is too small for the configured table sizes")
    out_dir = Path(out_dir) / cfg["run"]
    identity = _identity(pool, descriptors, candidates, config["game"],
                         config["embedding_net"], cfg, iteration, anchor_size)
    check_identity(out_dir, identity)
    modes = [("cold", int(cfg["cold_blocks"])), ("warm", int(cfg["warm_sessions"]))]
    jobs = []
    for benchmark in cfg["benchmarks"]:
        ids = list(range(anchor_size if benchmark == "anchor" else len(pool)))
        for mode, count in modes:
            if count:
                units = _plan(cfg, config["game"], iteration, benchmark, mode, count, ids)
                jobs.append((benchmark, mode, units))
    total = sum(2*u["players"]*u["length"] for _, _, us in jobs for u in us)
    bar = progress(total=total, desc="pool evaluation", unit="hand")
    report = {"version": VERSION, "run": cfg["run"], "iteration": iteration,
              "candidates": [c.member.name for c in candidates],
              "confidence": cfg["confidence"], "interval": "normal approximation over independent units",
              "target_halfwidth_bb100": cfg["target_halfwidth_bb100"],
              "note": "Fixed-budget evaluation. Size recommendations are for a fresh independent run.",
              "benchmarks": {}}
    networks = {id(net): net for m in list(pool) + [c.member for c in candidates]
                for net in (getattr(m, "net", getattr(m, "agent", None)),
                            getattr(m, "embed_net", None)) if net is not None}
    states = {key: net.training for key, net in networks.items()}
    for net in networks.values():
        net.eval()
    try:
        for benchmark, mode, units in jobs:
            directory = out_dir / benchmark / mode
            directory.mkdir(parents=True, exist_ok=True)
            rows, pending = [], []
            for unit in units:
                path = directory / f"{unit['block']:08d}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    rows.append(row)
                    bar.update(row["hands"])
                else:
                    pending.append(unit)
            width = int(cfg["blocks_per_batch"])
            for lo in range(0, len(pending), width):
                chunk = pending[lo:lo+width]
                t0 = time.perf_counter()
                completed = _play_units(chunk, pool, candidates, config, cfg, mode, bar)
                elapsed = time.perf_counter() - t0
                n_hands = sum(r["hands"] for r in completed)
                for row in completed:
                    row["seconds"] = elapsed * row["hands"] / n_hands
                    atomic_json(directory / f"{row['block']:08d}.json", row)
                    rows.append(row)
            rows.sort(key=lambda r: r["block"])
            summary = summarise(rows, cfg["confidence"], cfg["target_halfwidth_bb100"])
            summary["unit"] = "complete_session" if mode == "warm" else "duplicate_block"
            summary["by_players"] = {str(n): summarise([r for r in rows if r["players"] == n],
                                                      cfg["confidence"], cfg["target_halfwidth_bb100"])
                                     for n in sorted({r["players"] for r in rows})}
            summary["by_opponent"] = {
                str(i): {"name": pool[i].name, **summarise([r for r in rows if i in r["opponents"]],
                         cfg["confidence"], cfg["target_halfwidth_bb100"])}
                for i in sorted({i for r in rows for i in r["opponents"]})}
            summary["opponent_note"] = "Hero return conditional on this opponent being seated; multiway rows overlap."
            report["benchmarks"].setdefault(benchmark, {})[mode] = summary
            chosen = summary["cv"] if summary["cv"] is not None else summary["raw"]
            delta = chosen["delta"]
            log(f"[pool-eval] {benchmark}/{mode}: {summary['hands']} hands, "
                f"{summary['units']} independent units; new={chosen['new']['bb_per_100']:+.2f}, "
                f"old={chosen['old']['bb_per_100']:+.2f}, delta={delta['bb_per_100']:+.2f} "
                f"BB/100, SE={delta['stderr_bb_per_100']}; {summary['seconds']:.1f}s")
    finally:
        bar.close()
        for key, net in networks.items():
            net.train(states[key])
    report["invocation_seconds"] = time.perf_counter() - started
    report["total_hands"] = total
    report["play_seconds"] = sum(s["seconds"] for modes in report["benchmarks"].values() for s in modes.values())
    report["hands_per_second"] = total/report["play_seconds"] if report["play_seconds"] else None
    atomic_json(out_dir / "report.json", report)
    return report
