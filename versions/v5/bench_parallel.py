"""
Benchmark for the parallel inference path. Sweeps `n_workers` ∈ {given list}
for MCTS collection and opponent_data generation, reports wall-clock time
and throughput. Designed to be run on a real GPU machine — locally on Mac/MPS
the absolute numbers are not representative, only correctness can be checked.

Typical use:

    cd versions/v5
    source ../../venv/bin/activate
    MCTS_TIMING=1 python bench_parallel.py \
        --mcts-hands 20 --mcts-sims 500 \
        --opp-hands 100 \
        --workers 1 2 8 16

For an A/B between the optimized (default) and legacy parallel paths, run
twice with `MCTS_PARALLEL_OPT=1` and `MCTS_PARALLEL_OPT=0`. Sequential
(`n_workers=1`) is unaffected by the flag — `LocalEvaluator` is the same
code in both modes.

Outputs a small ASCII table to stdout and, when `MCTS_TIMING=1`, parses the
per-process jsonl files and prints a server bucket-size + per-group latency
summary so you can see whether large cross-actor batches actually formed.
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _build_logger():
    """Minimal stdout logger compatible with `pipeline.py`'s `Logger`."""
    class L:
        init_time = "bench"

        def __call__(self, msg):
            print(msg)

        def run_dir(self, name):
            d = os.path.join("/tmp", f"bench_{name}_{int(time.time())}")
            os.makedirs(d, exist_ok=True)
            return d

    return L()


def _build_agents(config, n_agents, device, log, agents_dir=None):
    """Load N agents from `agents_dir` or build random-init copies.

    Returns the same shape the parallel collectors expect:
      list[{"agent": ASI, "norm_stats": dict, "name": str, "temperature": float}]
    """
    from agent.agent import ASI

    multi = config.get("multi_agent", {})
    names = [a["name"] for a in multi.get("agents", [])][:n_agents]
    if not names:
        names = [f"a{i}" for i in range(n_agents)]
    names = names[:n_agents] or ["a0"]

    fallback_temp = config.get("solver", {}).get("gto_temperature", 1.0)
    identity_norm = {
        "pot_mean": 0.0, "pot_std": 1.0,
        "stack_mean": 0.0, "stack_std": 1.0,
        "bets_mean": 0.0, "bets_std": 1.0,
        "blind_mean": 0.0, "blind_std": 1.0,
        "ev_mean": 0.0, "ev_std": 1.0,
    }

    agents = []
    for name in names:
        a = ASI(log, config)
        a.set_device(device)

        norm = dict(identity_norm)
        temp = fallback_temp

        if agents_dir:
            agent_path = os.path.join(agents_dir, name)
            ckpt = ASI._find_best_checkpoint(agent_path)
            if ckpt:
                a.load_checkpoint(ckpt)
                loaded = torch.load(ckpt, weights_only=False, map_location=device)
                norm = loaded.get("norm_stats", norm) or norm
                temp = loaded.get("temperature", temp)
                log(f"  loaded {name} from {ckpt}")
            else:
                log(f"  no checkpoint for {name}; random init")
        else:
            log(f"  {name}: random init")

        a.eval()
        agents.append({"agent": a, "norm_stats": norm,
                       "name": name, "temperature": temp})
    return agents


def _bench_mcts(config, n_workers, n_hands, n_sims, device, log, agents_dir):
    """Run a single-cycle MCTS collection and return (wall_seconds, n_examples)."""
    from agent.mcts.collect import run_mcts_collection

    cfg = copy.deepcopy(config)
    cfg["mcts"]["n_simulations"] = n_sims
    cfg["mcts_train"]["n_workers"] = n_workers

    n_agents = len(config.get("multi_agent", {}).get("agents", [])) or 1
    log(f"\n--- MCTS bench: n_workers={n_workers}, n_hands={n_hands}, "
        f"n_sims={n_sims}, n_agents={n_agents} ---")

    agents = _build_agents(cfg, n_agents, device, log, agents_dir)

    t0 = time.monotonic()
    per_agent_examples = run_mcts_collection(
        agents, cfg, device, log, n_hands, cycle_idx=0, n_cycles=1)
    t1 = time.monotonic()
    n_examples = sum(len(v) for v in per_agent_examples.values())
    return t1 - t0, n_examples


def _bench_opponent(config, n_workers, n_hands, device, log, agents_dir):
    """Run a single opponent_data generation and return (wall_seconds, n_scenarios)."""
    from agent.train_scenarios.generation.generate_opponent import (
        generate_opponent_dataset)

    cfg = copy.deepcopy(config)
    cfg["opponent_data"]["n_workers"] = n_workers
    cfg["opponent_data"]["n_hands"] = n_hands
    if agents_dir:
        cfg["opponent_data"]["agents_dir"] = agents_dir
    # Always regen — point to a fresh tmp dir so the cache hit doesn't skip work.
    save_dir = f"/tmp/bench_opp_{n_workers}_{int(time.monotonic()*1000)}"
    cfg["opponent_data"]["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    log(f"\n--- Opponent bench: n_workers={n_workers}, n_hands={n_hands} ---")
    t0 = time.monotonic()
    scenarios = generate_opponent_dataset(cfg, save_dir, device, log)
    t1 = time.monotonic()
    return t1 - t0, len(scenarios)


def _aggregate_timing(jsonl_glob):
    """Read all per-process jsonl files and summarize."""
    if not os.environ.get("MCTS_TIMING", "0") == "1":
        return None
    files = glob.glob(jsonl_glob)
    if not files:
        return None
    bucket_sizes = []
    bucket_linger = []
    bucket_hit_max = 0
    group_durations = {}  # rtype -> [ms]
    group_items = {}      # rtype -> [items]
    rpc_waits = {}        # rtype -> [ms]
    rpc_puts = {}
    for p in files:
        try:
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    n = r.get("name")
                    if n == "server_bucket_formed":
                        bucket_sizes.append(r["n_reqs"])
                        bucket_linger.append(r["linger_used_ms"])
                        if r.get("hit_max"):
                            bucket_hit_max += 1
                    elif n == "server_run_group":
                        rtype = r["rtype"]
                        group_durations.setdefault(rtype, []).append(r["duration_ms"])
                        group_items.setdefault(rtype, []).append(r["n_items"])
                    elif n == "actor_rpc":
                        rtype = r["rtype"]
                        rpc_waits.setdefault(rtype, []).append(r["wait_ms"])
                        rpc_puts.setdefault(rtype, []).append(r["put_ms"])
        except Exception:
            pass

    def _stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        n = len(xs)
        return dict(
            n=n,
            mean=sum(xs) / n,
            p50=xs[n // 2],
            p95=xs[int(n * 0.95)],
            min=xs[0],
            max=xs[-1],
        )

    return {
        "server_bucket_size": _stats(bucket_sizes),
        "server_linger_ms": _stats(bucket_linger),
        "server_hit_max_pct": (100.0 * bucket_hit_max / max(1, len(bucket_sizes))),
        "server_group_ms": {k: _stats(v) for k, v in group_durations.items()},
        "server_group_items": {k: _stats(v) for k, v in group_items.items()},
        "actor_rpc_wait_ms": {k: _stats(v) for k, v in rpc_waits.items()},
        "actor_rpc_put_ms": {k: _stats(v) for k, v in rpc_puts.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--agents-dir", default="",
                    help="Path to dir with agent subdirs (loads checkpoints). "
                         "Empty = random-init agents.")
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 8],
                    help="n_workers values to sweep")
    ap.add_argument("--mcts-hands", type=int, default=20)
    ap.add_argument("--mcts-sims", type=int, default=500)
    ap.add_argument("--opp-hands", type=int, default=100)
    ap.add_argument("--skip-mcts", action="store_true")
    ap.add_argument("--skip-opp", action="store_true")
    ap.add_argument("--timing-dir", default="/tmp/mcts_timing_bench",
                    help="If MCTS_TIMING=1 is set, jsonls go here.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log = _build_logger()
    log(f"Device: {device}")
    log(f"MCTS_PARALLEL_OPT={os.environ.get('MCTS_PARALLEL_OPT', '1')}")
    log(f"MCTS_TIMING={os.environ.get('MCTS_TIMING', '0')}")

    if os.environ.get("MCTS_TIMING") == "1":
        os.makedirs(args.timing_dir, exist_ok=True)
        os.environ["MCTS_TIMING_PATH"] = os.path.join(args.timing_dir, "t.jsonl")

    results = {"mcts": [], "opp": []}

    for nw in args.workers:
        if os.environ.get("MCTS_TIMING") == "1":
            # New timing path per run
            run_dir = os.path.join(args.timing_dir, f"workers_{nw}")
            os.makedirs(run_dir, exist_ok=True)
            os.environ["MCTS_TIMING_PATH"] = os.path.join(run_dir, "t.jsonl")

        if not args.skip_mcts:
            wall, n_ex = _bench_mcts(
                config, nw, args.mcts_hands, args.mcts_sims, device, log,
                args.agents_dir or None)
            ex_per_s = n_ex / wall if wall > 0 else 0
            results["mcts"].append((nw, wall, n_ex, ex_per_s))
            log(f"  MCTS n_workers={nw}: {wall:.2f}s, {n_ex} examples, "
                f"{ex_per_s:.1f} ex/s")

        if not args.skip_opp:
            wall, n_sc = _bench_opponent(
                config, nw, args.opp_hands, device, log,
                args.agents_dir or None)
            sc_per_s = n_sc / wall if wall > 0 else 0
            results["opp"].append((nw, wall, n_sc, sc_per_s))
            log(f"  Opp  n_workers={nw}: {wall:.2f}s, {n_sc} scenarios, "
                f"{sc_per_s:.1f} sc/s")

    print("\n" + "=" * 60)
    print("MCTS benchmark")
    print(f"{'n_workers':>10} {'wall_s':>10} {'examples':>10} {'ex/s':>10}")
    for nw, wall, n, eps in results["mcts"]:
        print(f"{nw:>10} {wall:>10.2f} {n:>10} {eps:>10.1f}")

    print("\nOpponent benchmark")
    print(f"{'n_workers':>10} {'wall_s':>10} {'scenarios':>10} {'sc/s':>10}")
    for nw, wall, n, sps in results["opp"]:
        print(f"{nw:>10} {wall:>10.2f} {n:>10} {sps:>10.1f}")

    if os.environ.get("MCTS_TIMING") == "1":
        print("\n" + "=" * 60)
        print("Timing aggregation (per n_workers)")
        for nw in args.workers:
            run_dir = os.path.join(args.timing_dir, f"workers_{nw}")
            agg = _aggregate_timing(os.path.join(run_dir, "*.jsonl"))
            if agg is None:
                continue
            print(f"\n--- n_workers={nw} ---")
            print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
