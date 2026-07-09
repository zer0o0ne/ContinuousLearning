"""Profile MCTS data collection to find bottlenecks.

Usage (from versions/v6):
    python analytics/profile_mcts_collect.py [--n_hands 3] [--device cuda]

Profiles the sequential (n_workers=1) collection path: MCTS search per
decision, post-hand terminal evaluation, equity outcome computation.
Reduces n_simulations to 200 by default for reasonable profiling time
(override with --n_simulations).

Prints cProfile breakdown + key-function summary.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import pstats
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hands", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--n_simulations", type=int, default=None,
                        help="Override mcts.n_simulations (default: use config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.n_simulations is not None:
        cfg["mcts"]["n_simulations"] = args.n_simulations
        print(f"n_simulations override: {args.n_simulations}")

    n_sims = cfg["mcts"]["n_simulations"]
    batch_sz = cfg["mcts"].get("batch_size", 16)
    print(f"Config: n_simulations={n_sims}, batch_size={batch_sz}, "
          f"max_players={cfg['mcts_train'].get('max_players', 6)}")

    from agent.agent import ASI
    from agent.mcts.collect import run_mcts_collection

    agent = ASI(lambda m: None, config=cfg)
    agent.set_device(device)
    agent.eval()

    norm_stats = {
        "pot_mean": 50.0, "pot_std": 30.0,
        "stack_mean": 500.0, "stack_std": 200.0,
        "bets_mean": 5.0, "bets_std": 10.0,
        "blind_mean": 10.0, "blind_std": 1.0,
        "ev_mean": 0.0, "ev_std": 1.0,
    }

    agents_list = [{
        "agent": agent, "name": "profiled",
        "norm_stats": norm_stats, "temperature": 1.0,
    }]

    log_lines = []
    def log(msg):
        log_lines.append(msg)

    n = args.n_hands

    # Warm up (1 hand)
    print("Warming up...")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_mcts_collection(agents_list, cfg, device, log, n_hands=1)
    if device == "cuda":
        torch.cuda.synchronize()
    log_lines.clear()

    # Profiled run
    print(f"Profiling {n} hands...")
    random.seed(123); np.random.seed(123); torch.manual_seed(123)

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()

    result = run_mcts_collection(agents_list, cfg, device, log, n_hands=n)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    pr.disable()

    total_examples = sum(len(v) for v in result.values())
    print(f"\n{'='*60}")
    print(f"{n} hands in {elapsed:.2f}s  ({elapsed/n:.2f}s per hand)")
    print(f"Total training examples: {total_examples}")
    print(f"{'='*60}")

    for line in log_lines:
        print(f"  LOG: {line}")

    stats = pstats.Stats(pr)

    stats.sort_stats("cumulative")
    print("\n--- Top 40 by CUMULATIVE time ---")
    stats.print_stats(40)

    stats.sort_stats("tottime")
    print("\n--- Top 40 by TOTAL (self) time ---")
    stats.print_stats(40)

    print("\n--- Key functions ---")
    for fn_name in [
        "run_mcts_collection",
        "_play_hands",
        "search",
        "_select_to_leaf",
        "_flush_pending",
        "_evaluate_root",
        "_build_context",
        "_pad_and_stack",
        "evaluate_leaves",
        "evaluate_root",
        "forward_batch",
        "_rebuild_events",
        "_normalize_events_inplace",
        "evaluate_all_terminals",
        "compute_equity_outcome",
        "gpu_equity_v2",
        "_finalize_value_targets",
        "collect_training_data",
        "_expand_node",
        "_backup_cached_terminal",
        "_deterministic_terminal_value",
    ]:
        stats.print_stats(fn_name)


if __name__ == "__main__":
    main()
