"""Profile opponent-data generation to find the real bottleneck.

Usage (from versions/v6):
    python analytics/profile_opponent_gen.py [--n_hands 5] [--device cuda]

Prints per-function timing breakdown (cProfile) and a per-section wall-clock
summary so you can see exactly where the 4s/hand goes.
"""

from __future__ import annotations

import argparse
import cProfile
import json
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
    parser.add_argument("--n_hands", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from agent.agent import ASI
    from agent.train_scenarios.generation.generate_opponent import (
        generate_opponent_hand,
    )
    from utils import get_amp_config

    agent = ASI(lambda m: None, config=cfg)
    agent.set_device(device)
    agent.eval()

    norm_stats = {
        "pot_mean": 50.0, "pot_std": 30.0,
        "stack_mean": 500.0, "stack_std": 200.0,
        "bets_mean": 5.0, "bets_std": 10.0,
        "blind_mean": 10.0, "blind_std": 1.0,
    }

    agents_list = [{
        "agent": agent, "name": "profiled",
        "norm_stats": norm_stats, "temperature": 0.3,
    }]

    game_cfg = cfg["game"]
    opp_cfg = cfg.get("opponent_data", {})
    merged = {**game_cfg, **opp_cfg}
    amp_config = get_amp_config(device)[:3]

    # Warm up (1 hand — JIT, CUDA kernels, etc.)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    generate_opponent_hand(merged, agents_list, device, amp_config)
    if device == "cuda":
        torch.cuda.synchronize()

    # ---- Profiled run ----
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    n = args.n_hands

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()

    scenario_counts = []
    for _ in range(n):
        r = generate_opponent_hand(merged, agents_list, device, amp_config)
        scenario_counts.append(len(r) if r else 0)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    pr.disable()

    print(f"\n{'='*60}")
    print(f"{n} hands in {elapsed:.2f}s  ({elapsed/n:.2f}s per hand)")
    print(f"Scenarios per hand: {scenario_counts}")
    print(f"{'='*60}")

    stats = pstats.Stats(pr)

    stats.sort_stats("cumulative")
    print("\n--- Top 30 by CUMULATIVE time ---")
    stats.print_stats(30)

    stats.sort_stats("tottime")
    print("\n--- Top 30 by TOTAL (self) time ---")
    stats.print_stats(30)

    # Grep for the interesting functions
    print("\n--- Key functions ---")
    for fn_name in [
        "generate_opponent_hand",
        "_compute_range_probs",
        "_tile_template_precomputed",
        "extract_event_tensors",
        "_build_batch_tensors",
        "_build_shared_events",
        "_shared_to_standard",
        "_normalize_events_inplace",
        "forward_batch",
        "_apply_joint_card_removal",
        "_dead_mask_array",
    ]:
        stats.print_stats(fn_name)


if __name__ == "__main__":
    main()
