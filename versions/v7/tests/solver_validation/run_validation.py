#!/usr/bin/env python3
"""Solver validation: compare gpu_solver_v3 EVs against reference implementation.

Generates ~9000 poker spots, computes reference EVs analytically, runs the
solver on the same spots, and reports discrepancies.

Usage (from versions/v6):
    python -m tests.solver_validation.run_validation [options]

Options:
    --streets STREET[,STREET]   Only validate these streets (0-3). Default: all.
    --max-spots N               Limit total spots (for quick smoke tests).
    --save PATH                 Save results to JSON file.
    --threshold PCT             Discrepancy threshold as fraction of pot (default: 0.05).
    --river-only                Shortcut for --streets 3 (fastest).
    --verbose                   Print per-spot details for discrepancies.
    --solver-device DEVICE      Device for solver MC (default: cpu).
"""

import sys
import os
import json
import time
import argparse
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np

from tests.solver_validation.spots import generate_all_spots, Spot, card_name
from tests.solver_validation.reference_ev import compute_reference_evs, ReferenceEVResult
from tests.solver_validation.cards import card_str

from agent.gto_utils.gpu_solver_v3 import compute_ev_v3
from agent.gto_utils.gpu_solver_v2 import HAND_RANKINGS, expand_range


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------

def run_solver_on_spot(spot, raise_fracs, solver_config, device="cpu"):
    """Run gpu_solver_v3 on a spot and return EVs for each action.

    Returns:
        dict with keys: 'fold_ev', 'call_ev', 'raise_evs' (dict frac->ev),
        'allin_ev', 'raw_equity'
    """
    hero_cards = torch.tensor(list(spot.hero_cards), dtype=torch.long)
    board_cards = torch.tensor(list(spot.board_cards), dtype=torch.long)

    dead = set(spot.hero_cards) | set(spot.board_cards)
    opponent_range = [HAND_RANKINGS[:]]  # full range per opponent
    n_opponents = len(spot.opponent_positions) if spot.opponent_positions else 1
    opponent_range_hand_types = [HAND_RANKINGS[:] for _ in range(n_opponents)]

    mc_iters = int(solver_config.get("mc_iterations", 5000))
    combo_response_iters = int(solver_config.get("combo_response_iters", 30))
    reraise_threshold = float(solver_config.get("reraise_threshold", 0.72))
    eqr_enabled = bool(solver_config.get("eqr_enabled", True))
    weighted_sampling = bool(solver_config.get("weighted_sampling", True))
    threshold_smoothing = solver_config.get("threshold_smoothing", None)
    polarized_reraise = solver_config.get("polarized_reraise", None)

    from agent.gto_utils.gpu_solver_v3 import _prepare_ev_state_v3, _compute_ev_v3_from_state

    state = _prepare_ev_state_v3(
        hero_cards, board_cards, opponent_range_hand_types,
        n_iters=mc_iters, device=device,
        hero_position=spot.hero_position,
        street=spot.street,
        n_players=spot.table_size,
        eqr_enabled=eqr_enabled,
        combo_response_iters=combo_response_iters,
        reraise_threshold=reraise_threshold,
        weighted_sampling=weighted_sampling,
        action_history=None,
        opponent_positions=spot.opponent_positions,
        threshold_smoothing=threshold_smoothing,
        polarized_reraise=polarized_reraise,
    )

    solver_results = {
        "raw_equity": state["raw_equity"],
        "raise_evs": {},
    }

    # Compute EVs for each raise frac
    for raise_frac in raise_fracs:
        fold_ev, call_ev, raise_ev, best_ev = _compute_ev_v3_from_state(
            state, spot.pot, spot.facing_bet, spot.stack, spot.hero_invested,
            raise_frac=raise_frac,
        )
        solver_results["fold_ev"] = fold_ev
        solver_results["call_ev"] = call_ev
        solver_results["raise_evs"][raise_frac] = raise_ev

    # All-in
    if spot.stack > spot.facing_bet:
        allin_frac = (spot.stack - spot.facing_bet) / max(spot.pot + spot.facing_bet, 1e-6)
        fold_ev, call_ev, raise_ev, best_ev = _compute_ev_v3_from_state(
            state, spot.pot, spot.facing_bet, spot.stack, spot.hero_invested,
            raise_frac=allin_frac,
        )
        solver_results["allin_ev"] = raise_ev
    else:
        solver_results["allin_ev"] = solver_results.get("call_ev", 0.0)

    return solver_results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Discrepancy:
    spot_id: int
    action: str
    ref_ev: float
    solver_ev: float
    diff: float
    diff_pct_pot: float  # as fraction of pot


def compare_spot(spot, ref_result, solver_result, threshold_pct=0.05):
    """Compare reference and solver EVs for one spot.

    Returns list of Discrepancy objects for actions exceeding threshold.
    """
    pot = max(spot.pot, 1.0)
    discrepancies = []

    # Fold EV
    ref_fold = ref_result.fold_ev
    sol_fold = solver_result["fold_ev"]
    diff = abs(ref_fold - sol_fold)
    if diff / pot > threshold_pct:
        discrepancies.append(Discrepancy(
            spot.spot_id, "fold", ref_fold, sol_fold, diff, diff / pot))

    # Call EV
    ref_call = ref_result.call_ev
    sol_call = solver_result["call_ev"]
    diff = abs(ref_call - sol_call)
    if diff / pot > threshold_pct:
        discrepancies.append(Discrepancy(
            spot.spot_id, "call", ref_call, sol_call, diff, diff / pot))

    # Raise EVs
    for frac in ref_result.raise_evs:
        if frac in solver_result["raise_evs"]:
            ref_raise = ref_result.raise_evs[frac]
            sol_raise = solver_result["raise_evs"][frac]
            diff = abs(ref_raise - sol_raise)
            if diff / pot > threshold_pct:
                discrepancies.append(Discrepancy(
                    spot.spot_id, f"raise_{frac}", ref_raise, sol_raise, diff, diff / pot))

    # All-in EV
    ref_allin = ref_result.allin_ev
    sol_allin = solver_result["allin_ev"]
    diff = abs(ref_allin - sol_allin)
    if diff / pot > threshold_pct:
        discrepancies.append(Discrepancy(
            spot.spot_id, "allin", ref_allin, sol_allin, diff, diff / pot))

    return discrepancies


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_discrepancies, spots, total_comparisons, elapsed):
    """Print a summary report of the validation results."""
    n_disc = len(all_discrepancies)
    print("\n" + "=" * 70)
    print("SOLVER VALIDATION REPORT")
    print("=" * 70)
    print(f"Spots validated:     {len(spots)}")
    print(f"Total comparisons:   {total_comparisons}")
    print(f"Discrepancies found: {n_disc}")
    print(f"Time elapsed:        {elapsed:.1f}s")
    print()

    if not all_discrepancies:
        print("ALL CHECKS PASSED — no discrepancies above threshold.")
        return

    # Group by action type
    by_action = defaultdict(list)
    for d in all_discrepancies:
        by_action[d.action].append(d)

    print("Discrepancies by action type:")
    print("-" * 50)
    for action in sorted(by_action.keys()):
        discs = by_action[action]
        avg_pct = sum(d.diff_pct_pot for d in discs) / len(discs)
        max_pct = max(d.diff_pct_pot for d in discs)
        print(f"  {action:20s}: {len(discs):4d} spots, "
              f"avg {avg_pct*100:.1f}% pot, max {max_pct*100:.1f}% pot")

    # Group by dimension
    spot_map = {s.spot_id: s for s in spots}
    disc_spot_ids = set(d.spot_id for d in all_discrepancies)

    print("\nDiscrepancies by street:")
    print("-" * 50)
    by_street = defaultdict(int)
    total_by_street = defaultdict(int)
    for s in spots:
        total_by_street[s.street] += 1
    for sid in disc_spot_ids:
        by_street[spot_map[sid].street] += 1
    for street in sorted(total_by_street.keys()):
        name = ["preflop", "flop", "turn", "river"][street]
        n = by_street[street]
        t = total_by_street[street]
        print(f"  {name:10s}: {n:4d}/{t} spots ({100*n/t:.1f}%)")

    print("\nDiscrepancies by hand category:")
    print("-" * 50)
    by_cat = defaultdict(int)
    total_by_cat = defaultdict(int)
    for s in spots:
        total_by_cat[s.hand_category] += 1
    for sid in disc_spot_ids:
        by_cat[spot_map[sid].hand_category] += 1
    for cat in sorted(total_by_cat.keys()):
        n = by_cat[cat]
        t = total_by_cat[cat]
        print(f"  {cat:12s}: {n:4d}/{t} spots ({100*n/t:.1f}%)")

    # Top 10 worst discrepancies
    print("\nTop 10 largest discrepancies:")
    print("-" * 70)
    sorted_disc = sorted(all_discrepancies, key=lambda d: d.diff_pct_pot, reverse=True)
    for d in sorted_disc[:10]:
        s = spot_map[d.spot_id]
        hero = f"{card_str(s.hero_cards[0])}{card_str(s.hero_cards[1])}"
        board = " ".join(card_str(c) for c in s.board_cards) if s.board_cards else "preflop"
        print(f"  spot {d.spot_id:5d} | {d.action:15s} | "
              f"ref={d.ref_ev:+8.1f} solver={d.solver_ev:+8.1f} "
              f"diff={d.diff_pct_pot*100:5.1f}% pot | "
              f"{hero} on {board}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solver validation")
    parser.add_argument("--streets", type=str, default=None,
                        help="Comma-separated streets to validate (0-3)")
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Discrepancy threshold as fraction of pot")
    parser.add_argument("--river-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--solver-device", type=str, default="cpu")
    args = parser.parse_args()

    # Generate spots
    print("Generating spots...")
    all_spots = generate_all_spots()

    # Filter by street
    if args.river_only:
        streets = {3}
    elif args.streets:
        streets = {int(s) for s in args.streets.split(",")}
    else:
        streets = {0, 1, 2, 3}

    spots = [s for s in all_spots if s.street in streets]
    if args.max_spots:
        spots = spots[:args.max_spots]

    print(f"Validating {len(spots)} spots (streets: {sorted(streets)})")

    # Solver config (from config.json defaults)
    solver_config = {
        "mc_iterations": 5000,
        "combo_response_iters": 30,
        "reraise_threshold": 0.72,
        "eqr_enabled": True,
        "weighted_sampling": True,
        "threshold_smoothing": {
            "enabled": True,
            "beta_fold": 0.07,
            "beta_reraise": 0.07,
        },
        "polarized_reraise": {
            "enabled": True,
            "bluff_threshold": 0.25,
            "beta_bluff": 0.10,
            "bluff_frequency": 0.30,
        },
    }

    # Raise fracs from config
    raise_fracs_by_street = {
        0: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        1: [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
        2: [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
        3: [0.1, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
    }

    all_discrepancies = []
    total_comparisons = 0
    t0 = time.time()

    for i, spot in enumerate(spots):
        raise_fracs = raise_fracs_by_street.get(spot.street, raise_fracs_by_street[1])

        # Compute reference EVs
        try:
            ref_result = compute_reference_evs(spot, raise_fracs, solver_config)
        except Exception as e:
            print(f"  [WARN] Reference failed on spot {spot.spot_id}: {e}")
            continue

        # Run solver
        try:
            solver_result = run_solver_on_spot(spot, raise_fracs, solver_config,
                                               device=args.solver_device)
        except Exception as e:
            print(f"  [WARN] Solver failed on spot {spot.spot_id}: {e}")
            continue

        # Compare
        discrepancies = compare_spot(spot, ref_result, solver_result, args.threshold)
        all_discrepancies.extend(discrepancies)

        # Count comparisons: fold + call + raises + allin
        total_comparisons += 2 + len(raise_fracs) + 1

        if args.verbose and discrepancies:
            hero = f"{card_str(spot.hero_cards[0])}{card_str(spot.hero_cards[1])}"
            board = " ".join(card_str(c) for c in spot.board_cards) if spot.board_cards else "preflop"
            print(f"\n  Spot {spot.spot_id}: {hero} on {board} "
                  f"(street={spot.street}, pot={spot.pot:.0f}, fb={spot.facing_bet:.0f})")
            print(f"  Equity: ref={ref_result.exact_equity:.4f} "
                  f"solver={solver_result['raw_equity']:.4f}")
            for d in discrepancies:
                print(f"    {d.action}: ref={d.ref_ev:+.1f} solver={d.solver_ev:+.1f} "
                      f"diff={d.diff_pct_pot*100:.1f}% pot")

        # Progress
        if (i + 1) % 100 == 0 or i == len(spots) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(spots) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(spots)}] {rate:.1f} spots/s, "
                  f"ETA {eta:.0f}s, {len(all_discrepancies)} discrepancies", end="\r")

    elapsed = time.time() - t0
    print_report(all_discrepancies, spots, total_comparisons, elapsed)

    # Save results
    if args.save:
        save_data = {
            "n_spots": len(spots),
            "n_discrepancies": len(all_discrepancies),
            "threshold": args.threshold,
            "elapsed_seconds": elapsed,
            "discrepancies": [
                {"spot_id": d.spot_id, "action": d.action,
                 "ref_ev": d.ref_ev, "solver_ev": d.solver_ev,
                 "diff": d.diff, "diff_pct_pot": d.diff_pct_pot}
                for d in all_discrepancies
            ],
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
