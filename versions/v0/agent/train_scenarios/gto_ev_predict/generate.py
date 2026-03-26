"""
Scenario generator for GTO EV prediction training.

Generates random poker situations, computes GTO equity and EV via
Monte Carlo simulation, and saves as a dataset (.pt file).

Can be run standalone:
    python -m agent.train_scenarios.gto_ev_predict.generate --config config.json
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

# Add gto_utils to path for direct import
_gto_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gto_utils")
if _gto_utils_dir not in sys.path:
    sys.path.insert(0, _gto_utils_dir)
from gto_helper import equity, decide_bets, cards as parse_cards


# Card ID (0-51) to GTO string format conversion
# env: rank = card_id // 4 + 2 (2..14), suit = card_id % 4 (0=d, 1=h, 2=c, 3=s)
SUIT_MAP = {0: "d", 1: "h", 2: "c", 3: "s"}
RANK_MAP = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
            9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}


def card_id_to_str(card_id):
    """Convert env card ID (0-51) to GTO string like 'Ah'."""
    rank = card_id // 4 + 2
    suit = card_id % 4
    return RANK_MAP[rank] + SUIT_MAP[suit]


def generate_scenario(config):
    """Generate one random poker scenario with GTO EV label.

    Returns dict with env_state and ev_target, or None if GTO computation fails.
    """
    n_scenarios_cfg = config.get("n_scenarios", 50000)
    mc_iters = config.get("mc_iterations", 10000)
    range_pct = config.get("opponent_range_pct", 50)
    default_stack = config.get("default_stack", 1000)
    big_blind = config.get("big_blind", 10)
    max_players = 6

    # Random deck
    deck = np.random.permutation(52)

    # Random street: 0=preflop, 1=flop, 2=turn, 3=river
    street = random.randint(0, 3)

    # Board cards (first 5 of deck, revealed based on street)
    if street == 0:
        board_ids = []
        table = [-1] * 5
    elif street == 1:
        board_ids = deck[:3].tolist()
        table = list(deck[:3]) + [-1, -1]
    elif street == 2:
        board_ids = deck[:4].tolist()
        table = list(deck[:4]) + [-1]
    else:
        board_ids = deck[:5].tolist()
        table = list(deck[:5])

    # Hero hand (cards 5-6 in deck, position 0)
    hero_pos = 0
    hand = deck[5 + 2 * hero_pos: 7 + 2 * hero_pos].tolist()

    # Random number of opponents (1-5)
    n_opponents = random.randint(1, 5)
    n_players = n_opponents + 1

    # Random pot size (scales with street and blinds)
    pot_multiplier = {0: (2, 8), 1: (4, 20), 2: (8, 40), 3: (12, 60)}
    lo, hi = pot_multiplier[street]
    pot = big_blind * random.uniform(lo, hi)

    # Random bet to face (0 = no bet, or some fraction of pot)
    if random.random() < 0.4:
        facing_bet = 0.0
    else:
        facing_bet = pot * random.uniform(0.25, 1.5)

    # Stack
    stack = max(default_stack - pot * random.uniform(0.1, 0.5), facing_bet + big_blind)

    # Bets array
    bets = np.zeros(max_players)
    if n_players >= 2:
        bets[0] = min(big_blind * 0.5, stack)
        bets[1] = min(big_blind, stack)

    # Active positions
    active_positions = np.arange(n_players)

    # Convert to GTO format for equity computation
    hero_str = "".join(card_id_to_str(c) for c in hand)
    board_str = "".join(card_id_to_str(c) for c in board_ids)

    try:
        hero_cards = parse_cards(hero_str)
        board_cards = parse_cards(board_str)
        eq, _ = equity(hero_cards, board_cards, n_opponents, pct=range_pct, iters=mc_iters)

        # Compute EV via decide_bets
        _, fold_ev, call_ev, raise_ev, _ = decide_bets(eq, pot, facing_bet, stack)
        ev_target = max(fold_ev, call_ev, raise_ev)
    except Exception:
        return None

    # Build minimal bet history for env_state
    # For training we create a simplified history with one initial step
    initial_action = np.zeros(13)
    initial_action[1] = 1.0  # call/check as placeholder
    history_step = {
        "pos": hero_pos,
        "pot": float(pot * 0.5),  # pot was smaller earlier
        "action": torch.tensor(initial_action, dtype=torch.float32),
        "table": table,
    }

    now = {
        "pos": hero_pos,
        "pot": float(pot),
        "bank": float(stack),
        "hand": hand,
        "table": table,
        "bets": bets,
        "active_positions": active_positions,
    }

    env_state = {
        "table_state": [history_step],
        "now": now,
        "active_positions": active_positions,
    }

    return {
        "env_state": env_state,
        "ev_target": float(ev_target),
        "equity": float(eq),
        "pot": float(pot),
        "facing_bet": float(facing_bet),
        "stack": float(stack),
        "street": street,
        "n_opponents": n_opponents,
    }


def _compute_norm_stats(scenarios):
    """Compute mean/std for targets and scalar inputs across all scenarios."""
    evs, pots, banks, all_bets = [], [], [], []
    for s in scenarios:
        evs.append(s["ev_target"])
        now = s["env_state"]["now"]
        pots.append(now["pot"])
        banks.append(now["bank"])
        all_bets.extend(float(b) for b in now["bets"])
        for step in s["env_state"]["table_state"]:
            pots.append(step["pot"])

    def _stats(vals):
        arr = np.array(vals, dtype=np.float64)
        m, s = float(arr.mean()), float(arr.std())
        if s < 1e-8:
            s = 1.0
        return m, s

    return {
        "ev_mean": _stats(evs)[0], "ev_std": _stats(evs)[1],
        "pot_mean": _stats(pots)[0], "pot_std": _stats(pots)[1],
        "bank_mean": _stats(banks)[0], "bank_std": _stats(banks)[1],
        "bets_mean": _stats(all_bets)[0], "bets_std": _stats(all_bets)[1],
    }


def _normalize_scenarios(scenarios, norm_stats):
    """Normalize targets and scalar inputs in-place."""
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    bank_m, bank_s = norm_stats["bank_mean"], norm_stats["bank_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]

    for s in scenarios:
        s["ev_target"] = (s["ev_target"] - ev_m) / ev_s
        now = s["env_state"]["now"]
        now["pot"] = (now["pot"] - pot_m) / pot_s
        now["bank"] = (now["bank"] - bank_m) / bank_s
        now["bets"] = (now["bets"] - bets_m) / bets_s
        for step in s["env_state"]["table_state"]:
            step["pot"] = (step["pot"] - pot_m) / pot_s


def generate_dataset(config, save_dir, log=None):
    """Generate full dataset of scenarios, normalize, and save to disk.

    Args:
        config: gto_ev_train config dict
        save_dir: directory to save dataset.pt and norm_stats.pt into
        log: optional logger

    Returns:
        (scenarios_list, norm_stats_dict)
    """
    n_scenarios = config.get("n_scenarios", 50000)
    dataset_path = os.path.join(save_dir, "dataset.pt")
    stats_path = os.path.join(save_dir, "norm_stats.pt")

    if os.path.exists(dataset_path) and os.path.exists(stats_path):
        if log:
            log(f"Dataset already exists at {dataset_path}, loading...")
        scenarios = torch.load(dataset_path, weights_only=False)
        norm_stats = torch.load(stats_path, weights_only=False)
        return scenarios, norm_stats

    if log:
        log(f"Generating {n_scenarios} scenarios...")

    scenarios = []
    failed = 0
    for _ in tqdm(range(n_scenarios), desc="Generating scenarios"):
        scenario = generate_scenario(config)
        if scenario is not None:
            scenarios.append(scenario)
        else:
            failed += 1

    if log:
        log(f"Generated {len(scenarios)} scenarios ({failed} failed)")

    # Compute normalization stats on raw data, then normalize
    norm_stats = _compute_norm_stats(scenarios)
    if log:
        log(f"Norm stats: " + ", ".join(f"{k}={v:.4f}" for k, v in norm_stats.items()))
    _normalize_scenarios(scenarios, norm_stats)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(scenarios, dataset_path)
    torch.save(norm_stats, stats_path)
    if log:
        log(f"Dataset saved to {dataset_path}")

    return scenarios, norm_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GTO EV training dataset")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--save-dir", default="data/standalone", help="Directory to save dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_cfg = config.get("gto_ev_train", config)
    scenarios, norm_stats = generate_dataset(train_cfg, args.save_dir, log=print)
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        evs = [s["ev_target"] for s in scenarios]
        print(f"Normalized EV range: [{min(evs):.2f}, {max(evs):.2f}], mean: {sum(evs)/len(evs):.4f}")
