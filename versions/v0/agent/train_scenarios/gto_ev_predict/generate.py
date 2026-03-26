"""
Scenario generator for GTO EV prediction training.

Generates poker hands using the Table simulator with GTO-based action
sampling. Each sample is a SEQUENCE of events from hand start to a
decision point, paired with a scalar EV target.

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
import torch.nn.functional as F
from tqdm.auto import tqdm

from env.table import Table

# Add gto_utils to path for direct import
_gto_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gto_utils")
if _gto_utils_dir not in sys.path:
    sys.path.insert(0, _gto_utils_dir)
from gpu_solver import gpu_equity, compute_ev


MAX_PLAYERS = 6
GTO_TEMPERATURE = 1.0


def _get_board_cards(table):
    """Extract revealed board card IDs from table state."""
    if table.turn == 0:
        return []
    elif table.turn == 1:
        return table.deck[:3].tolist()
    elif table.turn == 2:
        return table.deck[:4].tolist()
    else:
        return table.deck[:5].tolist()


def _get_table_display(table):
    """Get 5-element board display with -1 for unrevealed cards."""
    if table.turn == 0:
        return [-1] * 5
    elif table.turn == 1:
        return list(table.deck[:3]) + [-1, -1]
    elif table.turn == 2:
        return list(table.deck[:4]) + [-1]
    else:
        return list(table.deck[:5])


def _compute_player_ev(table, player_pos, device="mps", mc_iters=10000):
    """Compute EV for a player at the current table state.

    Returns (equity, fold_ev, call_ev, raise_ev, best_ev, hero_invested, facing_bet)
    or None on failure.
    """
    hand = table.deck[5 + 2 * player_pos: 7 + 2 * player_pos]
    board_ids = _get_board_cards(table)

    hero_t = torch.tensor(hand.tolist(), dtype=torch.long)
    board_t = torch.tensor(board_ids, dtype=torch.long) if board_ids else torch.tensor([], dtype=torch.long)

    n_active = int((table.players_state >= 0).sum())
    n_opponents = max(1, n_active - 1)

    try:
        eq = gpu_equity(hero_t, board_t, n_opponents, n_iters=mc_iters, device=device)
    except Exception:
        return None

    # Hero's investment: start_credits - current credits
    hero_invested = table.start_credits - table.credits[player_pos]

    # Facing bet: difference between high bet and our current bet
    facing_bet = max(0, table.high_bet - table.bets[player_pos])

    stack = table.credits[player_pos]
    pot = table.pot

    fold_ev, call_ev, raise_ev, best_ev = compute_ev(
        eq, pot, facing_bet, stack, hero_invested
    )

    return eq, fold_ev, call_ev, raise_ev, best_ev, hero_invested, facing_bet


def _sample_gto_action(fold_ev, call_ev, raise_ev, table, player_pos):
    """Sample an action based on GTO EV distribution.

    Maps EVs to action indices: fold=0, call=1, raise(pot-size)=5, all-in=12.
    Returns one-hot action tensor (13,).
    """
    evs = torch.tensor([fold_ev, call_ev, raise_ev], dtype=torch.float)
    probs = F.softmax(evs / GTO_TEMPERATURE, dim=0)

    choice = torch.multinomial(probs, 1).item()

    action = torch.zeros(13, dtype=torch.float32)
    if choice == 0:
        action[0] = 1.0  # fold
    elif choice == 1:
        action[1] = 1.0  # call/check
    else:
        # Raise: pick a random raise size bin (2-11) or all-in (12)
        if random.random() < 0.15:
            action[12] = 1.0  # all-in
        else:
            raise_bin = random.randint(2, 11)
            action[raise_bin] = 1.0

    return action


def _build_event(table, hero_pos, acting_pos, action, num_players, big_blind, small_blind):
    """Build one event dict from current table state."""
    hand = table.deck[5 + 2 * hero_pos: 7 + 2 * hero_pos].tolist()
    hero_stack = float(table.credits[hero_pos])
    table_cards = _get_table_display(table)
    bets = np.copy(table.bets)

    return {
        "hand": hand,
        "num_players": num_players,
        "hero_pos": hero_pos,
        "acting_pos": acting_pos,
        "big_blind": float(big_blind),
        "small_blind": float(small_blind),
        "stack": hero_stack,
        "table": table_cards,
        "pot": float(table.pot),
        "bets": bets,
        "action": action,
    }


def generate_scenario(config, device="mps"):
    """Generate one scenario: a full event sequence + EV target.

    Simulates a poker hand using Table with GTO-sampled actions.
    Picks a random decision point where hero must act, and records
    the sequence up to that point.

    Returns dict with events list and ev_target, or None on failure.
    """
    mc_iters = config.get("mc_iterations", 10000)
    big_blind = config.get("big_blind", 10)
    small_blind = big_blind // 2
    default_stack = config.get("default_stack", 1000)

    # Random number of players (2-6)
    num_players = random.randint(2, MAX_PLAYERS)

    table = Table(
        num_players=num_players,
        bins=10,
        max_bet=2,
        start_credits=default_stack,
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.start_table()

    # Track all events and hero decision points
    all_events = []
    hero_decision_indices = []  # indices in all_events where hero had to act

    # First event: post-blinds state (no action yet)
    hero_pos = random.randint(0, num_players - 1)
    initial_action = torch.zeros(13, dtype=torch.float32)
    first_event = _build_event(
        table, hero_pos, table.active_player, initial_action,
        num_players, big_blind, small_blind
    )
    all_events.append(first_event)

    # Simulate the hand
    max_actions = 4 * num_players  # safety limit
    hand_ended = False

    for _ in range(max_actions):
        active_pos = table.active_player

        # Check if this player can act (state == 1)
        if table.players_state[active_pos] != 1:
            break

        # Compute EV for the active player to decide their action
        ev_result = _compute_player_ev(table, active_pos, device=device, mc_iters=mc_iters)
        if ev_result is None:
            return None

        eq, fold_ev, call_ev, raise_ev, best_ev, hero_invested, facing_bet = ev_result

        # If this is the hero's turn, record it as a potential decision point
        if active_pos == hero_pos:
            # Record decision event (action=zeros, hero must decide)
            decision_event = _build_event(
                table, hero_pos, active_pos, torch.zeros(13, dtype=torch.float32),
                num_players, big_blind, small_blind
            )
            hero_decision_indices.append(len(all_events))
            all_events.append(decision_event)

        # Sample GTO action
        action = _sample_gto_action(fold_ev, call_ev, raise_ev, table, active_pos)

        # Record the action event (what actually happened)
        action_event = _build_event(
            table, hero_pos, active_pos, action,
            num_players, big_blind, small_blind
        )

        # If this was hero's decision, replace the decision event with the action
        # (we keep the decision event at its index, add action as next event)
        all_events.append(action_event)

        # Execute action on table
        end, several_all_in, state, bet = table.step(action)

        if end:
            hand_ended = True
            break

        # If all-in situation, table handles auto-advance
        if several_all_in:
            break

    # Need at least one hero decision point
    if not hero_decision_indices:
        return None

    # Pick a random hero decision point
    decision_idx = random.choice(hero_decision_indices)

    # The sequence is everything up to and including the decision event
    events = all_events[:decision_idx + 1]

    if len(events) < 2:
        return None

    # Recompute EV at the chosen decision point
    # We need to re-simulate up to that point to get the table state
    # Instead, we stored the state in the event — reconstruct from it
    decision_event = events[-1]

    # Use stored table state to compute EV
    hand = decision_event["hand"]
    hero_t = torch.tensor(hand, dtype=torch.long)

    board_ids = [int(c) for c in decision_event["table"] if int(c) >= 0]
    board_t = torch.tensor(board_ids, dtype=torch.long) if board_ids else torch.tensor([], dtype=torch.long)

    n_active = num_players  # approximate: use all players at decision point
    n_opponents = max(1, n_active - 1)

    try:
        eq = gpu_equity(hero_t, board_t, n_opponents, n_iters=mc_iters, device=device)
    except Exception:
        return None

    hero_invested = default_stack - decision_event["stack"]
    # Estimate facing bet from bets array
    bets_arr = decision_event["bets"]
    if isinstance(bets_arr, np.ndarray):
        bets_arr = bets_arr.tolist()
    max_bet = max(bets_arr) if bets_arr else 0
    hero_bet = bets_arr[hero_pos] if hero_pos < len(bets_arr) else 0
    facing_bet = max(0, max_bet - hero_bet)

    stack = decision_event["stack"]
    pot = decision_event["pot"]

    fold_ev, call_ev, raise_ev, best_ev = compute_ev(
        eq, pot, facing_bet, stack, hero_invested
    )
    ev_target = best_ev

    return {
        "events": events,
        "ev_target": float(ev_target),
        "equity": float(eq),
        "pot": float(pot),
        "facing_bet": float(facing_bet),
        "stack": float(stack),
        "hero_invested": float(hero_invested),
        "num_players": num_players,
        "n_events": len(events),
    }


def _compute_norm_stats(scenarios):
    """Compute mean/std for normalization across all scenarios."""
    evs, pots, stacks, all_bets, blinds = [], [], [], [], []
    for s in scenarios:
        evs.append(s["ev_target"])
        for event in s["events"]:
            pots.append(event["pot"])
            stacks.append(event["stack"])
            blinds.append(event["big_blind"])
            raw_bets = event["bets"]
            if isinstance(raw_bets, np.ndarray):
                raw_bets = raw_bets.tolist()
            all_bets.extend(float(b) for b in raw_bets)

    def _stats(vals):
        arr = np.array(vals, dtype=np.float64)
        m, s = float(arr.mean()), float(arr.std())
        if s < 1e-8:
            s = 1.0
        return m, s

    return {
        "ev_mean": _stats(evs)[0], "ev_std": _stats(evs)[1],
        "pot_mean": _stats(pots)[0], "pot_std": _stats(pots)[1],
        "stack_mean": _stats(stacks)[0], "stack_std": _stats(stacks)[1],
        "bets_mean": _stats(all_bets)[0], "bets_std": _stats(all_bets)[1],
        "blind_mean": _stats(blinds)[0], "blind_std": _stats(blinds)[1],
    }


def _normalize_scenarios(scenarios, norm_stats):
    """Normalize targets and scalar inputs in-place."""
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    stack_m, stack_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]
    blind_m, blind_s = norm_stats["blind_mean"], norm_stats["blind_std"]

    for s in scenarios:
        s["ev_target"] = (s["ev_target"] - ev_m) / ev_s
        for event in s["events"]:
            event["pot"] = (event["pot"] - pot_m) / pot_s
            event["stack"] = (event["stack"] - stack_m) / stack_s
            event["big_blind"] = (event["big_blind"] - blind_m) / blind_s
            event["small_blind"] = (event["small_blind"] - blind_m) / blind_s
            if isinstance(event["bets"], np.ndarray):
                event["bets"] = (event["bets"] - bets_m) / bets_s
            else:
                event["bets"] = [(b - bets_m) / bets_s for b in event["bets"]]


def generate_dataset(config, save_dir, log=None):
    """Generate full dataset of event-sequence scenarios, normalize, and save.

    Args:
        config: gto_ev_train config dict
        save_dir: directory to save dataset.pt and norm_stats.pt
        log: optional logger

    Returns:
        (scenarios_list, norm_stats_dict)
    """
    n_scenarios = config.get("n_scenarios", 50000)
    device = config.get("device", "mps")
    dataset_path = os.path.join(save_dir, "dataset.pt")
    stats_path = os.path.join(save_dir, "norm_stats.pt")

    if os.path.exists(dataset_path) and os.path.exists(stats_path):
        if log:
            log(f"Dataset already exists at {dataset_path}, loading...")
        scenarios = torch.load(dataset_path, weights_only=False)
        norm_stats = torch.load(stats_path, weights_only=False)
        return scenarios, norm_stats

    if log:
        log(f"Generating {n_scenarios} scenarios on {device}...")

    scenarios = []
    failed = 0
    for _ in tqdm(range(n_scenarios), desc="Generating scenarios"):
        scenario = generate_scenario(config, device=device)
        if scenario is not None:
            scenarios.append(scenario)
        else:
            failed += 1

    if log:
        log(f"Generated {len(scenarios)} scenarios ({failed} failed)")
        if scenarios:
            lengths = [s["n_events"] for s in scenarios]
            log(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

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
