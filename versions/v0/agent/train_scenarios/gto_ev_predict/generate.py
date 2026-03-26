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


def _get_solver(solver_name):
    """Import and return solver functions based on config choice.

    Args:
        solver_name: "v1" (random ranges), "v2" (range-aware)

    Returns:
        (gpu_equity_fn, compute_ev_fn, extra_modules_dict)
    """
    if solver_name == "v1":
        from gpu_solver import gpu_equity, compute_ev
        return gpu_equity, compute_ev, {}
    elif solver_name == "v2":
        from gpu_solver_v2 import gpu_equity_v2, compute_ev_v2, get_position_range, narrow_range, expand_range
        return gpu_equity_v2, compute_ev_v2, {
            "get_position_range": get_position_range,
            "narrow_range": narrow_range,
            "expand_range": expand_range,
        }
    else:
        raise ValueError(f"Unknown solver: {solver_name}. Use 'v1' or 'v2'.")


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


def _build_opponent_ranges(table, player_pos, action_history, solver_modules):
    """Build opponent range hand types based on positions and past actions.

    Args:
        table: Table instance
        player_pos: current player's position (excluded from opponents)
        action_history: list of (position, action_type) tuples for the hand
        solver_modules: dict with get_position_range, narrow_range (from v2 solver)

    Returns:
        list of lists of hand type strings, one per active opponent
    """
    get_position_range = solver_modules["get_position_range"]
    narrow_range = solver_modules["narrow_range"]

    opponent_ranges = []
    for pos in range(table.num_players):
        if pos == player_pos:
            continue
        if table.players_state[pos] < 0:
            continue  # folded

        hand_types = get_position_range(pos, table.num_players)
        for act_pos, act_type in action_history:
            if act_pos == pos:
                hand_types = narrow_range(hand_types, act_type)

        opponent_ranges.append(hand_types)

    return opponent_ranges


def _compute_player_ev(table, player_pos, action_history, solver_name="v2",
                       device="mps", mc_iters=10000, n_raise_samples=3,
                       mdf_max_fold=0.7, reraise_pct=0.15, reraise_cap=0.10):
    """Compute EV for a player. Supports both v1 (random range) and v2 (range-aware).

    Returns dict with keys: equity, fold_ev, call_ev, raise_evs, etc.
    Returns None on failure.
    """
    gpu_equity_fn, compute_ev_fn, solver_modules = _get_solver(solver_name)

    hand = table.deck[5 + 2 * player_pos: 7 + 2 * player_pos]
    board_ids = _get_board_cards(table)

    hero_t = torch.tensor(hand.tolist(), dtype=torch.long)
    board_t = torch.tensor(board_ids, dtype=torch.long) if board_ids else torch.tensor([], dtype=torch.long)

    hero_invested = table.start_credits - table.credits[player_pos]
    facing_bet = max(0, table.high_bet - table.bets[player_pos])
    stack = table.credits[player_pos]
    pot = table.pot

    n_active = int((table.players_state >= 0).sum())
    n_opponents = max(1, n_active - 1)

    if solver_name == "v1":
        # V1: random ranges, simple EV
        try:
            eq = gpu_equity_fn(hero_t, board_t, n_opponents, n_iters=mc_iters, device=device)
            fold_ev, call_ev, _, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested)
        except Exception:
            return None

        # Raise EVs
        raise_evs = []
        max_bet_mult = table.max_bet
        bins = table.bins
        available_bins = list(range(2, bins + 2))
        n_samples = min(n_raise_samples, len(available_bins))
        for b in random.sample(available_bins, n_samples):
            raise_frac = (b - 1) * max_bet_mult / bins
            actual_raise = min(facing_bet + raise_frac * pot, stack)
            if actual_raise >= stack:
                continue
            _, _, r_ev, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested, raise_frac=raise_frac)
            raise_evs.append((b, raise_frac, r_ev))

        allin_frac = stack / max(pot + facing_bet, 1e-6)
        _, _, allin_ev, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested, raise_frac=allin_frac)
        raise_evs.append((12, allin_frac, allin_ev))

    else:
        # V2: range-aware equity + opponent response model
        expand_range = solver_modules["expand_range"]
        opp_range_types = _build_opponent_ranges(table, player_pos, action_history, solver_modules)

        if not opp_range_types:
            return {
                "equity": 1.0, "fold_ev": -hero_invested,
                "call_ev": pot - hero_invested,
                "raise_evs": [], "hero_invested": hero_invested,
                "facing_bet": facing_bet, "stack": stack, "pot": pot,
            }

        ev_extra = {"mdf_max_fold": mdf_max_fold, "reraise_pct": reraise_pct, "reraise_cap": reraise_cap}

        try:
            fold_ev, call_ev, _, _ = compute_ev_fn(
                hero_t, board_t, opp_range_types,
                pot, facing_bet, stack, hero_invested,
                raise_frac=1.0, n_iters=mc_iters, device=device, **ev_extra
            )
            dead = set(hero_t.tolist())
            if len(board_t) > 0:
                dead.update(board_t.tolist())
            opp_combos = [expand_range(ht_list, dead) for ht_list in opp_range_types]
            eq = gpu_equity_fn(hero_t, board_t, opp_combos, mc_iters, device)
        except Exception:
            return None

        # Raise EVs
        raise_evs = []
        max_bet_mult = table.max_bet
        bins = table.bins
        available_bins = list(range(2, bins + 2))
        n_samples = min(n_raise_samples, len(available_bins))
        for b in random.sample(available_bins, n_samples):
            raise_frac = (b - 1) * max_bet_mult / bins
            actual_raise = min(facing_bet + raise_frac * pot, stack)
            if actual_raise >= stack:
                continue
            try:
                _, _, r_ev, _ = compute_ev_fn(
                    hero_t, board_t, opp_range_types,
                    pot, facing_bet, stack, hero_invested,
                    raise_frac=raise_frac, n_iters=mc_iters, device=device, **ev_extra
                )
                raise_evs.append((b, raise_frac, r_ev))
            except Exception:
                continue

        allin_frac = stack / max(pot + facing_bet, 1e-6)
        try:
            _, _, allin_ev, _ = compute_ev_fn(
                hero_t, board_t, opp_range_types,
                pot, facing_bet, stack, hero_invested,
                raise_frac=allin_frac, n_iters=mc_iters, device=device, **ev_extra
            )
            raise_evs.append((12, allin_frac, allin_ev))
        except Exception:
            pass

    return {
        "equity": eq,
        "fold_ev": fold_ev,
        "call_ev": call_ev,
        "raise_evs": raise_evs,
        "hero_invested": hero_invested,
        "facing_bet": facing_bet,
        "stack": stack,
        "pot": pot,
    }


def _sample_gto_action(ev_result, temperature=1.0):
    """Sample an action based on GTO EV distribution over all options.

    ev_result: dict from _compute_player_ev with fold_ev, call_ev, raise_evs.
    temperature: softmax temperature for action sampling.
    Returns one-hot action tensor (13,).
    """
    # Build EV list: [fold, call, raise_size_1, raise_size_2, ..., all-in]
    options = []  # (action_bin, ev)
    options.append((0, ev_result["fold_ev"]))
    options.append((1, ev_result["call_ev"]))
    for action_bin, _, ev in ev_result["raise_evs"]:
        options.append((action_bin, ev))

    evs = torch.tensor([ev for _, ev in options], dtype=torch.float)
    probs = F.softmax(evs / temperature, dim=0)

    choice_idx = torch.multinomial(probs, 1).item()
    chosen_bin = options[choice_idx][0]

    action = torch.zeros(13, dtype=torch.float32)
    action[chosen_bin] = 1.0
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
    n_raise_samples = config.get("n_raise_samples", 3)
    big_blind = config.get("big_blind", 10)
    small_blind = big_blind // 2
    default_stack = config.get("default_stack", 1000)
    max_players = config.get("max_players", 9)
    temperature = config.get("gto_temperature", 1.0)
    table_bins = config.get("table_bins", 10)
    table_max_bet = config.get("table_max_bet", 2)
    solver_name = config.get("solver", "v2")
    mdf_max_fold = config.get("mdf_max_fold", 0.7)
    reraise_pct = config.get("reraise_pct", 0.15)
    reraise_cap = config.get("reraise_cap", 0.10)

    # Random number of players (2 to max_players)
    num_players = random.randint(2, max_players)

    table = Table(
        num_players=num_players,
        bins=table_bins,
        max_bet=table_max_bet,
        start_credits=default_stack,
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.start_table()

    # Track all events and hero decision points (with stored EV results)
    all_events = []
    hero_decisions = []  # list of (event_index, ev_result) for hero turns

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
    action_history = []  # list of (position, action_type) for range narrowing

    # Solver-specific params
    ev_kwargs = {"device": device, "mc_iters": mc_iters, "n_raise_samples": n_raise_samples}
    if solver_name == "v2":
        ev_kwargs.update({
            "mdf_max_fold": mdf_max_fold,
            "reraise_pct": reraise_pct,
            "reraise_cap": reraise_cap,
        })

    for _ in range(max_actions):
        active_pos = table.active_player

        # Check if this player can act (state == 1)
        if table.players_state[active_pos] != 1:
            break

        # Compute EV for the active player
        ev_result = _compute_player_ev(
            table, active_pos, action_history, solver_name=solver_name, **ev_kwargs
        )
        if ev_result is None:
            return None

        # If this is the hero's turn, record decision point with EV data
        if active_pos == hero_pos:
            decision_event = _build_event(
                table, hero_pos, active_pos, torch.zeros(13, dtype=torch.float32),
                num_players, big_blind, small_blind
            )
            hero_decisions.append((len(all_events), ev_result))
            all_events.append(decision_event)

        # Sample GTO action from EV distribution (fold, call, multiple raise sizes)
        action = _sample_gto_action(ev_result, temperature=temperature)

        # Classify action for range narrowing
        action_idx = torch.argmax(action).item()
        if action_idx == 0:
            act_type = None  # fold — player leaves, no range update needed
        elif action_idx == 1:
            act_type = "call" if table.turn == 0 else "call_postflop"
        elif action_idx == 12:
            act_type = "3bet" if table.turn == 0 else "bet_postflop"  # all-in as strong action
        else:
            act_type = "open" if table.turn == 0 else "bet_postflop"  # raise

        if act_type is not None:
            action_history.append((active_pos, act_type))

        # Record the action event
        action_event = _build_event(
            table, hero_pos, active_pos, action,
            num_players, big_blind, small_blind
        )
        all_events.append(action_event)

        # Execute action on table
        end, several_all_in, state, bet = table.step(action)

        if end:
            break

        if several_all_in:
            break

    # Need at least one hero decision point
    if not hero_decisions:
        return None

    # Pick a random hero decision point
    decision_idx, ev_result = random.choice(hero_decisions)

    # The sequence is everything up to and including the decision event
    events = all_events[:decision_idx + 1]

    if len(events) < 2:
        return None

    # best_ev = max over fold, call, and all raise sizes
    all_evs = [ev_result["fold_ev"], ev_result["call_ev"]]
    all_evs.extend(ev for _, _, ev in ev_result["raise_evs"])
    best_ev = max(all_evs)

    return {
        "events": events,
        "ev_target": float(best_ev),
        "equity": float(ev_result["equity"]),
        "pot": float(ev_result["pot"]),
        "facing_bet": float(ev_result["facing_bet"]),
        "stack": float(ev_result["stack"]),
        "hero_invested": float(ev_result["hero_invested"]),
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
