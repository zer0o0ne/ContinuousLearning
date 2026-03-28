"""
Unified scenario generator for all GTO training scenarios.

Generates poker hands using the Table simulator with GTO-based action
sampling. Each sample is a sequence of events from hand start to a
decision point, paired with:
  - ev_target: scalar best EV (for value head training)
  - action_evs: per-action EV vector (for action probability training)
  - action_probs: softmax(action_evs / (big_blind * gto_temperature))

Both labels are computed in a single simulation pass to avoid
duplicating expensive solver calls.

Can be run standalone:
    python -m agent.train_scenarios.generation.generate --config config.json
"""

import os
import sys
import json
import copy
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


def _sample_gto_action(ev_result, n_actions, big_blind=10, temperature=1.0):
    """Sample an action based on GTO EV distribution over all options.

    ev_result: dict from _compute_player_ev with fold_ev, call_ev, raise_evs.
    n_actions: size of the action space (table_bins + 3).
    big_blind: big blind size for EV normalization.
    temperature: softmax temperature for action sampling.
    Returns one-hot action tensor (n_actions,).
    """
    options = []  # (action_bin, ev)
    options.append((0, ev_result["fold_ev"]))
    options.append((1, ev_result["call_ev"]))
    for action_bin, _, ev in ev_result["raise_evs"]:
        options.append((action_bin, ev))

    evs = torch.tensor([ev for _, ev in options], dtype=torch.float)
    probs = F.softmax(evs / (big_blind * temperature), dim=0)

    choice_idx = torch.multinomial(probs, 1).item()
    chosen_bin = options[choice_idx][0]

    action = torch.zeros(n_actions, dtype=torch.float32)
    action[chosen_bin] = 1.0
    return action


def _compute_player_ev(table, player_pos, action_history, solver_name="v2",
                       device="mps", mc_iters=10000, n_raise_samples=3,
                       mdf_max_fold=0.7, reraise_pct=0.15, reraise_cap=0.10):
    """Compute EV for a player (sampled raise bins for GTO action sampling).

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
        try:
            eq = gpu_equity_fn(hero_t, board_t, n_opponents, n_iters=mc_iters, device=device)
            fold_ev, call_ev, _, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested)
        except Exception:
            return None

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
        allin_idx = bins + 2
        raise_evs.append((allin_idx, allin_frac, allin_ev))

    else:
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
            allin_idx = bins + 2
            raise_evs.append((allin_idx, allin_frac, allin_ev))
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


def _compute_all_action_evs(table, player_pos, action_history, n_actions,
                            solver_name="v2", device="mps", mc_iters=10000,
                            mdf_max_fold=0.7, reraise_pct=0.15, reraise_cap=0.10):
    """Compute EV for ALL possible actions (fold, call, each raise bin, all-in).

    Returns:
        tensor of shape (n_actions,) with EV per action, or None on failure.
        Also returns metadata dict with equity, pot, etc.
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

    bins = table.bins
    max_bet_mult = table.max_bet

    evs = torch.zeros(n_actions, dtype=torch.float)

    # Fold EV
    evs[0] = -hero_invested

    if solver_name == "v1":
        try:
            eq = gpu_equity_fn(hero_t, board_t, n_opponents, n_iters=mc_iters, device=device)
            _, call_ev, _, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested)
        except Exception:
            return None, None
        evs[1] = call_ev

        for b in range(2, bins + 2):
            raise_frac = (b - 1) * max_bet_mult / bins
            actual_raise = min(facing_bet + raise_frac * pot, stack)
            if actual_raise >= stack:
                allin_frac = stack / max(pot + facing_bet, 1e-6)
                _, _, r_ev, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested, raise_frac=allin_frac)
            else:
                _, _, r_ev, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested, raise_frac=raise_frac)
            evs[b] = r_ev

        allin_frac = stack / max(pot + facing_bet, 1e-6)
        _, _, allin_ev, _ = compute_ev_fn(eq, pot, facing_bet, stack, hero_invested, raise_frac=allin_frac)
        evs[bins + 2] = allin_ev

    else:
        expand_range = solver_modules["expand_range"]
        opp_range_types = _build_opponent_ranges(table, player_pos, action_history, solver_modules)

        if not opp_range_types:
            evs[1] = pot - hero_invested
            for b in range(2, n_actions):
                evs[b] = pot - hero_invested
            meta = {"equity": 1.0, "hero_invested": hero_invested,
                    "facing_bet": facing_bet, "stack": stack, "pot": pot}
            return evs, meta

        ev_extra = {"mdf_max_fold": mdf_max_fold, "reraise_pct": reraise_pct, "reraise_cap": reraise_cap}

        try:
            _, call_ev, _, _ = compute_ev_fn(
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
            return None, None
        evs[1] = call_ev

        allin_frac = stack / max(pot + facing_bet, 1e-6)
        allin_ev = None
        try:
            _, _, allin_ev_val, _ = compute_ev_fn(
                hero_t, board_t, opp_range_types,
                pot, facing_bet, stack, hero_invested,
                raise_frac=allin_frac, n_iters=mc_iters, device=device, **ev_extra
            )
            allin_ev = allin_ev_val
        except Exception:
            allin_ev = evs[0]

        for b in range(2, bins + 2):
            raise_frac = (b - 1) * max_bet_mult / bins
            actual_raise = min(facing_bet + raise_frac * pot, stack)
            if actual_raise >= stack:
                evs[b] = allin_ev
            else:
                try:
                    _, _, r_ev, _ = compute_ev_fn(
                        hero_t, board_t, opp_range_types,
                        pot, facing_bet, stack, hero_invested,
                        raise_frac=raise_frac, n_iters=mc_iters, device=device, **ev_extra
                    )
                    evs[b] = r_ev
                except Exception:
                    evs[b] = allin_ev

        evs[bins + 2] = allin_ev

    meta = {
        "equity": eq,
        "hero_invested": hero_invested,
        "facing_bet": facing_bet,
        "stack": stack,
        "pot": pot,
    }
    return evs, meta


def _get_table_display_from_turn(deck, turn):
    """Get 5-element board display from deck and turn number."""
    if turn == 0:
        return [-1] * 5
    elif turn == 1:
        return list(deck[:3]) + [-1, -1]
    elif turn == 2:
        return list(deck[:4]) + [-1]
    else:
        return list(deck[:5])


def _rebuild_events(snapshots, deck, hero_pos, num_players, big_blind, small_blind, n_actions, up_to):
    """Rebuild event sequence from a specific player's perspective.

    Args:
        snapshots: list of dicts with table state at each step
        deck: table deck (constant throughout hand)
        hero_pos: the player whose perspective to use
        num_players: number of players
        big_blind, small_blind: blind sizes
        n_actions: action space size
        up_to: include snapshots[0..up_to] inclusive

    Returns:
        list of event dicts
    """
    hand = deck[5 + 2 * hero_pos: 7 + 2 * hero_pos].tolist()
    events = []
    for snap in snapshots[:up_to + 1]:
        table_cards = _get_table_display_from_turn(deck, snap["turn"])
        action = snap["action"]
        if action is None:
            action = torch.zeros(n_actions, dtype=torch.float32)
        events.append({
            "hand": hand,
            "num_players": num_players,
            "hero_pos": hero_pos,
            "acting_pos": snap["active_pos"],
            "big_blind": float(big_blind),
            "small_blind": float(small_blind),
            "stack": float(snap["credits"][hero_pos]),
            "table": table_cards,
            "pot": float(snap["pot"]),
            "bets": np.copy(snap["bets"]),
            "action": action,
        })
    return events


def generate_scenario(config, device="mps"):
    """Generate multiple samples from one poker hand.

    Simulates a poker hand using Table with GTO-sampled actions.
    At EVERY player's decision point, computes EV for all actions.
    Each decision point becomes a separate training sample with:
      - events: sequence from that player's perspective up to the decision
      - ev_target: max(EVs)
      - action_probs: softmax(EVs / (big_blind * gto_temperature))

    Returns list of scenario dicts, or None on failure.
    """
    mc_iters = config.get("mc_iterations", 10000)
    big_blind = config.get("big_blind", 10)
    small_blind = big_blind // 2
    max_stack = config.get("max_stack", 1000)
    max_players = config.get("max_players", 9)
    temperature = config.get("gto_temperature", 1.0)
    table_bins = config.get("table_bins", 10)
    table_max_bet = config.get("table_max_bet", 2)
    n_actions = table_bins + 3
    solver_name = config.get("solver", "v2")
    mdf_max_fold = config.get("mdf_max_fold", 0.7)
    reraise_pct = config.get("reraise_pct", 0.15)
    reraise_cap = config.get("reraise_cap", 0.10)

    num_players = random.randint(2, max_players)
    start_credits = random.randint(big_blind * 2, max_stack)

    table = Table(
        num_players=num_players,
        bins=table_bins,
        max_bet=table_max_bet,
        start_credits=start_credits,
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.start_table()

    # Save table state snapshots for rebuilding events from any player's perspective
    snapshots = []  # list of {pot, bets, credits, turn, active_pos, action}
    decisions = []  # list of (snapshot_index, player_pos, all_evs, meta)

    # Initial snapshot (no action yet)
    snapshots.append({
        "pot": table.pot,
        "bets": np.copy(table.bets),
        "credits": list(table.credits),
        "turn": table.turn,
        "active_pos": table.active_player,
        "action": None,
    })

    max_actions = 4 * num_players
    action_history = []

    ev_kwargs = {"device": device, "mc_iters": mc_iters}
    if solver_name == "v2":
        ev_kwargs.update({
            "mdf_max_fold": mdf_max_fold,
            "reraise_pct": reraise_pct,
            "reraise_cap": reraise_cap,
        })

    for _ in range(max_actions):
        active_pos = table.active_player

        if table.players_state[active_pos] != 1:
            break

        # Compute ALL action EVs for the active player
        all_evs, meta = _compute_all_action_evs(
            table, active_pos, action_history, n_actions,
            solver_name=solver_name, **ev_kwargs
        )
        if all_evs is None:
            return None

        # Record decision point (snapshot before action = current last snapshot)
        # The decision snapshot is a new one showing the "decision moment"
        decision_snap_idx = len(snapshots)
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": active_pos,
            "action": None,  # no action yet at decision point
        })
        decisions.append((decision_snap_idx, active_pos, all_evs, meta))

        # Sample action from full EVs
        normalizer = big_blind * temperature
        probs = F.softmax(all_evs / normalizer, dim=0)
        choice_idx = torch.multinomial(probs, 1).item()
        action = torch.zeros(n_actions, dtype=torch.float32)
        action[choice_idx] = 1.0

        # Classify action for range narrowing
        if choice_idx == 0:
            act_type = None
        elif choice_idx == 1:
            act_type = "call" if table.turn == 0 else "call_postflop"
        elif choice_idx == table_bins + 2:
            act_type = "3bet" if table.turn == 0 else "bet_postflop"
        else:
            act_type = "open" if table.turn == 0 else "bet_postflop"

        if act_type is not None:
            action_history.append((active_pos, act_type))

        # Execute action on table
        end, several_all_in, state, bet = table.step(action)

        # Save post-action snapshot
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": active_pos,
            "action": action,
        })

        if end or several_all_in:
            break

    if not decisions:
        return None

    # Build one sample per decision point
    normalizer = big_blind * temperature
    results = []
    for snap_idx, player_pos, all_evs, meta in decisions:
        # Rebuild events from this player's perspective, up to the decision
        events = _rebuild_events(
            snapshots, table.deck, player_pos,
            num_players, big_blind, small_blind, n_actions,
            up_to=snap_idx,
        )

        if len(events) < 2:
            continue

        best_ev = float(all_evs.max().item())
        action_probs = F.softmax(all_evs / normalizer, dim=0)

        results.append({
            "events": events,
            "ev_target": best_ev,
            "action_probs": action_probs.tolist(),
            "action_evs": all_evs.tolist(),
            "equity": float(meta["equity"]),
            "pot": float(meta["pot"]),
            "facing_bet": float(meta["facing_bet"]),
            "stack": float(meta["stack"]),
            "hero_invested": float(meta["hero_invested"]),
            "num_players": num_players,
            "n_events": len(events),
        })

    return results if results else None


def _compute_norm_stats(scenarios):
    """Compute mean/std for normalization across all scenarios.

    EV targets are first scaled by (pot + facing_bet) to remove pot-size
    dependence, then z-score stats are computed on the ratio.
    """
    evs, pots, stacks, all_bets, blinds = [], [], [], [], []
    for s in scenarios:
        # Scale EV by pot + facing_bet before computing stats
        denom = max(s.get("pot", 0) + s.get("facing_bet", 0),
                    s["events"][-1]["big_blind"])
        evs.append(s["ev_target"] / denom)
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
    """Normalize ev_target and event scalar inputs in-place.

    EV is first scaled by (pot + facing_bet), then z-scored.
    """
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    stack_m, stack_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]
    blind_m, blind_s = norm_stats["blind_mean"], norm_stats["blind_std"]

    for s in scenarios:
        denom = max(s.get("pot", 0) + s.get("facing_bet", 0),
                    s["events"][-1]["big_blind"])
        s["ev_target"] = (s["ev_target"] / denom - ev_m) / ev_s
        for event in s["events"]:
            event["pot"] = (event["pot"] - pot_m) / pot_s
            event["stack"] = (event["stack"] - stack_m) / stack_s
            event["big_blind"] = (event["big_blind"] - blind_m) / blind_s
            event["small_blind"] = (event["small_blind"] - blind_m) / blind_s
            if isinstance(event["bets"], np.ndarray):
                event["bets"] = (event["bets"] - bets_m) / bets_s
            else:
                event["bets"] = [(b - bets_m) / bets_s for b in event["bets"]]


def load_dataset(dataset_dir, log=None):
    """Load a raw dataset from a directory.

    Args:
        dataset_dir: directory containing dataset.pt
        log: optional logger

    Returns:
        scenarios list, or None if dataset.pt not found
    """
    dataset_path = os.path.join(dataset_dir, "dataset.pt")

    if not os.path.exists(dataset_path):
        return None

    scenarios = torch.load(dataset_path, weights_only=False)
    if log:
        log(f"Loaded dataset from {dataset_dir} ({len(scenarios)} samples)")

    return scenarios


def generate_dataset(config, save_dir, log=None):
    """Generate full dataset of scenarios with both EV and action prob labels.

    Saves raw (unnormalized) data. Normalization is done at training time
    per-agent so each agent can have its own norm_stats.

    Args:
        config: merged config dict (game + solver + scenario-specific)
        save_dir: directory to save dataset.pt
        log: optional logger

    Returns:
        scenarios list
    """
    # Try loading existing dataset first
    existing = load_dataset(save_dir, log=log)
    if existing is not None:
        return existing

    n_scenarios = config.get("n_scenarios", 50000)
    batch_size = config.get("batch_size", 64)
    val_every = config.get("val_every", 10)
    save_every = val_every * batch_size  # incremental save interval (in samples)
    import torch as _torch
    _default_device = "cuda" if _torch.cuda.is_available() else ("mps" if _torch.backends.mps.is_available() else "cpu")
    device = config.get("device", _default_device)
    dataset_path = os.path.join(save_dir, "dataset.pt")

    os.makedirs(save_dir, exist_ok=True)

    if log:
        log(f"Generating {n_scenarios} hands on {device} (saving every {save_every} samples)...")

    scenarios = []
    failed = 0
    last_save_count = 0
    for _ in tqdm(range(n_scenarios), desc="Generating hands"):
        result = generate_scenario(config, device=device)
        if result is not None:
            scenarios.extend(result)
        else:
            failed += 1

        # Incremental save (raw data)
        if len(scenarios) - last_save_count >= save_every:
            torch.save(scenarios, dataset_path)
            last_save_count = len(scenarios)
            if log:
                log(f"  Incremental save: {len(scenarios)} samples")

    if log:
        log(f"Generated {len(scenarios)} samples from {n_scenarios - failed} hands ({failed} failed)")
        if scenarios:
            lengths = [s["n_events"] for s in scenarios]
            log(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    torch.save(scenarios, dataset_path)
    if log:
        log(f"Dataset saved to {dataset_path}")

    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unified GTO training dataset")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--save-dir", default="data/standalone", help="Directory to save dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Merge config sections for standalone use
    train_cfg = {}
    train_cfg.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    train_cfg["solver"] = solver_cfg.pop("type", "v2")
    train_cfg.update(solver_cfg)
    # Use gto_ev_train for n_scenarios etc. (or any scenario config)
    train_cfg.update(config.get("gto_ev_train", {}))

    scenarios, norm_stats = generate_dataset(train_cfg, args.save_dir, log=print)
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        evs = [s["ev_target"] for s in scenarios]
        print(f"Normalized EV range: [{min(evs):.2f}, {max(evs):.2f}]")
        print(f"Action probs sample: {scenarios[0]['action_probs']}")
