"""
Agent-vs-agent evaluation module.

Loads trained agent checkpoints, seats them at a poker table, plays N hands,
and reports BB/100 for each agent.

When len(agents) > num_players, agents rotate in/out so each gets equal
playtime and no agent occupies more than one seat simultaneously.

Can be run standalone:
    python -m evaluation.evaluate --config config.json
"""

import os
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from agent.agent import ASI
from env.table import Table


def _find_best_checkpoint(agent_dir):
    """Find the best checkpoint in an agent directory.

    Prefers gto_probs_predict (action head trained) over gto_ev_predict.
    Within each, picks the most recent timestamped subdirectory.
    """
    for scenario in ("gto_probs_predict", "gto_ev_predict"):
        scenario_dir = os.path.join(agent_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        subdirs = sorted(
            [d for d in os.listdir(scenario_dir)
             if os.path.isdir(os.path.join(scenario_dir, d))],
            reverse=True,
        )
        for subdir in subdirs:
            ckpt_path = os.path.join(scenario_dir, subdir, "best.pt")
            if os.path.exists(ckpt_path):
                return ckpt_path
    return None


def _load_agents(agents_dir, config, device, log, fallback_temperature):
    """Load all agents from subdirectories.

    Returns list of dicts: {"agent": ASI, "norm_stats": dict, "name": str,
                            "temperature": float, "stack": float}
    """
    if not os.path.isdir(agents_dir):
        log(f"ERROR: agents_dir not found: {agents_dir}")
        return []

    agent_names = sorted(
        d for d in os.listdir(agents_dir)
        if os.path.isdir(os.path.join(agents_dir, d))
    )

    if not agent_names:
        log(f"ERROR: no agent subdirectories found in {agents_dir}")
        return []

    agents = []
    for name in agent_names:
        agent_path = os.path.join(agents_dir, name)
        ckpt_path = _find_best_checkpoint(agent_path)
        if ckpt_path is None:
            log(f"WARNING: no checkpoint found for agent '{name}', skipping")
            continue

        agent = ASI(log, config)
        agent.set_device(device)
        agent.load_checkpoint(ckpt_path)
        agent.eval()

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        norm_stats = ckpt.get("norm_stats")
        if norm_stats is None:
            log(f"WARNING: no norm_stats in checkpoint for '{name}', using identity normalization")
            norm_stats = {
                "pot_mean": 0.0, "pot_std": 1.0,
                "stack_mean": 0.0, "stack_std": 1.0,
                "bets_mean": 0.0, "bets_std": 1.0,
                "blind_mean": 0.0, "blind_std": 1.0,
            }

        temperature = ckpt.get("temperature")
        if temperature is None:
            log(f"WARNING: no temperature in checkpoint for '{name}', using config fallback ({fallback_temperature})")
            temperature = fallback_temperature

        agents.append({
            "agent": agent,
            "norm_stats": norm_stats,
            "name": name,
            "temperature": temperature,
            "stack": 0.0,  # initialized later
        })
        log(f"Loaded agent '{name}' from {ckpt_path} (temperature={temperature})")

    return agents


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
    """Rebuild event sequence from a player's perspective using snapshots."""
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


def _normalize_events_inplace(events, norm_stats):
    """Apply z-score normalization to events in-place.

    Safe to call on events from _rebuild_events (which already creates
    fresh dicts and np.copy of bets), avoiding an expensive deepcopy.
    """
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    stack_m, stack_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]
    blind_m, blind_s = norm_stats["blind_mean"], norm_stats["blind_std"]

    for event in events:
        event["pot"] = (event["pot"] - pot_m) / pot_s
        event["stack"] = (event["stack"] - stack_m) / stack_s
        event["big_blind"] = (event["big_blind"] - blind_m) / blind_s
        event["small_blind"] = (event["small_blind"] - blind_m) / blind_s
        if isinstance(event["bets"], np.ndarray):
            event["bets"] = (event["bets"] - bets_m) / bets_s
        else:
            event["bets"] = [(b - bets_m) / bets_s for b in event["bets"]]


@torch.no_grad()
def _select_action(agent_info, events, n_actions, device):
    """Get action from agent: normalize → forward → per-agent temperature → sample.

    Returns one-hot action tensor of shape (n_actions,).
    """
    _normalize_events_inplace(events, agent_info["norm_stats"])
    out = agent_info["agent"].forward_batch([events], skip_memory=True)
    logits = out["action_logits"][0]  # (n_actions,)
    probs = F.softmax(logits / agent_info["temperature"], dim=0)
    action_idx = torch.multinomial(probs, 1).item()

    action = torch.zeros(n_actions, dtype=torch.float32)
    action[action_idx] = 1.0
    return action


def run_evaluation(config, device, log):
    """Run agent-vs-agent evaluation.

    Args:
        config: full config dict (with evaluation, game, architecture sections)
        device: torch device string
        log: Logger instance
    """
    eval_cfg = config.get("evaluation", {})
    game_cfg = config.get("game", {})

    agents_dir = eval_cfg.get("agents_dir", "")
    if agents_dir and not os.path.isabs(agents_dir):
        version = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        agents_dir = os.path.join(project_root, "data", version, agents_dir)

    n_hands = eval_cfg.get("n_hands", 10000)
    big_blind = eval_cfg.get("big_blind", game_cfg.get("big_blind", 10))
    small_blind = big_blind // 2
    start_stack = eval_cfg.get("start_stack", 1500)
    min_rebuy = eval_cfg.get("min_rebuy_stack", 500)
    max_rebuy = eval_cfg.get("max_rebuy_stack", 3000)
    max_stack_cap = eval_cfg.get("max_stack_cap", 5000)
    log_every = eval_cfg.get("log_every", 100)
    fallback_temperature = eval_cfg.get("action_temperature", 0.5)
    table_bins = game_cfg.get("table_bins", 50)
    table_max_bet = game_cfg.get("table_max_bet", 5)
    n_actions = table_bins + 3

    log("=== Evaluation ===")
    log(f"Loading agents from {agents_dir}")

    agents = _load_agents(agents_dir, config, device, log, fallback_temperature)
    if not agents:
        log("No agents loaded. Aborting evaluation.")
        return

    # num_players = min(config value, number of loaded agents)
    cfg_num_players = eval_cfg.get("num_players", 0)
    if cfg_num_players <= 0:
        num_players = len(agents)
    else:
        num_players = min(cfg_num_players, len(agents))
    num_players = max(2, num_players)

    # Initialize per-agent stacks
    for a in agents:
        a["stack"] = float(start_stack)

    log(f"Table: {num_players} seats, {len(agents)} agents, {n_hands} hands")
    log(f"BB={big_blind}, start_stack={start_stack}, rebuy=[{min_rebuy},{max_rebuy}], cap={max_stack_cap}")
    agent_summary = [(a["name"], a["temperature"]) for a in agents]
    log(f"Agents: {agent_summary}")

    # Agent rotation queue — rotates so different subsets are seated
    agent_queue = deque(range(len(agents)))

    # History save path
    version = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    exp_name = config.get("name", "default")
    results_dir = os.path.join(project_root, "data", version, exp_name, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, f"{log.init_time}.pt")

    # Tracking
    total_profit = {a["name"]: 0.0 for a in agents}
    hands_count = {a["name"]: 0 for a in agents}
    history = {"bb100": [], "profit": [], "hands": []}

    for hand_idx in tqdm(range(n_hands), desc="Evaluating"):
        # Select which agents play this hand (first num_players from queue)
        seated_indices = [agent_queue[i] for i in range(num_players)]
        seated = [agents[idx] for idx in seated_indices]

        # Load per-agent stacks into a fresh table
        # Table needs a start_credits value; use max current stack as reference
        # but we override credits immediately
        table = Table(
            num_players=num_players,
            bins=table_bins,
            max_bet=table_max_bet,
            start_credits=int(max(a["stack"] for a in seated)),
            big_blind=big_blind,
            small_blind=small_blind,
        )
        # Set per-player credits from agent stacks
        table.credits = [a["stack"] for a in seated]
        table.start_table()

        # Record pre-hand credits (after blinds posted)
        pre_credits = list(table.credits)
        # But we want profit = final - initial (before blinds), so re-record
        pre_credits = [seated[i]["stack"] for i in range(num_players)]

        # Snapshots for event rebuilding
        snapshots = [{
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": table.active_player,
            "action": None,
        }]

        MAX_ACTIONS = 10000
        hand_stuck = False
        for action_step in range(MAX_ACTIONS):
            active_pos = table.active_player

            if not table.several_all_in:
                if table.players_state[active_pos] != 1:
                    break

                # Decision snapshot
                snapshots.append({
                    "pot": table.pot,
                    "bets": np.copy(table.bets),
                    "credits": list(table.credits),
                    "turn": table.turn,
                    "active_pos": active_pos,
                    "action": None,
                })

                agent_info = seated[active_pos]
                events = _rebuild_events(
                    snapshots, table.deck, active_pos,
                    num_players, big_blind, small_blind, n_actions,
                    up_to=len(snapshots) - 1,
                )

                action = _select_action(agent_info, events, n_actions, device)

                end, several_all_in, state, bet = table.step(action)

                # Post-action snapshot
                snapshots.append({
                    "pot": table.pot,
                    "bets": np.copy(table.bets),
                    "credits": list(table.credits),
                    "turn": table.turn,
                    "active_pos": active_pos,
                    "action": action,
                })
            else:
                # All-in runout: step with dummy action to advance streets
                dummy = torch.zeros(n_actions, dtype=torch.float32)
                end, several_all_in, state, bet = table.step(dummy)

            if end:
                break
        else:
            # for-loop exhausted without break — hand did not terminate
            hand_stuck = True

        if hand_stuck:
            log(f"ERROR: hand {hand_idx} did not terminate after {MAX_ACTIONS} actions. "
                f"Table state: turn={table.turn}, pot={table.pot}, "
                f"players_state={table.players_state.tolist()}, "
                f"credits={table.credits}")
            log("Aborting evaluation early.")
            break

        # Write back credits to agent stacks and compute profit
        for pos in range(num_players):
            agent_info = seated[pos]
            new_stack = table.credits[pos]
            profit = new_stack - pre_credits[pos]
            total_profit[agent_info["name"]] += profit
            hands_count[agent_info["name"]] += 1
            agent_info["stack"] = new_stack

        # Rebuy busted agents + cap oversize stacks
        for pos in range(num_players):
            agent_info = seated[pos]
            if agent_info["stack"] <= 0:
                agent_info["stack"] = float(random.randint(min_rebuy, max_rebuy))
            elif agent_info["stack"] > max_stack_cap:
                agent_info["stack"] = float(random.randint(min_rebuy, max_rebuy))

        # Rotate dealer position + periodically shuffle relative seating
        agent_queue.rotate(-1)
        if (hand_idx + 1) % num_players == 0:
            # Full orbit complete — shuffle to break fixed relative positions
            queue_list = list(agent_queue)
            random.shuffle(queue_list)
            agent_queue = deque(queue_list)

        # Periodic logging + history save
        if (hand_idx + 1) % log_every == 0:
            snapshot_bb100 = {}
            log(f"  Hand {hand_idx + 1}/{n_hands}")
            for name in sorted(total_profit.keys()):
                n = hands_count[name]
                bb100 = (total_profit[name] / big_blind) / (n / 100) if n > 0 else 0.0
                snapshot_bb100[name] = round(bb100, 4)
                log(f"    {name}: {bb100:+.2f} BB/100 ({n} hands)")
            history["bb100"].append((hand_idx + 1, snapshot_bb100))
            history["profit"].append((hand_idx + 1, dict(total_profit)))
            history["hands"].append((hand_idx + 1, dict(hands_count)))
            torch.save(history, history_path)

    # Final results
    hands_actually_played = sum(hands_count.values()) // num_players  # total hands dealt
    log(f"\n=== Final Results ({hands_actually_played} hands dealt) ===")
    results = {}
    snapshot_bb100 = {}
    for name in sorted(total_profit.keys()):
        n = hands_count[name]
        bb100 = (total_profit[name] / big_blind) / (n / 100) if n > 0 else 0.0
        snapshot_bb100[name] = round(bb100, 4)
        results[name] = {
            "bb_per_100": round(bb100, 2),
            "total_profit": round(total_profit[name], 2),
            "hands_played": n,
        }
        log(f"  {name}: {bb100:+.2f} BB/100 ({n} hands, total: {total_profit[name]:+.0f} chips)")

    # Final history snapshot + save
    history["bb100"].append((hands_actually_played, snapshot_bb100))
    history["profit"].append((hands_actually_played, dict(total_profit)))
    history["hands"].append((hands_actually_played, dict(hands_count)))
    torch.save(history, history_path)
    log(f"History saved to {history_path}")

    # Save results JSON
    results_json_path = os.path.join(results_dir, f"{log.init_time}.json")
    with open(results_json_path, "w") as f:
        json.dump({
            "n_hands": hands_actually_played,
            "num_players": num_players,
            "num_agents": len(agents),
            "big_blind": big_blind,
            "max_stack_cap": max_stack_cap,
            "agents": results,
            "config": eval_cfg,
        }, f, indent=4)
    log(f"Results saved to {results_json_path}")


if __name__ == "__main__":
    import argparse
    from utils import Logger

    parser = argparse.ArgumentParser(description="Evaluate agents against each other")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    version = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    name = config.get("name", "default")
    base_dir = os.path.join(project_root, "data", version, name)
    log = Logger(base_dir)

    run_evaluation(config, device, log)
