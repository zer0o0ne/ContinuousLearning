"""
Range-based opponent action data generation.

Simulates poker hands with trained agents, tracking per-player hand ranges.
At each decision point:
1. Runs action inference for every hand in the acting player's range
2. Averages the action distributions -> training target for opponent_action_head
3. Picks a concrete hand from range, executes a sampled action on the table
4. Narrows range: removes hands where P(chosen)/P(best) < threshold

Events are stored in shared format (all hands unmasked). During training,
the data loader converts to per-observer format with appropriate masking.

Can be run standalone:
    python -m agent.train_scenarios.generation.generate_opponent --config config.json
"""

import os
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from env.table import Table
from evaluation.evaluate import _normalize_events_inplace
from agent.train_scenarios.generation.generate import _get_raise_sizes, load_dataset
from agent.agent import ASI
from agent.mcts.game_state import GameState
from utils import get_amp_config


# ---------------------------------------------------------------------------
# Agent loading (extended checkpoint search)
# ---------------------------------------------------------------------------

def _find_best_checkpoint(agent_dir):
    """Find best checkpoint, searching gto_predict in addition to probs/ev."""
    for scenario in ("gto_probs_predict", "gto_predict", "modelling_predict", "gto_ev_predict"):
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
    """Load all agents from subdirectories."""
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
            log(f"WARNING: no norm_stats for '{name}', using identity")
            norm_stats = {
                "pot_mean": 0.0, "pot_std": 1.0,
                "stack_mean": 0.0, "stack_std": 1.0,
                "bets_mean": 0.0, "bets_std": 1.0,
                "blind_mean": 0.0, "blind_std": 1.0,
            }

        temperature = ckpt.get("temperature")
        if temperature is None:
            temperature = fallback_temperature

        agents.append({
            "agent": agent,
            "norm_stats": norm_stats,
            "name": name,
            "temperature": temperature,
        })
        log(f"Loaded agent '{name}' from {ckpt_path} (temperature={temperature})")

    return agents


# ---------------------------------------------------------------------------
# Combo utilities
# ---------------------------------------------------------------------------

_ALL_COMBOS = None


def _get_all_combos():
    """All C(52,2) = 1326 two-card combos, cached. Returns list of (c1,c2), c1<c2."""
    global _ALL_COMBOS
    if _ALL_COMBOS is None:
        _ALL_COMBOS = [(c1, c2) for c1 in range(52) for c2 in range(c1 + 1, 52)]
    return _ALL_COMBOS


def _filter_dead(combos, dead_cards):
    """Remove combos containing any dead card."""
    if not dead_cards:
        return combos
    return [(c1, c2) for c1, c2 in combos if c1 not in dead_cards and c2 not in dead_cards]


def _get_board_dead(table):
    """Set of revealed board card IDs."""
    if table.turn == 0:
        return set()
    elif table.turn == 1:
        return set(table.deck[:3].tolist())
    elif table.turn == 2:
        return set(table.deck[:4].tolist())
    else:
        return set(table.deck[:5].tolist())


# ---------------------------------------------------------------------------
# Shared event construction
# ---------------------------------------------------------------------------

def _get_table_display(deck, turn):
    """5-element board display with -1 for unrevealed cards."""
    if turn == 0:
        return [-1] * 5
    elif turn == 1:
        return list(deck[:3]) + [-1, -1]
    elif turn == 2:
        return list(deck[:4]) + [-1]
    else:
        return list(deck[:5])


def _build_shared_events(snapshots, deck, fixed_hands, num_players,
                         big_blind, small_blind, n_actions, up_to,
                         opponent_ids=None):
    """Build shared event sequence (unmasked, all players' hands included).

    Args:
        snapshots: list of state snapshots
        deck: table deck array (for board cards)
        fixed_hands: dict {pos: (c1,c2)} for players with fixed hands
        num_players: number of players
        big_blind, small_blind: blind sizes
        n_actions: action space size
        up_to: include snapshots[0..up_to] inclusive
        opponent_ids: optional dict {pos: str} — persistent player identifiers

    Returns:
        list of shared event dicts
    """
    events = []
    for snap in snapshots[:up_to + 1]:
        table_cards = _get_table_display(deck, snap["turn"])

        action = snap["action"]
        if action is None:
            action = torch.zeros(n_actions, dtype=torch.float32)

        hands = {}
        for pos, hand in fixed_hands.items():
            if hand is not None:
                hands[pos] = list(hand)

        event = {
            "hands": hands,
            "num_players": num_players,
            "acting_pos": snap["active_pos"],
            "big_blind": float(big_blind),
            "small_blind": float(small_blind),
            "pot": float(snap["pot"]),
            "bets": np.copy(snap["bets"]),
            "table": table_cards,
            "action": action,
            "stacks": list(snap["credits"]),
        }
        if opponent_ids is not None:
            event["opponent_ids"] = dict(opponent_ids)
        events.append(event)
    return events


def _shared_to_standard(shared_events, hero_pos, hero_hand):
    """Convert shared events to standard format for a specific hero.

    Creates lightweight copies — only hand/hero_pos/stack differ.
    """
    events = []
    for e in shared_events:
        events.append({
            "hand": list(hero_hand),
            "num_players": e["num_players"],
            "hero_pos": hero_pos,
            "acting_pos": e["acting_pos"],
            "big_blind": e["big_blind"],
            "small_blind": e["small_blind"],
            "stack": float(e["stacks"][hero_pos]),
            "table": list(e["table"]),
            "pot": e["pot"],
            "bets": np.copy(e["bets"]),
            "action": e["action"],
        })
    return events


# ---------------------------------------------------------------------------
# Range inference
# ---------------------------------------------------------------------------

def _compute_range_probs(agent, shared_events, combos, active_pos,
                         norm_stats, temperature, device, n_actions,
                         max_batch, amp_config):
    """Compute action distributions for all combos in range via batched inference.

    Args:
        agent: ASI model in eval mode
        shared_events: shared event sequence up to decision point
        combos: list of (c1,c2) — acting player's live range
        active_pos: acting player's position
        norm_stats: z-score normalization stats from checkpoint
        temperature: softmax temperature
        device: torch device
        n_actions: action space size
        max_batch: max combos per forward pass
        amp_config: (amp_enabled, device_type, amp_dtype)

    Returns:
        avg_probs: (n_actions,) averaged action distribution (the TARGET)
        per_combo_probs: (n_combos, n_actions) per-combo distributions
    """
    amp_enabled, device_type, amp_dtype = amp_config

    # Build and normalize template events (placeholder hand, will be replaced)
    template = _shared_to_standard(shared_events, active_pos, [0, 1])
    _normalize_events_inplace(template, norm_stats)

    all_probs = []

    for start in range(0, len(combos), max_batch):
        batch_combos = combos[start:start + max_batch]

        # Build event sequences — shallow copy template, replace hand
        batch_events = []
        for c1, c2 in batch_combos:
            events = [dict(e) for e in template]
            for e in events:
                e["hand"] = [c1, c2]
            batch_events.append(events)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = agent.forward_batch(batch_events, skip_memory=True, heads={"action"})
            logits = out["action_logits"]  # (batch, n_actions)
            probs = F.softmax(logits / temperature, dim=-1)
            all_probs.append(probs.cpu())

    per_combo_probs = torch.cat(all_probs, dim=0)  # (n_combos, n_actions)
    avg_probs = per_combo_probs.mean(dim=0)          # (n_actions,)

    return avg_probs, per_combo_probs


# ---------------------------------------------------------------------------
# Hand generation
# ---------------------------------------------------------------------------

def generate_opponent_hand(config, agents_list, device, amp_config, player_ids=None):
    """Generate training scenarios from one poker hand with range tracking.

    Args:
        config: merged config dict (game + opponent_data params)
        agents_list: list of agent info dicts from _load_agents
        device: torch device string
        amp_config: (amp_enabled, device_type, amp_dtype)
        player_ids: optional list of persistent player IDs (length >= max_players)

    Returns:
        list of scenario dicts, or None on failure
    """
    big_blind = config.get("big_blind", 10)
    small_blind = big_blind // 2
    max_stack = config.get("max_stack", 1000)
    max_players = config.get("max_players", 9)
    range_threshold = config.get("range_threshold", 0.5)
    max_batch = config.get("max_batch_combos", 256)
    raise_sizes = _get_raise_sizes(config)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3

    min_stack = config.get("min_stack", big_blind * 10)
    num_players = random.randint(2, max_players)
    start_credits = random.randint(min_stack, max_stack)

    # Build opponent_ids mapping for this hand
    if player_ids is not None:
        opponent_ids = {pos: player_ids[pos] for pos in range(num_players)}
    else:
        opponent_ids = {pos: f"anon_{pos}" for pos in range(num_players)}

    table = Table(
        num_players=num_players,
        raise_sizes=raise_sizes,
        start_credits=start_credits,
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.start_table()

    # Select agents for each seat WITH REPLACEMENT
    seated = random.choices(agents_list, k=num_players)

    # Per-player state
    all_combos = _get_all_combos()
    player_ranges = {pos: list(all_combos) for pos in range(num_players)}
    fixed_hands = {pos: None for pos in range(num_players)}

    # Snapshots for event reconstruction
    snapshots = [{
        "pot": table.pot,
        "bets": np.copy(table.bets),
        "credits": list(table.credits),
        "turn": table.turn,
        "active_pos": table.active_player,
        "action": None,
    }]

    scenarios = []
    max_actions = 4 * num_players

    for _ in range(max_actions):
        active_pos = table.active_player

        if table.players_state[active_pos] != 1:
            break

        agent_info = seated[active_pos]
        agent = agent_info["agent"]

        # Filter range by dead board cards
        dead = _get_board_dead(table)
        live_range = _filter_dead(player_ranges[active_pos], dead)

        if not live_range:
            break

        # Decision snapshot (before action)
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": active_pos,
            "action": None,
        })
        decision_snap_idx = len(snapshots) - 1

        # Build shared events up to this decision
        shared_events = _build_shared_events(
            snapshots, table.deck, fixed_hands, num_players,
            big_blind, small_blind, n_actions, up_to=decision_snap_idx,
            opponent_ids=opponent_ids,
        )

        if len(shared_events) < 2:
            break

        # Compute action distributions for entire range
        avg_probs, per_combo_probs = _compute_range_probs(
            agent, shared_events, live_range, active_pos,
            agent_info["norm_stats"], agent_info["temperature"],
            device, n_actions, max_batch, amp_config,
        )

        # Mask out unplayable actions (dominated fold / raises that collapse
        # to call or all-in) so the training target and the sampled action
        # both respect the playable set. Re-normalize per row.
        gs = GameState.from_table(table, active_pos)
        legal_mask = torch.tensor(
            gs.get_legal_action_mask(n_actions), dtype=torch.bool,
        )
        per_combo_probs = per_combo_probs.masked_fill(~legal_mask, 0.0)
        row_sums = per_combo_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        per_combo_probs = per_combo_probs / row_sums
        avg_probs = per_combo_probs.mean(dim=0)

        # Fix hand at first action of this player
        if fixed_hands[active_pos] is None:
            combo_idx = random.randint(0, len(live_range) - 1)
            fixed_hands[active_pos] = live_range[combo_idx]

        # Get probs for the fixed hand
        fixed = fixed_hands[active_pos]
        try:
            fixed_idx = live_range.index(fixed)
        except ValueError:
            # Fixed hand removed by dead cards — degenerate
            break

        fixed_probs = per_combo_probs[fixed_idx]

        # Sample action from fixed hand's distribution
        chosen_action = torch.multinomial(fixed_probs, 1).item()

        action = torch.zeros(n_actions, dtype=torch.float32)
        action[chosen_action] = 1.0

        # Active observers (non-folded, non-acting)
        hero_positions = [
            pos for pos in range(num_players)
            if pos != active_pos and table.players_state[pos] >= 0
        ]

        facing_bet = max(0, table.high_bet - table.bets[active_pos])

        # Record scenario
        scenarios.append({
            "events": copy.deepcopy(shared_events),
            "opponent_action_probs": avg_probs.tolist(),
            "acting_pos": active_pos,
            "hero_positions": hero_positions,
            "num_players": num_players,
            "pot": float(table.pot),
            "facing_bet": float(facing_bet),
            "n_events": len(shared_events),
            "range_size": len(live_range),
            "opponent_ids": dict(opponent_ids),
        })

        # Execute action on table
        end, several_all_in, state, bet = table.step(action)

        # Post-action snapshot
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": table.active_player,
            "action": action,
        })

        # Narrow range: remove combos where P(chosen)/P(best) < threshold
        if range_threshold > 0:
            new_range = []
            for i, combo in enumerate(live_range):
                best_prob = per_combo_probs[i].max().item()
                action_prob = per_combo_probs[i][chosen_action].item()
                if best_prob <= 0 or action_prob / best_prob >= range_threshold:
                    new_range.append(combo)
            player_ranges[active_pos] = new_range

        if end or several_all_in:
            break

    return scenarios if scenarios else None


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_opponent_dataset(config, save_dir, device, log):
    """Generate full opponent action dataset.

    Loads trained agents, plays them against each other with range tracking,
    and records range-averaged action distributions as training targets.

    Args:
        config: full config dict
        save_dir: directory to save dataset.pt
        device: torch device string
        log: logger callable

    Returns:
        list of scenario dicts
    """
    existing = load_dataset(save_dir, log=log)
    if existing is not None:
        return existing

    opp_cfg = config.get("opponent_data", {})
    game_cfg = config.get("game", {})

    agents_dir = opp_cfg.get("agents_dir", "")
    if agents_dir and not os.path.isabs(agents_dir):
        version = os.path.basename(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
        agents_dir = os.path.join(project_root, "data", version, agents_dir)

    n_hands = opp_cfg.get("n_hands", 5000)
    fallback_temperature = opp_cfg.get("action_temperature", 0.3)
    save_every = opp_cfg.get("save_every_hands", 500)

    # Merge game params into generation config
    gen_cfg = {}
    gen_cfg.update(game_cfg)
    gen_cfg.update(opp_cfg)

    amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)
    amp_config = (amp_enabled, device_type, amp_dtype)

    log("=== Opponent Action Data Generation ===")
    log(f"Loading agents from {agents_dir}")

    agents_list = _load_agents(agents_dir, config, device, log, fallback_temperature)
    if not agents_list:
        log("No agents loaded. Aborting.")
        return []

    log(f"Loaded {len(agents_list)} agents: {[a['name'] for a in agents_list]}")
    log(f"Generating {n_hands} hands, threshold={gen_cfg.get('range_threshold', 0.5)}, "
        f"max_batch={gen_cfg.get('max_batch_combos', 256)}")

    # Persistent player IDs — simulate realistic table dynamics
    max_players = gen_cfg.get("max_players", 9)
    n_player_pool = opp_cfg.get("n_player_pool", max_players * 3)
    swap_prob = opp_cfg.get("player_swap_prob", 0.05)
    player_pool = [f"p_{i}" for i in range(n_player_pool)]
    table_roster = list(player_pool[:max_players])
    log(f"Player pool: {n_player_pool} IDs, swap_prob={swap_prob}")

    os.makedirs(save_dir, exist_ok=True)
    dataset_path = os.path.join(save_dir, "dataset.pt")

    scenarios = []
    failed = 0

    for hand_i in tqdm(range(n_hands), desc="Generating opponent data"):
        # Simulate player rotation — occasionally swap seat identities
        for pos in range(max_players):
            if random.random() < swap_prob:
                table_roster[pos] = random.choice(player_pool)

        result = generate_opponent_hand(gen_cfg, agents_list, device, amp_config,
                                        player_ids=table_roster)
        if result is not None:
            for s in result:
                s["hand_id"] = hand_i
            scenarios.extend(result)
        else:
            failed += 1

        if (hand_i + 1) % save_every == 0:
            torch.save(scenarios, dataset_path)
            log(f"  Incremental save: {len(scenarios)} scenarios ({hand_i + 1} hands)")

    log(f"Generated {len(scenarios)} scenarios from {n_hands - failed} hands "
        f"({failed} failed)")

    if scenarios:
        range_sizes = [s["range_size"] for s in scenarios]
        log(f"Range sizes: min={min(range_sizes)}, max={max(range_sizes)}, "
            f"avg={sum(range_sizes) / len(range_sizes):.0f}")

    torch.save(scenarios, dataset_path)
    log(f"Dataset saved to {dataset_path}")

    return scenarios


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from utils import Logger

    parser = argparse.ArgumentParser(
        description="Generate range-based opponent action dataset")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    ver = os.path.basename(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    proj_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    exp_name = cfg.get("name", "default")

    opp_save = cfg.get("opponent_data", {}).get("save_dir", "")
    if opp_save and os.path.isabs(opp_save):
        save_dir = opp_save
    else:
        save_dir = os.path.join(proj_root, "data", ver, exp_name, "opponent_dataset")

    logger = Logger(os.path.join(proj_root, "data", ver, exp_name))
    generate_opponent_dataset(cfg, save_dir, dev, logger)
