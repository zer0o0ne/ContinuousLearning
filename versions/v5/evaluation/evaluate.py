"""
Agent-vs-agent evaluation module.

Loads trained agent checkpoints, seats them at a poker table, plays N hands,
and reports BB/100 for each agent.

When len(agents) > num_players, agents rotate in/out so each gets equal
playtime and no agent occupies more than one seat simultaneously.

Uses multi-table batching: runs multiple tables in parallel and batches
agent decisions across tables for efficient GPU utilization.

Can be run standalone:
    python -m evaluation.evaluate --config config.json
"""

import os
import json
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from agent.agent import ASI
from agent.perception.opponent_embeddings import OpponentEmbeddingTable
from env.table import Table
from utils import get_amp_config

# NOTE: agent.mcts.game_state and agent.mcts.mcts are imported lazily inside
# functions to avoid a circular import: agent/mcts/__init__.py loads collect,
# which imports from evaluation.evaluate.


def _find_best_checkpoint(agent_dir):
    """Find the best checkpoint in an agent directory.

    Prefers gto_probs_predict (action head trained) over gto_ev_predict.
    Within each, picks the most recent timestamped subdirectory.
    """
    for scenario in ("mcts_predict", "opponent_action_predict", "modelling_predict",
                      "gto_probs_predict", "gto_predict", "gto_ev_predict"):
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


def _resolve_checkpoint_path(path):
    """Resolve a user-supplied path to a concrete .pt file.

    Accepted forms:
      - file ending with .pt → returned as-is
      - directory with best.pt directly inside → that best.pt
      - agent directory (contains scenario subdirs like mcts_predict/) →
            _find_best_checkpoint priority search
      - scenario directory (contains timestamped subdirs with best.pt) →
            pick latest timestamp
    Returns absolute path or None if nothing found.
    """
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return path if path.endswith(".pt") else None
    # Direct best.pt
    direct = os.path.join(path, "best.pt")
    if os.path.isfile(direct):
        return direct
    # Agent dir (priority search across known scenarios)
    found = _find_best_checkpoint(path)
    if found:
        return found
    # Scenario dir: any subdir/best.pt? pick latest by name
    candidates = []
    for sub in os.listdir(path):
        sub_path = os.path.join(path, sub)
        if not os.path.isdir(sub_path):
            continue
        ckpt = os.path.join(sub_path, "best.pt")
        if os.path.isfile(ckpt):
            candidates.append((sub, ckpt))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def _build_agent_bundle(name, ckpt_path, config, device, log,
                        fallback_temperature, entry_temperature_override=None,
                        use_opp_emb=False, use_mcts=False, mcts_cfg=None):
    """Build a single agent bundle from a checkpoint path.

    Temperature precedence: checkpoint > entry override > fallback.

    Per-agent options:
      use_opp_emb: attach a fresh OpponentEmbeddingTable (only effective if
        the agent's perception has opp_emb_enabled).
      use_mcts: attach an MCTS instance for decision-making. When True, the
        agent's decisions are produced via MCTS.search instead of action-head
        sampling. The opp_table (if any) is passed into MCTS so opponent
        embeddings work at the search root.
    """
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

    ckpt_temp = ckpt.get("temperature")
    if ckpt_temp is not None:
        temperature = float(ckpt_temp)
    elif entry_temperature_override is not None:
        temperature = float(entry_temperature_override)
    else:
        log(f"WARNING: no temperature in checkpoint for '{name}', using config fallback ({fallback_temperature})")
        temperature = float(fallback_temperature)

    opp_table = None
    if use_opp_emb and agent.perception.opp_emb_enabled:
        opp_table = OpponentEmbeddingTable(agent.perception.d_model)

    mcts = None
    if use_mcts:
        from agent.mcts.mcts import MCTS  # lazy import — see top-of-module note
        mcts = MCTS(agent, device, mcts_cfg or {}, opponent_emb_table=opp_table)

    return {
        "agent": agent,
        "norm_stats": norm_stats,
        "name": name,
        "temperature": temperature,
        "stack": 0.0,
        "opp_table": opp_table,
        "mcts": mcts,
        "use_opp_emb": opp_table is not None,
        "use_mcts": mcts is not None,
    }


def _resolve_agent_path(path, project_root, version):
    if not path:
        raise ValueError("agent entry missing 'path'")
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, "data", version, path)


def _load_agents_from_list(agent_entries, config, device, log, fallback_temperature):
    """Load agents from an explicit list (eval_pipeline-style).

    Each entry: {
        "name": str,
        "path": str,
        "action_temperature": float (optional),
        "use_opponent_embedding": bool (optional, default False),
        "use_mcts": bool (optional, default False),
    }
    """
    here = os.path.dirname(os.path.abspath(__file__))
    version = os.path.basename(os.path.abspath(os.path.join(here, "..")))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    mcts_cfg = config.get("mcts", {})

    agents = []
    for entry in agent_entries:
        name = entry.get("name") or os.path.basename(entry.get("path", "").rstrip("/"))
        path = _resolve_agent_path(entry["path"], project_root, version)
        ckpt_path = _resolve_checkpoint_path(path)
        if ckpt_path is None:
            log(f"WARNING: no checkpoint for agent '{name}' at {path}, skipping")
            continue
        use_opp_emb = bool(entry.get("use_opponent_embedding", False))
        use_mcts = bool(entry.get("use_mcts", False))
        bundle = _build_agent_bundle(
            name, ckpt_path, config, device, log,
            fallback_temperature,
            entry_temperature_override=entry.get("action_temperature"),
            use_opp_emb=use_opp_emb,
            use_mcts=use_mcts,
            mcts_cfg=mcts_cfg,
        )
        agents.append(bundle)
        log(f"Loaded agent '{name}' from {ckpt_path} "
            f"(temperature={bundle['temperature']}, "
            f"opp_emb={bundle['use_opp_emb']}, mcts={bundle['use_mcts']})")
    return agents


def _load_agents(agents_dir, config, device, log, fallback_temperature,
                 use_opp_emb=False):
    """Load all agents from subdirectories (legacy dir-based path).

    Applies the same global flag `use_opp_emb` to every agent loaded.
    No MCTS in this path (dir-based config doesn't have per-agent flags).
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

    mcts_cfg = config.get("mcts", {})
    agents = []
    for name in agent_names:
        agent_path = os.path.join(agents_dir, name)
        ckpt_path = _find_best_checkpoint(agent_path)
        if ckpt_path is None:
            log(f"WARNING: no checkpoint found for agent '{name}', skipping")
            continue
        bundle = _build_agent_bundle(
            name, ckpt_path, config, device, log, fallback_temperature,
            use_opp_emb=use_opp_emb,
            use_mcts=False,
            mcts_cfg=mcts_cfg,
        )
        agents.append(bundle)
        log(f"Loaded agent '{name}' from {ckpt_path} "
            f"(temperature={bundle['temperature']}, opp_emb={bundle['use_opp_emb']})")

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


def _rebuild_events(snapshots, deck, hero_pos, num_players, big_blind, small_blind, n_actions, up_to,
                    seated_names=None):
    """Rebuild event sequence from a player's perspective using snapshots.

    Args:
        seated_names: optional list of agent names by seat position. When provided,
            each event gets an 'opponent_id' field with the name of the acting agent.
    """
    hand = deck[5 + 2 * hero_pos: 7 + 2 * hero_pos].tolist()
    events = []
    for snap in snapshots[:up_to + 1]:
        table_cards = _get_table_display_from_turn(deck, snap["turn"])
        action = snap["action"]
        if action is None:
            action = torch.zeros(n_actions, dtype=torch.float32)
        event = {
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
        }
        if seated_names is not None:
            event["opponent_id"] = seated_names[snap["active_pos"]]
        events.append(event)
    return events


def _normalize_events_inplace(events, norm_stats):
    """Apply z-score normalization to events in-place."""
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


def _init_table_state(agents, agent_queue, num_players, raise_sizes,
                      big_blind, small_blind, n_actions, player_ids=None):
    """Initialize a new hand at a table. Returns table state dict."""
    seated_indices = [agent_queue[i] for i in range(num_players)]
    seated = [agents[idx] for idx in seated_indices]

    table = Table(
        num_players=num_players,
        raise_sizes=raise_sizes,
        start_credits=int(max(a["stack"] for a in seated)),
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.credits = [a["stack"] for a in seated]
    table.start_table()

    pre_credits = [seated[i]["stack"] for i in range(num_players)]

    snapshots = [{
        "pot": table.pot,
        "bets": np.copy(table.bets),
        "credits": list(table.credits),
        "turn": table.turn,
        "active_pos": table.active_player,
        "action": None,
    }]

    seated_names = [a["name"] for a in seated]

    return {
        "table": table,
        "seated": seated,
        "seated_names": seated_names,
        "player_ids": player_ids,
        "pre_credits": pre_credits,
        "snapshots": snapshots,
        "action_step": 0,
        "finished": False,
    }


def run_evaluation(config, device, log, results_dir_override=None):
    """Run agent-vs-agent evaluation with multi-table batching.

    Runs multiple tables in parallel, batching agent forward passes across
    tables for efficient GPU utilization.

    Two ways to specify agents (mutually exclusive — list takes priority):
      * eval_cfg["agents"]: explicit list of {"name", "path", "action_temperature"}
      * eval_cfg["agents_dir"]: directory whose subdirs are agent names
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
    from agent.train_scenarios.generation.generate import _get_raise_sizes
    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3
    n_tables = eval_cfg.get("n_tables", 16)
    # AMP config
    amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)

    log("=== Evaluation ===")
    agent_entries = eval_cfg.get("agents")
    if isinstance(agent_entries, list) and agent_entries:
        log(f"Loading {len(agent_entries)} agents from explicit list")
        agents = _load_agents_from_list(agent_entries, config, device, log,
                                        fallback_temperature)
    else:
        # Legacy dir-based path: apply global use_opponent_emb to all agents
        global_use_opp_emb = bool(eval_cfg.get("use_opponent_emb", False))
        log(f"Loading agents from {agents_dir} (global use_opponent_emb={global_use_opp_emb})")
        agents = _load_agents(agents_dir, config, device, log,
                              fallback_temperature,
                              use_opp_emb=global_use_opp_emb)
    if not agents:
        log("No agents loaded. Aborting evaluation.")
        return

    cfg_num_players = eval_cfg.get("num_players", 0)
    if cfg_num_players <= 0:
        num_players = len(agents)
    else:
        num_players = min(cfg_num_players, len(agents))
    num_players = max(2, num_players)

    for a in agents:
        a["stack"] = float(start_stack)

    n_opp_emb = sum(1 for a in agents if a.get("opp_table") is not None)
    n_mcts = sum(1 for a in agents if a.get("mcts") is not None)
    log(f"Per-agent opponent_embedding active for {n_opp_emb}/{len(agents)} agent(s)")
    log(f"Per-agent MCTS active for {n_mcts}/{len(agents)} agent(s)")

    # Player identity pool + swap config (for opponent embedding tracking)
    opp_data_cfg = config.get("opponent_data", {})
    swap_prob = eval_cfg.get("player_swap_prob", opp_data_cfg.get("player_swap_prob", 0.02))
    n_player_pool = eval_cfg.get("n_player_pool",
                                  opp_data_cfg.get("n_player_pool", num_players * 3))
    player_pool = [f"p_{i}" for i in range(n_player_pool)]
    if n_opp_emb > 0:
        log(f"Player pool: {n_player_pool} IDs, swap_prob={swap_prob}")

    n_tables = min(n_tables, n_hands)

    log(f"Table: {num_players} seats, {len(agents)} agents, {n_hands} hands, {n_tables} parallel tables")
    log(f"BB={big_blind}, start_stack={start_stack}, rebuy=[{min_rebuy},{max_rebuy}], cap={max_stack_cap}")
    if amp_enabled:
        log(f"AMP enabled: {device_type}, dtype={amp_dtype}")
    agent_summary = [(a["name"], a["temperature"]) for a in agents]
    log(f"Agents: {agent_summary}")

    agent_queue = deque(range(len(agents)))

    # History save path
    if results_dir_override:
        results_dir = results_dir_override
    else:
        version = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        exp_name = config.get("name", "default")
        results_dir = os.path.join(project_root, "data", version, exp_name, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, f"{log.init_time}.pt")

    # Tracking — two parallel sets of arrays:
    #   * raw_*       — actual chip delta from Table.credits (the env may leak
    #                   chips through showdown / multi-street accounting; sum
    #                   across agents over many hands can drift negative).
    #   * total_*     — leak-corrected: each hand's chip leak is split equally
    #                   across seated agents so per-hand sum is exactly zero.
    #                   This is what we use for BB/100 — the metric reflects
    #                   pure win/loss between agents, not env artefacts.
    raw_profit = {a["name"]: 0.0 for a in agents}
    raw_per_hand_chips = {a["name"]: [] for a in agents}
    total_profit = {a["name"]: 0.0 for a in agents}
    per_hand_chips = {a["name"]: [] for a in agents}
    hands_count = {a["name"]: 0 for a in agents}
    # Per-agent action histograms for analytics:
    #   action_hist[name] = np.array(n_actions,) — overall counts
    #   action_hist_by_street[name] = np.array((4, n_actions),) — per-street
    action_hist = {a["name"]: np.zeros(n_actions, dtype=np.int64) for a in agents}
    action_hist_by_street = {a["name"]: np.zeros((4, n_actions), dtype=np.int64)
                              for a in agents}
    history = {"bb100": [], "profit": [], "hands": []}

    MAX_ACTIONS = 10000
    hands_completed = 0
    hands_started = 0
    last_log_at = 0

    # Initialize all tables
    table_states = []
    for _ in range(n_tables):
        if hands_started >= n_hands:
            break
        init_pids = [random.choice(player_pool) for _ in range(num_players)]
        ts = _init_table_state(agents, agent_queue, num_players, raise_sizes,
                               big_blind, small_blind, n_actions,
                               player_ids=init_pids)
        table_states.append(ts)
        hands_started += 1

    pbar = tqdm(total=n_hands, desc="Evaluating")

    dummy_action = torch.zeros(n_actions, dtype=torch.float32)

    while hands_completed < n_hands and table_states:
        # --- Phase 1: Advance all-in runouts (no model call needed) ---
        for ts in table_states:
            if ts["finished"]:
                continue
            while ts["table"].several_all_in and ts["action_step"] < MAX_ACTIONS:
                end, several_all_in, state, bet = ts["table"].step(dummy_action)
                ts["action_step"] += 1
                if end:
                    ts["finished"] = True
                    break

        # --- Phase 2: Collect pending decisions ---
        # Each active table contributes at most one decision (the current active player)
        pending = []  # list of (table_idx, agent_info, events)
        for ti, ts in enumerate(table_states):
            if ts["finished"]:
                continue
            if ts["action_step"] >= MAX_ACTIONS:
                log(f"WARNING: table {ti} stuck after {MAX_ACTIONS} actions, skipping hand")
                ts["finished"] = True
                continue

            table = ts["table"]
            active_pos = table.active_player

            if table.players_state[active_pos] != 1:
                ts["finished"] = True
                continue

            # Decision snapshot
            ts["snapshots"].append({
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": active_pos,
                "action": None,
            })

            agent_info = ts["seated"][active_pos]
            events = _rebuild_events(
                ts["snapshots"], table.deck, active_pos,
                num_players, big_blind, small_blind, n_actions,
                up_to=len(ts["snapshots"]) - 1,
                seated_names=ts["player_ids"],
            )
            decision_street = int(table.turn)
            pending.append((ti, agent_info, events, decision_street))

        # --- Phase 3: Decisions ---
        # Split pending into MCTS (one-by-one) vs non-MCTS (batched by model).
        # MCTS agents can't share batch with non-MCTS — MCTS.search runs its
        # own internal forward passes and returns an action_idx directly.
        if pending:
            actions_out = [None] * len(pending)

            mcts_pending = []   # decisions handled via MCTS.search
            batch_pending = []  # decisions handled via action-head sampling
            for pidx, (ti, agent_info, events, street) in enumerate(pending):
                if agent_info.get("mcts") is not None:
                    mcts_pending.append((pidx, ti, agent_info, events, street))
                else:
                    batch_pending.append((pidx, ti, agent_info, events, street))

            with torch.no_grad():
                # MCTS path — sequential per decision (each search has its own
                # forward passes; cannot batch across tables).
                if mcts_pending:
                    from agent.mcts.game_state import GameState  # lazy import
                for pidx, ti, agent_info, events, street in mcts_pending:
                    table = table_states[ti]["table"]
                    active_pos = table.active_player
                    _normalize_events_inplace(events, agent_info["norm_stats"])
                    gs = GameState.from_table(table, active_pos)
                    action_idx = int(agent_info["mcts"].search([events], gs))
                    action = torch.zeros(n_actions, dtype=torch.float32)
                    action[action_idx] = 1.0
                    actions_out[pidx] = action
                    action_hist[agent_info["name"]][action_idx] += 1
                    action_hist_by_street[agent_info["name"]][street, action_idx] += 1

                # Batched action-head path — group by model_id for efficient
                # forward; opp_table is per-agent so events from agents with
                # different opp_emb settings would need different forward
                # configs, but each agent has a unique model so model_id is a
                # natural grouping unit.
                groups = defaultdict(list)
                for pidx, ti, agent_info, events, street in batch_pending:
                    model_id = id(agent_info["agent"])
                    groups[model_id].append((pidx, ti, agent_info, events, street))

                # Single lazy import outside the per-decision loop.
                if groups:
                    from agent.mcts.game_state import GameState  # noqa: F401 (used below)

                for model_id, group_items in groups.items():
                    agent_model = group_items[0][2]["agent"]
                    # All entries in a group share the same agent → same opp_table
                    opp_table = group_items[0][2].get("opp_table")

                    all_events = []
                    temperatures = []
                    for pidx, ti, agent_info, events, street in group_items:
                        _normalize_events_inplace(events, agent_info["norm_stats"])
                        all_events.append(events)
                        temperatures.append(agent_info["temperature"])

                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                        out = agent_model.forward_batch(
                            all_events, skip_memory=True,
                            skip_opponent_emb=(opp_table is None),
                            opponent_emb_table=opp_table,
                        )
                    all_logits = out["action_logits"]

                    for local_idx, (pidx, ti, agent_info, events, street) in enumerate(group_items):
                        logits = all_logits[local_idx]
                        temp = temperatures[local_idx]
                        # Mask out unplayable actions (dominated fold, raises
                        # that collapse to call/all-in) so the policy can't
                        # sample them at inference.
                        table = table_states[ti]["table"]
                        gs = GameState.from_table(table, table.active_player)
                        legal_mask = torch.tensor(
                            gs.get_legal_action_mask(n_actions),
                            dtype=torch.bool, device=logits.device,
                        )
                        logits = logits.masked_fill(~legal_mask, float("-inf"))
                        probs = F.softmax(logits / temp, dim=0)
                        action_idx = torch.multinomial(probs, 1).item()
                        action = torch.zeros(n_actions, dtype=torch.float32)
                        action[action_idx] = 1.0
                        actions_out[pidx] = action
                        action_hist[agent_info["name"]][action_idx] += 1
                        action_hist_by_street[agent_info["name"]][street, action_idx] += 1

            # Step each table with its action
            for pidx, (ti, agent_info, events, _street) in enumerate(pending):
                ts = table_states[ti]
                table = ts["table"]
                action = actions_out[pidx]

                end, several_all_in, state, bet = table.step(action)
                ts["action_step"] += 1

                # Post-action snapshot
                ts["snapshots"].append({
                    "pot": table.pot,
                    "bets": np.copy(table.bets),
                    "credits": list(table.credits),
                    "turn": table.turn,
                    "active_pos": table.active_player,
                    "action": action,
                })

                if end:
                    ts["finished"] = True

        # --- Phase 4: Finalize completed hands, start replacements ---
        new_table_states = []
        for ts in table_states:
            if not ts["finished"]:
                new_table_states.append(ts)
                continue

            # Profit accounting — two passes:
            #   1) raw_profit: actual chip delta from Table.credits.
            #   2) total_profit: leak-corrected (zero-sum per hand) — used
            #      for BB/100. Any chip delta that vanished inside the env
            #      (showdown / multi-street accounting bugs, max-stack caps,
            #      etc.) is split equally across seated agents so the metric
            #      reflects only actual win/loss transfers between players.
            table = ts["table"]
            seated = ts["seated"]
            pre_credits = ts["pre_credits"]

            raw_deltas = []
            for pos in range(num_players):
                agent_info = seated[pos]
                new_stack = table.credits[pos]
                raw_deltas.append(new_stack - pre_credits[pos])
                agent_info["stack"] = new_stack

            leak = -sum(raw_deltas)  # > 0 if env lost chips this hand
            per_player_correction = leak / num_players if num_players else 0.0

            for pos in range(num_players):
                agent_info = seated[pos]
                raw_d = float(raw_deltas[pos])
                corrected_d = raw_d + per_player_correction
                # Raw (uncorrected): for diagnostics
                raw_profit[agent_info["name"]] += raw_d
                raw_per_hand_chips[agent_info["name"]].append(raw_d)
                # Corrected (zero-sum): primary metric for BB/100
                total_profit[agent_info["name"]] += corrected_d
                per_hand_chips[agent_info["name"]].append(float(corrected_d))
                hands_count[agent_info["name"]] += 1

            # Rebuy busted agents + cap oversize stacks
            for pos in range(num_players):
                agent_info = seated[pos]
                if agent_info["stack"] <= 0:
                    agent_info["stack"] = float(random.randint(min_rebuy, max_rebuy))
                elif agent_info["stack"] > max_stack_cap:
                    agent_info["stack"] = float(random.randint(min_rebuy, max_rebuy))

            hands_completed += 1
            pbar.update(1)

            # Rotate + periodic shuffle
            agent_queue.rotate(-1)
            if hands_completed % num_players == 0:
                queue_list = list(agent_queue)
                random.shuffle(queue_list)
                agent_queue = deque(queue_list)

            # Periodic logging
            if hands_completed >= last_log_at + log_every:
                last_log_at = hands_completed
                snapshot_bb100 = {}
                log(f"  Hand {hands_completed}/{n_hands}")
                for name in sorted(total_profit.keys()):
                    n = hands_count[name]
                    bb100 = (total_profit[name] / big_blind) / (n / 100) if n > 0 else 0.0
                    snapshot_bb100[name] = round(bb100, 4)
                    log(f"    {name}: {bb100:+.2f} BB/100 ({n} hands)")
                history["bb100"].append((hands_completed, snapshot_bb100))
                history["profit"].append((hands_completed, dict(total_profit)))
                history["hands"].append((hands_completed, dict(hands_count)))
                torch.save(history, history_path)

            # Start new hand if quota not reached
            if hands_started < n_hands:
                # Carry forward player IDs with random swaps
                prev_pids = list(ts["player_ids"])
                for pos in range(num_players):
                    if random.random() < swap_prob:
                        prev_pids[pos] = random.choice(player_pool)
                new_ts = _init_table_state(agents, agent_queue, num_players, raise_sizes,
                                           big_blind, small_blind, n_actions,
                                           player_ids=prev_pids)
                new_table_states.append(new_ts)
                hands_started += 1

        table_states = new_table_states

    pbar.close()

    # Final results
    hands_actually_played = sum(hands_count.values()) // num_players
    log(f"\n=== Final Results ({hands_actually_played} hands dealt) ===")
    results = {}
    snapshot_bb100 = {}
    by_name = {a["name"]: a for a in agents}
    for name in sorted(total_profit.keys()):
        n = hands_count[name]
        bb100 = (total_profit[name] / big_blind) / (n / 100) if n > 0 else 0.0
        snapshot_bb100[name] = round(bb100, 4)
        # Per-agent action distribution + per-street breakdown
        hist = action_hist[name]
        total_actions = int(hist.sum())
        action_dist = (hist / max(total_actions, 1)).tolist()
        action_dist_by_street = (
            action_hist_by_street[name]
            / np.maximum(action_hist_by_street[name].sum(axis=1, keepdims=True), 1)
        ).tolist()
        # Aggregate: fold rate, all-in rate
        fold_rate = action_dist[0] if total_actions > 0 else 0.0
        allin_rate = action_dist[n_actions - 1] if total_actions > 0 else 0.0
        # Variance metrics
        chips_arr = np.asarray(per_hand_chips[name], dtype=np.float64) if per_hand_chips[name] else np.zeros(0)
        stderr_bb100 = (
            float(chips_arr.std(ddof=1) / np.sqrt(len(chips_arr)) / big_blind * 100)
            if len(chips_arr) > 1 else 0.0
        )
        # Raw (uncorrected) BB/100 — for diagnostics. Difference between
        # this and the corrected one shows how much the env is leaking chips.
        bb100_raw = ((raw_profit[name] / big_blind) / (n / 100)) if n > 0 else 0.0
        bundle = by_name.get(name, {})
        results[name] = {
            "bb_per_100": round(bb100, 2),
            "bb_per_100_raw": round(bb100_raw, 2),
            "total_profit": round(total_profit[name], 2),
            "total_profit_raw": round(raw_profit[name], 2),
            "hands_played": n,
            "stderr_bb_per_100": round(stderr_bb100, 4),
            "decisions_made": total_actions,
            "fold_rate": round(fold_rate, 4),
            "allin_rate": round(allin_rate, 4),
            "use_opponent_embedding": bool(bundle.get("use_opp_emb", False)),
            "use_mcts": bool(bundle.get("use_mcts", False)),
            "temperature": float(bundle.get("temperature", 0.0)),
            "action_distribution": [round(p, 4) for p in action_dist],
            "action_distribution_by_street": [
                [round(p, 4) for p in row] for row in action_dist_by_street
            ],
            "action_counts_total": [int(c) for c in hist],
            "action_counts_by_street": [
                [int(c) for c in row] for row in action_hist_by_street[name]
            ],
        }
        log(f"  {name}: {bb100:+.2f} BB/100 (corrected; raw={bb100_raw:+.2f})"
            f" — {n} hands, total={total_profit[name]:+.0f} chips,"
            f" fold={fold_rate:.3f}, allin={allin_rate:.3f},"
            f" decisions={total_actions}")

    # Final history snapshot + save
    history["bb100"].append((hands_actually_played, snapshot_bb100))
    history["profit"].append((hands_actually_played, dict(total_profit)))
    history["hands"].append((hands_actually_played, dict(hands_count)))
    history["per_hand_chips"] = per_hand_chips           # leak-corrected
    history["per_hand_chips_raw"] = raw_per_hand_chips   # uncorrected
    history["raw_profit_total"] = dict(raw_profit)
    history["action_hist"] = {n: arr.tolist() for n, arr in action_hist.items()}
    history["action_hist_by_street"] = {
        n: arr.tolist() for n, arr in action_hist_by_street.items()
    }
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
            "n_tables": n_tables,
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
