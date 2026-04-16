"""
Post-hand terminal node evaluation for MCTS trees.

After a hand completes, evaluates all terminal nodes across all MCTS trees
using MC equity with opponent range narrowing via each player's action head.
Must run post-hand because range narrowing uses ALL agents' action heads.
"""

import torch
import torch.nn.functional as F
import numpy as np

from agent.mcts.mcts import _collect_terminals, re_backup_terminals
from agent.mcts.game_state import GameState
from agent.gto_utils.gpu_solver_v2 import (
    get_position_range, expand_range, narrow_range, gpu_equity_v2,
)


def evaluate_all_terminals(hand_record, agents_by_position, device, config=None):
    """Evaluate terminal nodes across all MCTS trees from one completed hand.

    For fold terminals: deterministic Q (pot distribution).
    For showdown terminals: MC equity with range narrowing via agents' action heads.
    After evaluation, re-backs up Q through each tree.

    Args:
        hand_record: dict with:
            decisions: list of dicts, each with player_pos, action_idx,
                       mcts_root (MCTSNode), events_at_root (list of event dicts),
                       game_state_at_root (GameState)
            deck: np.array(52,) — the shuffled deck
            hero_hands: dict {pos: [c1, c2]} — actual hands dealt
            num_players: int
            big_blind: float
        agents_by_position: dict {pos: ASI} — each player's agent model
        device: torch device string
        config: optional dict with n_equity_iters (default 3000), max_batch (default 128)
    """
    cfg = config or {}
    n_equity_iters = cfg.get("n_equity_iters", 3000)
    max_batch = cfg.get("max_batch", 128)

    decisions = hand_record["decisions"]
    deck = hand_record["deck"]
    hero_hands = hand_record["hero_hands"]
    num_players = hand_record["num_players"]

    # Pre-compute board cards visible at each decision's root
    # (based on the street at that decision point)
    def _board_at_turn(turn):
        if turn == 0:
            return torch.tensor([], dtype=torch.long)
        elif turn == 1:
            return torch.tensor(deck[:3].tolist(), dtype=torch.long)
        elif turn == 2:
            return torch.tensor(deck[:4].tolist(), dtype=torch.long)
        else:
            return torch.tensor(deck[:5].tolist(), dtype=torch.long)

    # Pre-compute per-combo action probs for each decision (for range narrowing)
    # combo_probs[decision_idx] = {pos: (combos, per_combo_probs)} or None
    combo_probs_cache = _precompute_combo_probs(
        decisions, hero_hands, agents_by_position, num_players, device, max_batch)

    # Process each tree
    for dec_idx, decision in enumerate(decisions):
        root = decision["mcts_root"]
        hero_pos = decision["player_pos"]
        root_gs = decision["game_state_at_root"]
        root_turn = root_gs.turn
        hero_hand = hero_hands[hero_pos]
        board_cards = _board_at_turn(root_turn)
        initial_stacks = list(root_gs.credits)
        # Hero invested at root = what hero has already put in before this decision
        # We track invested as initial_stack - current_credits along each path

        terminals = _collect_terminals(root)
        for terminal in terminals:
            # Replay game state to terminal
            path = _path_to_root(terminal)
            gs = root_gs.clone()
            for node in path[1:]:
                if node.action_idx is not None:
                    gs.step(node.action_idx)

            active = [i for i in range(num_players) if gs.players_state[i] >= 0]

            if len(active) <= 1:
                # FOLD terminal: one player wins
                if len(active) == 1:
                    winner = active[0]
                    if winner == hero_pos:
                        terminal.Q = gs.pot - (initial_stacks[hero_pos] - gs.credits[hero_pos])
                    else:
                        terminal.Q = -(initial_stacks[hero_pos] - gs.credits[hero_pos])
                else:
                    terminal.Q = 0.0
            else:
                # SHOWDOWN terminal: MC equity needed
                hero_invested = initial_stacks[hero_pos] - gs.credits[hero_pos]
                hero_cards_t = torch.tensor(hero_hand, dtype=torch.long)

                # Build narrowed opponent ranges
                dead_cards = set(hero_hand)
                dead_cards.update(board_cards.tolist())
                opponent_combos = []

                for opp_pos in active:
                    if opp_pos == hero_pos:
                        continue
                    range_types = get_position_range(opp_pos, num_players)
                    range_types = _narrow_by_real_actions(
                        range_types, opp_pos, dec_idx, decisions, combo_probs_cache)
                    range_types = _narrow_by_simulated_actions(
                        range_types, opp_pos, hero_pos, path, root_gs)
                    combos = expand_range(range_types, dead_cards)
                    if len(combos) == 0:
                        # Fallback: use full position range
                        combos = expand_range(
                            get_position_range(opp_pos, num_players), dead_cards)
                    opponent_combos.append(combos)

                if opponent_combos:
                    equity = gpu_equity_v2(
                        hero_cards_t, board_cards, opponent_combos,
                        n_iters=n_equity_iters, device=device)
                    terminal.Q = equity * gs.pot - hero_invested
                else:
                    terminal.Q = gs.pot - hero_invested  # no opponents

        re_backup_terminals(root)


def _path_to_root(node):
    """Build path from root to node (inclusive)."""
    path = []
    n = node
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path


def _precompute_combo_probs(decisions, hero_hands, agents_by_position, num_players,
                            device, max_batch):
    """Pre-compute P(action | combo) for each decision point.

    For each decision, runs the acting player's action_head on every combo
    in their position range. Used for Bayesian range narrowing.

    Returns:
        list of dicts, one per decision. Each dict maps the acting player's
        position to (combos_list, per_combo_probs tensor).
    """
    cache = [None] * len(decisions)

    for dec_idx, decision in enumerate(decisions):
        acting_pos = decision["player_pos"]
        agent = agents_by_position.get(acting_pos)
        if agent is None:
            continue

        events_template = decision["events_at_root"]
        acting_hand = hero_hands[acting_pos]

        # Get range for this position
        range_types = get_position_range(acting_pos, num_players)
        dead_cards = set()
        # Dead: board cards visible at this decision
        for e in events_template:
            for c in e.get("table", []):
                if isinstance(c, (int, np.integer)) and c >= 0:
                    dead_cards.add(int(c))
        combos = expand_range(range_types, dead_cards)
        if len(combos) == 0:
            continue

        # Batched per-combo inference
        all_probs = []
        for start in range(0, len(combos), max_batch):
            batch_combos = combos[start:start + max_batch]
            batch_events = []
            for c1, c2 in batch_combos.tolist():
                events_copy = [dict(e) for e in events_template]
                for e in events_copy:
                    e["hand"] = [c1, c2]
                batch_events.append(events_copy)

            with torch.no_grad():
                out = agent.forward_batch(batch_events, skip_memory=True, heads={"action"})
                logits = out["action_logits"]
                probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

        per_combo_probs = torch.cat(all_probs, dim=0)  # (n_combos, n_actions)
        cache[dec_idx] = {
            "combos": combos,
            "probs": per_combo_probs,
            "action_idx": decision["action_idx"],
        }

    return cache


def _narrow_by_real_actions(range_types, opp_pos, current_dec_idx, decisions,
                            combo_probs_cache):
    """Narrow an opponent's range using their per-combo action probs from real decisions.

    For each prior decision by this opponent, uses Bayesian update:
    weight(combo) *= P(actual_action | combo) from their action_head.
    Returns the top-weighted hand types.
    """
    # Collect all prior decisions by this opponent
    prior_decisions = []
    for i in range(current_dec_idx):
        if decisions[i]["player_pos"] == opp_pos and combo_probs_cache[i] is not None:
            prior_decisions.append(combo_probs_cache[i])

    if not prior_decisions:
        return range_types

    # Build combo weights via Bayesian updates
    # Reference: latest combo set (narrowest, most dead cards)
    latest = prior_decisions[-1]
    ref_combos = latest["combos"]  # (n_ref, 2)
    n_combos = len(ref_combos)
    weights = torch.ones(n_combos)

    # Build reference lookup: (c1, c2) → index
    ref_lookup = {}
    for i in range(n_combos):
        key = (int(ref_combos[i, 0]), int(ref_combos[i, 1]))
        ref_lookup[key] = i

    for cached in prior_decisions:
        action_idx = cached["action_idx"]
        combos = cached["combos"]    # (n_cached, 2)
        probs = cached["probs"]      # (n_cached, n_actions)
        likelihoods = probs[:, action_idx]  # (n_cached,)

        # Build cached lookup: (c1, c2) → likelihood
        cached_lookup = {}
        for j in range(len(combos)):
            key = (int(combos[j, 0]), int(combos[j, 1]))
            cached_lookup[key] = likelihoods[j].item()

        # Apply to reference combos (unmatched combos keep weight unchanged)
        for key, idx in ref_lookup.items():
            if key in cached_lookup:
                weights[idx] *= cached_lookup[key]

    # Normalize
    total = weights.sum()
    if total < 1e-8:
        return range_types

    weights /= total

    # Keep combos with weight above threshold (top ~70%)
    sorted_w, sorted_idx = weights.sort(descending=True)
    cumsum = sorted_w.cumsum(dim=0)
    cutoff = (cumsum >= 0.95).nonzero(as_tuple=True)[0]
    if len(cutoff) > 0:
        keep = min(len(weights), cutoff[0].item() + 1)
    else:
        keep = len(weights)

    # Map back to hand types (heuristic: keep the top fraction of range)
    frac = keep / max(n_combos, 1)
    n_keep = max(1, int(len(range_types) * frac))
    return range_types[:n_keep]


def _narrow_by_simulated_actions(range_types, opp_pos, hero_pos, path, root_gs):
    """Narrow range by simulated actions in the MCTS tree path.

    Uses heuristic narrow_range() based on action category.
    """
    gs = root_gs.clone()
    for node in path[1:]:
        if node.action_idx is None:
            continue
        # Check if this action was by the opponent
        if gs.active_player == opp_pos and not gs.is_terminal:
            action_cat = _action_to_category(node.action_idx, gs)
            if action_cat is not None:
                range_types = narrow_range(range_types, action_cat)
        gs.step(node.action_idx)

    return range_types


def _action_to_category(action_idx, game_state):
    """Map action index to narrow_range category. Returns None for fold."""
    if action_idx == 0:
        return None  # fold — no range narrowing
    elif action_idx == 1:
        if game_state.turn == 0:
            return "call"
        else:
            return "call_postflop"
    else:
        # Raise or all-in
        if game_state.turn == 0:
            return "3bet"
        else:
            return "bet_postflop"
