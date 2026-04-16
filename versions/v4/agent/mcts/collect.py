"""
Training data extraction and self-play collection for MCTS.

Provides:
- collect_training_data(): extract MCTSTrainingExample from hand records
- run_mcts_collection(): play hands with MCTS, produce training data per agent
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from agent.mcts.mcts import MCTS, get_n_distribution
from agent.mcts.game_state import GameState
from env.table import Table
from evaluation.evaluate import (
    _rebuild_events, _normalize_events_inplace, _get_table_display_from_turn,
)


@dataclass
class ChainStep:
    """One step in the modelling chain: action taken + target distribution."""
    action_taken: int
    target_distribution: list
    is_hero: bool  # True → action_head predicts, False → opponent_action_head


@dataclass
class MCTSTrainingExample:
    """Training data from one MCTS tree.

    events: normalized event sequence at this tree's root
    value_target: root Q after terminal re-backup
    action_target: root N-distribution (visit count proportions)
    chain: list of ChainSteps for all subsequent decisions in the hand
    """
    events: list = field(default_factory=list)
    value_target: float = 0.0
    action_target: list = field(default_factory=list)
    chain: list = field(default_factory=list)


def collect_training_data(hand_record, n_actions):
    """Extract training examples from all MCTS trees in a completed hand.

    For each tree (one per decision point), produces an MCTSTrainingExample with:
    - value_target: root.Q (after terminal re-backup reflects MC equity)
    - action_target: normalized visit count distribution at root
    - chain: for each subsequent decision in the hand, the action taken and
      the target distribution (N-distribution from that decision's MCTS tree)

    Args:
        hand_record: dict with "decisions" list — each entry has:
            player_pos, action_idx, mcts_root, events_at_root
        n_actions: int, action space size

    Returns:
        list of MCTSTrainingExample, one per tree
    """
    decisions = hand_record["decisions"]
    examples = []

    for t, decision in enumerate(decisions):
        hero_pos = decision["player_pos"]
        root = decision["mcts_root"]

        # Value target: root Q (should include backed-up terminal equity)
        value_target = root.Q

        # Action target: N-distribution at root
        action_target = get_n_distribution(root, n_actions)

        # Modelling chain: all subsequent decisions in the hand
        chain = []
        for future_dec in decisions[t + 1:]:
            future_root = future_dec["mcts_root"]
            target_dist = get_n_distribution(future_root, n_actions)
            chain.append(ChainStep(
                action_taken=future_dec["action_idx"],
                target_distribution=target_dist,
                is_hero=(future_dec["player_pos"] == hero_pos),
            ))

        examples.append(MCTSTrainingExample(
            events=decision["events_at_root"],
            value_target=value_target,
            action_target=action_target,
            chain=chain,
        ))

    return examples


def run_mcts_collection(agents_list, config, device, log, n_hands):
    """Play hands with MCTS decisions and collect training examples.

    Each agent uses MCTS for its decisions. After each hand, training
    examples are extracted using actual game outcomes as value targets.

    Args:
        agents_list: list of dicts with keys:
            "agent" (ASI), "norm_stats" (dict), "name" (str), "temperature" (float)
        config: full config dict (game, mcts, etc.)
        device: torch device string
        log: logger callable
        n_hands: number of hands to play

    Returns:
        dict mapping agent_name -> list[MCTSTrainingExample]
    """
    from agent.train_scenarios.generation.generate import _get_raise_sizes

    game_cfg = config.get("game", {})
    mcts_cfg = config.get("mcts", {})
    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3
    big_blind = game_cfg.get("big_blind", 10)
    small_blind = big_blind // 2
    num_players = min(len(agents_list), game_cfg.get("max_players", 9))
    num_players = max(2, num_players)
    start_stack = config.get("evaluation", {}).get("start_stack", 1000)

    per_agent_examples = {a["name"]: [] for a in agents_list}
    dummy_action = torch.zeros(n_actions, dtype=torch.float32)
    MAX_ACTIONS = 10000

    log(f"MCTS collection: {n_hands} hands, {num_players} players, "
        f"{mcts_cfg.get('n_simulations', 1000)} simulations/decision")

    for hand_i in tqdm(range(n_hands), desc="MCTS collection"):
        # Seat agents
        seated = random.choices(agents_list, k=num_players)

        table = Table(
            num_players=num_players,
            raise_sizes=raise_sizes,
            start_credits=start_stack,
            big_blind=big_blind,
            small_blind=small_blind,
        )
        table.start_table()
        initial_credits = list(table.credits)

        # Snapshots for event reconstruction
        snapshots = [{
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": table.active_player,
            "action": None,
        }]

        decisions = []
        action_step = 0
        hand_done = False

        while not hand_done and action_step < MAX_ACTIONS:
            # Advance all-in runouts without MCTS
            while table.several_all_in and action_step < MAX_ACTIONS:
                end, _, _, _ = table.step(dummy_action)
                action_step += 1
                if end:
                    hand_done = True
                    break
            if hand_done:
                break

            active_pos = table.active_player
            if table.players_state[active_pos] != 1:
                hand_done = True
                break

            # Pre-decision snapshot
            snapshots.append({
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": active_pos,
                "action": None,
            })

            agent_info = seated[active_pos]

            # Build and normalize events for this agent
            events = _rebuild_events(
                snapshots, table.deck, active_pos,
                num_players, big_blind, small_blind, n_actions,
                up_to=len(snapshots) - 1,
            )
            norm_events = [dict(e) for e in events]  # shallow copy before mutation
            for e in norm_events:
                if isinstance(e["bets"], np.ndarray):
                    e["bets"] = np.copy(e["bets"])
            _normalize_events_inplace(norm_events, agent_info["norm_stats"])

            # MCTS search
            gs = GameState.from_table(table, active_pos)
            mcts = MCTS(agent_info["agent"], device, mcts_cfg)
            agent_info["agent"].eval()
            action_idx = mcts.search([norm_events], gs)

            decisions.append({
                "player_pos": active_pos,
                "action_idx": action_idx,
                "mcts_root": mcts.last_root,
                "events_at_root": norm_events,
            })

            # Step table
            action_vec = torch.zeros(n_actions, dtype=torch.float32)
            action_vec[action_idx] = 1.0
            end, _, _, _ = table.step(action_vec)
            action_step += 1

            # Post-action snapshot
            snapshots.append({
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": table.active_player,
                "action": action_vec,
            })

            if end:
                hand_done = True

        if not decisions:
            continue

        # Actual outcomes (normalized by big blind)
        outcomes = {}
        for pos in range(num_players):
            outcomes[pos] = (table.credits[pos] - initial_credits[pos]) / big_blind

        # Extract training examples and override value_target with actual outcome
        hand_record = {"decisions": decisions}
        examples = collect_training_data(hand_record, n_actions)

        for ex, dec in zip(examples, decisions):
            pos = dec["player_pos"]
            ex.value_target = outcomes[pos]
            agent_name = seated[pos]["name"]
            per_agent_examples[agent_name].append(ex)

    for name, exs in per_agent_examples.items():
        log(f"  {name}: {len(exs)} training examples")

    return per_agent_examples
