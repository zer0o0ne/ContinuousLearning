"""
Training data extraction from MCTS trees after a completed hand.

Collects value targets, action targets, and modelling chain targets
from the MCTS trees produced during a hand.
"""

from dataclasses import dataclass, field

from agent.mcts.mcts import get_n_distribution


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
