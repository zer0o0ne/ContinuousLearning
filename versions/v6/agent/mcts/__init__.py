from agent.mcts.mcts import MCTS, re_backup_terminals, get_n_distribution
from agent.mcts.game_state import GameState
from agent.mcts.collect import collect_training_data, MCTSTrainingExample, ChainStep


def evaluate_all_terminals(hand_record, agents_by_position, device, config=None,
                           value_scales_by_position=None):
    """Lazy import to avoid gto_utils import chain at module load time."""
    from agent.mcts.terminal_eval import evaluate_all_terminals as _eval
    return _eval(hand_record, agents_by_position, device, config,
                 value_scales_by_position=value_scales_by_position)


def compute_equity_outcome(hand_record, agents_by_position, device,
                           ref_credits_by_decision, config=None):
    """Lazy import: equity-based realised outcome at the actual final state."""
    from agent.mcts.terminal_eval import compute_equity_outcome as _coe
    return _coe(hand_record, agents_by_position, device,
                ref_credits_by_decision, config)
