from agent.mcts.mcts import MCTS, re_backup_terminals, get_n_distribution
from agent.mcts.game_state import GameState
from agent.mcts.collect import collect_training_data, MCTSTrainingExample, ChainStep


def evaluate_all_terminals(hand_record, agents_by_position, device, config=None):
    """Lazy import to avoid gto_utils import chain at module load time."""
    from agent.mcts.terminal_eval import evaluate_all_terminals as _eval
    return _eval(hand_record, agents_by_position, device, config)
