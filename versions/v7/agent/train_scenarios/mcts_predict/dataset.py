"""
PyTorch Dataset for MCTS-derived training data.

Each example contains an event sequence, value/action targets,
and a modelling chain of subsequent decisions.
"""

import torch
from torch.utils.data import Dataset


class MCTSDataset(Dataset):
    """Dataset of MCTS training examples.

    Each item is (events, value_target, action_target, chain) where:
    - events: list of event dicts (input to perception)
    - value_target: scalar float
    - action_target: (n_actions,) float tensor
    - chain: list of ChainStep dataclass instances
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            ex.events,
            ex.value_target,
            torch.tensor(ex.action_target, dtype=torch.float32),
            ex.chain,
            # New: tree-terminal targets — list of (action_path, equity_Q).
            # Older pickled examples without this field default to empty via
            # `getattr` so the loader stays backwards-compatible.
            list(getattr(ex, "terminal_targets", []) or []),
        )


def batch_collate(batch):
    """Collate MCTS training examples into a batch.

    Events stay as list-of-lists (variable length).
    Value targets stacked. Action targets stacked.
    Chains stay as list (variable length per example).
    Terminal targets stay as list-of-lists (variable count per example).
    """
    event_sequences = [item[0] for item in batch]
    value_targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    action_targets = torch.stack([item[2] for item in batch])
    chains = [item[3] for item in batch]
    terminal_targets = [item[4] for item in batch]
    return event_sequences, value_targets, action_targets, chains, terminal_targets


class _TensorCollate:
    def __init__(self, max_players):
        self.max_players = max_players

    def __call__(self, batch):
        from agent.perception.perception import extract_event_tensors
        event_sequences = [item[0] for item in batch]
        value_targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        action_targets = torch.stack([item[2] for item in batch])
        chains = [item[3] for item in batch]
        terminal_targets = [item[4] for item in batch]
        precomputed = extract_event_tensors(event_sequences, self.max_players)
        return event_sequences, precomputed, value_targets, action_targets, chains, terminal_targets


def make_tensor_collate(max_players):
    """Return a collate that pre-extracts event tensors on CPU."""
    return _TensorCollate(max_players)
