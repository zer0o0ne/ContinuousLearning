"""
PyTorch Dataset for GTO action probability prediction training.

Each scenario contains an event sequence (list of event dicts)
and a target probability distribution over actions.
"""

import torch
from torch.utils.data import Dataset


class GTOProbsDataset(Dataset):
    """Dataset of poker event sequences with GTO action probability labels.

    Each item is a (events_list, action_probs) pair where events_list
    is a list of event dicts and action_probs is a probability distribution
    over all possible actions.
    """

    def __init__(self, scenarios):
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return scenario["events"], torch.tensor(scenario["action_probs"], dtype=torch.float32)


def batch_collate(batch):
    """Collate that separates event sequences and stacks action probability targets."""
    event_sequences = [item[0] for item in batch]
    action_probs = torch.stack([item[1] for item in batch])  # (B, n_actions)
    return event_sequences, action_probs
