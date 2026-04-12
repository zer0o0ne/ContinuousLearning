"""
PyTorch Dataset for combined GTO prediction training.

Each scenario contains an event sequence (list of event dicts),
a scalar EV target, and a target probability distribution over actions.
"""

import torch
from torch.utils.data import Dataset


class GTODataset(Dataset):
    """Dataset of poker event sequences with both EV and action probability labels.

    Each item is a (events_list, ev_target, action_probs) triple.
    """

    def __init__(self, scenarios):
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return (
            scenario["events"],
            torch.tensor(scenario["ev_target"], dtype=torch.float32),
            torch.tensor(scenario["action_probs"], dtype=torch.float32),
        )


def batch_collate(batch):
    """Collate that separates event sequences, EV targets, and action probs."""
    event_sequences = [item[0] for item in batch]
    ev_targets = torch.stack([item[1] for item in batch])
    action_probs = torch.stack([item[2] for item in batch])
    return event_sequences, ev_targets, action_probs
