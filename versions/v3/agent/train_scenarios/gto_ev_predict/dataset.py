"""
PyTorch Dataset for GTO EV prediction training.

Each scenario contains an event sequence (list of event dicts)
and a scalar EV target.
"""

import torch
from torch.utils.data import Dataset


class GTOEVDataset(Dataset):
    """Dataset of poker event sequences with GTO EV labels.

    Each item is a (events_list, ev_target) pair where events_list
    is a list of event dicts and ev_target is a float.
    """

    def __init__(self, scenarios):
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return scenario["events"], torch.tensor(scenario["ev_target"], dtype=torch.float32)


def batch_collate(batch):
    """Collate that separates event sequences and stacks EV targets."""
    event_sequences = [item[0] for item in batch]
    ev_targets = torch.stack([item[1] for item in batch])
    return event_sequences, ev_targets
