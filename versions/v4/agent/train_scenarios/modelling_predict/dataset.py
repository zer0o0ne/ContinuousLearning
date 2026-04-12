"""
PyTorch Dataset for modelling head training.

Each scenario contains an event sequence (list of event dicts)
and a per-action EV vector (normalized identically to ev_target).
"""

import torch
from torch.utils.data import Dataset


class GTOModellingDataset(Dataset):
    """Dataset of poker event sequences with per-action EV labels.

    Each item is a (events_list, action_evs) pair where events_list
    is a list of event dicts and action_evs is a float vector of size n_actions.
    """

    def __init__(self, scenarios):
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return scenario["events"], torch.tensor(scenario["action_evs"], dtype=torch.float32)


def batch_collate(batch):
    """Collate that separates event sequences and stacks per-action EV targets."""
    event_sequences = [item[0] for item in batch]
    action_evs = torch.stack([item[1] for item in batch])  # (B, n_actions)
    return event_sequences, action_evs
