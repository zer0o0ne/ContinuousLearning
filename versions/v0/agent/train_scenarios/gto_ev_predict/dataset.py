"""
PyTorch Dataset for GTO EV prediction training.

Loads pre-generated scenarios from a .pt file. Each scenario contains
an env_state dict and a scalar EV target.
"""

import torch
from torch.utils.data import Dataset


class GTOEVDataset(Dataset):
    """Dataset of poker scenarios with GTO EV labels.

    Each item is a (env_state, ev_target) pair where env_state is a dict
    compatible with StateEmbedder and ev_target is a float.
    """

    def __init__(self, scenarios):
        """
        Args:
            scenarios: list of dicts with 'env_state' and 'ev_target' keys
        """
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return scenario["env_state"], torch.tensor(scenario["ev_target"], dtype=torch.float32)


def no_collate(batch):
    """Custom collate that doesn't stack env_states (they're dicts with variable structure).

    Returns list of (env_state, ev_target) pairs.
    """
    return batch


def batch_collate(batch):
    """Collate that separates env_states and stacks EV targets for batch forward."""
    env_states = [item[0] for item in batch]
    ev_targets = torch.stack([item[1] for item in batch])
    return env_states, ev_targets
