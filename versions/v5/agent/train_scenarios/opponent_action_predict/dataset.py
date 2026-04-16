"""
PyTorch Dataset for opponent action prediction training.

Scenarios store shared events (unmasked, all players' hands).
The dataset expands each scenario into per-observer samples, converting
shared events to standard format with appropriate hand assignment.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


class OpponentActionDataset(Dataset):
    """Dataset of poker scenarios with range-averaged opponent action targets.

    Each raw scenario has shared events and a list of valid hero_positions.
    The dataset pre-expands these into (scenario_idx, hero_pos) pairs so
    each __getitem__ returns one observer's view.

    Args:
        scenarios: list of scenario dicts from generate_opponent
        norm_stats: dict with pot/stack/bets/blind mean/std for z-score normalization.
            If None, no normalization is applied.
    """

    def __init__(self, scenarios, norm_stats=None):
        self.scenarios = scenarios
        self.norm_stats = norm_stats
        # Pre-expand into (scenario_idx, hero_pos) index
        self.indices = []
        for s_idx, s in enumerate(scenarios):
            for hero_pos in s["hero_positions"]:
                self.indices.append((s_idx, hero_pos))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s_idx, hero_pos = self.indices[idx]
        scenario = self.scenarios[s_idx]
        events = self._to_standard(scenario["events"], hero_pos)
        if self.norm_stats is not None:
            _normalize_events_inplace(events, self.norm_stats)
        target = torch.tensor(scenario["opponent_action_probs"], dtype=torch.float32)
        return events, target

    def _to_standard(self, shared_events, hero_pos):
        """Convert shared events to standard per-hero format."""
        # Find hero's hand (last known from any event)
        hero_hand = None
        for e in shared_events:
            hands = e.get("hands", {})
            if hero_pos in hands:
                hero_hand = hands[hero_pos]

        if hero_hand is None:
            # Preflop observer without fixed hand — sample random
            dead = set()
            for e in shared_events:
                for c in e.get("table", []):
                    if c >= 0:
                        dead.add(c)
                for pos, h in e.get("hands", {}).items():
                    dead.update(h)
            available = [c for c in range(52) if c not in dead]
            if len(available) >= 2:
                hero_hand = random.sample(available, 2)
            else:
                hero_hand = [0, 1]

        result = []
        for e in shared_events:
            event = {
                "hand": list(hero_hand),
                "num_players": e["num_players"],
                "hero_pos": hero_pos,
                "acting_pos": e["acting_pos"],
                "big_blind": e["big_blind"],
                "small_blind": e["small_blind"],
                "stack": float(e["stacks"][hero_pos]),
                "table": list(e["table"]),
                "pot": e["pot"],
                "bets": np.copy(e["bets"]),
                "action": e["action"],
            }
            # Add opponent_id for the acting player (for GRU-based opponent modeling)
            opp_ids = e.get("opponent_ids")
            if opp_ids is not None:
                acting = e["acting_pos"]
                event["opponent_id"] = opp_ids.get(acting)
            result.append(event)
        return result


def _normalize_events_inplace(events, norm_stats):
    """Apply z-score normalization to standard events in-place."""
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    stack_m, stack_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]
    blind_m, blind_s = norm_stats["blind_mean"], norm_stats["blind_std"]

    for event in events:
        event["pot"] = (event["pot"] - pot_m) / pot_s
        event["stack"] = (event["stack"] - stack_m) / stack_s
        event["big_blind"] = (event["big_blind"] - blind_m) / blind_s
        event["small_blind"] = (event["small_blind"] - blind_m) / blind_s
        if isinstance(event["bets"], np.ndarray):
            event["bets"] = (event["bets"] - bets_m) / bets_s
        else:
            event["bets"] = [(b - bets_m) / bets_s for b in event["bets"]]


def batch_collate(batch):
    """Collate: separate event sequences and stack target tensors."""
    event_sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])  # (B, n_actions)
    return event_sequences, targets
