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
        # D.5.2: seed by idx so val samples get deterministic hole cards
        hero_hand = self._resolve_hero_hand(scenario["events"], hero_pos,
                                             seed=idx)
        events = self._to_standard(scenario["events"], hero_pos, hero_hand)
        if self.norm_stats is not None:
            _normalize_events_inplace(events, self.norm_stats)
        target = self._observer_target(scenario, hero_hand)
        return events, target

    def _resolve_hero_hand(self, shared_events, hero_pos, seed=None):
        """Hero's hand: the fixed hand if known, else a random board/hand-free one."""
        hero_hand = None
        for e in shared_events:
            hands = e.get("hands", {})
            if hero_pos in hands:
                hero_hand = hands[hero_pos]

        if hero_hand is None:
            # Preflop observer without fixed hand — sample a collision-free hand.
            dead = set()
            for e in shared_events:
                for c in e.get("table", []):
                    if c >= 0:
                        dead.add(c)
                for pos, h in e.get("hands", {}).items():
                    dead.update(h)
            available = [c for c in range(52) if c not in dead]
            if len(available) >= 2:
                # D.5.2: deterministic sampling seeded by index so val
                # evaluations are reproducible across epochs.
                rng = random.Random(seed) if seed is not None else random
                hero_hand = rng.sample(available, 2)
            else:
                hero_hand = [0, 1]
        return list(hero_hand)

    def _observer_target(self, scenario, hero_hand):
        """Per-observer opponent-action target (Audit B.7 — observer blockers).

        Re-averages the actor's stored per-combo distributions, excluding combos
        that contain the observer's own cards (those combos are physically
        impossible from this observer's view, so they should not contribute to
        the actor's range-averaged target). Falls back to the stored un-blocked
        average when per-combo data is absent (legacy datasets) or every combo
        is blocked. Re-averaging over ALL combos reproduces the stored average.
        """
        fc = scenario.get("forward_combos")
        if not fc:
            return torch.tensor(scenario["opponent_action_probs"], dtype=torch.float32)
        pcp = torch.tensor(scenario["per_combo_probs"], dtype=torch.float32)  # (K, n_actions)
        w = torch.tensor(scenario["forward_weights"], dtype=torch.float32)    # (K,)
        blockers = {int(hero_hand[0]), int(hero_hand[1])}
        keep = torch.tensor(
            [c[0] not in blockers and c[1] not in blockers for c in fc],
            dtype=torch.bool,
        )
        w = w * keep.to(w.dtype)
        if float(w.sum()) < 1e-9:
            return torch.tensor(scenario["opponent_action_probs"], dtype=torch.float32)
        w = w / w.sum()
        t = (pcp * w.unsqueeze(-1)).sum(dim=0)
        return t / t.sum().clamp(min=1e-9)

    def _to_standard(self, shared_events, hero_pos, hero_hand):
        """Convert shared events to standard per-hero format (given hero_hand)."""
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
                # B.6.2: carry the full per-position stacks vector through to
                # the standard event (shared events already track it).
                "stacks": list(e["stacks"]),
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
        # B.6.2: normalize the per-position stacks vector on the hero-stack scale.
        if "stacks" in event:
            if isinstance(event["stacks"], np.ndarray):
                event["stacks"] = (event["stacks"] - stack_m) / stack_s
            else:
                event["stacks"] = [(c - stack_m) / stack_s for c in event["stacks"]]


def batch_collate(batch):
    """Collate: separate event sequences and stack target tensors."""
    event_sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])  # (B, n_actions)
    return event_sequences, targets


class _TensorCollate:
    def __init__(self, max_players):
        self.max_players = max_players

    def __call__(self, batch):
        from agent.perception.perception import extract_event_tensors
        event_sequences = [item[0] for item in batch]
        targets = torch.stack([item[1] for item in batch])
        precomputed = extract_event_tensors(event_sequences, self.max_players)
        return event_sequences, precomputed, targets


def make_tensor_collate(max_players):
    """Return a collate that pre-extracts event tensors on CPU."""
    return _TensorCollate(max_players)
