"""
Hand-aware train/val split for poker training datasets.

Ensures all scenarios from the same hand stay in the same split,
preventing data leakage between train and validation sets.
"""

import random
from collections import defaultdict
from torch.utils.data import Subset


def _infer_hand_ids(scenarios):
    """Infer hand boundaries from contiguous scenario data.

    Scenarios from the same hand are always contiguous in the dataset
    (generate_scenario returns a list per hand, extended sequentially).

    Detects boundaries using three signals:
    1. num_players changes → definitely new hand
    2. Community cards contradict known board → new hand
    3. Same hero_pos appears with different hole cards → new hand

    Returns:
        list of int hand_ids, one per scenario
    """
    if not scenarios:
        return []

    def _get_fingerprint(scenario):
        """Extract (hero_pos, hand_tuple) from a scenario."""
        last_event = scenario["events"][-1]
        return last_event["hero_pos"], tuple(last_event["hand"])

    hand_ids = [0]
    hand_id = 0
    known_board = [None] * 5
    # Track hero hands seen in current hand group: {hero_pos: hand_tuple}
    seen_heroes = {}

    # Initialize from first scenario
    first_table = scenarios[0]["events"][-1]["table"]
    for j in range(5):
        if first_table[j] != -1:
            known_board[j] = first_table[j]

    hero_pos, hero_hand = _get_fingerprint(scenarios[0])
    seen_heroes[hero_pos] = hero_hand
    prev_num_players = scenarios[0]["num_players"]

    for i in range(1, len(scenarios)):
        table = scenarios[i]["events"][-1]["table"]
        num_players = scenarios[i]["num_players"]
        hero_pos, hero_hand = _get_fingerprint(scenarios[i])

        # Check compatibility
        new_hand = False

        if num_players != prev_num_players:
            new_hand = True
        else:
            # Check community cards
            for j in range(5):
                if table[j] != -1 and known_board[j] is not None and known_board[j] != table[j]:
                    new_hand = True
                    break

            # Check hero identity: same position must have same cards
            if not new_hand and hero_pos in seen_heroes:
                if seen_heroes[hero_pos] != hero_hand:
                    new_hand = True

        if new_hand:
            hand_id += 1
            known_board = [None] * 5
            seen_heroes = {}

        # Update known board
        for j in range(5):
            if table[j] != -1:
                known_board[j] = table[j]

        seen_heroes[hero_pos] = hero_hand
        prev_num_players = num_players
        hand_ids.append(hand_id)

    return hand_ids


def hand_aware_split(dataset, scenarios, val_split, seed=42):
    """Split dataset by hand so no hand appears in both train and val.

    Uses explicit hand_id if present in scenarios, otherwise infers
    hand boundaries from contiguity (backward compat with old datasets).

    Args:
        dataset: PyTorch Dataset wrapping scenarios
        scenarios: raw scenario list (for hand_id extraction)
        val_split: fraction of hands to use for validation
        seed: random seed for reproducibility

    Returns:
        (train_dataset, val_dataset) as Subset objects
    """
    # Get hand_ids
    if scenarios and "hand_id" in scenarios[0]:
        hand_ids = [s["hand_id"] for s in scenarios]
    else:
        hand_ids = _infer_hand_ids(scenarios)

    # Group scenario indices by hand_id
    hand_to_indices = defaultdict(list)
    for idx, hid in enumerate(hand_ids):
        hand_to_indices[hid].append(idx)

    # Shuffle and split hand_ids
    unique_hands = list(hand_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_hands)

    val_n_hands = max(1, int(len(unique_hands) * val_split))
    val_hands = set(unique_hands[:val_n_hands])

    # Partition indices
    train_indices = []
    val_indices = []
    for hid, indices in hand_to_indices.items():
        if hid in val_hands:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
