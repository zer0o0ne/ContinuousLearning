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

    Only compares consecutive pairs (no accumulated state) to guarantee
    that scenarios from the same hand are NEVER split into different groups.
    This may merge consecutive hands (false negatives), but never falsely
    splits a hand (no false positives = no data leakage).

    Signals checked between consecutive scenarios:
    1. num_players changes → new hand
    2. Community cards contradict → new hand
    3. Same hero_pos with different hole cards → new hand

    Returns:
        list of int hand_ids, one per scenario
    """
    if not scenarios:
        return []

    hand_ids = [0]
    hand_id = 0

    for i in range(1, len(scenarios)):
        prev = scenarios[i - 1]
        curr = scenarios[i]

        new_hand = False

        # 1. num_players changed
        if curr["num_players"] != prev["num_players"]:
            new_hand = True

        # 2. Community cards conflict between consecutive scenarios
        if not new_hand:
            prev_table = prev["events"][-1]["table"]
            curr_table = curr["events"][-1]["table"]
            for j in range(5):
                if prev_table[j] != -1 and curr_table[j] != -1 and prev_table[j] != curr_table[j]:
                    new_hand = True
                    break

        # 3. Same hero_pos with different hole cards
        if not new_hand:
            prev_hero = prev["events"][-1]["hero_pos"]
            curr_hero = curr["events"][-1]["hero_pos"]
            if prev_hero == curr_hero:
                if tuple(prev["events"][-1]["hand"]) != tuple(curr["events"][-1]["hand"]):
                    new_hand = True

        if new_hand:
            hand_id += 1

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
