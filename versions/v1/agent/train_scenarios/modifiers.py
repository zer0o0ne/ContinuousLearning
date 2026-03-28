"""Target modifiers for multi-agent training.

Modifies action_evs to create agent "personalities", then recomputes
ev_target and action_probs consistently from the modified EVs.
"""

import copy
import torch
import torch.nn.functional as F


def resolve_actions(selector, n_actions):
    """Convert action selector to list of action indices.

    Action layout: [fold, call, raise_0 .. raise_(bins-1), all-in]
    where bins = n_actions - 3.

    Selector can be:
      - str: named group ("fold", "call", "raises", "allin", "small_raises",
             "big_raises", "aggressive")
      - list[int]: explicit indices, e.g. [0, 2, 5]
      - str slice: "start:stop" or "start:stop:step", e.g. "2:10", "2:52:2"
    """
    # List of ints — explicit indices
    if isinstance(selector, list):
        return [int(i) for i in selector]

    # String slice — "start:stop" or "start:stop:step"
    if isinstance(selector, str) and ":" in selector:
        parts = selector.split(":")
        args = [int(p) if p else None for p in parts]
        return list(range(*slice(*args).indices(n_actions)))

    # Named group
    bins = n_actions - 3
    mid = bins // 2

    selectors = {
        "fold": [0],
        "call": [1],
        "raises": list(range(2, 2 + bins)),
        "allin": [n_actions - 1],
        "small_raises": list(range(2, 2 + mid)),
        "big_raises": list(range(2 + mid, n_actions)),
        "aggressive": list(range(2, n_actions)),
    }

    if selector not in selectors:
        raise ValueError(f"Unknown action selector: {selector!r}. "
                         f"Valid: {list(selectors.keys())} or list of ints or 'start:stop[:step]'")
    return selectors[selector]


def _parse_condition(condition_str):
    """Parse condition string like 'equity < 0.3' or 'pos > 4'.

    Returns: (field, op, threshold)
    """
    parts = condition_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Condition must be 'field op value', got: {condition_str!r}")

    field, op, value = parts
    if field not in ("equity", "pos"):
        raise ValueError(f"Condition field must be 'equity' or 'pos', got: {field!r}")
    if op not in ("<", ">"):
        raise ValueError(f"Condition op must be '<' or '>', got: {op!r}")

    return field, op, float(value)


def _check_condition(scenario, field, op, threshold):
    """Check if a scenario matches a condition."""
    if field == "equity":
        val = scenario.get("equity", 0.0)
    elif field == "pos":
        val = scenario["events"][-1]["hero_pos"]
    else:
        return False

    if op == "<":
        return val < threshold
    return val > threshold


def apply_modifiers(scenarios, modifiers, n_actions, big_blind, temperature):
    """Apply modifiers to scenarios, returning a modified deepcopy.

    Modifies action_evs, then recomputes ev_target and action_probs.
    Original scenarios are not touched.

    Args:
        scenarios: list of scenario dicts (with action_evs, ev_target, action_probs)
        modifiers: list of modifier dicts from config
        n_actions: number of actions (e.g. 53)
        big_blind: big blind size (for temperature scaling)
        temperature: base GTO temperature

    Returns:
        list of modified scenario dicts (deepcopy)
    """
    scenarios = copy.deepcopy(scenarios)

    if not modifiers:
        return scenarios

    # Separate temperature modifier (applied at the end)
    temp = temperature
    bias_modifiers = []
    for mod in modifiers:
        if mod["type"] == "temperature":
            temp = mod["value"]
        else:
            bias_modifiers.append(mod)

    # Pre-parse conditions
    parsed_mods = []
    for mod in bias_modifiers:
        entry = {
            "type": mod["type"],
            "actions": resolve_actions(mod["actions"], n_actions),
            "factor": mod["factor"],
        }
        if mod["type"] == "conditional_bias":
            field, op, thresh = _parse_condition(mod["condition"])
            entry["cond"] = (field, op, thresh)
        parsed_mods.append(entry)

    normalizer = big_blind * temp

    for s in scenarios:
        evs = s["action_evs"]
        if isinstance(evs, list):
            evs = [float(e) for e in evs]
        else:
            evs = list(evs)

        # Accumulate total factor per action from all modifiers,
        # then apply once: ev[i] = ev[i] + |ev[i]| * total_factor[i]
        total_factor = [0.0] * len(evs)

        for mod in parsed_mods:
            if mod["type"] == "conditional_bias":
                field, op, thresh = mod["cond"]
                if not _check_condition(s, field, op, thresh):
                    continue

            for idx in mod["actions"]:
                if idx < len(evs):
                    total_factor[idx] += mod["factor"]

        # Apply accumulated factors once (from original values)
        for i in range(len(evs)):
            if total_factor[i] != 0.0:
                evs[i] = evs[i] + abs(evs[i]) * total_factor[i]

        # Recompute targets from modified EVs
        evs_t = torch.tensor(evs, dtype=torch.float32)
        s["action_evs"] = evs
        s["ev_target"] = float(evs_t.max().item())
        s["action_probs"] = F.softmax(evs_t / normalizer, dim=0).tolist()

    return scenarios
