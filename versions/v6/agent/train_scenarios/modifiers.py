"""Target modifiers for multi-agent training.

Modifies action_evs to create agent "personalities", then recomputes
ev_target and action_probs consistently from the modified EVs.
"""

import copy
import warnings

import torch
import torch.nn.functional as F

# Warn only once per run when old-format scenarios (no `legal_mask`) are seen.
_LEGAL_MASK_WARNED = False


def _warn_missing_legal_mask():
    global _LEGAL_MASK_WARNED
    if _LEGAL_MASK_WARNED:
        return
    _LEGAL_MASK_WARNED = True
    warnings.warn(
        "Scenario has no 'legal_mask' (old-format dataset): recomputing "
        "action_probs UNMASKED — fold regains probability when checking is "
        "free and capped raise bins duplicate the all-in mass. Regenerate the "
        "dataset to fix. (warned once per run)",
        stacklevel=3,
    )


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


STYLE_DIMS = 16


def build_style_vector(modifiers, n_actions, base_temperature, max_players=9):
    """Canonical real-valued style encoding of a modifier list (§2.1 of
    PLAN_OPPONENT_ADAPTATION). Deterministic — one vector per agent config.

    Layout (STYLE_DIMS = 16):
      [0:5]   unconditional category biases (fold, call, small raise,
              big raise, all-in — same split as resolve_actions)
      [5:10]  low-equity conditional biases ("equity < t"), scaled by the
              region measure t
      [10:15] high-equity conditional biases ("equity > t"), scaled by 1 - t
      [15]    log(effective temperature)

    Every bias modifier distributes its factor over categories
    coverage-weighted: contribution to category c =
    factor * |resolved_actions ∩ c| / |c|; multiple modifiers accumulate
    additively (mirrors apply_modifiers).  `pos` conditions fold into the
    UNCONDITIONAL block weighted by the fraction of positions
    (0..max_players-1) satisfying the condition.
    """
    import math

    bins = n_actions - 3
    mid = bins // 2
    cat_members = [
        [0],                                   # fold
        [1],                                   # call
        list(range(2, 2 + mid)),               # small raises
        list(range(2 + mid, n_actions - 1)),   # big raises
        [n_actions - 1],                       # all-in
    ]

    vec = [0.0] * STYLE_DIMS
    temp = float(base_temperature)

    for mod in modifiers or []:
        if mod["type"] == "temperature":
            temp = float(mod["value"])
            continue

        actions = set(resolve_actions(mod["actions"], n_actions))
        factor = float(mod["factor"])

        block = 0
        region = 1.0
        if mod["type"] == "conditional_bias":
            field, op, thresh = _parse_condition(mod["condition"])
            if field == "equity":
                if op == "<":
                    block, region = 1, thresh
                else:
                    block, region = 2, 1.0 - thresh
            else:  # pos — fold into the unconditional block, region-weighted
                if op == "<":
                    n_sat = sum(1 for p in range(max_players) if p < thresh)
                else:
                    n_sat = sum(1 for p in range(max_players) if p > thresh)
                block, region = 0, n_sat / max_players

        for c in range(5):
            members = cat_members[c]
            if not members:
                continue
            coverage = len(actions.intersection(members)) / len(members)
            if coverage:
                vec[block * 5 + c] += factor * region * coverage

    vec[15] = math.log(temp)
    return vec


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
        list of modified scenario dicts (shallow copy; only action_evs/action_probs/ev_target are mutated)
    """
    if not modifiers:
        return scenarios

    scenarios = [{**s, "action_evs": list(s["action_evs"]),
                  "action_probs": list(s["action_probs"])} for s in scenarios]

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

    for s in scenarios:
        # Audit B.3: use the SAME per-scenario normalizer as generation
        # (generate.py:834/913), not `big_blind * temp`. Generation softmaxes
        # action_evs by `max(pot + facing_bet, big_blind) * temperature`; using
        # only `big_blind * temp` here recomputed action_probs on a different
        # (much sharper) scale than the base dataset, so an identity-temperature
        # modifier would NOT reproduce the base targets. GTO scenarios carry
        # top-level `pot`/`facing_bet` (generate.py:922-923).
        normalizer = max(s["pot"] + s["facing_bet"], big_blind) * temp

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
        # Apply the scenario's legal mask exactly like generation does
        # (generate.py:997-999): illegal/dominated actions get -inf before the
        # tempered softmax, so they carry zero probability in the target.
        legal_mask = s.get("legal_mask")
        if legal_mask is not None:
            mask_t = torch.tensor(legal_mask, dtype=torch.bool)
            probs_evs = evs_t.masked_fill(~mask_t, float("-inf"))
        else:
            _warn_missing_legal_mask()
            probs_evs = evs_t
        s["action_probs"] = F.softmax(probs_evs / normalizer, dim=0).tolist()

    return scenarios
