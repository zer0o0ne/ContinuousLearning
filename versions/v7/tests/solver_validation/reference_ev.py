"""Reference EV calculator — clean-room reimplementation of gpu_solver_v3 formulas.

Computes fold/call/raise EVs for a poker decision point using exact equity
(not MC) and the same opponent response model as the solver. Differences
from gpu_solver_v3 should only arise from:
  1. Exact vs MC equity (< ~2% for river, slightly more for earlier streets)
  2. Floating-point order of operations

Any LARGE discrepancy (> 5% of pot) indicates a formula bug in either this
reference or the solver.

EV convention (matches the solver):
  - EVs are net profit/loss in chips relative to the start of the hand
  - Positive = hero profits, negative = hero loses
  - fold_ev = -hero_invested (hero loses everything already put in)
"""

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# EQR table — matches gpu_solver_v3.py: _get_eqr exactly (binary IP/OOP model)
# ---------------------------------------------------------------------------

_EQR_TABLE = {
    (True, 0): 1.05,    # IP preflop
    (False, 0): 0.90,   # OOP preflop
    (True, 1): 1.03,    # IP flop
    (False, 1): 0.93,   # OOP flop
    (True, 2): 1.01,    # IP turn
    (False, 2): 0.97,   # OOP turn
    (True, 3): 1.0,     # IP river
    (False, 3): 1.0,    # OOP river
}

_STREET_DISCOUNT = {0: 0.92, 1: 0.95, 2: 0.98, 3: 1.0}


def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def get_eqr(hero_position, street, n_players, active_positions=None):
    if active_positions is not None and len(active_positions) > 1:
        if street == 0:
            is_ip = (hero_position == 1)
        else:
            is_ip = hero_position == max(active_positions)
    else:
        if n_players == 2:
            is_ip = (hero_position == 0) if street > 0 else (hero_position == 1)
        elif n_players <= 6:
            is_ip = (hero_position == 5)
        else:
            is_ip = (hero_position == 8)
    return _EQR_TABLE.get((is_ip, street), 1.0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReferenceEVResult:
    """Complete EV breakdown for one spot."""
    fold_ev: float = 0.0
    call_ev: float = 0.0
    raise_evs: dict = field(default_factory=dict)
    allin_ev: float = 0.0
    best_action: str = ""
    best_ev: float = 0.0

    # Diagnostics
    exact_equity: float = 0.0
    eff_equity_call: float = 0.0
    eqr: float = 1.0

    # Per-raise diagnostics
    raise_details: dict = field(default_factory=dict)


@dataclass
class RaiseDetail:
    """Diagnostics for a single raise size."""
    raise_frac: float = 0.0
    raise_amount: float = 0.0
    new_pot: float = 0.0
    fold_threshold: float = 0.0
    p_fold: float = 0.0
    p_call: float = 0.0
    p_reraise: float = 0.0
    eq_vs_callers: float = 0.0
    eq_vs_reraisers: float = 0.0
    showdown_ev: float = 0.0
    ev_on_reraise: float = 0.0
    raise_ev: float = 0.0


# ---------------------------------------------------------------------------
# Per-combo equity computation
# ---------------------------------------------------------------------------

def _compute_per_combo_equity(hero_cards, board_cards, dead_cards):
    """Compute hero's equity against every possible opponent 2-card combo.

    Returns:
        list of (combo_tuple, hero_equity) where combo is (c1, c2)
    """
    from tests.solver_validation.cards import _JUDGER
    import numpy as np
    from itertools import combinations

    all_dead = set(hero_cards) | set(board_cards) | set(dead_cards)
    available = [c for c in range(52) if c not in all_dead]

    results = []
    street = len(board_cards)

    if street == 5:
        # River: exact evaluation
        hero_7 = list(board_cards) + list(hero_cards)
        hero_power, hero_bord = _JUDGER.compute_power(np.sort(np.array(hero_7)))

        for c1, c2 in combinations(available, 2):
            opp_7 = list(board_cards) + [c1, c2]
            opp_power, opp_bord = _JUDGER.compute_power(np.sort(np.array(opp_7)))

            if hero_power > opp_power:
                eq = 1.0
            elif hero_power < opp_power:
                eq = 0.0
            else:
                if hero_bord > opp_bord:
                    eq = 1.0
                elif hero_bord < opp_bord:
                    eq = 0.0
                else:
                    eq = 0.5
            results.append(((c1, c2), eq))

    elif street == 4:
        # Turn: enumerate 1 river card
        remaining = [c for c in available]
        for river_card in remaining:
            full_board = list(board_cards) + [river_card]
            hero_7 = full_board + list(hero_cards)
            hero_power, hero_bord = _JUDGER.compute_power(np.sort(np.array(hero_7)))

            avail_after_river = [c for c in available if c != river_card]
            for c1, c2 in combinations(avail_after_river, 2):
                opp_7 = full_board + [c1, c2]
                opp_power, opp_bord = _JUDGER.compute_power(np.sort(np.array(opp_7)))

                if hero_power > opp_power:
                    eq = 1.0
                elif hero_power < opp_power:
                    eq = 0.0
                else:
                    eq = 1.0 if hero_bord > opp_bord else (0.0 if hero_bord < opp_bord else 0.5)

                # We're computing per-river-card, but we need per-combo average
                # For simplicity, aggregate later
                results.append(((c1, c2), eq))

    elif street == 3:
        # Flop: enumerate 2 remaining cards (turn + river)
        remaining = [c for c in available]
        for t_card, r_card in combinations(remaining, 2):
            full_board = list(board_cards) + [t_card, r_card]
            hero_7 = full_board + list(hero_cards)
            hero_power, hero_bord = _JUDGER.compute_power(np.sort(np.array(hero_7)))

            avail_after = [c for c in available if c not in (t_card, r_card)]
            for c1, c2 in combinations(avail_after, 2):
                opp_7 = full_board + [c1, c2]
                opp_power, opp_bord = _JUDGER.compute_power(np.sort(np.array(opp_7)))

                if hero_power > opp_power:
                    eq = 1.0
                elif hero_power < opp_power:
                    eq = 0.0
                else:
                    eq = 1.0 if hero_bord > opp_bord else (0.0 if hero_bord < opp_bord else 0.5)
                results.append(((c1, c2), eq))

    return results


def compute_overall_equity_and_per_combo(hero_cards, board_cards, dead_cards=None):
    """Compute overall equity AND per-opponent-combo equity.

    For river: returns (overall_equity, combo_equities) where combo_equities
    is a dict mapping (c1, c2) -> hero_equity_vs_that_combo.

    For earlier streets: combo equities are AVERAGED over future boards.

    Returns:
        (overall_equity: float, combo_equities: dict[(int,int), float])
    """
    if dead_cards is None:
        dead_cards = set()

    results = _compute_per_combo_equity(hero_cards, board_cards, dead_cards)

    if not results:
        return 0.5, {}

    street = len(board_cards)

    if street == 5:
        # River: one result per combo
        combo_eq = {combo: eq for combo, eq in results}
        overall = sum(eq for _, eq in results) / len(results)
        return overall, combo_eq

    elif street == 4:
        # Turn: average across river cards per combo
        from collections import defaultdict
        combo_sums = defaultdict(float)
        combo_counts = defaultdict(int)
        for combo, eq in results:
            combo_sums[combo] += eq
            combo_counts[combo] += 1

        combo_eq = {}
        total_eq = 0.0
        total_n = 0
        for combo in combo_sums:
            avg = combo_sums[combo] / combo_counts[combo]
            combo_eq[combo] = avg
            total_eq += avg
            total_n += 1

        overall = total_eq / total_n if total_n > 0 else 0.5
        return overall, combo_eq

    elif street == 3:
        # Flop: average across turn+river combos per opponent combo
        from collections import defaultdict
        combo_sums = defaultdict(float)
        combo_counts = defaultdict(int)
        for combo, eq in results:
            combo_sums[combo] += eq
            combo_counts[combo] += 1

        combo_eq = {}
        total_eq = 0.0
        total_n = 0
        for combo in combo_sums:
            avg = combo_sums[combo] / combo_counts[combo]
            combo_eq[combo] = avg
            total_eq += avg
            total_n += 1

        overall = total_eq / total_n if total_n > 0 else 0.5
        return overall, combo_eq

    else:
        # Preflop: too expensive for exact, use MC from cards.py
        from tests.solver_validation.cards import exact_equity_preflop_mc
        eq = exact_equity_preflop_mc(hero_cards, n_iters=50000, seed=hash(hero_cards) & 0x7FFFFFFF)
        return eq, {}


def compute_multiway_equity_mc(hero_cards, board_cards, n_opponents, n_iters=10000, seed=42):
    """Compute hero equity against n_opponents via MC, matching gpu_equity_v3.

    Hero must beat ALL opponents to win. Ties split proportionally.
    Each iteration samples non-overlapping opponent hands from the full range.
    """
    import random
    from itertools import combinations
    from tests.solver_validation.cards import _JUDGER
    import numpy as np

    rng = random.Random(seed)

    dead = set(hero_cards) | set(board_cards)
    available = [c for c in range(52) if c not in dead]
    all_combos = list(combinations(available, 2))

    hero_7 = list(board_cards) + list(hero_cards)
    hero_power, hero_bord = _JUDGER.compute_power(np.sort(np.array(hero_7)))

    total_eq = 0.0
    valid_count = 0

    for _ in range(n_iters):
        opp_cards_used = set()
        opp_powers = []
        opp_bords = []
        valid = True

        for _ in range(n_opponents):
            attempts = 0
            while attempts < 50:
                combo = rng.choice(all_combos)
                if combo[0] not in opp_cards_used and combo[1] not in opp_cards_used:
                    break
                attempts += 1
            else:
                valid = False
                break

            opp_cards_used.add(combo[0])
            opp_cards_used.add(combo[1])
            opp_7 = list(board_cards) + list(combo)
            opp_pow, opp_b = _JUDGER.compute_power(np.sort(np.array(opp_7)))
            opp_powers.append(opp_pow)
            opp_bords.append(opp_b)

        if not valid:
            continue

        best_opp_pow = max(opp_powers)
        best_indices = [i for i, p in enumerate(opp_powers) if p == best_opp_pow]
        best_opp_bord = max(opp_bords[i] for i in best_indices)

        if hero_power > best_opp_pow:
            total_eq += 1.0
        elif hero_power == best_opp_pow:
            if hero_bord > best_opp_bord:
                total_eq += 1.0
            elif hero_bord == best_opp_bord:
                n_tied = sum(1 for i in best_indices if opp_bords[i] == hero_bord)
                total_eq += 1.0 / (n_tied + 1.0)
        valid_count += 1

    return total_eq / valid_count if valid_count > 0 else 0.5


# ---------------------------------------------------------------------------
# Opponent response model (mirrors gpu_solver_v3 exactly)
# ---------------------------------------------------------------------------

def _compute_fold_threshold(call_cost, pot_after_raise, hero_position,
                            opponent_positions, street):
    """Fold threshold: what equity opponent needs to call profitably."""
    if pot_after_raise <= 0:
        return 0.5
    if call_cost <= 0:
        return 0.0
    raw = call_cost / pot_after_raise
    threshold = raw ** 0.85

    if opponent_positions and street > 0:
        max_opp_pos = max(opponent_positions)
        if max_opp_pos < hero_position:
            threshold *= 1.08
        elif min(opponent_positions) > hero_position:
            threshold *= 0.95

    return threshold


def _compute_reraise_threshold(call_cost, pot_after_raise, street, stack, pot,
                               static_threshold=0.72, dynamic=False):
    """Reraise threshold: above this equity, opponent reraises."""
    if not dynamic:
        return static_threshold
    # Dynamic threshold (mirrors _compute_reraise_threshold in gpu_solver_v3)
    if pot_after_raise <= 0:
        return 0.8
    raw = call_cost / pot_after_raise
    spr = stack / max(pot, 1e-6)
    street_mult = {0: 1.0, 1: 1.1, 2: 1.15, 3: 1.2}.get(street, 1.1)
    spr_factor = min(1.0, spr / 4.0) if spr < 4.0 else 1.0
    threshold = max(0.5, min(0.95, (1.0 - raw) * street_mult * spr_factor))
    return threshold


def _compute_opponent_response(combo_equities, fold_threshold, reraise_threshold,
                               smoothing_enabled=True, beta_fold=0.07,
                               beta_reraise=0.07, combo_weights=None,
                               polarized_reraise=None):
    """Compute p_fold, p_call, p_reraise and eq_vs_callers/reraisers.

    Mirrors gpu_solver_v3 lines 789-913.

    Args:
        combo_equities: dict of (c1,c2) -> opponent_equity (NOT hero equity)
        fold_threshold: float
        reraise_threshold: float
        smoothing_enabled: use sigmoid or hard mask
        beta_fold, beta_reraise: sigmoid steepness
        combo_weights: optional dict of (c1,c2) -> weight

    Returns:
        (p_fold, p_call, p_reraise, eq_vs_callers, eq_vs_reraisers)
    """
    if not combo_equities:
        return 0.0, 1.0, 0.0, 0.5, 0.5

    combos = list(combo_equities.keys())
    opp_eqs = [combo_equities[c] for c in combos]
    hero_eqs = [1.0 - oe for oe in opp_eqs]

    n = len(combos)
    if combo_weights:
        weights = [combo_weights.get(c, 1.0 / n) for c in combos]
        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]
    else:
        weights = [1.0 / n] * n

    if smoothing_enabled:
        # Sigmoid soft weights
        p_fold_per = [_sigmoid((fold_threshold - oe) / beta_fold) for oe in opp_eqs]
        p_reraise_per = [_sigmoid((oe - reraise_threshold) / beta_reraise) for oe in opp_eqs]

        # Polarized reraise: add bluff component
        if polarized_reraise and polarized_reraise.get("enabled", False):
            bluff_thresh = polarized_reraise.get("bluff_threshold", 0.25)
            beta_bluff = polarized_reraise.get("beta_bluff", 0.10)
            bluff_freq = polarized_reraise.get("bluff_frequency", 0.30)
            for i, oe in enumerate(opp_eqs):
                blocker_strength = max(0.0, 0.5 - oe)
                bluff_score = _sigmoid((blocker_strength - bluff_thresh) / beta_bluff)
                p_reraise_per[i] = min(1.0, p_reraise_per[i] + bluff_freq * bluff_score)

        # Clamp fold + reraise <= 1
        p_call_per = []
        for i in range(n):
            denom = max(1.0, p_fold_per[i] + p_reraise_per[i])
            p_fold_per[i] /= denom
            p_reraise_per[i] /= denom
            p_call_per.append(max(0.0, 1.0 - p_fold_per[i] - p_reraise_per[i]))

        # Weighted aggregation
        p_fold = sum(pf * w for pf, w in zip(p_fold_per, weights))
        p_reraise = sum(pr * w for pr, w in zip(p_reraise_per, weights))
        p_call = sum(pc * w for pc, w in zip(p_call_per, weights))

        # eq_vs_callers: weighted by p_call_per * combo_weight
        call_weight_sum = sum(pc * w for pc, w in zip(p_call_per, weights))
        if call_weight_sum > 1e-8:
            eq_vs_callers = sum(he * pc * w for he, pc, w in
                                zip(hero_eqs, p_call_per, weights)) / call_weight_sum
        else:
            eq_vs_callers = sum(he * w for he, w in zip(hero_eqs, weights))

        # eq_vs_reraisers: weighted by p_reraise_per * combo_weight
        reraise_weight_sum = sum(pr * w for pr, w in zip(p_reraise_per, weights))
        if reraise_weight_sum > 1e-8:
            eq_vs_reraisers = sum(he * pr * w for he, pr, w in
                                  zip(hero_eqs, p_reraise_per, weights)) / reraise_weight_sum
        else:
            eq_vs_reraisers = sum(he * w for he, w in zip(hero_eqs, weights))

    else:
        # Hard mask path
        fold_count = 0.0
        call_count = 0.0
        reraise_count = 0.0
        eq_callers_sum = 0.0
        eq_reraisers_sum = 0.0

        for i in range(n):
            oe = opp_eqs[i]
            he = hero_eqs[i]
            w = weights[i]
            if oe < fold_threshold:
                fold_count += w
            elif oe > reraise_threshold:
                reraise_count += w
                eq_reraisers_sum += he * w
            else:
                call_count += w
                eq_callers_sum += he * w

        p_fold = fold_count
        p_call = call_count
        p_reraise = reraise_count

        eq_vs_callers = eq_callers_sum / call_count if call_count > 1e-8 else \
            sum(he * w for he, w in zip(hero_eqs, weights))
        eq_vs_reraisers = eq_reraisers_sum / reraise_count if reraise_count > 1e-8 else \
            sum(he * w for he, w in zip(hero_eqs, weights))

    # Blocker adjustment
    mean_opp_eq = sum(oe * w for oe, w in zip(opp_eqs, weights))
    blocker_adj = 1.0 + (0.5 - mean_opp_eq) * 0.12
    p_fold *= blocker_adj
    p_total = p_fold + p_call + p_reraise
    if p_total > 1e-6:
        p_fold /= p_total
        p_call /= p_total
        p_reraise /= p_total

    return p_fold, p_call, p_reraise, eq_vs_callers, eq_vs_reraisers


# ---------------------------------------------------------------------------
# Main EV computation
# ---------------------------------------------------------------------------

def compute_reference_evs(spot, raise_fracs, solver_config=None):
    """Compute reference EVs for all actions at a given spot.

    Args:
        spot: Spot object with hero_cards, board_cards, pot, facing_bet, etc.
        raise_fracs: list of raise fractions (e.g. [0.33, 0.5, 0.67, 1.0])
        solver_config: dict with solver parameters (thresholds, smoothing, etc.)

    Returns:
        ReferenceEVResult with fold_ev, call_ev, raise_evs, allin_ev
    """
    if solver_config is None:
        solver_config = {}

    hero_cards = tuple(spot.hero_cards)
    board_cards = list(spot.board_cards)
    pot = spot.pot
    facing_bet = spot.facing_bet
    stack = spot.stack
    hero_invested = spot.hero_invested
    street = spot.street
    hero_position = spot.hero_position
    opponent_positions = spot.opponent_positions
    n_players = spot.table_size

    # EQR
    eqr_enabled = solver_config.get("eqr_enabled", True)
    active_positions = list(opponent_positions) + [hero_position]
    eqr_raw = get_eqr(hero_position, street, n_players, active_positions) if eqr_enabled else 1.0

    # SPR-adjusted EQR
    spr = stack / max(pot, 1e-6)
    spr_eqr_factor = min(1.0, spr / 6.0)
    eqr = 1.0 + (eqr_raw - 1.0) * spr_eqr_factor

    # Equity: per-combo HU equity for opponent response model, multiway for raw_equity
    n_opponents = len(opponent_positions) if opponent_positions else 1
    if street == 0:
        from tests.solver_validation.cards import exact_equity_preflop_mc
        raw_equity = exact_equity_preflop_mc(hero_cards, n_iters=50000,
                                             seed=hash(hero_cards) & 0x7FFFFFFF)
        combo_equities = {}
    else:
        hu_equity, hero_combo_eq = compute_overall_equity_and_per_combo(
            hero_cards, board_cards)
        combo_equities = {c: 1.0 - he for c, he in hero_combo_eq.items()}

        if n_opponents > 1:
            raw_equity = compute_multiway_equity_mc(
                hero_cards, board_cards, n_opponents,
                n_iters=10000, seed=hash(hero_cards) & 0x7FFFFFFF)
        else:
            raw_equity = hu_equity

    result = ReferenceEVResult()
    result.exact_equity = raw_equity
    result.eqr = eqr

    # --- Fold EV ---
    result.fold_ev = -hero_invested

    # --- Call EV ---
    eff_equity = max(0.0, min(1.0, raw_equity * eqr))
    result.eff_equity_call = eff_equity
    total_call_investment = hero_invested + facing_bet
    call_ev = eff_equity * (pot - hero_invested) + (1 - eff_equity) * (-total_call_investment)
    street_discount = _STREET_DISCOUNT.get(street, 1.0)
    # B.5.5: discount only positive call EV
    call_ev = min(call_ev, call_ev * street_discount)
    result.call_ev = call_ev

    # --- Raise EVs ---
    smoothing_cfg = solver_config.get("threshold_smoothing", {})
    smoothing_enabled = bool(smoothing_cfg.get("enabled", False))
    beta_fold = float(smoothing_cfg.get("beta_fold", 0.07))
    beta_reraise = float(smoothing_cfg.get("beta_reraise", 0.07))
    static_reraise_threshold = float(solver_config.get("reraise_threshold", 0.72))
    dynamic_reraise = bool(solver_config.get("dynamic_reraise", False))
    polarized_reraise = solver_config.get("polarized_reraise", None)

    for raise_frac in raise_fracs:
        detail = RaiseDetail(raise_frac=raise_frac)

        raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
        total_raise = hero_invested + raise_amount
        # B.1: pot after opponent calls hero's raise
        new_pot = pot + raise_amount + (raise_amount - facing_bet)
        call_cost = raise_amount - facing_bet

        detail.raise_amount = raise_amount
        detail.new_pot = new_pot

        # Fold threshold
        fold_threshold = _compute_fold_threshold(
            call_cost, new_pot, hero_position, opponent_positions, street)
        detail.fold_threshold = fold_threshold

        # Reraise threshold
        reraise_threshold = _compute_reraise_threshold(
            call_cost, new_pot, street, stack, pot,
            static_threshold=static_reraise_threshold,
            dynamic=dynamic_reraise)

        # Opponent response
        if combo_equities:
            p_fold, p_call, p_reraise, eq_vs_callers, eq_vs_reraisers = \
                _compute_opponent_response(
                    combo_equities, fold_threshold, reraise_threshold,
                    smoothing_enabled=smoothing_enabled,
                    beta_fold=beta_fold, beta_reraise=beta_reraise,
                    polarized_reraise=polarized_reraise)
        else:
            # Preflop with no per-combo data: use simple model
            # Aggressive opponents fold more
            profile = getattr(spot, 'opponent_profile', 'passive')
            if profile == 'aggressive':
                p_fold = min(0.7, fold_threshold * 1.3)
            else:
                p_fold = min(0.5, fold_threshold * 0.8)
            p_reraise = max(0.0, min(0.3, (raw_equity - reraise_threshold) * 2.0))
            if raw_equity < reraise_threshold:
                p_reraise = 0.05
            p_call = max(0.0, 1.0 - p_fold - p_reraise)
            eq_vs_callers = raw_equity * 0.95
            eq_vs_reraisers = raw_equity * 0.85

        # When raise_amount reaches stack, hero is all-in — no reraise possible
        is_allin_raise = raise_amount >= stack - 1e-6
        if is_allin_raise and p_reraise > 0:
            if eq_vs_reraisers is not None and (p_call + p_reraise) > 1e-8:
                eq_vs_callers = (p_call * eq_vs_callers + p_reraise * eq_vs_reraisers) / (p_call + p_reraise)
            p_call = p_call + p_reraise
            p_reraise = 0.0

        detail.p_fold = p_fold
        detail.p_call = p_call
        detail.p_reraise = p_reraise
        detail.eq_vs_callers = eq_vs_callers
        detail.eq_vs_reraisers = eq_vs_reraisers

        # Showdown EV when called
        eff_eq_callers = max(0.0, min(1.0, eq_vs_callers * eqr)) if eqr_enabled else eq_vs_callers
        showdown_ev = eff_eq_callers * (new_pot - total_raise) + (1 - eff_eq_callers) * (-total_raise)
        detail.showdown_ev = showdown_ev

        # Reraise EV
        if p_reraise > 0 and eq_vs_reraisers is not None:
            _reraise_spr = stack / max(pot, 1e-6)
            if _reraise_spr < 3.0:
                reraise_size = stack
            else:
                _street_mult = {0: 3.0, 1: 2.5, 2: 2.2, 3: 2.0}.get(street, 2.5)
                reraise_size = min(raise_amount * _street_mult, stack)
            total_reraise_cost = hero_invested + reraise_size
            reraise_pot = new_pot + reraise_size
            hero_call_cost = reraise_size - raise_amount
            hero_continue_threshold = (
                hero_call_cost / (reraise_pot + hero_call_cost)
                if (reraise_pot + hero_call_cost) > 0 else 0.5
            )
            p_hero_continues = _sigmoid(15.0 * (eq_vs_reraisers - hero_continue_threshold))
            eff_eq_reraise = max(0.0, min(1.0, eq_vs_reraisers * eqr)) if eqr_enabled else eq_vs_reraisers
            ev_continue = eff_eq_reraise * (reraise_pot - total_reraise_cost) + \
                          (1 - eff_eq_reraise) * (-total_reraise_cost)
            ev_on_reraise = p_hero_continues * ev_continue + (1 - p_hero_continues) * (-total_raise)
            reraise_discount = 0.3
            geometric_factor = min(1.5, 1.0 / max(0.5, 1.0 - p_reraise * reraise_discount))
            if ev_on_reraise > 0:
                ev_on_reraise *= geometric_factor
        else:
            ev_on_reraise = -total_raise
        detail.ev_on_reraise = ev_on_reraise

        # Multiway fold: per-opponent raw sigmoid fold rate (no blocker/normalization)
        # matches solver's opp_eq_by_opp path in _compute_ev_v3_from_state
        n_opponents = len(opponent_positions) if opponent_positions else 1
        if n_opponents <= 1:
            p_others_all_fold = 1.0
        else:
            p_others_all_fold = 1.0
            if combo_equities:
                opp_eqs_list = list(combo_equities.values())
                raw_fold_probs = [_sigmoid((fold_threshold - oe) / beta_fold) for oe in opp_eqs_list]
                avg_raw_fold = sum(raw_fold_probs) / len(raw_fold_probs)
                for _ in range(n_opponents - 1):
                    p_others_all_fold *= avg_raw_fold
            else:
                for _ in range(n_opponents - 1):
                    p_others_all_fold *= p_fold

        # Final raise EV (B.5.4 multiway-aware)
        fold_term = p_fold * (
            p_others_all_fold * (pot - hero_invested)
            + (1.0 - p_others_all_fold) * showdown_ev
        )
        raise_ev = fold_term + p_call * showdown_ev + p_reraise * ev_on_reraise
        detail.raise_ev = raise_ev

        result.raise_evs[raise_frac] = raise_ev
        result.raise_details[raise_frac] = detail

    # --- All-in EV ---
    # All-in is just a raise with raise_amount = stack
    if stack > facing_bet:
        allin_frac = (stack - facing_bet) / max(pot + facing_bet, 1e-6)
        allin_detail = RaiseDetail(raise_frac=allin_frac)
        raise_amount = stack
        total_raise = hero_invested + raise_amount
        new_pot = pot + raise_amount + (raise_amount - facing_bet)
        call_cost = raise_amount - facing_bet

        fold_threshold = _compute_fold_threshold(
            call_cost, new_pot, hero_position, opponent_positions, street)

        reraise_threshold = _compute_reraise_threshold(
            call_cost, new_pot, street, stack, pot,
            static_threshold=static_reraise_threshold,
            dynamic=dynamic_reraise)

        if combo_equities:
            p_fold, p_call, p_reraise, eq_vs_callers, eq_vs_reraisers = \
                _compute_opponent_response(
                    combo_equities, fold_threshold, reraise_threshold,
                    smoothing_enabled=smoothing_enabled,
                    beta_fold=beta_fold, beta_reraise=beta_reraise,
                    polarized_reraise=polarized_reraise)
            # All-in: opponents can't reraise — merge reraise mass into call
            if p_reraise > 0 and (p_call + p_reraise) > 1e-8:
                merged_eq = (p_call * eq_vs_callers + p_reraise * eq_vs_reraisers) / (p_call + p_reraise)
                p_call = p_call + p_reraise
                eq_vs_callers = merged_eq
                p_reraise = 0.0
        else:
            profile = getattr(spot, 'opponent_profile', 'passive')
            if profile == 'aggressive':
                p_fold = min(0.8, fold_threshold * 1.5)
            else:
                p_fold = min(0.6, fold_threshold * 0.9)
            p_reraise = 0.0
            p_call = 1.0 - p_fold
            eq_vs_callers = raw_equity * 0.92
            eq_vs_reraisers = raw_equity

        eff_eq_callers = max(0.0, min(1.0, eq_vs_callers * eqr)) if eqr_enabled else eq_vs_callers
        showdown_ev = eff_eq_callers * (new_pot - total_raise) + (1 - eff_eq_callers) * (-total_raise)

        # Multiway fold: per-opponent raw fold rate (same as raise section)
        n_opponents_ai = len(opponent_positions) if opponent_positions else 1
        if n_opponents_ai <= 1:
            p_others_all_fold = 1.0
        else:
            p_others_all_fold = 1.0
            if combo_equities:
                opp_eqs_list = list(combo_equities.values())
                raw_fold_probs = [_sigmoid((fold_threshold - oe) / beta_fold) for oe in opp_eqs_list]
                avg_raw_fold = sum(raw_fold_probs) / len(raw_fold_probs)
                for _ in range(n_opponents_ai - 1):
                    p_others_all_fold *= avg_raw_fold
            else:
                p_others_all_fold = p_fold ** max(0, n_opponents_ai - 1)

        fold_term = p_fold * (
            p_others_all_fold * (pot - hero_invested)
            + (1.0 - p_others_all_fold) * showdown_ev
        )
        result.allin_ev = fold_term + p_call * showdown_ev

        allin_detail.raise_amount = raise_amount
        allin_detail.new_pot = new_pot
        allin_detail.fold_threshold = fold_threshold
        allin_detail.p_fold = p_fold
        allin_detail.p_call = p_call
        allin_detail.p_reraise = 0.0
        allin_detail.eq_vs_callers = eq_vs_callers
        allin_detail.showdown_ev = showdown_ev
        allin_detail.raise_ev = result.allin_ev
        result.raise_details["allin"] = allin_detail
    else:
        result.allin_ev = result.call_ev  # stack <= facing_bet means call = all-in

    # Best action
    all_evs = {"fold": result.fold_ev, "call": result.call_ev, "allin": result.allin_ev}
    for frac, ev in result.raise_evs.items():
        all_evs[f"raise_{frac}"] = ev
    result.best_action = max(all_evs, key=all_evs.get)
    result.best_ev = all_evs[result.best_action]

    return result
