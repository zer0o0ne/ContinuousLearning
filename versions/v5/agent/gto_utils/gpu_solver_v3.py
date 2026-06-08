"""
GPU poker solver v3 — per-combo opponent response, EQR, weighted sampling.

Improvements over gpu_solver_v2:
1. Per-combo opponent response: computes hero equity vs each opponent combo
   individually, then classifies fold/call/reraise by pot-odds thresholds
   instead of blanket MDF percentages.
2. Equity Realization (EQR) multipliers: corrects raw equity for positional
   advantage (IP vs OOP) and street.
3. Weighted combo sampling: action-consistent weighting so opponent combos
   that match observed actions are sampled more frequently.

Card encoding: card_id 0-51, rank = card_id // 4 (0=2, 12=A), suit = card_id % 4.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# Ensure gto_utils is importable (same pattern as generate.py)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from gpu_solver import evaluate_hands
from gpu_solver_v2 import (
    HAND_RANKINGS,
    POSITION_RANGE_PCT,
    ACTION_NARROWING,
    _ALL_COMBOS_CACHE,
    get_position_range,
    narrow_range,
    expand_range,
    expand_hand_type,
)


# ---------------------------------------------------------------------------
# Equity Realization (EQR) lookup table
# (is_in_position, street) → multiplier
# ---------------------------------------------------------------------------

EQR_TABLE = {
    (True, 0): 1.05,    # IP preflop
    (False, 0): 0.90,   # OOP preflop
    (True, 1): 1.03,    # IP flop
    (False, 1): 0.93,   # OOP flop
    (True, 2): 1.01,    # IP turn
    (False, 2): 0.97,   # OOP turn
    (True, 3): 1.0,     # IP river
    (False, 3): 1.0,    # OOP river
}


# ---------------------------------------------------------------------------
# Improvement 2: Equity Realization
# ---------------------------------------------------------------------------

def _get_eqr(hero_position, street, n_players, active_positions=None):
    """Get equity realization multiplier based on position and street.

    Args:
        hero_position: int, hero's seat index (0-based)
        street: int, 0=preflop, 1=flop, 2=turn, 3=river
        n_players: int, table size (2-9)
        active_positions: optional list of active player seat indices.
            If provided, hero is IP if they act last among active players postflop.

    Returns:
        float multiplier (typically 0.90-1.05)
    """
    if active_positions is not None and len(active_positions) > 1:
        if street == 0:
            # Preflop: higher position index = later to act = more IP
            # (in standard rotation, BTN acts last preflop in 2-player, BB last in multi)
            is_ip = hero_position == max(active_positions)
        else:
            # Postflop: highest seat index acts last (BTN position)
            is_ip = hero_position == max(active_positions)
    else:
        # Fallback: simple BTN check
        if n_players == 2:
            is_ip = (hero_position == 0) if street > 0 else (hero_position == 1)
        elif n_players <= 6:
            is_ip = (hero_position == 5)  # BTN for 6-max
        else:
            is_ip = (hero_position == 8)  # BTN for 9-max
    return EQR_TABLE.get((is_ip, street), 1.0)


# ---------------------------------------------------------------------------
# Improvement 3: Weighted combo sampling
# ---------------------------------------------------------------------------

def compute_combo_weights(hand_types, action_history, dead_cards=None):
    """Compute sampling weights for opponent hand types based on action consistency.

    For each observed action in the opponent's history, applies a multiplicative
    weighting curve to the hand type list (ordered strongest-first):
    - "call"/"call_postflop": bell curve centered on middle of range
    - "3bet"/"bet_postflop": exponential decay from top (strongest most likely)
    - "open": uniform (no adjustment)

    Args:
        hand_types: list of hand type strings (ordered strongest-first)
        action_history: list of action_type strings for this specific opponent
        dead_cards: optional set of dead card IDs for combo expansion

    Returns:
        combo_weights: (n_combos,) tensor normalized to sum to 1.0, or None if empty
    """
    n = len(hand_types)
    if n == 0:
        return None

    type_weights = torch.ones(n, dtype=torch.float32)

    for action in action_history:
        if action in ("call", "call_postflop"):
            # Bell curve: middle of range weighted highest
            center = n / 2.0
            for i in range(n):
                dist = abs(i - center) / max(n, 1)
                type_weights[i] *= max(0.1, 1.0 - dist * 1.5)
        elif action in ("3bet", "bet_postflop"):
            # Skew to top: strongest hands most likely
            for i in range(n):
                frac = i / max(n - 1, 1)
                type_weights[i] *= max(0.1, 1.0 - frac * 0.9)
        # "open" → uniform, no change

    # Expand type-level weights to combo-level weights
    dead = dead_cards or set()
    combo_weights_list = []
    for i, ht in enumerate(hand_types):
        combos = _ALL_COMBOS_CACHE[ht]
        n_valid = sum(1 for c1, c2 in combos if c1 not in dead and c2 not in dead)
        combo_weights_list.extend([type_weights[i].item()] * n_valid)

    if not combo_weights_list:
        return None

    w = torch.tensor(combo_weights_list, dtype=torch.float32)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Improvement 3 applied: Range-aware MC equity with weighted sampling
# ---------------------------------------------------------------------------

def gpu_equity_v3(hero_cards, board_cards, opponent_range_combos,
                  n_iters=10000, device="mps", combo_weights=None):
    """Compute hero equity vs opponent ranges via Monte Carlo with optional weighted sampling.

    Args:
        hero_cards: (2,) int64 tensor
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_range_combos: list of (n_combos_i, 2) tensors, one per opponent
        n_iters: MC iterations
        device: torch device
        combo_weights: optional list of (n_combos_i,) weight tensors per opponent.
            If None or entry is None, uses uniform sampling for that opponent.

    Returns:
        equity: float (0-1)
    """
    hero_cards = hero_cards.to(device)
    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    n_board = len(board_cards)
    n_board_needed = 5 - n_board
    n_opponents = len(opponent_range_combos)

    if n_opponents == 0:
        return 1.0

    # Move range combos to device
    opp_ranges = [r.to(device) for r in opponent_range_combos]

    # Empty range fallback
    for i, r in enumerate(opp_ranges):
        if r.shape[0] == 0:
            all_cards = set(range(52))
            dead = set(hero_cards.tolist())
            if n_board > 0:
                dead.update(board_cards.tolist())
            available = sorted(all_cards - dead)
            fallback = []
            for j in range(len(available)):
                for k in range(j + 1, len(available)):
                    fallback.append((available[j], available[k]))
            opp_ranges[i] = torch.tensor(fallback[:200], dtype=torch.long, device=device)

    # Dead cards from hero and board
    hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    hero_board_dead[hero_cards] = True
    if n_board > 0:
        hero_board_dead[board_cards] = True

    # Sample opponent hands from ranges (weighted or uniform)
    opp_hands = []  # list of (n_iters, 2) tensors
    for i, r in enumerate(opp_ranges):
        n_combos = r.shape[0]
        if combo_weights is not None and i < len(combo_weights) and combo_weights[i] is not None:
            w = combo_weights[i].to(device)
            if len(w) != n_combos:
                # Mismatch — fall back to uniform
                indices = torch.randint(0, n_combos, (n_iters,), device=device)
            else:
                indices = torch.multinomial(w, n_iters, replacement=True)
        else:
            indices = torch.randint(0, n_combos, (n_iters,), device=device)
        opp_hands.append(r[indices])  # (n_iters, 2)

    # Build dead-card mask per iteration
    iter_dead = hero_board_dead.unsqueeze(0).expand(n_iters, -1).clone()  # (n_iters, 52)
    for oh in opp_hands:
        iter_dead.scatter_(1, oh, True)

    # Validity mask: no card conflicts between opponents
    valid = torch.ones(n_iters, dtype=torch.bool, device=device)
    if n_opponents > 1:
        all_opp_cards = torch.cat(opp_hands, dim=1)  # (n_iters, 2*n_opponents)
        opp_onehot = F.one_hot(all_opp_cards, 52).sum(dim=1)  # (n_iters, 52)
        valid = valid & (opp_onehot.max(dim=1).values <= 1)

    # Check opponent cards don't overlap with hero/board
    for oh in opp_hands:
        for ci in range(2):
            card = oh[:, ci]
            valid = valid & ~hero_board_dead[card]

    # Board completion via Gumbel-top-k
    if n_board_needed > 0:
        available_mask = ~iter_dead  # (n_iters, 52)
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_indices = keys.topk(n_board_needed, dim=1)

        if n_board > 0:
            full_board = torch.cat([
                board_cards.unsqueeze(0).expand(n_iters, -1),
                board_indices
            ], dim=1)
        else:
            full_board = board_indices
    else:
        full_board = board_cards.unsqueeze(0).expand(n_iters, -1)

    # Hero 7-card hands
    hero_7 = torch.cat([
        hero_cards.unsqueeze(0).expand(n_iters, -1),
        full_board
    ], dim=1)  # (n_iters, 7)

    # Opponent 7-card hands
    opp_hands_stacked = torch.stack(opp_hands, dim=1)  # (n_iters, n_opp, 2)
    opp_7 = torch.cat([
        opp_hands_stacked,
        full_board.unsqueeze(1).expand(-1, n_opponents, -1)
    ], dim=2)  # (n_iters, n_opp, 7)

    # Evaluate hands
    hero_power = evaluate_hands(hero_7)  # (n_iters,)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7)).reshape(n_iters, n_opponents)

    # Win/tie/loss
    best_opp = opp_power.max(dim=1).values
    hero_wins = (hero_power > best_opp).float()

    hero_ties = (hero_power == best_opp)
    n_tied_opps = (opp_power == hero_power.unsqueeze(1)).sum(dim=1)
    tie_share = hero_ties.float() / (n_tied_opps.float() + 1.0)

    results = hero_wins + tie_share  # (n_iters,)

    # Only count valid iterations
    if valid.sum() < 10:
        return results.mean().item()

    return results[valid].mean().item()


# ---------------------------------------------------------------------------
# Improvement 1: Per-combo equity for opponent response modeling
# ---------------------------------------------------------------------------

MAX_BATCH = 5000  # Max evaluations per GPU batch (tune for MPS memory)


def gpu_equity_per_combo(hero_cards, board_cards, opponent_combos,
                         n_iters_per_combo=30, device="mps"):
    """Compute hero equity against each individual opponent combo.

    Instead of sampling random combos, fixes each opponent combo and samples
    board completions. Total GPU evaluations = n_combos * n_iters_per_combo.

    Args:
        hero_cards: (2,) int64 tensor
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_combos: (n_combos, 2) int64 tensor (single opponent's range)
        n_iters_per_combo: board samples per combo (default 30)
        device: torch device

    Returns:
        hero_eq: (n_combos,) tensor of hero equity vs each combo
    """
    hero_cards = hero_cards.to(device)
    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    opponent_combos = opponent_combos.to(device)
    n_combos = opponent_combos.shape[0]

    if n_combos == 0:
        return torch.tensor([], dtype=torch.float32, device=device)

    n_board = len(board_cards)
    n_board_needed = 5 - n_board
    total = n_combos * n_iters_per_combo

    # Process in batches if too large
    if total > MAX_BATCH:
        results = []
        batch_combos = max(1, MAX_BATCH // n_iters_per_combo)
        for start in range(0, n_combos, batch_combos):
            end = min(start + batch_combos, n_combos)
            chunk = opponent_combos[start:end]
            chunk_eq = _equity_per_combo_batch(
                hero_cards, board_cards, chunk,
                n_iters_per_combo, n_board, n_board_needed, device
            )
            results.append(chunk_eq)
        return torch.cat(results, dim=0)
    else:
        return _equity_per_combo_batch(
            hero_cards, board_cards, opponent_combos,
            n_iters_per_combo, n_board, n_board_needed, device
        )


def _equity_per_combo_batch(hero_cards, board_cards, combos,
                            n_iters_per_combo, n_board, n_board_needed, device):
    """Internal: compute per-combo equity for a batch of combos.

    Args:
        hero_cards: (2,) on device
        board_cards: (B,) on device
        combos: (batch_combos, 2) on device
        n_iters_per_combo: int
        n_board: int, current board card count
        n_board_needed: int, cards to complete board
        device: torch device

    Returns:
        (batch_combos,) tensor of hero equity per combo
    """
    n_combos = combos.shape[0]
    total = n_combos * n_iters_per_combo

    # Expand: each combo repeated n_iters_per_combo times
    opp_hands = combos.repeat_interleave(n_iters_per_combo, dim=0)  # (total, 2)

    # Filter out combos that conflict with hero/board
    hero_set = set(hero_cards.tolist())
    board_set = set(board_cards.tolist()) if n_board > 0 else set()
    dead_set = hero_set | board_set

    # Build per-row dead mask: hero + board + that row's opp cards
    hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    hero_board_dead[hero_cards] = True
    if n_board > 0:
        hero_board_dead[board_cards] = True

    iter_dead = hero_board_dead.unsqueeze(0).expand(total, -1).clone()  # (total, 52)
    iter_dead.scatter_(1, opp_hands, True)

    # Validity: opponent cards must not conflict with hero/board
    valid = torch.ones(total, dtype=torch.bool, device=device)
    for ci in range(2):
        card = opp_hands[:, ci]
        valid = valid & ~hero_board_dead[card]

    # Board completion via Gumbel-top-k with per-row dead masks
    if n_board_needed > 0:
        available_mask = ~iter_dead  # (total, 52)
        keys = torch.rand(total, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_indices = keys.topk(n_board_needed, dim=1)  # (total, n_board_needed)

        if n_board > 0:
            full_board = torch.cat([
                board_cards.unsqueeze(0).expand(total, -1),
                board_indices
            ], dim=1)
        else:
            full_board = board_indices
    else:
        full_board = board_cards.unsqueeze(0).expand(total, -1)

    # Build 7-card hands
    hero_7 = torch.cat([
        hero_cards.unsqueeze(0).expand(total, -1),
        full_board
    ], dim=1)  # (total, 7)

    opp_7 = torch.cat([opp_hands, full_board], dim=1)  # (total, 7)

    # Evaluate
    hero_power = evaluate_hands(hero_7)  # (total,)
    opp_power = evaluate_hands(opp_7)    # (total,)

    wins = (hero_power > opp_power).float()
    ties = (hero_power == opp_power).float() * 0.5

    results = wins + ties  # (total,)

    # Invalidate conflicting rows
    results[~valid] = 0.5  # Neutral equity for invalid combos

    # Reshape and average per combo
    results = results.view(n_combos, n_iters_per_combo)
    valid_reshaped = valid.view(n_combos, n_iters_per_combo)

    # Per-combo mean (only valid iterations)
    valid_counts = valid_reshaped.float().sum(dim=1).clamp(min=1)
    per_combo_eq = (results * valid_reshaped.float()).sum(dim=1) / valid_counts

    return per_combo_eq  # (n_combos,)


# ---------------------------------------------------------------------------
# Main EV calculator
# ---------------------------------------------------------------------------

def _compute_reraise_threshold(call_cost, pot_after_raise, street, stack, pot):
    """Compute dynamic reraise threshold based on pot odds, street, and SPR.

    Returns a threshold (0.5-0.95) for opponent equity above which they reraise.
    """
    # Base: inverse pot odds the opponent gets
    if pot_after_raise + call_cost > 0:
        base = 1.0 - (call_cost / (pot_after_raise + call_cost))
    else:
        base = 0.75

    # Street factor: tighter reraise on later streets
    street_factors = {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.0}
    street_factor = street_factors.get(street, 1.0)

    # SPR factor: more shoving when shallow
    spr = stack / max(pot, 1e-6)
    if spr < 2.0:
        spr_factor = 0.85
    elif spr < 4.0:
        spr_factor = 0.92
    else:
        spr_factor = 1.0

    return max(0.5, min(0.95, base * street_factor * spr_factor))


def _prepare_ev_state_v3(hero_cards, board_cards, opponent_range_hand_types,
                          n_iters=3000, device="mps",
                          hero_position=0, street=0, n_players=6,
                          eqr_enabled=True,
                          combo_response_iters=30,
                          reraise_threshold=0.75,
                          weighted_sampling=True,
                          action_history=None,
                          opponent_positions=None):
    """Pre-compute all raise_frac-independent quantities for compute_ev_v3.

    Splits compute_ev_v3's expensive MC work into a one-shot prep step so a
    caller that wants EVs at many raise sizes (e.g. dataset generation
    iterating all raise bins for the same decision point) can amortise the
    MC cost. Pair with _compute_ev_v3_from_state.

    Returns a state dict carrying:
      - raw_equity (scalar): hero equity vs full opp ranges (any raise size)
      - opp_eq_cpu (tensor or None): per-primary-combo hero equity, used for
        fold/call/reraise classification (raise size only changes the
        threshold cuts, not these values)
      - reraise_mask (tensor or None): reraise_threshold is constant, so
        which primary combos reraise is raise-size independent
      - eq_vs_reraisers (scalar or None): hero equity vs reraisers, also
        raise-size independent for the same reason

    NOTE: eq_vs_callers is NOT cached here — call_mask depends on
    fold_threshold which depends on raise_frac. It is computed inside
    _compute_ev_v3_from_state.
    """
    dead = set(hero_cards.tolist())
    if len(board_cards) > 0:
        dead.update(board_cards.tolist())

    opp_combos = [expand_range(ht_list, dead) for ht_list in opponent_range_hand_types]

    combo_weights = None
    if weighted_sampling and action_history and opponent_positions:
        combo_weights = []
        for i, ht_list in enumerate(opponent_range_hand_types):
            if i < len(opponent_positions):
                opp_pos = opponent_positions[i]
                opp_actions = [a for p, a in action_history if p == opp_pos]
            else:
                opp_actions = []
            w = compute_combo_weights(ht_list, opp_actions, dead_cards=dead)
            combo_weights.append(w)

    active_positions = list(opponent_positions) + [hero_position] if opponent_positions else None
    eqr_raw = _get_eqr(hero_position, street, n_players, active_positions) if eqr_enabled else 1.0

    raw_equity = gpu_equity_v3(hero_cards, board_cards, opp_combos, n_iters, device, combo_weights)

    state = {
        "hero_cards": hero_cards,
        "board_cards": board_cards,
        "opp_combos": opp_combos,
        "combo_weights": combo_weights,
        "eqr_raw": eqr_raw,
        "eqr_enabled": eqr_enabled,
        "raw_equity": raw_equity,
        "reraise_threshold": reraise_threshold,
        "street": street,
        "hero_position": hero_position,
        "opponent_positions": opponent_positions,
        "n_iters": n_iters,
        "device": device,
        "primary_idx": 0,
        "opp_eq_cpu": None,
        "reraise_mask": None,
        "primary_combos": None,
        "eq_vs_reraisers": None,
    }

    primary_idx = 0
    if len(opp_combos) > 0 and opp_combos[primary_idx].shape[0] > 0:
        primary_combos = opp_combos[primary_idx]

        hero_eq_per_combo = gpu_equity_per_combo(
            hero_cards, board_cards, primary_combos,
            combo_response_iters, device
        )
        opp_eq_per_combo = 1.0 - hero_eq_per_combo
        opp_eq_cpu = opp_eq_per_combo.cpu()
        reraise_mask = opp_eq_cpu > reraise_threshold

        state["primary_combos"] = primary_combos
        state["opp_eq_cpu"] = opp_eq_cpu
        state["reraise_mask"] = reraise_mask

        if reraise_mask.any():
            reraising_combos = primary_combos[reraise_mask]
            all_reraise_combos = [
                reraising_combos if j == primary_idx else c
                for j, c in enumerate(opp_combos)
            ]
            reraise_weights = None
            if combo_weights is not None and combo_weights[primary_idx] is not None:
                rw = combo_weights[primary_idx][reraise_mask]
                rw_sum = rw.sum()
                if rw_sum > 0:
                    rw = rw / rw_sum
                reraise_weights = [
                    rw if j == primary_idx else
                    (combo_weights[j] if combo_weights is not None and j < len(combo_weights) else None)
                    for j in range(len(opp_combos))
                ]
            state["eq_vs_reraisers"] = gpu_equity_v3(
                hero_cards, board_cards, all_reraise_combos,
                n_iters, device, reraise_weights
            )

    return state


def _compute_ev_v3_from_state(state, pot, facing_bet, stack, hero_invested,
                               raise_frac=1.0, dynamic_reraise=False):
    """Compute (fold_ev, call_ev, raise_ev, best_ev) from precomputed state.

    Reuses raw_equity, opp_eq_cpu, reraise_mask, eq_vs_reraisers from state.
    Only re-runs MC for eq_vs_callers (call_mask depends on raise_frac).
    """
    eqr_raw = state["eqr_raw"]
    eqr_enabled = state["eqr_enabled"]
    raw_equity = state["raw_equity"]
    street = state["street"]
    opp_combos = state["opp_combos"]
    hero_cards = state["hero_cards"]
    board_cards = state["board_cards"]
    n_iters = state["n_iters"]
    device = state["device"]
    combo_weights = state["combo_weights"]
    opponent_positions = state["opponent_positions"]
    hero_position = state["hero_position"]

    spr = stack / max(pot, 1e-6)
    spr_eqr_factor = min(1.0, spr / 6.0)
    eqr = 1.0 + (eqr_raw - 1.0) * spr_eqr_factor

    fold_ev = -hero_invested

    eff_equity = max(0.0, min(1.0, raw_equity * eqr))
    total_call_investment = hero_invested + facing_bet
    call_ev = eff_equity * (pot - hero_invested) + (1 - eff_equity) * (-total_call_investment)
    _street_discount = {0: 0.92, 1: 0.95, 2: 0.98, 3: 1.0}.get(street, 1.0)
    call_ev *= _street_discount

    raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
    total_raise = hero_invested + raise_amount
    new_pot = pot + facing_bet + raise_amount

    call_cost = raise_amount
    pot_after_raise = new_pot
    raw_fold_threshold = call_cost / pot_after_raise if pot_after_raise > 0 else 0.5
    fold_threshold = raw_fold_threshold ** 0.85
    if opponent_positions and street > 0:
        max_opp_pos = max(opponent_positions)
        if max_opp_pos < hero_position:
            fold_threshold *= 1.08
        elif min(opponent_positions) > hero_position:
            fold_threshold *= 0.95

    opp_eq_cpu = state["opp_eq_cpu"]
    reraise_mask = state["reraise_mask"]
    primary_combos = state["primary_combos"]
    primary_idx = state["primary_idx"]
    eq_vs_reraisers = state["eq_vs_reraisers"]

    if opp_eq_cpu is not None:
        fold_mask = opp_eq_cpu < fold_threshold
        call_mask = ~fold_mask & ~reraise_mask

        n_total = float(len(opp_eq_cpu))
        p_fold = fold_mask.float().sum().item() / n_total if n_total > 0 else 0.0
        p_reraise = reraise_mask.float().sum().item() / n_total if n_total > 0 else 0.0
        p_call = call_mask.float().sum().item() / n_total if n_total > 0 else 1.0

        mean_opp_eq = opp_eq_cpu.mean().item()
        blocker_adj = 1.0 + (0.5 - mean_opp_eq) * 0.12
        p_fold = p_fold * blocker_adj
        p_total = p_fold + p_call + p_reraise
        if p_total > 1e-6:
            p_fold /= p_total
            p_call /= p_total
            p_reraise /= p_total

        if call_mask.any():
            calling_combos = primary_combos[call_mask]
            all_calling_combos = []
            for j, c in enumerate(opp_combos):
                if j == primary_idx:
                    all_calling_combos.append(calling_combos)
                else:
                    all_calling_combos.append(c)
            calling_weights = None
            if combo_weights is not None and combo_weights[primary_idx] is not None:
                w = combo_weights[primary_idx]
                calling_w = w[call_mask]
                w_sum = calling_w.sum()
                if w_sum > 0:
                    calling_w = calling_w / w_sum
                calling_weights = []
                for j in range(len(opp_combos)):
                    if j == primary_idx:
                        calling_weights.append(calling_w)
                    elif combo_weights is not None and j < len(combo_weights):
                        calling_weights.append(combo_weights[j])
                    else:
                        calling_weights.append(None)
            eq_vs_callers = gpu_equity_v3(
                hero_cards, board_cards, all_calling_combos,
                n_iters, device, calling_weights
            )
        else:
            eq_vs_callers = raw_equity

        eff_eq_callers = max(0.0, min(1.0, eq_vs_callers * eqr)) if eqr_enabled else eq_vs_callers
        showdown_ev = eff_eq_callers * (new_pot - total_raise) + (1 - eff_eq_callers) * (-total_raise)

        if p_reraise > 0 and eq_vs_reraisers is not None:
            if dynamic_reraise:
                actual_reraise_threshold_val = _compute_reraise_threshold(
                    call_cost, pot_after_raise, street, stack, pot)
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
            p_hero_continues = 1.0 / (1.0 + math.exp(-15.0 * (eq_vs_reraisers - hero_continue_threshold)))
            eff_eq_reraise = max(0.0, min(1.0, eq_vs_reraisers * eqr)) if eqr_enabled else eq_vs_reraisers
            ev_continue = eff_eq_reraise * (reraise_pot - total_reraise_cost) + \
                          (1 - eff_eq_reraise) * (-total_reraise_cost)
            ev_on_reraise = p_hero_continues * ev_continue + (1 - p_hero_continues) * (-total_raise)
            reraise_discount = 0.3
            geometric_factor = min(1.5, 1.0 / max(0.5, 1.0 - p_reraise * reraise_discount))
            ev_on_reraise *= geometric_factor
        else:
            ev_on_reraise = -total_raise

        raise_ev = (
            p_fold * (pot - hero_invested)
            + p_call * showdown_ev
            + p_reraise * ev_on_reraise
        )
    else:
        raise_ev = pot - hero_invested

    best_ev = max(fold_ev, call_ev, raise_ev)
    return fold_ev, call_ev, raise_ev, best_ev


def compute_ev_v3(hero_cards, board_cards, opponent_range_hand_types,
                  pot, facing_bet, stack, hero_invested,
                  raise_frac=1.0, n_iters=3000, device="mps",
                  hero_position=0, street=0, n_players=6,
                  eqr_enabled=True,
                  combo_response_iters=30,
                  reraise_threshold=0.75,
                  weighted_sampling=True,
                  action_history=None,
                  opponent_positions=None,
                  dynamic_reraise=False):
    """Compute EV for fold/call/raise with per-combo response and EQR.

    Thin wrapper that prepares state and computes EV for a single raise_frac.
    For multi-raise_frac use cases (e.g. dataset generation), call
    _prepare_ev_state_v3 once and _compute_ev_v3_from_state per raise_frac.
    """
    state = _prepare_ev_state_v3(
        hero_cards, board_cards, opponent_range_hand_types,
        n_iters=n_iters, device=device,
        hero_position=hero_position, street=street, n_players=n_players,
        eqr_enabled=eqr_enabled,
        combo_response_iters=combo_response_iters,
        reraise_threshold=reraise_threshold,
        weighted_sampling=weighted_sampling,
        action_history=action_history,
        opponent_positions=opponent_positions,
    )
    return _compute_ev_v3_from_state(
        state, pot, facing_bet, stack, hero_invested,
        raise_frac=raise_frac, dynamic_reraise=dynamic_reraise,
    )
