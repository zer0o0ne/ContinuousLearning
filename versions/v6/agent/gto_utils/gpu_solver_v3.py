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
            # B.5.9: preflop the BB (seat 1) closes the action — it acts LAST,
            # hence in position. The old `max(active_positions)` (e.g. the BTN)
            # acts BEFORE the blinds preflop, so it was the wrong closer.
            is_ip = (hero_position == 1)
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

    idx = torch.arange(n, dtype=torch.float32)
    type_weights = torch.ones(n, dtype=torch.float32)

    for action in action_history:
        if action in ("call", "call_postflop"):
            center = n / 2.0
            dist = (idx - center).abs() / max(n, 1)
            type_weights *= (1.0 - dist * 1.5).clamp(min=0.1)
        elif action in ("3bet", "bet_postflop"):
            frac = idx / max(n - 1, 1)
            type_weights *= (1.0 - frac * 0.9).clamp(min=0.1)

    # Vectorized combo expansion and dead-card filtering
    dead = dead_cards or set()

    # 1. Pre-build flat tensor of all combos and a counts tensor for repeat_interleave
    all_combos_flat = []
    combo_counts = []
    for ht in hand_types:
        combos = _ALL_COMBOS_CACHE[ht]
        all_combos_flat.extend(combos)
        combo_counts.append(len(combos))

    if not all_combos_flat:
        return None

    # (total_combos, 2) tensor of card pairs
    all_combos = torch.tensor(all_combos_flat, dtype=torch.long)
    combo_counts_t = torch.tensor(combo_counts, dtype=torch.long)

    # 2. Create dead-card boolean mask (size 52)
    dead_mask = torch.zeros(52, dtype=torch.bool)
    if dead:
        dead_indices = torch.tensor(sorted(dead), dtype=torch.long)
        dead_mask[dead_indices] = True

    # 3. Filter combos: neither card is dead
    c1_dead = dead_mask[all_combos[:, 0]]
    c2_dead = dead_mask[all_combos[:, 1]]
    alive = ~c1_dead & ~c2_dead

    if not alive.any():
        return None

    # 4. Repeat type_weights per combo count, then apply alive filter
    w = torch.repeat_interleave(type_weights, combo_counts_t)
    w = w[alive]

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
                          opponent_positions=None,
                          threshold_smoothing=None,
                          dynamic_reraise=False,
                          polarized_reraise=None):
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

    NOTE post-R2: `p_reraise_per_combo`, `reraise_mask` and `eq_vs_reraisers`
    were precomputed here in R1 because `reraise_threshold` was static. R2
    introduces dynamic per-raise_frac thresholds (`call_cost` depends on
    `raise_frac`), so the reraise weights/mask + downstream
    `eq_vs_reraisers` MC must move into `_compute_ev_v3_from_state`. These
    slots stay in state as `None` placeholders for backward-compat with any
    external code that reads them; the per-call function recomputes locally.

    `eq_vs_callers` is also NOT cached here — call_mask depends on
    fold_threshold which depends on raise_frac.

    The `dynamic_reraise` flag is stored on state so the per-call function
    can derive the threshold from `(call_cost, pot_after_raise, street,
    stack, pot)` instead of using the static `reraise_threshold` fallback.
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

    # B.5.4: model the AGGRESSOR (the last opponent to raise this hand) as the
    # primary opponent for the detailed call/reraise response, not seat 0. The
    # aggressor is whom a raise most directly contests. Fall back to the first
    # opponent when no opponent has raised yet (hero is opening the pot).
    primary_idx = 0
    if opponent_positions and action_history:
        _raise_acts = {"open", "3bet", "bet_postflop"}
        _opp_pos_list = list(opponent_positions)
        for p, a in action_history:
            if a in _raise_acts and p in _opp_pos_list:
                primary_idx = _opp_pos_list.index(p)  # keep the LAST such raiser

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
        "primary_idx": primary_idx,
        "opp_eq_cpu": None,
        "opp_eq_by_opp": None,
        "reraise_mask": None,
        "p_reraise_per_combo": None,
        "primary_combos": None,
        "eq_vs_reraisers": None,
        "threshold_smoothing": threshold_smoothing,
        "dynamic_reraise": bool(dynamic_reraise),
        "polarized_reraise": polarized_reraise,
    }

    # B.5.4: per-opponent hero-equity-per-combo for EVERY live opponent (not
    # just the primary), so `_compute_ev_v3_from_state` can derive each
    # opponent's fold probability and combine them as Π p_fold. ~N× the
    # per-combo MC of the old primary-only path. The primary's slice still
    # drives the detailed call/reraise response.
    opp_eq_by_opp = [None] * len(opp_combos)
    for j, combos_j in enumerate(opp_combos):
        if combos_j.shape[0] > 0:
            hero_eq_j = gpu_equity_per_combo(
                hero_cards, board_cards, combos_j, combo_response_iters, device
            )
            opp_eq_by_opp[j] = (1.0 - hero_eq_j).cpu()
    state["opp_eq_by_opp"] = opp_eq_by_opp

    if len(opp_combos) > primary_idx and opp_combos[primary_idx].shape[0] > 0:
        # State carries the precomputed per-combo equity only. Reraise
        # weights/mask + eq_vs_reraisers depend on raise_frac (post-R2) and
        # are recomputed per-call in _compute_ev_v3_from_state.
        state["primary_combos"] = opp_combos[primary_idx]
        state["opp_eq_cpu"] = opp_eq_by_opp[primary_idx]

    return state


def _compute_ev_v3_from_state(state, pot, facing_bet, stack, hero_invested,
                               raise_frac=1.0, dynamic_reraise=False):
    """Compute (fold_ev, call_ev, raise_ev, best_ev) from precomputed state.

    Reuses raw_equity and opp_eq_cpu from state. Recomputes reraise weights/
    eq_vs_reraisers per raise_frac because the reraise threshold is now
    raise_frac-dependent when `state["dynamic_reraise"]` is True
    (call_cost = raise_amount enters `_compute_reraise_threshold`). When
    `state["dynamic_reraise"]` is False the static `state["reraise_threshold"]`
    is reused on every call, matching R1 numerics for the static path.
    Also runs MC for eq_vs_callers (call_mask depends on fold_threshold which
    depends on raise_frac).
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
    # B.5.5: apply the street discount only to a POSITIVE call EV. The discount
    # models equity-realization risk on a marginal call; multiplying a NEGATIVE
    # call_ev by <1 would make a losing call look BETTER. `min(x, x*d)` discounts
    # when x > 0 and is a no-op when x <= 0.
    call_ev = min(call_ev, call_ev * _street_discount)

    raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
    total_raise = hero_invested + raise_amount
    # Audit B.1: `pot` already contains the opponent's `facing_bet` (see call_ev
    # above: a win pays `pot - hero_invested`). When the opponent calls hero's
    # raise they add `raise_amount - facing_bet` on top of what they already
    # have in. So the final pot is the existing `pot` (hero matched + dead money)
    # plus hero's `raise_amount` plus the opponent's call of the raise.
    # The old `pot + facing_bet + raise_amount` double-counted facing_bet and
    # omitted the opponent's call, making EV(raise) - EV(check) = -(1-eq)*b < 0
    # for any equity at facing_bet=0 (value bets never paid). Fixed:
    #   EV(raise|call) - EV(check) = b*(2*eq - 1)  (> 0 for eq > 0.5).
    new_pot = pot + raise_amount + (raise_amount - facing_bet)

    call_cost = raise_amount - facing_bet
    pot_after_raise = new_pot
    raw_fold_threshold = call_cost / pot_after_raise if pot_after_raise > 0 else 0.5
    fold_threshold = raw_fold_threshold ** 0.85
    if opponent_positions and street > 0:
        max_opp_pos = max(opponent_positions)
        if max_opp_pos < hero_position:
            fold_threshold *= 1.08
        elif min(opponent_positions) > hero_position:
            fold_threshold *= 0.95

    # ------------------------------------------------------------------
    # R2: derive the effective reraise threshold for this raise_frac.
    # When dynamic_reraise is on, `_compute_reraise_threshold` consumes
    # `(call_cost=raise_amount, pot_after_raise=new_pot, street, stack, pot)`
    # → threshold varies across raise sizes (clamped to [0.5, 0.95]).
    # When off, fall back to the static config value for backward-compat.
    # The flag is read from state (set by `_prepare_ev_state_v3`). If a
    # caller also passes `dynamic_reraise` as a kwarg, the state value
    # takes precedence (and the kwarg is ignored) — same source of truth.
    # ------------------------------------------------------------------
    dyn_th = bool(state.get("dynamic_reraise", False))
    static_th = state["reraise_threshold"]
    if dyn_th:
        actual_reraise_threshold = _compute_reraise_threshold(
            call_cost, pot_after_raise, street, stack, pot)
    else:
        actual_reraise_threshold = static_th

    opp_eq_cpu = state["opp_eq_cpu"]
    primary_combos = state["primary_combos"]
    primary_idx = state["primary_idx"]
    smoothing_cfg = state.get("threshold_smoothing") or {}
    smoothing_enabled = bool(smoothing_cfg.get("enabled", False))
    beta_reraise = float(smoothing_cfg.get("beta_reraise", 0.07))

    # Per-raise_frac reraise weights/mask. Moved here from
    # `_prepare_ev_state_v3` (R2) because `actual_reraise_threshold` now
    # depends on `raise_frac`.
    if opp_eq_cpu is not None:
        if smoothing_enabled:
            p_reraise_per_combo = torch.sigmoid(
                (opp_eq_cpu - actual_reraise_threshold) / beta_reraise
            )
            reraise_mask = None
        else:
            p_reraise_per_combo = None
            reraise_mask = opp_eq_cpu > actual_reraise_threshold
    else:
        p_reraise_per_combo = None
        reraise_mask = None

    # ------------------------------------------------------------------
    # R3: polarize the reraise range — add a bluff component (low-equity
    # blockers) on top of the value side just computed above. Real GTO
    # 3-bet ranges are top-equity value + low-equity blockers; the pure
    # sigmoid above captures only the value side, so trained agents see
    # under-bluffed opponents and converge to over-folds in response.
    #
    # blocker_strength is high when opp_eq is low (good blocker / weak
    # hand). bluff_score sigmoids the blocker_strength against a
    # bluff_threshold; bluff_frequency caps the bluff additive mass.
    # The final `p_reraise_per_combo` may exceed 1.0 on some combos when
    # value and bluff overlap (rare) — clamp at 1.0 here; cross-combo
    # bookkeeping is handled by the `denom.clamp(min=1.0)` step below
    # in the smoothing block, so the per-action probability sum stays in
    # [0, 1].
    #
    # Polarization is structurally tied to smoothing: hard masks have no
    # concept of fractional weight, so when `threshold_smoothing.enabled`
    # is False we skip polarization regardless of `polarized_reraise.enabled`.
    # ------------------------------------------------------------------
    polar_cfg = state.get("polarized_reraise") or {}
    polar_enabled = bool(polar_cfg.get("enabled", False))

    if smoothing_enabled and polar_enabled and opp_eq_cpu is not None:
        value_score = p_reraise_per_combo
        bluff_thresh = float(polar_cfg.get("bluff_threshold", 0.25))
        beta_bluff = float(polar_cfg.get("beta_bluff", 0.10))
        bluff_freq = float(polar_cfg.get("bluff_frequency", 0.30))

        blocker_strength = (0.5 - opp_eq_cpu).clamp(min=0.0)
        bluff_score = torch.sigmoid((blocker_strength - bluff_thresh) / beta_bluff)

        p_reraise_per_combo = (value_score + bluff_freq * bluff_score).clamp(max=1.0)

    # Per-raise_frac eq_vs_reraisers MC. Was cached in state pre-R2 when
    # the reraise threshold was static; must move here for the dynamic
    # threshold to take effect. One extra MC per raise_frac is acceptable
    # — see plan §"Edge cases / safety".
    eq_vs_reraisers = None
    if opp_eq_cpu is not None:
        if smoothing_enabled and p_reraise_per_combo.sum().item() > 1e-6:
            if combo_weights is not None and combo_weights[primary_idx] is not None:
                base_w = combo_weights[primary_idx]
                combined = base_w * p_reraise_per_combo
                csum = combined.sum()
                if csum.item() > 0:
                    combined = combined / csum
            else:
                combined = p_reraise_per_combo / p_reraise_per_combo.sum()
            # E.3.1: analytical equity from per-combo table
            hero_eq_per_combo = 1.0 - opp_eq_cpu
            eq_vs_reraisers = float((hero_eq_per_combo * combined).sum().item())
        elif (not smoothing_enabled) and reraise_mask is not None and reraise_mask.any():
            # E.3.1: analytical equity from per-combo table
            hero_eq_per_combo = 1.0 - opp_eq_cpu
            masked_eq = hero_eq_per_combo[reraise_mask]
            if combo_weights is not None and combo_weights[primary_idx] is not None:
                rw = combo_weights[primary_idx][reraise_mask]
                rw_sum = rw.sum()
                if rw_sum.item() > 0:
                    eq_vs_reraisers = float((masked_eq * rw / rw_sum).sum().item())
                else:
                    eq_vs_reraisers = float(masked_eq.mean().item())
            else:
                eq_vs_reraisers = float(masked_eq.mean().item())

    if opp_eq_cpu is not None:
        if smoothing_enabled:
            beta_fold = float(smoothing_cfg.get("beta_fold", 0.07))

            # Sigmoid soft weights — symmetric across combos. Fold dominates
            # when opp_eq is well below fold_threshold; reraise dominates well
            # above actual_reraise_threshold; call fills the rest. The clamp
            # below prevents the edge case where both fold and reraise sigmoids
            # exceed 1 on the same combo (rare but possible if the two
            # thresholds overlap).
            p_fold_per_combo = torch.sigmoid(
                (fold_threshold - opp_eq_cpu) / beta_fold
            )
            denom = (p_fold_per_combo + p_reraise_per_combo).clamp(min=1.0)
            p_fold_per_combo = p_fold_per_combo / denom
            p_reraise_per_combo = p_reraise_per_combo / denom
            p_call_per_combo = (1.0 - p_fold_per_combo - p_reraise_per_combo).clamp(min=0.0)

            # B.5.8: aggregate the per-combo response probabilities with the
            # SAME combo weights that gate the equities (weighted_sampling),
            # not a flat mean over combos. `combo_weights[primary_idx]` is
            # already normalized to sum 1; fall back to a flat mean when no
            # weights are present.
            w_primary = (combo_weights[primary_idx]
                         if combo_weights is not None
                         and combo_weights[primary_idx] is not None
                         else None)
            if w_primary is not None and len(w_primary) == len(p_fold_per_combo):
                p_fold = float((p_fold_per_combo * w_primary).sum().item())
                p_reraise = float((p_reraise_per_combo * w_primary).sum().item())
                p_call = float((p_call_per_combo * w_primary).sum().item())
            else:
                p_fold = p_fold_per_combo.mean().item()
                p_reraise = p_reraise_per_combo.mean().item()
                p_call = p_call_per_combo.mean().item()
        else:
            # Legacy hard-mask path (preserved bit-for-bit vs R1).
            fold_mask = opp_eq_cpu < fold_threshold
            call_mask = ~fold_mask & ~reraise_mask

            # B.5.8: weight the fold/call/reraise fractions by combo weights
            # (consistent with the equities they gate), not a flat combo count.
            w_primary = (combo_weights[primary_idx]
                         if combo_weights is not None
                         and combo_weights[primary_idx] is not None
                         else None)
            n_total = float(len(opp_eq_cpu))
            if w_primary is not None and len(w_primary) == len(opp_eq_cpu):
                p_fold = float((fold_mask.float() * w_primary).sum().item())
                p_reraise = float((reraise_mask.float() * w_primary).sum().item())
                p_call = float((call_mask.float() * w_primary).sum().item())
            elif n_total > 0:
                p_fold = fold_mask.float().sum().item() / n_total
                p_reraise = reraise_mask.float().sum().item() / n_total
                p_call = call_mask.float().sum().item() / n_total
            else:
                p_fold, p_reraise, p_call = 0.0, 0.0, 1.0
            p_call_per_combo = None  # unused on legacy path

        mean_opp_eq = opp_eq_cpu.mean().item()
        blocker_adj = 1.0 + (0.5 - mean_opp_eq) * 0.12
        p_fold = p_fold * blocker_adj
        p_total = p_fold + p_call + p_reraise
        if p_total > 1e-6:
            p_fold /= p_total
            p_call /= p_total
            p_reraise /= p_total

        # B.5.4: probability that EVERY OTHER live opponent (besides the
        # primary) also folds to the raise. Together with the primary's fold
        # probability this yields Π p_fold_i — the only way hero wins the pot
        # uncontested with >1 opponent. With a single opponent this product is
        # empty (== 1) and the raise EV below reduces EXACTLY to the heads-up
        # model. Each opponent's fold prob uses the same fold_threshold and
        # smoothing as the primary, weighted by that opponent's combo weights.
        opp_eq_by_opp = state.get("opp_eq_by_opp") or []
        beta_fold_mw = float((state.get("threshold_smoothing") or {}).get("beta_fold", 0.07))
        p_others_all_fold = 1.0
        for j, oeq in enumerate(opp_eq_by_opp):
            if j == primary_idx or oeq is None:
                continue
            if smoothing_enabled:
                pf_j = torch.sigmoid((fold_threshold - oeq) / beta_fold_mw)
            else:
                pf_j = (oeq < fold_threshold).float()
            wj = (combo_weights[j]
                  if combo_weights is not None and j < len(combo_weights)
                  and combo_weights[j] is not None else None)
            if wj is not None and len(wj) == len(pf_j):
                p_fold_j = float((pf_j * wj).sum().item())
            else:
                p_fold_j = float(pf_j.mean().item())
            p_others_all_fold *= p_fold_j

        if smoothing_enabled:
            if p_call_per_combo.sum().item() > 1e-6:
                if combo_weights is not None and combo_weights[primary_idx] is not None:
                    base_w = combo_weights[primary_idx]
                    combined = base_w * p_call_per_combo
                    csum = combined.sum()
                    if csum.item() > 0:
                        combined = combined / csum
                else:
                    combined = p_call_per_combo / p_call_per_combo.sum()

                # E.3.1: analytical equity from per-combo table
                hero_eq_per_combo = 1.0 - opp_eq_cpu
                eq_vs_callers = float((hero_eq_per_combo * combined).sum().item())
            else:
                eq_vs_callers = raw_equity
        elif call_mask.any():
            # E.3.1: analytical equity from per-combo table
            hero_eq_per_combo = 1.0 - opp_eq_cpu
            masked_eq = hero_eq_per_combo[call_mask]
            if combo_weights is not None and combo_weights[primary_idx] is not None:
                cw = combo_weights[primary_idx][call_mask]
                cw_sum = cw.sum()
                if cw_sum.item() > 0:
                    eq_vs_callers = float((masked_eq * cw / cw_sum).sum().item())
                else:
                    eq_vs_callers = float(masked_eq.mean().item())
            else:
                eq_vs_callers = float(masked_eq.mean().item())
        else:
            eq_vs_callers = raw_equity

        eff_eq_callers = max(0.0, min(1.0, eq_vs_callers * eqr)) if eqr_enabled else eq_vs_callers
        showdown_ev = eff_eq_callers * (new_pot - total_raise) + (1 - eff_eq_callers) * (-total_raise)

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
            p_hero_continues = 1.0 / (1.0 + math.exp(-15.0 * (eq_vs_reraisers - hero_continue_threshold)))
            eff_eq_reraise = max(0.0, min(1.0, eq_vs_reraisers * eqr)) if eqr_enabled else eq_vs_reraisers
            ev_continue = eff_eq_reraise * (reraise_pot - total_reraise_cost) + \
                          (1 - eff_eq_reraise) * (-total_reraise_cost)
            ev_on_reraise = p_hero_continues * ev_continue + (1 - p_hero_continues) * (-total_raise)
            reraise_discount = 0.3
            geometric_factor = min(1.5, 1.0 / max(0.5, 1.0 - p_reraise * reraise_discount))
            # B.5.7: the geometric pot-growth bonus only amplifies a POSITIVE
            # ev_on_reraise; applying the (>= 1) factor to a negative EV would
            # over-penalize a fold-to-reraise line.
            if ev_on_reraise > 0:
                ev_on_reraise *= geometric_factor
        else:
            ev_on_reraise = -total_raise

        # B.5.4: multiway-aware fold term. When the primary folds, hero wins the
        # pot uncontested ONLY if every other opponent folds too
        # (p_others_all_fold); otherwise it is a showdown vs the remaining field
        # (approximated by showdown_ev — the equity-vs-callers term that already
        # includes the non-primary opponents' full ranges). For a single
        # opponent p_others_all_fold == 1 and this is exactly the old model:
        #   p_fold*(pot-hero_invested) + p_call*showdown + p_reraise*ev_reraise.
        fold_term = p_fold * (
            p_others_all_fold * (pot - hero_invested)
            + (1.0 - p_others_all_fold) * showdown_ev
        )
        raise_ev = fold_term + p_call * showdown_ev + p_reraise * ev_on_reraise
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
                  dynamic_reraise=False,
                  threshold_smoothing=None,
                  polarized_reraise=None):
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
        threshold_smoothing=threshold_smoothing,
        dynamic_reraise=dynamic_reraise,
        polarized_reraise=polarized_reraise,
    )
    return _compute_ev_v3_from_state(
        state, pot, facing_bet, stack, hero_invested,
        raise_frac=raise_frac, dynamic_reraise=dynamic_reraise,
    )
