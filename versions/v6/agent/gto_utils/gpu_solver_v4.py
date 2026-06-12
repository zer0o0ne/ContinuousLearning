"""
GPU poker solver v4 — batched marginalized opponent modeling with Bayesian range updates.

Extends v3 with:
1. Batched equity: compute equity for N hero hands in one GPU pass,
   sharing opponent sampling and board completion.
2. Batched per-combo equity: (N, M) equity matrix for response modeling.
3. Batched EV: vectorized fold/call/raise EV across N heroes.
4. Marginalized action probs: weighted average of action distributions
   over an opponent's range.
5. Bayesian range update: posterior weight update after observed action.

Card encoding: card_id 0-51, rank = card_id // 4 (0=2, 12=A), suit = card_id % 4.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Ensure gto_utils is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Re-export everything from v3 (which re-exports from v2/v1)
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
from gpu_solver_v3 import (
    EQR_TABLE,
    _get_eqr,
    compute_combo_weights,
    gpu_equity_v3,
    gpu_equity_per_combo,
    compute_ev_v3,
    MAX_BATCH,
)

# Larger batch for v4 batched operations (tune for GPU memory)
MAX_BATCH_V4 = 50000


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


# ---------------------------------------------------------------------------
# 1. Batched equity: N hero hands, shared MC work
# ---------------------------------------------------------------------------

def gpu_equity_batched(hero_cards_batch, board_cards, opponent_range_combos,
                       n_iters=3000, device="mps", combo_weights=None):
    """Compute equity for N hero hands simultaneously, sharing opponent/board sampling.

    Args:
        hero_cards_batch: (N, 2) int64 tensor — multiple hero hands
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_range_combos: list of (n_combos_i, 2) tensors, one per opponent
        n_iters: MC iterations (shared across all heroes)
        device: torch device
        combo_weights: optional list of (n_combos_i,) weight tensors per opponent

    Returns:
        (N,) tensor of equities, one per hero hand
    """
    hero_cards_batch = hero_cards_batch.to(device)
    N = hero_cards_batch.shape[0]

    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    n_board = len(board_cards)
    n_board_needed = 5 - n_board
    n_opponents = len(opponent_range_combos)

    if n_opponents == 0:
        return torch.ones(N, dtype=torch.float32, device=device)

    # Move ranges to device, fallback for empty
    opp_ranges = []
    for r in opponent_range_combos:
        r = r.to(device)
        if r.shape[0] == 0:
            available = sorted(set(range(52)) - set(board_cards.tolist()))
            fallback = []
            for j in range(len(available)):
                for k in range(j + 1, len(available)):
                    fallback.append((available[j], available[k]))
            r = torch.tensor(fallback[:200], dtype=torch.long, device=device)
        opp_ranges.append(r)

    # --- Shared opponent sampling ---
    opp_hands = []
    for i, r in enumerate(opp_ranges):
        n_combos = r.shape[0]
        if combo_weights is not None and i < len(combo_weights) and combo_weights[i] is not None:
            w = combo_weights[i].to(device)
            if len(w) != n_combos:
                indices = torch.randint(0, n_combos, (n_iters,), device=device)
            else:
                indices = torch.multinomial(w, n_iters, replacement=True)
        else:
            indices = torch.randint(0, n_combos, (n_iters,), device=device)
        opp_hands.append(r[indices])  # (n_iters, 2)

    # --- Shared dead mask: board + opponent cards ---
    shared_dead = torch.zeros(n_iters, 52, dtype=torch.bool, device=device)
    if n_board > 0:
        shared_dead[:, board_cards] = True

    # Fix 1: for small N, add all hero cards to shared_dead before board completion
    # to avoid wasting iterations where board cards overlap with hero cards.
    # For large N, compensate by increasing n_iters.
    small_n_optimization = (N <= 8)
    if small_n_optimization:
        all_hero_cards = hero_cards_batch.reshape(-1).unique()
        shared_dead[:, all_hero_cards] = True
    else:
        n_available = 52 - n_board - 2 * n_opponents
        expected_valid_ratio = max(0.3, 1.0 - (2 * N) / max(n_available, 1))
        n_iters = min(n_iters * 2, int(n_iters / expected_valid_ratio))
        # Re-sample opponent hands with new n_iters
        opp_hands = []
        for i, r in enumerate(opp_ranges):
            n_combos = r.shape[0]
            if combo_weights is not None and i < len(combo_weights) and combo_weights[i] is not None:
                w = combo_weights[i].to(device)
                if len(w) != n_combos:
                    indices = torch.randint(0, n_combos, (n_iters,), device=device)
                else:
                    indices = torch.multinomial(w, n_iters, replacement=True)
            else:
                indices = torch.randint(0, n_combos, (n_iters,), device=device)
            opp_hands.append(r[indices])
        shared_dead = torch.zeros(n_iters, 52, dtype=torch.bool, device=device)
        if n_board > 0:
            shared_dead[:, board_cards] = True

    for oh in opp_hands:
        shared_dead.scatter_(1, oh, True)

    # --- Shared validity: opponent card conflicts ---
    shared_valid = torch.ones(n_iters, dtype=torch.bool, device=device)
    if n_opponents > 1:
        all_opp_cards = torch.cat(opp_hands, dim=1)  # (n_iters, 2*n_opp)
        opp_onehot = F.one_hot(all_opp_cards.long(), 52).sum(dim=1)
        shared_valid = shared_valid & (opp_onehot.max(dim=1).values <= 1)

    # Check opp cards don't overlap with board
    if n_board > 0:
        board_dead = torch.zeros(52, dtype=torch.bool, device=device)
        board_dead[board_cards] = True
        for oh in opp_hands:
            for ci in range(2):
                shared_valid = shared_valid & ~board_dead[oh[:, ci]]

    # --- Shared board completion (avoiding board + opp cards, NOT hero cards) ---
    if n_board_needed > 0:
        available_mask = ~shared_dead
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_indices = keys.topk(n_board_needed, dim=1)

        if n_board > 0:
            full_board = torch.cat([
                board_cards.unsqueeze(0).expand(n_iters, -1),
                board_indices
            ], dim=1)  # (n_iters, 5)
        else:
            full_board = board_indices
    else:
        full_board = board_cards.unsqueeze(0).expand(n_iters, -1)

    # --- Shared opponent evaluation ---
    opp_hands_stacked = torch.stack(opp_hands, dim=1)  # (n_iters, n_opp, 2)
    opp_7 = torch.cat([
        opp_hands_stacked,
        full_board.unsqueeze(1).expand(-1, n_opponents, -1)
    ], dim=2)  # (n_iters, n_opp, 7)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7)).reshape(n_iters, n_opponents)
    best_opp = opp_power.max(dim=1).values  # (n_iters,)

    # --- Per-hero validity: hero cards not in used cards ---
    # used_onehot: board + opp + completed board (NOT hero cards from small-N opt)
    used_onehot = torch.zeros(n_iters, 52, dtype=torch.bool, device=device)
    if n_board > 0:
        used_onehot[:, board_cards] = True
    for oh in opp_hands:
        used_onehot.scatter_(1, oh, True)
    if n_board_needed > 0:
        used_onehot.scatter_(1, board_indices, True)

    # hero_cards_batch: (N, 2)
    c0_conflict = used_onehot[:, hero_cards_batch[:, 0]]  # (n_iters, N)
    c1_conflict = used_onehot[:, hero_cards_batch[:, 1]]  # (n_iters, N)
    hero_valid = ~(c0_conflict | c1_conflict)  # (n_iters, N)
    hero_valid = hero_valid.T  # (N, n_iters)

    # Also check hero cards don't conflict with each other (they shouldn't, but safety)
    # Combined validity
    combined_valid = hero_valid & shared_valid.unsqueeze(0)  # (N, n_iters)

    # --- Hero hand evaluation (batched) ---
    # hero_7: (N, n_iters, 7)
    hero_expanded = hero_cards_batch.unsqueeze(1).expand(N, n_iters, 2)
    board_expanded = full_board.unsqueeze(0).expand(N, n_iters, 5)
    hero_7 = torch.cat([hero_expanded, board_expanded], dim=2)

    # Process in chunks if too large
    total_evals = N * n_iters
    if total_evals > MAX_BATCH_V4:
        chunk_size = max(1, MAX_BATCH_V4 // n_iters)
        hero_powers = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_7 = hero_7[start:end].reshape(-1, 7)
            chunk_power = evaluate_hands(chunk_7).reshape(end - start, n_iters)
            hero_powers.append(chunk_power)
        hero_power = torch.cat(hero_powers, dim=0)  # (N, n_iters)
    else:
        hero_power = evaluate_hands(hero_7.reshape(-1, 7)).reshape(N, n_iters)

    # --- Win/tie computation ---
    best_opp_expanded = best_opp.unsqueeze(0)  # (1, n_iters)
    hero_wins = (hero_power > best_opp_expanded).float()

    hero_ties = (hero_power == best_opp_expanded)
    # Tie share: count how many opponents tied
    n_tied_opps = (opp_power == hero_power.unsqueeze(2)).sum(dim=2)  # wait, shapes
    # opp_power: (n_iters, n_opp), hero_power: (N, n_iters)
    # Need (N, n_iters, n_opp) comparison
    n_tied_opps = (opp_power.unsqueeze(0) == hero_power.unsqueeze(2)).sum(dim=2)  # (N, n_iters)
    tie_share = hero_ties.float() / (n_tied_opps.float() + 1.0)

    results = hero_wins + tie_share  # (N, n_iters)

    # --- Per-hero equity ---
    valid_counts = combined_valid.float().sum(dim=1).clamp(min=1)
    equity = (results * combined_valid.float()).sum(dim=1) / valid_counts  # (N,)

    return equity


# ---------------------------------------------------------------------------
# 2. Batched per-combo equity: (N, M) matrix
# ---------------------------------------------------------------------------

def gpu_equity_per_combo_batched(hero_cards_batch, board_cards, opponent_combos,
                                 n_iters_per_combo=30, device="mps"):
    """Compute equity matrix: each hero hand vs each opponent combo.

    Args:
        hero_cards_batch: (N, 2) int64 tensor
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_combos: (M, 2) int64 tensor (single opponent's range)
        n_iters_per_combo: board samples per combo
        device: torch device

    Returns:
        (N, M) tensor — hero_i equity vs combo_j
    """
    hero_cards_batch = hero_cards_batch.to(device)
    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    opponent_combos = opponent_combos.to(device)
    N = hero_cards_batch.shape[0]
    M = opponent_combos.shape[0]

    if M == 0:
        return torch.full((N, 0), 0.5, dtype=torch.float32, device=device)

    n_board = len(board_cards)
    n_board_needed = 5 - n_board
    n_iters = n_iters_per_combo

    # Process heroes in chunks to control memory
    chunk_heroes = max(1, MAX_BATCH_V4 // (M * n_iters))
    if chunk_heroes >= N:
        return _equity_per_combo_batched_inner(
            hero_cards_batch, board_cards, opponent_combos,
            n_iters, n_board, n_board_needed, device
        )

    results = []
    for start in range(0, N, chunk_heroes):
        end = min(start + chunk_heroes, N)
        chunk = hero_cards_batch[start:end]
        chunk_eq = _equity_per_combo_batched_inner(
            chunk, board_cards, opponent_combos,
            n_iters, n_board, n_board_needed, device
        )
        results.append(chunk_eq)
    return torch.cat(results, dim=0)


def _equity_per_combo_batched_inner(hero_batch, board_cards, opp_combos,
                                    n_iters, n_board, n_board_needed, device):
    """Inner function: compute (N_chunk, M) equity matrix.

    Shares board completions across all (hero, combo) pairs per iteration.
    Each iteration: sample one board, evaluate all N*M pairs.
    """
    N = hero_batch.shape[0]
    M = opp_combos.shape[0]

    # --- Sample board completions: (n_iters, n_board_needed) ---
    # Dead cards for board sampling: only known board cards
    board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    if n_board > 0:
        board_dead[board_cards] = True

    if n_board_needed > 0:
        available_mask = (~board_dead).unsqueeze(0).expand(n_iters, -1)
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_indices = keys.topk(n_board_needed, dim=1)  # (n_iters, n_board_needed)

        if n_board > 0:
            full_board = torch.cat([
                board_cards.unsqueeze(0).expand(n_iters, -1),
                board_indices
            ], dim=1)  # (n_iters, 5)
        else:
            full_board = board_indices
    else:
        full_board = board_cards.unsqueeze(0).expand(n_iters, -1)
        board_indices = None

    # --- Build used-card mask per iteration: board + completed board ---
    used_onehot = board_dead.unsqueeze(0).expand(n_iters, -1).clone()
    if board_indices is not None:
        used_onehot.scatter_(1, board_indices, True)

    # --- Per-pair validity: (N, M, n_iters) ---
    # Hero card conflicts with board/completed-board
    h_c0_conflict = used_onehot[:, hero_batch[:, 0]]  # (n_iters, N)
    h_c1_conflict = used_onehot[:, hero_batch[:, 1]]  # (n_iters, N)
    hero_board_valid = ~(h_c0_conflict | h_c1_conflict)  # (n_iters, N)

    # Opp combo conflicts with board/completed-board
    o_c0_conflict = used_onehot[:, opp_combos[:, 0]]  # (n_iters, M)
    o_c1_conflict = used_onehot[:, opp_combos[:, 1]]  # (n_iters, M)
    opp_board_valid = ~(o_c0_conflict | o_c1_conflict)  # (n_iters, M)

    # Hero-opp card conflicts: hero_i shares a card with combo_j
    # hero_batch: (N, 2), opp_combos: (M, 2)
    # Check all 4 pairs of cards
    h0 = hero_batch[:, 0].unsqueeze(1)  # (N, 1)
    h1 = hero_batch[:, 1].unsqueeze(1)  # (N, 1)
    o0 = opp_combos[:, 0].unsqueeze(0)  # (1, M)
    o1 = opp_combos[:, 1].unsqueeze(0)  # (1, M)
    hero_opp_conflict = (h0 == o0) | (h0 == o1) | (h1 == o0) | (h1 == o1)  # (N, M)

    # Combined validity: (N, M, n_iters)
    # hero_board_valid: (n_iters, N) → (N, n_iters) → (N, 1, n_iters)
    # opp_board_valid: (n_iters, M) → (M, n_iters) → (1, M, n_iters)
    # hero_opp_conflict: (N, M) → (N, M, 1)
    valid = (
        hero_board_valid.T.unsqueeze(1) &  # (N, 1, n_iters)
        opp_board_valid.T.unsqueeze(0) &   # (1, M, n_iters)
        ~hero_opp_conflict.unsqueeze(2)    # (N, M, 1)
    )  # (N, M, n_iters)

    # --- Evaluate hands ---
    # Hero hands: (N, n_iters, 7) — hero cards + full board
    hero_expanded = hero_batch.unsqueeze(1).expand(N, n_iters, 2)
    board_exp_h = full_board.unsqueeze(0).expand(N, n_iters, 5)
    hero_7 = torch.cat([hero_expanded, board_exp_h], dim=2)  # (N, n_iters, 7)
    hero_power = evaluate_hands(hero_7.reshape(-1, 7)).reshape(N, n_iters)

    # Opp hands: (M, n_iters, 7) — opp cards + full board
    opp_expanded = opp_combos.unsqueeze(1).expand(M, n_iters, 2)
    board_exp_o = full_board.unsqueeze(0).expand(M, n_iters, 5)
    opp_7 = torch.cat([opp_expanded, board_exp_o], dim=2)  # (M, n_iters, 7)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7)).reshape(M, n_iters)

    # --- Win/tie: (N, M, n_iters) ---
    # hero_power: (N, n_iters) → (N, 1, n_iters)
    # opp_power: (M, n_iters) → (1, M, n_iters)
    hp = hero_power.unsqueeze(1)
    op = opp_power.unsqueeze(0)
    wins = (hp > op).float()
    ties = (hp == op).float() * 0.5
    results = wins + ties  # (N, M, n_iters)

    # Invalidate conflicts
    results = results * valid.float()

    # Per-pair equity
    valid_counts = valid.float().sum(dim=2).clamp(min=1)  # (N, M)
    equity = results.sum(dim=2) / valid_counts  # (N, M)

    return equity


# ---------------------------------------------------------------------------
# Helper: classify single opponent's response
# ---------------------------------------------------------------------------

def _classify_opponent_response(hero_cards_batch, board_cards, opp_combos_single,
                                fold_threshold, reraise_threshold,
                                combo_response_iters, device):
    """Classify a single opponent's combos into fold/call/reraise per hero hand.

    Returns:
        (p_fold, p_call, p_reraise, call_mask, reraise_mask, mean_opp_eq)
        p_*: (N,) tensors, call/reraise_mask: (N, M) bool, mean_opp_eq: (N,)
    """
    eq_matrix = gpu_equity_per_combo_batched(
        hero_cards_batch, board_cards, opp_combos_single,
        combo_response_iters, device)
    opp_eq = 1.0 - eq_matrix

    fold_mask = opp_eq < fold_threshold
    reraise_mask = opp_eq > reraise_threshold
    call_mask = ~fold_mask & ~reraise_mask

    M = float(opp_combos_single.shape[0])
    p_fold = fold_mask.float().sum(dim=1) / M
    p_reraise = reraise_mask.float().sum(dim=1) / M
    p_call = call_mask.float().sum(dim=1) / M
    mean_opp_eq = opp_eq.mean(dim=1)  # (N,) for blocker adjustment

    return p_fold, p_call, p_reraise, call_mask, reraise_mask, mean_opp_eq


# ---------------------------------------------------------------------------
# 3. Batched EV computation
# ---------------------------------------------------------------------------

def compute_ev_batched(hero_cards_batch, board_cards, opp_range_hand_types,
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
    """Batched EV computation for N hero hands with shared game state.

    Same interface as compute_ev_v3 but hero_cards is a batch.

    Args:
        hero_cards_batch: (N, 2) int64 tensor
        (all other args same as compute_ev_v3)

    Returns:
        (fold_evs, call_evs, raise_evs): each (N,) tensor
    """
    N = hero_cards_batch.shape[0]
    hero_cards_batch = hero_cards_batch.to(device)

    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    # Dead cards for range expansion: board only (hero-specific dead handled via validity)
    dead = set()
    if len(board_cards) > 0:
        dead.update(board_cards.tolist())

    # Expand opponent ranges (move to device for indexing with GPU masks)
    opp_combos = [expand_range(ht_list, dead).to(device) for ht_list in opp_range_hand_types]

    # Combo weights
    cw = None
    if weighted_sampling and action_history and opponent_positions:
        cw = []
        for i, ht_list in enumerate(opp_range_hand_types):
            if i < len(opponent_positions):
                opp_pos = opponent_positions[i]
                opp_actions = [a for p, a in action_history if p == opp_pos]
            else:
                opp_actions = []
            w = compute_combo_weights(ht_list, opp_actions, dead_cards=dead)
            cw.append(w)

    # EQR
    active_positions = list(opponent_positions) + [hero_position] if opponent_positions else None
    eqr = _get_eqr(hero_position, street, n_players, active_positions) if eqr_enabled else 1.0

    # Fix 11: SPR-adjusted EQR — positional advantage vanishes at shallow SPR
    spr = stack / max(pot, 1e-6)
    spr_eqr_factor = min(1.0, spr / 6.0)
    eqr = 1.0 + (eqr - 1.0) * spr_eqr_factor

    # --- Fold EV ---
    fold_evs = torch.full((N,), -hero_invested, dtype=torch.float32, device=device)

    # --- Call EV ---
    raw_equities = gpu_equity_batched(
        hero_cards_batch, board_cards, opp_combos, n_iters, device, cw
    )  # (N,)
    eff_equities = (raw_equities * eqr).clamp(0.0, 1.0)
    total_call_investment = hero_invested + facing_bet
    call_evs = eff_equities * (pot - hero_invested) + (1 - eff_equities) * (-total_call_investment)
    # Fix 13: street discount — earlier streets have more uncertainty ahead
    _street_discount = {0: 0.92, 1: 0.95, 2: 0.98, 3: 1.0}.get(street, 1.0)
    call_evs = call_evs * _street_discount

    # --- Raise EV with batched per-combo response ---
    raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
    total_raise = hero_invested + raise_amount
    new_pot = pot + facing_bet + raise_amount

    call_cost = raise_amount
    pot_after_raise = new_pot
    # Fix 8: nonlinear fold threshold — S-curve closer to real solver outputs
    raw_fold_threshold = call_cost / pot_after_raise if pot_after_raise > 0 else 0.5
    fold_threshold = raw_fold_threshold ** 0.85

    # Fix 14: IP/OOP fold threshold correction
    # OOP opponents fold ~8% more (harder to realize equity), IP fold ~5% less
    if opponent_positions and street > 0:
        # Postflop: highest position acts last (IP)
        max_opp_pos = max(opponent_positions)
        if max_opp_pos < hero_position:
            # All opponents OOP vs hero → they fold more
            fold_threshold *= 1.08
        elif min(opponent_positions) > hero_position:
            # All opponents IP vs hero → they fold less
            fold_threshold *= 0.95

    # Fix 4: dynamic reraise threshold
    if dynamic_reraise:
        actual_reraise_threshold = _compute_reraise_threshold(
            call_cost, pot_after_raise, street, stack, pot)
    else:
        actual_reraise_threshold = reraise_threshold

    has_opponents = any(c.shape[0] > 0 for c in opp_combos) if opp_combos else False

    if has_opponents:
        # Fix 6: multiway — classify each opponent independently
        n_opponents = len(opp_combos)
        use_multiway = n_opponents > 1 and n_opponents <= 3

        if use_multiway:
            per_opp_p_fold = []
            per_opp_p_reraise = []
            per_opp_call_masks = []  # (opp_idx, call_mask (N, M_i))
            per_opp_reraise_masks = []
            per_opp_mean_eq = []

            for opp_idx, opp_c in enumerate(opp_combos):
                if opp_c.shape[0] == 0:
                    per_opp_p_fold.append(torch.ones(N, device=device))
                    per_opp_p_reraise.append(torch.zeros(N, device=device))
                    per_opp_mean_eq.append(torch.full((N,), 0.5, device=device))
                    continue
                p_f, p_c, p_r, c_mask, r_mask, m_eq = _classify_opponent_response(
                    hero_cards_batch, board_cards, opp_c,
                    fold_threshold, actual_reraise_threshold,
                    combo_response_iters, device)
                per_opp_p_fold.append(p_f)
                per_opp_p_reraise.append(p_r)
                per_opp_call_masks.append((opp_idx, c_mask))
                per_opp_reraise_masks.append((opp_idx, r_mask))
                per_opp_mean_eq.append(m_eq)

            # Combined probabilities (independent decisions)
            p_all_fold = torch.ones(N, dtype=torch.float32, device=device)
            for p_f in per_opp_p_fold:
                p_all_fold = p_all_fold * p_f
            p_no_reraise = torch.ones(N, dtype=torch.float32, device=device)
            for p_r in per_opp_p_reraise:
                p_no_reraise = p_no_reraise * (1 - p_r)
            p_any_reraise = 1 - p_no_reraise

            # Fix 9: squeeze correction — when one opponent calls, others fold more often
            # ~15% penalty to fold equity per extra opponent
            squeeze_factor = 0.85 ** (n_opponents - 1)
            p_all_fold = p_all_fold * squeeze_factor

            # Fix 15 (multiway): blocker-adjusted fold equity
            # Average blocker effect across opponents
            avg_mean_opp_eq = torch.stack(per_opp_mean_eq).mean(dim=0)  # (N,)
            blocker_adj = 1.0 + (0.5 - avg_mean_opp_eq) * 0.12
            p_all_fold = (p_all_fold * blocker_adj).clamp(0.0, 1.0)

            p_at_least_one_calls = (1 - p_all_fold - p_any_reraise).clamp(0, 1)

            p_fold = p_all_fold
            p_call = p_at_least_one_calls
            p_reraise = p_any_reraise

            # Fix 3: equity vs callers with card removal — build reduced calling ranges
            all_calling_combos = []
            for opp_idx, opp_c in enumerate(opp_combos):
                matched = False
                for idx, c_mask in per_opp_call_masks:
                    if idx == opp_idx and c_mask.any(dim=0).any():
                        any_caller = c_mask.any(dim=0)
                        all_calling_combos.append(opp_c[any_caller])
                        matched = True
                        break
                if not matched:
                    all_calling_combos.append(opp_c)

            eq_vs_callers = gpu_equity_batched(
                hero_cards_batch, board_cards, all_calling_combos,
                n_iters, device, cw)

            # Build reraising range for Fix 2
            all_reraise_combos = []
            for opp_idx, opp_c in enumerate(opp_combos):
                matched = False
                for idx, r_mask in per_opp_reraise_masks:
                    if idx == opp_idx and r_mask.any(dim=0).any():
                        any_reraiser = r_mask.any(dim=0)
                        all_reraise_combos.append(opp_c[any_reraiser])
                        matched = True
                        break
                if not matched:
                    all_reraise_combos.append(opp_c)

        else:
            # Single opponent (or >3 opponents: primary-only fallback)
            primary_idx = 0
            primary_combos = opp_combos[primary_idx]

            eq_matrix = gpu_equity_per_combo_batched(
                hero_cards_batch, board_cards, primary_combos,
                combo_response_iters, device)
            opp_eq_matrix = 1.0 - eq_matrix

            fold_mask = opp_eq_matrix < fold_threshold
            reraise_mask = opp_eq_matrix > actual_reraise_threshold
            call_mask = ~fold_mask & ~reraise_mask

            M = float(primary_combos.shape[0])
            p_fold = fold_mask.float().sum(dim=1) / M
            p_reraise = reraise_mask.float().sum(dim=1) / M
            p_call = call_mask.float().sum(dim=1) / M

            # Fix 15: blocker-adjusted fold equity
            # If hero blocks strong opponent combos, opp average equity is lower → folds more
            # mean_opp_eq < 0.5 means hero holds blockers; > 0.5 means hero is dominated
            mean_opp_eq = opp_eq_matrix.mean(dim=1)  # (N,)
            # Center at 0.5 (neutral), scale adjustment ±6%
            blocker_adj = 1.0 + (0.5 - mean_opp_eq) * 0.12  # blockers → >1, dominated → <1
            p_fold = (p_fold * blocker_adj).clamp(0.0, 1.0)
            # Redistribute to keep p_fold + p_call + p_reraise == 1
            p_total = p_fold + p_call + p_reraise
            p_fold = p_fold / p_total.clamp(min=1e-6)
            p_call = p_call / p_total.clamp(min=1e-6)
            p_reraise = p_reraise / p_total.clamp(min=1e-6)

            # Fix 3: equity vs callers with card removal via MC
            any_caller = call_mask.any(dim=0)
            if any_caller.any():
                calling_combos = primary_combos[any_caller]
                all_calling_combos = [
                    calling_combos if j == primary_idx else c
                    for j, c in enumerate(opp_combos)]
                eq_vs_callers = gpu_equity_batched(
                    hero_cards_batch, board_cards, all_calling_combos,
                    n_iters, device, cw)
            else:
                eq_vs_callers = raw_equities.clone()

            no_callers = (call_mask.float().sum(dim=1) == 0)
            eq_vs_callers[no_callers] = raw_equities[no_callers]

            # Build reraising range for Fix 2
            any_reraiser = reraise_mask.any(dim=0)
            if any_reraiser.any():
                reraising_combos = primary_combos[any_reraiser]
                all_reraise_combos = [
                    reraising_combos if j == primary_idx else c
                    for j, c in enumerate(opp_combos)]
            else:
                all_reraise_combos = opp_combos

        # Showdown EV vs callers
        eff_eq_callers = (eq_vs_callers * eqr).clamp(0.0, 1.0) if eqr_enabled else eq_vs_callers
        showdown_ev = eff_eq_callers * (new_pot - total_raise) + (1 - eff_eq_callers) * (-total_raise)

        # Fix 2: hero continuing range on reraise
        has_reraisers = any(c.shape[0] > 0 for c in all_reraise_combos)
        if p_reraise.sum() > 0 and has_reraisers:
            eq_vs_reraisers = gpu_equity_batched(
                hero_cards_batch, board_cards, all_reraise_combos,
                n_iters, device, cw)

            # Fix 12: dynamic reraise sizing — SPR/street-aware
            _reraise_spr = stack / max(pot, 1e-6)
            if _reraise_spr < 3.0:
                reraise_size = stack  # shallow → jam
            else:
                _street_mult = {0: 3.0, 1: 2.5, 2: 2.2, 3: 2.0}.get(street, 2.5)
                reraise_size = min(raise_amount * _street_mult, stack)
            total_reraise_cost = hero_invested + reraise_size
            reraise_pot = new_pot + reraise_size
            hero_call_cost = reraise_size - raise_amount
            hero_continue_threshold = hero_call_cost / (reraise_pot + hero_call_cost) \
                if (reraise_pot + hero_call_cost) > 0 else 0.5

            # Fix 7: soft continue — sigmoid instead of binary threshold
            continue_steepness = 15.0
            p_hero_continues = torch.sigmoid(
                continue_steepness * (eq_vs_reraisers - hero_continue_threshold)
            )
            eff_eq_reraise = (eq_vs_reraisers * eqr).clamp(0.0, 1.0) if eqr_enabled else eq_vs_reraisers
            ev_continue = eff_eq_reraise * (reraise_pot - total_reraise_cost) + \
                          (1 - eff_eq_reraise) * (-total_reraise_cost)
            ev_on_reraise_one_level = p_hero_continues * ev_continue + (1 - p_hero_continues) * (-total_raise)
            # Fix 10: geometric approximation for infinite reraise tree
            # Each subsequent reraise level is less likely by discount factor
            reraise_discount = 0.3
            geometric_factor = (1.0 / (1.0 - p_reraise * reraise_discount).clamp(min=0.5)).clamp(max=1.5)
            ev_on_reraise = ev_on_reraise_one_level * geometric_factor
        else:
            ev_on_reraise = torch.full((N,), -total_raise, dtype=torch.float32, device=device)

        raise_evs = (
            p_fold * (pot - hero_invested)
            + p_call * showdown_ev
            + p_reraise * ev_on_reraise
        )
    else:
        raise_evs = torch.full((N,), pot - hero_invested, dtype=torch.float32, device=device)

    return fold_evs, call_evs, raise_evs


# ---------------------------------------------------------------------------
# 4. Marginalized action probabilities
# ---------------------------------------------------------------------------

def compute_marginalized_action_probs(acting_range_combos, acting_weights,
                                      board_cards, opp_range_hand_types,
                                      pot, facing_bet, stack, hero_invested,
                                      n_actions, street_raises, effective_pot,
                                      temperature, big_blind,
                                      n_iters=3000, device="mps",
                                      hero_position=0, street=0, n_players=6,
                                      eqr_enabled=True,
                                      combo_response_iters=30,
                                      reraise_threshold=0.75,
                                      weighted_sampling=True,
                                      action_history=None,
                                      opponent_positions=None,
                                      dynamic_reraise=False):
    """Compute action probabilities marginalized over the acting player's range.

    For each combo in the acting player's range, computes action EVs
    (fold, call, each raise bin, all-in), converts to probabilities via softmax,
    then averages weighted by Bayesian range weights.

    Args:
        acting_range_combos: (N, 2) int64 tensor — combos in acting player's range
        acting_weights: (N,) float tensor — Bayesian weights (sum to 1)
        board_cards: (B,) int64 tensor
        opp_range_hand_types: list of lists of hand type strings (opponents of acting player)
        pot, facing_bet, stack, hero_invested: floats (game state)
        n_actions: int (fold + call + raise_bins + all-in)
        street_raises: list of raise fractions for current street
        effective_pot: float (pot - hero_bets, for raise fraction conversion)
        temperature: float (softmax temperature)
        big_blind: float
        n_iters: MC iterations for equity
        device: torch device
        hero_position, street, n_players: position/street info for EQR
        (remaining args: same as compute_ev_v3)

    Returns:
        marginalized_probs: (n_actions,) tensor — weighted action distribution
        per_combo_probs: (N, n_actions) tensor — per-combo action distributions
    """
    N = acting_range_combos.shape[0]
    n_raise_bins = n_actions - 3  # fold + call + raise_bins + all-in

    if len(board_cards) > 0:
        board_t = board_cards.to(device) if not board_cards.is_cuda else board_cards
    else:
        board_t = torch.tensor([], dtype=torch.long, device=device)

    acting_range_combos = acting_range_combos.to(device)
    acting_weights = acting_weights.to(device)

    # Common EV kwargs
    ev_kwargs = dict(
        n_iters=n_iters, device=device,
        hero_position=hero_position, street=street, n_players=n_players,
        eqr_enabled=eqr_enabled,
        combo_response_iters=combo_response_iters,
        reraise_threshold=reraise_threshold,
        weighted_sampling=weighted_sampling,
        action_history=action_history,
        opponent_positions=opponent_positions,
        dynamic_reraise=dynamic_reraise,
    )

    all_evs = torch.zeros(N, n_actions, dtype=torch.float32, device=device)

    # Fold EV (same for all combos)
    all_evs[:, 0] = -hero_invested

    # Helper to convert raise_pct to solver_frac
    def _raise_to_solver_frac(raise_pct):
        return raise_pct * effective_pot / max(pot, 1e-6)

    # Call EV
    fold_evs, call_evs, _ = compute_ev_batched(
        acting_range_combos, board_t, opp_range_hand_types,
        pot, facing_bet, stack, hero_invested,
        raise_frac=1.0, **ev_kwargs
    )
    all_evs[:, 1] = call_evs

    # All-in EV
    allin_frac = stack / max(pot + facing_bet, 1e-6)
    _, _, allin_evs = compute_ev_batched(
        acting_range_combos, board_t, opp_range_hand_types,
        pot, facing_bet, stack, hero_invested,
        raise_frac=allin_frac, **ev_kwargs
    )
    all_evs[:, n_raise_bins + 2] = allin_evs

    # Raise bins
    for b in range(n_raise_bins):
        raise_pct = street_raises[b]
        actual_bet = facing_bet + raise_pct * effective_pot
        if actual_bet >= stack:
            all_evs[:, b + 2] = allin_evs
        else:
            solver_frac = _raise_to_solver_frac(raise_pct)
            _, _, raise_evs = compute_ev_batched(
                acting_range_combos, board_t, opp_range_hand_types,
                pot, facing_bet, stack, hero_invested,
                raise_frac=solver_frac, **ev_kwargs
            )
            all_evs[:, b + 2] = raise_evs

    # Fix 5: normalize by pot size, not just big blind
    normalizer = max(pot + facing_bet, big_blind) * temperature
    per_combo_probs = F.softmax(all_evs / normalizer, dim=1)  # (N, n_actions)

    # Marginalize: weighted sum
    marginalized_probs = (acting_weights.unsqueeze(1) * per_combo_probs).sum(dim=0)  # (n_actions,)

    return marginalized_probs, per_combo_probs


# ---------------------------------------------------------------------------
# 5. Bayesian range update
# ---------------------------------------------------------------------------

def bayesian_range_update(combo_weights, per_combo_action_probs, observed_action_idx,
                          combos=None, dead_cards=None):
    """Update range weights after an observed action via Bayes' rule.

    P(combo | action) ∝ P(action | combo) × P(combo)

    Args:
        combo_weights: (N,) current weights (sum to 1)
        per_combo_action_probs: (N, n_actions) — P(action | combo) for each combo
        observed_action_idx: int — which action was observed
        combos: optional (N, 2) int64 tensor — card IDs for each combo
        dead_cards: optional set of int card IDs to zero out before update

    Returns:
        (N,) updated weights on CPU (renormalized to sum to 1)
    """
    likelihood = per_combo_action_probs[:, observed_action_idx].cpu()  # (N,)
    combo_weights = combo_weights.cpu()

    # Zero out combos that conflict with dead cards
    if combos is not None and dead_cards:
        dead_t = torch.tensor(sorted(dead_cards), dtype=torch.long)
        c0 = combos[:, 0].cpu()
        c1 = combos[:, 1].cpu()
        c0_dead = (c0.unsqueeze(1) == dead_t.unsqueeze(0)).any(dim=1)
        c1_dead = (c1.unsqueeze(1) == dead_t.unsqueeze(0)).any(dim=1)
        dead_mask = c0_dead | c1_dead
        combo_weights = combo_weights.clone()
        combo_weights[dead_mask] = 0.0

    updated = combo_weights * likelihood
    total = updated.sum()
    if total > 0:
        return updated / total
    return combo_weights  # fallback: no update if degenerate


# ---------------------------------------------------------------------------
# 6. Utility: filter combos by dead cards
# ---------------------------------------------------------------------------

def filter_dead_combos(combos, weights, dead_cards):
    """Filter out combos that conflict with dead cards, renormalize weights.

    Args:
        combos: (N, 2) int64 tensor
        weights: (N,) float tensor
        dead_cards: set of int card IDs

    Returns:
        (valid_combos, valid_weights) — filtered and renormalized
    """
    if not dead_cards:
        return combos, weights

    dead_t = torch.tensor(sorted(dead_cards), dtype=torch.long)
    # Check each combo card against dead cards
    c0 = combos[:, 0]  # (N,)
    c1 = combos[:, 1]  # (N,)

    c0_dead = (c0.unsqueeze(1) == dead_t.unsqueeze(0)).any(dim=1)  # (N,)
    c1_dead = (c1.unsqueeze(1) == dead_t.unsqueeze(0)).any(dim=1)  # (N,)
    valid_mask = ~(c0_dead | c1_dead)

    valid_combos = combos[valid_mask]
    valid_weights = weights[valid_mask]

    total = valid_weights.sum()
    if total > 0:
        valid_weights = valid_weights / total

    return valid_combos, valid_weights
