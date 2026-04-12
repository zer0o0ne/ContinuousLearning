"""
Range-aware GPU poker solver with opponent response modeling.

Improvements over gpu_solver.py:
1. Opponent hands sampled from position-based ranges (not random)
2. Ranges narrow based on actions (raise/call/3-bet)
3. Raise EV accounts for fold/call/reraise probabilities (MDF-based)

Card encoding: card_id 0-51, rank = card_id // 4 (0=2, 12=A), suit = card_id % 4.
"""

import torch
import torch.nn.functional as F
from gpu_solver import evaluate_hands


# ---------------------------------------------------------------------------
# 169 canonical hand types ordered by preflop strength (strongest first)
# Standard ranking used by most GTO solvers.
# "s" = suited, "o" = offsuit, pairs have no suffix.
# ---------------------------------------------------------------------------

HAND_RANKINGS = [
    # Premium
    "AA", "KK", "QQ", "AKs", "JJ", "AQs", "KQs", "AJs", "KJs", "TT",
    "AKo", "ATs", "QJs", "KTs", "QTs", "JTs", "99", "AQo", "A9s", "KQo",
    # Strong
    "88", "K9s", "T9s", "A8s", "Q9s", "J9s", "AJo", "A5s", "77", "A7s",
    "KJo", "A4s", "A3s", "A6s", "QJo", "66", "K8s", "T8s", "A2s", "98s",
    # Playable
    "J8s", "ATo", "Q8s", "K7s", "KTo", "55", "JTo", "87s", "QTo", "44",
    "33", "22", "K6s", "97s", "K5s", "76s", "T7s", "K4s", "K3s", "K2s",
    "Q7s", "86s", "65s", "J7s", "54s", "Q6s", "75s", "96s", "Q5s", "64s",
    # Marginal
    "Q4s", "Q3s", "T9o", "T6s", "Q2s", "A9o", "53s", "85s", "J6s", "J9o",
    "K9o", "J5s", "Q9o", "43s", "74s", "J4s", "J3s", "95s", "J2s", "63s",
    "A8o", "52s", "T5s", "84s", "T4s", "T3s", "42s", "T2s", "98o", "T8o",
    # Weak
    "A5o", "A7o", "73s", "A4o", "32s", "94s", "93s", "J8o", "A3o", "62s",
    "92s", "K8o", "A6o", "87o", "Q8o", "83s", "A2o", "82s", "97o", "72s",
    "76o", "K7o", "65o", "T7o", "K6o", "86o", "54o", "K5o", "J7o", "75o",
    "Q7o", "K4o", "K3o", "96o", "K2o", "64o", "Q6o", "53o", "85o", "T6o",
    "Q5o", "43o", "Q4o", "Q3o", "74o", "Q2o", "J6o", "63o", "J5o", "95o",
    "52o", "J4o", "J3o", "42o", "J2o", "84o", "T5o", "T4o", "32o", "T3o",
    "73o", "T2o", "62o", "94o", "93o", "92o", "83o", "82o", "72o",
]

# Total: 169 hand types

# ---------------------------------------------------------------------------
# Position-based opening range percentages
# 9-max: 0=SB, 1=BB, 2=UTG, 3=UTG+1, 4=UTG+2, 5=LJ, 6=HJ, 7=CO, 8=BTN
# 6-max: 0=SB, 1=BB, 2=UTG, 3=MP, 4=CO, 5=BTN
# For fewer players, later positions are used first (tighter early seats)
# ---------------------------------------------------------------------------

POSITION_RANGE_PCT = {
    0: 0.35,   # SB
    1: 0.40,   # BB (defending)
    2: 0.12,   # UTG (9-max: very tight)
    3: 0.14,   # UTG+1
    4: 0.16,   # UTG+2 / MP (6-max UTG)
    5: 0.20,   # LJ (Lojack)
    6: 0.24,   # HJ (Hijack)
    7: 0.30,   # CO (Cutoff)
    8: 0.42,   # BTN (Button)
}

# ---------------------------------------------------------------------------
# Action-based range narrowing multipliers
# After an action, keep only a portion of the original range.
# Format: (keep_from_pct, keep_to_pct) — percentile slice of range to keep
# where 0% = strongest hands, 100% = weakest hands
# ---------------------------------------------------------------------------

# "open"    = opened/raised preflop → full opening range
# "call"    = called a raise → remove top 10% (would 3-bet) and bottom 30% (would fold)
# "3bet"    = 3-bet → only top 8%
# "call_postflop" = called a bet postflop → top 70%
# "bet_postflop"  = bet/raised postflop → top 40%

ACTION_NARROWING = {
    "open":           (0.0, 1.0),     # full opening range
    "call":           (0.10, 0.70),   # middle portion
    "3bet":           (0.0, 0.08),    # very top
    "call_postflop":  (0.0, 0.70),    # top 70%
    "bet_postflop":   (0.0, 0.40),    # top 40%
}


# ---------------------------------------------------------------------------
# Hand type → card combos expansion
# ---------------------------------------------------------------------------

# Rank char to rank index (0=2, 1=3, ..., 12=A)
_RANK_CHAR_TO_IDX = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}


def _parse_hand_type(hand_type):
    """Parse canonical hand type string into (rank1, rank2, suited_flag).

    Returns:
        (r1, r2, is_suited) where r1 >= r2, is_suited is bool.
        For pairs, is_suited is None.
    """
    if len(hand_type) == 2:
        # Pair: "AA", "KK", etc.
        r = _RANK_CHAR_TO_IDX[hand_type[0]]
        return r, r, None
    elif len(hand_type) == 3:
        r1 = _RANK_CHAR_TO_IDX[hand_type[0]]
        r2 = _RANK_CHAR_TO_IDX[hand_type[1]]
        is_suited = hand_type[2] == "s"
        return max(r1, r2), min(r1, r2), is_suited
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")


def expand_hand_type(hand_type):
    """Expand canonical hand type to all specific card combos.

    Returns list of (card_id_1, card_id_2) tuples.
    Card ID = rank * 4 + suit (suit 0-3).
    """
    r1, r2, is_suited = _parse_hand_type(hand_type)
    combos = []

    if is_suited is None:
        # Pair: 6 combos (4 choose 2)
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                combos.append((r1 * 4 + s1, r2 * 4 + s2))
    elif is_suited:
        # Suited: 4 combos (same suit)
        for s in range(4):
            combos.append((r1 * 4 + s, r2 * 4 + s))
    else:
        # Offsuit: 12 combos (different suits)
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    combos.append((r1 * 4 + s1, r2 * 4 + s2))

    return combos


# Pre-compute all combos for all 169 hand types
_ALL_COMBOS_CACHE = {}
for _ht in HAND_RANKINGS:
    _ALL_COMBOS_CACHE[_ht] = expand_hand_type(_ht)


# ---------------------------------------------------------------------------
# Range construction and narrowing
# ---------------------------------------------------------------------------

def get_position_range(position, n_players=6):
    """Get canonical hand types for a position's opening range.

    Maps seat index to a range percentage based on table size.
    For 9-max, positions 0-8 map directly to POSITION_RANGE_PCT.
    For 6-max, positions 0-5 are remapped: 0=SB, 1=BB, 2→4(UTG+2), 3→6(HJ), 4→7(CO), 5→8(BTN).
    For fewer players, ranges widen.

    Args:
        position: seat index (0-based)
        n_players: number of players at table (2-9)

    Returns:
        list of hand type strings from HAND_RANKINGS
    """
    if n_players <= 3:
        pct = max(POSITION_RANGE_PCT.get(position, 0.30), 0.35)
    elif n_players <= 6:
        # Remap 6-max seats to 9-max equivalents
        remap_6max = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7, 5: 8}
        mapped = remap_6max.get(position, position)
        pct = POSITION_RANGE_PCT.get(mapped, 0.25)
    else:
        # 7-9 max: direct mapping
        pct = POSITION_RANGE_PCT.get(position, 0.25)

    n_types = max(1, int(len(HAND_RANKINGS) * pct))
    return HAND_RANKINGS[:n_types]


def narrow_range(hand_types, action):
    """Narrow a range based on an observed action.

    Args:
        hand_types: list of canonical hand type strings (ordered by strength)
        action: one of "open", "call", "3bet", "call_postflop", "bet_postflop"

    Returns:
        narrowed list of hand type strings
    """
    if action not in ACTION_NARROWING:
        return hand_types

    from_pct, to_pct = ACTION_NARROWING[action]
    n = len(hand_types)
    start = int(n * from_pct)
    end = int(n * to_pct)
    return hand_types[start:max(start + 1, end)]


def expand_range(hand_types, dead_cards):
    """Expand hand types to concrete card combos, filtering dead cards.

    Args:
        hand_types: list of canonical hand type strings
        dead_cards: set of card IDs that are already dealt

    Returns:
        tensor of shape (n_combos, 2) with card IDs, or empty (0, 2) tensor
    """
    combos = []
    for ht in hand_types:
        for c1, c2 in _ALL_COMBOS_CACHE[ht]:
            if c1 not in dead_cards and c2 not in dead_cards:
                combos.append((c1, c2))

    if not combos:
        return torch.zeros(0, 2, dtype=torch.long)

    return torch.tensor(combos, dtype=torch.long)


# ---------------------------------------------------------------------------
# Range-aware Monte Carlo equity
# ---------------------------------------------------------------------------

def gpu_equity_v2(hero_cards, board_cards, opponent_range_combos, n_iters=10000, device="mps"):
    """Compute hero equity vs opponent ranges via Monte Carlo.

    Args:
        hero_cards: (2,) int64 tensor
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_range_combos: list of (n_combos_i, 2) tensors, one per opponent
        n_iters: MC iterations
        device: torch device

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

    # Check all ranges have combos
    for i, r in enumerate(opp_ranges):
        if r.shape[0] == 0:
            # Empty range — treat as random (fallback)
            all_cards = set(range(52))
            dead = set(hero_cards.tolist())
            if n_board > 0:
                dead.update(board_cards.tolist())
            available = sorted(all_cards - dead)
            # Generate random pairs
            fallback = []
            for j in range(len(available)):
                for k in range(j + 1, len(available)):
                    fallback.append((available[j], available[k]))
            opp_ranges[i] = torch.tensor(fallback[:200], dtype=torch.long, device=device)

    # Dead cards from hero and board
    dead_set = set(hero_cards.tolist())
    if n_board > 0:
        dead_set.update(board_cards.tolist())

    # Available deck for board completion (excludes hero + board only;
    # opponent cards excluded per-iteration)
    all_cards = torch.arange(52, dtype=torch.long, device=device)
    hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    hero_board_dead[hero_cards] = True
    if n_board > 0:
        hero_board_dead[board_cards] = True

    # Sample opponent hands from ranges
    # For each opponent, sample n_iters indices into their range
    opp_hands = []  # list of (n_iters, 2) tensors
    for r in opp_ranges:
        n_combos = r.shape[0]
        indices = torch.randint(0, n_combos, (n_iters,), device=device)
        opp_hands.append(r[indices])  # (n_iters, 2)

    # Build dead-card mask per iteration (hero + board + all opponent cards)
    iter_dead = hero_board_dead.unsqueeze(0).expand(n_iters, -1).clone()  # (n_iters, 52)
    for oh in opp_hands:
        # Mark opponent cards as dead
        iter_dead.scatter_(1, oh, True)

    # For card conflicts between opponents: rejection via validity mask
    # Check that no two opponents share cards
    valid = torch.ones(n_iters, dtype=torch.bool, device=device)
    if n_opponents > 1:
        all_opp_cards = torch.cat(opp_hands, dim=1)  # (n_iters, 2*n_opponents)
        n_opp_cards = all_opp_cards.shape[1]
        # Check uniqueness: one-hot sum should have no entry > 1
        opp_onehot = F.one_hot(all_opp_cards, 52).sum(dim=1)  # (n_iters, 52)
        valid = valid & (opp_onehot.max(dim=1).values <= 1)

    # Also check opponent cards don't overlap with hero/board
    for oh in opp_hands:
        for ci in range(2):
            card = oh[:, ci]
            valid = valid & ~hero_board_dead[card]

    # Available cards per iteration for board completion
    if n_board_needed > 0:
        available_mask = ~iter_dead  # (n_iters, 52)
        # Gumbel-top-k on available cards only
        # Set unavailable cards to -inf so they're never picked
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_indices = keys.topk(n_board_needed, dim=1)  # (n_iters, n_board_needed)
        board_completion = board_indices  # these are card IDs directly (since we index all 52)

        if n_board > 0:
            full_board = torch.cat([
                board_cards.unsqueeze(0).expand(n_iters, -1),
                board_completion
            ], dim=1)
        else:
            full_board = board_completion
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

    # Win/tie/loss (only on valid iterations)
    best_opp = opp_power.max(dim=1).values
    hero_wins = (hero_power > best_opp).float()

    hero_ties = (hero_power == best_opp)
    n_tied_opps = (opp_power == hero_power.unsqueeze(1)).sum(dim=1)
    tie_share = hero_ties.float() / (n_tied_opps.float() + 1.0)

    results = hero_wins + tie_share  # (n_iters,)

    # Only count valid iterations
    if valid.sum() < 10:
        # Not enough valid samples — fallback to all
        return results.mean().item()

    equity = results[valid].mean().item()
    return equity


# ---------------------------------------------------------------------------
# EV calculator with opponent response modeling
# ---------------------------------------------------------------------------

def compute_ev_v2(hero_cards, board_cards, opponent_range_hand_types,
                  pot, facing_bet, stack, hero_invested,
                  raise_frac=1.0, n_iters=3000, device="mps",
                  mdf_max_fold=0.7, reraise_pct=0.15, reraise_cap=0.10):
    """Compute EV for fold/call/raise with range-aware equity and opponent responses.

    Args:
        hero_cards: (2,) int64 tensor
        board_cards: (B,) int64 tensor
        opponent_range_hand_types: list of lists of hand type strings, one per opponent
        pot, facing_bet, stack, hero_invested: floats
        raise_frac: raise size as fraction of (pot + facing_bet)
        n_iters: MC iterations
        device: torch device
        mdf_max_fold: maximum fold probability cap
        reraise_pct: fraction of continuing range that reraises
        reraise_cap: absolute cap on reraise probability

    Returns:
        (fold_ev, call_ev, raise_ev, best_ev) all floats
    """
    # Dead cards
    dead = set(hero_cards.tolist())
    if len(board_cards) > 0:
        dead.update(board_cards.tolist())

    # Expand opponent ranges to card combos
    opp_combos = [expand_range(ht_list, dead) for ht_list in opponent_range_hand_types]

    # --- Fold EV ---
    fold_ev = -hero_invested

    # --- Call EV (equity vs full opponent ranges) ---
    full_equity = gpu_equity_v2(hero_cards, board_cards, opp_combos, n_iters, device)
    total_call_investment = hero_invested + facing_bet
    call_ev = full_equity * (pot - hero_invested) + (1 - full_equity) * (-total_call_investment)

    # --- Raise EV with opponent fold/call/reraise ---
    raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
    total_raise = hero_invested + raise_amount

    # MDF-based fold/call/reraise probabilities
    bet_into_pot = raise_amount
    if (pot + bet_into_pot) > 0:
        mdf = pot / (pot + bet_into_pot)
        p_fold = max(0.0, min(mdf_max_fold, 1.0 - mdf))
    else:
        p_fold = 0.0

    p_reraise = min(reraise_cap, (1.0 - p_fold) * reraise_pct)
    p_call = 1.0 - p_fold - p_reraise

    # Narrow ranges to calling portion (remove bottom folders and top reraisers)
    calling_range_types = []
    for ht_list in opponent_range_hand_types:
        n = len(ht_list)
        # Remove top reraise portion and bottom fold portion
        reraise_end = max(1, int(n * p_reraise / (1.0 - p_fold))) if p_fold < 1.0 else 0
        fold_start = max(reraise_end + 1, int(n * (1.0 - p_fold)))
        calling_types = ht_list[reraise_end:fold_start]
        if not calling_types:
            calling_types = ht_list[:max(1, n // 2)]
        calling_range_types.append(calling_types)

    calling_combos = [expand_range(ht_list, dead) for ht_list in calling_range_types]

    # Equity vs calling range
    call_range_equity = gpu_equity_v2(hero_cards, board_cards, calling_combos, n_iters, device)

    new_pot = pot + facing_bet + raise_amount
    showdown_ev = call_range_equity * (new_pot - total_raise) + (1 - call_range_equity) * (-total_raise)

    raise_ev = (
        p_fold * (pot - hero_invested)  # opponent folds, we win pot
        + p_call * showdown_ev            # opponent calls, showdown
        + p_reraise * (-total_raise)      # opponent reraises, we fold (simplified)
    )

    best_ev = max(fold_ev, call_ev, raise_ev)
    return fold_ev, call_ev, raise_ev, best_ev
