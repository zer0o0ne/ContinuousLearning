"""
GPU-accelerated poker Monte Carlo solver for MPS (Apple Silicon).

Replaces gto_helper.py with fully vectorized tensor operations.
All MC simulations run in parallel on GPU.
EV is calculated relative to hand start (fold_ev = -hero_invested).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hand evaluator – vectorised over N hands on GPU
# ---------------------------------------------------------------------------

def _top_k_ranks(rank_counts, k):
    """Return top-k rank indices (descending) from rank histogram.

    Args:
        rank_counts: (N, 13) count of each rank
        k: number of top ranks to return

    Returns:
        (N, k) tensor of rank indices (0-12), highest first
    """
    # Multiply each rank's count indicator by rank value for sorting
    # We want the highest ranks that have count > 0
    present = (rank_counts > 0).long()  # (N, 13)
    # Sort by presence (1 vs 0), breaking ties by rank index (higher = better)
    # Use: present * 14 + rank_index as sort key
    rank_indices = torch.arange(13, device=rank_counts.device).unsqueeze(0)  # (1, 13)
    sort_key = present * 14 + rank_indices  # (N, 13)
    _, top_idx = sort_key.topk(k, dim=1)  # (N, k) - indices of top-k ranks
    return top_idx


def _get_kickers(rank_counts, exclude_mask, k):
    """Get top-k kicker ranks, excluding certain ranks.

    Args:
        rank_counts: (N, 13)
        exclude_mask: (N, 13) bool - True for ranks to exclude
        k: number of kickers

    Returns:
        (N, k) tensor of kicker rank indices, highest first
    """
    available = rank_counts.clone()
    available[exclude_mask] = 0
    present = (available > 0).long()
    rank_indices = torch.arange(13, device=rank_counts.device).unsqueeze(0)
    sort_key = present * 14 + rank_indices
    _, top_idx = sort_key.topk(k, dim=1)
    return top_idx


def _detect_straight(rank_present):
    """Detect straights from rank presence bitmap.

    Args:
        rank_present: (N, 13) bool/int - which ranks are present

    Returns:
        has_straight: (N,) bool
        straight_high: (N,) int - highest rank in best straight (0-12)
    """
    N = rank_present.shape[0]
    device = rank_present.device
    rp = rank_present.long()

    straight_high = torch.full((N,), -1, dtype=torch.long, device=device)

    # Check windows of 5 consecutive ranks (start 0..8 → high 4..12)
    for start in range(9):
        window = rp[:, start:start + 5]
        is_str = (window.sum(dim=1) == 5)
        high = start + 4
        straight_high = torch.where(is_str, torch.tensor(high, device=device), straight_high)

    # Wheel: A(12)-2(0)-3(1)-4(2)-5(3) → high card = 3 (rank of 5)
    wheel = rp[:, 0] & rp[:, 1] & rp[:, 2] & rp[:, 3] & rp[:, 12]
    wheel = wheel.bool()
    # Only use wheel if no better straight found
    straight_high = torch.where(wheel & (straight_high < 3), torch.tensor(3, device=device), straight_high)

    has_straight = straight_high >= 0
    return has_straight, straight_high


def _encode_bord(b0, b1, b2, b3, b4):
    """Encode 5 bord values into single int: b0*13^4 + b1*13^3 + b2*13^2 + b3*13 + b4."""
    return b0 * 28561 + b1 * 2197 + b2 * 169 + b3 * 13 + b4


def evaluate_hands(cards):
    """Evaluate N 7-card poker hands in parallel on GPU.

    Args:
        cards: (N, 7) int64 tensor of card IDs (0-51)

    Returns:
        (N,) int64 tensor of hand scores. Higher = better.
        Score = category * 13^5 + bord_encoding
        Categories: 0=high card, 1=pair, 2=two pair, 3=trips,
                    4=straight, 5=flush, 6=full house, 7=quads, 8=straight flush
    """
    N = cards.shape[0]
    device = cards.device

    ranks = cards // 4   # (N, 7) values 0-12
    suits = cards % 4    # (N, 7) values 0-3

    # Histograms
    rank_counts = F.one_hot(ranks, 13).sum(dim=1)  # (N, 13)
    suit_counts = F.one_hot(suits, 4).sum(dim=1)    # (N, 4)

    rank_present = (rank_counts > 0)  # (N, 13) bool

    BASE = 371293  # 13^5

    # -----------------------------------------------------------------------
    # High card (category 0)
    # -----------------------------------------------------------------------
    top5 = _top_k_ranks(rank_counts, 5)  # (N, 5)
    score = _encode_bord(top5[:, 0], top5[:, 1], top5[:, 2], top5[:, 3], top5[:, 4])

    # -----------------------------------------------------------------------
    # Pair detection (category 1)
    # -----------------------------------------------------------------------
    pair_mask = (rank_counts >= 2)  # (N, 13)
    num_pairs = pair_mask.long().sum(dim=1)  # (N,)
    has_pair = num_pairs >= 1

    # Highest pair rank
    pair_rank_key = pair_mask.long() * (torch.arange(13, device=device).unsqueeze(0) + 1)  # 1-13 if pair, 0 otherwise
    pair_rank_1 = pair_rank_key.argmax(dim=1)  # highest pair rank (0-12)

    # Kickers for single pair (top 3 non-pair ranks)
    p_exclude = F.one_hot(pair_rank_1, 13).bool()  # (N, 13)
    p_kickers = _get_kickers(rank_counts, p_exclude, 3)  # (N, 3)

    pair_score = 1 * BASE + _encode_bord(pair_rank_1, p_kickers[:, 0], p_kickers[:, 1], p_kickers[:, 2],
                                         torch.zeros(N, dtype=torch.long, device=device))
    score = torch.where(has_pair, pair_score, score)

    # -----------------------------------------------------------------------
    # Two pair (category 2)
    # -----------------------------------------------------------------------
    has_two_pair = num_pairs >= 2

    # Second highest pair
    pair_rank_key_2 = pair_rank_key.clone()
    pair_rank_key_2.scatter_(1, pair_rank_1.unsqueeze(1), 0)
    pair_rank_2 = pair_rank_key_2.argmax(dim=1)  # second pair rank

    # Ensure pair_rank_1 > pair_rank_2 (high pair first)
    high_pair = torch.max(pair_rank_1, pair_rank_2)
    low_pair = torch.min(pair_rank_1, pair_rank_2)

    # Kicker for two pair
    tp_exclude = F.one_hot(high_pair, 13).bool() | F.one_hot(low_pair, 13).bool()
    tp_kicker = _get_kickers(rank_counts, tp_exclude, 1)[:, 0]  # (N,)

    two_pair_score = 2 * BASE + _encode_bord(high_pair, low_pair, tp_kicker,
                                             torch.zeros(N, dtype=torch.long, device=device),
                                             torch.zeros(N, dtype=torch.long, device=device))
    score = torch.where(has_two_pair, two_pair_score, score)

    # -----------------------------------------------------------------------
    # Three of a kind (category 3)
    # -----------------------------------------------------------------------
    trip_mask = (rank_counts >= 3)  # (N, 13)
    has_trips = trip_mask.any(dim=1)

    trip_rank_key = trip_mask.long() * (torch.arange(13, device=device).unsqueeze(0) + 1)
    trip_rank = trip_rank_key.argmax(dim=1)  # highest trip rank

    t_exclude = F.one_hot(trip_rank, 13).bool()
    t_kickers = _get_kickers(rank_counts, t_exclude, 2)

    trip_score = 3 * BASE + _encode_bord(trip_rank, t_kickers[:, 0], t_kickers[:, 1],
                                         torch.zeros(N, dtype=torch.long, device=device),
                                         torch.zeros(N, dtype=torch.long, device=device))
    score = torch.where(has_trips & ~has_two_pair, trip_score, score)
    # Note: trips with a pair = full house (handled later), but trips alone overwrites two_pair only if no 2nd pair

    # Actually: if has_trips, it always beats two pair. But we need to check if it's really
    # trips without a second pair (which would be full house). Handle after full house detection.

    # -----------------------------------------------------------------------
    # Straight (category 4)
    # -----------------------------------------------------------------------
    has_straight, straight_high = _detect_straight(rank_present)

    straight_score = 4 * BASE + _encode_bord(straight_high,
                                             torch.zeros(N, dtype=torch.long, device=device),
                                             torch.zeros(N, dtype=torch.long, device=device),
                                             torch.zeros(N, dtype=torch.long, device=device),
                                             torch.zeros(N, dtype=torch.long, device=device))
    # Will be overwritten by flush/full house/etc. if applicable

    # -----------------------------------------------------------------------
    # Flush (category 5)
    # -----------------------------------------------------------------------
    flush_suit_mask = (suit_counts >= 5)  # (N, 4)
    has_flush = flush_suit_mask.any(dim=1)

    # Which suit is the flush suit (take highest count, or first if tie)
    flush_suit_id = flush_suit_mask.long().argmax(dim=1)  # (N,)

    # Extract ranks of flush-suit cards
    # card_is_flush: (N, 7) bool
    card_is_flush = (suits == flush_suit_id.unsqueeze(1))
    flush_ranks = torch.where(card_is_flush, ranks, torch.tensor(-1, device=device))
    flush_ranks_sorted, _ = flush_ranks.sort(dim=1, descending=True)
    flush_top5 = flush_ranks_sorted[:, :5]  # (N, 5) top 5 flush ranks

    flush_score = 5 * BASE + _encode_bord(flush_top5[:, 0], flush_top5[:, 1],
                                          flush_top5[:, 2], flush_top5[:, 3], flush_top5[:, 4])

    # -----------------------------------------------------------------------
    # Full house (category 6)
    # -----------------------------------------------------------------------
    # Has trips AND at least one other pair (could be from remaining ranks or another trip)
    fh_remaining = rank_counts.clone()
    # Zero out the trip rank
    fh_remaining.scatter_(1, trip_rank.unsqueeze(1), 0)
    fh_has_pair = (fh_remaining >= 2).any(dim=1)
    has_full = has_trips & fh_has_pair

    # Best pair rank for full house (from remaining)
    fh_pair_key = (fh_remaining >= 2).long() * (torch.arange(13, device=device).unsqueeze(0) + 1)
    fh_pair_rank = fh_pair_key.argmax(dim=1)

    full_score = 6 * BASE + _encode_bord(trip_rank, fh_pair_rank,
                                         torch.zeros(N, dtype=torch.long, device=device),
                                         torch.zeros(N, dtype=torch.long, device=device),
                                         torch.zeros(N, dtype=torch.long, device=device))

    # -----------------------------------------------------------------------
    # Four of a kind (category 7)
    # -----------------------------------------------------------------------
    quad_mask = (rank_counts >= 4)
    has_quads = quad_mask.any(dim=1)
    quad_rank_key = quad_mask.long() * (torch.arange(13, device=device).unsqueeze(0) + 1)
    quad_rank = quad_rank_key.argmax(dim=1)

    q_exclude = F.one_hot(quad_rank, 13).bool()
    q_kicker = _get_kickers(rank_counts, q_exclude, 1)[:, 0]

    quad_score = 7 * BASE + _encode_bord(quad_rank, q_kicker,
                                         torch.zeros(N, dtype=torch.long, device=device),
                                         torch.zeros(N, dtype=torch.long, device=device),
                                         torch.zeros(N, dtype=torch.long, device=device))

    # -----------------------------------------------------------------------
    # Straight flush (category 8)
    # -----------------------------------------------------------------------
    # Build rank presence for flush suit only using one-hot (no nested loops)
    flush_suit_ranks = torch.where(card_is_flush, ranks, torch.tensor(13, device=device))  # 13 = sentinel
    flush_rank_onehot = F.one_hot(flush_suit_ranks, 14)[:, :, :13]  # (N, 7, 13), drop sentinel col
    flush_rank_counts = flush_rank_onehot.sum(dim=1)  # (N, 13)

    flush_rank_present = (flush_rank_counts > 0)
    has_sf, sf_high = _detect_straight(flush_rank_present)
    has_sf = has_sf & has_flush  # must also have flush

    sf_score = 8 * BASE + _encode_bord(sf_high,
                                       torch.zeros(N, dtype=torch.long, device=device),
                                       torch.zeros(N, dtype=torch.long, device=device),
                                       torch.zeros(N, dtype=torch.long, device=device),
                                       torch.zeros(N, dtype=torch.long, device=device))

    # -----------------------------------------------------------------------
    # Assemble final scores (apply from weakest to strongest)
    # -----------------------------------------------------------------------
    # Already have high card and pair/two-pair/trips assigned above.
    # Now layer on straight, flush, full house, quads, SF — each overwrites if present.

    # Straight (beats trips/two pair/pair/high card)
    score = torch.where(has_straight, straight_score, score)

    # Flush (beats straight)
    score = torch.where(has_flush, flush_score, score)

    # Full house (beats flush) — also overwrites trips that were set earlier
    score = torch.where(has_full, full_score, score)

    # Four of a kind (beats full house)
    score = torch.where(has_quads, quad_score, score)

    # Straight flush (beats everything)
    score = torch.where(has_sf, sf_score, score)

    # Fix: trips that also have a pair should be full house (already handled above).
    # But pure trips (no extra pair) should overwrite straight only if no straight.
    # Re-apply trips for cases where has_trips but NOT has_full and NOT has_straight and NOT has_flush:
    pure_trips = has_trips & ~has_full & ~has_straight & ~has_flush & ~has_quads
    score = torch.where(pure_trips, trip_score, score)

    return score


# ---------------------------------------------------------------------------
# Monte Carlo equity calculator
# ---------------------------------------------------------------------------

def gpu_equity(hero_cards, board_cards, n_opponents, n_iters=10000, device="mps"):
    """Compute hero equity via Monte Carlo simulation on GPU.

    Args:
        hero_cards: (2,) int64 tensor of hero card IDs (0-51)
        board_cards: (B,) int64 tensor of board card IDs, B in {0,3,4,5}
        n_opponents: int, 1-5
        n_iters: number of MC iterations (all run in parallel)
        device: torch device string

    Returns:
        equity: float (0-1), hero's win probability
    """
    hero_cards = hero_cards.to(device)
    if len(board_cards) > 0:
        board_cards = board_cards.to(device)
    else:
        board_cards = torch.tensor([], dtype=torch.long, device=device)

    n_board = len(board_cards)
    n_board_needed = 5 - n_board
    n_opp_cards = 2 * n_opponents
    n_needed = n_board_needed + n_opp_cards

    # Available deck (remove dead cards)
    all_cards = torch.arange(52, dtype=torch.long, device=device)
    dead = torch.zeros(52, dtype=torch.bool, device=device)
    dead[hero_cards] = True
    if n_board > 0:
        dead[board_cards] = True
    available = all_cards[~dead]  # (D,) where D = 52 - len(dead)

    D = available.shape[0]

    # Gumbel-top-k sampling: random keys → topk gives sampling without replacement
    keys = torch.rand(n_iters, D, device=device)
    _, indices = keys.topk(n_needed, dim=1)  # (n_iters, n_needed)
    dealt = available[indices]  # (n_iters, n_needed)

    # Split dealt cards
    board_completion = dealt[:, :n_board_needed]  # (n_iters, n_board_needed)

    if n_board > 0:
        full_board = torch.cat([board_cards.unsqueeze(0).expand(n_iters, -1),
                                board_completion], dim=1)  # (n_iters, 5)
    else:
        full_board = board_completion  # (n_iters, 5)

    opp_section = dealt[:, n_board_needed:]  # (n_iters, n_opp_cards)
    opp_cards = opp_section.reshape(n_iters, n_opponents, 2)  # (n_iters, n_opp, 2)

    # Hero 7-card hands
    hero_7 = torch.cat([hero_cards.unsqueeze(0).expand(n_iters, -1),
                        full_board], dim=1)  # (n_iters, 7)

    # Opponent 7-card hands
    opp_7 = torch.cat([opp_cards,
                       full_board.unsqueeze(1).expand(-1, n_opponents, -1)], dim=2)  # (n_iters, n_opp, 7)

    # Evaluate hands
    hero_power = evaluate_hands(hero_7)  # (n_iters,)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7))  # (n_iters * n_opp,)
    opp_power = opp_power.reshape(n_iters, n_opponents)  # (n_iters, n_opp)

    # Win/tie/loss
    best_opp = opp_power.max(dim=1).values  # (n_iters,)

    hero_wins = (hero_power > best_opp).float()

    # Ties: hero ties with best opponent → split equally
    hero_ties = (hero_power == best_opp)
    # Count how many opponents also have the same score as hero
    n_tied_opps = (opp_power == hero_power.unsqueeze(1)).sum(dim=1)  # (n_iters,)
    tie_share = hero_ties.float() / (n_tied_opps.float() + 1.0)  # +1 for hero

    equity = (hero_wins + tie_share).mean().item()
    return equity


# ---------------------------------------------------------------------------
# EV calculator — relative to hand start
# ---------------------------------------------------------------------------

def compute_ev(equity, pot, facing_bet, stack, hero_invested, raise_frac=1.0):
    """Compute EV for fold/call/raise relative to hand start.

    All EVs represent net profit/loss from the beginning of the hand.
    fold_ev = -hero_invested (lose everything already put in).

    Args:
        equity: float (0-1)
        pot: float, current total pot
        facing_bet: float, bet hero must call (0 if check)
        stack: float, hero's remaining stack
        hero_invested: float, total hero has contributed to pot so far
        raise_frac: float, raise size as fraction of (pot + facing_bet)

    Returns:
        (fold_ev, call_ev, raise_ev, best_ev) all floats
    """
    # Fold: lose everything invested
    fold_ev = -hero_invested

    # Call: total investment = hero_invested + facing_bet
    # If win: gain entire pot (pot + facing_bet) minus our total investment
    # If lose: lose total investment
    total_call = hero_invested + facing_bet
    # Net if win = (pot + facing_bet) - total_call = pot - hero_invested
    # Net if lose = -total_call
    call_ev = equity * (pot - hero_invested) + (1 - equity) * (-total_call)
    # Simplifies to: equity * (pot + facing_bet) - total_call

    # Raise
    raise_amount = min(facing_bet + raise_frac * (pot + facing_bet), stack)
    total_raise = hero_invested + raise_amount
    # Assume one opponent calls the raise (simplified model)
    # New pot = pot + facing_bet + raise_amount (hero adds) + (raise_amount - facing_bet) (opponent calls)
    # More precisely: pot already has opponent's bet, hero adds raise_amount,
    # opponent needs to call the raise difference.
    # Simplified: assume opponent calls → new_pot = pot + raise_amount + (raise_amount - facing_bet)
    # But this is complex. Use simpler model: pot + facing_bet + raise_amount
    new_pot = pot + facing_bet + raise_amount
    raise_ev = equity * (new_pot - total_raise) + (1 - equity) * (-total_raise)
    # Simplifies to: equity * new_pot - total_raise

    best_ev = max(fold_ev, call_ev, raise_ev)
    return fold_ev, call_ev, raise_ev, best_ev
