"""Card encoding, board texture classification, and exact equity computation.

Card encoding convention (matches the project):
  card_id = rank * 4 + suit
  rank: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=T, 9=J, 10=Q, 11=K, 12=A
  suit: 0=d, 1=h, 2=c, 3=s
  52 = no card
"""

import sys
import os
import itertools
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from env.judger import Judger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RANK_CHARS = '23456789TJQKA'
_SUIT_CHARS = 'dhcs'

_RANK_INDEX = {c: i for i, c in enumerate(_RANK_CHARS)}
_SUIT_INDEX = {c: i for i, c in enumerate(_SUIT_CHARS)}

_ALL_CARDS = list(range(52))

# Singleton Judger for re-use across calls (stateless after __init__).
_JUDGER = Judger()


# ---------------------------------------------------------------------------
# Card encoding / decoding
# ---------------------------------------------------------------------------

def card(rank_char: str, suit_char: str) -> int:
    """Convert rank/suit characters to card_id.

    Args:
        rank_char: one of '23456789TJQKA'
        suit_char: one of 'dhcs'

    Returns:
        card_id in [0, 51]

    Examples:
        >>> card('A', 'h')
        49
        >>> card('2', 'd')
        0
    """
    rank = _RANK_INDEX[rank_char]
    suit = _SUIT_INDEX[suit_char]
    return rank * 4 + suit


def card_str(card_id: int) -> str:
    """Convert card_id to human-readable string.

    Args:
        card_id: integer in [0, 51]; 52 is treated as 'XX' (no card)

    Returns:
        Two-character string like 'Ah', '2d', 'Ts', or 'XX' for 52.

    Examples:
        >>> card_str(49)
        'Ah'
        >>> card_str(0)
        '2d'
        >>> card_str(52)
        'XX'
    """
    if card_id == 52:
        return 'XX'
    rank = card_id // 4
    suit = card_id % 4
    return _RANK_CHARS[rank] + _SUIT_CHARS[suit]


# ---------------------------------------------------------------------------
# Board texture classification
# ---------------------------------------------------------------------------

def classify_board_texture(board_cards: list) -> str:
    """Classify board as 'dry', 'draw_heavy', or 'connected'.

    Rules applied in this priority order:
      connected  — board has a pair, OR (flush draw AND straight draw),
                   OR monotone (3+ cards of the same suit)
      dry        — rainbow (all different suits) AND no two cards within
                   2 ranks of each other AND no pair
      draw_heavy — has flush draw (2+ same suit, but NOT 3+ on a 3-card
                   flop) OR has straight draw (2+ cards within 4 ranks of
                   each other), with no pair

    Args:
        board_cards: list of card_ids (3–5 cards, no-card entries ignored)

    Returns:
        'dry', 'draw_heavy', or 'connected'
    """
    cards = [c for c in board_cards if c != 52]
    if not cards:
        return 'dry'

    ranks = [c // 4 for c in cards]
    suits = [c % 4 for c in cards]

    # Pair detection
    has_pair = len(ranks) != len(set(ranks))

    # Suit counts
    suit_counts = [suits.count(s) for s in range(4)]
    max_suit_count = max(suit_counts)

    # Monotone: 3+ cards of same suit
    is_monotone = max_suit_count >= 3

    # Flush draw: 2+ same suit. On a 3-card flop, require exactly 2 (not 3+,
    # which is monotone). On later streets, 2+ same suit still counts.
    n_cards = len(cards)
    if n_cards == 3:
        has_flush_draw = (max_suit_count == 2)
    else:
        has_flush_draw = (max_suit_count >= 2)

    # Straight draw: any two cards within 4 ranks of each other
    sorted_ranks = sorted(set(ranks))
    has_straight_draw = False
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] <= 4:
                has_straight_draw = True
                break
        if has_straight_draw:
            break

    # No two cards within 2 ranks (for dry classification)
    cards_within_2 = False
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] <= 2:
                cards_within_2 = True
                break
        if cards_within_2:
            break

    # Apply priority rules
    if has_pair or (has_flush_draw and has_straight_draw) or is_monotone:
        return 'connected'

    if not cards_within_2 and not has_flush_draw and len(set(suits)) == len(suits):
        return 'dry'

    return 'draw_heavy'


# ---------------------------------------------------------------------------
# Hand strength classification
# ---------------------------------------------------------------------------

def classify_hand_strength(equity: float) -> str:
    """Classify hand strength by equity.

    Args:
        equity: win probability in [0, 1]

    Returns:
        'made'    if equity > 0.65
        'drawing' if equity > 0.35
        'bluff'   otherwise
    """
    if equity > 0.65:
        return 'made'
    if equity > 0.35:
        return 'drawing'
    return 'bluff'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compare_hero_vs_opp(board: list, hero: tuple, opp: tuple) -> float:
    """Return hero's share of the pot: 1.0 win, 0.5 tie, 0.0 loss.

    Args:
        board: list of 5 card_ids (river board)
        hero:  2-tuple of card_ids
        opp:   2-tuple of card_ids
    """
    h_hand = np.array([*board, *hero], dtype=np.int64)
    o_hand = np.array([*board, *opp], dtype=np.int64)
    h_win, o_win = _JUDGER.compare_hands(h_hand, o_hand)
    if h_win == 1 and o_win == 0:
        return 1.0
    if h_win == 0 and o_win == 1:
        return 0.0
    return 0.5  # tie (h_win == 1 and o_win == 1)


# ---------------------------------------------------------------------------
# Equity computation
# ---------------------------------------------------------------------------

def exact_equity_river(hero: tuple, board: list) -> float:
    """Exact equity on river: enumerate all unblocked opponent hands.

    Args:
        hero:  (card_id_1, card_id_2) — hero's hole cards
        board: list of 5 card_ids — river board

    Returns:
        Hero's average win rate in [0, 1] against a uniform opponent range.

    Raises:
        ValueError: if board does not contain exactly 5 valid cards or hero
                    cards overlap with the board.
    """
    dead = set(hero) | set(board)
    live = [c for c in _ALL_CARDS if c not in dead]

    total_weight = 0.0
    hero_share = 0.0

    for opp in itertools.combinations(live, 2):
        share = _compare_hero_vs_opp(board, hero, opp)
        hero_share += share
        total_weight += 1.0

    if total_weight == 0.0:
        return 0.0
    return hero_share / total_weight


def exact_equity_turn(hero: tuple, board: list) -> float:
    """Exact equity on turn: enumerate all river cards and opponent hands.

    Args:
        hero:  (card_id_1, card_id_2) — hero's hole cards
        board: list of 4 card_ids — turn board

    Returns:
        Hero's average equity over all possible river cards and opponent hands.
    """
    dead_base = set(hero) | set(board)
    live_base = [c for c in _ALL_CARDS if c not in dead_base]

    total_weight = 0.0
    hero_share = 0.0

    for river_card in live_base:
        full_board = list(board) + [river_card]
        dead = dead_base | {river_card}
        live = [c for c in live_base if c != river_card]

        for opp in itertools.combinations(live, 2):
            share = _compare_hero_vs_opp(full_board, hero, opp)
            hero_share += share
            total_weight += 1.0

    if total_weight == 0.0:
        return 0.0
    return hero_share / total_weight


def exact_equity_flop(hero: tuple, board: list) -> float:
    """Exact equity on flop: enumerate all turn+river combos and opponent hands.

    Expensive (~1 million evaluations per spot). Progress is printed to stdout.

    Args:
        hero:  (card_id_1, card_id_2) — hero's hole cards
        board: list of 3 card_ids — flop board

    Returns:
        Hero's average equity over all runouts and opponent hands.
    """
    dead_base = set(hero) | set(board)
    live_base = [c for c in _ALL_CARDS if c not in dead_base]

    runouts = list(itertools.combinations(live_base, 2))
    n_runouts = len(runouts)

    total_weight = 0.0
    hero_share = 0.0

    print(f"exact_equity_flop: {n_runouts} runouts to process...")

    for idx, (turn_card, river_card) in enumerate(runouts):
        if idx % 500 == 0:
            print(f"  runout {idx}/{n_runouts} ({100.0 * idx / n_runouts:.1f}%)")

        full_board = list(board) + [turn_card, river_card]
        dead = dead_base | {turn_card, river_card}
        live = [c for c in live_base if c not in (turn_card, river_card)]

        for opp in itertools.combinations(live, 2):
            share = _compare_hero_vs_opp(full_board, hero, opp)
            hero_share += share
            total_weight += 1.0

    print(f"exact_equity_flop: done ({total_weight:.0f} evaluations)")

    if total_weight == 0.0:
        return 0.0
    return hero_share / total_weight


def exact_equity_preflop_mc(hero: tuple, n_iters: int = 50000, seed: int = 42) -> float:
    """Monte Carlo equity for preflop hole cards.

    Deterministic given the same seed. Each iteration deals a random 5-card
    board and a random opponent hand from the remaining deck.

    Args:
        hero:   (card_id_1, card_id_2) — hero's hole cards
        n_iters: number of Monte Carlo iterations (default 50000)
        seed:   RNG seed for determinism (default 42)

    Returns:
        Hero's estimated win rate in [0, 1].
    """
    rng = random.Random(seed)
    dead_base = list(set(hero))
    live_base = [c for c in _ALL_CARDS if c not in dead_base]

    hero_share = 0.0

    for _ in range(n_iters):
        sample = rng.sample(live_base, 7)  # 5 board + 2 opp
        board = sample[:5]
        opp = (sample[5], sample[6])
        share = _compare_hero_vs_opp(board, hero, opp)
        hero_share += share

    return hero_share / n_iters


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== cards.py self-test ===")

    # --- Encoding round-trip ---
    test_cases = [('A', 'h'), ('2', 'd'), ('K', 's'), ('T', 'c'), ('7', 'h')]
    for r, s in test_cases:
        cid = card(r, s)
        back = card_str(cid)
        assert back == r + s, f"Round-trip failed: {r}{s} -> {cid} -> {back}"
    print("Encoding round-trip: OK")

    # card_str(52) == 'XX'
    assert card_str(52) == 'XX'
    print("card_str(52)='XX': OK")

    # --- Board texture ---
    # Dry board: 2d 7h Ks — rainbow, no two within 2 ranks, no pair
    dry_board = [card('2', 'd'), card('7', 'h'), card('K', 's')]
    texture = classify_board_texture(dry_board)
    assert texture == 'dry', f"Expected 'dry', got '{texture}'"
    print(f"Dry board {[card_str(c) for c in dry_board]}: '{texture}' OK")

    # Draw-heavy: flush draw 2d 7d Kh (2 diamonds, no pair)
    flushdraw_board = [card('2', 'd'), card('7', 'd'), card('K', 'h')]
    texture = classify_board_texture(flushdraw_board)
    assert texture == 'draw_heavy', f"Expected 'draw_heavy', got '{texture}'"
    print(f"Flush draw board {[card_str(c) for c in flushdraw_board]}: '{texture}' OK")

    # Connected: paired board 2d 2h Ks
    paired_board = [card('2', 'd'), card('2', 'h'), card('K', 's')]
    texture = classify_board_texture(paired_board)
    assert texture == 'connected', f"Expected 'connected', got '{texture}'"
    print(f"Paired board {[card_str(c) for c in paired_board]}: '{texture}' OK")

    # Monotone: 3 cards same suit
    monotone_board = [card('2', 'd'), card('7', 'd'), card('K', 'd')]
    texture = classify_board_texture(monotone_board)
    assert texture == 'connected', f"Expected 'connected' (monotone), got '{texture}'"
    print(f"Monotone board {[card_str(c) for c in monotone_board]}: '{texture}' OK")

    # --- Hand strength ---
    assert classify_hand_strength(0.80) == 'made'
    assert classify_hand_strength(0.50) == 'drawing'
    assert classify_hand_strength(0.20) == 'bluff'
    assert classify_hand_strength(0.65) == 'drawing'  # boundary: not > 0.65
    assert classify_hand_strength(0.35) == 'bluff'    # boundary: not > 0.35
    print("Hand strength classification: OK")

    # --- River equity ---
    # AA vs random on a low rainbow board: expect ~85%+ equity
    hero_aa = (card('A', 'd'), card('A', 'h'))
    board_river = [
        card('2', 'c'), card('5', 's'), card('7', 'd'),
        card('9', 'h'), card('J', 's'),
    ]
    eq_aa = exact_equity_river(hero_aa, board_river)
    print(f"AA vs random (river, low board): equity = {eq_aa:.4f}")
    assert eq_aa > 0.85, f"AA river equity too low: {eq_aa:.4f}"

    # 72o vs random on a board of all overcards — pure air, expect <20%
    hero_72 = (card('7', 'h'), card('2', 's'))
    board_river_72 = [
        card('8', 'd'), card('T', 'c'), card('Q', 'h'),
        card('K', 's'), card('A', 'd'),
    ]
    eq_72 = exact_equity_river(hero_72, board_river_72)
    print(f"72o vs random (river, all overcards): equity = {eq_72:.4f}")
    assert eq_72 < 0.20, f"72o river equity too high: {eq_72:.4f}"

    # --- Turn equity ---
    hero_kk = (card('K', 'd'), card('K', 'h'))
    board_turn = [
        card('2', 'c'), card('5', 's'), card('7', 'd'), card('9', 'h'),
    ]
    eq_kk = exact_equity_turn(hero_kk, board_turn)
    print(f"KK vs random (turn): equity = {eq_kk:.4f}")
    assert eq_kk > 0.70, f"KK turn equity too low: {eq_kk:.4f}"

    # --- Preflop MC ---
    hero_aa_pf = (card('A', 'c'), card('A', 's'))
    eq_aa_pf = exact_equity_preflop_mc(hero_aa_pf, n_iters=50000, seed=42)
    print(f"AA preflop MC (50k iters, seed=42): equity = {eq_aa_pf:.4f}")
    assert eq_aa_pf > 0.80, f"AA preflop MC equity too low: {eq_aa_pf:.4f}"

    # Determinism check
    eq_aa_pf_2 = exact_equity_preflop_mc(hero_aa_pf, n_iters=50000, seed=42)
    assert eq_aa_pf == eq_aa_pf_2, "MC equity not deterministic with same seed"
    print("MC determinism (same seed): OK")

    # Different seed should give close but possibly different result
    eq_aa_pf_3 = exact_equity_preflop_mc(hero_aa_pf, n_iters=50000, seed=99)
    print(f"AA preflop MC (50k iters, seed=99): equity = {eq_aa_pf_3:.4f}")
    assert abs(eq_aa_pf - eq_aa_pf_3) < 0.01, (
        f"MC equity varies too much across seeds: {eq_aa_pf:.4f} vs {eq_aa_pf_3:.4f}"
    )

    print("\nAll tests passed.")
