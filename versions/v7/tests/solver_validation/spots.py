"""
spots.py — Comprehensive poker spot generator for solver validation.

Generates ~9024 unique poker situations covering all combinations of:
  - Street (4): preflop / flop / turn / river
  - Board texture (3, postflop): dry / draw_heavy / connected
  - Opponent profile (2): aggressive / passive
  - Stack depth (4): short / medium / deep / allin
  - Table size (4): 2 / 4 / 6 / 8
  - Active players postflop (3): 2 / 3 / 4  (≤ table_size)
  - Hand strength (3): made / drawing / bluff

Card encoding (matches project convention):
    card_id = rank * 4 + suit
    rank: 0=2, 1=3, ..., 12=A
    suit: 0=d, 1=h, 2=c, 3=s

Run:
    python -m tests.solver_validation.spots
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Card helper
# ---------------------------------------------------------------------------

_RANK_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT_TO_IDX: Dict[str, int] = {"d": 0, "h": 1, "c": 2, "s": 3}


def c(name: str) -> int:
    """Convert a card name like 'Ah', 'Ks', 'Td' to card_id (0-51).

    Encoding: rank * 4 + suit  (rank 0=2..12=A, suit d=0 h=1 c=2 s=3)
    """
    rank_char = name[0].upper()
    suit_char = name[1].lower()
    if rank_char not in _RANK_TO_IDX:
        raise ValueError(f"Unknown rank '{rank_char}' in '{name}'")
    if suit_char not in _SUIT_TO_IDX:
        raise ValueError(f"Unknown suit '{suit_char}' in '{name}'")
    return _RANK_TO_IDX[rank_char] * 4 + _SUIT_TO_IDX[suit_char]


def card_name(card_id: int) -> str:
    """Reverse: card_id → 'Ah' style string (e.g. 49 → 'Ah')."""
    rank_char = "23456789TJQKA"[card_id // 4]
    suit_char = "dhcs"[card_id % 4]
    return rank_char + suit_char


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_BLIND = 10.0

STACK_RANGES: Dict[str, Tuple[float, float]] = {
    "short":  (150.0, 250.0),    # 15-25 bb
    "medium": (400.0, 600.0),    # 40-60 bb
    "deep":   (800.0, 1500.0),   # 80-150 bb
    "allin":  (20.0,  50.0),     # ≤5 bb behind after calling
}

# Realistic pot ranges by street
_POT_RANGES: Dict[int, Tuple[float, float]] = {
    0: (15.0,  30.0),    # preflop
    1: (40.0,  200.0),   # flop
    2: (100.0, 500.0),   # turn
    3: (200.0, 1000.0),  # river
}

STREETS = [0, 1, 2, 3]
TEXTURES = ["dry", "draw_heavy", "connected"]
OPP_PROFILES = ["aggressive", "passive"]
STACK_CATS = ["short", "medium", "deep", "allin"]
TABLE_SIZES = [2, 4, 6, 8]
HAND_CATS = ["made", "drawing", "bluff"]

# Number of variants per combination (4 different boards/hands each)
N_VARIANTS = 4

# ---------------------------------------------------------------------------
# Spot dataclass
# ---------------------------------------------------------------------------

@dataclass
class Spot:
    # Dimension labels
    street: int
    board_texture: str           # 'dry', 'draw_heavy', 'connected', 'none' (preflop)
    opponent_profile: str        # 'aggressive', 'passive'
    stack_category: str          # 'short', 'medium', 'deep', 'allin'
    table_size: int              # 2, 4, 6, 8
    active_players: int          # 2, 3, 4 (postflop); equals table_size for preflop
    hand_category: str           # 'made', 'drawing', 'bluff'

    # Cards
    hero_cards: Tuple[int, int]
    board_cards: List[int]       # length 0 (preflop), 3 (flop), 4 (turn), 5 (river)

    # Game state (in internal units, big_blind=10)
    pot: float
    facing_bet: float
    hero_invested: float
    stack: float                 # hero's remaining chips
    hero_position: int           # seat index
    opponent_positions: List[int]

    # Metadata
    spot_id: int = 0
    combo_id: int = 0            # which dimension combination (unique per combo)
    variant: int = 0             # variant within combination (0..N_VARIANTS-1)

    # Precomputed (filled in later by equity engine)
    exact_equity: float = -1.0


# ---------------------------------------------------------------------------
# Board + hand templates
# ---------------------------------------------------------------------------
# Structure: list of (board_3cards, hand_dict)
#   board_3cards: flop only (3 card_ids)
#   hand_dict: {hand_cat: [hero_hand_0, hero_hand_1, hero_hand_2, hero_hand_3]}
#
# Each hero_hand must be disjoint from its board. All hands per category have
# exactly 4 entries so variant % 4 always selects a distinct hand.
#
# Turn/river cards are appended from separate pools (_*_TURN_CARDS / _*_RIVER_CARDS).

# ---------------------------------------------------------------------------
# DRY boards (rainbow, disconnected, no flush/straight with 2 random holecards)
# ---------------------------------------------------------------------------

_DRY_FLOP_TEMPLATES: List[Tuple[List[int], Dict[str, List[Tuple[int, int]]]]] = [
    # Template 0: K♠7♥2♦ — top card king, two small gaps
    (
        [c("Ks"), c("7h"), c("2d")],
        {
            "made":    [(c("Kh"), c("Qh")),   # top pair top kicker
                        (c("Kd"), c("Jc")),   # top pair
                        (c("7d"), c("7c")),   # middle set
                        (c("Kc"), c("7s")),   # two pair (K+7)
                        ],
            "drawing": [(c("Jd"), c("Th")),   # gutshot + overcards
                        (c("Qc"), c("Js")),   # two overcards
                        (c("9h"), c("8s")),   # gutshot (8-9-T-J → 10 fills)
                        (c("Ac"), c("Jh")),   # two overcards (A+J)
                        ],
            "bluff":   [(c("6s"), c("4h")),   # complete air
                        (c("5c"), c("3s")),   # bottom of range
                        (c("9d"), c("4c")),   # no pair no draw
                        (c("8h"), c("3d")),   # trash
                        ],
        },
    ),
    # Template 1: Q♣8♦3♥
    (
        [c("Qc"), c("8d"), c("3h")],
        {
            "made":    [(c("Qh"), c("Qs")),   # top set
                        (c("Qd"), c("Ac")),   # top pair top kicker
                        (c("8c"), c("8s")),   # middle set
                        (c("Qs"), c("8h")),   # two pair (Q+8) — Qs not Qc
                        ],
            "drawing": [(c("Jh"), c("Tc")),   # oesd (J-T-9-8 or T-9-8-7)
                        (c("Kd"), c("Js")),   # overcards + backdoor
                        (c("Ah"), c("Kc")),   # two overcards
                        (c("7s"), c("6d")),   # gutshot (7-8-9-T → 9 or J fills)
                        ],
            "bluff":   [(c("5h"), c("4s")),   # air low
                        (c("Td"), c("2s")),   # backdoor only
                        (c("9c"), c("6h")),   # nothing
                        (c("4d"), c("2c")),   # trash
                        ],
        },
    ),
    # Template 2: A♠5♥2♣ — ace high dry
    (
        [c("As"), c("5h"), c("2c")],
        {
            "made":    [(c("Ac"), c("Kd")),   # top pair top kicker
                        (c("Ah"), c("Qc")),   # top pair
                        (c("5d"), c("5s")),   # middle set
                        (c("Ad"), c("5c")),   # two pair (A+5)
                        ],
            "drawing": [(c("4h"), c("3d")),   # oesd (A-2-3-4-5 wheel)
                        (c("Kh"), c("Qd")),   # two overcards to 5,2
                        (c("6c"), c("4s")),   # gutshot (3-4-5-6 → 3 or 7 fills)
                        (c("8s"), c("6d")),   # backdoor + two overcards to 5
                        ],
            "bluff":   [(c("Jd"), c("9s")),   # air middle
                        (c("Tc"), c("7h")),   # unconnected
                        (c("8c"), c("3s")),   # nothing
                        (c("9h"), c("6c")),   # air
                        ],
        },
    ),
    # Template 3: T♦6♣3♠
    (
        [c("Td"), c("6c"), c("3s")],
        {
            "made":    [(c("Th"), c("Ac")),   # top pair top kicker
                        (c("Tc"), c("Kd")),   # top pair good kicker
                        (c("6d"), c("6h")),   # middle set
                        (c("Ts"), c("6s")),   # two pair (T+6)
                        ],
            "drawing": [(c("9h"), c("8s")),   # oesd
                        (c("Jd"), c("9c")),   # oesd high end
                        (c("5h"), c("4d")),   # gutshot low (2-3-4-5 → 2 or A fills)
                        (c("Ah"), c("Qc")),   # two overcards
                        ],
            "bluff":   [(c("8c"), c("4h")),   # air
                        (c("Ks"), c("2h")),   # overs disconnected
                        (c("Jh"), c("4s")),   # overcard + nothing
                        (c("7c"), c("2d")),   # trash
                        ],
        },
    ),
    # Template 4: J♥4♦2♠
    (
        [c("Jh"), c("4d"), c("2s")],
        {
            "made":    [(c("Jd"), c("Jc")),   # top set
                        (c("Js"), c("Ac")),   # top pair top kicker
                        (c("4h"), c("4s")),   # middle set
                        (c("Jc"), c("4c")),   # two pair (J+4)
                        ],
            "drawing": [(c("Ah"), c("Kd")),   # two overcards
                        (c("Qh"), c("Ts")),   # gutshot + overcard
                        (c("5h"), c("3c")),   # wheel gutshot (A-2-3-4-5)
                        (c("Tc"), c("9d")),   # oesd (9-T-J-Q → Q or 8 fills)
                        ],
            "bluff":   [(c("9s"), c("7h")),   # air
                        (c("8d"), c("6c")),   # nothing
                        (c("Kh"), c("6s")),   # overcard + nothing
                        (c("Qc"), c("5d")),   # overcard air
                        ],
        },
    ),
    # Template 5: 9♣7♦2♥
    (
        [c("9c"), c("7d"), c("2h")],
        {
            "made":    [(c("9h"), c("9s")),   # top set
                        (c("9d"), c("Ac")),   # top pair top kicker
                        (c("7h"), c("7s")),   # middle set
                        (c("9h"), c("7h")),   # two pair (9+7)
                        ],
            "drawing": [(c("8h"), c("6d")),   # oesd
                        (c("Jd"), c("8s")),   # gutshot + overcard
                        (c("Tc"), c("8d")),   # gutshot (T-9-8-7 → J or 6)
                        (c("As"), c("Kh")),   # two overcards
                        ],
            "bluff":   [(c("Qd"), c("5c")),   # air
                        (c("Kc"), c("4h")),   # overcard trash
                        (c("4d"), c("3s")),   # nothing
                        (c("6c"), c("5s")),   # near-wheel air
                        ],
        },
    ),
]

# Cards to append for turn / river on dry boards (non-connecting, brick)
_DRY_TURN_CARDS  = [c("Kd"), c("As"), c("Qh"), c("Jc"), c("Td"), c("9s")]
_DRY_RIVER_CARDS = [c("Kh"), c("Ac"), c("Qd"), c("Js"), c("Tc"), c("8d")]

# ---------------------------------------------------------------------------
# DRAW-HEAVY boards (flush draws, straight draws, combo draws)
# ---------------------------------------------------------------------------

_DRAW_FLOP_TEMPLATES: List[Tuple[List[int], Dict[str, List[Tuple[int, int]]]]] = [
    # Template 0: T♠9♠5♥ — two-tone with straight draw
    (
        [c("Ts"), c("9s"), c("5h")],
        {
            "made":    [(c("Tc"), c("Td")),   # top set
                        (c("Th"), c("9h")),   # two pair (T+9)
                        (c("9c"), c("9d")),   # middle set
                        (c("5c"), c("5d")),   # bottom set
                        ],
            "drawing": [(c("Jd"), c("8c")),   # oesd
                        (c("Ks"), c("Qs")),   # flush draw overcards
                        (c("8s"), c("7s")),   # flush draw + oesd combo
                        (c("Qh"), c("Jc")),   # gutshot + two overcards
                        ],
            "bluff":   [(c("Ah"), c("2d")),   # ace-high air
                        (c("3c"), c("2h")),   # nothing low
                        (c("Kd"), c("4h")),   # overcard air
                        (c("6c"), c("2c")),   # nothing
                        ],
        },
    ),
    # Template 1: J♥T♥2♦ — two-tone hearts with straight potential
    (
        [c("Jh"), c("Th"), c("2d")],
        {
            "made":    [(c("Jd"), c("Ac")),   # top pair top kicker
                        (c("Tc"), c("Ts")),   # middle set
                        (c("Jc"), c("Js")),   # top set
                        (c("Jd"), c("Tc")),   # two pair (J+T)
                        ],
            "drawing": [(c("Kh"), c("Qh")),   # flush draw + straight draw combo
                        (c("9d"), c("8c")),   # oesd (8-9-T-J)
                        (c("Ah"), c("8h")),   # flush draw
                        (c("Qs"), c("9h")),   # gutshot (9-T-J-Q)
                        ],
            "bluff":   [(c("5s"), c("4d")),   # air low
                        (c("8c"), c("3s")),   # nothing
                        (c("6d"), c("4h")),   # trash
                        (c("7c"), c("3d")),   # nothing
                        ],
        },
    ),
    # Template 2: 8♣7♣3♠ — two-tone clubs with connected low board
    (
        [c("8c"), c("7c"), c("3s")],
        {
            "made":    [(c("8h"), c("8d")),   # top set
                        (c("7h"), c("7d")),   # middle set
                        (c("8s"), c("7s")),   # two pair (8+7)
                        (c("3h"), c("3d")),   # bottom set
                        ],
            "drawing": [(c("9h"), c("6d")),   # oesd (6-7-8-9)
                        (c("Ac"), c("5c")),   # flush draw + wheel draw
                        (c("6c"), c("5c")),   # flush draw + oesd low
                        (c("Jc"), c("Tc")),   # flush draw with overcards
                        ],
            "bluff":   [(c("Kd"), c("4h")),   # air overcard
                        (c("Ah"), c("2d")),   # ace-high nothing
                        (c("Qs"), c("Jh")),   # overcards no draw
                        (c("Td"), c("9s")),   # overcards connected air
                        ],
        },
    ),
    # Template 3: Q♦J♦4♣ — two-tone diamonds high connected
    (
        [c("Qd"), c("Jd"), c("4c")],
        {
            "made":    [(c("Qh"), c("Qs")),   # top set
                        (c("Qs"), c("Jh")),   # two pair (Q+J)
                        (c("Jc"), c("Js")),   # middle set
                        (c("4h"), c("4s")),   # bottom set
                        ],
            "drawing": [(c("Kd"), c("Td")),   # flush draw + straight draw (KQJT → straight)
                        (c("Th"), c("9c")),   # oesd (T-J-Q-K)
                        (c("Ad"), c("9d")),   # flush draw
                        (c("9d"), c("8h")),   # oesd + flush potential
                        ],
            "bluff":   [(c("2s"), c("3h")),   # air low
                        (c("7c"), c("5d")),   # nothing middle
                        (c("8h"), c("6s")),   # air
                        (c("Ah"), c("2c")),   # ace-high nothing
                        ],
        },
    ),
    # Template 4: 9♥8♥K♣ — two-tone hearts with top king
    (
        [c("9h"), c("8h"), c("Kc")],
        {
            "made":    [(c("9d"), c("9c")),   # middle set
                        (c("8d"), c("8s")),   # lower set
                        (c("Ks"), c("Kd")),   # top set
                        (c("Kd"), c("9s")),   # two pair (K+9)
                        ],
            "drawing": [(c("Jh"), c("Th")),   # flush draw + oesd combo (T-J or 7-8-9-T)
                        (c("Qd"), c("Jc")),   # gutshot + overcards
                        (c("7h"), c("6h")),   # flush draw + oesd low
                        (c("Ah"), c("5h")),   # flush draw with ace
                        ],
            "bluff":   [(c("2d"), c("4s")),   # air low
                        (c("5c"), c("3d")),   # nothing
                        (c("Qc"), c("2h")),   # overcard nothing
                        (c("Jd"), c("4c")),   # overcard trash
                        ],
        },
    ),
    # Template 5: 7♠6♠A♥ — two-tone spades with ace overcard
    (
        [c("7s"), c("6s"), c("Ah")],
        {
            "made":    [(c("7h"), c("7d")),   # middle set
                        (c("6h"), c("6c")),   # lower set
                        (c("Ac"), c("Ad")),   # top set
                        (c("Ac"), c("7h")),   # two pair (A+7)
                        ],
            "drawing": [(c("8h"), c("5d")),   # oesd (5-6-7-8)
                        (c("5s"), c("4s")),   # flush draw + gutshot
                        (c("9s"), c("8s")),   # flush draw + oesd
                        (c("Ks"), c("Qs")),   # flush draw overcards
                        ],
            "bluff":   [(c("Jd"), c("3c")),   # air
                        (c("Tc"), c("4h")),   # nothing middle
                        (c("Qd"), c("2c")),   # overcard trash
                        (c("2d"), c("3h")),   # trash low
                        ],
        },
    ),
]

_DRAW_TURN_CARDS  = [c("2c"), c("3d"), c("Kh"), c("As"), c("Jc"), c("4d")]
_DRAW_RIVER_CARDS = [c("2d"), c("Ks"), c("Ah"), c("3c"), c("Qd"), c("5h")]

# ---------------------------------------------------------------------------
# CONNECTED boards (paired, monotone, highly coordinated)
# ---------------------------------------------------------------------------

_CONN_FLOP_TEMPLATES: List[Tuple[List[int], Dict[str, List[Tuple[int, int]]]]] = [
    # Template 0: Q♠Q♥5♦ — paired board
    (
        [c("Qs"), c("Qh"), c("5d")],
        {
            "made":    [(c("Qd"), c("Qc")),   # quads
                        (c("Qc"), c("5h")),   # full house (Q+5)
                        (c("5c"), c("5s")),   # lower full house
                        (c("Kd"), c("Kh")),   # overpair
                        ],
            "drawing": [(c("Jd"), c("Tc")),   # oesd (T-J-Q-K) + overcards to 5
                        (c("Ah"), c("Kc")),   # two overcards to Q
                        (c("9h"), c("8d")),   # gutshot low
                        (c("Kh"), c("Jc")),   # overcards strong
                        ],
            "bluff":   [(c("7s"), c("6h")),   # air
                        (c("3d"), c("2c")),   # trash
                        (c("8c"), c("4s")),   # nothing
                        (c("6d"), c("2h")),   # air low
                        ],
        },
    ),
    # Template 1: T♣9♦8♥ — highly connected three-way
    (
        [c("Tc"), c("9d"), c("8h")],
        {
            "made":    [(c("Jh"), c("7d")),   # flopped straight (7-8-9-T-J)
                        (c("Ts"), c("Th")),   # top set
                        (c("9h"), c("9s")),   # middle set
                        (c("8d"), c("8s")),   # bottom set
                        ],
            "drawing": [(c("Qs"), c("Jh")),   # oesd (J-Q on top) + overcard
                        (c("7h"), c("6s")),   # oesd low (6-7-8-9)
                        (c("Kd"), c("Qc")),   # two overcards
                        (c("Jd"), c("6c")),   # gutshot both ways
                        ],
            "bluff":   [(c("3s"), c("2d")),   # air low
                        (c("5h"), c("4c")),   # air low connected
                        (c("Ac"), c("5d")),   # ace-high nothing
                        (c("Kc"), c("4h")),   # king-high nothing
                        ],
        },
    ),
    # Template 2: A♠K♠Q♠ — monotone broadway
    (
        [c("As"), c("Ks"), c("Qs")],
        {
            "made":    [(c("Jh"), c("Th")),   # flopped broadway straight (off-suit)
                        (c("Ah"), c("Ad")),   # top set (off-suit)
                        (c("Kh"), c("Kd")),   # middle set (off-suit)
                        (c("Qh"), c("Qd")),   # bottom set (off-suit)
                        ],
            "drawing": [(c("Js"), c("Ts")),   # nut flush draw + nut straight draw
                        (c("Ts"), c("9s")),   # flush draw + gutshot
                        (c("Jh"), c("9h")),   # off-suit straight draw
                        (c("8s"), c("7s")),   # flush draw low end
                        ],
            "bluff":   [(c("5h"), c("2d")),   # air low
                        (c("6c"), c("3h")),   # trash
                        (c("7d"), c("4c")),   # nothing
                        (c("9c"), c("2h")),   # nothing
                        ],
        },
    ),
    # Template 3: J♥J♦7♣ — paired board mid-high
    (
        [c("Jh"), c("Jd"), c("7c")],
        {
            "made":    [(c("Jc"), c("Js")),   # quads
                        (c("Jc"), c("7h")),   # full house (J+7)
                        (c("7h"), c("7d")),   # lower full house
                        (c("Kd"), c("Kh")),   # overpair
                        ],
            "drawing": [(c("Qh"), c("Th")),   # gutshot (Q-J fills K or 9)
                        (c("9d"), c("8s")),   # oesd low (7-8-9-T)
                        (c("Ah"), c("Kc")),   # two overcards
                        (c("8h"), c("6d")),   # gutshot low
                        ],
            "bluff":   [(c("5s"), c("4d")),   # air
                        (c("3c"), c("2h")),   # trash
                        (c("6h"), c("2c")),   # nothing low
                        (c("9c"), c("3d")),   # nothing
                        ],
        },
    ),
    # Template 4: 8♠7♠6♠ — monotone low connected
    (
        [c("8s"), c("7s"), c("6s")],
        {
            "made":    [(c("As"), c("Ks")),   # nut flush (no pair needed)
                        (c("9d"), c("5c")),   # flopped straight (5-6-7-8-9)
                        (c("8h"), c("8d")),   # top set
                        (c("7h"), c("7d")),   # middle set
                        ],
            "drawing": [(c("Ts"), c("4s")),   # flush draw + oesd
                        (c("5d"), c("4h")),   # oesd (4-5-6-7 or 5-6-7-8)
                        (c("9h"), c("5c")),   # straight draw (5-6-7-8-9)
                        (c("Ah"), c("Kd")),   # two overcards
                        ],
            "bluff":   [(c("Kd"), c("2c")),   # king-high air
                        (c("Ah"), c("3c")),   # ace-high nothing
                        (c("Jh"), c("2d")),   # nothing
                        (c("4c"), c("3d")),   # low nothing
                        ],
        },
    ),
    # Template 5: K♦Q♦J♦ — monotone high connected
    (
        [c("Kd"), c("Qd"), c("Jd")],
        {
            "made":    [(c("Ad"), c("9d")),   # flush
                        (c("Th"), c("9h")),   # flopped straight off-suit (T-J-Q-K)
                        (c("Kh"), c("Kc")),   # top set
                        (c("Qh"), c("Qs")),   # middle set
                        ],
            "drawing": [(c("Td"), c("8d")),   # flush draw + gutshot
                        (c("Th"), c("9c")),   # oesd off-suit (T-J-Q-K)
                        (c("Ah"), c("Th")),   # oesd + ace overcard
                        (c("9d"), c("8d")),   # flush draw + oesd low
                        ],
            "bluff":   [(c("5h"), c("2c")),   # air low
                        (c("7s"), c("4h")),   # nothing
                        (c("8c"), c("3s")),   # nothing
                        (c("2h"), c("3c")),   # trash
                        ],
        },
    ),
]

_CONN_TURN_CARDS  = [c("2c"), c("3s"), c("4h"), c("5d"), c("Ac"), c("6c")]
_CONN_RIVER_CARDS = [c("2h"), c("3c"), c("4s"), c("Td"), c("5s"), c("9h")]

# ---------------------------------------------------------------------------
# PREFLOP hand templates (board is empty; board_texture = 'none')
# ---------------------------------------------------------------------------

_PREFLOP_HAND_TEMPLATES: Dict[str, List[Tuple[int, int]]] = {
    "made": [
        (c("As"), c("Ah")),   # pocket aces
        (c("Ks"), c("Kh")),   # pocket kings
        (c("Qs"), c("Qh")),   # pocket queens
        (c("As"), c("Ks")),   # AKs
        (c("As"), c("Kd")),   # AKo
        (c("Jh"), c("Jd")),   # pocket jacks
        (c("Ts"), c("Th")),   # pocket tens
        (c("9s"), c("9h")),   # pocket nines
    ],
    "drawing": [
        (c("As"), c("Qh")),   # AQo
        (c("Kh"), c("Qd")),   # KQo
        (c("Jd"), c("Tc")),   # JTo
        (c("8s"), c("7s")),   # 87s
        (c("Ah"), c("Jd")),   # AJo
        (c("Qd"), c("Jh")),   # QJo
        (c("9h"), c("8d")),   # 98o
        (c("Kd"), c("Jc")),   # KJo
    ],
    "bluff": [
        (c("7s"), c("2d")),   # 72o — classic trash hand
        (c("8d"), c("3h")),   # 83o
        (c("9c"), c("4h")),   # 94o
        (c("Jh"), c("3d")),   # J3o
        (c("Qc"), c("2s")),   # Q2o
        (c("5d"), c("2h")),   # 52o
        (c("6s"), c("3d")),   # 63o
        (c("Tc"), c("2h")),   # T2o
    ],
}

# ---------------------------------------------------------------------------
# Board card extension helpers
# ---------------------------------------------------------------------------

_TEXTURE_DATA: Dict[str, Tuple] = {
    "dry":        (_DRY_FLOP_TEMPLATES,  _DRY_TURN_CARDS,  _DRY_RIVER_CARDS),
    "draw_heavy": (_DRAW_FLOP_TEMPLATES, _DRAW_TURN_CARDS, _DRAW_RIVER_CARDS),
    "connected":  (_CONN_FLOP_TEMPLATES, _CONN_TURN_CARDS, _CONN_RIVER_CARDS),
}

_N_TEMPLATES = len(_DRY_FLOP_TEMPLATES)   # 6 — all texture lists have same length


def _extend_board(
    flop: List[int],
    hero_hand: Tuple[int, int],
    turn_pool: List[int],
    river_pool: List[int],
    variant: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (flop_cards, turn_cards, river_cards) with no card collisions.

    Scans turn_pool / river_pool starting from variant-offset position,
    skipping any card already in use, to guarantee a clean board.
    """
    used: set = set(flop) | set(hero_hand)

    # Turn card
    turn_card = turn_pool[variant % len(turn_pool)]
    for i in range(len(turn_pool)):
        candidate = turn_pool[(variant + i) % len(turn_pool)]
        if candidate not in used:
            turn_card = candidate
            break
    used.add(turn_card)

    # River card
    river_start = (variant + 1) % len(river_pool)
    river_card = river_pool[river_start]
    for i in range(len(river_pool)):
        candidate = river_pool[(river_start + i) % len(river_pool)]
        if candidate not in used:
            river_card = candidate
            break

    flop_cards  = list(flop)
    turn_cards  = flop_cards + [turn_card]
    river_cards = turn_cards + [river_card]
    return flop_cards, turn_cards, river_cards


def _get_board_hand(
    texture: str,
    street: int,
    hand_cat: str,
    template_idx: int,
    variant: int,
) -> Tuple[List[int], Tuple[int, int]]:
    """Return (board_cards, hero_hand) for the given parameters.

    template_idx selects which of the 6 flop templates to use.
    variant (0-3) selects which hero hand within that template's hand_cat list.
    A card-collision fallback scans alternative hands if the primary selection
    overlaps with the board.
    """
    flop_templates, turn_pool, river_pool = _TEXTURE_DATA[texture]
    tidx = template_idx % len(flop_templates)
    flop_board, hand_dict = flop_templates[tidx]

    hands_for_cat: List[Tuple[int, int]] = hand_dict.get(hand_cat, [])
    if not hands_for_cat:
        for cat_name in ("made", "drawing", "bluff"):
            if hand_dict.get(cat_name):
                hands_for_cat = hand_dict[cat_name]
                break

    # Primary hand selection: spread variants across all available hands
    n = len(hands_for_cat)
    hero_hand = hands_for_cat[variant % n]

    # Extend board for turn/river
    flop_cards, turn_cards, river_cards = _extend_board(
        flop_board, hero_hand, turn_pool, river_pool, variant
    )
    board = {1: flop_cards, 2: turn_cards, 3: river_cards}[street]

    # Collision check — try alternative hands if hero overlaps board
    board_set = set(board)
    if set(hero_hand) & board_set:
        for alt_idx in range(n):
            alt_hand = hands_for_cat[alt_idx]
            if not (set(alt_hand) & board_set):
                hero_hand = alt_hand
                break

    return board, hero_hand


def _get_preflop_hand(hand_cat: str, variant: int) -> Tuple[int, int]:
    """Return a preflop hero hand for the given category and variant."""
    hands = _PREFLOP_HAND_TEMPLATES[hand_cat]
    return hands[variant % len(hands)]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _hero_and_opponents(
    table_size: int,
    active_players: int,
    rng: random.Random,
) -> Tuple[int, List[int]]:
    """Return (hero_position, opponent_positions) from the active seat pool."""
    all_seats = list(range(table_size))
    chosen = rng.sample(all_seats, active_players)
    hero_pos = chosen[0]
    opp_positions = chosen[1:]
    return hero_pos, opp_positions


# ---------------------------------------------------------------------------
# Game state generator
# ---------------------------------------------------------------------------

def _game_state(
    street: int,
    stack_cat: str,
    variant: int,
    local_rng: random.Random,
) -> Tuple[float, float, float, float]:
    """Return (pot, facing_bet, hero_invested, stack).

    Each variant selects a different point in the realistic range, so 4
    variants produce 4 meaningfully different chip configurations.
    """
    stack_lo, stack_hi = STACK_RANGES[stack_cat]
    # Linear spread over variants
    stack_frac = variant / max(N_VARIANTS - 1, 1)
    stack = stack_lo + (stack_hi - stack_lo) * stack_frac

    pot_lo, pot_hi = _POT_RANGES[street]
    pot_fracs = [0.20, 0.45, 0.65, 0.90]
    pot = pot_lo + (pot_hi - pot_lo) * pot_fracs[variant % 4]

    # Alternate between checked-to (facing_bet=0) and facing a bet
    if variant % 2 == 0:
        facing_bet = 0.0
    else:
        facing_bet = pot * local_rng.uniform(0.30, 0.70)

    # hero_invested grows with street depth
    invest_bb = [1.0, 1.5, 2.5, 4.0][street]
    hero_invested = BIG_BLIND * invest_bb * (1.0 + variant * 0.3)

    if stack_cat == "allin":
        # Ensure hero has ≤5bb remaining after calling
        hero_invested = max(stack + facing_bet - local_rng.uniform(20.0, 50.0), 0.0)

    return pot, facing_bet, hero_invested, stack


# ---------------------------------------------------------------------------
# Valid (table_size, active_players) combinations
# ---------------------------------------------------------------------------

def _valid_table_active_combos() -> List[Tuple[int, int]]:
    """All valid (table_size, active_players) pairs for postflop."""
    combos = []
    for ts in TABLE_SIZES:
        for ap in [2, 3, 4]:
            if ap <= ts:
                combos.append((ts, ap))
    return combos  # 10 combos: (2,2),(4,2),(4,3),(4,4),(6,2),(6,3),(6,4),(8,2),(8,3),(8,4)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_all_spots() -> List[Spot]:
    """Generate 9024 spots covering all dimension combinations.

    Preflop : 4 stacks × 4 tables × 3 strengths × 2 aggression
              = 96 combos × 4 variants = 384

    Postflop: 3 streets × 3 textures × 4 stacks × 2 aggression × 3 strengths
              × 10 (table_size, active_players) pairs
              = 2160 combos × 4 variants = 8640

    Total: 9024
    """
    spots: List[Spot] = []
    spot_id = 0
    combo_id = 0

    # ------------------------------------------------------------------ #
    # PREFLOP                                                              #
    # ------------------------------------------------------------------ #
    preflop_table_active = [(ts, ts) for ts in TABLE_SIZES]

    for stack_cat, opp_profile, hand_cat, (ts, ap) in product(
        STACK_CATS, OPP_PROFILES, HAND_CATS, preflop_table_active
    ):
        for variant in range(N_VARIANTS):
            hero_hand = _get_preflop_hand(hand_cat, variant)
            local_rng = random.Random(12345 + spot_id)

            pot, facing_bet, hero_invested, stack = _game_state(
                street=0, stack_cat=stack_cat, variant=variant, local_rng=local_rng,
            )
            hero_pos, opp_positions = _hero_and_opponents(
                ts, ap, random.Random(12345 + spot_id + 1_000_000)
            )

            spots.append(Spot(
                street=0,
                board_texture="none",
                opponent_profile=opp_profile,
                stack_category=stack_cat,
                table_size=ts,
                active_players=ap,
                hand_category=hand_cat,
                hero_cards=hero_hand,
                board_cards=[],
                pot=pot,
                facing_bet=facing_bet,
                hero_invested=hero_invested,
                stack=stack,
                hero_position=hero_pos,
                opponent_positions=opp_positions,
                spot_id=spot_id,
                combo_id=combo_id,
                variant=variant,
                exact_equity=-1.0,
            ))
            spot_id += 1

        combo_id += 1

    # ------------------------------------------------------------------ #
    # POSTFLOP: flop / turn / river                                        #
    # ------------------------------------------------------------------ #
    valid_ta = _valid_table_active_combos()

    for street, texture, stack_cat, opp_profile, hand_cat, (ts, ap) in product(
        [1, 2, 3], TEXTURES, STACK_CATS, OPP_PROFILES, HAND_CATS, valid_ta
    ):
        for variant in range(N_VARIANTS):
            # Rotate template by (combo_id + variant) so each variant gets a
            # different board (different flop template mod 6)
            template_idx = (combo_id + variant) % _N_TEMPLATES

            board, hero_hand = _get_board_hand(
                texture, street, hand_cat,
                template_idx=template_idx,
                variant=variant,
            )

            local_rng = random.Random(12345 + spot_id)
            pot, facing_bet, hero_invested, stack = _game_state(
                street=street, stack_cat=stack_cat, variant=variant, local_rng=local_rng,
            )
            hero_pos, opp_positions = _hero_and_opponents(
                ts, ap, random.Random(12345 + spot_id + 1_000_000)
            )

            spots.append(Spot(
                street=street,
                board_texture=texture,
                opponent_profile=opp_profile,
                stack_category=stack_cat,
                table_size=ts,
                active_players=ap,
                hand_category=hand_cat,
                hero_cards=hero_hand,
                board_cards=board,
                pot=pot,
                facing_bet=facing_bet,
                hero_invested=hero_invested,
                stack=stack,
                hero_position=hero_pos,
                opponent_positions=opp_positions,
                spot_id=spot_id,
                combo_id=combo_id,
                variant=variant,
                exact_equity=-1.0,
            ))
            spot_id += 1

        combo_id += 1

    return spots


# ---------------------------------------------------------------------------
# Grouping and summary helpers
# ---------------------------------------------------------------------------

def get_spots_by_combo(spots: List[Spot]) -> Dict[tuple, List[Spot]]:
    """Group spots by their dimension combination tuple.

    Key: (street, board_texture, opponent_profile, stack_category,
          table_size, active_players, hand_category)
    """
    result: Dict[tuple, List[Spot]] = {}
    for spot in spots:
        key = (
            spot.street,
            spot.board_texture,
            spot.opponent_profile,
            spot.stack_category,
            spot.table_size,
            spot.active_players,
            spot.hand_category,
        )
        result.setdefault(key, []).append(spot)
    return result


def summarize_coverage(spots: List[Spot]) -> str:
    """Return a formatted multi-line summary of spot counts per dimension value."""
    lines: List[str] = []
    lines.append(f"Total spots: {len(spots)}")
    lines.append("")

    def _count(attr: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in spots:
            v = str(getattr(s, attr))
            counts[v] = counts.get(v, 0) + 1
        return counts

    def _section(title: str, attr: str) -> None:
        lines.append(f"  {title}:")
        for val, cnt in sorted(_count(attr).items(), key=lambda x: x[0]):
            lines.append(f"    {val:20s}: {cnt:5d}")

    _section("Street", "street")
    lines.append("")
    _section("Board texture", "board_texture")
    lines.append("")
    _section("Opponent profile", "opponent_profile")
    lines.append("")
    _section("Stack category", "stack_category")
    lines.append("")
    _section("Table size", "table_size")
    lines.append("")
    _section("Active players", "active_players")
    lines.append("")
    _section("Hand category", "hand_category")
    lines.append("")

    # Combo statistics
    by_combo = get_spots_by_combo(spots)
    lines.append(f"  Distinct combinations: {len(by_combo)}")
    variant_counts = [len(v) for v in by_combo.values()]
    if variant_counts:
        lines.append(
            f"  Variants per combo   : min={min(variant_counts)}, max={max(variant_counts)}"
        )

    # Diversity: combos where all 4 variants have distinct hero hands
    diverse = sum(
        1 for v in by_combo.values() if len(set(s.hero_cards for s in v)) == len(v)
    )
    lines.append(
        f"  Combos with all-distinct hero hands: {diverse}/{len(by_combo)}"
    )

    # Combos where all 4 variants have distinct boards (postflop)
    post_combos = {k: v for k, v in by_combo.items() if k[0] != 0}
    diverse_boards = sum(
        1 for v in post_combos.values()
        if len(set(tuple(s.board_cards) for s in v)) == len(v)
    )
    lines.append(
        f"  Postflop combos with all-distinct boards: "
        f"{diverse_boards}/{len(post_combos)}"
    )

    # Card collision check
    collisions = sum(
        1 for s in spots if set(s.board_cards) & set(s.hero_cards)
    )
    lines.append(f"  Card collisions      : {collisions}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    spots = generate_all_spots()
    print(summarize_coverage(spots))
