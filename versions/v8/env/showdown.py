"""Showdown reveals and their labels (CONCEPT.md §5.1a).

A showdown is the only place where an opponent's line is tied to an actual
holding, so it is the sharpest style signal available. It reaches the embedding
through a **terminal token** appended after every decision of the hand
(`nets/features.py`), whose targets are computed here.

Two targets, deliberately different in kind:

``strength``
    the exact percentile of the revealed hand on the final board — what
    fraction of the combos an unknown opponent could hold it beats, ties at a
    half. Board-relative: it answers "how strong was this player's holding on
    *this* runout", which is what calibrates a range against a line.
``class_169``
    the standard 169-way preflop class of the revealed hand (pair / suited /
    offsuit × ranks). Board-independent: it answers "what does this player
    show up with", which is the part of style that does not wash out with the
    board. It is a lookup, not a computation.

The strength percentile is computed by **exact enumeration**, not Monte Carlo.
v7's showdown anchor (`versions/v7/PLAN_OPPONENT_ADAPTATION` §4) sampled 256
combos and needed a seed to stay reproducible; enumerating all 990 is both
cheaper here and exact, so there is nothing to seed.

**The label is observer-independent.** The combos it is scored against are
everything not on the board and not in the revealed hand itself — other players'
holdings are not removed, even when the observer saw them at the same showdown.
Making the label depend on who was watching would give one hand several
different labels, and the label is a property of the hand.
"""

import numpy as np
import torch

from gto_utils.gpu_solver import evaluate_hands
from utils import progress

N_HAND_CLASSES = 169


def showdown_positions(players_state):
    """Seats whose cards are revealed: the live seats, when there are ≥ 2.

    A hand that ends with one player left is a fold-out and reveals nothing.
    The engine's `Judger` settles by comparing every live seat's hand, so "live
    at the end" is exactly "shown" as far as this codebase is concerned. Real
    poker lets a beaten player muck; treating those as shown is the more
    generous convention and the one the Slumbot protocol also gives us.
    """
    live = [i for i, s in enumerate(players_state) if s >= 0]
    return live if len(live) >= 2 else []


def hand_class_169(card_a, card_b):
    """The 169-way preflop class of two cards, as a flat 13×13 grid index.

    Pairs land on the diagonal, suited hands below it and offsuit hands above,
    so the 169 classes are distinct by construction and no table is needed.
    """
    r_a, s_a = card_a // 4, card_a % 4
    r_b, s_b = card_b // 4, card_b % 4
    hi, lo = max(r_a, r_b), min(r_a, r_b)
    if hi == lo:
        return hi * 13 + lo
    if s_a == s_b:
        return hi * 13 + lo
    return lo * 13 + hi


def _combos_excluding(board):
    """All 2-card combos of the 47 cards not on the board. (1081, 2)."""
    live = np.setdiff1d(np.arange(52), np.asarray(board, dtype=np.int64))
    i, j = np.triu_indices(len(live), k=1)
    return np.stack([live[i], live[j]], axis=1)


def strength_percentiles(board, holes, device="cpu"):
    """Exact percentile in [0, 1] of each revealed hand on `board`.

    Args:
        board: the 5 community cards.
        holes: (N, 2) revealed hole cards.

    Returns:
        (N,) float array. 1.0 = beats every combo an unknown opponent could
        hold, 0.5 = ties everything, 0.0 = loses to everything.
    """
    board = np.asarray(board, dtype=np.int64)
    holes = np.asarray(holes, dtype=np.int64).reshape(-1, 2)
    assert board.shape == (5,), f"a showdown needs a full board, got {board}"

    combos = _combos_excluding(board)                       # (1081, 2)
    board_t = torch.as_tensor(board, device=device)

    opp = torch.cat([board_t.expand(len(combos), 5),
                     torch.as_tensor(combos, device=device)], dim=1)
    hero = torch.cat([board_t.expand(len(holes), 5),
                      torch.as_tensor(holes, device=device)], dim=1)
    opp_scores = evaluate_hands(opp)                        # (1081,)
    hero_scores = evaluate_hands(hero)                      # (N,)

    combos_t = torch.as_tensor(combos, device=device)
    out = np.zeros(len(holes), dtype=np.float64)
    for k in range(len(holes)):
        # Drop the combos that conflict with this hand's own cards.
        conflict = ((combos_t[:, 0] == int(holes[k, 0]))
                    | (combos_t[:, 0] == int(holes[k, 1]))
                    | (combos_t[:, 1] == int(holes[k, 0]))
                    | (combos_t[:, 1] == int(holes[k, 1])))
        scores = opp_scores[~conflict]
        wins = (hero_scores[k] > scores).sum()
        ties = (hero_scores[k] == scores).sum()
        out[k] = float((wins + 0.5 * ties) / len(scores))
    return out


def label_showdowns(records, device="cpu", desc=None):
    """Fill `record.showdown_strength` / `record.showdown_class` in place.

    Called once over a corpus, outside any inner loop: the labels depend only on
    the cards, so they never have to be recomputed.

    `desc` labels the progress bar (`CLAUDE.md` §5); omitting it runs silently.
    The unit is the hand, not the reveal, so the bar reaches its total even
    though most hands carry no label.
    """
    labelled = 0
    for record in progress(records, desc=desc, unit="hand",
                           disable=desc is None):
        if not record.showdown:
            continue
        holes = np.array([record.hole_cards(p) for p in record.showdown],
                         dtype=np.int64)
        strengths = strength_percentiles(record.deck[:5], holes, device=device)
        record.showdown_strength = {
            p: float(s) for p, s in zip(record.showdown, strengths)}
        record.showdown_class = {
            p: hand_class_169(int(h[0]), int(h[1]))
            for p, h in zip(record.showdown, holes)}
        labelled += len(record.showdown)
    return labelled
