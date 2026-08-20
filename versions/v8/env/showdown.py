"""Hand-strength labels: the showdown reveals, and every seat's own hand
(CONCEPT.md §5.1a, §5.6).

A showdown is the only place where an opponent's line is tied to an actual
holding, so it is the sharpest style signal available. It reaches the embedding
through a **terminal token** appended after every decision of the hand
(`nets/features.py`), whose targets are computed here.

**The same enumeration answers a second question (§5.6).** The percentile of the
*observer's own* hand on the final board is the target of the strength head —
the poker prior the trunk is pretrained on, so that the oracle's noisy EV labels
do not also have to teach hand evaluation from scratch. It is wanted on **every**
hand and for **every dealt seat**, not only the ones that showed, so this module
computes the percentile for all of them in one pass and `showdown_strength` is
that dict restricted to the revealed seats. One enumeration, two consumers; the
showdown labels are bit-identical to what a showdown-only pass produced.

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
    """Fill `record.hand_strength`, `showdown_strength` and `showdown_class`.

    Called once over a corpus, outside any inner loop: the labels depend only on
    the cards, so they never have to be recomputed.

    Every dealt seat gets a `hand_strength` percentile (§5.6) — including seats
    that folded, whose "final board" is the runout the deck had pinned for the
    hand and never turned over. That is a legitimate target and not a leak: it
    is the *label* of a prediction made from the visible board, exactly as a
    value target is, and `nets/features.py` never writes it into a token. It is
    one sample of the runout rather than an expectation over runouts, so the
    head's MSE floor is the conditional variance of the runout and not zero —
    the number to read it against is the marginal variance of the target, never
    zero (`CONCEPT.md` §5.6).

    `showdown_strength` is `hand_strength` restricted to the revealed seats, so
    the §5.1a labels are bit-identical to what a showdown-only pass produced.

    `desc` labels the progress bar (`CLAUDE.md` §5); omitting it runs silently.
    The unit is the hand, not the reveal, so the bar reaches its total even
    though most hands carry no reveal. Returns the number of reveals labelled.
    """
    labelled = 0
    for record in progress(records, desc=desc, unit="hand",
                           disable=desc is None):
        seats = list(range(record.num_players))
        holes = np.array([record.hole_cards(p) for p in seats], dtype=np.int64)
        strengths = strength_percentiles(record.deck[:5], holes, device=device)
        record.hand_strength = {p: float(s) for p, s in zip(seats, strengths)}
        if not record.showdown:
            continue
        record.showdown_strength = {p: record.hand_strength[p]
                                    for p in record.showdown}
        record.showdown_class = {
            p: hand_class_169(*record.hole_cards(p))
            for p in record.showdown}
        labelled += len(record.showdown)
    return labelled
