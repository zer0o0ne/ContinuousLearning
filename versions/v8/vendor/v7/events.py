"""The v7 event format (CONCEPT.md §4.3, §16 OI-6).

A v7 checkpoint cannot be fed anything but v7 events, so the format is vendored
alongside the model. This is `_rebuild_events` / `_get_table_display_from_turn`
from `versions/v7/agent/train_scenarios/generation/generate.py`, cross-checked
against the copy that came into v8 with
`evaluation/slumbot_eval.py::_build_events` — the two agree field for field.

Two conventions matter and are load-bearing:

* **Board masking.** An event shows the board as of *its own* street, never the
  final board. Stamping the river into a preflop event leaks future cards.
* **`acting_pos` is the next-player convention** (audit B.2): in a post-action
  snapshot `acting_pos` is whoever acts *after* the action, and `action` is the
  action that was just taken. In a pre-decision snapshot `acting_pos` is the
  player on turn and `action` is all-zeros. The caller (`env/driver.py`) records
  snapshots in that shape; this module only reads them.

Events are built from **one player's perspective**: `hand` is that player's own
hole cards and everything else is public. There is no path here through which a
member could see another player's cards.
"""

import numpy as np


def table_display_from_turn(deck, turn):
    """5-element board as visible on `turn`; unseen cards are -1."""
    if turn == 0:
        return [-1] * 5
    if turn == 1:
        return list(deck[:3]) + [-1, -1]
    if turn == 2:
        return list(deck[:4]) + [-1]
    return list(deck[:5])


def build_v7_events(snapshots, deck, hero_pos, num_players, big_blind,
                    small_blind, n_actions, up_to):
    """Rebuild the v7 event sequence for `hero_pos` over `snapshots[0..up_to]`.

    Args:
        snapshots: list of dicts with keys ``pot``, ``bets``, ``credits``,
            ``turn``, ``active_pos``, ``action`` (one-hot list or None).
        deck: the hand's 52-card permutation (board = deck[:5], hole cards of
            seat p = deck[5 + 2p : 7 + 2p]).
        hero_pos: seat whose perspective the events are built from.
        up_to: inclusive index of the last snapshot to include.

    Returns:
        list of v7 event dicts.
    """
    hole = list(np.asarray(deck)[5 + 2 * hero_pos: 7 + 2 * hero_pos])
    hand = [int(c) for c in hole]
    events = []
    for snap in snapshots[:up_to + 1]:
        action = snap["action"]
        if action is None:
            action = [0.0] * n_actions
        else:
            action = [float(a) for a in action]
        events.append({
            "hand": hand,
            "num_players": int(num_players),
            "hero_pos": int(hero_pos),
            "acting_pos": int(snap["active_pos"]),
            "big_blind": float(big_blind),
            "small_blind": float(small_blind),
            "stack": float(snap["credits"][hero_pos]),
            "stacks": [float(c) for c in snap["credits"]],
            "table": table_display_from_turn(deck, snap["turn"]),
            "pot": float(snap["pot"]),
            "bets": np.asarray(snap["bets"], dtype=np.float32).copy(),
            "action": action,
        })
    return events
