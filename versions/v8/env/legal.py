"""The one legality rule (CONCEPT.md §6.2: "one implementation, not two").

Every place in v8 that needs to know which actions are playable — the driver
that samples pool-member actions, the tokeniser that masks the action-prediction
logits, and later the oracle and the agent's target construction — calls this.
A second copy is how the played distribution and the trained distribution drift
apart without anyone noticing.

The rule is v7's, ported from `versions/v7/agent/mcts/game_state.py`
::`get_legal_actions` onto `env.table.Table` (v8 has no `GameState`). It is
"playable", not merely rule-legal: strictly-dominated branches are dropped
because they add no distinct choice and would each carry probability mass.

Action layout: ``[fold, call, raise_0 … raise_{bins-1}, all-in]``.
"""

import numpy as np


def legal_actions(table):
    """Playable action indices for `table.active_player`.

    Filters, all inherited from v7:

    * **fold** only when facing a bet — checking strictly dominates folding for
      free;
    * **raise bins** that would not exceed the call amount (``step()`` collapses
      them into a call), that already commit the whole stack (duplicate of the
      explicit all-in), or whose increment is below the NLHE min-raise;
    * **every raise** when all other live players are already all-in (C.5) —
      the chips could only come back uncalled;
    * **every raise** for a player who had already matched the last full raise
      level when a short all-in pushed the high bet past it (C.7.5 — not
      reopened).
    """
    pos = table.active_player
    call_amount = table.high_bet - table.bets[pos]
    credits_pos = table.credits[pos]
    effective_pot = table.pot - table.bets[pos]
    facing_bet = call_amount > 0

    others_live = [i for i in range(table.num_players)
                   if i != pos and table.players_state[i] >= 0]
    all_others_allin = bool(others_live) and all(
        table.players_state[i] == 2 for i in others_live)

    actions = []
    if facing_bet:
        actions.append(0)
    actions.append(1)

    short_allin_restricted = (
        table._last_full_raise_level > 0
        and table.bets[pos] >= table._last_full_raise_level
        and table.high_bet > table._last_full_raise_level
    )
    can_raise = (credits_pos > call_amount
                 and not all_others_allin
                 and not short_allin_restricted)
    if can_raise:
        for i in range(table.n_raise_bins):
            raise_pct = table.raise_sizes[table.turn][i]
            bet = call_amount + raise_pct * effective_pot
            if bet <= call_amount:
                continue
            if bet >= credits_pos:
                continue
            if raise_pct * effective_pot < table.last_raise_size:
                continue
            actions.append(i + 2)
        actions.append(table.n_raise_bins + 2)
    elif (credits_pos > 0 and credits_pos > call_amount
          and not all_others_allin
          and not short_allin_restricted):
        actions.append(table.n_raise_bins + 2)

    return actions


def legal_action_mask(table, n_actions):
    """Boolean numpy mask of length `n_actions`, True where playable."""
    mask = np.zeros(n_actions, dtype=bool)
    mask[legal_actions(table)] = True
    return mask
