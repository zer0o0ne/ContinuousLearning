"""
Lightweight game state tracker for MCTS.

Mirrors env/table.py step() + next_turn() betting logic without
cards, deck, or judger. Tracks only what MCTS needs: pot, stacks,
bets, player states, active player, street, and terminal conditions.
"""


class GameState:
    __slots__ = [
        "num_players", "hero_pos", "active_player", "players_state",
        "credits", "bets", "pot", "high_bet", "turn",
        "raise_sizes", "n_raise_bins", "is_terminal", "several_all_in",
        "last_raise_size", "big_blind", "_last_full_raise_level",
    ]

    def __init__(self, num_players, hero_pos, active_player, players_state,
                 credits, bets, pot, high_bet, turn, raise_sizes,
                 n_raise_bins, is_terminal=False, several_all_in=False,
                 last_raise_size=None, big_blind=None,
                 last_full_raise_level=None):
        self.num_players = num_players
        self.hero_pos = hero_pos
        self.active_player = active_player
        self.players_state = players_state
        self.credits = credits
        self.bets = bets
        self.pot = pot
        self.high_bet = high_bet
        self.turn = turn
        self.raise_sizes = raise_sizes  # shared, read-only
        self.n_raise_bins = n_raise_bins
        self.is_terminal = is_terminal
        self.several_all_in = several_all_in
        # C.7.5: NLHE min-raise tracking
        self.big_blind = float(big_blind) if big_blind is not None else (float(high_bet) if high_bet > 0 else 10.0)
        if last_raise_size is not None:
            self.last_raise_size = float(last_raise_size)
        else:
            self.last_raise_size = self.big_blind
        # C.7.5: bet level at which the last FULL raise occurred. Players
        # whose bets >= this level can only call/fold (not re-raise) if a
        # subsequent short all-in pushed high_bet above this level.
        if last_full_raise_level is not None:
            self._last_full_raise_level = float(last_full_raise_level)
        else:
            self._last_full_raise_level = float(high_bet) if high_bet > 0 else 0.0

    @classmethod
    def from_table(cls, table, hero_pos):
        """Snapshot a Table instance into a GameState."""
        return cls(
            num_players=table.num_players,
            hero_pos=hero_pos,
            active_player=table.active_player,
            players_state=list(table.players_state),
            credits=list(table.credits),
            bets=list(table.bets),
            pot=float(table.pot),
            high_bet=float(table.high_bet),
            turn=int(table.turn),
            raise_sizes=table.raise_sizes,
            n_raise_bins=table.n_raise_bins,
            is_terminal=False,
            several_all_in=bool(table.several_all_in),
            last_raise_size=float(getattr(table, 'last_raise_size', table.big_blind)),
            big_blind=float(table.big_blind),
            last_full_raise_level=float(getattr(table, '_last_full_raise_level', table.big_blind)),
        )

    def clone(self):
        return GameState(
            num_players=self.num_players,
            hero_pos=self.hero_pos,
            active_player=self.active_player,
            players_state=list(self.players_state),
            credits=list(self.credits),
            bets=list(self.bets),
            pot=self.pot,
            high_bet=self.high_bet,
            turn=self.turn,
            raise_sizes=self.raise_sizes,  # shared reference
            n_raise_bins=self.n_raise_bins,
            is_terminal=self.is_terminal,
            several_all_in=self.several_all_in,
            last_raise_size=self.last_raise_size,
            big_blind=self.big_blind,
            last_full_raise_level=self._last_full_raise_level,
        )

    def is_hero_turn(self):
        return self.active_player == self.hero_pos

    def step(self, action_idx):
        """Apply action and advance game state. Mirrors Table.step() + next_turn()."""
        if not self.several_all_in:
            pos = self.active_player

            if action_idx == 0:
                # Fold
                self.players_state[pos] = -1

            elif action_idx == 1:
                # Call / check
                bet = min(self.high_bet - self.bets[pos], self.credits[pos])
                self.pot += bet
                self.credits[pos] -= bet
                self.bets[pos] += bet
                self.players_state[pos] = 0
                if self.credits[pos] == 0:
                    self.players_state[pos] = 2

            elif 2 <= action_idx < self.n_raise_bins + 2:
                # Raise
                old_high = self.high_bet
                raise_pct = self.raise_sizes[self.turn][action_idx - 2]
                effective_pot = self.pot - self.bets[pos]
                call_amount = self.high_bet - self.bets[pos]
                bet = min(call_amount + raise_pct * effective_pot, self.credits[pos])
                self.pot += bet
                self.credits[pos] -= bet
                self.bets[pos] += bet
                self.high_bet = max(self.high_bet, self.bets[pos])
                # C.7.5: track raise increment for min-raise / reopen logic
                raise_inc = self.high_bet - old_high
                if raise_inc > 0:
                    if raise_inc >= self.last_raise_size:
                        self.last_raise_size = float(raise_inc)
                        self._last_full_raise_level = self.high_bet
                self.players_state[pos] = 0
                if self.credits[pos] == 0:
                    self.players_state[pos] = 2

            elif action_idx == self.n_raise_bins + 2:
                # All-in
                old_high = self.high_bet
                bet = self.credits[pos]
                self.pot += bet
                self.credits[pos] -= bet
                self.bets[pos] += bet
                self.high_bet = max(self.high_bet, self.bets[pos])
                # C.7.5: short all-in — update last_raise_size only on full raise
                raise_inc = self.high_bet - old_high
                if raise_inc > 0:
                    if raise_inc >= self.last_raise_size:
                        self.last_raise_size = float(raise_inc)
                        self._last_full_raise_level = self.high_bet
                self.players_state[pos] = 2

        self._next_turn()

    def _next_turn(self):
        """Advance to next player/street or mark terminal. Mirrors Table.next_turn()."""
        active_count = sum(1 for s in self.players_state if s >= 0)
        if active_count <= 1:
            self.is_terminal = True
            return

        # Mark players behind on bets as needing to act
        for i in range(self.num_players):
            if self.bets[i] < self.high_bet and self.players_state[i] == 0:
                self.players_state[i] = 1

        waiting = [i for i in range(self.num_players) if self.players_state[i] == 0]
        moving = [i for i in range(self.num_players) if self.players_state[i] == 1]

        if not moving:
            # Street action complete
            if self.turn == 3:
                # River complete — terminal (showdown, but no judger)
                self.is_terminal = True
                return

            # Advance to next street
            self.turn += 1
            self.bets = [0.0] * self.num_players
            self.high_bet = 0.0
            # C.7.5: reset min-raise to big blind on new street.
            self.last_raise_size = self.big_blind
            self._last_full_raise_level = 0.0
            for i in waiting:
                self.players_state[i] = 1

            if not waiting:
                # All remaining players are all-in — no more decisions
                self.several_all_in = True
                self.is_terminal = True
                return

            # Find first player to act on new street
            start_pos = 1 if self.num_players == 2 else 0
            for offset in range(self.num_players):
                pos = (start_pos + offset) % self.num_players
                if self.players_state[pos] == 1:
                    self.active_player = pos
                    break
        else:
            # Advance to next acting player
            self.active_player = (self.active_player + 1) % self.num_players
            while self.players_state[self.active_player] != 1:
                self.active_player = (self.active_player + 1) % self.num_players

    def get_legal_actions(self):
        """Return list of *playable* action indices.

        Beyond rule-legality, also filters strictly-dominated branches that
        only inflate the search space without adding distinct game-theoretic
        choices:
          - fold when there's nothing to call (check dominates),
          - raise sizes whose resulting bet does not exceed the call amount
            (collapses to a call in step()),
          - raise sizes that already commit the full stack (duplicate of
            the explicit all-in action),
          - any raise when every other live player is already all-in (C.5):
            there is no one left to call, so a raise only piles in chips that
            must be returned uncalled — it cannot change the outcome and
            distorts the pot. Only check/call (and fold when facing a bet)
            remain.
        """
        pos = self.active_player
        call_amount = self.high_bet - self.bets[pos]
        credits_pos = self.credits[pos]
        effective_pot = self.pot - self.bets[pos]
        facing_bet = call_amount > 0

        # C.5: are all OTHER live players already all-in? Then raising is moot.
        others_live = [i for i in range(self.num_players)
                       if i != pos and self.players_state[i] >= 0]
        all_others_allin = bool(others_live) and all(
            self.players_state[i] == 2 for i in others_live)

        actions = []
        # Fold only when facing a bet (otherwise check strictly dominates).
        if facing_bet:
            actions.append(0)
        # Call/check is always playable.
        actions.append(1)

        # C.7.5: a player who already matched the last full raise level can
        # only call/fold after a short all-in pushed high_bet above that level.
        # They were "not reopened" per NLHE rules.
        short_allin_restricted = (
            self._last_full_raise_level > 0
            and self.bets[pos] >= self._last_full_raise_level
            and self.high_bet > self._last_full_raise_level
        )
        can_raise = (credits_pos > call_amount
                     and not all_others_allin
                     and not short_allin_restricted)
        if can_raise:
            # Keep distinct raise sizes; drop those that collapse to a call,
            # already match/exceed all-in, or fall below the NLHE min-raise.
            for i in range(self.n_raise_bins):
                raise_pct = self.raise_sizes[self.turn][i]
                bet = call_amount + raise_pct * effective_pot
                if bet <= call_amount:
                    # Degenerates to call in step() — skip.
                    continue
                if bet >= credits_pos:
                    # Collapses into the explicit all-in action.
                    continue
                # C.7.5: raise increment above high_bet must meet min-raise.
                raise_increment = raise_pct * effective_pot
                if raise_increment < self.last_raise_size:
                    continue
                actions.append(i + 2)
            actions.append(self.n_raise_bins + 2)  # all-in
        elif (credits_pos > 0
              and not all_others_allin
              and not short_allin_restricted):
            # Can only go all-in (not enough to raise, but has chips). Suppressed
            # when every other live player is all-in (C.5), or when the player
            # can't re-raise after a short all-in (C.7.5).
            actions.append(self.n_raise_bins + 2)

        return actions

    def get_legal_action_mask(self, n_actions):
        """Boolean mask of length n_actions: True where action is playable.

        Convenience for inference sites that need to zero out logits without
        constructing the legal list. Matches get_legal_actions() semantics.
        """
        legal = set(self.get_legal_actions())
        return [a in legal for a in range(n_actions)]
