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
    ]

    def __init__(self, num_players, hero_pos, active_player, players_state,
                 credits, bets, pot, high_bet, turn, raise_sizes,
                 n_raise_bins, is_terminal=False, several_all_in=False):
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
                raise_pct = self.raise_sizes[self.turn][action_idx - 2]
                effective_pot = self.pot - self.bets[pos]
                call_amount = self.high_bet - self.bets[pos]
                bet = min(call_amount + raise_pct * effective_pot, self.credits[pos])
                self.pot += bet
                self.credits[pos] -= bet
                self.bets[pos] += bet
                self.high_bet = max(self.high_bet, self.bets[pos])
                self.players_state[pos] = 0
                if self.credits[pos] == 0:
                    self.players_state[pos] = 2

            elif action_idx == self.n_raise_bins + 2:
                # All-in
                bet = self.credits[pos]
                self.pot += bet
                self.credits[pos] -= bet
                self.bets[pos] += bet
                self.high_bet = max(self.high_bet, self.bets[pos])
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
        """Return list of legal action indices."""
        pos = self.active_player
        call_amount = self.high_bet - self.bets[pos]
        can_raise = self.credits[pos] > call_amount

        actions = [0, 1]  # fold and call/check always legal

        if can_raise:
            for i in range(self.n_raise_bins):
                actions.append(i + 2)
            actions.append(self.n_raise_bins + 2)  # all-in
        elif self.credits[pos] > 0:
            # Can only go all-in (not enough to raise, but has chips)
            actions.append(self.n_raise_bins + 2)

        return actions
