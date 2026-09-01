import torch
from torch import nn
import numpy as np
from math import *
from env.judger import Judger

class Table:
    def __init__(self, num_players, raise_sizes, start_credits = 1000, big_blind = 10, small_blind = 5):
        self.num_players = num_players
        self.big_blind = big_blind
        self.small_blind = small_blind
        # B.6.1: per-seat starting stacks. Accept a scalar (same stack for all
        # seats) OR a per-seat sequence. Stored as a float array so downstream
        # `hero_invested = start_credits[pos] - credits[pos]` works for the
        # asymmetric (per-seat) stacks the data generators now sample.
        if np.ndim(start_credits) == 0:
            self.start_credits = np.full(num_players, float(start_credits))
        else:
            self.start_credits = np.asarray(start_credits, dtype=float)
        self.credits = list(self.start_credits)
        self.raise_sizes = raise_sizes  # list of 4 lists (one per street)
        # The widest street fixes the action layout: bins are a global index
        # space (`env/legal.py`, `nets/features.py`), and the all-in slot sits
        # after the last of them. A street that defines fewer sizes simply has
        # the trailing bins illegal on it — which is how every other
        # unavailable action is already expressed.
        self.n_raise_bins = max(len(raise_sizes[t]) for t in range(4))
        self.judger = Judger()

    def reset(self, position = None):
        if position is None:
            self.credits = list(self.start_credits)
        else:
            self.credits[position] = self.start_credits[position]

    def get_hand(self):
        pos = self.active_player
        return self.deck[5 + 2 * pos : 7 + 2 * pos]

    def rotate(self):
        idx = [self.num_players - 1] + list(range(self.num_players - 1))
        self.credits = list(np.array(self.credits)[idx])

    def start_table(self, for_history = False):
        self.players_state = np.ones((self.num_players,))
        self.active_player = 2 % self.num_players
        self.several_all_in = False
        sb = min(self.small_blind, self.credits[0])
        bb = min(self.big_blind, self.credits[1])
        if not for_history:
            self.credits[0] -= sb
            self.credits[1] -= bb
            if self.credits[0] == 0: self.players_state[0] = 2
            if self.credits[1] == 0: self.players_state[1] = 2
        self.deck = np.random.permutation(52)
        self.pot = sb + bb
        self.high_bet = self.big_blind
        self.bets = np.zeros((self.num_players,))
        self.bets[0], self.bets[1] = sb, bb
        # Per-hand cumulative contributions per player. Unlike self.bets (which
        # next_turn resets to zero on each street change), this accumulates over
        # the whole hand. Seeded with the blinds here; step() adds each action's
        # chips. Used at showdown so chips from earlier streets are not lost.
        self.cumulative_bets = np.copy(self.bets)
        self.turn = 0
        # C.7.5: NLHE min-raise tracking. The big blind is the initial "raise".
        self.last_raise_size = float(self.big_blind)
        self._last_full_raise_level = float(self.big_blind)

    def step(self, action):
        bet = 0
        if not self.several_all_in:
            action = torch.argmax(action).item()
            if action == 0:
                self.players_state[self.active_player] = -1

            if action == 1:
                bet = min(self.high_bet - self.bets[self.active_player], self.credits[self.active_player])
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.players_state[self.active_player] = 0
                if self.credits[self.active_player] == 0: self.players_state[self.active_player] = 2

            if action > 1 and action < self.n_raise_bins + 2:
                old_high = self.high_bet
                raise_pct = self.raise_sizes[self.turn][action - 2]
                effective_pot = self.pot - self.bets[self.active_player]
                call_amount = self.high_bet - self.bets[self.active_player]
                bet = min(call_amount + raise_pct * effective_pot, self.credits[self.active_player])
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.high_bet = max(self.high_bet, self.bets[self.active_player])
                # C.7.5: track raise increment for min-raise logic
                raise_inc = self.high_bet - old_high
                if raise_inc > 0 and raise_inc >= self.last_raise_size:
                    self.last_raise_size = float(raise_inc)
                    self._last_full_raise_level = float(self.high_bet)
                self.players_state[self.active_player] = 0
                if self.credits[self.active_player] == 0: self.players_state[self.active_player] = 2

            if action == self.n_raise_bins + 2:
                old_high = self.high_bet
                bet = self.credits[self.active_player]
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.high_bet = max(self.high_bet, self.bets[self.active_player])
                # C.7.5: only update on full raise
                raise_inc = self.high_bet - old_high
                if raise_inc > 0 and raise_inc >= self.last_raise_size:
                    self.last_raise_size = float(raise_inc)
                    self._last_full_raise_level = float(self.high_bet)
                self.players_state[self.active_player] = 2

        self.cumulative_bets[self.active_player] += bet
        end = self.next_turn()
        return end, self.several_all_in, self.get_state(), bet

    def next_turn(self):
        active_players = self.players_state >= 0
        if active_players.sum() == 1:
            self.credits[np.argmax(active_players)] += self.pot
            return True

        for i in range(self.num_players):
            if self.bets[i] < self.high_bet and self.players_state[i] == 0:
                self.players_state[i] = 1

        waiting_players = self.players_state == 0
        moving_players = self.players_state == 1
        if moving_players.sum() == 0:
            if self.turn == 3:
                rewards = self.judger.get_reward(self.deck, self.players_state, self.cumulative_bets)
                for i in range(self.num_players):
                    self.credits[i] += rewards[i] + self.cumulative_bets[i]
                return True
            else:
                self.turn += 1
                self.bets = np.zeros((self.num_players,))
                self.high_bet = 0
                # C.7.5: reset min-raise to big blind on new street
                self.last_raise_size = float(self.big_blind)
                self._last_full_raise_level = 0.0
                self.players_state[waiting_players] = 1
                start_pos = 1 if self.num_players == 2 else 0
                for offset in range(self.num_players):
                    pos = (start_pos + offset) % self.num_players
                    if waiting_players[pos]:
                        self.active_player = pos
                        break
                if waiting_players.sum() == 0:
                    self.several_all_in = True
        
        else:
            self.active_player = (self.active_player + 1) % self.num_players
            while self.players_state[self.active_player] != 1:
                self.active_player = (self.active_player + 1) % self.num_players

        return False 

    def get_state(self):
        active_positions = np.arange(self.num_players)[self.players_state >= 0]
        pos = self.active_player
        pot = self.pot
        bank = self.credits[pos]
        hand = self.deck[5 + 2 * pos : 7 + 2 * pos]
        bets = np.copy(self.bets)
        if self.turn == 0: table = [-1] * 5
        if self.turn == 1: table = list(self.deck[:3]) + [-1] * 2
        if self.turn == 2: table = list(self.deck[:4]) + [-1]
        if self.turn == 3: table = list(self.deck[:5]) 
        return {"active_positions": active_positions, "pos": pos, "pot": pot, "bank": bank, "hand": hand, "table": table, "bets": bets}

    def get_reward(self):
        active_positions = np.arange(self.num_players)[self.players_state >= 0]
        hands = [self.deck[5 + 2 * pos : 7 + 2 * pos] for pos in np.arange(self.num_players)]
        table = [self.deck[:3], [self.deck[3]], [self.deck[4]]]
        rewards = self.judger.get_reward(self.deck, self.players_state, self.bets)
        return {"table": table, "rewards": rewards, "active_positions": active_positions, "hands": hands}


