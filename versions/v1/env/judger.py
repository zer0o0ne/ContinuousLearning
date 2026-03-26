import numpy as np
from copy import deepcopy

class Judger:
    def __init__(self):
        pass

    def get_reward(self, deck, players_state, bets):
        deck, players_state, bets = deepcopy(deck), deepcopy(players_state), deepcopy(bets)
        active_players = players_state >= 0
        if active_players.sum() == 1:
            bets *= -1
            bets[np.argmax(active_players)] = np.sum(bets) * -1
        else:
            active_players_numbers = np.arange(len(players_state))[active_players]
            bets = self.share_out(deck, active_players_numbers, bets)
        
        return bets

    def share_out(self, deck, active_players_numbers, bets):
        n_players = len(bets)
        rewards = np.zeros((n_players,))
        deck = list(deck)
        hands = np.array([deck[:5] + deck[5 + i * 2 : 7 + i * 2] for i in active_players_numbers])
        powers = self.eval_hands(active_players_numbers, n_players, hands)
        while np.max(powers) > 0:
            potential_winners = powers.sum(axis = 1) == np.max(powers.sum(axis = 1))
            winners = np.arange(n_players)[potential_winners]
            while len(winners) > 0:
                pie = np.min(bets[winners])
                pay_now = winners[bets[winners] >= pie]
                bets[pay_now] -= pie
                for i in range(n_players):
                    if i in pay_now:
                        continue
                    pie_ = min(bets[i], pie)
                    rewards[i] -= pie_
                    bets[i] -= pie_
                    rewards[pay_now] += pie_ / len(pay_now)
                winners = winners[bets[winners] > 0]
            powers[potential_winners] -= 1
        
        return rewards

    def eval_hands(self, active_players_numbers, n_players, hands):
        powers = np.zeros((n_players, n_players)) - 1
        for i in range(len(active_players_numbers)):
            for j in range(len(active_players_numbers)):
                if (i >= j): continue
                powers[active_players_numbers[i]][active_players_numbers[j]], powers[active_players_numbers[j]][active_players_numbers[i]] = self.compare_hands(hands[i], hands[j])

        return powers

    def compare_hands(self, hand_1, hand_2):
        hand_1, hand_2 = np.sort(hand_1), np.sort(hand_2)
        (p_1, bord_1), (p_2, bord_2) = self.compute_power(hand_1), self.compute_power(hand_2)
        if p_1 > p_2: return 1, 0
        elif p_2 > p_1: return 0, 1
        else:
            temp_1, temp_2 = bord_1 > bord_2, bord_1 == bord_2
            if temp_1: return 1, 0
            elif temp_2: return 1, 1
            else: return 0, 1

    def compute_power(self, hand):
        rank, suit = hand // 4, hand % 4
        flush, straight, fourakind, threeakind, twoakind, twopairs, full, straightflush, count = False, False, False, False, False, False, False, False, 1
        for i in range(4):
            if (suit == i).sum() >= 5: 
                flush = True
                maybe_straight = rank[suit == i]

                ranks_set = set(maybe_straight)
                if {0, 1, 2, 3, 12}.issubset(ranks_set):                                                                                                                
                    straightflush = True

                for j in range(len(maybe_straight) - 1):
                    if maybe_straight[j] + 1 == maybe_straight[j + 1]: 
                        count += 1
                        if count == 5: straightflush = True
                    elif maybe_straight[j] == maybe_straight[j + 1]:
                        continue
                    else: count = 1

        ranks_set = set(rank)
        if {0, 1, 2, 3, 12}.issubset(ranks_set):                                                                                                                
            straight = True

        count = 1
        for i in range(6):
            if rank[i] + 1 == rank[i + 1]: 
                count += 1
                if count == 5: straight = True
            elif rank[i] == rank[i + 1]:
                continue
            else: count = 1

        for r in np.flip(np.unique(rank)):
            if (rank == r).sum() == 4:
                fourakind = True
                fourakind_coef = r * 13
            if (rank == r).sum() >= 2 and threeakind:
                full = True
                full_coef = threeakind_coef + r
            if (rank == r).sum() == 3 and not threeakind:
                threeakind = True
                threeakind_coef = r * 13
            if (rank == r).sum() == 2 and twoakind and not twopairs:
                twopairs = True
                twopairs_coef = twoakind_coef + r
            if (rank == r).sum() == 2 and not twoakind:
                twoakind = True
                twoakind_coef = r * 13

        if threeakind and not full:
            three_rank = threeakind_coef // 13
            for r in np.flip(np.unique(rank)):
                if r != three_rank and (rank == r).sum() >= 2:
                    full = True
                    full_coef = threeakind_coef + r
                    break

        if straightflush: return 8 * 13 * 13, self.get_bord(8, rank, suit)
        if fourakind: return 7 * 13 * 13 + fourakind_coef, self.get_bord(7, rank, suit)
        if full: return 6 * 13 * 13 + full_coef, self.get_bord(6, rank, suit)
        if flush: return 5 * 13 * 13, self.get_bord(5, rank, suit)
        if straight: return 4 * 13 * 13, self.get_bord(4, rank, suit)
        if threeakind: return 3 * 13 * 13 + threeakind_coef, self.get_bord(3, rank, suit)
        if twopairs: return 2 * 13 * 13 + twopairs_coef, self.get_bord(2, rank, suit)
        if twoakind: return 1 * 13 * 13 + twoakind_coef, self.get_bord(1, rank, suit)

        return 0, list(np.flip(np.sort(rank)))[:5]

    def _find_best_straight(self, ranks):
        unique = sorted(set(ranks))
        best = []
        run = [unique[0]]
        for k in range(1, len(unique)):
            if unique[k] == unique[k - 1] + 1:
                run.append(unique[k])
            else:
                if len(run) >= 5:
                    best = run[-5:]
                run = [unique[k]]
        if len(run) >= 5:
            best = run[-5:]
        if {0, 1, 2, 3, 12}.issubset(set(ranks)) and len(best) < 5:
            best = [-1, 0, 1, 2, 3]
        return best

    def get_bord(self, power, rank, suit):
        bord = []

        if power == 8:
            for s in range(4):
                if (suit == s).sum() >= 5:
                    bord = self._find_best_straight(rank[suit == s])
                    break

        if power == 7:
            for r in np.flip(np.unique(rank)):
                if (rank == r).sum() == 4:
                    kicker = np.max(rank[rank != r])
                    bord = [r, r, r, r, kicker]
                    break

        if power == 6:
            three_r, two_r = None, None
            for r in np.flip(np.unique(rank)):
                if (rank == r).sum() == 3 and three_r is None:
                    three_r = r
                elif (rank == r).sum() >= 2 and two_r is None:
                    two_r = r
            bord = [three_r, three_r, three_r, two_r, two_r]

        if power == 5:
            for s in range(4):
                if (suit == s).sum() >= 5:
                    bord = list(np.sort(rank[suit == s])[-5:])
                    break

        if power == 4:
            bord = self._find_best_straight(rank)

        if power == 3:
            for r in np.flip(np.unique(rank)):
                if (rank == r).sum() == 3:
                    kickers = list(np.sort(rank[rank != r])[-2:])
                    bord = [r, r, r] + kickers
                    break

        if power == 2:
            pairs = []
            for r in np.flip(np.unique(rank)):
                if (rank == r).sum() >= 2 and len(pairs) < 2:
                    pairs.append(r)
            kicker = int(np.max(rank[(rank != pairs[0]) & (rank != pairs[1])]))
            bord = [pairs[0], pairs[0], pairs[1], pairs[1], kicker]

        if power == 1:
            for r in np.flip(np.unique(rank)):
                if (rank == r).sum() == 2:
                    kickers = list(np.sort(rank[rank != r])[-3:])
                    bord = [r, r] + kickers
                    break

        return list(np.flip(np.sort(bord)))
            
                


        
        






