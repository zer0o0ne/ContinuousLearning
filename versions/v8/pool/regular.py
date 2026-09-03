"""A regular, as a cascade of rules over a board (`PLAN_PROCEDURAL_POOL.md` §P3).

The five degenerate members pin the corners of the style space and never look at
the board; the ten archetypes of §P4 are this class with ten sets of numbers.
What it is *not* is a solver: it plays recognisably, not well, and the whole
point of a pool is a fixed diverse population to best-respond to.

**Shape before content.** Everything below is a `(1326,)` numpy expression, one
row per possible holding, because that is how a member is asked: the §7.2
posterior wants "what would you have done holding *this*" for every combo
consistent with the board, ~1 225 of them at one decision. The situation — pot,
stacks, who raised — is computed once per `(record, snap_idx)` group and shared
by all of them (`pool/situation.py`); the strength table is computed once per
board and shared by every member and every decision on it (`pool/strength.py`).
A driver batch of many one-row groups and a posterior batch of one 1 225-row
group run the same code.

**Intents, then bins.** The rules produce a distribution over five *intents* —
fold, check/call, a small bet, a big one, all-in — and only then does the intent
become an action index. That split is what lets one set of rules serve any raise
grid: a size is a target fraction of the pot, and the member plays the nearest
legal bin to it. Where a size cannot be expressed the mass goes to all-in if
that is legal and to check/call otherwise, and an illegal intent (folding when
checking is free) is redistributed over the legal ones before anything is
logged.

**What the numbers mean.** Preflop thresholds are fractions of *combos* in the
ordering of §P1 — "the top 14 %" — so a range written once tightens by itself as
the table fills up. Postflop thresholds are on `q = hs^(n_opponents)`, the
probability of currently holding the best hand, with the potential of a draw
folded in separately through `ehs`. Hand classes (top pair, overpair, set) are
never thresholded on: they are what a human rule *says*, and the numbers are
what it does.
"""

from dataclasses import dataclass

import numpy as np

from pool.action_map import nearest_bin
from pool.base import PoolMember
from pool.situation import (OPEN_BUTTON, OPEN_HU_SB, PreflopAction,
                            combo_percentile, opening_position,
                            opponent_ranges, preflop_range, situation,
                            street_order)
from pool.strength import (N_COMBOS, HandClass, combo_index, preflop_rank_pct)

#: The five things a member can mean, before a raise grid is involved.
FOLD, CHECK_CALL, BET_SMALL, BET_BIG, ALL_IN = range(5)
N_INTENTS = 5

EPS = 1e-12
#: Quantile of `q` above which a hand is unraisable value, and the two rungs
#: below the archetype's own `value_hs`. §P3 fixes all three.
NUTS_Q, MEDIUM_Q, WEAK_Q = 0.97, 0.55, 0.35

#: Heads-up push/fold Nash, as a share of combos, from
#: https://pailiku.com/push-fold (read 2026-09-03; its 10 BB figures — the small
#: blind jamming 52.9 % and the big blind calling 34.2 % — agree with the other
#: published tables, and HoldemResources' own chart gives per-hand stack
#: thresholds rather than aggregates). Below 5 BB and above 20 BB the value is
#: held flat, which understates the jam at 2 BB, where equilibrium is close to
#: any two cards.
PUSH_BB = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0)
PUSH_SHARE = (0.719, 0.572, 0.529, 0.478, 0.412, 0.363)
CALL_SHARE = (0.620, 0.403, 0.342, 0.300, 0.252, 0.199)


def push_range(eff_stack_bb, n_behind):
    """Share of combos a short stack open-shoves, with `n_behind` seats to act.

    The published table is heads-up, where exactly one seat acts behind, so the
    multiway correction divides by the seats behind *beyond* that one — at a
    full table the first seat shoves a third of what the small blind shoves
    heads-up. Crude, and deliberately explicit about being crude.
    """
    share = float(np.interp(float(eff_stack_bb), PUSH_BB, PUSH_SHARE))
    return share / (1.0 + 0.5 * max(0, int(n_behind) - 1))


def call_range(eff_stack_bb):
    """Share of combos that calls a shove at this depth."""
    return float(np.interp(float(eff_stack_bb), PUSH_BB, CALL_SHARE))


@dataclass
class RegularParams:
    """One archetype's numbers. Frequencies are in [0, 1] unless said otherwise.

    Preflop thresholds are fractions of combos; postflop sizes are fractions of
    the pot; `open_size_bb` and `push_fold_bb` are in big blinds.
    """

    # preflop
    open_early: float = 0.14
    open_late: float = 0.45
    limp_share: float = 0.0
    call_open: float = 0.15
    threebet_value: float = 0.06
    threebet_bluff: float = 0.04
    call_threebet: float = 0.08
    fourbet: float = 0.025
    open_size_bb: float = 2.5
    threebet_mult: float = 3.2
    push_fold_bb: float = 12.0
    # postflop
    value_hs: float = 0.75
    cbet_dry: float = 0.75
    cbet_wet: float = 0.55
    oop_factor: float = 0.75
    size_dry: float = 0.33
    size_wet: float = 0.67
    size_river: float = 0.75
    barrel_turn: float = 0.55
    barrel_river: float = 0.45
    bluff_ratio: float = 1.0
    semi_bluff: float = 0.55
    defend_factor: float = 1.0
    raise_value: float = 0.6
    raise_bluff: float = 0.2
    slowplay: float = 0.1
    donk: float = 0.05
    overbet: float = 0.1
    allin_spr: float = 1.5
    multiway_tighten: float = 0.12


class RegularMember(PoolMember):
    """A parametric human-shaped regular that reads the board."""

    def __init__(self, name, n_actions, params, cache, preflop_table,
                 raise_sizes, style=None):
        super().__init__(name, n_actions, style)
        self.params = params
        self.cache = cache
        self.preflop_table = np.asarray(preflop_table, dtype=np.float64)
        self.raise_sizes = [list(s) for s in raise_sizes]
        assert len(self.raise_sizes) == 4, "a raise grid is one list per street"
        assert n_actions == max(len(s) for s in self.raise_sizes) + 3, (
            f"{n_actions} actions do not match a grid of "
            f"{max(len(s) for s in self.raise_sizes)} raise bins")

    # -- the pool-member contract -------------------------------------------

    def logits(self, contexts):
        out = np.zeros((len(contexts), self.n_actions), dtype=np.float64)
        groups = {}
        for i, ctx in enumerate(contexts):
            groups.setdefault((id(ctx.record), int(ctx.snap_idx)), []).append(i)
        for members in groups.values():
            ctx = contexts[members[0]]
            intents, small, big = self._intents(ctx)
            rows = [combo_index(*contexts[i].hole_cards) for i in members]
            out[members] = np.log(
                self._actions(intents[rows], ctx, small, big) + 1e-9)
        return out

    def _rank_pct(self, n_opps):
        return preflop_rank_pct(self.preflop_table, n_opps)

    def _intents(self, ctx):
        """`(1326, 5)` intent distribution, and the two size targets."""
        sit = situation(ctx)
        if sit.street == 0:
            return self._preflop(sit, ctx)
        return self._postflop(sit, ctx)

    # -- intents to the action grid -----------------------------------------

    def _actions(self, intents, ctx, small, big):
        """`(B, n_actions)` playable distribution from `(B, 5)` intents."""
        legal = np.asarray(ctx.legal_mask, dtype=bool)
        allin = self.n_actions - 1
        p = np.zeros((intents.shape[0], self.n_actions), dtype=np.float64)
        p[:, FOLD] += intents[:, FOLD]
        p[:, CHECK_CALL] += intents[:, CHECK_CALL]
        p[:, self._bin(small, ctx, legal)] += intents[:, BET_SMALL]
        p[:, self._bin(big, ctx, legal)] += intents[:, BET_BIG]
        p[:, allin if legal[allin] else CHECK_CALL] += intents[:, ALL_IN]

        p *= legal
        total = p.sum(axis=1, keepdims=True)
        assert (total > 0).all(), (
            f"{self.name} put every intent on an illegal action at seat "
            f"{ctx.acting_pos} (legal={np.flatnonzero(legal).tolist()})")
        return p / total

    def _bin(self, fraction, ctx, legal):
        """The legal raise bin nearest a target fraction of the effective pot.

        Falling back to all-in when the street offers no legal raise, and to
        check/call when even that is gone — a member that meant to bet has to
        put its mass somewhere playable, and shoving is the nearer of the two to
        what it meant.
        """
        fracs = self.raise_sizes[int(ctx.turn)]
        playable = [i for i in range(len(fracs)) if legal[i + 2]]
        if not playable:
            return self.n_actions - 1 if legal[self.n_actions - 1] else CHECK_CALL
        pick = int(nearest_bin([max(0.0, float(fraction))],
                               [fracs[i] for i in playable])[0])
        return playable[pick] + 2

    def _raise_to(self, ctx, total_chips):
        """The pot fraction that raises hero's own bet up to `total_chips`.

        The engine reads a bin as `call + fraction · (pot − my bet)`, so a size
        a human states in big blinds ("open to 2.5") or as a multiple of the
        raise it faces ("3-bet to 3.2×") has to be converted against the live
        pot before a bin can be chosen — and the size a member actually plays
        then drifts with the number of limpers, which is what happens at a real
        table too.
        """
        bets = ctx.bets
        effective_pot = float(ctx.pot) - float(bets[ctx.acting_pos])
        return (float(total_chips) - float(bets.max())) / max(effective_pot, EPS)

    # -- preflop -------------------------------------------------------------

    def _preflop(self, sit, ctx):
        p = self.params
        pct = combo_percentile(self._rank_pct, sit.n_players)
        intents = np.zeros((N_COMBOS, N_INTENTS), dtype=np.float64)
        # Folding is not playable when checking is free, and a rule that says
        # "give up" then means "check".
        give_up = FOLD if sit.facing > 0 else CHECK_CALL

        entered = sum(1 for s in sit.live if s != sit.hero
                      and sit.opp_pf_action[s] in (PreflopAction.LIMP,
                                                   PreflopAction.CALL,
                                                   PreflopAction.RAISE,
                                                   PreflopAction.RERAISE))
        tighten = (1.0 - p.multiway_tighten) ** max(0, entered - 1)

        open_to = (p.open_size_bb + sit.n_limpers) * float(ctx.big_blind)
        small = self._raise_to(ctx, open_to)

        if sit.eff_stack_bb <= p.push_fold_bb:
            if sit.n_raises_pre == 0:
                shove = pct <= push_range(sit.eff_stack_bb, sit.n_behind) * tighten
                intents[shove, ALL_IN] = 1.0
                intents[~shove, give_up] = 1.0
            else:
                call = pct <= call_range(sit.eff_stack_bb) * tighten
                intents[call, CHECK_CALL] = 1.0
                intents[~call, give_up] = 1.0
            return intents, small, small

        if sit.n_raises_pre == 0:
            if sit.facing == 0:
                # The big blind's option: nothing to call, so the choice is
                # between raising the very top of the range and checking.
                top = pct <= p.open_late * 0.5 * tighten
                intents[top, BET_SMALL] = 1.0
                intents[~top, CHECK_CALL] = 1.0
                return intents, small, small
            enter = pct <= self._open_threshold(sit) * tighten
            intents[enter, CHECK_CALL] = p.limp_share
            intents[enter, BET_SMALL] = 1.0 - p.limp_share
            intents[~enter, give_up] = 1.0
            return intents, small, small

        # Facing a raise, or a re-raise. Both are the same shape: a value band
        # that raises, a band below it that calls, and — for the first raise
        # only — a polar bluff band below that.
        high = float(ctx.bets.max())
        small = self._raise_to(ctx, p.threebet_mult * high)
        if sit.n_raises_pre == 1:
            value = pct <= p.threebet_value * tighten
            call = ~value & (pct <= p.call_open * tighten)
            bluff = ~value & ~call & (
                pct <= (p.call_open + p.threebet_bluff) * tighten)
        else:
            value = pct <= p.fourbet * tighten
            call = ~value & (pct <= p.call_threebet * tighten)
            bluff = np.zeros(N_COMBOS, dtype=bool)

        # A raise that would commit a third of the stack is played as the shove
        # it already is.
        commit = p.threebet_mult * high - float(ctx.bets[ctx.acting_pos])
        raising = ALL_IN if commit > sit.eff_stack_bb * float(ctx.big_blind) / 3.0 \
            else BET_SMALL
        intents[value | bluff, raising] = 1.0
        intents[call, CHECK_CALL] = 1.0
        intents[~(value | bluff | call), FOLD] = 1.0
        return intents, small, small

    def _open_threshold(self, sit):
        """The share of combos this archetype enters an unraised pot with.

        The archetype's two numbers are a first seat and a button, interpolated
        by position. Heads-up the small blind *is* the button but plays a far
        wider one — a real heads-up button opens most of the deck, not the
        fifth of it a six-handed button does — so the button number is scaled to
        the heads-up norm the same table already credits an opponent in that
        seat with. Without it every regular would open 45 % heads-up while
        assigning its opponent 75 % in the same seat, which is not a style, it
        is a bug that only shows up at the table size the run is training on.

        Limpers widen it: a regular attacks a limped pot. And the seat is read in
        *preflop* order, which is not postflop order — see `opening_position`.
        """
        p = self.params
        base = p.open_early + (p.open_late - p.open_early) * opening_position(
            sit.hero, sit.n_players)
        if sit.n_players == 2 and sit.is_sb:
            base *= OPEN_HU_SB / OPEN_BUTTON
        return min(1.0, base + 0.1 * sit.n_limpers)

    # -- postflop ------------------------------------------------------------

    def _postflop(self, sit, ctx):
        p = self.params
        board = tuple(c for c in ctx.board if c >= 0)
        table = self.cache.get(board)
        n_opps = max(1, sit.n_live - 1)

        value_hs = min(0.95, p.value_hs + 0.05 * max(0, sit.n_live - 2))
        multiway = (1.0 - p.multiway_tighten) ** max(0, sit.n_live - 2)
        q = table.hs ** n_opps
        h = table.ehs(n_opps)
        strong_draw = table.flush_draw | table.oesd
        value = q >= value_hs
        medium = ~value & (q >= MEDIUM_Q)
        weak_air = q < MEDIUM_Q

        pct = combo_percentile(self._rank_pct, sit.n_players)
        hero_w = preflop_range(sit.opp_pf_action[sit.hero], sit.hero,
                               sit.n_players, pct)
        opp_w = opponent_ranges(sit, self._rank_pct)

        if sit.facing > 0:
            return self._defend(sit, board, table, p, h, value, weak_air,
                                strong_draw, hero_w, opp_w, multiway)

        f = self._cbet_frequency(sit, table, p, hero_w, opp_w)
        if sit.last_aggressor == sit.hero:
            bet = self._as_aggressor(sit, board, table, p, value, medium,
                                     weak_air, strong_draw, hero_w, f)
        else:
            bet = self._as_caller(sit, p, value, medium, weak_air, strong_draw,
                                  f)
        bet = np.clip(bet * multiway, 0.0, 1.0)
        return self._bet_intents(sit, board, p, bet, value, CHECK_CALL)

    def _cbet_frequency(self, sit, table, p, hero_w, opp_w):
        """The archetype's bet frequency on this texture, from this seat."""
        f = {"dry": p.cbet_dry, "wet": p.cbet_wet,
             "mid": 0.5 * (p.cbet_dry + p.cbet_wet)}[table.texture]
        if sit.pos_frac < 0.5:
            f *= p.oop_factor
        # Whose preflop range this board favours: hero's equity against each
        # opponent's range, averaged over hero's own range, worst case.
        mass = hero_w.sum()
        if opp_w and mass > 0:
            advantage = min(float((hero_w * table.range_equity(w)).sum() / mass)
                            for w in opp_w.values())
        else:
            advantage = 0.5
        return float(np.clip(f + 0.15 * (advantage - 0.5), 0.0, 1.0))

    def _as_aggressor(self, sit, board, table, p, value, medium, weak_air,
                      strong_draw, hero_w, f):
        """Branch 1 — hero took the previous street's initiative and may bet."""
        if sit.street == 3:
            return self._river(board, table, p, value, weak_air, hero_w)

        p_bet = np.zeros(N_COMBOS, dtype=np.float64)
        p_bet[value] = 1.0 - p.slowplay
        if sit.street == 1 or sit.hero_barrels == 0:
            p_bet[medium] = f
            p_bet[~value & strong_draw] = p.semi_bluff
            p_bet[weak_air & ~strong_draw] = f * p.bluff_ratio
            return p_bet

        # A turn barrel: the bluffs that continue are the draws and the share of
        # the air the archetype keeps firing, plus a bonus when the card that
        # came beats the whole flop and every range missed it.
        scare = 0.2 if _is_overcard(board) else 0.0
        p_bet[~value & strong_draw] = p.semi_bluff * p.barrel_turn
        p_bet[weak_air & ~strong_draw] = np.clip(
            p.barrel_turn * p.bluff_ratio * f + scare, 0.0, 1.0)
        return p_bet

    def _river(self, board, table, p, value, weak_air, hero_w):
        """Branch 1 on the river — polar, and balanced against its own range.

        The bluffs are the hands that were drawing on the turn and missed, and
        how many of them fire is set by the *ratio* the archetype wants between
        bluffs and value over its own range, not by a per-hand frequency. That
        is the one place a rule cascade can be balanced at all, and it is what
        `bluff_ratio` means: 1 is the size's own indifference ratio, 0 never
        bluffs, 2 bluffs twice as often as balance does.
        """
        turn = self.cache.get(board[:4])
        missed = weak_air & (turn.flush_draw | turn.oesd | turn.gutshot)

        s = p.size_river
        alpha = float(np.clip(p.bluff_ratio * s / (1.0 + s), 0.0, 0.99))
        betting_value = float((hero_w * value).sum()) * (1.0 - p.slowplay)
        candidates = float((hero_w * missed).sum())
        share = (betting_value * alpha / (1.0 - alpha)
                 / max(candidates, EPS)) if candidates > 0 else 0.0

        p_bet = np.zeros(N_COMBOS, dtype=np.float64)
        p_bet[value] = 1.0 - p.slowplay
        p_bet[missed] = min(1.0, share) * p.barrel_river
        return p_bet

    def _as_caller(self, sit, p, value, medium, weak_air, strong_draw, f):
        """Branch 3 — nobody bet and hero did not take the initiative.

        Betting into a player who still has to act is a donk bet and is rarer
        than stabbing at a pot the aggressor has already given up on; `donk`
        scales the first and leaves the second alone.
        """
        order = street_order(sit.street, sit.n_players)
        aggressor = sit.last_aggressor
        into = (aggressor in sit.live
                and order.index(aggressor) > order.index(sit.hero))
        scale = p.donk if into else 1.0

        p_bet = np.zeros(N_COMBOS, dtype=np.float64)
        p_bet[value] = 1.0 - p.slowplay
        p_bet[medium] = 0.5 * f
        p_bet[~value & strong_draw] = p.semi_bluff * 0.6
        p_bet[weak_air & ~strong_draw] = f * p.bluff_ratio * 0.4
        return p_bet * scale

    def _bet_intents(self, sit, board, p, bet, value, otherwise):
        """Split a bet frequency into small / big / shove, rest to `otherwise`.

        Value is what gets overbet and what gets shoved: an archetype that
        overbets does it with the top of its range, and a stack short enough
        relative to the pot has no second bet to make anyway.
        """
        intents = np.zeros((N_COMBOS, N_INTENTS), dtype=np.float64)
        zero = np.zeros(N_COMBOS, dtype=np.float64)
        shove = bet * value if sit.spr <= p.allin_spr else zero
        rest = bet - shove
        big = rest * p.overbet * value if sit.street >= 2 else zero
        intents[:, ALL_IN] = shove
        intents[:, BET_BIG] = big
        intents[:, BET_SMALL] = rest - big
        intents[:, otherwise] = 1.0 - bet
        return intents, self._normal_size(sit, board, p), _overbet_size(sit)

    def _normal_size(self, sit, board, p):
        """The pot fraction this archetype bets on this street and texture."""
        if sit.street == 3:
            return p.size_river
        if sit.street == 2:
            return p.size_wet if self._completed(board) else p.size_dry
        return {"dry": p.size_dry, "wet": p.size_wet,
                "mid": 0.5 * (p.size_dry + p.size_wet)}[
                    self.cache.get(board).texture]

    def _completed(self, board):
        """Did the card that just came fill a draw?

        Measured rather than pattern-matched: the share of possible holdings
        that now make a straight or better, against the same share one street
        ago. A flush card and a straight card both move it; a blank does not.
        """
        return (_made_share(self.cache.get(board))
                > _made_share(self.cache.get(board[:-1])) + 1e-9)

    # -- facing a bet --------------------------------------------------------

    def _defend(self, sit, board, table, p, h, value, weak_air, strong_draw,
                hero_w, opp_w, multiway):
        """Branch 2 — somebody bet, and the answer is a defence frequency.

        The continuing range is the top `defend_factor · mdf` of hero's *own*
        range by equity against the opponents still in, not the top `mdf` of all
        1 326 combos: a regular defends a range, and which hands are in it is
        the only thing position and preflop action left it. On top of that sits
        the human floor — **a draw getting the right price never folds**,
        whatever the frequency says.

        The floor is on the draw's own odds and not on the hand's equity, and
        that is a correction to §P3 rather than a reading of it. Measured: with
        the floor on total equity, a pot-sized bet is called with 93 % of the
        range by *every* archetype — a nit at `defend_factor` 0.55 and a station
        at 1.35 both — because most of a preflop range beats a random hand a
        third of the time. That is not a floor, it is a ceiling on the style
        axis, and `defend_factor` stops meaning anything. On the draw's odds it
        binds where a player would say it out loud ("I'm getting three to one
        with a flush draw") and nowhere else: on a dry board against a pot-sized
        bet it adds nothing at all.
        """
        pot_odds = sit.pot_odds
        intents = np.zeros((N_COMBOS, N_INTENTS), dtype=np.float64)
        if sit.facing_allin:
            call = h >= pot_odds * (1.0 + 0.1 * (1.0 - p.defend_factor))
            intents[call, CHECK_CALL] = 1.0
            intents[~call, FOLD] = 1.0
            return intents, p.size_wet, _overbet_size(sit)

        pooled = sum(opp_w.values()) if opp_w else np.ones(N_COMBOS)
        equity = table.range_equity(pooled)
        mdf = float(np.clip(p.defend_factor / (1.0 + sit.facing), 0.0, 1.0))
        keep = (_top_mass(equity, hero_w, mdf)
                | (table.p_improve >= pot_odds)).astype(np.float64)

        raise_p = np.zeros(N_COMBOS, dtype=np.float64)
        raise_p[weak_air] = p.raise_bluff * p.bluff_ratio * 0.3
        raise_p[~value & strong_draw & (table.p_improve < pot_odds)] = (
            p.raise_bluff * p.semi_bluff)
        raise_p[value] = p.raise_value
        raise_p = np.clip(raise_p * multiway, 0.0, 1.0) * keep

        zero = np.zeros(N_COMBOS, dtype=np.float64)
        shove = raise_p * value if sit.spr <= p.allin_spr else zero
        intents[:, ALL_IN] = shove
        intents[:, BET_SMALL] = raise_p - shove
        intents[:, CHECK_CALL] = keep - raise_p
        intents[:, FOLD] = 1.0 - keep
        return intents, p.size_wet, _overbet_size(sit)


def _overbet_size(sit):
    """The pot fraction an overbet uses: pot-and-a-half, or double on the river."""
    return 2.0 if sit.street == 3 else 1.5


def _is_overcard(board):
    """True when the last card dealt is above every card before it."""
    ranks = [c // 4 for c in board]
    return len(ranks) >= 2 and ranks[-1] > max(ranks[:-1])


def _made_share(table):
    """Share of live combos already making a straight or better."""
    live = table.live
    if not live.any():
        return 0.0
    return float((table.hand_class[live] >= HandClass.STRAIGHT).mean())


def _top_mass(value, weights, fraction):
    """Boolean rows holding the top `fraction` of `weights` mass, by `value`."""
    total = float(weights.sum())
    if total <= 0 or fraction <= 0:
        return np.zeros(N_COMBOS, dtype=bool)
    if fraction >= 1:
        return weights > 0
    order = np.argsort(-np.asarray(value, dtype=np.float64), kind="stable")
    cum = np.cumsum(weights[order]) / total
    keep = np.zeros(N_COMBOS, dtype=bool)
    # `<= fraction` alone drops the row that straddles the boundary; including
    # it is what makes the kept mass reach the frequency rather than fall just
    # under it.
    reached = cum <= fraction
    reached[np.argmax(cum > fraction)] = True
    keep[order] = reached
    return keep & (weights > 0)
