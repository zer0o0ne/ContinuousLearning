"""What a decision looks like to a player who is not reading tokens
(`PLAN_PROCEDURAL_POOL.md` §P2).

`nets/tokeniser.py` already reads a `HandRecord` for the network, but into
tokens. A rule cascade needs the same record as *scalars* — am I the preflop
aggressor, how many players are still behind me, what fraction of the pot am I
being asked for — and none of those exist anywhere in the tree.

**Everything here is per decision, and that is the point.** §P1's tables are per
combo, ~1 225 rows of them; a `Situation` is the other half, identical for every
one of those rows, so the cascade computes it once per `(record, snap_idx)` and
then runs numpy over the combos. Keeping the split explicit is what stops a rule
from accidentally costing 1 225 times what it should.

**It is a function of the prefix.** Nothing below reads a snapshot past the
decision being asked about, so a `Situation` built on a finished record and one
built on the record truncated at that decision are the same object — asserted by
a test, because a leak here is exactly §9's future leak in a different costume,
and the posterior asks about decisions from the middle of finished hands.

**Ranges are what a regular *assigns*, not what an opponent holds.** They are
read off the opponent's preflop action and position and nothing else — no
postflop narrowing (that is the posterior's job and it costs a per-decision
update over 1 326 combos per opponent), and deliberately the same table for
every archetype: how well someone reads ranges is not a style axis in v1.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from pool.strength import N_COMBOS

FOLD = 0
#: The `bets` of a snapshot are per street; a preflop high bet still at the big
#: blind means nobody has raised yet.
EPS = 1e-9


class PreflopAction(IntEnum):
    """A seat's preflop line, as its *last* preflop decision leaves it."""

    UNOPENED = 0        # checked its option, or has not acted yet
    FOLD = 1
    LIMP = 2
    CALL = 3
    RAISE = 4
    RERAISE = 5


@dataclass(frozen=True)
class Move:
    """One recorded decision, read for what it did rather than for its index."""

    seat: int
    street: int
    action_idx: int
    folded: bool
    put: float                # chips this action added
    raised: bool              # it lifted the high bet — a bet or a raise
    to_call: float            # what it was facing before acting
    pot_before: float
    n_bets_street: int        # raises already made on this street before it
    at_blind_level: bool      # the high bet was still the big blind


def moves(record, upto_snap=None):
    """Every decision of `record` before `upto_snap`, as `Move`s.

    One pass, because half the fields ("how many raises came before this one")
    are properties of the sequence rather than of the decision. `upto_snap` is
    what keeps `situation` a function of the prefix; `None` reads the whole
    record, which is what the stats want.

    **What an action put in is read off the pot, not off the bets.** A decision
    that closes a street is stepped before its snapshot is taken, and the engine
    zeroes `bets` on a street change and pays the pot into `credits` at the end
    of a hand — so both of those read as nonsense for exactly the decisions that
    end something. The pot only ever grows by the chips an action adds, on every
    street and on the last action of a hand alike.
    """
    snaps = record.snapshots
    big_blind = float(record.spec.big_blind)
    out, per_street = [], {}
    for d in record.decisions:
        snap_idx = int(d["snap_idx"])
        if upto_snap is not None and snap_idx >= int(upto_snap):
            break
        before, after = snaps[snap_idx], snaps[snap_idx + 1]
        bets = np.asarray(before["bets"], dtype=np.float64)
        seat, street = int(d["acting_pos"]), int(before["turn"])
        high = float(bets.max())
        to_call = max(0.0, high - float(bets[seat]))
        put = float(after["pot"]) - float(before["pot"])
        raised = put > to_call + EPS
        out.append(Move(
            seat=seat, street=street, action_idx=int(d["action_idx"]),
            folded=int(d["action_idx"]) == FOLD, put=put, raised=raised,
            to_call=to_call, pot_before=float(before["pot"]),
            n_bets_street=per_street.get(street, 0),
            at_blind_level=high <= big_blind + EPS))
        if raised:
            per_street[street] = per_street.get(street, 0) + 1
    return out


def position_fraction(seat, n_players):
    """0 = first to act postflop, 1 = the button.

    Postflop order is seat order from the small blind, so the fraction is the
    seat itself — except heads-up, where the engine acts the big blind first
    postflop and the small blind *is* the button.
    """
    n = int(n_players)
    if n == 2:
        return 1.0 if int(seat) == 0 else 0.0
    return float(seat) / float(n - 1)


def street_order(street, n_players):
    """The seats of one street in the order the engine acts them.

    Preflop starts two seats past the small blind (the button heads-up, where
    `2 % 2` is seat 0); every later street starts at the small blind, and
    heads-up at the big blind. Both come straight out of `env/table.py`.
    """
    n = int(n_players)
    first = (2 % n) if int(street) == 0 else (1 if n == 2 else 0)
    return [(first + k) % n for k in range(n)]


@dataclass
class Situation:
    """One decision, as a rule cascade reads it."""

    street: int
    n_players: int
    hero: int
    live: tuple
    n_live: int
    n_behind: int
    pos_frac: float
    is_sb: bool
    is_bb: bool
    n_raises_pre: int
    pf_aggressor: int
    hero_is_pf_aggressor: bool
    n_limpers: int
    n_bets_street: int
    facing: float
    facing_allin: bool
    checked_to_hero: bool
    hero_bet_prev_street: bool
    hero_barrels: int
    last_aggressor: int
    pot_bb: float
    eff_stack_bb: float
    spr: float
    pot_odds: float
    opp_pf_action: dict


def situation(ctx):
    """Read the pending decision of `ctx` into scalars."""
    record = ctx.record
    n = int(record.num_players)
    hero = int(ctx.acting_pos)
    street = int(ctx.turn)
    big_blind = float(record.spec.big_blind)
    snap = record.snapshots[int(ctx.snap_idx)]
    credits = np.asarray(snap["credits"], dtype=np.float64)
    bets = np.asarray(snap["bets"], dtype=np.float64)

    past = moves(record, upto_snap=int(ctx.snap_idx))
    live = tuple(s for s in range(n)
                 if not any(m.folded for m in past if m.seat == s))
    assert hero in live, (
        f"seat {hero} folded earlier in this hand and cannot be acting")

    pre = [m for m in past if m.street == 0]
    raises_pre = [m for m in pre if m.raised]
    pf_aggressor = raises_pre[-1].seat if raises_pre else -1
    limpers = {m.seat for m in pre
               if m.put > 0 and not m.raised and m.at_blind_level}

    this_street = [m for m in past if m.street == street]
    previous = [m for m in past if m.street == street - 1 and m.raised]

    to_call = float(ctx.to_call)
    pot = float(ctx.pot)
    # `facing` is the bet as a fraction of the pot it was bet *into*, which is
    # what makes `mdf = 1/(1 + facing)` the minimum defence frequency: the pot
    # a snapshot carries already contains the outstanding bet.
    facing = to_call / max(pot - to_call, EPS) if to_call > 0 else 0.0
    high = float(bets.max())
    shoved = any(credits[s] == 0 and bets[s] >= high - EPS
                 for s in live if s != hero)

    opponents = [s for s in live if s != hero]
    eff_stack = (min(credits[hero], max(credits[s] for s in opponents))
                 if opponents else credits[hero])
    pot_bb = pot / big_blind

    barrels, s = 0, street - 1
    while s >= 1 and any(m.raised for m in past
                         if m.seat == hero and m.street == s):
        barrels += 1
        s -= 1

    order = street_order(street, n)
    behind = order[order.index(hero) + 1:]

    return Situation(
        street=street, n_players=n, hero=hero, live=live, n_live=len(live),
        n_behind=sum(1 for s in behind if s in live and credits[s] > 0),
        pos_frac=position_fraction(hero, n),
        is_sb=hero == 0, is_bb=hero == 1,
        n_raises_pre=len(raises_pre), pf_aggressor=pf_aggressor,
        hero_is_pf_aggressor=pf_aggressor == hero, n_limpers=len(limpers),
        n_bets_street=sum(1 for m in this_street if m.raised),
        facing=facing,
        # Nothing can be bet after this call: either the bet was somebody's
        # whole stack, or calling it is hero's.
        facing_allin=to_call > 0 and (shoved or to_call >= credits[hero]),
        checked_to_hero=to_call == 0 and bool(this_street),
        hero_bet_prev_street=any(m.raised for m in previous if m.seat == hero),
        hero_barrels=barrels,
        last_aggressor=previous[-1].seat if previous else -1,
        pot_bb=pot_bb, eff_stack_bb=eff_stack / big_blind,
        spr=(eff_stack / big_blind) / pot_bb,
        pot_odds=to_call / (pot + to_call) if to_call > 0 else 0.0,
        opp_pf_action=_preflop_actions(pre, n))


def _preflop_actions(pre, n_players):
    """Seat → its preflop line, from that seat's *last* preflop decision.

    A seat that has not acted yet reads as `UNOPENED`, the same as the big blind
    checking its option: in both cases nothing has been learned about it, and
    `opponent_ranges` gives both the whole range.
    """
    out = {s: PreflopAction.UNOPENED for s in range(int(n_players))}
    for m in pre:
        if m.folded:
            out[m.seat] = PreflopAction.FOLD
        elif m.raised:
            out[m.seat] = (PreflopAction.RAISE if m.at_blind_level
                           else PreflopAction.RERAISE)
        elif m.put > 0:
            out[m.seat] = (PreflopAction.LIMP if m.at_blind_level
                           else PreflopAction.CALL)
        else:
            out[m.seat] = PreflopAction.UNOPENED
    return out


#: What a regular assigns to each preflop line, as percentile bands of the
#: combo-weighted preflop ordering (`pool/strength.py::preflop_rank_pct`).
OPEN_FIRST, OPEN_BUTTON, OPEN_HU_SB = 0.15, 0.45, 0.75
RERAISE_VALUE, RERAISE_BLUFF = 0.08, (0.25, 0.35)
CALL_BAND = (0.05, 0.30)
LIMP_TOP = 0.55


def opening_position(seat, n_players):
    """How late a seat is **preflop**, for an opening range.

    Preflop and postflop are not the same seat order, and the small blind is
    where they differ most: it acts *first* on every street after the first and
    *second to last* on the first, so a postflop rule reads it as the earliest
    seat and an opening range has to read it as one of the latest — it has one
    player left to act behind it. Everything else keeps the postflop fraction,
    which is already in preflop order for seats 2 upward.

    Without this a small blind opens the range an under-the-gun seat does, which
    is not a style: it is the worst preflop leak in poker, given to every
    archetype at once.
    """
    n = int(n_players)
    if n >= 3 and int(seat) == 0:
        return 1.0
    return position_fraction(seat, n)


def opening_range(seat, n_players):
    """The fraction of combos a seat is credited with opening from there."""
    if int(n_players) == 2 and int(seat) == 0:
        return OPEN_HU_SB
    return OPEN_FIRST + (OPEN_BUTTON - OPEN_FIRST) * opening_position(
        seat, n_players)


def preflop_range(action, seat, n_players, pct):
    """`(1326,)` weights a regular credits one seat's preflop line with.

    Hero's own range is the same table read for hero's own line, which is what
    the cascade needs to say "the top `mdf` of *my* range" or "my bluffs are
    this share of my value bets".
    """
    w = np.zeros(N_COMBOS, dtype=np.float64)
    if action == PreflopAction.RAISE:
        w[pct <= opening_range(seat, n_players)] = 1.0
    elif action == PreflopAction.RERAISE:
        w[pct <= RERAISE_VALUE] = 1.0
        w[(pct > RERAISE_BLUFF[0]) & (pct <= RERAISE_BLUFF[1])] = 0.5
    elif action == PreflopAction.CALL:
        w[(pct > CALL_BAND[0]) & (pct <= CALL_BAND[1])] = 1.0
    elif action == PreflopAction.LIMP:
        w[pct <= LIMP_TOP] = 1.0
    else:
        w[:] = 1.0
    return w


def combo_percentile(rank_pct_fn, n_players):
    """The combo percentile array for the table hero is sitting at."""
    pct = np.asarray(rank_pct_fn(max(1, int(n_players) - 1)), dtype=np.float64)
    assert pct.shape == (N_COMBOS,), (
        f"a percentile is one value per combo, {N_COMBOS} of them; got "
        f"{pct.shape}")
    return pct


def opponent_ranges(sit, rank_pct_fn):
    """Seat → `(1326,)` weights, for hero's live opponents.

    `rank_pct_fn(n_opps)` returns the combo percentile array of
    `pool/strength.py::preflop_rank_pct`; it is called once, for the table hero
    is sitting at, so a range written as a fraction of combos tightens by itself
    as the table fills up.
    """
    pct = combo_percentile(rank_pct_fn, sit.n_players)
    return {seat: preflop_range(sit.opp_pf_action[seat], seat, sit.n_players,
                                pct)
            for seat in sit.live if seat != sit.hero}
