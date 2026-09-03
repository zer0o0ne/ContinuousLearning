"""The stat line a player would be read by (`PLAN_PROCEDURAL_POOL.md` §P2, ⚠6).

§P4's realism gate asks a question no existing gate asks: *does this member play
like the kind of player it is named after*. G1 reports losses and G3 reports
label cost; neither says whether a seat opened 8 % of hands or 60 % of them, and
that is the only evidence available for the eight table sizes and the whole
stack range where no benchmark exists.

The definitions are PokerTracker's, and every stat comes back as
`(numerator, denominator)` rather than as a ratio. Two reasons, and the second
is the one that matters: a band can then be checked against a *count* when the
denominator is small — a c-bet frequency over eleven opportunities is not a
frequency — and a denominator of zero is visible instead of being a `nan` or a
silently-invented 0.

Everything is read through `pool/situation.py::moves`, the same pass the
cascade's own `situation` uses, so "this decision was a bet" cannot come to mean
one thing in a member's rules and another in the report on it.
"""

import numpy as np

from pool.situation import moves

#: Every stat this module reports, in the order the report lists them.
STAT_NAMES = ("vpip", "pfr", "threebet", "fold_to_threebet", "limp",
              "cbet_flop", "fold_to_cbet", "barrel_turn", "barrel_river",
              "af", "agg_pct", "wtsd", "wsd", "check_raise", "steal",
              "overbet_pct")

FLOP, TURN, RIVER = 1, 2, 3


def steal_seats(n_players):
    """The seats a steal is attempted from: cutoff, button and small blind.

    Seat 0 is the small blind and the highest seat is the button
    (`env/table.py`), so the cutoff is one seat below the button — and only
    exists once the table has four seats, since at three the seat below the
    button is the big blind and heads-up the small blind *is* the button.
    """
    n = int(n_players)
    return {0} | {s for s in (n - 1, n - 2) if s >= 2}


def _blank():
    return {name: [0, 0] for name in STAT_NAMES}


def hud_stats(records, seat_filter=None):
    """`{stat: (numerator, denominator)}` over `records`.

    `seat_filter(record, seat) -> bool` selects whose decisions are counted;
    `None` counts every seat of every hand. A filter is per record because the
    seat a member occupies rotates with the button — to follow one pool member
    rather than one chair, pass
    `lambda r, s: r.spec.seat_members[s] == member_idx`.
    """
    counts = _blank()
    for record in records:
        played = moves(record)
        n = int(record.num_players)
        for seat in range(n):
            if seat_filter is not None and not seat_filter(record, seat):
                continue
            _one_seat(counts, record, played, seat)
    return {name: (int(num), int(den)) for name, (num, den) in counts.items()}


def _add(counts, name, num, den):
    counts[name][0] += int(num)
    counts[name][1] += int(den)


def _one_seat(counts, record, played, seat):
    own = [m for m in played if m.seat == seat]
    if not own:
        return
    pre = [m for m in own if m.street == 0]
    post = [m for m in own if m.street >= FLOP]

    _preflop(counts, record, played, seat, pre)
    _postflop(counts, record, played, seat, post)
    _showdown(counts, record, played, seat, pre)


def _preflop(counts, record, played, seat, pre):
    if not pre:
        return
    _add(counts, "vpip", any(m.put > 0 for m in pre), 1)
    _add(counts, "pfr", any(m.raised for m in pre), 1)
    _add(counts, "limp",
         any(m.put > 0 and not m.raised and m.at_blind_level for m in pre), 1)

    # Facing exactly one raise is the 3-bet spot; facing two or more *after
    # having raised yourself* is the spot a 3-bet was aimed at.
    has_raised = False
    for m in pre:
        if m.n_bets_street == 1:
            _add(counts, "threebet", m.raised, 1)
        if has_raised and m.n_bets_street >= 2:
            _add(counts, "fold_to_threebet", m.folded, 1)
        has_raised = has_raised or m.raised

    if seat in steal_seats(record.num_players):
        chance = _steal_chance(played, seat)
        if chance is not None:
            _add(counts, "steal", chance.raised, 1)


def _steal_chance(played, seat):
    """This seat's decision in a pot nobody has entered yet, if it had one.

    A steal is an attempt on an *unentered* pot: a limper ahead of the button
    ends the opportunity rather than creating one. Folding the opportunity is
    still an opportunity — that is what makes this an attempt frequency and not
    "of the pots you played, how many did you raise".
    """
    for m in played:
        if m.street != 0:
            break
        if m.seat == seat:
            return m
        if m.put > 0:
            break
    return None


def _postflop(counts, record, played, seat, post):
    raises = sum(1 for m in post if m.raised)
    calls = sum(1 for m in post if m.put > 0 and not m.raised and not m.folded)
    checks = sum(1 for m in post
                 if m.put == 0 and not m.raised and not m.folded)
    _add(counts, "af", raises, calls)
    _add(counts, "agg_pct", raises, raises + calls + checks)

    for m in post:
        if m.raised:
            # A bet is an overbet when it is bigger than the pot; a *raise* is
            # one when the part beyond the call is bigger than the pot the call
            # would leave. Without the second half every pot-sized raise counts
            # as an overbet and the stat measures how often a member raises.
            _add(counts, "overbet_pct",
                 m.put - m.to_call > m.pot_before + m.to_call, 1)

    aggressor = _preflop_aggressor(played)
    if seat == aggressor:
        opener = _first_free(post, FLOP)
        if opener is not None:
            _add(counts, "cbet_flop", opener.raised, 1)
    else:
        facing = _facing_the_cbet(played, seat, aggressor)
        if facing is not None:
            _add(counts, "fold_to_cbet", facing.folded, 1)

    for street, name in ((TURN, "barrel_turn"), (RIVER, "barrel_river")):
        if not any(m.raised for m in post if m.street == street - 1):
            continue
        follow = _first_free(post, street)
        if follow is not None:
            _add(counts, name, follow.raised, 1)

    for street in (FLOP, TURN, RIVER):
        _check_raise(counts, [m for m in post if m.street == street])


def _preflop_aggressor(played):
    raises = [m for m in played if m.street == 0 and m.raised]
    return raises[-1].seat if raises else -1


def _first_free(own, street):
    """This seat's first decision on `street` with nobody having bet yet."""
    for m in own:
        if m.street == street and m.n_bets_street == 0:
            return m
    return None


def _facing_the_cbet(played, seat, aggressor):
    """This seat's first flop decision facing a bet made by the aggressor."""
    bet = None
    for m in played:
        if m.street != FLOP:
            continue
        if bet is None:
            if m.raised:
                bet = m if m.seat == aggressor else False
                if bet is False:
                    return None
            continue
        if m.seat == seat and m.to_call > 0:
            return m
    return None


def _check_raise(counts, street_moves):
    """A check, then a raise on the same street — of the times a bet came back.

    The denominator is the times the check was *answered*: checking behind and
    seeing a free card is not a missed check-raise.
    """
    checked = False
    for m in street_moves:
        if not checked:
            checked = m.put == 0 and not m.raised and not m.folded
            continue
        if m.to_call > 0:
            _add(counts, "check_raise", m.raised, 1)
            return


def _showdown(counts, record, played, seat, pre):
    saw_flop = (not any(m.folded for m in pre)
                and any(int(s["turn"]) >= FLOP for s in record.snapshots))
    showed = seat in record.showdown
    _add(counts, "wtsd", showed, saw_flop)
    _add(counts, "wsd", showed and float(np.asarray(record.rewards)[seat]) > 0,
         showed)
