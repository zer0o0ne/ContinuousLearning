"""gto_helper.py – advanced GTO advisor (Python 3)
Copied from https://github.com/sol5000/gto
Features: equity simulation, EV-based decisions, strict mode
"""
import csv, json, math, sys, random, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import eval7

try:
    _ = eval7.Card("Ah").rank_char
except AttributeError:
    def _rank_char(self):
        return str(self)[0]

try:
    import numpy as np
except ImportError:
    np = None

SUITS, RANKS = "shdc", "23456789TJQKA"

CACHE = {}


def card_rank_char(card: eval7.Card) -> str:
    rc = getattr(card, "rank_char", str(card)[0])
    rc = rc.upper()
    return "T" if rc == "0" else rc


STRICT_THRESH = {"raise": 0.65, "check": 0.4}

ACCURACY_ITERS = {"Fast": 10000, "Balanced": 25000, "Detailed": 100000}

GAMES = ["Holdem", "Short Deck"]


def cards(txt: str):
    """Parse cards. Accepts spaced ("Ah Ks") or concatenated ("AhKs7c")."""
    txt = txt.strip()
    if not txt:
        return []
    if " " in txt:
        raw = txt.split()
    else:
        if len(txt) % 2:
            raise ValueError("Card string length must be even (pairs of chars).")
        raw = [txt[i:i+2] for i in range(0, len(txt), 2)]
    toks = []
    for t in raw:
        if len(t) != 2:
            raise ValueError(f"Bad card: {t}")
        rank, suit = t[0].upper(), t[1].lower()
        if rank == '0':
            rank = 'T'
        if rank not in RANKS or suit not in SUITS:
            raise ValueError(f"Bad card: {t}")
        toks.append(rank + suit)
    if len(toks) != len(set(toks)):
        raise ValueError("Duplicate cards detected.")
    return [eval7.Card(t) for t in toks]


_order, ORDER = 169, {}
for r1 in RANKS[::-1]:
    for r2 in RANKS[::-1]:
        if r1 < r2:
            continue
        for suited in (True, False):
            ORDER[(r1, r2, suited)] = _order
            _order -= 1


def top_range(p):
    keep = math.ceil(169 * p / 100)
    return set(k for k, _ in zip(sorted(ORDER, key=ORDER.get, reverse=True), range(keep)))


def deck_for_game(game: str) -> eval7.Deck:
    deck = eval7.Deck()
    if game.lower().startswith("short"):
        for r in "2345":
            for s in SUITS:
                card = eval7.Card(r + s)
                deck.cards.remove(card)
    return deck


def _simulate(hero, board, villains, rng, weighted, iters, game):
    wins, buckets = 0.0, Counter()
    for _ in range(iters):
        deck = deck_for_game(game)
        [deck.cards.remove(c) for c in hero + board]
        deck.shuffle()
        opp = []
        while len(opp) < villains:
            a, b = deck.deal(2)
            r1 = max(card_rank_char(a), card_rank_char(b))
            r2 = min(card_rank_char(a), card_rank_char(b))
            s = a.suit == b.suit
            if weighted is not None:
                w = weighted.get((r1, r2, s), 0.0)
                if random.random() > w:
                    deck.cards.extend([a, b])
                    deck.shuffle()
                    continue
            elif rng:
                if (r1, r2, s) not in rng:
                    deck.cards.extend([a, b])
                    deck.shuffle()
                    continue
            opp.append([a, b])
        sim_board = board + deck.deal(5 - len(board))
        hero_sc = eval7.evaluate(hero + sim_board)
        best, ties, hero_best = hero_sc, 1, True
        for v in opp:
            vs = eval7.evaluate(v + sim_board)
            if vs > best:
                best, ties, hero_best = vs, 1, False
            elif vs == best:
                ties += 1
                hero_best |= vs == hero_sc
        if hero_best and hero_sc == best:
            wins += 1 / ties
            buckets[int((1 / ties) * 10)] += 1
        else:
            buckets[0] += 1
    return wins, buckets


def equity(hero, board, villains, pct=None, custom=None, iters=25000, weighted=None, multiprocess=False, show_progress=False, game="Holdem"):
    """Simulate equity via Monte Carlo.

    Args:
        hero: list of eval7.Card (2 cards)
        board: list of eval7.Card (0, 3, 4, or 5 cards)
        villains: int, number of opponents
        pct: top X percent of hands for villain range
        custom: optional set defining a custom range
        iters: number of Monte Carlo iterations
        game: "Holdem" or "Short Deck"

    Returns:
        (equity_float, histogram_list)
    """
    rng = custom if custom is not None else (top_range(pct) if (pct or 0) > 0 else None)

    key = (
        tuple(str(c) for c in hero),
        tuple(str(c) for c in board),
        villains,
        pct if custom is None else tuple(sorted(custom)),
        iters,
        game,
    )
    if key in CACHE:
        return CACHE[key]
    if multiprocess:
        try:
            import multiprocessing as mp
            procs = min(mp.cpu_count(), 4)
            chunk = iters // procs
            todo = [chunk + (1 if i < iters % procs else 0) for i in range(procs)]
            with mp.Pool(procs) as pool:
                args = [(hero, board, villains, rng, weighted, n, game) for n in todo]
                results = pool.starmap(_simulate, args)
        except Exception:
            results = [_simulate(hero, board, villains, rng, weighted, iters, game)]
    else:
        results = [_simulate(hero, board, villains, rng, weighted, iters, game)]
    wins, buckets = 0.0, Counter()
    for w, b in results:
        wins += w
        buckets.update(b)
    eq = wins / iters
    hist = (np.bincount([min(k, 9) for k in buckets.elements()], minlength=10) / iters).tolist() if np else [buckets[i] / iters for i in range(10)]
    CACHE[key] = (eq, hist)
    return eq, hist


def strict_action(eq):
    return (
        "RAISE"
        if eq >= STRICT_THRESH["raise"]
        else "CHECK" if eq >= STRICT_THRESH["check"] else "FOLD"
    )


RAISE_SIZES = {"0.5": .5, "1": 1.0, "2": 2.0, "shove": None}


def decide_bets(eq, pot, bet, stack, pref="1"):
    """Calculate EV for fold/call/raise and return best action.

    Args:
        eq: equity float (0-1)
        pot: current pot size
        bet: facing bet amount
        stack: hero's remaining stack
        pref: raise size preference ("0.5", "1", "2", "shove")

    Returns:
        (action_str, fold_ev, call_ev, raise_ev, move_details)
    """
    call_ev = eq * (pot + bet) - (1 - eq) * bet
    fold_ev = 0.0
    raise_total = stack if pref == "shove" or stack <= bet else min(bet + (pot + 2 * bet) * RAISE_SIZES[pref], stack)
    raise_ev = eq * (pot + bet + raise_total) - (1 - eq) * raise_total
    best = max(fold_ev, call_ev, raise_ev)
    if best == raise_ev and raise_total > bet:
        act = "ALL_IN" if raise_total == stack else "RAISE"
        mv = {"call": bet, "raise": round(raise_total - bet, 2), "total": round(raise_total, 2)}
    elif best == call_ev and stack >= bet:
        act, mv = "CALL", {"call": bet, "raise": 0, "total": bet}
    else:
        act, mv = "FOLD", {}
    return act, fold_ev, call_ev, raise_ev, mv
