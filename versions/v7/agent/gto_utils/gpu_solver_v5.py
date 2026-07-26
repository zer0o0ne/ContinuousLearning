"""
Poker solver v5 — chance-sampled vector CFR over the real betting tree.

Unlike v1-v4 (one-step heuristic EV models: call -> immediate showdown,
raise -> one level of threshold-based opponent response), v5 builds the actual
betting tree from the current decision point to the END of the hand (all
remaining streets) and runs regret-matching CFR (linear CFR with regret
flooring, CFR+-style) over it:

  - vector form: strategies/regrets are kept per-combo over each player's
    whole range (like commercial range-vs-range solvers). The root is a real
    CFR node for hero, so opponents respond to hero's per-action RANGE
    composition (raising range vs calling range), not to hero's full range.
  - chance sampling: each iteration samples a batch of board runouts;
    showdown utilities use exact 7-card ranks on those runouts via
    `evaluate_hands` — no card abstraction, so no bucketing bias;
  - the hero's root decision uses the game's full action set (every raise
    bin); non-root decisions use a coarse action set (`future_bet_sizes`,
    `raise_cap`). This trades tree size for mild action-abstraction
    coarseness, but every line still ends in a fold terminal or a river
    showdown — the early-showdown bias of v1-v4 does not exist here in any
    configuration.

Root action EVs are extracted as iteration-averaged counterfactual values of
hero's actual combo, normalized by the opponents' root reach mass (standard
CFV / reach-mass ratio estimator). Fold EV is exact (-hero_invested).

Stacks and side pots are modeled fully: every solver player plays their
real remaining stack (`opponent_stacks`), short calls are all-in-for-less,
and terminal pots are decomposed into main/side-pot layers from the
players' total investments (`opponent_invested` + in-solve chips) — an
uncalled bet comes back as a singleton refund layer. Callers that pass no
stacks fall back to hero's stack for everyone (the old v3-style model).

Approximations (documented; noise- or coarseness-type, not structural):
  - at the root street, opponents are treated as having already matched
    `facing_bet`; hero's call closes the street (they respond only to a
    raise), while at `facing_bet == 0` opponents still act behind. Same
    within-street model as v3, but subsequent streets ARE played out.
  - act order is seat order on every street (no preflop blind-order special
    case).
  - at most `max_opponents` opponents are CFR players (primary = last
    aggressor, then range order); further opponents' chips stay in the pot
    as dead money.
  - multiway (3+ live at showdown) win probabilities multiply pairwise
    reach-weighted win rates (independence approximation; heads-up terminals
    are exact, including per-combo card removal via disjointness masks).
  - ranges larger than `max_combos` are subsampled (weighted, without
    replacement) per solve.

Main accuracy/speed knob: `iterations` (CFR iterations). Secondary:
`batch_runouts` (variance per iteration), `future_bet_sizes`/`raise_cap`
(tree width). Cost per solve ~ iterations x tree_nodes; more iterations
means less noise, never less structural bias.

Card encoding: card_id 0-51, rank = card_id // 4, suit = card_id % 4.
"""

import os
import sys
import warnings

import numpy as np
import torch

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from gpu_solver import evaluate_hands
from gpu_solver_v2 import get_position_range, narrow_range, expand_range
from gpu_solver_v3 import compute_combo_weights

V5_DEFAULTS = {
    "iterations": 64,         # CFR iterations — the main accuracy/speed knob
    "batch_runouts": 4,       # board runouts averaged per iteration
    "future_bet_sizes": [1.0],  # pot fractions for non-root bets/raises
    "raise_cap": 2,           # max bets+raises per street below the root
    "allin_spr": 4.0,         # non-root all-in only when stack <= spr * pot
    "max_root_sizes": 5,      # distinct root raise subtrees; other bins are
                              # EV-interpolated over the raise amount
    "max_opponents": 2,       # CFR-modeled opponents (rest = dead money)
    "max_combos": 120,        # per-range combo cap (weighted subsample)
    "max_tree_nodes": 8000,   # safety: rebuild coarser above this
}

# filled by solve_spot for diagnostics/tests: {"nodes": int, "terminals": int}
_last_stats = {}


class _TreeTooBig(Exception):
    pass


# ---------------------------------------------------------------------------
# Betting tree
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("player", "children", "terminal", "pot", "invested",
                 "alive", "showdown", "layers")

    def __init__(self):
        self.player = -1        # acting player index (decision nodes)
        self.children = []      # list of node ids
        self.terminal = False
        self.pot = 0.0
        self.invested = None    # tuple per player (chips incl. priors)
        self.alive = None       # tuple of player indices live at terminal
        self.showdown = False   # True: rank payoff; False: single winner
        self.layers = None      # list of (amount, eligible frozenset)


def _pot_layers(pot, invested, alive):
    """Decompose a terminal pot into main/side-pot layers.

    invested: per-player TOTAL contributions (priors + in-solve).
    Untracked dead money (pot - sum(invested): folded blinds, opponents
    dropped by max_opponents) goes into the bottom layer, contested by
    every live player. Tracked folded chips above the top live investment
    go into the top layer. A live player's investment nobody could match
    forms a singleton layer — an uncalled-bet refund.

    Returns list of (amount, eligible frozenset of player idx); amounts
    sum to pot.
    """
    n = len(invested)
    alive = sorted(alive)
    caps = sorted({round(invested[p], 6) for p in alive})
    layers = []
    prev = 0.0
    dead_untracked = max(0.0, pot - sum(invested))
    for i, t in enumerate(caps):
        elig = frozenset(p for p in alive if invested[p] >= t - 1e-6)
        amt = sum(max(0.0, min(invested[p], t) - prev) for p in range(n))
        if i == 0:
            amt += dead_untracked
        layers.append([amt, elig])
        prev = t
    # tracked folded chips above the top live investment -> top layer
    alive_set = set(alive)
    extra = sum(max(0.0, invested[p] - prev)
                for p in range(n) if p not in alive_set)
    if extra > 0:
        layers[-1][0] += extra
    return [(amt, elig) for amt, elig in layers if amt > 1e-9]


def _build_tree(n_players, order, root_pot, root_facing, root_stacks,
                invested0, street0, root_action_specs, future_bet_sizes,
                raise_cap, allin_spr, big_blind, max_nodes):
    """Build the betting tree. Returns (nodes, root_id).

    order: player indices in seat order (act rotation). Player 0 is hero.
    root_stacks: per-player remaining stacks (dict player -> chips).
    root_action_specs: list of (kind, put_in) for hero's root node, kind in
        {"fold", "call", "raise"}.
    """
    nodes = []

    def new_node():
        if len(nodes) >= max_nodes:
            raise _TreeTooBig()
        nodes.append(_Node())
        return len(nodes) - 1

    def make_terminal(pot, invested, alive, showdown):
        nid = new_node()
        n = nodes[nid]
        n.terminal = True
        n.pot = pot
        n.invested = tuple(invested)
        n.alive = tuple(sorted(alive))
        n.showdown = showdown
        n.layers = _pot_layers(pot, n.invested, n.alive)
        return nid

    def next_actor(alive, allin, acted, street_bets, high, from_player):
        """Next player after from_player (seat order, wrapping) who still
        needs to act this street. None -> betting closed."""
        k = order.index(from_player)
        for step in range(1, n_players + 1):
            p = order[(k + step) % n_players]
            if p not in alive or p in allin:
                continue
            if p not in acted or street_bets.get(p, 0.0) < high - 1e-9:
                return p
        return None

    def advance_street(pot, invested, alive, allin, stacks_rem, street):
        if street >= 3:
            return make_terminal(pot, invested, alive, showdown=True)
        actors = [p for p in order if p in alive and p not in allin]
        if len(actors) <= 1:
            # all-in runout — no more betting
            return make_terminal(pot, invested, alive, showdown=True)
        return decision(pot, invested, alive, allin, stacks_rem,
                        street + 1, {p: 0.0 for p in alive}, 0.0,
                        set(allin), 0, actors[0], None)

    def apply_action(kind, put_in, p, pot, invested, alive, allin, stacks_rem,
                     street, street_bets, high, acted, raises):
        pot2 = pot
        invested2 = list(invested)
        alive2 = set(alive)
        allin2 = set(allin)
        stacks2 = dict(stacks_rem)
        sb2 = dict(street_bets)
        acted2 = set(acted)
        high2 = high
        raises2 = raises

        if kind == "fold":
            alive2.discard(p)
            if len(alive2) == 1:
                return make_terminal(pot2, invested2, alive2, showdown=False)
            nxt = next_actor(alive2, allin2, acted2, sb2, high2, p)
            if nxt is None:
                return advance_street(pot2, invested2, alive2, allin2,
                                      stacks2, street)
            return decision(pot2, invested2, alive2, allin2, stacks2, street,
                            sb2, high2, acted2, raises2, nxt, None)

        put_in = min(put_in, stacks2[p])
        pot2 += put_in
        invested2[p] += put_in
        stacks2[p] -= put_in
        sb2[p] = sb2.get(p, 0.0) + put_in
        acted2.add(p)
        if stacks2[p] <= 1e-9:
            allin2.add(p)

        if kind == "raise":
            high2 = max(high2, sb2[p])
            raises2 += 1
            acted2 = {p} | (allin2 & alive2)  # others must respond again

        nxt = next_actor(alive2, allin2, acted2, sb2, high2, p)
        if nxt is None:
            return advance_street(pot2, invested2, alive2, allin2,
                                  stacks2, street)
        return decision(pot2, invested2, alive2, allin2, stacks2, street,
                        sb2, high2, acted2, raises2, nxt, None)

    def decision(pot, invested, alive, allin, stacks_rem, street,
                 street_bets, high, acted, raises, p, action_specs):
        nid = new_node()
        nodes[nid].player = p

        if action_specs is None:
            cost = max(0.0, high - street_bets.get(p, 0.0))
            action_specs = []
            if cost > 1e-9:
                action_specs.append(("fold", 0.0))
            action_specs.append(("call", cost))
            # graded abstraction: streets beyond the NEXT one (relative to
            # the root street) matter least for the root EVs — cap them at
            # one bet per street to keep the tree tractable
            cap_here = raise_cap if street <= street0 + 1 else min(raise_cap, 1)
            if raises < cap_here and stacks_rem[p] > cost + 1e-9:
                amounts = []
                for f in future_bet_sizes:
                    incr = max(f * (pot + cost), big_blind)
                    amounts.append(cost + incr)
                # deep-stack all-ins add subtrees without GTO mass; include
                # the shove only when stacks are shallow relative to the pot
                if stacks_rem[p] <= allin_spr * max(pot, big_blind):
                    amounts.append(stacks_rem[p])
                seen = set()
                for amt in amounts:
                    amt = min(amt, stacks_rem[p])
                    key = round(amt, 6)
                    if key in seen or amt <= cost + 1e-9:
                        continue
                    seen.add(key)
                    action_specs.append(("raise", amt))

        children = []
        for kind, amt in action_specs:
            children.append(apply_action(
                kind, amt, p, pot, invested, alive, allin, stacks_rem,
                street, street_bets, high, acted, raises))
        nodes[nid].children = children
        return nid

    alive = set(range(n_players))
    stacks_rem = dict(root_stacks)
    # players with no chips behind (already all-in in the real hand) never
    # act in the tree but stay live for showdown
    allin = {p for p in range(n_players) if stacks_rem[p] <= 1e-9}
    street_bets = {p: (0.0 if p == 0 else root_facing)
                   for p in range(n_players)}
    # opponents already matched the current high; they respond only to a
    # raise. At facing 0 they still act behind hero.
    acted = set(range(1, n_players)) if root_facing > 1e-9 else set()

    root_id = decision(root_pot, invested0, alive, allin, stacks_rem,
                       street0, street_bets, root_facing, acted, 0, 0,
                       root_action_specs)
    return nodes, root_id


# ---------------------------------------------------------------------------
# Range preparation
# ---------------------------------------------------------------------------

def _prepare_range(hand_types, actions, dead, max_combos, rng):
    """Expand + Bayes-weight + subsample one player's range.

    Returns (combos (M,2) int64 numpy, weights (M,) float64 numpy, sum=1).
    """
    combos_t = expand_range(hand_types, dead)
    combos = combos_t.cpu().numpy().astype(np.int64)
    M = combos.shape[0]
    if M == 0:
        return combos, np.zeros(0, dtype=np.float64)

    w_t = compute_combo_weights(hand_types, actions, dead_cards=dead)
    if w_t is not None and len(w_t) == M:
        w = w_t.cpu().numpy().astype(np.float64)
    else:
        w = np.ones(M, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else np.full(M, 1.0 / M)

    if M > max_combos:
        idx = rng.choice(M, size=max_combos, replace=False, p=w)
        combos = combos[idx]
        w = w[idx]
        w = w / w.sum()
    return combos, w


def _disjoint_mask(combos_a, combos_b):
    """(Ma, Mb) float64: 1.0 where combos share no card."""
    a0 = combos_a[:, 0][:, None]
    a1 = combos_a[:, 1][:, None]
    b0 = combos_b[:, 0][None, :]
    b1 = combos_b[:, 1][None, :]
    clash = (a0 == b0) | (a0 == b1) | (a1 == b0) | (a1 == b1)
    return (~clash).astype(np.float64)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_spot(hero_cards, board_cards, opponent_range_hand_types,
               pot, facing_bet, stack, hero_invested,
               street_raises, effective_pot, n_actions,
               street=0, hero_position=0, n_players=6,
               action_history=None, opponent_positions=None,
               big_blind=10.0, v5_params=None, seed=None,
               hero_range_hand_types=None,
               opponent_stacks=None, opponent_invested=None):
    """Solve the current spot with vector CFR and return per-action EVs.

    Args:
        hero_cards: (2,) int64 tensor — hero's actual hand
        board_cards: (B,) int64 tensor, B in {0,3,4,5}
        opponent_range_hand_types: list of hand-type lists (live opponents)
        pot / facing_bet / stack / hero_invested: current game state (chips);
            same conventions as compute_ev_v3
        street_raises: raise fractions of the current street (the game's bins)
        effective_pot: pot - hero street bets (table raise convention)
        n_actions: fold + call + len(street_raises) + all-in
        street: 0..3
        action_history: list of (position, act_type) — Bayes narrowing input
        opponent_positions: seat positions aligned with the ranges list
        v5_params: dict overriding V5_DEFAULTS
        seed: int — deterministic solve; None -> seeded from global numpy RNG
        opponent_stacks: per-opponent remaining chips aligned with the
            ranges list; None -> every opponent plays hero's stack
        opponent_invested: per-opponent chips already invested this hand
            (start - current stack), aligned with the ranges list; used for
            side-pot layering. None -> assumed equal to hero_invested

    Returns:
        (evs, equity): evs (n_actions,) float32 torch tensor (raw chip EVs,
        same convention as v3: fold EV = -hero_invested); equity — hero's
        actual-hand equity vs the modeled ranges (float). (None, None) on a
        degenerate spot.
    """
    p = dict(V5_DEFAULTS)
    if v5_params:
        p.update({k: v for k, v in v5_params.items() if v is not None})
    iterations = max(1, int(p["iterations"]))
    batch_runouts = max(1, int(p["batch_runouts"]))
    future_bet_sizes = list(p["future_bet_sizes"])
    raise_cap = int(p["raise_cap"])
    allin_spr = float(p["allin_spr"])
    max_root_sizes = max(2, int(p["max_root_sizes"]))
    max_opponents = max(1, int(p["max_opponents"]))
    max_combos = max(2, int(p["max_combos"]))
    max_tree_nodes = int(p["max_tree_nodes"])

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    hero_list = [int(c) for c in hero_cards.tolist()]
    board_list = [int(c) for c in board_cards.tolist()] if len(board_cards) else []
    action_history = action_history or []

    n_raise_bins = n_actions - 3
    evs = torch.zeros(n_actions, dtype=torch.float32)
    evs[0] = -float(hero_invested)

    # ---- opponent selection: primary aggressor first ----
    n_opp_all = len(opponent_range_hand_types)
    opp_positions = list(opponent_positions) if opponent_positions else list(range(n_opp_all))
    primary_idx = 0
    _raise_acts = {"open", "3bet", "bet_postflop"}
    for pos, act in action_history:
        if act in _raise_acts and pos in opp_positions:
            primary_idx = opp_positions.index(pos)
    keep = [primary_idx] + [i for i in range(n_opp_all) if i != primary_idx]
    keep = keep[:max_opponents]

    # ---- ranges ----
    dead_opp = set(hero_list) | set(board_list)
    dead_hero = set(board_list)

    hero_acts = [a for q, a in action_history if q == hero_position]
    if hero_range_hand_types is not None:
        hero_types = list(hero_range_hand_types)
    else:
        hero_types = get_position_range(hero_position, n_players)
        for a in hero_acts:
            hero_types = narrow_range(hero_types, a)
    hero_combos, hero_w = _prepare_range(hero_types, hero_acts, dead_hero,
                                         max_combos, rng)

    # hero's actual combo must be present (EVs are read off it)
    hc_lo, hc_hi = min(hero_list), max(hero_list)
    if hero_combos.shape[0] > 0:
        lo = np.minimum(hero_combos[:, 0], hero_combos[:, 1])
        hi = np.maximum(hero_combos[:, 0], hero_combos[:, 1])
        match = np.nonzero((lo == hc_lo) & (hi == hc_hi))[0]
    else:
        match = []
    if len(match):
        h_star = int(match[0])
    else:
        hero_combos = np.vstack([hero_combos.reshape(-1, 2),
                                 np.array([hero_list], dtype=np.int64)])
        add_w = hero_w.min() if len(hero_w) else 1.0
        hero_w = np.append(hero_w, add_w)
        hero_w = hero_w / hero_w.sum()
        h_star = hero_combos.shape[0] - 1

    ranges = [(hero_combos, hero_w)]
    kept_positions = []
    kept_stacks = []
    kept_invested = []
    for i in keep:
        pos_i = opp_positions[i] if i < len(opp_positions) else 100 + i
        acts_i = [a for q, a in action_history if q == pos_i]
        combos_i, w_i = _prepare_range(opponent_range_hand_types[i], acts_i,
                                       dead_opp, max_combos, rng)
        if combos_i.shape[0] > 0:
            ranges.append((combos_i, w_i))
            kept_positions.append(pos_i)
            kept_stacks.append(
                float(opponent_stacks[i])
                if opponent_stacks is not None and i < len(opponent_stacks)
                else float(stack))
            kept_invested.append(
                float(opponent_invested[i])
                if opponent_invested is not None and i < len(opponent_invested)
                else float(hero_invested))
    n_solver_players = len(ranges)
    if n_solver_players == 1:
        win = float(pot) - float(hero_invested)
        evs[1:] = win
        return evs, 1.0

    positions = [hero_position] + kept_positions
    order = sorted(range(n_solver_players), key=lambda j: positions[j])

    # ---- root actions ----
    stack = float(stack)
    facing_bet = float(facing_bet)
    pot = float(pot)
    call_put = min(facing_bet, stack)

    root_specs = [("fold", 0.0), ("call", call_put)]
    # spec index (into root children) per output action
    FOLD_CI, CALL_CI = 0, 1
    bin_amts = {}   # bin action idx -> raise amount, None -> all-in
    distinct = []
    for b in range(n_raise_bins):
        amt = facing_bet + float(street_raises[b]) * float(effective_pot)
        if amt >= stack - 1e-9:
            bin_amts[b + 2] = None
            continue
        amt = round(amt, 6)
        bin_amts[b + 2] = amt
        if amt not in distinct:
            distinct.append(amt)
    distinct.sort()
    # cap the number of root raise subtrees; remaining bins are interpolated
    # over the raise amount (EV is smooth in size) after the solve
    if len(distinct) > max_root_sizes:
        sel = np.unique(np.round(np.linspace(0, len(distinct) - 1,
                                             max_root_sizes)).astype(int))
        solved_amts = [distinct[i] for i in sel]
    else:
        solved_amts = distinct
    amt_ci = {}
    for amt in solved_amts:
        amt_ci[amt] = len(root_specs)
        root_specs.append(("raise", amt))
    if stack > call_put + 1e-9:
        allin_ci = len(root_specs)
        root_specs.append(("raise", stack))
    else:
        allin_ci = CALL_CI  # facing an all-in: raising == calling all-in

    invested0 = [float(hero_invested)] + kept_invested
    stacks0 = {0: stack}
    for j, s_j in enumerate(kept_stacks):
        stacks0[j + 1] = s_j

    build_args = (n_solver_players, order, pot, facing_bet, stacks0,
                  invested0, int(street), root_specs)
    try:
        nodes, root_id = _build_tree(
            *build_args, future_bet_sizes, raise_cap, allin_spr,
            float(big_blind), max_tree_nodes)
    except _TreeTooBig:
        warnings.warn(
            f"solver v5: tree exceeded {max_tree_nodes} nodes; "
            f"rebuilding with raise_cap=1 and a single bet size", stacklevel=2)
        nodes, root_id = _build_tree(
            *build_args, future_bet_sizes[:1], 1, allin_spr,
            float(big_blind), max_tree_nodes)
    root_children = nodes[root_id].children

    # ---- static pairwise disjointness masks ----
    M = [r[0].shape[0] for r in ranges]
    combos_np = [r[0] for r in ranges]
    weights_np = [r[1].astype(np.float32) for r in ranges]
    disjoint = {}
    for a in range(n_solver_players):
        for b in range(n_solver_players):
            if a != b:
                disjoint[(a, b)] = _disjoint_mask(
                    combos_np[a], combos_np[b]).astype(np.float32)

    # node ids are assigned pre-order: every child id > parent id
    n_nodes = len(nodes)
    P = n_solver_players
    terminal_ids = np.array([nid for nid, n in enumerate(nodes) if n.terminal],
                            dtype=np.int64)
    decision_ids = [nid for nid, n in enumerate(nodes) if not n.terminal]
    T = len(terminal_ids)
    _last_stats.clear()
    _last_stats.update({"nodes": n_nodes, "terminals": T,
                        "decisions": len(decision_ids)})

    term_inv = np.array([nodes[nid].invested for nid in terminal_ids],
                        dtype=np.float32)
    term_alive = np.zeros((T, P), dtype=bool)
    term_showdown = np.zeros(T, dtype=bool)
    for t, nid in enumerate(terminal_ids):
        term_showdown[t] = nodes[nid].showdown
        for j in nodes[nid].alive:
            term_alive[t, j] = True

    # side-pot layer slots: slot s of terminal t = t's s-th pot layer
    # (amount 0 where the terminal has fewer layers). Equal-investment
    # terminals have exactly one layer == the whole pot.
    n_slots = max(len(nodes[nid].layers) for nid in terminal_ids)
    slot_amt = np.zeros((T, n_slots), dtype=np.float32)
    slot_elig = np.zeros((T, n_slots, P), dtype=bool)
    for t, nid in enumerate(terminal_ids):
        for s, (amt, elig) in enumerate(nodes[nid].layers):
            slot_amt[t, s] = amt
            for j in elig:
                slot_elig[t, s, j] = True

    # ---- level-vectorized sweep structure ----
    # Group decision nodes by (depth, player, n_children): each group's
    # forward/backward step is a handful of large fancy-indexed numpy ops
    # instead of one small op per node.
    depth = np.zeros(n_nodes, dtype=np.int64)
    for nid in range(n_nodes):
        for cid in nodes[nid].children:
            depth[cid] = depth[nid] + 1
    group_map = {}
    for nid in decision_ids:
        key = (int(depth[nid]), nodes[nid].player, len(nodes[nid].children))
        group_map.setdefault(key, []).append(nid)
    groups = []
    for (d, pj, A), ids in sorted(group_map.items()):
        I = np.array(ids, dtype=np.int64)
        C = np.array([nodes[nid].children for nid in ids], dtype=np.int64)
        groups.append({
            "depth": d, "p": pj, "A": A, "I": I, "C": C,
            "R": np.zeros((len(ids), M[pj], A), dtype=np.float64),
        })

    remaining = np.array([c for c in range(52) if c not in set(board_list)],
                         dtype=np.int64)
    n_missing = 5 - len(board_list)
    B = 1 if n_missing == 0 else batch_runouts
    board_np = np.array(board_list, dtype=np.int64)

    # ---- pre-sample ALL runouts and evaluate ALL ranks in one chunked
    # pass: evaluate_hands has a large per-call constant cost, so per-
    # iteration calls dominate the solve if done naively ----
    KB = iterations * B if n_missing > 0 else 1
    if n_missing > 0:
        draws_all = np.stack([rng.choice(remaining, size=n_missing,
                                         replace=False) for _ in range(KB)])
        if len(board_list):
            boards_all = np.concatenate(
                [np.broadcast_to(board_np, (KB, len(board_list))), draws_all],
                axis=1)
        else:
            boards_all = draws_all
    else:
        draws_all = None
        boards_all = board_np.reshape(1, 5)

    _EVAL_CHUNK = 50000
    boards_all_t = torch.from_numpy(np.ascontiguousarray(boards_all))
    ranks_all = []
    alive_all = []
    for j in range(n_solver_players):
        cj = combos_np[j]
        Mj = cj.shape[0]
        hands7 = torch.cat([
            torch.from_numpy(cj).unsqueeze(0).expand(KB, Mj, 2),
            boards_all_t.unsqueeze(1).expand(KB, Mj, 5),
        ], dim=2).reshape(-1, 7)
        parts = []
        for s0 in range(0, hands7.shape[0], _EVAL_CHUNK):
            parts.append(evaluate_hands(hands7[s0:s0 + _EVAL_CHUNK]))
        ranks_all.append(torch.cat(parts).reshape(KB, Mj).cpu().numpy())
        if draws_all is not None:
            clash = np.zeros((KB, Mj), dtype=bool)
            for k in range(n_missing):
                d = draws_all[:, k][:, None]
                clash |= (cj[None, :, 0] == d) | (cj[None, :, 1] == d)
            alive_all.append((~clash).astype(np.float32))
        else:
            alive_all.append(np.ones((KB, Mj), dtype=np.float32))

    ev_num = np.zeros(len(root_specs), dtype=np.float64)
    eq_num = 0.0
    denom = 0.0

    for it in range(1, iterations + 1):
        # this iteration's slice of pre-evaluated runouts
        if n_missing > 0:
            s0 = (it - 1) * B
            ranks = [ranks_all[j][s0:s0 + B] for j in range(P)]
            alive_mask = [alive_all[j][s0:s0 + B] for j in range(P)]
        else:
            ranks = [ranks_all[j] for j in range(P)]
            alive_mask = [alive_all[j] for j in range(P)]

        win_cache = {}

        def get_win(a, b):
            """(B, Ma, Mb) float32 win share of a's combos vs b's, masked by
            disjointness."""
            if (a, b) not in win_cache:
                ra = ranks[a][:, :, None]
                rb = ranks[b][:, None, :]
                w = (ra > rb).astype(np.float32)
                w += 0.5 * (ra == rb)
                w *= disjoint[(a, b)][None, :, :]
                win_cache[(a, b)] = w
            return win_cache[(a, b)]

        # ---- forward pass: reaches (level-vectorized) ----
        REACH = [np.empty((n_nodes, B, M[q]), dtype=np.float32)
                 for q in range(P)]
        for q in range(P):
            REACH[q][root_id] = weights_np[q][None, :] * alive_mask[q]

        for g in groups:
            pj, A, I, C = g["p"], g["A"], g["I"], g["C"]
            R = g["R"]  # (G, Mp, A), floored (>= 0) by construction
            s = R.sum(axis=2, keepdims=True)
            sigma = np.where(s > 0, R / np.where(s > 0, s, 1.0), 1.0 / A)
            sigma = sigma.astype(np.float32)
            g["sigma"] = sigma
            rp = REACH[pj][I]  # (G, B, Mp)
            for a in range(A):
                REACH[pj][C[:, a]] = rp * sigma[:, None, :, a]
            for q in range(P):
                if q == pj:
                    continue
                src = REACH[q][I]
                for a in range(A):
                    REACH[q][C[:, a]] = src

        # ---- batched terminal utilities ----
        # All terminals' mass / win-mass vectors via a handful of BLAS calls.
        # Terminals whose reach mass is negligible for every player are
        # pruned this iteration (their utilities are ~0 anyway) — after the
        # first few iterations most deep raise-war lines collapse, cutting
        # the dominant BLAS cost several-fold.
        UTIL = [np.empty((n_nodes, B, M[q]), dtype=np.float32)
                for q in range(P)]
        R_stack = {q: REACH[q][terminal_ids] for q in range(P)}  # (T, B, Mq)

        tmass = np.stack(
            [R_stack[q].sum(axis=(1, 2)).astype(np.float64) for q in range(P)],
            axis=1)  # (T, P)
        tm = np.maximum(tmass, 1e-30)
        prod_all = np.prod(tm, axis=1)
        crit = (prod_all[:, None] / tm).max(axis=1)
        active = np.nonzero(crit > crit.max() * 1e-7)[0]
        Ta = len(active)
        act_ids = terminal_ids[active]

        Rs_act = {q: R_stack[q][active] for q in range(P)}
        MV = {}
        for j in range(P):
            for q in range(P):
                if j != q:
                    Rq2 = Rs_act[q].reshape(Ta * B, M[q])
                    MV[(j, q)] = (Rq2 @ disjoint[(j, q)].T).reshape(Ta, B, M[j])

        showdown_a = term_showdown[active]
        alive_a = term_alive[active]
        inv_a = term_inv[active]
        slot_amt_a = slot_amt[active]
        slot_elig_a = slot_elig[active]

        for j in range(P):
            UTIL[j][terminal_ids] = 0.0
            show_j = showdown_a & alive_a[:, j]
            # F[(q)] = per-combo win-mass vs q where both live at a showdown,
            # plain reach mass otherwise (folded q / fold-out terminals)
            massprod = None
            Fq = {}
            for q in range(P):
                if q == j:
                    continue
                mv = MV[(j, q)]
                massprod = mv if massprod is None else massprod * mv
                F = mv
                both = show_j & alive_a[:, q]
                idx = np.nonzero(both)[0]
                if len(idx):
                    F = mv.copy()
                    W = get_win(j, q)          # (B, Mj, Mq)
                    Rs = Rs_act[q][idx]        # (Ts, B, Mq)
                    for b in range(B):
                        F[idx, b, :] = Rs[:, b, :] @ W[b].T
                Fq[q] = F

            # layered payoff: j wins layer s (vs the players eligible for
            # it) iff j is eligible; non-eligible players only contribute
            # reach mass. Sum over layers == pot; a layer where j is the
            # sole eligible player is an uncalled-bet refund.
            U = (-inv_a[:, j][:, None, None]) * massprod
            for s in range(n_slots):
                sel = np.nonzero((slot_amt_a[:, s] > 0)
                                 & slot_elig_a[:, s, j])[0]
                if not len(sel):
                    continue
                prod = None
                for q in range(P):
                    if q == j:
                        continue
                    cond = slot_elig_a[sel, s, q][:, None, None]
                    fac = np.where(cond, Fq[q][sel], MV[(j, q)][sel])
                    prod = fac if prod is None else prod * fac
                U[sel] += slot_amt_a[sel, s][:, None, None] * prod
            UTIL[j][act_ids] = U

        # ---- backward pass + regret updates (level-vectorized) ----
        for g in reversed(groups):
            pj, A, I, C = g["p"], g["A"], g["I"], g["C"]
            sigma = g["sigma"]
            acc = None
            for a in range(A):
                term = UTIL[pj][C[:, a]] * sigma[:, None, :, a]
                acc = term if acc is None else acc + term
            UTIL[pj][I] = acc
            for q in range(P):
                if q == pj:
                    continue
                accq = None
                for a in range(A):
                    cu = UTIL[q][C[:, a]]
                    accq = cu.copy() if accq is None else accq + cu
                UTIL[q][I] = accq

            # linear CFR regret update with flooring. Combos colliding with
            # the sampled runout carry garbage ranks — mask their deltas so
            # collision noise never leaks into the strategy used on valid
            # runouts.
            R = g["R"]
            am = alive_mask[pj][None]  # (1, B, Mp)
            for a in range(A):
                delta = ((UTIL[pj][C[:, a]] - acc) * am).mean(axis=1)
                R[:, :, a] = np.maximum(R[:, :, a] + it * delta, 0.0)

        # ---- root EV / equity accumulation (linear weighting) ----
        hstar_alive = alive_mask[0][:, h_star].astype(np.float64)
        mass_root = hstar_alive.copy()
        winprod_root = hstar_alive.copy()
        for q in range(1, n_solver_players):
            rq = REACH[q][root_id]  # (B, Mq)
            mv = (disjoint[(0, q)][h_star][None, :] * rq).sum(axis=1)
            mass_root *= mv
            wv = (get_win(0, q)[:, h_star, :] * rq).sum(axis=1)
            winprod_root *= wv

        w_it = float(it)
        denom += w_it * mass_root.sum()
        eq_num += w_it * winprod_root.sum()
        for ci, cid in enumerate(root_children):
            ev_num[ci] += w_it * float(
                (UTIL[0][cid][:, h_star] * alive_mask[0][:, h_star]).sum())

    if denom <= 0:
        warnings.warn("solver v5: zero opponent reach mass at root",
                      stacklevel=2)
        return None, None

    root_evs = ev_num / denom
    # equity denominator: mass_root already includes only pairwise-disjoint
    # opponent mass; winprod over the same mass gives P(win) * mass
    equity = float(min(1.0, max(0.0, eq_num / denom)))

    evs[1] = float(root_evs[CALL_CI])
    allin_ev = float(root_evs[allin_ci])
    solved_evs = [float(root_evs[amt_ci[a]]) for a in solved_amts]
    for b in range(2, n_raise_bins + 2):
        amt = bin_amts.get(b)
        if amt is None:
            evs[b] = allin_ev
        elif amt in amt_ci:
            evs[b] = float(root_evs[amt_ci[amt]])
        else:
            evs[b] = float(np.interp(amt, solved_amts, solved_evs))
    evs[n_raise_bins + 2] = allin_ev
    return evs, equity
