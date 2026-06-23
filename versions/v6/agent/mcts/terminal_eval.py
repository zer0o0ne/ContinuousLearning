"""
Post-hand terminal node evaluation for MCTS trees.

After a hand completes, evaluates all terminal nodes across all MCTS trees
using MC equity with opponent range narrowing via each player's action head.
Must run post-hand because range narrowing uses ALL agents' action heads.

Also exposes `compute_equity_outcome` — the same equity + range-narrowing
machinery applied to the **actual played-out final state** of the hand,
producing the equity-based realized outcome used as the (1−α)-half of the
hybrid value target (see ``versions/v5/PLAN_MCTS_VALUE_REDESIGN.md`` §6.2).
"""

import torch
import torch.nn.functional as F
import numpy as np

from agent.mcts.mcts import _collect_terminals, re_backup_terminals
from agent.mcts.game_state import GameState
from agent.gto_utils.gpu_solver_v2 import (
    get_position_range, expand_range, narrow_range, gpu_equity_v2,
)


def evaluate_all_terminals(hand_record, agents_by_position, device, config=None,
                           value_scales_by_position=None,
                           proxy=None, agent_name_by_pos=None, equity_device=None,
                           combo_probs_cache=None):
    """Evaluate terminal nodes across all MCTS trees from one completed hand.

    For fold terminals: deterministic Q (pot distribution).
    For showdown terminals: MC equity with range narrowing via agents' action heads.
    After evaluation, re-backs up Q through each tree.

    Args:
        hand_record: dict with:
            decisions: list of dicts, each with player_pos, action_idx,
                       mcts_root (MCTSNode), events_at_root (list of event dicts),
                       game_state_at_root (GameState)
            deck: np.array(52,) — the shuffled deck
            hero_hands: dict {pos: [c1, c2]} — actual hands dealt
            num_players: int
            big_blind: float
        agents_by_position: dict {pos: ASI} — each player's agent model
        device: torch device string
        config: optional dict with `n_equity_iters` (default 3000),
            `max_batch` (default 128) and `terminal_eval_prob_floor`
            (default 0.05 — uniform mix on per-combo action probs before
            the Bayesian range narrowing).
        value_scales_by_position: optional dict {pos: float}. When provided,
            terminal Q values are divided by `value_scales_by_position[hero_pos]`
            so they live on the same `mcts_value_scale` axis as the value
            head's outputs at non-terminal leaves; ``re_backup_terminals``
            then propagates them up consistently. When omitted, terminal Q
            is left in raw chips (legacy / debugging behaviour).
        combo_probs_cache: optional pre-computed cache from
            ``_precompute_combo_probs``. When provided, skips the expensive
            per-combo action-head inference (E.1.1 dedup).

    Returns:
        combo_probs_cache — the cache (computed or passed through), so the
        caller can forward it to ``compute_equity_outcome``.
    """
    cfg = config or {}
    n_equity_iters = cfg.get("n_equity_iters", 3000)
    max_batch = cfg.get("max_batch", 128)
    prob_floor = float(cfg.get("terminal_eval_prob_floor", 0.05))
    eq_dev = equity_device if equity_device is not None else device

    decisions = hand_record["decisions"]
    deck = hand_record["deck"]
    hero_hands = hand_record["hero_hands"]
    num_players = hand_record["num_players"]

    if combo_probs_cache is None:
        combo_probs_cache = _precompute_combo_probs(
            decisions, hero_hands, agents_by_position, num_players, device,
            max_batch, proxy=proxy, agent_name_by_pos=agent_name_by_pos)

    narrow_cache = {}

    # Process each tree
    equity_cache = {}
    for dec_idx, decision in enumerate(decisions):
        root = decision["mcts_root"]
        if root is None:
            continue
        hero_pos = decision["player_pos"]
        root_gs = decision["game_state_at_root"]
        root_turn = root_gs.turn
        hero_hand = hero_hands[hero_pos]
        board_cards = _board_at_turn(deck, root_turn)
        initial_stacks = list(root_gs.credits)
        root_pot = float(root_gs.pot)

        if value_scales_by_position is not None:
            scale = float(value_scales_by_position.get(hero_pos, 1.0))
        else:
            scale = 1.0
        if scale <= 0.0:
            scale = 1.0

        terminals = _collect_terminals(root)
        for terminal in terminals:
            path = _path_to_root(terminal)
            gs = root_gs.clone()
            for node in path[1:]:
                if node.action_idx is not None:
                    gs.step(node.action_idx)

            active = [i for i in range(num_players) if gs.players_state[i] >= 0]
            hero_invested = initial_stacks[hero_pos] - gs.credits[hero_pos]
            contributions = [initial_stacks[p] - gs.credits[p]
                             for p in range(num_players)]

            if len(active) <= 1:
                if len(active) == 1:
                    winner = active[0]
                    if winner == hero_pos:
                        q_chips = float(gs.pot) - hero_invested
                    else:
                        q_chips = -hero_invested
                else:
                    q_chips = 0.0
            elif hero_pos not in active:
                q_chips = -hero_invested
            else:
                q_chips = _equity_terminal_chips(
                    hero_pos=hero_pos,
                    hero_hand=hero_hand,
                    board_cards=board_cards,
                    contributions=contributions,
                    active_players=active,
                    num_players=num_players,
                    dec_idx=dec_idx,
                    decisions=decisions,
                    combo_probs_cache=combo_probs_cache,
                    path=path,
                    root_gs=root_gs,
                    n_equity_iters=n_equity_iters,
                    device=eq_dev,
                    prob_floor=prob_floor,
                    narrow_cache=narrow_cache,
                    equity_cache=equity_cache,
                    dead_money=root_pot,
                )

            terminal.Q = q_chips / scale

        re_backup_terminals(root)

    return combo_probs_cache


def compute_equity_outcome(hand_record, agents_by_position, device,
                           ref_credits_by_decision, config=None,
                           proxy=None, agent_name_by_pos=None, equity_device=None,
                           combo_probs_cache=None):
    """Equity-based realized outcome at the played-out final state of the hand.

    Replaces the noisy single-sample chip delta `final_credits[hero] −
    credits_at(t)[hero]` with the **expected** chip delta given the actual
    final board and narrowed opponent ranges. The same equity + Bayesian
    range-narrowing machinery as :func:`evaluate_all_terminals` is reused so
    realised outcome and terminal Qs share a consistent EV definition.

    Behaviour:
      - Fold-terminated hands (single survivor): deterministic, identical to
        what `final_credits[hero] − credits_at(t)[hero]` would yield.
      - Showdown-terminated hands with ≥2 active players at hand-end: the
        outcome becomes ``equity_hero * final_pot − chips_invested_from(t)``
        where `equity_hero` is hero's equity vs narrowed opponent ranges on
        the full 5-card board. `equity_hero` is the **same** for every chain
        step of one root example (same hand, same cards, same opponents) —
        only `chips_invested_from(t)` differs per chain step.
      - Players who folded along the actual sequence are excluded from the
        equity computation regardless of the rooted decision's perspective.

    Args:
        hand_record: dict with keys ``decisions, deck, hero_hands,
            num_players, big_blind, final_pot, final_active_positions,
            credits_pre_distribution``. ``credits_pre_distribution`` is the
            per-player credit vector AFTER all betting but BEFORE the pot
            was redistributed (i.e. ``initial_credits − cumulative_bets``).
            Required to compute ``chips_invested_from(decision)`` without
            relying on `final_credits`, which mixes in winnings.
        agents_by_position: same as in :func:`evaluate_all_terminals` —
            ``{pos: ASI}``. Used for range narrowing only.
        device: torch device string.
        ref_credits_by_decision: list aligned with ``hand_record["decisions"]``.
            Each entry is the credits vector AT that decision (raw chips).
            Used to compute ``chips_invested_from(decision)`` per hero.
        config: optional dict with `n_equity_iters` (default 3000),
            `max_batch` (default 128).

    Returns:
        dict with two keys:
          ``"realized_by_decision"``: ``{dec_idx: equity_realized_chips}`` —
            equity-based realized chip delta from the perspective of
            ``decisions[dec_idx]["player_pos"]`` (root example use).
          ``"equity_by_hero"``: ``{hero_pos: float | None}`` — hero's equity
            vs narrowed opponent ranges. ``None`` for heroes who folded
            before showdown or when the hand ended by fold. Chain steps of
            an example reuse this equity with a per-step ``chips_invested``
            recomputation: same cards, same opponents, only the reference
            credits change. Computed at the actual hand-end board.
          ``"final_pot"``, ``"credits_pre_distribution"``,
            ``"final_active_positions"``: copied through for the caller's
            chain-step bookkeeping (so it doesn't need to re-derive them).

        All chip values are **raw** (un-normalized).
    """
    cfg = config or {}
    n_equity_iters = cfg.get("n_equity_iters", 3000)
    max_batch = cfg.get("max_batch", 128)
    prob_floor = float(cfg.get("terminal_eval_prob_floor", 0.05))
    eq_dev = equity_device if equity_device is not None else device

    decisions = hand_record["decisions"]
    deck = hand_record["deck"]
    hero_hands = hand_record["hero_hands"]
    num_players = hand_record["num_players"]
    final_pot = float(hand_record["final_pot"])
    final_active = list(hand_record["final_active_positions"])
    credits_pre_dist = list(hand_record["credits_pre_distribution"])
    initial_credits = hand_record.get("initial_credits")

    # C.5: per-player total contributions for the whole hand (for side-pot cap).
    # contributions[p] = chips player p put in across all streets INCLUDING blinds.
    # Use start_stacks (pre-blind) when available so blind money is counted;
    # fall back to initial_credits (post-blind) for backward compat.
    start_stacks = hand_record.get("start_stacks")
    if start_stacks is not None:
        contributions = [float(start_stacks[p]) - float(credits_pre_dist[p])
                         for p in range(num_players)]
    elif initial_credits is not None:
        contributions = [float(initial_credits[p]) - float(credits_pre_dist[p])
                         for p in range(num_players)]
    else:
        contributions = None

    def _invested(dec_idx, hero_pos):
        return (float(ref_credits_by_decision[dec_idx][hero_pos])
                - float(credits_pre_dist[hero_pos]))

    realized_by_decision = {}
    equity_by_hero = {}
    # C.5: cache side-pot-capped hero base (excess + equity * hero_share_pot)
    # per hero so each decision only recomputes invested_from_t.
    _hero_base_cache = {}

    base = {
        "final_pot": final_pot,
        "credits_pre_distribution": credits_pre_dist,
        "final_active_positions": final_active,
    }

    # Fold-terminated path: deterministic for every hero.
    if len(final_active) <= 1:
        for dec_idx, decision in enumerate(decisions):
            hero_pos = decision["player_pos"]
            invested = _invested(dec_idx, hero_pos)
            if len(final_active) == 1 and final_active[0] == hero_pos:
                realized_by_decision[dec_idx] = final_pot - invested
            else:
                realized_by_decision[dec_idx] = -invested
            equity_by_hero.setdefault(hero_pos, None)
        return {
            "realized_by_decision": realized_by_decision,
            "equity_by_hero": equity_by_hero,
            **base,
        }

    # Showdown path: equity-based per hero. Cache equity per hero across all
    # decisions made by that hero in this hand (cards & narrowing identical).
    board_cards = torch.tensor(deck[:5].tolist(), dtype=torch.long)

    if combo_probs_cache is None:
        combo_probs_cache = _precompute_combo_probs(
            decisions, hero_hands, agents_by_position, num_players, device,
            max_batch, proxy=proxy, agent_name_by_pos=agent_name_by_pos)

    for dec_idx, decision in enumerate(decisions):
        hero_pos = decision["player_pos"]
        invested = _invested(dec_idx, hero_pos)

        # If this player folded before showdown, outcome is deterministic.
        if hero_pos not in final_active:
            realized_by_decision[dec_idx] = -invested
            equity_by_hero.setdefault(hero_pos, None)
            continue

        if hero_pos not in equity_by_hero or equity_by_hero[hero_pos] is None:
            equity_by_hero[hero_pos] = _hero_equity_at_showdown(
                hero_pos=hero_pos,
                hero_hand=hero_hands[hero_pos],
                board_cards=board_cards,
                final_active=final_active,
                num_players=num_players,
                decisions=decisions,
                combo_probs_cache=combo_probs_cache,
                n_equity_iters=n_equity_iters,
                device=eq_dev,
                prob_floor=prob_floor,
            )

        equity = equity_by_hero[hero_pos]

        # C.5: side-pot-correct chip delta. hero_base = excess + equity *
        # hero_share_pot (same for all decisions by this hero); only
        # invested_from_t varies per decision.
        if contributions is not None and hero_pos not in _hero_base_cache:
            invested_hero_total = float(contributions[hero_pos])
            opp_contribs = [float(contributions[p]) for p in final_active
                            if p != hero_pos]
            max_opp = max(opp_contribs) if opp_contribs else 0.0
            effective_hero = min(invested_hero_total, max_opp)
            excess = invested_hero_total - effective_hero
            hero_share_pot = sum(min(float(c), effective_hero)
                                 for c in contributions)
            _hero_base_cache[hero_pos] = excess + equity * hero_share_pot

        if contributions is not None:
            realized_by_decision[dec_idx] = _hero_base_cache[hero_pos] - invested
        else:
            realized_by_decision[dec_idx] = equity * final_pot - invested

    return {
        "realized_by_decision": realized_by_decision,
        "equity_by_hero": equity_by_hero,
        **base,
    }


def _board_at_turn(deck, turn):
    """Visible board cards at a given turn (0=preflop,1=flop,2=turn,3=river)."""
    if turn == 0:
        return torch.tensor([], dtype=torch.long)
    elif turn == 1:
        return torch.tensor(deck[:3].tolist(), dtype=torch.long)
    elif turn == 2:
        return torch.tensor(deck[:4].tolist(), dtype=torch.long)
    else:
        return torch.tensor(deck[:5].tolist(), dtype=torch.long)


def _capped_showdown_chips(equity, contributions, hero_pos, active_players,
                           dead_money=0.0):
    """Side-pot-correct hero chip delta at a showdown (C.5).

    `contributions[p]` = chips player `p` put in the pot from the reference
    point onward. `dead_money` = pot that existed before the reference point
    (e.g. blinds, or the pot at the MCTS root) — contested by all active
    players at equal equity but not subject to side-pot caps.

    Net = ``excess + equity · (hero_share_pot + dead_money) − invested_hero``.
    """
    invested_hero = float(contributions[hero_pos])
    opp_contribs = [float(contributions[p]) for p in active_players
                    if p != hero_pos]
    max_opp = max(opp_contribs) if opp_contribs else 0.0
    effective_hero = min(invested_hero, max_opp)
    excess = invested_hero - effective_hero
    hero_share_pot = sum(min(float(c), effective_hero) for c in contributions)
    return excess + equity * (hero_share_pot + dead_money) - invested_hero


def _equity_terminal_chips(hero_pos, hero_hand, board_cards, contributions,
                           active_players, num_players, dec_idx, decisions,
                           combo_probs_cache, path, root_gs, n_equity_iters,
                           device, prob_floor=0.05,
                           narrow_cache=None, equity_cache=None,
                           dead_money=0.0):
    """Compute hero's chip-units terminal Q via equity vs narrowed ranges.

    Used inside :func:`evaluate_all_terminals` for showdown terminals. Shared
    with :func:`_hero_equity_at_showdown` via the helpers below — only the
    "what counts as active" and "where to apply path-narrowing" differ. The
    equity is converted to chips with the side-pot-correct
    :func:`_capped_showdown_chips` (C.5).
    """
    hero_cards_t = torch.tensor(hero_hand, dtype=torch.long)
    dead_cards = set(hero_hand)
    dead_cards.update(board_cards.tolist())

    opponent_combos = []
    range_key_parts = []
    for opp_pos in active_players:
        if opp_pos == hero_pos:
            continue
        range_types = get_position_range(opp_pos, num_players)
        narrow_key = (opp_pos, dec_idx, frozenset(dead_cards))
        if narrow_cache is not None and narrow_key in narrow_cache:
            range_types = narrow_cache[narrow_key]
        else:
            range_types = _narrow_by_real_actions(
                range_types, opp_pos, dec_idx, decisions, combo_probs_cache,
                prob_floor=prob_floor, dead_cards=dead_cards)
            if narrow_cache is not None:
                narrow_cache[narrow_key] = range_types
        range_types = _narrow_by_simulated_actions(
            range_types, opp_pos, hero_pos, path, root_gs)
        range_key_parts.append((opp_pos, tuple(range_types)))
        combos = expand_range(range_types, dead_cards)
        if len(combos) == 0:
            combos = expand_range(
                get_position_range(opp_pos, num_players), dead_cards)
        opponent_combos.append(combos)

    if not opponent_combos:
        return _capped_showdown_chips(1.0, contributions, hero_pos,
                                      active_players, dead_money=dead_money)

    eq_key = (hero_pos, frozenset(active_players), tuple(range_key_parts))
    if equity_cache is not None and eq_key in equity_cache:
        equity = equity_cache[eq_key]
    else:
        equity = gpu_equity_v2(
            hero_cards_t, board_cards, opponent_combos,
            n_iters=n_equity_iters, device=device)
        if equity_cache is not None:
            equity_cache[eq_key] = equity
    return _capped_showdown_chips(equity, contributions, hero_pos,
                                  active_players, dead_money=dead_money)


def _hero_equity_at_showdown(hero_pos, hero_hand, board_cards, final_active,
                             num_players, decisions, combo_probs_cache,
                             n_equity_iters, device, prob_floor=0.05):
    """Hero equity at the real hand's showdown. Narrowing uses **all** of the
    hand's decisions (not just those before some `dec_idx`), since by the time
    we compute the realized outcome the whole sequence is known.
    """
    hero_cards_t = torch.tensor(hero_hand, dtype=torch.long)
    dead_cards = set(hero_hand)
    dead_cards.update(board_cards.tolist())

    opponent_combos = []
    for opp_pos in final_active:
        if opp_pos == hero_pos:
            continue
        range_types = get_position_range(opp_pos, num_players)
        # Use the entire decision sequence to narrow this opponent's range.
        # C.7.3: pass hero's dead cards so impossible combos don't distort
        # hand-type aggregation inside the Bayesian narrowing.
        range_types = _narrow_by_real_actions(
            range_types, opp_pos, current_dec_idx=len(decisions),
            decisions=decisions, combo_probs_cache=combo_probs_cache,
            prob_floor=prob_floor, dead_cards=dead_cards)
        combos = expand_range(range_types, dead_cards)
        if len(combos) == 0:
            combos = expand_range(
                get_position_range(opp_pos, num_players), dead_cards)
        opponent_combos.append(combos)

    if not opponent_combos:
        return 1.0  # hero alone is uncontested → wins pot for certainty

    return gpu_equity_v2(
        hero_cards_t, board_cards, opponent_combos,
        n_iters=n_equity_iters, device=device)


def _path_to_root(node):
    """Build path from root to node (inclusive)."""
    path = []
    n = node
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path


def _precompute_combo_probs(decisions, hero_hands, agents_by_position, num_players,
                            device, max_batch, proxy=None, agent_name_by_pos=None):
    """Pre-compute P(action | combo) for each decision point.

    For each decision, runs the acting player's action_head on every combo
    in their position range. Used for Bayesian range narrowing.

    Returns:
        list of dicts, one per decision. Each dict maps the acting player's
        position to (combos_list, per_combo_probs tensor).
    """
    cache = [None] * len(decisions)

    for dec_idx, decision in enumerate(decisions):
        acting_pos = decision["player_pos"]
        # In parallel/actor mode the model lives on the inference server, so
        # `agents_by_position` is empty and we route the action-head forward
        # through `proxy` keyed by the acting position's agent name. In
        # sequential mode `proxy is None` and we use the live ASI directly.
        if proxy is not None:
            agent = None
            if agent_name_by_pos is None or acting_pos not in agent_name_by_pos:
                continue
        else:
            agent = agents_by_position.get(acting_pos)
            if agent is None:
                continue

        events_template = decision["events_at_root"]

        # Get range for this position
        range_types = get_position_range(acting_pos, num_players)
        dead_cards = set()
        # Dead: board cards visible at this decision
        for e in events_template:
            for c in e.get("table", []):
                if isinstance(c, (int, np.integer)) and c >= 0:
                    dead_cards.add(int(c))
        combos = expand_range(range_types, dead_cards)
        if len(combos) == 0:
            continue

        # Batched per-combo inference
        all_probs = []
        for start in range(0, len(combos), max_batch):
            batch_combos = combos[start:start + max_batch]

            if proxy is not None:
                logits = proxy.forward_batch_templated(
                    agent_name_by_pos[acting_pos], events_template,
                    batch_combos.tolist(), heads=("action",))
                probs = F.softmax(logits, dim=-1)
            else:
                batch_events = []
                for c1, c2 in batch_combos.tolist():
                    events_copy = [dict(e) for e in events_template]
                    for e in events_copy:
                        e["hand"] = [c1, c2]
                    batch_events.append(events_copy)
                with torch.no_grad():
                    out = agent.forward_batch(batch_events, skip_memory=True, heads={"action"})
                    logits = out["action_logits"]
                    probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

        per_combo_probs = torch.cat(all_probs, dim=0)  # (n_combos, n_actions)
        cache[dec_idx] = {
            "combos": combos,
            "probs": per_combo_probs,
            "action_idx": decision["action_idx"],
        }

    return cache


_RANK_CHARS = "23456789TJQKA"


def _combo_to_hand_type(c1, c2):
    """Map a (card_id, card_id) combo (0..51) to a canonical hand type string.

    Card encoding: ``card_id = rank * 4 + suit`` (matches
    ``expand_hand_type`` in `gpu_solver_v2.py`). Output format matches
    `HAND_RANKINGS`: pairs ``"AA"``, suited ``"AKs"``, offsuit ``"AKo"``.
    """
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if r1 == r2:
        return _RANK_CHARS[r1] + _RANK_CHARS[r2]
    if r1 < r2:
        r1, r2 = r2, r1
    suffix = "s" if s1 == s2 else "o"
    return _RANK_CHARS[r1] + _RANK_CHARS[r2] + suffix


def _narrow_by_real_actions(range_types, opp_pos, current_dec_idx, decisions,
                            combo_probs_cache, prob_floor=0.05, dead_cards=None):
    """Narrow an opponent's range using their per-combo action probs from
    real decisions, with two robustness improvements over the legacy logic:

    1. **Likelihood floor.** Per-combo ``P(action | combo)`` from
       `action_head` is mixed with uniform over actions before multiplication:
       ``L = (1 − prob_floor) · p + prob_floor / n_actions``. This caps the
       influence of an over-confident or under-trained `action_head` so a
       single sharp predictor on one decision can't crush combo weights to
       ≈0; multi-decision Bayesian products stay well-conditioned. Set
       ``prob_floor=0`` to recover the raw-likelihood behaviour.

    2. **Correct combo → hand-type remap.** The legacy code computed Bayesian
       weights per combo, then ignored *which* combos won mass — it returned
       ``range_types[:n_keep]`` where ``n_keep`` was a function only of the
       ``count`` of high-mass combos. The result was always "keep the top
       X% of the position range" regardless of what the opponent actually
       did, which biased every showdown equity computation toward strong
       opp ranges (premium pairs / top broadways). This rewrite:

         - aggregates the (smoothed, Bayesian) combo weights by canonical
           hand type via `_combo_to_hand_type`;
         - re-ranks `range_types` by aggregated mass (descending);
         - keeps the prefix whose cumulative mass covers ≥95% of the total.

       Information from the action_head likelihoods now actually flows into
       which hand types survive, so the narrowed range reflects what
       opponent's pattern of actions implies (e.g. a multi-check line on a
       dry board narrows toward weak / showdown-value hands, not
       premium pairs).

    Returns the narrowed list of hand-type strings. Falls back to the
    original ``range_types`` when there is nothing to narrow on or when all
    weights collapse to ≈0.
    """
    # Collect all prior decisions by this opponent
    prior_decisions = []
    for i in range(current_dec_idx):
        if decisions[i]["player_pos"] == opp_pos and combo_probs_cache[i] is not None:
            prior_decisions.append(combo_probs_cache[i])

    if not prior_decisions:
        return range_types

    # Build combo weights via Bayesian updates
    # Reference: latest combo set (narrowest, most dead cards)
    latest = prior_decisions[-1]
    ref_combos = latest["combos"]  # (n_ref, 2)
    n_combos = len(ref_combos)
    weights = torch.ones(n_combos)

    # Build reference lookup: (c1, c2) → index
    ref_lookup = {}
    for i in range(n_combos):
        key = (int(ref_combos[i, 0]), int(ref_combos[i, 1]))
        ref_lookup[key] = i

    for cached in prior_decisions:
        action_idx = cached["action_idx"]
        combos = cached["combos"]    # (n_cached, 2)
        probs = cached["probs"]      # (n_cached, n_actions)
        n_actions_local = probs.shape[1]
        # Likelihood floor (see docstring §1)
        if prob_floor > 0.0:
            likelihoods = ((1.0 - prob_floor) * probs[:, action_idx]
                           + prob_floor / float(n_actions_local))
        else:
            likelihoods = probs[:, action_idx]

        # Build cached lookup: (c1, c2) → likelihood
        cached_lookup = {}
        for j in range(len(combos)):
            key = (int(combos[j, 0]), int(combos[j, 1]))
            cached_lookup[key] = likelihoods[j].item()

        # Apply to reference combos (unmatched combos keep weight unchanged)
        for key, idx in ref_lookup.items():
            if key in cached_lookup:
                weights[idx] *= cached_lookup[key]

    # Normalize
    total = weights.sum()
    if total < 1e-8:
        return range_types

    weights /= total

    # ── Correct remap: aggregate per-combo mass into per-hand-type mass ──
    # (see docstring §2)
    type_mass = {}
    for i in range(n_combos):
        c1 = int(ref_combos[i, 0])
        c2 = int(ref_combos[i, 1])
        # C.7.3: skip combos containing observer's dead cards (hero hand +
        # board) — impossible for the opponent, they distort which hand types
        # survive before the final expand_range filters them out.
        if dead_cards and (c1 in dead_cards or c2 in dead_cards):
            continue
        ht = _combo_to_hand_type(c1, c2)
        type_mass[ht] = type_mass.get(ht, 0.0) + float(weights[i])

    # Sort range_types by their aggregated mass (descending). Hand types
    # absent from the reference combo set get mass 0 (typically blocked by
    # dead cards) — they end up at the tail and get pruned by the cumulative
    # threshold.
    ranked = sorted(range_types, key=lambda t: type_mass.get(t, 0.0),
                    reverse=True)
    total_mass = sum(type_mass.get(t, 0.0) for t in ranked)
    if total_mass < 1e-8:
        return range_types

    cum = 0.0
    keep = 0
    for t in ranked:
        cum += type_mass.get(t, 0.0)
        keep += 1
        if cum / total_mass >= 0.95:
            break
    return ranked[:max(1, keep)]


def _narrow_by_simulated_actions(range_types, opp_pos, hero_pos, path, root_gs):
    """Narrow range by simulated actions in the MCTS tree path.

    Uses heuristic narrow_range() based on action category.
    """
    gs = root_gs.clone()
    for node in path[1:]:
        if node.action_idx is None:
            continue
        # Check if this action was by the opponent
        if gs.active_player == opp_pos and not gs.is_terminal:
            action_cat = _action_to_category(node.action_idx, gs)
            if action_cat is not None:
                range_types = narrow_range(range_types, action_cat)
        gs.step(node.action_idx)

    return range_types


def _action_to_category(action_idx, game_state):
    """Map action index to narrow_range category. Returns None for fold."""
    if action_idx == 0:
        return None  # fold — no range narrowing
    elif action_idx == 1:
        if game_state.turn == 0:
            return "call"
        else:
            return "call_postflop"
    else:
        # Raise or all-in
        if game_state.turn == 0:
            return "3bet"
        else:
            return "bet_postflop"
