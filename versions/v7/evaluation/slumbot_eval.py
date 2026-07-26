"""
Slumbot HU NLHE evaluation module.

Plays a list of trained agents heads-up against the public Slumbot HTTP API
(slumbot.com/slumbot/api). Reports raw and baseline-corrected BB/100.

Per-agent options (config.slumbot_eval.agents[*]):
  - type: "model" (default) or "solver". Selects whether to play with a trained
          ASI checkpoint or directly with the GTO solver used for dataset gen.
  - name: display name in logs / result JSON

  Model entries (type == "model"):
    - path: directory containing a checkpoint (file or scenario subdirs)
    - use_opponent_embedding: bool — feed events through OpponentEmbeddingTable
    - use_mcts: bool — pick actions via MCTS.search instead of action-head
                       sampling. If use_opponent_embedding is also true, the
                       opp_emb_table is injected at the root perception call
                       (inner-tree nodes don't re-run perception).
    - action_temperature: float (optional) — overrides checkpoint temperature

  Solver entries (type == "solver"):
    - solver_overrides: dict — any keys override the top-level config.solver
                               (e.g. {"mc_iterations": 20000, "gto_temperature":
                               0.15}). Defaults come from the top-level "solver"
                               and "game" sections, so the solver plays with the
                               same EV machinery used for dataset generation.
    - action_temperature: float (optional) — used only if neither solver_overrides
                                              nor config.solver provide
                                              "gto_temperature".
    Notes:
      * Solver entries always run in the sequential session path even when
        slumbot_eval.n_workers > 1 — the solver doesn't need replicated GPU
        state, and the parallel worker is hard-wired for ASI.

Standalone:
    python -m evaluation.slumbot_eval --config config.json
"""

import argparse
import json
import os
import random
import traceback
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import requests
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from agent.agent import ASI
from agent.mcts.game_state import GameState
from agent.mcts.mcts import MCTS
from agent.perception.opponent_embeddings import OpponentEmbeddingTable
from agent.train_scenarios.generation.generate import (
    _compute_all_action_evs,
    _get_raise_sizes,
    _get_solver,
)
from evaluation.evaluate import (
    _find_best_checkpoint,
    _get_table_display_from_turn,
    _normalize_events_inplace,
    _resolve_checkpoint_path,
)
from utils import get_amp_config


# Slumbot fixed parameters (from sample_api.py)
SLUMBOT_HOST = "slumbot.com"
SLUMBOT_NUM_STREETS = 4
SLUMBOT_SMALL_BLIND = 50
SLUMBOT_BIG_BLIND = 100
SLUMBOT_STACK_SIZE = 20000

_RANK_TO_IDX = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT_TO_IDX = {"c": 0, "d": 1, "h": 2, "s": 3}


# ============================================================================
# Card translation
# ============================================================================

def _card_to_int(card_str):
    """Slumbot card "Ac"/"Td"/etc → int 0..51 with rank*4 + suit encoding."""
    if len(card_str) != 2:
        raise ValueError(f"Invalid card '{card_str}'")
    rank, suit = card_str[0], card_str[1].lower()
    if rank not in _RANK_TO_IDX:
        raise ValueError(f"Invalid rank in '{card_str}'")
    if suit not in _SUIT_TO_IDX:
        raise ValueError(f"Invalid suit in '{card_str}'")
    return _RANK_TO_IDX[rank] * 4 + _SUIT_TO_IDX[suit]


def _board_to_ints(board_strs):
    """5-card board (-1 padded). board_strs may be empty / 3 / 4 / 5 long."""
    ints = [_card_to_int(c) for c in board_strs]
    while len(ints) < 5:
        ints.append(-1)
    return ints


# ============================================================================
# Action <-> discrete index translation
# ============================================================================

def _token_to_action_idx(state_pre, token, raise_sizes_for_street, n_raise_bins):
    """Encode a Slumbot action token into our discrete action index.

    state_pre: dict with 'bets', 'credits', 'pot', 'high_bet', 'active_pos'
               in Slumbot chip units (Slumbot frame: pos 0=BB, pos 1=SB).
    """
    if token == "f":
        return 0
    if token == "k" or token == "c":
        return 1
    assert token.startswith("b"), f"unknown token {token!r}"
    new_total = int(token[1:])
    pos = state_pre["active_pos"]
    bets_pos = state_pre["bets"][pos]
    credits_pos = state_pre["credits"][pos]
    added = new_total - bets_pos
    if added >= credits_pos - 1e-9:
        return n_raise_bins + 2  # all-in
    call_amount = state_pre["high_bet"] - bets_pos
    effective_pot = state_pre["pot"] - bets_pos
    raise_pct = max(0.0, (added - call_amount) / max(effective_pot, 1.0))
    diffs = [abs(raise_pct - rs) for rs in raise_sizes_for_street]
    return 2 + int(np.argmin(diffs))


def _action_idx_to_incr(state_pre, action_idx, raise_sizes_for_street,
                        n_raise_bins, hero_slumbot_pos, clamp_counters):
    """Translate our discrete action_idx into a legal Slumbot 'incr' string.

    Applies fallback clamping ("аккуратно"):
      - fold without facing a bet → check
      - raise below min legal → bump to min legal
      - raise above stack → cap at all-in
      - raise that doesn't exceed call → emit call/check
    Each clamp increments the corresponding counter.
    """
    bets = state_pre["bets"]
    credits = state_pre["credits"]
    high_bet = state_pre["high_bet"]
    last_bet_size = state_pre["last_bet_size"]
    bets_hero = bets[hero_slumbot_pos]
    cap = bets_hero + credits[hero_slumbot_pos]
    facing_bet = high_bet > bets_hero + 1e-9

    if action_idx == 0:
        if facing_bet:
            return "f"
        clamp_counters["fold_to_check"] += 1
        return "k"

    if action_idx == 1:
        return "c" if facing_bet else "k"

    if action_idx == n_raise_bins + 2:
        return f"b{int(round(cap))}"

    raise_pct = raise_sizes_for_street[action_idx - 2]
    call_amount = high_bet - bets_hero
    effective_pot = state_pre["pot"] - bets_hero
    added = round(call_amount + raise_pct * effective_pot)
    new_total = bets_hero + added

    # Slumbot min raise rule (sample_api.py:188-204)
    min_legal_total = high_bet + max(SLUMBOT_BIG_BLIND, last_bet_size)
    if min_legal_total > cap:
        min_legal_total = cap

    # If the computed "raise" is actually a call/check (added <= call_amount),
    # gracefully degrade to 'c'/'k' rather than emit an illegal bet.
    if added <= call_amount + 1e-9:
        clamp_counters["raise_to_call"] += 1
        return "c" if facing_bet else "k"

    if new_total < min_legal_total:
        call_total = high_bet
        dist_to_call = new_total - call_total
        dist_to_min = min_legal_total - new_total
        if dist_to_call <= dist_to_min:
            clamp_counters["raise_rounded_to_call"] += 1
            return "c" if facing_bet else "k"
        clamp_counters["raise_bumped"] += 1
        new_total = min_legal_total
    if new_total > cap:
        clamp_counters["raise_to_allin"] += 1
        new_total = cap

    return f"b{int(round(new_total))}"


# ============================================================================
# Solver helpers (used by type == "solver" agent entries)
# ============================================================================

def _action_idx_to_history_act_type(action_idx, n_raise_bins, street,
                                    street_raise_count):
    """Map an action_idx applied at the given street to generate.py's act_type.

    Mirrors the classification in generate.py (B.5.6) so the solver's
    opponent-range narrowing matches what the dataset generator does. Folds
    return None and are not appended to action_history (narrow_range has no
    "fold" key).

    B.5.6: preflop raises are classified by the number of raises ALREADY made
    this street, not by all-in-vs-sized. Any raise (sized bin or all-in) is an
    "open" when it is the first raise of the street and a "3bet" when it
    raises over a raise.
    """
    if action_idx == 0:
        return None
    if action_idx == 1:
        return "call" if street == 0 else "call_postflop"
    # Any raise (sized bin or all-in).
    if street == 0:
        return "3bet" if street_raise_count >= 1 else "open"
    return "bet_postflop"


def _make_solver_table_stub(hero_user_pos, hole_cards_int, board_ints, state,
                            raise_sizes, big_blind_internal, small_blind_internal,
                            chip_scale, num_players=2):
    """Build a Table-like SimpleNamespace from current Slumbot state.

    Only carries the attributes that the solver actually reads
    (deck / credits / bets / players_state / pot / high_bet / turn / ...).
    Chip values are scaled into training units (divided by chip_scale).
    Hero hole cards go to deck[5+2*hero_user_pos : 7+2*hero_user_pos]; the
    board occupies deck[:5]. Slumbot frame pos 0 (BB) maps to user pos 1, and
    pos 1 (SB) maps to user pos 0 — consistent with the rest of the eval.
    """
    inv = 1.0 / chip_scale

    placed = set()
    for c in hole_cards_int:
        placed.add(int(c))
    for c in board_ints:
        if c >= 0:
            placed.add(int(c))
    remaining = [c for c in range(52) if c not in placed]
    random.shuffle(remaining)

    deck = [0] * 52
    for i in range(5):
        if i < len(board_ints) and board_ints[i] >= 0:
            deck[i] = int(board_ints[i])
        else:
            deck[i] = remaining.pop()
    deck[5 + 2 * hero_user_pos] = int(hole_cards_int[0])
    deck[5 + 2 * hero_user_pos + 1] = int(hole_cards_int[1])
    for i in range(5, 5 + 2 * num_players):
        if i == 5 + 2 * hero_user_pos or i == 5 + 2 * hero_user_pos + 1:
            continue
        deck[i] = remaining.pop()
    for i in range(5 + 2 * num_players, 52):
        if remaining:
            deck[i] = remaining.pop()

    # Use float64 to match env.table.Table's default dtype — np.float32 elements
    # propagate through the solver and break the torch tensor assignment in
    # _compute_all_action_evs on torch>=2.5 (only np.float32 is rejected).
    bets_user = np.zeros(num_players, dtype=np.float64)
    credits_user = [0.0] * num_players
    players_state_user = np.ones(num_players, dtype=np.float64)
    for slumbot_pos in range(2):
        u = 1 - slumbot_pos
        bets_user[u] = float(state["bets"][slumbot_pos]) * inv
        credits_user[u] = float(state["credits"][slumbot_pos]) * inv
        ss = int(state["players_state"][slumbot_pos])
        if ss < 0:
            players_state_user[u] = -1.0
        elif ss == 2:
            players_state_user[u] = 2.0
        else:
            players_state_user[u] = 1.0

    start_credits_scaled = float(SLUMBOT_STACK_SIZE) * inv

    # active_player: Slumbot pos → user pos (0↔1 swap)
    active_player_user = 1 - int(state["active_pos"])

    # several_all_in: True when ≥2 players are all-in (state 2)
    several_all_in = int(np.sum(players_state_user == 2)) >= 2

    return SimpleNamespace(
        deck=np.array(deck, dtype=np.int64),
        num_players=num_players,
        start_credits=np.full(num_players, start_credits_scaled, dtype=np.float64),
        credits=credits_user,
        bets=bets_user,
        players_state=players_state_user,
        pot=float(state["pot"]) * inv,
        high_bet=float(state["high_bet"]) * inv,
        turn=int(state["turn"]),
        raise_sizes=raise_sizes,
        n_raise_bins=len(raise_sizes[0]),
        big_blind=float(big_blind_internal),
        small_blind=float(small_blind_internal),
        active_player=active_player_user,
        several_all_in=several_all_in,
        last_raise_size=float(state.get("last_bet_size", 0)) * inv or float(big_blind_internal),
        # C.7.5: level of the last FULL raise, mirroring env/table.py semantics:
        # preflop before any raise it equals the big blind (= high_bet), each
        # full raise moves it to the new high_bet, and a street change resets
        # it to 0.0 (= high_bet after bets reset). The harness cannot observe
        # short-all-in raises separately, so every raise is treated as full —
        # `high_bet` scaled covers all three cases. Without this attribute,
        # GameState.from_table defaults it to big_blind, which would set
        # short_allin_restricted and strip raises from the legal mask whenever
        # hero's street bet >= 1BB while facing a raise.
        _last_full_raise_level=float(state["high_bet"]) * inv,
    )


def _solver_choose_action(bundle, state, hero_user_pos, hole_cards_int,
                          board_ints, action_history, raise_sizes, n_raise_bins,
                          n_actions, chip_scale, big_blind_internal,
                          small_blind_internal):
    """Run the GTO solver on the current Slumbot state and sample an action.

    Sampling mirrors generate.py exactly:
        normalizer = max(pot + facing_bet, big_blind) * temperature
        probs = softmax(all_evs / normalizer)
    Legal-action mask is applied before the softmax so the sampled index can
    be played as-is (with the usual clamp in _action_idx_to_incr for edge
    cases like raise-degraded-to-call).
    """
    scfg = bundle["solver_cfg"]
    table_stub = _make_solver_table_stub(
        hero_user_pos, hole_cards_int, board_ints, state,
        raise_sizes, big_blind_internal, small_blind_internal, chip_scale,
        num_players=2,
    )

    evs, meta = _compute_all_action_evs(
        table_stub, hero_user_pos, action_history, n_actions,
        solver_name=scfg.get("type", "v3"),
        device=scfg.get("device", "cpu"),
        mc_iters=int(scfg.get("mc_iterations", 5000)),
        eqr_enabled=bool(scfg.get("eqr_enabled", True)),
        combo_response_iters=int(scfg.get("combo_response_iters", 30)),
        reraise_threshold=float(scfg.get("reraise_threshold", 0.72)),
        weighted_sampling=bool(scfg.get("weighted_sampling", True)),
        threshold_smoothing=scfg.get("threshold_smoothing"),
        polarized_reraise=scfg.get("polarized_reraise"),
        v5_params=scfg.get("v5"),
    )
    if evs is None or meta is None:
        # Solver failure (rare — usually MC OOM/timeout). Fall back to
        # check/call: matches the dataset-gen behaviour of skipping the
        # sample, but here we must still return something playable.
        return 1

    gs = _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins,
                           chip_scale, big_blind_internal=big_blind_internal)
    legal_mask = torch.tensor(
        gs.get_legal_action_mask(n_actions), dtype=torch.bool,
    )

    temperature = max(float(bundle["temperature"]), 1e-3)
    normalizer = max(float(meta["pot"]) + float(meta["facing_bet"]),
                     float(big_blind_internal)) * temperature
    evs_masked = evs.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(evs_masked / max(normalizer, 1e-6), dim=0)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        # All legal actions masked or numerical blow-up — pick first legal.
        legal_idx = torch.nonzero(legal_mask, as_tuple=False).flatten()
        return int(legal_idx[0].item()) if len(legal_idx) else 1
    return int(torch.multinomial(probs, 1).item())


def _v4_init_bayesian_state(hero_user_pos, num_players=2):
    """Initialize per-hand Bayesian opponent range for v4 solver."""
    _, _, v4_mods = _get_solver("v4")
    get_pos_range = v4_mods["get_position_range"]
    expand_range = v4_mods["expand_range"]

    opp_state = {}
    for pos in range(num_players):
        if pos == hero_user_pos:
            continue
        ht = get_pos_range(pos, num_players)
        combos = expand_range(ht, set())
        weights = torch.ones(combos.shape[0], dtype=torch.float32)
        weights = weights / weights.sum()
        opp_state[pos] = {"hand_types": ht, "combos": combos, "weights": weights}
    return opp_state


def _v4_update_opponent_range(bayesian_state, opp_user_pos, observed_action_idx,
                              board_ints, hero_cards_int, table_stub,
                              action_history, n_actions, scfg):
    """Bayesian update of opponent range after observing their action."""
    if opp_user_pos not in bayesian_state:
        return

    _, _, v4_mods = _get_solver("v4")
    compute_marg = v4_mods["compute_marginalized_action_probs"]
    bayes_update = v4_mods["bayesian_range_update"]
    filter_dead = v4_mods["filter_dead_combos"]
    expand_range = v4_mods["expand_range"]

    bs = bayesian_state[opp_user_pos]
    device = scfg.get("device", "cpu")

    board_ids = [int(b) for b in board_ints if b >= 0]
    dead = set(board_ids) | set(int(c) for c in hero_cards_int)

    valid_combos, valid_weights = filter_dead(bs["combos"], bs["weights"], dead)
    if valid_combos.shape[0] == 0:
        return

    board_t = torch.tensor(board_ids, dtype=torch.long) if board_ids else torch.tensor([], dtype=torch.long)

    opp_ranges_ht = []
    opp_positions = []
    for pos in range(table_stub.num_players):
        if pos == opp_user_pos:
            continue
        opp_ranges_ht.append(
            bayesian_state[pos]["hand_types"] if pos in bayesian_state
            else [])
        opp_positions.append(pos)

    hero_invested = float(table_stub.start_credits[opp_user_pos]) - float(table_stub.credits[opp_user_pos])
    facing_bet = max(0.0, float(table_stub.high_bet) - float(table_stub.bets[opp_user_pos]))
    stack = float(table_stub.credits[opp_user_pos])
    pot = float(table_stub.pot)
    hero_bets = float(table_stub.bets[opp_user_pos])
    effective_pot = max(pot - hero_bets, 1e-6)
    street_raises = table_stub.raise_sizes[table_stub.turn]
    temperature = max(float(scfg.get("gto_temperature", 0.2)), 1e-3)
    big_blind = float(table_stub.big_blind)

    try:
        _, per_combo_probs = compute_marg(
            valid_combos, valid_weights, board_t, opp_ranges_ht,
            pot, facing_bet, stack, hero_invested,
            n_actions, street_raises, effective_pot,
            temperature, big_blind,
            n_iters=int(scfg.get("marginal_mc_iters", 3000)),
            device=device,
            hero_position=opp_user_pos,
            street=table_stub.turn,
            n_players=table_stub.num_players,
            eqr_enabled=bool(scfg.get("eqr_enabled", True)),
            combo_response_iters=int(scfg.get("marginal_response_iters", 30)),
            reraise_threshold=float(scfg.get("reraise_threshold", 0.72)),
            weighted_sampling=bool(scfg.get("weighted_sampling", True)),
            action_history=action_history,
            opponent_positions=opp_positions,
            dynamic_reraise=True,
        )

        updated = bayes_update(
            valid_weights, per_combo_probs, observed_action_idx,
            combos=valid_combos, dead_cards=dead,
        )
        bs["weights"] = updated
        bs["combos"] = valid_combos
    except Exception:
        pass


def _v4_solver_choose_action(bundle, state, hero_user_pos, hole_cards_int,
                             board_ints, action_history, raise_sizes, n_raise_bins,
                             n_actions, chip_scale, big_blind_internal,
                             small_blind_internal):
    """v4 solver: uses compute_ev_batched with Bayesian-narrowed opponent range."""
    from agent.gto_utils.gpu_solver_v4 import compute_ev_batched
    from agent.gto_utils.gpu_solver_v3 import get_position_range, expand_range

    scfg = bundle["solver_cfg"]
    device = scfg.get("device", "cpu")
    inv = 1.0 / chip_scale

    hero_t = torch.tensor(hole_cards_int, dtype=torch.long).unsqueeze(0)
    board_ids = [int(b) for b in board_ints if b >= 0]
    board_t = torch.tensor(board_ids, dtype=torch.long) if board_ids else torch.tensor([], dtype=torch.long)

    hero_invested = float(state["bets"][1 - hero_user_pos]) * inv
    # Hero's bets in user frame
    hero_bets_slumbot = state["bets"][1 - hero_user_pos]
    hero_credits_slumbot = state["credits"][1 - hero_user_pos]
    pot = float(state["pot"]) * inv
    high_bet = float(state["high_bet"]) * inv
    hero_bets_u = float(hero_bets_slumbot) * inv
    facing_bet = max(0.0, high_bet - hero_bets_u)
    stack = float(hero_credits_slumbot) * inv
    start_credits = float(SLUMBOT_STACK_SIZE) * inv
    hero_invested = start_credits - stack

    dead = set(hole_cards_int) | set(board_ids)
    bayesian = bundle.get("_v4_bayesian")
    opp_user_pos = 1 - hero_user_pos

    if bayesian and opp_user_pos in bayesian:
        bs = bayesian[opp_user_pos]
        from agent.gto_utils.gpu_solver_v4 import filter_dead_combos
        valid_combos, valid_weights = filter_dead_combos(bs["combos"], bs["weights"], dead)
        if valid_combos.shape[0] > 0:
            opp_range_combos = [valid_combos]
        else:
            opp_ht = get_position_range(opp_user_pos, 2)
            opp_range_combos = [expand_range(opp_ht, dead)]
    else:
        opp_ht = get_position_range(opp_user_pos, 2)
        opp_range_combos = [expand_range(opp_ht, dead)]

    opp_range_hand_types = [get_position_range(opp_user_pos, 2)]
    opp_positions = [opp_user_pos]

    street = int(state["turn"])
    street_raises = raise_sizes[street]
    n_raise_bins_actual = len(street_raises)
    effective_pot = max(pot - hero_bets_u, 1e-6)

    evs = torch.zeros(n_actions, dtype=torch.float)
    evs[0] = -hero_invested

    ev_kwargs = dict(
        n_iters=int(scfg.get("mc_iterations", 5000)),
        device=device,
        hero_position=hero_user_pos,
        street=street,
        n_players=2,
        eqr_enabled=bool(scfg.get("eqr_enabled", True)),
        combo_response_iters=int(scfg.get("combo_response_iters", 30)),
        reraise_threshold=float(scfg.get("reraise_threshold", 0.72)),
        weighted_sampling=bool(scfg.get("weighted_sampling", True)),
        action_history=action_history,
        opponent_positions=opp_positions,
        dynamic_reraise=True,
    )

    def _raise_to_solver_frac(raise_pct):
        return raise_pct * effective_pot / max(pot + facing_bet, 1e-6)

    try:
        fold_evs, call_evs, _ = compute_ev_batched(
            hero_t, board_t, opp_range_hand_types,
            pot, facing_bet, stack, hero_invested,
            raise_frac=1.0, **ev_kwargs,
        )
        evs[1] = call_evs[0].item()
    except Exception:
        return 1

    allin_frac = stack / max(pot + facing_bet, 1e-6)
    try:
        _, _, allin_evs = compute_ev_batched(
            hero_t, board_t, opp_range_hand_types,
            pot, facing_bet, stack, hero_invested,
            raise_frac=allin_frac, **ev_kwargs,
        )
        evs[n_raise_bins + 2] = allin_evs[0].item()
    except Exception:
        evs[n_raise_bins + 2] = evs[0]

    for b in range(n_raise_bins_actual):
        raise_pct = street_raises[b]
        actual_bet = facing_bet + raise_pct * effective_pot
        if actual_bet >= stack:
            evs[b + 2] = evs[n_raise_bins + 2]
        else:
            try:
                solver_frac = _raise_to_solver_frac(raise_pct)
                _, _, raise_evs = compute_ev_batched(
                    hero_t, board_t, opp_range_hand_types,
                    pot, facing_bet, stack, hero_invested,
                    raise_frac=solver_frac, **ev_kwargs,
                )
                evs[b + 2] = raise_evs[0].item()
            except Exception:
                evs[b + 2] = evs[n_raise_bins + 2]

    gs = _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins,
                           chip_scale, big_blind_internal=big_blind_internal)
    legal_mask = torch.tensor(gs.get_legal_action_mask(n_actions), dtype=torch.bool)

    temperature = max(float(bundle["temperature"]), 1e-3)
    normalizer = max(pot + facing_bet, float(big_blind_internal)) * temperature
    evs_masked = evs.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(evs_masked / max(normalizer, 1e-6), dim=0)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        legal_idx = torch.nonzero(legal_mask, as_tuple=False).flatten()
        return int(legal_idx[0].item()) if len(legal_idx) else 1
    return int(torch.multinomial(probs, 1).item())


# ============================================================================
# Action-string replay
# ============================================================================

def _initial_state(hole_cards_int, board_ints):
    """Initial Slumbot-frame state before any action (blinds posted).

    Slumbot frame: pos 0 = BB, pos 1 = SB. SB acts first preflop.
    """
    bets = [SLUMBOT_BIG_BLIND, SLUMBOT_SMALL_BLIND]
    credits = [SLUMBOT_STACK_SIZE - SLUMBOT_BIG_BLIND,
               SLUMBOT_STACK_SIZE - SLUMBOT_SMALL_BLIND]
    return {
        "pot": SLUMBOT_BIG_BLIND + SLUMBOT_SMALL_BLIND,
        "bets": bets,
        "credits": credits,
        "high_bet": SLUMBOT_BIG_BLIND,
        "last_bet_size": SLUMBOT_BIG_BLIND - SLUMBOT_SMALL_BLIND,
        "turn": 0,
        "active_pos": 1,            # Slumbot frame — SB first preflop
        "players_state": [1, 1],    # both active and "moving"
        "is_terminal": False,
        "hole_cards": hole_cards_int,
        "board": board_ints,
    }


def _make_snapshot(state, n_actions, action_idx, hero_slumbot_pos):
    """Build a snapshot dict in evaluate.py-compatible format.

    Stores active_pos in USER frame (1 - slumbot_pos). Chip values stay in
    Slumbot units; events are scaled at build time.
    """
    if action_idx is None:
        action_tensor = None
    else:
        action_tensor = torch.zeros(n_actions, dtype=torch.float32)
        action_tensor[action_idx] = 1.0
    return {
        "pot": float(state["pot"]),
        "bets": np.array(state["bets"], dtype=np.float32),
        "credits": list(state["credits"]),
        "turn": int(state["turn"]),
        "active_pos": 1 - state["active_pos"],
        "action": action_tensor,
    }


def _apply_token(state, token, raise_sizes, n_raise_bins):
    """Apply a single Slumbot token (k/c/f/b<N>) to state.
    Returns the action_idx that this token represents.
    """
    pos = state["active_pos"]
    raise_sizes_for_street = raise_sizes[state["turn"]]

    # Encode action_idx BEFORE mutating state (state_pre semantics)
    action_idx = _token_to_action_idx(
        state, token, raise_sizes_for_street, n_raise_bins
    )

    if token == "f":
        state["players_state"][pos] = -1
        state["is_terminal"] = True
        return action_idx

    if token == "k":
        # check (no chip movement)
        pass
    elif token == "c":
        # call: match high_bet
        amount = state["high_bet"] - state["bets"][pos]
        amount = min(amount, state["credits"][pos])
        state["bets"][pos] += amount
        state["credits"][pos] -= amount
        state["pot"] += amount
        if state["credits"][pos] <= 0:
            state["players_state"][pos] = 2  # all-in
        state["last_bet_size"] = 0
    elif token.startswith("b"):
        new_total = int(token[1:])
        added = new_total - state["bets"][pos]
        added = min(added, state["credits"][pos])
        state["bets"][pos] += added
        state["credits"][pos] -= added
        state["pot"] += added
        new_last_bet_size = state["bets"][pos] - state["high_bet"]
        if new_last_bet_size > 0:
            state["last_bet_size"] = new_last_bet_size
        state["high_bet"] = max(state["high_bet"], state["bets"][pos])
        if state["credits"][pos] <= 0:
            state["players_state"][pos] = 2
    else:
        raise ValueError(f"Unknown token: {token!r}")

    return action_idx


def _advance_after_action(state, token):
    """After applying a non-fold action, decide whether the street ends or
    play continues. Mirrors sample_api.py ParseAction transitions."""
    pos = state["active_pos"]
    other = 1 - pos
    # Did this action close the street?
    if token == "k":
        # check by SB preflop never closes (BB still to act); BB check
        # postflop closes after both checked. We track via "both checked".
        # Simpler: check ends street if the other player has already acted
        # this street (i.e., the other was last to act and it was a check).
        # State: we infer using last_bet_size (0 means no outstanding bet).
        # If high_bet equals current bets, opponent has matched (or both 0):
        opp_done = (state["bets"][other] == state["high_bet"]
                    and (state["turn"] > 0 or state["bets"][other] == SLUMBOT_BIG_BLIND))
        # Special: postflop opening check from BB (first to act) does NOT
        # close — opponent hasn't acted yet. We detect "first action of street"
        # via a flag passed in via state['_first_in_street'].
        if state.pop("_first_in_street", False):
            opp_done = False
        if opp_done:
            _street_advance_or_terminal(state)
            return
        # otherwise continue: pass turn
        state["active_pos"] = other
        return

    if token == "c":
        # Call closes the street UNLESS preflop BB option (SB limps then BB
        # still has option). sample_api: after a call, next is opponent's
        # turn but `check_or_call_ends_street=True` — meaning if the OPPONENT
        # next acts and check/calls, the street is closed. Practically, a
        # call always closes the street EXCEPT preflop SB-limp (SB calls BB
        # for 50→100; BB still has option). We handle via _first_in_street.
        if state.pop("_first_in_street", False):
            # SB limp preflop — BB still gets option
            state["active_pos"] = other
            return
        _street_advance_or_terminal(state)
        return

    # bet/raise: opponent must respond
    state.pop("_first_in_street", None)
    state["active_pos"] = other
    state["players_state"][pos] = 0  # already acted (waiting for response)
    state["players_state"][other] = 1
    return


def _street_advance_or_terminal(state):
    """Move to next street or mark terminal at showdown."""
    if state["turn"] >= SLUMBOT_NUM_STREETS - 1:
        state["is_terminal"] = True
        return
    state["turn"] += 1
    state["bets"] = [0, 0]
    state["high_bet"] = 0
    state["last_bet_size"] = 0
    # Postflop: BB (Slumbot pos 0) acts first
    state["active_pos"] = 0
    state["players_state"] = [1, 1]
    state["_first_in_street"] = True


def _replay_action_string(action_str, hole_cards_int, board_ints,
                          client_pos, raise_sizes, n_raise_bins, n_actions,
                          hero_action_indices):
    """Re-parse the entire Slumbot action string from scratch, building
    SlumbotState + snapshot list.

    The snapshot pattern matches the generate.py/evaluate.py training
    distribution: initial snap (action=None), then for EACH action a
    (pre-decision action=None, post-action) pair — so sequences always open
    with TWO action=None snapshots (initial + first pre-decision), exactly
    like generate.py:723-731/813-822 and evaluate.py:565-572/798-806.

    Also returns `action_history`: list of (user_pos, act_type) tuples in the
    same shape generate.py produces for the GTO solver. Folds are excluded
    (narrow_range has no "fold" key). Solver-mode decision making consumes
    this; model mode ignores it.

    Returns: (state, snapshots, hero_moves_seen, action_history)
    """
    state = _initial_state(hole_cards_int, board_ints)
    state["_first_in_street"] = True
    # Initial snapshot (no action yet) — mirrors generate.py:723-731
    snapshots = [_make_snapshot(state, n_actions, None, client_pos)]
    hero_moves_seen = 0
    action_history = []

    # B.5.6: number of raises made so far on the CURRENT street, used to
    # classify a preflop raise as an open (first raise) vs a 3bet+ (raise over
    # a raise). Reset whenever the street advances.
    street_raise_count = 0
    cur_street = int(state["turn"])

    if not action_str:
        return state, snapshots, hero_moves_seen, action_history

    i = 0
    sz = len(action_str)
    while i < sz:
        c = action_str[i]
        if c == "/":
            # Tolerate stray slashes (e.g., "b20000c///" — all-in runout).
            i += 1
            continue

        if c in ("k", "c", "f"):
            token = c
            i += 1
        elif c == "b":
            j = i + 1
            while j < sz and action_str[j].isdigit():
                j += 1
            token = action_str[i:j]
            i = j
        else:
            raise ValueError(f"Unknown char {c!r} at offset {i} in {action_str!r}")

        # Pre-decision snap (action=None) — one for EVERY action, including
        # the hand's first token. Training sequences open with TWO action=None
        # snapshots (initial + first pre-decision): generate.py:723-731 posts
        # the initial snap and :813-822 a decision snap before every action.
        snapshots.append(_make_snapshot(state, n_actions, None, client_pos))

        # Capture the acting position and street BEFORE the token mutates state,
        # so action_history is built in the same frame generate.py uses.
        acting_pos_user = 1 - state["active_pos"]
        street_pre = int(state["turn"])

        # B.5.6: reset the per-street raise counter when the street advances.
        if street_pre != cur_street:
            cur_street = street_pre
            street_raise_count = 0

        is_hero = (state["active_pos"] == client_pos)
        if is_hero and hero_moves_seen < len(hero_action_indices):
            action_idx = hero_action_indices[hero_moves_seen]
            hero_moves_seen += 1
            _apply_token(state, token, raise_sizes, n_raise_bins)
        else:
            action_idx = _apply_token(state, token, raise_sizes, n_raise_bins)
            if is_hero:
                hero_moves_seen += 1

        act_type = _action_idx_to_history_act_type(
            action_idx, n_raise_bins, street_pre, street_raise_count,
        )
        if act_type is not None:
            action_history.append((acting_pos_user, act_type))

        # Count this raise toward the current street's raise tally (B.5.6).
        if action_idx >= 2:
            street_raise_count += 1

        if state["is_terminal"]:
            snapshots.append(_make_snapshot(state, n_actions, action_idx, client_pos))
            break

        _advance_after_action(state, token)
        snapshots.append(_make_snapshot(state, n_actions, action_idx, client_pos))

        if state["is_terminal"]:
            break

    return state, snapshots, hero_moves_seen, action_history


# ============================================================================
# Event building
# ============================================================================

def _build_events(snapshots, hole_cards_int, board_ints, hero_user_pos,
                  client_pos, num_players, big_blind_internal,
                  small_blind_internal, chip_scale, n_actions,
                  hero_id="hero", opp_id="slumbot"):
    """Convert snapshots → list of evaluate.py-compatible event dicts.

    Chip values are divided by chip_scale to land in training-time units.
    """
    events = []
    inv_scale = 1.0 / chip_scale

    for snap in snapshots:
        # Board masked to the snapshot's street — matches the training
        # convention (_get_table_display_from_turn in generate.py/evaluate.py):
        # preflop events show [-1]*5, flop events 3 cards, etc. Stamping the
        # CURRENT board into past events would leak future cards.
        table = _get_table_display_from_turn(board_ints, snap["turn"])
        action = snap["action"]
        if action is None:
            action = torch.zeros(n_actions, dtype=torch.float32)
        # active_pos in snap is already in user frame (1 - slumbot_pos)
        acting_pos_user = snap["active_pos"]
        opponent_id = hero_id if acting_pos_user == hero_user_pos else opp_id
        # Reorder bets to user frame: user_pos = 1 - slumbot_pos
        bets_user = np.zeros(num_players, dtype=np.float32)
        for slumbot_pos in range(2):
            user_pos = 1 - slumbot_pos
            bets_user[user_pos] = float(snap["bets"][slumbot_pos]) * inv_scale
        stack_user_hero = float(snap["credits"][1 - hero_user_pos]) * inv_scale
        # B.6.2: per-position stacks vector in user frame (mirrors bets_user).
        stacks_user = np.zeros(num_players, dtype=np.float32)
        for slumbot_pos in range(2):
            user_pos = 1 - slumbot_pos
            stacks_user[user_pos] = float(snap["credits"][slumbot_pos]) * inv_scale
        events.append({
            "hand": list(hole_cards_int),
            "num_players": num_players,
            "hero_pos": hero_user_pos,
            "acting_pos": acting_pos_user,
            "big_blind": float(big_blind_internal),
            "small_blind": float(small_blind_internal),
            "stack": stack_user_hero,
            "stacks": stacks_user,
            "table": table,
            "pot": float(snap["pot"]) * inv_scale,
            "bets": bets_user,
            "action": action,
            "opponent_id": opponent_id,
        })
    return events


def _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins, chip_scale,
                      big_blind_internal=None):
    """Construct a GameState (in training chip units) from SlumbotState for MCTS."""
    inv_scale = 1.0 / chip_scale
    # Reorder to user frame
    bets_user = [0.0, 0.0]
    credits_user = [0.0, 0.0]
    players_state_user = [0, 0]
    for slumbot_pos in range(2):
        user_pos = 1 - slumbot_pos
        bets_user[user_pos] = float(state["bets"][slumbot_pos]) * inv_scale
        credits_user[user_pos] = float(state["credits"][slumbot_pos]) * inv_scale
        players_state_user[user_pos] = int(state["players_state"][slumbot_pos])

    bb_internal = (float(big_blind_internal) if big_blind_internal is not None
                   else float(state.get("high_bet", 10.0)) * inv_scale)
    # C.7.5: propagate the min-raise state. The harness tracks the last raise
    # increment in state["last_bet_size"] (Slumbot chip units); scale it into
    # internal chips. Table semantics keep last_raise_size >= big_blind
    # (preflop it starts at BB; a street change resets it to BB), and Slumbot's
    # own min-raise rule is high_bet + max(BB, last_bet_size) — so floor at BB.
    # Without this, GameState defaults last_raise_size to big_blind and the
    # legal mask admits sub-min-raise bins when facing a large bet.
    last_raise_size = max(bb_internal,
                          float(state.get("last_bet_size", 0.0)) * inv_scale)

    return GameState(
        num_players=2,
        hero_pos=hero_user_pos,
        active_player=1 - state["active_pos"],
        players_state=players_state_user,
        credits=credits_user,
        bets=bets_user,
        pot=float(state["pot"]) * inv_scale,
        high_bet=float(state["high_bet"]) * inv_scale,
        turn=int(state["turn"]),
        raise_sizes=raise_sizes,
        n_raise_bins=n_raise_bins,
        is_terminal=False,
        several_all_in=False,
        last_raise_size=last_raise_size,
        big_blind=bb_internal,
    )


# ============================================================================
# HTTP client
# ============================================================================

class SlumbotClient:
    """Thin wrapper over Slumbot's HTTP API.

    Retries transient network failures (ConnectTimeout / ReadTimeout /
    ConnectionError) with exponential backoff. Uses a persistent
    requests.Session so TCP+TLS handshakes are reused across requests
    (significant speedup on long evals).
    """

    def __init__(self, host=SLUMBOT_HOST, username="", password="",
                 timeout=30, retries=4, backoff=1.0, log=None):
        self.host = host
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self.log = log
        self.session = requests.Session()
        self.token = None
        if username and password:
            self.token = self._login(username, password)

    def _post(self, endpoint, data):
        import time
        url = f"https://{self.host}/slumbot/api/{endpoint}"
        # Retry on transient network errors (timeout / connection reset).
        # Other errors (HTTP 4xx/5xx, error_msg in body) propagate immediately.
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(url, json=data, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Slumbot {endpoint} HTTP {r.status_code}: {r.text}")
                body = r.json()
                if "error_msg" in body:
                    raise RuntimeError(
                        f"Slumbot {endpoint} error: {body['error_msg']}")
                new_tok = body.get("token")
                if new_tok:
                    self.token = new_tok
                return body
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt >= self.retries:
                    break
                wait = self.backoff * (2 ** attempt)
                if self.log is not None:
                    self.log(
                        f"  Slumbot {endpoint} {type(e).__name__}, "
                        f"retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.retries})")
                time.sleep(wait)
        raise RuntimeError(
            f"Slumbot {endpoint} failed after {self.retries + 1} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def _login(self, username, password):
        body = self._post("login", {"username": username, "password": password})
        tok = body.get("token")
        if not tok:
            raise RuntimeError("Slumbot login: no token in response")
        return tok

    def new_hand(self):
        data = {}
        if self.token:
            data["token"] = self.token
        return self._post("new_hand", data)

    def act(self, incr):
        if not self.token:
            raise RuntimeError("act() before token established (call new_hand first)")
        return self._post("act", {"token": self.token, "incr": incr})


# ============================================================================
# Agent loading
# ============================================================================

def _resolve_agent_path(path, project_root, version):
    """Resolve a relative agent path against data/<version>/."""
    if not path:
        raise ValueError("agent entry missing 'path'")
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, "data", version, path)


def _load_solver_bundle(agent_entry, config, device, fallback_temperature, log):
    """Build a solver bundle (no ASI / no MCTS / no opp embedding).

    Defaults come from config.solver and config.game; any key in the entry's
    `solver_overrides` dict takes precedence. Sampling temperature priority:
        solver_overrides.gto_temperature
        > config.solver.gto_temperature
        > agent_entry.action_temperature
        > slumbot_eval.action_temperature
    """
    name = agent_entry.get("name") or "solver"
    base_solver_cfg = dict(config.get("solver", {}))
    overrides = dict(agent_entry.get("solver_overrides", {}) or {})
    for k, v in overrides.items():
        base_solver_cfg[k] = v
    base_solver_cfg["device"] = device

    if "gto_temperature" in overrides:
        temperature = float(overrides["gto_temperature"])
    elif "gto_temperature" in base_solver_cfg:
        temperature = float(base_solver_cfg["gto_temperature"])
    else:
        entry_temp = agent_entry.get("action_temperature")
        temperature = float(entry_temp) if entry_temp is not None else fallback_temperature

    log(f"Loaded solver '{name}' (type={base_solver_cfg.get('type', 'v3')}, "
        f"mc_iters={base_solver_cfg.get('mc_iterations', 5000)}, "
        f"combo_response_iters={base_solver_cfg.get('combo_response_iters', 30)}, "
        f"gto_temperature={temperature}, device={device})")

    return {
        "name": name,
        "type": "solver",
        "agent": None,
        "norm_stats": None,
        "temperature": float(temperature),
        "opp_table": None,
        "mcts": None,
        "solver_cfg": base_solver_cfg,
        "use_opp_emb": False,
    }


def _load_one_agent(agent_entry, config, device, project_root, version,
                    fallback_temperature, log):
    """Load one agent from a path. Returns dict bundle or None on failure."""
    entry_type = (agent_entry.get("type") or "model").lower()
    if entry_type == "solver":
        return _load_solver_bundle(agent_entry, config, device,
                                   fallback_temperature, log)
    if entry_type != "model":
        log(f"WARNING: unknown agent type {entry_type!r}, treating as 'model'")

    path = _resolve_agent_path(agent_entry["path"], project_root, version)
    name = agent_entry.get("name") or os.path.basename(path.rstrip("/"))

    ckpt_path = _resolve_checkpoint_path(path)
    if ckpt_path is None:
        log(f"WARNING: no checkpoint for agent '{name}' at {path}, skipping")
        return None

    asi = ASI(log, config)
    asi.set_device(device)
    asi.load_checkpoint(ckpt_path)
    asi.eval()

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        log(f"WARNING: no norm_stats in '{name}', using identity normalization")
        norm_stats = {
            "pot_mean": 0.0, "pot_std": 1.0,
            "stack_mean": 0.0, "stack_std": 1.0,
            "bets_mean": 0.0, "bets_std": 1.0,
            "blind_mean": 0.0, "blind_std": 1.0,
        }
    # Temperature precedence: checkpoint > per-agent override > section fallback.
    # Checkpoint wins by default; entry override only kicks in if checkpoint
    # didn't store one.
    ckpt_temp = ckpt.get("temperature")
    if ckpt_temp is not None:
        temperature = float(ckpt_temp)
    else:
        entry_temp = agent_entry.get("action_temperature")
        temperature = float(entry_temp) if entry_temp is not None else fallback_temperature

    use_opp_emb = bool(agent_entry.get("use_opponent_embedding", False))
    use_mcts = bool(agent_entry.get("use_mcts", False))

    opp_table = None
    if use_opp_emb and asi.perception.opp_emb_enabled:
        opp_table = OpponentEmbeddingTable(asi.perception.d_model)

    mcts = None
    if use_mcts:
        mcts_cfg = config.get("mcts", {})
        # Same derivation as collect.py:52 — deterministic terminals (fold/
        # uncontested win) must be projected onto the value head's
        # mcts_value_scale axis, not left in raw chips (which breaks PUCT).
        big_blind_internal = float(config.get("game", {}).get("big_blind", 10))
        search_scale = float((norm_stats or {}).get("mcts_value_scale",
                                                    big_blind_internal))
        mcts = MCTS(asi, device, mcts_cfg, opponent_emb_table=opp_table,
                    search_scale=search_scale)

    log(f"Loaded '{name}' from {ckpt_path} "
        f"(temp={temperature}, mcts={use_mcts}, opp_emb={opp_table is not None})")

    return {
        "name": name,
        "type": "model",
        "agent": asi,
        "norm_stats": norm_stats,
        "temperature": float(temperature),
        "opp_table": opp_table,
        "mcts": mcts,
        "use_opp_emb": use_opp_emb,
    }


# ============================================================================
# Decision making
# ============================================================================

def _choose_action(bundle, events, state, hero_user_pos, hole_cards_int,
                   board_ints, action_history, raise_sizes, n_raise_bins,
                   n_actions, chip_scale, big_blind_internal,
                   small_blind_internal, amp_enabled, device_type, amp_dtype):
    """Run the agent (model / MCTS / solver) and return the chosen action_idx.

    Solver path uses hole_cards/board/action_history; model path ignores them.
    """
    if bundle.get("type") == "solver":
        scfg = bundle.get("solver_cfg", {})
        if scfg.get("type") == "v4":
            return _v4_solver_choose_action(
                bundle, state, hero_user_pos, hole_cards_int, board_ints,
                action_history, raise_sizes, n_raise_bins, n_actions, chip_scale,
                big_blind_internal, small_blind_internal,
            )
        return _solver_choose_action(
            bundle, state, hero_user_pos, hole_cards_int, board_ints,
            action_history, raise_sizes, n_raise_bins, n_actions, chip_scale,
            big_blind_internal, small_blind_internal,
        )

    asi = bundle["agent"]
    norm_stats = bundle["norm_stats"]
    # Normalize a copy in place (events are local to this hand)
    _normalize_events_inplace(events, norm_stats)

    gs = _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins, chip_scale,
                           big_blind_internal=big_blind_internal)

    if bundle["mcts"] is not None:
        return int(bundle["mcts"].search([events], gs))

    opp_table = bundle["opp_table"]
    skip_opp = (opp_table is None)
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            out = asi.forward_batch(
                [events], skip_memory=True,
                skip_opponent_emb=skip_opp,
                opponent_emb_table=opp_table,
                heads={"action"},
            )
    logits = out["action_logits"][0]
    legal_mask = torch.tensor(
        gs.get_legal_action_mask(n_actions), dtype=torch.bool, device=logits.device,
    )
    logits = logits.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(logits / max(bundle["temperature"], 1e-3), dim=0)
    return int(torch.multinomial(probs, 1).item())


# ============================================================================
# Hand loop
# ============================================================================

def _play_one_hand(client, bundle, config, raise_sizes, n_raise_bins, n_actions,
                   chip_scale, big_blind_internal, small_blind_internal,
                   amp_enabled, device_type, amp_dtype, clamp_counters, log,
                   action_hist=None, action_hist_by_street=None):
    """Play one Slumbot hand. Returns (winnings, baseline_winnings).

    Optional analytics buffers (mutated in place):
      action_hist: np.ndarray (n_actions,) — counts per chosen idx (post-clamp)
      action_hist_by_street: np.ndarray (4, n_actions) — same, split by street
    """
    r = client.new_hand()
    client_pos = r["client_pos"]
    hero_user_pos = 1 - client_pos
    hole_cards_int = [_card_to_int(c) for c in r["hole_cards"]]
    board_ints = _board_to_ints(r.get("board") or [])
    hero_action_indices = []

    is_v4 = (bundle.get("type") == "solver"
             and bundle.get("solver_cfg", {}).get("type") == "v4")
    if is_v4:
        bundle["_v4_bayesian"] = _v4_init_bayesian_state(hero_user_pos, num_players=2)
    prev_action_history_len = 0

    while True:
        # Update board if Slumbot revealed more cards
        if r.get("board"):
            board_ints = _board_to_ints(r["board"])
        action_str = r.get("action") or ""

        # Check for hand end
        if "winnings" in r and r["winnings"] is not None:
            return float(r["winnings"]), float(r.get("baseline_winnings") or 0.0)

        state, snapshots, _, action_history = _replay_action_string(
            action_str, hole_cards_int, board_ints, client_pos,
            raise_sizes, n_raise_bins, n_actions, hero_action_indices,
        )

        # v4 Bayesian update: process any new opponent actions since last iteration
        if is_v4 and len(action_history) > prev_action_history_len:
            scfg = bundle["solver_cfg"]
            opp_user_pos = 1 - hero_user_pos
            for ah_idx in range(prev_action_history_len, len(action_history)):
                acting_pos, act_type = action_history[ah_idx]
                if acting_pos == opp_user_pos and act_type is not None:
                    opp_action_idx = {
                        "call": 1, "call_postflop": 1,
                        "open": 2, "3bet": n_raise_bins + 2,
                        "bet_postflop": 2,
                    }.get(act_type, 1)
                    table_stub = _make_solver_table_stub(
                        hero_user_pos, hole_cards_int, board_ints, state,
                        raise_sizes, big_blind_internal, small_blind_internal,
                        chip_scale, num_players=2,
                    )
                    _v4_update_opponent_range(
                        bundle["_v4_bayesian"], opp_user_pos, opp_action_idx,
                        board_ints, hole_cards_int, table_stub,
                        action_history[:ah_idx], n_actions, scfg,
                    )
            prev_action_history_len = len(action_history)

        if state["is_terminal"]:
            # Server should respond with winnings on next /act, but we shouldn't
            # have an action to send. This branch can only be hit if our local
            # parser disagrees with server — break to safety.
            log(f"  WARN: local state terminal but no winnings in response, action={action_str!r}")
            return 0.0, 0.0

        # Whose turn?
        if state["active_pos"] != client_pos:
            # Slumbot's turn but no winnings: shouldn't happen (server would have
            # taken its action before responding). Defensive: re-fetch by acting
            # as a check/call placeholder is illegal. Just break.
            log(f"  WARN: opponent's turn in response, action={action_str!r}")
            return 0.0, 0.0

        # Pre-decision snapshot (action=None) — match training distribution
        snapshots_with_pre = list(snapshots)
        # Append a "we're about to act" snapshot mirroring evaluate.py:371-378
        pre_snap = _make_snapshot(state, n_actions, None, client_pos)
        snapshots_with_pre.append(pre_snap)

        events = _build_events(
            snapshots_with_pre, hole_cards_int, board_ints, hero_user_pos,
            client_pos, num_players=2,
            big_blind_internal=big_blind_internal,
            small_blind_internal=small_blind_internal,
            chip_scale=chip_scale, n_actions=n_actions,
        )

        action_idx = _choose_action(
            bundle, events, state, hero_user_pos, hole_cards_int, board_ints,
            action_history, raise_sizes, n_raise_bins, n_actions, chip_scale,
            big_blind_internal, small_blind_internal,
            amp_enabled, device_type, amp_dtype,
        )

        incr = _action_idx_to_incr(
            state, action_idx, raise_sizes[state["turn"]], n_raise_bins,
            hero_slumbot_pos=client_pos, clamp_counters=clamp_counters,
        )

        # If the clamp degraded a raise to call/check, _token_to_action_idx on
        # the resulting token will give a different idx than what we chose.
        # Cache the EFFECTIVE idx so the replay produces consistent events.
        if incr == "f":
            effective_idx = 0
        elif incr in ("c", "k"):
            effective_idx = 1
        elif incr.startswith("b"):
            effective_idx = _token_to_action_idx(
                state, incr, raise_sizes[state["turn"]], n_raise_bins,
            )
        else:
            effective_idx = action_idx
        hero_action_indices.append(effective_idx)

        # Analytics: record action distribution per street
        if action_hist is not None:
            action_hist[effective_idx] += 1
        if action_hist_by_street is not None:
            action_hist_by_street[int(state["turn"]), effective_idx] += 1

        r = client.act(incr)


# ============================================================================
# Main runner
# ============================================================================

def _resolve_paths(config):
    here = os.path.dirname(os.path.abspath(__file__))
    version = os.path.basename(os.path.abspath(os.path.join(here, "..")))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return version, project_root


def _bb_per_100(chip_winnings_sum, n_hands):
    if n_hands <= 0:
        return 0.0
    return (chip_winnings_sum / SLUMBOT_BIG_BLIND) / (n_hands / 100.0)


def _stderr_bb_per_100(per_hand_chips):
    if len(per_hand_chips) < 2:
        return 0.0
    arr = np.asarray(per_hand_chips, dtype=np.float64) / SLUMBOT_BIG_BLIND
    return float(arr.std(ddof=1) / np.sqrt(len(arr)) * 100.0)


def _stderr_bb_per_100_online(welford_n, welford_M2):
    """O(1) stderr of BB/100 from Welford's online variance state.

    welford_n:  number of samples incorporated so far
    welford_M2: running sum of squared deviations (already in BB units)
    Returns the same value as _stderr_bb_per_100 but without iterating the list.
    """
    if welford_n < 2:
        return 0.0
    variance = welford_M2 / (welford_n - 1)  # sample variance (ddof=1)
    return float(np.sqrt(variance / welford_n) * 100.0)


def _make_session_buffers(n_actions):
    """Fresh accumulators for one agent session."""
    return {
        "per_hand_chips": [],
        "per_hand_baseline": [],
        "sum_chips": 0.0,
        "sum_baseline": 0.0,
        # Welford's online variance accumulators (values in BB units).
        # chips:
        "welford_chips_n": 0,
        "welford_chips_mean": 0.0,
        "welford_chips_M2": 0.0,
        # baseline:
        "welford_baseline_n": 0,
        "welford_baseline_mean": 0.0,
        "welford_baseline_M2": 0.0,
        "clamp_counters": defaultdict(int),
        "action_hist": np.zeros(n_actions, dtype=np.int64),
        "action_hist_by_street": np.zeros((4, n_actions), dtype=np.int64),
        "history": {"bb100_raw": [], "bb100_baseline": []},
        "hands_failed": 0,
    }


def _update_progress(buffers, name, n_hands, log_every, pbar, log,
                     post_log=True):
    """Update tqdm and emit log_every-aligned history row from current buffers.

    Called from the main thread only (both sequential and parallel paths)."""
    per_hand_chips = buffers["per_hand_chips"]
    per_hand_baseline = buffers["per_hand_baseline"]
    done = len(per_hand_chips)
    if done > 0:
        latest_chips = per_hand_chips[-1]
        latest_baseline = per_hand_baseline[-1]
        buffers["sum_chips"] += latest_chips
        buffers["sum_baseline"] += latest_baseline

        # Welford online update (in BB units) — O(1) per hand.
        x_chips = latest_chips / SLUMBOT_BIG_BLIND
        n = buffers["welford_chips_n"] + 1
        delta = x_chips - buffers["welford_chips_mean"]
        new_mean = buffers["welford_chips_mean"] + delta / n
        delta2 = x_chips - new_mean
        buffers["welford_chips_n"] = n
        buffers["welford_chips_mean"] = new_mean
        buffers["welford_chips_M2"] += delta * delta2

        x_baseline = latest_baseline / SLUMBOT_BIG_BLIND
        n_b = buffers["welford_baseline_n"] + 1
        delta_b = x_baseline - buffers["welford_baseline_mean"]
        new_mean_b = buffers["welford_baseline_mean"] + delta_b / n_b
        delta2_b = x_baseline - new_mean_b
        buffers["welford_baseline_n"] = n_b
        buffers["welford_baseline_mean"] = new_mean_b
        buffers["welford_baseline_M2"] += delta_b * delta2_b

    running_bcorr = _bb_per_100(buffers["sum_baseline"], done)
    running_residual = _bb_per_100(
        buffers["sum_chips"] - buffers["sum_baseline"], done)
    pbar.set_postfix({
        "BB/100 base": f"{running_bcorr:+.1f}",
        "failed": buffers["hands_failed"],
    }, refresh=False)
    pbar.update(1)
    if post_log and done > 0 and done % log_every == 0:
        raw = _bb_per_100(buffers["sum_chips"], done)
        stderr_raw = _stderr_bb_per_100_online(
            buffers["welford_chips_n"], buffers["welford_chips_M2"])
        stderr_bcorr = _stderr_bb_per_100_online(
            buffers["welford_baseline_n"], buffers["welford_baseline_M2"])
        buffers["history"]["bb100_raw"].append((done, raw))
        buffers["history"]["bb100_baseline"].append((done, running_bcorr))
        log(f"  [{name}] {done}/{n_hands}: "
            f"raw={raw:+.2f} BB/100 (stderr={stderr_raw:.2f}), "
            f"baseline_corrected={running_bcorr:+.2f} BB/100 (stderr={stderr_bcorr:.2f}), "
            f"aivat_residual={running_residual:+.2f} BB/100, "
            f"clamps={dict(buffers['clamp_counters'])}")


def _run_agent_session_sequential(
    bundle, n_hands, log_every, host, username, password, timeout, retries,
    backoff, config, raise_sizes, n_raise_bins, n_actions, chip_scale,
    big_blind_internal, small_blind_internal, amp_enabled, device_type,
    amp_dtype, log,
):
    """Single-threaded path — bit-for-bit identical to the pre-parallel code."""
    name = bundle["name"]
    client = SlumbotClient(host=host, username=username, password=password,
                           timeout=timeout, retries=retries, backoff=backoff,
                           log=log)
    buffers = _make_session_buffers(n_actions)
    pbar = tqdm(total=n_hands, desc=f"Slumbot/{name}", unit="hand",
                smoothing=0)
    for hand_idx in range(n_hands):
        try:
            w, b = _play_one_hand(
                client, bundle, config, raise_sizes, n_raise_bins, n_actions,
                chip_scale, big_blind_internal, small_blind_internal,
                amp_enabled, device_type, amp_dtype, buffers["clamp_counters"], log,
                action_hist=buffers["action_hist"],
                action_hist_by_street=buffers["action_hist_by_street"],
            )
        except Exception as e:
            buffers["hands_failed"] += 1
            log(f"  hand {hand_idx + 1} failed: {type(e).__name__}: {e}")
            pbar.update(1)
            continue
        buffers["per_hand_chips"].append(w)
        buffers["per_hand_baseline"].append(b)
        _update_progress(buffers, name, n_hands, log_every, pbar, log)
    pbar.close()
    return buffers


# ─────────────────────────────────────────────────────────────────────────────
# Parallel path — MULTIPROCESS (not threads)
#
# Threads do NOT speed up this workload. Two reasons that both bite:
#   1. Python dispatch holds the GIL. Each PyTorch op (Linear, attention,
#      LayerNorm, ...) needs Python-side dispatch *before* the GPU kernel
#      launches. For batch=1 forwards (one hand at a time), dispatch is
#      50-90% of wall time, and GIL serializes it across threads.
#   2. All threads' CUDA kernels queue on the default stream — even when the
#      GIL is briefly released, the GPU executes them sequentially.
# Even though `requests.Session.post` releases the GIL during socket recv,
# the Python overhead between HTTP calls (event building, MCTS tree, model
# dispatch) is the actual bottleneck, and that part is fully serialized.
#
# Multiprocess solves both: N independent processes, each with its OWN Python
# interpreter (no shared GIL) and its OWN model copy on the GPU (own CUDA
# context, own stream). Scales linearly until either Slumbot rate-limits or
# the GPU runs out of memory for replicas.
#
# Per-worker isolation (implicit by being a separate process):
#   - SlumbotClient (own requests.Session + token)
#   - ASI model + opp_table + MCTS instance
#   - action_hist / action_hist_by_street / clamp_counters
# Cross-worker comms:
#   - shared mp.Value counter for dynamic hand allocation
#   - mp.Queue for per-hand outcomes + warnings + completion
# Workers never call the parent's `log` directly — they push warnings to the
# queue and the main process renders them.
# ─────────────────────────────────────────────────────────────────────────────


# Inter-process message tags.
_MSG_OK = "OK"           # (tag, worker_id, w, b)
_MSG_HAND_FAIL = "HFAIL" # (tag, worker_id, hand_idx, err_str)
_MSG_WORKER_ABORT = "ABORT"  # (tag, worker_id, reason)
_MSG_FATAL = "FATAL"     # (tag, worker_id, traceback_str)
_MSG_DONE = "DONE"       # (tag, worker_id, clamp_counters, action_hist_list,
                         #                  action_hist_by_street_list)
_MSG_WARN = "WARN"       # (tag, worker_id, text) — informational, not counted


def _is_rate_limit_error(exc):
    """Heuristic: rate-limit signal in the message → exponential backoff."""
    msg = str(exc).lower()
    return ("429" in msg or "too many" in msg or "rate limit" in msg
            or "throttle" in msg)


def _no_op_logger(*_args, **_kwargs):
    """Logger stub for ASI constructed inside worker processes."""
    pass


def _slumbot_worker_process(
    worker_id, state_dict, norm_stats, temperature, agent_name,
    use_opp_emb, use_mcts, device, config,
    raise_sizes, n_raise_bins, n_actions, chip_scale,
    big_blind_internal, small_blind_internal,
    host, username, password, timeout, retries, backoff,
    counter, n_hands_total, stop_event, result_q,
    max_consecutive_failures,
):
    """Worker process entry point. Builds its own ASI from `state_dict` on
    `device`, then loops: claim next hand from `counter`, play it, push
    outcome. Sends DONE with reduced per-worker buffers on exit.

    Runs in a fresh Python interpreter (spawn start method), so this function
    must be top-level (picklable) and rebuild everything from primitives.
    """
    import sys as _sys
    import torch as _torch

    def worker_log(text):
        try:
            result_q.put((_MSG_WARN, worker_id, str(text)))
        except Exception:
            pass

    try:
        # CPU/BLAS threading: each worker is one process with one model on
        # GPU. Don't oversubscribe the host CPU when N workers run in parallel
        # — torch defaults to OMP_NUM_THREADS=#cores per process, which means
        # 16 workers × 16 threads = 256 contended OS threads.
        _torch.set_num_threads(max(1, int(os.environ.get("SLUMBOT_TORCH_THREADS", "2"))))

        # Build agent on this process's GPU context.
        asi = ASI(_no_op_logger, config)
        asi.load_state_dict(state_dict, strict=False)
        asi.set_device(device)
        asi.eval()

        amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)

        opp_table = None
        if use_opp_emb and asi.perception.opp_emb_enabled:
            opp_table = OpponentEmbeddingTable(asi.perception.d_model)
        mcts = None
        if use_mcts:
            mcts_cfg = config.get("mcts", {})
            # Same derivation as collect.py:52 (see _load_one_agent).
            big_blind_internal = float(
                config.get("game", {}).get("big_blind", 10))
            search_scale = float((norm_stats or {}).get(
                "mcts_value_scale", big_blind_internal))
            mcts = MCTS(asi, device, mcts_cfg, opponent_emb_table=opp_table,
                        search_scale=search_scale)

        bundle = {
            "name": agent_name,
            "agent": asi,
            "norm_stats": norm_stats,
            "temperature": float(temperature),
            "opp_table": opp_table,
            "mcts": mcts,
        }

        client = SlumbotClient(host=host, username=username, password=password,
                               timeout=timeout, retries=retries, backoff=backoff,
                               log=worker_log)

        local_clamps = defaultdict(int)
        local_hist = np.zeros(n_actions, dtype=np.int64)
        local_hist_by_street = np.zeros((4, n_actions), dtype=np.int64)
        consecutive_failures = 0

        while not stop_event.is_set():
            # Claim next hand from shared atomic counter.
            with counter.get_lock():
                if counter.value >= n_hands_total:
                    hand_idx = -1
                else:
                    hand_idx = counter.value
                    counter.value += 1
            if hand_idx < 0:
                break

            try:
                w, b = _play_one_hand(
                    client, bundle, config, raise_sizes, n_raise_bins,
                    n_actions, chip_scale, big_blind_internal,
                    small_blind_internal, amp_enabled, device_type, amp_dtype,
                    local_clamps, log=worker_log,
                    action_hist=local_hist,
                    action_hist_by_street=local_hist_by_street,
                )
            except Exception as e:
                consecutive_failures += 1
                err_str = f"{type(e).__name__}: {e}"
                result_q.put((_MSG_HAND_FAIL, worker_id, hand_idx, err_str))
                # Per-worker exponential backoff on transient failures.
                # Bounded at 30 s. Rate-limit errors trigger backoff immediately;
                # other errors trigger after 2+ in a row.
                if _is_rate_limit_error(e) or consecutive_failures > 1:
                    wait = min(backoff * (2 ** min(consecutive_failures, 6)), 30.0)
                    if stop_event.wait(timeout=wait):
                        break
                if consecutive_failures >= max_consecutive_failures:
                    result_q.put((
                        _MSG_WORKER_ABORT, worker_id,
                        f"{consecutive_failures} consecutive failures, "
                        f"last={err_str}"))
                    # Send DONE so the parent folds in whatever this worker
                    # managed before giving up.
                    result_q.put((_MSG_DONE, worker_id, dict(local_clamps),
                                  local_hist.tolist(), local_hist_by_street.tolist()))
                    return
                continue
            consecutive_failures = 0
            result_q.put((_MSG_OK, worker_id, float(w), float(b)))

        # Normal exit. Ship aggregates back as plain Python types so pickling
        # through mp.Queue stays cheap and avoids any shared-memory tensor
        # storage paths (per project memory: tensors in mp IPC payloads
        # exhaust vm.max_map_count at scale).
        result_q.put((_MSG_DONE, worker_id, dict(local_clamps),
                      local_hist.tolist(), local_hist_by_street.tolist()))
    except BaseException:
        tb = traceback.format_exc()
        try:
            result_q.put((_MSG_FATAL, worker_id, tb))
        except Exception:
            pass
        _sys.stderr.write(f"[slumbot worker {worker_id}] FATAL:\n{tb}\n")
        _sys.stderr.flush()
    finally:
        # Flush the queue's feeder thread so the parent definitely sees our
        # last message before this process exits.
        try:
            result_q.close()
            result_q.join_thread()
        except Exception:
            pass


def _run_agent_session_parallel(
    bundle, n_hands, n_workers, log_every, max_consecutive_failures,
    host, username, password, timeout, retries, backoff,
    config, raise_sizes, n_raise_bins, n_actions, chip_scale,
    big_blind_internal, small_blind_internal, amp_enabled, device_type,
    amp_dtype, device, log,
):
    """Multiprocess parallel path. Each worker has its OWN model copy on the
    GPU device — that's the only way to bypass the GIL + single-CUDA-stream
    bottleneck that kills threaded inference on small batches. Per-hand
    outcomes flow back through an mp.Queue."""
    import torch as _torch
    import torch.multiprocessing as tmp

    name = bundle["name"]
    asi = bundle["agent"]
    use_opp_emb = bundle["opp_table"] is not None
    use_mcts = bundle["mcts"] is not None

    buffers = _make_session_buffers(n_actions)

    # Pull state_dict to CPU once — workers receive it via spawn-pickle and
    # rebuild their own ASI on the target device. norm_stats / temperature
    # ride along.
    state_dict = {k: v.detach().cpu() for k, v in asi.state_dict().items()}
    norm_stats = bundle["norm_stats"]
    temperature = bundle["temperature"]

    # Move parent's model off the GPU during the parallel session. We don't
    # use it while workers run, and dropping it frees ~one model's worth of
    # VRAM so the N worker copies (+ N CUDA contexts) have room.
    parent_device = getattr(asi, "device_", device)
    moved_parent = False
    if str(device).startswith("cuda"):
        try:
            asi.cpu()
            asi.device_ = "cpu"
            _torch.cuda.empty_cache()
            moved_parent = True
        except BaseException:
            moved_parent = False

    log(f"  multiprocess mode: {n_workers} workers, "
        f"max_consecutive_failures={max_consecutive_failures}, "
        f"use_mcts={use_mcts}, use_opp_emb={use_opp_emb}, device={device}")

    ctx = tmp.get_context("spawn")
    counter = ctx.Value("i", 0)  # has its own internal Lock; workers use .get_lock()
    stop_event = ctx.Event()
    result_q = ctx.Queue(maxsize=max(256, n_workers * 8))

    procs = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_slumbot_worker_process,
            args=(
                wid, state_dict, norm_stats, temperature, name,
                use_opp_emb, use_mcts, device, config,
                raise_sizes, n_raise_bins, n_actions, chip_scale,
                big_blind_internal, small_blind_internal,
                host, username, password, timeout, retries, backoff,
                counter, n_hands, stop_event, result_q,
                max_consecutive_failures,
            ),
            daemon=False,  # let workers finish their current hand on parent exit
        )
        p.start()
        procs.append(p)

    pbar = tqdm(total=n_hands, desc=f"Slumbot/{name}", unit="hand",
                smoothing=0)
    workers_alive = n_workers
    fatal_tb = None
    aborted = []

    def _fold_done(msg):
        _, _wid, clamps, hist, hist_by_street = msg
        for k, v in clamps.items():
            buffers["clamp_counters"][k] += v
        buffers["action_hist"] += np.asarray(hist, dtype=np.int64)
        buffers["action_hist_by_street"] += np.asarray(hist_by_street,
                                                       dtype=np.int64)

    try:
        while workers_alive > 0:
            try:
                msg = result_q.get(timeout=2.0)
            except Exception:
                # Idle timeout: if every worker process has died without
                # signalling DONE/FATAL (e.g. SIGKILL from OOM-killer),
                # break out to avoid hanging forever.
                if all(not p.is_alive() for p in procs):
                    silent = [p.pid for p in procs
                              if p.exitcode not in (0, None)]
                    if silent and fatal_tb is None:
                        fatal_tb = (
                            f"worker(s) died without reporting "
                            f"(pids/exitcodes={[(p.pid, p.exitcode) for p in procs]})")
                    break
                continue

            tag = msg[0]
            if tag == _MSG_OK:
                _, _wid, w, b = msg
                buffers["per_hand_chips"].append(w)
                buffers["per_hand_baseline"].append(b)
                _update_progress(buffers, name, n_hands, log_every, pbar, log)
            elif tag == _MSG_HAND_FAIL:
                _, wid, hand_idx, err_str = msg
                buffers["hands_failed"] += 1
                log(f"  [{name} w{wid}] hand {hand_idx + 1} failed: {err_str}")
                pbar.update(1)
            elif tag == _MSG_WARN:
                _, wid, text = msg
                log(f"  [{name} w{wid}] {text}")
            elif tag == _MSG_WORKER_ABORT:
                _, wid, reason = msg
                log(f"  [{name} w{wid}] aborted: {reason}")
                aborted.append(wid)
            elif tag == _MSG_DONE:
                _fold_done(msg)
                workers_alive -= 1
            elif tag == _MSG_FATAL:
                _, wid, tb = msg
                if fatal_tb is None:
                    fatal_tb = f"worker {wid} fatal:\n{tb}"
                stop_event.set()
                workers_alive -= 1
                # Don't break — let surviving workers send DONE for clean
                # buffer accounting.
    except KeyboardInterrupt:
        log("  KeyboardInterrupt — signalling workers to stop")
        stop_event.set()
        raise
    finally:
        pbar.close()
        stop_event.set()
        # Join workers. They check stop_event only between hands so we may
        # wait up to one HTTP RTT per worker. Hard-terminate after 60 s.
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                log(f"  worker pid={p.pid} still alive after 60s, terminating")
                p.terminate()
                p.join(timeout=5)

        # Final drain — pick up any DONE / FATAL messages emitted during
        # shutdown so the buffers reflect every hand that did get played.
        while True:
            try:
                msg = result_q.get_nowait()
            except Exception:
                break
            tag = msg[0]
            if tag == _MSG_OK:
                _, _wid, w, b = msg
                buffers["per_hand_chips"].append(w)
                buffers["per_hand_baseline"].append(b)
            elif tag == _MSG_HAND_FAIL:
                buffers["hands_failed"] += 1
            elif tag == _MSG_DONE:
                _fold_done(msg)
            elif tag == _MSG_FATAL and fatal_tb is None:
                _, wid, tb = msg
                fatal_tb = f"worker {wid} fatal:\n{tb}"

        # Restore parent's model to its original device.
        if moved_parent:
            try:
                asi.set_device(parent_device)
            except BaseException:
                pass

    if fatal_tb is not None:
        raise RuntimeError(f"slumbot eval parallel session failed: {fatal_tb}")
    if aborted and len(aborted) == n_workers:
        raise RuntimeError(
            f"all {n_workers} workers aborted — check Slumbot connectivity / "
            f"auth / rate-limit (aborted workers: {aborted})")

    return buffers


# ─────────────────────────────────────────────────────────────────────────────
# Parallel solver path
#
# The ASI parallel worker replicates `state_dict` to each worker process; the
# solver has no model state, only an MC equity kernel that lives in
# `agent.gto_utils.gpu_solver_v3`. We still benefit from N processes because:
#   1. Each hand spends ~half its wall time blocked on Slumbot HTTP — fully
#      parallel across processes (no shared GIL).
#   2. CUDA contexts created per worker allow the MC kernels to interleave on
#      the GPU. Throughput scales sub-linearly (contention on a single device),
#      typically 2-4x with 6 workers, but that's a large win over sequential.
# Sequential bit-for-bit behaviour is preserved on `n_workers <= 1`.
# ─────────────────────────────────────────────────────────────────────────────


def _solver_worker_process(
    worker_id, name, solver_cfg, temperature, device, config,
    raise_sizes, n_raise_bins, n_actions, chip_scale,
    big_blind_internal, small_blind_internal,
    host, username, password, timeout, retries, backoff,
    counter, n_hands_total, stop_event, result_q,
    max_consecutive_failures,
):
    """Worker process for `type=="solver"` entries.

    No ASI / no MCTS / no opp embedding — just a SlumbotClient and a solver
    bundle. Same hand-claim / report / shutdown protocol as the ASI worker so
    the parent's drain loop is shared in spirit (parallel/_consume helpers).
    """
    import sys as _sys
    import torch as _torch

    def worker_log(text):
        try:
            result_q.put((_MSG_WARN, worker_id, str(text)))
        except Exception:
            pass

    try:
        _torch.set_num_threads(max(1, int(os.environ.get(
            "SLUMBOT_TORCH_THREADS", "2"))))

        bundle = {
            "name": name,
            "type": "solver",
            "agent": None,
            "norm_stats": None,
            "temperature": float(temperature),
            "opp_table": None,
            "mcts": None,
            "solver_cfg": dict(solver_cfg),
        }

        # Solver path doesn't use autocast — set inert values so `_play_one_hand`
        # can still pass them through `_choose_action`.
        amp_enabled, device_type, amp_dtype = False, "cpu", _torch.float32

        client = SlumbotClient(host=host, username=username, password=password,
                               timeout=timeout, retries=retries, backoff=backoff,
                               log=worker_log)

        local_clamps = defaultdict(int)
        local_hist = np.zeros(n_actions, dtype=np.int64)
        local_hist_by_street = np.zeros((4, n_actions), dtype=np.int64)
        consecutive_failures = 0

        while not stop_event.is_set():
            with counter.get_lock():
                if counter.value >= n_hands_total:
                    hand_idx = -1
                else:
                    hand_idx = counter.value
                    counter.value += 1
            if hand_idx < 0:
                break

            try:
                w, b = _play_one_hand(
                    client, bundle, config, raise_sizes, n_raise_bins,
                    n_actions, chip_scale, big_blind_internal,
                    small_blind_internal, amp_enabled, device_type, amp_dtype,
                    local_clamps, log=worker_log,
                    action_hist=local_hist,
                    action_hist_by_street=local_hist_by_street,
                )
            except Exception as e:
                consecutive_failures += 1
                err_str = f"{type(e).__name__}: {e}"
                result_q.put((_MSG_HAND_FAIL, worker_id, hand_idx, err_str))
                if _is_rate_limit_error(e) or consecutive_failures > 1:
                    wait = min(backoff * (2 ** min(consecutive_failures, 6)), 30.0)
                    if stop_event.wait(timeout=wait):
                        break
                if consecutive_failures >= max_consecutive_failures:
                    result_q.put((
                        _MSG_WORKER_ABORT, worker_id,
                        f"{consecutive_failures} consecutive failures, "
                        f"last={err_str}"))
                    result_q.put((_MSG_DONE, worker_id, dict(local_clamps),
                                  local_hist.tolist(),
                                  local_hist_by_street.tolist()))
                    return
                continue
            consecutive_failures = 0
            result_q.put((_MSG_OK, worker_id, float(w), float(b)))

        result_q.put((_MSG_DONE, worker_id, dict(local_clamps),
                      local_hist.tolist(), local_hist_by_street.tolist()))
    except BaseException:
        tb = traceback.format_exc()
        try:
            result_q.put((_MSG_FATAL, worker_id, tb))
        except Exception:
            pass
        _sys.stderr.write(f"[solver worker {worker_id}] FATAL:\n{tb}\n")
        _sys.stderr.flush()
    finally:
        try:
            result_q.close()
            result_q.join_thread()
        except Exception:
            pass


def _run_agent_session_parallel_solver(
    bundle, n_hands, n_workers, log_every, max_consecutive_failures,
    host, username, password, timeout, retries, backoff,
    config, raise_sizes, n_raise_bins, n_actions, chip_scale,
    big_blind_internal, small_blind_internal, device, log,
):
    """Multiprocess parallel path for solver entries — no model replication.

    Each worker process opens its own CUDA context (if device is cuda) and
    runs its own solver instance. Throughput scales sub-linearly because all
    workers share one GPU; the wall-time win comes from overlapping Slumbot
    HTTP RTT across workers (each hand is half blocked on the network)."""
    import torch.multiprocessing as tmp

    name = bundle["name"]
    solver_cfg = bundle["solver_cfg"]
    temperature = bundle["temperature"]

    buffers = _make_session_buffers(n_actions)

    log(f"  multiprocess solver: {n_workers} workers, "
        f"max_consecutive_failures={max_consecutive_failures}, device={device}, "
        f"mc_iters={solver_cfg.get('mc_iterations')}, "
        f"combo_response_iters={solver_cfg.get('combo_response_iters')}")

    ctx = tmp.get_context("spawn")
    counter = ctx.Value("i", 0)
    stop_event = ctx.Event()
    result_q = ctx.Queue(maxsize=max(256, n_workers * 8))

    procs = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_solver_worker_process,
            args=(
                wid, name, solver_cfg, temperature, device, config,
                raise_sizes, n_raise_bins, n_actions, chip_scale,
                big_blind_internal, small_blind_internal,
                host, username, password, timeout, retries, backoff,
                counter, n_hands, stop_event, result_q,
                max_consecutive_failures,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    pbar = tqdm(total=n_hands, desc=f"Slumbot/{name}", unit="hand",
                smoothing=0)
    workers_alive = n_workers
    fatal_tb = None
    aborted = []

    def _fold_done(msg):
        _, _wid, clamps, hist, hist_by_street = msg
        for k, v in clamps.items():
            buffers["clamp_counters"][k] += v
        buffers["action_hist"] += np.asarray(hist, dtype=np.int64)
        buffers["action_hist_by_street"] += np.asarray(hist_by_street,
                                                       dtype=np.int64)

    try:
        while workers_alive > 0:
            try:
                msg = result_q.get(timeout=2.0)
            except Exception:
                if all(not p.is_alive() for p in procs):
                    silent = [p.pid for p in procs
                              if p.exitcode not in (0, None)]
                    if silent and fatal_tb is None:
                        fatal_tb = (
                            f"solver worker(s) died without reporting "
                            f"(pids/exitcodes="
                            f"{[(p.pid, p.exitcode) for p in procs]})")
                    break
                continue

            tag = msg[0]
            if tag == _MSG_OK:
                _, _wid, w, b = msg
                buffers["per_hand_chips"].append(w)
                buffers["per_hand_baseline"].append(b)
                _update_progress(buffers, name, n_hands, log_every, pbar, log)
            elif tag == _MSG_HAND_FAIL:
                _, wid, hand_idx, err_str = msg
                buffers["hands_failed"] += 1
                log(f"  [{name} w{wid}] hand {hand_idx + 1} failed: {err_str}")
                pbar.update(1)
            elif tag == _MSG_WARN:
                _, wid, text = msg
                log(f"  [{name} w{wid}] {text}")
            elif tag == _MSG_WORKER_ABORT:
                _, wid, reason = msg
                log(f"  [{name} w{wid}] aborted: {reason}")
                aborted.append(wid)
            elif tag == _MSG_DONE:
                _fold_done(msg)
                workers_alive -= 1
            elif tag == _MSG_FATAL:
                _, wid, tb = msg
                if fatal_tb is None:
                    fatal_tb = f"solver worker {wid} fatal:\n{tb}"
                stop_event.set()
                workers_alive -= 1
    except KeyboardInterrupt:
        log("  KeyboardInterrupt — signalling workers to stop")
        stop_event.set()
        raise
    finally:
        pbar.close()
        stop_event.set()
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                log(f"  solver worker pid={p.pid} still alive after 60s, "
                    f"terminating")
                p.terminate()
                p.join(timeout=5)

        while True:
            try:
                msg = result_q.get_nowait()
            except Exception:
                break
            tag = msg[0]
            if tag == _MSG_OK:
                _, _wid, w, b = msg
                buffers["per_hand_chips"].append(w)
                buffers["per_hand_baseline"].append(b)
            elif tag == _MSG_HAND_FAIL:
                buffers["hands_failed"] += 1
            elif tag == _MSG_DONE:
                _fold_done(msg)
            elif tag == _MSG_FATAL and fatal_tb is None:
                _, wid, tb = msg
                fatal_tb = f"solver worker {wid} fatal:\n{tb}"

    if fatal_tb is not None:
        raise RuntimeError(
            f"slumbot solver eval parallel session failed: {fatal_tb}")
    if aborted and len(aborted) == n_workers:
        raise RuntimeError(
            f"all {n_workers} solver workers aborted — check Slumbot "
            f"connectivity / auth / rate-limit (aborted workers: {aborted})")

    return buffers


def _run_agent_session(
    bundle, n_hands, n_workers, log_every, max_consecutive_failures, host,
    username, password, timeout, retries, backoff, config, raise_sizes,
    n_raise_bins, n_actions, chip_scale, big_blind_internal,
    small_blind_internal, amp_enabled, device_type, amp_dtype, device, log,
):
    """Run `n_hands` for one agent. Dispatch sequential vs parallel.

    Sequential (`n_workers <= 1`) is bit-for-bit equivalent to the
    pre-parallel implementation. For `n_workers > 1`, solver and model
    entries use distinct parallel paths because the ASI worker replicates
    `state_dict` to each process (irrelevant + costly for the solver)."""
    if n_workers <= 1:
        return _run_agent_session_sequential(
            bundle, n_hands, log_every, host, username, password, timeout,
            retries, backoff, config, raise_sizes, n_raise_bins, n_actions,
            chip_scale, big_blind_internal, small_blind_internal, amp_enabled,
            device_type, amp_dtype, log,
        )
    if bundle.get("type") == "solver":
        return _run_agent_session_parallel_solver(
            bundle, n_hands, n_workers, log_every, max_consecutive_failures,
            host, username, password, timeout, retries, backoff, config,
            raise_sizes, n_raise_bins, n_actions, chip_scale,
            big_blind_internal, small_blind_internal, device, log,
        )
    return _run_agent_session_parallel(
        bundle, n_hands, n_workers, log_every, max_consecutive_failures,
        host, username, password, timeout, retries, backoff, config,
        raise_sizes, n_raise_bins, n_actions, chip_scale, big_blind_internal,
        small_blind_internal, amp_enabled, device_type, amp_dtype, device, log,
    )


def run_slumbot_evaluation(config, device, log, results_dir_override=None):
    cfg = config.get("slumbot_eval", {})
    if not cfg:
        log("slumbot_eval: no config section, skipping")
        return

    agents_cfg = cfg.get("agents", [])
    if not agents_cfg:
        log("slumbot_eval: agents list empty, skipping")
        return

    n_hands = int(cfg.get("n_hands", 1000))
    n_workers = max(1, int(cfg.get("n_workers", 1)))
    max_consecutive_failures = max(1, int(cfg.get("max_consecutive_failures", 20)))
    log_every = int(cfg.get("log_every", 50))
    fallback_temperature = float(cfg.get("action_temperature", 0.5))
    host = cfg.get("host", SLUMBOT_HOST)
    username = cfg.get("username", "") or ""
    password = cfg.get("password", "") or ""
    timeout = int(cfg.get("request_timeout", 30))
    retries = int(cfg.get("retries", 4))
    backoff = float(cfg.get("backoff", 1.0))

    game_cfg = config.get("game", {})
    big_blind_internal = float(game_cfg.get("big_blind", 10))
    small_blind_internal = big_blind_internal / 2.0
    chip_scale = SLUMBOT_BIG_BLIND / big_blind_internal

    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3

    amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)

    version, project_root = _resolve_paths(config)
    if results_dir_override:
        results_dir = results_dir_override
    else:
        exp_name = config.get("name", "default")
        results_dir = os.path.join(project_root, "data", version, exp_name, "slumbot_eval")
    os.makedirs(results_dir, exist_ok=True)

    log("=== Slumbot evaluation ===")
    log(f"host={host}, n_hands={n_hands}/agent, n_workers={n_workers}, "
        f"chip_scale={chip_scale} (BB internal={big_blind_internal})")

    all_results = {}

    for agent_entry in agents_cfg:
        bundle = _load_one_agent(
            agent_entry, config, device, project_root, version,
            fallback_temperature, log,
        )
        if bundle is None:
            continue

        name = bundle["name"]
        log(f"\n--- Playing {name} for {n_hands} hands ---")
        buffers = _run_agent_session(
            bundle, n_hands, n_workers, log_every, max_consecutive_failures,
            host, username, password, timeout, retries, backoff, config,
            raise_sizes, n_raise_bins, n_actions, chip_scale,
            big_blind_internal, small_blind_internal, amp_enabled, device_type,
            amp_dtype, device, log,
        )
        per_hand_chips = buffers["per_hand_chips"]
        per_hand_baseline = buffers["per_hand_baseline"]
        clamp_counters = buffers["clamp_counters"]
        action_hist = buffers["action_hist"]
        action_hist_by_street = buffers["action_hist_by_street"]
        history = buffers["history"]
        hands_failed = buffers["hands_failed"]

        n_played = len(per_hand_chips)
        total_chips = float(sum(per_hand_chips))
        total_baseline = float(sum(per_hand_baseline))
        bb100_raw = _bb_per_100(total_chips, n_played)
        bb100_bcorr = _bb_per_100(total_baseline, n_played)
        bb100_residual = _bb_per_100(total_chips - total_baseline, n_played)
        stderr_bb100 = _stderr_bb_per_100_online(
            buffers["welford_chips_n"], buffers["welford_chips_M2"])
        stderr_bb100_bcorr = _stderr_bb_per_100_online(
            buffers["welford_baseline_n"], buffers["welford_baseline_M2"])
        mbb_per_hand_raw = bb100_raw * 10.0  # 1000 mbb / 100 hands

        # Action distribution analytics
        total_actions = int(action_hist.sum())
        action_dist = (action_hist / max(total_actions, 1)).tolist()
        action_dist_by_street = (
            action_hist_by_street
            / np.maximum(action_hist_by_street.sum(axis=1, keepdims=True), 1)
        ).tolist()

        agent_result = {
            "hands_played": n_played,
            "hands_failed": hands_failed,
            "session_total_chips": total_chips,
            "session_baseline_total_chips": total_baseline,
            "bb_per_100_raw": round(bb100_raw, 4),
            "bb_per_100_baseline_corrected": round(bb100_bcorr, 4),
            "bb_per_100_aivat_residual": round(bb100_residual, 4),
            "mbb_per_hand_raw": round(mbb_per_hand_raw, 4),
            "stderr_bb_per_100": round(stderr_bb100, 4),
            "stderr_bb_per_100_baseline_corrected": round(stderr_bb100_bcorr, 4),
            "clamps": dict(clamp_counters),
            "type": bundle.get("type", "model"),
            "use_mcts": bundle["mcts"] is not None,
            "use_opponent_embedding": bundle["opp_table"] is not None,
            "temperature": bundle["temperature"],
            "solver_cfg": bundle.get("solver_cfg"),
            "decisions_made": total_actions,
            "fold_rate": round(action_dist[0], 4) if total_actions else 0.0,
            "allin_rate": round(action_dist[n_actions - 1], 4) if total_actions else 0.0,
            "action_distribution": [round(p, 4) for p in action_dist],
            "action_distribution_by_street": [
                [round(p, 4) for p in row] for row in action_dist_by_street
            ],
            "action_counts_total": [int(c) for c in action_hist],
            "action_counts_by_street": [
                [int(c) for c in row] for row in action_hist_by_street
            ],
        }
        all_results[name] = agent_result

        # Save per-agent history (raw per-hand outcomes for variance analysis)
        hist_path = os.path.join(results_dir, f"{log.init_time}_{name}.history.pt")
        torch.save({
            "per_hand_chips": per_hand_chips,
            "per_hand_baseline": per_hand_baseline,
            "history": history,
            "action_hist": action_hist.tolist(),
            "action_hist_by_street": action_hist_by_street.tolist(),
        }, hist_path)

        log(f"\n[{name}] FINAL: raw={bb100_raw:+.2f} BB/100 (stderr={stderr_bb100:.2f}), "
            f"baseline_corrected={bb100_bcorr:+.2f} BB/100 (stderr={stderr_bb100_bcorr:.2f}), "
            f"aivat_residual={bb100_residual:+.2f} BB/100 "
            f"({n_played} hands, {hands_failed} failed)")
        log(f"[{name}] Clamps: {dict(clamp_counters)}")

    # Save aggregate JSON
    out_json = os.path.join(results_dir, f"{log.init_time}.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_hands_per_agent": n_hands,
            "chip_scale": chip_scale,
            "big_blind_internal": big_blind_internal,
            "agents": all_results,
            "config": cfg,
        }, f, indent=4)
    log(f"\nResults saved to {out_json}")


# ============================================================================
# CLI
# ============================================================================

def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent against Slumbot")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = _pick_device()

    from utils import Logger
    version, project_root = _resolve_paths(config)
    name = config.get("name", "default")
    base_dir = os.path.join(project_root, "data", version, name)
    log = Logger(base_dir)
    log(f"Slumbot eval — version={version}, experiment={name}, device={device}")

    run_slumbot_evaluation(config, device, log)


if __name__ == "__main__":
    main()
