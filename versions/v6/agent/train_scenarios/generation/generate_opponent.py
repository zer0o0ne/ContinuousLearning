"""
Range-based opponent action data generation.

Simulates poker hands with trained agents, tracking per-player hand ranges.
At each decision point:
1. Runs action inference for every hand in the acting player's range
2. Averages the action distributions -> training target for opponent_action_head
3. Picks a concrete hand from range, executes a sampled action on the table
4. Narrows range: removes hands where P(chosen)/P(best) < threshold

Events are stored in shared format (all hands unmasked). During training,
the data loader converts to per-observer format with appropriate masking.

Can be run standalone:
    python -m agent.train_scenarios.generation.generate_opponent --config config.json
"""

import os
import copy
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from env.table import Table
from evaluation.evaluate import _normalize_events_inplace
from agent.train_scenarios.generation.generate import (
    _get_raise_sizes, load_dataset, _read_meta, _write_meta, _meta_path,
)
from agent.agent import ASI
from agent.mcts.game_state import GameState
from agent.resume import atomic_torch_save
from utils import get_amp_config


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _load_agents(agents_dir, config, device, log, fallback_temperature):
    """Load all agents from subdirectories."""
    if not os.path.isdir(agents_dir):
        log(f"ERROR: agents_dir not found: {agents_dir}")
        return []

    agent_names = sorted(
        d for d in os.listdir(agents_dir)
        if os.path.isdir(os.path.join(agents_dir, d))
    )
    if not agent_names:
        log(f"ERROR: no agent subdirectories found in {agents_dir}")
        return []

    agents = []
    for name in agent_names:
        agent_path = os.path.join(agents_dir, name)
        ckpt_path = ASI._find_best_checkpoint(agent_path)
        if ckpt_path is None:
            log(f"WARNING: no checkpoint found for agent '{name}', skipping")
            continue

        agent = ASI(log, config)
        agent.set_device(device)
        agent.load_checkpoint(ckpt_path)
        agent.eval()

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        norm_stats = ckpt.get("norm_stats")
        if norm_stats is None:
            log(f"WARNING: no norm_stats for '{name}', using identity")
            norm_stats = {
                "pot_mean": 0.0, "pot_std": 1.0,
                "stack_mean": 0.0, "stack_std": 1.0,
                "bets_mean": 0.0, "bets_std": 1.0,
                "blind_mean": 0.0, "blind_std": 1.0,
            }

        temperature = ckpt.get("temperature")
        if temperature is None:
            temperature = fallback_temperature

        agents.append({
            "agent": agent,
            "norm_stats": norm_stats,
            "name": name,
            "temperature": temperature,
        })
        log(f"Loaded agent '{name}' from {ckpt_path} (temperature={temperature})")

    return agents


# ---------------------------------------------------------------------------
# Combo utilities
# ---------------------------------------------------------------------------

_ALL_COMBOS = None
_COMBO_TO_IDX = None


def _get_all_combos():
    """All C(52,2) = 1326 two-card combos, cached. Returns list of (c1,c2), c1<c2."""
    global _ALL_COMBOS
    if _ALL_COMBOS is None:
        _ALL_COMBOS = [(c1, c2) for c1 in range(52) for c2 in range(c1 + 1, 52)]
    return _ALL_COMBOS


def _get_combo_to_idx():
    """Map from (c1,c2) with c1<c2 to combo index 0..1325."""
    global _COMBO_TO_IDX
    if _COMBO_TO_IDX is None:
        _COMBO_TO_IDX = {c: i for i, c in enumerate(_get_all_combos())}
    return _COMBO_TO_IDX


def _dead_mask_array(dead_cards):
    """Return shape (1326,) bool array: True where the combo conflicts with a dead card."""
    combos = _get_all_combos()
    if not dead_cards:
        return np.zeros(len(combos), dtype=bool)
    dead_set = set(dead_cards)
    return np.array(
        [c1 in dead_set or c2 in dead_set for (c1, c2) in combos],
        dtype=bool,
    )


_M_COMPAT = None


def _get_compat_matrix():
    """1326x1326 bool matrix. M[i,j] = True iff combos i and j share NO card.

    Used for factored card-removal correction across players' belief ranges.
    Computed once (vectorized), cached. Float32 storage (~7 MB) for fast CPU
    matmul — numpy bool matmul is awkward and the result is a (1326,) float
    vector anyway.
    """
    global _M_COMPAT
    if _M_COMPAT is None:
        combos = np.array(_get_all_combos(), dtype=np.int16)  # (1326, 2)
        ci = combos[:, None, :]  # (1326, 1, 2)
        cj = combos[None, :, :]  # (1, 1326, 2)
        overlap = (
            (ci[..., 0:1] == cj[..., 0:1]) |
            (ci[..., 0:1] == cj[..., 1:2]) |
            (ci[..., 1:2] == cj[..., 0:1]) |
            (ci[..., 1:2] == cj[..., 1:2])
        ).squeeze(-1)  # (1326, 1326) bool
        _M_COMPAT = (~overlap).astype(np.float32)
    return _M_COMPAT


def _apply_joint_card_removal(w_live, player_weights, active_pos, players_state):
    """Factored card-removal correction across opponents' belief ranges.

    For the acting player at `active_pos`: multiply `w_live` by, for each
    other LIVE (non-folded, non-acting) player j, the compatibility mass
    `M @ player_weights[j]`. The result is the observer's view of acting
    player's effective range, accounting for the fact that combos held by
    other players cannot be held by the actor.

    IMPORTANT: this is an INFERENCE-VIEW correction only. It does NOT
    update any per-player marginal posterior `w_i`. Use the returned
    `w_eff` for ESS top-K subset selection, target weighted averaging,
    and fixed_hand sampling — but use the original `w_live` to drive
    the Bayes posterior update on the acting player.

    Args:
        w_live: (1326,) np.float32 — acting player's live (dead-masked,
                normalized) weights.
        player_weights: dict[pos -> np.ndarray(1326,)] — per-player
                marginals (raw, NOT dead-masked here).
        active_pos: int — acting player position.
        players_state: indexable — table.players_state; values >= 0 mean
                the player is live (not folded).

    Returns:
        (1326,) np.float32, normalized. Falls back to `w_live` if the
        correction collapses (rare numerical edge case where opponents'
        beliefs are joint-incompatible with all of acting's combos).
    """
    M = _get_compat_matrix()
    w_eff = w_live.copy()
    for j, w_j in player_weights.items():
        if j == active_pos:
            continue
        if players_state[j] < 0:  # folded
            continue
        compat_mass = M @ w_j  # (1326,) float32 in [0, 1]
        w_eff = w_eff * compat_mass
    s = float(w_eff.sum())
    if s < 1e-9:
        # Degenerate: opponents' beliefs are joint-incompatible with all
        # of acting's live combos. Fall back to plain w_live (normalized).
        return w_live
    return w_eff / s


def _filter_dead(combos, dead_cards):
    """Remove combos containing any dead card."""
    if not dead_cards:
        return combos
    return [(c1, c2) for c1, c2 in combos if c1 not in dead_cards and c2 not in dead_cards]


def _get_board_dead(table):
    """Set of revealed board card IDs."""
    if table.turn == 0:
        return set()
    elif table.turn == 1:
        return set(table.deck[:3].tolist())
    elif table.turn == 2:
        return set(table.deck[:4].tolist())
    else:
        return set(table.deck[:5].tolist())


# ---------------------------------------------------------------------------
# Shared event construction
# ---------------------------------------------------------------------------

def _get_table_display(deck, turn):
    """5-element board display with -1 for unrevealed cards."""
    if turn == 0:
        return [-1] * 5
    elif turn == 1:
        return list(deck[:3]) + [-1, -1]
    elif turn == 2:
        return list(deck[:4]) + [-1]
    else:
        return list(deck[:5])


def _build_shared_events(snapshots, deck, fixed_hands, num_players,
                         big_blind, small_blind, n_actions, up_to,
                         opponent_ids=None):
    """Build shared event sequence (unmasked, all players' hands included).

    Args:
        snapshots: list of state snapshots
        deck: table deck array (for board cards)
        fixed_hands: dict {pos: (c1,c2)} for players with fixed hands
        num_players: number of players
        big_blind, small_blind: blind sizes
        n_actions: action space size
        up_to: include snapshots[0..up_to] inclusive
        opponent_ids: optional dict {pos: str} — persistent player identifiers

    Returns:
        list of shared event dicts
    """
    events = []
    for snap in snapshots[:up_to + 1]:
        table_cards = _get_table_display(deck, snap["turn"])

        action = snap["action"]
        if action is None:
            action = [0.0] * n_actions
        elif isinstance(action, torch.Tensor):
            # Plain-Python action vector — see `generate.py:_rebuild_events`
            # for why: torch.Tensor in event dicts forces shared-memory IPC
            # which exhausts `vm.max_map_count` after ~10k hands when
            # results stream back to the main process.
            action = action.detach().cpu().tolist()

        hands = {}
        for pos, hand in fixed_hands.items():
            if hand is not None:
                hands[pos] = list(hand)

        event = {
            "hands": hands,
            "num_players": num_players,
            "acting_pos": snap["active_pos"],
            "big_blind": float(big_blind),
            "small_blind": float(small_blind),
            "pot": float(snap["pot"]),
            "bets": np.copy(snap["bets"]),
            "table": table_cards,
            "action": action,
            "stacks": list(snap["credits"]),
        }
        if opponent_ids is not None:
            event["opponent_ids"] = dict(opponent_ids)
        events.append(event)
    return events


def _shared_to_standard(shared_events, hero_pos, hero_hand):
    """Convert shared events to standard format for a specific hero.

    Creates lightweight copies — only hand/hero_pos/stack differ.
    """
    events = []
    for e in shared_events:
        events.append({
            "hand": list(hero_hand),
            "num_players": e["num_players"],
            "hero_pos": hero_pos,
            "acting_pos": e["acting_pos"],
            "big_blind": e["big_blind"],
            "small_blind": e["small_blind"],
            "stack": float(e["stacks"][hero_pos]),
            "table": list(e["table"]),
            "pot": e["pot"],
            "bets": np.copy(e["bets"]),
            "action": e["action"],
        })
    return events


# ---------------------------------------------------------------------------
# Range inference
# ---------------------------------------------------------------------------

def _compute_range_probs(agent, shared_events, combos, active_pos,
                         norm_stats, temperature, device, n_actions,
                         max_batch, amp_config, proxy=None, agent_name=None):
    """Compute action distributions for the given combo subset via batched inference.

    The caller (`generate_opponent_hand`) decides which combos to forward
    (typically the top-K-by-mass subset under the current belief). Caller
    also performs the weighted average — this function only returns the
    per-combo distributions.

    Args:
        agent: ASI model in eval mode
        shared_events: shared event sequence up to decision point
        combos: list of (c1,c2) — combos to forward
        active_pos: acting player's position
        norm_stats: z-score normalization stats from checkpoint
        temperature: softmax temperature
        device: torch device
        n_actions: action space size
        max_batch: max combos per forward pass
        amp_config: (amp_enabled, device_type, amp_dtype)

    Returns:
        per_combo_probs: (len(combos), n_actions) per-combo distributions (CPU fp32)
    """
    amp_enabled, device_type, amp_dtype = amp_config

    # Build and normalize template events (placeholder hand, will be replaced)
    template = _shared_to_standard(shared_events, active_pos, [0, 1])
    _normalize_events_inplace(template, norm_stats)

    # In parallel/actor mode we can use the server's templated forward path
    # (Phase 3): ship the template ONCE per `_compute_range_probs` call and
    # have the server replicate per combo. This avoids pickling N copies of
    # the (~100-event) template per chunk for IPC. Controlled by the global
    # PARALLEL_OPT_ENABLED flag in evaluator.py.
    use_templated = False
    if proxy is not None:
        try:
            from agent.mcts.evaluator import PARALLEL_OPT_ENABLED
            use_templated = bool(PARALLEL_OPT_ENABLED) and hasattr(
                proxy, "forward_batch_templated")
        except ImportError:
            use_templated = False

    all_probs = []

    for start in range(0, len(combos), max_batch):
        batch_combos = combos[start:start + max_batch]

        if use_templated:
            # Send template once + N hand pairs. Server replicates and runs.
            logits = proxy.forward_batch_templated(
                agent_name, template, batch_combos, heads=("action",))
        elif proxy is not None:
            # Legacy parallel path: actor builds N event copies and ships them.
            batch_events = []
            for c1, c2 in batch_combos:
                events = [dict(e) for e in template]
                for e in events:
                    e["hand"] = [c1, c2]
                batch_events.append(events)
            logits = proxy.forward_batch(agent_name, batch_events, heads=("action",))
        else:
            # Sequential path: in-process forward, unchanged.
            batch_events = []
            for c1, c2 in batch_combos:
                events = [dict(e) for e in template]
                for e in events:
                    e["hand"] = [c1, c2]
                batch_events.append(events)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    out = agent.forward_batch(batch_events, skip_memory=True, heads={"action"})
                logits = out["action_logits"]  # (batch, n_actions)

        # Softmax in fp32 even when the forward ran under fp16 autocast.
        # fp16 softmax can emit slight negatives / nan at the edge, which
        # torch.multinomial rejects ("element < 0 / nan / inf"). Casting
        # logits → fp32 here also turns finite-but-very-large fp16 values
        # into safe fp32 numbers; only true ±inf survives and is handled
        # by the row-level fallback downstream in generate_opponent_hand.
        logits = logits.float()
        probs = F.softmax(logits / temperature, dim=-1)
        all_probs.append(probs.cpu())

    per_combo_probs = torch.cat(all_probs, dim=0)  # (n_combos, n_actions)
    return per_combo_probs


# ---------------------------------------------------------------------------
# Hand generation
# ---------------------------------------------------------------------------

def generate_opponent_hand(config, agents_list, device, amp_config, player_ids=None,
                           proxy=None):
    """Generate training scenarios from one poker hand with range tracking.

    Args:
        config: merged config dict (game + opponent_data params)
        agents_list: list of agent info dicts from _load_agents
        device: torch device string
        amp_config: (amp_enabled, device_type, amp_dtype)
        player_ids: optional list of persistent player IDs (length >= max_players)

    Returns:
        list of scenario dicts, or None on failure
    """
    big_blind = config.get("big_blind", 10)
    small_blind = big_blind // 2
    max_stack = config.get("max_stack", 1000)
    max_players = config.get("max_players", 9)
    max_batch = config.get("max_batch_combos", 256)
    raise_sizes = _get_raise_sizes(config)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3

    # Soft-Bayes belief update config (R5/R6/R8).
    bayes_cfg = config.get("bayes") or {}
    bayes_enabled = bool(bayes_cfg.get("enabled", True))
    if not bayes_enabled:
        raise NotImplementedError(
            "Legacy hard-cut opponent_data path was removed in Stage 4 "
            "refactor. Set opponent_data.bayes.enabled=true."
        )
    tau_belief = float(bayes_cfg.get("tau_belief", 2.0))
    ess_mass = float(bayes_cfg.get("ess_truncation_mass", 0.995))

    min_stack = config.get("min_stack", big_blind * 10)
    num_players = random.randint(2, max_players)
    start_credits = random.randint(min_stack, max_stack)

    # Build opponent_ids mapping for this hand
    if player_ids is not None:
        opponent_ids = {pos: player_ids[pos] for pos in range(num_players)}
    else:
        opponent_ids = {pos: f"anon_{pos}" for pos in range(num_players)}

    table = Table(
        num_players=num_players,
        raise_sizes=raise_sizes,
        start_credits=start_credits,
        big_blind=big_blind,
        small_blind=small_blind,
    )
    table.start_table()

    # Select agents for each seat WITH REPLACEMENT
    seated = random.choices(agents_list, k=num_players)

    # Per-player state: soft belief = float32 weight per combo (uniform prior).
    n_combos = 1326
    all_combos = _get_all_combos()
    player_weights = {
        pos: np.ones(n_combos, dtype=np.float32) / n_combos
        for pos in range(num_players)
    }
    fixed_hands = {pos: None for pos in range(num_players)}

    # Snapshots for event reconstruction
    snapshots = [{
        "pot": table.pot,
        "bets": np.copy(table.bets),
        "credits": list(table.credits),
        "turn": table.turn,
        "active_pos": table.active_player,
        "action": None,
    }]

    scenarios = []
    max_actions = 4 * num_players

    for _ in range(max_actions):
        active_pos = table.active_player

        if table.players_state[active_pos] != 1:
            break

        agent_info = seated[active_pos]
        # In parallel/actor mode there is no live model (it lives on the
        # inference server); the forward is routed via `proxy` by agent name.
        agent = None if proxy is not None else agent_info["agent"]
        agent_name = agent_info["name"]

        # ----- Soft-belief live weights (mask dead, renormalize) ---------
        dead = _get_board_dead(table)
        dead_mask = _dead_mask_array(dead)
        w_live = player_weights[active_pos].copy()
        w_live[dead_mask] = 0.0
        live_mass = float(w_live.sum())
        if live_mass < 1e-9:
            break
        w_live /= live_mass

        # ----- Joint card-removal correction (R7) -----------------------
        # IMPORTANT: `w_live` drives the POSTERIOR (per-player marginal w_i).
        # `w_eff` drives INFERENCE (target, top-K subset, sampling, metrics).
        # Card-removal correction is a marginalization step over OTHER
        # players' beliefs — it gives the observer's view of the acting
        # player's effective range — but does NOT update the marginal
        # posterior for the acting player (their own actions only update
        # their own w). Keep this split — do not merge `w_eff` back into
        # `player_weights[active_pos]`.
        w_eff = _apply_joint_card_removal(
            w_live, player_weights, active_pos, table.players_state,
        )

        # ----- ESS-based top-K forward subset ---------------------------
        sorted_idx = np.argsort(-w_eff)
        cumsum = np.cumsum(w_eff[sorted_idx])
        top_k = int(np.searchsorted(cumsum, ess_mass)) + 1
        top_k = max(1, min(top_k, n_combos))
        forward_indices = sorted_idx[:top_k]
        forward_combos = [all_combos[i] for i in forward_indices]
        forward_weights = w_eff[forward_indices].copy()
        fw_sum = float(forward_weights.sum())
        if fw_sum < 1e-12:
            break
        forward_weights /= fw_sum  # renormalize over forward subset

        # Decision snapshot (before action)
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": active_pos,
            "action": None,
        })
        decision_snap_idx = len(snapshots) - 1

        # Build shared events up to this decision
        shared_events = _build_shared_events(
            snapshots, table.deck, fixed_hands, num_players,
            big_blind, small_blind, n_actions, up_to=decision_snap_idx,
            opponent_ids=opponent_ids,
        )

        if len(shared_events) < 2:
            break

        # ----- Per-combo action distributions for forward subset --------
        per_combo_probs = _compute_range_probs(
            agent, shared_events, forward_combos, active_pos,
            agent_info["norm_stats"], agent_info["temperature"],
            device, n_actions, max_batch, amp_config,
            proxy=proxy, agent_name=agent_name,
        )

        # Mask out unplayable actions (dominated fold / raises that collapse
        # to call or all-in) so the training target and the sampled action
        # both respect the playable set. Re-normalize per row.
        gs = GameState.from_table(table, active_pos)
        legal_mask = torch.tensor(
            gs.get_legal_action_mask(n_actions), dtype=torch.bool,
        )
        # Sanitize: any nan/±inf from the network forward becomes 0, then
        # clip negatives (fp16 softmax can produce small negatives). Without
        # this `torch.multinomial` rejects the row with "inf / nan / element
        # < 0" — exactly the crash we hit on long-event-sequence hands.
        per_combo_probs = torch.nan_to_num(
            per_combo_probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        per_combo_probs = per_combo_probs.masked_fill(~legal_mask, 0.0)
        row_sums = per_combo_probs.sum(dim=-1, keepdim=True)
        # Any row whose probabilities all zeroed out (either everything was
        # nan/inf, or the network put all its mass on illegal actions) falls
        # back to uniform-over-legal so we can still sample an action and the
        # training target stays well-defined.
        legal_count = int(legal_mask.sum().item())
        if legal_count == 0:
            # Defensive: caller already checks via GameState, but stay safe.
            break
        bad_rows = (row_sums.squeeze(-1) <= 1e-12)
        if bool(bad_rows.any()):
            uniform_legal = (legal_mask.float() / legal_count).unsqueeze(0)
            per_combo_probs[bad_rows] = uniform_legal
            row_sums = per_combo_probs.sum(dim=-1, keepdim=True)
        per_combo_probs = per_combo_probs / row_sums.clamp(min=1e-12)

        # ----- Belief-weighted training target --------------------------
        # `forward_weights` is float32 on CPU; lift to torch and match the
        # device per_combo_probs lives on (CPU after `_compute_range_probs`).
        forward_weights_t = torch.from_numpy(forward_weights).to(
            per_combo_probs.device, dtype=per_combo_probs.dtype)
        # If any combo's row collapsed to all-zero before the uniform-legal
        # rescue (shouldn't happen after the rescue, but stay defensive),
        # zero its weight contribution and renormalize.
        row_mask = per_combo_probs.sum(dim=-1) > 1e-9
        if not bool(row_mask.all()):
            forward_weights_t = forward_weights_t * row_mask.to(
                forward_weights_t.dtype)
            fw_sum_t = forward_weights_t.sum().clamp(min=1e-9)
            forward_weights_t = forward_weights_t / fw_sum_t

        target = (per_combo_probs * forward_weights_t.unsqueeze(-1)).sum(dim=0)
        target = target / target.sum().clamp(min=1e-9)  # safety renorm
        avg_probs = target  # name kept for downstream scenario field

        # ----- Fix hand at first action of this player ------------------
        if fixed_hands[active_pos] is None:
            chosen_idx = int(np.random.choice(top_k, p=forward_weights))
            fixed_hands[active_pos] = forward_combos[chosen_idx]

        fixed = fixed_hands[active_pos]
        try:
            fixed_idx_in_forward = forward_combos.index(fixed)
        except ValueError:
            # Fixed hand was sampled before but dropped out of the forward
            # subset (its weight collapsed below the ESS truncation mass).
            # Cannot honestly sample an action for the fixed hand → stop.
            break

        fixed_probs = per_combo_probs[fixed_idx_in_forward]
        # Sample action from fixed hand's distribution
        chosen_action = torch.multinomial(fixed_probs, 1).item()

        action = torch.zeros(n_actions, dtype=torch.float32)
        action[chosen_action] = 1.0

        # Active observers (non-folded, non-acting)
        hero_positions = [
            pos for pos in range(num_players)
            if pos != active_pos and table.players_state[pos] >= 0
        ]

        facing_bet = max(0, table.high_bet - table.bets[active_pos])

        # ----- ESS / top-mass scenario metadata (R8) --------------------
        # Computed over `w_eff` (observer's view) so metrics reflect the
        # belief the training target actually averages against. With joint
        # card removal, ESS tends to be slightly LOWER than over `w_live`
        # because the correction concentrates mass on combos compatible
        # with other players' holdings.
        ess = float(1.0 / float(np.power(w_eff, 2).sum().clip(min=1e-9)))
        max_w = float(w_eff.max())
        top_mass_size = int((w_eff >= 0.99 * max_w).sum()) if max_w > 0 else 0

        # Record scenario
        scenarios.append({
            "events": copy.deepcopy(shared_events),
            "opponent_action_probs": avg_probs.tolist(),
            "acting_pos": active_pos,
            "hero_positions": hero_positions,
            "num_players": num_players,
            "pot": float(table.pot),
            "facing_bet": float(facing_bet),
            "n_events": len(shared_events),
            "range_ess": ess,
            "range_top_mass_size": top_mass_size,
            "opponent_ids": dict(opponent_ids),
        })

        # Execute action on table
        end, several_all_in, state, bet = table.step(action)

        # Post-action snapshot
        snapshots.append({
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": table.active_player,
            "action": action,
        })

        # ----- Soft Bayes posterior update on the active player ---------
        # Likelihood π(a*|c) for each forward combo; tempered by τ. Combos
        # outside the forward subset have no observation — keep their
        # current weight under the live (dead-masked) belief and renormalize.
        likelihood = per_combo_probs[:, chosen_action].cpu().numpy().astype(np.float32)
        tempered = np.power(np.clip(likelihood, 1e-12, None), 1.0 / tau_belief)

        new_w_full = np.zeros(n_combos, dtype=np.float32)
        new_w_full[forward_indices] = w_live[forward_indices] * tempered
        # Carry non-forward live combos at their renormalized prior weight
        # (no observation → keep as-is, just under the renormalized scale).
        non_forward_live_mask = np.ones(n_combos, dtype=bool)
        non_forward_live_mask[forward_indices] = False
        non_forward_live_mask &= ~dead_mask
        new_w_full[non_forward_live_mask] = w_live[non_forward_live_mask]

        total = float(new_w_full.sum())
        if total < 1e-12:
            # Pathological: posterior collapsed everywhere. Fall back to
            # the renormalized live belief (no update this step).
            new_w_full = w_live.astype(np.float32, copy=True)
            total = float(new_w_full.sum())
            if total < 1e-12:
                break
        new_w_full /= total
        player_weights[active_pos] = new_w_full

        if end or several_all_in:
            break

    return scenarios if scenarios else None


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_opponent_dataset(config, save_dir, device, log,
                              agents_override=None, resume=False,
                              config_hash=None):
    """Generate full opponent action dataset.

    Loads trained agents, plays them against each other with range tracking,
    and records range-averaged action distributions as training targets.

    Args:
        config: full config dict
        save_dir: directory to save dataset.pt
        device: torch device string
        log: logger callable
        agents_override: optional pre-built list of agent dicts
            (`[{"agent": ASI, "norm_stats": dict, "name": str,
                 "temperature": float}, ...]`). When provided, skips
            `_load_agents` — useful for benchmarking where checkpoints
            don't exist or for unit tests where you want to pin specific
            weights. Data quality with random-init agents is meaningless
            but throughput numbers are valid.
        resume: if True, look at `meta.json` and continue from where a
            prior interrupted run left off (sequential mode only — parallel
            mode is all-or-nothing). On `config_hash` mismatch the existing
            dataset is renamed to `dataset.pt.stale.<ts>` and generation
            restarts.
        config_hash: hash of game.* + solver.* — see
            `agent.resume.compute_config_hash`. Stored in `meta.json` so
            subsequent resumes can detect stale data.

    Returns:
        list of scenario dicts
    """
    opp_cfg = config.get("opponent_data", {})
    game_cfg = config.get("game", {})
    n_hands = opp_cfg.get("n_hands", 5000)
    fallback_temperature = opp_cfg.get("action_temperature", 0.3)
    save_every = opp_cfg.get("save_every_hands", 500)

    os.makedirs(save_dir, exist_ok=True)
    dataset_path = os.path.join(save_dir, "dataset.pt")

    # -----------------------------------------------------------------
    # Resume bookkeeping (sequential path only; parallel mode is
    # all-or-nothing because actors don't checkpoint individually).
    # -----------------------------------------------------------------
    prior_scenarios = []
    start_attempts = 0
    n_workers_cfg = int(opp_cfg.get("n_workers", 1) or 1)

    if resume:
        meta = _read_meta(save_dir)
        if meta is not None:
            if config_hash is not None \
                    and meta.get("config_hash") != config_hash:
                stale = (f"{dataset_path}.stale."
                         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                log(f"Opponent dataset config_hash mismatch (was "
                    f"{meta.get('config_hash')!r}, now {config_hash!r}). "
                    f"Renaming existing to {os.path.basename(stale)} and "
                    f"starting fresh.")
                if os.path.exists(dataset_path):
                    os.replace(dataset_path, stale)
                try:
                    os.unlink(_meta_path(save_dir))
                except OSError:
                    pass
            elif meta.get("done") and meta.get("target", 0) >= n_hands:
                log(f"Opponent dataset already complete at {save_dir} "
                    f"(target={meta['target']} >= {n_hands})")
                return torch.load(dataset_path, weights_only=False)
            elif os.path.exists(dataset_path):
                if n_workers_cfg > 1:
                    log(f"Partial opponent dataset present but n_workers>1; "
                        f"parallel mode does not support mid-generation "
                        f"resume — regenerating from scratch.")
                    os.unlink(dataset_path)
                    try:
                        os.unlink(_meta_path(save_dir))
                    except OSError:
                        pass
                else:
                    prior_scenarios = torch.load(
                        dataset_path, weights_only=False)
                    start_attempts = int(meta.get("completed_attempts", 0))
                    log(f"Resuming opponent dataset: "
                        f"{start_attempts}/{n_hands} attempts done, "
                        f"{len(prior_scenarios)} scenarios on disk")
    else:
        # Legacy non-resume: prefer any existing dataset.pt as-is.
        existing = load_dataset(save_dir, log=log)
        if existing is not None:
            return existing

    # Merge game params into generation config
    gen_cfg = {}
    gen_cfg.update(game_cfg)
    gen_cfg.update(opp_cfg)

    amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)
    amp_config = (amp_enabled, device_type, amp_dtype)

    log("=== Opponent Action Data Generation ===")

    if agents_override is not None:
        agents_list = agents_override
        log(f"Using {len(agents_list)} caller-provided agent(s) "
            f"(skipping disk load): {[a['name'] for a in agents_list]}")
    else:
        agents_dir = opp_cfg.get("agents_dir", "")
        if agents_dir and not os.path.isabs(agents_dir):
            version = os.path.basename(os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")))
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "..", ".."))
            agents_dir = os.path.join(project_root, "data", version, agents_dir)
        log(f"Loading agents from {agents_dir}")
        agents_list = _load_agents(agents_dir, config, device, log,
                                   fallback_temperature)
        if not agents_list:
            log("No agents loaded. Aborting.")
            return []

    log(f"Loaded {len(agents_list)} agents: {[a['name'] for a in agents_list]}")
    bayes_log_cfg = gen_cfg.get("bayes") or {}
    log(f"Generating {n_hands} hands, "
        f"tau_belief={bayes_log_cfg.get('tau_belief', 2.0)}, "
        f"ess_truncation_mass={bayes_log_cfg.get('ess_truncation_mass', 0.995)}, "
        f"max_batch={gen_cfg.get('max_batch_combos', 256)}")

    # Persistent player IDs — simulate realistic table dynamics
    max_players = gen_cfg.get("max_players", 9)
    n_player_pool = opp_cfg.get("n_player_pool", max_players * 3)
    swap_prob = opp_cfg.get("player_swap_prob", 0.05)
    player_pool = [f"p_{i}" for i in range(n_player_pool)]
    table_roster = list(player_pool[:max_players])
    log(f"Player pool: {n_player_pool} IDs, swap_prob={swap_prob}")

    def _persist(scenarios_list, completed_attempts, meta_done):
        atomic_torch_save(scenarios_list, dataset_path)
        if config_hash is not None or resume:
            _write_meta(save_dir, {
                "version":            1,
                "target":             n_hands,
                "completed_attempts": completed_attempts,
                "completed_hands":    len(scenarios_list),
                "done":               bool(meta_done),
                "config_hash":        config_hash,
                "n_workers":          n_workers_cfg,
            })

    # Dispatch: parallel (opponent_data.n_workers > 1) reuses the MCTS GPU
    # inference server — CPU actors play hands + range bookkeeping and offload
    # the action-head combo inference (FORWARD_BATCH) to one GPU server. The
    # default (n_workers <= 1) keeps the original sequential path unchanged.
    n_workers = n_workers_cfg
    if n_workers > 1:
        # All-or-nothing parallel path. No prior partial scenarios are used.
        scenarios = _run_parallel_opponent(
            agents_list, config, gen_cfg, device, log, n_hands, n_workers,
            max_players, n_player_pool, swap_prob, player_pool)
        log(f"Generated {len(scenarios)} scenarios (parallel, {n_workers} actors)")
    else:
        scenarios = list(prior_scenarios)
        failed = max(0, start_attempts - len(prior_scenarios))
        remaining = n_hands - start_attempts
        if start_attempts > 0:
            log(f"Resuming sequential opponent generation: "
                f"{remaining} attempts remaining")

        for offset in tqdm(range(remaining), desc="Generating opponent data"):
            hand_i = start_attempts + offset
            # Simulate player rotation — occasionally swap seat identities
            for pos in range(max_players):
                if random.random() < swap_prob:
                    table_roster[pos] = random.choice(player_pool)

            result = generate_opponent_hand(gen_cfg, agents_list, device, amp_config,
                                            player_ids=table_roster)
            if result is not None:
                for s in result:
                    s["hand_id"] = hand_i
                scenarios.extend(result)
            else:
                failed += 1

            if (hand_i + 1) % save_every == 0:
                _persist(scenarios, hand_i + 1, meta_done=False)
                log(f"  Incremental save: {len(scenarios)} scenarios ({hand_i + 1} hands)")

        log(f"Generated {len(scenarios)} scenarios from "
            f"{n_hands - failed} hands ({failed} failed)")

    if scenarios:
        ess_values = [s["range_ess"] for s in scenarios]
        top_mass_values = [s["range_top_mass_size"] for s in scenarios]
        log(f"Range ESS: min={min(ess_values):.1f}, max={max(ess_values):.1f}, "
            f"avg={sum(ess_values) / len(ess_values):.1f}")
        log(f"Range top_mass: min={min(top_mass_values)}, "
            f"max={max(top_mass_values)}, "
            f"avg={sum(top_mass_values) / len(top_mass_values):.1f}")

    _persist(scenarios, n_hands, meta_done=True)
    log(f"Dataset saved to {dataset_path}")

    return scenarios


# ---------------------------------------------------------------------------
# Parallel generation (CPU actors + reused GPU inference server)
# ---------------------------------------------------------------------------

def _opp_actor_main(worker_id, agents_meta, gen_cfg, n_hands, hand_id_offset,
                    seed, max_players, player_pool, swap_prob,
                    req_q, resp_q, result_q, progress, output_dir,
                    progress_counter=None):
    """Actor process: play `n_hands` opponent hands with an EvalProxy (action
    inference offloaded to the server), write scenarios to a per-actor pickle
    file on disk, then signal completion via `result_q` (path only, no payload).

    Why disk transport: mp.Queue with large pickled payloads silently lost the
    OK message in Python 3.14 (observed across multiple runs — actor confirmed
    `close()+join_thread()` flushed the feeder, yet parent's `get()` never
    returned the message). Disk-based payload + tiny-signal-via-queue removes
    the queue from the failure path entirely.

    Never touches CUDA — the model lives on the server; all CPU here."""
    import os as _os
    import pickle as _pickle
    import sys as _sys
    import traceback as _tb
    _sys.stderr.write(f"[opp actor {worker_id}] starting, n_hands={n_hands}\n")
    _sys.stderr.flush()
    try:
        import torch as _torch
        _torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        _torch.manual_seed(seed)
        from agent.mcts.evaluator import EvalProxy

        proxy = EvalProxy(worker_id, req_q, resp_q)
        amp_config = (False, "cpu", _torch.float32)  # unused on the proxy path
        table_roster = list(player_pool[:max_players])
        scenarios = []
        # When a `progress_counter` is set (parallel mode), forward per-hand
        # progress to the shared counter that the parent's tqdm thread polls,
        # and skip our own per-actor tqdm.
        if progress_counter is not None:
            it = range(n_hands)
        else:
            it = tqdm(range(n_hands), desc=f"opp actor {worker_id}") \
                if progress else range(n_hands)
        for hand_i in it:
            for pos in range(max_players):
                if random.random() < swap_prob:
                    table_roster[pos] = random.choice(player_pool)
            result = generate_opponent_hand(
                gen_cfg, agents_meta, "cpu", amp_config,
                player_ids=table_roster, proxy=proxy)
            if result is not None:
                hid = hand_id_offset + hand_i
                for s in result:
                    s["hand_id"] = hid
                scenarios.extend(result)
            if progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 1

        # Persist payload to disk atomically (write to .tmp then rename).
        out_path = _os.path.join(output_dir, f"actor_{worker_id}.pkl")
        tmp_path = out_path + ".tmp"
        _sys.stderr.write(
            f"[opp actor {worker_id}] loop done, writing {len(scenarios)} "
            f"scenarios to {out_path}\n")
        _sys.stderr.flush()
        with open(tmp_path, "wb") as _f:
            _pickle.dump(scenarios, _f, protocol=_pickle.HIGHEST_PROTOCOL)
            _f.flush()
            _os.fsync(_f.fileno())
        _os.rename(tmp_path, out_path)
        size_mb = _os.path.getsize(out_path) / (1024 * 1024)
        _sys.stderr.write(
            f"[opp actor {worker_id}] wrote {size_mb:.1f} MB, signalling OK\n")
        _sys.stderr.flush()
        # Tiny signal via queue (path string, no big payload).
        result_q.put(("OK", worker_id, out_path))
        result_q.close()
        result_q.join_thread()
        _sys.stderr.write(f"[opp actor {worker_id}] OK signal flushed, exiting\n")
        _sys.stderr.flush()
    except BaseException:
        # BaseException catches SystemExit / KeyboardInterrupt too — otherwise
        # the actor can exit silently with code 0 and parent waits forever.
        tb = _tb.format_exc()
        _sys.stderr.write(f"[opp actor {worker_id}] FAILED:\n{tb}\n")
        _sys.stderr.flush()
        try:
            result_q.put(("ACTOR_ERROR", worker_id, tb))
            result_q.close()
            result_q.join_thread()
        except BaseException:
            pass


def _run_parallel_opponent(agents_list, config, gen_cfg, device, log, n_hands,
                           n_workers, max_players, n_player_pool, swap_prob,
                           player_pool):
    """Spawn the inference server + `n_workers` CPU actors, gather and merge
    their scenarios. Hand ids are contiguous across actors (offset per actor).
    """
    import torch.multiprocessing as tmp
    from agent.mcts.inference_server import server_main

    opp_cfg = config.get("opponent_data", {})
    server_cfg = {
        "device": device,
        "server_max_batch": int(opp_cfg.get("server_max_batch", 256)),
        "server_linger_ms": float(opp_cfg.get("server_linger_ms", 2)),
    }

    spec = []
    for a in agents_list:
        sd = {k: v.detach().cpu() for k, v in a["agent"].state_dict().items()}
        spec.append({"name": a["name"], "config": config,
                     "state_dict": sd, "norm_stats": a.get("norm_stats")})
    agents_meta = [{"name": a["name"], "norm_stats": a.get("norm_stats"),
                    "temperature": a.get("temperature")} for a in agents_list]

    # Move parent's agents off the GPU for the duration of parallel generation.
    # Mirrors `agent/mcts/collect.py:_run_parallel_mcts`: the inference server
    # holds its own GPU copies built from `spec`, parent does not use the live
    # models while waiting for actor results, so keeping both on CUDA doubles
    # pressure and OOMs the server during agent build (24GB cards, multi-agent).
    # Restored in the `finally` block below so downstream code finds them where
    # it left them.
    parent_devices = []
    if str(device).startswith("cuda"):
        for a in agents_list:
            parent_devices.append(getattr(a["agent"], "device_", "cpu"))
            a["agent"].cpu()
            a["agent"].device_ = "cpu"
        torch.cuda.empty_cache()
        log(f"  parallel opp_data: moved {len(agents_list)} parent agent(s) "
            f"to CPU during server lifetime (freed CUDA cache)")

    ctx = tmp.get_context("spawn")
    req_q = ctx.Queue(maxsize=max(64, 8 * n_workers))
    resp_qs = [ctx.Queue() for _ in range(n_workers)]
    # When an actor exits, the server's feeder for that resp_q will see EPIPE
    # on the next push. ignore_epipe makes it return silently instead of
    # spamming a traceback from a daemon thread.
    for q in resp_qs:
        q._ignore_epipe = True
    result_q = ctx.Queue()
    ready_event = ctx.Event()
    stop_event = ctx.Event()

    # Per-actor payload directory (disk transport — avoids the Python-3.14
    # mp.Queue large-payload loss). Cleaned at end of run.
    import tempfile as _tempfile
    import shutil as _shutil
    payload_dir = _tempfile.mkdtemp(prefix="opp_actor_payloads_")

    server = ctx.Process(
        target=server_main,
        args=(spec, req_q, resp_qs, ready_event, stop_event, server_cfg),
        daemon=True)
    server.start()
    if not ready_event.wait(timeout=600):
        stop_event.set()
        server.terminate()
        # Restore parent devices before propagating — caller may try to
        # reuse `agents_list` (e.g. retry, fallback to sequential).
        if parent_devices:
            for a, dev in zip(agents_list, parent_devices):
                try:
                    a["agent"].set_device(dev)
                except BaseException:
                    pass
        raise RuntimeError("inference server failed to become ready in 600s")

    base = n_hands // n_workers
    rem = n_hands % n_workers
    hands_per = [base + (1 if i < rem else 0) for i in range(n_workers)]
    offsets, acc = [], 0
    for h in hands_per:
        offsets.append(acc)
        acc += h

    # Shared progress counter — actors increment after each completed hand
    # under its built-in lock; a parent daemon thread polls and updates a
    # single tqdm for the whole generation. `progress=False` on actors
    # disables their local per-actor tqdm.
    progress_counter = ctx.Value("i", 0)

    actors = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_opp_actor_main,
            args=(wid, agents_meta, gen_cfg, hands_per[wid], offsets[wid],
                  1000 + wid, max_players, player_pool, swap_prob,
                  req_q, resp_qs[wid], result_q, False, payload_dir,
                  progress_counter),
            daemon=True)
        p.start()
        actors.append(p)

    log(f"Opponent parallel generation: {n_workers} actors, server on {device}, "
        f"max_batch={server_cfg['server_max_batch']}, hands/actor={hands_per}, "
        f"payload_dir={payload_dir}")

    # Single unified tqdm for the whole generation; updated by a daemon
    # thread reading `progress_counter`. Refreshes every 100ms.
    import threading as _threading
    pbar = tqdm(total=n_hands, desc="Generating opponent data")
    pbar_stop = _threading.Event()

    def _pbar_loop():
        while not pbar_stop.is_set():
            try:
                cur = progress_counter.value
                if cur != pbar.n:
                    pbar.n = cur
                    pbar.refresh()
            except BaseException:
                pass
            pbar_stop.wait(0.1)

    pbar_thread = _threading.Thread(target=_pbar_loop, daemon=True)
    pbar_thread.start()

    import pickle as _pickle
    import os as _os
    scenarios = []
    received_from = [False] * n_workers
    error = None

    def _consume(msg):
        nonlocal error
        tag = msg[0]
        if tag == "OK":
            _, wid_done, path = msg
            # Disk-transported payload: read & delete the file immediately.
            try:
                with open(path, "rb") as _f:
                    partial = _pickle.load(_f)
            except BaseException as _e:
                error = (f"opponent actor {wid_done} OK file unreadable "
                         f"({path}): {type(_e).__name__}: {_e}")
                return
            try:
                _os.unlink(path)
            except OSError:
                pass
            scenarios.extend(partial)
            received_from[wid_done] = True
        else:
            _, wid_done, tb = msg
            error = f"opponent actor {wid_done} failed:\n{tb}"

    try:
        while not all(received_from) and error is None:
            try:
                msg = result_q.get(timeout=1.0)
            except Exception:
                # Drain pending messages: an actor may have put its OK
                # on the pipe and exited cleanly between our get() timeout
                # and the is_alive() check below.
                while True:
                    try:
                        drained = result_q.get_nowait()
                    except Exception:
                        break
                    _consume(drained)
                    if error is not None:
                        break
                if error is not None or all(received_from):
                    break
                silent_dead = [
                    (wid, actors[wid].exitcode)
                    for wid, ok in enumerate(received_from)
                    if not ok and not actors[wid].is_alive()
                ]
                if silent_dead:
                    details = ", ".join(
                        f"actor {w} (exitcode={ec})" for w, ec in silent_dead)
                    error = (f"opponent actor(s) exited without reporting: "
                             f"{details} — check stderr above for traceback")
                    break
                if not server.is_alive() and server.exitcode not in (0, None):
                    error = "the inference server died unexpectedly"
                    break
                continue
            _consume(msg)

        stop_event.set()
        try:
            req_q.put(None)
        except Exception:
            pass
        for p in actors:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        server.join(timeout=30)
        if server.is_alive():
            server.terminate()
    finally:
        # Stop the tqdm refresh thread, flush the final count, close the bar.
        pbar_stop.set()
        try:
            pbar_thread.join(timeout=2)
        except BaseException:
            pass
        try:
            pbar.n = progress_counter.value
            pbar.refresh()
            pbar.close()
        except BaseException:
            pass
        # Always clean up the payload directory, even on error / Ctrl+C.
        try:
            _shutil.rmtree(payload_dir, ignore_errors=True)
        except BaseException:
            pass
        # Restore parent's agents to their original device(s). Done in
        # `finally` so even on RuntimeError above (server died, actor died)
        # the caller's downstream code finds the agents where it expects them.
        if parent_devices:
            for a, dev in zip(agents_list, parent_devices):
                try:
                    a["agent"].set_device(dev)
                except BaseException:
                    pass

    if error is not None:
        raise RuntimeError(error)

    return scenarios


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from utils import Logger

    parser = argparse.ArgumentParser(
        description="Generate range-based opponent action dataset")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    ver = os.path.basename(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    proj_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    exp_name = cfg.get("name", "default")

    opp_save = cfg.get("opponent_data", {}).get("save_dir", "")
    if opp_save and os.path.isabs(opp_save):
        save_dir = opp_save
    else:
        save_dir = os.path.join(proj_root, "data", ver, exp_name, "opponent_dataset")

    logger = Logger(os.path.join(proj_root, "data", ver, exp_name))
    generate_opponent_dataset(cfg, save_dir, dev, logger)
