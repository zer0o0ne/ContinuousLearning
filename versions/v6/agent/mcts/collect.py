"""
Training data extraction and self-play collection for MCTS.

Provides:
- collect_training_data(): extract MCTSTrainingExample from hand records
- run_mcts_collection(): play hands with MCTS, produce training data per agent
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from agent.mcts.mcts import (
    MCTS, get_n_distribution, _collect_terminals, action_path_from_root,
)
from agent.mcts.game_state import GameState
from env.table import Table
from evaluation.evaluate import _rebuild_events, _normalize_events_inplace

# `terminal_eval` pulls `agent.gto_utils.gpu_solver_v2`, which at import time
# requires `gpu_solver` on sys.path (a pipeline-level setup, not always done
# in unit-test contexts). We need its helpers only inside `run_mcts_collection`
# at runtime, so import lazily there instead of at module load — keeps
# `collect.py` importable for legacy / training-script callers that never
# trigger equity evaluation.


def _materialize_past(spec, config, device, big_blind, log):
    """Load weights from `spec['ckpt_path']` into a fresh ASI on `device`.

    Returns an agent_info dict shaped like entries in `agents_list` so the
    collection code path treats past snapshots identically to active agents.

    Cheap GRU table omitted: past agents do NOT share an opponent_emb_table —
    they're seated only briefly between reshuffles, no GRU accumulation is
    meaningful.
    """
    from agent.agent import ASI
    asi = ASI(lambda *a, **k: None, config)
    asi.set_device(device)
    ckpt = torch.load(
        spec["ckpt_path"], weights_only=False, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    asi.load_state_dict(state_dict, strict=False)
    asi.eval()
    ns = ckpt.get("norm_stats") or {}
    temp = ckpt.get("temperature", 1.0)
    search_scale = float(ns.get("mcts_value_scale", float(big_blind)))
    log(f"  [past] materialised {spec['name']} on {device} "
        f"(temp={temp:.3f}, search_scale={search_scale:.2f})")
    return {
        "agent": asi,
        "norm_stats": ns,
        "name": spec["name"],
        "temperature": float(temp),
        "is_active": False,
        "is_past": True,
        "search_scale": search_scale,
    }


def _sample_past_action(agent_info, norm_events, legal_actions, n_actions,
                        device):
    """Sample an action for a past-snapshot agent: one action_head forward,
    softmax with temperature + legal-action mask, multinomial sample.

    Returns ``(action_idx, action_distribution_over_full_space)`` where
    `action_distribution_over_full_space` is a length-`n_actions` list of
    floats (illegal slots are 0). The distribution serves as the chain-
    target for opponent_action_head when this state is referenced in any
    active hero's chain.
    """
    import torch.nn.functional as F
    asi = agent_info["agent"]
    with torch.no_grad():
        out = asi.forward_batch(
            [norm_events], skip_memory=True, heads={"action"},
            skip_opponent_emb=True, opponent_emb_table=None)
    logits = out["action_logits"][0].to("cpu")  # (n_actions,)
    legal_mask = torch.zeros(n_actions, dtype=torch.bool)
    legal_mask[list(legal_actions)] = True
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    temp = max(1e-3, float(agent_info.get("temperature", 1.0)))
    probs = F.softmax(masked / temp, dim=0)
    if not torch.isfinite(probs).all() or probs.sum().item() <= 0.0:
        # Pathological case (all illegal → -inf everywhere). Fall back to
        # uniform over legal.
        probs = legal_mask.float()
        probs = probs / probs.sum().clamp(min=1e-8)
    action_idx = int(torch.multinomial(probs, 1).item())
    return action_idx, probs.tolist()


def _reseat_with_past(agents_list, past_specs, min_players, max_players,
                       materialized_past, device, config, big_blind, log):
    """Pick a new num_players + seating subject to floor(n/2) active.

    Updates `materialized_past` in place: loads past snapshots needed for the
    new seating; drops past snapshots no longer seated and frees CUDA cache.
    Moves active agents on/off `device` based on whether they're seated.

    Returns the new `hand_seated` list (length num_players) of agent_info
    dicts (active live entries OR materialized past entries).
    """
    num_players = random.randint(min_players, max_players)
    n_forced_active = num_players // 2
    n_free = num_players - n_forced_active

    # Forced active seats: uniform over active pool (with replacement).
    forced = random.choices(agents_list, k=n_forced_active)

    # Free seats: uniform over (active ∪ past_specs). We mix the two pools
    # element-wise (each past spec is one candidate); resolution to a live
    # ASI happens after we know which spec ids made it.
    pool = list(agents_list) + list(past_specs or [])
    free_picks = random.choices(pool, k=n_free) if pool else []

    # Resolve seating: anything that looks like a spec dict (has "ckpt_path")
    # is a past snapshot we need to materialize.
    seated = []
    seated_past_names = set()
    for entry in forced + free_picks:
        if isinstance(entry, dict) and "ckpt_path" in entry:
            name = entry["name"]
            seated_past_names.add(name)
            if name not in materialized_past:
                materialized_past[name] = _materialize_past(
                    entry, config, device, big_blind, log)
            seated.append(materialized_past[name])
        else:
            seated.append(entry)

    # Drop past snapshots no longer seated → frees their GPU memory.
    stale = [n for n in materialized_past.keys() if n not in seated_past_names]
    for n in stale:
        log(f"  [past] dropping {n} from materialized pool")
        del materialized_past[n]
    if stale and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Move active agents on/off device based on seating membership. The
    # `device_` attribute tracks where ASI parameters currently live (set by
    # ASI.set_device); we mirror the parallel-mode pattern in
    # `_run_parallel_collection` for keeping it consistent with a manual
    # `.cpu()` move.
    active_on_table = {
        a["name"] for a in seated
        if not isinstance(a, dict) or "ckpt_path" not in a
    }
    # ^^ a in `seated` is always a dict here (live agent_info); the
    # ckpt_path test will be False for active entries.
    for a in agents_list:
        cur = getattr(a["agent"], "device_", "cpu")
        if a["name"] in active_on_table:
            if str(cur) != str(device):
                a["agent"].set_device(device)
        else:
            if str(cur) != "cpu":
                a["agent"].cpu()
                a["agent"].device_ = "cpu"
    if str(device).startswith("cuda"):
        # Cleanup after CPU offloads + past drops.
        torch.cuda.empty_cache()

    return seated, num_players


@dataclass
class ChainStep:
    """One step in the modelling chain: action taken + target distribution.

    Fields populated for new training signals (back-compat with older pickles
    via defaults — empty `events_at_step` disables recon/TF/value-at-step for
    that ChainStep):
      events_at_step: events_at_root of decisions[t+1+i] (normalized, ready
        to feed into perception). Used for the reconstruction target
        (modelling embedding ≈ pooled perception of this state) AND for
        teacher forcing (substitute rolled ctx with real perception of this
        state). May be empty when loaded from legacy collections.
      value_target: raw outcome ratio for the ORIGINAL hero (the player who
        decided at the root state) measured at state(t+1+i), **scaled by
        the SAME denom as the root example** so root + chain targets stay on
        a single shared scale. MCTS value_head outputs across nodes at
        different depths are all on the same per-agent `mcts_value_scale`;
        using a local denom per chain step would create a per-depth scale
        mismatch. Same mcts_ev_* stats then apply to root + chain uniformly.
        Pre-z-score stored here; the caller z-scores at the end.
      root_q_ratio: MCTS `root.Q` of the **future** tree at decisions[t+1+i],
        de-z-scored (back to ratio space) and rescaled into THIS example's
        root_denom — so it lives on the SAME scale as `value_target`. After
        z-scoring with the (possibly new) mcts_ev_* stats, blended with the
        realized ratio via `value_target_alpha` to produce the final
        value-head training target. NaN ⇒ TD-blend unavailable (legacy
        data) → caller falls back to pure MC for this step.
    """
    action_taken: int
    target_distribution: list
    is_hero: bool  # True → action_head predicts, False → opponent_action_head
    events_at_step: list = field(default_factory=list)
    value_target: float = 0.0
    root_q_ratio: float = float("nan")


@dataclass
class MCTSTrainingExample:
    """Training data from one MCTS tree.

    events: normalized event sequence at this tree's root
    value_target: per-decision realised outcome, z-scored using
        MCTS-specific bootstrapped stats:
            ratio = outcome_chips_from_decision / max(pot + facing_bet, BB)
            value_target = (ratio - mcts_ev_mean) / mcts_ev_std
        Outcome is the chip delta from THIS decision to hand end (excludes
        sunk costs prior to the decision). The z-score stats live as new
        top-level keys in `norm_stats` (`mcts_ev_mean`, `mcts_ev_std`,
        `mcts_ev_n_samples`, `mcts_ev_ratio_min`, `mcts_ev_ratio_max`),
        added on the first collection that has data and persisted in
        checkpoints. The original gto_ev_predict stats (ev_mean/ev_std/
        pot/stack/etc) stay untouched — they keep driving event-input
        z-scoring. MCTS realised-outcome variance is much wider than
        solver-EV variance, so MCTS needs its own calibration.
    action_target: root N-distribution (visit count proportions)
    chain: list of ChainSteps. chain[i].action_taken is the action chosen at
        the decision *i steps before* chain[i]'s state — i.e. for chain[0] it
        is the root decision's action (the one that advances state(t)→state(t+1));
        for chain[i] (i≥1) it is decisions[t+i].action_idx (advances state(t+i)
        →state(t+i+1)). target_distribution at chain[i] is the N-distribution
        AT decisions[t+1+i] (i.e. the state we land in after applying
        action_taken at the prior context). This matches MCTS inner-node
        semantics (mcts.py: child.action_embedding = modelling(parent_ctx)[a]).
    """
    events: list = field(default_factory=list)
    value_target: float = 0.0
    action_target: list = field(default_factory=list)
    chain: list = field(default_factory=list)
    # MCTS root.Q for THIS decision, de-z-scored into the example's root_denom
    # ratio space. Used for the TD half of the hybrid `α·Q + (1−α)·realized`
    # value-head target (see run_mcts_collection). NaN ⇒ TD-blend unavailable
    # (legacy data) → caller falls back to pure MC.
    root_q_ratio: float = float("nan")
    # Tree-terminal value targets for direct value_head supervision. Each entry
    # is `(action_path_from_root, equity_Q)` where `action_path_from_root` is
    # the list of `action_idx` to walk from root to that terminal, and
    # `equity_Q` is the equity-based terminal value (in `mcts_value_scale`
    # units, matching the rest of the value-target axis). Selected at
    # collection time as K_worst lowest-Q + K_best highest-Q terminals from
    # the tree — sampling at both tails avoids biasing the value head toward
    # only "good" outcomes. Empty when k_worst+k_best=0 (disabled).
    terminal_targets: list = field(default_factory=list)


def _select_terminal_targets(root, k_worst, k_best, clip_val=None):
    """Pick `k_worst` lowest-Q + `k_best` highest-Q terminals from the tree.

    Returns a list of `(action_path_from_root, equity_Q)` tuples ready to
    drop into `MCTSTrainingExample.terminal_targets`. Terminals are sorted
    by `terminal.Q` (which `evaluate_all_terminals` has set to the
    equity-based value in `mcts_value_scale` units). Both tails are taken
    to avoid biasing value-head training toward only successful outcomes;
    overlap (small trees) is handled by union of indices.

    `clip_val`: if not None, each `equity_Q` is clamped to `[-clip_val,
    clip_val]` — matches the `value_target_clip` applied to root/chain
    value targets in `_finalize_value_targets`, so the value head's tail
    supervision lives on the same bounded axis as its other targets.

    Returns `[]` when `k_worst + k_best == 0`.
    """
    k_worst = max(0, int(k_worst))
    k_best = max(0, int(k_best))
    if k_worst + k_best == 0:
        return []
    terminals = _collect_terminals(root)
    if not terminals:
        return []
    # Stable sort: ascending Q. Tail K_best are the highest.
    sorted_terms = sorted(terminals, key=lambda t: float(t.Q))
    n = len(sorted_terms)
    picked = set()
    for i in range(min(k_worst, n)):
        picked.add(i)
    for i in range(max(0, n - k_best), n):
        picked.add(i)
    out = []
    for i in sorted(picked):
        t = sorted_terms[i]
        q = float(t.Q)
        if clip_val is not None:
            q = max(-float(clip_val), min(float(clip_val), q))
        out.append((action_path_from_root(t), q))
    return out


def collect_training_data(hand_record, n_actions, max_chain_depth=None,
                          final_credits=None, big_blind=None,
                          action_label_smoothing=0.0,
                          terminal_k_worst=0, terminal_k_best=0,
                          terminal_clip_val=None):
    """Extract training examples from all MCTS trees in a completed hand.

    For each tree (one per decision point), produces an MCTSTrainingExample.
    `value_target` is in BB units: `chip_delta_from_state_to_end / BB`. The
    same scale is used at root and at every chain step — no state-dependent
    denom, no mean shift, no z-score. `run_mcts_collection` then optionally
    blends with `root_q_ratio` via `value_target_alpha`.

    Chain semantics: chain[i].action_taken advances context from state(t+i)
    to state(t+i+1). chain[i].target_distribution is the predicted distribution
    AT state(t+i+1) (after applying that action).

    `root_q_ratio` stores MCTS `root.Q` (the backed-up value from the tree at
    that decision). After `evaluate_all_terminals` + `re_backup_terminals`
    have run, `root.Q` is a mix of value-head outputs at non-terminal leaves
    and equity-based terminal contributions, both in `search_scale` units
    (= the cycle's `mcts_value_scale`). `_finalize_value_targets` then
    rescales by `search_scale / new_scale` onto the current target axis.

    Args:
        hand_record: dict with "decisions" list — each entry has:
            player_pos, action_idx, mcts_root, events_at_root,
            pot_at_decision, all_credits_before_decision (optional)
        n_actions: int, action space size
        max_chain_depth: optional int — truncate chain to first N future
            decisions. None = full chain.
        final_credits: optional list[float] of credits at hand end. Required
            to compute per-step chain value targets.
        big_blind: optional int. Constant scale factor for value_target.
        action_label_smoothing: ε for the KL target on both root and chain
            visit distributions. Mixes `N/total` with uniform over legal
            actions to keep mass on rarely-visited but legal moves; see
            ``mcts.get_n_distribution``. Same ε is applied to root
            (action_head) and chain (action_head / opponent_action_head)
            targets so they live on the same calibrated scale.

    Returns:
        list of (decision_idx, MCTSTrainingExample) tuples — one per tree.
        Past-opponent decisions (where ``decision["mcts_root"] is None``)
        are skipped, so the list may be shorter than ``decisions``; the
        original index lets callers align with ``realized_by_dec`` /
        ``decisions``.
    """
    decisions = hand_record["decisions"]
    examples = []
    has_value_targets = (final_credits is not None and big_blind is not None)

    for t, decision in enumerate(decisions):
        hero_pos = decision["player_pos"]
        root = decision["mcts_root"]
        # Past-opponent decisions have no MCTS tree — no training example
        # is produced for them (they're not trained), but they DO contribute
        # to game-state evolution and chain steps of other (active) examples.
        if root is None:
            continue
        action_target = get_n_distribution(
            root, n_actions, label_smoothing=action_label_smoothing)

        # `root.Q` is in `search_scale` units after re_backup_terminals
        # (terminals divided by `value_scales_by_position[hero_pos]` in
        # `evaluate_all_terminals`, value-head outputs trained on the same
        # axis). `_finalize_value_targets` lifts it to the new target scale.
        example_root_q_ratio = float(root.Q)

        # Modelling chain: chain[i] predicts distribution at decisions[t+1+i],
        # using the action taken at decisions[t+i] to advance context.
        chain = []
        for i, future_dec in enumerate(decisions[t + 1:]):
            if max_chain_depth is not None and i >= max_chain_depth:
                break
            future_root = future_dec["mcts_root"]
            # action that advances state(t+i) → state(t+i+1):
            action_taken = decisions[t + i]["action_idx"]
            events_at_step = future_dec.get("events_at_root", []) or []
            if future_root is None:
                # Future decision was taken by a past-snapshot opponent —
                # no MCTS tree exists. The opponent's actual policy at that
                # state is their action_head distribution (computed when the
                # decision was made) and stored as fallback_action_distribution.
                # That IS their "true" policy in this collection, so it's the
                # right KL target for opponent_action_head.
                target_dist = list(future_dec.get("fallback_action_distribution") or [])
                if len(target_dist) != n_actions:
                    target_dist = [1.0 / n_actions] * n_actions
                # No root.Q to TD-blend against → caller falls back to pure MC.
                step_root_q_ratio = float("nan")
            else:
                target_dist = get_n_distribution(
                    future_root, n_actions, label_smoothing=action_label_smoothing)
                # C.2: `future_root.Q` is the backed-up value of the FUTURE
                # tree — from the perspective of whoever decided there and on
                # THAT agent's value scale. It is a valid TD anchor for THIS
                # example's value target only when the future decision is the
                # same hero/seat (within a hand a seat is one fixed agent, so
                # same seat ⇒ same perspective AND same scale). Otherwise write
                # NaN → pure-MC fallback in `_finalize_value_targets` (no
                # cross-perspective / cross-scale TD contamination).
                if future_dec["player_pos"] == hero_pos:
                    step_root_q_ratio = float(future_root.Q)
                else:
                    step_root_q_ratio = float("nan")

            value_target_step = 0.0
            if has_value_targets:
                all_creds = future_dec.get("all_credits_before_decision")
                if all_creds is not None and hero_pos < len(all_creds):
                    hero_credits = float(all_creds[hero_pos])
                    # Store RAW chip delta. Normalization happens later in
                    # run_mcts_collection where we have access to the
                    # bootstrapped value scale.
                    value_target_step = float(final_credits[hero_pos]
                                                - hero_credits)

            chain.append(ChainStep(
                action_taken=action_taken,
                target_distribution=target_dist,
                is_hero=(future_dec["player_pos"] == hero_pos),
                events_at_step=events_at_step,
                value_target=value_target_step,
                root_q_ratio=step_root_q_ratio,
            ))

        # C.3: store terminal Q RAW (in `search_scale` units, like `root.Q`).
        # `_finalize_value_targets` rescales by `search_scale/new_scale` onto
        # the fresh target axis and clips THERE — clipping on the stale axis at
        # collection time would turn a scale mismatch into sign-only noise. The
        # pipeline therefore passes `terminal_clip_val=None` here (no
        # collection-time clip); the param stays for explicit callers/tests.
        terminal_targets = _select_terminal_targets(
            root, terminal_k_worst, terminal_k_best,
            clip_val=terminal_clip_val)

        # Tuple keeps the original decision index `t` so callers can map back
        # into `decisions` / `realized_by_dec` even when past-opponent
        # decisions skip example creation (no positional alignment with
        # `decisions`).
        examples.append((t, MCTSTrainingExample(
            events=decision["events_at_root"],
            value_target=root.Q,  # placeholder; overwritten in run_mcts_collection
            action_target=action_target,
            chain=chain,
            root_q_ratio=example_root_q_ratio,
            terminal_targets=terminal_targets,
        )))

    return examples


def run_mcts_collection(agents_list, config, device, log, n_hands,
                          cycle_idx=0, n_cycles=1,
                          past_snapshot_specs=None):
    """Play hands with MCTS decisions and collect training examples.

    Each agent uses MCTS for its decisions. After each hand, training
    examples are extracted using actual game outcomes as value targets.

    Number of players is randomized per hand between min_players and
    max_players. Seated agents are drawn from the pool and swapped
    with player_swap_prob between hands (like opponent_action generation).

    `cycle_idx` / `n_cycles` drive the strange-traversal schedule:
        ``C(t) = C_start * 0.5 * (1 + cos(π * t / n_cycles))``
    starts at `mcts.strange_traversal_C_start` (config) and decays to 0 by
    the last cycle. Per-agent probability is
        ``p_strange = C(t) * exp(-last_action_loss)``
    where `last_action_loss` is the mean KL of `action_head` over the
    previous training cycle, stored in `agent_info["norm_stats"]
    ["last_action_loss"]` by `train_mcts`. First-ever cycle (no prior loss):
    `p_strange = 0`.

    Args:
        agents_list: list of dicts with keys:
            "agent" (ASI), "norm_stats" (dict), "name" (str), "temperature" (float)
        config: full config dict (game, mcts, mcts_train, etc.)
        device: torch device string
        log: logger callable
        n_hands: number of hands to play
        cycle_idx: current MCTS cycle index (0-based)
        n_cycles: total number of MCTS cycles in this run
        past_snapshot_specs: optional list of past-snapshot specs (each:
            ``{"name", "agent_name", "ckpt_path", "cycle_id"}``). When
            non-empty AND running in sequential mode (``mcts_train.n_workers
            <= 1``), past snapshots join active agents in seating: each
            reshuffle keeps at least ``floor(num_players / 2)`` seats for
            active agents and samples remaining seats uniformly from
            active∪past. Past agents pick actions via
            ``softmax(action_head)`` (no MCTS) and generate no training
            examples (their model never trains in this loop). Disabled with
            a warning when ``n_workers > 1`` — the inference server's spec
            is frozen at startup, so on-demand past loading is not yet
            supported there.

    Returns:
        dict mapping agent_name -> list[MCTSTrainingExample]
    """
    from agent.train_scenarios.generation.generate import _get_raise_sizes
    from agent.mcts.terminal_eval import (
        evaluate_all_terminals, compute_equity_outcome,
    )

    game_cfg = config.get("game", {})
    mcts_cfg = config.get("mcts", {})
    mcts_train_cfg = config.get("mcts_train", {})
    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3
    big_blind = game_cfg.get("big_blind", 10)
    small_blind = big_blind // 2
    abs_max_players = game_cfg.get("max_players", 9)
    min_players = max(2, mcts_train_cfg.get("min_players", 2))
    max_players = min(abs_max_players, mcts_train_cfg.get("max_players", 6))
    max_players = max(min_players, max_players)
    swap_prob = mcts_train_cfg.get("player_swap_prob", 0.05)
    min_stack = mcts_train_cfg.get("min_stack", big_blind * 10)
    max_stack = mcts_train_cfg.get("max_stack", game_cfg.get("max_stack", 3000))
    raw_max_chain_depth = mcts_train_cfg.get("max_chain_depth", None)
    max_chain_depth = None if not raw_max_chain_depth else int(raw_max_chain_depth)
    action_label_smoothing = float(
        mcts_train_cfg.get("action_label_smoothing", 0.0))
    terminal_k_worst = int(mcts_train_cfg.get("terminal_value_k_worst", 0))
    terminal_k_best = int(mcts_train_cfg.get("terminal_value_k_best", 0))

    per_agent_examples = {a["name"]: [] for a in agents_list}
    MAX_ACTIONS = 10000

    # Per-agent OpponentEmbeddingTable. Used at MCTS root perception call so
    # search sees the same opponent context as supervised training does. Each
    # agent has its own table because GRU updates are agent-specific (the GRU
    # lives inside Perception). Tables persist across hands within this call,
    # mirroring how training accumulates embeddings within an epoch.
    opp_tables = {}
    for a in agents_list:
        asi = a["agent"]
        if asi.perception.opp_emb_enabled:
            from agent.perception.opponent_embeddings import OpponentEmbeddingTable
            opp_tables[a["name"]] = OpponentEmbeddingTable(asi.perception.d_model)
    if opp_tables:
        log(f"MCTS collection: opponent_embedding active at root for "
            f"{len(opp_tables)}/{len(agents_list)} agent(s)")

    # Per-agent search-time value scale: the per-cycle `mcts_value_scale`
    # snapshot used by `evaluate_all_terminals` (via `value_scales_by_position`)
    # to divide equity-based terminal Q into the same axis as the value head's
    # outputs at non-terminal leaves. Captured once at the start so the final
    # normalization can convert root.Q back into chips even if
    # mcts_value_scale is re-bootstrapped later in the same call. Fallback to
    # BB on the first-ever cycle (no bootstrap yet).
    search_scales = {}
    for a in agents_list:
        ns = a.get("norm_stats") or {}
        search_scales[a["name"]] = float(
            ns.get("mcts_value_scale", float(big_blind)))

    log(f"MCTS collection: {n_hands} hands, players={min_players}-{max_players}, "
        f"stack={min_stack}-{max_stack}, swap_prob={swap_prob}, "
        f"{mcts_cfg.get('n_simulations', 1000)} simulations/decision")
    for name_, sc_ in search_scales.items():
        log(f"  {name_}: search_scale = {sc_:.2f} chips "
            f"({'bootstrapped' if sc_ != float(big_blind) else 'BB fallback'})")

    # ── Strange-traversal schedule ───────────────────────────────────────
    # C(t) = C_start * 0.5 * (1 + cos(π * t / n_cycles)) — cosine decay from
    # C_start (at cycle 0) to 0 (at the final cycle). Per-agent
    # `p_strange = C(t) * exp(-last_action_loss)` where `last_action_loss`
    # comes from `norm_stats`. Missing key (first cycle, fresh agent) →
    # p_strange = 0.
    strange_C_start = float(mcts_cfg.get("strange_traversal_C_start", 0.3))
    nc = max(1, int(n_cycles))
    if nc > 1:
        frac = max(0.0, min(1.0, float(cycle_idx) / float(nc)))
        C_t = strange_C_start * 0.5 * (1.0 + math.cos(math.pi * frac))
    else:
        C_t = strange_C_start
    strange_p_by_agent = {}
    for a in agents_list:
        last_loss = (a.get("norm_stats") or {}).get("last_action_loss")
        if last_loss is None:
            strange_p_by_agent[a["name"]] = 0.0
        else:
            strange_p_by_agent[a["name"]] = float(C_t) * float(
                math.exp(-float(last_loss)))
    log(f"  strange traversal: C_start={strange_C_start:.3f}, "
        f"cycle={cycle_idx}/{nc}, C_t={C_t:.4f}")
    for a in agents_list:
        name_ = a["name"]
        last_loss = (a.get("norm_stats") or {}).get("last_action_loss")
        loss_str = f"{last_loss:.4f}" if isinstance(last_loss, (int, float)) else "n/a"
        log(f"  {name_}: p_strange = {strange_p_by_agent[name_]:.4f} "
            f"(last_action_loss={loss_str})")

    # ── Dispatch: sequential (n_workers<=1) vs parallel CPU actors + GPU server ──
    n_workers = int(mcts_train_cfg.get("n_workers", 1) or 1)
    if n_workers > 1:
        if past_snapshot_specs:
            log(f"  WARNING: past_opponents pool ({len(past_snapshot_specs)} "
                f"snapshot(s)) is disabled in parallel mode (n_workers={n_workers}) "
                f"— inference-server spec is frozen at startup. Seating "
                f"falls back to active-only.")
        per_agent_examples = _run_parallel_collection(
            agents_list, config, device, log, n_hands, n_workers,
            search_scales, strange_p_by_agent, cycle_idx, n_cycles)
    else:
        def make_mcts(agent_info, gs):
            asi = agent_info["agent"]
            asi.eval()
            return MCTS(asi, device, mcts_cfg,
                        opponent_emb_table=opp_tables.get(agent_info["name"]),
                        strange_p=strange_p_by_agent[agent_info["name"]],
                        search_scale=search_scales.get(
                            agent_info["name"], float(big_blind)))

        per_agent_examples = _play_hands(
            agents_list, config, device, n_hands, make_mcts=make_mcts,
            terminal_proxy=None, equity_device=device,
            search_scales=search_scales, strange_p_by_agent=strange_p_by_agent,
            log=log, progress=True,
            past_snapshot_specs=past_snapshot_specs)

    # Per-agent bootstrap + hybrid + clip. See
    # `versions/v5/PLAN_MCTS_VALUE_REDESIGN.md` §4 + §5 for math.
    alpha = float(mcts_train_cfg.get("value_target_alpha", 0.5))
    clip_val = float(mcts_train_cfg.get("value_target_clip", 5.0))
    _finalize_value_targets(
        per_agent_examples=per_agent_examples,
        agents_list=agents_list,
        search_scales=search_scales,
        alpha=alpha,
        clip_val=clip_val,
        big_blind=float(big_blind),
        log=log,
    )

    for name, exs in per_agent_examples.items():
        log(f"  {name}: {len(exs)} training examples")

    return per_agent_examples


def _play_hands(agents_list, config, device, n_hands, make_mcts,
                terminal_proxy, equity_device, search_scales,
                strange_p_by_agent, log, progress=True,
                progress_counter=None, past_snapshot_specs=None):
    """Play `n_hands` hands and return per-agent MCTSTrainingExamples.

    Value targets are RAW chip deltas at this stage — `_finalize_value_targets`
    runs ONCE in the parent over all merged examples (so the per-agent
    `mcts_value_scale` bootstrap sees every example). `make_mcts(agent_info, gs)`
    builds the MCTS for the acting agent with the appropriate evaluator
    (LocalEvaluator in sequential mode, RemoteEvaluator in an actor).
    `terminal_proxy` is None in sequential mode (terminal equity uses the live
    ASI on `device`); in an actor it is an EvalProxy routing range-narrowing
    forwards to the server, with the equity Monte Carlo on `equity_device`.

    Past-snapshot opponents (`past_snapshot_specs` non-empty, sequential mode
    only) are seated alongside active agents on reshuffle. Each reshuffle
    keeps at least ``floor(num_players / 2)`` active seats; remaining seats
    sample uniformly from active∪past. Past agents pick actions by
    softmax-sampling their `action_head` (no MCTS), and contribute to
    chain-step targets via that distribution; they produce no training
    examples themselves.
    """
    from agent.train_scenarios.generation.generate import _get_raise_sizes
    from agent.mcts.terminal_eval import (
        evaluate_all_terminals, compute_equity_outcome,
    )

    game_cfg = config.get("game", {})
    mcts_cfg = config.get("mcts", {})
    mcts_train_cfg = config.get("mcts_train", {})
    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3
    big_blind = game_cfg.get("big_blind", 10)
    small_blind = big_blind // 2
    abs_max_players = game_cfg.get("max_players", 9)
    min_players = max(2, mcts_train_cfg.get("min_players", 2))
    max_players = min(abs_max_players, mcts_train_cfg.get("max_players", 6))
    max_players = max(min_players, max_players)
    swap_prob = mcts_train_cfg.get("player_swap_prob", 0.05)
    min_stack = mcts_train_cfg.get("min_stack", big_blind * 10)
    max_stack = mcts_train_cfg.get("max_stack", game_cfg.get("max_stack", 3000))
    raw_max_chain_depth = mcts_train_cfg.get("max_chain_depth", None)
    max_chain_depth = None if not raw_max_chain_depth else int(raw_max_chain_depth)
    action_label_smoothing = float(
        mcts_train_cfg.get("action_label_smoothing", 0.0))
    terminal_k_worst = int(mcts_train_cfg.get("terminal_value_k_worst", 0))
    terminal_k_best = int(mcts_train_cfg.get("terminal_value_k_best", 0))

    per_agent_examples = {a["name"]: [] for a in agents_list}
    MAX_ACTIONS = 10000

    # Tag live agents as `is_active` so seating / decision dispatch can
    # distinguish them from materialised past snapshots. We mutate the dicts
    # in place — the same objects are shared with the caller, and the flag
    # is harmless for downstream consumers.
    for a in agents_list:
        a.setdefault("is_active", True)
        a.setdefault("is_past", False)

    past_pool = list(past_snapshot_specs or [])
    # Registry of currently-materialised past snapshots: name → agent_info.
    # Updated only by `_reseat_with_past` (so creation cost is one-shot per
    # reshuffle, not per hand).
    materialized_past = {}
    has_past = bool(past_pool)

    # Initial table: respect the floor(n/2)-active constraint from the start
    # so the first hand isn't trivially active-only by accident.
    if has_past:
        hand_seated, num_players = _reseat_with_past(
            agents_list, past_pool, min_players, max_players,
            materialized_past, device, config, big_blind, log)
    else:
        num_players = random.randint(min_players, max_players)
        hand_seated = random.choices(agents_list, k=num_players)

    # When `progress_counter` is set (parallel actor), forward per-hand
    # progress to the shared counter that the parent's tqdm thread polls,
    # and skip our own tqdm. Sequential mode keeps its local tqdm.
    if progress_counter is not None:
        hand_iter = range(n_hands)
    else:
        hand_iter = (tqdm(range(n_hands), desc="MCTS collection")
                     if progress else range(n_hands))
    for hand_i in hand_iter:
        # With swap_prob, reshuffle the entire table: new count + new agents
        if random.random() < swap_prob:
            if has_past:
                hand_seated, num_players = _reseat_with_past(
                    agents_list, past_pool, min_players, max_players,
                    materialized_past, device, config, big_blind, log)
            else:
                num_players = random.randint(min_players, max_players)
                hand_seated = random.choices(agents_list, k=num_players)
        seated_names = [a["name"] for a in hand_seated]

        dummy_action = torch.zeros(n_actions, dtype=torch.float32)
        # B.6.1: independent per-seat starting stacks (asymmetric effective
        # stacks). collect derives invested chips from initial_credits +
        # cumulative_bets (not table.start_credits), so per-seat credits suffice.
        start_stacks = [random.randint(min_stack, max_stack) for _ in range(num_players)]

        table = Table(
            num_players=num_players,
            raise_sizes=raise_sizes,
            start_credits=start_stacks,
            big_blind=big_blind,
            small_blind=small_blind,
        )
        table.start_table()
        initial_credits = list(table.credits)
        # `cumulative_bets[p]` tracks every chip player `p` puts into the pot
        # AFTER `start_table()` (excludes the blinds, which are already baked
        # into `initial_credits`). Used post-hand to derive
        # `credits_pre_distribution = initial_credits − cumulative_bets`,
        # i.e. credits after all betting but BEFORE the pot is redistributed.
        # `final_credits` mixes in winnings (and the engine's pot-distribution
        # has a known mis-accounting on showdowns where prior-street chips
        # never reach winners) — so we rely on `credits_pre_distribution`
        # to compute hero's `chips_invested_from(t)_to_end`, which is what
        # the equity-based realized outcome needs.
        cumulative_bets = np.zeros(num_players, dtype=np.float64)
        # Snapshot of the actual hands dealt — needed for equity-based
        # terminal/realized evaluation in `compute_equity_outcome` &
        # `evaluate_all_terminals` (`agents_by_position` derives opponent
        # ranges; hero cards come from here).
        hero_hands = {
            p: list(table.deck[5 + 2 * p: 7 + 2 * p])
            for p in range(num_players)
        }
        deck_snapshot = np.array(table.deck, copy=True)

        # Snapshots for event reconstruction
        snapshots = [{
            "pot": table.pot,
            "bets": np.copy(table.bets),
            "credits": list(table.credits),
            "turn": table.turn,
            "active_pos": table.active_player,
            "action": None,
        }]

        decisions = []
        action_step = 0
        hand_done = False

        while not hand_done and action_step < MAX_ACTIONS:
            # Advance all-in runouts without MCTS — no chips move in this
            # mode (Table.step short-circuits on several_all_in), so we
            # don't update cumulative_bets here.
            while table.several_all_in and action_step < MAX_ACTIONS:
                end, _, _, _ = table.step(dummy_action)
                action_step += 1
                if end:
                    hand_done = True
                    break
            if hand_done:
                break

            active_pos = table.active_player
            if table.players_state[active_pos] != 1:
                hand_done = True
                break

            # Pre-decision snapshot
            snapshots.append({
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": active_pos,
                "action": None,
            })

            agent_info = hand_seated[active_pos]

            # Build and normalize events for this agent
            events = _rebuild_events(
                snapshots, table.deck, active_pos,
                num_players, big_blind, small_blind, n_actions,
                up_to=len(snapshots) - 1,
                seated_names=seated_names,
            )
            norm_events = [dict(e) for e in events]  # shallow copy before mutation
            for e in norm_events:
                if isinstance(e["bets"], np.ndarray):
                    e["bets"] = np.copy(e["bets"])
            _normalize_events_inplace(norm_events, agent_info["norm_stats"])

            # Capture state at decision time (in raw chips, BEFORE action
            # is applied) — needed to normalize value_target identically to
            # gto_ev_predict: (outcome / max(pot + facing_bet, BB) - ev_m)/ev_s.
            pot_at_dec = float(table.pot)
            facing_at_dec = float(table.high_bet - table.bets[active_pos])
            credits_at_dec = float(table.credits[active_pos])

            # MCTS search (opponent_emb at root only — see MCTS._evaluate_root).
            # Terminals DO contribute value during search (C.4): fold terminals
            # get a deterministic chip value (no NN, the anti fold-spiral
            # anchor), showdown terminals get the value head. Their proper
            # game-theoretic Q is then recomputed post-hand by
            # `evaluate_all_terminals` (equity + range narrowing) and propagated
            # via `re_backup_terminals`, overriding the search-time estimate.
            gs = GameState.from_table(table, active_pos)
            is_past = bool(agent_info.get("is_past", False))
            if is_past:
                # Past snapshots act via `softmax(action_head)` sampling — no
                # MCTS, no tree. The sampled distribution is stored on the
                # decision as a fallback target for any active hero's chain
                # step that references this state (see
                # `collect_training_data` chain branch on `future_root is None`).
                legal = gs.get_legal_actions()
                action_idx, fallback_dist = _sample_past_action(
                    agent_info, norm_events, legal, n_actions, device)
                last_root = None
            else:
                mcts = make_mcts(agent_info, gs)
                action_idx = mcts.search([norm_events], gs)
                last_root = mcts.last_root
                fallback_dist = None

            decisions.append({
                "player_pos": active_pos,
                "action_idx": action_idx,
                "mcts_root": last_root,
                "fallback_action_distribution": fallback_dist,
                "events_at_root": norm_events,
                "pot_at_decision": pot_at_dec,
                "facing_bet_at_decision": facing_at_dec,
                "credits_before_decision": credits_at_dec,
                # Snapshot of ALL players' credits/bets at this decision
                # — needed by collect_training_data to compute per-step chain
                # value targets from the original hero's perspective (point 11)
                # regardless of who is acting at the future state.
                "all_credits_before_decision": list(table.credits),
                "bets_before_decision": np.copy(table.bets),
                "agent_norm_stats": agent_info["norm_stats"],
                # GameState snapshot at decision time — needed by Step 2's
                # `evaluate_all_terminals` to replay each MCTS terminal node
                # forward from the root for equity-based showdown valuation.
                # Cloned because MCTS holds a reference to the same `gs`
                # that subsequent decisions / steps would mutate in place.
                "game_state_at_root": gs.clone(),
            })

            # Step table — capture `bet` so we can update cumulative_bets and
            # later reconstruct credits_pre_distribution. Engine's bet return
            # is the chips moved by this action only (0 in several_all_in mode).
            action_vec = torch.zeros(n_actions, dtype=torch.float32)
            action_vec[action_idx] = 1.0
            end, _, _, bet = table.step(action_vec)
            cumulative_bets[active_pos] += float(bet)
            action_step += 1

            # Post-action snapshot
            snapshots.append({
                "pot": table.pot,
                "bets": np.copy(table.bets),
                "credits": list(table.credits),
                "turn": table.turn,
                "active_pos": table.active_player,
                "action": action_vec,
            })

            if end:
                hand_done = True

        if not decisions:
            continue

        # ── Post-hand bookkeeping for Step 2 (equity-based Q + realized) ──
        # `credits_pre_distribution` = chips each player has AFTER all
        # betting but BEFORE the engine redistributes the pot. We don't
        # trust `final_credits` for `chips_invested` because the engine's
        # showdown distribution mixes winnings into credits AND has a known
        # mis-accounting on multi-street pots (prior-street chips never
        # reach winners when judger() runs). `cumulative_bets` is the
        # ground truth: every chip that left a player's stack via Table.step.
        final_pot = float(table.pot)
        final_active = [i for i in range(num_players)
                        if table.players_state[i] >= 0]
        credits_pre_dist = [
            float(initial_credits[p] - cumulative_bets[p])
            for p in range(num_players)
        ]

        # Build agents_by_position / value_scales_by_position for the
        # equity machinery. Position → ASI (used for action-head range
        # narrowing) and position → per-hero `mcts_value_scale` (used so
        # `evaluate_all_terminals` emits terminal Q in the SAME scale as the
        # value head's outputs at non-terminal leaves — otherwise
        # `re_backup_terminals` would mix raw-chip terminal Q with
        # normalized-scale W in ancestors).
        if terminal_proxy is None:
            agents_by_position = {p: hand_seated[p]["agent"]
                                  for p in range(num_players)}
            agent_name_by_pos = None
        else:
            agents_by_position = {}
            agent_name_by_pos = {p: hand_seated[p]["name"]
                                 for p in range(num_players)}
        value_scales_by_position = {
            p: float(search_scales.get(hand_seated[p]["name"], float(big_blind)))
            for p in range(num_players)
        }

        hand_record = {
            "decisions": decisions,
            "deck": deck_snapshot,
            "hero_hands": hero_hands,
            "num_players": num_players,
            "big_blind": big_blind,
            "final_pot": final_pot,
            "final_active_positions": final_active,
            "credits_pre_distribution": credits_pre_dist,
            "initial_credits": initial_credits,
            "start_stacks": start_stacks,
        }

        # 1) Backfill equity-based Q on EVERY terminal across every tree in
        # this hand, then re_backup_terminals propagates the override up so
        # `decision["mcts_root"].Q` reflects equity. Done before
        # collect_training_data because that function reads `root.Q` to fill
        # `root_q_ratio`.
        combo_probs_cache = evaluate_all_terminals(
            hand_record, agents_by_position, device,
            config=mcts_cfg,
            value_scales_by_position=value_scales_by_position,
            proxy=terminal_proxy, agent_name_by_pos=agent_name_by_pos,
            equity_device=equity_device,
        )

        # 2) Equity-based realized outcome at the actual played-out final
        # state of the hand. Replaces the noisy single-sample chip delta
        # `final_credits[hero] − credits_at(t)[hero]`. Same equity machinery
        # as (1); the only difference is the board is fully revealed (river)
        # and opponent ranges are narrowed by ALL of the hand's real decisions.
        # E.1.1: reuse combo_probs_cache from (1) — exact same data.
        ref_credits = [dec["all_credits_before_decision"] for dec in decisions]
        equity_pkt = compute_equity_outcome(
            hand_record, agents_by_position, device,
            ref_credits_by_decision=ref_credits, config=mcts_cfg,
            proxy=terminal_proxy, agent_name_by_pos=agent_name_by_pos,
            equity_device=equity_device,
            combo_probs_cache=combo_probs_cache,
        )
        realized_by_dec = equity_pkt["realized_by_decision"]
        equity_by_hero = equity_pkt["equity_by_hero"]

        # C.5: per-player total contributions for side-pot cap.
        # Use start_stacks (pre-blind) so blinds are included in contributions.
        contributions = [float(start_stacks[p]) - float(credits_pre_dist[p])
                         for p in range(num_players)]
        # Cache side-pot-capped hero_base per hero (same across all
        # decisions by that hero — only invested_from_t varies).
        _chain_hero_base = {}

        # Build per-(hero_pos, dec_idx) lookup of realized chips for chain
        # steps. Chain step i of example t uses HERO_T's perspective at
        # state(t+1+i). Equity is identical (same cards, same opponents);
        # only `chips_invested_from(t+1+i)` differs.
        def _chain_realized(hero_pos, dec_idx_chain):
            invested = (float(ref_credits[dec_idx_chain][hero_pos])
                        - credits_pre_dist[hero_pos])
            if len(final_active) <= 1:
                if (len(final_active) == 1
                        and final_active[0] == hero_pos):
                    return final_pot - invested
                return -invested
            if hero_pos not in final_active:
                return -invested
            equity = equity_by_hero.get(hero_pos)
            if equity is None:
                return -invested
            # C.5: side-pot-correct chip delta
            if hero_pos not in _chain_hero_base:
                inv_total = float(contributions[hero_pos])
                opp_c = [float(contributions[p]) for p in final_active
                         if p != hero_pos]
                max_opp = max(opp_c) if opp_c else 0.0
                eff = min(inv_total, max_opp)
                excess = inv_total - eff
                share_pot = sum(min(float(c), eff) for c in contributions)
                _chain_hero_base[hero_pos] = excess + equity * share_pot
            return _chain_hero_base[hero_pos] - invested

        examples = collect_training_data(
            hand_record, n_actions,
            max_chain_depth=max_chain_depth,
            final_credits=None,  # equity-realized supersedes raw final_credits
            big_blind=big_blind,
            action_label_smoothing=action_label_smoothing,
            terminal_k_worst=terminal_k_worst,
            terminal_k_best=terminal_k_best,
            # C.3: do NOT clip terminal Q at collection — _finalize_value_targets
            # rescales to the fresh value axis and clips there.
            terminal_clip_val=None,
        )

        # Overwrite raw realized targets with equity-based ones.
        # `examples` is now [(orig_t, MCTSTrainingExample), ...] — past-
        # opponent decisions skipped, so the original index `t` is required
        # to index `realized_by_dec` / `decisions` correctly.
        for t, ex in examples:
            dec = decisions[t]
            hero_pos = dec["player_pos"]
            ex.value_target = float(realized_by_dec[t])
            for i, step in enumerate(ex.chain):
                dec_idx_chain = t + 1 + i
                if dec_idx_chain < len(decisions):
                    step.value_target = float(
                        _chain_realized(hero_pos, dec_idx_chain))
                else:
                    # Defensive: chain truncated past last decision —
                    # shouldn't happen because collect_training_data stops
                    # at len(decisions), but guard anyway.
                    step.value_target = 0.0
            agent_name = hand_seated[hero_pos]["name"]
            per_agent_examples[agent_name].append(ex)

        if progress_counter is not None:
            with progress_counter.get_lock():
                progress_counter.value += 1

    # Cleanup: drop any still-materialised past snapshots and restore every
    # active agent to `device` (some were offloaded to CPU during reshuffles
    # while not seated; the caller expects them ready for training).
    if has_past:
        if materialized_past:
            log(f"  [past] releasing {len(materialized_past)} "
                f"materialised snapshot(s) at end of collection")
            materialized_past.clear()
        for a in agents_list:
            cur = getattr(a["agent"], "device_", "cpu")
            if str(cur) != str(device):
                a["agent"].set_device(device)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return per_agent_examples


def _robust_scale(chips, fallback):
    """Robust scale (in chips) for the value-target normalizer (C.7.4).

    Self-play chip-delta distributions are fat-tailed (rare all-in swings) and
    the bootstrap sample is small at low `n_hands_per_cycle`, so a plain `std`
    over-inflates from a handful of extreme hands — which then shrinks every
    example's normalized target. Use a robust estimator instead: MAD·1.4826
    (≈ σ for a Gaussian), falling back to IQR/1.349, then `std`, then
    `fallback` (BB) — the first that is finite and > 0.
    """
    chips = np.asarray(chips, dtype=np.float64)
    if chips.size == 0:
        return float(fallback)
    med = np.median(chips)
    mad = float(np.median(np.abs(chips - med)))
    scale = 1.4826 * mad
    if scale >= 1e-8:
        return float(scale)
    q75, q25 = np.percentile(chips, [75, 25])
    iqr_scale = float(q75 - q25) / 1.349
    if iqr_scale >= 1e-8:
        return iqr_scale
    std_c = float(chips.std())
    if std_c >= 1e-8:
        return std_c
    return float(fallback)


def _finalize_value_targets(per_agent_examples, agents_list, search_scales,
                              alpha, clip_val, big_blind, log):
    """Bootstrap `mcts_value_scale` and apply hybrid blend + clip in place.

    The value_target field of each MCTSTrainingExample / ChainStep enters
    as a **raw chip delta** (from that state to hand end, hero POV) and
    exits as a **scalar in [−clip, +clip]** equal to:
        clamp(α · (root_q_ratio · search_scale / new_scale)
              + (1 − α) · (realized_chips / new_scale),
              ±clip)
    where:
      - `search_scale` is the per-agent `mcts_value_scale` snapshot taken
        before this cycle's collection — the same scale `evaluate_all_terminals`
        divided terminal Q by, so `root.Q` lives on `search_scale` axis
        (fallback `big_blind` on the first-ever cycle, before any bootstrap).
      - `new_scale` is either the existing `mcts_value_scale` from
        `norm_stats`, or freshly bootstrapped as a robust scale (MAD/IQR, see
        `_robust_scale` — C.7.4) of the chip deltas if the key is absent
        (first cycle or post-rebootstrap).
      - `root_q_ratio` is stored in `search_scale`-units; rescaling by
        `search_scale / new_scale` lifts it onto the same target axis as
        `realized_chips / new_scale`. The multiplication is **not** a
        double normalization — it undoes the search-time division so we
        can re-apply the cycle's fresh scale (see PLAN §4.1).
      - C.2: the chain TD half uses `step.root_q_ratio` only for hero-owned
        steps (NaN / opp steps fall back to pure MC).
      - C.3: `ex.terminal_targets` (RAW `search_scale`-unit equity Q) are
        rescaled by the same `search_scale / new_scale` and clipped here, on
        the fresh axis — not at collection time.

    Operates in place. Returns nothing.
    """
    bb = float(big_blind)

    for agent_info in agents_list:
        name = agent_info["name"]
        examples = per_agent_examples.get(name, [])
        if not examples:
            continue

        ns = agent_info.get("norm_stats")
        if ns is None:
            ns = {}
            agent_info["norm_stats"] = ns

        # 1. Bootstrap new_scale from raw chip-delta std if absent OR if the
        # pipeline asked for a forced rebootstrap this cycle (flag set by
        # `value_norm_rebootstrap_every`). The flag path keeps the OLD scale
        # in `ns` until this moment so MCTS search & inference earlier in the
        # cycle used a coherent value scale; we only now swap to the fresh one.
        pending = bool(ns.pop("_mcts_value_scale_pending_rebootstrap", False))
        need_bootstrap = ("mcts_value_scale" not in ns) or pending
        if need_bootstrap:
            chips = np.array(
                [float(ex.value_target) for ex in examples],
                dtype=np.float64,
            )
            # C.7.4: robust scale (MAD/IQR) instead of std — fat tails + tiny
            # samples make std over-inflate.
            scale_c = _robust_scale(chips, bb)
            old_scale = ns.get("mcts_value_scale")
            ns["mcts_value_scale"] = scale_c
            ns["mcts_value_scale_n_samples"] = int(len(chips))
            ns["mcts_value_chip_min"] = float(chips.min())
            ns["mcts_value_chip_max"] = float(chips.max())
            origin = "rebootstrapped" if pending else "bootstrapped"
            old_s = (f" (was {old_scale:.2f})"
                      if isinstance(old_scale, (int, float)) else "")
            log(f"  {name}: {origin} mcts_value_scale = {scale_c:.2f} "
                f"chips (robust MAD/IQR){old_s} (n={len(chips)}, range "
                f"[{chips.min():.1f}, {chips.max():.1f}])")
        else:
            log(f"  {name}: reusing mcts_value_scale = "
                f"{ns['mcts_value_scale']:.2f} chips "
                f"(n={ns.get('mcts_value_scale_n_samples', '?')})")

        new_scale = float(ns["mcts_value_scale"])
        search_scale = float(search_scales.get(name, bb))
        rescale_q = search_scale / new_scale

        # 2. Hybrid blend + clip per example and per chain step.
        n_root = n_clipped_root = 0
        n_chain = n_clipped_chain = 0
        for ex in examples:
            realized = float(ex.value_target)
            root_q_raw = float(ex.root_q_ratio)
            realized_in_new = realized / new_scale
            if np.isnan(root_q_raw):
                blend = realized_in_new
            else:
                q_in_new = root_q_raw * rescale_q
                blend = alpha * q_in_new + (1.0 - alpha) * realized_in_new
            ex.value_target = max(-clip_val, min(clip_val, blend))
            n_root += 1
            if abs(blend) > clip_val:
                n_clipped_root += 1

            for step in ex.chain:
                step_realized = float(step.value_target)
                step_q_raw = float(step.root_q_ratio)
                # C.2: only TD-blend the future tree's root.Q when that future
                # decision is the SAME hero (same perspective AND scale). Opp
                # chain steps (and legacy/past-opp steps with NaN) → pure-MC
                # realized target.
                if step.is_hero and not np.isnan(step_q_raw):
                    step_q_in_new = step_q_raw * rescale_q
                else:
                    step_q_in_new = step_realized / new_scale
                step_blend = (
                    alpha * step_q_in_new
                    + (1.0 - alpha) * (step_realized / new_scale)
                )
                step.value_target = max(-clip_val, min(clip_val, step_blend))
                n_chain += 1
                if abs(step_blend) > clip_val:
                    n_clipped_chain += 1

            # C.3: lift tree-terminal value targets onto the SAME fresh axis as
            # root/chain (they were stored RAW in `search_scale` units), THEN
            # clip — same `rescale_q` and `clip_val` as above.
            if ex.terminal_targets:
                rescaled_terms = []
                for action_path, q_search in ex.terminal_targets:
                    q_new = float(q_search) * rescale_q
                    q_new = max(-clip_val, min(clip_val, q_new))
                    rescaled_terms.append((action_path, q_new))
                ex.terminal_targets = rescaled_terms

        log(f"  {name}: blended (α={alpha:.2f}, clip=±{clip_val:.1f}) — "
            f"root: {n_clipped_root}/{n_root} clipped; "
            f"chain: {n_clipped_chain}/{n_chain} clipped")


# ─────────────────────────── Parallel collection ───────────────────────────
# CPU actor processes run the tree search + game logic; one GPU inference
# server (agent/mcts/inference_server.py) batches their forward requests. The
# within-tree mcts.batch_size (16) is unchanged — cross-actor batching at the
# server provides GPU efficiency on top. opponent_embedding uses a per-actor
# table on the server (variant B). See versions/v5 CLAUDE.md / the plan.


def _actor_main(worker_id, agents_meta, config, n_hands, seed,
                search_scales, strange_p_by_agent, req_q, resp_q,
                result_q, progress, output_dir, progress_counter=None):
    """Actor process: play `n_hands` with a RemoteEvaluator (forwards offloaded
    to the inference server), write per_agent examples to a per-actor pickle
    file on disk, signal completion via `result_q` (path only, no payload).

    Why disk transport: mp.Queue with large pickled payloads silently lost the
    OK message in Python 3.14 (observed in the opponent-gen actors — feeder
    confirmed flushed, parent never received). Disk-based payload + tiny-
    signal-via-queue removes the queue from the failure path entirely.

    Never touches CUDA (model lives on the server; equity MC runs on CPU)."""
    import os as _os
    import pickle as _pickle
    import sys as _sys
    import traceback as _tb
    _sys.stderr.write(f"[mcts actor {worker_id}] starting, n_hands={n_hands}\n")
    _sys.stderr.flush()
    try:
        import torch as _torch
        # One BLAS thread per actor so N actors don't oversubscribe the cores.
        _torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        _torch.manual_seed(seed)

        from agent.mcts.evaluator import RemoteEvaluator, EvalProxy
        from agent.train_scenarios.generation.generate import _get_raise_sizes

        game_cfg = config.get("game", {})
        mcts_cfg = config.get("mcts", {})
        raise_sizes = _get_raise_sizes(game_cfg)
        n_actions = len(raise_sizes[0]) + 3

        terminal_proxy = EvalProxy(worker_id, req_q, resp_q)

        def make_mcts(agent_info, gs):
            name = agent_info["name"]
            ev = RemoteEvaluator(worker_id, name, req_q, resp_q, n_actions)
            return MCTS(None, "cpu", mcts_cfg,
                        strange_p=strange_p_by_agent[name], evaluator=ev,
                        search_scale=search_scales.get(
                            name, float(game_cfg.get("big_blind", 10))))

        per_agent = _play_hands(
            agents_meta, config, "cpu", n_hands, make_mcts=make_mcts,
            terminal_proxy=terminal_proxy, equity_device="cpu",
            search_scales=search_scales, strange_p_by_agent=strange_p_by_agent,
            log=lambda *a, **k: None, progress=progress,
            progress_counter=progress_counter)
        n_items = sum(len(v) for v in per_agent.values())

        # Persist payload to disk atomically (write to .tmp then rename).
        out_path = _os.path.join(output_dir, f"actor_{worker_id}.pkl")
        tmp_path = out_path + ".tmp"
        _sys.stderr.write(
            f"[mcts actor {worker_id}] loop done, writing {n_items} examples "
            f"to {out_path}\n")
        _sys.stderr.flush()
        with open(tmp_path, "wb") as _f:
            _pickle.dump(per_agent, _f, protocol=_pickle.HIGHEST_PROTOCOL)
            _f.flush()
            _os.fsync(_f.fileno())
        _os.rename(tmp_path, out_path)
        size_mb = _os.path.getsize(out_path) / (1024 * 1024)
        _sys.stderr.write(
            f"[mcts actor {worker_id}] wrote {size_mb:.1f} MB, signalling OK\n")
        _sys.stderr.flush()
        # Tiny signal via queue (path string, no big payload).
        result_q.put(("OK", worker_id, out_path))
        result_q.close()
        result_q.join_thread()
        _sys.stderr.write(
            f"[mcts actor {worker_id}] OK signal flushed, exiting\n")
        _sys.stderr.flush()
    except BaseException:
        # BaseException catches SystemExit / KeyboardInterrupt too — otherwise
        # the actor can exit silently with code 0 and parent waits forever.
        tb = _tb.format_exc()
        _sys.stderr.write(f"[mcts actor {worker_id}] FAILED:\n{tb}\n")
        _sys.stderr.flush()
        try:
            result_q.put(("ACTOR_ERROR", worker_id, tb))
            result_q.close()
            result_q.join_thread()
        except BaseException:
            pass


def _run_parallel_collection(agents_list, config, device, log, n_hands,
                             n_workers, search_scales, strange_p_by_agent,
                             cycle_idx, n_cycles=1):
    """Spawn the inference server + `n_workers` CPU actors, gather and merge
    their per-agent examples. Returns the merged (still RAW) per_agent_examples
    dict; the caller runs `_finalize_value_targets` once over it."""
    import torch.multiprocessing as tmp
    from agent.mcts.inference_server import server_main

    mcts_train_cfg = config.get("mcts_train", {})
    server_cfg = {
        "device": device,
        "server_max_batch": int(mcts_train_cfg.get("server_max_batch", 256)),
        "server_linger_ms": float(mcts_train_cfg.get("server_linger_ms", 2)),
        "n_workers": n_workers,
    }

    # Server spec: each agent's config + CPU state_dict + norm_stats.
    spec = []
    for a in agents_list:
        sd = {k: v.detach().cpu() for k, v in a["agent"].state_dict().items()}
        spec.append({"name": a["name"], "config": config,
                     "state_dict": sd, "norm_stats": a.get("norm_stats")})

    # Move parent's agents off the GPU for the duration of parallel collection.
    # The inference server holds its own GPU copies built from `spec`; the
    # parent does not use the live models while waiting for actor results, so
    # keeping them on CUDA just doubles GPU memory pressure and was OOM'ing
    # the server at agent build time (24GB cards, multi-agent runs).
    # Restored to their original device in the `finally` block below so that
    # downstream training in `pipeline.py` finds them where it left them.
    parent_devices = []
    if str(device).startswith("cuda"):
        import torch as _torch_local
        for a in agents_list:
            parent_devices.append(getattr(a["agent"], "device_", "cpu"))
            a["agent"].cpu()
            # `device_` is a custom attr on ASI maintained by set_device; keep
            # it in sync so any inadvertent forward in the parent doesn't try
            # to target the previous CUDA device.
            a["agent"].device_ = "cpu"
        _torch_local.cuda.empty_cache()
        log(f"  parallel collect: moved {len(agents_list)} parent agent(s) "
            f"to CPU during server lifetime (freed CUDA cache)")

    # Model-free metadata for actors (the server holds the live model).
    agents_meta = [{"name": a["name"], "norm_stats": a.get("norm_stats"),
                    "temperature": a.get("temperature")} for a in agents_list]

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
    payload_dir = _tempfile.mkdtemp(prefix=f"mcts_actor_payloads_cycle{cycle_idx}_")

    server = ctx.Process(
        target=server_main,
        args=(spec, req_q, resp_qs, ready_event, stop_event, server_cfg),
        daemon=True)
    server.start()
    if not ready_event.wait(timeout=600):
        stop_event.set()
        server.terminate()
        raise RuntimeError("inference server failed to become ready in 600s")

    # Split hands across actors; per-actor seed varies by cycle and worker.
    base = n_hands // n_workers
    rem = n_hands % n_workers
    hands_per = [base + (1 if i < rem else 0) for i in range(n_workers)]
    base_seed = 1000 * (int(cycle_idx) + 1)

    # Shared progress counter — actors increment after each completed hand
    # under its built-in lock; a parent daemon thread polls and updates a
    # single tqdm for the whole collection. `progress=False` on actors
    # disables their local tqdm.
    progress_counter = ctx.Value("i", 0)

    actors = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_actor_main,
            args=(wid, agents_meta, config, hands_per[wid],
                  base_seed + wid, search_scales, strange_p_by_agent,
                  req_q, resp_qs[wid], result_q, False, payload_dir,
                  progress_counter),
            daemon=True)
        p.start()
        actors.append(p)

    log(f"MCTS parallel collection: {n_workers} actors, server on {device}, "
        f"max_batch={server_cfg['server_max_batch']}, hands/actor={hands_per}, "
        f"payload_dir={payload_dir}")

    # Single unified tqdm for the whole collection; updated by a daemon
    # thread reading `progress_counter`. Refreshes every 100ms.
    import threading as _threading
    pbar = tqdm(total=n_hands,
                desc=f"MCTS collect cycle {cycle_idx + 1}/{n_cycles}")
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

    # Gather results; fail loudly on any actor error or unexpected death.
    import pickle as _pickle
    import os as _os
    per_agent_examples = {a["name"]: [] for a in agents_list}
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
                error = (f"mcts actor {wid_done} OK file unreadable "
                         f"({path}): {type(_e).__name__}: {_e}")
                return
            try:
                _os.unlink(path)
            except OSError:
                pass
            for name, exs in partial.items():
                per_agent_examples.setdefault(name, []).extend(exs)
            received_from[wid_done] = True
        else:
            _, wid_done, tb = msg
            error = f"actor {wid_done} failed:\n{tb}"

    try:
        while not all(received_from) and error is None:
            try:
                msg = result_q.get(timeout=1.0)
            except Exception:
                # Drain pending messages: an actor may have already put its
                # OK on the pipe and exited cleanly between our get() timeout
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
                    error = (f"mcts actor(s) exited without reporting: "
                             f"{details} — check stderr above for traceback")
                    break
                if not server.is_alive() and server.exitcode not in (0, None):
                    error = "the inference server died unexpectedly"
                    break
                continue
            _consume(msg)

        # Shutdown: stop server, join everything (terminate stragglers).
        stop_event.set()
        try:
            req_q.put(None)  # SENTINEL
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
        # the caller's downstream code (training) still finds the agents
        # where it expects them.
        if parent_devices:
            for a, dev in zip(agents_list, parent_devices):
                try:
                    a["agent"].set_device(dev)
                except BaseException:
                    pass

    if error is not None:
        raise RuntimeError(error)

    return per_agent_examples
