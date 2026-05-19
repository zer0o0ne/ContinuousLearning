"""
Training data extraction and self-play collection for MCTS.

Provides:
- collect_training_data(): extract MCTSTrainingExample from hand records
- run_mcts_collection(): play hands with MCTS, produce training data per agent
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from agent.mcts.mcts import MCTS, get_n_distribution
from agent.mcts.game_state import GameState
from env.table import Table
from evaluation.evaluate import _rebuild_events, _normalize_events_inplace

# `terminal_eval` pulls `agent.gto_utils.gpu_solver_v2`, which at import time
# requires `gpu_solver` on sys.path (a pipeline-level setup, not always done
# in unit-test contexts). We need its helpers only inside `run_mcts_collection`
# at runtime, so import lazily there instead of at module load — keeps
# `collect.py` importable for legacy / training-script callers that never
# trigger equity evaluation.


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
        a single shared scale. MCTS backup mixes value_head outputs across
        nodes at different depths in the tree, all normalized by
        `denom_at_root` via `_make_terminal_evaluator`; using a local denom
        per chain step would create a per-depth scale mismatch. Same
        mcts_ev_* stats then apply to root + chain uniformly. Pre-z-score
        stored here; the caller z-scores at the end.
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


def _make_terminal_evaluator(hero_pos, hero_initial_credits, value_scale):
    """Build a closure that scores any terminal GameState from hero's POV.

    Returns a callable `evaluate(gs) -> float` whose output is the chip
    delta from hero's perspective scaled by a single per-cycle `value_scale`
    (in chips). No `-mean` shift and no state-dependent `(pot + facing)`
    denom — that combination created the fold-bias (fold's `0` mapping to
    `(0 − μ)/σ = +0.15` when μ<0). Single-scale division preserves
    zero-sum across players and keeps fold at exactly 0.

    `value_scale` semantics:
      - In `run_mcts_collection`: pass `norm_stats["mcts_value_scale"]` if
        present (the bootstrapped per-agent std of realized chip deltas),
        else `big_blind` as a fallback (used only on the first cycle
        before any bootstrap exists).

    Fold terminals are deterministic: hero either scoops the pot or loses
    what they invested along this path. Showdown terminals use a fair-
    share-of-pot heuristic (`pot / n_active`) because MCTS doesn't carry
    cards into the tree — `evaluate_all_terminals` (Step 2) will replace
    showdown values with proper equity calculations.
    """
    value_scale = max(float(value_scale), 1.0)

    def evaluate(gs):
        active = [i for i in range(gs.num_players) if gs.players_state[i] >= 0]
        hero_invested = hero_initial_credits - float(gs.credits[hero_pos])
        if len(active) == 1:
            if active[0] == hero_pos:
                outcome = float(gs.pot) - hero_invested
            else:
                outcome = -hero_invested
        elif len(active) >= 2:
            outcome = float(gs.pot) / len(active) - hero_invested
        else:
            outcome = 0.0
        return outcome / value_scale

    return evaluate


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


def collect_training_data(hand_record, n_actions, max_chain_depth=None,
                          final_credits=None, big_blind=None):
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
    that decision). Because `_make_terminal_evaluator` now returns
    `outcome / BB` directly, **all `root.Q` values are already in BB units**
    — no de-z-scoring or rescaling required for root or chain.

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

    Returns:
        list of MCTSTrainingExample, one per tree
    """
    decisions = hand_record["decisions"]
    examples = []
    has_value_targets = (final_credits is not None and big_blind is not None)

    for t, decision in enumerate(decisions):
        hero_pos = decision["player_pos"]
        root = decision["mcts_root"]
        action_target = get_n_distribution(root, n_actions)

        # `_make_terminal_evaluator` returns chip_delta in BB units, so
        # `root.Q` is already in BB units. No transform needed.
        example_root_q_ratio = float(root.Q)

        # Modelling chain: chain[i] predicts distribution at decisions[t+1+i],
        # using the action taken at decisions[t+i] to advance context.
        chain = []
        for i, future_dec in enumerate(decisions[t + 1:]):
            if max_chain_depth is not None and i >= max_chain_depth:
                break
            future_root = future_dec["mcts_root"]
            target_dist = get_n_distribution(future_root, n_actions)
            # action that advances state(t+i) → state(t+i+1):
            action_taken = decisions[t + i]["action_idx"]
            events_at_step = future_dec.get("events_at_root", []) or []

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

            # future_root.Q is already in BB units (terminal_evaluator returns
            # chip_delta/BB uniformly across all trees in this hand).
            step_root_q_ratio = float(future_root.Q)

            chain.append(ChainStep(
                action_taken=action_taken,
                target_distribution=target_dist,
                is_hero=(future_dec["player_pos"] == hero_pos),
                events_at_step=events_at_step,
                value_target=value_target_step,
                root_q_ratio=step_root_q_ratio,
            ))

        examples.append(MCTSTrainingExample(
            events=decision["events_at_root"],
            value_target=root.Q,  # placeholder; overwritten in run_mcts_collection
            action_target=action_target,
            chain=chain,
            root_q_ratio=example_root_q_ratio,
        ))

    return examples


def run_mcts_collection(agents_list, config, device, log, n_hands):
    """Play hands with MCTS decisions and collect training examples.

    Each agent uses MCTS for its decisions. After each hand, training
    examples are extracted using actual game outcomes as value targets.

    Number of players is randomized per hand between min_players and
    max_players. Seated agents are drawn from the pool and swapped
    with player_swap_prob between hands (like opponent_action generation).

    Args:
        agents_list: list of dicts with keys:
            "agent" (ASI), "norm_stats" (dict), "name" (str), "temperature" (float)
        config: full config dict (game, mcts, mcts_train, etc.)
        device: torch device string
        log: logger callable
        n_hands: number of hands to play

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

    # Per-agent search-time value scale: what `terminal_evaluator` divided
    # outcomes by during this cycle's MCTS searches. Snapshot once at the
    # start so the final normalization can convert root.Q (in search_scale
    # units) back into chips even if mcts_value_scale is re-bootstrapped
    # later in the same call. Fallback to BB on the first-ever cycle (no
    # bootstrap yet).
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

    # Initial table: random player count + random agents from pool (with replacement)
    num_players = random.randint(min_players, max_players)
    hand_seated = random.choices(agents_list, k=num_players)

    for hand_i in tqdm(range(n_hands), desc="MCTS collection"):
        # With swap_prob, reshuffle the entire table: new count + new agents
        if random.random() < swap_prob:
            num_players = random.randint(min_players, max_players)
            hand_seated = random.choices(agents_list, k=num_players)
        seated_names = [a["name"] for a in hand_seated]

        dummy_action = torch.zeros(n_actions, dtype=torch.float32)
        start_stack = random.randint(min_stack, max_stack)

        table = Table(
            num_players=num_players,
            raise_sizes=raise_sizes,
            start_credits=start_stack,
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

            # MCTS search (opponent_emb at root only — see MCTS._evaluate_root)
            gs = GameState.from_table(table, active_pos)
            # Use the agent's bootstrapped value scale if available, else
            # fall back to BB (first-ever cycle has no scale yet). The same
            # scale is captured per-agent below for the final normalization
            # of value_target so the search-time scale and the target-time
            # scale are converted into a consistent target space.
            ns_search = agent_info.get("norm_stats") or {}
            search_scale = float(ns_search.get("mcts_value_scale",
                                                float(big_blind)))
            term_eval = _make_terminal_evaluator(
                hero_pos=active_pos,
                hero_initial_credits=credits_at_dec,
                value_scale=search_scale,
            )
            mcts = MCTS(agent_info["agent"], device, mcts_cfg,
                        opponent_emb_table=opp_tables.get(agent_info["name"]),
                        terminal_evaluator=term_eval)
            agent_info["agent"].eval()
            action_idx = mcts.search([norm_events], gs)

            decisions.append({
                "player_pos": active_pos,
                "action_idx": action_idx,
                "mcts_root": mcts.last_root,
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
        # `evaluate_all_terminals` emits terminal Q in the SAME scale the
        # search-time terminal_evaluator did — otherwise re_backup_terminals
        # would mix raw-chip terminal Q with normalized-scale W in ancestors).
        agents_by_position = {p: hand_seated[p]["agent"]
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
        }

        # 1) Backfill equity-based Q on EVERY terminal across every tree in
        # this hand, then re_backup_terminals propagates the override up so
        # `decision["mcts_root"].Q` reflects equity. Done before
        # collect_training_data because that function reads `root.Q` to fill
        # `root_q_ratio`.
        evaluate_all_terminals(
            hand_record, agents_by_position, device,
            config=mcts_cfg,
            value_scales_by_position=value_scales_by_position,
        )

        # 2) Equity-based realized outcome at the actual played-out final
        # state of the hand. Replaces the noisy single-sample chip delta
        # `final_credits[hero] − credits_at(t)[hero]`. Same equity machinery
        # as (1); the only difference is the board is fully revealed (river)
        # and opponent ranges are narrowed by ALL of the hand's real decisions.
        ref_credits = [dec["all_credits_before_decision"] for dec in decisions]
        equity_pkt = compute_equity_outcome(
            hand_record, agents_by_position, device,
            ref_credits_by_decision=ref_credits, config=mcts_cfg,
        )
        realized_by_dec = equity_pkt["realized_by_decision"]
        equity_by_hero = equity_pkt["equity_by_hero"]

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
            return equity * final_pot - invested

        examples = collect_training_data(
            hand_record, n_actions,
            max_chain_depth=max_chain_depth,
            final_credits=None,  # equity-realized supersedes raw final_credits
            big_blind=big_blind,
        )

        # Overwrite raw realized targets with equity-based ones.
        for t, (ex, dec) in enumerate(zip(examples, decisions)):
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
      - `search_scale` is the scale `_make_terminal_evaluator` used during
        this cycle's MCTS search for this agent (from `norm_stats` snapshot
        before any clearing, fallback `big_blind`).
      - `new_scale` is either the existing `mcts_value_scale` from
        `norm_stats`, or freshly bootstrapped as `std(chip_deltas)` if the
        key is absent (first cycle or post-rebootstrap).
      - `root_q_ratio` is stored in `search_scale`-units; rescaling by
        `search_scale / new_scale` lifts it onto the same target axis as
        `realized_chips / new_scale`. The multiplication is **not** a
        double normalization — it undoes the search-time division so we
        can re-apply the cycle's fresh scale (see PLAN §4.1).

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

        # 1. Bootstrap new_scale from raw chip-delta std if absent.
        if "mcts_value_scale" not in ns:
            chips = np.array(
                [float(ex.value_target) for ex in examples],
                dtype=np.float64,
            )
            std_c = float(chips.std())
            if std_c < 1e-8:
                std_c = bb
            ns["mcts_value_scale"] = std_c
            ns["mcts_value_scale_n_samples"] = int(len(chips))
            ns["mcts_value_chip_min"] = float(chips.min())
            ns["mcts_value_chip_max"] = float(chips.max())
            log(f"  {name}: bootstrapped mcts_value_scale = {std_c:.2f} "
                f"chips (n={len(chips)}, range "
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
            q_in_new = root_q_raw * rescale_q
            realized_in_new = realized / new_scale
            blend = alpha * q_in_new + (1.0 - alpha) * realized_in_new
            ex.value_target = max(-clip_val, min(clip_val, blend))
            n_root += 1
            if abs(blend) > clip_val:
                n_clipped_root += 1

            for step in ex.chain:
                step_realized = float(step.value_target)
                step_q_raw = float(step.root_q_ratio)
                if not np.isnan(step_q_raw):
                    step_q_in_new = step_q_raw * rescale_q
                else:
                    # Legacy step without root.Q → fall back to pure MC.
                    step_q_in_new = step_realized / new_scale
                step_blend = (
                    alpha * step_q_in_new
                    + (1.0 - alpha) * (step_realized / new_scale)
                )
                step.value_target = max(-clip_val, min(clip_val, step_blend))
                n_chain += 1
                if abs(step_blend) > clip_val:
                    n_clipped_chain += 1

        log(f"  {name}: blended (α={alpha:.2f}, clip=±{clip_val:.1f}) — "
            f"root: {n_clipped_root}/{n_root} clipped; "
            f"chain: {n_clipped_chain}/{n_chain} clipped")
