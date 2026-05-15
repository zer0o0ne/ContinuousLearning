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


@dataclass
class ChainStep:
    """One step in the modelling chain: action taken + target distribution."""
    action_taken: int
    target_distribution: list
    is_hero: bool  # True → action_head predicts, False → opponent_action_head


def _make_terminal_evaluator(hero_pos, hero_initial_credits, denom,
                              ev_mean, ev_std):
    """Build a closure that scores any terminal GameState from hero's POV.

    Returns a callable `evaluate(gs) -> float` whose output is on the same
    scale as the value head's training target — `(outcome_ratio - ev_mean)
    / ev_std` where `outcome_ratio = outcome_chips / denom`.

    Fold terminals are deterministic: hero either scoops the pot or loses
    what they invested along this path. Showdown terminals use a fair-
    share-of-pot heuristic (`pot / n_active`) because MCTS doesn't carry
    cards into the tree — this is biased but gives a sensible neutral
    estimate that scales with pot size.

    When `ev_mean`/`ev_std` are unavailable (first cycle, before
    `mcts_ev_*` is bootstrapped), the raw outcome ratio is returned.
    """
    has_norm = (ev_mean is not None and ev_std is not None and ev_std > 1e-8)
    denom = max(float(denom), 1.0)

    def evaluate(gs):
        active = [i for i in range(gs.num_players) if gs.players_state[i] >= 0]
        hero_invested = hero_initial_credits - float(gs.credits[hero_pos])
        if len(active) == 1:
            if active[0] == hero_pos:
                outcome = float(gs.pot) - hero_invested
            else:
                outcome = -hero_invested
        elif len(active) >= 2:
            # Showdown: fair-share-of-pot heuristic (cards not available
            # in MCTS GameState; equity calc happens post-hand in
            # terminal_eval.py if needed).
            outcome = float(gs.pot) / len(active) - hero_invested
        else:
            outcome = 0.0
        ratio = outcome / denom
        if has_norm:
            return (ratio - ev_mean) / ev_std
        return ratio

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


def collect_training_data(hand_record, n_actions):
    """Extract training examples from all MCTS trees in a completed hand.

    For each tree (one per decision point), produces an MCTSTrainingExample.
    `value_target` is left as raw `root.Q` here; the caller (run_mcts_collection)
    overrides it with the actual normalized hand outcome.

    Chain semantics: chain[i].action_taken advances context from state(t+i)
    to state(t+i+1). chain[i].target_distribution is the predicted distribution
    AT state(t+i+1) (after applying that action).

    Args:
        hand_record: dict with "decisions" list — each entry has:
            player_pos, action_idx, mcts_root, events_at_root
        n_actions: int, action space size

    Returns:
        list of MCTSTrainingExample, one per tree
    """
    decisions = hand_record["decisions"]
    examples = []

    for t, decision in enumerate(decisions):
        hero_pos = decision["player_pos"]
        root = decision["mcts_root"]
        action_target = get_n_distribution(root, n_actions)

        # Modelling chain: chain[i] predicts distribution at decisions[t+1+i],
        # using the action taken at decisions[t+i] to advance context.
        chain = []
        for i, future_dec in enumerate(decisions[t + 1:]):
            future_root = future_dec["mcts_root"]
            target_dist = get_n_distribution(future_root, n_actions)
            # action that advances state(t+i) → state(t+i+1):
            action_taken = decisions[t + i]["action_idx"]
            chain.append(ChainStep(
                action_taken=action_taken,
                target_distribution=target_dist,
                is_hero=(future_dec["player_pos"] == hero_pos),
            ))

        examples.append(MCTSTrainingExample(
            events=decision["events_at_root"],
            value_target=root.Q,  # placeholder; overwritten in run_mcts_collection
            action_target=action_target,
            chain=chain,
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

    log(f"MCTS collection: {n_hands} hands, players={min_players}-{max_players}, "
        f"stack={min_stack}-{max_stack}, swap_prob={swap_prob}, "
        f"{mcts_cfg.get('n_simulations', 1000)} simulations/decision")

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
            # Advance all-in runouts without MCTS
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
            ns = agent_info.get("norm_stats") or {}
            term_eval = _make_terminal_evaluator(
                hero_pos=active_pos,
                hero_initial_credits=credits_at_dec,
                denom=max(pot_at_dec + facing_at_dec, float(big_blind)),
                ev_mean=ns.get("mcts_ev_mean"),
                ev_std=ns.get("mcts_ev_std"),
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
                "agent_norm_stats": agent_info["norm_stats"],
            })

            # Step table
            action_vec = torch.zeros(n_actions, dtype=torch.float32)
            action_vec[action_idx] = 1.0
            end, _, _, _ = table.step(action_vec)
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

        # Build training examples; store RAW ratio in value_target as a
        # temporary scalar. We z-score AFTER all hands are collected because
        # MCTS realised-outcome distribution is much wider than gto_ev_predict's
        # solver-EV distribution, so reusing gto_ev's ev_mean/ev_std produces
        # outliers in z-score space (∼±200). Instead we bootstrap MCTS-
        # specific stats on the first collection that has data, then reuse
        # them across cycles (persisted via norm_stats inside the checkpoint).
        hand_record = {"decisions": decisions}
        examples = collect_training_data(hand_record, n_actions)

        final_credits = list(table.credits)
        for ex, dec in zip(examples, decisions):
            pos = dec["player_pos"]
            outcome_chips = float(final_credits[pos] - dec["credits_before_decision"])
            denom = max(dec["pot_at_decision"] + dec["facing_bet_at_decision"],
                        float(big_blind))
            ex.value_target = outcome_chips / denom  # raw ratio, normalized below
            agent_name = hand_seated[pos]["name"]
            per_agent_examples[agent_name].append(ex)

    # Per-agent: bootstrap MCTS-specific value norm stats on first collection
    # that has data, then reuse them across cycles. Stored as new top-level
    # keys with `mcts_ev_*` prefix in norm_stats; the original
    # gto_ev_predict stats (ev_mean/ev_std/pot/stack/etc) stay untouched.
    # Mutating norm_stats in place propagates back to
    # agent._checkpoint_norm_stats and gets persisted by _save_best.
    for agent_info in agents_list:
        name = agent_info["name"]
        examples = per_agent_examples.get(name, [])
        if not examples:
            continue

        ns = agent_info.get("norm_stats")
        if ns is None:
            ns = {}
            agent_info["norm_stats"] = ns

        if "mcts_ev_mean" not in ns or "mcts_ev_std" not in ns:
            ratios = np.array([float(ex.value_target) for ex in examples],
                              dtype=np.float64)
            mean_r = float(ratios.mean())
            std_r = float(ratios.std())
            if std_r < 1e-8:
                std_r = 1.0
            ns["mcts_ev_mean"] = mean_r
            ns["mcts_ev_std"] = std_r
            ns["mcts_ev_n_samples"] = int(len(ratios))
            ns["mcts_ev_ratio_min"] = float(ratios.min())
            ns["mcts_ev_ratio_max"] = float(ratios.max())
            log(f"  {name}: bootstrapped mcts_ev_* "
                f"({len(ratios)} ratios) — mean={mean_r:.4f}, std={std_r:.4f} "
                f"(min={ns['mcts_ev_ratio_min']:.2f}, "
                f"max={ns['mcts_ev_ratio_max']:.2f})")
        else:
            log(f"  {name}: reusing mcts_ev_* "
                f"(mean={ns['mcts_ev_mean']:.4f}, std={ns['mcts_ev_std']:.4f}, "
                f"n_bootstrap={ns.get('mcts_ev_n_samples', '?')})")

        m = float(ns["mcts_ev_mean"])
        s = float(ns["mcts_ev_std"])
        for ex in examples:
            ex.value_target = (float(ex.value_target) - m) / s

    for name, exs in per_agent_examples.items():
        log(f"  {name}: {len(exs)} training examples")

    return per_agent_examples
