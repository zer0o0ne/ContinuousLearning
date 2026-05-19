import os
import json
import copy

from utils import Logger
from agent.agent import ASI


def _find_latest_best_ckpt(scenario_dir):
    """Return path to best.pt in the latest timestamp subdir of scenario_dir, or None."""
    if not os.path.isdir(scenario_dir):
        return None
    candidates = []
    for name in os.listdir(scenario_dir):
        best = os.path.join(scenario_dir, name, "best.pt")
        if os.path.isfile(best):
            candidates.append((name, best))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _run_or_skip_phase(scenario_name, agent, agent_base, agent_log, train_fn):
    """If scenario already has a best.pt, load it and skip. Otherwise run train_fn()
    and load its best.pt. Returns the path to the loaded best.pt (or None)."""
    existing = _find_latest_best_ckpt(os.path.join(agent_base, scenario_name))
    if existing is not None:
        agent_log(f"  [resume] {scenario_name}: loading existing {existing}, "
                  f"skipping training")
        agent.load_checkpoint(existing)
        return existing
    _, run_dir = train_fn()
    if run_dir:
        best_ckpt = os.path.join(run_dir, "best.pt")
        if os.path.exists(best_ckpt):
            agent.load_checkpoint(best_ckpt)
            return best_ckpt
    return None


def _merge_train_config(config, scenario_key):
    """Merge game + solver + dataset + scenario-specific config into a flat dict."""
    merged = {}
    merged.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    merged["solver"] = solver_cfg.pop("type", "v2")
    merged.update(solver_cfg)
    merged.update(config.get("dataset", {}))
    merged.update(config.get(scenario_key, {}))
    return merged


def _load_or_generate_dataset(config, base_dir, device, log):
    """Load dataset from dataset_dir, or generate into it / base_dir.

    Returns raw (unnormalized) scenarios list.
    """
    import torch
    from agent.train_scenarios.generation.generate import generate_dataset, load_dataset

    dataset_cfg = config.get("dataset", {})
    dataset_dir = dataset_cfg.get("dataset_dir", "")
    save_dir = dataset_cfg.get("save_dir", "")

    # Merge generation params (game + solver + dataset)
    gen_cfg = {}
    gen_cfg.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    gen_cfg["solver"] = solver_cfg.pop("type", "v2")
    gen_cfg.update(solver_cfg)
    gen_cfg.update(dataset_cfg)

    if dataset_dir:
        # Try loading from specified dir
        scenarios = load_dataset(dataset_dir, log=log)
        if scenarios is not None:
            return scenarios
        log(f"Dataset not found at {dataset_dir}, generating...")

    # Determine where to save the generated dataset
    if save_dir:
        dataset_save_dir = save_dir
    elif dataset_dir:
        dataset_save_dir = dataset_dir
    else:
        dataset_save_dir = os.path.join(base_dir, "dataset", log.init_time)

    os.makedirs(dataset_save_dir, exist_ok=True)
    return generate_dataset(gen_cfg, dataset_save_dir, log=log)


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    version = os.path.basename(os.path.abspath(os.path.dirname(__file__)))
    name = config.get("name", "default")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    base_dir = os.path.join(project_root, "data", version, name)

    log = Logger(base_dir)
    log(f"Version: {version}, Experiment: {name}, Device: {device}")

    # Save config snapshot
    configs_dir = os.path.join(base_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    config_snapshot_path = os.path.join(configs_dir, f"{log.init_time}.json")
    with open(config_snapshot_path, "w") as f:
        json.dump(config, f, indent=4)
    log(f"Config saved to {config_snapshot_path}")

    from agent.train_scenarios.gto_ev_predict.train import train_gto_ev
    from agent.train_scenarios.gto_probs_predict.train import train_gto_probs
    from agent.train_scenarios.gto_predict.train import train_gto
    from agent.train_scenarios.modelling_predict.train import train_modelling
    from evaluation.evaluate import run_evaluation

    agent_dir = config.get("agent_dir", "")
    pipeline_cfg = config.get("pipeline", {})
    multi_agent = config.get("multi_agent")

    needs_training = (pipeline_cfg.get("run_gto_ev", True)
                      or pipeline_cfg.get("run_gto_probs", False)
                      or pipeline_cfg.get("run_gto_training", False)
                      or pipeline_cfg.get("run_modelling", False))

    # --- Load/generate dataset only if training is enabled ---
    base_scenarios = None
    if needs_training:
        base_scenarios = _load_or_generate_dataset(config, base_dir, device, log)
        if not base_scenarios:
            log("No dataset available. Aborting.")
            return

    dataset_cfg = config.get("dataset", {})
    val_split = dataset_cfg.get("val_split", 0.1)

    if multi_agent and needs_training:
        # --- Multi-agent training ---
        from agent.train_scenarios.modifiers import apply_modifiers

        save_dir_cfg = multi_agent.get("save_dir", "")
        if save_dir_cfg and os.path.isabs(save_dir_cfg):
            save_base_dir = save_dir_cfg
        else:
            save_base_dir = os.path.join(project_root, "data", version, save_dir_cfg or name)

        game_cfg = config.get("game", {})
        solver_cfg = config.get("solver", {})
        raise_sizes = game_cfg.get("raise_sizes")
        n_actions = len(next(iter(raise_sizes.values()))) + 3 if raise_sizes else game_cfg.get("table_bins", 50) + 3
        big_blind = game_cfg.get("big_blind", 10)
        temperature = solver_cfg.get("gto_temperature", 1.0)

        skip_training = set(multi_agent.get("skip_training", []))

        for agent_cfg in multi_agent["agents"]:
            agent_name = agent_cfg["name"]
            modifiers = agent_cfg.get("modifiers", [])

            if agent_name in skip_training:
                log(f"\n=== Agent: {agent_name} — skipped "
                    f"(listed in multi_agent.skip_training) ===")
                continue

            # Extract effective temperature for this agent
            agent_temperature = temperature
            for mod in modifiers:
                if mod.get("type") == "temperature":
                    agent_temperature = mod["value"]

            agent_base = os.path.join(save_base_dir, agent_name)
            agent_log = Logger(agent_base)
            agent_log(f"\n=== Agent: {agent_name} ===")
            agent_log(f"Modifiers: {json.dumps(modifiers)}")
            agent_log(f"Effective temperature: {agent_temperature}")

            agent = ASI(agent_log, config)
            agent.set_device(device)
            if agent_dir:
                # Try per-agent checkpoint first, fall back to shared agent_dir
                per_agent_dir = os.path.join(agent_dir, agent_name)
                if os.path.isdir(per_agent_dir):
                    agent.load_checkpoint(per_agent_dir)
                else:
                    agent.load_checkpoint(agent_dir)
            else:
                agent_log("Agent initialized randomly")

            modified = apply_modifiers(base_scenarios, modifiers, n_actions,
                                       big_blind, temperature)

            ev_train_cfg = _merge_train_config(config, "gto_ev_train")
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")

            if pipeline_cfg.get("run_gto_ev", True):
                _run_or_skip_phase(
                    "gto_ev_predict", agent, agent_base, agent_log,
                    lambda: train_gto_ev(agent, ev_train_cfg, device, agent_log,
                                         scenarios_override=modified,
                                         temperature=agent_temperature))

            if pipeline_cfg.get("run_gto_probs", False):
                _run_or_skip_phase(
                    "gto_probs_predict", agent, agent_base, agent_log,
                    lambda: train_gto_probs(agent, probs_train_cfg, device, agent_log,
                                            scenarios_override=modified,
                                            temperature=agent_temperature))

            if pipeline_cfg.get("run_gto_training", False):
                gto_train_cfg = _merge_train_config(config, "gto_train")
                _run_or_skip_phase(
                    "gto_predict", agent, agent_base, agent_log,
                    lambda: train_gto(agent, gto_train_cfg, device, agent_log,
                                      scenarios_override=modified,
                                      temperature=agent_temperature))

            if pipeline_cfg.get("run_modelling", False):
                modelling_cfg = _merge_train_config(config, "modelling_train")
                _run_or_skip_phase(
                    "modelling_predict", agent, agent_base, agent_log,
                    lambda: train_modelling(agent, modelling_cfg, device, agent_log,
                                            scenarios_override=modified,
                                            temperature=agent_temperature))

    elif needs_training:
        # --- Single-agent training ---
        agent = ASI(log, config)
        agent.set_device(device)
        if agent_dir:
            agent.load_checkpoint(agent_dir)
        else:
            log("No agent_dir specified, agent initialized randomly")

        single_temperature = config.get("solver", {}).get("gto_temperature", 1.0)
        ev_train_cfg = _merge_train_config(config, "gto_ev_train")

        if pipeline_cfg.get("run_gto_ev", True):
            _, ev_run_dir = train_gto_ev(agent, ev_train_cfg, device, log,
                         scenarios_override=base_scenarios, temperature=single_temperature)
            if ev_run_dir:
                best_ckpt = os.path.join(ev_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_gto_probs", False):
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")
            _, probs_run_dir = train_gto_probs(agent, probs_train_cfg, device, log,
                            scenarios_override=base_scenarios, temperature=single_temperature)
            if probs_run_dir:
                best_ckpt = os.path.join(probs_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_gto_training", False):
            gto_train_cfg = _merge_train_config(config, "gto_train")
            _, gto_run_dir = train_gto(agent, gto_train_cfg, device, log,
                            scenarios_override=base_scenarios, temperature=single_temperature)
            if gto_run_dir:
                best_ckpt = os.path.join(gto_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_modelling", False):
            modelling_cfg = _merge_train_config(config, "modelling_train")
            train_modelling(agent, modelling_cfg, device, log,
                            scenarios_override=base_scenarios, temperature=single_temperature)

    # --- Opponent data generation + training ---
    if pipeline_cfg.get("run_opponent_data", False):
        from agent.train_scenarios.generation.generate_opponent import generate_opponent_dataset

        opp_cfg = config.get("opponent_data", {})
        opp_save_cfg = opp_cfg.get("save_dir", "")
        if opp_save_cfg and os.path.isabs(opp_save_cfg):
            opp_save_dir = opp_save_cfg
        else:
            opp_save_dir = os.path.join(base_dir, "opponent_dataset", log.init_time)

        opp_scenarios = generate_opponent_dataset(config, opp_save_dir, device, log)

        if pipeline_cfg.get("run_opponent_action_train", False) and opp_scenarios:
            from agent.train_scenarios.opponent_action_predict.train import train_opponent_action

            opp_train_cfg = config.get("opponent_action_train", {})

            if multi_agent:
                # Train each agent's opponent_action_head on shared opponent data
                save_dir_cfg = multi_agent.get("save_dir", "")
                if save_dir_cfg and os.path.isabs(save_dir_cfg):
                    save_base_dir_opp = save_dir_cfg
                else:
                    save_base_dir_opp = os.path.join(project_root, "data", version,
                                                     save_dir_cfg or name)

                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    modifiers = agent_cfg.get("modifiers", [])
                    agent_temperature = config.get("solver", {}).get("gto_temperature", 1.0)
                    for mod in modifiers:
                        if mod.get("type") == "temperature":
                            agent_temperature = mod["value"]

                    agent_base = os.path.join(save_base_dir_opp, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== Opponent Action Training: {agent_name} ===")

                    agent = ASI(agent_log, config)
                    agent.set_device(device)
                    # Load best checkpoint for this agent
                    agent.load_checkpoint(agent_base)

                    _, opp_run_dir = train_opponent_action(
                        agent, opp_train_cfg, device, agent_log,
                        scenarios_override=opp_scenarios,
                        temperature=agent_temperature,
                    )
                    if opp_run_dir:
                        best_ckpt = os.path.join(opp_run_dir, "best.pt")
                        if os.path.exists(best_ckpt):
                            agent.load_checkpoint(best_ckpt)
            else:
                # Single-agent
                agent = ASI(log, config)
                agent.set_device(device)
                if agent_dir:
                    agent.load_checkpoint(agent_dir)

                single_temp = config.get("solver", {}).get("gto_temperature", 1.0)
                _, opp_run_dir = train_opponent_action(
                    agent, opp_train_cfg, device, log,
                    scenarios_override=opp_scenarios,
                    temperature=single_temp,
                )
                if opp_run_dir:
                    best_ckpt = os.path.join(opp_run_dir, "best.pt")
                    if os.path.exists(best_ckpt):
                        agent.load_checkpoint(best_ckpt)

    # --- MCTS cyclic collect → train ---
    if pipeline_cfg.get("run_mcts_train", False):
        from agent.train_scenarios.mcts_predict.train import train_mcts
        from agent.mcts.collect import run_mcts_collection

        mcts_train_cfg = config.get("mcts_train", {})
        examples_dir = mcts_train_cfg.get("examples_dir", "")
        n_cycles = mcts_train_cfg.get("n_cycles", 1)
        n_hands_per_cycle = mcts_train_cfg.get("n_hands_per_cycle", 500)

        # Resolve agent save directory
        if multi_agent:
            save_dir_cfg = multi_agent.get("save_dir", "")
            if save_dir_cfg and os.path.isabs(save_dir_cfg):
                save_base_dir_mcts = save_dir_cfg
            else:
                save_base_dir_mcts = os.path.join(project_root, "data", version,
                                                   save_dir_cfg or name)
        else:
            save_base_dir_mcts = base_dir

        fallback_temp = config.get("solver", {}).get("gto_temperature", 1.0)

        if examples_dir:
            # Backwards compatible: pre-generated examples, single pass
            mcts_examples = _load_mcts_examples(examples_dir, log)
            if not mcts_examples:
                log("MCTS training skipped: no examples at examples_dir")
            elif multi_agent:
                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    agent_temp = fallback_temp
                    for mod in agent_cfg.get("modifiers", []):
                        if mod.get("type") == "temperature":
                            agent_temp = mod["value"]
                    agent_base = os.path.join(save_base_dir_mcts, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== MCTS Training: {agent_name} ===")
                    agent = ASI(agent_log, config)
                    agent.set_device(device)
                    agent.load_checkpoint(agent_base)
                    train_mcts(agent, mcts_train_cfg, device, agent_log,
                               mcts_examples, temperature=agent_temp)
            else:
                agent = ASI(log, config)
                agent.set_device(device)
                if agent_dir:
                    agent.load_checkpoint(agent_dir)
                train_mcts(agent, mcts_train_cfg, device, log,
                           mcts_examples, temperature=fallback_temp)
        else:
            # Cyclic self-play: collect → train → repeat.
            # Agents are loaded ONCE from disk and trained in-place across
            # cycles so we can save checkpoints less frequently than every
            # cycle without losing training progress. History is appended to
            # a single file per agent for continuous loss curves.

            save_every_cycles = max(1, mcts_train_cfg.get("save_every_cycles", 1))

            def _checkpoint_metadata(agent_obj, fallback_norm_stats):
                """Pull norm_stats / temperature from agent's checkpoint."""
                norm_stats = getattr(agent_obj, '_checkpoint_norm_stats', None)
                if norm_stats is None:
                    norm_stats = fallback_norm_stats
                return norm_stats

            _identity_norm = {"pot_mean": 0, "pot_std": 1,
                              "stack_mean": 0, "stack_std": 1,
                              "bets_mean": 0, "bets_std": 1,
                              "blind_mean": 0, "blind_std": 1,
                              "ev_mean": 0, "ev_std": 1}

            def _build_persistent_optim(agent_obj, ckpt, mcts_train_cfg,
                                         n_cycles, agent_log):
                """Construct a single Adam + warmup→cosine that survives
                across all cycles AND across pipeline runs.

                Restores Adam moments and scheduler state from the
                checkpoint when present, so cosine doesn't restart and β2
                moments don't reset every cycle.
                """
                from torch.optim.lr_scheduler import (
                    LinearLR, CosineAnnealingLR, SequentialLR)
                lr = float(mcts_train_cfg.get("lr", 1e-5))
                opt = torch.optim.Adam(agent_obj.parameters(), lr=lr)

                est_steps = int(mcts_train_cfg.get(
                    "estimated_steps_per_cycle", 20))
                # Generous horizon — slight over-estimation just delays
                # eta_min, under-estimation freezes early.
                total_steps = max(1, n_cycles * est_steps)
                warmup_steps = min(100, max(1, total_steps // 5))
                eta_min = float(mcts_train_cfg.get(
                    "scheduler_eta_min", 1e-6))

                warmup = LinearLR(opt, start_factor=0.01,
                                  total_iters=warmup_steps)
                cosine = CosineAnnealingLR(
                    opt, T_max=max(1, total_steps - warmup_steps),
                    eta_min=eta_min)
                sched = SequentialLR(opt, [warmup, cosine],
                                     milestones=[warmup_steps])

                restored_opt = False
                restored_sched = False
                if ckpt is not None:
                    opt_state = ckpt.get("optimizer_state_dict")
                    if opt_state is not None:
                        try:
                            opt.load_state_dict(opt_state)
                            restored_opt = True
                        except Exception as e:
                            agent_log(f"  optimizer state ignored: {e}")
                    sched_state = ckpt.get("scheduler_state_dict")
                    if sched_state is not None:
                        try:
                            sched.load_state_dict(sched_state)
                            restored_sched = True
                        except Exception as e:
                            agent_log(f"  scheduler state ignored: {e}")
                msg = (
                    f"  Persistent optim: lr_now={opt.param_groups[0]['lr']:.2e}, "
                    f"horizon={total_steps} steps "
                    f"(={n_cycles} cycles × {est_steps}), "
                    f"warmup={warmup_steps}, eta_min={eta_min:.0e}, "
                    f"adam_restored={restored_opt}, "
                    f"sched_restored={restored_sched}")
                agent_log(msg)
                return opt, sched

            # Build the persistent agent registry once
            trained_agents = []  # list of dicts kept across cycles
            if multi_agent:
                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    agent_base = os.path.join(save_base_dir_mcts, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== Loading {agent_name} for cyclic MCTS ===")

                    agent_obj = ASI(agent_log, config)
                    agent_obj.set_device(device)
                    agent_obj.load_checkpoint(agent_base)
                    agent_obj.eval()

                    # Resolve temperature from checkpoint or modifiers
                    ckpt_path = agent_obj._find_best_checkpoint(agent_base)
                    temp = fallback_temp
                    loaded_ckpt = None
                    if ckpt_path:
                        loaded_ckpt = torch.load(
                            ckpt_path, weights_only=False, map_location=device)
                        temp = loaded_ckpt.get("temperature", fallback_temp)
                    for mod in agent_cfg.get("modifiers", []):
                        if mod.get("type") == "temperature":
                            temp = mod["value"]

                    # New timestamp dir for THIS run's best.pt (cycle-aware
                    # save). History lives at scenario level (one above the
                    # timestamp dir) so it accumulates across pipeline runs
                    # — required for continuous loss curves.
                    run_dir = agent_log.run_dir("mcts_predict")
                    scenario_dir = os.path.dirname(run_dir)
                    history_path = os.path.join(scenario_dir, "history.pt")
                    cumulative_step = 0
                    if os.path.exists(history_path):
                        existing = torch.load(history_path, weights_only=False)
                        steps = existing.get("step_loss", []) or []
                        if steps and isinstance(steps[-1], dict):
                            cumulative_step = int(steps[-1].get("step", 0)) + 1

                    optimizer, scheduler = _build_persistent_optim(
                        agent_obj, loaded_ckpt, mcts_train_cfg,
                        n_cycles, agent_log)

                    trained_agents.append({
                        "agent": agent_obj,
                        "norm_stats": _checkpoint_metadata(agent_obj, _identity_norm),
                        "name": agent_name,
                        "temperature": temp,
                        "agent_log": agent_log,
                        "run_dir": run_dir,
                        "history_path": history_path,
                        "cumulative_step": cumulative_step,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                    })
            else:
                single_load_dir = agent_dir or save_base_dir_mcts
                agent_obj = ASI(log, config)
                agent_obj.set_device(device)
                agent_obj.load_checkpoint(single_load_dir)
                agent_obj.eval()
                run_dir = log.run_dir("mcts_predict")
                scenario_dir = os.path.dirname(run_dir)
                history_path = os.path.join(scenario_dir, "history.pt")
                cumulative_step = 0
                if os.path.exists(history_path):
                    existing = torch.load(history_path, weights_only=False)
                    steps = existing.get("step_loss", []) or []
                    if steps and isinstance(steps[-1], dict):
                        cumulative_step = int(steps[-1].get("step", 0)) + 1

                # Re-find checkpoint for optimizer/scheduler state
                single_ckpt_path = agent_obj._find_best_checkpoint(single_load_dir)
                single_ckpt = None
                if single_ckpt_path:
                    single_ckpt = torch.load(
                        single_ckpt_path, weights_only=False, map_location=device)
                optimizer, scheduler = _build_persistent_optim(
                    agent_obj, single_ckpt, mcts_train_cfg, n_cycles, log)

                trained_agents.append({
                    "agent": agent_obj,
                    "norm_stats": _checkpoint_metadata(agent_obj, _identity_norm),
                    "name": name,
                    "temperature": fallback_temp,
                    "agent_log": log,
                    "run_dir": run_dir,
                    "history_path": history_path,
                    "cumulative_step": cumulative_step,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                })

            # Periodic re-bootstrap of MCTS value norm. The agent's policy
            # drifts during cyclic training, so realised-outcome distribution
            # shifts. Clearing the mcts_ev_* keys forces run_mcts_collection
            # to recompute fresh stats from the current cycle's data. Set to
            # 0/None to disable (norm stats stay frozen from first bootstrap).
            value_norm_rebootstrap_every = mcts_train_cfg.get(
                "value_norm_rebootstrap_every", 0) or 0

            log(f"\nCyclic MCTS: {n_cycles} cycles, "
                f"save_every_cycles={save_every_cycles}, "
                f"value_norm_rebootstrap_every={value_norm_rebootstrap_every}")

            # New normalization key (post-Step-1 redesign): single per-agent
            # std of realized chip deltas, used as `value_scale` in
            # `_make_terminal_evaluator` and the final hybrid target. Legacy
            # `mcts_ev_*` keys are kept in the clear list for backward-compat
            # with checkpoints saved before the redesign.
            # See versions/v5/PLAN_MCTS_VALUE_REDESIGN.md for math.
            _MCTS_NORM_KEYS = ("mcts_ev_mean", "mcts_ev_std",
                               "mcts_ev_n_samples", "mcts_ev_ratio_min",
                               "mcts_ev_ratio_max",
                               "mcts_value_scale",
                               "mcts_value_scale_n_samples",
                               "mcts_value_chip_min",
                               "mcts_value_chip_max")

            for cycle in range(n_cycles):
                log(f"\n=== MCTS Cycle {cycle + 1}/{n_cycles} ===")

                # Re-bootstrap value norm? (only on non-zero cycle id, every N)
                if (value_norm_rebootstrap_every > 0
                        and cycle > 0
                        and cycle % value_norm_rebootstrap_every == 0):
                    for agent_info in trained_agents:
                        ns = agent_info.get("norm_stats")
                        if ns is not None:
                            had = any(k in ns for k in _MCTS_NORM_KEYS)
                            for k in _MCTS_NORM_KEYS:
                                ns.pop(k, None)
                            if had:
                                agent_info["agent_log"](
                                    f"  [{agent_info['name']}] cleared "
                                    f"mcts_ev_* — will re-bootstrap this cycle")

                # Use in-memory agents directly for collection (no disk reload)
                agents_for_play = [
                    {"agent": a["agent"], "norm_stats": a["norm_stats"],
                     "name": a["name"], "temperature": a["temperature"]}
                    for a in trained_agents
                ]

                per_agent_examples = run_mcts_collection(
                    agents_for_play, config, device, log, n_hands_per_cycle)

                is_save_cycle = (cycle % save_every_cycles == 0
                                 or cycle == n_cycles - 1)

                for agent_info in trained_agents:
                    agent_name = agent_info["name"]
                    examples = per_agent_examples.get(agent_name, [])
                    if not examples:
                        agent_info["agent_log"](
                            f"  {agent_name}: no examples this cycle, skipping training")
                        continue

                    agent_info["agent_log"](
                        f"\n--- MCTS Train: {agent_name} "
                        f"(cycle {cycle + 1}/{n_cycles}, {len(examples)} examples, "
                        f"save={is_save_cycle}) ---")

                    _, _, new_step = train_mcts(
                        agent_info["agent"], mcts_train_cfg, device,
                        agent_info["agent_log"], examples,
                        temperature=agent_info["temperature"],
                        run_dir=agent_info["run_dir"],
                        history_path=agent_info["history_path"],
                        cycle_id=cycle,
                        global_step_offset=agent_info["cumulative_step"],
                        save_checkpoint=is_save_cycle,
                        run_timestamp=getattr(agent_info["agent_log"],
                                               "init_time", None),
                        optimizer=agent_info["optimizer"],
                        scheduler=agent_info["scheduler"],
                    )
                    agent_info["cumulative_step"] = new_step

    # --- Evaluation (after all training stages) ---
    if pipeline_cfg.get("run_evaluation", False):
        run_evaluation(config, device, log)

    # --- Slumbot evaluation (external HU benchmark) ---
    if pipeline_cfg.get("run_slumbot_eval", False):
        from evaluation.slumbot_eval import run_slumbot_evaluation
        run_slumbot_evaluation(config, device, log)


def _load_mcts_examples(examples_dir, log):
    """Load pre-generated MCTS training examples from a directory."""
    import torch

    if not examples_dir:
        return None
    if not os.path.isdir(examples_dir):
        # Try as direct file path
        if os.path.isfile(examples_dir):
            log(f"Loading MCTS examples from {examples_dir}")
            return torch.load(examples_dir, weights_only=False)
        log(f"MCTS examples path not found: {examples_dir}")
        return None

    # Search for latest examples file in directory
    candidates = sorted(
        [f for f in os.listdir(examples_dir) if f.endswith(".pt")],
        reverse=True,
    )
    if not candidates:
        log(f"No .pt files found in {examples_dir}")
        return None

    path = os.path.join(examples_dir, candidates[0])
    log(f"Loading MCTS examples from {path}")
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    main()
