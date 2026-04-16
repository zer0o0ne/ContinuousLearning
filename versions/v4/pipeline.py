import os
import json
import copy

from utils import Logger
from agent.agent import ASI


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

        for agent_cfg in multi_agent["agents"]:
            agent_name = agent_cfg["name"]
            modifiers = agent_cfg.get("modifiers", [])

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
                _, ev_run_dir = train_gto_ev(agent, ev_train_cfg, device, agent_log,
                             scenarios_override=modified, temperature=agent_temperature)
                # Reload best EV checkpoint so probs training starts from best weights
                if ev_run_dir:
                    best_ckpt = os.path.join(ev_run_dir, "best.pt")
                    if os.path.exists(best_ckpt):
                        agent.load_checkpoint(best_ckpt)

            if pipeline_cfg.get("run_gto_probs", False):
                _, probs_run_dir = train_gto_probs(agent, probs_train_cfg, device, agent_log,
                                scenarios_override=modified, temperature=agent_temperature)
                if probs_run_dir:
                    best_ckpt = os.path.join(probs_run_dir, "best.pt")
                    if os.path.exists(best_ckpt):
                        agent.load_checkpoint(best_ckpt)

            if pipeline_cfg.get("run_gto_training", False):
                gto_train_cfg = _merge_train_config(config, "gto_train")
                _, gto_run_dir = train_gto(agent, gto_train_cfg, device, agent_log,
                                scenarios_override=modified, temperature=agent_temperature)
                if gto_run_dir:
                    best_ckpt = os.path.join(gto_run_dir, "best.pt")
                    if os.path.exists(best_ckpt):
                        agent.load_checkpoint(best_ckpt)

            if pipeline_cfg.get("run_modelling", False):
                modelling_cfg = _merge_train_config(config, "modelling_train")
                train_modelling(agent, modelling_cfg, device, agent_log,
                                scenarios_override=modified, temperature=agent_temperature)

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

        if examples_dir:
            # Backwards compatible: pre-generated examples, single pass
            mcts_examples = _load_mcts_examples(examples_dir, log)
            if not mcts_examples:
                log("MCTS training skipped: no examples at examples_dir")
            elif multi_agent:
                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    agent_base = os.path.join(save_base_dir_mcts, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== MCTS Training: {agent_name} ===")
                    agent = ASI(agent_log, config)
                    agent.set_device(device)
                    agent.load_checkpoint(agent_base)
                    train_mcts(agent, mcts_train_cfg, device, agent_log, mcts_examples)
            else:
                agent = ASI(log, config)
                agent.set_device(device)
                if agent_dir:
                    agent.load_checkpoint(agent_dir)
                train_mcts(agent, mcts_train_cfg, device, log, mcts_examples)
        else:
            # Cyclic self-play: collect → train → repeat
            fallback_temp = config.get("solver", {}).get("gto_temperature", 1.0)

            for cycle in range(n_cycles):
                log(f"\n=== MCTS Cycle {cycle + 1}/{n_cycles} ===")

                # Load latest agent checkpoints for collection
                agents_for_play = []
                if multi_agent:
                    for agent_cfg in multi_agent["agents"]:
                        agent_name = agent_cfg["name"]
                        agent_base = os.path.join(save_base_dir_mcts, agent_name)
                        agent_obj = ASI(log, config)
                        agent_obj.set_device(device)
                        agent_obj.load_checkpoint(agent_base)
                        agent_obj.eval()

                        # Get temperature + norm_stats from checkpoint
                        ckpt_path = agent_obj._find_best_checkpoint(agent_base)
                        if ckpt_path:
                            ckpt = torch.load(ckpt_path, weights_only=False,
                                              map_location=device)
                            norm_stats = ckpt.get("norm_stats")
                            temp = ckpt.get("temperature", fallback_temp)
                        else:
                            norm_stats = None
                            temp = fallback_temp
                        if norm_stats is None:
                            norm_stats = {"pot_mean": 0, "pot_std": 1,
                                          "stack_mean": 0, "stack_std": 1,
                                          "bets_mean": 0, "bets_std": 1,
                                          "blind_mean": 0, "blind_std": 1}

                        agents_for_play.append({
                            "agent": agent_obj, "norm_stats": norm_stats,
                            "name": agent_name, "temperature": temp,
                        })
                else:
                    agent_obj = ASI(log, config)
                    agent_obj.set_device(device)
                    if agent_dir:
                        agent_obj.load_checkpoint(agent_dir)
                    agent_obj.eval()
                    norm_stats = getattr(agent_obj, '_checkpoint_norm_stats', None)
                    if norm_stats is None:
                        norm_stats = {"pot_mean": 0, "pot_std": 1,
                                      "stack_mean": 0, "stack_std": 1,
                                      "bets_mean": 0, "bets_std": 1,
                                      "blind_mean": 0, "blind_std": 1}
                    agents_for_play.append({
                        "agent": agent_obj, "norm_stats": norm_stats,
                        "name": name, "temperature": fallback_temp,
                    })

                # Collect
                per_agent_examples = run_mcts_collection(
                    agents_for_play, config, device, log, n_hands_per_cycle)

                # Train each agent on its collected examples
                if multi_agent:
                    for agent_cfg in multi_agent["agents"]:
                        agent_name = agent_cfg["name"]
                        examples = per_agent_examples.get(agent_name, [])
                        if not examples:
                            log(f"  {agent_name}: no examples, skipping training")
                            continue

                        agent_base = os.path.join(save_base_dir_mcts, agent_name)
                        agent_log = Logger(agent_base)
                        agent_log(f"\n=== MCTS Train: {agent_name} "
                                  f"(cycle {cycle + 1}, {len(examples)} examples) ===")

                        agent = ASI(agent_log, config)
                        agent.set_device(device)
                        agent.load_checkpoint(agent_base)

                        _, mcts_run_dir = train_mcts(
                            agent, mcts_train_cfg, device, agent_log, examples)
                        if mcts_run_dir:
                            best_ckpt = os.path.join(mcts_run_dir, "best.pt")
                            if os.path.exists(best_ckpt):
                                agent.load_checkpoint(best_ckpt)
                else:
                    examples = per_agent_examples.get(name, [])
                    if examples:
                        log(f"\n=== MCTS Train (cycle {cycle + 1}, "
                            f"{len(examples)} examples) ===")
                        agent = ASI(log, config)
                        agent.set_device(device)
                        if agent_dir:
                            agent.load_checkpoint(agent_dir)
                        train_mcts(agent, mcts_train_cfg, device, log, examples)

    # --- Evaluation (after all training stages) ---
    if pipeline_cfg.get("run_evaluation", False):
        run_evaluation(config, device, log)


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
