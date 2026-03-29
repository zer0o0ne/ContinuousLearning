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
        # Not found — generate into dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        log(f"Dataset not found at {dataset_dir}, generating there...")
        return generate_dataset(gen_cfg, dataset_dir, log=log)

    # No dataset_dir — generate into base_dir/dataset/<timestamp>
    dataset_save_dir = os.path.join(base_dir, "dataset", log.init_time)
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
    from evaluation.evaluate import run_evaluation

    agent_dir = config.get("agent_dir", "")
    pipeline_cfg = config.get("pipeline", {})
    multi_agent = config.get("multi_agent")

    needs_training = pipeline_cfg.get("run_gto_ev", True) or pipeline_cfg.get("run_gto_probs", False)

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
        n_actions = game_cfg.get("table_bins", 50) + 3
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
                agent.load_checkpoint(agent_dir)
            else:
                agent_log("Agent initialized randomly")

            modified = apply_modifiers(base_scenarios, modifiers, n_actions,
                                       big_blind, temperature)

            ev_train_cfg = _merge_train_config(config, "gto_ev_train")
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")

            if pipeline_cfg.get("run_gto_ev", True):
                train_gto_ev(agent, ev_train_cfg, device, agent_log,
                             scenarios_override=modified, temperature=agent_temperature)

            if pipeline_cfg.get("run_gto_probs", False):
                train_gto_probs(agent, probs_train_cfg, device, agent_log,
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
            train_gto_ev(agent, ev_train_cfg, device, log,
                         scenarios_override=base_scenarios, temperature=single_temperature)

        if pipeline_cfg.get("run_gto_probs", False):
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")
            train_gto_probs(agent, probs_train_cfg, device, log,
                            scenarios_override=base_scenarios, temperature=single_temperature)

    # --- Evaluation ---
    if pipeline_cfg.get("run_evaluation", False):
        run_evaluation(config, device, log)


if __name__ == "__main__":
    main()
