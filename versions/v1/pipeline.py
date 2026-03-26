import os
import json

from utils import Logger
from agent.agent import ASI


def _merge_train_config(config, scenario_key):
    """Merge game + solver + scenario-specific config into a flat dict.

    Args:
        config: full config dict
        scenario_key: e.g. "gto_ev_train" or "gto_probs_train"

    Returns:
        merged flat dict for the training scenario
    """
    merged = {}
    merged.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    # Rename "type" to "solver" for backward compat with generate.py
    merged["solver"] = solver_cfg.pop("type", "v2")
    merged.update(solver_cfg)
    merged.update(config.get(scenario_key, {}))
    return merged


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

    # Parse version from directory name (e.g. "v0" from "versions/v0")
    version = os.path.basename(os.path.abspath(os.path.dirname(__file__)))
    name = config.get("name", "default")

    # All run artifacts go under <project_root>/data/<version>/<name>/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    base_dir = os.path.join(project_root, "data", version, name)

    log = Logger(base_dir)
    log(f"Version: {version}, Experiment: {name}, Device: {device}")

    agent = ASI(log, config)
    agent.set_device(device)

    agent_dir = config.get("agent_dir", "")
    if agent_dir:
        agent.load_checkpoint(agent_dir)
    else:
        log("No agent_dir specified, agent initialized randomly")

    pipeline_cfg = config.get("pipeline", {})
    ev_run_dir = None

    # Step 1: GTO EV prediction training
    if pipeline_cfg.get("run_gto_ev", True):
        train_cfg = _merge_train_config(config, "gto_ev_train")
        from agent.train_scenarios.gto_ev_predict.train import train_gto_ev
        result = train_gto_ev(agent, train_cfg, device, log)
        if result is not None:
            _, ev_run_dir = result

    # Step 2: GTO action probability prediction training
    if pipeline_cfg.get("run_gto_probs", False):
        train_cfg = _merge_train_config(config, "gto_probs_train")
        # Reuse dataset from step 1 if no dataset_dir specified
        if not train_cfg.get("dataset_dir") and ev_run_dir:
            dataset_path = os.path.join(ev_run_dir, "dataset.pt")
            if os.path.exists(dataset_path):
                train_cfg["dataset_dir"] = ev_run_dir
                log(f"Reusing dataset from gto_ev_predict: {ev_run_dir}")
        from agent.train_scenarios.gto_probs_predict.train import train_gto_probs
        train_gto_probs(agent, train_cfg, device, log)


if __name__ == "__main__":
    main()
