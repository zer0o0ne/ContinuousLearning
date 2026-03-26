import os
import json

from utils import Logger
from agent.agent import ASI


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

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

    # Step 1: GTO EV prediction training
    train_cfg = config["gto_ev_train"]
    from agent.train_scenarios.gto_ev_predict.train import train_gto_ev
    train_gto_ev(agent, train_cfg, device, log)


if __name__ == "__main__":
    main()
