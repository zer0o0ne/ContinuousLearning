"""Evaluate saved iterations without collecting labels or training.

python pool_eval_pipeline.py --iteration 7 --reference-iteration 3 --run compare_7_3
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from pipeline import (agent_variant_members, embedding_vintages,
                      frozen_agent_net, pool_evaluation)
from pool.build import build_pool
from utils import Logger, resolve_device


def run_saved(config, exp_dir, iteration, reference=None, log=print):
    if iteration < 0 or (reference is not None and not -1 <= reference < iteration):
        raise ValueError("Choose a nonnegative iteration and an earlier reference (-1 = agent_init)")
    config = copy.deepcopy(config)
    config.setdefault("pool_evaluation", {})["enabled"] = True
    game = config["game"]
    device = resolve_device(config.get("device", "auto"))
    seed = int(config.get("seed", 0))
    pool, descriptors = build_pool(config, np.random.default_rng(seed), device=device, log=log)
    bootstrap_size = len(pool)
    generations = embedding_vintages(str(exp_dir), iteration+1, config, game, device, True, log)
    conditioned = config["embedding_net"].get("pool_agent_vectors", "zero") == "amortised"
    new_net = measured = None
    for k in range(iteration+1):
        path = Path(exp_dir) / f"iter_{k:04d}" / "agent.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        measured = checkpoint.get("config") or config
        net = frozen_agent_net(checkpoint["model_state_dict"], measured, game, device)
        if k == iteration:
            new_net = net
        else:
            members, desc = agent_variant_members(net, k, config, game, device, seed,
                                                  embed_net=generations[k] if conditioned else None)
            pool.extend(members)
            descriptors.extend(desc)
    return pool_evaluation(pool, descriptors, new_net, generations[iteration], measured,
                            config, str(exp_dir), iteration, bootstrap_size, device, log,
                            reference_iteration=reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--reference-iteration", type=int)
    parser.add_argument("--run", help="New output/seed namespace, e.g. compare_7_3")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.run:
        config.setdefault("pool_evaluation", {})["run"] = args.run
    root = Path(config.get("out_dir", "../../data/v8"))
    log = Logger(str(root))
    try:
        run_saved(config, root/config["experiment"], args.iteration,
                   args.reference_iteration, log)
    finally:
        log.close()


if __name__ == "__main__":
    main()
