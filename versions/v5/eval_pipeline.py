"""
Evaluation-only pipeline. Loads eval_config.json and runs the configured
evaluation flows. Results land in data/<version>/evaluation/<datetime>/.

Two independent sections in eval_config.json:
  * internal_evaluation — agent-vs-agent self-play (multi-table batched)
  * slumbot_evaluation  — heads-up vs Slumbot HTTP API

Each section toggles with `enabled: true|false` and carries its own table/bot
parameters plus an explicit list of agents (path + optional behavior flags).

action_temperature precedence inside each agent bundle:
  checkpoint.temperature > entry.action_temperature > section.action_temperature
"""

import json
import os
from datetime import datetime

import torch

from utils import Logger
from evaluation.evaluate import run_evaluation
from evaluation.slumbot_eval import run_slumbot_evaluation


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    version = os.path.basename(here)
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    return version, project_root, here


def _build_internal_config(eval_config, internal_cfg):
    """Build a dict-shape that evaluate.run_evaluation expects.

    Maps internal_evaluation -> evaluation key, preserves architecture/game.
    """
    return {
        "name": eval_config.get("name", "eval"),
        "architecture": eval_config.get("architecture", {}),
        "game": eval_config.get("game", {}),
        "mcts": eval_config.get("mcts", {}),
        "evaluation": internal_cfg,
        # opponent_data section is optional in evaluate.py for swap defaults; if
        # absent it falls back to per-section keys.
        "opponent_data": {},
    }


def _build_slumbot_config(eval_config, slumbot_cfg):
    """Map slumbot_evaluation -> slumbot_eval key for slumbot_eval module."""
    return {
        "name": eval_config.get("name", "eval"),
        "architecture": eval_config.get("architecture", {}),
        "game": eval_config.get("game", {}),
        "mcts": eval_config.get("mcts", {}),
        "slumbot_eval": slumbot_cfg,
    }


def main():
    version, project_root, here = _resolve_paths()
    config_path = os.path.join(here, "eval_config.json")
    with open(config_path, "r") as f:
        eval_config = json.load(f)

    device = _pick_device()

    # data/<version>/evaluation/<datetime>/
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_results_dir = os.path.join(
        project_root, "data", version, "evaluation", timestamp,
    )
    os.makedirs(base_results_dir, exist_ok=True)

    log = Logger(base_results_dir)
    log(f"Eval pipeline — version={version}, device={device}")
    log(f"Results dir: {base_results_dir}")

    # Snapshot config
    snapshot_path = os.path.join(base_results_dir, "eval_config.json")
    with open(snapshot_path, "w") as f:
        json.dump(eval_config, f, indent=4)
    log(f"Config snapshot saved to {snapshot_path}")

    # --- Internal evaluation ---
    internal_cfg = eval_config.get("internal_evaluation", {})
    if internal_cfg.get("enabled", False):
        log("\n=========================================")
        log("=== Internal evaluation ===")
        log("=========================================")
        internal_results_dir = os.path.join(base_results_dir, "internal")
        os.makedirs(internal_results_dir, exist_ok=True)
        cfg = _build_internal_config(eval_config, internal_cfg)
        run_evaluation(cfg, device, log, results_dir_override=internal_results_dir)
    else:
        log("\nInternal evaluation: disabled, skipping")

    # --- Slumbot evaluation ---
    slumbot_cfg = eval_config.get("slumbot_evaluation", {})
    if slumbot_cfg.get("enabled", False):
        log("\n=========================================")
        log("=== Slumbot evaluation ===")
        log("=========================================")
        slumbot_results_dir = os.path.join(base_results_dir, "slumbot")
        os.makedirs(slumbot_results_dir, exist_ok=True)
        cfg = _build_slumbot_config(eval_config, slumbot_cfg)
        run_slumbot_evaluation(cfg, device, log, results_dir_override=slumbot_results_dir)
    else:
        log("\nSlumbot evaluation: disabled, skipping")

    log("\nEval pipeline complete.")


if __name__ == "__main__":
    main()
