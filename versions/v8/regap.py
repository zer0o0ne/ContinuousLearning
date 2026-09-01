"""Recompute §8's oracle gap over iterations that already ran (`pipeline.py` D).

Phase D is a `softmax` over the `q` stored in the label shards and one batched
agent forward — no rollouts and no embedding network — so it can be re-run after
the fact for any iteration whose `labels/` shards and `agent.pt` are still on
disk. That is what this script does, for the sole purpose of reporting the terms
`gap_terms` computes — warm and cold, exactly as the loop's phase D does:
nothing here trains, and nothing here touches the labels, the weights, or the
sampler state.

Determinism is what makes the result the same number the run would have written:
the held-out split is `split_heldout(n, fraction, seed, iteration)` and reads no
running RNG, and the agent is the checkpoint that iteration saved. Every number
the old `metrics.json` already carries is therefore checked against the
recomputed one, and a mismatch is an error rather than a silent overwrite.

**The config is the one the iteration ran with, not the one on disk now.**
`agent.pt` carries the config of its own iteration, and that is what phase D
read: `oracle.temperature` shapes `π_oracle`, so re-measuring an old checkpoint
under a temperature that was edited since would silently report `kl` and
`ev_gap_target` for a target that agent was never trained toward. `--config`
therefore only locates the experiment (`out_dir`, `experiment`, `n_iterations`);
every number that enters the measurement comes from the checkpoint, and a
difference between the two is logged per iteration.

Run::

    cd versions/v8 && python3 regap.py --config config.json
"""

import argparse
import json
import os

import torch

from pipeline import (GAP_KEYS, _iter_dir, _read_json, _write_json,
                      frozen_agent_net, oracle_gap, split_heldout,
                      winrate_line)
from train.generate import load_shard
from utils import Logger, resolve_device


def _flat(gap, prefix=()):
    """`{(scope, group, key): number}` — one flat view of a gap report."""
    out = {}
    for key, value in gap.items():
        if isinstance(value, dict):
            out.update(_flat(value, prefix + (str(key),)))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix + (str(key),)] = float(value)
    return out


def regap_iteration(config, exp_dir, iteration, device, log):
    """Recompute one iteration's gap in place. Returns its `metrics.json`."""
    it_dir = _iter_dir(exp_dir, iteration)
    labels_path = os.path.join(it_dir, "labels.json")
    agent_path = os.path.join(it_dir, "agent.pt")
    if not (os.path.exists(labels_path) and os.path.exists(agent_path)):
        log(f"[regap] iteration {iteration}: no labels.json or agent.pt — "
            f"skipped")
        return None

    manifest = _read_json(labels_path)["manifest"]
    labels = [lab for path in manifest["shards"] for lab in load_shard(path)]
    assert len(labels) == manifest["n_labels"], (
        f"{len(labels)} labels on disk, manifest says {manifest['n_labels']}")

    ckpt = torch.load(agent_path, map_location=device, weights_only=False)
    run_config = ckpt.get("config") or config
    for section, key in (("oracle", "temperature"), ("oracle", "divisor"),
                         ("agent_train", "heldout_fraction"), (None, "seed")):
        here = (config if section is None else config.get(section, {})).get(key)
        then = (run_config if section is None
                else run_config.get(section, {})).get(key)
        if here != then:
            log(f"[regap] iteration {iteration}: {section or ''}.{key} was "
                f"{then!r} when the run measured it and is {here!r} in the "
                f"config on disk now — measuring with {then!r}")

    train_cfg = run_config["agent_train"]
    oracle_cfg = run_config["oracle"]
    _train_idx, held_idx = split_heldout(
        len(labels), train_cfg.get("heldout_fraction", 0.0),
        int(run_config.get("seed", 0)), iteration)

    net = frozen_agent_net(ckpt["model_state_dict"], run_config,
                           run_config["game"], device)
    gap_args = (net, [labels[i] for i in held_idx],
                float(oracle_cfg["temperature"]),
                oracle_cfg.get("divisor", "pot_plus_bet"),
                int(train_cfg["batch_hands"]), device, log)
    gaps = {"gap": oracle_gap(*gap_args),
            "gap_cold": oracle_gap(*gap_args, cold=True)}

    metrics_path = os.path.join(it_dir, "metrics.json")
    metrics = _read_json(metrics_path) if os.path.exists(metrics_path) else {}
    for name, gap in gaps.items():
        was, now = _flat(metrics.get(name, {})), _flat(gap)
        for key in sorted(set(was) & set(now)):
            assert abs(was[key] - now[key]) < 1e-9, (
                f"iteration {iteration}: recomputing {name}/{'/'.join(key)} "
                f"gives {now[key]} where metrics.json says {was[key]} — the "
                f"recomputation is not reproducing the run, so none of its "
                f"other numbers can be trusted either")
    metrics.update({"iteration": iteration, **gaps})
    _write_json(metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="recompute §8's oracle gap")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)
    base_dir = config.get("out_dir", "../../data/v8")
    exp_dir = os.path.join(base_dir, config["experiment"])
    log = Logger(base_dir)
    device = resolve_device(config.get("device", "auto"))
    log(f"[regap] {exp_dir} on {device}")

    try:
        metrics_all = []
        for k in range(int(config["n_iterations"])):
            metrics = regap_iteration(config, exp_dir, k, device, log)
            if metrics is None:
                continue
            metrics_all.append(metrics)
            o = metrics["gap"].get("overall")
            if o:
                log("[regap] iteration {}: ".format(k)
                    + " ".join(f"{key}={o[key]:+.4f}" for key in GAP_KEYS))
            log(f"[regap] iteration {k}: "
                + winrate_line(metrics["gap"], metrics["gap_cold"]))
        report_path = os.path.join(exp_dir, "report.json")
        if metrics_all and os.path.exists(report_path):
            report = _read_json(report_path)
            report["metrics"] = metrics_all
            _write_json(report_path, report)
            log(f"[regap] rewrote {report_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
