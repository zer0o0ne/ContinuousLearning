"""MCTS training analytics for a single agent.

Reads `<agent_mcts_dir>/history.pt` (the cross-cycle accumulated history file
produced by the cyclic MCTS training pipeline) and produces diagnostic plots
plus a per-cycle summary table.

Usage:
    python -m analytics.mcts_training --agent_dir <path>
    # or directly:
    python analytics/mcts_training.py \
        --agent_dir /Users/.../data/v5/5_final_agents/gto_pure/mcts_predict

`agent_dir` must point to the `mcts_predict` scenario folder (the one that
contains `history.pt` and per-run `<timestamp>/best.pt`). Output PNGs are
written to `<agent_dir>/analysis/` by default.

Plots produced:
    1. mcts_training_overview.png   — 2x3 panel: train/val components, best val
                                       per cycle, LR schedule, dataset sizes,
                                       train-vs-val gap per epoch.
    2. mcts_component_balance.png   — stacked train components + val component
                                       shares (where each loss term dominates).
    3. mcts_cycle_progression.png   — train losses colour-coded by cycle id +
                                       per-cycle best val markers.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _VERSION_DIR not in sys.path:
    sys.path.insert(0, _VERSION_DIR)

from agent.train_scenarios._history import IncrementalHistory  # noqa: E402


# ---------- Loading helpers ----------

def _column(records, key):
    return np.asarray([r[key] for r in records], dtype=float)


def _smooth(x, w):
    """Moving-average smoothing with reflection at edges."""
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad-2:-1]])
    return np.convolve(xp, np.ones(w) / w, mode="valid")[: len(x)]


_HISTORY_KEYS = ["step_loss", "val_loss", "epoch_train_loss",
                 "epoch_val_loss", "cycles"]


def load_history(agent_dir):
    hist_path = os.path.join(agent_dir, "history.pt")
    shard_dir = os.path.join(agent_dir, "history_shards")
    if not os.path.isfile(hist_path) and not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"Neither history.pt nor history_shards/ found at {agent_dir}")
    hist = IncrementalHistory(agent_dir, keys=_HISTORY_KEYS)
    return hist.data


# ---------- Plot 1: 2x3 overview ----------

def plot_overview(hist, out_path, agent_name, smooth_window=11):
    step_loss = hist["step_loss"]
    val_loss = hist["val_loss"]
    epoch_train = hist["epoch_train_loss"]
    epoch_val = hist["epoch_val_loss"]
    cycles = hist["cycles"]

    s_step = _column(step_loss, "step")
    s_total = _column(step_loss, "total")
    s_value = _column(step_loss, "value")
    s_action = _column(step_loss, "action")
    s_chain = _column(step_loss, "chain")
    s_lr = _column(step_loss, "lr")

    v_step = _column(val_loss, "step")
    v_total = _column(val_loss, "total")
    v_value = _column(val_loss, "value")
    v_action = _column(val_loss, "action")
    v_chain = _column(val_loss, "chain")

    c_id = np.asarray([c["cycle_id"] for c in cycles], dtype=int)
    c_train = np.asarray([c["train_size"] for c in cycles], dtype=int)
    c_val = np.asarray([c["val_size"] for c in cycles], dtype=int)
    c_best_val = np.asarray(
        [c["best_val_loss_in_cycle"] if c["best_val_loss_in_cycle"] is not None
         else np.nan for c in cycles], dtype=float)
    c_saved = np.asarray([c["saved_checkpoint"] for c in cycles], dtype=bool)
    c_step_end = np.asarray([c["step_end"] for c in cycles], dtype=int)
    cycle_boundaries = c_step_end[:-1] if len(c_step_end) > 1 else np.array([])

    e_train = _column(epoch_train, "total") if epoch_train else np.array([])
    e_val = _column(epoch_val, "total") if epoch_val else np.array([])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"MCTS training analytics — {agent_name} "
        f"({len(cycles)} cycles, {len(step_loss)} training steps)",
        fontsize=14,
    )

    # (0,0) Train loss components
    ax = axes[0, 0]
    W = smooth_window
    ax.plot(s_step, _smooth(s_total, W), label="total", color="black", lw=1.5)
    ax.plot(s_step, _smooth(s_value, W), label="value", color="C0", alpha=0.8)
    ax.plot(s_step, _smooth(s_action, W), label="action", color="C1", alpha=0.8)
    ax.plot(s_step, _smooth(s_chain, W), label="chain", color="C2", alpha=0.8)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step (cumulative)")
    ax.set_ylabel("loss")
    ax.set_title(f"Train loss components (smooth window={W})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (0,1) Validation loss components
    ax = axes[0, 1]
    if len(v_step) > 0:
        ax.plot(v_step, v_total, "o-", label="total", color="black", lw=1.5, ms=4)
        ax.plot(v_step, v_value, "o-", label="value", color="C0", alpha=0.8, ms=3)
        ax.plot(v_step, v_action, "o-", label="action", color="C1", alpha=0.8, ms=3)
        ax.plot(v_step, v_chain, "o-", label="chain", color="C2", alpha=0.8, ms=3)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss components")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (0,2) Best val per cycle
    ax = axes[0, 2]
    ax.plot(c_id, c_best_val, "o-", color="darkblue", lw=1.2, ms=5,
            label="best val in cycle")
    saved_idx = np.where(c_saved)[0]
    if len(saved_idx) > 0:
        ax.scatter(c_id[saved_idx], c_best_val[saved_idx], marker="*", s=180,
                   color="orange", edgecolor="black", linewidth=0.5, zorder=5,
                   label=f"saved ({c_saved.sum()})")
    ax.set_xlabel("cycle id")
    ax.set_ylabel("best val loss")
    ax.set_title("Per-cycle best val loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (1,0) LR schedule
    ax = axes[1, 0]
    ax.plot(s_step, s_lr, color="purple", lw=0.7)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (warmup + cosine per cycle)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # (1,1) Examples per cycle
    ax = axes[1, 1]
    width = 0.35
    ax.bar(c_id - width / 2, c_train, width, label="train", color="C0")
    ax.bar(c_id + width / 2, c_val, width, label="val", color="C1")
    ax.set_xlabel("cycle id")
    ax.set_ylabel("examples")
    ax.set_title(
        f"Dataset size per cycle "
        f"(mean: train={c_train.mean():.0f}, val={c_val.mean():.0f})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # (1,2) Train vs Val per epoch
    ax = axes[1, 2]
    if len(e_train) > 0 and len(e_train) == len(e_val):
        epoch_idx = np.arange(len(e_train))
        ax.plot(epoch_idx, e_train, "o-", label="train", color="C0", lw=1.2, ms=4)
        ax.plot(epoch_idx, e_val, "o-", label="val", color="C3", lw=1.2, ms=4)
        # Cycle boundaries (each cycle has epochs_run epochs)
        ep_cumulative = 0
        for cyc in cycles:
            n_eps = cyc.get("epochs_run") or 1
            ep_cumulative += n_eps
            if ep_cumulative < len(e_train):
                ax.axvline(ep_cumulative - 0.5, color="gray", lw=0.3, alpha=0.5)
        ax.set_xlabel("epoch index (cumulative)")
        ax.set_ylabel("loss")
        ax.set_title("Train vs Val per epoch (gap = overfitting signal)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 2: component balance ----------

def plot_component_balance(hist, out_path, smooth_window=11):
    step_loss = hist["step_loss"]
    val_loss = hist["val_loss"]

    s_step = _column(step_loss, "step")
    s_value = _column(step_loss, "value")
    s_action = _column(step_loss, "action")
    s_chain = _column(step_loss, "chain")

    v_step = _column(val_loss, "step")
    v_total = _column(val_loss, "total")
    v_value = _column(val_loss, "value")
    v_action = _column(val_loss, "action")
    v_chain = _column(val_loss, "chain")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Loss component balance over training", fontsize=14)

    # Train: stacked area
    ax = axes[0]
    W = smooth_window
    v_smooth = _smooth(s_value, W)
    a_smooth = _smooth(s_action, W)
    c_smooth = _smooth(s_chain, W)
    ax.fill_between(s_step, 0, v_smooth, color="C0", alpha=0.6, label="value")
    ax.fill_between(s_step, v_smooth, v_smooth + a_smooth,
                    color="C1", alpha=0.6, label="action")
    ax.fill_between(s_step, v_smooth + a_smooth,
                    v_smooth + a_smooth + c_smooth,
                    color="C2", alpha=0.6, label="chain")
    ax.set_xlabel("training step")
    ax.set_ylabel("contribution to total loss (smoothed)")
    ax.set_title(f"Train: components stacked (window={W})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Val: relative shares
    ax = axes[1]
    if len(v_step) > 0:
        v_share = v_value / np.maximum(v_total, 1e-9)
        a_share = v_action / np.maximum(v_total, 1e-9)
        c_share = v_chain / np.maximum(v_total, 1e-9)
        ax.plot(v_step, v_share, "o-", color="C0", label="value", ms=3)
        ax.plot(v_step, a_share, "o-", color="C1", label="action", ms=3)
        ax.plot(v_step, c_share, "o-", color="C2", label="chain", ms=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of total val loss")
    ax.set_title("Val: relative weight of each component")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 3: cycle progression ----------

def plot_cycle_progression(hist, out_path):
    step_loss = hist["step_loss"]
    cycles = hist["cycles"]

    s_step = _column(step_loss, "step")
    s_total = _column(step_loss, "total")
    s_cycle = np.asarray([r["cycle_id"] for r in step_loss], dtype=int)

    c_step_end = np.asarray([c["step_end"] for c in cycles], dtype=int)
    c_best_val = np.asarray(
        [c["best_val_loss_in_cycle"] if c["best_val_loss_in_cycle"] is not None
         else np.nan for c in cycles], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    n_cyc = max(1, len(cycles) - 1)
    for c_idx, cyc in enumerate(cycles):
        mask = s_cycle == cyc["cycle_id"]
        if mask.any():
            ax.plot(s_step[mask], s_total[mask], lw=1, alpha=0.7,
                    color=plt.cm.viridis(c_idx / n_cyc))
        if cyc.get("saved_checkpoint"):
            ax.axvline(cyc["step_end"], color="orange", lw=0.6, alpha=0.5)

    ax.scatter(c_step_end, c_best_val, marker="D", color="red", s=40,
               edgecolor="black", linewidth=0.5, zorder=5,
               label=f"cycle best val ({len(cycles)} cycles)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Train loss by cycle (gradient color) + per-cycle best val")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=n_cyc))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03)
    cb.set_label("cycle id")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Numeric summary ----------

def print_summary(hist, log=print):
    step_loss = hist["step_loss"]
    val_loss = hist["val_loss"]
    epoch_train = hist["epoch_train_loss"]
    epoch_val = hist["epoch_val_loss"]
    cycles = hist["cycles"]

    e_train = _column(epoch_train, "total") if epoch_train else np.array([])
    e_val = _column(epoch_val, "total") if epoch_val else np.array([])
    e_cycle = (np.asarray([r.get("cycle_id", -1) for r in epoch_train], dtype=int)
               if epoch_train else np.array([], dtype=int))

    s_total = _column(step_loss, "total")
    s_value = _column(step_loss, "value")
    s_action = _column(step_loss, "action")
    s_chain = _column(step_loss, "chain")

    v_total = _column(val_loss, "total")
    v_value = _column(val_loss, "value")
    v_action = _column(val_loss, "action")
    v_chain = _column(val_loss, "chain")

    c_id = np.asarray([c["cycle_id"] for c in cycles], dtype=int)
    c_best_val = np.asarray(
        [c["best_val_loss_in_cycle"] if c["best_val_loss_in_cycle"] is not None
         else np.nan for c in cycles], dtype=float)
    c_saved = np.asarray([c["saved_checkpoint"] for c in cycles], dtype=bool)

    log("=" * 75)
    log(f"{'Cycle':>5}  {'Saved':>5}  {'Examples':>8}  {'Train':>5}  {'Val':>4}  "
        f"{'Train→':>9}  {'Val→':>9}  {'Best val':>9}")
    log("=" * 75)
    for cyc in cycles:
        cid = cyc["cycle_id"]
        saved = "✓" if cyc["saved_checkpoint"] else " "
        ep_mask = e_cycle == cid
        if ep_mask.any():
            train_str = f"{e_train[ep_mask][-1]:.4f}"
            val_str = f"{e_val[ep_mask][-1]:.4f}"
        else:
            train_str = "-"
            val_str = "-"
        bv = cyc.get("best_val_loss_in_cycle")
        bv_str = f"{bv:.4f}" if bv is not None else "-"
        log(f"{cid:>5}  {saved:>5}  {cyc['examples_count']:>8}  "
            f"{cyc['train_size']:>5}  {cyc['val_size']:>4}  "
            f"{train_str:>9}  {val_str:>9}  {bv_str:>9}")
    log("=" * 75)
    log(f"Saved checkpoints: {c_saved.sum()}/{len(cycles)} cycles")
    if len(c_best_val) and not np.all(np.isnan(c_best_val)):
        best_idx = int(np.nanargmin(c_best_val))
        log(f"Best val overall: {np.nanmin(c_best_val):.4f} (cycle {c_id[best_idx]})")
        log(f"First cycle best val: {c_best_val[0]:.4f}")
        log(f"Last cycle best val:  {c_best_val[-1]:.4f}")
        improvement = c_best_val[0] - c_best_val[-1]
        log(f"Improvement: {improvement:.4f} "
            f"({improvement / c_best_val[0] * 100:.1f}%)")
    if len(s_total):
        log("")
        log(f"Train loss: first={s_total[0]:.4f}, last={s_total[-1]:.4f}, "
            f"mean={s_total.mean():.4f}")
        log(f"  Component means: value={s_value.mean():.4f}, "
            f"action={s_action.mean():.4f}, chain={s_chain.mean():.4f}")
    if len(v_total):
        log(f"Val loss: first={v_total[0]:.4f}, last={v_total[-1]:.4f}, "
            f"mean={v_total.mean():.4f}")
        log(f"  Component means: value={v_value.mean():.4f}, "
            f"action={v_action.mean():.4f}, chain={v_chain.mean():.4f}")


# ---------- Main entrypoint ----------

def analyze(agent_dir, out_dir=None, smooth_window=11):
    """Run full analytics: 3 plots + console summary."""
    agent_dir = os.path.abspath(agent_dir)
    if out_dir is None:
        out_dir = os.path.join(agent_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    hist = load_history(agent_dir)
    # Agent name = the directory two levels up (mcts_predict / agent_name / ...)
    agent_name = os.path.basename(os.path.dirname(agent_dir)) or "agent"

    paths = {
        "overview":      os.path.join(out_dir, "mcts_training_overview.png"),
        "balance":       os.path.join(out_dir, "mcts_component_balance.png"),
        "progression":   os.path.join(out_dir, "mcts_cycle_progression.png"),
    }
    plot_overview(hist, paths["overview"], agent_name, smooth_window)
    plot_component_balance(hist, paths["balance"], smooth_window)
    plot_cycle_progression(hist, paths["progression"])

    for name, p in paths.items():
        print(f"Saved: {p}")
    print()
    print_summary(hist)
    return paths


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent_dir", required=True,
                   help="Path to mcts_predict scenario folder containing history.pt")
    p.add_argument("--out_dir", default=None,
                   help="Where to write PNGs (default: <agent_dir>/analysis)")
    p.add_argument("--smooth_window", type=int, default=11,
                   help="Moving-average window for train-loss smoothing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(args.agent_dir, out_dir=args.out_dir,
            smooth_window=args.smooth_window)
