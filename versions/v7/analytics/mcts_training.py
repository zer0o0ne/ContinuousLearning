"""MCTS training analytics for a single agent.

Reads `<agent_mcts_dir>/history.pt` (the cross-cycle accumulated history file
produced by the cyclic MCTS training pipeline) and produces diagnostic plots
plus a per-cycle summary table.

Usage:
    python -m analytics.mcts_training --agent_dir <path>
    # or directly:
    python analytics/mcts_training.py \
        --agent_dir /path/to/data/v7/<save_dir>/<agent>/mcts_predict

`agent_dir` must point to the `mcts_predict` scenario folder (the one that
contains `history.pt` and per-run `<timestamp>/best.pt`). Output PNGs are
written to `<agent_dir>/analysis/` by default.

Plots produced:
    1. mcts_training_overview.png   — 3x3 panel: train/val components (primary
                                       + auxiliary), best val per cycle, LR
                                       schedule, dataset sizes, train-vs-val
                                       gap, teacher-forcing probability.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _VERSION_DIR not in sys.path:
    sys.path.insert(0, _VERSION_DIR)

from agent.train_scenarios._history import IncrementalHistory  # noqa: E402


# ---------- Loss component definitions ----------

_PRIMARY_COMPONENTS = ["value", "action", "chain"]
_AUX_COMPONENTS = ["chain_value", "recon", "terminal_value", "action_entropy"]
_RECON_SUB = ["recon_mse", "recon_infonce"]
_ALL_COMPONENTS = _PRIMARY_COMPONENTS + _AUX_COMPONENTS

_COMPONENT_COLORS = {
    "value": "C0",
    "action": "C1",
    "chain": "C2",
    "chain_value": "C3",
    "recon": "C4",
    "terminal_value": "C5",
    "action_entropy": "C6",
    "recon_mse": "C4",
    "recon_infonce": "C7",
}

# ---------- Loading helpers ----------

def _column(records, key, default=0.0):
    return np.asarray([r.get(key, default) for r in records], dtype=float)


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


def _has_component(records, key):
    """Check if any record has a non-zero value for key."""
    for r in records:
        v = r.get(key)
        if v is not None and v != 0:
            return True
    return False


# ---------- Plot 1: 3x3 overview ----------

def plot_overview(hist, out_path, agent_name, smooth_window=11):
    step_loss = hist["step_loss"]
    val_loss = hist["val_loss"]
    epoch_train = hist["epoch_train_loss"]
    epoch_val = hist["epoch_val_loss"]
    cycles = hist["cycles"]

    s_step = _column(step_loss, "step")
    s_total = _column(step_loss, "total")
    s_lr = _column(step_loss, "lr")
    s_p_tf = _column(step_loss, "p_tf", default=np.nan)

    v_step = _column(val_loss, "step")
    v_total = _column(val_loss, "total")

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

    has_p_tf = not np.all(np.isnan(s_p_tf))
    active_aux = [c for c in _AUX_COMPONENTS if _has_component(step_loss, c)]
    has_aux = len(active_aux) > 0
    has_recon_sub = _has_component(step_loss, "recon_mse")

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14))
    fig.suptitle(
        f"MCTS training analytics — {agent_name} "
        f"({len(cycles)} cycles, {len(step_loss)} training steps)",
        fontsize=14,
    )
    W = smooth_window

    # (0,0) Train primary loss components
    ax = axes[0, 0]
    ax.plot(s_step, _smooth(s_total, W), label="total", color="black", lw=1.5)
    for comp in _PRIMARY_COMPONENTS:
        ax.plot(s_step, _smooth(_column(step_loss, comp), W),
                label=comp, color=_COMPONENT_COLORS[comp], alpha=0.8)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"Train: primary components (smooth={W})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) Train auxiliary loss components
    ax = axes[0, 1]
    if has_aux:
        for comp in active_aux:
            ax.plot(s_step, _smooth(_column(step_loss, comp), W),
                    label=comp, color=_COMPONENT_COLORS[comp], alpha=0.8)
        if has_recon_sub:
            for sub in _RECON_SUB:
                ax.plot(s_step, _smooth(_column(step_loss, sub), W),
                        label=sub, color=_COMPONENT_COLORS[sub],
                        alpha=0.5, ls="--", lw=0.8)
        for b in cycle_boundaries:
            ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
        ax.set_title(f"Train: auxiliary components (smooth={W})")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_title("Train: auxiliary components (none active)")
        ax.text(0.5, 0.5, "No auxiliary losses recorded",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    # (0,2) Validation loss components
    ax = axes[0, 2]
    if len(v_step) > 0:
        ax.plot(v_step, v_total, "o-", label="total", color="black", lw=1.5, ms=4)
        for comp in _PRIMARY_COMPONENTS:
            ax.plot(v_step, _column(val_loss, comp), "o-",
                    label=comp, color=_COMPONENT_COLORS[comp], alpha=0.8, ms=3)
        active_val_aux = [c for c in _AUX_COMPONENTS if _has_component(val_loss, c)]
        for comp in active_val_aux:
            ax.plot(v_step, _column(val_loss, comp), "s-",
                    label=comp, color=_COMPONENT_COLORS[comp], alpha=0.6, ms=2)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss components")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    # (1,0) Best val per cycle
    ax = axes[1, 0]
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

    # (1,1) LR schedule
    ax = axes[1, 1]
    ax.plot(s_step, s_lr, color="purple", lw=0.7)
    for b in cycle_boundaries:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (warmup + cosine per cycle)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # (1,2) Teacher-forcing probability
    ax = axes[1, 2]
    if has_p_tf:
        ax.plot(s_step, s_p_tf, color="teal", lw=0.7)
        for b in cycle_boundaries:
            ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
        ax.set_title("Teacher-forcing probability")
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_title("Teacher-forcing probability (not recorded)")
        ax.text(0.5, 0.5, "p_tf not in history",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("training step")
    ax.set_ylabel("p_tf")
    ax.grid(alpha=0.3)

    # (2,0) Examples per cycle
    ax = axes[2, 0]
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

    # (2,1) Train vs Val per epoch
    ax = axes[2, 1]
    if len(e_train) > 0 and len(e_train) == len(e_val):
        epoch_idx = np.arange(len(e_train))
        ax.plot(epoch_idx, e_train, "o-", label="train", color="C0", lw=1.2, ms=4)
        ax.plot(epoch_idx, e_val, "o-", label="val", color="C3", lw=1.2, ms=4)
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

    # (2,2) Val component breakdown per cycle
    ax = axes[2, 2]
    if len(cycles) > 0 and len(val_loss) > 0:
        cycle_ids_val = _column(val_loss, "cycle_id")
        for comp in _PRIMARY_COMPONENTS + active_aux:
            comp_vals = _column(val_loss, comp)
            per_cycle_means = []
            for cid in c_id:
                mask = cycle_ids_val == cid
                if mask.any():
                    per_cycle_means.append(comp_vals[mask].mean())
                else:
                    per_cycle_means.append(np.nan)
            ax.plot(c_id, per_cycle_means, "o-", label=comp,
                    color=_COMPONENT_COLORS[comp], ms=3, alpha=0.8)
        ax.set_xlabel("cycle id")
        ax.set_ylabel("mean val loss")
        ax.set_title("Val components per cycle (mean)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 2: component balance ----------

def plot_component_balance(hist, out_path, smooth_window=11):
    step_loss = hist["step_loss"]
    val_loss = hist["val_loss"]

    s_step = _column(step_loss, "step")
    v_step = _column(val_loss, "step")
    v_total = _column(val_loss, "total")

    active_components = [c for c in _ALL_COMPONENTS if _has_component(step_loss, c)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Loss component balance over training", fontsize=14)
    W = smooth_window

    # Train: stacked area
    ax = axes[0]
    smoothed = {}
    for comp in active_components:
        smoothed[comp] = np.abs(_smooth(_column(step_loss, comp), W))

    bottom = np.zeros_like(s_step)
    for comp in active_components:
        ax.fill_between(s_step, bottom, bottom + smoothed[comp],
                        color=_COMPONENT_COLORS[comp], alpha=0.6, label=comp)
        bottom = bottom + smoothed[comp]
    ax.set_xlabel("training step")
    ax.set_ylabel("contribution to total loss (smoothed)")
    ax.set_title(f"Train: components stacked (window={W})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Val: relative shares
    ax = axes[1]
    if len(v_step) > 0:
        active_val = [c for c in _ALL_COMPONENTS if _has_component(val_loss, c)]
        for comp in active_val:
            share = np.abs(_column(val_loss, comp)) / np.maximum(np.abs(v_total), 1e-9)
            ax.plot(v_step, share, "o-", color=_COMPONENT_COLORS[comp],
                    label=comp, ms=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of total val loss")
    ax.set_title("Val: relative weight of each component")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 3: cycle progression ----------

def plot_cycle_progression(hist, out_path):
    step_loss = hist["step_loss"]
    cycles = hist["cycles"]

    s_step = _column(step_loss, "step")
    s_total = _column(step_loss, "total")
    s_cycle = np.asarray([r.get("cycle_id", 0) for r in step_loss], dtype=int)

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

    c_id = np.asarray([c["cycle_id"] for c in cycles], dtype=int)
    c_best_val = np.asarray(
        [c["best_val_loss_in_cycle"] if c["best_val_loss_in_cycle"] is not None
         else np.nan for c in cycles], dtype=float)
    c_saved = np.asarray([c["saved_checkpoint"] for c in cycles], dtype=bool)

    log("=" * 90)
    log(f"{'Cycle':>5}  {'Saved':>5}  {'Examples':>8}  {'Train':>5}  {'Val':>4}  "
        f"{'Train->':>9}  {'Val->':>9}  {'Best val':>9}")
    log("=" * 90)
    for cyc in cycles:
        cid = cyc["cycle_id"]
        saved = "Y" if cyc["saved_checkpoint"] else " "
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
    log("=" * 90)
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
        active = [c for c in _ALL_COMPONENTS if _has_component(step_loss, c)]
        means_str = ", ".join(
            f"{c}={_column(step_loss, c).mean():.4f}" for c in active)
        log(f"  Component means: {means_str}")

        active_sub = [c for c in _RECON_SUB if _has_component(step_loss, c)]
        if active_sub:
            sub_str = ", ".join(
                f"{c}={_column(step_loss, c).mean():.4f}" for c in active_sub)
            log(f"  Recon sub-components: {sub_str}")

    v_total_arr = _column(val_loss, "total") if val_loss else np.array([])
    if len(v_total_arr):
        log(f"Val loss: first={v_total_arr[0]:.4f}, last={v_total_arr[-1]:.4f}, "
            f"mean={v_total_arr.mean():.4f}")
        active_val = [c for c in _ALL_COMPONENTS if _has_component(val_loss, c)]
        val_means_str = ", ".join(
            f"{c}={_column(val_loss, c).mean():.4f}" for c in active_val)
        log(f"  Component means: {val_means_str}")


# ---------- Main entrypoint ----------

def analyze(agent_dir, out_dir=None, smooth_window=11):
    """Run full analytics: 3 plots + console summary."""
    agent_dir = os.path.abspath(agent_dir)
    if out_dir is None:
        out_dir = os.path.join(agent_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    hist = load_history(agent_dir)
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
