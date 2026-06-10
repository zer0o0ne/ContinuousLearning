"""Shared checkpoint serialization helpers.

All training phases (gto_ev, gto_probs, gto, modelling, opponent_action,
mcts) write Adam + SequentialLR state through these helpers so that cross-
phase contamination is impossible. The bug that motivated this module:
`pipeline._build_persistent_optim` blindly restored an opponent_action
scheduler state_dict into the freshly built MCTS scheduler, overwriting its
`T_max` / `_milestones` and driving lr negative through CosineAnnealingLR's
recursive get_lr.

Two tags travel with every checkpoint:

* `phase` — directory name of the saving scenario (e.g. "mcts_predict").
  Restore is gated by phase equality.
* `param_signature` / `trainable_signature` — md5 over (name, shape) of all
  / requires_grad params. Catches architecture drift that would otherwise
  desync Adam's positional state.

Legacy checkpoints written before this module existed have none of these
keys; they are detected and treated as "do not restore optim/sched, only
the model weights".
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import torch
import torch.nn as nn


VALID_PHASES = (
    "gto_ev_predict",
    "gto_probs_predict",
    "gto_predict",
    "modelling_predict",
    "opponent_action_predict",
    "mcts_predict",
)


def _signature(named_params) -> str:
    parts = sorted(f"{name}:{tuple(p.shape)}" for name, p in named_params)
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def param_signature(model: nn.Module) -> str:
    return _signature(model.named_parameters())


def trainable_signature(model: nn.Module) -> str:
    return _signature(
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    )


def make_checkpoint(
    *,
    phase: str,
    model: nn.Module,
    optimizer,
    scheduler,
    norm_stats,
    val_loss,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in VALID_PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {VALID_PHASES}")
    ckpt: dict[str, Any] = {
        "phase": phase,
        "param_signature": param_signature(model),
        "trainable_signature": trainable_signature(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_stats": norm_stats,
        "val_loss": val_loss,
    }
    if extra:
        overlap = set(extra) & set(ckpt)
        if overlap:
            raise ValueError(
                f"extra checkpoint keys collide with reserved names: {sorted(overlap)}"
            )
        ckpt.update(extra)
    return ckpt


def _log(log: Callable[[str], None] | None, msg: str) -> None:
    if log is not None:
        log(msg)


def restore_optim_sched(
    *,
    optimizer,
    scheduler,
    ckpt: dict[str, Any],
    expected_phase: str,
    model: nn.Module,
    strict: bool,
    log: Callable[[str], None] | None = None,
    legacy_path_phase_hint: str | None = None,
) -> tuple[bool, bool, str]:
    """Restore optimizer and scheduler state with phase + signature gating.

    Returns ``(restored_opt, restored_sched, reason)``. ``reason`` is a
    human-readable note that callers can fold into their own log line.

    Parameters
    ----------
    expected_phase
        The phase the caller is currently executing. State is only loaded
        when ``ckpt["phase"] == expected_phase``.
    strict
        ``True``  — the caller knows the checkpoint *should* match (e.g.
        per-phase resume from ``<run_dir>/latest.pt``). Any mismatch raises
        ``RuntimeError`` rather than silently starting over, so genuine
        architecture / phase mistakes are noisy.
        ``False`` — best-effort restore (e.g. first MCTS cycle that may
        find an opponent_action checkpoint). Mismatch is a normal skip.
    legacy_path_phase_hint
        Only consulted when the checkpoint has no ``phase`` tag at all
        (i.e. was written before this module existed). Setting this to a
        valid phase name asserts to ``restore_optim_sched`` that the
        checkpoint's location on disk already proves it belongs to that
        phase — used by phase 1–5 resume paths where ``latest.pt`` lives
        inside the phase's own ``run_dir``. If the hint matches
        ``expected_phase``, the legacy checkpoint is accepted; otherwise
        treated as untagged.
    """
    ckpt_phase = ckpt.get("phase")

    # --- Phase gate ---------------------------------------------------
    if ckpt_phase is None:
        if (
            legacy_path_phase_hint is not None
            and legacy_path_phase_hint == expected_phase
        ):
            reason = "legacy untagged ckpt, phase inferred from path"
            _log(log, f"  [restore] {reason}; attempting optim/sched restore")
            # Fall through to the actual loads.
        else:
            reason = "legacy untagged ckpt (no phase tag) — skip optim/sched"
            if strict:
                raise RuntimeError(
                    "strict restore requested but ckpt has no phase tag and no "
                    "path-based phase hint; refusing to silently continue"
                )
            _log(log, f"  [restore] {reason}")
            return False, False, reason
    elif ckpt_phase != expected_phase:
        reason = (
            f"phase mismatch: ckpt={ckpt_phase!r}, expected={expected_phase!r}"
        )
        if strict:
            raise RuntimeError(
                f"strict restore expected phase {expected_phase!r}, got {ckpt_phase!r}"
            )
        _log(log, f"  [restore] {reason} — fresh optim/sched")
        return False, False, reason

    # --- Optimizer (gated additionally by trainable_signature) --------
    restored_opt = False
    cur_train_sig = trainable_signature(model)
    ckpt_train_sig = ckpt.get("trainable_signature")
    opt_state = ckpt.get("optimizer_state_dict")

    if opt_state is None:
        opt_skip_reason = "no optimizer_state_dict in ckpt"
    elif ckpt_train_sig is None:
        # legacy w/ path hint — checked above, ckpt_phase was None and we
        # already accepted. Trust path-hint and load.
        if ckpt_phase is None:
            optimizer.load_state_dict(opt_state)
            restored_opt = True
            opt_skip_reason = "legacy ckpt accepted via path hint"
        else:
            opt_skip_reason = "ckpt missing trainable_signature"
            if strict:
                raise RuntimeError(
                    "strict restore but ckpt has phase tag without "
                    "trainable_signature — refusing to load partial metadata"
                )
    elif ckpt_train_sig != cur_train_sig:
        opt_skip_reason = (
            "trainable_signature mismatch (architecture or freeze pattern "
            "changed)"
        )
        if strict:
            raise RuntimeError(
                "strict restore: trainable_signature mismatch — model "
                "architecture or requires_grad mask differs from checkpoint"
            )
    else:
        optimizer.load_state_dict(opt_state)
        restored_opt = True
        opt_skip_reason = ""

    # --- Scheduler ----------------------------------------------------
    restored_sched = False
    sched_state = ckpt.get("scheduler_state_dict")
    if sched_state is None:
        sched_skip_reason = "no scheduler_state_dict in ckpt"
    else:
        scheduler.load_state_dict(sched_state)
        restored_sched = True
        sched_skip_reason = ""

    bits = []
    if restored_opt:
        bits.append("opt=ok")
    else:
        bits.append(f"opt=skip({opt_skip_reason})")
    if restored_sched:
        bits.append("sched=ok")
    else:
        bits.append(f"sched=skip({sched_skip_reason})")
    reason = ", ".join(bits)
    _log(log, f"  [restore] {reason}")
    return restored_opt, restored_sched, reason


def is_legacy_mcts_ckpt(ckpt_path: str | None, ckpt: dict[str, Any] | None) -> bool:
    """True iff the checkpoint lives under .../mcts_predict/... but has no
    phase tag — i.e. was written by the pre-fix code path that produced
    corrupted weights via negative LR. Callers should warn loudly: the
    user almost certainly wants to delete that subdirectory and restart
    MCTS from the prior phase's checkpoint."""
    if ckpt_path is None or ckpt is None:
        return False
    if ckpt.get("phase") is not None:
        return False
    sep = os.sep
    return f"{sep}mcts_predict{sep}" in ckpt_path
