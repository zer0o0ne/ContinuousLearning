"""Append-only history persistence for training phases.

E.5.3: instead of rewriting the entire history dict on every val/save,
append only new entries as numbered shard files. On load, replay shards
to rebuild the full dict.  Backward-compatible with legacy history.pt.

Usage (drop-in replacement for the old _save_history pattern)::

    hist = IncrementalHistory(run_dir, keys=["step_loss", "val_loss", ...])
    # hist.data is the dict — use hist.data["step_loss"].append(...)
    # When ready to persist:
    hist.save()
    # At training end, compact into one file:
    hist.compact()
"""

from __future__ import annotations

import glob
import os
import tempfile

import torch


# Compact shards into a single file every N saves to bound shard count.
_COMPACT_EVERY = 50


class IncrementalHistory:
    """Append-only history with shard-based persistence.

    Parameters
    ----------
    run_dir : str
        Directory where history shards are stored.
    keys : list[str]
        Expected keys in the history dict (each maps to a list).
    """

    def __init__(self, run_dir: str, keys: list[str]):
        self.run_dir = run_dir
        self._shard_dir = os.path.join(run_dir, "history_shards")
        self.data: dict[str, list] = {}
        self._persisted_counts: dict[str, int] = {}
        self._save_count = 0

        # Load existing data (backward compat + shard replay)
        self._load(keys)

    def _load(self, keys: list[str]) -> None:
        """Load history from legacy history.pt and/or shard files."""
        legacy_path = os.path.join(self.run_dir, "history.pt")
        base = {}

        # 1. Try legacy history.pt first
        if os.path.exists(legacy_path):
            try:
                base = torch.load(legacy_path, weights_only=False)
            except Exception:
                base = {}

        # Ensure all expected keys exist
        for k in keys:
            base.setdefault(k, [])

        self.data = base
        # After loading base, record current lengths as persisted
        self._persisted_counts = {k: len(v) for k, v in self.data.items()}

        # 2. Replay shard files on top (if any)
        if os.path.isdir(self._shard_dir):
            shard_files = sorted(glob.glob(
                os.path.join(self._shard_dir, "shard_*.pt")
            ))
            for sf in shard_files:
                try:
                    delta = torch.load(sf, weights_only=False)
                except Exception:
                    continue
                for k, new_entries in delta.items():
                    if k not in self.data:
                        self.data[k] = []
                    self.data[k].extend(new_entries)

            # After replaying shards, all entries are "persisted"
            self._persisted_counts = {k: len(v) for k, v in self.data.items()}

    def save(self) -> None:
        """Persist only new entries since last save as a numbered shard."""
        delta = {}
        for k, lst in self.data.items():
            prev = self._persisted_counts.get(k, 0)
            if len(lst) > prev:
                delta[k] = lst[prev:]

        if not delta:
            return  # nothing new

        os.makedirs(self._shard_dir, exist_ok=True)

        # Determine next shard number
        existing = glob.glob(os.path.join(self._shard_dir, "shard_*.pt"))
        next_idx = len(existing)
        shard_path = os.path.join(self._shard_dir, f"shard_{next_idx:06d}.pt")

        # Atomic write
        fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=self._shard_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(delta, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, shard_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        # Update persisted counts
        for k, lst in self.data.items():
            self._persisted_counts[k] = len(lst)

        self._save_count += 1
        if self._save_count % _COMPACT_EVERY == 0:
            self.compact()

    def compact(self) -> None:
        """Merge all shards + base into a single history.pt, remove shards.

        Called periodically (every _COMPACT_EVERY saves) and explicitly at
        training end.
        """
        # Write the full dict as history.pt (atomic)
        history_path = os.path.join(self.run_dir, "history.pt")
        fd, tmp = tempfile.mkstemp(prefix=".tmp.",
                                   dir=os.path.dirname(history_path) or ".")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(self.data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, history_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        # Remove shard files now that everything is in history.pt
        if os.path.isdir(self._shard_dir):
            for sf in glob.glob(os.path.join(self._shard_dir, "shard_*.pt")):
                try:
                    os.unlink(sf)
                except OSError:
                    pass
            # Try to remove the directory (only succeeds if empty)
            try:
                os.rmdir(self._shard_dir)
            except OSError:
                pass

        # Reset: everything is now in history.pt, zero shards
        self._persisted_counts = {k: len(v) for k, v in self.data.items()}
        self._save_count = 0
