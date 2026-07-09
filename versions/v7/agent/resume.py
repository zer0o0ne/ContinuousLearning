"""Resumable pipeline state + atomic I/O helpers.

Used by `pipeline.py`, both dataset generators, and all phase-1..5 training
scripts to keep `<save_base_dir>/pipeline_state.json` consistent across
interrupted runs.

When `pipeline.resume = false` in the config, `PipelineState` is constructed
in-memory and never touches disk — behaviour matches the legacy pipeline
exactly. When `resume = true`, every status change is flushed atomically so
a SIGKILL can never leave the file half-written.
"""

import hashlib
import json
import os
import tempfile

import torch


def atomic_torch_save(obj, path):
    """torch.save with crash safety: write to .tmp, fsync, then os.replace.

    os.replace is atomic on POSIX so the destination either has the previous
    contents or the new contents — never a half-written file.
    """
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_json_dump(obj, path):
    """json.dump with the same crash-safety guarantees as atomic_torch_save."""
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def compute_config_hash(config):
    """SHA256 over `game.*` + `solver.*` + `multi_agent.agents`.

    Datasets become stale on game/solver changes. Training phases become
    stale on modifier changes (multi_agent.agents). Including all three
    in a single hash means a modifier change also triggers dataset
    regeneration (safe, slightly wasteful) — D.5.3.
    """
    payload = {
        "game":   config.get("game", {}),
        "solver": config.get("solver", {}),
        "multi_agent_agents": config.get("multi_agent", {}).get("agents", []),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# Scenario directories in the per-agent folder. Used for bootstrap discovery
# (mapping existing best.pt → status="done").
_PHASE_DIRS = (
    "gto_ev_predict",
    "gto_probs_predict",
    "gto_predict",
    "modelling_predict",
    "opponent_action_predict",
)


def _find_latest_run_dir(scenario_dir):
    """Return the most recent timestamp subdir of `scenario_dir` containing
    a best.pt, or None."""
    if not os.path.isdir(scenario_dir):
        return None
    candidates = []
    for name in os.listdir(scenario_dir):
        sub = os.path.join(scenario_dir, name)
        if not os.path.isdir(sub):
            continue
        if os.path.isfile(os.path.join(sub, "best.pt")):
            candidates.append((name, sub))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


class PipelineState:
    """Single source of truth for pipeline progress across interrupted runs.

    Schema (only fields actually consulted at runtime are documented; extras
    are preserved on load and round-tripped on save):

      version:        int (currently 1)
      config_hash:    str — hash of game.* + solver.*
      datasets:       {name: {path, target, completed_hands, done,
                              config_hash}}
      agents:         {agent_name: {phase_name: {status, run_dir, next_epoch,
                                                  global_step, best_val_loss,
                                                  fails_since_best}}}
      mcts:           {next_cycle, stage, run_dirs, examples_paths,
                       cumulative_steps}

    `status` is one of "pending" | "in_progress" | "done".
    """

    def __init__(self, path, resume, config_hash, log,
                 force_phases=None, force_agents=None):
        self.path = path
        self.resume = bool(resume)
        self.current_config_hash = config_hash
        self.log = log
        self.force_phases = set(force_phases or [])
        self.force_agents = set(force_agents or [])
        self.data = {
            "version": 1,
            "config_hash": config_hash,
            "datasets": {},
            "agents": {},
            "mcts": {},
        }
        # When the on-disk config_hash doesn't match the current one we keep
        # training state (model weights are independent of these params) but
        # invalidate the datasets — see `is_dataset_compatible`.
        self.config_hash_mismatch = False

    @classmethod
    def load_or_create(cls, path, resume, config_hash, log,
                       force_phases=None, force_agents=None):
        st = cls(path, resume, config_hash, log,
                 force_phases=force_phases, force_agents=force_agents)
        if not resume:
            return st
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if loaded.get("version") != 1:
                    log(f"  [resume] unknown pipeline_state version "
                        f"{loaded.get('version')}, ignoring file")
                else:
                    st.data = loaded
                    st.data.setdefault("datasets", {})
                    st.data.setdefault("agents", {})
                    st.data.setdefault("mcts", {})
                    if loaded.get("config_hash") != config_hash:
                        log("  [resume] WARNING: config_hash mismatch. "
                            "Datasets will be regenerated; agent training "
                            "state preserved.")
                        st.config_hash_mismatch = True
                    log(f"  [resume] loaded pipeline_state from {path}")
            except Exception as e:
                log(f"  [resume] failed to parse {path}: {e}. "
                    f"Starting from scratch.")
        else:
            log(f"  [resume] no pipeline_state at {path}, creating fresh")
        return st

    def save(self):
        """Atomic flush. No-op when resume is disabled."""
        if not self.resume:
            return
        # Always overwrite config_hash with the live one so a successful save
        # after a mismatch warning records that we've moved past the warning.
        self.data["config_hash"] = self.current_config_hash
        atomic_json_dump(self.data, self.path)

    # ----- datasets ---------------------------------------------------

    def get_dataset(self, name):
        return self.data["datasets"].get(name)

    def set_dataset(self, name, **fields):
        ds = self.data["datasets"].setdefault(name, {})
        ds.update(fields)
        self.save()

    def is_dataset_compatible(self, name, current_hash):
        """True iff the on-disk dataset state was produced with the same
        game/solver settings as the current config. Used by pipeline to
        decide whether to regenerate from scratch."""
        ds = self.get_dataset(name)
        if ds is None:
            return False
        return ds.get("config_hash") == current_hash

    # ----- phases (per-agent training scenarios) ----------------------

    def get_phase(self, agent_name, phase_name):
        return self.data["agents"].get(agent_name, {}).get(phase_name)

    def set_phase(self, agent_name, phase_name, **fields):
        agent = self.data["agents"].setdefault(agent_name, {})
        phase = agent.setdefault(phase_name, {})
        phase.update(fields)
        self.save()

    def should_force_restart_phase(self, agent_name, phase_name):
        if not self.resume:
            return False
        if phase_name in self.force_phases:
            return True
        if agent_name in self.force_agents:
            return True
        return False

    # ----- MCTS -------------------------------------------------------

    def get_mcts(self):
        return self.data.get("mcts") or None

    def set_mcts(self, **fields):
        mcts = self.data.setdefault("mcts", {})
        mcts.update(fields)
        self.save()

    # ----- bootstrap from existing on-disk artefacts ------------------

    def bootstrap_from_disk(self, save_base_dir, multi_agent_names):
        """First-time resume: scan `<save_base_dir>/<agent>/<phase>/<ts>/best.pt`
        and mark each found pair as status=done.

        This lets the user flip `pipeline.resume = true` on an experiment
        that was previously run without resume support without losing any
        completed phases. Phases without a best.pt are left as pending.
        """
        if not self.resume:
            return
        if self.data["agents"]:
            return  # already populated — nothing to bootstrap
        if not os.path.isdir(save_base_dir):
            return
        bootstrapped = 0
        for agent_name in multi_agent_names:
            agent_dir = os.path.join(save_base_dir, agent_name)
            if not os.path.isdir(agent_dir):
                continue
            for phase_name in _PHASE_DIRS:
                run_dir = _find_latest_run_dir(
                    os.path.join(agent_dir, phase_name))
                if run_dir is None:
                    continue
                self.data["agents"].setdefault(agent_name, {})[phase_name] = {
                    "status":  "done",
                    "run_dir": run_dir,
                }
                bootstrapped += 1
        if bootstrapped:
            self.log(f"  [resume] bootstrap: found {bootstrapped} completed "
                     f"phase(s) on disk")
            self.save()


# Re-exported helpers (used by training scripts via direct import).
__all__ = [
    "atomic_torch_save",
    "atomic_json_dump",
    "compute_config_hash",
    "PipelineState",
]
