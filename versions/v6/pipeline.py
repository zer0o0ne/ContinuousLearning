import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import json
import copy
from math import ceil

from utils import Logger
from agent.agent import ASI
from agent.resume import (
    PipelineState, compute_config_hash, atomic_torch_save,
)
from agent.train_scenarios._checkpoint_io import (
    is_legacy_mcts_ckpt,
    restore_optim_sched,
)
from agent.train_scenarios._history import IncrementalHistory


def _release_memory(device="cpu"):
    """Force Python to return freed memory to the OS.

    CPython's pymalloc holds freed arenas indefinitely; on unified-memory
    systems (DGX Spark) this inflates RSS by tens of GB after large
    dataset operations.  gc.collect() breaks reference cycles, then
    malloc_trim asks glibc to release free pages back to the kernel.
    """
    gc.collect()
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _find_latest_best_ckpt(scenario_dir):
    """Return path to best.pt in the latest timestamp subdir of scenario_dir, or None."""
    if not os.path.isdir(scenario_dir):
        return None
    candidates = []
    for name in os.listdir(scenario_dir):
        best = os.path.join(scenario_dir, name, "best.pt")
        if os.path.isfile(best):
            candidates.append((name, best))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _discover_past_snapshots(trained_agents, max_per_agent, log):
    """Scan ``<run_dir>/cycles/cycle_NNNN.pt`` for each active agent.

    Returns a list of dicts suitable for `run_mcts_collection(...
    past_snapshot_specs=...)`:

        {"name": "agent_a@cycle_0015",
         "agent_name": "agent_a",
         "cycle_id": 15,
         "ckpt_path": "<run_dir>/cycles/cycle_0015.pt"}

    For each active agent, if more than ``max_per_agent`` snapshots are
    available, picks ``max_per_agent`` uniformly at random WITHOUT
    replacement (so older / newer snapshots both get represented over
    cycles). ``max_per_agent <= 0`` or ``None`` → use all available.
    """
    import random as _random
    snapshots = []
    for a in trained_agents:
        run_dir = a.get("run_dir")
        if not run_dir:
            continue
        cycles_dir = os.path.join(run_dir, "cycles")
        if not os.path.isdir(cycles_dir):
            continue
        agent_name = a["name"]
        candidates = []
        for fname in os.listdir(cycles_dir):
            if not (fname.startswith("cycle_") and fname.endswith(".pt")):
                continue
            try:
                cid = int(fname[len("cycle_"):-3])
            except ValueError:
                continue
            candidates.append((cid, os.path.join(cycles_dir, fname)))
        if not candidates:
            continue
        if max_per_agent and len(candidates) > int(max_per_agent):
            candidates = _random.sample(candidates, int(max_per_agent))
        for cid, path in candidates:
            snapshots.append({
                "name": f"{agent_name}@cycle_{cid:04d}",
                "agent_name": agent_name,
                "cycle_id": cid,
                "ckpt_path": path,
            })
    if snapshots:
        log(f"  past_opponents pool: {len(snapshots)} snapshot(s) across "
            f"{len(trained_agents)} active agent(s)")
        # Compact per-agent breakdown for quick eyeballing.
        per_agent = {}
        for s in snapshots:
            per_agent.setdefault(s["agent_name"], []).append(s["cycle_id"])
        for name, cids in per_agent.items():
            log(f"    {name}: cycles {sorted(cids)}")
    else:
        log("  past_opponents pool: empty (no cycle snapshots on disk yet)")
    return snapshots


def _run_or_skip_phase(scenario_name, agent, agent_base, agent_log, train_fn):
    """Legacy non-resume helper: if scenario already has a best.pt, load it
    and skip. Otherwise run train_fn() and load its best.pt. Returns the
    path to the loaded best.pt (or None).

    Used only when `pipeline.resume` is False. The resume-aware variant
    `_run_or_resume_phase` lives next to it.
    """
    existing = _find_latest_best_ckpt(os.path.join(agent_base, scenario_name))
    if existing is not None:
        agent_log(f"  [resume] {scenario_name}: loading existing {existing}, "
                  f"skipping training")
        agent.load_checkpoint(existing)
        return existing
    _, run_dir = train_fn()
    if run_dir:
        best_ckpt = os.path.join(run_dir, "best.pt")
        if os.path.exists(best_ckpt):
            agent.load_checkpoint(best_ckpt)
            return best_ckpt
    return None


def _run_or_resume_phase(state, agent, agent_name, scenario_name,
                         agent_base, agent_log, device, train_fn):
    """Resume-aware phase runner.

    `train_fn(run_dir, resume_state) -> (history, run_dir)` is invoked with
    the run directory and (when applicable) a `resume_state` dict that the
    train scripts use to restore optimizer/scheduler/global_step/etc. The
    pipeline state is updated to `in_progress` before the call and `done`
    after a successful return.

    Status transitions:
      pending     → run from scratch, fresh run_dir
      in_progress → load latest.pt, continue from saved epoch
      done        → skip; load best.pt and return immediately
      force flag  → treat as `pending` (ignore on-disk progress)
    """
    import torch

    force = state.should_force_restart_phase(agent_name, scenario_name)
    phase = state.get_phase(agent_name, scenario_name)

    if not force and phase and phase.get("status") == "done":
        # Already complete in a prior run — load the recorded best.pt and skip.
        # We pass the FILE path (not the agent dir) so ASI.load_checkpoint
        # doesn't fall into _find_best_checkpoint's scenario-priority search,
        # which would either miss this phase entirely or pick up a later
        # phase's checkpoint instead.
        saved_run_dir = phase.get("run_dir")
        best_ckpt = None
        if saved_run_dir:
            candidate = os.path.join(saved_run_dir, "best.pt")
            if os.path.exists(candidate):
                best_ckpt = candidate
        if best_ckpt is None:
            # State recorded done but file is gone (manual move, etc.).
            # Fall back to scanning the scenario dir for any best.pt.
            best_ckpt = _find_latest_best_ckpt(
                os.path.join(agent_base, scenario_name))
        if best_ckpt is None:
            agent_log(
                f"  [resume] {scenario_name}: status=done but no best.pt on "
                f"disk (run_dir={saved_run_dir}). Re-running from scratch.")
            state.set_phase(agent_name, scenario_name, status="pending")
            # fall through to fresh-run path below
        else:
            agent_log(f"  [resume] {scenario_name}: status=done, "
                      f"loading {best_ckpt}")
            agent.load_checkpoint(best_ckpt)
            return

    resume_state = None
    run_dir = None
    if not force and phase and phase.get("status") == "in_progress":
        run_dir = phase.get("run_dir")
        if run_dir and os.path.isdir(run_dir):
            latest_path = os.path.join(run_dir, "latest.pt")
            if os.path.exists(latest_path):
                ckpt = torch.load(
                    latest_path, weights_only=False, map_location=device)
                agent.load_state_dict(ckpt["model_state_dict"], strict=False)
                del ckpt["model_state_dict"]
                ns = ckpt.get("norm_stats")
                if ns is not None:
                    agent._checkpoint_norm_stats = ns
                resume_state = {
                    "optimizer_state_dict": ckpt.pop("optimizer_state_dict"),
                    "scheduler_state_dict": ckpt.pop("scheduler_state_dict"),
                    "start_epoch":     int(ckpt.get("next_epoch", 0)),
                    "global_step":     int(ckpt.get("global_step", 0)),
                    "best_val_loss":   float(ckpt.get("best_val_loss",
                                                       float("inf"))),
                    "fails_since_best": int(ckpt.get("fails_since_best", 0)),
                    "norm_stats":      ns,
                }
                del ckpt
                agent_log(
                    f"  [resume] {scenario_name}: starting from "
                    f"epoch={resume_state['start_epoch']}, "
                    f"step={resume_state['global_step']}")
            else:
                agent_log(
                    f"  [resume] {scenario_name}: status=in_progress but "
                    f"no latest.pt at {run_dir} — restarting fresh")
                run_dir = None

    if run_dir is None:
        run_dir = agent_log.run_dir(scenario_name)
    state.set_phase(agent_name, scenario_name,
                    status="in_progress", run_dir=run_dir)

    train_fn(run_dir=run_dir, resume_state=resume_state)

    best_ckpt = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_ckpt):
        agent.load_checkpoint(best_ckpt)
    state.set_phase(agent_name, scenario_name,
                    status="done", run_dir=run_dir)


def _run_phase(state, scenario_name, agent, agent_name, agent_base,
               agent_log, device, train_fn):
    """Unified phase entry point.

    `train_fn(run_dir, resume_state) -> (history, run_dir)` must accept the
    two kwargs — resume-aware train scripts already do.

    Dispatches:
      state.resume=True  -> resume-aware path (per-epoch latest.pt,
                            optimizer/scheduler restore, pipeline_state.json).
      state.resume=False -> legacy behaviour (skip if any best.pt exists in
                            the scenario dir; otherwise fresh run).
    """
    if state.resume:
        _run_or_resume_phase(state, agent, agent_name, scenario_name,
                             agent_base, agent_log, device, train_fn)
    else:
        _run_or_skip_phase(
            scenario_name, agent, agent_base, agent_log,
            lambda: train_fn(run_dir=None, resume_state=None))


def _merge_train_config(config, scenario_key):
    """Merge game + solver + dataset + scenario-specific config into a flat dict."""
    merged = {}
    merged.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    merged["solver"] = solver_cfg.pop("type", "v2")
    merged.update(solver_cfg)
    merged.update(config.get("dataset", {}))
    merged.update(config.get(scenario_key, {}))
    return merged


def _load_or_generate_dataset(config, base_dir, device, log,
                              state=None, config_hash=None, resume=False):
    """Load or generate the GTO dataset.

    Returns the save directory path (string).  The dataset is stored as
    numbered shards on disk and never fully loaded into memory.

    When `resume=True`:
    - The default save directory is `<base_dir>/dataset/` (no timestamp), so
      successive runs find the same file. Pass `dataset.save_dir` or
      `dataset.dataset_dir` to override.
    - `generate_dataset` is invoked with `resume=True, config_hash=...`. It
      reads `meta.json`, returns immediately if `done=true` and target is
      sufficient, otherwise continues a partial run (sequential mode only).
    - The pipeline-state file mirrors the per-dataset status so other
      operations (force_restart, hash mismatch) see one source of truth.
    """
    from agent.train_scenarios.generation.generate import generate_dataset, load_dataset

    dataset_cfg = config.get("dataset", {})
    dataset_dir = dataset_cfg.get("dataset_dir", "")
    save_dir = dataset_cfg.get("save_dir", "")

    # Merge generation params (game + solver + dataset)
    gen_cfg = {}
    gen_cfg.update(config.get("game", {}))
    solver_cfg = dict(config.get("solver", {}))
    gen_cfg["solver"] = solver_cfg.pop("type", "v2")
    gen_cfg.update(solver_cfg)
    gen_cfg.update(dataset_cfg)

    # Determine where to read/save the dataset.
    if save_dir:
        dataset_save_dir = save_dir
    elif dataset_dir:
        dataset_save_dir = dataset_dir
    elif resume:
        dataset_save_dir = os.path.join(base_dir, "dataset")
    else:
        dataset_save_dir = os.path.join(base_dir, "dataset", log.init_time)

    if not resume and dataset_dir:
        existing = load_dataset(dataset_dir, log=log)
        if existing is not None:
            return dataset_dir
        log(f"Dataset not found at {dataset_dir}, generating...")

    os.makedirs(dataset_save_dir, exist_ok=True)

    result = generate_dataset(
        gen_cfg, dataset_save_dir, log=log,
        resume=resume, config_hash=config_hash)

    if state is not None and result:
        state.set_dataset(
            "gto", path=dataset_save_dir, target=int(gen_cfg.get("n_scenarios", 0)),
            done=True, config_hash=config_hash)
    return dataset_save_dir


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

    version = os.path.basename(os.path.abspath(os.path.dirname(__file__)))
    name = config.get("name", "default")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    base_dir = os.path.join(project_root, "data", version, name)

    log = Logger(base_dir)
    log(f"Version: {version}, Experiment: {name}, Device: {device}")

    # Save config snapshot
    configs_dir = os.path.join(base_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    config_snapshot_path = os.path.join(configs_dir, f"{log.init_time}.json")
    with open(config_snapshot_path, "w") as f:
        json.dump(config, f, indent=4)
    log(f"Config saved to {config_snapshot_path}")

    from agent.train_scenarios.gto_ev_predict.train import train_gto_ev
    from agent.train_scenarios.gto_probs_predict.train import train_gto_probs
    from agent.train_scenarios.gto_predict.train import train_gto
    from agent.train_scenarios.modelling_predict.train import train_modelling
    from evaluation.evaluate import run_evaluation

    agent_dir = config.get("agent_dir", "")
    pipeline_cfg = config.get("pipeline", {})
    multi_agent = config.get("multi_agent")

    # ----- resume configuration --------------------------------------
    resume = bool(pipeline_cfg.get("resume", False))
    force_phases = pipeline_cfg.get("force_restart_phases", []) or []
    force_agents = pipeline_cfg.get("force_restart_agents", []) or []
    config_hash = compute_config_hash(config)

    # State file lives next to the per-agent save dirs in multi-agent runs;
    # for single-agent it sits in the experiment base_dir.
    if multi_agent:
        save_dir_cfg = multi_agent.get("save_dir", "")
        if save_dir_cfg and os.path.isabs(save_dir_cfg):
            state_root = save_dir_cfg
        else:
            state_root = os.path.join(project_root, "data", version,
                                       save_dir_cfg or name)
    else:
        state_root = base_dir
    os.makedirs(state_root, exist_ok=True)
    state_path = os.path.join(state_root, "pipeline_state.json")

    state = PipelineState.load_or_create(
        state_path, resume=resume, config_hash=config_hash, log=log,
        force_phases=force_phases, force_agents=force_agents)

    # First-time bootstrap: scan on-disk best.pt artefacts and pre-fill
    # `done` statuses so the user can enable resume on an already-trained
    # experiment without losing prior phases.
    if resume and multi_agent:
        state.bootstrap_from_disk(
            state_root, [a["name"] for a in multi_agent.get("agents", [])])
    if resume:
        log(f"Resume mode: ON. force_phases={force_phases}, "
            f"force_agents={force_agents}")

    needs_training = (pipeline_cfg.get("run_gto_ev", True)
                      or pipeline_cfg.get("run_gto_probs", False)
                      or pipeline_cfg.get("run_gto_training", False)
                      or pipeline_cfg.get("run_modelling", False))

    # In resume mode, skip loading the (potentially huge) GTO dataset if
    # every enabled training phase is already done for every agent. The
    # dataset is only consumed by the GTO training loop; when all phases
    # are complete the loop will `_run_or_resume_phase` → return
    # immediately for each phase anyway, so loading is pure waste.
    if resume and needs_training:
        _PHASE_FLAGS = [
            ("gto_ev_predict",    "run_gto_ev",       True),
            ("gto_probs_predict", "run_gto_probs",    False),
            ("gto_predict",       "run_gto_training", False),
            ("modelling_predict", "run_modelling",    False),
        ]
        all_done = True
        agent_names = []
        if multi_agent:
            skip_training = set(multi_agent.get("skip_training", []))
            agent_names = [
                a["name"] for a in multi_agent.get("agents", [])
                if a["name"] not in skip_training
            ]
        else:
            agent_names = [name]

        for aname in agent_names:
            for phase_name, flag_key, flag_default in _PHASE_FLAGS:
                if not pipeline_cfg.get(flag_key, flag_default):
                    continue
                if state.should_force_restart_phase(aname, phase_name):
                    all_done = False
                    break
                phase = state.get_phase(aname, phase_name)
                if not phase or phase.get("status") != "done":
                    all_done = False
                    break
            if not all_done:
                break

        if all_done:
            log("[resume] All GTO training phases already done — "
                "skipping dataset load")
            needs_training = False

    # --- Load/generate dataset only if training is enabled ---
    scenarios_dir = None
    if needs_training:
        scenarios_dir = _load_or_generate_dataset(
            config, base_dir, device, log,
            state=state, config_hash=config_hash, resume=resume)
        if not scenarios_dir:
            log("No dataset available. Aborting.")
            return

    dataset_cfg = config.get("dataset", {})
    val_split = dataset_cfg.get("val_split", 0.1)

    if multi_agent and needs_training:
        # --- Multi-agent training ---
        save_dir_cfg = multi_agent.get("save_dir", "")
        if save_dir_cfg and os.path.isabs(save_dir_cfg):
            save_base_dir = save_dir_cfg
        else:
            save_base_dir = os.path.join(project_root, "data", version, save_dir_cfg or name)

        game_cfg = config.get("game", {})
        solver_cfg = config.get("solver", {})
        raise_sizes = game_cfg.get("raise_sizes")
        n_actions = len(next(iter(raise_sizes.values()))) + 3 if raise_sizes else game_cfg.get("table_bins", 50) + 3
        big_blind = game_cfg.get("big_blind", 10)
        temperature = solver_cfg.get("gto_temperature", 1.0)

        skip_training = set(multi_agent.get("skip_training", []))

        for agent_cfg in multi_agent["agents"]:
            agent_name = agent_cfg["name"]
            modifiers = agent_cfg.get("modifiers", [])

            if agent_name in skip_training:
                log(f"\n=== Agent: {agent_name} — skipped "
                    f"(listed in multi_agent.skip_training) ===")
                continue

            agent_temperature = temperature
            for mod in modifiers:
                if mod.get("type") == "temperature":
                    agent_temperature = mod["value"]

            agent_base = os.path.join(save_base_dir, agent_name)
            agent_log = Logger(agent_base)
            agent_log(f"\n=== Agent: {agent_name} ===")
            agent_log(f"Modifiers: {json.dumps(modifiers)}")
            agent_log(f"Effective temperature: {agent_temperature}")

            agent = ASI(agent_log, config)
            agent.set_device(device)
            if agent_dir:
                per_agent_dir = os.path.join(agent_dir, agent_name)
                if os.path.isdir(per_agent_dir):
                    agent.load_checkpoint(per_agent_dir)
                else:
                    agent.load_checkpoint(agent_dir)
            else:
                agent_log("Agent initialized randomly")

            mod_params = (n_actions, big_blind, temperature)

            ev_train_cfg = _merge_train_config(config, "gto_ev_train")
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")

            if pipeline_cfg.get("run_gto_ev", True):
                _run_phase(
                    state, "gto_ev_predict", agent, agent_name,
                    agent_base, agent_log, device,
                    lambda run_dir, resume_state: train_gto_ev(
                        agent, ev_train_cfg, device, agent_log,
                        scenarios_dir=scenarios_dir,
                        modifiers=modifiers, mod_params=mod_params,
                        temperature=agent_temperature,
                        run_dir=run_dir, resume_state=resume_state))

            if pipeline_cfg.get("run_gto_probs", False):
                _run_phase(
                    state, "gto_probs_predict", agent, agent_name,
                    agent_base, agent_log, device,
                    lambda run_dir, resume_state: train_gto_probs(
                        agent, probs_train_cfg, device, agent_log,
                        scenarios_dir=scenarios_dir,
                        modifiers=modifiers, mod_params=mod_params,
                        temperature=agent_temperature,
                        run_dir=run_dir, resume_state=resume_state))

            if pipeline_cfg.get("run_gto_training", False):
                gto_train_cfg = _merge_train_config(config, "gto_train")
                _run_phase(
                    state, "gto_predict", agent, agent_name,
                    agent_base, agent_log, device,
                    lambda run_dir, resume_state: train_gto(
                        agent, gto_train_cfg, device, agent_log,
                        scenarios_dir=scenarios_dir,
                        modifiers=modifiers, mod_params=mod_params,
                        temperature=agent_temperature,
                        run_dir=run_dir, resume_state=resume_state))

            if pipeline_cfg.get("run_modelling", False):
                modelling_cfg = _merge_train_config(config, "modelling_train")
                _run_phase(
                    state, "modelling_predict", agent, agent_name,
                    agent_base, agent_log, device,
                    lambda run_dir, resume_state: train_modelling(
                        agent, modelling_cfg, device, agent_log,
                        scenarios_dir=scenarios_dir,
                        modifiers=modifiers, mod_params=mod_params,
                        temperature=agent_temperature,
                        run_dir=run_dir, resume_state=resume_state))

            del agent
            _release_memory(device)

    elif needs_training:
        # --- Single-agent training ---
        agent = ASI(log, config)
        agent.set_device(device)
        if agent_dir:
            agent.load_checkpoint(agent_dir)
        else:
            log("No agent_dir specified, agent initialized randomly")

        single_temperature = config.get("solver", {}).get("gto_temperature", 1.0)
        ev_train_cfg = _merge_train_config(config, "gto_ev_train")

        if pipeline_cfg.get("run_gto_ev", True):
            _, ev_run_dir = train_gto_ev(agent, ev_train_cfg, device, log,
                         scenarios_dir=scenarios_dir, temperature=single_temperature)
            if ev_run_dir:
                best_ckpt = os.path.join(ev_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_gto_probs", False):
            probs_train_cfg = _merge_train_config(config, "gto_probs_train")
            _, probs_run_dir = train_gto_probs(agent, probs_train_cfg, device, log,
                            scenarios_dir=scenarios_dir, temperature=single_temperature)
            if probs_run_dir:
                best_ckpt = os.path.join(probs_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_gto_training", False):
            gto_train_cfg = _merge_train_config(config, "gto_train")
            _, gto_run_dir = train_gto(agent, gto_train_cfg, device, log,
                            scenarios_dir=scenarios_dir, temperature=single_temperature)
            if gto_run_dir:
                best_ckpt = os.path.join(gto_run_dir, "best.pt")
                if os.path.exists(best_ckpt):
                    agent.load_checkpoint(best_ckpt)

        if pipeline_cfg.get("run_modelling", False):
            modelling_cfg = _merge_train_config(config, "modelling_train")
            train_modelling(agent, modelling_cfg, device, log,
                            scenarios_dir=scenarios_dir, temperature=single_temperature)

    scenarios_dir = None
    _release_memory(device)

    # --- Opponent data generation + training ---
    if pipeline_cfg.get("run_opponent_data", False):
        from agent.train_scenarios.generation.generate_opponent import (
            generate_opponent_dataset, load_opponent_shards,
        )

        opp_cfg = config.get("opponent_data", {})
        opp_save_cfg = opp_cfg.get("save_dir", "")
        if opp_save_cfg and os.path.isabs(opp_save_cfg):
            opp_save_dir = opp_save_cfg
        elif opp_save_cfg:
            opp_save_dir = os.path.join(project_root, "data", version, opp_save_cfg)
        elif resume:
            opp_save_dir = os.path.join(base_dir, "opponent_dataset")
        else:
            opp_save_dir = os.path.join(base_dir, "opponent_dataset", log.init_time)

        # generate_opponent_dataset returns save_dir (string), NOT the
        # loaded list.  Data stays on disk as shards until we explicitly
        # call load_opponent_shards — only when training actually needs it.
        opp_data_dir = generate_opponent_dataset(
            config, opp_save_dir, device, log,
            resume=resume, config_hash=config_hash)

        _release_memory(device)

        if resume and opp_data_dir:
            state.set_dataset(
                "opponent", path=opp_save_dir,
                target=int(opp_cfg.get("n_hands", 0)),
                done=True, config_hash=config_hash)

        if pipeline_cfg.get("run_opponent_action_train", False) and opp_data_dir:
            from agent.train_scenarios.opponent_action_predict.train import train_opponent_action

            all_opp_train_done = False
            if resume:
                all_opp_train_done = True
                _opp_agents = (
                    [a["name"] for a in multi_agent.get("agents", [])]
                    if multi_agent else [name]
                )
                for aname in _opp_agents:
                    if state.should_force_restart_phase(
                            aname, "opponent_action_predict"):
                        all_opp_train_done = False
                        break
                    phase = state.get_phase(aname, "opponent_action_predict")
                    if not phase or phase.get("status") != "done":
                        all_opp_train_done = False
                        break

            if all_opp_train_done:
                log("[resume] All opponent_action_predict phases done — "
                    "skipping data load")
            else:
                opp_scenarios_dir = (opp_data_dir if isinstance(opp_data_dir, str)
                                     else opp_save_dir)

                opp_train_cfg = config.get("opponent_action_train", {})

                if multi_agent:
                    save_dir_cfg = multi_agent.get("save_dir", "")
                    if save_dir_cfg and os.path.isabs(save_dir_cfg):
                        save_base_dir_opp = save_dir_cfg
                    else:
                        save_base_dir_opp = os.path.join(project_root, "data", version,
                                                         save_dir_cfg or name)

                    for agent_cfg in multi_agent["agents"]:
                        agent_name = agent_cfg["name"]
                        agent_temperature = config.get("solver", {}).get("gto_temperature", 1.0)
                        for mod in agent_cfg.get("modifiers", []):
                            if mod.get("type") == "temperature":
                                agent_temperature = mod["value"]

                        agent_base = os.path.join(save_base_dir_opp, agent_name)
                        agent_log = Logger(agent_base)
                        agent_log(f"\n=== Opponent Action Training: {agent_name} ===")

                        agent = ASI(agent_log, config)
                        agent.set_device(device)
                        agent.load_checkpoint(agent_base)

                        _run_phase(
                            state, "opponent_action_predict", agent, agent_name,
                            agent_base, agent_log, device,
                            lambda run_dir, resume_state, _agent=agent,
                                   _temp=agent_temperature: train_opponent_action(
                                _agent, opp_train_cfg, device, agent_log,
                                scenarios_dir=opp_scenarios_dir,
                                temperature=_temp,
                                run_dir=run_dir, resume_state=resume_state))

                        del agent
                        _release_memory(device)

                else:
                    agent = ASI(log, config)
                    agent.set_device(device)
                    single_opp_load = base_dir
                    if os.path.isdir(single_opp_load):
                        agent.load_checkpoint(single_opp_load)
                    elif agent_dir:
                        agent.load_checkpoint(agent_dir)

                    single_temp = config.get("solver", {}).get("gto_temperature", 1.0)
                    _, opp_run_dir = train_opponent_action(
                        agent, opp_train_cfg, device, log,
                        scenarios_dir=opp_scenarios_dir,
                        temperature=single_temp,
                    )
                    if opp_run_dir:
                        best_ckpt = os.path.join(opp_run_dir, "best.pt")
                        if os.path.exists(best_ckpt):
                            agent.load_checkpoint(best_ckpt)

                    del agent
                    _release_memory(device)

            _release_memory(device)

    _release_memory(device)

    # --- MCTS cyclic collect → train ---
    if pipeline_cfg.get("run_mcts_train", False):
        from agent.train_scenarios.mcts_predict.train import train_mcts
        from agent.mcts.collect import run_mcts_collection

        mcts_train_cfg = config.get("mcts_train", {})
        examples_dir = mcts_train_cfg.get("examples_dir", "")
        n_cycles = mcts_train_cfg.get("n_cycles", 1)
        n_hands_per_cycle = mcts_train_cfg.get("n_hands_per_cycle", 500)

        # Resolve agent save directory
        if multi_agent:
            save_dir_cfg = multi_agent.get("save_dir", "")
            if save_dir_cfg and os.path.isabs(save_dir_cfg):
                save_base_dir_mcts = save_dir_cfg
            else:
                save_base_dir_mcts = os.path.join(project_root, "data", version,
                                                   save_dir_cfg or name)
        else:
            save_base_dir_mcts = base_dir

        fallback_temp = config.get("solver", {}).get("gto_temperature", 1.0)

        if examples_dir:
            # Backwards compatible: pre-generated examples, single pass
            mcts_examples = _load_mcts_examples(examples_dir, log)
            if not mcts_examples:
                log("MCTS training skipped: no examples at examples_dir")
            elif multi_agent:
                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    agent_temp = fallback_temp
                    for mod in agent_cfg.get("modifiers", []):
                        if mod.get("type") == "temperature":
                            agent_temp = mod["value"]
                    agent_base = os.path.join(save_base_dir_mcts, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== MCTS Training: {agent_name} ===")
                    agent = ASI(agent_log, config)
                    agent.set_device(device)
                    agent.load_checkpoint(agent_base)
                    train_mcts(agent, mcts_train_cfg, device, agent_log,
                               mcts_examples, temperature=agent_temp)
            else:
                agent = ASI(log, config)
                agent.set_device(device)
                if agent_dir:
                    agent.load_checkpoint(agent_dir)
                train_mcts(agent, mcts_train_cfg, device, log,
                           mcts_examples, temperature=fallback_temp)
        else:
            # Cyclic self-play: collect → train → repeat.
            # Agents are loaded ONCE from disk and trained in-place across
            # cycles so we can save checkpoints less frequently than every
            # cycle without losing training progress. History is appended to
            # a single file per agent for continuous loss curves.

            save_every_cycles = max(1, mcts_train_cfg.get("save_every_cycles", 1))

            def _make_identity_norm():
                return {"pot_mean": 0, "pot_std": 1,
                        "stack_mean": 0, "stack_std": 1,
                        "bets_mean": 0, "bets_std": 1,
                        "blind_mean": 0, "blind_std": 1,
                        "ev_mean": 0, "ev_std": 1}

            def _checkpoint_metadata(agent_obj):
                """Return agent's norm_stats dict. If the checkpoint had none,
                create a per-agent identity-norm dict and attach it to the
                agent so mutations (e.g. `_finalize_value_targets` bootstrapping
                `mcts_value_scale`) propagate through to `train_mcts._save_best`
                — which reads `agent._checkpoint_norm_stats` when writing the
                next `best.pt`. Without this, freshly bootstrapped MCTS norms
                would never reach disk and restarts would fall back to BB."""
                norm_stats = getattr(agent_obj, '_checkpoint_norm_stats', None)
                if norm_stats is None:
                    norm_stats = _make_identity_norm()
                    agent_obj._checkpoint_norm_stats = norm_stats
                return norm_stats

            def _build_persistent_optim(agent_obj, ckpt, ckpt_path,
                                         mcts_train_cfg, n_cycles, agent_log):
                """Construct a single Adam + warmup→clamped-cosine that survives
                across all cycles AND across pipeline runs.

                D.1: T_max is derived from config params (not the old
                estimated_steps_per_cycle). A clamped cosine holds eta_min
                once T_max is reached — no oscillation on late cycles.
                Warmup runs only on cycle 0 (first warmup_steps global
                steps). Optimizer / scheduler state is only restored when
                the source checkpoint was written by a previous MCTS run
                (phase tag == "mcts_predict") AND the agent's trainable
                parameter signature matches.
                """
                from torch.optim.lr_scheduler import (
                    LinearLR, CosineAnnealingLR, SequentialLR)

                # D.1: clamped cosine — holds eta_min instead of wrapping
                class _ClampedCosineAnnealingLR(CosineAnnealingLR):
                    def get_lr(self):
                        if self.last_epoch >= self.T_max:
                            return [self.eta_min
                                    for _ in self.base_lrs]
                        return super().get_lr()

                lr = float(mcts_train_cfg.get("lr", 1e-5))
                opt = torch.optim.Adam(agent_obj.parameters(), lr=lr)

                # D.1: compute T_max from actual config (no
                # estimated_steps_per_cycle). Upper-bound examples per
                # cycle from n_hands × ~6 decisions, then steps from
                # epochs × ceil(train_examples / batch_size).
                epochs = int(mcts_train_cfg.get("epochs", 5))
                batch_size = int(mcts_train_cfg.get("batch_size", 16))
                n_hands = int(mcts_train_cfg.get(
                    "n_hands_per_cycle", 100))
                val_split = float(mcts_train_cfg.get("val_split", 0.1))
                est_examples = n_hands * 6
                est_steps_per_cycle = epochs * ceil(
                    est_examples * (1.0 - val_split) / batch_size)
                total_steps = max(1, n_cycles * est_steps_per_cycle)
                warmup_steps = min(100, max(1, total_steps // 5))
                eta_min = float(mcts_train_cfg.get(
                    "scheduler_eta_min", 1e-6))

                warmup = LinearLR(opt, start_factor=0.01,
                                  total_iters=warmup_steps)
                cosine = _ClampedCosineAnnealingLR(
                    opt, T_max=max(1, total_steps - warmup_steps),
                    eta_min=eta_min)
                sched = SequentialLR(opt, [warmup, cosine],
                                     milestones=[warmup_steps])

                restored_opt = False
                restored_sched = False
                if ckpt is not None:
                    if is_legacy_mcts_ckpt(ckpt_path, ckpt):
                        agent_log(
                            "  WARNING: loaded model weights from a legacy "
                            "untagged mcts_predict checkpoint at "
                            f"{ckpt_path}. These weights were almost "
                            "certainly produced by the negative-LR bug — "
                            "delete this mcts_predict/ subdirectory and "
                            "restart MCTS from the prior phase's checkpoint."
                        )
                    restored_opt, restored_sched, _ = restore_optim_sched(
                        optimizer=opt, scheduler=sched, ckpt=ckpt,
                        expected_phase="mcts_predict", model=agent_obj,
                        strict=False, log=agent_log,
                    )
                msg = (
                    f"  Persistent optim: lr_now={opt.param_groups[0]['lr']:.2e}, "
                    f"horizon={total_steps} steps "
                    f"(={n_cycles} cycles × ~{est_steps_per_cycle}), "
                    f"warmup={warmup_steps}, eta_min={eta_min:.0e}, "
                    f"clamped_cosine=True, "
                    f"adam_restored={restored_opt}, "
                    f"sched_restored={restored_sched}")
                agent_log(msg)
                return opt, sched

            _release_memory(device)

            # Resume state for MCTS (no-op when state.resume == False).
            mcts_state = state.get_mcts() or {}
            saved_run_dirs = mcts_state.get("run_dirs", {}) or {}
            saved_cum_steps = mcts_state.get("cumulative_steps", {}) or {}
            saved_examples_paths = mcts_state.get("examples_paths", {}) or {}
            start_cycle = int(mcts_state.get("next_cycle", 0))
            resume_stage = mcts_state.get("stage", "collecting")

            # Build the persistent agent registry once
            trained_agents = []  # list of dicts kept across cycles
            # Opponent-embedding tables persist ACROSS cycles (collection dict
            # keyed by agent name; per-agent training table lives in
            # agent_info["opp_emb_table"]) so long-run context about players
            # the agent has met many times accumulates instead of resetting
            # every cycle.
            mcts_collection_opp_tables = {}
            if multi_agent:
                for agent_cfg in multi_agent["agents"]:
                    agent_name = agent_cfg["name"]
                    agent_base = os.path.join(save_base_dir_mcts, agent_name)
                    agent_log = Logger(agent_base)
                    agent_log(f"\n=== Loading {agent_name} for cyclic MCTS ===")

                    agent_obj = ASI(agent_log, config)
                    agent_obj.set_device(device)
                    agent_obj.load_checkpoint(agent_base)
                    agent_obj.eval()

                    # Resolve temperature from checkpoint or modifiers
                    ckpt_path = agent_obj._find_best_checkpoint(agent_base)
                    temp = fallback_temp
                    loaded_ckpt = None
                    if ckpt_path:
                        loaded_ckpt = torch.load(
                            ckpt_path, weights_only=False, map_location=device)
                        temp = loaded_ckpt.get("temperature", fallback_temp)
                    for mod in agent_cfg.get("modifiers", []):
                        if mod.get("type") == "temperature":
                            temp = mod["value"]

                    # Reuse run_dir from pipeline_state when resuming — keeps
                    # best.pt and history.pt in the same place across runs.
                    saved_run_dir = saved_run_dirs.get(agent_name)
                    if state.resume and saved_run_dir and os.path.isdir(saved_run_dir):
                        run_dir = saved_run_dir
                        agent_log(f"  [resume] reusing run_dir {run_dir}")
                    else:
                        run_dir = agent_log.run_dir("mcts_predict")
                    scenario_dir = os.path.dirname(run_dir)
                    history_path = os.path.join(scenario_dir, "history.pt")
                    cumulative_step = 0
                    if state.resume and agent_name in saved_cum_steps:
                        cumulative_step = int(saved_cum_steps[agent_name])
                    elif (os.path.exists(history_path)
                          or os.path.isdir(os.path.join(scenario_dir,
                                                        "history_shards"))):
                        existing = IncrementalHistory(
                            scenario_dir,
                            keys=["step_loss", "val_loss",
                                  "epoch_train_loss", "epoch_val_loss",
                                  "cycles"]).data
                        steps = existing.get("step_loss", []) or []
                        if steps and isinstance(steps[-1], dict):
                            cumulative_step = int(steps[-1].get("step", 0)) + 1

                    optimizer, scheduler = _build_persistent_optim(
                        agent_obj, loaded_ckpt, ckpt_path,
                        mcts_train_cfg, n_cycles, agent_log)

                    from utils import get_amp_config
                    _, _, _, use_scaler = get_amp_config(device)
                    agent_scaler = torch.amp.GradScaler(enabled=use_scaler)

                    trained_agents.append({
                        "agent": agent_obj,
                        "norm_stats": _checkpoint_metadata(agent_obj),
                        "name": agent_name,
                        "temperature": temp,
                        "agent_log": agent_log,
                        "run_dir": run_dir,
                        "history_path": history_path,
                        "cumulative_step": cumulative_step,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "scaler": agent_scaler,
                    })
            else:
                single_load_dir = save_base_dir_mcts or agent_dir
                agent_obj = ASI(log, config)
                agent_obj.set_device(device)
                agent_obj.load_checkpoint(single_load_dir)
                agent_obj.eval()
                saved_run_dir_single = saved_run_dirs.get(name)
                if state.resume and saved_run_dir_single \
                        and os.path.isdir(saved_run_dir_single):
                    run_dir = saved_run_dir_single
                    log(f"  [resume] reusing single-agent MCTS run_dir {run_dir}")
                else:
                    run_dir = log.run_dir("mcts_predict")
                scenario_dir = os.path.dirname(run_dir)
                history_path = os.path.join(scenario_dir, "history.pt")
                cumulative_step = 0
                if state.resume and name in saved_cum_steps:
                    cumulative_step = int(saved_cum_steps[name])
                elif (os.path.exists(history_path)
                      or os.path.isdir(os.path.join(scenario_dir,
                                                    "history_shards"))):
                    existing = IncrementalHistory(
                        scenario_dir,
                        keys=["step_loss", "val_loss",
                              "epoch_train_loss", "epoch_val_loss",
                              "cycles"]).data
                    steps = existing.get("step_loss", []) or []
                    if steps and isinstance(steps[-1], dict):
                        cumulative_step = int(steps[-1].get("step", 0)) + 1

                # Re-find checkpoint for optimizer/scheduler state
                single_ckpt_path = agent_obj._find_best_checkpoint(single_load_dir)
                single_ckpt = None
                if single_ckpt_path:
                    single_ckpt = torch.load(
                        single_ckpt_path, weights_only=False, map_location=device)
                optimizer, scheduler = _build_persistent_optim(
                    agent_obj, single_ckpt, single_ckpt_path,
                    mcts_train_cfg, n_cycles, log)

                trained_agents.append({
                    "agent": agent_obj,
                    "norm_stats": _checkpoint_metadata(agent_obj),
                    "name": name,
                    "temperature": fallback_temp,
                    "agent_log": log,
                    "run_dir": run_dir,
                    "history_path": history_path,
                    "cumulative_step": cumulative_step,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                })

            # MCTS value-norm drift is handled per cycle inside
            # `_finalize_value_targets` (collect.py): the agent's policy
            # drifts during cyclic training, so `mcts_value_scale` is
            # EMA-smoothed toward each cycle's freshly observed robust scale
            # (`mcts_train.value_scale_ema`). The old one-shot forced
            # rebootstrap (`value_norm_rebootstrap_every`) was removed — it
            # shocked the target axis faster than the value head (few grad
            # steps/cycle) could re-map. `mcts_value_scale` is NEVER popped
            # here: search and terminal eval must keep one coherent axis
            # within a cycle (popping it made `evaluate_all_terminals` fall
            # back to BB while the value head was still on the old scale →
            # PUCT compared apples to oranges → fold collapse).
            log(f"\nCyclic MCTS: {n_cycles} cycles, "
                f"save_every_cycles={save_every_cycles}")

            # Legacy `mcts_ev_*` keys (pre-Step-1 redesign) are unused; the
            # tuple is kept for checkpoint backward-compat (older best.pt
            # files still carry these keys).
            # See versions/v5/PLAN_MCTS_VALUE_REDESIGN.md for math.
            _LEGACY_MCTS_NORM_KEYS = ("mcts_ev_mean", "mcts_ev_std",
                                       "mcts_ev_n_samples",
                                       "mcts_ev_ratio_min",
                                       "mcts_ev_ratio_max")

            if state.resume and start_cycle > 0:
                log(f"\n[resume] MCTS resuming at cycle {start_cycle}/{n_cycles} "
                    f"(stage={resume_stage})")

            for cycle in range(start_cycle, n_cycles):
                log(f"\n=== MCTS Cycle {cycle + 1}/{n_cycles} ===")

                # --- Collection (or resume from saved examples) ---
                per_agent_examples = None
                if (state.resume and cycle == start_cycle
                        and resume_stage == "training"):
                    # Prior run crashed mid-training of this cycle — reload
                    # the examples we saved before training started.
                    per_agent_examples = {}
                    all_loaded = True
                    for a in trained_agents:
                        p = saved_examples_paths.get(a["name"])
                        if p and os.path.exists(p):
                            per_agent_examples[a["name"]] = torch.load(
                                p, weights_only=False)
                            a["agent_log"](
                                f"  [resume] loaded {len(per_agent_examples[a['name']])} "
                                f"examples from {os.path.basename(p)}")
                        else:
                            all_loaded = False
                            a["agent_log"](
                                f"  [resume] examples missing for {a['name']}; "
                                f"will recollect")
                    if not all_loaded:
                        per_agent_examples = None

                if per_agent_examples is None:
                    # Fresh collection. Mark stage so a crash here is recovered
                    # by re-collecting (collection is non-deterministic; we
                    # don't try to checkpoint mid-collection).
                    state.set_mcts(next_cycle=cycle, stage="collecting")

                    agents_for_play = [
                        {"agent": a["agent"], "norm_stats": a["norm_stats"],
                         "name": a["name"], "temperature": a["temperature"]}
                        for a in trained_agents
                    ]

                    # Past-opponent snapshots: fresh disk scan each cycle so
                    # newly-written cycle_NNNN.pt files are picked up. Empty
                    # the first time (no snapshots exist before the first
                    # save-cycle).
                    past_snapshot_specs = []
                    past_cfg = (mcts_train_cfg.get("past_opponents") or {})
                    n_workers_cfg = int(
                        mcts_train_cfg.get("n_workers", 1) or 1)
                    if (past_cfg.get("enabled", False)
                            and n_workers_cfg <= 1):
                        past_snapshot_specs = _discover_past_snapshots(
                            trained_agents,
                            past_cfg.get("max_snapshots_per_agent"),
                            log)
                    elif (past_cfg.get("enabled", False)
                          and n_workers_cfg > 1):
                        log("  past_opponents enabled but n_workers>1 — "
                            "disabled (inference-server spec is frozen).")

                    per_agent_examples = run_mcts_collection(
                        agents_for_play, config, device, log, n_hands_per_cycle,
                        cycle_idx=cycle, n_cycles=n_cycles,
                        past_snapshot_specs=past_snapshot_specs,
                        opp_tables=mcts_collection_opp_tables)

                    # Persist examples atomically BEFORE training so a crash
                    # during train_mcts can resume the SAME data on next start.
                    new_examples_paths = {}
                    for a in trained_agents:
                        exs = per_agent_examples.get(a["name"], [])
                        if not exs:
                            continue
                        ex_dir = os.path.join(a["run_dir"], "examples")
                        os.makedirs(ex_dir, exist_ok=True)
                        p = os.path.join(ex_dir, f"cycle_{cycle:04d}.pt")
                        atomic_torch_save(exs, p)
                        new_examples_paths[a["name"]] = p
                    state.set_mcts(stage="training",
                                   examples_paths=new_examples_paths)
                    saved_examples_paths = new_examples_paths

                is_save_cycle = (cycle % save_every_cycles == 0
                                 or cycle == n_cycles - 1)

                for agent_info in trained_agents:
                    agent_name = agent_info["name"]
                    examples = per_agent_examples.get(agent_name, [])
                    if not examples:
                        agent_info["agent_log"](
                            f"  {agent_name}: no examples this cycle, skipping training")
                        continue

                    agent_info["agent_log"](
                        f"\n--- MCTS Train: {agent_name} "
                        f"(cycle {cycle + 1}/{n_cycles}, {len(examples)} examples, "
                        f"save={is_save_cycle}) ---")

                    if (agent_info.get("opp_emb_table") is None
                            and agent_info["agent"].perception.opp_emb_enabled):
                        from agent.perception.opponent_embeddings import (
                            OpponentEmbeddingTable,
                        )
                        agent_info["opp_emb_table"] = OpponentEmbeddingTable(
                            agent_info["agent"].perception.d_model)

                    _, _, new_step = train_mcts(
                        agent_info["agent"], mcts_train_cfg, device,
                        agent_info["agent_log"], examples,
                        temperature=agent_info["temperature"],
                        run_dir=agent_info["run_dir"],
                        history_path=agent_info["history_path"],
                        cycle_id=cycle,
                        global_step_offset=agent_info["cumulative_step"],
                        save_checkpoint=is_save_cycle,
                        run_timestamp=getattr(agent_info["agent_log"],
                                               "init_time", None),
                        optimizer=agent_info["optimizer"],
                        scheduler=agent_info["scheduler"],
                        scaler=agent_info.get("scaler"),
                        save_every_cycles=save_every_cycles,
                        opponent_emb_table=agent_info.get("opp_emb_table"),
                    )
                    agent_info["cumulative_step"] = new_step

                # End of cycle — record progress so a crash in the NEXT
                # cycle starts at cycle+1 rather than re-running this one.
                state.set_mcts(
                    next_cycle=cycle + 1,
                    stage="collecting",
                    run_dirs={a["name"]: a["run_dir"]
                              for a in trained_agents},
                    cumulative_steps={a["name"]: a["cumulative_step"]
                                       for a in trained_agents},
                    examples_paths=saved_examples_paths,
                )
                # When this cycle wrote a permanent checkpoint, the saved
                # examples for any cycle through this one are no longer
                # needed (the model state subsumes them). Delete them to
                # avoid unbounded disk usage.
                if is_save_cycle:
                    for a in trained_agents:
                        ex_dir = os.path.join(a["run_dir"], "examples")
                        if not os.path.isdir(ex_dir):
                            continue
                        for fname in os.listdir(ex_dir):
                            if not (fname.startswith("cycle_")
                                    and fname.endswith(".pt")):
                                continue
                            try:
                                prev_cid = int(fname[len("cycle_"):-3])
                            except ValueError:
                                continue
                            if prev_cid <= cycle:
                                try:
                                    os.remove(os.path.join(ex_dir, fname))
                                except OSError:
                                    pass
                    # Clear examples_paths in state — they're gone from disk.
                    state.set_mcts(examples_paths={})
                    saved_examples_paths = {}

    # --- Evaluation (after all training stages) ---
    if pipeline_cfg.get("run_evaluation", False):
        run_evaluation(config, device, log)

    # --- Slumbot evaluation (external HU benchmark) ---
    if pipeline_cfg.get("run_slumbot_eval", False):
        from evaluation.slumbot_eval import run_slumbot_evaluation
        run_slumbot_evaluation(config, device, log)


def _load_mcts_examples(examples_dir, log):
    """Load pre-generated MCTS training examples from a directory."""
    import torch

    if not examples_dir:
        return None
    if not os.path.isdir(examples_dir):
        # Try as direct file path
        if os.path.isfile(examples_dir):
            log(f"Loading MCTS examples from {examples_dir}")
            return torch.load(examples_dir, weights_only=False)
        log(f"MCTS examples path not found: {examples_dir}")
        return None

    # Search for latest examples file in directory
    candidates = sorted(
        [f for f in os.listdir(examples_dir) if f.endswith(".pt")],
        reverse=True,
    )
    if not candidates:
        log(f"No .pt files found in {examples_dir}")
        return None

    path = os.path.join(examples_dir, candidates[0])
    log(f"Loading MCTS examples from {path}")
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    main()
