"""
Training loop for opponent action prediction.

Trains opponent_action_head to predict range-averaged opponent action
distributions from the observer's game state.

Perception, value_head, action_head, and modelling_head are all FROZEN.
opponent_action_head is trained via KL divergence loss.
When opponent_embedding is enabled, opponent_gru is also trained —
it learns to update per-opponent embeddings from observed behavior.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from agent.train_scenarios.opponent_action_predict.dataset import (
    OpponentActionDataset, batch_collate,
)
from agent.train_scenarios._checkpoint_io import (
    make_checkpoint,
    restore_optim_sched,
)
from agent.resume import atomic_torch_save
from agent.train_scenarios._history import IncrementalHistory


class _CachedDataset(torch.utils.data.Dataset):
    """Fully in-RAM dataset: perception outputs and targets are pre-extracted,
    so training never touches the sharded base dataset (whose LRU-1 shard
    cache thrashes under the length-grouped sampler's shard-random access
    order — one torch.load per item).
    """
    def __init__(self, indices, p_outs, p_masks, targets):
        self.indices = list(indices)
        self.p_outs = p_outs
        self.p_masks = p_masks
        self.targets = targets
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        oidx = self.indices[idx]
        return self.p_outs[oidx], self.p_masks[oidx], self.targets[oidx]


def _cached_collate(batch):
    p_outs_b, masks_b, targets_b = zip(*batch)
    max_len = max(p.shape[0] for p in p_outs_b)
    B = len(batch)
    d = p_outs_b[0].shape[-1]
    padded_p = torch.zeros(B, max_len, d)
    padded_m = torch.zeros(B, max_len, dtype=masks_b[0].dtype)
    for i, (p, m) in enumerate(zip(p_outs_b, masks_b)):
        L = p.shape[0]
        padded_p[i, :L] = p
        padded_m[i, :L] = m
    return padded_p, padded_m, torch.stack(targets_b)


_PHASE = "opponent_action_predict"


class LengthGroupedBatchSampler(Sampler):
    """Groups samples by sequence length into batches, shuffles batch order.

    Batch composition is re-randomized every epoch: indices are shuffled
    before stable-sorting by length, so same-length samples get different
    neighbours each time (E.5.6).
    """

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.n = len(dataset)
        self.lengths = [len(dataset[i][0]) for i in range(self.n)]

    def __iter__(self):
        indices = list(range(self.n))
        random.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i])
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def _compute_norm_stats(scenarios):
    """Compute normalization stats from shared-format opponent scenarios."""
    pots, stacks, all_bets, blinds = [], [], [], []
    for s in tqdm(scenarios, desc="Computing norm stats", leave=False, smoothing=0):
        for event in s["events"]:
            pots.append(event["pot"])
            stacks.extend(event["stacks"])
            blinds.append(event["big_blind"])
            raw_bets = event["bets"]
            if isinstance(raw_bets, np.ndarray):
                raw_bets = raw_bets.tolist()
            all_bets.extend(float(b) for b in raw_bets)

    def _stats(vals):
        arr = np.array(vals, dtype=np.float64)
        m, s = float(arr.mean()), float(arr.std())
        if s < 1e-8:
            s = 1.0
        return m, s

    return {
        "pot_mean": _stats(pots)[0], "pot_std": _stats(pots)[1],
        "stack_mean": _stats(stacks)[0], "stack_std": _stats(stacks)[1],
        "bets_mean": _stats(all_bets)[0], "bets_std": _stats(all_bets)[1],
        "blind_mean": _stats(blinds)[0], "blind_std": _stats(blinds)[1],
    }


def _kl_loss(logits, target_probs):
    """KL divergence: target || softmax(logits)."""
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def _build_probe_ctx(agent, train_cfg, use_opp_emb, log):
    """Probe supervision context (PLAN_OPPONENT_ADAPTATION §2/§4).

    Returns None when neither probe is active. Style targets are z-scored
    per dim across the pool; pool-constant dims are masked out of the loss.
    """
    style_w = float(train_cfg.get("style_probe_weight", 0.0) or 0.0)
    sd_w = float(train_cfg.get("showdown_probe_weight", 0.0) or 0.0)
    style_targets_raw = train_cfg.get("style_targets") or {}

    use_style = (use_opp_emb and style_w > 0
                 and getattr(agent, "style_probe", None) is not None
                 and len(style_targets_raw) >= 2)
    use_showdown = (use_opp_emb and sd_w > 0
                    and getattr(agent, "showdown_probe", None) is not None)

    ctx = {"use_style": False, "use_showdown": use_showdown,
           "style_w": style_w, "sd_w": sd_w}
    if use_style:
        names = sorted(style_targets_raw)
        t = torch.tensor([style_targets_raw[n] for n in names],
                         dtype=torch.float32)                    # (K, 16)
        mean = t.mean(dim=0)
        std = t.std(dim=0, unbiased=False)
        dim_mask = std >= 1e-8
        if bool(dim_mask.any()):
            tz = (t - mean) / std.clamp(min=1e-8)
            tz[:, ~dim_mask] = 0.0
            ctx.update(use_style=True,
                       style_names=names,
                       style_targets_z=tz,
                       style_by_name={n: i for i, n in enumerate(names)},
                       dim_mask=dim_mask)
            log(f"Style probe: {len(names)} pool styles, "
                f"{int(dim_mask.sum())}/{dim_mask.numel()} informative dims, "
                f"weight={style_w}")
        else:
            log("Style probe: all target dims constant across pool — disabled")
    if use_showdown:
        log(f"Showdown probe: enabled, weight={sd_w}")

    if not ctx["use_style"] and not ctx["use_showdown"]:
        return None
    return ctx


def _probe_losses(agent, out, aux, ctx, device):
    """Weighted probe losses for one batch.

    Returns (weighted_loss, style_loss_float, sd_loss_float,
    nn_correct, nn_total). Samples without a target / without opponent
    events are masked out; empty selections contribute exactly 0.
    """
    states = out["opp_last_states"]
    smask = out["opp_states_mask"] >= 0.5
    loss = states.sum() * 0.0  # scalar zero, right device/dtype, in-graph
    style_loss_val = None
    sd_loss_val = None
    nn_correct = 0
    nn_total = 0

    if ctx["use_style"]:
        rows, tgt_rows = [], []
        for i, a in enumerate(aux):
            r = ctx["style_by_name"].get(a.get("acting_agent"))
            if r is not None and bool(smask[i]):
                rows.append(i)
                tgt_rows.append(r)
        if rows:
            dm = ctx["dim_mask"].to(device)
            pred = agent.style_probe(states[rows])
            tgt = ctx["style_targets_z"].to(device)[tgt_rows]
            style_loss = F.mse_loss(pred[:, dm], tgt[:, dm])
            loss = loss + ctx["style_w"] * style_loss
            style_loss_val = float(style_loss.detach())
            with torch.no_grad():
                pool = ctx["style_targets_z"].to(device)[:, dm].float()
                d2 = torch.cdist(pred[:, dm].float(), pool)
                nn_idx = d2.argmin(dim=1)
                nn_correct = int((nn_idx == torch.tensor(
                    tgt_rows, device=device)).sum())
                nn_total = len(rows)

    if ctx["use_showdown"]:
        sd_t = torch.tensor([a.get("showdown_target", float("nan"))
                             for a in aux],
                            dtype=torch.float32, device=device)
        valid = (~torch.isnan(sd_t)) & smask
        if bool(valid.any()):
            pred = agent.showdown_probe(states[valid]).squeeze(-1)
            sd_loss = F.mse_loss(pred.float(), sd_t[valid])
            loss = loss + ctx["sd_w"] * sd_loss
            sd_loss_val = float(sd_loss.detach())

    return loss, style_loss_val, sd_loss_val, nn_correct, nn_total


def _run_validation(agent, val_loader, device, amp_config=None,
                    opponent_emb_table=None, gru_window=1,
                    use_perception_cache=False, val_group_seq=None,
                    probe_ctx=None):
    """Run validation. Returns (avg_loss, top1_accuracy, style_nn_acc).

    avg_loss includes the weighted probe losses when probe_ctx is set, so
    best-model selection optimizes the same objective as training.
    style_nn_acc is None when the style probe is inactive.
    """
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    skip_opp = (opponent_emb_table is None)
    if opponent_emb_table is not None:
        opponent_emb_table = opponent_emb_table.clone()
    agent.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    nn_correct_sum = 0
    nn_total_sum = 0
    _grp_offset = 0
    with torch.no_grad():
        for batch in val_loader:
            if use_perception_cache:
                cached_p, cached_m, target_probs = batch
                cached_p = cached_p.to(device)
                cached_m = cached_m.to(device)
                target_probs = target_probs.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = agent.opponent_action_head(cached_p, mask=cached_m)
                    batch_loss = _kl_loss(logits, target_probs)
                n_samples = cached_p.shape[0]
            else:
                event_sequences, precomputed, target_probs, aux = batch
                target_probs = target_probs.to(device)
                n_samples = precomputed["B"] if precomputed is not None else 0
                # A.4.5: val loader is sequential (shuffle=False), so the
                # aligned group sequence advances by batch length.
                _groups = None
                if val_group_seq is not None:
                    _groups = val_group_seq[
                        _grp_offset:_grp_offset + len(event_sequences)]
                _grp_offset += len(event_sequences)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    out = agent.forward_batch(event_sequences, skip_memory=True,
                                              heads={"opponent_action"},
                                              skip_opponent_emb=skip_opp,
                                              opponent_emb_table=opponent_emb_table,
                                              gru_window=gru_window,
                                              precomputed=precomputed,
                                              gru_sample_groups=_groups,
                                              collect_opp_states=(probe_ctx is not None))
                    logits = out["opponent_action_logits"]
                    batch_loss = _kl_loss(logits, target_probs)
                    if probe_ctx is not None:
                        aux_loss, _, _, nn_c, nn_t = _probe_losses(
                            agent, out, aux, probe_ctx, device)
                        batch_loss = batch_loss + aux_loss
                        nn_correct_sum += nn_c
                        nn_total_sum += nn_t
            loss_sum += batch_loss.item() * n_samples
            correct += (logits.argmax(dim=-1) == target_probs.argmax(dim=-1)).sum().item()
            count += n_samples
    agent.train()
    n = max(count, 1)
    nn_acc = (nn_correct_sum / nn_total_sum) if nn_total_sum else None
    return loss_sum / n, correct / n, nn_acc


def _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
               global_step, epoch, val_loss, log, temperature=None):
    """Save best model checkpoint."""
    best_path = os.path.join(ckpt_dir, "best.pt")
    extra = {"step": global_step, "epoch": epoch + 1}
    if temperature is not None:
        extra["temperature"] = temperature
    ckpt = make_checkpoint(
        phase=_PHASE, model=agent, optimizer=optimizer, scheduler=scheduler,
        norm_stats=norm_stats, val_loss=val_loss, extra=extra,
    )
    torch.save(ckpt, best_path)
    log(f"  New best model (val loss: {val_loss:.6f})")


def _save_history(hist):
    """Save training history incrementally (E.5.3: append-only shards)."""
    hist.save()


def _save_latest(agent, optimizer, scheduler, norm_stats, run_dir,
                 next_epoch, global_step, best_val_loss, fails_since_best,
                 val_loss, temperature=None):
    """Atomic per-epoch checkpoint for pipeline.resume."""
    extra = {
        "next_epoch":       next_epoch,
        "global_step":      global_step,
        "best_val_loss":    best_val_loss,
        "fails_since_best": fails_since_best,
    }
    if temperature is not None:
        extra["temperature"] = temperature
    ckpt = make_checkpoint(
        phase=_PHASE, model=agent, optimizer=optimizer, scheduler=scheduler,
        norm_stats=norm_stats, val_loss=val_loss, extra=extra,
    )
    atomic_torch_save(ckpt, os.path.join(run_dir, "latest.pt"))


def train_opponent_action(agent, train_cfg, device, log,
                          scenarios_override=None, scenarios_dir=None,
                          temperature=None,
                          run_dir=None, resume_state=None):
    """Train opponent_action_head on range-based opponent action data.

    Freezes perception, value_head, action_head, modelling_head.
    Trains only opponent_action_head via KL divergence.

    Args:
        agent: ASI model instance (already on device)
        train_cfg: dict with training hyperparameters
        device: torch device string
        log: logger callable
        scenarios_override: raw opponent scenarios (shared event format)
        scenarios_dir: path to sharded opponent data directory
        temperature: agent temperature (saved in checkpoint)
        run_dir: pre-existing run directory (reused on resume).
        resume_state: optimizer/scheduler/counters from a prior interrupted
            run — see gto_ev_predict.train for the schema.

    Returns:
        (history, run_dir)
    """
    lr = train_cfg.get("lr", 1e-4)
    batch_size = train_cfg.get("batch_size", 64)
    epochs = train_cfg.get("epochs", 5)
    val_split = train_cfg.get("val_split", 0.1)
    log_every = train_cfg.get("log_every", 10)
    val_every = train_cfg.get("val_every", None)
    interrupt_after_fails = train_cfg.get("interrupt_after_fails", None)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    gru_window = max(1, int(train_cfg.get("gru_window", 1)))

    log("=== Opponent Action Prediction Training ===")

    # Freeze everything except opponent_action_head (+ opponent_gru if enabled)
    for param in agent.perception.parameters():
        param.requires_grad = False
    for param in agent.value_head.parameters():
        param.requires_grad = False
    for param in agent.action_head.parameters():
        param.requires_grad = False
    for param in agent.modelling_head.parameters():
        param.requires_grad = False
    for param in agent.opponent_action_head.parameters():
        param.requires_grad = True

    # Opponent GRU: unfreeze and include in optimizer if enabled
    use_opp_emb = agent.perception.opp_emb_enabled
    if use_opp_emb:
        for param in agent.perception.opponent_gru.parameters():
            param.requires_grad = True

    trainable_params = list(agent.opponent_action_head.parameters())
    if use_opp_emb:
        trainable_params += list(agent.perception.opponent_gru.parameters())

    # §3: HUD stats projection trains alongside the GRU (it was frozen by the
    # blanket perception freeze above).
    _stats_proj = getattr(agent.perception, "opp_stats_proj", None)
    if use_opp_emb and _stats_proj is not None:
        for param in _stats_proj.parameters():
            param.requires_grad = True
        trainable_params += list(_stats_proj.parameters())

    # §2/§4: probe supervision. Probes train only here; requires_grad is set
    # explicitly either way so no stale grads accumulate in other phases.
    probe_ctx = _build_probe_ctx(agent, train_cfg, use_opp_emb, log)
    _use_style = bool(probe_ctx and probe_ctx["use_style"])
    _use_showdown = bool(probe_ctx and probe_ctx["use_showdown"])
    if getattr(agent, "style_probe", None) is not None:
        for param in agent.style_probe.parameters():
            param.requires_grad = _use_style
        if _use_style:
            trainable_params += list(agent.style_probe.parameters())
    if getattr(agent, "showdown_probe", None) is not None:
        for param in agent.showdown_probe.parameters():
            param.requires_grad = _use_showdown
        if _use_showdown:
            trainable_params += list(agent.showdown_probe.parameters())

    log("Frozen: perception (encoder/decoder/embedder), value_head, action_head, modelling_head")
    log(f"Training: opponent_action_head" +
        (", opponent_gru" if use_opp_emb else "") +
        (", opp_stats_proj" if (use_opp_emb and _stats_proj is not None) else "") +
        (", style_probe" if _use_style else "") +
        (", showdown_probe" if _use_showdown else ""))
    if use_opp_emb:
        log(f"GRU window: {gru_window} step(s)")

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if run_dir is None:
        run_dir = log.run_dir("opponent_action_predict")
    else:
        os.makedirs(run_dir, exist_ok=True)

    # AMP
    from utils import get_amp_config
    amp_enabled, device_type, amp_dtype, use_scaler = get_amp_config(device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    amp_cfg = (amp_enabled, device_type, amp_dtype)
    if amp_enabled:
        log(f"AMP enabled: {device_type}, dtype={amp_dtype}, scaler={use_scaler}")

    # Dataset — sharded path (scenarios_dir) or legacy list path (scenarios_override)
    _sharded = scenarios_dir is not None
    if _sharded:
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedOpponentDataset, ShardBatchSampler,
            scan_opponent_metadata, compute_opponent_norm_stats_from_shards,
            shard_aware_split_expanded,
        )
        shards = ShardedScenarios(scenarios_dir, shard_subdir="shards")
        log(f"Using sharded opponent data from {scenarios_dir} "
            f"({len(shards)} scenarios, {shards.n_shards} shards)")
        hand_ids, n_events, expanded_indices = scan_opponent_metadata(shards)
        log(f"Expanded samples: {len(expanded_indices)}")

        if resume_state is not None and resume_state.get("norm_stats"):
            norm_stats = resume_state["norm_stats"]
            log("Using norm stats from resume_state")
        else:
            checkpoint_norm_stats = getattr(agent, '_checkpoint_norm_stats', None)
            if checkpoint_norm_stats is not None:
                norm_stats = checkpoint_norm_stats
                log("Using norm stats from agent checkpoint (matches perception training)")
            else:
                norm_stats = compute_opponent_norm_stats_from_shards(shards)
                log("No checkpoint norm stats — computed from opponent shards")
        log("Norm stats: " + ", ".join(f"{k}={v:.4f}" for k, v in norm_stats.items()))

        train_exp_idx, val_exp_idx = shard_aware_split_expanded(
            hand_ids, expanded_indices, val_split)
        log(f"Split: train {len(train_exp_idx)}, val {len(val_exp_idx)}")

        dataset = ShardedOpponentDataset(shards, expanded_indices, norm_stats=norm_stats)
        train_dataset = ShardedOpponentDataset(shards, expanded_indices,
                                                norm_stats=norm_stats,
                                                subset_indices=train_exp_idx)
        val_dataset = ShardedOpponentDataset(shards, expanded_indices,
                                              norm_stats=norm_stats,
                                              subset_indices=val_exp_idx)

    elif scenarios_override is not None:
        scenarios = scenarios_override
        log(f"Using {len(scenarios)} raw scenarios")

        if resume_state is not None and resume_state.get("norm_stats"):
            norm_stats = resume_state["norm_stats"]
            log("Using norm stats from resume_state")
        else:
            checkpoint_norm_stats = getattr(agent, '_checkpoint_norm_stats', None)
            if checkpoint_norm_stats is not None:
                norm_stats = checkpoint_norm_stats
                log("Using norm stats from agent checkpoint (matches perception training)")
            else:
                norm_stats = _compute_norm_stats(scenarios)
                log("No checkpoint norm stats — computing from opponent scenarios")
        log("Norm stats: " + ", ".join(f"{k}={v:.4f}" for k, v in norm_stats.items()))

        from agent.train_scenarios.split import hand_aware_split_expanded
        dataset = OpponentActionDataset(scenarios, norm_stats=norm_stats)
        train_dataset, val_dataset = hand_aware_split_expanded(
            dataset, scenarios, val_split, dataset.indices)
    else:
        log("ERROR: scenarios_override or scenarios_dir required for opponent action training")
        return None, run_dir

    log(f"Expanded samples: {len(dataset)} (train: {len(train_dataset)}, val: {len(val_dataset)})")

    opp_table = None
    val_group_seq = None
    if use_opp_emb:
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        opp_table = OpponentEmbeddingTable(agent.perception.d_model)
        log("Opponent GRU embedding enabled")

    _use_perception_cache = not use_opp_emb

    if use_opp_emb:
        from agent.train_scenarios.split import OrderedBatchSampler

        if _sharded:
            keyed = []
            for pos, exp_idx in enumerate(train_exp_idx):
                s_idx = expanded_indices[exp_idx][0]
                hid = hand_ids[s_idx] if s_idx < len(hand_ids) else s_idx
                keyed.append((hid, exp_idx, pos, s_idx))
            val_group_seq = [expanded_indices[exp_idx][0]
                             for exp_idx in val_exp_idx]
        else:
            base_ds = train_dataset.dataset
            expanded = base_ds.indices
            scens = base_ds.scenarios
            keyed = []
            for pos, exp_idx in enumerate(train_dataset.indices):
                s_idx = expanded[exp_idx][0]
                hid = scens[s_idx].get("hand_id", s_idx)
                keyed.append((hid, exp_idx, pos, s_idx))
            val_group_seq = [expanded[exp_idx][0]
                             for exp_idx in val_dataset.indices]

        keyed.sort(key=lambda t: (t[0], t[1]))
        order = [pos for _, _, pos, _ in keyed]
        # A.4.5: scenario id per yielded sample, aligned with `order` — the
        # sampler batches `order` consecutively, so batch b covers
        # train_group_seq[b*batch_size : b*batch_size + len(batch)]. Observer
        # copies of one scenario share an id; the GRU pass advances the table
        # once per scenario instead of once per copy (train/deploy cadence).
        train_group_seq = [s_idx for _, _, _, s_idx in keyed]
        train_sampler = OrderedBatchSampler(order, batch_size)
        log("Opponent GRU active → chronological (hand_id) batch order")
        from agent.train_scenarios.opponent_action_predict.dataset import make_tensor_collate
        max_players = agent.perception.embedder.max_players
        _tc = make_tensor_collate(max_players)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  collate_fn=_tc, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=_tc, num_workers=0)
    else:
        # Targets extracted in the same pass so training never touches the
        # shards again (see _CachedDataset docstring). In the sharded path
        # the cache is spilled to disk shards (DiskShardedCache) — neither
        # the raw dataset nor the perception cache is ever fully in RAM.
        log("Pre-computing frozen perception outputs...")
        if _sharded:
            from agent.train_scenarios.sharded import (
                DiskShardedCache, CachedSubsetDataset,
            )
            _p_cache = DiskShardedCache(
                os.path.join(run_dir, "perception_cache"))
        else:
            _p_outs = []
            _p_masks = []
            _targets = []
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(dataset), batch_size),
                              total=n_batches, desc="Caching perception", smoothing=0):
                end = min(start + batch_size, len(dataset))
                items = [dataset[j] for j in range(start, end)]
                batch_events = [it[0] for it in items]
                p_out, _, m = agent.perception.forward_batch(
                    batch_events, device=device, skip_memory=True)
                for k in range(p_out.shape[0]):
                    L = int(m[k].sum().item())
                    p_k = p_out[k, :L].detach().cpu()
                    m_k = m[k, :L].detach().cpu()
                    if _sharded:
                        _p_cache.append((p_k, m_k, items[k][1]))
                    else:
                        _p_outs.append(p_k)
                        _p_masks.append(m_k)
                        _targets.append(items[k][1])

        if _sharded:
            _p_cache.finalize()
            shards.clear_cache()
            log(f"Cached {len(_p_cache)} perception outputs → {_p_cache.cache_dir}")

            cached_train = CachedSubsetDataset(_p_cache, train_exp_idx)
            cached_val = CachedSubsetDataset(_p_cache, val_exp_idx)
            exp_n_events = [n_events[s_idx] for s_idx, _ in expanded_indices]
            train_sampler = ShardBatchSampler(
                _p_cache.shard_sizes, _p_cache.shard_offsets, batch_size,
                indices=train_exp_idx, n_events=exp_n_events)
            train_loader = DataLoader(cached_train, batch_sampler=train_sampler,
                                      collate_fn=_cached_collate, num_workers=0)
            val_sampler = ShardBatchSampler(
                _p_cache.shard_sizes, _p_cache.shard_offsets, batch_size,
                indices=val_exp_idx)
            val_loader = DataLoader(cached_val, batch_sampler=val_sampler,
                                    collate_fn=_cached_collate, num_workers=0)
        else:
            log(f"Cached {len(_p_outs)} perception outputs")
            cached_train = _CachedDataset(train_dataset.indices, _p_outs, _p_masks, _targets)
            cached_val = _CachedDataset(val_dataset.indices, _p_outs, _p_masks, _targets)

            train_sampler = LengthGroupedBatchSampler(cached_train, batch_size)
            train_loader = DataLoader(cached_train, batch_sampler=train_sampler,
                                      collate_fn=_cached_collate, num_workers=0)
            val_loader = DataLoader(cached_val, batch_size=batch_size, shuffle=False,
                                    collate_fn=_cached_collate, num_workers=0)

    log(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")

    # Scheduler: warmup + cosine
    total_steps = epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 5)
    eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                               eta_min=eta_min)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    best_val_loss = float("inf")
    fails_since_best = 0
    global_step = 0
    start_epoch = 0

    hist = IncrementalHistory(run_dir,
                              keys=["step_loss", "val_loss", "val_accuracy",
                                    "epoch_train_loss", "epoch_val_loss",
                                    "style_loss", "showdown_loss",
                                    "val_style_nn_acc"])
    history = hist.data
    log(f"Loaded history (step_loss n={len(history['step_loss'])})")

    if resume_state is not None:
        restore_optim_sched(
            optimizer=optimizer, scheduler=scheduler,
            ckpt=resume_state, expected_phase=_PHASE, model=agent,
            strict=True, log=log,
            legacy_path_phase_hint=_PHASE,
        )
        start_epoch = int(resume_state.get("start_epoch", 0))
        global_step = int(resume_state.get("global_step", 0))
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        fails_since_best = int(resume_state.get("fails_since_best", 0))
        log(f"  [resume] start_epoch={start_epoch}, global_step={global_step}, "
            f"best_val_loss={best_val_loss:.6f}, "
            f"fails_since_best={fails_since_best}")

    stopped_early = False

    for epoch in range(start_epoch, epochs):
        if stopped_early:
            break

        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Opp action epoch {epoch+1}/{epochs}", leave=False, smoothing=0)):
            if _use_perception_cache:
                cached_p, cached_m, target_probs = batch
                cached_p = cached_p.to(device)
                cached_m = cached_m.to(device)
                target_probs = target_probs.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    logits = agent.opponent_action_head(cached_p, mask=cached_m)
                    batch_loss = _kl_loss(logits, target_probs)
                n_samples = cached_p.shape[0]
            else:
                event_sequences, precomputed, target_probs, aux = batch
                target_probs = target_probs.to(device)
                n_samples = precomputed["B"] if precomputed is not None else 0
                # A.4.5: scenario ids for this batch — OrderedBatchSampler
                # yields consecutive slices of `order`, so the aligned
                # group sequence slices the same way.
                _b0 = batch_idx * batch_size
                _groups = train_group_seq[_b0:_b0 + len(event_sequences)]
                with torch.autocast(device_type=device_type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    out = agent.forward_batch(event_sequences, skip_memory=True,
                                              heads={"opponent_action"},
                                              skip_opponent_emb=(opp_table is None),
                                              opponent_emb_table=opp_table,
                                              gru_window=gru_window,
                                              precomputed=precomputed,
                                              gru_sample_groups=_groups,
                                              collect_opp_states=(probe_ctx is not None))
                    logits = out["opponent_action_logits"]
                    batch_loss = _kl_loss(logits, target_probs)
                    if probe_ctx is not None:
                        aux_loss, _style_l, _sd_l, _, _ = _probe_losses(
                            agent, out, aux, probe_ctx, device)
                        batch_loss = batch_loss + aux_loss
                        if _style_l is not None:
                            history["style_loss"].append((global_step, _style_l))
                        if _sd_l is not None:
                            history["showdown_loss"].append((global_step, _sd_l))

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            if opp_table is not None:
                opp_table.detach_all()

            step_loss = batch_loss.item()
            history["step_loss"].append((global_step, step_loss))
            global_step += 1

            train_loss_sum += step_loss * n_samples
            train_count += n_samples

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, "
                    f"Loss: {avg:.6f}, LR: {cur_lr:.2e}")

            # Intra-epoch validation
            if val_every and (global_step % val_every == 0):
                val_loss, val_acc, val_nn_acc = _run_validation(
                    agent, val_loader, device, amp_config=amp_cfg,
                    opponent_emb_table=opp_table, gru_window=gru_window,
                    use_perception_cache=_use_perception_cache,
                    val_group_seq=val_group_seq, probe_ctx=probe_ctx)
                history["val_loss"].append((global_step, val_loss))
                history["val_accuracy"].append((global_step, val_acc))
                if val_nn_acc is not None:
                    history["val_style_nn_acc"].append((global_step, val_nn_acc))
                _save_history(hist)
                log(f"  [Step {global_step}] Val Loss: {val_loss:.6f}, "
                    f"Acc: {val_acc:.4f}"
                    + (f", StyleNN: {val_nn_acc:.4f}"
                       if val_nn_acc is not None else ""))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fails_since_best = 0
                    _save_best(agent, optimizer, scheduler, norm_stats, run_dir,
                               global_step, epoch, val_loss, log,
                               temperature=temperature)
                else:
                    fails_since_best += 1
                    if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
                        log(f"  Early stopping: {fails_since_best} validations "
                            f"without improvement")
                        stopped_early = True
                        break

        if stopped_early:
            break

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # End-of-epoch validation
        val_loss, val_acc, val_nn_acc = _run_validation(
            agent, val_loader, device, amp_config=amp_cfg,
            opponent_emb_table=opp_table, gru_window=gru_window,
            use_perception_cache=_use_perception_cache,
            val_group_seq=val_group_seq, probe_ctx=probe_ctx)
        history["val_loss"].append((global_step, val_loss))
        history["val_accuracy"].append((global_step, val_acc))
        if val_nn_acc is not None:
            history["val_style_nn_acc"].append((global_step, val_nn_acc))
        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss)
        _save_history(hist)

        log(f"Epoch {epoch + 1}/{epochs} — Train: {train_loss_avg:.6f}, "
            f"Val: {val_loss:.6f}, Acc: {val_acc:.4f}"
            + (f", StyleNN: {val_nn_acc:.4f}" if val_nn_acc is not None else ""))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            fails_since_best = 0
            _save_best(agent, optimizer, scheduler, norm_stats, run_dir,
                       global_step, epoch, val_loss, log, temperature=temperature)
            should_break_after_latest = False
        else:
            fails_since_best += 1
            should_break_after_latest = (interrupt_after_fails
                                          and fails_since_best >= interrupt_after_fails)
            if should_break_after_latest:
                log(f"  Early stopping: {fails_since_best} validations "
                    f"without improvement")

        _save_latest(agent, optimizer, scheduler, norm_stats, run_dir,
                     next_epoch=epoch + 1, global_step=global_step,
                     best_val_loss=best_val_loss,
                     fails_since_best=fails_since_best,
                     val_loss=val_loss, temperature=temperature)

        if should_break_after_latest:
            break

    # Unfreeze all
    for param in agent.perception.parameters():
        param.requires_grad = True
    for param in agent.value_head.parameters():
        param.requires_grad = True
    for param in agent.action_head.parameters():
        param.requires_grad = True
    for param in agent.modelling_head.parameters():
        param.requires_grad = True

    log(f"=== Opponent Action Training Complete. Best Val Loss: {best_val_loss:.6f} ===")
    hist.compact()

    if _sharded and _use_perception_cache:
        _p_cache.cleanup()

    return history, run_dir
