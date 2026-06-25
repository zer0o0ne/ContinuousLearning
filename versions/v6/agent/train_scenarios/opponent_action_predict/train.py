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
    def __init__(self, indices, p_outs, p_masks, base_dataset):
        self.indices = list(indices)
        self.p_outs = p_outs
        self.p_masks = p_masks
        self.base = base_dataset
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        oidx = self.indices[idx]
        return self.p_outs[oidx], self.p_masks[oidx], self.base[oidx][1]


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
    for s in scenarios:
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


def _run_validation(agent, val_loader, device, amp_config=None,
                    opponent_emb_table=None, gru_window=1,
                    use_perception_cache=False):
    """Run validation. Returns (avg_loss, top1_accuracy)."""
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    skip_opp = (opponent_emb_table is None)
    if opponent_emb_table is not None:
        opponent_emb_table = opponent_emb_table.clone()
    agent.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
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
                event_sequences, precomputed, target_probs = batch
                target_probs = target_probs.to(device)
                n_samples = precomputed["B"] if precomputed is not None else 0
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    out = agent.forward_batch(event_sequences, skip_memory=True,
                                              heads={"opponent_action"},
                                              skip_opponent_emb=skip_opp,
                                              opponent_emb_table=opponent_emb_table,
                                              gru_window=gru_window,
                                              precomputed=precomputed)
                    logits = out["opponent_action_logits"]
                    batch_loss = _kl_loss(logits, target_probs)
            loss_sum += batch_loss.item() * n_samples
            correct += (logits.argmax(dim=-1) == target_probs.argmax(dim=-1)).sum().item()
            count += n_samples
    agent.train()
    n = max(count, 1)
    return loss_sum / n, correct / n


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
                          scenarios_override=None, temperature=None,
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

    log("Frozen: perception (encoder/decoder/embedder), value_head, action_head, modelling_head")
    log(f"Training: opponent_action_head" +
        (", opponent_gru" if use_opp_emb else ""))
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

    # Dataset
    if scenarios_override is None:
        log("ERROR: scenarios_override required for opponent action training")
        return None, run_dir
    scenarios = scenarios_override
    log(f"Using {len(scenarios)} raw scenarios")

    # Use norm stats from checkpoint (same distribution perception was trained on)
    # to avoid distribution mismatch with frozen perception. On resume, the
    # stats in `resume_state` take precedence.
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

    # Train/val split (hand-aware, mapped to expanded indices)
    from agent.train_scenarios.split import hand_aware_split_expanded
    dataset = OpponentActionDataset(scenarios, norm_stats=norm_stats)
    train_dataset, val_dataset = hand_aware_split_expanded(
        dataset, scenarios, val_split, dataset.indices)

    log(f"Expanded samples: {len(dataset)} (train: {len(train_dataset)}, val: {len(val_dataset)})")

    # Opponent embedding table (persistent across batches, detached after each backward)
    opp_table = None
    if use_opp_emb:
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        opp_table = OpponentEmbeddingTable(agent.perception.d_model)
        log("Opponent GRU embedding enabled")

    _use_perception_cache = not use_opp_emb

    if use_opp_emb:
        # A.4.4: feed in hand_id (chronological) order so the opponent GRU table
        # accumulates exactly as it would at inference — not length-shuffled.
        from agent.train_scenarios.split import OrderedBatchSampler
        base_ds = train_dataset.dataset       # OpponentActionDataset
        expanded = base_ds.indices            # [(s_idx, hero_pos), ...]
        scens = base_ds.scenarios
        keyed = []
        for pos, exp_idx in enumerate(train_dataset.indices):
            s_idx = expanded[exp_idx][0]
            hid = scens[s_idx].get("hand_id", s_idx)
            keyed.append((hid, exp_idx, pos))
        keyed.sort(key=lambda t: (t[0], t[1]))   # hand_id, then scenario order
        order = [pos for _, _, pos in keyed]
        train_sampler = OrderedBatchSampler(order, batch_size)
        log("Opponent GRU active → chronological (hand_id) batch order")
        from agent.train_scenarios.opponent_action_predict.dataset import make_tensor_collate
        max_players = agent.perception.embedder.max_players
        _tc = make_tensor_collate(max_players)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  collate_fn=_tc, num_workers=2,
                                  persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=_tc, num_workers=2,
                                persistent_workers=True)
    else:
        # E.5.1: pre-compute frozen perception outputs (all perception params
        # frozen when opp_emb disabled — outputs are deterministic)
        log("Pre-computing frozen perception outputs...")
        _p_outs = []
        _p_masks = []
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch_events = [dataset[j][0] for j in range(start, end)]
                p_out, _, m = agent.perception.forward_batch(
                    batch_events, device=device, skip_memory=True)
                for k in range(p_out.shape[0]):
                    L = int(m[k].sum().item())
                    _p_outs.append(p_out[k, :L].detach().cpu())
                    _p_masks.append(m[k, :L].detach().cpu())
        log(f"Cached {len(_p_outs)} perception outputs")

        cached_train = _CachedDataset(train_dataset.indices, _p_outs, _p_masks, dataset)
        cached_val = _CachedDataset(val_dataset.indices, _p_outs, _p_masks, dataset)

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
                                    "epoch_train_loss", "epoch_val_loss"])
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

        for batch_idx, batch in enumerate(train_loader):
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
                event_sequences, precomputed, target_probs = batch
                target_probs = target_probs.to(device)
                n_samples = precomputed["B"] if precomputed is not None else 0
                with torch.autocast(device_type=device_type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    out = agent.forward_batch(event_sequences, skip_memory=True,
                                              heads={"opponent_action"},
                                              skip_opponent_emb=(opp_table is None),
                                              opponent_emb_table=opp_table,
                                              gru_window=gru_window,
                                              precomputed=precomputed)
                    logits = out["opponent_action_logits"]
                    batch_loss = _kl_loss(logits, target_probs)

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
                val_loss, val_acc = _run_validation(
                    agent, val_loader, device, amp_config=amp_cfg,
                    opponent_emb_table=opp_table, gru_window=gru_window,
                    use_perception_cache=_use_perception_cache)
                history["val_loss"].append((global_step, val_loss))
                history["val_accuracy"].append((global_step, val_acc))
                _save_history(hist)
                log(f"  [Step {global_step}] Val Loss: {val_loss:.6f}, "
                    f"Acc: {val_acc:.4f}")

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
        val_loss, val_acc = _run_validation(
            agent, val_loader, device, amp_config=amp_cfg,
            opponent_emb_table=opp_table, gru_window=gru_window,
            use_perception_cache=_use_perception_cache)
        history["val_loss"].append((global_step, val_loss))
        history["val_accuracy"].append((global_step, val_acc))
        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss)
        _save_history(hist)

        log(f"Epoch {epoch + 1}/{epochs} — Train: {train_loss_avg:.6f}, "
            f"Val: {val_loss:.6f}, Acc: {val_acc:.4f}")

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

    return history, run_dir
