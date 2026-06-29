"""
Training loop for GTO action probability prediction (Recipe Step 2).

Trains action_head to predict GTO action probabilities via KL divergence.
Perception and value_head are FROZEN — only action_head learns.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from agent.train_scenarios.generation.generate import generate_dataset, load_dataset, \
    _compute_norm_stats, _normalize_scenarios, _shallow_copy_scenarios
from agent.train_scenarios.gto_probs_predict.dataset import GTOProbsDataset, batch_collate
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


_PHASE = "gto_probs_predict"


class LengthGroupedBatchSampler(Sampler):
    """Sampler that groups samples by sequence length into batches.

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


def _kl_loss(logits, target_probs):
    """KL divergence loss: target_probs || softmax(logits).

    Args:
        logits: (B, n_actions) raw action logits
        target_probs: (B, n_actions) target probability distribution

    Returns: scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def _weighted_rank_concordance(logits, target_probs):
    """Weighted pairwise ranking concordance.

    For each pair (i,j) where target[i] > target[j], checks if logits agree.
    Weight = target[i] - target[j], so swapping high-diff actions penalizes
    much more than swapping near-equal ones.

    Returns: (B,) scores in [0, 1]. 1.0 = perfect, 0.5 = random.
    """
    target_diff = target_probs.unsqueeze(-1) - target_probs.unsqueeze(-2)
    logit_diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    mask = target_diff > 0
    weights = target_diff * mask
    concordant = (logit_diff > 0).float() * mask
    w_sum = weights.sum(dim=(-1, -2))
    c_sum = (weights * concordant).sum(dim=(-1, -2))
    return torch.where(w_sum > 0, c_sum / w_sum, torch.ones_like(w_sum))


def _run_validation(agent, val_loader, device, amp_config=None):
    """Run validation and return (avg_loss, accuracy, wrc).

    E.5.1: val_loader yields (cached_p_out, cached_mask, target_probs)
    when perception caching is active — action_head runs directly on
    pre-computed perception outputs.
    """
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    agent.eval()
    val_loss_sum = 0.0
    correct = 0
    wrc_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for cached_p, cached_m, target_probs in val_loader:
            cached_p = cached_p.to(device)
            cached_m = cached_m.to(device)
            target_probs = target_probs.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                action_logits = agent.action_head(cached_p, mask=cached_m)
                batch_loss = _kl_loss(action_logits, target_probs)
            val_loss_sum += batch_loss.item() * cached_p.shape[0]
            correct += (action_logits.argmax(dim=-1) == target_probs.argmax(dim=-1)).sum().item()
            wrc_sum += _weighted_rank_concordance(action_logits, target_probs).sum().item()
            val_count += cached_p.shape[0]
    agent.train()
    n = max(val_count, 1)
    return val_loss_sum / n, correct / n, wrc_sum / n


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
    """Atomic per-epoch checkpoint for pipeline.resume — see gto_ev_predict.train."""
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


def _check_val(val_loss, best_val_loss, fails_since_best, interrupt_after_fails, log):
    """Check validation result. Returns (new_best_val_loss, new_fails_count, should_stop)."""
    if val_loss < best_val_loss:
        return val_loss, 0, False

    fails_since_best += 1
    if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
        log(f"  Early stopping: {fails_since_best} validations without improvement")
        return best_val_loss, fails_since_best, True

    return best_val_loss, fails_since_best, False


def train_gto_probs(agent, train_cfg, device, log, scenarios_override=None,
                    scenarios_dir=None, modifiers=None, mod_params=None,
                    temperature=None, run_dir=None, resume_state=None):
    """Main training entry point for GTO action probability prediction.

    Freezes perception + value_head, trains only action_head with KL divergence.

    Args:
        agent: ASI model instance (already on device)
        train_cfg: dict with training hyperparameters (merged game + solver + gto_probs_train)
        device: torch device string
        log: logger callable
        scenarios_override: if provided, use these raw scenarios instead of loading/generating
        temperature: effective temperature for this agent (saved in checkpoint)
        run_dir: pre-existing run directory (reused on resume).
        resume_state: optimizer/scheduler/counters from a prior interrupted
            run — see gto_ev_predict.train for the schema.
    """
    lr = train_cfg.get("lr", 1e-4)
    batch_size = train_cfg.get("batch_size", 64)
    epochs = train_cfg.get("epochs", 10)
    val_split = train_cfg.get("val_split", 0.1)
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("val_every", None)
    interrupt_after_fails = train_cfg.get("interrupt_after_fails", None)

    log("=== GTO Action Probability Prediction Training (Step 2) ===")
    log("Frozen: perception, value_head. Training: action_head only")

    # Freeze perception + value_head
    for param in agent.perception.parameters():
        param.requires_grad = False
    for param in agent.value_head.parameters():
        param.requires_grad = False

    # Ensure action_head is unfrozen
    for param in agent.action_head.parameters():
        param.requires_grad = True

    # Optimizer over action_head parameters only
    trainable_params = list(agent.action_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # Run directory: reuse the one passed by the pipeline (resume) or
    # create a fresh timestamped directory.
    if run_dir is None:
        run_dir = log.run_dir("gto_probs_predict")
    else:
        os.makedirs(run_dir, exist_ok=True)

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # AMP setup
    from utils import get_amp_config
    amp_enabled, device_type, amp_dtype, use_scaler = get_amp_config(device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    amp_cfg = (amp_enabled, device_type, amp_dtype)
    if amp_enabled:
        log(f"AMP enabled: {device_type}, dtype={amp_dtype}, scaler={use_scaler}")

    # Dataset setup
    if scenarios_dir is not None:
        from agent.train_scenarios.sharded import (
            ShardedScenarios, ShardedGTODataset, ShardBatchSampler,
            scan_shard_metadata, compute_norm_stats_from_shards,
            shard_aware_split,
        )
        shards = ShardedScenarios(scenarios_dir)
        log(f"Using sharded scenarios from {scenarios_dir} ({len(shards)} samples, {shards.n_shards} shards)")

        hand_ids, n_events = scan_shard_metadata(shards)

        if resume_state is not None and resume_state.get("norm_stats"):
            norm_stats = resume_state["norm_stats"]
            log(f"Norm stats (resumed): " + ", ".join(
                f"{k}={v:.4f}" for k, v in norm_stats.items()))
        else:
            norm_stats = compute_norm_stats_from_shards(shards, modifiers=modifiers, mod_params=mod_params)
            log(f"Norm stats: " + ", ".join(
                f"{k}={v:.4f}" for k, v in norm_stats.items()))

        train_indices, val_indices = shard_aware_split(hand_ids, val_split)

        # E.5.1: pre-compute frozen perception outputs using a full dataset
        # (sequential access — shard LRU cache handles ordering)
        full_dataset = ShardedGTODataset(shards, norm_stats, phase="probs",
                                         modifiers=modifiers, mod_params=mod_params)
        log("Pre-computing frozen perception outputs...")
        _p_outs = []
        _p_masks = []
        n_batches = (len(full_dataset) + batch_size - 1) // batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(full_dataset), batch_size),
                              total=n_batches, desc="Caching perception", smoothing=0):
                end = min(start + batch_size, len(full_dataset))
                batch_events = [full_dataset[j][0] for j in range(start, end)]
                p_out, _, m = agent.perception.forward_batch(
                    batch_events, device=device, skip_memory=True)
                for k in range(p_out.shape[0]):
                    L = int(m[k].sum().item())
                    _p_outs.append(p_out[k, :L].detach().cpu())
                    _p_masks.append(m[k, :L].detach().cpu())
        log(f"Cached {len(_p_outs)} perception outputs")

        cached_train = _CachedDataset(train_indices, _p_outs, _p_masks, full_dataset)
        cached_val = _CachedDataset(val_indices, _p_outs, _p_masks, full_dataset)

        train_dataset = cached_train
        val_dataset = cached_val

        train_sampler = LengthGroupedBatchSampler(cached_train, batch_size)
        train_loader = DataLoader(cached_train, batch_sampler=train_sampler,
                                  collate_fn=_cached_collate, num_workers=0)
        val_loader = DataLoader(cached_val, batch_size=batch_size, shuffle=False,
                                collate_fn=_cached_collate, num_workers=0)
    else:
        # Original in-memory path
        if scenarios_override is not None:
            scenarios = scenarios_override
            log(f"Using provided scenarios ({len(scenarios)} samples)")
        else:
            scenarios = generate_dataset(train_cfg, run_dir, log=log)
            if not scenarios:
                log("No scenarios generated. Aborting training.")
                return None, None

        # Compute norm_stats and normalize (per-agent). On resume, reuse the
        # stats from the prior run — see gto_ev_predict.train for why.
        scenarios = _shallow_copy_scenarios(scenarios)
        if resume_state is not None and resume_state.get("norm_stats"):
            norm_stats = resume_state["norm_stats"]
            log(f"Norm stats (resumed): " + ", ".join(
                f"{k}={v:.4f}" for k, v in norm_stats.items()))
        else:
            norm_stats = _compute_norm_stats(scenarios)
            log(f"Norm stats: " + ", ".join(
                f"{k}={v:.4f}" for k, v in norm_stats.items()))
        _normalize_scenarios(scenarios, norm_stats)

        # Train/val split (hand-aware: no hand leaks between sets)
        from agent.train_scenarios.split import hand_aware_split
        dataset = GTOProbsDataset(scenarios)
        train_dataset, val_dataset = hand_aware_split(dataset, scenarios, val_split)

        # E.5.1: pre-compute frozen perception outputs once (deterministic with
        # frozen weights — avoids re-running encoder+decoder every epoch)
        log("Pre-computing frozen perception outputs...")
        _p_outs = []
        _p_masks = []
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(dataset), batch_size),
                              total=n_batches, desc="Caching perception", smoothing=0):
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

    log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    if val_every:
        log(f"Validation every {val_every} steps")
    if interrupt_after_fails:
        log(f"Early stopping after {interrupt_after_fails} failed validations")

    # Linear warmup + cosine annealing scheduler
    total_steps = epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 5)
    eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    ckpt_dir = run_dir

    best_val_loss = float("inf")
    fails_since_best = 0
    global_step = 0
    start_epoch = 0

    hist = IncrementalHistory(run_dir,
                              keys=["step_loss", "val_loss", "val_accuracy",
                                    "val_wrc", "epoch_train_loss",
                                    "epoch_val_loss"])
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

        # --- Training ---
        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (cached_p, cached_m, target_probs) in enumerate(
                tqdm(train_loader, desc=f"GTO probs epoch {epoch+1}/{epochs}", leave=False, smoothing=0)):
            cached_p = cached_p.to(device)
            cached_m = cached_m.to(device)
            target_probs = target_probs.to(device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                action_logits = agent.action_head(cached_p, mask=cached_m)
                batch_loss = _kl_loss(action_logits, target_probs)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step_loss = batch_loss.item()
            history["step_loss"].append((global_step, step_loss))
            global_step += 1

            train_loss_sum += step_loss * cached_p.shape[0]
            train_count += cached_p.shape[0]

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Train Loss: {avg:.6f}, LR: {cur_lr:.2e}")

            # Intra-epoch validation
            if val_every and (global_step % val_every == 0):
                val_loss, val_acc, val_wrc = _run_validation(agent, val_loader, device, amp_config=amp_cfg)
                history["val_loss"].append((global_step, val_loss))
                history["val_accuracy"].append((global_step, val_acc))
                history["val_wrc"].append((global_step, val_wrc))
                _save_history(hist)
                log(f"  [Step {global_step}] Val Loss: {val_loss:.6f}, Acc: {val_acc:.4f}, WRC: {val_wrc:.4f}")

                prev_best = best_val_loss
                best_val_loss, fails_since_best, should_stop = _check_val(
                    val_loss, best_val_loss, fails_since_best, interrupt_after_fails, log
                )
                if val_loss < prev_best:
                    _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
                               global_step, epoch, val_loss, log, temperature=temperature)

                if should_stop:
                    stopped_early = True
                    break

        train_loss_avg = train_loss_sum / max(train_count, 1)

        if stopped_early:
            break

        # --- End-of-epoch validation ---
        val_loss_avg, val_acc, val_wrc = _run_validation(agent, val_loader, device, amp_config=amp_cfg)
        history["val_loss"].append((global_step, val_loss_avg))
        history["val_accuracy"].append((global_step, val_acc))
        history["val_wrc"].append((global_step, val_wrc))

        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss_avg)

        _save_history(hist)
        log(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}, Acc: {val_acc:.4f}, WRC: {val_wrc:.4f}")

        prev_best = best_val_loss
        best_val_loss, fails_since_best, should_stop = _check_val(
            val_loss_avg, best_val_loss, fails_since_best, interrupt_after_fails, log
        )
        if val_loss_avg < prev_best:
            _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
                       global_step, epoch, val_loss_avg, log, temperature=temperature)

        _save_latest(agent, optimizer, scheduler, norm_stats, run_dir,
                     next_epoch=epoch + 1, global_step=global_step,
                     best_val_loss=best_val_loss,
                     fails_since_best=fails_since_best,
                     val_loss=val_loss_avg, temperature=temperature)

        if should_stop:
            break

    # Unfreeze all modules after training
    for param in agent.perception.parameters():
        param.requires_grad = True
    for param in agent.value_head.parameters():
        param.requires_grad = True

    log(f"=== GTO Action Prob Training Complete. Best Val Loss: {best_val_loss:.6f} ===")
    hist.compact()

    return history, run_dir
