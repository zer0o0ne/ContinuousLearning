"""
Training loop for modelling head (Recipe Step 3).

Trains modelling_head to produce per-action state embeddings such that
the frozen value_head can predict each action's EV from them.

Perception, value_head, and action_head are all frozen.
Gradients flow through value_head (params frozen, graph intact) back to modelling_head.
"""

import os
import random
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from agent.train_scenarios.generation.generate import generate_dataset, load_dataset, \
    _compute_norm_stats, _normalize_scenarios
from agent.train_scenarios.modelling_predict.dataset import GTOModellingDataset, batch_collate


class LengthGroupedBatchSampler(Sampler):
    """Sampler that groups samples by sequence length into batches.

    Sorts by n_events, chunks into batches, shuffles batch order each epoch.
    """

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        indices = list(range(len(dataset)))
        lengths = []
        for i in indices:
            sample = dataset[i]
            lengths.append(len(sample[0]))
        sorted_indices = sorted(indices, key=lambda i: lengths[i])
        self.batches = [sorted_indices[i:i + batch_size]
                        for i in range(0, len(sorted_indices), batch_size)]

    def __iter__(self):
        batch_order = list(range(len(self.batches)))
        random.shuffle(batch_order)
        for idx in batch_order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


def _compress_targets(targets):
    """Compress extreme negative targets: values < -1 mapped to -1 + (target+1)*0.03."""
    return torch.where(targets >= -1, targets, -1 + (targets + 1) * 0.03)


def _modelling_forward(agent, event_sequences, device):
    """Run the modelling forward pass: perception → modelling_head → value_head.

    Returns:
        predicted_evs: (B, K) — predicted EV for each action
    """
    # Perception is frozen — no_grad + detach
    with torch.no_grad():
        perception_out, encoded, mask = agent.perception.forward_batch(
            event_sequences, device=device, skip_memory=True
        )
    perception_out = perception_out.detach()

    # Modelling head — TRAINABLE, produces per-action embeddings
    action_embs = agent.modelling_head(perception_out, mask=mask)  # (B, K, D)

    B, N, D = perception_out.shape
    K = agent.n_actions

    # Expand perception_out: (B, N, D) → (B*K, N, D)
    p_expanded = perception_out.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)

    # Reshape action embeddings: (B, K, D) → (B*K, 1, D)
    a_flat = action_embs.reshape(B * K, 1, D)

    # Concat: (B*K, N+1, D)
    combined = torch.cat([p_expanded, a_flat], dim=1)

    # Extend mask: (B, N) → (B*K, N+1) with 1 appended for the action token
    mask_expanded = mask.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
    mask_combined = torch.cat([mask_expanded,
                               torch.ones(B * K, 1, device=device)], dim=1)

    # Value head — frozen params, but gradients flow through to action_embs
    values = agent.value_head(combined, mask=mask_combined)  # (B*K, 1)
    return values.squeeze(-1).reshape(B, K)  # (B, K)


def _run_validation(agent, val_loader, loss_fn, device, amp_config=None):
    """Run validation and return average loss."""
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    agent.eval()
    val_loss_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for event_sequences, action_evs in val_loader:
            action_evs = _compress_targets(action_evs.to(device))
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                predicted_evs = _modelling_forward(agent, event_sequences, device)
                batch_loss = loss_fn(predicted_evs, action_evs)
            val_loss_sum += batch_loss.item() * len(event_sequences)
            val_count += len(event_sequences)
    agent.train()
    return val_loss_sum / max(val_count, 1)


def _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
               global_step, epoch, val_loss, log, temperature=None):
    """Save best model checkpoint."""
    best_path = os.path.join(ckpt_dir, "best.pt")
    ckpt = {
        "step": global_step, "epoch": epoch + 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_stats": norm_stats,
        "val_loss": val_loss,
    }
    if temperature is not None:
        ckpt["temperature"] = temperature
    torch.save(ckpt, best_path)
    log(f"  New best model (val loss: {val_loss:.6f})")


def _save_history(history, run_dir):
    """Save training history incrementally for real-time monitoring."""
    torch.save(history, os.path.join(run_dir, "history.pt"))


def _check_val(val_loss, best_val_loss, fails_since_best, interrupt_after_fails, log):
    """Check validation result. Returns (new_best_val_loss, new_fails_count, should_stop)."""
    if val_loss < best_val_loss:
        return val_loss, 0, False

    fails_since_best += 1
    if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
        log(f"  Early stopping: {fails_since_best} validations without improvement")
        return best_val_loss, fails_since_best, True

    return best_val_loss, fails_since_best, False


def _normalize_action_evs(scenarios, norm_stats):
    """Normalize action_evs identically to ev_target: scale by (pot+facing_bet), then z-score."""
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    for s in scenarios:
        denom = max(s.get("pot", 0) + s.get("facing_bet", 0),
                    s["events"][-1]["big_blind"])
        evs = s["action_evs"]
        if isinstance(evs, np.ndarray):
            s["action_evs"] = (evs / denom - ev_m) / ev_s
        else:
            s["action_evs"] = [(ev / denom - ev_m) / ev_s for ev in evs]


def train_modelling(agent, train_cfg, device, log, scenarios_override=None, temperature=None):
    """Main training entry point for modelling head.

    Args:
        agent: ASI model instance (already on device)
        train_cfg: dict with training hyperparameters from config["modelling_train"]
        device: torch device string
        log: logger callable
        scenarios_override: if provided, use these raw scenarios instead of loading/generating
        temperature: effective temperature for this agent (saved in checkpoint)
    """
    lr = train_cfg.get("lr", 1e-4)
    batch_size = train_cfg.get("batch_size", 64)
    epochs = train_cfg.get("epochs", 10)
    val_split = train_cfg.get("val_split", 0.1)
    log_every = train_cfg.get("log_every", 10)
    val_every = train_cfg.get("val_every", None)
    interrupt_after_fails = train_cfg.get("interrupt_after_fails", None)

    log("=== Modelling Head Training (Step 3) ===")
    log("Memory: DISABLED (skip_memory=True)")

    # Freeze everything except modelling_head
    for param in agent.perception.parameters():
        param.requires_grad = False
    for param in agent.value_head.parameters():
        param.requires_grad = False
    for param in agent.action_head.parameters():
        param.requires_grad = False
    for param in agent.modelling_head.parameters():
        param.requires_grad = True

    # Optimizer over modelling_head parameters only
    trainable_params = list(agent.modelling_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # Run directory
    run_dir = log.run_dir("modelling_predict")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # AMP setup
    from utils import get_amp_config
    amp_enabled, device_type, amp_dtype, use_scaler = get_amp_config(device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    amp_cfg = (amp_enabled, device_type, amp_dtype)
    if amp_enabled:
        log(f"AMP enabled: {device_type}, dtype={amp_dtype}, scaler={use_scaler}")

    # Dataset
    if scenarios_override is not None:
        scenarios = scenarios_override
        log(f"Using provided scenarios ({len(scenarios)} samples)")
    else:
        scenarios = generate_dataset(train_cfg, run_dir, log=log)
        if not scenarios:
            log("No scenarios generated. Aborting training.")
            return None, run_dir

    # Normalize events (same as gto_ev), then normalize action_evs with same stats
    scenarios = copy.deepcopy(scenarios)
    norm_stats = _compute_norm_stats(scenarios)
    log(f"Norm stats: " + ", ".join(f"{k}={v:.4f}" for k, v in norm_stats.items()))
    _normalize_action_evs(scenarios, norm_stats)  # BEFORE _normalize_scenarios (uses raw big_blind)
    _normalize_scenarios(scenarios, norm_stats)

    # Train/val split
    dataset = GTOModellingDataset(scenarios)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_sampler = LengthGroupedBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=batch_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate)

    log(f"Train: {train_size}, Val: {val_size}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
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
    history = {"step_loss": [], "val_loss": [], "epoch_train_loss": [], "epoch_val_loss": []}
    global_step = 0
    stopped_early = False

    for epoch in range(epochs):
        if stopped_early:
            break

        # --- Training ---
        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (event_sequences, action_evs) in enumerate(train_loader):
            action_evs = _compress_targets(action_evs.to(device))

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                predicted_evs = _modelling_forward(agent, event_sequences, device)
                batch_loss = loss_fn(predicted_evs, action_evs)

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

            train_loss_sum += step_loss * len(event_sequences)
            train_count += len(event_sequences)

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Train Loss: {avg:.6f}, LR: {cur_lr:.2e}")

            # Intra-epoch validation
            if val_every and (global_step % val_every == 0):
                val_loss = _run_validation(agent, val_loader, loss_fn, device, amp_config=amp_cfg)
                history["val_loss"].append((global_step, val_loss))
                _save_history(history, run_dir)
                log(f"  [Step {global_step}] Val Loss: {val_loss:.6f}")

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
        val_loss_avg = _run_validation(agent, val_loader, loss_fn, device, amp_config=amp_cfg)
        history["val_loss"].append((global_step, val_loss_avg))

        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss_avg)

        _save_history(history, run_dir)
        log(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}")

        prev_best = best_val_loss
        best_val_loss, fails_since_best, should_stop = _check_val(
            val_loss_avg, best_val_loss, fails_since_best, interrupt_after_fails, log
        )
        if val_loss_avg < prev_best:
            _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
                       global_step, epoch, val_loss_avg, log, temperature=temperature)

        if should_stop:
            break

    # Unfreeze all after training
    for param in agent.perception.parameters():
        param.requires_grad = True
    for param in agent.value_head.parameters():
        param.requires_grad = True
    for param in agent.action_head.parameters():
        param.requires_grad = True

    log(f"=== Modelling Training Complete. Best Val Loss: {best_val_loss:.6f} ===")
    _save_history(history, run_dir)

    return history, run_dir
