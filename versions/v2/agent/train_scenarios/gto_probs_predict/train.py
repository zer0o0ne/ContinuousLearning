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
from torch.utils.data import DataLoader, random_split, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from agent.train_scenarios.generation.generate import generate_dataset, load_dataset, \
    _compute_norm_stats, _normalize_scenarios
from agent.train_scenarios.gto_probs_predict.dataset import GTOProbsDataset, batch_collate


class LengthGroupedBatchSampler(Sampler):
    """Sampler that groups samples by sequence length into batches."""

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


def _run_validation(agent, val_loader, device):
    """Run validation and return (avg_loss, accuracy, wrc)."""
    agent.eval()
    val_loss_sum = 0.0
    correct = 0
    wrc_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for event_sequences, target_probs in val_loader:
            target_probs = target_probs.to(device)
            result = agent.forward_batch(event_sequences, skip_memory=True)
            action_logits = result["action_logits"]
            batch_loss = _kl_loss(action_logits, target_probs)
            val_loss_sum += batch_loss.item() * len(event_sequences)
            correct += (action_logits.argmax(dim=-1) == target_probs.argmax(dim=-1)).sum().item()
            wrc_sum += _weighted_rank_concordance(action_logits, target_probs).sum().item()
            val_count += len(event_sequences)
    agent.train()
    n = max(val_count, 1)
    return val_loss_sum / n, correct / n, wrc_sum / n


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


def train_gto_probs(agent, train_cfg, device, log, scenarios_override=None, temperature=None):
    """Main training entry point for GTO action probability prediction.

    Freezes perception + value_head, trains only action_head with KL divergence.

    Args:
        agent: ASI model instance (already on device)
        train_cfg: dict with training hyperparameters (merged game + solver + gto_probs_train)
        device: torch device string
        log: logger callable
        scenarios_override: if provided, use these raw scenarios instead of loading/generating
        temperature: effective temperature for this agent (saved in checkpoint)
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

    # Run directory
    run_dir = log.run_dir("gto_probs_predict")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # Dataset is provided by pipeline (loaded/generated there)
    if scenarios_override is not None:
        scenarios = scenarios_override
        log(f"Using provided scenarios ({len(scenarios)} samples)")
    else:
        scenarios = generate_dataset(train_cfg, run_dir, log=log)
        if not scenarios:
            log("No scenarios generated. Aborting training.")
            return

    # Compute norm_stats and normalize (per-agent)
    import copy
    scenarios = copy.deepcopy(scenarios)
    norm_stats = _compute_norm_stats(scenarios)
    log(f"Norm stats: " + ", ".join(f"{k}={v:.4f}" for k, v in norm_stats.items()))
    _normalize_scenarios(scenarios, norm_stats)

    # Train/val split
    dataset = GTOProbsDataset(scenarios)
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
    history = {"step_loss": [], "val_loss": [], "val_accuracy": [], "val_wrc": [], "epoch_train_loss": [], "epoch_val_loss": []}
    global_step = 0
    stopped_early = False

    for epoch in range(epochs):
        if stopped_early:
            break

        # --- Training ---
        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (event_sequences, target_probs) in enumerate(train_loader):
            target_probs = target_probs.to(device)

            result = agent.forward_batch(event_sequences, skip_memory=True)
            action_logits = result["action_logits"]  # (B, n_actions)
            batch_loss = _kl_loss(action_logits, target_probs)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
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
                val_loss, val_acc, val_wrc = _run_validation(agent, val_loader, device)
                history["val_loss"].append((global_step, val_loss))
                history["val_accuracy"].append((global_step, val_acc))
                history["val_wrc"].append((global_step, val_wrc))
                _save_history(history, run_dir)
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
        val_loss_avg, val_acc, val_wrc = _run_validation(agent, val_loader, device)
        history["val_loss"].append((global_step, val_loss_avg))
        history["val_accuracy"].append((global_step, val_acc))
        history["val_wrc"].append((global_step, val_wrc))

        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss_avg)

        _save_history(history, run_dir)
        log(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}, Acc: {val_acc:.4f}, WRC: {val_wrc:.4f}")

        prev_best = best_val_loss
        best_val_loss, fails_since_best, should_stop = _check_val(
            val_loss_avg, best_val_loss, fails_since_best, interrupt_after_fails, log
        )
        if val_loss_avg < prev_best:
            _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
                       global_step, epoch, val_loss_avg, log, temperature=temperature)

        if should_stop:
            break

    # Unfreeze all modules after training
    for param in agent.perception.parameters():
        param.requires_grad = True
    for param in agent.value_head.parameters():
        param.requires_grad = True

    log(f"=== GTO Action Prob Training Complete. Best Val Loss: {best_val_loss:.6f} ===")
    _save_history(history, run_dir)

    return history
