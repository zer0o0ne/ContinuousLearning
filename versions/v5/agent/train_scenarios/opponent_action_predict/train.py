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
from agent.resume import atomic_torch_save


class LengthGroupedBatchSampler(Sampler):
    """Groups samples by sequence length into batches, shuffles batch order."""

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        indices = list(range(len(dataset)))
        lengths = [len(dataset[i][0]) for i in indices]
        sorted_indices = sorted(indices, key=lambda i: lengths[i])
        self.batches = [sorted_indices[i:i + batch_size]
                        for i in range(0, len(sorted_indices), batch_size)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.shuffle(order)
        for idx in order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


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


def _run_validation(agent, val_loader, device, amp_config=None, opponent_emb_table=None):
    """Run validation. Returns (avg_loss, top1_accuracy)."""
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    skip_opp = (opponent_emb_table is None)
    agent.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for event_sequences, target_probs in val_loader:
            target_probs = target_probs.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = agent.forward_batch(event_sequences, skip_memory=True,
                                          heads={"opponent_action"},
                                          skip_opponent_emb=skip_opp,
                                          opponent_emb_table=opponent_emb_table)
                logits = out["opponent_action_logits"]
                batch_loss = _kl_loss(logits, target_probs)
            loss_sum += batch_loss.item() * len(event_sequences)
            correct += (logits.argmax(dim=-1) == target_probs.argmax(dim=-1)).sum().item()
            count += len(event_sequences)
    agent.train()
    n = max(count, 1)
    return loss_sum / n, correct / n


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
    atomic_torch_save(history, os.path.join(run_dir, "history.pt"))


def _save_latest(agent, optimizer, scheduler, norm_stats, run_dir,
                 next_epoch, global_step, best_val_loss, fails_since_best,
                 val_loss, temperature=None):
    """Atomic per-epoch checkpoint for pipeline.resume."""
    ckpt = {
        "model_state_dict":     agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "next_epoch":           next_epoch,
        "global_step":          global_step,
        "best_val_loss":        best_val_loss,
        "fails_since_best":     fails_since_best,
        "norm_stats":           norm_stats,
        "val_loss":             val_loss,
    }
    if temperature is not None:
        ckpt["temperature"] = temperature
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

    train_sampler = LengthGroupedBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=batch_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=batch_collate)

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

    history_path = os.path.join(run_dir, "history.pt")
    if os.path.exists(history_path):
        history = torch.load(history_path, weights_only=False)
        for k in ("step_loss", "val_loss", "val_accuracy",
                  "epoch_train_loss", "epoch_val_loss"):
            history.setdefault(k, [])
        log(f"Resumed history from {history_path} "
            f"(step_loss n={len(history['step_loss'])})")
    else:
        history = {"step_loss": [], "val_loss": [], "val_accuracy": [],
                   "epoch_train_loss": [], "epoch_val_loss": []}

    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        except Exception as e:
            log(f"  Optimizer/scheduler restore failed: {e}. "
                f"Continuing with fresh ones.")
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

        for batch_idx, (event_sequences, target_probs) in enumerate(train_loader):
            target_probs = target_probs.to(device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype,
                                enabled=amp_enabled):
                out = agent.forward_batch(event_sequences, skip_memory=True,
                                          heads={"opponent_action"},
                                          skip_opponent_emb=(opp_table is None),
                                          opponent_emb_table=opp_table)
                logits = out["opponent_action_logits"]
                batch_loss = _kl_loss(logits, target_probs)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if opp_table is not None:
                opp_table.detach_all()

            step_loss = batch_loss.item()
            history["step_loss"].append((global_step, step_loss))
            global_step += 1

            train_loss_sum += step_loss * len(event_sequences)
            train_count += len(event_sequences)

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, "
                    f"Loss: {avg:.6f}, LR: {cur_lr:.2e}")

            # Intra-epoch validation
            if val_every and (global_step % val_every == 0):
                val_loss, val_acc = _run_validation(
                    agent, val_loader, device, amp_config=amp_cfg,
                    opponent_emb_table=opp_table)
                history["val_loss"].append((global_step, val_loss))
                history["val_accuracy"].append((global_step, val_acc))
                _save_history(history, run_dir)
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
            opponent_emb_table=opp_table)
        history["val_loss"].append((global_step, val_loss))
        history["val_accuracy"].append((global_step, val_acc))
        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss)
        _save_history(history, run_dir)

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
    _save_history(history, run_dir)

    return history, run_dir
