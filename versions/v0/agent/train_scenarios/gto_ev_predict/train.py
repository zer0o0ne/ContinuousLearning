"""
Training loop for GTO EV prediction (Recipe Step 1).

Trains perception + value_head to predict GTO EV of poker situations.
Memory is active: each encoded state is stored and retrieved via beam search.
Action head is frozen. Memory resets each epoch.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.train_scenarios.gto_ev_predict.generate import generate_dataset
from agent.train_scenarios.gto_ev_predict.dataset import GTOEVDataset, batch_collate


def train_gto_ev(agent, train_cfg, device, log):
    """Main training entry point for GTO EV prediction.

    Args:
        agent: ASI model instance (already on device)
        train_cfg: dict with training hyperparameters from config["gto_ev_train"]
        device: torch device string
        log: logger callable
    """
    lr = train_cfg.get("lr", 3e-4)
    batch_size = train_cfg.get("batch_size", 64)
    epochs = train_cfg.get("epochs", 50)
    val_split = train_cfg.get("val_split", 0.1)
    log_every = train_cfg.get("log_every", 100)
    checkpoint_every = train_cfg.get("checkpoint_every", 5)

    log("=== GTO EV Prediction Training (Step 1) ===")

    # Freeze action head — only train perception + value
    for param in agent.action_head.parameters():
        param.requires_grad = False

    # Optimizer over trainable parameters only
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.MSELoss()

    # Run directory: data/<version>/<name>/gto_ev_predict/<time>/
    run_dir = log.run_dir("gto_ev_predict")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # Load dataset from existing dir or generate new one
    dataset_dir = train_cfg.get("dataset_dir", None)
    if dataset_dir:
        dataset_path = os.path.join(dataset_dir, "dataset.pt")
        stats_path = os.path.join(dataset_dir, "norm_stats.pt")
        if not os.path.exists(dataset_path) or not os.path.exists(stats_path):
            log(f"Dataset not found at {dataset_dir}. Aborting.")
            return
        scenarios = torch.load(dataset_path, weights_only=False)
        norm_stats = torch.load(stats_path, weights_only=False)
        log(f"Loaded dataset from {dataset_dir} ({len(scenarios)} scenarios)")
    else:
        result = generate_dataset(train_cfg, run_dir, log=log)
        if not result or not result[0]:
            log("No scenarios generated. Aborting training.")
            return
        scenarios, norm_stats = result

    # Train/val split
    dataset = GTOEVDataset(scenarios)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate)

    log(f"Train: {train_size}, Val: {val_size}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")

    # Cosine annealing scheduler
    total_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Checkpoints saved inside run_dir
    ckpt_dir = run_dir

    best_val_loss = float("inf")
    history = {"step_loss": [], "epoch_train_loss": [], "epoch_val_loss": []}
    global_step = 0

    for epoch in range(epochs):
        # --- Training ---
        agent.perception.memory.clear()
        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (env_states, ev_targets) in enumerate(train_loader):
            ev_targets = ev_targets.to(device)

            result = agent.forward_batch(env_states, skip_memory=False, store=True)
            predicted_ev = result["value"].squeeze(-1)  # (B,)
            batch_loss = loss_fn(predicted_ev, ev_targets)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()

            step_loss = batch_loss.item()
            history["step_loss"].append((global_step, step_loss))
            global_step += 1

            train_loss_sum += step_loss * len(env_states)
            train_count += len(env_states)

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Train Loss: {avg:.6f}, LR: {cur_lr:.2e}")

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # --- Validation ---
        agent.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for env_states, ev_targets in val_loader:
                ev_targets = ev_targets.to(device)
                result = agent.forward_batch(env_states, skip_memory=False, store=False)
                predicted_ev = result["value"].squeeze(-1)
                batch_loss = loss_fn(predicted_ev, ev_targets)
                val_loss_sum += batch_loss.item() * len(env_states)
                val_count += len(env_states)

        val_loss_avg = val_loss_sum / max(val_count, 1)

        history["epoch_train_loss"].append(train_loss_avg)
        history["epoch_val_loss"].append(val_loss_avg)

        log(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}")

        # Checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({"epoch": epoch + 1, "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "memory_state": agent.perception.memory.state_dict(),
                        "norm_stats": norm_stats,
                        "train_loss": train_loss_avg, "val_loss": val_loss_avg}, ckpt_path)
            log(f"  Checkpoint saved: {ckpt_path}")

        # Best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({"epoch": epoch + 1, "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "memory_state": agent.perception.memory.state_dict(),
                        "norm_stats": norm_stats,
                        "train_loss": train_loss_avg, "val_loss": val_loss_avg}, best_path)
            log(f"  New best model (val loss: {val_loss_avg:.6f})")

    # Unfreeze action head after training
    for param in agent.action_head.parameters():
        param.requires_grad = True

    log(f"=== GTO EV Training Complete. Best Val Loss: {best_val_loss:.6f} ===")

    # Save history
    history_path = os.path.join(run_dir, "history.pt")
    torch.save(history, history_path)

    return history
