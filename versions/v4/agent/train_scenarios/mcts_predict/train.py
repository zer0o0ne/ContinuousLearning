"""
Training loop for MCTS-derived data.

Trains all heads simultaneously through the modelling chain:
- value_head: SmoothL1 on root Q (with backed-up terminal equity)
- action_head: KL div on root N-distribution + chain steps where is_hero=True
- opponent_action_head: KL div on chain steps where is_hero=False
- modelling_head: gradients flow back through the chain from all predictions
- perception: unfrozen, gradients from everything
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from agent.train_scenarios.mcts_predict.dataset import MCTSDataset, batch_collate


class LengthGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        lengths = [len(dataset[i][0]) for i in range(len(dataset))]
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        self.batches = [sorted_indices[i:i + batch_size]
                        for i in range(0, len(sorted_indices), batch_size)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.shuffle(order)
        for idx in order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


def _mcts_forward(agent, event_sequences, chains, device, opponent_emb_table=None):
    """Forward pass: perception → root predictions → modelling chain.

    Processes one batch. Chains may have different lengths across examples,
    so we process them per-example (batch_size=1 for chain steps).

    Returns:
        value_preds: (B, 1)
        action_preds: (B, n_actions)
        chain_preds: list of length B, each is a list of (n_actions,) tensors
        chain_targets: list of length B, each is a list of (n_actions,) tensors
        chain_is_hero: list of length B, each is a list of bools
    """
    # Perception
    skip_opp = (opponent_emb_table is None)
    perception_out, _, mask = agent.perception.forward_batch(
        event_sequences, device=device, skip_memory=True,
        skip_opponent_emb=skip_opp, opponent_emb_table=opponent_emb_table)

    # Root predictions
    value_preds = agent.value_head(perception_out, mask=mask)
    action_preds = agent.action_head(perception_out, mask=mask)

    B = perception_out.shape[0]
    all_chain_preds = []
    all_chain_targets = []
    all_chain_is_hero = []

    for b in range(B):
        chain = chains[b]
        if not chain:
            all_chain_preds.append([])
            all_chain_targets.append([])
            all_chain_is_hero.append([])
            continue

        # Per-example context: (1, seq_len, d_model)
        ctx = perception_out[b:b+1]
        ctx_mask = mask[b:b+1]
        preds = []
        targets = []
        is_hero_list = []

        for step in chain:
            # Modelling: get action embeddings for current context
            action_embs = agent.modelling_head(ctx, mask=ctx_mask)

            # Select embedding for the action actually taken
            emb = action_embs[:, step.action_taken, :]  # (1, d_model)

            # Extend context
            ctx = torch.cat([ctx, emb.unsqueeze(1)], dim=1)
            ctx_mask = torch.cat([
                ctx_mask,
                torch.ones(1, 1, dtype=ctx_mask.dtype, device=device)
            ], dim=1)

            # Predict distribution
            if step.is_hero:
                pred = agent.action_head(ctx, mask=ctx_mask)
            else:
                pred = agent.opponent_action_head(ctx, mask=ctx_mask)

            preds.append(pred.squeeze(0))  # (n_actions,)
            targets.append(torch.tensor(
                step.target_distribution, dtype=torch.float32, device=device))
            is_hero_list.append(step.is_hero)

        all_chain_preds.append(preds)
        all_chain_targets.append(targets)
        all_chain_is_hero.append(is_hero_list)

    return value_preds, action_preds, all_chain_preds, all_chain_targets, all_chain_is_hero


def _compute_loss(value_preds, value_targets, action_preds, action_targets,
                  chain_preds, chain_targets, chain_is_hero,
                  value_weight=1.0, action_weight=1.0, chain_weight=1.0):
    """Compute combined loss across all heads.

    Returns:
        total_loss, loss_dict (for logging)
    """
    # Value loss: SmoothL1
    value_loss = F.smooth_l1_loss(value_preds.squeeze(-1), value_targets)

    # Action loss: KL divergence (root action distribution)
    action_log_probs = F.log_softmax(action_preds, dim=-1)
    action_loss = F.kl_div(action_log_probs, action_targets, reduction="batchmean")

    # Chain loss: KL divergence at each step
    chain_losses = []
    for b_preds, b_targets in zip(chain_preds, chain_targets):
        for pred, target in zip(b_preds, b_targets):
            log_probs = F.log_softmax(pred, dim=-1)
            chain_losses.append(F.kl_div(log_probs, target, reduction="sum"))

    if chain_losses:
        chain_loss = torch.stack(chain_losses).sum() / len(chain_losses)
    else:
        chain_loss = torch.tensor(0.0, device=value_preds.device)

    total = value_weight * value_loss + action_weight * action_loss + chain_weight * chain_loss

    return total, {
        "value": value_loss.item(),
        "action": action_loss.item(),
        "chain": chain_loss.item(),
        "total": total.item(),
    }


def _run_validation(agent, val_loader, device, weights, amp_config=None,
                    opponent_emb_table=None):
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    agent.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for event_seqs, val_targets, act_targets, chains in val_loader:
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                v_pred, a_pred, c_preds, c_tgts, c_hero = _mcts_forward(
                    agent, event_seqs, chains, device,
                    opponent_emb_table=opponent_emb_table)
                loss, _ = _compute_loss(v_pred, val_targets, a_pred, act_targets,
                                        c_preds, c_tgts, c_hero, **weights)
            loss_sum += loss.item() * len(event_seqs)
            count += len(event_seqs)
    agent.train()
    return loss_sum / max(count, 1)


def train_mcts(agent, train_cfg, device, log, examples):
    """Train all agent heads on MCTS-derived training data.

    All heads are unfrozen and trained simultaneously:
    - perception: gradients from modelling chain
    - value_head: SmoothL1 on root Q
    - action_head: KL on root + chain hero steps
    - opponent_action_head: KL on chain opponent steps
    - modelling_head: gradients from chain predictions

    Args:
        agent: ASI model instance (on device)
        train_cfg: dict with lr, batch_size, epochs, etc.
        device: torch device string
        log: logger callable
        examples: list of MCTSTrainingExample
    """
    lr = train_cfg.get("lr", 3e-5)
    batch_size = train_cfg.get("batch_size", 16)
    epochs = train_cfg.get("epochs", 5)
    val_split = train_cfg.get("val_split", 0.1)
    log_every = train_cfg.get("log_every", 10)
    val_every = train_cfg.get("val_every", None)
    interrupt_after_fails = train_cfg.get("interrupt_after_fails", None)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    value_weight = train_cfg.get("value_weight", 1.0)
    action_weight = train_cfg.get("action_weight", 1.0)
    chain_weight = train_cfg.get("chain_weight", 1.0)
    weights = {"value_weight": value_weight, "action_weight": action_weight,
               "chain_weight": chain_weight}

    log("=== MCTS Training (All Heads) ===")

    # All parameters trainable
    for param in agent.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    run_dir = log.run_dir("mcts_predict")

    # AMP
    from utils import get_amp_config
    amp_enabled, device_type, amp_dtype, use_scaler = get_amp_config(device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    amp_cfg = (amp_enabled, device_type, amp_dtype)

    # Opponent embedding table
    opp_table = None
    if agent.perception.opp_emb_enabled:
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        opp_table = OpponentEmbeddingTable(agent.perception.d_model)
        log("Opponent GRU embedding enabled")

    # Dataset
    dataset = MCTSDataset(examples)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_sampler = LengthGroupedBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=batch_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=batch_collate)

    log(f"Train: {n_train}, Val: {n_val}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    log(f"Weights: value={value_weight}, action={action_weight}, chain={chain_weight}")

    # Scheduler
    total_steps = epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 5)
    eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                               eta_min=eta_min)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    best_val_loss = float("inf")
    fails_since_best = 0
    history = {"step_loss": [], "val_loss": [], "epoch_train_loss": [], "epoch_val_loss": []}
    global_step = 0
    stopped_early = False

    for epoch in range(epochs):
        if stopped_early:
            break

        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (event_seqs, val_targets, act_targets, chains) in enumerate(train_loader):
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                v_pred, a_pred, c_preds, c_tgts, c_hero = _mcts_forward(
                    agent, event_seqs, chains, device,
                    opponent_emb_table=opp_table)
                loss, loss_dict = _compute_loss(
                    v_pred, val_targets, a_pred, act_targets,
                    c_preds, c_tgts, c_hero, **weights)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if opp_table is not None:
                opp_table.detach_all()

            step_loss = loss.item()
            history["step_loss"].append((global_step, step_loss))
            global_step += 1
            train_loss_sum += step_loss * len(event_seqs)
            train_count += len(event_seqs)

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                log(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Loss: {avg:.6f} (v={loss_dict['value']:.4f} "
                    f"a={loss_dict['action']:.4f} c={loss_dict['chain']:.4f})")

            if val_every and (global_step % val_every == 0):
                vl = _run_validation(agent, val_loader, device, weights, amp_cfg,
                                     opponent_emb_table=opp_table)
                history["val_loss"].append((global_step, vl))
                log(f"  [Step {global_step}] Val Loss: {vl:.6f}")
                if vl < best_val_loss:
                    best_val_loss = vl
                    fails_since_best = 0
                    _save_best(agent, optimizer, scheduler, run_dir, global_step,
                               epoch, vl, log)
                else:
                    fails_since_best += 1
                    if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
                        log(f"  Early stopping: {fails_since_best} failed validations")
                        stopped_early = True
                        break

        if stopped_early:
            break

        train_avg = train_loss_sum / max(train_count, 1)
        val_avg = _run_validation(agent, val_loader, device, weights, amp_cfg,
                                   opponent_emb_table=opp_table)
        history["epoch_train_loss"].append(train_avg)
        history["epoch_val_loss"].append(val_avg)
        history["val_loss"].append((global_step, val_avg))
        log(f"Epoch {epoch+1}/{epochs} — Train: {train_avg:.6f}, Val: {val_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            fails_since_best = 0
            _save_best(agent, optimizer, scheduler, run_dir, global_step,
                       epoch, val_avg, log)
        else:
            fails_since_best += 1
            if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
                log(f"  Early stopping: {fails_since_best} failed validations")
                break

    torch.save(history, os.path.join(run_dir, "history.pt"))
    log(f"=== MCTS Training Complete. Best Val Loss: {best_val_loss:.6f} ===")
    return history, run_dir


def _save_best(agent, optimizer, scheduler, run_dir, step, epoch, val_loss, log):
    best_path = os.path.join(run_dir, "best.pt")
    torch.save({
        "step": step, "epoch": epoch + 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, best_path)
    log(f"  New best model (val loss: {val_loss:.6f})")
