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

    Chain semantics (matches `collect.py:MCTSTrainingExample`):
      step.action_taken is the action chosen at the PRIOR state. We extend
      ctx by appending modelling_head(ctx)[step.action_taken] (which encodes
      "next state if action_taken was taken at current ctx") then predict
      the distribution at that extended state and compare with
      step.target_distribution. This mirrors MCTS inner-node expansion
      (mcts.py: child.action_embedding = modelling_head(parent_ctx)[a]).

    Returns:
        value_preds: (B, 1) — predicted at root context only
        action_preds: (B, n_actions) — root predictions
        chain_preds, chain_targets, chain_is_hero: per-batch list of lists
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
    """Run validation. Returns dict with total + component losses."""
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    agent.eval()
    sums = {"total": 0.0, "value": 0.0, "action": 0.0, "chain": 0.0}
    count = 0
    with torch.no_grad():
        for event_seqs, val_targets, act_targets, chains in val_loader:
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                v_pred, a_pred, c_preds, c_tgts, c_hero = _mcts_forward(
                    agent, event_seqs, chains, device,
                    opponent_emb_table=opponent_emb_table)
                loss, ldict = _compute_loss(v_pred, val_targets, a_pred, act_targets,
                                            c_preds, c_tgts, c_hero, **weights)
            n = len(event_seqs)
            sums["total"] += ldict["total"] * n
            sums["value"] += ldict["value"] * n
            sums["action"] += ldict["action"] * n
            sums["chain"] += ldict["chain"] * n
            count += n
    agent.train()
    if count == 0:
        return {"total": 0.0, "value": 0.0, "action": 0.0, "chain": 0.0}
    return {k: v / count for k, v in sums.items()}


def train_mcts(agent, train_cfg, device, log, examples, temperature=None,
               run_dir=None, history_path=None, cycle_id=0,
               global_step_offset=0, save_checkpoint=True,
               run_timestamp=None,
               optimizer=None, scheduler=None):
    """Train all agent heads on MCTS-derived training data.

    Supports cross-cycle continuity for cyclic self-play training:
    - run_dir: pre-existing run dir; if None, creates a new one. Pass the same
        run_dir across cycles so best.pt and history.pt are co-located.
    - history_path: shared history.pt path across cycles. If file exists, its
        content is loaded and new entries are appended (continuous loss curves).
    - cycle_id: outer cycle index, recorded with each step/val entry for
        analytics.
    - global_step_offset: starting step counter (cumulative across cycles).
    - save_checkpoint: when True, the permanent per-cycle snapshot
        `cycles/cycle_<N>.pt` is written this cycle. The rolling `best.pt`
        is overwritten every cycle regardless. History is always persisted.
    - run_timestamp: timestamp string of the pipeline run, attached to cycle
        summary so multi-run analyses can group entries.

    All heads are unfrozen and trained simultaneously:
    - perception: gradients from modelling chain
    - value_head: SmoothL1 on root Q
    - action_head: KL on root + chain hero steps
    - opponent_action_head: KL on chain opponent steps
    - modelling_head: gradients from chain predictions

    Returns:
        history: dict (in-memory, also persisted to history_path)
        run_dir: path used for outputs
        new_global_step: cumulative step counter to feed back into next cycle
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

    log(f"=== MCTS Training (cycle {cycle_id}, save={save_checkpoint}) ===")

    # Preserve norm_stats from checkpoint for saving
    norm_stats = getattr(agent, '_checkpoint_norm_stats', None) or {}

    # All parameters trainable
    for param in agent.parameters():
        param.requires_grad = True

    # External optimizer/scheduler take precedence (option 2: single global
    # instance kept alive across cycles so Adam moments and the long-horizon
    # cosine schedule both survive). Fall back to fresh per-cycle instances
    # only when called without externals (legacy usage and one-shot tests).
    external_optim = optimizer is not None
    if optimizer is None:
        optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    if run_dir is None:
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
    if external_optim:
        cur_lr = optimizer.param_groups[0]["lr"]
        log(f"Using external optimizer/scheduler (current lr={cur_lr:.2e})")

    # Scheduler: legacy per-cycle warmup+cosine only when no external one
    # was supplied. With external schedulers the LR plan spans all cycles.
    if scheduler is None:
        total_steps = epochs * len(train_loader)
        warmup_steps = min(100, total_steps // 5)
        eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                   eta_min=eta_min)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # Cross-cycle history: load existing if present, else fresh
    if history_path is None:
        history_path = os.path.join(run_dir, "history.pt")
    if os.path.exists(history_path):
        history = torch.load(history_path, weights_only=False)
        for k in ("step_loss", "val_loss", "epoch_train_loss",
                  "epoch_val_loss", "cycles"):
            history.setdefault(k, [])
    else:
        history = {"step_loss": [], "val_loss": [], "epoch_train_loss": [],
                   "epoch_val_loss": [], "cycles": []}

    best_val_loss = float("inf")
    fails_since_best = 0
    global_step = int(global_step_offset)
    cycle_step_start = global_step
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
            current_lr = optimizer.param_groups[0]["lr"]
            history["step_loss"].append({
                "step": global_step,
                "cycle_id": cycle_id,
                "total": step_loss,
                "value": loss_dict["value"],
                "action": loss_dict["action"],
                "chain": loss_dict["chain"],
                "lr": current_lr,
                "batch_size": len(event_seqs),
            })
            global_step += 1
            train_loss_sum += step_loss * len(event_seqs)
            train_count += len(event_seqs)

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                log(f"  Cycle {cycle_id} Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Loss: {avg:.6f} (v={loss_dict['value']:.4f} "
                    f"a={loss_dict['action']:.4f} c={loss_dict['chain']:.4f}) "
                    f"lr={current_lr:.2e}")

            if val_every and (global_step % val_every == 0):
                val_dict = _run_validation(agent, val_loader, device, weights,
                                            amp_cfg, opponent_emb_table=opp_table)
                vl = val_dict["total"]
                history["val_loss"].append({
                    "step": global_step,
                    "cycle_id": cycle_id,
                    **val_dict,
                })
                log(f"  [Cycle {cycle_id} Step {global_step}] Val: "
                    f"total={vl:.6f} v={val_dict['value']:.4f} "
                    f"a={val_dict['action']:.4f} c={val_dict['chain']:.4f}")
                if vl < best_val_loss:
                    best_val_loss = vl
                    fails_since_best = 0
                    _save_best(agent, optimizer, scheduler, norm_stats,
                               run_dir, global_step, epoch, vl, log,
                               temperature=temperature, cycle_id=cycle_id,
                               examples_in_cycle=len(examples),
                               write_snapshot=save_checkpoint)
                else:
                    fails_since_best += 1
                    if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
                        log(f"  Early stopping: {fails_since_best} failed validations")
                        stopped_early = True
                        break

        if stopped_early:
            break

        train_avg = train_loss_sum / max(train_count, 1)
        val_dict = _run_validation(agent, val_loader, device, weights, amp_cfg,
                                    opponent_emb_table=opp_table)
        val_avg = val_dict["total"]
        history["epoch_train_loss"].append({
            "step": global_step, "cycle_id": cycle_id,
            "total": train_avg, "epoch": epoch + 1,
        })
        history["epoch_val_loss"].append({
            "step": global_step, "cycle_id": cycle_id,
            "epoch": epoch + 1, **val_dict,
        })
        history["val_loss"].append({
            "step": global_step, "cycle_id": cycle_id, **val_dict,
        })
        log(f"Cycle {cycle_id} Epoch {epoch+1}/{epochs} — "
            f"Train: {train_avg:.6f}, Val: {val_avg:.6f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            fails_since_best = 0
            _save_best(agent, optimizer, scheduler, norm_stats, run_dir,
                       global_step, epoch, val_avg, log,
                       temperature=temperature, cycle_id=cycle_id,
                       examples_in_cycle=len(examples),
                       write_snapshot=save_checkpoint)
        else:
            fails_since_best += 1
            if interrupt_after_fails and fails_since_best >= interrupt_after_fails:
                log(f"  Early stopping: {fails_since_best} failed validations")
                break

    # best.pt is always rolled forward (latest-after-cycle), but the
    # permanent per-cycle snapshot under cycles/ is gated by
    # save_every_cycles via save_checkpoint.
    final_val = float(val_avg) if 'val_avg' in locals() else float(best_val_loss)
    final_epoch = epoch if 'epoch' in locals() else 0
    _save_best(agent, optimizer, scheduler, norm_stats, run_dir,
               global_step, final_epoch, final_val, log,
               temperature=temperature, cycle_id=cycle_id,
               examples_in_cycle=len(examples), reason="cycle_end",
               write_snapshot=save_checkpoint)

    # Cycle summary for analytics
    history["cycles"].append({
        "cycle_id": cycle_id,
        "run_timestamp": run_timestamp or getattr(log, "init_time", None),
        "examples_count": len(examples),
        "train_size": n_train,
        "val_size": n_val,
        "step_start": cycle_step_start,
        "step_end": global_step,
        "best_val_loss_in_cycle": (best_val_loss if best_val_loss != float("inf") else None),
        "saved_checkpoint": True,
        "lr_start": train_cfg.get("lr", 3e-5),
        "epochs_run": epochs,
    })

    torch.save(history, history_path)
    log(f"=== MCTS Cycle {cycle_id} Complete. "
        f"Best val: {best_val_loss:.6f}, saved=cycle_end ===")
    return history, run_dir, global_step


def _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
               global_step, epoch, val_loss, log, temperature=None,
               cycle_id=None, examples_in_cycle=None, reason="val_improved",
               write_snapshot=True):
    """Save checkpoint to a rolling `best.pt` (always) and optionally a
    per-cycle snapshot `cycles/cycle_<N>.pt`. The rolling file is what
    `_find_best_checkpoint` loads (latest state); the per-cycle snapshot is
    permanent and gated by `save_every_cycles` via `write_snapshot`.
    """
    ckpt = {
        "step": global_step, "epoch": epoch + 1,
        "cycle_id": cycle_id,
        "examples_in_cycle": examples_in_cycle,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_stats": norm_stats,
        "val_loss": val_loss,
        "save_reason": reason,
    }
    if temperature is not None:
        ckpt["temperature"] = temperature

    best_path = os.path.join(ckpt_dir, "best.pt")
    torch.save(ckpt, best_path)

    snapshot_written = False
    if cycle_id is not None and write_snapshot:
        snapshots_dir = os.path.join(ckpt_dir, "cycles")
        os.makedirs(snapshots_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshots_dir, f"cycle_{cycle_id:04d}.pt")
        torch.save(ckpt, snapshot_path)
        snapshot_written = True

    snap_str = " + snapshot" if snapshot_written else ""
    log(f"  Saved best.pt{snap_str} ({reason}, val={val_loss:.6f}, "
        f"cycle={cycle_id})")
