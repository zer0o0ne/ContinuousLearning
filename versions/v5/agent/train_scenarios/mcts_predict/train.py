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


def _mcts_forward(agent, event_sequences, chains, device,
                  opponent_emb_table=None, p_tf=0.0, stop_grad_old_embs=True):
    """Forward pass: perception → root predictions → modelling chain.

    Chain semantics (matches `collect.py:MCTSTrainingExample`):
      step.action_taken is the action chosen at the PRIOR state. We extend
      ctx by appending modelling_head(ctx)[step.action_taken] (which encodes
      "next state if action_taken was taken at current ctx") then predict
      the distribution at that extended state and compare with
      step.target_distribution. This mirrors MCTS inner-node expansion
      (mcts.py: child.action_embedding = modelling_head(parent_ctx)[a]).

    New training signals (controllable via train_cfg / cycle-scheduled p_tf):
      - Teacher forcing (point 2): with probability p_tf, the prediction heads
        receive the REAL perception of decisions[t+1+i] as ctx instead of the
        recursively-rolled ctx. The rolled ctx is still maintained so modelling
        and downstream chain steps stay defined; only the prediction-head input
        is swapped this step.
      - Stop-gradient on old modelling embeddings (point 6): when extending
        ctx_rolled for the next iteration, the appended emb is detached so the
        gradient at step i only updates modelling_head/perception via the
        CURRENT step's emb (and via the original root perception_out).
      - Reconstruction (point 1) and per-step value (point 11) targets are
        produced here and consumed by `_compute_loss`.

    Returns: dict with keys
      value_preds: (B, 1) — root prediction only
      action_preds: (B, n_actions) — root prediction only
      chain_action_preds / chain_action_targets / chain_is_hero:
          per-batch list of lists (n_actions,) / (n_actions,) / bool
      chain_value_preds / chain_value_targets:
          per-batch list of lists, scalar tensors
      chain_recon_preds / chain_recon_targets / chain_recon_depths:
          per-batch list of lists, (D,) tensors / (D,) tensors / int — the
          depth `i` of each recon entry in the original chain. Targets
          detached. Only populated for steps whose ChainStep.events_at_step
          is non-empty; the depth list lets `_compute_loss` apply the same
          `chain_depth_gamma` weighting as the action/value chain losses
          even though recon is sparse over chain positions.
    """
    skip_opp = (opponent_emb_table is None)

    # 1. Root perception (updates opp_table once)
    perception_out, _, mask = agent.perception.forward_batch(
        event_sequences, device=device, skip_memory=True,
        skip_opponent_emb=skip_opp, opponent_emb_table=opponent_emb_table)

    # 2. Root predictions
    value_preds = agent.value_head(perception_out, mask=mask)
    action_preds = agent.action_head(perception_out, mask=mask)

    B = perception_out.shape[0]

    # 3. Collect chain events for ONE batched perception forward
    flat_step_events = []
    step_index = {}  # (b, i) -> index into flat_step_events / chain_pooled
    for b, chain in enumerate(chains):
        for i, step in enumerate(chain):
            evts = getattr(step, "events_at_step", None) or []
            if evts:
                step_index[(b, i)] = len(flat_step_events)
                flat_step_events.append(evts)

    chain_perception_out = None
    chain_perception_mask = None
    chain_pooled_detached = None
    need_grad_chain_perc = bool(flat_step_events) and p_tf > 0.0
    if flat_step_events:
        # Snapshot opp_table embeddings BEFORE chain perception so the GRU
        # state seen by next batch matches what root-only would have left
        # behind (no double-update from chain perception).
        opp_snapshot = None
        if opponent_emb_table is not None:
            opp_snapshot = dict(opponent_emb_table.embeddings)

        if need_grad_chain_perc:
            chain_perception_out, _, chain_perception_mask = (
                agent.perception.forward_batch(
                    flat_step_events, device=device, skip_memory=True,
                    skip_opponent_emb=skip_opp,
                    opponent_emb_table=opponent_emb_table)
            )
        else:
            with torch.no_grad():
                chain_perception_out, _, chain_perception_mask = (
                    agent.perception.forward_batch(
                        flat_step_events, device=device, skip_memory=True,
                        skip_opponent_emb=skip_opp,
                        opponent_emb_table=opponent_emb_table)
                )

        if opp_snapshot is not None:
            opponent_emb_table.embeddings = opp_snapshot

        # Pool real-perception output to a single (D,) per step for recon target
        m_float = chain_perception_mask.float().unsqueeze(-1)  # (M, S, 1)
        sums = (chain_perception_out * m_float).sum(dim=1)     # (M, D)
        counts = m_float.sum(dim=1).clamp(min=1.0)             # (M, 1)
        chain_pooled_detached = (sums / counts).detach()       # (M, D)

    # 4. Per-example chain loop
    chain_action_preds = []
    chain_action_targets = []
    chain_is_hero = []
    chain_value_preds = []
    chain_value_targets = []
    chain_recon_preds = []
    chain_recon_targets = []
    chain_recon_depths = []

    for b in range(B):
        chain = chains[b]
        if not chain:
            for L in (chain_action_preds, chain_action_targets, chain_is_hero,
                      chain_value_preds, chain_value_targets,
                      chain_recon_preds, chain_recon_targets,
                      chain_recon_depths):
                L.append([])
            continue

        ctx_rolled = perception_out[b:b+1]
        ctx_mask = mask[b:b+1]

        b_action_preds = []
        b_action_tgts = []
        b_is_hero = []
        b_value_preds = []
        b_value_tgts = []
        b_recon_preds = []
        b_recon_tgts = []
        b_recon_depths = []

        for i, step in enumerate(chain):
            action_embs = agent.modelling_head(ctx_rolled, mask=ctx_mask)
            emb = action_embs[:, step.action_taken, :]  # (1, D)
            new_emb_token = emb.unsqueeze(1)            # (1, 1, D)
            ones_mask = torch.ones(1, 1, dtype=ctx_mask.dtype, device=device)
            ctx_with_new_emb = torch.cat([ctx_rolled, new_emb_token], dim=1)
            ctx_with_new_emb_mask = torch.cat([ctx_mask, ones_mask], dim=1)

            tf_idx = step_index.get((b, i))
            use_tf = (tf_idx is not None and p_tf > 0.0
                      and chain_perception_out is not None
                      and random.random() < p_tf)
            if use_tf:
                ctx_for_pred = chain_perception_out[tf_idx:tf_idx+1]
                ctx_for_pred_mask = chain_perception_mask[tf_idx:tf_idx+1]
            else:
                ctx_for_pred = ctx_with_new_emb
                ctx_for_pred_mask = ctx_with_new_emb_mask

            if step.is_hero:
                pred = agent.action_head(ctx_for_pred, mask=ctx_for_pred_mask)
            else:
                pred = agent.opponent_action_head(ctx_for_pred,
                                                   mask=ctx_for_pred_mask)
            value_pred = agent.value_head(ctx_for_pred, mask=ctx_for_pred_mask)

            b_action_preds.append(pred.squeeze(0))
            b_action_tgts.append(torch.tensor(
                step.target_distribution, dtype=torch.float32, device=device))
            b_is_hero.append(bool(step.is_hero))
            b_value_preds.append(value_pred.reshape(()))
            b_value_tgts.append(torch.tensor(
                float(getattr(step, "value_target", 0.0)),
                dtype=torch.float32, device=device))

            if tf_idx is not None and chain_pooled_detached is not None:
                m_float = ctx_with_new_emb_mask.float().unsqueeze(-1)
                sums_e = (ctx_with_new_emb * m_float).sum(dim=1)  # (1, D)
                counts_e = m_float.sum(dim=1).clamp(min=1.0)      # (1, 1)
                recon_pred = (sums_e / counts_e).squeeze(0)       # (D,)
                b_recon_preds.append(recon_pred)
                b_recon_tgts.append(chain_pooled_detached[tf_idx])
                b_recon_depths.append(i)

            if stop_grad_old_embs:
                ctx_rolled = torch.cat([ctx_rolled, new_emb_token.detach()],
                                        dim=1)
            else:
                ctx_rolled = torch.cat([ctx_rolled, new_emb_token], dim=1)
            ctx_mask = ctx_with_new_emb_mask

        chain_action_preds.append(b_action_preds)
        chain_action_targets.append(b_action_tgts)
        chain_is_hero.append(b_is_hero)
        chain_value_preds.append(b_value_preds)
        chain_value_targets.append(b_value_tgts)
        chain_recon_preds.append(b_recon_preds)
        chain_recon_targets.append(b_recon_tgts)
        chain_recon_depths.append(b_recon_depths)

    return {
        "value_preds": value_preds,
        "action_preds": action_preds,
        "chain_action_preds": chain_action_preds,
        "chain_action_targets": chain_action_targets,
        "chain_is_hero": chain_is_hero,
        "chain_value_preds": chain_value_preds,
        "chain_value_targets": chain_value_targets,
        "chain_recon_preds": chain_recon_preds,
        "chain_recon_targets": chain_recon_targets,
        "chain_recon_depths": chain_recon_depths,
    }


def _compute_loss(forward_out, value_targets, action_targets,
                  value_weight=1.0, action_weight=1.0, chain_weight=1.0,
                  recon_weight=0.5, value_chain_weight=0.5,
                  chain_depth_gamma=0.7):
    """Compute combined loss across all heads.

    Aggregation policy (post-modifications):
      - chain KL, chain value AND reconstruction are first weighted by
        `gamma^i` within an example, normalized by the sum of weights for
        that example, then averaged across examples (point 5 + point 3).
        For recon, `i` is the original chain depth (not the position in
        the filtered recon list) — recon entries exist only for steps with
        non-empty `events_at_step`, so the gamma exponent must come from
        `chain_recon_depths`, not from sequential indexing. This stops a
        single long-chain example from dominating the gradient AND aligns
        recon weighting with the action/value chain losses (deeper
        modelling rollouts compound prediction error, so their recon
        target is also intrinsically noisier).
      - Empty chains contribute nothing.

    Args:
        forward_out: dict returned by `_mcts_forward`.
        value_targets / action_targets: root targets (B,) / (B, n_actions).

    Returns: (total_loss, loss_dict) where loss_dict has float entries for
        value / action / chain / chain_value / recon / total.
    """
    value_preds = forward_out["value_preds"]
    action_preds = forward_out["action_preds"]
    device = value_preds.device

    # --- Root losses (unchanged semantics) ---
    value_loss = F.smooth_l1_loss(value_preds.squeeze(-1), value_targets)
    action_log_probs = F.log_softmax(action_preds, dim=-1)
    action_loss = F.kl_div(action_log_probs, action_targets, reduction="batchmean")

    # --- Helper: per-example weighted mean with gamma^i ---
    def _per_example_weighted(per_example_steps, gamma):
        """Each item: list of scalar tensors (one per chain step).
        Returns batch-mean of (sum gamma^i * loss_i / sum gamma^i)."""
        per_ex = []
        for steps in per_example_steps:
            if not steps:
                continue
            weights = [gamma ** i for i in range(len(steps))]
            total_w = sum(weights) or 1.0
            weighted_sum = torch.stack(
                [w * s for w, s in zip(weights, steps)]
            ).sum()
            per_ex.append(weighted_sum / total_w)
        if not per_ex:
            return torch.tensor(0.0, device=device)
        return torch.stack(per_ex).mean()

    # --- Chain action KL (depth-weighted per example) ---
    chain_kl_per_example = []
    for b_preds, b_targets in zip(forward_out["chain_action_preds"],
                                   forward_out["chain_action_targets"]):
        steps = []
        for pred, target in zip(b_preds, b_targets):
            log_probs = F.log_softmax(pred, dim=-1)
            steps.append(F.kl_div(log_probs, target, reduction="sum"))
        chain_kl_per_example.append(steps)
    chain_loss = _per_example_weighted(chain_kl_per_example, chain_depth_gamma)

    # --- Chain value (depth-weighted per example, SmoothL1) ---
    chain_value_per_example = []
    for b_preds, b_targets in zip(forward_out["chain_value_preds"],
                                   forward_out["chain_value_targets"]):
        steps = []
        for pred, target in zip(b_preds, b_targets):
            steps.append(F.smooth_l1_loss(pred, target))
        chain_value_per_example.append(steps)
    chain_value_loss = _per_example_weighted(chain_value_per_example,
                                              chain_depth_gamma)

    # --- Reconstruction MSE (depth-weighted per example using ORIGINAL
    # chain index, not the position within the recon-only sublist) ---
    recon_per_example = []
    for b_preds, b_targets, b_depths in zip(
            forward_out["chain_recon_preds"],
            forward_out["chain_recon_targets"],
            forward_out["chain_recon_depths"]):
        if not b_preds:
            continue
        step_losses = [F.mse_loss(p, t) for p, t in zip(b_preds, b_targets)]
        weights = [chain_depth_gamma ** d for d in b_depths]
        total_w = sum(weights) or 1.0
        weighted_sum = torch.stack(
            [w * s for w, s in zip(weights, step_losses)]
        ).sum()
        recon_per_example.append(weighted_sum / total_w)
    if recon_per_example:
        recon_loss = torch.stack(recon_per_example).mean()
    else:
        recon_loss = torch.tensor(0.0, device=device)

    total = (value_weight * value_loss
             + action_weight * action_loss
             + chain_weight * chain_loss
             + value_chain_weight * chain_value_loss
             + recon_weight * recon_loss)

    return total, {
        "value": value_loss.item(),
        "action": action_loss.item(),
        "chain": chain_loss.item(),
        "chain_value": chain_value_loss.item(),
        "recon": recon_loss.item(),
        "total": total.item(),
    }


_LOSS_KEYS = ("total", "value", "action", "chain", "chain_value", "recon")


def _run_validation(agent, val_loader, device, weights, amp_config=None,
                    opponent_emb_table=None, stop_grad_old_embs=True):
    """Run validation. Returns dict with total + component losses.

    Validation always uses p_tf=0 (fully rolled chain) so the metric stays
    comparable across cycles regardless of the current teacher-forcing
    schedule.
    """
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    agent.eval()
    sums = {k: 0.0 for k in _LOSS_KEYS}
    count = 0
    with torch.no_grad():
        for event_seqs, val_targets, act_targets, chains in val_loader:
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                forward_out = _mcts_forward(
                    agent, event_seqs, chains, device,
                    opponent_emb_table=opponent_emb_table,
                    p_tf=0.0,
                    stop_grad_old_embs=stop_grad_old_embs)
                _, ldict = _compute_loss(
                    forward_out, val_targets, act_targets, **weights)
            n = len(event_seqs)
            for k in _LOSS_KEYS:
                sums[k] += ldict[k] * n
            count += n
    agent.train()
    if count == 0:
        return {k: 0.0 for k in _LOSS_KEYS}
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
    recon_weight = train_cfg.get("recon_weight", 0.0)
    value_chain_weight = train_cfg.get("value_chain_weight", 0.0)
    chain_depth_gamma = float(train_cfg.get("chain_depth_gamma", 1.0))
    stop_grad_old_embs = bool(train_cfg.get("stop_grad_old_embs", True))
    weights = {"value_weight": value_weight, "action_weight": action_weight,
               "chain_weight": chain_weight, "recon_weight": recon_weight,
               "value_chain_weight": value_chain_weight,
               "chain_depth_gamma": chain_depth_gamma}

    # Teacher-forcing probability decays linearly from p_start → p_end over
    # `decay_cycles` cycles. Disabled (p_tf=0) when section is missing.
    tf_cfg = train_cfg.get("teacher_forcing", {}) or {}
    tf_p_start = float(tf_cfg.get("p_start", 0.0))
    tf_p_end = float(tf_cfg.get("p_end", 0.0))
    tf_decay_cycles = max(1, int(tf_cfg.get("decay_cycles", 1)))
    if tf_p_start <= 0.0 and tf_p_end <= 0.0:
        p_tf = 0.0
    else:
        frac = min(1.0, max(0.0, cycle_id / tf_decay_cycles))
        p_tf = tf_p_start + (tf_p_end - tf_p_start) * frac

    log(f"=== MCTS Training (cycle {cycle_id}, save={save_checkpoint}) ===")

    # Preserve norm_stats from checkpoint for saving. Use `is None` rather
    # than `or {}` so an empty-but-present dict still shares its identity
    # with the pipeline's `agent_info["norm_stats"]` — that's the dict
    # `_finalize_value_targets` mutates to bootstrap `mcts_value_scale`,
    # and we need those mutations to flow into the saved best.pt.
    norm_stats = getattr(agent, '_checkpoint_norm_stats', None)
    if norm_stats is None:
        norm_stats = {}
        agent._checkpoint_norm_stats = norm_stats

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
    log(f"Weights: value={value_weight}, action={action_weight}, "
        f"chain={chain_weight}, recon={recon_weight}, "
        f"chain_value={value_chain_weight}, depth_gamma={chain_depth_gamma}")
    log(f"Chain extras: p_tf={p_tf:.3f} "
        f"(start={tf_p_start}, end={tf_p_end}, decay={tf_decay_cycles}), "
        f"stop_grad_old_embs={stop_grad_old_embs}")
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
                forward_out = _mcts_forward(
                    agent, event_seqs, chains, device,
                    opponent_emb_table=opp_table,
                    p_tf=p_tf,
                    stop_grad_old_embs=stop_grad_old_embs)
                loss, loss_dict = _compute_loss(
                    forward_out, val_targets, act_targets, **weights)

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
                "chain_value": loss_dict["chain_value"],
                "recon": loss_dict["recon"],
                "p_tf": p_tf,
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
                    f"a={loss_dict['action']:.4f} c={loss_dict['chain']:.4f} "
                    f"cv={loss_dict['chain_value']:.4f} "
                    f"r={loss_dict['recon']:.4f}) "
                    f"lr={current_lr:.2e}")

            if val_every and (global_step % val_every == 0):
                val_dict = _run_validation(
                    agent, val_loader, device, weights, amp_cfg,
                    opponent_emb_table=opp_table,
                    stop_grad_old_embs=stop_grad_old_embs)
                vl = val_dict["total"]
                history["val_loss"].append({
                    "step": global_step,
                    "cycle_id": cycle_id,
                    **val_dict,
                })
                log(f"  [Cycle {cycle_id} Step {global_step}] Val: "
                    f"total={vl:.6f} v={val_dict['value']:.4f} "
                    f"a={val_dict['action']:.4f} c={val_dict['chain']:.4f} "
                    f"cv={val_dict['chain_value']:.4f} "
                    f"r={val_dict['recon']:.4f}")
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
        val_dict = _run_validation(
            agent, val_loader, device, weights, amp_cfg,
            opponent_emb_table=opp_table,
            stop_grad_old_embs=stop_grad_old_embs)
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
            f"Train: {train_avg:.6f}, Val: {val_avg:.6f} "
            f"(v={val_dict['value']:.4f} a={val_dict['action']:.4f} "
            f"c={val_dict['chain']:.4f} cv={val_dict['chain_value']:.4f} "
            f"r={val_dict['recon']:.4f})")

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
