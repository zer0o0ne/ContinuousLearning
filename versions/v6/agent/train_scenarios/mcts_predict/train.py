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
from tqdm.auto import tqdm

from agent.train_scenarios.mcts_predict.dataset import MCTSDataset, batch_collate, make_tensor_collate
from agent.train_scenarios._checkpoint_io import make_checkpoint
from agent.resume import atomic_torch_save
from agent.train_scenarios._history import IncrementalHistory


_PHASE = "mcts_predict"


class LengthGroupedBatchSampler(Sampler):
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


def _mcts_forward(agent, event_sequences, chains, device,
                  opponent_emb_table=None, p_tf=0.0, stop_grad_old_embs=True,
                  examples_per_batch_terminals=None, precomputed=None):
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
        skip_opponent_emb=skip_opp, opponent_emb_table=opponent_emb_table,
        precomputed=precomputed)

    # 2. Root predictions
    value_preds = agent.value_head(perception_out, mask=mask)
    action_preds = agent.action_head(perception_out, mask=mask)

    B = perception_out.shape[0]

    # 3. Collect chain events for ONE batched perception forward.
    # D.3: only hero-owned steps — opp-step events contain the opponent's
    # private hand cards which are unavailable during search. Teacher
    # forcing and recon on opp-steps would train on unreachable inputs.
    flat_step_events = []
    step_index = {}  # (b, i) -> index into flat_step_events / chain_pooled
    for b, chain in enumerate(chains):
        for i, step in enumerate(chain):
            evts = getattr(step, "events_at_step", None) or []
            if evts and step.is_hero:
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

    # 4. E.1.5: depth-batched chain loop — process chain steps of the same
    #    depth across examples in one padded forward instead of B=1 per step.
    chain_action_preds = [[] for _ in range(B)]
    chain_action_targets = [[] for _ in range(B)]
    chain_is_hero = [[] for _ in range(B)]
    chain_value_preds = [[] for _ in range(B)]
    chain_value_targets = [[] for _ in range(B)]
    chain_recon_preds = [[] for _ in range(B)]
    chain_recon_targets = [[] for _ in range(B)]
    chain_recon_depths = [[] for _ in range(B)]

    # A.5.1: trim each example to its true length
    per_ex_ctx = [None] * B
    per_ex_mask = [None] * B
    for b in range(B):
        if chains[b]:
            L_b = int(mask[b].sum().item())
            per_ex_ctx[b] = perception_out[b:b+1, :L_b]
            per_ex_mask[b] = mask[b:b+1, :L_b]

    max_depth = max((len(chains[b]) for b in range(B) if chains[b]), default=0)
    D_model = perception_out.shape[2]

    for d in range(max_depth):
        active_bs = [b for b in range(B) if chains[b] and d < len(chains[b])]
        if not active_bs:
            continue
        N_act = len(active_bs)

        # --- Modelling head: batched across active examples ---
        ctxs = [per_ex_ctx[b] for b in active_bs]
        ctx_masks = [per_ex_mask[b] for b in active_bs]
        max_ctx_len = max(c.size(1) for c in ctxs)

        padded_c = torch.zeros(N_act, max_ctx_len, D_model,
                               dtype=ctxs[0].dtype, device=device)
        padded_m = torch.zeros(N_act, max_ctx_len,
                               dtype=ctx_masks[0].dtype, device=device)
        for idx, (c, m) in enumerate(zip(ctxs, ctx_masks)):
            Lc = c.size(1)
            padded_c[idx, :Lc] = c[0]
            padded_m[idx, :Lc] = m[0]

        batch_embs = agent.modelling_head(padded_c, mask=padded_m)

        # Extract per-example embedding, build prediction contexts
        steps_d = []
        new_emb_tokens = []
        pred_ctxs = []
        pred_masks = []

        for idx, b in enumerate(active_bs):
            step = chains[b][d]
            steps_d.append(step)
            emb = batch_embs[idx:idx+1, step.action_taken, :]
            new_tok = emb.unsqueeze(1)
            new_emb_tokens.append(new_tok)

            ones = torch.ones(1, 1, dtype=per_ex_mask[b].dtype, device=device)
            ctx_with_new = torch.cat([per_ex_ctx[b], new_tok], dim=1)
            mask_with_new = torch.cat([per_ex_mask[b], ones], dim=1)

            tf_idx_val = step_index.get((b, d))
            use_tf = (tf_idx_val is not None and p_tf > 0.0
                      and step.is_hero
                      and chain_perception_out is not None
                      and random.random() < p_tf)
            if use_tf:
                pred_ctxs.append(chain_perception_out[tf_idx_val:tf_idx_val+1])
                pred_masks.append(chain_perception_mask[tf_idx_val:tf_idx_val+1])
            else:
                pred_ctxs.append(ctx_with_new)
                pred_masks.append(mask_with_new)

        # --- Prediction heads: batched ---
        max_pred_len = max(pc.size(1) for pc in pred_ctxs)
        all_pred = torch.zeros(N_act, max_pred_len, D_model,
                               dtype=pred_ctxs[0].dtype, device=device)
        all_pred_m = torch.zeros(N_act, max_pred_len,
                                 dtype=pred_masks[0].dtype, device=device)
        for idx, (pc, pm) in enumerate(zip(pred_ctxs, pred_masks)):
            Lp = pc.size(1)
            all_pred[idx, :Lp] = pc[0]
            all_pred_m[idx, :Lp] = pm[0]

        batch_values = agent.value_head(all_pred, mask=all_pred_m)

        hero_idxs = [idx for idx, s in enumerate(steps_d) if s.is_hero]
        opp_idxs = [idx for idx, s in enumerate(steps_d) if not s.is_hero]

        batch_hero_preds = None
        if hero_idxs:
            h_idx = torch.tensor(hero_idxs, dtype=torch.long)
            batch_hero_preds = agent.action_head(
                all_pred[h_idx], mask=all_pred_m[h_idx])

        batch_opp_preds = None
        if opp_idxs:
            o_idx = torch.tensor(opp_idxs, dtype=torch.long)
            batch_opp_preds = agent.opponent_action_head(
                all_pred[o_idx], mask=all_pred_m[o_idx])

        # --- Scatter results back ---
        hero_ctr = 0
        opp_ctr = 0
        for idx, b in enumerate(active_bs):
            step = steps_d[idx]
            if step.is_hero:
                act_pred = batch_hero_preds[hero_ctr:hero_ctr+1]
                hero_ctr += 1
            else:
                act_pred = batch_opp_preds[opp_ctr:opp_ctr+1]
                opp_ctr += 1

            chain_action_preds[b].append(act_pred.squeeze(0))
            chain_action_targets[b].append(torch.tensor(
                step.target_distribution, dtype=torch.float32, device=device))
            chain_is_hero[b].append(bool(step.is_hero))
            chain_value_preds[b].append(batch_values[idx].reshape(()))
            chain_value_targets[b].append(torch.tensor(
                float(getattr(step, "value_target", 0.0)),
                dtype=torch.float32, device=device))

            # D.3: recon only for hero-owned steps
            tf_idx_val = step_index.get((b, d))
            if (tf_idx_val is not None and chain_pooled_detached is not None
                    and step.is_hero):
                ones = torch.ones(1, 1, dtype=per_ex_mask[b].dtype,
                                  device=device)
                ctx_wn = torch.cat([per_ex_ctx[b], new_emb_tokens[idx]],
                                   dim=1)
                msk_wn = torch.cat([per_ex_mask[b], ones], dim=1)
                m_fl = msk_wn.float().unsqueeze(-1)
                sums_e = (ctx_wn * m_fl).sum(dim=1)
                counts_e = m_fl.sum(dim=1).clamp(min=1.0)
                recon_pred = (sums_e / counts_e).squeeze(0)
                chain_recon_preds[b].append(recon_pred)
                chain_recon_targets[b].append(chain_pooled_detached[tf_idx_val])
                chain_recon_depths[b].append(d)

        # --- Update contexts for next depth ---
        for idx, b in enumerate(active_bs):
            ones = torch.ones(1, 1, dtype=per_ex_mask[b].dtype, device=device)
            tok = new_emb_tokens[idx].detach() if stop_grad_old_embs \
                else new_emb_tokens[idx]
            per_ex_ctx[b] = torch.cat([per_ex_ctx[b], tok], dim=1)
            per_ex_mask[b] = torch.cat([per_ex_mask[b], ones], dim=1)

    # 5. Terminal value supervision: for each example, roll out the modelling
    #    chain along each terminal's action_path_from_root and predict
    #    value_head at the terminal context. MSE'd against equity_Q in
    #    `_compute_loss`. Per-terminal sequential rollout (within an example).
    #    K_worst+K_best caps the count per example so this stays bounded.
    terminal_value_preds = []
    terminal_value_targets = []
    if examples_per_batch_terminals is None:
        examples_per_batch_terminals = [[] for _ in range(B)]
    for b in range(B):
        # `examples_per_batch_terminals[b]` is one list per example, each
        # entry a (action_path, equity_Q) tuple.
        t_list = examples_per_batch_terminals[b] or []
        if not t_list:
            terminal_value_preds.append([])
            terminal_value_targets.append([])
            continue

        b_t_preds = []
        b_t_tgts = []
        ctx_init = perception_out[b:b+1]
        mask_init = mask[b:b+1]
        for action_path, equity_Q in t_list:
            ctx = ctx_init
            ctx_mask = mask_init
            for a in action_path:
                a_embs = agent.modelling_head(ctx, mask=ctx_mask)  # (1, n_actions, d)
                emb_tok = a_embs[:, int(a), :].unsqueeze(1).detach()  # (1, 1, d)
                ones = torch.ones(1, 1, dtype=ctx_mask.dtype, device=device)
                ctx = torch.cat([ctx, emb_tok], dim=1)
                ctx_mask = torch.cat([ctx_mask, ones], dim=1)
            v_pred = agent.value_head(ctx, mask=ctx_mask).reshape(())
            b_t_preds.append(v_pred)
            b_t_tgts.append(torch.tensor(
                float(equity_Q), dtype=torch.float32, device=device))
        terminal_value_preds.append(b_t_preds)
        terminal_value_targets.append(b_t_tgts)

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
        "terminal_value_preds": terminal_value_preds,
        "terminal_value_targets": terminal_value_targets,
    }


def _compute_loss(forward_out, value_targets, action_targets,
                  value_weight=1.0, action_weight=1.0, chain_weight=1.0,
                  recon_weight=0.5, value_chain_weight=0.5,
                  chain_depth_gamma=0.7, entropy_weight=0.0,
                  terminal_value_weight=0.0):
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
    # Entropy regularization on root action_head — encourages mixed strategies
    # so policy doesn't collapse onto 1–2 deterministic actions during
    # self-play. `entropy_weight * Σ p log p` adds a NEGATIVE entropy term to
    # the loss; minimization therefore maximizes H. The bare KL action_loss
    # value (above) is what gets reported / averaged into `last_action_loss`
    # — entropy term is logged separately so the strange-traversal schedule
    # stays calibrated to actual policy convergence, not the regularizer.
    action_probs = action_log_probs.exp()
    action_entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
    entropy_penalty = -float(entropy_weight) * action_entropy

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

    # --- Terminal value MSE: direct supervision at tree-terminal states.
    # `equity_Q` (from `evaluate_all_terminals`) is the target; current
    # value_head on the rolled terminal context is the prediction. K_worst +
    # K_best terminals per example (sampled from both tails of Q to avoid
    # bias toward only "good" outcomes — see `_select_terminal_targets`).
    # Per-example mean of per-terminal MSE; batch-mean across examples.
    terminal_per_example = []
    for b_preds, b_tgts in zip(forward_out.get("terminal_value_preds", []),
                                forward_out.get("terminal_value_targets", [])):
        if not b_preds:
            continue
        step_losses = torch.stack(
            [F.smooth_l1_loss(p, t) for p, t in zip(b_preds, b_tgts)])
        terminal_per_example.append(step_losses.mean())
    if terminal_per_example:
        terminal_value_loss = torch.stack(terminal_per_example).mean()
    else:
        terminal_value_loss = torch.tensor(0.0, device=device)

    total = (value_weight * value_loss
             + action_weight * action_loss
             + chain_weight * chain_loss
             + value_chain_weight * chain_value_loss
             + recon_weight * recon_loss
             + terminal_value_weight * terminal_value_loss
             + entropy_penalty)

    return total, {
        "value": value_loss.item(),
        "action": action_loss.item(),
        "chain": chain_loss.item(),
        "chain_value": chain_value_loss.item(),
        "recon": recon_loss.item(),
        "terminal_value": terminal_value_loss.item(),
        "action_entropy": action_entropy.item(),
        "total": total.item(),
    }


_LOSS_KEYS = ("total", "value", "action", "chain", "chain_value", "recon",
              "terminal_value", "action_entropy")


def _run_validation(agent, val_loader, device, weights, amp_config=None,
                    opponent_emb_table=None, stop_grad_old_embs=True):
    """Run validation. Returns dict with total + component losses.

    Validation always uses p_tf=0 (fully rolled chain) so the metric stays
    comparable across cycles regardless of the current teacher-forcing
    schedule.
    """
    amp_enabled, device_type, amp_dtype = amp_config or (False, "cpu", torch.float32)
    # A.4.3: validation forwards mutate the table — work on a clone so the live
    # training table is not advanced by the val set.
    if opponent_emb_table is not None:
        opponent_emb_table = opponent_emb_table.clone()
    agent.eval()
    sums = {k: 0.0 for k in _LOSS_KEYS}
    count = 0
    with torch.no_grad():
        for event_sequences, precomputed, val_targets, act_targets, chains, term_tgts in val_loader:
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                forward_out = _mcts_forward(
                    agent, event_sequences, chains, device,
                    opponent_emb_table=opponent_emb_table,
                    p_tf=0.0,
                    stop_grad_old_embs=stop_grad_old_embs,
                    examples_per_batch_terminals=term_tgts,
                    precomputed=precomputed)
                _, ldict = _compute_loss(
                    forward_out, val_targets, act_targets, **weights)
            n = precomputed["B"] if precomputed is not None else 0
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
               optimizer=None, scheduler=None, scaler=None):
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
    entropy_weight = float(train_cfg.get("entropy_weight", 0.0))
    terminal_value_weight = float(train_cfg.get("terminal_value_weight", 0.0))
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    weights = {"value_weight": value_weight, "action_weight": action_weight,
               "chain_weight": chain_weight, "recon_weight": recon_weight,
               "value_chain_weight": value_chain_weight,
               "chain_depth_gamma": chain_depth_gamma,
               "entropy_weight": entropy_weight,
               "terminal_value_weight": terminal_value_weight}

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
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=use_scaler)
    amp_cfg = (amp_enabled, device_type, amp_dtype)

    # Opponent embedding table
    opp_table = None
    if agent.perception.opp_emb_enabled:
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        opp_table = OpponentEmbeddingTable(agent.perception.d_model)
        log("Opponent GRU embedding enabled")

    # Dataset — D.2: hand-aware split prevents chain leakage (chain of
    # example t contains roots of examples t+1..t+5 from the same hand;
    # random split = direct train↔val leak).
    dataset = MCTSDataset(examples)
    from agent.train_scenarios.split import hand_aware_split
    scenarios_for_split = [
        {"events": ex.events,
         "num_players": ex.events[0]["num_players"] if ex.events else 2}
        for ex in examples
    ]
    train_dataset, val_dataset = hand_aware_split(
        dataset, scenarios_for_split, val_split, seed=42)
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    if opp_table is not None:
        # A.4.4: feed in collection (chronological) order so the opponent GRU
        # table accumulates as at inference. Examples are appended per decision
        # in hand order, so the original example index is a chronological proxy;
        # sort subset positions by it (the random split scrambles which examples
        # land in train, but their relative order is restored here).
        from agent.train_scenarios.split import OrderedBatchSampler
        order = sorted(range(len(train_indices)), key=lambda pos: train_indices[pos])
        train_sampler = OrderedBatchSampler(order, batch_size)
        log("Opponent GRU active → chronological (collection-order) batch order")
    else:
        train_sampler = LengthGroupedBatchSampler(train_dataset, batch_size)
    max_players = agent.perception.embedder.max_players
    _tensor_collate = make_tensor_collate(max_players)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=_tensor_collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=_tensor_collate, num_workers=0)

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
    history_dir = os.path.dirname(history_path)
    hist = IncrementalHistory(history_dir,
                              keys=["step_loss", "val_loss",
                                    "epoch_train_loss", "epoch_val_loss",
                                    "cycles"])
    history = hist.data

    best_val_loss = float("inf")
    fails_since_best = 0
    # Accumulate `action_loss` (KL only — entropy term excluded) across all
    # training steps of this cycle. Mean stored into
    # `norm_stats["last_action_loss"]` at cycle end and consumed by
    # `run_mcts_collection` next cycle to set `p_strange = C(t) * exp(-loss)`.
    cycle_action_sum = 0.0
    cycle_action_count = 0
    global_step = int(global_step_offset)
    cycle_step_start = global_step
    stopped_early = False

    agent.set_gradient_checkpointing(gradient_checkpointing)
    if gradient_checkpointing:
        log("Gradient checkpointing enabled on perception + all heads")

    for epoch in range(epochs):
        if stopped_early:
            break

        agent.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (event_sequences, precomputed, val_targets, act_targets, chains, term_tgts) in enumerate(
                tqdm(train_loader, desc=f"MCTS cycle {cycle_id} epoch {epoch+1}/{epochs}", leave=False, smoothing=0)):
            val_targets = val_targets.to(device)
            act_targets = act_targets.to(device)
            n_samples = precomputed["B"] if precomputed is not None else 0

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                forward_out = _mcts_forward(
                    agent, event_sequences, chains, device,
                    opponent_emb_table=opp_table,
                    p_tf=p_tf,
                    stop_grad_old_embs=stop_grad_old_embs,
                    examples_per_batch_terminals=term_tgts,
                    precomputed=precomputed)
                loss, loss_dict = _compute_loss(
                    forward_out, val_targets, act_targets, **weights)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
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
                "terminal_value": loss_dict["terminal_value"],
                "action_entropy": loss_dict["action_entropy"],
                "p_tf": p_tf,
                "lr": current_lr,
                "batch_size": n_samples,
            })
            global_step += 1
            train_loss_sum += step_loss * n_samples
            train_count += n_samples
            cycle_action_sum += loss_dict["action"] * n_samples
            cycle_action_count += n_samples

            if (batch_idx + 1) % log_every == 0:
                avg = train_loss_sum / train_count
                log(f"  Cycle {cycle_id} Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, "
                    f"Loss: {avg:.6f} (v={loss_dict['value']:.4f} "
                    f"a={loss_dict['action']:.4f} c={loss_dict['chain']:.4f} "
                    f"cv={loss_dict['chain_value']:.4f} "
                    f"r={loss_dict['recon']:.4f} "
                    f"tv={loss_dict['terminal_value']:.4f} "
                    f"H={loss_dict['action_entropy']:.4f}) "
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

    # D.5.4: best.pt is **latest by design** — always rolled forward
    # (overwritten at cycle end). Within a single cycle the val-gated save
    # (above) is also written when val improves, but the cycle_end save
    # overwrites it unconditionally. Rationale: each MCTS cycle collects
    # NEW self-play data from the updated agent, so intra-cycle val is a
    # noisy proxy; the latest weights are the most meaningful input for the
    # next collection. With D.2 (hand-aware split) the val signal is now
    # reliable enough that intra-cycle val-gating could be reconsidered in
    # a future enhancement. Permanent per-cycle snapshots under cycles/
    # are gated by save_every_cycles via save_checkpoint.
    final_val = float(val_avg) if 'val_avg' in locals() else float(best_val_loss)
    final_epoch = epoch if 'epoch' in locals() else 0

    # Record this cycle's mean action_loss (KL only) into norm_stats so the
    # NEXT cycle's `run_mcts_collection` can compute strange-traversal
    # probability. Mutating the same `norm_stats` dict the pipeline holds
    # makes it available immediately AND it lands in best.pt via _save_best.
    if cycle_action_count > 0 and norm_stats is not None:
        norm_stats["last_action_loss"] = float(
            cycle_action_sum / cycle_action_count)
        log(f"  last_action_loss = {norm_stats['last_action_loss']:.6f} "
            f"(mean over {cycle_action_count} samples)")

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

    hist.save()
    log(f"=== MCTS Cycle {cycle_id} Complete. "
        f"Best val: {best_val_loss:.6f}, saved=cycle_end ===")
    agent.set_gradient_checkpointing(False)
    return history, run_dir, global_step


def _cleanup_old_snapshots(snapshots_dir, keep_last=3, keep_every=10, log=None):
    """Remove old cycle snapshots, keeping the last `keep_last` plus every
    `keep_every`-th (e.g. cycle 0, 10, 20, ...). Prevents unbounded disk growth.
    """
    if not os.path.isdir(snapshots_dir):
        return
    files = sorted(f for f in os.listdir(snapshots_dir) if f.startswith("cycle_") and f.endswith(".pt"))
    if len(files) <= keep_last:
        return
    keep_set = set(files[-keep_last:])
    for f in files:
        try:
            cid = int(f.replace("cycle_", "").replace(".pt", ""))
        except ValueError:
            continue
        if cid % keep_every == 0:
            keep_set.add(f)
    removed = 0
    for f in files:
        if f not in keep_set:
            os.remove(os.path.join(snapshots_dir, f))
            removed += 1
    if removed and log:
        log(f"  Cleaned up {removed} old cycle snapshot(s), kept {len(files) - removed}")


def _save_best(agent, optimizer, scheduler, norm_stats, ckpt_dir,
               global_step, epoch, val_loss, log, temperature=None,
               cycle_id=None, examples_in_cycle=None, reason="val_improved",
               write_snapshot=True):
    """Save checkpoint to a rolling `best.pt` (always) and optionally a
    per-cycle snapshot `cycles/cycle_<N>.pt`. The rolling file is what
    `_find_best_checkpoint` loads (latest state); the per-cycle snapshot is
    permanent and gated by `save_every_cycles` via `write_snapshot`.
    """
    extra = {
        "step": global_step, "epoch": epoch + 1,
        "cycle_id": cycle_id,
        "examples_in_cycle": examples_in_cycle,
        "save_reason": reason,
    }
    if temperature is not None:
        extra["temperature"] = temperature
    ckpt = make_checkpoint(
        phase=_PHASE, model=agent, optimizer=optimizer, scheduler=scheduler,
        norm_stats=norm_stats, val_loss=val_loss, extra=extra,
    )

    best_path = os.path.join(ckpt_dir, "best.pt")
    atomic_torch_save(ckpt, best_path)

    snapshot_written = False
    if cycle_id is not None and write_snapshot:
        snapshots_dir = os.path.join(ckpt_dir, "cycles")
        os.makedirs(snapshots_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshots_dir, f"cycle_{cycle_id:04d}.pt")
        atomic_torch_save(ckpt, snapshot_path)
        snapshot_written = True

    if snapshot_written:
        _cleanup_old_snapshots(snapshots_dir, keep_last=3, keep_every=10, log=log)

    snap_str = " + snapshot" if snapshot_written else ""
    log(f"  Saved best.pt{snap_str} ({reason}, val={val_loss:.6f}, "
        f"cycle={cycle_id})")
