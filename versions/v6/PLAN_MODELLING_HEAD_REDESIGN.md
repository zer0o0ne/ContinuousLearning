# PLAN: Modelling Head Redesign — autoregressive next-decision-state prediction

Status: APPROVED 2026-07-02. Decisions fixed with the owner:
(1) loss = MSE + InfoNCE; (2) target = next DECISION-POINT decoder token
(runout marginalized); (3) fixed teacher-forcing mix 0.7/0.3 (no decay);
(4) stop-grad targets only, no EMA target network.

## 1. Motivation

`analytics/modelling_head_collapse_analysis.pdf` proved the collapsed state
(all per-action embeddings equal) is the only attractor of the old training
dynamics: (1) O(1/(L+d+1)) mean-pool gradient dilution, (2) action-agnostic
recon target, (3) self-reinforcing MCTS visit targets, (4) no restoring force.

The redesign makes the head an autoregressive, action-conditioned
next-decision-state predictor trained LM-style with teacher forcing:

- **Requirement 1 (O(1) gradient)**: the LM loss applies directly to the
  head's output at each position — no concatenation + mean-pool on the
  gradient path.
- **Requirement 2 (action-conditional targets)**: h(t, a_t) is supervised
  against the REAL state that followed a_t; different actions in similar
  states pull toward different targets. InfoNCE adds the contrastive
  coupling that plain regression lacks (in LM the softmax couples the
  vocabulary; independent regression vectors give zero gradient to untaken
  actions — InfoNCE restores separation pressure and prevents norm
  collapse).
- **Requirement 3 (decoupled from MCTS targets)**: teacher-forced
  next-state prediction on real trajectories is collapse-independent.

## 2. New architecture (`agent/modelling/modelling.py`)

Replace cross-attention-with-action-queries by:

```
context (B, N, D)  = decoder output (perception_out)
  → causal Qwen3 self-attn stack, n_modelling_layers × Qwen3DecoderLayer
    (RoPE position_ids = arange(N), causal + padding mask — mirrors
    perception/decoder.py conventions)                    → s (B, N, D)
  → action conditioning: e_a = Embedding(n_actions, D)
    h(t, a) = W_out( GELU( W_in( cat(s_t, e_a) ) ) )      # W_in: 2D→d_ff,
                                                          # W_out: d_ff→D
  → final Qwen3RMSNorm on h
```

Causality is inherent: s_t sees only positions ≤ t. The full B×N×A×D tensor
is NEVER materialized — h is computed only at requested (position, action)
pairs.

### Forward modes

1. `forward(context, mask)` → `(B, n_actions, D)` — h at each example's
   LAST true position (lengths−1) for ALL actions. **Signature and output
   identical to the old head** — MCTS (`evaluator.py`, `inference_server.py`,
   `mcts.py`), `agent.forward_batch` ("action_embeddings"), and the phase-4
   EV probe keep working unchanged.
2. `forward_positions(context, mask, batch_idx, positions, actions)` →
   `(M, D)` — training mode: compute s once over the full sequence, gather
   `s[batch_idx, positions]`, condition on `actions`. Used by the LM loss.

### state_dict

New param tree under `modelling_head.*` (action_embeddings kept by name;
`self_attn_layers.{i}.*` kept; `cross_norms`/`cross_attns` removed;
new `mlp_in`/`mlp_out`). Old checkpoints load with `strict=False`: the
modelling head reinitializes, everything else is preserved. Phase 4 must be
retrained (was planned anyway — dataset regeneration for `legal_mask`).

## 3. Pair construction (shared convention)

Event stream (collect.py `_play_hands`): index 0 initial snapshot
(action=None); per decision: pre-decision snapshot (action=None) at p, then
post-action snapshot (one-hot) at p+1; next decision's pre-decision at p+2.
Board reveals are implicit in snapshots (no extra tokens); all-in runouts
append no events.

For every event index q whose `action` one-hot has max ≥ 0.5:
- source position = q − 1 (the pre-decision token — the state the decision
  was made in)
- action = argmax(events[q].action)
- target position = q + 1 (the NEXT decision-point token), **only if q+1
  exists** (last decision of a sequence has no target → skipped)

Target vector = `perception_out[bi, q+1].detach()` (stop-grad; the root
value/action losses anchor perception against constant-token degeneracy).
Chance between decisions (new streets) is marginalized: MSE learns the
runout-averaged next state — accepted by design decision (2).

## 4. LM loss (shared helper)

Given predictions P (M, D) and detached targets T (M, D):

- `L_mse = MSE(P, T)`
- `L_nce = CrossEntropy( cos(P_i, T_j) / τ , diag )` over in-batch
  negatives (all targets of the mini-batch; skipped when M < 2)
- `L_lm = L_mse + infonce_weight · L_nce`

Config: `infonce_weight` (0.5), `infonce_temperature` τ (0.1) — added to
BOTH `modelling_train` and `mcts_train`.

## 5. Phase 4 (`modelling_predict/train.py`)

- **EV loss unchanged**: `forward` mode 1 → append h(last, a) at the true
  length position per action → frozen value head → SmoothL1 vs
  `action_evs`. (This supervises ALL actions per state via solver EVs —
  the strongest anti-collapse signal; kept as-is.)
- **`_reconstruction_loss` replaced** by the LM loss of §3–4 over all
  decision positions of every sequence (dense supervision; the old code
  supervised only the final-position embedding against a t+1 target).
  Weight: existing `recon_weight` (0.8) gates `L_lm`.

## 6. Phase 6 (`mcts_predict/train.py`)

- **Root-sequence LM loss (new, always teacher-forced)**: pairs from §3 on
  the root `events` prefix, `forward_positions` on root perception_out.
  Dense, collapse-independent signal every step.
- **Rolled-context recon replaced**: at chain depth d, the already-computed
  `emb = batch_embs[idx, step.action_taken]` (h at the rolled context's
  last position) is supervised directly against the LAST true token of
  `chain_perception_out[tf_idx]` (the real next decision's pre-decision
  token, detached) — no ctx_wn mean-pool, no pooled target. This is the
  0.3 "self-extended" share: the head learns to correct its own drift.
- Both pair sets share one `L_lm` (MSE + InfoNCE), gated by `recon_weight`
  (0.5).
- **Teacher forcing**: config `teacher_forcing.p_start = p_end = 0.7`
  (fixed mix, no code change — the linear decay formula yields a
  constant). The TF gate keeps deciding whether chain PREDICTION contexts
  are real or rolled, exactly as now.
- `stop_grad_old_embs` stays true; terminal-value loss unchanged
  (embeddings were already detached there); chain KL / chain value /
  entropy / root losses unchanged.

## 7. What does NOT change

- `mcts.py` / `evaluator.py` / `inference_server.py` — mode-1 signature is
  identical; nodes still store `act_embs[0, a].detach()`; `_build_context`
  still appends embedding tokens.
- `agent.py forward_batch` — same call, same "action_embeddings" key.
- Datasets, collect.py, GameState, value/action/opponent heads.

## 8. Residual risks (accepted, monitored)

- Untaken actions still get no MSE gradient; InfoNCE + the phase-4 EV
  probe (all 14 actions supervised) cover this. Watch per-action embedding
  cosine-spread in analytics.
- Exposure bias beyond depth 1 in rolled contexts — mitigated by the fixed
  0.3 rolled share; embeddings appended during rollout remain detached
  (stop_grad_old_embs), so depth>1 drift is corrected only via the
  depth-d supervision, not through BPTT.
- Moving perception targets in phase 6 — anchored by root losses;
  if analytics show target drift, add an EMA target net later (explicitly
  deferred by owner decision 4).
