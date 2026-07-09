# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
All project was written by Claude Code, so you are responsible for every bug in the code

## Running

Current working <version> - v6

```bash
# Always use the venv
source venv/bin/activate

# From repo root
./run.sh --version=<version>
```

Device auto-detected: CUDA → MPS → CPU. All Python commands must use the venv.

**Linux CUDA requirement**: `vm.max_map_count` must be ≥ 1048576 (default 65530 is too low for `expandable_segments:True`). Check: `sysctl vm.max_map_count`. Fix: `sudo sysctl -w vm.max_map_count=1048576` (or persist in `/etc/sysctl.conf`). Without this, the inference server may crash with ENOMEM.
ALL CODE EDITS MUST MODIFY ONLY DIRECTORY /versoins/<version>!
ALL DATA MUST STORE OUTSIDE THE /versions DIRECTORY INTO /data DIRECTORY!

## Project Overview

Poker AI agent (ASI) trained in 6 phases:
1. **GTO EV** (`gto_ev_predict`) — perception + value head learn to predict GTO Expected Value
2. **GTO Probs** (`gto_probs_predict`) — action head learns GTO action distributions (perception + value frozen)
3. **GTO Combined** (`gto_predict`) — perception + value + action trained jointly (modelling + opponent frozen)
4. **Modelling** (`modelling_predict`) — modelling head learns per-action state embeddings for MCTS rollout (all else frozen)
5. **Opponent Action** (`opponent_action_predict`) — opponent_action_head learns range-averaged opponent distributions (all else frozen)
6. **MCTS Self-Play** (`mcts_predict`) — all heads fine-tuned via cyclic collect → train on self-play data

Each phase saves to its own subdirectory. `_find_best_checkpoint()` searches in reverse phase order (mcts first, gto_ev last) to always load the latest trained stage.

## Architecture (`versions/<version>/agent/`)

```
Event sequence (N events)
  → EventSequenceEmbedder: each event → 7 per-card vectors (5 table + 2 hand)
  → + source_embed (table=0 / hand=1) → LayerNorm
  → + opponent_embed (optional GRU-updated per-opponent embedding on hand cards)
  → Encoder (Qwen3 causal, N×7 tokens) + padding mask
  → Mean pool window=7: N×7 → N vectors
  → Decoder (Qwen3 self-attention, N tokens) + padding mask
  → ValueHead (Qwen3 self-attn → masked mean pool → Linear → scalar)
  → ActionHead (Qwen3 self-attn → masked mean pool → Linear → n_actions logits)
  → OpponentActionHead (same architecture as ActionHead)
  → ModellingHead (causal Qwen3 self-attn over decoder output → action-conditioned MLP → (B, n_actions, d_model) at last true position)
```

### EventSequenceEmbedder (`perception/perception.py`)

Each event produces **7 vectors** (CARDS_PER_EVENT=7), one per card slot: `[table_0..4, hand_0, hand_1]`. Each vector combines:
- **Card embedding**: `Embedding(53, d_model)` — indices 0-51 = cards, 52 = no-card. Clamped to [0, 52].
- **7 context embeddings** (shared across all 7 cards): hero_pos, acting_pos, num_players (Embedding lookups), pot+hero-stack (Linear(2→d)), bets (Linear(max_players→d)), action (Linear(n_actions→d)), **per-position stacks vector** (Linear(max_players→d) — B.6.2; a per-seat effective-stack signal, without which stack-aware play is unlearnable). Events must carry a `stacks` list; missing → treated as zeros.
- Combined: `cat(card_emb, 7 context embs)` → `Linear(8d → d)` → `+ source_embed` → `LayerNorm(d)`

**Opponent embedding** (optional, `architecture.opponent_embedding.enabled`): for hand card positions (5, 6), adds a per-opponent GRU-updated embedding. `OpponentEmbeddingTable` stores embeddings by opponent_id; `opponent_gru` advances them from **pre-encoder embedder features** (mean over the table-card slots 0–4 only — A.4.1), processed causally per event in flat sample/event order: each event injects the running state as of that event (no future leak, no cross-sample interleaving). The stored tensor stays in the autograd graph within a forward (truncated BPTT, depth = `gru_window`) and is detached between steps via `detach_all()`; it is never an optimizer parameter.

**Opponent stats vector — HUD** (`architecture.opponent_embedding.stats_enabled`; `versions/v6/PLAN_OPPONENT_ADAPTATION.md` §3): count-based per-opponent action frequencies stored as raw counts (8 buckets = street×facing × 5 action categories) in the same `OpponentEmbeddingTable` (`.stats`, plain numpy, copy-on-write so A.4.5 group-rewind / `clone()` snapshots stay valid). A 49-dim feature vector (Laplace-smoothed in-bucket frequencies + log-count confidence terms — explicit "how well do I know this player" signal) goes through `opp_stats_proj: Linear(49→d)` and is added at the same slots (5,6) on top of the GRU state. Attribution respects the B.2 next-player convention: an event's one-hot action is counted for the PREVIOUS event's acting player, in the previous event's street/facing bucket. Stats ride along in every table flow (phase-6 cross-cycle persistence, eval, server per-(worker,agent) tables). `opp_stats_proj` trains in phases 5 and 6. Off → bit-for-bit legacy path.

**Phase-5 probes** (`architecture.opponent_embedding.style_probe` / `showdown_probe`; plan §2/§4): training-only heads on the acting opponent's GRU state at the decision point (`collect_opp_states=True` → `out["opp_last_states"]`, `out["opp_states_mask"]`). `StyleProbe` (`agent/opponent_action/probes.py`) regresses the canonical 16-dim style vector of the generating agent (`modifiers.build_style_vector`: coverage-weighted category biases [unconditional / low-equity / high-equity] + log-temperature; z-scored across the pool, pool-constant dims masked; targets passed by pipeline as `opponent_action_train.style_targets` keyed by agent NAME). `ShowdownStrengthProbe` regresses the revealed-hand strength percentile (`actor_showdown_strength` labels). Probes train ONLY in phase 5 (owner decision 2026-07-09: by phase 6 agents drift from their initial modifiers, labels stale) and are never used at deployment; they persist in checkpoints, old checkpoints load fine (strict=False).

### Encoder (`perception/encoder.py`)

Qwen3 causal transformer. Receives N×7 per-card tokens. Padding mask → 4D attention mask `(B, 1, 1, S)` with `-inf` for padding. RoPE over full N×7 sequence.

### Perception (`perception/perception.py`)

After encoder: **non-overlapping mean pool with window 7** → `(B, N, d_model)`. Optional GRU opponent embedding update. Then decoder.

When `skip_memory=False`: memory vectors prepended to encoder output → fed to decoder. **Known latent bug**: returned mask doesn't include memory positions; will cause shape mismatch when memory is enabled. Currently `skip_memory=True` everywhere.

`forward_batch` returns `(decoder_output, encoded, mask)`. All heads receive `decoder_output`.

### Decoder (`perception/decoder.py`)

Qwen3 self-attention over encoder output (or `[memory_vectors, encoder_output]`). Own RoPE + padding mask.

### ValueHead (`value/value.py`)

Qwen3 self-attention layers → masked mean pool (clamp(min=1) prevents div-by-zero) → `Linear(d_model, 1)` → scalar.

### ActionHead (`action/action.py`)

Same as ValueHead but `Linear(d_model, n_actions)` → logits.

### OpponentActionHead (`opponent_action/opponent_action.py`)

Same architecture as ActionHead. Predicts range-averaged opponent action distributions from observer's perspective.

### ModellingHead (`modelling/modelling.py`)

**Autoregressive action-conditioned next-decision-state predictor** (redesigned 2026-07, see `versions/v6/PLAN_MODELLING_HEAD_REDESIGN.md` — the old cross-attention-query head had a proven collapse attractor, `analytics/modelling_head_collapse_analysis.pdf`). Causal Qwen3 self-attn stack over decoder output → per-position state `s_t` → `h(t, a) = norm(mlp_out(GELU(mlp_in(cat(s_t, e_a)))))` with `e_a = Embedding(n_actions, d_model)`. Two forwards: `forward(context, mask)` → `(B, n_actions, d_model)` at each example's last true position (same signature/consumers as before — MCTS unchanged); `forward_positions(context, mask, batch_idx, positions, actions)` → `(M, d_model)` for LM-style training. Module-level helpers `build_lm_pairs` (pair convention: source = pre-decision token q−1, action = argmax at post-action q, target = next decision-point token q+1) and `lm_loss` (MSE + `infonce_weight`·InfoNCE over in-batch negatives, τ = `infonce_temperature`) are shared by phases 4 and 6. Targets are always detached (stop-grad); actions conditioned independently (perturbing e_a does not affect other actions' outputs).

### Agent (`agent/agent.py`)

`forward_batch(event_sequences, skip_memory, heads, skip_opponent_emb, opponent_emb_table)`:
- Routes through perception → selected heads (filterable via `heads` set: `"action"`, `"value"`, `"opponent_action"`, `"modelling"`)
- Frozen perception auto-detected via `requires_grad` check → `torch.no_grad()` + `.detach()`

`load_checkpoint(path)`: loads from file or directory. For directories, calls `_find_best_checkpoint()` which searches scenario subdirectories in priority order: mcts_predict → opponent_action_predict → modelling_predict → gto_predict → gto_probs_predict → gto_ev_predict. Within each, picks latest timestamp. Partial loading with `strict=False`. Preserves `norm_stats` from checkpoint as `_checkpoint_norm_stats`.

### GPU Solvers (`gto_utils/`)

4 solver versions (v1–v4). Config param `solver.type` selects which. v3 adds EQR table + weighted combo sampling. v4 reserved.

### HierarchicalMemory (`perception/memory.py`)

Beam-search clustering memory. Currently disabled (`skip_memory=True` everywhere).

### MCTS (`agent/mcts/`)

- `mcts.py`: `MCTS` class — PUCT for hero nodes, prior sampling for opponents. Context built by appending modelling-head action embeddings along the path. `@torch.no_grad()` during search. Dirichlet noise at root. Temperature-scaled visit counts for action selection.
- `game_state.py`: `GameState` — lightweight table tracker (no cards/judger). Mirrors `Table.step()` + `next_turn()` betting logic.
- `collect.py`: `run_mcts_collection()` — plays hands with MCTS decisions, extracts `MCTSTrainingExample` per decision point (events, value_target from actual outcome, action_target from visit distribution, chain of future decisions).
- `terminal_eval.py`: terminal equity evaluation for MCTS nodes.
- `evaluator.py`: neural-net boundary for MCTS. `LocalEvaluator` wraps a live ASI (in-process; bit-for-bit identical to the old direct head calls). `RemoteEvaluator` / `EvalProxy` marshal forwards to the inference server. MCTS calls only `evaluate_root` / `evaluate_leaves`; context assembly (`_build_context`/`_pad_and_stack`) stays in the tree as cheap CPU ops.
- `inference_server.py`: single GPU process holding all agents; batches `ROOT`/`LEAF`/`FORWARD_BATCH` requests ACROSS actors into large forwards.

**Parallel collection** (`mcts_train.n_workers > 1`): CPU actor processes run the tree search + game logic and offload neural-net forwards to the GPU inference server. Within-tree `mcts.batch_size` (16) is **unchanged** — cross-actor batching at the server provides GPU efficiency on top, so search faithfulness is preserved. `n_workers <= 1` runs the original sequential path (LocalEvaluator), bit-for-bit. Actors split `n_hands`, each with its own RNG seed and independent hand stream; the parent merges all examples and runs `_finalize_value_targets` ONCE over them (so the `mcts_value_scale` bootstrap sees every example). `opponent_embedding` uses a per-`(worker, agent)` GRU table on the server (variant B — each actor accumulates over its own stream; deterministic, no cross-actor races). Equity (`terminal_eval`) runs on CPU in actors with action-head range-narrowing routed to the server (`FORWARD_BATCH`); actors never init CUDA. Config knobs (`mcts_train`): `n_workers` (default 1), `server_max_batch` (256), `server_linger_ms` (2). **Caveat**: parallel runs are NOT bitwise-reproducible vs sequential (fp16 context transport + cross-actor batch composition), only distributionally equivalent. IPC sends full contexts in fp16; if IPC-bound, a server-side root cache is the planned v2 optimization.

### Evaluation (`evaluation/evaluate.py`)

Agent-vs-agent evaluation. Loads agents from subdirectories, seats them at tables, plays N hands, reports BB/100. Multi-table batching for GPU efficiency. Agent rotation when pool > table size. Can run standalone: `python -m evaluation.evaluate --config config.json`.

**E.4.2: parallel MCTS in eval** — when MCTS agents are present and `n_tables > 1`, an inference server (`_EvalInferenceServer`) is spun up reusing `agent/mcts/inference_server.py`. MCTS decisions across tables are dispatched concurrently via a `ThreadPoolExecutor`; each thread creates a fresh `MCTS` + `RemoteEvaluator` routing NN forwards to the server. Non-MCTS agents continue using batched action-head forward in the main thread. MCTS agents' models are moved to CPU during eval (the server holds its own GPU copies). The thread pool overlaps MCTS searches with the batched action-head path, improving GPU utilization at high `n_tables`. Config: `evaluation.server_max_batch` (default: `mcts.server_max_batch` or 256), `evaluation.server_linger_ms` (default: `mcts.server_linger_ms` or 2). Falls back to sequential MCTS if the server fails to start or `n_tables <= 1`.

## Config (`versions/<version>/config.json`)

| Section | Purpose |
|---|---|
| `name` | Experiment name, used for save directory |
| `agent_dir` | Path to checkpoint for initial weight loading (`strict=False`) |
| `architecture` | Model dims, layer counts, memory config, opponent_embedding, max_players |
| `dataset` | Data generation: `dataset_dir`, `save_dir`, `n_scenarios`, `val_split`, `n_workers` |
| `game` | `raise_sizes` (dict per street), `big_blind`, `max_stack`, `max_players` |
| `solver` | `type` (v1–v4), `mc_iterations`, `gto_temperature`, `eqr_enabled`, `combo_response_iters`, `weighted_sampling` |
| `gto_ev_train` | Phase 1 hyperparams |
| `gto_probs_train` | Phase 2 hyperparams |
| `gto_train` | Phase 3 hyperparams + `action_loss_weight` |
| `modelling_train` | Phase 4 hyperparams + `recon_weight`, `infonce_weight`, `infonce_temperature` |
| `opponent_data` | Opponent data generation: `agents_dir`, `n_hands`, `action_temperature` (dead `range_threshold` removed — B.7.3). Pool IDs are round-robin-BOUND to agents (P0, PLAN_OPPONENT_ADAPTATION §1): one persistent ID = one style across hands; binding persisted in `meta.json`, per-scenario `acting_agent`/`acting_agent_idx` labels + `actor_showdown_strength` (§4, seeded MC) |
| `opponent_action_train` | Phase 5 hyperparams + `style_probe_weight`, `showdown_probe_weight` (probe aux losses; `style_targets` injected by pipeline from `multi_agent` modifiers) |
| `mcts` | MCTS search params: `n_simulations`, `c_puct`, `dirichlet_alpha/epsilon`, `temperature` |
| `mcts_train` | Phase 6: `n_cycles`, `n_hands_per_cycle`, `value/action/chain_weight`, `gru_window` (opponent-GRU truncated-BPTT depth in phase-6 forwards; default 1), `n_terminal_values` (uniformly RANDOM tree terminals per example for direct value-head supervision; replaced `terminal_value_k_worst`/`k_best` tails-only selection, which trained the value head only on extreme outcomes). Opponent-embedding tables persist ACROSS cycles (pipeline owns them: collection dict + per-agent `agent_info["opp_emb_table"]`) — long-run context about repeat opponents accumulates instead of resetting every cycle |
| `evaluation` | `agents_dir`, `n_hands`, `n_tables`, `use_opponent_emb`, `server_max_batch`, `server_linger_ms` |
| `pipeline` | Flags: `run_gto_ev`, `run_gto_probs`, `run_gto_training`, `run_modelling`, `run_opponent_data`, `run_opponent_action_train`, `run_mcts_train`, `run_evaluation` |
| `multi_agent` | Agent pool with per-agent modifiers (see below) |

Key params:
- `game.raise_sizes`: dict with keys `preflop`, `flop`, `turn`, `river` → list of raise fractions. `n_actions = len(raise_sizes[street]) + 3` (fold + call + raises + all-in).
- `dataset.dataset_dir`: path to existing raw dataset. Empty → generate new.
- `solver.gto_temperature`: action distribution sharpness in data generation.

## Multi-Agent Training (`multi_agent` config section)

Each agent gets full independent training on modifier-adjusted targets. `save_dir`: relative → `data/<version>/<save_dir>/<agent_name>/`, absolute → `<save_dir>/<agent_name>/`.

### Modifier types (`agent/train_scenarios/modifiers.py`)

**`action_bias`** — `modified_ev[i] = ev[i] + |ev[i]| * factor`. Factor > 0 boosts, < 0 penalizes.
**`conditional_bias`** — same formula, gated by condition (`"equity < 0.3"`, `"pos > 5"`, etc.).
**`temperature`** — overrides `gto_temperature` for softmax recomputation.

All bias modifiers accumulate factors per action from original values, then apply once. `apply_modifiers()` shallow-copies each scenario (`{**s}` + fresh lists for mutated fields); originals stay unmutated. When recomputing `action_probs` from modified EVs, the scenario's persisted `legal_mask` is applied (masked_fill −inf before the tempered softmax) exactly like generation does; old-format scenarios without `legal_mask` warn once per run and proceed unmasked — regenerate the dataset.

### Action selectors

Named: `"fold"`, `"call"`, `"raises"`, `"allin"`, `"small_raises"`, `"big_raises"`, `"aggressive"`.
Explicit: `[0, 1, 52]`. Slice: `"15:35"`, `"2:52:2"`.

## Data Generation

### GTO Data (`generation/generate.py`)

Simulates poker hands with GTO-sampled actions. Each sample: `events` (variable-length event dicts, each with a per-position `stacks` vector — B.6.2), `ev_target`, `action_evs` (raw, unmasked), `action_probs`, `legal_mask` (per-action bools — the mask used for the policy softmax, persisted so training-time modifiers can recompute `action_probs` with the same mask), metadata (`equity`, `pot`, `facing_bet`, `stack`, etc.). Saved **raw** (unnormalized).

Audit-B fixes baked in here: per-seat independent starting stacks (B.6.1); event `acting_pos` uses the next-player convention, consistent with collect/eval (B.2); `action_probs`/sampling mask illegal+dominated actions via the same `GameState.get_legal_action_mask` the MCTS/eval paths use — fold is dropped when checking is free and capped raise bins collapse into the single all-in (B.5.2/B.5.3); preflop raises are classified open vs 3bet+ by raises-this-street (B.5.6); raise-size→solver-frac conversion divides by `pot+facing_bet` (B.5.1); `max_actions = 6*num_players+8`, truncations warned not silently dropped (B.6.3). The v3 solver's value-bet pot is `pot + raise_amount + (raise_amount − facing_bet)` (B.1) and raise EV is multiway-aware (`Π p_fold` to win uncontested, else showdown vs callers — B.5.4).

### Opponent Data (`generation/generate_opponent.py`)

Simulates hands with trained agents, tracking per-player hand ranges via a soft-Bayes belief (`opponent_data.bayes`). At each decision: runs inference over the ESS-truncated forward subset of the acting player's range → belief-weighted average action distribution → training target. Shared event format (all hands unmasked, with a per-position `stacks` vector); data loader converts to per-observer masking.

P0 (PLAN_OPPONENT_ADAPTATION §1): persistent pool IDs are deterministically round-robin-bound to agents — `seated` comes from the binding, NOT a per-hand random draw — so cross-hand GRU/stats accumulation per ID sees one consistent style (train/deploy parity: at deployment `opponent_id` IS the agent name). Scenarios persist `acting_agent`/`acting_agent_idx` (style-probe labels) and, on showdown-ended hands, `actor_showdown_strength` on each live actor's LAST decision scenario (`_label_showdown_strengths`: strength percentile of the fixed hand vs a random combo on the full board, seeded MC over 256 combos, ties 0.5, board-colliding hands skipped).

Audit-B fixes: fixed hands are physically consistent — disjoint from the revealed board and from other players' fixed hands; an acting hand that collides with a later board card is re-sampled, and observers whose fixed hand lands on the board are dropped from `hero_positions` (B.4). Un-observed (non-forward) combos in the Bayes update get the **likelihood floor** (min observed tempered likelihood), not an implicit 1.0, so the excluded tail does not inflate (B.7.1). Each scenario persists the per-combo distributions + weights + combo card-pairs; the loader **re-averages the target per observer**, excluding combos blocked by that observer's own cards (B.7.2).

**Parallel generation** (`opponent_data.n_workers > 1`): reuses the same GPU inference server as MCTS (`agent/mcts/inference_server.py`). CPU actor processes play hands + range bookkeeping; the only model forward — the action-head combo inference in `_compute_range_probs` — is routed to the server via `EvalProxy` (`FORWARD_BATCH`). No opponent embedding here (`skip_opponent_emb=True`), so no server-side state. Hands split across actors with contiguous `hand_id` offsets; parent merges scenarios and saves once (no per-`save_every_hands` incremental save in parallel mode). Config: `opponent_data.n_workers` (default 1 = sequential, unchanged), `server_max_batch` (256), `server_linger_ms` (2).

## Normalization

Computed per-agent at training time. Norm stats saved in `best.pt` as `_checkpoint_norm_stats`. Downstream phases (modelling, opponent_action) reuse checkpoint norm stats to avoid distribution mismatch with frozen perception.

**EV**: `ev_target / max(pot + facing_bet, big_blind)` → z-score.
**Events**: pot, stack, bets → z-score. The per-position `stacks` vector (B.6.2) reuses the hero-`stack` mean/std (same units). Blinds → z-score.
**Action EVs** (modelling only): same scale as EV target, normalized separately via `_normalize_action_evs()`.

## Pipeline Flow (`pipeline.py`)

```
config.json → pipeline.py
  │
  ├─ Load/generate raw GTO dataset (shared)
  │
  ├─ [multi-agent] For each agent:                    ← saves to <save_dir>/<agent>/
  │    ├─ apply_modifiers → modified scenarios
  │    ├─ GTO EV training → gto_ev_predict/<ts>/best.pt       (perception + value)
  │    ├─ GTO Probs training → gto_probs_predict/<ts>/best.pt (action head)
  │    ├─ GTO Combined → gto_predict/<ts>/best.pt             (perception + value + action)
  │    └─ Modelling → modelling_predict/<ts>/best.pt          (modelling head only)
  │
  ├─ Opponent data generation (loads agents from <save_dir>/<agent>/)
  │
  ├─ [multi-agent] Opponent action training per agent:
  │    └─ → opponent_action_predict/<ts>/best.pt              (opponent_action head)
  │
  └─ MCTS cyclic self-play (n_cycles):
       For each cycle:
         ├─ Load latest checkpoints for all agents (from <save_dir>/<agent>/)
         ├─ Collect: play hands with MCTS → MCTSTrainingExample per decision
         └─ Train each agent → mcts_predict/<ts>/best.pt      (all heads unfrozen)
       Next cycle loads updated agents ✓
```

**Checkpoint chain**: each phase saves to its own scenario subdirectory. `_find_best_checkpoint()` always finds the latest phase. After each training step, pipeline reloads the best checkpoint (including modelling).

**Between phases**: fresh ASI is created and loaded from `agent_base` directory → `_find_best_checkpoint` finds the most recent stage's best.pt. No stale weights.

### Training details per phase

| Phase | Loss | Frozen | Trainable |
|---|---|---|---|
| GTO EV | SmoothL1 (Huber) | action, modelling, opponent_action | perception, value |
| GTO Probs | KL divergence | perception, value, modelling, opponent_action | action |
| GTO Combined | SmoothL1 + `action_loss_weight` * KL | modelling, opponent_action | perception, value, action |
| Modelling | SmoothL1(predicted_evs) + `recon_weight` * (MSE + `infonce_weight`·InfoNCE)(LM next-decision-state pairs) | perception, value, action | modelling |
| Opponent Action | KL + `style_probe_weight`·MSE(style vector, z-scored, masked dims) + `showdown_probe_weight`·MSE(showdown strength) — probe losses on `opp_last_states`, absent labels masked to 0 | perception, value, action, modelling | opponent_action (+ opponent_gru, opp_stats_proj, style_probe, showdown_probe if enabled) |
| MCTS | `value_weight`·SmoothL1(root) + `action_weight`·KL(root) + `chain_weight`·KL(chain) + `recon_weight`·(MSE + `infonce_weight`·InfoNCE)(LM pairs: root-sequence pairs [always teacher-forced] + rolled chain-step embeddings vs last true token of the real next decision — one pooled InfoNCE batch, no depth gamma) + `value_chain_weight`·SmoothL1(chain). Value-target = `clip(α·(root.Q rescaled) + (1−α)·(equity_realized/new_scale), ±clip)`; equity-anchored root.Q (terminal_eval) + equity-based realized outcome at actual hand-end. `teacher_forcing` p_start=p_end=0.7 (fixed 0.7/0.3 mix, no decay) | — | all heads |

All phases: LR warmup (100 steps) + cosine decay, gradient clipping (max_norm=1.0), early stopping, intra-epoch validation, AMP when available.

### MCTS value-target normalization — IMPORTANT

Step 1 and Step 2 both implemented (2026-05). Full plan + math + reasoning
lives in [`versions/v5/PLAN_MCTS_VALUE_REDESIGN.md`](versions/v5/PLAN_MCTS_VALUE_REDESIGN.md).
Read it before changing anything in `_make_terminal_evaluator`,
`run_mcts_collection`, `_finalize_value_targets`, `terminal_eval.py`, or
`re_backup_terminals`. Highlights:

- **The 97% fold rate of trained agents was a normalization bug, not an
  architecture bug.** Per-agent z-score `(ratio − μ)/σ` with `μ < 0` made
  fold's deterministic `outcome = 0` land at `−μ/σ ≈ +0.13…+0.25` in
  z-space while internal value_head outputs averaged to 0 → fold won PUCT
  systematically. Step 1 removed the mean shift.
- **`value_target = root.Q` in `collect_training_data` is a placeholder**
  overwritten by `run_mcts_collection`. Value head trains on the hybrid
  `α · root_q_ratio + (1−α) · realized` after Step 1; the realized half is
  the **equity-based** chip delta after Step 2 (was: single-sample MC).
- **Terminal Q has two stages**:
  - During search (v6): `_deterministic_terminal_value` in `mcts.py` —
    fold-family terminals exact (chips / `search_scale`), showdown via
    value head. (The v5 `_make_terminal_evaluator` heuristic
    `pot/n_active − invested` no longer exists in v6.)
  - Post-hand: `evaluate_all_terminals` in `terminal_eval.py` overrides
    every terminal across every MCTS tree. Fold = deterministic; showdown
    = `equity * pot − hero_invested` via `gpu_equity_v2` with opponent
    range narrowing through each player's `action_head`. Output is
    divided by `value_scales_by_position[hero_pos]` so it matches
    ancestor-W scale.
  - `re_backup_terminals` propagates the override as **delta**
    `(new_Q − old_Q) * N` — replaces heuristic contribution, doesn't add
    on top of it.
- **Equity-based realized outcome** (Step 2B):
  `realized = equity_hero * final_pot − chips_invested_from(t)` for
  showdown hands; deterministic for fold-terminated hands.
  `equity_by_hero` is computed once per hero and reused across all chain
  steps of all that hero's examples in the hand (only
  `chips_invested_from(t+1+i)` varies per step).
- **`final_credits` is NOT used for showdown realized outcome.** The
  engine has a known pot-distribution mis-accounting on multi-street
  showdowns (`Table.next_turn` resets `self.bets` at each street, then
  passes river-only `self.bets` to `Judger.get_reward`). Step 2 tracks
  `cumulative_bets` via `Table.step`'s 4th return value and derives
  `credits_pre_distribution = initial_credits − cumulative_bets`.
- **Single scale across root + chain**. `_make_terminal_evaluator` and
  `_finalize_value_targets` use one bootstrapped per-agent
  `mcts_value_scale = std(realized_chip_deltas)` in chips (state-
  dependent denoms cause extreme tails). Stored in `norm_stats`.
- **Hybrid `α·Q + (1−α)·realized`** controlled by
  `mcts_train.value_target_alpha`. Pure MC (`α=0`) = AlphaZero-style and
  safe. Pure TD (`α=1`) = self-bootstrap, risks policy fixed-point
  drift. Default `0.5`.
- **Q normalization** in `_finalize_value_targets` passes through
  `search_scale → chips → new_scale`. NOT double-division: the
  multiplication undoes terminal_evaluator's division so we can re-apply
  the cycle's updated `new_scale`. When stable
  (`search_scale == new_scale`), the ratio is 1 → identity. Plan §4.1.
- **Per-cycle EMA scale update** (`mcts_train.value_scale_ema`, default
  0.1): every cycle, `mcts_value_scale ← (1−ema)·old + ema·cycle_scale`
  where `cycle_scale` = `_robust_scale` (MAD·1.4826) of the cycle's
  realized chip deltas. Cycle 0 (no existing scale) does a full
  bootstrap; `ema = 0` reuses the stored scale. This replaced the
  one-shot `value_norm_rebootstrap_every` rebootstrap (removed 2026-07:
  the axis shock at the rebootstrap cycle degraded search and collapsed
  agents — the value head gets only ~10–25 grad steps/cycle and cannot
  re-map a jumped target axis).
- **Legacy `norm_stats` keys** `mcts_ev_mean`, `mcts_ev_std`,
  `mcts_ev_n_samples`, `mcts_ev_ratio_min`, `mcts_ev_ratio_max` stay in
  checkpoints for backward-compat but are unused. New keys:
  `mcts_value_scale` (chips), `mcts_value_scale_n_samples`,
  `mcts_value_chip_min`, `mcts_value_chip_max` (last three now record
  the current cycle's values). `pipeline.py:_LEGACY_MCTS_NORM_KEYS`
  kept for checkpoint backward-compat.
- **Belief-based equity ≠ per-hand zero-sum.** Each hero's equity is vs
  the OPPONENT'S range (not opponent's actual cards), so
  `Σ_p equity_p ≠ 1` on a single hand. Zero-sum holds in expectation
  across hands. This is correct: training signal must come from what
  hero can estimate at decision time, not omniscient ground truth.

**Modelling forward**: perception(frozen) → modelling_head produces per-action embeddings (at last true position) → each action embedding appended to perception output → value_head(frozen params, grad flows through) predicts EV per action. LM loss (replaced the old reconstruction loss): `forward_positions` at every decision position vs the next decision-point token (detached), MSE + InfoNCE.

**MCTS chain forward**: perception → root value + action predictions. Then for each subsequent decision in hand: modelling_head → action embedding for taken action → extend context → predict next action distribution (action_head if hero, opponent_action_head if opponent).

### Output structure

```
data/<version>/<save_dir>/<agent_name>/
  ├─ gto_ev_predict/<timestamp>/best.pt, history.pt
  ├─ gto_probs_predict/<timestamp>/best.pt, history.pt
  ├─ gto_predict/<timestamp>/best.pt, history.pt
  ├─ modelling_predict/<timestamp>/best.pt, history.pt
  ├─ opponent_action_predict/<timestamp>/best.pt, history.pt
  └─ mcts_predict/<timestamp>/best.pt, history.pt     (one per cycle)

data/<version>/<name>/
  ├─ configs/<timestamp>.json
  ├─ logs/<timestamp>.txt
  └─ dataset/<timestamp>/dataset.pt                    (raw, shared)
```

`best.pt`: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `norm_stats`, `val_loss`, `step`, `epoch`, optional `temperature`.
