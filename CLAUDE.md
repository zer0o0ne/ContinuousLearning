# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
All project was written by Claude Code, so you are responsible for every bug in the code

## Running

Current working <version> - v5

```bash
# Always use the venv
source venv/bin/activate

# From repo root
./run.sh --version=<version>
```

Device auto-detected: CUDA → MPS → CPU. All Python commands must use the venv.
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
  → ModellingHead (learnable action queries → cross-attn to decoder output → Qwen3 self-attn → (B, n_actions, d_model))
```

### EventSequenceEmbedder (`perception/perception.py`)

Each event produces **7 vectors** (CARDS_PER_EVENT=7), one per card slot: `[table_0..4, hand_0, hand_1]`. Each vector combines:
- **Card embedding**: `Embedding(53, d_model)` — indices 0-51 = cards, 52 = no-card. Clamped to [0, 52].
- **6 context embeddings** (shared across all 7 cards): hero_pos, acting_pos, num_players (Embedding lookups), pot+stack (Linear(2→d)), bets (Linear(max_players→d)), action (Linear(n_actions→d))
- Combined: `cat(card_emb, 6 context embs)` → `Linear(7d → d)` → `+ source_embed` → `LayerNorm(d)`

**Opponent embedding** (optional, `architecture.opponent_embedding.enabled`): for hand card positions (5, 6), adds a per-opponent GRU-updated embedding. `OpponentEmbeddingTable` stores embeddings by opponent_id; `opponent_gru` updates them from encoder output after each forward. Embeddings do NOT require grad — updated only by GRU output replacement.

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

**Cross-attention architecture**: learnable `Embedding(n_actions, d_model)` as queries attend to decoder output (keys/values) via Qwen3-style cross-attention (GQA + QK-norm + RoPE), then refine via Qwen3 self-attention + FFN. Output: `(B, n_actions, d_model)` — one embedding vector per action. Used during MCTS rollout to extend context with action representations.

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
| `modelling_train` | Phase 4 hyperparams + `recon_weight` |
| `opponent_data` | Opponent data generation: `agents_dir`, `n_hands`, `action_temperature`, `range_threshold` |
| `opponent_action_train` | Phase 5 hyperparams |
| `mcts` | MCTS search params: `n_simulations`, `c_puct`, `dirichlet_alpha/epsilon`, `temperature` |
| `mcts_train` | Phase 6: `n_cycles`, `n_hands_per_cycle`, `value/action/chain_weight` |
| `evaluation` | `agents_dir`, `n_hands`, `n_tables`, `use_opponent_emb` |
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

All bias modifiers accumulate factors per action from original values, then apply once. `apply_modifiers()` always deepcopies first.

### Action selectors

Named: `"fold"`, `"call"`, `"raises"`, `"allin"`, `"small_raises"`, `"big_raises"`, `"aggressive"`.
Explicit: `[0, 1, 52]`. Slice: `"15:35"`, `"2:52:2"`.

## Data Generation

### GTO Data (`generation/generate.py`)

Simulates poker hands with GTO-sampled actions. Each sample: `events` (variable-length event dicts), `ev_target`, `action_evs`, `action_probs`, metadata (`equity`, `pot`, `facing_bet`, `stack`, etc.). Saved **raw** (unnormalized).

### Opponent Data (`generation/generate_opponent.py`)

Simulates hands with trained agents, tracking per-player hand ranges. At each decision: runs inference for every hand in acting player's range → averages action distributions → training target. Range narrowing: removes hands where P(chosen)/P(best) < threshold. Shared event format (all hands unmasked); data loader converts to per-observer masking.

**Parallel generation** (`opponent_data.n_workers > 1`): reuses the same GPU inference server as MCTS (`agent/mcts/inference_server.py`). CPU actor processes play hands + range bookkeeping; the only model forward — the action-head combo inference in `_compute_range_probs` — is routed to the server via `EvalProxy` (`FORWARD_BATCH`). No opponent embedding here (`skip_opponent_emb=True`), so no server-side state. Hands split across actors with contiguous `hand_id` offsets; parent merges scenarios and saves once (no per-`save_every_hands` incremental save in parallel mode). Config: `opponent_data.n_workers` (default 1 = sequential, unchanged), `server_max_batch` (256), `server_linger_ms` (2).

## Normalization

Computed per-agent at training time. Norm stats saved in `best.pt` as `_checkpoint_norm_stats`. Downstream phases (modelling, opponent_action) reuse checkpoint norm stats to avoid distribution mismatch with frozen perception.

**EV**: `ev_target / max(pot + facing_bet, big_blind)` → z-score.
**Events**: pot, stack, bets → z-score. Blinds → z-score.
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
| Modelling | SmoothL1(predicted_evs) + `recon_weight` * MSE(state_reconstruction) | perception, value, action | modelling |
| Opponent Action | KL divergence | perception, value, action, modelling | opponent_action (+ opponent_gru if enabled) |
| MCTS | `value_weight`·SmoothL1(root) + `action_weight`·KL(root) + `chain_weight`·KL(chain) + `recon_weight`·MSE + `value_chain_weight`·SmoothL1(chain). Value-target = `clip(α·(root.Q rescaled) + (1−α)·(equity_realized/new_scale), ±clip)`; equity-anchored root.Q (terminal_eval) + equity-based realized outcome at actual hand-end | — | all heads |

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
  - During search: `_make_terminal_evaluator` uses fast heuristic
    `pot/n_active − invested` (chips / `search_scale`). Good enough for
    selection.
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
  the cycle's freshly bootstrapped `new_scale`. When stable
  (`search_scale == new_scale`), the ratio is 1 → identity. Plan §4.1.
- **Legacy `norm_stats` keys** `mcts_ev_mean`, `mcts_ev_std`,
  `mcts_ev_n_samples`, `mcts_ev_ratio_min`, `mcts_ev_ratio_max` stay in
  checkpoints for backward-compat but are unused. New keys:
  `mcts_value_scale` (chips), `mcts_value_scale_n_samples`,
  `mcts_value_chip_min`, `mcts_value_chip_max`.
  `pipeline.py:_MCTS_NORM_KEYS` controls clearing on
  `value_norm_rebootstrap_every`.
- **Belief-based equity ≠ per-hand zero-sum.** Each hero's equity is vs
  the OPPONENT'S range (not opponent's actual cards), so
  `Σ_p equity_p ≠ 1` on a single hand. Zero-sum holds in expectation
  across hands. This is correct: training signal must come from what
  hero can estimate at decision time, not omniscient ground truth.

**Modelling forward**: perception(frozen) → modelling_head produces per-action embeddings → each action embedding appended to perception output → value_head(frozen params, grad flows through) predicts EV per action. Reconstruction loss: action embedding for taken action ≈ next perception state.

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
