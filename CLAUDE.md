# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

Current working <version> - v1

```bash
# Always use the venv
source venv/bin/activate

# From repo root
./run.sh --version=<version>
```

Device auto-detected: CUDA → MPS → CPU. All Python commands must use the venv.
ALL CODE EDITS MUST MODIFY ONLY DIRECTORY /versoins/<version>!

## Project Overview

Poker AI agent (ASI) trained in 3 phases:
1. **GTO EV prediction** — perception + value head learn to predict GTO Expected Value
2. **Action training** — cross-entropy on MCTS best solutions
3. **Full pipeline** — perception + action + value with MCTS-based loss on game history

Current working <version> - v1. ALL CODE EDITS MUST MODIFY ONLY DIRECTORY /versoins/<version>!

## Architecture (`versions/<version>/agent/`)

ALL CODE EDITS MUST MODIFY ONLY DIRECTORY /versoins/<version>!

```
Event sequence (N events)
  → EventSequenceEmbedder: each event → 7 per-card vectors (5 table + 2 hand)
  → + source_embed (table=0 / hand=1) → LayerNorm
  → Encoder (Qwen3 causal, N×7 tokens) + padding mask
  → Mean pool window=7: N×7 → N vectors
  → Decoder (Qwen3 self-attention, N tokens) + padding mask
  → ValueHead (Qwen3 → masked mean pool → Linear → scalar)
  → ActionHead (Qwen3 → masked mean pool → Linear → n_actions logits)
```

### EventSequenceEmbedder (`perception/perception.py`)

Each event produces **7 vectors** (CARDS_PER_EVENT=7), one per card slot in fixed order: `[table_0..4, hand_0, hand_1]`. Each vector combines:
- **Card embedding**: `Embedding(53, d_model)` — indices 0-51 = cards, 52 = no-card (unrevealed). Clamped to [0, 52].
- **6 context embeddings** (shared across all 7 cards): hero_pos, acting_pos, num_players (Embedding lookups), pot+stack (Linear(2→d)), bets (Linear(max_players→d)), action (Linear(n_actions→d))
- Combined: `cat(card_emb, 6 context embs)` → `Linear(7d → d)` → `+ source_embed` → `LayerNorm(d)`

**Source embedding**: `Embedding(2, d)` — distinguishes table cards (index 0) from hand cards (index 1). Added before LayerNorm.

**No blind embedding**: blinds are fixed and normalize to 0; used only during data generation.

`forward_batch` outputs `(B, max_events×7, d_model)` with mask `(B, max_events×7)`.

### Encoder (`perception/encoder.py`)

Qwen3 causal transformer (decoder layers with built-in causal masking). Receives N×7 per-card tokens. Padding mask converted to 4D attention mask `(B, 1, 1, S)` with `-inf` for padding positions. RoPE positional encoding over full N×7 sequence — cards within an event cross-attend, and all previous events' cards are visible.

### Perception (`perception/perception.py`)

After encoder: **non-overlapping mean pool with window 7** compresses each event's 7 card vectors into 1:
```python
encoded = encoded.view(B, S // 7, 7, D).mean(dim=2)  # (B, N, d_model)
mask = mask[:, ::7]  # (B, N)
```
Output to decoder/memory/heads is `(B, N, d_model)` — same interface as before per-card change.

When `skip_memory=False`: memory vectors are prepended to encoder output with mask of all-ones, then fed to decoder.

### Decoder (`perception/decoder.py`)

Qwen3 self-attention over encoder output (or `[memory_vectors, encoder_output]`). Has its own RoPE and padding mask. Returns `(B, seq_len, d_model)`.

### ValueHead (`value/value.py`)

Qwen3 layers → **masked mean pool** (only real positions, not padding) → `Linear(d_model, 1)` → scalar. Padding mask passed through attention layers and pooling.

### ActionHead (`action/action.py`)

Same structure as ValueHead but outputs `Linear(d_model, n_actions)` logits. Also uses masked mean pool + padding mask in attention.

### Agent (`agent/agent.py`)

`forward_batch` routes data: perception → value_head + action_head. When perception is frozen (detected via `requires_grad` check), wraps perception forward in `torch.no_grad()` + `.detach()` to save memory/compute.

### GPU Solvers (`gto_utils/`)

`gpu_solver.py` (v1, random opponent ranges) and `gpu_solver_v2.py` (v2, position-based ranges with MDF opponent response modeling). Solver choice is a config param.

### HierarchicalMemory (`perception/memory.py`)

Beam-search clustering memory. Disabled during GTO training (`skip_memory=True`), activated in later phases.

## Config (`versions/<version>/config.json`)

Top-level sections:

| Section | Purpose |
|---|---|
| `name` | Experiment name, used for save directory |
| `agent_dir` | Path to checkpoint for weight loading (partial load supported, `strict=False`) |
| `architecture` | Model dimensions, layer counts, memory config, max_players (up to 9-max) |
| `dataset` | Data generation and split params: `dataset_dir`, `n_scenarios`, `val_split` |
| `game` | Table params: `table_bins` (raise quantization), `big_blind`, `max_stack`, `max_players` |
| `solver` | GTO solver: `type` (v1/v2), `mc_iterations`, `gto_temperature`, MDF params |
| `gto_ev_train` | Step 1 training hyperparams: `lr`, `batch_size`, `epochs`, `val_every`, `interrupt_after_fails` |
| `gto_probs_train` | Step 2 training hyperparams (same keys as gto_ev_train) |
| `pipeline` | Which steps to run: `run_gto_ev`, `run_gto_probs` |
| `multi_agent` | Optional: multi-agent training with target modifiers (see below) |

Key params:
- `dataset.dataset_dir`: path to existing raw dataset. If empty, generates new. If path doesn't exist, creates dir and generates there.
- `n_actions` = `game.table_bins` + 3 (fold + call + bins + all-in). Currently 53.
- `solver.gto_temperature`: controls action distribution sharpness in data generation.

## Multi-Agent Training (`multi_agent` config section)

Trains multiple agent "personalities" from one shared dataset. Each agent gets **full independent training** (perception + value + action) on modified targets.

```json
"multi_agent": {
    "save_dir": "pool",
    "agents": [
        {"name": "gto_pure", "modifiers": []},
        {"name": "lag", "modifiers": [
            {"type": "action_bias", "actions": "aggressive", "factor": 0.35}
        ]}
    ]
}
```

`save_dir`: relative → `data/<version>/<save_dir>/<agent_name>/`, absolute → `<save_dir>/<agent_name>/`.

### Modifier types (`agent/train_scenarios/modifiers.py`)

**`action_bias`** — `modified_ev[i] = ev[i] + |ev[i]| * factor`. Factor > 0 boosts, < 0 penalizes.

**`conditional_bias`** — same formula, gated by condition. Conditions: `"equity < 0.3"`, `"equity > 0.6"`, `"pos < 3"`, `"pos > 5"`.

**`temperature`** — overrides `gto_temperature` for softmax recomputation.

All bias modifiers accumulate factors per action from original values, then apply once (no cascading).

### Action selectors

Named: `"fold"`, `"call"`, `"raises"`, `"allin"`, `"small_raises"`, `"big_raises"`, `"aggressive"` (raises+allin).
Explicit list: `[0, 1, 52]`.
Slice: `"15:35"`, `"2:52:2"`.

### Modifier pipeline

`apply_modifiers(scenarios, modifiers)` → deepcopy always → modify `action_evs` → recompute `ev_target = max(modified_evs)` + `action_probs = softmax(modified_evs / (bb * temp))`. Original dataset untouched.

## Data Generation (`agent/train_scenarios/generation/generate.py`)

Uses `env/table.py` to simulate poker hands with GTO-sampled actions. Each sample contains:
- `events`: sequence of event dicts (variable length, 2-36 events)
- `ev_target`: max EV across all actions (scalar, raw chips)
- `action_evs`: per-action EV vector (n_actions values)
- `action_probs`: softmax(action_evs / normalizer)
- Metadata: `equity`, `pot`, `facing_bet`, `stack`, `hero_invested`, `num_players`

Dataset is always saved **raw** (unnormalized). Normalization is per-agent at training time.

The v2 solver narrows opponent ranges by position and action history, and models fold/call/reraise responses via MDF.

## Normalization (`_compute_norm_stats` / `_normalize_scenarios`)

Computed per-agent at training time (not at generation). Norm stats saved in `best.pt` checkpoint as agent parameter.

**EV normalization**: `ev_target / max(pot + facing_bet, big_blind)` → z-score. Removes pot-size dependence.

**Event-level**: pot, stack, bets → z-score. Blinds → z-score (kept in data but not used by perception).

## Data Flow

```
config.json → pipeline.py
  ├─ Load/generate raw dataset (once, shared)
  ├─ Save config snapshot to data/<version>/<name>/configs/<timestamp>.json
  │
  ├─ [single-agent] Pass raw scenarios → train_gto_ev → train_gto_probs
  │
  └─ [multi-agent] For each agent:
       ├─ apply_modifiers(raw_scenarios, agent.modifiers) → modified copy
       ├─ Fresh ASI + train_gto_ev(modified) → perception + value trained
       └─ train_gto_probs(modified) → action head trained (perception + value frozen)

Each train function:
  scenarios_override → deepcopy → _compute_norm_stats → _normalize_scenarios
  → train/val split (seed=42) → stratified batching → training loop
  → save best.pt (model + norm_stats) + history.pt
```

### Training details

- **Step 1 (EV)**: SmoothL1Loss (Huber, beta=1.0). LR warmup (100 steps) + cosine decay. Stratified batching by sequence length. Action head frozen.
- **Step 2 (Action)**: KL divergence loss. Same LR warmup + cosine + stratified batching. Perception + value head frozen. Metrics: accuracy (top-1), WRC (weighted rank concordance).
- Both steps: gradient clipping (max_norm=1.0), early stopping, intra-epoch validation.

### Output structure

```
data/<version>/<name>/
  ├─ configs/<timestamp>.json
  ├─ logs/<timestamp>.txt
  ├─ dataset/<timestamp>/dataset.pt          (raw, shared)
  └─ [single-agent]
       ├─ gto_ev_predict/<timestamp>/best.pt, history.pt
       └─ gto_probs_predict/<timestamp>/best.pt, history.pt

data/<version>/<save_dir>/
  └─ <agent_name>/
       ├─ gto_ev_predict/<timestamp>/best.pt, history.pt
       └─ gto_probs_predict/<timestamp>/best.pt, history.pt
```

`best.pt` contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `norm_stats`, `val_loss`, `step`, `epoch`.
