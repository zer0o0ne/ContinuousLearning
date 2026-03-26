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
Event sequence → EventSequenceEmbedder → Encoder (Qwen3 causal transformer)
    → Decoder (Qwen3 self-attention) → ValueHead (Qwen3 → mean pool → scalar)
                                      → ActionHead (frozen during phase 1)
```

- **EventSequenceEmbedder** (`perception/perception.py`): Embeds each poker event (cards, positions, pot, bets, action, blinds, num_players) into d_model vectors. One sample = full sequence of events from hand start to decision point.
- **Encoder** (`perception/encoder.py`): Qwen3 causal transformer over event sequence.
- **HierarchicalMemory** (`perception/memory.py`): Beam-search clustering memory. Disabled during GTO EV training (`skip_memory=True`), activated in later phases.
- **Decoder** (`perception/decoder.py`): Qwen3 self-attention over encoder output (or memory + encoder when memory active).
- **ValueHead** (`value/value.py`): Qwen3 layers → mean pool → Linear → scalar EV.
- **GPU Solvers** (`gto_utils/`): `gpu_solver.py` (v1, random opponent ranges) and `gpu_solver_v2.py` (v2, position-based ranges with MDF opponent response modeling). Solver choice is a config param.

## Config (`versions/<version>/config.json`)

Two top-level sections:
- `architecture` — model dimensions, layer counts, memory config, max_players (supports up to 9-max)
- `gto_ev_train` — training hyperparams, solver choice (`"v1"`/`"v2"`), dataset/agent loading, early stopping

Key config params:
- `agent_dir` (root level): path to checkpoint dir for loading pretrained weights (partial loading supported)
- `gto_ev_train.dataset_dir`: path to pre-generated dataset (skips generation if valid)
- `gto_ev_train.solver`: `"v1"` (random ranges) or `"v2"` (range-aware with opponent response model)
- `gto_ev_train.val_every`: validate every N batches (null = end-of-epoch only)
- `gto_ev_train.interrupt_after_fails`: early stopping after N validations without improvement

## Data Generation (`agent/train_scenarios/gto_ev_predict/generate.py`)

Uses `env/table.py` to simulate poker hands with GTO-sampled actions (softmax over fold/call/raise EVs). Each sample: event sequence + scalar EV target. The v2 solver narrows opponent ranges by position and action history, and models fold/call/reraise responses via MDF.

## Data Flow

`config.json` → `pipeline.py` → creates ASI, optionally loads checkpoint → `train.py` → loads/generates dataset → trains with MSE loss on EV targets → saves `best.pt` + `history.pt` to `data/<version>/<experiment_name>/gto_ev_predict/<timestamp>/`
