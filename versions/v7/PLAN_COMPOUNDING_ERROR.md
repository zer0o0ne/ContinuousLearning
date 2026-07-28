# PLAN: Compounding-error fixes for the modelling head / MCTS

Date: 2026-07-26. Owner decisions recorded in-session (validated the earlier
diagnosis; four mechanisms approved). Supersedes nothing; complements
`versions/v6/PLAN_MODELLING_HEAD_REDESIGN.md` (the LM-loss redesign) and
partially REVISES one decision of the 2026-07 audit (`stop_grad_old_embs`,
see §3).

## Diagnosis (validated against code)

1. **Stochastic transitions across street boundaries.** `mcts.py` has no
   notion of streets/cards/chance. When an action closes a street, the true
   next decision state depends on dealt cards, but the tree extends the path
   with ONE deterministic embedding `h(s, a)`. In training, the LM target of
   such a step is the real perception WITH the actually dealt cards → a
   deterministic head trained with MSE converges to the conditional MEAN
   embedding over runouts — an off-manifold point ("the average flop is not
   a flop"). Every tree path crossing a street then compounds from a broken
   base. Jensen: `π(E[s']) ≠ E[π(s')]` — the action/opponent heads suffer
   most; the value head least (`V(s) = E_c[V(s')]` is exactly what a
   pre-chance value estimate should mean).
2. **Support mismatch.** Training chains: ≤ `max_chain_depth=5` rolled steps,
   70 % teacher forcing (hero steps only). Search: unbounded depth (no
   "depth" anywhere in `mcts.py`), 0 % TF. Deep contexts are outside the
   training support → uncontrolled extrapolation.
3. **Maximization bias.** PUCT is a max over noisy/biased Q̂;
   `E[max Q̂] ≥ max E[Q̂]`, the gap grows with the number of evaluated
   branches (≈ n_simulations) and with per-node noise (grows with depth via
   1–2). The search adversarially harvests the modelling head's own errors —
   this is why MORE simulations make agents WORSE.
4. **No self-consistency training.** `stop_grad_old_embs=true` detaches every
   appended embedding between chain steps: the composition `h∘h` never
   appears in any loss, so the head is never optimized to produce outputs
   that are good inputs to itself (MuZero trains THROUGH the unroll with
   0.5/step gradient scaling; that is its key stability mechanism).

## Approved mechanisms

### 1. Street-boundary leaf in MCTS search (`mcts.py`)

Config: `mcts.street_boundary_leaf` (bool, default `false` = legacy).

- `MCTSNode` gets `is_street_leaf` (slot, default False).
- `_select_to_leaf`: on first visit (after `_replay_game_state`), when the
  flag is enabled and `not gs.is_terminal and gs.turn != root_gs.turn`, mark
  `node.is_street_leaf = True`. Known street leaves return early
  (`path, None`) like terminals — no replay on repeat visits.
- `search()` / `_simulate`: street leaves share the terminal handling path —
  first visit tries `_deterministic_terminal_value(gs)` (hero already folded
  → exact `−invested`, no NN; hero live → `None` → queued for ONE value_head
  forward, cached in `_term_value`), repeat visits back up the cached value.
  Never expanded — the modelling head is never asked to predict across a
  chance event.
- Street leaves are NOT terminals: `_collect_terminals` skips them (they
  fail `is_terminal` and have no children), so `evaluate_all_terminals` /
  `re_backup_terminals` / `_select_terminal_targets` are untouched. Their
  value_head Q stays as backed up — semantically `V(afterstate) =
  E_cards[V(next street)]`, which is exactly what the value head is trained
  to represent.
- Works unchanged in sequential, parallel-actor (flag rides in `mcts` config)
  and eval paths.

### 2. Street-aligned training targets

a) **Chains** (`collect.py`): `collect_training_data` gains
   `chain_street_boundary` (from `mcts_train.chain_street_boundary`, default
   `false`). The chain loop breaks before appending step `i` when
   `decisions[t+1+i]["game_state_at_root"].turn !=
   decisions[t]["game_state_at_root"].turn` (both snapshots already exist on
   every decision — no new fields). `max_chain_depth` remains as a hard cap
   on top (min of the two).

b) **LM pairs** (`modelling.py::build_lm_pairs`): new arg
   `street_boundary=False`. Street of an event = count of revealed table
   cards (`0 <= c <= 51` — excludes both `-1` and `52` no-card encodings;
   same idiom as `perception.py:33`). A pair is dropped when
   `street(events[tgt]) != street(events[src])` — those targets contain
   newly dealt cards and are unpredictable in principle (mean-regression
   pairs). Plumbed from `mcts_train.lm_street_boundary` (phase-6 root pairs;
   chain pairs are already street-pure after (a)) and
   `modelling_train.lm_street_boundary` (phase 4 — both perception-caching
   call sites + `_reconstruction_loss` fallback).

With (1)+(2) the supports match by construction: the head is trained and
queried ONLY on within-street rollouts.

### 3. Truncated BPTT through the rolled chain (`mcts_predict/train.py`)

REVISES the 2026-07 audit decision `stop_grad_old_embs=true` (owner approved
in-session). Config: `mcts_train.bptt_depth` (int, default `0` = legacy),
`mcts_train.bptt_grad_scale` (float, default `0.5`).

- `_mcts_forward` keeps, per example, the trimmed base context plus a LIST of
  appended rolled tokens instead of one incrementally-cat'ed tensor. At chain
  depth `d`, the context is rebuilt as `cat(base, tok_0 … tok_{d−1})` where
  token `j` is attached iff its age `d − j <= bptt_depth`, else `.detach()`.
- Each token is stored once through `_scale_grad(x, s) = s·x + (1−s)·x.detach()`
  (MuZero-style): a gradient path passing through `k` autoregressive hops is
  damped by `s^k` — self-consistency is trained, explosion is geometrically
  suppressed.
- Precedence: `bptt_depth > 0` → truncated scheme; else legacy
  (`stop_grad_old_embs` true → all detached, false → full BPTT, both
  bit-for-bit as before).
- Memory: bounded by `bptt_depth ≤ max_chain_depth = 5`;
  `gradient_checkpointing=true` already on.

### 4. Teacher-forcing decay (config only)

`mcts_train.teacher_forcing.p_end: 0.7 → 0.35` (schedule code already exists
and is currently dead because `p_start == p_end`; `decay_cycles=50` stays).
Validation already runs at `p_tf=0`, so the effect is directly visible in
`val_loss`.

### 5. Spectral clamp on the modelling conditioning MLP (`modelling.py`)

Config: `architecture.modelling_spectral_clamp = {enabled, max_sigma,
n_power_iterations}` (absent/disabled = bit-for-bit legacy).

- Scope: `mlp_in` / `mlp_out` of `_condition` only (the map that produces the
  appended embedding). The `_encode` attention stack is NOT clamped —
  deliberately: full-Lipschitz control of attention is invasive and the
  output `RMSNorm` already bounds scale. This is a partial Lipschitz control
  (announced limitation).
- Implementation is a CLAMP, not a normalization: effective weight
  `W_eff = W · min(1, max_sigma/σ(W))` with σ estimated by 1-step power
  iteration on persistent `u` buffers (updated under `no_grad` in training
  mode only; gradient flows through σ when the clamp is active, as in
  Miyato et al.). Below the cap the layer is EXACTLY identity-equal to
  legacy — no expressiveness loss until σ exceeds `max_sigma`.
- Checkpoint compatibility both ways: the parameter key stays `weight`
  (torch's parametrized `spectral_norm` would rename it and silently drop
  pretrained weights under `strict=False` — that is why it is not used).
  New `u` buffers are missing in old checkpoints → freshly initialized,
  power iteration re-converges in a few forwards.

## Config values for the next v7 run

```
mcts.street_boundary_leaf         = true
mcts_train.chain_street_boundary  = true
mcts_train.lm_street_boundary     = true
mcts_train.bptt_depth             = 2      # conservative; 3 if val stable
mcts_train.bptt_grad_scale        = 0.5
mcts_train.teacher_forcing.p_end  = 0.35
modelling_train.lm_street_boundary = true
architecture.modelling_spectral_clamp = {enabled: true, max_sigma: 1.0,
                                         n_power_iterations: 1}
```

All new knobs default to legacy behaviour when absent — old configs and
checkpoints run unchanged.

## What is deliberately NOT done (deferred, needs measurement first)

- Ensemble value heads / LCB pessimism at PUCT (problem-4 residual): add only
  if, after the street cap, deep-branch overestimation is still visible.
- Chance-node expansion / Stochastic-MuZero codebook: the principled fix for
  cross-street prediction; the street cap removes the need at 1-street
  horizon. Revisit if within-street horizons prove insufficient.
- Noise injection on rolled embeddings: owner deselected; std would need the
  rolled-vs-real MSE measurement anyway.

## Interactions / invariants to keep

- Street leaves must never enter `evaluate_all_terminals` (they are not
  fold/showdown states; equity replay would mis-handle them).
- `n_terminal_values` rollouts use tree terminals only — with the cap these
  are within-street terminals (folds/all-in-runouts); paths stay inside the
  trained support automatically.
- `get_n_distribution` (root visit targets) is unaffected — root children are
  always on the root street.
- Parallel mode: flag plumbing only via config sections already shipped to
  actors; no inference-server protocol change.

## Success criteria

- `val_loss` (p_tf=0) chain/recon components stop degrading with depth;
- dose–response flips: eval BB/100 no longer decreases as `n_simulations`
  grows (the original divergence symptom);
- no regression in phase-4/6 unit+e2e tests; new deterministic tests cover
  each mechanism on/off.
