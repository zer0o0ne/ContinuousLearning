# PLAN: Opponent Adaptation Upgrade — style supervision, count-based stats, showdown anchor

Status: IMPLEMENTED 2026-07-09 (approved same day; D1 switched to regression
per owner). Tests: `tests/test_opponent_adaptation.py` (17 tests, all
deterministic). Deviations from the plan as written: §7 test 2 (parallel
binding e2e) is covered by `test_pool_binding_hand_level` (the exact actor
call shape) + `test_pool_binding_legacy_fallback` instead of a full
inference-server run; showdown labeling extracted into
`_label_showdown_strengths` for honest unit coverage; MC seeds are strings
`f"{hand_seed}:{pos}"`. Bundles three changes (delivered together, per
owner):

1. **Style-recovery auxiliary loss** for the opponent GRU (phase 5) —
   supervised by the KNOWN generating agent of every observed action.
2. **Count-based opponent stats vector (HUD)** injected alongside the GRU
   embedding — sample-efficient frequencies + explicit confidence signal.
3. **Showdown anchor** — auxiliary loss tying the GRU state to the revealed
   hand strength of showdown-reaching opponents.

Plus one **prerequisite bug fix (P0)** discovered while scoping: without it,
items 1–3 train against noise.

Decisions fixed in this plan (flag on review if you disagree):
- **(D1)** Style supervision = **regression of a canonical real-valued style
  vector** derived from the generating agent's modifiers — OWNER DECISION
  (2026-07-09): real players are not K discrete classes; their parameters
  are real-valued, and regression generalizes to the style continuum.
  Canonical encoding in §2.1 resolves the identifiability concerns.
- **(D2)** No extra "chosen-action CE" loss: the phase-5 KL target already
  back-propagates next-action pressure through the GRU (`gru_window=32`),
  and the range-averaged target is a strict superset of the single sampled
  action. Becomes effective once P0 is fixed.
- **(D3)** Item 1 (style supervision) lives in **phase 5 only** — OWNER
  REQUIREMENT (2026-07-09): by phase 6 the agents have drifted away from
  their initial modifiers through self-play fine-tuning, so agent-class
  labels are stale there. No style labels in MCTS collection, no probe
  losses in `mcts_train`; the showdown probe is likewise phase-5-only.
  Phase 6 keeps training the GRU/stats-proj through its existing losses;
  probes stay frozen there.
- **(D4)** All new scenario fields are optional; loaders tolerate their
  absence (labels masked out → aux losses contribute 0). Old checkpoints
  load via the existing `strict=False` path.

---

## 1. P0 — pool-ID ↔ agent binding fix (`generation/generate_opponent.py`)

### 1.1 The bug

`generate_opponent_hand` seats agents with

```python
seated = random.choices(agents_list, k=num_players)   # re-drawn EVERY hand
```

while `opponent_ids` come from the persistent player pool (`p_0..p_26`,
`player_swap_prob=0.01`). So the same persistent ID `p_3` is played by a
**different random personality each hand**. Cross-hand GRU accumulation per
ID — the entire point of the opponent embedding — trains against an
iid mixture of all styles. At deployment (phase 6 collect, evaluation) the
`opponent_id` IS the agent name (`evaluate.py:_rebuild_events`), i.e. one ID
= one consistent style. Train/deploy mismatch: the GRU can at best learn
within-hand adaptation; everything it accumulates across hands is noise.

### 1.2 The fix

- At `generate_opponent_dataset` start, build a deterministic binding
  `pool_binding = {f"p_{i}": agents_list[i % len(agents_list)]["name"]}`.
  Round-robin, no RNG — reproducible, and every agent gets ⌈pool/K⌉ IDs.
- In `generate_opponent_hand`, replace `random.choices` with a lookup:
  `seated[pos] = agent_by_name[pool_binding[table_roster[pos]]]`. Seat
  changes still happen via the existing roster swap mechanism; the STYLE
  behind an ID never changes.
- Parallel mode: pass `pool_binding` to `_opp_actor_main` (it already
  receives `player_pool`); actors use the same lookup.
- Persist:
  - per scenario: `"acting_agent"` (name, str) and `"acting_agent_idx"`
    (index into the sorted agent-name list — the order `_load_agents`
    produces) — labels for §2;
  - in `meta.json`: `"pool_binding"` (dict) — audit/debug.

No config flag: this is a bug fix, not a feature.

### 1.3 Consequence

**The opponent dataset must be regenerated** (25k hands with the current
config). Old opponent datasets remain loadable (D4) but carry no style
labels and keep the broken binding — training on them wastes items 1–3.

---

## 2. Item 1 — style-recovery auxiliary loss (phase 5)

### 2.1 D1: canonical style vector (regression target)

Raw modifier params are not directly regressable: they live in
heterogeneous spaces (factors over arbitrary action index sets like
`"4:10"` / `[0, 1, 13]`, conditions on unobservable equity, a temperature
composing multiplicatively) and the map params → policy is many-to-one.
Fix: a **deterministic canonical encoding** `build_style_vector(modifiers,
n_actions) -> list[float]` (new pure function in
`agent/train_scenarios/modifiers.py` — it reuses `resolve_actions` and
`_parse_condition`), n_style_dims = **16**:

- **Dims 0–4 — unconditional category biases**: 5 action categories (fold,
  call, small raise, big raise, all-in — same split as
  `resolve_actions`). Every `action_bias` modifier distributes its `factor`
  over categories **coverage-weighted**: contribution to category c =
  `factor × |resolved_actions ∩ c| / |c|`. Factors from multiple modifiers
  accumulate additively (mirrors `apply_modifiers` accumulation).
- **Dims 5–9 — low-equity conditional biases**: `conditional_bias` with
  condition `equity < t` contributes `factor × t × coverage(c)` (region
  measure `t` scales the contribution — "equity < 0.3" affects less of the
  play than "equity < 0.6").
- **Dims 10–14 — high-equity conditional biases**: condition `equity > t`
  contributes `factor × (1 − t) × coverage(c)`.
  `pos`-conditions (unused in the current config) fold into the
  UNCONDITIONAL block weighted by the fraction of positions satisfying the
  condition (documented in the function docstring).
- **Dim 15 — `log(temperature)`** (the effective one after modifier
  override; base `gto_temperature` if no temperature modifier).

Identifiability: the encoding is canonical (one vector per agent config,
computed the same way every time), and the probe regresses the ENCODING,
not the free-form params — two param sets producing the same encoding are
treated as the same style, which is exactly the desired equivalence.

**Per-dim z-scoring**: at phase-5 training start, compute mean/std of each
dim across the pool's K target vectors; z-score targets (std < 1e-8 → that
dim is constant across the pool → excluded from the loss via a fixed mask).
Keeps MSE balanced across dims of very different scales (factors vs log-T).

### 2.2 New module (`agent/opponent_action/probes.py`)

```python
class StyleProbe(nn.Module):        # d_model → n_style_dims (16)
    def __init__(self, d_model, n_style_dims=16):
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_style_dims))
```

Registered on `ASI` as `self.style_probe`, created iff
`architecture.opponent_embedding.style_probe` is true (bool, default false —
old configs unaffected). Saved in every checkpoint like any submodule;
frozen implicitly in phases 1–4 and 6 (their optimizers enumerate explicit
param lists; the probe is in none of them).

### 2.3 Exposing per-sample GRU states

`Perception.forward_batch` already computes, per flat event with an
`opponent_id`, the post-update hidden `h` (perception.py:653–657). Add:

- kwarg `collect_opp_states=False`. When True, track `last_state[b_i] = h`
  in the existing causal loop (the LAST event of every phase-5 sample is the
  decision event of the acting player, so `last_state[b_i]` is exactly the
  actor's state at the decision). Return an extra pair
  `(opp_last_states (B, d_model), opp_states_mask (B,))` — zeros/0 for
  samples with no opponent-id events.
- `ASI.forward_batch` forwards the kwarg and, when set, adds
  `out["opp_last_states"]`, `out["opp_states_mask"]`.

No behavior change when the kwarg is absent (default False everywhere
except the phase-5 train/val loops).

### 2.4 Targets and loss (phase 5, `opponent_action_predict/`)

- **Target plumbing**: the pipeline builds
  `style_targets = {agent_name: build_style_vector(...)}` from
  `multi_agent.agents` (+ base `solver.gto_temperature`) and passes it into
  `train_opponent_action` via `train_cfg["style_targets"]`. Scenarios carry
  the acting agent's NAME (P0, §1.2); matching by name — not by index —
  avoids any ordering mismatch between config order and `_load_agents`'s
  sorted-directory order.
- `OpponentActionDataset.__getitem__` / `ShardedOpponentDataset` also return
  `acting_agent` = `scenario.get("acting_agent")` (str or None). Collates
  pass it through as a list.
- Loss in `train_opponent_action` (train + validation paths): resolve each
  sample's z-scored target vector by name (None / unknown name → masked
  out), then

```
valid = has_target & opp_states_mask.bool()
pred  = style_probe(opp_last_states[valid])              # (M, 16)
loss_style = MSE(pred[:, dim_mask], target_z[valid][:, dim_mask])
total = kl + style_probe_weight * loss_style             # 0 when M == 0
```

  where `dim_mask` excludes pool-constant dims (§2.1 z-scoring).
- `style_probe_weight` from `opponent_action_train` config (default 0.0 —
  off unless configured; config.json sets 0.3).
- Trainable params += `style_probe.parameters()`.
- History/logging: track `style_loss` and `style_nn_acc` — nearest-neighbor
  accuracy (predicted vector's nearest pool target vector == true agent), a
  readable "how fast does the embedding localize a style" metric derived
  from the regression, logged on validation.

Gradient path: MSE → probe → `h` → GRU (+ its inputs) with the existing
truncated BPTT (`gru_window`). This is the direct "the embedding must
identify the opponent's real-valued style" pressure the current setup
lacks — and because the target space is continuous, an unseen opponent
lands BETWEEN pool styles instead of being forced into a class.

---

## 3. Item 2 — count-based opponent stats vector (HUD)

### 3.1 Feature spec (frozen; n_stats = 49)

Per opponent ID, plain counters over that opponent's OBSERVED actions
(events where their action one-hot has `max ≥ 0.5`; decision snapshots with
zero action vectors update nothing):

- **Buckets**: street (preflop/flop/turn/river — from the number of revealed
  table cards in the event: 0/3/4/5) × facing (`facing_bet == 0` vs `> 0` —
  from the event's bets vector: `max(bets) − bets[acting_pos] > 1e-6`;
  z-scoring is affine with a shared mean/std, so the sign of differences is
  preserved on normalized events).
- **Action categories** (5): fold `[0]`, call `[1]`, small raise
  `[2 .. 2+bins//2)`, big raise `[2+bins//2 .. n_actions−1)`, all-in
  `[n_actions−1]` — same split as `modifiers.resolve_actions`.
- **Features**:
  - 4 × 2 × 5 = 40 Laplace-smoothed in-bucket frequencies
    `(count + 1) / (bucket_total + 5)`;
  - 8 per-bucket confidence terms `log1p(bucket_total) / 5.0`;
  - 1 global `log1p(total_observed_actions) / 5.0`.

All features are bounded and dimensionless — no norm_stats involvement.
The confidence terms are the explicit "how much do I know about this player"
signal the GRU state cannot express (a fresh opponent and a well-known
zero-ish-style opponent are currently indistinguishable).

### 3.2 Storage (`perception/opponent_embeddings.py`)

Extend `OpponentEmbeddingTable`:

- `self.stats = {}` — `opp_id -> np.ndarray(40+... raw COUNTS, float64)`;
  store raw counts (shape `(8, 5)` bucket×category), derive the 49-dim
  feature vector on demand via a module-level
  `stats_features(counts) -> np.ndarray(49, float32)` (pure function —
  unit-testable).
- `get`/eviction: stats created/evicted together with the embedding entry.
- `clone()`: deep-copies stats (validation isolation — same reason as
  embeddings, A.4.3).
- `detach_all()`: no-op for stats (plain numpy, never in the graph).
- `state_dict()`/`load_state_dict()`: include stats under a `"__stats__"`
  sub-dict; loading old state without it → empty stats (D4).

### 3.3 Update + injection (`perception/perception.py`)

In the existing causal flat loop of `forward_batch` (the ONLY place opponent
state advances — chronology, sample-major order and A.4.5 group rewind are
already correct there):

1. Decode the event's discrete features from `precomputed` tensors already
   on hand: street from `card_ids[flat_idx, :5]` (count of 52-tokens),
   category from `actions[flat_idx].argmax()` when `max ≥ 0.5`, facing flag
   from `bets[flat_idx]` + `acting_pos[flat_idx]`.
2. Update counters for `opp_id` (post-update convention — same as the GRU:
   the injected state reflects the current event, whose content the event
   token itself already carries; no leak).
3. Injection: `inject = h + self.opp_stats_proj(feat)` where
   `feat = stats_features(counts)` as a device tensor and
   `opp_stats_proj = nn.Linear(49, d_model)` — new Perception submodule,
   created iff `architecture.opponent_embedding.stats_enabled` (bool,
   default false). When disabled, injection stays `h` exactly (bit-for-bit
   legacy).
4. **A.4.5 group rewind**: snapshot/restore `table.stats` (deep copy of the
   touched dict) exactly where `group_start_table` snapshots embeddings —
   otherwise observer copies multi-count every decision.

Trainability: `opp_stats_proj` params join the phase-5 trainable list next
to `opponent_gru`; phase 6 trains everything already. Features themselves
are constants (counters) — gradient flows only through the projection.

### 3.4 Persistence — rides along for free

Everything downstream already passes `OpponentEmbeddingTable` objects around
(phase-5 train table, phase-6 pipeline-owned per-agent tables persisting
across cycles, eval per-agent tables, server-side per-(worker, agent)
tables). Stats live inside the same object, so every existing flow carries
them with zero call-site changes. The only serialization surface is
`state_dict`/`load_state_dict` (§3.2).

---

## 4. Item 4 — showdown anchor (phase 5)

### 4.1 Label generation (`generation/generate_opponent.py`)

After the betting loop of `generate_opponent_hand`, when the hand reached
showdown (≥ 2 players with `players_state >= 0`):

- For each such player `pos` with `fixed_hands[pos] is not None` and the
  fixed hand disjoint from the full board `deck[:5]` (B.4.2 collisions →
  skip that player, honest):
  - `strength ∈ [0, 1]` = Monte-Carlo estimate of
    `P(fixed hand beats a random opponent combo on the full board)`, ties
    counted 0.5. K = 256 combos sampled without replacement from the 1326
    minus board/own-hand conflicts, RNG seeded by `(hand_id_seed, pos)` —
    fully deterministic. Uses the same 7-card evaluator the engine's
    `Judger` uses (CPU, no CUDA in actors).
  - Annotate that player's LAST recorded scenario of this hand (the latest
    `scenarios[i]` with `acting_pos == pos`):
    `scenario["actor_showdown_strength"] = strength`.
- Fold-ended hands, players without fixed hands, colliding hands → no label.

Note `hand_id` is assigned by the caller; pass a per-hand `seed` into
`generate_opponent_hand` (sequential: `hand_i`; parallel: `hand_id_offset +
hand_i`) so the label MC is reproducible in both modes.

### 4.2 Probe + loss (phase 5)

- `ShowdownStrengthProbe` in `agent/opponent_action/probes.py`:
  `Linear(d, d//2) → GELU → Linear(d//2, 1)`, raw output, target in [0,1].
  Registered on `ASI` as `self.showdown_probe`, gated on its own flag
  `architecture.opponent_embedding.showdown_probe: bool` (default false) —
  independent of the style probe.
- Dataset returns `showdown_target` = `scenario.get("actor_showdown_strength",
  float("nan"))`; loss masks NaN:

```
valid = ~isnan(showdown_target) & opp_states_mask.bool()
loss_sd = MSE(showdown_probe(opp_last_states[valid]).squeeze(-1),
              showdown_target[valid])
total = kl + style_probe_weight * loss_style + showdown_probe_weight * loss_sd
```

- `showdown_probe_weight` in `opponent_action_train` (default 0.0;
  config.json sets 0.3).
- Input is the same `opp_last_states` as §2 — `h` has seen every event of
  the hand (board, line) plus the cross-hand style memory, so it carries
  exactly the information "how strong is THIS player's range on THIS line".
  This is the only loss connecting actions → actual holdings; it teaches
  per-style range calibration (e.g. `lag_bluffer`'s river aggression maps to
  weaker revealed strength than `value_heavy`'s).

Why the LAST decision of the hand: it is the point where the full line is
known and closest to the reveal; earlier decisions would supervise `h`
against outcomes it cannot yet know (the runout).

---

## 5. Config changes (`versions/v6/config.json`)

```jsonc
"architecture": {
  "opponent_embedding": {
    "enabled": true,
    "stats_enabled": true,            // §3 — new
    "style_probe": true,              // §2 — new (16-dim regression probe)
    "showdown_probe": true            // §4 — new
  }
},
"opponent_action_train": {
  // ... existing ...
  "style_probe_weight": 0.3,          // §2 — new
  "showdown_probe_weight": 0.3        // §4 — new
}
```

All new keys default to off/0 → absent keys reproduce current behavior
exactly (old configs, old checkpoints keep working).

## 6. Files touched

| File | Change |
|---|---|
| `generation/generate_opponent.py` | P0 binding; `acting_agent`/`acting_agent_idx`/`pool_binding` persistence; showdown strength labels; per-hand seed plumbing (sequential + parallel actor) |
| `perception/opponent_embeddings.py` | stats storage, `stats_features()`, clone/state_dict/eviction |
| `perception/perception.py` | `opp_stats_proj`; stats update + injection in the causal loop; group-rewind of stats; `collect_opp_states` kwarg |
| `agent/agent.py` | probe submodules (gated); forward kwarg passthrough + `opp_last_states` outputs |
| `agent/opponent_action/probes.py` | new: `StyleProbe`, `ShowdownStrengthProbe` |
| `train_scenarios/modifiers.py` | new pure function `build_style_vector(modifiers, n_actions, base_temperature)` (§2.1) |
| `pipeline.py` | build `style_targets` from `multi_agent` config, pass via `train_cfg` |
| `opponent_action_predict/dataset.py` | emit `acting_agent`, `showdown_target`; collates |
| `train_scenarios/sharded.py` | same two fields through `ShardedOpponentDataset` |
| `opponent_action_predict/train.py` | aux losses, weights, target z-scoring + dim mask, trainable params, history keys, validation path |
| `config.json` | §5 keys |
| `CLAUDE.md` | architecture + phase-5 loss row + opponent_data notes |

Explicitly NOT touched: MCTS search (`mcts.py`, `evaluator.py`,
`inference_server.py` — tables ride along), modelling head, GTO phases,
value normalization.

## 7. Tests (`versions/v6/tests/`) — deterministic, e2e-style

All tests: fixed seeds, no probabilistic assertions, no order dependence.
New file `test_opponent_adaptation.py` unless noted.

**P0 binding**
1. `test_pool_binding_consistency` — generate a small opponent dataset
   (2 tiny random-init agents via `agents_override`, ~30 hands, seeded):
   for every persistent ID appearing in ≥2 hands, assert `acting_agent` is
   identical across ALL its scenarios; assert `acting_agent_idx` matches the
   sorted-name index; assert `meta.json` contains the binding and it is the
   round-robin one.
2. `test_pool_binding_parallel_matches_spec` — same generation with
   `n_workers=2`: every scenario's `acting_agent` equals
   `pool_binding[opponent_ids[acting_pos]]` (actors received the binding).

**Stats table (§3)**
3. `test_stats_features_pure` — feed `stats_features()` hand-built count
   arrays; assert exact expected 49-dim vectors (Laplace smoothing, log
   terms) via `torch.testing.assert_close` on literal values.
4. `test_stats_update_counts` — craft an event sequence with known actions
   (fold preflop unraised, big raise on flop facing a bet, zero-action
   decision snapshot), run `Perception.forward_batch` with a table; assert
   the table's raw counts equal the hand-computed ones exactly, and that
   the zero-action event contributed nothing.
5. `test_stats_group_rewind` — two observer copies of one scenario with
   matching `gru_sample_groups`: counts advance ONCE; without groups: twice.
   Exact integer assertions.
6. `test_stats_disabled_bitwise_legacy` — `stats_enabled=false`: forward
   output is bitwise-identical to the pre-change path (golden comparison of
   two forwards with identical seeds/weights on the same build, stats flag
   on-vs-absent config with fresh table and zeroed proj is NOT required —
   compare `enabled:false` vs a config without the key).
7. `test_stats_state_dict_roundtrip` — populate table (embeddings + stats),
   `state_dict` → `load_state_dict`; assert equality; load a legacy state
   (no `__stats__`) → empty stats, no crash.

**Style probe (§2)**
8. `test_build_style_vector` — pure-function test: encode all 5 config
   agents' modifier lists; assert exact expected 16-dim vectors computed by
   hand (coverage weighting for `"4:10"` / `[0, 1, 13]`, region measure for
   `equity < 0.3` / `> 0.6`, `log(T)` dim, additive accumulation); assert
   an empty modifier list encodes to zeros + `log(base_temperature)`; assert
   two calls are identical (determinism).
9. `test_collect_opp_states_alignment` — batch of 3 samples, one WITHOUT
   any `opponent_id`: `opp_last_states[i]` equals the h injected at each
   sample's last event (recompute by hand from a sequential single-sample
   forward with the same table clone); mask is 0 for the no-id sample.
10. `test_style_loss_e2e_learns_styles` — e2e scenario: two synthetic
    "styles" (always-fold vs always-raise event streams) under two
    persistent IDs with distinct target vectors, tiny agent, run N seeded
    phase-5 steps with `style_probe_weight=1.0`; assert final MSE on a
    held-out fixed batch is below the initial MSE (exact recorded floats —
    deterministic given seeds, CPU) and each prediction's nearest target
    vector is the true agent's (`style_nn_acc == 1.0` on the fixed batch).
11. `test_style_target_absent_is_masked` — dataset without `acting_agent`
    (legacy scenarios): total loss equals plain KL exactly; no NaN. Also:
    a pool-constant dim (identical across all targets) is excluded by the
    dim mask and contributes exactly 0.

**Showdown anchor (§4)**
12. `test_showdown_strength_labels` — seeded generation where one player
    holds the nuts-ish fixed hand and another air (construct by seeding and
    reading back `fixed_hands` via the scenario's persisted combos): assert
    labels exist only on showdown-reaching actors' last scenarios, values in
    [0,1], nuts label > air label, and exact float equality across two runs
    with the same seed (MC determinism).
13. `test_showdown_label_skipped_on_fold_and_collision` — fold-ended hand →
    no labels; hand where the fixed hand collides with the runout board →
    that player unlabeled.
14. `test_showdown_loss_masking` — batch with all-NaN targets → total loss
    == KL + style part exactly.

**Integration / compat**
15. `test_phase5_e2e_all_losses` — full `train_opponent_action` run on a
    tiny generated dataset (both aux weights > 0, `stats_enabled=true`,
    2 epochs, seeded): completes, `best.pt` exists, history contains the new
    keys, checkpoint loads back into a fresh ASI (`strict` path) and a
    forward runs; run twice with identical seeds → identical `best.pt`
    val_loss.
16. `test_old_checkpoint_loads` — build a state_dict WITHOUT probe/proj keys
    (delete them), `load_checkpoint` with the new code → no error, new
    modules freshly initialized, old weights preserved (spot-check one
    tensor).
17. `test_phase6_and_eval_paths_unaffected` — with probes present on the
    ASI: phase-6-style forward (`heads={"action","value"}`,
    `skip_opponent_emb=False`, table with stats) and an eval-style batched
    action forward both run and are seed-reproducible; MCTS
    `LocalEvaluator.evaluate_root` smoke on a toy tree.

## 8. Implementation order

1. P0 binding + persistence fields + tests 1–2.
2. Stats table + injection + rewind + tests 3–7.
3. `build_style_vector` + `collect_opp_states` + probes + phase-5 losses +
   tests 8–11, 14.
4. Showdown labels in generation + tests 12–13.
5. Integration tests 15–17, config.json, CLAUDE.md.
6. Regenerate the opponent dataset (25k hands), rerun phase 5, then phase 6.

## 9. Risks & mitigations

- **Dataset regeneration is mandatory** for the new signal (P0 + labels).
  Cost ≈ one current `opponent_data` run. Old data still trains (D4) but
  items 1–3 are inert on it.
- **Probe overfitting to the K pool styles**: only K distinct target
  vectors exist in the data, so the regression could still degenerate into
  a K-point lookup. Mitigation: small weight (0.3), KL remains the dominant
  loss, and the continuous target space at least keeps the probe's output
  geometry style-shaped; watch `style_nn_acc` vs KL — if KL degrades while
  style_nn_acc saturates, lower the weight. If more style diversity is ever
  needed, the clean lever is more pool agents with jittered modifier
  factors (data change, no code change).
- **Stats injection distribution shift at phase-6 start**: cycle-0 tables
  are empty → features are the smoothed-uniform vector; same cold-start the
  GRU already has, and `post_embed_norm` (LayerNorm) bounds the injection
  magnitude. No extra handling.
- **Facing-bet flag on normalized bets** relies on shared affine z-scoring
  (true today, `_normalize_events_inplace`); test 4 pins it.
- **Showdown labels are range-consistent, not omniscient**: fixed hands are
  belief-sampled, which is exactly the quantity the observer could learn —
  consistent with the "training signal from decision-time-estimable
  quantities" principle of the MCTS value redesign.
