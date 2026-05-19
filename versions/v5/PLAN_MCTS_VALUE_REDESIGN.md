# MCTS Value-Target Redesign

Status: **Step 1 and Step 2 both implemented and smoke-tested** (2026-05).
Both steps live behind the same `_finalize_value_targets` hybrid, so no flag
is needed to toggle between them — Step 2 simply replaces the realized half
and the terminal Qs that feed root.Q with equity-based estimates.

This document captures the full plan, the reasoning behind every decision,
and the math so a future session (or another engineer) can pick it up
without losing context. Steps that are **already done** are marked ✓; the
rest is still TODO and ordered for implementation.

---

## 1. Symptoms

Trained agents (`data/v5/5_final_agents/gto_pure/mcts_predict/2026_05_16_*`)
converge to fold-rate ≈ **0.97** in slumbot evaluation. Concurrently:

- `mcts_ev_mean` drifts negative across cycles: −3.24 → −1.29 → −0.81.
- `mcts_ev_std` is bloated: 12.8 → 8.3 → 6.1.
- `ratio_max` collapses: +5.4 (cycle 1) → +0.81 (cycle 20+). The model has
  stopped seeing winning outcomes in its own self-play.
- Per-cycle component losses: `chain` loss stagnates at ≈0.4 even after
  `value` and `action` drop to near-zero; explodes again at cycle 20.

## 2. Root causes (ranked by impact)

### 2.1 Per-agent mean shift creates a fold-bias

Current value-target formula:
```
ratio = outcome_chips / max(pot + facing_bet, BB)
value_target = (ratio - mcts_ev_mean) / mcts_ev_std       ← the bug
```

When `mcts_ev_mean < 0` (agent loses on average in self-play), fold's
deterministic `outcome = 0` becomes:
```
z(fold) = (0 - μ) / σ = -μ/σ > 0  → e.g. +0.155 at cycle 10
```

But `value_head` trained on `(ratio - μ)/σ` outputs ≈ 0 for "typical"
internal states (where `true_EV ≈ μ`). So in PUCT comparison at the root,
the fold child has `Q ≈ +0.155` while all other children have
`Q ≈ 0` — fold systematically wins.

Worse: fold is depth-1 terminal directly under the root. Its `Q` is
locked in after one visit (deterministic outcome, zero variance). Other
children's `Q` mixes value_head outputs (≈0) with deeper terminal
contributions (also ≈+0.155 because the same z-score offset). The
fold-bias does **not** average out — it concentrates at fold.

This forms a death-spiral:
```
losing self-play → μ < 0 → fold gets +0.155 → MCTS picks fold more
   → blinds lost without play → μ more negative → fold bias stronger → ...
```

### 2.2 State-dependent denom inflates left tail

`denom = max(pot+facing, BB)`. On early-street decisions with no facing bet,
`denom = BB = 10`. If hand escalates and hero loses 980 chips later, the
training ratio becomes `−980 / 10 = −98`. This single sample blows up the
empirical std (12.8 in cycle 1), which then propagates into every example's
z-score normalizer.

Wins are bounded by opponent stacks **at decision time**; losses are bounded
by **hero's full stack going forward**. Asymmetry → fat left tail → mean
drifts negative even for a 50/50 agent.

### 2.3 Heuristic terminal evaluator

`_make_terminal_evaluator` uses `pot / n_active − invested` for showdown
terminals. This is a fair-share heuristic because MCTS `GameState` doesn't
carry cards. Strong hands get under-estimated terminal value, weak hands
get over-estimated. The heuristic also carries the same z-score offset, so
its outputs are biased too.

`terminal_eval.py:evaluate_all_terminals` implements proper equity-based
terminal evaluation (with opponent range narrowing via action_heads) — but
it is **never called** from the pipeline (verified via grep).

### 2.4 High-variance MC realized outcome

Each example's `value_target` is the **single realized chip delta** from
that decision to hand end. In poker this is bimodal at showdowns (win whole
pot vs lose investment). Std of realized outcomes ≫ std of true EV.

## 3. Design principles

These constrain every design choice below:

1. **Zero-sum preserved**. At any state s, `Σ_p EV(s, p) = 0` (chip
   conservation). Any per-agent mean shift breaks this. Constants and
   single-scale normalization preserve it.

2. **No self-bootstrap as primary signal**. AlphaZero's robust
   convergence relies on game-outcome (`z`) anchoring. Pure `root.Q`
   target = self-bootstrap, drifts toward policy fixed point regardless
   of true EV.

3. **Single scale across all training examples**. State-dependent denom
   makes value_head learn different "EV units" per state — incompatible
   with MCTS backup, which mixes Q values across tree depth.

4. **Compatible with value-head pretraining transfer**. Don't pick a
   scale so different from GTO that pretraining is wasted.

5. **Variance reduction via equity averaging**, not just z-scoring. Card
   randomness is the dominant noise source; equity calculations remove it
   analytically.

## 4. Hybrid target formula

The target for `value_head` becomes:
```
target = clip( (α·q_chips + (1−α)·realized_chips) / new_scale, ±clip )
```

Where:
- `q_chips` = MCTS `root.Q` (after equity backfill in Step 2)
  converted to chip units.
- `realized_chips` = chip-delta from decision to end of hand (Step 1: raw
  Monte Carlo; Step 2: equity-based on actual final game state).
- `new_scale` = bootstrapped `std(realized_chips)` over collected hands of
  current cycle (in chips). Stored as `norm_stats["mcts_value_scale"]`.
- `α` = `mcts_train.value_target_alpha` (config). 0 = pure MC, 1 = pure TD.
- `clip` = `mcts_train.value_target_clip` (config, default 5.0). Same value
  used for both root and chain steps. SmoothL1/Huber handles the truncated
  tail gracefully.

Applied identically to **root example** and **every chain step**, with
chain steps using the original hero's perspective. The example's
`chips_invested_from_decision_to_end` differs per chain step but the scaler
`new_scale` is one global per-agent value, so all targets in one example
live on a single shared scale → MCTS backup stays consistent.

**No `−mean` shift anywhere.** This is the critical fix for the fold bias.

### 4.1 The "double division" question

Q passes through two normalizations in the implementation:
- `terminal_evaluator(gs) = outcome / search_scale` during MCTS search.
- Final target: `q_chips / new_scale` where `q_chips = root.Q * search_scale`.

Net effect: **Q is divided exactly once by `new_scale`.** The
multiplication by `search_scale` simply undoes terminal_evaluator's
division so we can re-normalize against the cycle's bootstrapped scale.
Equivalent compact formula:
```
q_in_new_scale = root.Q * (search_scale / new_scale)
target = clip(α·q_in_new_scale + (1−α)·realized_in_new_scale, ±clip)
realized_in_new_scale = realized_chips / new_scale
```

When `search_scale == new_scale` (stable cycle, no re-bootstrap), the ratio
is 1 — Q passes through identity. Only after a re-bootstrap or on cycle 0
(fallback `search_scale = BB`) does the scaling factor diverge from 1, and
in those cases the explicit conversion is **needed** to put Q in the
right scale.

## 5. Step 1 — drop mean shift, BB-bootstrap, hybrid blend

Goal: fix the fold-bias source (mean shift) and the bimodal-MC noise
problem without yet touching equity. Should remove 97% fold rate by
itself in a few cycles.

### 5.1 Done already in this branch ✓
- `mcts_train.value_target_alpha` in `config.json`.
- `MCTSTrainingExample.root_q_ratio` and `ChainStep.root_q_ratio` fields.
- `collect_training_data` stores raw chip outcome in `ChainStep.value_target`
  (no normalization) and raw `future_root.Q` in `ChainStep.root_q_ratio`.
- `_make_terminal_evaluator` simplified to BB-only (`outcome / BB`).
  **TODO**: make it scale-aware (next item).

### 5.2 To finish for Step 1

| # | What | Where |
|---|---|---|
| 1 | `_make_terminal_evaluator(hero_pos, init_credits, value_scale)`: returns `outcome / value_scale`. Caller passes `value_scale` from `norm_stats["mcts_value_scale"]` (fallback `BB`). | `collect.py` |
| 2 | `run_mcts_collection`: at start of each agent's search loop, read `search_scale = norm_stats.get("mcts_value_scale", BB)`. Pass to terminal_evaluator. | `collect.py` |
| 3 | `run_mcts_collection`: store raw chip-delta in `ex.value_target` (already done). Store raw `root.Q` in `ex.root_q_ratio` (already done). | `collect.py` |
| 4 | `run_mcts_collection`: after all hands, **bootstrap** `new_scale = std(realized_chip_deltas)` for each agent, write to `norm_stats["mcts_value_scale"]`. | `collect.py` |
| 5 | `run_mcts_collection`: blend + clip per example: `target = clip(α·(root_q_ratio·search_scale + ... etc), ±clip)`. Apply for root AND every chain step. | `collect.py` |
| 6 | `value_target_clip` in `mcts_train` config (default `5.0`). | `config.json` |
| 7 | `pipeline.py:_MCTS_NORM_KEYS` add `"mcts_value_scale"` so the periodic re-bootstrap clears it. Optionally also clear legacy `mcts_ev_*` keys (kept for backward-compat but unused). | `pipeline.py` |
| 8 | Smoke tests: verify (a) cycle 0 with no prior stats falls back to `BB` during search but produces clean targets after bootstrap; (b) α=0 reproduces pure-MC behavior; (c) α=1 produces TD-only targets; (d) bootstrap recomputes scale on demand; (e) clip activates on outliers. | `/tmp/smoke_mcts_train.py` |

### 5.3 Cycle 0 cleanliness

Bootstrap happens **after** the hand-collection phase but **before**
training. So cycle-0 training targets are computed against the freshly
bootstrapped `new_scale` — not against the fallback `BB`. The fallback is
used **only** by `terminal_evaluator` inside the MCTS search of cycle 0
(when no prior `mcts_value_scale` exists yet). That creates a one-cycle
search-time inconsistency between `value_head` (still in GTO scale from
pretraining) and `terminal_evaluator` (in BB-scale). This rapidly
self-corrects within cycle 0's training, after which both are aligned.

## 6. Step 2 — equity backfill ✓ (implemented)

Goal: replace the heuristic terminal evaluator AND the noisy MC realized
outcome with equity-based estimates. After Step 2, value-target variance
drops sharply (mostly only the hand-specific equity remains), and
`new_scale` auto-adapts via the bootstrap mechanism.

**Implementation summary (delivered 2026-05):**
- `agent/mcts/terminal_eval.py`: added `compute_equity_outcome(...)` helper
  that returns both `realized_by_decision` (per-root realized) and
  `equity_by_hero` (cached per-hero equity reused for chain steps); shared
  range-narrowing helpers (`_precompute_combo_probs`,
  `_narrow_by_real_actions`, `_narrow_by_simulated_actions`) between the
  terminal-Q and realized-outcome paths so the math is consistent.
- `agent/mcts/terminal_eval.py:evaluate_all_terminals` now takes
  `value_scales_by_position`; terminal Q is **chip / value_scale** so it
  lives on the same axis the search-time `_make_terminal_evaluator` used.
  Without this, `re_backup_terminals` would mix raw-chip terminal Q into
  normalized-scale ancestor W. Hero-folded-along-path terminals are
  pinned at `−hero_invested` (no equity needed; locked in).
- `agent/mcts/mcts.py:re_backup_terminals` switched to **delta**
  propagation `(new_Q − old_Q) * N`. Old behaviour (`add new_Q * N`)
  double-counted when terminals already had non-zero Q from search-time
  `terminal_evaluator`. Both legacy (Q=0 during search) and current
  (heuristic during search) trees now back up correctly.
- `agent/mcts/collect.py:run_mcts_collection`:
  - tracks `cumulative_bets` per hand using `Table.step`'s 4th return
    value, then derives `credits_pre_distribution = initial − cumulative`.
    Independent of `final_credits`, which the engine's pot-distribution
    code mis-accounts on multi-street showdowns;
  - snapshots `game_state_at_root = gs.clone()` on every decision (needed
    to replay terminal paths in `evaluate_all_terminals`);
  - snapshots `hero_hands` from `table.deck[5+2p : 7+2p]` and the deck
    itself at hand start;
  - post-hand calls `evaluate_all_terminals(...)` BEFORE
    `collect_training_data` so `root.Q` (read into `root_q_ratio`)
    reflects equity backfill;
  - post-hand calls `compute_equity_outcome(...)` and overrides
    `ex.value_target` with the equity-based realized chips; chain steps
    reuse the same `equity_by_hero` with their own
    `chips_invested_from(t+1+i)` for the realized half.
- The original `terminal_eval.py` is loaded lazily inside
  `run_mcts_collection` (not at module import) because
  `gpu_solver_v2` requires `gto_utils` on `sys.path` — a pipeline-time
  setup that unit tests don't always perform.

### 6.1 Component 2A — wire `evaluate_all_terminals` ✓

`terminal_eval.py:evaluate_all_terminals` exists but is never called.

Effect when wired in: for every MCTS terminal in every decision's tree,
fold terminals stay deterministic; showdown terminals get equity computed
from hero's actual cards + opponent ranges narrowed by their action_heads.
The function then calls `re_backup_terminals(root)` to propagate the
corrected terminal Qs back up the tree, so `decisions[t].mcts_root.Q`
reflects equity reality.

Data we need to plumb through:
- `hero_hands: dict[pos -> [c1, c2]]` — actual cards dealt each player.
- `deck: np.array(52,)` — to derive board cards visible at each turn.
- `decisions[t]["game_state_at_root"]: GameState` — needed for terminal
  replay inside `evaluate_all_terminals`. Currently the dict has
  `events_at_root` (perception input) but not the game state object.

Implementation in `run_mcts_collection` (per-hand loop):
1. After `start_table()`: capture `hero_hands` from the shuffled deck.
2. At each decision: store `game_state_at_root = GameState.from_table(table, active_pos)`.
3. After hand ends: build `hand_record = {decisions, deck, hero_hands, num_players, big_blind}`.
4. Build `agents_by_position` mapping from `hand_seated`.
5. Call `evaluate_all_terminals(hand_record, agents_by_position, device, mcts_cfg)`.
6. Now `mcts_root.Q` (root + every chain decision's future_root) reflects equity.
7. Continue with existing `collect_training_data` + Step 1 normalization.

### 6.2 Component 2B — equity-based realized outcome ✓

The realized half of the hybrid currently uses
`final_credits[hero] − credits_at_decision[hero]`. In showdowns this is one
random sample of who actually won, dominated by card randomness.

Replacement: for each example, compute the **expected** outcome at the
actual played-out final state of the hand, using the same equity machinery:
- **Fold-terminated hand** (one survivor): deterministic. Same as today.
- **Showdown-terminated hand**: `realized_eq = equity_hero * final_pot − chips_invested_from_decision`,
  where `equity_hero` comes from `gpu_equity_v2(hero_cards, final_board, opp_ranges)`
  with opp ranges narrowed by the same logic as `evaluate_all_terminals`.

For chain step `i` (representing state `t+1+i`):
- Same equity, same `final_pot` (it's the actual hand's final pot).
- `chips_invested` differs: `credits_at(t+1+i)[hero_pos] − credits_at_showdown[hero_pos]`.
  `hero_pos` stays = the original hero of this example's root.

Implementation:
1. Extract shared helper `_compute_equity_outcome(...)` from
   `evaluate_all_terminals` (probably most of `gpu_equity_v2 + range
   narrowing` block) so the same code serves both 2A and 2B.
2. In `run_mcts_collection`, after the hand: for each example compute
   `realized_eq_chips` (root) and per-chain-step `realized_eq_chips`.
3. Substitute these into the Step 1 blend formula.

### 6.3 Component 2C — re-bootstrap behavior ✓

After 2A + 2B land, the std of `realized_chips` (now equity-based) will
drop sharply — possibly 3-5×. Existing `value_norm_rebootstrap_every` will
recompute `mcts_value_scale` on schedule, automatically adapting the
target scale. Value head needs ~1 cycle to re-converge to the new scale
(unavoidable). The hybrid `α` and `clip` don't need to change.

No code change required for 2C; it's a consequence of the existing
`_finalize_value_targets` bootstrap path reading the new equity-based
realized chips.

### 6.4 What is intentionally NOT zero-sum per hand

Belief-based equity (each hero vs the OPPONENT's position range, not vs
the opponent's actual cards) does **not** satisfy `Σ_p eq_p = 1` on a
single hand. Each hero uses a different opponent distribution; ranges
overlap. That means realized-outcome zero-sum holds **in expectation
across hands**, not per-hand. This is correct for training: hero's
training signal must come from what hero can estimate at decision time
(opp = range), not from ground-truth omniscience. The smoke test
`test_compute_equity_outcome_showdown_zero_sum` was rewritten to verify
the formula `realized = equity_hero * pot − invested` directly and to
sanity-check magnitudes (stronger hand → higher equity), not per-hand
zero-sum.

### 6.5 Engine pot-distribution bug, side-stepped

`env/judger.py:get_reward` is called with `self.bets`, which is
**river-only** at showdown (`Table.next_turn` resets `bets` at each street
advance). So showdown winners get only the river street redistributed,
not the full pot. This is a pre-existing engine bug; the Step 1 code path
that read `final_credits[hero] − credits_at(t)[hero]` was silently
training on this broken reward.

Step 2 sidesteps the bug because:
- terminal Q uses `equity * gs.pot − hero_invested` (full pot);
- realized outcome uses `equity * final_pot − chips_invested_from(t)`
  with `chips_invested` derived from `cumulative_bets`, not
  `final_credits`.

Neither path reads `final_credits`. The fold-terminated branch is
deterministic and matches the engine's correct fold-distribution path.

## 7. Open questions / future work

- **`α` schedule vs. constant**: at cycle 0, `root.Q` is partially
  corrupted by value_head's GTO-scale outputs mixing into MCTS backup.
  Could schedule `α: 0 → 0.5` over the first 2-3 cycles to give Q time
  to stabilize. Currently fixed `α` from config. Revisit after Step 1
  results.
- **Per-agent vs pool-wide bootstrap**: `mcts_value_scale` is per-agent.
  Pool-wide std would preserve zero-sum exactly. But per-agent matches
  current pipeline structure. Not urgent.
- **`clip` value**: 5.0 (= 5 σ) is a guess. Plot the actual distribution
  of `(blend / new_scale)` after Step 1 and tune.
- **CLAUDE.md MCTS row**: the line "value_head: SmoothL1 on root Q (with
  backed-up terminal equity)" is **wrong** for the current code (which
  uses realized outcome, not root.Q with equity). After Step 2 lands,
  CLAUDE.md description finally matches reality.

## 8. Reference: math summary

Notation:
- `Δ_p = final_credits[p] − credits_at_decision[p]` — realized chip delta
  for player `p` from this decision to hand end.
- `s_search` = `norm_stats["mcts_value_scale"]` at search time
  (fallback `BB` if absent).
- `s_new` = freshly bootstrapped scale after collection
  (= `std(Δ_hero)` over current cycle's training examples).
- `Q_raw` = `mcts_root.Q` as stored in the tree (in `s_search`-units).
- `α` = `value_target_alpha`.
- `c` = `value_target_clip`.

Target formula (root and every chain step, identical structure):
```
q_in_new_scale = Q_raw * (s_search / s_new)
realized_in_new_scale = Δ / s_new
target = clip(α · q_in_new_scale + (1−α) · realized_in_new_scale, ±c)
```

For chain step `i`:
- `Q_raw = decisions[t+1+i].mcts_root.Q`
- `Δ = final_credits[hero_root] − credits_at(t+1+i)[hero_root]`
  (hero is the **root example's** hero, not the player acting at `t+1+i`)

For the root example:
- `Q_raw = decisions[t].mcts_root.Q`
- `Δ = final_credits[hero_root] − credits_at(t)[hero_root]`

In stable conditions (no re-bootstrap), `s_search == s_new` and the Q
rescaling is identity. The explicit form is needed only on cycle 0 (where
`s_search = BB` fallback) and immediately after re-bootstrap events.

## 9. What this fixes — expected outcome

After Step 1: 97% fold rate should drop within a few cycles. `mcts_value_scale`
stabilizes around the agent's realized-outcome std. No artificial fold
bonus → value head learns honest EV → MCTS plays balanced strategies.

After Step 2: value-target variance drops further (no card randomness). Q
becomes equity-anchored, providing a strong TD signal that doesn't drift.
Self-play training becomes much more stable across cycles.
