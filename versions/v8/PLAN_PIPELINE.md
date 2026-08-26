# v8 — Implementation plan: from G1 to the full pipeline

**Status: plan only. Nothing here is implemented.**
**Scope: everything `ARCHITECTURE.md` §6 lists as missing.**

This document is the build order for the rest of `CONCEPT.md`. It exists because the work is
too large for one session and the pieces have hard dependencies: the oracle cannot be written
before the driver can resume a hand, the agent cannot be trained before targets exist, and
`CONCEPT.md` §13 says the whole shape of the pipeline may change depending on what one
measurement (G3) returns.

- **Why** each piece exists → `CONCEPT.md`
- **What is on disk today** → `ARCHITECTURE.md`
- **In what order to build the rest, and exactly what each step is** → this file

Project-wide rules — hardware, version discipline, testing budget, engineering principles —
are in the root `CLAUDE.md` and are not restated here. Two of them are load-bearing for this
plan and are worth naming anyway:

- **Composition over new code.** Every new low-level primitive in this plan is marked ⚠ and
  needs owner sign-off *before* it is written (`CLAUDE.md` §5). There are seven of them, and
  they are collected in §1 so the sign-off is one conversation, not seven.
- **No GPU on the dev box.** Every cost figure below is a hypothesis. Two sessions (S4, and
  the runs of S9/S11) execute on the Spark; everything else must be fully exercised on CPU at
  toy scale or it is unverified.

---

## 0. How to use this document

Each numbered section from §3 onward is **one Claude Code session**. Open a fresh session and
say:

> Реализуй раздел S<n> из `versions/v8/PLAN_PIPELINE.md`.

Every session section carries the same eight fields, and they are meant to be sufficient on
their own — a fresh session should not need this conversation's history:

| Field | Meaning |
|---|---|
| **Depends on** | sessions that must be finished first |
| **Reads first** | the files and `CONCEPT.md` sections to read before writing anything |
| **Deliverables** | exact paths, exact public signatures |
| **Design** | the algorithm, the decisions already taken, and why |
| ⚠ **Sign-off** | new primitives or edits to working code — stop and ask if not already agreed |
| **Tests** | the test file and the cases it must contain |
| **Acceptance** | the condition under which the session is done |
| **Non-goals** | what must *not* be built, so scope does not creep |

### Session close-out protocol — every session, no exceptions

1. `cd versions/v8 && python3 -m pytest tests/ -q` — the **whole** battery, green, and report
   its wall clock. The `CLAUDE.md` §4 budget is 30 minutes; today it is ~26 s, and this plan
   roughly triples the test count. If a session pushes past ~10 minutes, say so.
2. Update `ARCHITECTURE.md`: the §1 tree, the §6 "what is still missing" table (strike the row),
   the §7 test table (add the file), and a new subsection under §2 describing what was built.
   `ARCHITECTURE.md` describes what is on disk; it is wrong the moment code lands without it.
3. Report honestly what is **untested because it needs a GPU or real scale**, per `CLAUDE.md` §3.
   Do not describe a GPU path as verified.
4. Do not commit unless asked.

---

## 1. Decisions required before S1 starts

These are the ⚠ items. All of them either introduce a new low-level primitive or edit code that
currently works, which `CLAUDE.md` §5 says to agree before implementing. Answers go into this
section as they are settled.

| # | Question | Recommendation | Blocks |
|---|---|---|---|
| **D1** | `HandSpec` gains an optional `deck` and an optional `forced_actions`. This edits the driver — the piece `CONCEPT.md` §3 calls the largest single item in v8. | **Yes.** A rollout is "this hand, these opponent cards, this prefix, then free play". Without it the oracle either rewrites the driver or replays through a second engine path, and a second path is exactly the silent divergence §5 warns about. The replay-identity test (S1) is strong enough to make the edit safe. | S1 |
| **D2** | `DecisionContext` gains an optional `hole_override`, so the posterior can ask a pool member "what would you have done holding *this*". | **Yes.** `DecisionContext` is already a thin view with `__slots__`; the alternative is fabricating a `HandRecord` per combo, which is both slower and a second construction path for observations. | S2 |
| **D3** | The Qwen3 trunk is extracted from `OpponentEmbeddingNet` into `nets/trunk.py::HandEncoder`, used by both networks. This renames parameters, so **the existing G1 checkpoint stops loading**. | **Extract, and accept the break.** G1 is finished and its report is on disk; `eval_corpus.pkl` and `fitted_vectors.npz` are what post-hoc analysis reads, not the weights. A key-remap shim is available if the owner wants to keep re-evaluating that checkpoint. | S5 |
| **D4** | `gates/g1.py`'s `Session`, `build_sessions`, `play`, `raise_sizes_from` move to `env/session.py`, and G1 imports them from there. | **Yes.** Label generation needs the identical session semantics (button rotation, uniform 2–9 × 10–300 BB, slot 0 is the observer). A copy would let the two drift, and the drift would show up as a train/deploy mismatch nobody could see. | S7 |
| **D5** | Does the agent get a value head now? `CONCEPT.md` §6.1 says no; §7.4's variant C — the first lever if the compute budget does not close — requires one. | **No head now.** Build the baseline literally; if G3 (S4) says variant A does not fit, adding the head is a contained change to `AgentNet` plus a new oracle module. Building it "just in case" is the unrequested addition §5 forbids. **Settled 2026-08-19: G3 ran, variant A stands, D5 does not flip — no value head is being added.** See S4's outcome block. | S5, S6 |
| **D6** | `max_combos` subsampling semantics. `CONCEPT.md` §7.3 says "v7's `gpu_solver_v5` already has this knob and its semantics". | **Read `gto_utils/gpu_solver_v5.py` and reuse whatever it does.** Do not invent a scheme; if v5's is not self-normalised importance sampling, say so and ask rather than silently improving it. | S2 |
| **D7** | The G1 run found both showdown heads memorising the corpus (held-out `class_ce` 5.75 against `ln 169` = 5.13; held-out strength MSE 0.095 against a target variance of ~0.085), and the showdown term contributing nothing to the inference fit (−0.007 ± 0.005 nats pooled). §5.1a's weights are `train` config. | **Leave §5.1a exactly as designed for now, and carry the finding as a risk (§13, R4).** Changing the embedding objective before the pipeline exists means the pipeline is built on a network nobody has measured. The weights are config; the experiment is cheap once there is something to run it against. | — |
| **D9** | The embedding table is `nn.Embedding(n_members, d_emb)` and §8 grows the pool by one member per iteration, so the agent needs a row and the corpus needs a member index for hero's tokens. | **Reserve the rows up front** (owner decision 2026-08-19): the table is sized `len(pool₀) + max_iterations × style.agent_variants`, iteration *k* owns the block starting at `len(pool₀) + k × agent_variants` (its first row is the agent itself, from the moment it is first seated as hero, the rest are its style draws — D11), and retraining is a continuation rather than a rebuild. See `CONCEPT.md` §5.4. | S9 |
| **D10** | PFSP scores accumulate over the whole run, but hero is replaced every iteration, so the quantity being estimated is non-stationary and a member the early agents beat keeps its score forever. | **Forget geometrically** (owner decision 2026-08-19): `pool_sampling.result_decay` scales the accumulated hands and BB once per iteration, and `PoolSampler.end_iteration()` is called by the loop. At 0.8, a member with 20 000 hands of history crosses back over zero twelve iterations after it stops losing; without the decay it takes about sixty. `result_decay = 1.0` is the old lifetime behaviour. | S9 |
| **D11** | Does a trained agent join the pool as one member or as several? `CONCEPT.md` §4.2 says the style layer "applies uniformly to any pool member, v7 or v8", but §4.1 only ever said "the agent joins the pool". | **Several, and the count is config** (owner decision 2026-08-19): `style.agent_variants` — the agent plus `agent_variants − 1` style draws off it, exactly as a v7 checkpoint is expanded at bootstrap. Thirty iterations of one lineage are the most correlated members the pool will ever hold (§11.3 turned on ourselves) and a style draw costs no forward and no parameter. `1` is the no-multiplication setting. Changes D9's arithmetic. | S9 |
| **D12** | A past agent seated as an *opponent* is not a plain `PoolMember`: `AgentPoolMember` is one member per seat and needs an opponent-embedding table of its own, so "what does agent *k−3* believe about its tablemates while it plays" has to be answered. | **Settled 2026-08-19: (a).** `agent/policy.py::FrozenAgentMember` seats past agents at `e = 0`. The recommendation below was taken as written when S9 was built; it is reversible — (b) is a change to one class — and `ARCHITECTURE.md` §5 carries what it costs. Two candidates. *(a)* Seat past agents at `e = 0`, the unconditional policy that §6.2's embedding dropout already trains explicitly: no extra fit, no recursion, and the pool member is a fixed policy like every other. *(b)* Give each past agent its own §5.5 inference fit over its tablemates, which is faithful but nests one fit inside another and multiplies the cost of every rollout. Recommendation is (a); (b) has no cheap form and no measurement asking for it yet. | S9 |

| **D13** | `build_pool` gives variant #0 of an entry a random style draw like every other variant, so an entry without an explicit `style` puts **no unmodified copy of its base** in the pool. In `config_g1.json` and `config_g3.json` that is the case for all seven v7 entries: 56 v7 members, not one of them the plain network. | **Every base is in the pool unmodified as well, and the number of style draws stays per-entry config** (owner decision 2026-08-19). The mechanism needs no code: a second `bootstrap` entry for the same base with `"style": "identity"`, which is exactly what the five degenerate strategies already do in both gate configs. `n_variants` on the styled entry keeps being the knob for how many modifications that base contributes. The one cost is that two entries naming the same checkpoint load it twice — `build_pool` calls `_load_v7_agent` per entry — so S9 should memoise the loaded agent by checkpoint path. That is a cache, not a semantic change: `with_style` already shares one loaded agent across all of an entry's variants. See `CONCEPT.md` §4.1. **Done 2026-08-19:** `_load_v7_agent` takes a per-`build_pool` cache keyed by (checkpoint, arch_config), and `config.json` carries a `"style": "identity"` sibling for each of the seven v7 bases. | S9 |

**D8 — plan-level.** Sessions S5/S6 (the agent) do not depend on S1–S4 (the oracle). The order
below puts the oracle first because `CONCEPT.md` §13 says it is the piece most likely to kill
the design's shape, and finding that out late is expensive. If the owner prefers to see an agent
running first, S5 → S6 → S1 → S2 → S3 → S4 is a valid reordering; nothing else moves.

---

## 2. Build order and dependency graph

```
                     ┌── S1 rollout plumbing ──┐
                     │                         │
  (driver, pool,     ├── S2 posterior ─────────┼── S3 oracle A ── S4 G3 ═══╗
   embedding net     │                         │                           ║ DECISION
   already exist)    │                                                     ║ GATE
                     │                                                     ║  A or C
                     ├── S5 trunk + agent net ── S6 targets + training ─────╢
                     │                                                     ║
                     └── S8 pool sampling ─────────────────────────────────╢
                                                                           ║
                                          S7 label generation ═════════════╣
                                                                           ║
                                          S9 outer loop ═══════════════════╣
                                                                           ║
                             S10 Slumbot adapter ── S11 eval pipeline ══════╝
```

| S | Title | Depends on | Runs on | New code (rough) | Owner action | Status |
|---|---|---|---|---|---|---|
| **S1** | Rollout plumbing in the driver | — | CPU | ~120 + ~250 test | D1 | **done** |
| **S2** | Opponent posterior (§7.2) | S1 | CPU | ~180 + ~300 test | D2, D6 | **done** |
| **S3** | BR oracle, variant A (§7.1) | S1, S2 | CPU (toy) | ~280 + ~400 test | — | **done** |
| **S4** | G3 — what a label costs (§13, §14) | S3 | **Spark** | ~250 + ~150 test | run it; **decide A/C** | **done — ran 2026-08-18 and 2026-08-19; variant A stands** |
| **S5** | Trunk extraction + agent network (§6.1) | — | CPU | ~250 + ~350 test | D3, D5 | **done** |
| **S6** | Targets and agent training (§6.2) | S5 | CPU (toy) | ~300 + ~400 test | D5 | **done** — two losses, see below |
| **S7** | Label generation end to end (§8) | S1–S3, S5, S6 | CPU (toy) | ~300 + ~250 test | D4 | **done** |
| **S8** | Pool sampling (§4.4) | — | CPU | ~200 + ~250 test | — | **done** |
| **S9** | Outer loop `pipeline.py` + `config.json` (§8) | S7, S8 | **Spark** | ~400 + ~300 test | run it | **done** — built 2026-08-19, CPU-only; see below |
| **S10** | Slumbot adapter rewrite (§12) | S5 | CPU | ~300 + ~250 test | — | **done** — 2026-08-19 |
| **S11** | `eval_pipeline.py`, cold/warm, BB/100 ± SE (§12) | S10, S9 | **Spark** | ~250 + ~200 test | run it | **done** — 2026-08-19 |

**Every session in this plan is built.** What remains is not code but runs, and they are the
expensive half: one iteration at size on the Spark, then a screening run against Slumbot, then a
reportable one. `ARCHITECTURE.md` §5 lists what that leaves unverified.

**Two owner requests landed 2026-08-19, after S11:**

* **`./run.sh --version=v8` resumes.** It already did at the phase boundary; labelling — the phase
  budgeted in days — now also writes `labels/progress.json` after every flushed shard, so a crash
  in its middle costs one shard instead of the phase. A mid-labelling resume re-plays the
  iteration's hands and refits the embeddings (minutes against days) because no corpus of records
  is kept on disk. `ARCHITECTURE.md` §2.10, §2.11.
* **The Slumbot run is multiprocess**, `evaluation.n_workers`, as v7's was and for the same
  reason: a hand is an HTTP round trip, and threads lose to the GIL and a single CUDA stream on
  batch-of-one forwards. Each worker is a *session* — its own table, its own §5.5 fit, its own
  file, its own resume — so the warm-up curve stays the per-session quantity §12 asks about and
  there is no shared state. `ARCHITECTURE.md` §2.13.

The **decision gate** after S4 was real and has been passed: **variant A stands, D5 does not
flip, there is no value head.** S4's "Outcome" block records why. Nothing downstream changes
shape.

### What S9 inherits from S4 and S6

`config.json` is S9's deliverable, and four of its values are already decided rather than open:

| Key | Value | Where it was decided |
|---|---|---|
| `oracle.samples_per_action` | **128** | S4 / `CONCEPT.md` §13, §14 — the point where the sample budget stops buying anything |
| `oracle.max_combos` | **`null`** (exact posterior) | `CONCEPT.md` §7.3, OI-9 — the cap's accuracy is unmeasured and deliberately deferred |
| `agent_train.loss` | ⚠ **owner's call** — `soft_q` or `kl`. `config.json` ships `soft_q`, which is what the noise analysis argues for; flipping it is a one-word edit | `CONCEPT.md` §6.2. Both are implemented; `soft_q` is the one the noise analysis argues for, `kl` is the mass-covering baseline. Whichever is chosen, `§11.4` records that the logged `soft_q` number has a noise floor and does not go to zero |
| `agent_train.temperature` | required by `soft_q`, unused by `kl` | `train/targets.py` |

Budget, so S9 sizes its cycles against something real: at 128 samples an iteration of 100 000
labels costs **122–141 Spark-hours** of labelling alone, before any training step. `CONCEPT.md`
§13 has the per-budget table.

---

## S1 — Rollout plumbing in the driver

**Depends on:** nothing. **Unblocks:** S2 (needs `hole_override`'s sibling), S3 (needs both).

### Reads first
`env/driver.py` in full (365 lines — it is the piece being edited), `env/table.py::start_table`,
`nets/features.py::hand_tokens`, `CONCEPT.md` §3 and §15 (the equivalence requirement).

### Deliverables
`env/driver.py`:
```python
@dataclass
class HandSpec:
    ...                                  # unchanged fields
    deck: np.ndarray | None = None       # 52 ints; overrides the dealt deck
    forced_actions: list | None = None   # action indices, consumed in decision order
```
`nets/features.py`:
```python
def hand_tokens(record, observer_pos, slot_of_seat, max_players, n_actions,
                pending=None):
    """... `pending`: a `DecisionContext` for a decision that has not been taken
    yet. Appends one extra token with `action = -1` and `legal` from the context.
    A hand with a pending decision has no showdown, and passing both is refused."""
```

### Design

**Deck override.** In `_start`, immediately after `table.start_table()`:
```python
if spec.deck is not None:
    deck = np.asarray(spec.deck, dtype=table.deck.dtype)
    assert sorted(deck.tolist()) == list(range(52)), "deck must be a permutation of 0..51"
    table.deck = deck
```
`Table` reads cards from `self.deck` everywhere (`table.py:33, 130, 163-175`) and caches nothing,
so this is complete. The layout is fixed by `HandRecord.hole_cards`: `deck[:5]` is the board,
`deck[5 + 2p : 7 + 2p]` is seat *p*'s hand.

**Forced prefix.** In `run`, split the per-round `queries` before grouping by member:
```python
forced, free = [], []
for state, ctx in queries:
    fa = ctx.record.spec.forced_actions
    k = len(ctx.record.decisions)
    (forced if fa is not None and k < len(fa) else free).append((state, ctx))
for state, ctx in forced:
    self._apply(state, ctx, None, ctx.record.spec.seat_members[ctx.acting_pos],
                action_idx=fa[len(ctx.record.decisions)])
```
and give `_apply` an `action_idx=None` parameter: when it is given, skip the sampling block
(`probs` is then unused and must be `None`), keep every line that records the decision and steps
the table. **One recording path, two ways of choosing the action** — that is the whole point.
Assert `ctx.legal_mask[action_idx]`; a forced action taken from a real record is legal by
construction, so a violation is a bug and must be loud.

Forced decisions cost no policy call, which is where the oracle's saving comes from: a rollout
replays a prefix of ~5–15 decisions without touching the network.

**Pending token.** `hand_tokens` currently builds `T = len(record.decisions) + len(revealed)`
tokens. With `pending` it builds one more, from `pending.snapshot` exactly as a decision token is
built, with `action[t] = -1`, `legal[t] = pending.legal_mask`, `token_type[t] = TOKEN_DECISION`.
`prev_action[t]` is the action of `decisions[-1]`, which is known. Assert
`pending is None or not record.showdown` — a hand in progress has not reached showdown, and the
combination would mean the caller is confused about which moment it is observing (`CONCEPT.md`
§9's two moments).

### ⚠ Sign-off
D1. Two optional fields on `HandSpec` and one branch in `_apply`'s action choice.

### Tests — `tests/test_rollout_plumbing.py`
1. **Replay identity (the strong one).** Play 20 hands across 2–9 players and 10–300 BB with the
   toy pool. For each, rebuild the spec with `deck=record.deck` and
   `forced_actions=[d["action_idx"] for d in record.decisions]` and re-run. Assert `deck`,
   `snapshots`, `decisions` and `rewards` are equal element for element. This pins both features
   at once and is the reason D1 is safe.
2. **Deck override deals what was asked.** A hand-built deck; assert `record.hole_cards(p)` and
   the board match it at every street through `DecisionContext.board`.
3. **Partial prefix.** Force the first *k* decisions of a recorded hand and let the rest run free:
   the first *k* decisions match the original, chips are conserved, and the record is complete.
4. **An illegal forced action raises**, with the seat and the mask in the message.
5. **Lock-step ≡ sequential still holds** with both features on — extend the existing property in
   `test_driver_lockstep.py` rather than writing a second one.
6. **Pending token.** `hand_tokens(record, ..., pending=ctx)` returns `len(decisions) + 1` tokens;
   the last has `action == -1` and `legal == ctx.legal_mask`; tokens `0..k-1` are bit-identical to
   the same call with `pending=None` on the truncated record. Also: `pending` together with a
   showdown is refused.

### Acceptance
Test 1 passes for every table size 2–9 and both stack extremes, and the full battery is green.

### Non-goals
No rollout logic, no oracle, no batching strategy. `run`'s public signature does not change.

---

## S2 — Opponent posterior (§7.2)

**Depends on:** S1 (for `DecisionContext` familiarity; strictly it only needs D2).
**Unblocks:** S3.

### Reads first
`CONCEPT.md` §7.2, §7.3. `env/driver.py::DecisionContext`, `pool/base.py::PoolMember`,
`gto_utils/gpu_solver_v5.py` (for D6's `max_combos` semantics — read it, do not invent).

### Deliverables
`oracle/__init__.py`, `oracle/posterior.py`:
```python
def combo_universe(dead_cards):
    """All 2-card combos of the cards not in `dead_cards`. (C, 2) int64."""

def opponent_posterior(record, opp_pos, observer_pos, pool, n_actions,
                       through_decision, floor=1e-6, max_combos=None, rng=None):
    """Reach-weighted posterior over `opp_pos`'s holding, from `observer_pos`'s view.

    Returns (combos, weights) — (C, 2) int64 and (C,) float64 summing to 1.
    `through_decision` is an index into `record.decisions`; only decisions at or
    before it are conditioned on, so the posterior is a prefix function.
    """
```
`env/driver.py`: `DecisionContext.__slots__` gains `hole_override`, and the `hole_cards` property
returns it when set.

### Design

```
w(combo) ∝ prior(combo) · Π_t max(floor, P_i(a_t | combo, history_t))
```

1. **Dead cards** = the board as visible at `through_decision`'s street, plus `observer_pos`'s own
   two cards. That is the card removal `CONCEPT.md` §7.2 requires and it is *relative to the
   observer* — hero knows its own cards and the board and nothing else.
2. **Prior** is uniform over `combo_universe(dead)`.
3. For each recorded decision *t* by `opp_pos` with index ≤ `through_decision`: build `C` contexts
   that are copies of the real `DecisionContext` differing only in `hole_override`, call
   `pool[member].policy(contexts)` **once** for all `C`, and multiply the weights by
   `p[:, a_t]`, floored.
4. Normalise. If every weight underflows, fall back to the prior and log it — that means the
   member assigns zero probability to what it did, which is a bug in the member, not in the data.
5. `max_combos`: apply D6's semantics from `gpu_solver_v5`.

**Cost.** `C` is 1326 minus removals, and a member is queried once per decision it made, so an
opponent with 5 decisions costs 5 batched forwards of ~1200 rows. `CONCEPT.md` §13 budgets this
as "the same order again" as the rollouts, and that is the figure S4 will replace with a
measurement.

**Independent marginals.** §7.3 declares the joint over several opponents to be approximated by
independent marginals with a card-removal correction. This module returns *one opponent's*
marginal; the correction lives in S3, where the joint sample is drawn. Keep them separate — the
approximation is easier to reason about when the exact part is not tangled with it.

### ⚠ Sign-off
D2 (`hole_override`), D6 (`max_combos` semantics).

### Tests — `tests/test_posterior.py`
1. **Hand-computed example.** Heads-up, a two-member toy pool whose policy is an explicit function
   of the hole cards (e.g. "raise iff the hand is a pair"). Two decisions. Compute the posterior by
   hand in the test and assert equality to `1e-12`.
2. **Card removal.** No returned combo contains a board card or one of the observer's cards; the
   returned count equals `C(52 − 5 − 2, 2)` on the river.
3. **Normalisation.** Weights sum to 1 for every prefix length, including zero decisions.
4. **A card-independent member leaves the prior alone.** Run `always_call` for any number of
   decisions and assert the posterior is still uniform to `1e-12`. This is the behavioural version
   of "the likelihood is constant in the combo" and it catches indexing bugs that a numeric
   example would not.
5. **Prefix property.** The posterior through decision *k* is bit-identical whether or not the
   record contains decisions after *k*. `CONCEPT.md` §9's no-future-leak rule, in the oracle.
6. **Degenerate cases.** Exactly one legal combo → weight 1. Every likelihood zero → the logged
   fallback, not a NaN. An opponent who never acted → uniform.
7. **`max_combos`** returns at most that many combos, still normalised, deterministic under a seed.

### Acceptance
Tests 1 and 4 pass exactly (not approximately), and the module never reads a card the observer
could not see — asserted by test 2 and by construction of `dead`.

### Non-goals
No joint sampling, no rollouts, no caching layer. One opponent, one marginal, one prefix.

---

## S3 — BR oracle, variant A (§7.1)

**Depends on:** S1, S2. **Unblocks:** S4, S7.

### Reads first
`CONCEPT.md` §7.1, §7.2, §7.3, §13, §15. `env/driver.py` (`run`, `HandSpec`), `oracle/posterior.py`.

### Deliverables
`oracle/rollout.py`:
```python
@dataclass
class OracleConfig:
    samples_per_action: int = 256
    max_combos: int | None = None
    likelihood_floor: float = 1e-6
    batch_hands: int = 2048
    max_collision_retries: int = 32

@dataclass
class LabelStats:
    forwards: int
    seconds: float
    collision_rate: float
    n_rollouts: int

def action_values(record, decision_idx, driver, pool, hero_member_idx, cfg, rng):
    """Q for every legal action at `record.decisions[decision_idx]`, in BB.

    Returns (q, legal, stats): q is (n_actions,) float64 with `nan` at illegal
    actions, legal is the recorded mask, stats is a `LabelStats`.
    """
```

### Design

At one hero decision (hero = the seat acting at `decision_idx`):

1. `legal` = the recorded `legal_mask`. Illegal actions are never rolled out and their `q` is
   `nan` — S6's target construction masks them, and `nan` makes a masking bug fail loudly instead
   of quietly contributing a zero.
2. For every **other live seat**, `opponent_posterior(..., through_decision=decision_idx − 1)`.
   Seats that have folded hold nothing that matters and are skipped.
3. **Joint sample** (§7.3's declared approximation): draw each opponent's combo independently from
   its marginal; if two opponents share a card, or a card is dead, redraw — up to
   `max_collision_retries`, then drop the sample. Report the fraction rejected in
   `stats.collision_rate`; it is the size of the approximation and it belongs in the log, not in a
   comment.
4. For each legal action *a* and each sample *s*, build a `HandSpec`:
   - `deck` = `record.deck[:5]` (the real board, since the runout is not hero's choice), hero's
     real cards at hero's seat, the sampled cards at the opponents' seats, and every remaining
     card filled in arbitrarily but **consistently**, so `Judger` sees a valid 52-card deck;
   - `forced_actions` = `[d["action_idx"] for d in record.decisions[:decision_idx]] + [a]`;
   - `seat_members` = the real members, except hero's seat, which is `hero_member_idx`;
   - `start_credits`, `big_blind`, `small_blind`, `raise_sizes` copied from `record.spec`;
   - `seed` = a deterministic hash of `(record.spec.seed, decision_idx, a, s)` so the label is
     reproducible and independent of batch composition.
5. **One `driver.run(specs, batch_size=cfg.batch_hands)` for the whole `|A| × S` set.** This is the
   composition that makes the oracle affordable: every (action, sample) pair is an independent
   hand, so a single hero decision is one lock-step batch of up to ~2 560 hands, which is the batch
   size the GB10 wants (`CLAUDE.md` §3 — memory-bound, prefer large batches).
6. `q[a] = mean over s of rewards[hero_seat] / big_blind`.

**The card-leak rule, stated once.** The opponents' cards are fixed *in the deck*. Hero's member
is queried through `DecisionContext`, whose `hole_cards` reads `record.hole_cards(acting_pos)` —
hero's own seat. There is no path by which hero's policy sees an opponent's holding, and test 3
asserts it by instrumentation rather than by reading the code.

**Why the board is the real one.** Hero's action does not change the runout, and the posterior was
computed with the real board dead. Re-dealing the board would make `Q` an average over runouts
that the posterior already conditioned away — a different and wrong quantity.

### Tests — `tests/test_oracle.py`

Determinism is not optional here (`CLAUDE.md` §4, and the owner's standing rule): every test
below is exact, none asserts "within a Monte-Carlo tolerance".

1. **Fold is exact and needs no rollout.** For any configuration, `q[FOLD]` must equal
   `−(hero's cumulative contribution at that decision) / big_blind`, to `1e-12`. This is a closed
   form, it exercises the whole pipeline end to end, and it holds at every table size and stack
   depth.
2. **Exact `Q` against enumeration, made deterministic.** A heads-up, 10 BB, preflop-only case with
   a pool of two *deterministic* members (a degenerate strategy plus a hand-thresholded one), so a
   rollout is a deterministic function of the opponent's combo. Enumerate the posterior in the test,
   compute `Q` as an exact weighted sum, and assert equality with `samples_per_action` large enough
   that every combo is drawn (or with `max_combos=None` and a sampler that enumerates). Any
   remaining gap is a bug, not noise.
3. **No card leak.** Wrap hero's member in a recorder that stores every `ctx.hole_cards` it is
   handed; assert the set equals `{hero's real hand}` over thousands of rollouts.
4. **Chip conservation** in every rollout record — reuse the existing engine assertion.
5. **Illegal actions carry `nan`**, and `legal` matches the record's mask exactly.
6. **Reproducibility.** Two calls with the same `rng` seed return bit-identical `q`.
7. **Prefix independence.** `q` at `decision_idx` is unchanged if the record's decisions after that
   index are deleted — the oracle must not read the future of the hand it is labelling.
8. **Collision accounting.** A 9-handed case where the marginals overlap heavily: assert
   `stats.collision_rate` is what a hand count says it should be, and that dropped samples reduce
   the divisor rather than being counted as zeros.

### Acceptance
Tests 1, 2, 3 and 7 pass exactly. `LabelStats` is populated — S4 reads it and nothing else.

### Non-goals
No variant C, no value bootstrapping, no caching of posteriors across hero decisions in one hand
(that is an optimisation S4 may justify, and §7.2 notes the posterior is amortisable — but
measure first). No dataset writing; that is S7.

---

## S4 — G3: what an oracle label actually costs

**Depends on:** S3. **Unblocks:** the decision gate. **Runs on the Spark.**

### Reads first
`CONCEPT.md` §13, §14 (G3), §7.4. `gates/g1.py` — G3 is the same shape of experiment and should
reuse its scaffolding (config loading, `Logger`, report writing, `utils.progress`).

### Deliverables
`gates/g3.py`, `config_g3.json`.
```bash
cd versions/v8 && python3 -m gates.g3 --config config_g3.json
```
Output `data/v8/g3/<timestamp>/g3_report.json` and a printed table.

### Design
Build a realistic pool (the G1 `bootstrap` section, real v7 checkpoints), play a few hundred hands
with the driver, then label a fixed set of hero decisions under a **sweep**:

| Axis | Values (config) |
|---|---|
| `samples_per_action` | 32, 64, 128, 256 |
| `max_combos` | None, 512, 128 |
| table size | 2, 6, 9 |
| stack depth | 20, 100, 300 BB |

Per cell report: **wall-clock per label**, **policy forwards per label** (from `LabelStats`),
rollout depth actually seen, collision rate, and the resulting **labels per GPU-hour**. The
headline output is the sentence `CONCEPT.md` §13 wants: at *X* labels per hour, an iteration of
*N* labels costs *T*, and that fixes `oracle.samples_per_action` in the pipeline config.

One global `tqdm` bar over labels across the whole sweep, `unit="label"` (`CLAUDE.md` §5).

⚠ **Optional, needs sign-off:** also record the **standard error of `q` against
`samples_per_action`** by splitting each cell's samples in half. Cost is zero — the rollouts are
already run — and without it the sweep says how much samples cost but not how many are needed. It
is an addition to what §14 asks for, so it is the owner's call.

### Tests — `tests/test_g3_gate.py`
Small and structural, like `test_g1_gate.py`: a toy config runs end to end on CPU; the report has
one entry per sweep cell; `LabelStats.forwards` is monotone in `samples_per_action`; the bar reaches
its total. Correctness of the labels themselves is S3's job and is not re-tested here.

### Acceptance
The gate runs on the Spark and produces the table. **Then stop and read it with the owner.**

### The decision gate
- **Variant A fits** → continue to S5 unchanged.
- **Variant A does not fit** → §7.4's variant C. Consequences, so they are not a surprise: D5
  flips (the agent gets a value head), S6 gains a bootstrapped target and a value loss, and a new
  `oracle/bootstrap.py` replaces `rollout.py` at the root of the label path. S3's posterior, S1's
  plumbing and S5's network are all unaffected — which is why the order in this plan is what it is.

### Outcome — read with the owner 2026-08-18 and 2026-08-19

**Variant A stands. Variant C is not being built.** The gate ran twice; `CONCEPT.md` §13 carries
both tables. What decided it:

1. **Cost fits.** 122–141 h per 100 000 labels at the chosen budget. Never the binding
   constraint.
2. **Noise was the real question, and the first run measured it wrong** — the runout was pinned,
   hero was whoever the hand seated, and the error was taken on levels rather than on the
   differences the target depends on. Three fixes landed between the runs (`CONCEPT.md` §7.1,
   §14) and the second run behaves as textbook Monte-Carlo: `SE ~ n^-0.5`, **no plateau**, so
   there is no irreducible floor left to escape from.
3. **`oracle.samples_per_action = 128`** — the point where the sample budget stops buying
   anything. S9 takes this as given.
4. **Variant C's advertised 10–20× cost saving does not exist** at the measured rollout depths;
   it is worth 20–40% of a label. Its case is variance, not budget, and `CONCEPT.md` §7.4 has
   been corrected. It also cannot touch the rollouts that end before hero acts again, which are
   the noisiest ones.
5. **The bias-from-noise problem was solved in the loss instead** (`CONCEPT.md` §6.2's `soft_q`),
   which is cheaper than a value head and does not add the value-error feedback loop §11.1 would
   have had to absorb. D5 therefore does **not** flip: no value head.

Two things the gate did *not* settle and that are recorded rather than closed: the accuracy of
`max_combos` (OI-9, deferred — the baseline uses the exact posterior) and the contrast error,
which is now recoverable offline from `g3_report.json` because the split-half gaps are stored
signed.

### Non-goals
No tuning, no pipeline, no attempt to make the number better. G3 measures; it does not optimise.

---

## S5 — Trunk extraction and the agent network (§6.1)

**Depends on:** nothing (can run in parallel with S1–S4). **Unblocks:** S6, S7, S10.

### Reads first
`CONCEPT.md` §6.1, OI-4, §5.1, §5.2. `nets/embedding_net.py` in full, `nets/tokeniser.py`,
`nets/features.py`, `pool/base.py`.

### Deliverables
`nets/trunk.py`:
```python
class HandEncoder(nn.Module):
    """Tokeniser + RoPE + Qwen3 layers + final norm. The §5.1/§5.2 trunk, shared
    as *code* by the embedding network and the agent (OI-4). Weights are not shared."""
    def __init__(self, cfg, n_actions, max_players): ...
    def forward(self, batch, emb):  # -> (B, T, d_model)
```
`nets/embedding_net.py`: `OpponentEmbeddingNet` holds a `HandEncoder`; `hidden()` delegates to it.
Everything else — the member table, the amortised head, the two showdown heads, `objective`,
`fit_embeddings` — is untouched.

`nets/agent_net.py`:
```python
class AgentNet(nn.Module):
    """Entity 3 (§6.1). HandEncoder + Linear(d_model, n_actions), read at each
    hand's last real token. No value head (§6.1, D5). No search."""
    def forward(self, batch, emb): ...   # -> (B, n_actions) logits at the last real token
```
`agent/policy.py`:
```python
class AgentPoolMember(PoolMember):
    """The agent wearing the pool-member interface, so the driver, the oracle and
    the Slumbot adapter all reach it through one surface and none of them needs a
    special case for 'hero is the agent' vs 'hero is a v7 member' (§7.1)."""
    def __init__(self, net, embeddings, slot_of_seat, max_players, n_actions,
                 observer_pos, device): ...
    def logits(self, contexts): ...
```

### Design

**The extraction (D3).** `OpponentEmbeddingNet.__init__` currently builds the tokeniser, the RoPE
module, the layer list and the norm inline (`embedding_net.py:104–125`), and `hidden()` runs them.
Move exactly that into `HandEncoder`, change nothing about the computation, and let both networks
own one. The acceptance criterion is that **`test_embedding_net_masking.py` and
`test_inference_fit.py` pass unchanged** — if a single assertion has to move, the refactor changed
behaviour and must be reverted.

**`AgentNet` reads the last real token.** The hand batch is padded; take the index of the last
`True` in the padding mask per row. Logits are returned **unmasked** — legality masking happens in
one place, `train/targets.py` and `AgentPoolMember.policy`, both of which call
`env.legal.legal_action_mask`'s output as carried on the token. Two masking sites in the network
would be the "one implementation, not two" violation §6.2 warns about.

**`AgentPoolMember.logits`.** For each pending `DecisionContext`:
`hand_tokens(ctx.record, observer_pos=ctx.acting_pos, ..., pending=ctx)` — hero observes from its
own seat — then `collate`, then one `AgentNet` forward for the whole group. The driver already
groups queries by member (`driver.py:215`), so this is one batched forward per lock-step round,
which is the whole reason the interface is shaped this way.

`embeddings` is a `(max_players, d_emb)` tensor of the currently fitted vectors, indexed by slot
via `slot_of_seat`. Hero's own vector sits at hero's slot (§5.3 — hero has an embedding too).
Cold start is zeros (§5.5).

### ⚠ Sign-off
D3 (the extraction breaks the G1 checkpoint's parameter names), D5 (no value head).

### Tests — `tests/test_agent_net.py`
1. **The refactor is behaviour-neutral.** The entire existing battery passes with no edits. State
   this explicitly in the session report; it is the acceptance criterion, not a formality.
2. **Shape and last-token selection.** Ragged hands in one batch; assert the logits come from each
   row's own last real token, by constructing a batch where the padded positions would give a
   different answer.
3. **Observation parity.** Run `AgentPoolMember` through the same harness
   `test_observation_parity.py` uses: no unrevealed hole card, board never ahead of the street, the
   pending token carries no action.
4. **A valid distribution over legal actions** for every table size 2–9 and stack depths 10 and
   300 BB, through `PoolMember.policy`.
5. **`e = 0` is seat-blind.** With every slot's embedding zero, permuting which member sits where
   (holding the hand fixed) leaves the logits bit-identical. This is the property embedding dropout
   (§6.2) exists to create, and it should hold structurally from the start.
6. **Determinism** under a fixed seed, and the network's parameters are unchanged by a `policy` call.

### Acceptance
Test 1 — the whole pre-existing battery green after the extraction — plus 3 and 5.

### Non-goals
No training, no targets, no value head, no search, no warm start from the embedding network's trunk
(§6.1 keeps that as a config flag, off by default; it is not this session's work).

---

## S6 — Targets and agent training (§6.2)

**Depends on:** S5. **Unblocks:** S7.

### Reads first
`CONCEPT.md` §6.2 in full, and `versions/v7/ARCHITECTURE.md`'s "MCTS value-target normalization"
section — that is the scar this session exists to avoid re-opening.

### Deliverables
`train/targets.py`:
```python
def policy_target(q, legal, pot_bb, facing_bet_bb, temperature, divisor="pot_plus_bet"):
    """softmax(Q_normalised / T) over legal actions (§6.2).

    `q` in BB with `nan` at illegal actions. Returns (n_actions,) summing to 1
    with exact zeros off `legal`.
    """

def kl_loss(logits, target, legal):
    """KL(target ‖ softmax(masked logits)), averaged over the batch."""
```
`train/agent_train.py`: the training loop — dataset iteration, embedding dropout, optimiser,
schedule, `tqdm` over gradient steps, `Logger` output.

### Design

**Normalisation, which is the whole session.** `q_norm = q / (pot_bb + facing_bet_bb)`. The divisor
is a config choice (`CONCEPT.md` §6.2) so the alternative can be tried without editing code. Raw
EVs in a 300 BB pot and a 10 BB pot differ by more than an order of magnitude; one temperature over
raw EVs gives a near-deterministic policy in big pots and a near-uniform one in small ones, and
v7's 97 % fold rate was exactly this bug. Test 5 below is its regression test and it is the most
important assertion in this session.

**Masking.** Illegal actions are set to `−inf` before the softmax so they receive *exact* zero, not
`1e-30`. The mask is the one the environment produced — carried on the token, from
`env.legal.legal_action_mask`. §6.2 also says "dominated"; `env/legal.py` already drops dominated
raise bins, so there is nothing extra to do and the plan records that rather than adding a second
rule.

**Embedding dropout (§6.2).** With probability `p`, a slot's embedding is replaced by zeros for
that hand, independently per slot per hand. This is what makes `e = 0` a *usable unconditional
policy* rather than "population average at best, arbitrary at worst" — the policy hero plays
against anyone it has not observed. Applied **only** in agent training; `CONCEPT.md` §5.4 forbids
it when training the embedding network.

### Tests — `tests/test_targets.py`
1. **Hand-computed target** for a small `q`, exact to `1e-12`.
2. **Illegal actions get exact zero**, and the legal mass sums to exactly 1.
3. **Temperature limits.** `T → 0` gives the argmax one-hot; `T → ∞` gives uniform over legal.
   Assert at the limits reachable in float, not "approximately".
4. **Degenerate inputs.** One legal action → one-hot. All legal `q` equal → uniform. Every `q`
   `nan` except one → one-hot.
5. **The v7 scar.** Two situations identical up to a scale factor of 30 on pot, facing bet and every
   `q` produce **the same target** to `1e-12`. If this fails, the normalisation is wrong and the
   agent will fold too much in small pots or too little in big ones.
6. **`kl_loss` is exactly zero** when the masked prediction equals the target, and strictly positive
   otherwise; gradient flows to the logits and not to the target.
7. **Embedding dropout.** `p = 0` leaves every vector intact, `p = 1` zeroes all of them, and a
   fixed seed reproduces the same mask; the dropout is per slot per hand, not per token.
8. **A toy training run** of ~50 steps on synthetic labels reduces the loss monotonically in the
   mean and leaves the pool members untouched.

### Acceptance
Test 5 passes, and the toy run in test 8 completes on CPU in seconds.

### Non-goals
No value loss (D5), no dataset format decisions (S7 owns those), no LR search.

---

## S7 — Label generation, end to end (§8)

**Depends on:** S1, S2, S3, S5, S6. **Unblocks:** S9.

### Reads first
`CONCEPT.md` §8, §5.4, §5.5, §9. `gates/g1.py:78–174` (`Session`, `build_sessions`, `play`) — this
session moves them.

### Deliverables
`env/session.py` — `Session`, `build_sessions`, `raise_sizes_from`, `play`, moved verbatim from
`gates/g1.py`, which then imports them (D4). No behaviour change; `test_g1_gate.py` must stay green
untouched, and that is how the move is verified.

`train/generate.py`:
```python
def generate_labels(driver, pool, sampler, embed_net, agent_member, cfg, out_dir, log):
    """Play sessions, fit opponent embeddings, label hero's decisions, write shards.

    Returns a manifest dict: shard paths, label count, aggregate `LabelStats`.
    """
```

### Design

Per session:
1. Sample the table configuration uniformly — 2–9 players, 10–300 BB (`CLAUDE.md` §1) — and the
   members through `sampler` (S8; a uniform sampler until then). Hero is slot 0, as in G1
   (`ARCHITECTURE.md` §4 interpretive decision 6).
2. Play `hands_per_session` hands with the driver, hero seated as `agent_member`.
3. **Refresh embeddings every `R` hands** (§5.5) by `fit_embeddings` over the hands hero was in —
   which, in a session, is all of them. Between refreshes the vectors are stale, deliberately.
   `R`, `K`, `fit_lr`, `fit_reg` come from the `embedding_net` config section.
4. For every decision in those hands **taken by hero**, call `oracle.action_values` and write a
   record: the tokenised prefix with the pending decision (S1), the legal mask, `q`, `pot_bb`,
   `facing_bet_bb`, the slot embeddings in force at that moment, and the table metadata.
5. Shard to `data/v8/<experiment>/iter_<n>/labels/shard_<k>.npz`.

**The bar.** One global `tqdm` over **hero decisions across the whole job** — that is the unit that
costs and the unit whose ETA anyone wants. Not one bar per session (`CLAUDE.md` §5).

**Order matters and is easy to get wrong.** The embedding used to label a decision must be the one
hero actually held when it acted — fitted from the hands *before* the refresh point, never from the
hand being labelled. Test 3 pins it.

### ⚠ Sign-off
D4 (moving G1's session machinery).

### Tests — `tests/test_label_generation.py`
1. **The move is behaviour-neutral:** `test_g1_gate.py` passes with no edits.
2. **A toy end-to-end run** on CPU: 4 sessions × 6 hands, tiny oracle settings, produces the
   expected number of labels, every target is a valid distribution, every legal mask matches the
   record.
3. **No future leak into the embedding.** The vector attached to a label is a function of the hands
   before the refresh only: replay with the later hands mutated and assert the stored vectors are
   bit-identical.
4. **Observation parity** on the stored token prefixes, through the existing harness.
5. **Determinism** — the same seed produces byte-identical shards.
6. **Uniformity** — over many sessions the sampled table sizes cover 2–9 and stacks cover 10–300 BB;
   assert the exact multiset produced by a fixed seed, not a statistical property.

### Acceptance
Tests 1, 3 and 5.

### Non-goals
No pool sampling policy (S8), no agent training call (S9 orchestrates), no distributed sharding.

---

## S8 — Pool sampling (§4.4)

**Depends on:** nothing. **Unblocks:** S9.

### Reads first
`CONCEPT.md` §4.4, §11.1, §11.3. `pool/build.py`.

### Deliverables
`pool/sampling.py`:
```python
class PoolSampler:
    """PFSP + embedding-space dedup + a uniform floor (§4.4)."""
    def __init__(self, n_members, cfg, rng): ...
    def update(self, member_idx, hero_bb_per_100): ...   # results feed PFSP
    def set_vectors(self, vectors): ...                  # (n_members, d_emb) for dedup
    def sample_table(self, num_players): ...             # -> list of member indices
    def state_dict(self) / load_state_dict(...)          # survives a pipeline restart
```

### Design
- **PFSP**: `P(i) ∝ f(loss rate against i)` with `f(x) = x ** exponent`, exponent from config.
  Members never played get the maximum weight, so a newly appended agent is sampled immediately.
- **Dedup**: k-means (or agglomerative) over the members' embedding vectors into `n_clusters`;
  sample a cluster first, then a member inside it. This is what stops 500 near-duplicate
  checkpoints of one lineage from each taking a full share (§11.3). Vectors come from the embedding
  network's trained table via `set_vectors`.
- **Uniform floor**: with probability `floor_fraction`, ignore both and sample uniformly, so no
  style is ever fully evicted. §4.4: "nothing is deleted, things are down-weighted."

### Tests — `tests/test_pool_sampling.py`
Deterministic throughout — assert exact sequences under a fixed seed, never a statistical property.
1. A fixed score vector and seed produce an exact sequence of sampled indices.
2. The uniform floor fires exactly `floor_fraction` of the time over a fixed-seed draw of known
   length.
3. An unplayed member is sampled within its first *n* draws.
4. Clustering a hand-built vector matrix with obvious structure recovers that structure; two
   identical vectors never both appear in one table more often than the cluster rule allows.
5. `state_dict` / `load_state_dict` round-trips and reproduces the next draw exactly.
6. A table of `num_players` distinct seats is returned for every size 2–9, hero excluded.

### Acceptance
Tests 1, 2 and 5.

### Non-goals
No exploiter agents, no league structure (§11.1 mentions AlphaStar's; it is not in `CONCEPT.md`'s
baseline and is not to be built).

---

## S9 — The outer loop: `pipeline.py` and `config.json` (§8)

**Depends on:** S7, S8. **Unblocks:** S11. **Runs on the Spark.**

### Reads first
`CONCEPT.md` §8, §8.1, §13. `gates/g1.py::run` (the phase/logging/report idiom to follow),
`utils.py::Logger`.

### Deliverables
`pipeline.py`, `config.json` with exactly the sections `CONCEPT.md` §8.1 names: `bootstrap`,
`style`, `agent_init`, `embedding_net`, `oracle`, `pool_sampling`, `game`, `agent_train`,
`evaluation`.

**`bootstrap` must put every base in the pool unmodified as well as styled** (D13): one entry
with `"style": "identity"` and one with `n_variants` style draws, per base — the pattern the five
degenerate strategies already follow in `config_g1.json` and `config_g3.json`, extended to the v7
checkpoints, which have no unmodified member in either gate config today. The plain networks are
the strongest members of the pool and the reference point every style draw is a perturbation of;
`CONCEPT.md` §4.1 lists them as a pool ingredient in their own right. Memoise the loaded v7 agent
by checkpoint path while doing this, or the two entries load the same weights twice.
```bash
./run.sh --version=v8      # → cd versions/v8 && python3 pipeline.py
```

### Design
Iteration *n*:
```
1. sample tables and members                             (S8)
2. play hands, hero = agent (n ≥ 1) or agent_init member (n = 0, §7.1)
3. fit / refresh opponent embeddings                     (§5.5)
4. oracle labels at hero decisions                       (S3, S7)
5. train the agent on KL to softmax(Q_norm / T)          (S6)
   — on all but `agent_train.heldout_fraction` of the labels
6. measure the oracle gap on the held-out labels         (§8, below)
7. every `embedding_retrain_every` iterations, retrain the embedding network
   on the enlarged history corpus                        (§8)
8. append the trained agent to the pool as `style.agent_variants`
   members — the agent plus its style draws              (§4.1, D11)
9. `sampler.end_iteration()` — age the PFSP results      (S8, D10)
```

**The oracle gap (§8, owner decision 2026-08-19).** Every checkpoint is measured against the
oracle that taught it, on the slice of that iteration's labels training never saw. Seven numbers
— `kl`, `ev_agent`, `ev_oracle`, `q_best`, `ev_gap_target`, `ev_gap_greedy`, `agreement` (the
`pipeline.GAP_KEYS` tuple) — plus the same seven broken down by table size and by stack depth,
into `iter_<n>/metrics.json`. The two gaps are differences of the three terms reported beside
them, which is why those terms are reported: a gap that moved does not say which of its sides
moved. It costs one agent forward per held-out
decision and **no new rollouts**: the oracle's answer is already in the shard, so this is a
`softmax` over stored `q` and one batched forward, not a second labelling pass. `CONCEPT.md` §8
carries the definitions and, importantly, what the number is *not* — it is scored by the oracle's
own noisy `Q` on hero's own state distribution, so it bounds nothing about exploitability and its
floor is G3's Monte-Carlo error rather than zero. What it answers is whether this iteration's
training absorbed this iteration's labels, separately from whether the labels were any good.

**Iteration 0 is different and only in one way** (§7.1, OI-2 revised): the agent is trained from
scratch, and a **v7 pool member sits in hero's seat**, named by the `agent_init` config section —
which is deliberately separate from `bootstrap`, because the strongest opponent to have in the pool
and the best policy to seat as hero are different questions. From iteration 1 hero is the agent and
labels are on-policy (OI-3).

**Layout.** `data/v8/<experiment>/iter_<n>/{labels/,agent.pt,embedding.pt,metrics.json}` and
`data/v8/logs/<timestamp>.txt`. v8 never writes into `data/v7/` (`CLAUDE.md` §2).

**Checkpoint and resume.** Every phase writes its artefact before the next starts, and the pipeline
can resume at a phase boundary. A run on the Spark is measured in days (§13); a crash in phase 5
must not cost phase 4.

**Progress and logging.** One bar per phase, unit named for what it counts (hands, labels, steps),
`smoothing=0` through `utils.progress`. Phase boundaries, counts and losses also go to the `Logger`
as text — a bar is not a record (`CLAUDE.md` §5).

### Tests — `tests/test_pipeline.py`
A 2-iteration toy pipeline on CPU, everything shrunk to seconds:
1. It runs to completion and writes every expected artefact.
2. The pool grows by exactly `style.agent_variants` members per iteration.
3. Iteration 0 seats the `agent_init` member and iteration 1 seats the agent — asserted by
   inspecting who was queried, not by reading config.
3b. The held-out labels reach the metric and never the optimiser: the gap is computed on
   decisions absent from every training batch, and a run with `heldout_fraction = 0` reports no
   gap rather than a gap on training data. On a hand-built pair of distributions the seven numbers
   match values computed by hand, and `ev_gap_greedy` is zero exactly when the agent puts all its
   mass on the oracle's best action.
3c. The pool grows by `style.agent_variants` members per iteration, each with its own embedding
   row, and the reserved table is exactly `len(pool₀) + max_iterations × agent_variants`.
4. Resume from an interrupted phase produces byte-identical artefacts to an uninterrupted run.
5. Table sizes and stack depths over the run cover 2–9 and 10–300 BB with no weighting toward
   heads-up or 200 BB — the `CLAUDE.md` §1 compliance test, asserted on the exact multiset.

### Acceptance
Tests 1, 3, 4 and 5 on CPU; then one real iteration on the Spark, with wall clock per phase
reported against G3's prediction.

### Outcome — built 2026-08-19, not yet run at size

`pipeline.py`, `config.json` and `tests/test_pipeline.py` are on disk; the battery is 331 tests
in ~84 s. `ARCHITECTURE.md` §2.11 describes what was built. Five things were decided while
building it that the plan left open or did not name at all, and each one is a place a later
session should look first if the loop misbehaves:

1. **D12 → (a).** `agent/policy.py::FrozenAgentMember` seats a past agent at `e = 0`. It also had
   to answer `hole_override` and observe a *historical* decision, because the §7.2 posterior asks
   that of every member it conditions on — `AgentPoolMember` refuses both by design. No signature
   of an existing module changed: the truncation and the deck swap happen inside the new class,
   sharing `pool/v7_member.py`'s `_deck_seen_by`.
2. **The embedding network runs first, not last.** §8's step list puts the retrain at the end of
   an iteration; taken literally that leaves iteration 0's labels carrying vectors fitted by a
   network still at its initialisation. It runs at the *start* of an iteration instead, which is
   what §8's own "that is both gate G1 and the first pipeline phase" says.
3. **Its corpus is pool self-play, replayed per retrain rather than accumulated.** The
   enlargement is the pool: iteration *k*'s corpus contains agents 0 … *k*−1 as opponents.
   Replaying is what makes resume byte-identical without keeping every hand ever played. The cost
   is that hero's own hands never enter the corpus — see `ARCHITECTURE.md` §2.11.
4. **`sampler.update` is called, though the step list does not name it.** PFSP with no results is
   dead weight, so `train/generate.py`'s manifest now returns hero's BB and hand count against
   every member it sat with. A hand is credited to every opponent at the table **in full**; the
   argument, and what it costs at nine-handed, is in `_results_by_member`.
5. **Config shape.** `config.json` carries exactly §8.1's sections; run sizes are top-level, as
   in `config_g1.json` and `config_g3.json`. The agent's trunk dims come from `embedding_net`
   (OI-4 shares the trunk as code, and `d_emb` must agree anyway) and §6.2's `T` lives once, in
   `oracle`, rather than being configured twice.

**What is untested, and it is the expensive half** (`CLAUDE.md` §3). Everything above ran on CPU
at toy scale: two iterations, three sessions of four hands, two rollout samples per action, a
degenerate pool. Nothing in the battery has seen a v7 checkpoint with real weights, a CUDA
device, or an iteration at the size `config.json` asks for. **At the shipped settings — 400
sessions × 100 hands ≈ 100 000 labels at `samples_per_action = 128` — one iteration costs
122–141 Spark-hours of labelling alone** (`CONCEPT.md` §13), so ~5 days before a gradient step,
and 30 iterations is not a thing anyone should start. The first run should shrink `n_sessions`
until an iteration fits in a day, and report wall clock per phase against G3's prediction.

### Non-goals
No distributed training, no multi-GPU, no hyperparameter search, no automatic promotion of a
candidate (`CONCEPT.md` OI-7: selection is a manual owner step, outside the loop).

---

## S10 — Slumbot adapter rewrite (§12)

**Depends on:** S5. **Unblocks:** S11.

### Reads first
`CONCEPT.md` §12, §10. `evaluation/slumbot_eval.py` in full — it does not import today
(`ARCHITECTURE.md` §5) and this session is where that is fixed.

### Deliverables
- `evaluation/protocol.py` — **kept**, as verbatim as possible: `SlumbotClient`, the action-string
  grammar, `_token_to_action_idx` / `_action_idx_to_incr`, state replay, the BB/100 and SE
  accounting.
- `evaluation/v8_adapter.py` — **new**: turns a replayed Slumbot state into a v8 `HandRecord`-shaped
  object and asks `AgentPoolMember` for an action.
- `evaluation/slumbot_eval.py` — **deleted**, its remaining v7 agent path removed with it.

### Design
The protocol layer is v7's and is architecture-independent — it maps a wire action string onto the
engine's state and back. What is v7-specific is `_build_events` and `_choose_action`, and those are
what `v8_adapter.py` replaces: build the §5.1 token sequence from hero's view with the pending
decision (S1), attach the fitted opponent embedding, one `AgentNet` forward.

**This is plumbing, not specialisation** (`CLAUDE.md` §1, `CONCEPT.md` §10). Adapting to Slumbot's
wire protocol and bet-size grammar at evaluation time is allowed and belongs here. What must not
appear anywhere in this session: a Slumbot-specific policy branch, an opponent model keyed on
"this is Slumbot", or any assumption that the table is heads-up or the stack is 200 BB. The adapter
takes table size and stack depth from the protocol layer like any other parameter.

### Tests — `tests/test_slumbot_adapter.py`
No network access — every case runs off canned strings.
1. A canned action string replays into v8 tokens whose legality mask matches `env.legal`.
2. Observation parity on the replayed tokens, through the existing harness.
3. The action-index ↔ wire-increment mapping round-trips for every legal action on every street.
4. BB/100 and its standard error, computed from a canned list of hand results, match a
   hand-computed value.
5. The adapter refuses a table configuration outside `game.players_range` / `stack_bb_range` rather
   than silently clamping.

### Acceptance
Tests 1–4, and `python3 -c "import evaluation.protocol, evaluation.v8_adapter"` succeeds — which is
the first time anything under `evaluation/` imports in v8.

### Outcome — built 2026-08-19

`evaluation/protocol.py`, `evaluation/v8_adapter.py` and `tests/test_slumbot_adapter.py` are on
disk; `evaluation/slumbot_eval.py` is deleted; the battery is 350 tests. `ARCHITECTURE.md` §2.12
describes what was built. Three things are worth carrying forward into S11:

1. **The record is built, not replayed through the engine — deliberately.** The engine could have
   replayed the hand from a pinned deck and a forced prefix (S1), one construction path fewer,
   but a forced action is an *index*: the engine would then size Slumbot's bets out of *our* raise
   bins, and the agent would read the nearest bin's pot instead of the table's. The abstraction is
   unavoidable in what hero can say and must not reach what hero sees, so the record carries
   Slumbot's own chip amounts and only the action indices on the tokens are abstracted.
2. **`act` returns two indices** — what the agent chose and what it can be said to have played
   once `action_idx_to_incr` has clamped. S11 must append the **effective** one to
   `hero_action_indices`, or every later observation in the hand carries an action hero did not
   take. The clamp counters are the abstraction gap reporting itself and belong in S11's report.
3. **Cold and warm are already separable**: `SlumbotAgent` starts from the zero table (§5.5's cold
   start, §12's *cold* run) and `set_embeddings` installs a fitted one. S11 owns the refresh
   cadence `R` and the warm-up hand count; nothing about the fit is Slumbot-specific, which is
   what `CONCEPT.md` §10 records as adaptation rather than specialisation.

**What is untested:** no socket is opened anywhere in the battery, so the client's retry policy,
the live grammar and the wire itself are exercised only against canned strings. The first
screening run is S11's, and it is also the first evidence that any of this talks to Slumbot at
all.

### Non-goals
No hand-count runs, no result reporting (S11), no bet-size abstraction changes.

---

## S11 — `eval_pipeline.py`: cold, warm, and the disclosure (§12)

**Depends on:** S10, S9. **Runs on the Spark.**

### Reads first
`CONCEPT.md` §12, OI-7, §10. `CLAUDE.md` §1 — the ≥1 000 000-hand rule and the standard-error rule.

### Deliverables
`eval_pipeline.py`, driven by the `evaluation` config section.
```bash
./evaluate.sh --version=v8   # → cd versions/v8 && python3 eval_pipeline.py
```

### Design
**Two numbers, always both** (§12):
- **cold** — embedding pinned to zero for the whole run; this measures the unconditional policy;
- **warm** — embedding fitted online by the generic §5.5 mechanism, plus **how many hands it took
  to warm up**.

If warm is worse than cold, the exploitation mechanism is a net negative. That is a result to
report, not a bug to tune away — say so in the output, in those words.

**Reporting discipline, enforced in code.** The report always carries: hand count, BB/100, its
standard error, and the §12 selection disclosure — **how many candidates were screened and over how
many hands each**. A run below `min_reportable_hands` (1 000 000) is stamped `SCREENING ONLY` in the
report and in the printed header. At 50 000 hands a session's SE is roughly ±2.7 BB/100, which is
the scale of the selection bias involved, so the two must never be confused.

One `tqdm` bar over hands. Resumable — a million hands against a remote API is a long run and it
will be interrupted.

### Tests — `tests/test_eval_pipeline.py`
Offline, against a stubbed client.
1. BB/100 and SE arithmetic against hand-computed values, including the single-hand and zero-hand
   edge cases.
2. A short run is stamped `SCREENING ONLY`; a run at the threshold is not.
3. Cold pins the embedding to zero for every decision — asserted by instrumentation.
4. Warm refreshes on the configured `R` and reports the warm-up hand count.
5. Resume produces the same accumulated statistics as an uninterrupted run.

### Acceptance
Tests 1, 2 and 3, then a screening run on the Spark end to end.

### Outcome — built 2026-08-19, not yet run against Slumbot

`eval_pipeline.py`, `tests/test_eval_pipeline.py` and the `evaluation` config section are on
disk; the battery is 366 tests. `ARCHITECTURE.md` §2.13 describes what was built. Four decisions
were taken while building it:

1. **"How many hands it took to warm up" is derived, and the derivation is written down.** No
   single number of that shape exists in the data, so every warm hand records how many hands its
   vector was fitted from, the run is bucketed by that count (`evaluation.warmup_buckets`), and
   `warmup_hands` is the first bucket whose BB/100 reaches cold's overall BB/100. `None` — never
   caught up — is the result, not a missing value.
2. **The §5.5 fit is windowed** (`evaluation.fit_window`). "The histories observed so far" is not
   computable over a million hands; the window is config and is set well past the range G1
   measured, and the warm curve simply stops growing once it saturates.
3. **A failed hand is written to the log, marked failed, and excluded from the statistics.** It
   keeps its slot so the per-hand seed does not shift under a resume, and its count is in the
   report — a run that is quietly failing must not read as a clean one. This is the client's own
   "one lost hand, no desync, no cascade" policy, applied one level up.
4. **The selection disclosure is required**, not conventional: `build_report` refuses without
   `candidates_screened` and `screening_hands_each`. `config.json` ships `1` and `0`, which is the
   honest description of a first run and must be updated by whoever screens candidates.

**What is untested:** no socket is opened anywhere in the battery. The stub speaks the real
grammar through `evaluation/protocol.py`, so the action strings are ones the adapter must parse —
but the retry policy, the live grammar and Slumbot's actual response fields (`bot_hole_cards` in
particular, which the warm fit's showdown anchor depends on) are unverified until the first
screening run.

### Non-goals
No candidate selection (a manual owner step, OI-7), no automatic feedback of any result into a
config, a loss, a target or a pool weight (`CLAUDE.md` §1).

---

## 3. What this plan deliberately does not build

Recorded so that a later "why isn't there…" has an answer.

| Not built | Where it is recorded | Why not |
|---|---|---|
| Variant B — PPO / actor-critic | §7.4 | the scaling path of last resort; a single hand's chip delta is dominated by card variance |
| Variant C — bootstrapped continuation | §7.4 | only if S4 says variant A does not fit; then it replaces S3's root |
| Search at deployment | §6.1 | baseline is one forward per decision; a v7-phase-6-style search is a later experiment |
| Value head | §6.1, D5 | not needed by variant A |
| Warm-starting the agent's trunk from the embedding network | §6.1, OI-4, §7.1 | config flag, off by default; an open question, not a decision |
| Distilling a v7 member into the agent | §7.1 | recorded alternative to the iteration-0 cold start; a training phase, not an initialisation |
| CFR solvers as pool members | §4.1 | seconds per situation on CPU — unusable inside rollouts |
| Equity-gated style conditions | §4.2 | an equity evaluation in the innermost rollout loop |
| LBR exploitability measurement | §11.2 | explicitly not planned |
| League / exploiter agents | §11.1 | not in the baseline |

---

## 4. Risk register

Carried from `CONCEPT.md` §11 plus what the G1 run measured. Each row names the cheapest thing that
would tell us it has happened.

| # | Risk | `CONCEPT.md` | Earliest signal |
|---|---|---|---|
| **R1** | ~~Variant A does not fit the compute budget~~ — **closed by S4, 2026-08-19.** It fits: 122–141 h per 100 000 labels at `samples_per_action = 128`. The risk that replaced it is R8 | §13 | measured, twice |
| **R2** | Fictitious play cycles; the last iterate is what we measure and it need not converge | §11.1 | agent *n* losing to agent *n−2* in the pool's own results, visible in S8's PFSP scores |
| **R3** | The pool spans one dimension, so the embedding classifies members instead of carrying style | §11.3 | the style probe on G1's `fitted_vectors.npz` — zero compute, already saved |
| **R4** | The showdown heads memorise instead of generalising (**measured**: held-out `class_ce` 5.75 vs `ln 169` = 5.13; strength MSE 0.095 vs a target variance of ~0.085), and the showdown term contributes nothing to the inference fit (−0.007 ± 0.005 nats) | §5.1a, D7 | already observed; the weights are config, so the ablation is a config change once there is a pipeline to run it in |
| **R5** | The fit is worse than `e = 0` at short observation — measured at −0.195 ± 0.027 relative gain for held-out members with ≤2 of their own decisions observed | §5.5, §11.2 | it is why embedding dropout (§6.2) is mandatory and why §12 reports cold and warm separately; if warm < cold at 1 M hands, this is the reason |
| **R6** | The joint opponent range is approximated by independent marginals; bias direction unknown. **Measured proxy (S4, second run): 0–5% collisions heads-up, 33–46% six-handed, 57–82% nine-handed** — so the correction is idle at two players, where it is provably exact, and carries most of the weight at nine. The benchmark slice is the unbiased one; the generality bet lives where it is not | §7.3, §11.4 | S3's `collision_rate`, logged per label — a high rate means the approximation is working hard |
| **R7** | Conditional architecture satisfies "beats the pool" with a lookup table and no generality | §11.2 | cold BB/100 in §12 — the unconditional policy is the part that cannot be a lookup |
| **R8** | Label noise is large in the units the loss sees — `SE/pot` ran 0.2–3.4 at 256 samples and the sample budget cannot fix it (`SE ~ n^-0.5`, and `n = 128` is where buying more stops paying) | §13, §6.2 | already measured. The mitigation is the `soft_q` loss, which turns the noise into gradient variance instead of target bias; the signal that it was not enough is the policy's entropy rising with stack depth and table size, which is the shape of "the labels said nothing here" |
| **R9** | The bootstrap pool is weak — v7 plays ≈ **−90 BB/100** against Slumbot — so every posterior the oracle conditions on is a bad strategy's range, and iteration 0's hero is a bad rollout policy | §4.1, §11.4 | the first §12 evaluation of a v8 agent; and, before that, any measurement whose answer depends on the *shape* of a v7 range should be treated as not transferring (which is why OI-9 is deferred) |

---

## 5. Conventions every session follows

- **Absolute imports from the version root** (`from env.driver import LockstepDriver`). Two versions
  never share a process (`CLAUDE.md` §2).
- **Device resolved once** through `utils.resolve_device` and passed down. No `.cuda()`, no assumed
  CUDA. Every path runs on CPU at toy scale or it is untestable before deployment (`CLAUDE.md` §3).
- **All data under `data/v8/`**, never inside `versions/`. v8 never writes into `data/v7/`.
- **Every loop over a minute carries one global `tqdm` bar** through `utils.progress`, with the unit
  named for what is counted and `smoothing=0`. Never nested bars; a skipped item still advances the
  bar by what it would have contributed (`CLAUDE.md` §5).
- **Tests are deterministic** — fixed seeds, no probabilistic assertions, no order dependence — and
  prefer end-to-end scenarios over unit tests of helpers (`CLAUDE.md` §4).
- **New low-level logic is proposed, not written.** If a session discovers it needs a primitive not
  in §1's table, stop and ask (`CLAUDE.md` §5).
- **Implement literally.** No extra config knobs, no fallbacks, no "while I was there" improvements —
  especially in the oracle, the targets and the fit, where an unrequested addition silently changes
  results.
