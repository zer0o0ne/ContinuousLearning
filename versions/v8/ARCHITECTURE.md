# v8 — Architecture

**Status: gate G1 implemented (CONCEPT.md §14). The agent, the BR oracle and the
training pipeline do not exist yet.**

This document describes **what the code in `versions/v8/` actually is**. The design it is
being built toward is in **[`CONCEPT.md`](CONCEPT.md)**: five entities (environment, opponent
pool, agent, opponent-embedding network, best-response oracle) and a loop that trains a best
response to a fixed pool, conditioned on a per-opponent embedding, then adds the result to the
pool. Read `CONCEPT.md` for *why*; read this file for *what is on disk*.

Project-wide rules (hardware, testing, version discipline, engineering principles) live in the
root `CLAUDE.md`. The previous architecture is documented in `versions/v7/ARCHITECTURE.md` —
read-only reference, do not edit.

**Goal:** `CLAUDE.md` §1 — a single agent covering 2–9 players and 10–300 BB, measured by BB/100
against Slumbot (heads-up, 200 BB) with no training-time specialization to that slice.

---

## 1. Current tree

```
versions/v8/
  CONCEPT.md            design record — the thing being built
  ARCHITECTURE.md       this file
  config_g1.json        the G1 experiment surface
  config_g1_pilot.json  a shrunk G1 for measuring throughput on the Spark first

  env/                  poker engine, from v7, verbatim
    legal.py            the one legality rule (§6.2)                    NEW
    driver.py           lock-step vectorised driver over Table (§3)     NEW
    showdown.py         reveal detection + the two showdown labels (§5.1a)  NEW
  pool/                 entity 2 — the opponent pool (§4)               NEW
    base.py             PoolMember: logits → styled, legal distribution
    style.py            live style modifiers, 32 scalars per member (§4.2)
    degenerate.py       always-fold / call / min-raise / maniac / nit
    v7_member.py        a vendored v7 checkpoint as a pool member (§4.3)
    build.py            pool construction from config, fresh style draws
  nets/                 v8's own networks                               NEW
    features.py         §5.1 token features — where observation parity lives
    tokeniser.py        the shared tokeniser MLP (§5.1, OI-4)
    embedding_net.py    entity 4 — the opponent-embedding network (§5)
  gates/
    g1.py               the G1 experiment (§14)                         NEW
  vendor/v7/            frozen snapshot of v7's agent code (§4.3)       NEW
    attn_utils.py, perception/*, action/*, modifiers.py   copied verbatim
    agent.py            perception + action-head subset of v7's ASI
    events.py           the v7 event format

  attn_utils.py         causal+padding mask helper, inherited from v7
  utils.py              Logger, get_amp_config, resolve_device
  gto_utils/            hand evaluation, equity, CFR solvers v1–v5, from v7
  evaluation/
    slumbot_eval.py     from v7, verbatim — does not import yet (§5)
  tests/                8 files, 99 tests, ~26 s
```

### What was inherited, and why

Only architecture-independent code was carried over from v7. Everything tied to the v7 agent
(`agent/`, `train_scenarios/`, `mcts/`, `pipeline.py`, `config.json`, `evaluation/evaluate.py`)
was deliberately **not** copied — it is being replaced. The one exception is `vendor/v7/`, which
is not inheritance but vendoring: see §3.

| Path | Origin | State |
|---|---|---|
| `env/table.py`, `judger.py`, `dealers.py` | `v7/env/` verbatim | `Table`, `Judger`, dealers. Game rules do not change with the agent architecture. Untouched by v8 — the driver wraps them from outside. |
| `gto_utils/` | `v7/agent/gto_utils/` | Hand evaluation, equity, range utilities, CFR solvers v1–v5. Not on the G1 path at all; needed later by the oracle. Requires `eval7`. |
| `utils.py` | `v7/utils.py` + `resolve_device` | `Logger`, `get_amp_config`, and the CUDA → MPS → CPU device resolution `CLAUDE.md` §3 requires. |
| `attn_utils.py` | `v7/agent/attn_utils.py` verbatim | Causal + padding attention mask. Architecture-independent. |
| `evaluation/slumbot_eval.py` | `v7/evaluation/slumbot_eval.py` verbatim | **Does not import yet** — see §5. |

---

## 2. What exists: the G1 path

G1 (`CONCEPT.md` §14) asks one question — *does the embedding carry style, and does it
generalise to a style never trained on?* — and it is the first thing built because it tests the
riskiest bet (B1) before anything depends on it. Everything below is what that needs.

### 2.1 `env/legal.py` — the one legality rule

`CONCEPT.md` §6.2: "one implementation, not two". The driver that samples pool-member actions,
the network that masks its prediction logits, and later the oracle and the agent's target
construction all call `legal_action_mask`. A second copy is how the played distribution and the
trained distribution drift apart unnoticed.

The rule is v7's, ported from `versions/v7/agent/mcts/game_state.py::get_legal_actions` onto
`Table` (v8 has no `GameState`). It is *playable*, not merely rule-legal: fold is dropped when
checking is free, raise bins that collapse to a call or to the all-in are dropped, sub-min-raises
are dropped, all raises are dropped when every other live player is already all-in (C.5) or when
a short all-in did not reopen the betting (C.7.5).

### 2.2 `env/driver.py` — the lock-step vectorised driver

`CONCEPT.md` §3 requires this and calls it the largest single piece of engineering in v8. It
advances N independent hands together, collects the pending decision of every live hand, groups
them by which pool member has to answer, makes **one** call per member, and scatters the sampled
actions back. `Table.step` and its chip accounting are untouched, so the existing engine tests
keep covering them.

Two rules make batching invisible, which is the §15 equivalence requirement:

* the deck comes from `np.random.seed(hand_seed)` immediately before that hand's
  `start_table()`, so it depends on the hand and nothing else;
* actions are sampled by inverse-CDF from a uniform drawn from that hand's own
  `np.random.Generator`, so the draw does not depend on batch composition.

`run(specs, batch_size=1)` is therefore the sequential path and `run(specs, batch_size=N)` the
lock-step one, and `test_driver_lockstep.py` asserts they agree bit for bit.

The snapshot record follows v7's convention exactly (initial snapshot, then a pre-decision /
post-action pair per decision, with `acting_pos` in a post-action snapshot naming whoever acts
*next* — audit B.2), because a vendored v7 checkpoint has to see the event stream it was trained
on. An all-in runout is played out with the engine's own no-op steps so final credits are
correct; those steps carry no decision and are not recorded.

### 2.3 `pool/` — entity 2

A member exposes `logits(contexts)`; turning logits into a played distribution happens once, in
`PoolMember.policy`, so every member goes through the same last mile.

**Style modifiers (`style.py`, §4.2, OI-5).**

```
p = (1 − λ) · softmax( (logits + b(s)) / T )  +  λ · uniform_over_legal
```

32 scalars per member: 5 unconditional category biases, 5 gated on the position bucket, 5×4 per
street, plus `T` and `λ`. The categorisation is v7's `resolve_actions`, reused verbatim from the
vendored `modifiers.py`. A style draw is free — no extra forward, no extra parameters — so the
opponent space is continuous and effectively infinite, which is the answer to §11.3.

**Degenerate strategies (`degenerate.py`, §4.1).** always-fold, always-call, always-min-raise,
maniac, nit. They emit finite-scale logits rather than one-hots so a style draw can still move
them. The nit's strength test is a hole-card lookup, not an equity evaluation — §4.2 drops
equity-gated conditions as far too expensive inside rollouts.

**`build.py`.** Builds the pool from the `bootstrap` config section and produces the descriptor
list the G1 report needs. `fresh_style_variants` draws the never-trained-on styles that
measurement §14.2 is made of.

### 2.4 `nets/` — the tokeniser and entity 4

`features.py` builds the §5.1 token features from played hands and **is** the observation-parity
boundary (`CONCEPT.md` §9): one token per decision, board as of that decision's street, hole
cards only for the observer, no post-decision information, everything monetary in BB.

`tokeniser.py` is the shared tokeniser MLP — one class, used by the embedding network and later
by the agent (OI-4). Weights are not shared; only the code is.

`embedding_net.py` is entity 4: a per-hand causal encoder, a trainable per-member embedding
table, an amortised head, two showdown heads, and the inference-time joint fit. Training is the
§5.4 baseline (no inner loop). The amortised head reads a second forward taken with `e = 0`,
because that is the only thing available at inference before any vector exists — that second pass
is the cost of the no-inner-loop baseline, and it is the reason `loss_terms` runs the trunk twice.

### 2.4a The showdown anchor (§5.1a)

A hand that reaches showdown appends one terminal token per revealed seat, after every decision
token. `env/showdown.py` computes its two targets — the exact strength percentile on the final
board (enumerating all 990 non-conflicting combos through `gto_utils.gpu_solver.evaluate_hands`,
no Monte Carlo and no seed) and the 169-way preflop class — once over the whole corpus, never
inside a loop. `OpponentEmbeddingNet.showdown_strength_out` and `.showdown_class_out` read those
tokens.

Two properties carry the design and both are tested behaviourally:

* **nothing flows backwards.** Attention is causal within the hand, so the terminal token sees
  every decision and no decision sees it. Writing the revealed cards into the decision tokens
  instead would tell each of them that the player reached showdown — i.e. was not going to
  fold — which is a future leak that lowers the loss and carries no style.
* **the terms are in both objectives.** `OpponentEmbeddingNet.objective` is used by training and
  by `fit_embeddings` alike. v7 could use a training-only probe because its embedding was a GRU
  state from a forward pass; v8's is a latent found by gradient descent at deployment, so a
  training-only term would be optimised straight back out of the fitted vector.

Weights live in the `train` config section (`showdown_strength_weight`, `showdown_class_weight`)
and are applied identically in both places; setting them to zero is the ablation.

### 2.5 `gates/g1.py` — the experiment

```bash
cd versions/v8 && python3 -m gates.g1 --config config_g1.json
```

The unit of the experiment is a **session**: a fixed set of pool members sitting down together
for a fixed number of hands at a fixed stack depth, with the button rotating each hand. Sessions
are used because that is what deployment looks like — hero sits at a table, observes the players
there, and fits their vectors from hands hero was in (§5.4). Rotation is not cosmetic: with fixed
seating a member would be identifiable by its seat and the network would learn seats, not styles.

In an evaluation session the first `max(observed_hand_counts)` hands are the observation window
and the rest are held out. A curve over "number of observed hands" is therefore a curve over
prefixes of one session. Table size and stack depth are sampled uniformly and independently over
2–9 and 10–300 BB (`CLAUDE.md` §1).

Three conditions are measured at every point: `e = 0` (the §14.1 baseline, and the policy the
agent will play against an unobserved opponent), the ablation (`K = 0`, the amortised head's
output with no gradient fit, §5.4), and the fit (`K` gradient steps from that initialisation,
§5.5). Only non-observer players are scored — the observer is hero.

Output is `data/v8/g1/<timestamp>/g1_report.json` plus the trained network, and a printed table
covering all four §14 measurements. The run log goes to `data/v8/logs/<timestamp>.txt`.

`config_g1_pilot.json` is the same experiment shrunk to a throughput probe: same pool shape and
same model, 3 200 corpus hands instead of 96 000 and 500 training steps instead of 20 000. Its
numbers are not results — it exists because every cost figure for the real run is a hypothesis
until it has been run on the Spark (`CLAUDE.md` §3), and this is the cheapest way to replace them
with measurements.

---

## 3. `vendor/v7/` — the frozen v7 snapshot

`CONCEPT.md` §4.3 / OI-6. Two version trees cannot be imported into one process (`CLAUDE.md`
§2 — they define the same top-level package names), so a v7 checkpoint can only be run from a
copy living inside v8. `attn_utils.py`, `perception/*`, `action/*` and `modifiers.py` are
verbatim copies with `agent.…` imports repointed to `vendor.v7.…`; **this package is never
edited to follow v8 changes.**

Two files are not copies:

* `agent.py` — `V7Agent`, the perception + action-head subset of v7's `ASI`. That is all a pool
  member needs (`heads={"action"}`, `skip_opponent_emb=True`, no MCTS). The other heads are not
  constructed, so their weights land in `unexpected` under the `strict=False` load v7 itself
  used. A missing checkpoint raises rather than yielding a randomly initialised "v7 member".

  **This is a pool member and nothing else.** An earlier draft of `CONCEPT.md` (OI-2) also had
  the v8 *agent* initialised from a v7 checkpoint. That was withdrawn by the owner once this code
  existed and made the mismatch concrete: a v7 checkpoint's weights are shaped for v7's input —
  seven card vectors per event through `combine: Linear(8·d → d)`, an encoder over N×7 positions,
  a mean-pool to N, then a decoder — while the v8 agent reads the §5.1 token through the OI-4
  tokeniser. There is no correspondence to load. What OI-2 wanted (a competent rollout policy and
  state distribution at iteration 0) is now bought by seating a v7 pool member in hero's seat for
  that iteration, which needs no weight transfer at all (`CONCEPT.md` §7.1).
* `events.py` — the v7 event format, from `generate.py::_rebuild_events`, cross-checked against
  `evaluation/slumbot_eval.py::_build_events`, which is the copy that came into v8. Events are
  built from the acting player's seat, so a v7 member sees its own hole cards and nothing else's.

---

## 4. Interpretive decisions

`CONCEPT.md` is a design record, not a specification, and seven points needed a reading before
they could be code. They are listed here so the owner can overrule any of them cheaply.

1. **~~Showdown reveals are vacuous~~ — superseded.** The original reading was that §5.1's
   "strictly after the reveal" exception had nothing to apply to, since with one token per
   decision every token precedes the reveal, so hole cards were masked unconditionally. The owner
   chose instead to add the terminal token that makes the exception real: §5.1a, implemented as
   described in §2.4a above. Hole cards are still masked in every *decision* token.
2. **The position block gates on a binary bucket.** §4.2 gives the position block size 5 while
   gating it on an "acting position bucket", which only adds up to the stated ~32 scalars if the
   bucket is binary. Implemented as `acting_pos >= ceil(num_players / 2)` — "second half of the
   table", defined for every size from 2 to 9.
3. **Block-diagonal attention is realised as batching.** §5.2 itself says hands are a batch
   dimension, not a sequence. Cutting cross-hand attention makes hands conditionally independent
   given the embedding, so a batch *is* the block-diagonal mask, and it is cheaper. The
   behavioural tests §15 asks for are written anyway.
4. **RoPE runs within a hand.** §5.2's "no global positional encoding" rules out positions that
   cross hand boundaries. Under (3) there are none: position ids are within-hand, and the token
   still carries the decision index as §5.1 requires.
5. **The §5.4 ablation is `K = 0`.** §5.4 asks for "mean-pooled hidden states of the transformer
   over the player's tokens, no gradient fit at all". A raw pooled `d_model` vector cannot be
   fed where a `d_emb` vector goes, and the amortised head *is* the mean-pool-plus-MLP that maps
   a history to a vector. So the ablation is that head's output used as the final embedding, with
   no fit. The comparison §5.4 wants — does the inference-time optimisation earn its keep — is
   exactly what this measures.
6. **One observer per hand.** §5.3 puts every player's decisions in one forward; §5.4 restricts a
   player's history to hands the observer sat in. Each hand is therefore tokenised from one
   observer's view, and in a session that observer is slot 0 (hero) throughout.
7. **G1 reports a `gain` metric alongside §14.2's raw gap.** The raw seen-style / unseen-style
   loss difference is confounded: the two session sets have different tables and different
   intrinsic predictability, so their losses are not on the same scale. `gain = CE(e=0) − CE(fit)`
   compares each set against its own baseline, and the difference of gains is the number that
   actually answers B1(b). Both are reported; nothing is dropped.

---

## 5. Known non-working state

`evaluation/slumbot_eval.py` **will not import as-is**: it depends on `agent.agent.ASI`,
`agent.mcts.*`, `evaluation.evaluate` and `agent.train_scenarios.generation`, none of which exist
in v8. Treat it as reference material. When v8's agent lands, split it as `CONCEPT.md` §12
describes: keep the protocol layer (HTTP client, action-string grammar, token ↔ action mapping,
state replay, BB/100 + SE accounting), rewrite the agent adapter against v8's observation format.

**The v7 pool member has never been run with real weights.** `data/v7/` does not exist on the dev
box and there is no GPU (`CLAUDE.md` §3), so `test_v7_pool_member.py` exercises a randomly
initialised network: it verifies that the vendored stack imports and runs under the installed
`transformers`, that the event format is built correctly, and that a v7 member obeys the
observation-parity and legality contracts. It says nothing about a trained checkpoint. Loading
real v7 weights, and everything about GPU throughput, stays a hypothesis until run on the Spark.

`config_g1.json` ships its `v7` bootstrap entry with placeholder paths
(`../../data/v7/FILL_ME/…`). They must be filled in before a real G1 run; without them the pool
is degenerate strategies and their style draws only, which weakens the base policies the styles
modulate but does not change what G1 measures.

---

## 6. What is still missing

In roughly the order `CONCEPT.md` §14 says to build it:

| Piece | `CONCEPT.md` | Notes |
|---|---|---|
| **G3** — what an oracle label costs | §14, §13 | wall-clock and forwards per label on the Spark, on the driver that now exists. Fixes the label budget before a pipeline is built around it |
| BR oracle (variant A) | §7 | posterior over opponent combos + per-action rollouts |
| agent v8 | §6 | one forward at deployment, no value head, no search |
| pool sampling: PFSP, embedding dedup, uniform floor | §4.4 | not needed by G1 — the G1 pool is fixed and small |
| `pipeline.py`, `config.json` | §8.1 | the outer loop |
| Slumbot adapter rewrite | §12 | protocol layer survives |

---

## 7. Tests

```bash
cd versions/v8 && python3 -m pytest tests/ -q
```

99 tests, ~26 s on the dev box (CPU-only). The 30-minute budget from `CLAUDE.md` §4 is barely
touched.

| File | Covers |
|---|---|
| `test_engine_conservation.py` | Chip conservation through the engine (from v7) |
| `test_audit_stage0.py` | Engine invariants: `cumulative_bets` monotonicity, betting/street advance (from v7) |
| `test_solver_value_bet.py` | Solver value-bet pot construction (from v7) |
| `test_driver_lockstep.py` | **Lock-step ≡ sequential**, chip conservation through the driver, the v7 snapshot convention, the max-actions cap, every table size and stack depth, and the legality rule's corner cases |
| `test_observation_parity.py` | **The fatal invariant**: only the observer's hole cards, board never ahead of the street, no token carries its own action, scalars from the pre-decision snapshot, prefixes independent of what came later |
| `test_embedding_net_masking.py` | Causal within a hand, block-diagonal across hands, hand order irrelevant, the embedding is what changes the prediction, padding inert, **a showdown token cannot reach back into any decision**, the action loss ignores showdown tokens, both showdown heads reach the embedding, zero weights reduce the objective to action CE |
| `test_inference_fit.py` | The joint fit reaches the loss of the vectors that generated the labels, determinism, `K = 0` is the ablation, network weights untouched, cold start, regularisation, **the showdown terms reach the fitted vector** and zero weights reproduce the action-only fit |
| `test_observation_parity.py` (§5.1a part) | Showdown tokens exist exactly for the revealed seats and never among the decisions, the revealed cards are the target and never an input, the labels match the cards shown, the two masks partition the real tokens, a showdown hand with no labels is refused |
| `test_pool_style.py` | The five categories partition the action set, 32-scalar round trip, identity style is a masked softmax, position and street gating, temperature, uniform mix, every draw is a valid distribution over legal actions, each degenerate strategy does what it says |
| `test_v7_pool_member.py` | The vendored v7 stack constructs and plays legal hands; the v7 event format is built from the acting seat, masked to the street, and stops at its decision |
| `test_g1_gate.py` | The gate end to end: button rotation, uniform 2–9 × 10–300 BB, the four report sections, cold start ≡ `e = 0`, and that the standard error's unit is the session |

`conftest.py` puts `gto_utils/` and the version root on `sys.path` and reseeds
`random`/`numpy`/`torch` to 42 before every test. `tests/g1_fixtures.py` holds the shared toy
pool, hand specs and network config.

---

## 8. Data layout

All data lives outside `versions/` under `data/v8/…` (gitignored). v8 never writes into
`data/v7/`; it only **reads** v7 checkpoints, by path, from config (`CONCEPT.md` §4.3, §8.1).

## 9. Running

```bash
cd versions/v8 && python3 -m gates.g1 --config config_g1.json   # gate G1 (§14)

./run.sh      --version=v8   # → python3 pipeline.py       (does not exist yet)
./evaluate.sh --version=v8   # → python3 eval_pipeline.py  (does not exist yet)
```
