# v8 — Implementation plan: amortised opponent vectors for the pool's own agents

**Status: plan only. Nothing here is implemented. No code was changed while writing it.**
**Scope: let a past agent seated as an *opponent* condition on its tablemates through the
amortised head (`K = 0`) instead of playing at `e = 0`, in both phases that seat it — label
generation and the embedding corpus — with the current behaviour kept as the default and
bit-identical.**

This document revisits one decision: `PLAN_PIPELINE.md` D12, settled 2026-08-19 as option (a),
"a past agent in the pool plays at `e = 0`". It was asked for on 2026-09-02 as "how hard is it
to give the pool's agents amortised embeddings"; the answer is "moderate, and D12's objection to
the alternative does not apply to this form", and this file is the build order.

- **Why** a past agent is in the pool at all, and what `e = 0` costs → `CONCEPT.md` §4.1, §5.4
- **What is on disk today** → `ARCHITECTURE.md` §2.8 (`FrozenAgentMember`), §2.10 (blocks and
  the fit), §2.11 (the loop), "Labelling in parallel" under §2.3 (the mirrors)
- **The one measurement that already bears on this** → `g1_report.json`, the `ce_ablation`
  curves (§0.3 below)
- **In what order to build it, and exactly what each step is** → this file

Project-wide rules are in the root `CLAUDE.md`. Three of them decide the shape of this plan:

- **Composition over new code.** Every edit to working code and every new primitive is marked
  ⚠ and collected in §1 for one sign-off conversation. The plan reuses hero's own machinery —
  the per-seat member, the per-block refresh, the block-vector table, the worker mirrors — and
  adds no second path for any of it.
- **The CPU battery is the only local gate.** Everything below is exercised on CPU at toy scale.
  Every cost figure in §1.1 is a hypothesis until it has run on the Spark.
- **Benchmark results must not feed back into training.** Whether this change is *kept* is
  decided by the pipeline's own measurements (P2, phase D's warm/cold gap), never by the
  Slumbot number.

---

## 0. What was decided, what is being revisited, and the constraints

### 0.1 D12, and the three options

A past agent enters the pool at the end of every iteration, as `agent_variants` members. Seated
as an opponent it is asked two things: *what do you do here* (the driver, in play and inside the
oracle's rollouts) and *what would you have done holding this* (the §7.2 posterior). Both need
the vectors it conditions on, and D12 asked where those come from.

| | Option | Vectors of a past agent's tablemates | Cost per (table, refresh, seat) |
|---|---|---|---|
| (a) | **zero — current** | `e = 0`, the unconditional policy §6.2's dropout trains | nothing |
| (b) | full §5.5 fit | amortised head, then `K` gradient steps | tokenise window + `1 + 2K` trunk passes |
| (c) | **amortised — this plan** | amortised head, no gradient steps (`K = 0`) | tokenise window + **1** trunk pass |

(a) was recommended and taken because (b) "nests one §5.5 fit inside another for every rollout".
That objection is about **where** the vectors are computed, and it does not survive a look at
how hero's own vectors are produced: they are fitted **once per block of `R` hands, at the
driver level**, and the rollouts and the posterior only *read* them (`ARCHITECTURE.md` §2.10,
"order is the whole difficulty"). Nothing is fitted inside a rollout for hero, and nothing would
be for a past agent either. What (b) actually costs is `1 + 2K` trunk passes per past-agent seat
per refresh; what (c) costs is one. Neither nests anything.

(c) is therefore the cheap form of (b) that D12 said did not exist, and it is the form this plan
builds. (b) is not built: once (c) exists, (b) is one config value (`K_pool > 0`) away, and
nothing has measured a need for it. See §1, ⚠6.

### 0.2 What does not change

- **The agent.** Its training data, its inputs, its loss. A label still stores **hero's** table
  (block vectors, observer slot 0) and nothing else about the vectors. Phase C is untouched.
- **Hero.** Its fit (`K`, `fit_lr`, `fit_reg`, `R`), its per-seat members, its recorder.
- **The Slumbot evaluation.** No past agent sits at that table.
- **The pool's interface.** A member still answers `policy(contexts)` for any legal situation,
  batched. The style layer, PFSP, dedup, `_results_by_member` are all indexed by *pool member*,
  and a past agent stays one pool member. What changes is which *object* sits in a seat for one
  block of one session — exactly as it already does for hero.
- **The default.** `e = 0` remains the default and remains bit-identical: every test that pins a
  past agent's behaviour today keeps passing with the switch off.

**One thing that does change and was not in the first draft of this plan: whose network produces
a past agent's vectors** (owner decision, 2026-09-03; ⚠11). The obvious implementation reads the
*current* embedding network, the one the loop holds. That is wrong twice over:

- **The input is out of distribution.** Nothing anchors the coordinates of the vector space. The
  network is retrained every `retrain_every` iterations, continuing from its own weights, so the
  meaning of an axis drifts. Hero is safe because it is retrained alongside; a frozen past agent
  is not — it would interpret a description written in coordinates it has never seen.
- **The pool would stop being fixed.** A member of the pool is supposed to be a *fixed
  algorithm*: the same history always produces the same action. If its vectors come from a
  network that is still training, then the same member plays differently at iteration 10 and at
  iteration 20 with nobody having changed it. Hero's accumulated results per member — what PFSP
  samples on — would be results against a moving target, and the embedding network would be
  learning to describe styles that move because *it* moved. That is a feedback loop with no
  symptom.

So **every past agent carries a pointer to the embedding network of its own generation**, frozen
at the moment that generation entered the pool, and its vectors are always computed with that
network. The cost is memory only — the number of forwards is unchanged, one per conditioned seat
per refresh either way — and it is about 200 MB per retrain generation, ~3 GB over a 30-iteration
run at `retrain_every = 2`, which the owner has confirmed is not a constraint.

### 0.3 The evidence there is

G1 measures three conditions per observed-hand count: `e = 0`, the ablation `K = 0` (the amortised
head's output, no gradient fit — exactly option (c)'s vectors) and the fit. Held-out and unseen
sets, cross-entropy in nats of the embedding network's action prediction:

| observed hands | heldout `e=0` | heldout `K=0` | heldout fit | unseen `e=0` | unseen `K=0` | unseen fit |
|---|---|---|---|---|---|---|
| 1 | 1.950 | 1.852 | 1.928 | 1.802 | 1.728 | 1.722 |
| 5 | 1.950 | 1.771 | 1.640 | 1.802 | 1.594 | 1.490 |
| 25 | 1.950 | 1.699 | 1.536 | 1.802 | 1.520 | 1.379 |
| 100 | 1.950 | 1.697 | 1.501 | 1.802 | 1.520 | 1.356 |
| 200 | 1.950 | 1.689 | 1.506 | 1.802 | 1.525 | 1.345 |

Two readings, and their limits:

- `K = 0` recovers **55–60 % of the fit's gain** over `e = 0`, on members the network never saw
  (unseen), and it is *better* than the fit after one hand. So the amortised vectors carry real
  information about the tablemates and are not noise.
- The `K = 0` curve is **flat from ~25 hands on**. This is what makes a window cap (⚠4) free
  rather than a compromise, and it is consistent with how the head is trained: `loss_terms`
  pools the `e = 0` hidden states over whatever hands of a member land in one 64-hand batch — a
  handful — so the head has never been asked to summarise 2000 hands and there is no reason to
  hand it 2000.

What this does **not** say: that a *past agent* plays better with these vectors than without.
G1 scores the embedding network, not the agent; and the agent was trained (§6.2) on fully fitted
vectors or on zeros, never on `K = 0` vectors, so its input distribution shifts slightly (a
shrunk, regression-to-the-mean estimate of the fitted vector).

### 0.3a The owner's decision, 2026-09-03: build it without the measurement

P2 was built and **not run**, and P3 no longer waits on it. The owner's reasoning, recorded here
because it replaces a measurement:

> The pool is strongly exploitable from the start, and reading the opponent is precisely the
> mechanism that finds those exploits. So conditioning may become worthless at the end of
> training — if the agents converge on something unexploitable — but stays valuable against any
> exploitable opponent. Measuring it now, on agents that are weak in exactly that way, answers a
> different question than the one that matters.

Two consequences worth writing down, because both are checkable later:

- **A falsifiable prediction.** If the argument is right, the advantage of a conditioned pool
  member over a blind one *shrinks monotonically* across iterations as the pool converges. It
  will never be measured by default, but P2 exists and can be run on any two checkpoints; a
  non-decaying or growing advantage would mean the pool is not converging toward anything
  unexploitable, which is a result about the *training*, not about this mechanism.
- **A cost the argument does not cover.** A conditioned pool member adapts *within* a session,
  and the whole embedding scheme assumes a player has **one** style vector for the whole session.
  An adaptive tablemate violates that assumption: the network is asked to summarise a
  non-stationary player in one vector and will average it. This is the thing to watch when P4
  turns the corpus on — the retrain's action cross-entropy on past-agent tokens should rise a
  little, and a large rise is a result to report rather than a bug to tune away.

### 0.4 The four constraints that shape the design

1. **The observer's perspective is per seat, and it is a leak otherwise.** `hand_tokens` reveals
   the observer's own hole cards and masks everyone else's. The amortised head pools hidden
   states by *slot* over tokens built from **one** observer's view. A past agent at slot *j*
   therefore needs the session's hands tokenised from *its* seats, not from hero's — reusing
   hero's tokenisation would show the past agent hero's cards, which is §9's parity broken from
   the pool side. Consequence: **one tokenisation and one trunk pass per past-agent slot per
   refresh**, and `Session.tokens` has to take the observer slot as an argument (⚠2).

2. **Block alignment is hero's, exactly.** The vectors a past agent acts with in block *b* are
   computed over blocks `0 … b−1` and over nothing else; block 0 is the zero cold start; the
   posterior conditions on the member with the vectors it acted under, which holds because the
   per-(session, block) member is rebuilt in the label loop and in every worker the same way
   hero's is (`_seat_hero`, `_worker_main`). `test_the_embedding_of_a_block_ignores_every_later_hand`
   is the model for the test that pins it (P3).

3. **Persist or lose reproducibility.** A resumed labelling run plays only the unlabelled tail
   and refits nothing, because a GPU forward does not reduce identically twice and one flipped
   action changes the hand (`ARCHITECTURE.md` §2.11). That argument applies verbatim to a past
   agent's vectors: they decide its actions, so they go into `labels/vectors.npz` beside hero's
   and are read back on resume (⚠5).

4. **One member is one seat at one table — but a past agent already knows its slot.** Hero is
   one member per *seat* because its `slot_of_seat` depends on which seat it holds. A past agent
   at slot *j* acting from seat *s* fixes the hand index modulo the table size,
   `(j − s) mod n`, and with it the whole `slot_of_seat` — so **one member per (session, slot,
   block)** suffices, and it derives `slot_of_seat` from its own slot and the acting seat. No
   hand index, no metadata, and it works unchanged inside a rollout (the acting seat is real
   there) and under a posterior query (the member's own seat).

### 0.5 Rejected alternatives

- **Giving a past agent the *true* table rows of its tablemates** (the "oracle embedding" of
  G1). Rejected: it conditions a pool member on information no deployable agent has, and hero
  has no converged row at all — its hands never enter the corpus (`ARCHITECTURE.md` §2.11) — so
  the one tablemate that matters most would get a dead row. It would also make the pool's agents
  stronger than any version of themselves that could ever be deployed, which is not what the
  labels should be measured against.
- **Sharing hero's tokenisation across observers.** Rejected — constraint 1; it is a card leak.
- **Building (b) directly.** Deferred — §0.1. (c) first, measured (P2); (b) is `K_pool > 0`
  afterwards if the measurement asks for it.
- **A single shared member reading vectors from the record's metadata** (session and hand index
  in `spec.meta`, as `_ObservedHero` does). Rejected in favour of constraint 4: it needs the
  rollout's `HandSpec` to carry the parent's metadata, which is an assumption about the driver
  the per-(session, slot) member does not need, and it puts a dictionary lookup on the hot path
  of every pool forward.

---

## 1. Sign-off list — every edit to working code and every new primitive

| # | Item | Where | Why nothing existing does it |
|---|---|---|---|
| ⚠1 | `FrozenAgentMember` gains an optional `(embeddings, own_slot)` pair and a `with_vectors(name, embeddings, own_slot)` sibling constructor; `logits` gathers `emb` / `seat_emb` from the table when present and keeps the zero tensors when absent | `agent/policy.py` | it is the class D12 said (b) would be "a change to one class" of; `AgentPoolMember` cannot serve — it refuses `hole_override` and historical snapshots by design |
| ⚠2 | `Session.tokens(max_players, n_actions, observer_slot=0, window=None)` — tokens from *any* slot's view, over the last `window` hands; `observer_slot=0, window=None` is today's call | `env/session.py` | today's method hardcodes slot 0; the observer-perspective rule (§0.4, 1) needs the other slots |
| ⚠3 | `amortised_vectors(embed_net, session, slot, ...)` — tokenise the window from `slot`'s view, one `amortised_init`, pad to `(max_players, d_emb)` | `train/generate.py` (beside `_pad_vectors`) | `fit_embeddings(steps=0)` already *is* `K = 0`; the new piece is the per-observer tokenisation around it, three lines, and it must exist once because both phases call it |
| ⚠4 | config: `embedding_net.pool_agent_vectors ∈ {"zero", "amortised"}` (default `"zero"`) and `embedding_net.pool_agent_window` (hands; `null` = the whole session so far) | `config.json`, `config_*.json` | the switch is the experiment surface `CLAUDE.md` §5 asks for; the window is what keeps P4 affordable (§1.1) and §0.3 says it costs nothing in signal |
| ⚠5 | `block_vectors` gains an observer-slot axis: `vectors[session][block]` becomes `(max_players_slots, max_players, d_emb)` with hero at index 0 and a past agent's table at its slot; `vectors.npz`, `_resume_state`'s shape assertion and the label's `"embeddings"` (which reads index 0) follow | `train/generate.py` | one array, one file, one resume path — a second file for the pool's vectors is the divergence §5 warns about |
| ⚠6 | **decision, not code:** (b) is not built. If P2 says (c) pays and the warm/cold gap suggests more is available, `K_pool` is added *then* as a config value read by ⚠3 | — | — |
| ⚠7 | `_seat_hero` generalises to `_seat_conditioned(play_pool, session, block_vectors, block, hero_slots, agent_member, pool_slots)` — hero's per-seat members **and** the per-slot past-agent members for one `(session, block)`; called from the play loop, the sequential label loop and `_worker_main` | `train/generate.py`, `oracle/parallel.py` | the three call sites already share `_seat_hero`; they must keep sharing one function |
| ⚠8 | `mirror_spec` / `build_mirror`: the `"frozen"` spec carries `own_slot` and the mirror is rebuilt per block from the worker's `block_vectors`, as hero is; the reserved per-(session, slot) entries are mirrored as holes like hero's | `oracle/parallel.py` | the slab already ships `emb` and `seat_emb` from the worker, so the **server and the runners do not change** |
| ⚠9 | `embedding_phase` plays the corpus in blocks of `R` with the same refresh, through the same loop as the labels phase (`_play_sessions` with no hero) | `pipeline.py`, `train/generate.py` | today it is one `play` call; a second block loop would be a copy of the first |
| ⚠10 | `gates/pool_conditioning.py` — the discriminating experiment of P2 | `gates/` | nothing measures a *pool member's* play, and the G1/G3 gates measure the embedding and the label |
| ⚠11 | a past agent carries **its own generation's** embedding network: `FrozenAgentMember` gains an `embed_net` reference; `pipeline.py` freezes one snapshot per retrain and hands it to the members that generation contributes, and rebuilds the same mapping on resume from the per-iteration checkpoints; every phase computes a member's vectors with *that* network | `agent/policy.py`, `pipeline.py` | §0.2 — the current network is a drifting basis and would make the pool a moving target; nothing else knows which generation a member belongs to |

Not new, reused as-is: `amortised_init` (the head), `fit_embeddings(steps=0)` (its contract for
`K = 0`), `collate`, `hand_tokens` (with a different `observer_pos`), `_pad_vectors`, `with_style`
(the sibling pattern ⚠1 copies), `label_chunks` / `_label_requests`, `PoolSampler`, `progress`.

---

### 1.1 Cost model — hypotheses until run on the Spark

Trunk: `d_model 512 × 16 layers`, ≈ 50 M parameters, ≤ 80 decision tokens per hand → one
forward ≈ 2 · 50 M · ~40 tokens ≈ **4 GFLOP per hand**. `S` below is the mean number of
past-agent slots at a table; at late iterations (90 agent members of ~200, mean table 5.5)
`S ≈ 2`.

| phase | refreshes | window | hand-forwards, no cap | hand-forwards, `window = 100` | for scale |
|---|---|---|---|---|---|
| labels (300 × 400, `R = 25`) | 300 × 15 | grows to 400, mean 200 | 300·15·200·S ≈ **1.8 × 10⁶** | ≈ 0.9 × 10⁶ | hero's fit: 300·15·200·(1 + 2·150) ≈ 2.7 × 10⁸ — the addition is **~1 %** |
| corpus (1000 × 2000, `R = 25`) | 1000 × 79 | grows to 2000, mean 1000 | 1000·79·1000·S ≈ **1.6 × 10⁸** | 1000·79·100·S ≈ 1.6 × 10⁷ | the uncapped corpus alone is ~60 % of hero's whole fit; capped it is ~6 % |

Two things the table does not show and P2 has to measure on the dev box:

- **Tokenisation is Python and per (hand, observer).** Hero re-tokenises its whole session at
  every refresh today (`s.tokens(...)` in `_play_sessions`); at 300 × 15 × 200 that is
  9 × 10⁵ calls per iteration and nobody has noticed. The corpus with `S = 2` and a 100-hand
  window is 1.6 × 10⁷ calls per retrain, and with no cap it is 1.6 × 10⁸. If `hand_tokens` costs
  ~0.3 ms that is 1.3 h / 13 h on one core — **the cap is mandatory for P4, and per-(hand,
  observer) tokens may need to be kept from the block they were played in rather than rebuilt**.
  Whether they do is a measurement (P2 acceptance), not a design choice made blind.
- **Memory.** A window batch of 100 hands × 9 seats is small; 2000 hands is `collate`d as one
  batch today for hero and would need chunking for the trunk pass. The cap removes the question.

---

## 2. How to use this document

Each section from §P1 on is one Claude Code session:

> Реализуй раздел P<n> из `versions/v8/PLAN_AMORTISED_POOL.md`.

Fields per session are those of `PLAN_PIPELINE.md` §0 (Depends on / Reads first / Deliverables /
Design / ⚠ Sign-off / Tests / Acceptance / Non-goals), and the close-out protocol is the same:
the whole battery green with its wall clock reported, `ARCHITECTURE.md` updated (§1 tree, §2
subsection, §7 test table), GPU-only and scale-only claims named as untested, no commit unless
asked.

**Order matters here more than in most plans.** P1 builds the member and the tokens; P2 measures
whether a past agent is any stronger with them — **if it is not, P3–P5 are not built** and the
report is the deliverable. P3 wires the labels phase, P4 the corpus, P5 records the decision.

---

## P1 — The conditioned past agent

**Depends on:** nothing.

**Reads first:** `agent/policy.py` (both classes, whole file); `env/session.py::Session`;
`nets/embedding_net.py::amortised_init`, `fit_embeddings`; `train/generate.py::_pad_vectors`,
`_slot_of_seat_at`; `CONCEPT.md` §5.3–5.5, §9; `ARCHITECTURE.md` §2.8.

**Deliverables**

- `agent/policy.py::FrozenAgentMember.__init__(name, n_actions, net, max_players, device,
  style=None, embeddings=None, own_slot=None)`. `embeddings` is `(max_players, d_emb)` or `None`;
  `own_slot` is the slot this member occupies in its session, required iff `embeddings` is given.
  `with_vectors(name, embeddings, own_slot) -> FrozenAgentMember` — a `copy.copy` sibling with
  the table swapped in, sharing `net` and `style` by reference, the way `with_style` shares the
  base.
- `FrozenAgentMember._slot_of_seat(ctx)`: identity when `embeddings is None` (today); otherwise
  the rotation in which `own_slot` sits at `ctx.acting_pos`, i.e. `_slot_of_seat_at(n,
  acting_pos)` shifted so that `slot_of_seat[acting_pos] == own_slot` — one expression, asserted
  against `Session.slot_of_seat` in the tests.
- `FrozenAgentMember.logits`: `emb = embeddings[batch["slot"]]`, `seat_emb =
  embeddings[batch["seat_slot"]]` when present — the same two gathers `AgentPoolMember` does —
  and the zero tensors otherwise. `hole_override`, truncation and `_deck_seen_by` unchanged.
- `env/session.py::Session.tokens(max_players, n_actions, observer_slot=0, window=None)`:
  `observer_pos = seat_of_slot(observer_slot, h)`, `slot_of_seat = slot_of_seat(h)`, over
  `records[-window:]` when `window` is given. `ranges` are passed **only for `observer_slot ==
  0`** — they are the §5.7 target from hero's view and `hand_tokens` refuses keys for seats
  another observer has no live opponent at; the amortised pass has no use for them.
- `train/generate.py::amortised_vectors(embed_net, session, slot, max_players, n_actions,
  window, device) -> (max_players, d_emb)`: `session.tokens(..., observer_slot=slot,
  window=window)`, drop empty hands, `collate`, `embed_net.amortised_init(batch,
  session.num_players)`, `_pad_vectors`. Returns the zero table when no hand has a token (cold
  start, §5.5).

**Design.** The member is the one place a past agent's vectors reach a forward, and it keeps
both of `FrozenAgentMember`'s contracts — any seat, any snapshot, any holding — because the
table is indexed by slot and the slot is derived from the acting seat (§0.4, 4). Nothing about
which vectors are *right* lives here; that is P3's block discipline. `amortised_vectors` is
composition around `amortised_init`; it exists as a function because P3 and P4 both call it and
`CLAUDE.md` §5 does not want it written twice.

⚠ **Sign-off:** ⚠1, ⚠2, ⚠3.

**Tests** — `tests/test_frozen_agent_vectors.py`, plus edits noted:

- with `embeddings=None` every existing assertion about `FrozenAgentMember` holds bit-identically
  (`test_pipeline.py::test_a_past_agent_answers_the_question_the_posterior_asks_of_a_member`,
  `…observes_the_moment_it_is_asked_about…` run unedited).
- `with_vectors` shares `net` and `style` by identity and changes neither the base's answers nor
  the base's `embeddings`.
- for every table size 2–9 and every hand index, the member's derived `slot_of_seat` equals
  `Session.slot_of_seat(h)` when seated at `Session.seat_of_slot(own_slot, h)`.
- a non-zero table changes the logits and a zero table reproduces the `embeddings=None` logits
  exactly; the table row read for the acting seat is `own_slot`'s (permute the other rows: the
  acting seat's answer moves only when its own tablemates' rows move — mirror of
  `test_with_a_zero_embedding_the_network_cannot_tell_the_players_apart`).
- under `hole_override` and at a historical snapshot the conditioned member still sees only the
  hypothetical holding and the prefix — `test_observation_parity.py`'s checks re-run against the
  conditioned member.
- `Session.tokens(observer_slot=j)` reveals exactly slot *j*'s hole cards in every hand and nobody
  else's; `observer_slot=0` is byte-identical to today's output; `window=W` returns the last `W`
  hands.
- `amortised_vectors` on an empty session is the zero table; on a session with hands it equals
  `fit_embeddings(..., steps=0)` over the same batch, padded; the observer axis matters — the
  vectors from slot 1's view differ from slot 0's on the same hands.

**Acceptance.** Battery green with no edit to any existing assertion; the new file covers every
bullet above; wall clock reported.

**Non-goals.** No change to `AgentPoolMember`, to the driver, to `generate.py`'s loop, to the
mirrors. No config key yet.

### Outcome — built 2026-09-03

`agent/policy.py`, `env/session.py`, `train/generate.py::amortised_vectors` and
`tests/test_frozen_agent_vectors.py` are on disk, ⚠1–⚠3 as signed off. No existing assertion
moved: the whole battery passes unedited, which is the acceptance criterion — a conditioned
member is a *new* answer only when it is given a table, and every caller in the tree still
builds it without one. **Nothing in the pipeline calls any of it yet**; P1 is the capability and
P2 is the measurement that decides whether the phases use it.

Four things are worth carrying into P2 and P3:

1. **The rotation is derivable, and that is what keeps this one class.** A member told its own
   slot needs no hand index and no metadata: `h ≡ own_slot − acting_pos (mod n)` fixes every
   seat's slot from the seat it is asked to act at. The test asserts it against
   `Session.slot_of_seat` over every table size 2–9 and every hand, and separately that the same
   tablemates relabelled from another slot give the *same answer* through a different
   tokenisation. So P3 needs one member per (session, slot, block) — not per seat — and it works
   unchanged inside a rollout and under a posterior query.
2. **A zero table is bit-identical to no table**, asserted on played contexts. Block 0's cold
   start is therefore literally the D12 policy, and `"zero"` in P3/P4 can be the same code path
   rather than a branch that has to be kept in step.
3. **The §5.7 targets are hero's alone.** They are keyed by a seat *hero* has a live opponent
   at, and `hand_tokens` refuses such a key from another observer's view — its own seat among
   them. `Session.tokens` passes them for slot 0 only; a test pins that the other slots come
   back with no targets rather than an assertion.
4. **The cost is where §1.1 predicted it.** `amortised_vectors` is one trunk pass, but it is one
   *tokenisation per (hand, observer)* around it, in Python. Nothing here measures that at
   corpus scale — P2's timing section is the first evidence, and P4 is the phase that cannot
   afford to be wrong about it.

**What is untested:** every claim above is CPU-only and toy-scale. Nothing has run on the Spark,
nothing has run at pipeline scale, and no measurement yet says a conditioned past agent plays
any better than one at `e = 0` — that is exactly what P2 is for.

---

## P2 — Does a past agent play better with them? The gate

**Status 2026-09-03: built, not run, and no longer a gate on P3** — see §0.3a. It stays here as
a measurement that can be pointed at any two checkpoints when there is a reason to.

**Depends on:** P1.

**Reads first:** `gates/g1.py` (the shape of a gate: sessions, seeds, a report, a printed
table), `env/session.py::build_sessions`, `play`; `pipeline.py::frozen_agent_net`,
`agent_variant_members`; `CONCEPT.md` §12 on reporting standard errors; `CLAUDE.md` §1 on what
must not feed back.

**Deliverables**

- `gates/pool_conditioning.py` + `config_pc.json`: given an `agent.pt` and an `embedding.pt`
  from a pipeline run (paths in config, under `/data/v8/`), plays the *same* sessions — same
  seeds, same tables, same opponents drawn from the bootstrap pool — twice: with the checkpoint
  seated at slot 0 as a `FrozenAgentMember` at `e = 0`, and as the conditioned member refreshed
  every `R` hands from its own view with `pool_agent_window`. Reports the agent's BB/100 in each
  condition, the paired difference per session, its standard error over sessions, and the same
  three numbers grouped by table size and by stack depth (the `CLAUDE.md` §1 falsifiable
  claim: no cliff across neighbours).
- The report is stamped `POOL MEMBER CONDITIONING — not an agent result` and lives under
  `/data/v8/pool_conditioning/<timestamp>/`, never beside `evaluation/`.
- A timing section: `hand_tokens` per call, `amortised_init` per hand of window, on the machine
  the gate ran on — the numbers §1.1 needs.

**Design.** The question is narrow: *is the same network stronger at a table when it reads
`K = 0` vectors of its tablemates than when it reads zeros?* Slumbot cannot answer it (feedback
rule, and a past agent never sits there); the labels phase cannot answer it cheaply (it is
days). A paired self-play comparison over identical hands can, in minutes on the Spark. The
opponents are the bootstrap pool only, so the gate needs no pipeline run beyond the two
checkpoints. Pairing by session and reporting the difference's standard error is what makes a
small number legible; `hands_per_session` and `n_sessions` are config.

**Pre-registered decision rule** (written here so it is not written after the number):

- difference ≥ 2 standard errors above zero overall, and no table-size or stack bucket more than
  2 SE *below* zero → **proceed to P3**;
- difference within ±2 SE → **(c) is not worth its plumbing at this checkpoint**: report, do
  not build P3–P5, leave D12 at (a) with the report cited;
- difference significantly negative → a result to report (`CONCEPT.md` §11.2 R5 predicts the
  fit can hurt on tiny windows; `K = 0` is the regime where G1 saw it help, so a negative here
  says something about the agent's `e ≠ 0` input distribution, not about the head).

⚠ **Sign-off:** ⚠10. And an **owner decision**: which checkpoint pair is the one this is judged
on — the gate's answer depends on the agent's iteration, and a rule that lets the plan pick the
best of several would be selection.

**Tests** — `tests/test_pool_conditioning_gate.py`: the gate runs end to end on CPU at toy scale
(tiny nets from `g1_fixtures`, 4 sessions × 6 hands); the two conditions play *identical* hands
when the vectors are forced to zero (paired difference exactly 0.0); the paired SE is the
hand-computed one; the report refuses to be written without the stamp and the standard error.

**Acceptance.** The toy test green; the gate run once on the Spark on the owner-named
checkpoints; the report on disk; the decision rule applied and the outcome written into §P2 of
this file as an "Outcome" block.

**Non-goals.** No change to the pipeline. No Slumbot. No tuning of `pool_agent_window` against
the gate's own number beyond the one value config names before the run.

### Outcome — built 2026-09-03, **not yet run**

`gates/pool_conditioning.py`, `config_pc.json` and `tests/test_pool_conditioning_gate.py` are on
disk, ⚠10 as signed off; nothing outside `gates/` changed. `ARCHITECTURE.md` §2.7a describes it.

**The acceptance criterion is only half met, and deliberately so.** The toy test is green, but
"the gate run once on the Spark on the owner-named checkpoints" cannot be done from the dev box:
there is no GPU here, no pipeline checkpoints, and *which* `(agent.pt, embedding.pt)` pair the
answer is judged on is the owner decision this section already asks for — picking it here would
be the selection the plan warns against. `config_pc.json` names
`../../data/v8/run1_ranges/iter_0000/{agent,embedding}.pt` as a placeholder to be replaced with
that pair before the run. **P3 is therefore blocked on a measurement, exactly as intended.**

Four things the build settled, and one it exposed:

1. **The gate mirrors the run's config, not G1's.** The pool it plays against, the action set
   and the raise grid all have to be the checkpoint's, so `config_pc.json` is `config.json`'s
   `game` / `style` / `bootstrap` / `embedding_net` plus a `pool_conditioning` section. Both
   checkpoints carry the config they were trained under and the gate asserts the load-bearing
   keys against its own, so the mismatch that would otherwise be a silent, uninterpretable
   number is an assertion.
2. **`pool_agent_window` already lives where ⚠4 puts it** — in `embedding_net`, beside `R`,
   which the gate reads for the refresh interval rather than inventing a second knob.
   `pool_agent_vectors` is *not* added: the gate plays both conditions by construction, and the
   switch is P3's.
3. **One reserved pool entry per session, not per seat** — P1's derived rotation paying for
   itself in the first thing that seats a conditioned member.
4. **The decision rule is printed, not encoded.** The gate prints the rule beside the number
   and writes no verdict field: mechanising it would put the plan's judgement inside the
   measurement.

**What the toy test cannot show, and this matters for reading P2's result later.** At toy scale
the *real* `K = 0` vectors move an untrained network's policy by about 5e-4, which flips no
sampled action over two dozen hands — so the toy run's paired difference is exactly zero through
the genuine path. The battery therefore pins the *plumbing* (pairing, zero-equivalence, the
arithmetic, the refusals) and a deliberately loud table pins that the vectors reach the driver
at all. Whether a *trained* agent's policy moves enough for the difference to be non-zero is the
question the Spark run answers, and a null result there will need this distinction: it can mean
"the conditioning does not help", and it cannot be read as "the wiring is broken".

**What is untested:** everything about scale and hardware. No GPU path has run, no checkpoint
has been loaded, the timing numbers §1.1 wants exist only as a section the report will fill in,
and the effect size is unmeasured.

---

## P3 — The labels phase: past agents refreshed per block

**Depends on:** P1, and the owner's decision of §0.3a (not P2's number, which was never taken).

**Reads first:** `train/generate.py` whole (`generate_labels`, `_play_sessions`, `_seat_hero`,
`_label_sessions`, `_label_in_parallel`, `_write_play`, `_resume_state`, `_stale_reason`);
`oracle/parallel.py::_worker_main`, `spawn_workers`, `mirror_spec`, `build_mirror`,
`hero_mirror`; `tests/test_label_generation.py::test_the_embedding_of_a_block_ignores_every_later_hand`,
`…first_block_is_the_cold_start`, `…resumed_call_plays_the_unlabelled_hands_and_no_others`;
`tests/test_parallel_labels.py` whole; `ARCHITECTURE.md` §2.10, §2.11 (resume).

**Deliverables**

- `config.json`: `embedding_net.pool_agent_vectors: "zero"` and `embedding_net.pool_agent_window:
  null` (⚠4), read in `generate_labels`. `"zero"` leaves every code path below inert.
- **The vintage network** (⚠11). A past agent's member holds the embedding network of its own
  generation; `pipeline.py` freezes a snapshot right after each retrain and gives it to the
  members that iteration appends, and reconstructs the same mapping on resume by loading the
  latest embedding checkpoint at or before each past iteration, one load per generation. The
  labels phase computes each conditioned slot's vectors with the network that slot's member
  carries — never with the loop's current one. Members whose generation has no network (nothing
  was trained before them) and every non-agent member are never conditioned.
- **The observer axis is one row wide when the switch is off.** `block_vectors[i][b]` is
  `(n_obs, max_players, d_emb)` with `n_obs = 1` under `"zero"` and `max_players` under
  `"amortised"`; index 0 is always hero's. The `"zero"` run therefore writes the same volume of
  vectors it writes today rather than nine times it, and the mode is in the play signature, so a
  directory cannot be resumed into under the other setting.
- **Reserved entries.** After hero's `hero_plain` / `hero_rec` blocks, `generate_labels` reserves
  one `play_pool` entry per `(session, slot ≥ 1)` whose `s.members[slot]` is a
  `FrozenAgentMember` — `pool_slots[i][slot]`, `None` for other members. `spec.seat_members` for
  hand `h` points a past-agent seat at its session's entry instead of the shared member.
- **Block vectors** (⚠5): index 0 of `block_vectors[i][b]` is hero's fitted table (unchanged),
  index `slot` is `amortised_vectors(member's own embed_net, s, slot, …)` for a past-agent slot
  and zero otherwise. Block 0 is all zero. Written to `vectors.npz` as one
  array; `_resume_state` asserts the new shape; the label's `"embeddings"` is
  `block_vectors[i][h // R][0]`.
- **Refresh.** In `_play_sessions`, after hero's fit for a session, the past-agent slots' tables
  are computed for the same block from the same records. The progress postfix names both
  ("fit block b: n/N sessions", then "amortise").
- **Seating** (⚠7): `_seat_conditioned` builds hero's per-seat members *and*
  `s_members[slot].with_vectors(f"{name}@s{i}", block_vectors[i][b][slot], slot)` for every
  reserved slot, and writes them into the play pool. Called where `_seat_hero` is today: the
  play loop (per block), the sequential label loop (per `(i, block)` change) and `_worker_main`.
- **Mirrors** (⚠8): reserved entries are holes in `mirror_pool`; the worker rebuilds them per
  block from its own `block_vectors` slice through the `"frozen"` spec of the base member, which
  gains the style and `own_slot`. Server, slab, runners untouched.
- `_play_signature` gains `pool_agent_vectors` and `pool_agent_window`: two runs that disagree on
  them played different hands and must not splice.

**Design.** Everything here is hero's discipline applied to more seats. The one genuinely new
consideration is the interplay with the posterior: a past agent's decisions in block *b* were
taken with the block-*b* table; when the oracle later asks "what would this member have done
holding *X*" at one of those decisions, the member it asks must hold the *same* table. That is
exactly why the seating happens per `(session, block)` in the label loop and in every worker,
and why the tables are persisted rather than recomputed on resume (§0.4, 2–3).

`_results_by_member` and `PoolSampler.update` are indexed by `s.members` and do not see the
reserved entries — PFSP keeps scoring the pool member, not its per-session copy.

⚠ **Sign-off:** ⚠4, ⚠5, ⚠7, ⚠8.

**Tests** — additions to `tests/test_label_generation.py` and `tests/test_parallel_labels.py`;
the existing files run with `"zero"` **unedited** and must stay green:

- with `"zero"`, shards are byte-identical to the shards written before this change (pin against
  a fixture written by the current code, once, before P3 starts).
- with `"amortised"`: a past agent's table for block *b* is the `amortised_vectors` over the
  records of blocks `0 … b−1` from *its* view — asserted by replaying with the session's tail
  changed and requiring bit-identical tables, the exact shape of
  `test_the_embedding_of_a_block_ignores_every_later_hand`; block 0 is zero for every slot.
- the member seated for a past-agent slot in block *b* holds `block_vectors[i][b][slot]` and the
  posterior's queries about its block-*b* decisions go through a member holding the same table
  (the sequential loop and a two-worker run agree label for label — extend
  `test_two_workers_produce_the_labels_one_process_produces` to a pool containing a past agent
  with `"amortised"` on).
- resume: `vectors.npz` round-trips the new shape; a resumed call plays the unlabelled tail with
  the *stored* past-agent tables and computes none; a directory whose `pool_agent_vectors`
  differs is set aside.
- a hand's `"embeddings"` in the shard is still hero's table only.
- the label file's shard round-trip (`test_a_shard_round_trips`) is unchanged.

**Acceptance.** Battery green; the `"zero"` fixture pin holds; a toy `"amortised"` run through
both the sequential and the parallel path produces identical shards; wall clock reported and
under the §4 budget.

**Non-goals.** No change to the corpus phase (P4). No `K_pool`. No change to what a label stores.

### Outcome — built 2026-09-03

The labels phase seats conditioned past agents, and `pipeline.py` gives every past agent the
frozen network of its own generation (⚠4, ⚠5, ⚠7, ⚠8, ⚠11). `"zero"` is the default and is the
same run byte for byte. Nothing in the corpus phase changed; that is P4.

Five things settled during the build, three of them departures from the text above:

1. **The observer axis is one row wide with the switch off** (already folded into the
   deliverables). A nine-row table per session per block would be 45 MB an iteration of zeros in
   the default mode. The mode is in the play signature, so a directory cannot be resumed into
   under the other setting and the two shapes never meet.
2. **The base member is looked up through the play pool, not passed in.** That one detail is
   what lets the same seating function run in the parent process and inside a label worker: in
   the parent the lookup finds the pool member, in the worker it finds that member's weightless
   mirror. The reserved per-(session, slot) entries are holes in the mirror, exactly as hero's
   are, so the inference server, the slab and the runners did not change at all.
3. **The `"zero"` pin is same-process, not a golden file.** The plan asked for a fixture written
   by the pre-change code; a stored `.npz` would break on a library upgrade and prove nothing
   about this change. Instead one test runs the toy phase twice in one process — once with the
   new keys absent entirely, once with them present and off, the window set to a value that must
   not be read — and compares the shards byte for byte. Combined with the whole pre-existing
   battery passing unedited, that is the property the fixture was for.
4. **The vintage snapshot is taken after each retrain and shared.** Iterations that were trained
   against the same network get the same object, so they also share one inference runner when
   they are mirrored into a worker. On resume the mapping is rebuilt by loading the latest
   embedding checkpoint at or before each past iteration, one load per generation, and a test
   asserts the resumed pool gets the same generations the live run gave. With the switch off no
   snapshot is taken at all.
5. **A member with no generation is never conditioned**, and neither is any non-agent member.
   Both fall through to `e = 0` with the switch on, and the run is byte-identical to the switch
   being off.

**What is untested:** the scale and the hardware, as ever — no GPU, no run at size. Two specific
unknowns the toy battery cannot reach: what the per-observer tokenisation costs at 300 × 400
hands with several past agents per table, and whether the extra tables meaningfully change what
the oracle labels look like. Both are visible on the first real iteration.

---

## P4 — The corpus phase: the same refresh, no hero

**Status 2026-09-03: DROPPED — not built, and deliberately not built.** The owner's reasoning,
which is a design argument and not a deferral:

> The embedding corpus is exactly where conditioning must *not* happen. It makes a player's
> style non-constant within a session, which is the one thing the whole single-vector scheme
> assumes away. And the vectors of a real opponent will be out of distribution for the network
> whatever we do. So it is *good* if the agent learns to use a description of a **static**
> player against strategies that in fact keep changing — that is the deployment situation.

What this buys and what it costs, so that a later reader does not re-open it by accident:

- **The corpus keeps a stationary target.** Every player in it plays one policy for the whole
  session, so a member's row means "this player's style" and not "this player's style averaged
  over how its tablemates happened to draw it out". R3 below is closed by construction rather
  than watched.
- **The mismatch is deliberate and is the realistic one.** The embedding network learns to
  describe static players; at label time it is asked to describe past agents that adapt. That is
  a train/deploy gap *for the network*, and it is the same gap that exists against any human or
  any real bot, none of which are stationary either. Training hero to act on a static summary of
  a moving opponent is training for the case that actually occurs.
- **What is given up:** nothing measurable is. The corpus was never the reason for this plan; the
  labels phase was.
- **The cost question dies with it.** §1.1's expensive row was the corpus (≈ 60 % of hero's fit
  uncapped, ≈ 6 % capped) and the per-(hand, observer) tokenisation volume that went with it. The
  labels phase's addition stays at ≈ 1 %, so `pool_agent_window` is no longer a cost lever and
  `null` — the whole session — is the natural value. R2 is closed with it.

Everything below is what P4 *would* have been. It is kept for the record and is not to be built
without the owner reopening the argument above.

**Depends on:** P3.

**Reads first:** `pipeline.py::embedding_phase`; `env/session.py::play`,
`label_showdowns` (called inside `play`); `train/generate.py::_play_sessions` as left by P3;
`train/embed_train.py` (how the corpus is read: `s.tokens(...)` at slot 0, `member` per token);
`ARCHITECTURE.md` §2.11 "the embedding network is the first phase".

**Deliverables**

- `_play_sessions` accepts `hero_rec=None, agent_member=None`: no hero recorders, no fit, only
  the past-agent refresh (⚠9). With `"zero"` and no hero it degenerates to `n_blocks` calls of
  `play` that concatenate to today's single call — asserted bit-identical.
- `embedding_phase` builds its sessions as today (all members drawn from the pool, slot 0 is
  still the corpus observer for training) and plays them through `_play_sessions` in blocks of
  `R`, with `pool_agent_window` applied. The corpus is then read by `train_embedding_net` exactly
  as before — the tokens it trains on are unchanged in form; only the past agents' *actions* in
  them differ.
- Progress: the corpus keeps one global bar over hands (`play:corpus` today) with the refresh in
  the postfix, as the labels phase does.

**Design.** This is where a conditioned past agent changes the *embedding network's* target
distribution: its row is now trained on play that depends on its tablemates, so the row becomes
"this agent, averaged over the tables it sat at". That is the same thing a fitted vector of a
human regular is, and §0.3's flat curve says the head does not need long windows to read it.
What has to be watched is the retrain's `action_ce` on past-agent tokens against the previous
iteration's: a *rise* is expected (an adaptive player is less predictable from one vector) and
should be small; a large one is a result for `ARCHITECTURE.md` §5, not a bug. `pool_agent_window`
is the single cost lever (§1.1); if the tokenisation timing from P2 says it is not enough, the
per-(hand, observer) token cache is the next item and it is a **new** ⚠ to raise then, not now.

⚠ **Sign-off:** ⚠9.

**Tests** — additions to `tests/test_pipeline.py`:

- with `"zero"`, `embedding_phase`'s corpus records are identical to the single-call corpus
  (same seeds, same hands, same decisions) — the block split is invisible.
- with `"amortised"` and a pool containing a past agent, the agent's decisions in block *b* were
  taken by a member holding the block-*b* table computed from its own view over blocks
  `0 … b−1` (same replay-the-tail shape as P3); other members' decisions are unchanged between
  the two settings on the hands before the first past-agent decision diverges.
- the full toy loop (`test_the_loop_runs_and_writes_every_artefact`) runs with `"amortised"` on
  and writes every artefact; resume (`…reproduces_an_uninterrupted_run`) holds with it on.

**Acceptance.** Battery green under both settings of the switch; wall clock reported.

**Non-goals.** No change to `train_embedding_net`, to the table, to the corpus observer (slot 0),
to `label_ranges`.

---

## P5 — Recording the decision

**Depends on:** P3, and the two owner decisions of §0.3a and P4.

**Deliverables**

- `PLAN_PIPELINE.md` D12: append "**Revisited 2026-09-xx: (c)** — amortised vectors, `K = 0`,
  refreshed per block; `PLAN_AMORTISED_POOL.md` P2 carries the measurement" — or, if P2 said
  stop, "revisited, measured, (a) stands" with the report path.
- `CONCEPT.md` §4.1: the paragraph "A past agent in the pool plays at `e = 0`" gains the (c)
  option and the reason D12's objection did not apply; the asymmetry sentence ("hero exploits its
  table and the pool's own agents never do") is corrected to what remains of it — the pool's
  agents now exploit at `K = 0` while hero does at `K`.
- `ARCHITECTURE.md`: §2.8's `FrozenAgentMember` bullets; §2.10 (blocks now carry every seat's
  vectors; the `vectors.npz` shape); §5 — reword "the pool's own agents never exploit the tables
  they sit at"; §7 the new test files. **Not** §2.11: the corpus is unchanged (P4).
- `config.json` carries the two keys; their values are a design decision (§0.3a, P4), not a
  measurement.

**Acceptance.** Every statement in the four documents about a past agent's vectors is true of
the code on disk.

### Outcome — done 2026-09-03

`PLAN_PIPELINE.md` D12 and its S9 outcome note, `CONCEPT.md` §4.1, `ARCHITECTURE.md` §2.7a, §2.8,
§2.10, §5 and §7 now say what the code does: a past agent plays at `e = 0` by default and can be
given the `K = 0` reading of its own tablemates through its own generation's network, the labels
phase is the only phase that does so, and the corpus deliberately does not. The asymmetry
`CONCEPT.md` §4.1 and `ARCHITECTURE.md` §5 recorded — "hero exploits its table and the pool's own
agents never do" — is now a *switchable* asymmetry rather than a fixed one, and both say so.

---

## 3. Risks and open questions

| # | Risk | Where it shows | What to do |
|---|---|---|---|
| R1 | The agent has never seen `K = 0` vectors as input; a past agent may play *worse* conditioned than cold | phase D's warm/cold gap across iterations | **open, and accepted** (§0.3a): the owner's argument is that an exploitable pool makes this positive by construction early on. The `"zero"` switch is the control if it ever needs settling, and P2 is the instrument |
| R2 | ~~Tokenisation cost in the corpus~~ | — | **closed by P4 being dropped.** The corpus is not conditioned, and the labels phase's addition is ≈ 1 % of hero's own fit |
| R3 | ~~The embedding network's rows for past agents become averages over tables~~ | — | **closed by P4 being dropped.** The corpus keeps every player stationary, so a row still means one style |
| R4 | The oracle's `Q` moves: labels are now EV against a slightly stronger pool, so `ev_oracle` is not comparable across the switch | phase D across iterations that straddle the change | never flip the switch mid-run; a run is one setting |
| R5 | Hero has no corpus row, so the amortised estimate of *hero's* vector is the head generalising to an unseen player | G1's unseen curve says it does (1.80 → 1.52 nats); the agent-specific answer is P2 | none beyond P2; if it matters, "hero's hands in the corpus" is `ARCHITECTURE.md` §2.11's own open question and a separate plan |
| R6 | Nine slots × hundreds of sessions of per-block members and mirror specs | worker start-up log, slab count unchanged | entries are `copy.copy` siblings sharing one net; mirror specs are small tuples; nothing per-decision grows |

**Owner decisions — all taken.** (i) D12 was opened to revision and revisited (§0.3a);
(ii) ⚠1–⚠5, ⚠7–⚠11 signed off and built; (iii) P2 was never run and needs no checkpoint pair
(§0.3a); (iv) `pool_agent_window` is `null` — the whole session — since dropping P4 removed the
only reason to cap it.

**What is left in this plan:** nothing to build. One thing to *decide when convenient* — whether
to turn `pool_agent_vectors` on for the next run, which is one config value, and whether to do it
at a run boundary (R4: never mid-run, a run is one setting). One thing to *watch* on the first
conditioned run: phase D's warm/cold gap, which is R1's only free instrument.
