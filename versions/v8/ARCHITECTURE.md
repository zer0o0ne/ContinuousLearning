# v8 — Architecture

**Status: gates G1 and G3 implemented (CONCEPT.md §14), the BR oracle (variant A) with them,
the agent network (§6.1), the target construction and training loop that fits it (§6.2), and
label generation end to end (§8) — play, refresh the embeddings, label hero's decisions, shard —
and the pool sampler that decides who sits at those tables (§4.4).
G3 has been exercised only at toy scale on CPU — it has not been run, so what a label costs is
still the hypothesis §13 wrote down, and the decision it is supposed to inform (variant A or the
value-bootstrapped variant C of §7.4) has not been taken. Everything below has run only at toy
scale on CPU; no agent has yet been trained on a generated corpus, because nothing ties the
pieces into a loop — `pipeline.py` is still missing.**

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
  config_g3.json        the G3 sweep surface — samples × combos × table × stack

  env/                  poker engine, from v7, verbatim
    legal.py            the one legality rule (§6.2)                    NEW
    driver.py           lock-step vectorised driver + rollout plumbing (§3) NEW
    session.py          sessions: rotation, uniform table draw, play (§8)    NEW
    showdown.py         reveal detection + the two showdown labels (§5.1a)  NEW
  pool/                 entity 2 — the opponent pool (§4)               NEW
    base.py             PoolMember: logits → styled, legal distribution
    style.py            live style modifiers, 32 scalars per member (§4.2)
    degenerate.py       always-fold / call / min-raise / maniac / nit
    v7_member.py        a vendored v7 checkpoint as a pool member (§4.3)
    build.py            pool construction from config, fresh style draws
    sampling.py         PFSP + embedding dedup + uniform floor (§4.4)   NEW
  nets/                 v8's own networks                               NEW
    features.py         §5.1 token features — where observation parity lives
    tokeniser.py        the shared tokeniser MLP (§5.1, OI-4)
    trunk.py            HandEncoder — the §5.1/§5.2 trunk, shared as code    NEW
    embedding_net.py    entity 4 — the opponent-embedding network (§5)
    agent_net.py        entity 3 — the agent's policy network (§6.1)         NEW
  agent/                                                                     NEW
    policy.py           the agent as a pool member (§6.1, §7.1)
  train/                                                                     NEW
    targets.py          softmax(Q_norm / T) and the KL loss (§6.2)
    agent_train.py      one training cycle of the agent (§6.2, §8)
    generate.py         label generation end to end + the shards (§8)        NEW
    embed_train.py      §5.4 training of the embedding network (§5.4, §8)     NEW
  oracle/               entity 5 — the BR oracle (§7)                  NEW
    posterior.py        reach-weighted opponent ranges (§7.2)
    rollout.py          variant A: Q(s, ·) by full rollout (§7.1)
  gates/
    g1.py               the G1 experiment (§14)                         NEW
    g1_analysis.py      section A — regrouping a finished g1_report.json NEW
    g3.py               what one oracle label costs (§14, §13)           NEW
  vendor/v7/            frozen snapshot of v7's agent code (§4.3)       NEW
    attn_utils.py, perception/*, action/*, modifiers.py   copied verbatim
    agent.py            perception + action-head subset of v7's ASI
    events.py           the v7 event format

  attn_utils.py         causal+padding mask helper, inherited from v7
  utils.py              Logger, get_amp_config, resolve_device, progress
  gto_utils/            hand evaluation, equity, CFR solvers v1–v5, from v7
  evaluation/
    slumbot_eval.py     from v7, verbatim — does not import yet (§5)
  tests/                19 files, 306 tests, ~58 s
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
| `utils.py` | `v7/utils.py` + `resolve_device`, `progress` | `Logger`, `get_amp_config`, the CUDA → MPS → CPU device resolution `CLAUDE.md` §3 requires, and the single `tqdm` wrapper §5 requires long loops to use. |
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

### 2.2a Rollout plumbing — pinned deck and forced prefix

A BR-oracle rollout (`CONCEPT.md` §7.1) is "this hand, these cards, this prefix of decisions,
then free play". Two optional `HandSpec` fields express it, and `nets/features.py` gained the
matching observation:

| Field | Meaning |
|---|---|
| `HandSpec.deck` | 52 ints, asserted to be a permutation of `0..51`. Written onto `table.deck` immediately after `start_table()`, before the record is built. `Table` reads every card from `self.deck` and caches nothing, so the layout — `deck[:5]` the board, `deck[5 + 2p : 7 + 2p]` seat *p*'s hand — is the whole contract |
| `HandSpec.forced_actions` | action indices consumed in decision order. Decision *k* is forced while `k < len(forced_actions)`; the rest of the hand runs free |
| `hand_tokens(..., pending=ctx)` | one extra decision token built from a `DecisionContext` whose action has **not** been chosen: `action = -1`, `legal` from the context. This is the moment the *agent* observes, as opposed to the completed hand the embedding network observes (§9's two moments) — so a hand with a pending decision has no showdown, and passing both is refused |

`run` splits each lock-step round's queries into forced and free **before** grouping by member:
a forced decision costs no policy call and no draw from the hand's generator, which is where the
oracle's saving comes from — a rollout replays 5–15 decisions without touching the network.
`_apply` gained one `action_idx` parameter and one branch: **one recording path, two ways of
choosing the action**. An illegal forced action is an assertion, naming the seat, the decision
index and the legal set, because a prefix taken from a real record is legal by construction.

The safety argument for editing the driver at all is `test_rollout_plumbing.py`'s first test:
replaying a recorded hand with its own deck and its own full action sequence reproduces that
hand element for element — deck, snapshots, decisions, rewards, showdown — at every table size
2–9 and at both stack extremes.

### 2.2b `oracle/posterior.py` — reach-weighted opponent ranges

`CONCEPT.md` §7.2: the oracle knows the opponents' strategies, so it knows their ranges — but
the range is the posterior implied by what they *did*, not the set of hands still possible.

```
w(combo) ∝ prior(combo) · Π_t  max(floor, P_i(a_t | combo, history_t))
```

`opponent_posterior(record, opp_pos, observer_pos, pool, n_actions, through_decision, floor,
max_combos, rng)` returns `(combos, weights)` — `(C, 2)` int64 and `(C,)` float64 summing to 1.
`combo_universe(dead_cards)` is the prior's support, in a fixed order that every weight vector
is indexed by.

Three properties, each of them a way of getting it wrong:

* **The observer's information and no more.** Dead cards are the board *as visible at
  `through_decision`'s street* plus `observer_pos`'s own two cards. The opponent's real holding
  stays in the universe — it is what the posterior is over — and nothing here reads it.
* **A prefix function.** Only decisions at or before `through_decision` are conditioned on
  (`-1` is the empty prefix, the prior), so the posterior at a decision is *bit-identical*
  whether or not the hand was played on. That is §9's no-future-leak rule inside the oracle.
* **One question per decision.** A member is asked once per decision it made, with all `C`
  combos in a single batch — five decisions cost five batched forwards of ~1200 rows.

A member is asked about a hypothetical holding through `DecisionContext.hole_override`
(D2): the same situation with other cards, so the observation it answers is built by exactly
the code that built the real one, rather than by fabricating a `HandRecord` per combo. The
override is a view — the record is untouched — and it is `None` on every context the driver
creates.

If every likelihood underflows, the module warns (`RuntimeWarning`) and falls back to the
prior: a member that assigns zero probability to what it did is a bug in the member, not a
reason to return NaN. The `floor` is what normally prevents it.

`max_combos` is **self-normalised importance sampling over the prior**, and it deliberately
departs from v7 `gto_utils/gpu_solver_v5.py::_prepare_range` (D6 said to reuse v5's semantics;
v5's are wrong in two ways and both were fixed at the owner's instruction). v5 draws the
subsample *from the posterior* and renormalises what it kept, which counts the mode twice —
once in the selection, once in the weight — and it does so *after* the likelihood pass, so it
shortens the answer without saving a single policy call.

Here the cap is spent on the prior instead: `m` combos are drawn uniformly without replacement,
each carried at `prior / inclusion probability` (`m / C`, exact for sampling without
replacement, hence a constant — written out rather than cancelled), and the likelihood pass then
runs over those `m` alone. Normalising at the end makes it the standard SNIS ratio estimator:
consistent in `m`, selection independent of the weights, and the members are asked about `m`
combos rather than `C`. The proposal has to be the prior — a posterior-weighted proposal cannot
be drawn before the likelihoods that define it exist, and its without-replacement inclusion
probabilities have no closed form to divide out. The price is variance: a sharply concentrated
range needs a larger `m` than v5's mode-seeking draw would. `test_posterior.py` measures both
sides of that trade — the mass on pocket pairs in a fixture whose true value is 0.488 comes back
at 0.491 over 200 seeds at `m = 128`, where v5's scheme returns 0.89.

Only *one* opponent's marginal lives here. §7.3's approximation of the joint by independent
marginals with a card-removal correction belongs where the joint sample is drawn (the oracle),
so the exact part stays untangled from the approximate one.

### 2.2c `oracle/rollout.py` — variant A, `Q(s, ·)` by full rollout

`CONCEPT.md` §7.1. `action_values(record, decision_idx, driver, pool, hero_member_idx, cfg, rng)`
returns `(q, legal, stats)`: `q` is `(n_actions,)` float64 in **big blinds**, `nan` at every
illegal action, `legal` is the recorded mask, `stats` is a `LabelStats`. `OracleConfig` carries
the five knobs (`samples_per_action`, `max_combos`, `likelihood_floor`, `batch_hands`,
`max_collision_retries`) and every one of them trades cost against noise.

One label is:

1. the recorded `legal_mask` — illegal actions are never rolled out, and `nan` rather than zero
   makes a downstream masking bug fail loudly instead of averaging in a value nobody computed;
2. one `opponent_posterior(..., through_decision=decision_idx − 1)` per **live** opponent (a seat
   that has folded holds nothing that can change a showdown and gets filler cards);
3. one **joint sample** — each opponent's combo drawn independently from its own marginal, and
   the draw rejected and redrawn if two opponents share a card or a card is already on the board
   or in hero's hand;
4. one `HandSpec` per (legal action, surviving sample): the real board and hero's real cards in
   the deck, the sampled cards at the opponents' seats, the rest filled in ascending order;
   `forced_actions` = the recorded prefix plus that action; `seat_members` = the real members
   with `hero_member_idx` in hero's seat; `seed` = a `blake2b` hash of
   `(spec.seed, decision_idx, action, sample)`, so a label does not depend on which other
   rollouts shared its batch;
5. **one `driver.run` for the whole `|A| × S` set** — that is what makes the oracle affordable,
   because every (action, sample) pair is an independent hand and a single hero decision becomes
   one lock-step batch of thousands (`CLAUDE.md` §3: memory-bound, prefer large batches);
6. `q[a]` = the mean of hero's chip delta over the surviving samples, in BB.

The joint sample is drawn **once** and every action is rolled out on the same assignments, so
the differences between the `Q`s — which is all a target is built from — are not swamped by the
card variance the actions share.

**The card-leak rule.** Opponents' cards are fixed *in the deck* and nowhere else; hero's member
is queried through `DecisionContext`, whose `hole_cards` reads hero's own seat. `test_oracle.py`
asserts it by recording every holding hero's member is handed over ~900 rollout queries, not by
reading the code.

**Why the board is the real one.** Hero's action does not change the runout, and the posterior
already conditioned on the visible board. Re-dealing it would average over runouts the posterior
has conditioned away. The price is paid in step 3: a combo the posterior still likes may contain
the turn or the river, and that draw has to die.

**Fold is not special-cased.** Hero's chip delta after folding is minus what hero has already
put in, whatever the opponents hold, so the rollout returns the closed form with zero variance —
and `test_oracle.py` checks it against that closed form at every table size from 2 to 9 and at
both stack extremes.

**`LabelStats` — what S4/G3 reads, and nothing else.** `forwards` counts policy *rows*
(posterior rows + rollout decisions that were not forced), `seconds` is wall clock,
`n_rollouts` is hands played, and `collision_rate` is the fraction of joint draws rejected.
That last one is the size of §7.3's approximation, and it is large: measured on a **9-handed**
preflop decision with eight live opponents, **~98.5 % of draws are rejected** — 16 cards drawn
independently from one 50-card deck almost always repeat. Heads-up on the river it is exactly
zero. So `max_collision_retries` is not a formality at a full ring, and the cost of a full-ring
label is roughly `1 / P(accept)` draws per usable sample. This is a CPU measurement of the
sampler, not of the GPU cost — G3 (S4) is what measures the label.

**Not built here** (S3's non-goals): variant C, value bootstrapping, caching a posterior across
the hero decisions of one hand, and dataset writing (S7).

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

**`sampling.py` — who sits at the table (§4.4).** `PoolSampler.sample_table(n_opponents)`
returns one member index per non-hero seat; hero is slot 0 of the session and is never drawn
(§2.10). Every draw passes the same three mechanisms:

- **PFSP** — weight `f(x) = x ** pfsp_exponent` on the member's *hardness*, so the batch
  concentrates on opponents hero loses to. `update(member_idx, hero_bb, n_hands)` feeds it; a
  member with no results yet has the maximum hardness, so an agent appended by §8 is sampled on
  the next draw rather than after the first result exists.
- **Embedding-space dedup** — `set_vectors` re-clusters the pool from the embedding network's own
  table (§5.4) and a *cluster* is drawn before a member inside it, at a probability proportional
  to the cluster's **mean** weight. The mean, not the sum: a sum would put the proportional-to-
  size behaviour straight back and there would be nothing left of §11.3's mitigation.
- **Uniform floor** — a fixed fraction of draws ignores both. This is the only route back for a
  member whose PFSP weight has gone to zero, which is what "nothing is deleted, things are
  down-weighted" means operationally.

**From BB/100 to a number PFSP can use.** §4.4 transplants AlphaStar's formula, which is written
in terms of a *loss rate*, and that quantity does not exist here: StarCraft results are binary,
poker results are money. `PoolSampler` accumulates hero's total BB and total hands against each
member, takes the mean BB/100, and min-maxes it over the pool — `(m_max − m_i) / (m_max − m_min)`,
with an unplayed member at 1. The scale comes out of the pool itself, so there is no fourth
config knob (§8.1 gives `pool_sampling` three) and no constant that stops meaning anything as the
agent gets stronger.

The cheaper reading — count the sessions whose BB/100 was negative — was implemented first and
then rejected, and the reason is worth keeping. A session's *sign* is close to a coin flip: at
σ ≈ 6 BB/hand a 200-hand session has SE ≈ 42 BB/100, so a member hero beats by 10 BB/100 still
loses 41 % of sessions and one that beats hero by 10 wins 59 % of them. Every rate bunches around
0.5, PFSP flattens towards uniform, and §4.4 stops doing anything. Thresholding once per session
discards information that averaging keeps, and discards it irreversibly. Min-max is in exchange
sensitive to one extreme member stretching the denominator; the replacement if that shows up is a
rank, which costs no config either.

**The floor is an independent coin flip per draw.** A deterministic schedule ("every fifth draw")
has the same mean and less variance and was the first implementation, but it aligns with table
structure: at a fixed table size and a floor of 0.25 the floor lands on the *same two slots of
every table*, forever, and the slot index is carried on the token. Uniform table sizes (§4.4)
make the phase drift, so it would not bite today — it is a trap laid for the first fixed-size
diagnostic anybody runs.

Clustering is Lloyd's algorithm with a farthest-point initialisation, written against numpy —
there is no scikit-learn in `requirements.txt` and adding one for k-means would be an aarch64
dependency (`CLAUDE.md` §3) bought for forty lines. It consumes no randomness at all, so the
same vectors always give the same partition; ties go to the lowest index and an empty cluster
keeps its previous centre.

Seats within one table are drawn **independently**, so a member may occupy two seats. Rejecting
that would bias the draw away from small clusters, which is the opposite of what the dedup is
for.

`state_dict` / `load_state_dict` carry the accumulated hands and BB, the cluster labels and the
rng state, so a restart resumes the same stream (§8's resume requirement). A stored state may
cover a **prefix** of the members: §8 appends one agent per iteration, so the sampler that
resumes is one member larger than the one that saved. The new members arrive unplayed, each in a
cluster of its own until the next `set_vectors`; a stored state *larger* than the pool is an
error rather than a truncation.

### 2.4 `nets/` — the tokeniser and entity 4

`features.py` builds the §5.1 token features from played hands and **is** the observation-parity
boundary (`CONCEPT.md` §9): one token per decision, board as of that decision's street, hole
cards only for the observer, no post-decision information, everything monetary in BB.

`tokeniser.py` is the shared tokeniser MLP and `trunk.py` the encoder around it — one class each,
used by the embedding network and by the agent (OI-4, §2.8). Weights are not shared; only the
code is.

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

**Progress.** Every long loop carries a bar, as `CLAUDE.md` §5 requires: corpus generation
(`play:<tag>`, per hand), showdown labelling (`showdown:<tag>`, per hand), training (`train`, per
step) and evaluation (`eval:<tag>`, per fit). The evaluation one is the case the rule is written
for — it is a **single global bar over sessions × observation windows**, not a bar per session,
and a skipped session advances it by the fits it would have contributed so it still reaches its
total. All of them go through `utils.progress`, which pins `smoothing=0` so the ETA is the
average over every completed iteration rather than tqdm's default moving average; that matters
most in the evaluation loop, where a fit over a 1-hand window and one over a 200-hand window
differ in cost by two orders of magnitude.

Bars are written to **stderr**, so they never enter the `Logger` file — which means a run under
`nohup` should redirect stderr somewhere it can be watched, not to `/dev/null`.

`config_g1_pilot.json` is the same experiment shrunk to a throughput probe: same pool shape and
same model, 3 200 corpus hands instead of 96 000 and 500 training steps instead of 20 000. Its
numbers are not results — it exists because every cost figure for the real run is a hypothesis
until it has been run on the Spark (`CLAUDE.md` §3), and this is the cheapest way to replace them
with measurements.

**Three evaluation sets.** `seen` are members the network trained on; `unseen` are fresh style
draws over the same bases (measurement 2); `heldout` are members of bases named in
`corpus.holdout_bases`, which no training session seats at all. The three are a ladder of
distance from the training distribution, and only the third asks the question B1(b) is actually
about: measurement 2 re-draws 32 style scalars over a base the network knows, which is
interpolation inside one parametric family, and Slumbot is not a point in that family.
`split_bases` refuses a partition whose either half cannot seat a 9-handed table — narrowing the
table-size range to make a holdout fit is not an option (`CLAUDE.md` §1) — and refuses a base
name that is not in the pool. The fresh draws are taken off trainable bases only, so §14.2 and
the holdout do not confound each other. Held-out members still occupy rows of the embedding
table; those rows never receive a gradient, which is what makes the `oracle` condition below a
leak check on that set.

**`eval_conditions`.** Measurements that ride along on the same evaluation sessions. Each is a
question the baseline report cannot answer, and each is a switch because each costs another fit:

| switch | what it answers |
|---|---|
| `oracle_embedding` | the trained table row as the fit's **ceiling** — separates "the fit is weak" from "the trunk cannot do better". On a set whose members were never trained it must read ≈ `e = 0`, which makes it a leak check |
| `fit_steps_sweep` | `K`, `fit_lr` and `fit_reg` have never been measured, and §14.1 shows the fit *losing* to the ablation on a one-hand window — the regime hero is in when it sits down |
| `zero_init_fit` | the reverse of §14.3: if fitting from zero ties fitting from the amortised head, the head comes out and takes the second trunk pass of `loss_terms` with it |
| `no_showdown_fit` | whether the §5.1a terms help or hurt the *inference* fit. They are a third the size of the action term in that objective, and their heads may have memorised boards |
| `showdown_holdout` | those heads scored on hands never trained on. A hand's final board is five specific cards and so very nearly a unique key; a training loss far below a held-out loss near `ln 169` means the head answered from the key |
| `save_fitted_vectors` | the vectors themselves, to `fitted_vectors.npz`. With the descriptors' true 32-scalar styles they are what a style-decoding probe reads — the direct form of "does the vector carry style or identity" that CE curves cannot give |
| `condition_windows` | **the cost lever.** Outside it only the §5.5 baseline fit runs |

Cost is linear in window length and in gradient steps, so it concentrates hard: with
`observed_hand_counts` summing to 393 hands, the `n = 200` window alone is 51 % of all fitting,
and any condition measured there roughly doubles the evaluation. The delivered `config_g1.json`
(`condition_windows` = 1, 10, 200; sweep 10 and 200) costs **4.3×** the baseline evaluation per
session and **6.5×** once the third set is counted. `evaluate_sessions` logs the projected fits
and hand-steps per session before it starts, so the bill is visible before it is paid.

`corpus.save_eval_corpus` writes the tokenised evaluation sessions to `eval_corpus.pkl`
(≈ 2.6 KB per hand, so ≈ 0.5 GB at the delivered scale). Everything measured after training is a
function of those tokens and the checkpoint, so with them a changed switch costs an evaluation
rather than a full replay — and replay is only *probably* exact, since pool members are sampled
through GPU forwards whose bitwise reproducibility nobody has promised.

`report["timings"]` carries wall clock per phase. G3 wants those numbers and the gate is where
they are free.

### 2.6 `gates/g1_analysis.py` — section A, reading the report back

`g1.py` writes every measurement row it took into `g1_report.json`, and three questions its
printed tables leave open need no further compute — only a different grouping of those rows.
This module is that grouping, and it runs from the JSON alone: no checkpoint, no pool, not one
replayed hand.

* **A1 — is §14.2's gap composition?** The seen set draws *members*, the unseen set is
  `fresh_style_variants` cycling over *bases*, so a pool whose network bases carry eight style
  variants each and whose degenerate bases carry one to six is mostly networks in one set and
  mostly degenerates in the other. `gain_fit` is broken out per base, and the gap is recomputed
  with the seen set's base mix.
* **A2 — style, or degenerate-vs-network?** The same curve split by base kind. `CONCEPT.md` §11.3
  is the risk that the pool spans fewer styles than members; the network bases are the only ones
  that condition on cards, so their curve is the one that answers it.
* **A3 — is §14.4's table-size skew a skew or a data budget?** The printed breakdown fixes the
  number of observed *hands*, which is a different number of observed tokens per opponent at every
  table size. The same rows re-plotted against tokens per player separate the two.

A2 and A3 report gain on two scales, absolute and as a fraction of each row's own `e = 0` loss,
because the groups being compared differ in headroom as well as in what is being asked about —
a 9-handed table starts near 1.9 nats where a heads-up one starts near 2.9, and on the absolute
scale that difference alone looks like a finding.

Two helpers (`_curve_over`, `_relative`) re-key rows so that `g1._curve` — the single
implementation of §14's aggregation rule, average a session's rows then take the standard error
over sessions — can be reused for a different x-axis and a derived metric. There is deliberately
no second aggregator here.

### 2.7 `gates/g3.py` — what one oracle label costs

```bash
cd versions/v8 && python3 -m gates.g3 --config config_g3.json
```

`CONCEPT.md` §14 (G3) and §13. A realistic pool plays a few hundred hands; a fixed set of hero
decisions out of those hands is then labelled by `oracle/rollout.py` (§2.2c) under a sweep over
the four axes that plausibly move the cost:

| Axis | `config_g3.json` |
|---|---|
| `samples_per_action` | 32, 64, 128, 256 |
| `max_combos` | `null` (the exact posterior), 512, 128 |
| table size | 2, 6, 9 |
| stack depth | 20, 100, 300 BB |

108 cells × `labels_per_cell` labels, one global `tqdm` bar over labels (`CLAUDE.md` §5); a cell
that runs short of decisions advances the bar by the labels it did not take, and the shortfall is
reported rather than hidden. Output is `data/v8/g3/<timestamp>/g3_report.json` — every per-label
row as well as the aggregates — plus the printed table.

Per cell: wall clock per label, forwards per label **split into the posterior's share and the
rollouts' share**, the rollout depth actually seen, the collision rate of the joint draw, and the
labels per hour that follows. The split matters because the two halves scale differently — §13
amortises the posterior over the hero decisions of a hand and does not amortise the rollouts — so
one total cannot say which one to attack. `LabelStats` reports totals, which is all the pipeline
needs; the two columns it cannot give (rollout depth, and the per-sample rewards behind `q`) are
read by `MeasuringDriver`, a `LockstepDriver` that keeps the records of its last `run`. The
oracle's return value was not widened for one experiment.

**The `se_q` column** is the one addition to what §14 asks for, and it is in because the owner
asked for it. The sweep says what a label *costs*; it says nothing about how many samples a label
*needs*, and both are required to fix `oracle.samples_per_action`. It is free: each label's
samples are split in half and the two halves compared, so with equal independent halves
`E[(q_A − q_B)²] = 4·Var(q)` and the reported figure is `sqrt(mean((q_A − q_B)²)) / 2` in BB,
pooled over every label and every legal action of the cell.

Every cell labels the **same** decisions, which is what makes the columns comparable and the
forwards count monotone in the sample budget for a reason other than luck. Hero's seat is played
by the member already sitting in it — at iteration 0 of §8 the oracle improves on a pool member,
so that is the cost of the real thing. G3 is also the one place where table size and stack depth
are pinned instead of sampled: the question is how cost varies along them, which needs cells, not
a uniform draw. Nothing here is training data, so `CLAUDE.md` §1's sampling rule is untouched.

The gate measures and stops. It tunes nothing and it does not try to make the number better; the
decision that follows — variant A as it stands, or §7.4's variant C — is read off the table by
the owner. **It has not been run** (§5).

---

### 2.8 The agent (§6.1) — `nets/trunk.py`, `nets/agent_net.py`, `agent/policy.py`

Entity 3 exists as a network and as a pool member. It is **untrained**: nothing in the tree
builds a target for it or takes a gradient step on it, so its output today is what a random
initialisation says.

**The trunk moved out (D3).** `HandEncoder` in `nets/trunk.py` is the tokeniser, the RoPE
module, the Qwen3 decoder layers and the final norm — exactly the block `OpponentEmbeddingNet`
used to build inline, moved without a change to the computation. Both networks now own one.
OI-4 says the two share this as *code* and not as weights: they read the same §5.1 observation
and run the same attention structure, so a change to either lands in both at once, but they
optimise different objectives on different cadences and each keeps its own parameters.

The extraction renames parameters (`tokeniser.*` → `encoder.tokeniser.*`, and the same for the
layers and the norm), so **the G1 checkpoint no longer loads** (§5). That break was accepted
rather than papered over with a key-remap shim, because the acceptance criterion for the move
was behavioural: the whole pre-existing battery — `test_embedding_net_masking.py` and
`test_inference_fit.py` above all — passes **with no edits to any test**. Had one assertion had
to move, the refactor would have changed behaviour and would have been reverted.

**`AgentNet` is the trunk plus one linear head.** No value head (§6.1, D5) and no search: the
search that produces its targets is the offline BR oracle of §7, not something this forward
does. §7.4's variant C would need a value head, and that is a decision G3 has to inform.

The logits are read at each hand's **own** last real token. The observation of a decision that
has not been taken yet is the hand so far plus one pending token (§9), and that token is last
in the row; rows in a batch have different lengths, so the read is a gather at each row's own
length. Reading a fixed index would answer about padding. The logits come out **unmasked** —
legality is applied once per consumer, from the mask carried on the token, which is the mask
the driver sampled with. A second masking site inside the network is how the two drift apart.

**`AgentPoolMember` is the agent wearing the §4 interface.** Everything that needs an action
asks a `PoolMember`: the driver during self-play, the driver again inside the oracle's
rollouts, and the Slumbot adapter of §12. Wrapping the agent in that interface is what stops
"hero is the agent" from becoming a special case in three call sites, each free to build the
observation slightly differently. One `AgentNet` forward serves a whole lock-step round,
because the driver already groups its queries by member.

One member is **one seat at one table**: `slot_of_seat` and `observer_pos` are fixed at
construction, since the slot decides which fitted vector each seat is conditioned on and the
seat decides whose hole cards the tokens may show. A session that rotates the button
constructs a new member per seat — cheap, since the network and the vectors are held by
reference. Embeddings are a `(max_players, d_emb)` tensor indexed by slot, hero's own vector
among them (§5.3); the cold start is zeros (§5.5).

Three assertions in `logits` guard §9 parity, and they are asserts rather than fallbacks
because every one of them fails silently otherwise — the loss keeps falling while the network
reads something it will not have at deployment:

* the context's acting seat must be the member's own seat;
* `hole_override` is refused — the agent's observation comes from the record's own cards, so a
  hypothetical holding would be ignored and the answer would be about the real hand;
* the pending decision must be the record's latest snapshot, otherwise the tokens would carry
  decisions taken *after* the one being asked about.

The second and third together mean the agent **cannot yet be an opponent in §7.2's posterior**
(see §5).

---

### 2.9 Targets and agent training (§6.2, §8) — `train/targets.py`, `train/agent_train.py`

`policy_target(q, legal, pot_bb, facing_bet_bb, temperature, divisor)` turns one decision's
oracle EVs into the distribution the agent is fitted to, and `kl_loss(logits, target, legal)`
is the loss. `train/agent_train.py` is one training cycle around them.

**The normalisation is the point of the module.** `q / (pot + facing_bet)`, then a softmax at
temperature `T`. Raw EVs in a 300 BB pot and a 10 BB pot differ by more than an order of
magnitude, so a single `T` over raw EVs gives a near-deterministic target in big pots and a
near-uniform one in small ones — v7's 97 % fold rate was a bug of exactly this family
(`versions/v7/ARCHITECTURE.md`, "MCTS value-target normalization"), not an architecture
failure. `test_targets.py` pins scale invariance to `1e-12` and shows what the same numbers do
without the divisor. The divisor is a config choice as §6.2 requires; two are implemented,
`pot_plus_bet` (baseline) and `pot`, which are the only two the signature's inputs can express.

Illegal actions receive **exact** zero, not a small number: they are dropped before the softmax
rather than suppressed inside it. The mask is the environment's own, carried on the token, and
§6.2's "dominated" needs no second rule — `env/legal.py` already drops dominated raise bins
when it builds it.

`kl_loss` is the full KL and not a cross-entropy, so it reads as a distance: it is exactly zero
when the masked prediction is the target. The target is detached — a gradient into it would be
a gradient into the oracle.

**Embedding dropout (§6.2)** zeroes a slot's vector for a whole hand with probability `p`. The
draw is taken on the per-hand `(B, n_slots, d_emb)` table *before* it is read per token, which
is what makes it per hand per slot: a slot the agent is blind to has to be blind for every
token of that hand. This is what makes `e = 0` a usable unconditional policy — the one hero
plays against an opponent it has not observed, including the first hands of a Slumbot session.
§5.4 forbids the same dropout when training the embedding network.

**One cycle, and cycles chain (owner decision 2026-08-18).** `train_agent` takes an `iteration`
and trains the module it is handed, in place. Two consequences, both of which are invisible in
a loss curve if they are got wrong:

* **iteration 0 gets its own step count.** `steps_for_iteration` reads `first_iteration_steps`
  on the first cycle and `steps` on every later one (omitting the key makes them equal). The
  first cycle is the hardest: it starts from a random network, it is the only cycle whose
  labels come from a hero seat the agent did not occupy (§7.1), and it has no policy to
  inherit — so it needs the most gradient steps, and that count is a config key of its own
  rather than a multiplier.
* **the agent at iteration *k* is the network iteration *k−1* produced.** The caller passes the
  same module from one cycle to the next; re-initialising between cycles would discard every
  earlier best response and turn policy iteration into a sequence of unrelated fits. The
  optimiser and the cosine schedule are fresh per cycle — each cycle is a training run against
  a different label set, and carrying a decayed learning rate into it would leave the late
  cycles unable to move.

The loop consumes labels as `(hands, targets, embeddings)`: `HandTokens` each ending in the
pending token of the labelled decision, one target distribution per hand, and one
`(max_players, d_emb)` table per hand — which is what `load_shard` (§2.10) returns. What is
asserted here is
that the last token carries no action, that every target is a distribution, and that no target
puts mass off the environment's mask. The embeddings are an input and are never updated — they
are fitted by the embedding network (§5.5), not by this loss.

---

### 2.10 Label generation (§8) — `env/session.py`, `train/generate.py`

One turn of the outer loop's middle three lines: play sessions, refresh the opponent
embeddings, label hero's decisions with the oracle, write shards. Nothing else — no agent
training (that is `train/agent_train.py`, called by the loop that does not exist yet) and no
pool sampling policy (§4.4 — that is `pool/sampling.py`, and it arrives here as the `sampler`
argument).

**`env/session.py` is G1's session machinery, moved.** `Session`, `build_sessions`, `play` and
`raise_sizes_from` were in `gates/g1.py`; they are now here and G1 imports them back
(`PLAN_PIPELINE.md` D4). Label generation needs the identical semantics — button rotation,
uniform 2–9 × 10–300 BB, slot 0 is the observer — and a copy would let the two drift, with the
drift showing up as a train/deploy mismatch nobody could see. The move is behaviour-neutral and
`test_g1_gate.py` passing unedited is what says so. `play` gained one optional argument,
`bar=False`, for a caller that plays the corpus in several calls and carries its own global bar.
`loss_weights` moved the same way, from `gates/g1.py` to `nets/embedding_net.py`, for the same
reason: the §5.5 fit that runs here has to weight the §5.1a terms exactly as the training that
produced the network did, and production code has no business importing a gate.

**Order is the whole difficulty.** Three things have to agree at every label: the vectors hero
*acted* with, the vectors *attached* to the label, and the vectors hero could have had — fitted
from hands already played, never from the hand being labelled. They agree by construction rather
than by care. A session is played in **blocks of `R` hands** (§5.5's refresh interval); hero's
member for block *b* is built from the vectors fitted over blocks `0 … b−1`; the label of a
decision in block *b* stores that same table. Block 0 is the zero cold start. A vector fitted
over the whole session and attached afterwards would be a future leak that no loss curve could
ever show, which is why `test_label_generation.py` pins it by replaying with the tail of the
session changed and requiring the block's vectors to come out bit-identical.

**The observation is kept, not rebuilt.** Hero's member is wrapped in a recorder that stores
`hand_tokens(..., pending=ctx)` for every decision it is asked about, and that stored prefix is
what goes into the shard. It is therefore the observation hero actually acted on, §9-correct for
the same reason the live one is — as opposed to re-tokenising a truncated copy of the finished
record afterwards, which would be a second construction path for observations and so a place for
the two to drift.

**Hero is one member per seat, twice over.** `agent/policy.py` fixes a member to one seat at one
table, and hero's seat rotates, so hero is `num_players` members per session, rebuilt at every
refresh. Two copies of each: the recorded one plays the sessions, the plain one plays hero inside
the oracle's rollouts, where recording tens of thousands of throw-away hands would be a leak with
nothing reading it. Consequence: hero's *play* forwards do not batch across sessions. That is
cheap and deliberate — a label costs `|A| × samples_per_action` rollout hands against a **single**
hero member, so the rollouts, which are essentially the whole cost, batch exactly as before.

**Hero is not a pool member.** It is built by the `agent_member` argument, which is a factory
`(observer_pos, slot_of_seat, embeddings) -> PoolMember` rather than a member — because of the
per-seat rule above and because the vectors it conditions on change every `R` hands. At iteration
0 the factory ignores both arguments and returns the §7.1 pool member; from iteration 1 it
returns an `AgentPoolMember` over the current network. The sampler is therefore asked for
`num_players − 1` opponents, not for a full table.

**The shards.** `shard_<k>.npz`, written into the directory the caller names, each holding: the
tokenised prefixes concatenated with an offset index (ragged, never padded), the legal mask, `q`
in BB with `nan` off the mask, `pot_bb` and `facing_bet_bb` for the §6.2 normalisation, the
embedding tables **stored once and referenced** (one row per label would dominate the file for no
information at all), and the table metadata. `load_shard` reads them back. The target itself is
not stored: `q` plus the divisor and temperature is strictly more, and it keeps a temperature
sweep from being a regeneration of the corpus.

They are written with a fixed zip timestamp. `np.savez` stamps every entry with the wall clock,
which would make "the same seed produces the same corpus" a claim nobody could check by
comparison; with the stamp fixed the files are byte-identical across runs, and the test asserts
exactly that.

**Progress.** Two sequential bars, never nested: `play` over hands (the driver's own per-call
bars are suppressed, and the block loop advances one global bar instead), then `label` over
**hero decisions across the whole job**, which is the unit that costs and the unit whose ETA
anyone wants (`CLAUDE.md` §5).

**A label whose joint draws all collided** carries `nan` on legal actions and no target can be
built from it, so it is dropped and counted; `n_dropped` is in the manifest, because a large one
means a broken run rather than a rounding detail.

**`train/embed_train.py` is §5.4's training loop, moved out of the gate.** `train_embedding_net`
was in `gates/g1.py`, which made the pipeline depend on an experiment; it is now production code
and G1 imports it back, verified the same way the rest of the move was — `test_g1_gate.py` green
with no edits. Nothing in `env/`, `nets/`, `pool/`, `oracle/`, `agent/` or `train/` imports
`gates/` any more, so the loop's only external dependency is the v7 checkpoints the first cycle
needs (§4.3).

**Hero's `member` index is a placeholder.** A token's `member` field comes from
`spec.seat_members`, and hero's seat holds a temporary member appended past the pool, so hero's
tokens carry an index that is not a row of the embedding table. Nothing reads it today — the
§5.5 fit indexes by `slot` — but the §8 retraining of the embedding network indexes by `member`,
and under the owner decision of 2026-08-19 (`CONCEPT.md` §5.4) hero's tokens have to carry the
agent's reserved row `len(pool₀) + k`. Stamping it is S9's job, and it is the one place where a
wrong index would be read as a different player rather than raised as an error.

**Untested here.** Everything above has run at four sessions of six hands with two rollout
samples per action, on CPU. The memory profile at real scale is a hypothesis: the phase holds
every played record and every stored prefix until the labelling pass ends, which at §13's 10⁵
labels per iteration is order a gigabyte and fine, and at 10⁶ is not — if the loop ever wants
that many, generation has to be chunked, and it is the caller's `n_sessions` that chunks it.

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

**G3 has never been run.** `gates/g3.py` is exercised by `test_g3_gate.py` at a scale that
proves the experiment is the right shape and nothing else: a few hands, two samples per action,
a degenerate pool, on CPU. Every number it prints — seconds per label, labels per hour, the
standard error of `q` — is a property of the Spark and does not exist yet (`CLAUDE.md` §3). Until
it runs, `CONCEPT.md` §13's ~3 s and ~10⁵ forwards per label remain a design figure, and the
choice between variant A and §7.4's variant C has not been informed by any measurement.

**The G1 checkpoint no longer loads** (§2.8, D3). Extracting the trunk renamed every parameter
under it, so a `state_dict` saved before the extraction no longer matches
`OpponentEmbeddingNet`. Nothing on the G1 path needs it — `g1_report.json`, `eval_corpus.pkl`
and `fitted_vectors.npz` are what post-hoc analysis reads — but re-evaluating those weights
would now need a key-remap shim, which does not exist.

**The agent is untrained and cannot answer a hypothetical.** `AgentPoolMember` refuses a
`hole_override` and refuses any decision that is not the record's latest snapshot, so it can
play — through the driver, and through the oracle's rollouts — but it cannot be the opponent
whose range §7.2's posterior is estimating. When the agent enters the pool (§4.4, §8) that
becomes a real gap: it needs an observation builder that truncates a finished record at the
decision being asked about and substitutes the hypothetical cards, which is what
`pool/v7_member.py` does through `up_to` and its deck copy. Until then the assertions make the
gap loud instead of silent.

`config_g1.json` ships its `v7` bootstrap entry with placeholder paths
(`../../data/v7/FILL_ME/…`). They must be filled in before a real G1 run; without them the pool
is degenerate strategies and their style draws only, which weakens the base policies the styles
modulate but does not change what G1 measures.

---

## 6. What is still missing

In roughly the order `CONCEPT.md` §14 says to build it:

| Piece | `CONCEPT.md` | Notes |
|---|---|---|
| `pipeline.py`, `config.json` | §8.1 | the outer loop |
| Slumbot adapter rewrite | §12 | protocol layer survives |

---

## 7. Tests

```bash
cd versions/v8 && python3 -m pytest tests/ -q
```

306 tests, ~58 s on the dev box (CPU-only). The 30-minute budget from `CLAUDE.md` §4 is barely
touched.

| File | Covers |
|---|---|
| `test_engine_conservation.py` | Chip conservation through the engine (from v7) |
| `test_audit_stage0.py` | Engine invariants: `cumulative_bets` monotonicity, betting/street advance (from v7) |
| `test_solver_value_bet.py` | Solver value-bet pot construction (from v7) |
| `test_driver_lockstep.py` | **Lock-step ≡ sequential** (also with a pinned deck and a forced prefix), chip conservation through the driver, the v7 snapshot convention, the max-actions cap, every table size and stack depth, and the legality rule's corner cases |
| `test_rollout_plumbing.py` | **Replay identity**: a recorded hand replayed from its own deck and action sequence reproduces itself element for element, at every table size and both stack extremes; a forced replay issues zero policy calls; the deck override deals exactly what was asked and is refused if it is not a permutation; a partial prefix is replayed and the rest runs free with chips conserved; an illegal forced action raises naming the seat and the mask; the pending token adds exactly one action-less token, leaves every earlier token bit-identical, shows only the observer's cards, and is refused together with a showdown |
| `test_posterior.py` | The opponent posterior: a hand-computed two-decision example to `1e-12`; one batched policy call per opponent decision; card removal relative to the observer (`C(45, 2)` on the river, the opponent's real holding still in the universe); every prefix length normalised; **a card-independent member leaves the prior exactly alone** and a card-dependent one does not; the posterior through *k* is bit-identical on a record truncated at *k*; an opponent who has not acted is the prior; a zero likelihood warns and falls back instead of returning NaN; `max_combos` caps, renormalises and is seeded, is spent **before** any member is asked (32-row batches, not 1081), draws the same combos under two different posteriors, reproduces the full posterior restricted to its draw, and recovers a functional of the full posterior to 0.02 over 200 seeds; `hole_override` changes the cards and nothing else |
| `test_oracle.py` | The BR oracle, every case exact rather than within a Monte-Carlo tolerance: `q[FOLD]` equals hero's own contribution to `1e-12` at every table size 2–9 and both stack extremes; `Q` equals an enumerated posterior-weighted sum on a fixture where hero's payoff is constant on the range's support and different off it; **hero is never handed a card it could not see** over ~900 rollout queries; every rollout conserves chips; illegal actions carry `nan` and the mask is the recorded one; the same seed gives a bit-identical label; the label is unchanged when the record is truncated at the labelled decision; a range made only of board cards collides every time and labels nothing; a heads-up river decision cannot collide; dropped samples reduce the divisor instead of counting as zeros; the forward count is a hand count |
| `test_observation_parity.py` | **The fatal invariant**: only the observer's hole cards, board never ahead of the street, no token carries its own action, scalars from the pre-decision snapshot, prefixes independent of what came later |
| `test_embedding_net_masking.py` | Causal within a hand, block-diagonal across hands, hand order irrelevant, the embedding is what changes the prediction, padding inert, **a showdown token cannot reach back into any decision**, the action loss ignores showdown tokens, both showdown heads reach the embedding, zero weights reduce the objective to action CE |
| `test_inference_fit.py` | The joint fit reaches the loss of the vectors that generated the labels, determinism, `K = 0` is the ablation, network weights untouched, cold start, regularisation, **the showdown terms reach the fitted vector** and zero weights reproduce the action-only fit |
| `test_observation_parity.py` (§5.1a part) | Showdown tokens exist exactly for the revealed seats and never among the decisions, the revealed cards are the target and never an input, the labels match the cards shown, the two masks partition the real tokens, a showdown hand with no labels is refused |
| `test_pool_style.py` | The five categories partition the action set, 32-scalar round trip, identity style is a masked softmax, position and street gating, temperature, uniform mix, every draw is a valid distribution over legal actions, each degenerate strategy does what it says |
| `test_v7_pool_member.py` | The vendored v7 stack constructs and plays legal hands; the v7 event format is built from the acting seat, masked to the street, and stops at its decision; **a `hole_override` reaches the network** — an override naming the real cards reproduces the plain answer, aces and deuce-trey do not, the record is untouched, and end to end a v7 opponent's posterior leaves the prior |
| `test_g1_gate.py` | The gate end to end: button rotation, uniform 2–9 × 10–300 BB, the four report sections, cold start ≡ `e = 0`, and that the standard error's unit is the session |
| `test_g3_gate.py` | The label-cost sweep end to end: one cell per point of the grid, every column the decision is taken on present and finite, forwards and rollouts monotone in the sample budget, the posterior's and the rollouts' shares adding up to the total, the bar reaching its total when a cell runs short of decisions, the split-half error finite, the headline built from the exact-posterior cells only, and a pinned table size that a hand cannot quietly leave |
| `test_agent_net.py` | The agent end to end: one row of logits per hand and every padded position inert; each row answers from **its own** last real token and no hand moves another; through the driver, at every table size 2–9 and both stack extremes, a valid distribution over legal actions and chips conserved; the observation obeys §9 parity along the agent's own call path — only its own cards, board never ahead of the street, no showdown token, the pending token action-less — and the observation does not grow as the record does; with `e = 0` permuting the players is bit-identical and a non-zero vector is not; determinism, weights untouched by a `policy` call, and both parity guards refusing what they are meant to refuse |
| `test_label_generation.py` | Label generation end to end: a toy run whose every label is a valid distribution over the environment's own mask with `nan` exactly off it; hero is slot 0 and every hero decision is labelled once; an ordinary pool member works as hero (iteration 0, §7.1); **the embedding of a block ignores every later hand** — hero jams from hand `R` on and the block's vectors come out bit-identical anyway, while block 0 is the zero cold start; the stored prefix stops at the labelled decision, carries only hero's cards and a board never ahead of the street; the same seed writes byte-identical shards and a shard round-trips; the table draw is the exact uniform multiset of a fixed seed; and the session machinery is shared with G1 rather than copied |
| `test_pool_sampling.py` | Pool sampling: a fixed history and seed produce an exact sequence; results accumulate across sessions of different lengths into a mean and the mean into a weight, with a pool of no results and a pool of no spread both flat, and the **magnitude** of a loss — not its sign — moving the weight; a member hero beats the most is reached **only** through the floor — never at `floor_fraction = 0`, every draw uniform at 1; a member nobody has played is drawn immediately; clustering recovers a hand-built structure and a ten-member blob of near-duplicates does not crowd out a lone style, while a duplicate pair splits one cluster's share; more clusters than members is no clustering; the state round-trips and reproduces the next draw, survives a pool that has since grown by one member and refuses one that has shrunk; and every table size 2–9 gets one member per non-hero seat |
| `test_targets.py` | Targets, loss and the training cycle: a hand-computed softmax to `1e-12`; exact zeros off the mask and what sits under it never read; the two temperature limits reached in float, not approached; one legal action, equal EVs, no legal action, a `nan` under the mask; **the v7 scar** — two situations differing by a factor of 30 give the same target to `1e-12`, and without the divisor one is near-uniform while the other is near-deterministic; the KL is exactly zero on a match, positive off it, blind to illegal logits, and its gradient reaches the logits and not the target; dropout at `p = 0` and `p = 1`, reproducible from its generator, and per hand per slot rather than per token; a toy run that reduces the loss, is deterministic, leaves the pool and the embeddings untouched and refuses a hand that is not a pending decision; and the cycle — the first iteration runs its own step count, a later one opens from the weights the previous one left, and three cycles in a row keep improving |

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
cd versions/v8 && python3 -m gates.g1_analysis \
    --report ../../data/v8/g1/<run>/g1_report.json              # G1 section A
cd versions/v8 && python3 -m gates.g3 --config config_g3.json   # gate G3 (§14)

./run.sh      --version=v8   # → python3 pipeline.py       (does not exist yet)
./evaluate.sh --version=v8   # → python3 eval_pipeline.py  (does not exist yet)
```
