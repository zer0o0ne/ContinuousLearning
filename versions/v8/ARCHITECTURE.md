# v8 — Architecture

**Status: gates G1 and G3 implemented (CONCEPT.md §14), the BR oracle (variant A) with them,
the agent network (§6.1), the target construction and training loop that fits it (§6.2), and
label generation end to end (§8) — play, refresh the embeddings, label hero's decisions, shard —
and the pool sampler that decides who sits at those tables (§4.4).
G3 has run twice on the Spark (2026-08-18, 2026-08-19), so what a label costs is measured rather
than assumed and the decision it exists to inform has been taken: **variant A stands, there is
no value head, and `oracle.samples_per_action = 128`** (`CONCEPT.md` §13, §14). The outer loop
that ties all of it together — `pipeline.py` and `config.json` (§8) — now exists as well, so the
tree is complete from the bootstrap pool to a trained agent joining it. The Slumbot
adapter and the evaluation runner (§12) exist as well, cold and warm, so **the tree is now
complete end to end — bootstrap pool → oracle labels → trained agent → BB/100 against Slumbot.**
Everything below has run only at toy scale on CPU: **no iteration has yet been run at size and
no hand has yet been played against Slumbot**, so nothing here has yet met an opponent stronger
than the −90 BB/100 bootstrap pool.**

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
  config_g3.json        the G3 sweep surface — samples × combos × table × stack × pool
  pipeline.py           the outer loop (§8) — labels, training, gap, pool growth   NEW
  eval_pipeline.py      Slumbot: cold and warm, BB/100 ± SE (§12)                  NEW
  config.json           the experiment surface of a run and its evaluation (§8.1)  NEW

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
    policy.py           the agent as hero, and a past agent as an opponent (§6.1, §7.1, §8)
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
    protocol.py         Slumbot's wire protocol, kept from v7 (§12)             NEW
    v8_adapter.py       the v8 agent as a Slumbot player (§12)                  NEW
  tests/                22 files, 373 tests, ~111 s
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
| `evaluation/protocol.py` | `v7/evaluation/slumbot_eval.py`, the architecture-independent half | The HTTP client and its retry policy, the action-string grammar, token ↔ action-index translation, the action-string replay, BB/100 + SE. Bodies verbatim; see §2.12. The v7-specific half — the event builder, the action chooser, the MCTS and solver paths — was deleted with the file. |

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
   the draw rejected and redrawn if two opponents share a card or a card is already on the
   **visible** board or in hero's hand;
4. one `HandSpec` per (legal action, surviving sample): the visible board and hero's real cards
   in the deck, the sampled cards at the opponents' seats, the streets still to come drawn from
   what the assignment left, the rest filled in ascending order;
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

**Why the runout is dealt and the visible board is not** (`CONCEPT.md` §7.1, owner decision
2026-08-19). The dead set in step 3 is hero's information exactly: `_visible_board(record,
decision_idx)` plus hero's own two cards. A card of a street still to come is **not** dead — it
is in the deck as far as hero knows — so `_rollout_deck` draws the missing board cards per
sample out of whatever the assignment left. The order matters and is the whole argument:
combos first, from `p(· | visible board)`, then the runout from the remainder, which is the
exact factorisation `p(hands | visible) · p(runout | hands)`.

An earlier version pinned the full five-card board of the hand the decision came out of. That
makes `q` an estimate of `E[chips | this exact river]`, an error no `samples_per_action` can
reduce because every sample shares the one runout — and one invisible to the split-half column
that is supposed to size the sample budget. The draw is per **sample**, not per action, so the
common-random-numbers property above is unaffected: the actions of one sample are still compared
on one board.

The residual collision source is now only the one-decision offset: the posterior conditions
`through_decision = decision_idx − 1`, so its dead set is the board visible at the *previous*
decision, while step 3 rejects against the board visible at *this* one. A combo containing a card
that turned over in between still has to die.

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
zero. (These are the numbers before the runout change above, which can only lower the rate:
fewer cards are dead.) So `max_collision_retries` is not a formality at a full ring, and the cost of a full-ring
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

**`v7_member.py` runs the network under autocast.** G3 measured a label's wall clock at roughly
75 % the v7 forward and under 6 % everything Python does to prepare it, so precision is the only
lever the pool side has, and `V7NetworkMember` was running fp32 with no autocast at all while
`utils.get_amp_config` — v8's existing answer to "what does this device want" — sat unused.
`logits` now wraps `agent.action_logits` in `torch.autocast` with what that helper returns: bf16
on CUDA, which is the regime v7 itself trained and evaluated under (v7's own
`slumbot_eval.py` does the same), and **disabled on CPU**, so the dev box keeps running the fp32 path the test
battery pins. Resolved in `__init__` from `agent.device_`, because `build_pool` calls
`set_device` before it constructs a member and never moves one afterwards. Whether bf16 actually
pays on sm_121 is a hypothesis until the Spark says so (`CLAUDE.md` §3); the `.float()` already
at the end of `logits` is there because v7 ran this way too.

**Measured 2026-08-19: about 1.5×, not the 2–4× the change was made on.** The two G3 runs are not
directly comparable — hero, pool mix and hand lengths all changed with them — so the honest
normaliser is the model's cost per unit of the `events` bucket, which is pure Python
proportional to sequence length: it fell from 15.3 to 10.0. The model is still 68.8% of a
label's wall clock, so the remaining engineering lever is the 13.9% in tensor packing, and it is
now a larger share of what is left than it was before autocast landed.

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

**The accumulation forgets (`end_iteration`, `result_decay`, D10).** A procedural member never
changes but hero does, so "hero's BB/100 against member *i*" is a property of a *pair* and a
lifetime mean is a mean over every past agent. Left alone it produces exactly the failure the
mechanism exists to prevent: a member the agents of iterations 1–10 crushed carries a positive
score built on tens of thousands of hands, the ~330 hands per iteration the uniform floor keeps
delivering move it by a few percent, and the loop does not notice it has stopped beating it until
about the sixtieth iteration — past the length of a run. `end_iteration` scales both accumulators
by `result_decay` at the loop's iteration boundary, giving an effective window of roughly
`hands_per_iteration / (1 − result_decay)`; at 0.8 the same member crosses back over zero at the
twelfth iteration and takes the top PFSP weight there. Two properties make this the right shape:
the mean of a member nobody sampled is **unchanged** (numerator and denominator decay together —
no new information, no new estimate), while its effective hand count shrinks, which is exactly
what lets the next session it plays move the estimate. `result_decay = 1.0` is the lifetime
behaviour, kept reachable as the honest way to switch the mechanism off.

The same smearing is why this is not only about recovery: `PLAN_PIPELINE.md` R2 reads
fictitious-play cycling (agent *n* losing to agent *n−2*) off these scores, and a lifetime mean
averages that signal away across every past hero.

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

`state_dict` / `load_state_dict` carry the accumulated (and decayed, hence fractional) hands and
BB, the cluster labels and the rng state, so a restart resumes the same stream (§8's resume requirement). A stored state may
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

**The observer's own showdown token is unmasked** (2026-08-20): its hole slots carry the
observer's own hand, so `nets/features.py` applies one rule to every token type — the observer's
own cards always, everybody else's never. Every *other* seat's showdown token stays masked,
because that masking is the anchor's entire mechanism: no decision token attends to a showdown
token, so the only channel from a reveal to that player's vector is the gradient of a loss whose
answer is not in the input. The cost lands on the metric — `showdown_strength_mse` now pools a
trivial population (the observer's, readable off its own input) with the hard one, so it is no
longer comparable to G1's 0.095 and `showdown_losses` says so in its docstring.

Weights live in the `train` config section (`showdown_strength_weight`, `showdown_class_weight`)
and are applied identically in both places; setting them to zero is the ablation.
`showdown_class_weight` is **0** in `config.json` as of 2026-08-20 — G1 measured the 169-way head
at 5.75 nats held out against a 5.005-nat marginal entropy, i.e. worse than a constant. The code
stays, because zero is the documented ablation and dropping the field would change the shard
format for nothing.

### 2.4b The strength head and the poker prior (§5.6)

The third head on the embedding network's trunk, and the only one that carries no style at all.
On every decision token of the **observer** it predicts the percentile the observer's own hand
reached on that hand's final board; MSE, weight `embedding_net.strength_weight`.

What it is for: the trunk learns hand evaluation, board texture and the value of a draw from a
corpus of free self-play, so that the oracle's EV labels — whose noise at depth is larger than
the EV differences they are meant to teach (§13, §11.4) — do not also have to teach it. §2.11's
`warm_start_trunk` is what hands the result to the agent.

Four properties, all tested in `tests/test_strength_head.py`:

* **one enumeration, two consumers.** `env/showdown.py::label_showdowns` now scores *every* dealt
  seat and `showdown_strength` is that dict restricted to the revealed ones, so §5.1a's labels
  are bit-identical and non-showdown hands are labelled too. `HandRecord.hand_strength` carries
  it; the cost is that every hand pays the enumeration, not only the ~⅓ that show.
* **a target and never an input.** `hand_tokens` writes it into `own_strength` and touches no
  other field; the tokeniser does not read it. Predicting a quantity the observer only learns
  later is what a value target does. `-1` is the sentinel and the mask (a percentile is in
  [0, 1]), so `collate` derives `strength_mask` from it and the padded tail is excluded.
* **no target where there is no final board.** A tokenisation with a pending decision carries
  none — the hand is still in progress — and neither does a record nobody labelled, which is the
  shape the Slumbot replay builds.
* **training only, unlike §5.1a.** `strength_loss` is called from `loss_terms` and deliberately
  not from `objective`, so `fit_embeddings` is bit-identical with and without it. The strength of
  the observer's *own* cards says nothing about anybody's style: its gradient into an opponent's
  vector is noise, and spending K fit steps on it would buy nothing.

**The MSE floor is not zero.** The target is one runout, not an expectation over runouts, so a
perfect head still pays the conditional variance of the runout — large at preflop, zero on the
river. The number to read it against is the marginal variance of the target on the same corpus,
never zero. This is the §11.4 trap in a second place, and the reason `train_embedding_net` logs
the strength-target count: a corpus nobody labelled would otherwise train the head on nothing
while the curve looked ordinary.

**The agent has no showdown head and no strength head.** `AgentNet` carries `action_out` and
nothing else, and `train_agent` computes `kl_loss` / `soft_q_loss` and nothing else. Every
auxiliary weight (`showdown_strength_weight`, `showdown_class_weight`, `strength_weight`) is read
by `loss_weights` alone, whose callers are the embedding trainer, G1, and the two §5.5 fit sites.
The only path from any of them into the agent is `warm_start_trunk`, once, through the weights.

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
the five axes that plausibly move the cost:

| Axis | `config_g3.json` |
|---|---|
| `samples_per_action` | 32, 64, 128, 256 |
| `max_combos` | `null` (the exact posterior), 512, 128 |
| table size | 2, 6, 9 |
| stack depth | 20, 100, 300 BB |
| `pool_kinds` | `all`, `v7` |

216 cells × `labels_per_cell` labels, one global `tqdm` bar over labels (`CLAUDE.md` §5); a cell
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

**The `se_q` columns** are the one addition to what §14 asks for, and they are in because the
owner asked for them. The sweep says what a label *costs*; it says nothing about how many
samples a label *needs*, and both are required to fix `oracle.samples_per_action`. They are
free: each label's samples are split in half and the two halves compared, so with equal
independent halves `E[(q_A − q_B)²] = 4·Var(q)` and the reported figure is
`sqrt(mean((q_A − q_B)²)) / 2`, pooled over every label and every legal action of the cell.

Two details of that column are load-bearing and were wrong in the first run:

* **the gaps are stored signed.** `se_q` only needs the squares, but the target is a softmax and
  a softmax is shift-invariant, so what reaches the loss is the noise on the *differences*
  between actions — and a difference of two gaps cannot be recovered once the signs are gone.
  With `half_gap` signed, the contrast error is computable offline from `g3_report.json` without
  replaying a single rollout.
* **the error is reported in pot units as well as BB** (`SE/pot`), because §6.2 divides `Q` by
  `pot + facing_bet` before the target is built: the same 12 BB is fatal in a 3 BB pot and
  irrelevant in a 300 BB one. The divisor is per label, so the normalisation happens per label
  and not on the aggregate.

**The profile block** splits a label's wall clock five ways — `build_v7_events`, tensor packing
in `EventSequenceEmbedder._build_batch_tensors`, the model itself, the style-and-legality last
mile, and everything left over as driver and engine — and reports the rows per policy call and
the share of queries that reached a network at all. `Profile` and the `profiling()` context
manager live in the gate and wrap the production paths rather than instrumenting them: a timer
inside `PoolMember.policy` would be exactly the "while I was there" addition `CLAUDE.md` §5
forbids, and the gate is the only caller that wants the numbers. The wrapping is entered once
per label, so a row can never carry a profile it did not measure, and the five buckets sum to
the label's wall clock by construction (`test_g3_gate.py` pins that).

Every cell labels the **same** decisions, which is what makes the columns comparable and the
forwards count monotone in the sample budget for a reason other than luck. **Hero's seat is
played by `sweep.hero_kind`, not by whoever the hand seated** — the first run took the seated
member, which on a pool that is 29 % degenerate meant roughly every third label had
`always_fold` or `maniac` as its rollout policy, and §8 never puts either there. `pool_kinds`
exists for the mirror image of that problem: §4.4 samples opponents by PFSP with a uniform
floor, while `build_hands` samples uniformly, so cost and noise were both measured against a
table the pipeline will not set. G3 is also the one place where table size and stack depth are
pinned instead of sampled: the question is how cost varies along them, which needs cells, not a
uniform draw. Nothing here is training data, so `CLAUDE.md` §1's sampling rule is untouched.

The gate measures and stops. It tunes nothing and it does not try to make the number better; the
decision that follows — variant A as it stands, or §7.4's variant C — is read off the table by
the owner, and on 2026-08-19 it was: **variant A stands.** **The gate has been run twice on the
Spark** (2026-08-18 and 2026-08-19); `CONCEPT.md` §13 records both, and the changes above plus the §7.1 runout fix and bf16 in the pool member are why
the second is not comparable to the first.

**The answer, for whoever writes `config.json`: `oracle.samples_per_action = 128`.** The
derivation is worth carrying here because it is a property of the two columns this gate prints
rather than of anything downstream. A label costs `a + b·n` with `a` the posterior, which does
not move with `n`; for a fixed second-budget the total label noise goes as `(a/n + b)/C`, so the
efficiency of a sample budget is *exactly* the share of the label's wall clock spent on rollouts
rather than on the posterior — 68% at n=32, 77% at 64, 85% at 128, 91% at 256. Read as samples
bought per Spark-hour the curve peaks at 128 and turns slightly down after it. Below that the
label re-derives a posterior it does not use enough; above it, nothing is bought.

Second-run headline: `SE ~ n^-0.5` with no plateau (median exponent −0.51 in BB, −0.46 in pot
units over the 18 cells), which is the direct evidence that the pinned board of the earlier
`_rollout_deck` had been an irreducible floor and is not one any more. Wall clock splits 6.9%
events, 13.9% packing, 68.8% model, 0.3% style, 10.3% driver at 584 µs per policy row.

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
does. §7.4's variant C would have needed a value head; G3 was run to inform that decision and the
owner took it on 2026-08-19 — **variant A stands, D5 does not flip, no head is added.** The
noise problem C was the candidate answer to was solved in the loss instead (§2.9's `soft_q`).

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

The second and third together mean this member cannot be an opponent in §7.2's posterior — and
that is what the second class exists for.

**`FrozenAgentMember` — a past agent, seated as an opponent (D12, §8's last line).** §8 appends
the agent to the pool at the end of every iteration, and from then on it is asked the two
questions every pool member is asked: *what do you do here* (the driver, in play and inside the
oracle's rollouts) and *what would you have done holding this* (the §7.2 posterior). Three things
follow, and each one is why this is a second class rather than a flag on the first:

* **The vectors are zero** — D12 option (a), the plan's recommendation, taken 2026-08-19. A past
  agent plays its *unconditional* policy, the one §6.2's embedding dropout trains explicitly, so
  it is a fixed policy like every other member: no fit nested inside a fit, no recursion, no
  answer needed to "what did agent *k−3* believe about its tablemates". The alternative — a §5.5
  fit per past agent per table — has no cheap form and no measurement asking for it.
* **One member serves every seat**, because with `e = 0` the slot only selects which zero vector
  is read. That is what `PoolMember` requires and what `AgentPoolMember` cannot give.
* **The observation is rebuilt for the moment being asked about**: the record is truncated at the
  asked-about snapshot and the hypothetical holding is swapped into a copy of the deck. This is
  the same pair of moves `pool/v7_member.py` makes for the same two reasons, and it shares
  `_deck_seen_by` with it rather than repeating it. On the hot path — the driver, acting now,
  real cards — no copy is made at all.

The style layer applies to it exactly as to a v7 checkpoint (§4.2), so the `agent_variants`
members one agent contributes are `with_style` siblings sharing one network by reference (D11).

---

### 2.9 Targets and agent training (§6.2, §8) — `train/targets.py`, `train/agent_train.py`

`normalised_q(q, legal, pot_bb, facing_bet_bb, divisor)` puts one decision's oracle EVs on the
scale of the situation; `policy_target(...)` softmaxes that at temperature `T` into the
distribution the agent is fitted to. Two losses consume them and `agent_train`'s `loss` key
picks one — `kl_loss(logits, target, legal)` or `soft_q_loss(logits, q_norm, legal, T)`.
`train/agent_train.py` is one training cycle around them.

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

**`soft_q_loss` — the same optimum, linear in the labels** (`CONCEPT.md` §6.2, owner decision
2026-08-19). It computes

```
−⟨π_θ, Q̂ₙ⟩ + T·Σ π_θ log π_θ + T·log Σ_legal exp(Q̂ₙ/T)   =   T·KL(π_θ ‖ softmax(Q̂ₙ/T))
```

`Q̂ₙ` — a `normalised_q` row, exact zeros off `legal` — enters the first term **linearly** and
the third term does not involve `θ` at all, so `E_ε[∇_θ L] = ∇_θ L|_{Q̂=Q}`: Monte-Carlo error in
the labels stays variance instead of becoming target bias, which is what it becomes when the
labels are pushed through a softmax first. The third term is added only so the printed number is
a KL — non-negative, zero at the optimum, readable the way `kl_loss` is — and it shifts the value
without touching the gradient.

`test_targets.py` pins the property rather than the formula: averaging the gradients at `Q + ε`
and `Q − ε` reproduces the gradient at `Q` to `1e-12` for `soft_q` and demonstrably does not for
`kl`. It also pins that adding a constant to every legal EV changes neither the value nor the
gradient — the same shift invariance the softmax has, and the reason the common part of the
oracle's noise, the part §2.2c's common random numbers already share across actions, cannot
reach the gradient at all.

Note the direction: this is `KL(π ‖ target)`, the reverse of `kl_loss`. Same minimiser, but
mode-seeking where the network cannot reach it, which is why both losses exist rather than one
replacing the other. Under `soft_q` the history's `kl` key is `T·KL`, so the two losses' curves
are not comparable in absolute size.

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

### 2.11 The outer loop (§8) — `pipeline.py`, `config.json`

The file that turns every piece above into a training run. Five phases per iteration, and only
the last two are new code:

| | Phase | Does | Artefact |
|---|---|---|---|
| A | embedding network | plays a corpus of pool self-play and runs §5.4 training over it, every `embedding_net.retrain_every` iterations | `embedding.pt` |
| B | labels | `train/generate.py` — play, refresh the vectors, label hero's decisions | `labels/*.npz`, `labels.json` |
| C | agent | `train/agent_train.py` over all but `agent_train.heldout_fraction` of them | `agent.pt` |
| D | oracle gap | §8's four numbers on the held-out slice | `metrics.json` |
| E | close | PFSP results in, `result_decay` applied, the agent appended to the pool | `state.json` |

**The agent's trunk is warm-started from phase A** (§6.1 OI-4, §5.6), at iteration 0 only, when
`agent_train.warm_start_trunk` is set. `pipeline.py::warm_start_trunk` copies
`OpponentEmbeddingNet.encoder`'s state dict into `AgentNet.encoder` — they are built from the
same `embedding_net` config over the same tokeniser and the same §5.1 token, which is exactly
what OI-4 bought by sharing the trunk as code, so the two state dicts correspond key for key. The
action head keeps its initialisation. It is an initialisation and not a tie: the weights are
separate objects and the agent's trunk moves under `soft_q` from its first gradient step. It runs
after phase A because there has to be a trained trunk to copy, and it is skipped when this
iteration already has an `agent.pt` on disk, because a resumed run loads that one instead.

**Iteration 0 differs in exactly one way** (§7.1): the agent trains from scratch and the
`agent_init` member — built through `build_pool` so it is an ordinary member, and deliberately
*not* part of `bootstrap` — sits in hero's seat, so the labelled states come from a competent
policy rather than a random walk. From iteration 1 hero is the agent and the labels are
on-policy. Training continues across iterations; only iteration 0 starts from random weights and
gets its own `first_iteration_steps`.

**The embedding network is the first phase, not a later one.** §8 says it "can be trained before
any v8 agent exists, on hands played by pool members among themselves — that is both gate G1 and
the first pipeline phase", so it runs at the *start* of an iteration: iteration 0's labels carry
the vectors it fitted, and a network still at its initialisation would attach noise to every one
of them. **Its corpus is pool self-play, replayed at each retrain over the pool as it stands** —
which is what grows, since iteration *k*'s pool contains the agents of iterations 0 … *k*−1 as
opponents, the case §5.4 cares about. Replaying rather than accumulating is a deliberate reading
of "the enlarged history corpus": it keeps a resumed run identical to an uninterrupted one
without carrying every hand ever played on disk, and it costs hero's own hands never entering the
corpus. Whether the network should also see hero's tokens — which is what §5.4's "the agent's row
belongs to it from the moment it is first seated as hero" points at — is not settled by anything
measured, and the row still exists and is still read the moment that agent is seated as an
opponent.

**Row *i* of the embedding table is pool member *i*.** The table is sized
`len(pool₀) + max_iterations × agent_variants` up front (D9) and the pool grows by exactly
`agent_variants` members per iteration (D11), so the two indices coincide by construction and
iteration *k*'s block starts at `len(pool₀) + k × agent_variants`. A row nobody occupies yet is a
dead parameter at its initialisation, because no token carries its index.

**The oracle gap** (`gap_terms`, `oracle_gap`) is §8's four numbers — `kl`, `ev_gap_target`,
`ev_gap_greedy`, `agreement` — on the held-out slice, overall and grouped by table size and by
stack depth. One agent forward per held-out decision and **no new rollouts**: the oracle's answer
is already in the shard. What it answers is whether this iteration's training absorbed this
iteration's labels; what it does not is anything about exploitability, and its floor is G3's
Monte-Carlo error rather than zero. `heldout_fraction = 0` reports no gap at all rather than one
measured on the data the optimiser just saw.

**PFSP is fed from the hands the label phase already played.** `train/generate.py`'s manifest now
carries hero's BB and hand count against every member it sat with, and phase E hands them to
`PoolSampler.update` before `end_iteration` ages them (D10). A hand is credited to **every**
opponent at the table in full: hero's chip delta in a multiway hand is not divisible between the
opponents who produced it, and splitting it by table size would make a nine-handed beating look
an eighth as bad as the heads-up one it is being compared against. The cost is that the
nine-handed number carries eight opponents' worth of noise, which is what `result_decay` keeps
from accumulating.

**Resume is per phase, and inside labelling per shard.** `./run.sh --version=v8` after a crash or
a stop continues where it left off: every phase writes its artefact before the next one starts and
skips itself if that artefact is already there, and phase B — the one measured in days — writes a
`labels/progress.json` after every flushed shard as well, so a crash in its middle costs at most
one shard rather than the phase (§2.10). What a mid-labelling resume does re-do is playing the
iteration's hands and refitting the embeddings — minutes against days, and the price of not
keeping a corpus of records on disk. What makes the resumed run *identical*
rather than merely valid is that no phase reads a running RNG: every stream is seeded from
`(seed, iteration)`, and the one piece of genuinely sequential state — the sampler, whose draws
are consumed in phase B — is written out with phase B's own artefact and restored from it. One
iteration owns one block of a million hand seeds, split in half between the corpus and the
labelled sessions, and both halves are asserted to fit rather than assumed to.

**One temperature.** §6.2's `T` builds the `kl` target and is the unit the `soft_q` loss is
written in; it lives once, in the `oracle` section (§8.1), and is handed to the trainer rather
than configured twice.

`config.json` carries exactly §8.1's sections. Sizes and paths that belong to the run rather than
to a section — `experiment`, `seed`, `device`, `out_dir`, `n_iterations`, `n_sessions`,
`hands_per_session`, `labels_per_shard`, `driver_batch_size` — are top-level, as they are in
`config_g1.json` and `config_g3.json`. Two readings worth knowing before editing it: the agent's
trunk dimensions come from the `embedding_net` section, because OI-4 shares the trunk as code and
the two must agree on `d_emb` anyway; and `evaluation` is read by `eval_pipeline.py` (S11) and by
nothing here.

### 2.12 Slumbot (§12) — `evaluation/protocol.py`, `evaluation/v8_adapter.py`

v7's `slumbot_eval.py` was split rather than ported. One half of it never knew what an agent
was — the HTTP client and its retry policy, the action-string grammar, the token ↔ action-index
translation, the replay of an action string into a betting state, and the BB/100 accounting —
and that half is `protocol.py`, bodies verbatim. The other half was the v7 event builder, the
action chooser and the MCTS/solver paths, and it was deleted with the file.

**This is plumbing, not specialisation** (`CLAUDE.md` §1, `CONCEPT.md` §10). What is forbidden
and is not here: a Slumbot-specific policy branch, an opponent model keyed on "this is Slumbot",
or an assumption that a table is heads-up or a stack 200 BB. The adapter takes both from `game`
config and **refuses** a table outside `players_range` / `stack_bb_range` rather than clamping —
a clamped table would report a number about a situation the agent never saw under the name of
one it did.

**The chips are the table's, not the abstraction's.** The engine could have replayed the hand
from a pinned deck and a forced prefix (§2.2a), which would be one construction path fewer. It
was not used, and the reason is the whole design of this file: a forced action is an *index*, so
the engine would size the opponent's bets out of *our* raise bins, and Slumbot does not bet in
our bins. The pot and the stacks the agent read would be the nearest bin's. The abstraction is
unavoidable in what hero can *say* (`action_idx_to_incr` snaps and, when it must, clamps); it
must not reach what hero *sees*. So the `HandRecord` is built from the replay with Slumbot's own
chip amounts, and only the action indices on the tokens are abstracted. `test_slumbot_adapter.py`
pins it with a `b250` — a bet that lands on no bin of the test's grid and still reads as a 5 BB
pot.

**One legality rule, one observation builder, one last mile.** The mask comes from
`env.legal.legal_action_mask`, which reads an `env.table.Table` — so the adapter *assigns* one
from the replayed state rather than re-deriving legality (`_table_view`). The tokens come from
`nets.features.hand_tokens`. Logits become a played distribution in `PoolMember.policy`, reached
through the ordinary `AgentPoolMember`. None of the three has a Slumbot branch, and the record
built here is the only second construction path the file introduces.

**The two seat frames meet here and nowhere else.** Slumbot numbers seats `pos 0 = BB`,
`pos 1 = SB` with the SB first preflop; v8's `env/table.py` posts the small blind at seat 0 and,
heads-up, has seat 1 act first postflop (`next_turn`'s `start_pos = 1 if num_players == 2`). The
conventions are exact mirrors, so `v8_seat = 1 - slumbot_pos` on every street and the flip is
applied once, in `_flip`. Hero is **slot 0**, as in every session it was trained on, so the
vector at slot 0 is hero's and slot 1 is the opponent's; the cold start is a table of zeros
(§5.5) and §12's *warm* run replaces it through `set_embeddings`.

**Two indices per decision, and confusing them is silent.** `act` returns the index the agent
*chose* and the index it can be said to have *played* once `action_idx_to_incr` has had its say.
They differ exactly when a clamp fires — a fold with nothing to call leaves as a check — and the
caller must append the *effective* one to `hero_action_indices`, or every later observation in
the hand carries an action hero did not take. `effective_action_idx` recovers it from the token
that actually went on the wire.

Two things are recorded rather than solved. A wire bet can map to an index our own legality rule
would not have offered — the opponent really did bet that much, so the token carries it, and the
`legal_mask` beside it can disagree; that is what an abstraction gap looks like from the inside.
And `_table_view` maps `last_raise_size` straight from Slumbot's own min-raise quantity while
making `_last_full_raise_level` inert, which is exact heads-up because a short all-in there makes
every other live player all-in and `legal_actions` already drops every raise on that branch.

### 2.13 The Slumbot run (§12) — `eval_pipeline.py`

The only number in this project measured against something that is not ours. It plays the hands,
keeps the books and writes the report; `evaluation/` does the talking.

**Two numbers, always both.** *Cold* pins the embedding to zero for the whole run — the
unconditional policy, the one §6.2's embedding dropout trains explicitly and the one §11.2 says
cannot be a lookup table. *Warm* fits the vector online through the generic §5.5 mechanism, with
the same `K`, `fit_lr`, `fit_reg` and `R` the pipeline uses against every other opponent, which is
why `fit_vectors` is four lines of composition rather than a mechanism of its own — and why
`CONCEPT.md` §10 records it as adaptation and not specialisation.

**"How many hands it took to warm up"** is not a number the run can read off directly, so it is
derived and the derivation is written down: every warm hand records how many hands its vector had
been fitted from, the run is bucketed by that count (`evaluation.warmup_buckets`), and
`warmup_hands` is the first bucket whose BB/100 reaches the cold run's overall BB/100. `None` —
never caught up — is a result, not a missing value, and §11.2's R5 already predicts its shape:
the fit was measurably *worse* than `e = 0` for members observed for two decisions or fewer.

**If warm is worse than cold, the report says so in §12's own words** — "the exploitation
mechanism is a net negative … a result to report, not a bug to tune away".

**Reporting discipline is code, not habit.** The report always carries hand count, BB/100, its
standard error, and §12's selection disclosure — how many candidates were screened over how many
hands each — and `build_report` **refuses to produce a report without it**, because without it the
headline reads as an unbiased measurement when it is the maximum of several noisy ones. A run
below `evaluation.min_reportable_hands` is stamped `SCREENING ONLY` in the JSON and in the printed
header (`CLAUDE.md` §1).

**Parallel, because a million hands is a million HTTP round trips.** A hand is dominated by
network latency, not by the forward, so `evaluation.n_workers` **processes** each play their own
share. Processes and not threads for the reason v7's evaluation was multiprocess as well: the GIL
and a single CUDA stream make threaded inference on batch-of-one forwards slower than the serial
path. Three properties follow from how the split is drawn:

* **Each worker is a session in §5.5's sense** — it sits down at its own Slumbot table, observes
  the same opponent from scratch and fits its own vector from its own window. That is the quantity
  §12's warm-up curve is about, and it means there is no shared state to make the fit
  irreproducible.
* **Each worker owns its own file** (`<mode>_w<k>.jsonl`) and its own share of the hand count, so a
  resume is per worker and needs no coordination.
* **The parent holds no network and touches no GPU** while workers run: the checkpoints are loaded
  inside each worker from their paths, `spawn` is the start method, and the parent only drains the
  queue, moves the one bar and adds the files up at the end. The refusals that belong to the run
  rather than to a worker — a missing checkpoint, warm with no embedding network — are still made
  in the parent, because learning about them one process deep is worse.

**Resumable at the hand boundary.** Every completed hand is appended to its worker's file before
the next one starts; a restart replays that file into the accumulators, rebuilds the fit window
from its tail and carries on. The agent's own generator is seeded per hand from
`(seed, mode, worker, hand index)` rather than carried, so a resumed run draws what an
uninterrupted one would have. **A hand the wire loses is written too, marked failed**, so the index
does not shift underneath a resume; failed hands are counted in the report and never enter the
statistics — a run that is quietly failing cannot look like a clean one. **The clamp counters ride
on each hand rather than being tallied in memory**, so a resumed run's totals are the run's totals
and not what happened since the restart; every reported number is added up from disk by
`aggregate` for the same reason.

Two things are bounded rather than exact and both are recorded. The §5.5 fit is over the most
recent `evaluation.fit_window` hands rather than "the histories observed so far", because an
unbounded fit over a million hands is not computable; the window is config and is set well past
the range G1 measured. And the standard error is accumulated by Welford's form through
`protocol.stderr_bb_per_100_online`, so a million-hand run keeps no list of hands — the formula
still lives in one place.

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
  the `_build_events` of the inherited `slumbot_eval.py` before that file was deleted (§2.12).
  Events are
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

**No hand has been played against Slumbot.** `evaluation/` and `eval_pipeline.py` import and are
tested against a canned Slumbot that speaks the real grammar, but **nothing in the battery opens
a socket**. The retry policy, the live grammar and the server's actual responses are first
exercised by a screening run on the Spark, and that run is also the first evidence that any of
this talks to Slumbot at all.

**The v7 pool member is not covered by the battery with real weights.** `data/v7/` does not
exist on the dev box and there is no GPU (`CLAUDE.md` §3), so `test_v7_pool_member.py` exercises
a randomly initialised network: it verifies that the vendored stack imports and runs under the
installed `transformers`, that the event format is built correctly, and that a v7 member obeys
the observation-parity and legality contracts. It says nothing about a trained checkpoint. Real
checkpoints have loaded and played on the Spark since 2026-08-18 (seven of them, 79 members,
through G3) and the bf16 path of §2.3 has run there since 2026-08-19, but nothing in the battery
sees either: on CPU `get_amp_config` disables autocast, which is deliberate — the dev box keeps
testing the fp32 path — and means the battery is silent about the path production uses.

**Nothing in this tree has been run against a strong opponent.** The bootstrap pool is v7
checkpoints, and v7 plays about **−90 BB/100** against Slumbot (`CONCEPT.md` §4.1). The oracle is
not wrong because of it — `Q` is the EV against whatever pool it is given — but every posterior
in `oracle/posterior.py` is a weak strategy's range, so any conclusion that depends on the
*shape* of a range does not transfer, and the first §12 evaluation is the first time anything
here meets a strategy worth the name.

**G3 has run twice on the Spark**, 2026-08-18 and 2026-08-19; `CONCEPT.md` §13 records what the
first run measured and §7.4 records the design claim it withdrew. What is still a hypothesis on
this box is everything the gate touches at scale: `test_g3_gate.py` exercises it at a few hands,
two samples per action, a degenerate pool, on CPU, which proves the experiment is the right
shape and nothing else. The second run also changed what is being measured — the §7.1 runout
draw, the configured hero seat, the `pool_kinds` axis and bf16 all landed together — so its
numbers replace the first run's rather than extending them.

**The G1 checkpoint no longer loads** (§2.8, D3). Extracting the trunk renamed every parameter
under it, so a `state_dict` saved before the extraction no longer matches
`OpponentEmbeddingNet`. Nothing on the G1 path needs it — `g1_report.json`, `eval_corpus.pkl`
and `fitted_vectors.npz` are what post-hoc analysis reads — but re-evaluating those weights
would now need a key-remap shim, which does not exist.

**A past agent in the pool plays its unconditional policy, and that is a decision, not a
detail.** `FrozenAgentMember` (§2.11) seats it at `e = 0` — D12 option (a). It is cheap and it
has no recursion, but it means the pool's own agents never *exploit* the tables they sit at,
while hero always does. Whether the loop should instead fit a vector for each past agent (D12
option (b)) is open; nothing has measured the difference, and it would nest one §5.5 fit inside
another for every rollout.

`config_g1.json` ships its `v7` bootstrap entry with placeholder paths
(`../../data/v7/FILL_ME/…`). They must be filled in before a real G1 run; without them the pool
is degenerate strategies and their style draws only, which weakens the base policies the styles
modulate but does not change what G1 measures.

---

## 6. What is still missing

In roughly the order `CONCEPT.md` §14 says to build it:

Nothing. Every piece `CONCEPT.md` describes is on disk (`PLAN_PIPELINE.md` S1–S11 are all done).
What is missing is not code but **runs**: no iteration at size, no hand against Slumbot, and
therefore no number measured against anything but the pool that produced it. §5 lists what that
leaves unverified.

---

## 7. Tests

```bash
cd versions/v8 && python3 -m pytest tests/ -q
```

388 tests, ~118 s on the dev box (CPU-only) — the parallel-evaluation cases spawn processes and
account for most of the increase. The 30-minute budget from `CLAUDE.md` §4 is barely touched.

| File | Covers |
|---|---|
| `test_engine_conservation.py` | Chip conservation through the engine (from v7) |
| `test_audit_stage0.py` | Engine invariants: `cumulative_bets` monotonicity, betting/street advance (from v7) |
| `test_solver_value_bet.py` | Solver value-bet pot construction (from v7) |
| `test_driver_lockstep.py` | **Lock-step ≡ sequential** (also with a pinned deck and a forced prefix), chip conservation through the driver, the v7 snapshot convention, the max-actions cap, every table size and stack depth, and the legality rule's corner cases |
| `test_rollout_plumbing.py` | **Replay identity**: a recorded hand replayed from its own deck and action sequence reproduces itself element for element, at every table size and both stack extremes; a forced replay issues zero policy calls; the deck override deals exactly what was asked and is refused if it is not a permutation; a partial prefix is replayed and the rest runs free with chips conserved; an illegal forced action raises naming the seat and the mask; the pending token adds exactly one action-less token, leaves every earlier token bit-identical, shows only the observer's cards, and is refused together with a showdown |
| `test_posterior.py` | The opponent posterior: a hand-computed two-decision example to `1e-12`; one batched policy call per opponent decision; card removal relative to the observer (`C(45, 2)` on the river, the opponent's real holding still in the universe); every prefix length normalised; **a card-independent member leaves the prior exactly alone** and a card-dependent one does not; the posterior through *k* is bit-identical on a record truncated at *k*; an opponent who has not acted is the prior; a zero likelihood warns and falls back instead of returning NaN; `max_combos` caps, renormalises and is seeded, is spent **before** any member is asked (32-row batches, not 1081), draws the same combos under two different posteriors, reproduces the full posterior restricted to its draw, and recovers a functional of the full posterior to 0.02 over 200 seeds; `hole_override` changes the cards and nothing else |
| `test_oracle.py` | The BR oracle, every case exact rather than within a Monte-Carlo tolerance: `q[FOLD]` equals hero's own contribution to `1e-12` at every table size 2–9 and both stack extremes; `Q` equals an enumerated posterior-weighted sum on a fixture where hero's payoff is constant on the range's support and different off it; **hero is never handed a card it could not see** over ~900 rollout queries; every rollout conserves chips; illegal actions carry `nan` and the mask is the recorded one; the same seed gives a bit-identical label; the label is unchanged when the record is truncated at the labelled decision; **the runout is dealt per sample and the visible board is not** — every rollout replays the flop, the turn and river differ between samples, and no opponent is ever handed a card off the visible board or out of hero's hand; eight ranges inside three cards make every joint draw collide by pigeonhole and the label is `nan`; a heads-up river decision cannot collide; dropped samples reduce the divisor instead of counting as zeros; the forward count is a hand count |
| `test_observation_parity.py` | **The fatal invariant**: only the observer's hole cards, board never ahead of the street, no token carries its own action, scalars from the pre-decision snapshot, prefixes independent of what came later |
| `test_embedding_net_masking.py` | Causal within a hand, block-diagonal across hands, hand order irrelevant, the embedding is what changes the prediction, padding inert, **a showdown token cannot reach back into any decision**, the action loss ignores showdown tokens, both showdown heads reach the embedding, zero weights reduce the objective to action CE |
| `test_inference_fit.py` | The joint fit reaches the loss of the vectors that generated the labels, determinism, `K = 0` is the ablation, network weights untouched, cold start, regularisation, **the showdown terms reach the fitted vector** and zero weights reproduce the action-only fit |
| `test_observation_parity.py` (§5.1a part) | Showdown tokens exist exactly for the revealed seats and never among the decisions; **another seat's** revealed cards are the target and appear nowhere in its token, while **the observer's own showdown token carries the observer's own hand** — one rule across every token type, checked on decision and showdown tokens together; the labels match the cards shown, the two masks partition the real tokens, a showdown hand with no labels is refused |
| `test_strength_head.py` | The §5.6 poker prior and the §6.1 warm start: the target is the observer's own percentile on the final board, on the observer's own decision tokens and `-1` everywhere else, matching an independently enumerated value to `1e-12`; §5.1a's showdown labels are that same dict restricted to the revealed seats, so they did not move when the two passes merged; **a hand still in progress carries no target** and neither does a record nobody labelled; **it is a target and not an input** — the same hands tokenised with and without the label differ in `own_strength` alone and the action logits are bit-identical; `collate` masks the padded tail through the sentinel and selects exactly the labelled tokens; the weight shifts the total by exactly its term and 0 removes it while leaving the action CE untouched; a batch with no target is a batch and not an error; the head can actually learn the target, below the variance that is the only baseline it is read against; **the fit never sees the term** — `fit_embeddings` is bit-identical with and without it, and it did move, so that is not two no-ops; `first_retrain_steps` selects the first retrain only, leaves the agent's own key alone, and the trainer runs the count the iteration asks for; and the warm start copies the trunk key for key, leaves the action head at its initialisation, and is an initialisation rather than a tie — one gradient step moves the agent's trunk and not the embedding network's |
| `test_pool_style.py` | The five categories partition the action set, 32-scalar round trip, identity style is a masked softmax, position and street gating, temperature, uniform mix, every draw is a valid distribution over legal actions, each degenerate strategy does what it says |
| `test_v7_pool_member.py` | The vendored v7 stack constructs and plays legal hands; the v7 event format is built from the acting seat, masked to the street, and stops at its decision; **a `hole_override` reaches the network** — an override naming the real cards reproduces the plain answer, aces and deuce-trey do not, the record is untouched, and end to end a v7 opponent's posterior leaves the prior |
| `test_g1_gate.py` | The gate end to end: button rotation, uniform 2–9 × 10–300 BB, the four report sections, cold start ≡ `e = 0`, and that the standard error's unit is the session |
| `test_g3_gate.py` | The label-cost sweep end to end: one cell per point of the grid, every column the decision is taken on present and finite, forwards and rollouts monotone in the sample budget, the posterior's and the rollouts' shares adding up to the total, the bar reaching its total when a cell runs short of decisions, the split-half error finite and its gaps **signed on both sides**, the pot-unit error recomputed per label from the stored rows, the hero seat being the configured member and not whoever sat there, the five profile buckets summing to the label's wall clock with the counters reset per label and the wrapping undone on exit, the headline built from the exact-posterior cells only, and a pinned table size that a hand cannot quietly leave |
| `test_agent_net.py` | The agent end to end: one row of logits per hand and every padded position inert; each row answers from **its own** last real token and no hand moves another; through the driver, at every table size 2–9 and both stack extremes, a valid distribution over legal actions and chips conserved; the observation obeys §9 parity along the agent's own call path — only its own cards, board never ahead of the street, no showdown token, the pending token action-less — and the observation does not grow as the record does; with `e = 0` permuting the players is bit-identical and a non-zero vector is not; determinism, weights untouched by a `policy` call, and both parity guards refusing what they are meant to refuse |
| `test_label_generation.py` | Label generation end to end, including **resume**: a run stopped in the middle of its second shard comes back with the same label set shard for shard and byte for byte, and a progress file written for different sessions is refused rather than spliced; a toy run whose every label is a valid distribution over the environment's own mask with `nan` exactly off it; hero is slot 0 and every hero decision is labelled once; an ordinary pool member works as hero (iteration 0, §7.1); **the embedding of a block ignores every later hand** — hero jams from hand `R` on and the block's vectors come out bit-identical anyway, while block 0 is the zero cold start; the stored prefix stops at the labelled decision, carries only hero's cards and a board never ahead of the street; the same seed writes byte-identical shards and a shard round-trips; the table draw is the exact uniform multiset of a fixed seed; and the session machinery is shared with G1 rather than copied |
| `test_pool_sampling.py` | Pool sampling: a fixed history and seed produce an exact sequence; results accumulate across sessions of different lengths into a mean and the mean into a weight, with a pool of no results and a pool of no spread both flat, and the **magnitude** of a loss — not its sign — moving the weight; a member hero beats the most is reached **only** through the floor — never at `floor_fraction = 0`, every draw uniform at 1; forgetting leaves an unsampled member's estimate exactly where it was and is what lets a member the early agents crushed climb back to the top PFSP weight at all — at `result_decay = 1` it is still winning after forty iterations, at 0.8 it crosses at the twelfth and at 0.5 at the fifth; a member nobody has played is drawn immediately; clustering recovers a hand-built structure and a ten-member blob of near-duplicates does not crowd out a lone style, while a duplicate pair splits one cluster's share; more clusters than members is no clustering; the state round-trips and reproduces the next draw, survives a pool that has since grown by one member and refuses one that has shrunk; and every table size 2–9 gets one member per non-hero seat |
| `test_targets.py` | Targets, loss and the training cycle: a hand-computed softmax to `1e-12`; exact zeros off the mask and what sits under it never read; the two temperature limits reached in float, not approached; one legal action, equal EVs, no legal action, a `nan` under the mask; **the v7 scar** — two situations differing by a factor of 30 give the same target to `1e-12`, and without the divisor one is near-uniform while the other is near-deterministic; the KL is exactly zero on a match, positive off it, blind to illegal logits, and its gradient reaches the logits and not the target; **the linear loss** — the two losses share a minimiser, averaging the gradients at `Q ± ε` reproduces the gradient at `Q` to `1e-12` for `soft_q` and demonstrably not for `kl`, a constant added to every legal EV changes neither value nor gradient, illegal logits are ignored, a label off the mask is refused, and a toy run reaches its target; dropout at `p = 0` and `p = 1`, reproducible from its generator, and per hand per slot rather than per token; a toy run that reduces the loss, is deterministic, leaves the pool and the embeddings untouched and refuses a hand that is not a pending decision; and the cycle — the first iteration runs its own step count, a later one opens from the weights the previous one left, and three cycles in a row keep improving |

| `test_pipeline.py` | The outer loop end to end at toy scale, including that **a crash in the middle of labelling costs a shard and not the phase** — re-running produces the artefacts an uninterrupted run would have left: two iterations run to completion and write every artefact of every phase; the pool grows by exactly `style.agent_variants` members per iteration, variant 0 unmodified and the rest style draws, with the embedding table reserving `len(pool₀) + max_iterations × agent_variants` rows; **iteration 0 seats the `agent_init` member and iteration 1 seats the agent**, asserted from who was actually asked for an action; the four §8 gap numbers computed by hand, including that `ev_gap_greedy` is zero exactly when the agent's mass sits on the oracle's best action; the held-out slice reaches the metric and never the optimiser, and `heldout_fraction = 0` reports **no gap** rather than one on training data; the split is a partition, deterministic in `(seed, iteration)` and different between iterations; **a run resumed from a crash in the middle of an iteration reproduces an uninterrupted one** — the same shards byte for byte, the same weights tensor for tensor, differing only in wall clocks; table size and stack depth span 2–9 and 10–300 BB with no weighting; and a past agent in the pool answers `hole_override` (two holdings, two answers, a posterior that moves off the prior) while observing only the moment it was asked about |

| `test_slumbot_adapter.py` | The Slumbot seam, entirely off canned action strings — no socket: the mask hero acts under **is `env.legal`'s** on every canned state and reaches the token unchanged, folding is offered facing the blind and refused with nothing to call, and an opponent's all-in leaves no raise; **the chips are Slumbot's and not the abstraction's** — a `b250` that lands on no bin of ours still reads as a 5 BB pot, and every canned state's pot, stack and amount-to-call match the wire to 1e-9; the two seat frames are mirrors, hero holds hero's cards, the first decision of a hand belongs to seat 0 and the first of the flop to seat 1; §9 parity on the built record — only hero's hole cards, a board never ahead of the token's street, no showdown token, the pending decision action-less, and a prefix independent of what came later; **index → wire → index round-trips for every legal action on all four streets with no clamp firing**, while a fold with nothing to call becomes a check, says so in the counter, and is read back as a call; hero's own clamped action is what the next replay sees, and `hero_action_indices` overrides hero's seat only; BB/100 and its standard error against hand-computed values, with Welford's online form agreeing and the zero- and one-hand cases returning zero; a table outside `players_range` / `stack_bb_range` is **refused, not clamped**; and end to end the agent answers every canned state with a legal token, deterministically under its own generator, cold from the zero table and warm from a fitted one, reaching the network through the ordinary `AgentPoolMember` and not a copy of it |

| `test_eval_pipeline.py` | The evaluation runner against a canned Slumbot that speaks the real grammar through `evaluation/protocol.py` — no socket anywhere; the parallel path really spawns processes, which reach the stub by name: shares add up to the hand count for every split, each worker plays its own share into its own file and the aggregate is their sum, a parallel run resumes per worker, and the clamp counters survive a resume because they ride on the hands rather than being tallied; BB/100 and its standard error hand-computed, agreeing with the batch form, with the zero- and one-hand cases returning zero rather than crashing; **a short run is stamped `SCREENING ONLY` and a run at the threshold is not**, and the selection disclosure is refused when absent or incomplete rather than quietly omitted; **cold pins `e = 0` at every decision** — asserted from the vector the member actually held — and never fits anything; warm refreshes exactly on the configured `R`, records per hand how many hands its vector was fitted from, buckets the run by it, and a fitted vector demonstrably reaches the policy; the warm-up hand count is the first bucket that caught cold up and `None` when it never did; **warm worse than cold is reported in §12's own words**; **a resumed run reproduces an uninterrupted one** — the same BB/100, the same standard error, the same per-bucket curve and the same hands byte for byte; a failed hand is counted, keeps its slot in the file so the index does not shift, and does not enter the statistics; and both modes off, or warm with no embedding checkpoint, are refused |

`conftest.py` puts `gto_utils/` and the version root on `sys.path` and reseeds
`random`/`numpy`/`torch` to 42 before every test. `tests/g1_fixtures.py` holds the shared toy
pool, hand specs and network config.

---


## 8. Data layout

All data lives outside `versions/` under `data/v8/…` (gitignored). v8 never writes into
`data/v7/`; it only **reads** v7 checkpoints, by path, from config (`CONCEPT.md` §4.3, §8.1).

```
data/v8/
  logs/<timestamp>.txt              one file per process, from `utils.Logger`
  g1/<timestamp>/…                  a G1 run
  g3/<timestamp>/…                  a G3 run
  <experiment>/                     one run of the loop — named, not timestamped,
    report.json                       because resume has to find what it is resuming
    iter_0000/
      labels/shard_0000.npz …       the labels (`train/generate.py`)
      labels/progress.json          how far labelling got; a crash costs one shard
      labels.json                   manifest, per-member results, sampler state after the draws
      embedding.pt                  only on the iterations that retrained it
      agent.pt                      the checkpoint this iteration produced
      metrics.json                  the §8 oracle gap, overall and by table size and stack depth
      state.json                    sampler state after the decay; the phase boundary resume reads
    iter_0001/ …
    slumbot/<run>/                one evaluation (`eval_pipeline.py`)
      cold_w00.jsonl …            one file per worker per mode; a resume reads these
      slumbot_report.json         BB/100 ± SE cold and warm, and the §12 disclosure
```

## 9. Running

```bash
cd versions/v8 && python3 -m gates.g1 --config config_g1.json   # gate G1 (§14)
cd versions/v8 && python3 -m gates.g1_analysis \
    --report ../../data/v8/g1/<run>/g1_report.json              # G1 section A
cd versions/v8 && python3 -m gates.g3 --config config_g3.json   # gate G3 (§14)

./run.sh      --version=v8   # → python3 pipeline.py --config config.json
./evaluate.sh --version=v8   # → python3 eval_pipeline.py --config config.json
./evaluate.sh --version=v8   # → python3 eval_pipeline.py  (does not exist yet)
```
