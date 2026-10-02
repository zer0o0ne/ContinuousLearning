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
The **range head (§5.7)** was added 2026-09-02: the trunk now predicts every live opponent's
range at every decision and feeds that belief back into its own later layers, in both networks.
It is off with one config key and nothing has yet been trained with it on.
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
  config_pc.json        the pool-conditioning gate's surface (P2)                   NEW
  pipeline.py           the outer loop (§8) — labels, training, gap, pool growth   NEW
  eval_pipeline.py      Slumbot: cold and warm, BB/100 ± SE (§12)                  NEW
  regap.py              re-runs phase D over iterations already on disk (§8)       NEW
  config.json           the experiment surface of a run and its evaluation (§8.1)  NEW

  env/                  poker engine, from v7, verbatim
    legal.py            the one legality rule (§6.2)                    NEW
    driver.py           lock-step vectorised driver + rollout plumbing (§3) NEW
    runout.py           the rollouts' control variate (§7.3)             NEW
    session.py          sessions: rotation, uniform table draw, play (§8)    NEW
    showdown.py         reveal detection + the two showdown labels (§5.1a)  NEW
  pool/                 entity 2 — the opponent pool (§4)               NEW
    base.py             PoolMember: logits → styled, legal distribution
    style.py            live style modifiers, 32 scalars per member (§4.2)
    degenerate.py       always-fold / call / min-raise / maniac / nit
    v7_member.py        a vendored v7 checkpoint as a pool member (§4.3)
    action_map.py       nearest-bin transport between two raise grids
    build.py            pool construction from config, fresh style draws
    sampling.py         PFSP + embedding dedup + uniform floor (§4.4)   NEW
    strength.py         per-board hand strength for procedural members (§2.3a) NEW
    situation.py        a decision as scalars + preflop-implied ranges (§2.3b) NEW
    stats.py            VPIP / PFR / c-bet / … over played records (§2.3b)   NEW
    regular.py          the human-shaped rule cascade (§2.3c)                NEW
    archetypes.py       ten presets, jitter, and their domains (§2.3d)      NEW
  nets/                 v8's own networks                               NEW
    features.py         §5.1 token features — where observation parity lives
    tokeniser.py        the shared tokeniser MLP (§5.1, OI-4)
    trunk.py            HandEncoder — the §5.1/§5.2 trunk, shared as code    NEW
    range_head.py       §5.7 — the belief module inside the trunk             NEW
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
    ranges.py           the same belief as a filter over a whole hand (§5.7) NEW
    rollout.py          variant A: Q(s, ·) by full rollout (§7.1)
    parallel.py         N CPU label workers + one GPU inference server
    transport.py        the shared-memory slab those two talk over
  gates/
    g1.py               the G1 experiment (§14)                         NEW
    g1_analysis.py      section A — regrouping a finished g1_report.json NEW
    g3.py               what one oracle label costs (§14, §13)           NEW
    pool_conditioning.py  is a past agent stronger reading its tablemates? NEW
    pool_realism.py     are the archetypes diverse, and better than the five? NEW
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
  tests/                31 files, 605 tests, ~317 s
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

**Streets may define different numbers of raise sizes.** `game.raise_sizes` is four lists and
they need not be the same length: the **widest** street fixes the action layout — `n_raise_bins`,
and with it the all-in slot and `n_actions = n_raise_bins + 3` — and a street that lists fewer
sizes simply has its trailing bins illegal, which is how every other unavailable action is
already expressed. Nothing downstream of the mask has to know that the grids differ; the
network's output width, the one-hot and the Slumbot adapter's `n_actions − 3` all key off the
widest street. Before this, the bin count was read off the preflop list alone and every street
was walked to that length, so a shorter later street raised `IndexError` on its first decision
and a longer one was silently truncated.

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

**The exact path is incremental across a hand.** `PosteriorCache` keeps the last range for each
live opponent while consecutive hero decisions of one record are labelled. When the board grows,
it removes combos containing the newly visible cards; then it multiplies only likelihoods of
opponent actions after the cached prefix. This is Bayes filtering, not an approximation: a test
compares every cached prefix with a fresh full computation, including all street transitions.
The cache owns only the current record, so memory is bounded by one range per opponent, and
`LabelStats.forwards` counts only policy rows actually evaluated. It is disabled whenever
`max_combos` is set, because that mode deliberately draws a new per-label prior subsample and
reusing one would change both the estimator and the label RNG stream.

Each returned range is a normalised reach factor, including a seat's fold if
observed. For fixed policies and hand-start memory the joint density is the
product of these factors restricted to disjoint cards compatible with hero's
information. Rejection samples that joint exactly; the factors themselves are
not its full marginal distributions. Floors and combo caps remain approximations.

### 2.2c `oracle/rollout.py` — variant A, `Q(s, ·)` by full rollout

`CONCEPT.md` §7.1. `action_values(record, decision_idx, driver, pool, hero_member_idx, cfg, rng)`
returns `(q, legal, stats)`: `q` is `(n_actions,)` float64 in **big blinds**, `nan` at every
illegal action, `legal` is the recorded mask, `stats` is a `LabelStats`. `OracleConfig` carries
the five knobs (`samples_per_action`, `max_combos`, `likelihood_floor`, `batch_hands`,
`max_collision_retries`) and every one of them trades cost against noise — plus
`control_variate` and `runout_samples`, which buy noise down at four to six times the rate a
sample does (§2.2d).

One label is:

1. the recorded `legal_mask` — illegal actions are never rolled out, and `nan` rather than zero
   makes a downstream masking bug fail loudly instead of averaging in a value nobody computed;
2. one `opponent_posterior(..., through_decision=decision_idx − 1)` per opponent,
   **including folded seats and their fold likelihoods**;
3. one **joint sample** — each opponent's combo proposed independently from its reach factor, and
   the draw rejected and redrawn if two opponents share a card or a card is already on the
   **visible** board or in hero's hand;
4. one `HandSpec` per (legal action, surviving sample): the visible board and hero's real cards
   in the deck, the sampled cards at the opponents' seats, the streets still to come drawn from
   what the assignment left, and every remaining card **dealt** — shuffled, not sorted — into
   the undealt stub;
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

In heads-up the residual collision source is the one-decision offset: the posterior conditions
`through_decision = decision_idx − 1`, so its dead set is the board visible at the *previous*
decision, while step 3 rejects against the board visible at *this* one. A combo containing a card
that turned over in between still has to die.

**Fold conditioning (2026-09-08).** Folded cards are no longer random filler.
The fold and every earlier observed action weight that seat's possible holdings,
which then block both live opponents' cards and the runout. This restores card
bunching. The forced prefix keeps those players folded, and sampled cards never
enter hero's observation. The control variate conditions its board completions
on the same full assignment, including folded seats. Tests use a policy that
folds only AA: every rollout must assign it AA and exclude those two aces from
the live hand and board, with or without variance reduction. Cached ranges are
checked against fresh conditioning after later board reveals.

**Fold is not special-cased.** Hero's chip delta after folding is minus what hero has already
put in, whatever the opponents hold, so the rollout returns the closed form with zero variance —
and `test_oracle.py` checks it against that closed form at every table size from 2 to 9 and at
both stack extremes.

**`LabelStats` — what S4/G3 reads, and nothing else.** `forwards` counts policy *rows*
(posterior rows + rollout decisions that were not forced), `seconds` is wall clock,
`n_rollouts` is hands played, and `collision_rate` is the fraction of joint draws rejected.
That last one measures rejection cost, not bias. It is large: measured on a **9-handed**
preflop decision with eight live opponents, **~98.5 % of draws are rejected** — 16 cards drawn
independently from one 50-card deck almost always repeat. Heads-up on the river it is exactly
zero. (These are the numbers before the runout change above, which can only lower the rate:
fewer cards are dead.) So `max_collision_retries` is not a formality at a full ring, and the cost of a full-ring
label is roughly `1 / P(accept)` draws per usable sample. This is a CPU measurement of the
sampler, not of the GPU cost — G3 (S4) is what measures the label.

**Not built here** (S3's non-goals): variant C and value bootstrapping. Dataset writing belongs
to S7. The originally deferred posterior cache is now implemented in §2.2b.

### 2.2d `env/runout.py` — the control variate in the rollouts

`CONCEPT.md` §7.3 lists Monte-Carlo noise as the rollouts' one reducible error and §13 measures
it: `SE(q)` between 0.65 BB and 25.7 BB at 256 samples, and a per-sample chip-delta deviation of
0.5–1.4 of the stack, because nearly every rollout is a stack-off. Samples buy that down at
`n^{-1/2}` and nothing else does — which is what this file changes. It is a **control variate**:
it changes the noise on a label, never what the label estimates.

**The baseline.** `b(s)` is every seat's share of the **matched** pot, weighted by how often it
holds the best hand over the boards that can still come, minus what it put in. Three properties
carry the whole design, and each buys one thing:

* **It is an expectation over the cards**, so dealing one and re-averaging gives it back. The
  correction at a chance node is therefore exactly `b(after) − b(before)` — the card's luck and
  nothing else — with no expectation left to evaluate.
* **It ignores chips beyond the call**, which the settlement refunds anyway. So `b` after a
  raise, an all-in and a call is one number: the correction at a decision node is two numbers
  rather than one per raise size, and it is **identically zero where folding is illegal**. The
  only thing a decision can surprise this baseline with is a seat leaving the showdown.
* **It is cheap** — one pass over the ranking matrix, no pot logic — so its cost does not grow
  with the table.

**Integrating the cards out is not a second mechanism.** Once a rollout has no decisions left,
the corrections for the streets still to come telescope into `b(final board) − b(that state)`,
and the rollout reports `R − that`. With no side pot, `b` on a complete board *is* the
settlement, the two `R`s cancel, and what is reported is the exact average over every runout that
could have happened. `tests/test_runout.py` pins that against replaying the hand once per
possible river through the untouched engine.

**Its declared limit.** With a side pot `b` prices a short all-in as if it could win the whole
matched pot, so the cancellation is partial and so is the reduction. It is a reduction that gets
smaller, not an estimate that moves: every correction has zero mean over the draw it corrects
whatever `b` is worth. The other limit is the same one the bet-size insensitivity buys: the
*size* of a bet is not something this removes variance from.

**Why not settle exactly.** An earlier version made `b` the engine's own settlement, averaged
over boards. It is exact everywhere, and it costs one pot settlement per **distinct ranking** of
the live seats: measured at 46 µs for two live seats and 250 µs for eight, against up to 60
distinct rankings over 64 boards at a full ring — 8.6 ms per hand of overhead at nine seats, and
a test battery that went from two minutes to over thirty. The cheap baseline is what a control
variate actually needs, and the exactness it gives up is exactly the side-pot case above.

**Cost, measured on the dev box** — CPU, which is where it runs: the label workers are CPU-only
processes. Added wall clock per hand, `samples = 16`, against the fixture pool:

| seats | hand without | added |
|---|---|---|
| 2 | 0.26 ms | +0.45 ms |
| 6 | 1.18 ms | +1.14 ms |
| 9 | 2.67 ms | +1.38 ms |

**What it buys, on the same rollouts** — the standard deviation of hero's per-rollout value, raw
against reduced, so the two are compared on identical cards and identical actions:

| table | sd raw (BB) | sd reduced | ratio | samples this is worth |
|---|---|---|---|---|
| 2 seats, 200 BB | 60.4 | 31.4 | 0.52 | 3.7× |
| 3 seats, 100 BB | 27.3 | 11.2 | 0.41 | 5.9× |
| 6 seats, 50 BB | 21.2 | 10.0 | 0.47 | 4.5× |
| 9 seats, 30 BB | 14.6 | 6.7 | 0.46 | 4.7× |

**End to end, on the label path** — which is the number that decides whether this is worth having,
because corpus hands are never reduced. Whole labels, fixture pool, `samples = 16`, the same
rollouts either way, and the sd is hero's per-rollout value averaged over labels on all four
streets:

| table | wall clock | sd ratio | effective samples | net |
|---|---|---|---|---|
| 2 seats, 200 BB | ×1.61 | 0.663 | ×2.28 | ×1.4 |
| 6 seats, 200 BB | ×1.53 | 0.566 | ×3.12 | ×2.0 |

Before the ranking was shared and pruned (§2.2c) the same measurement read ×2.15 and ×1.98 of wall
clock for the same variance — the estimator was a wash at six seats and a small loss heads-up.
The knob is still there: `runout_samples = 4` costs ×1.47 / ×1.33 and buys ×2.02 / ×2.28, so most
of the reduction survives a quarter of the ranking, and 64 buys almost nothing more than 16.

So a label's standard error roughly halves, which is two to six times the sample budget, for
50–60% more wall clock. **All three tables are dev-box measurements against the fixture pool, not
the real one** — G3 is what produces the number that counts, and its split-half column now reads the
reduced value, so re-running the gate reports both the new `SE(q)` and the new seconds per label.

`samples` is how many board completions `b` averages over — exhaustive when there are no more
than that many, a uniform draw otherwise. The runout's variance comes out reduced by about
`1 − 1/samples`, so 8 buys 88% and 64 buys 98% while costing linearly; 16 is where the curve
flattens and is the default.

**Ranking is shared, and primed for the streets that will be asked.** The evaluator is what this
estimator actually costs — ranking `samples` boards per street per seat, against the one showdown
a plain hand pays for — so what is *not* ranked is the whole performance story.

A ranking depends on the cards and the street and on nothing else, and one label rolls every legal
action out on the **same deck**: an action changes the forced prefix, not a card. So the `|A|`
hands of one sample are one ranking between them rather than `|A|`, and every hand of a run reads
and writes one cache keyed by `(deck, street)`. Entries are dropped as the hands that own them
finish, so what it holds is bounded by the hands in flight and not by the length of the run.

Sharing is what fixes how the completions are drawn: from a generator seeded by **the cards that
street has already shown** — its board prefix, the holdings, the table size. Two things need
exactly that. No result may depend on which hand ranked a street first, or `q` would move with
`batch_hands` (§15, and `test_runout.py` pins it at three batch sizes). And a chance node's
correction has zero mean only if the boards `b(before)` averaged over were chosen independently of
the card that node turns over — so a seed lying downstream of that card, the hand's own seed
included where that seed is what dealt the deck, is the one thing the draw must not use.

What gets primed is what this round's decisions can ask for: their own street, the next one — where
a card correction lands — and the river, where the corrections telescope to if everybody is left
all-in. A decision still inside the *suppressed* part of a forced prefix asks for nothing at all,
so a rollout never pays for the streets that were already visible when the labelled decision was
taken. Priming remains a batching device and nothing more: a street nobody primed is ranked on
demand, one call at a time, with the same numbers.

Both together, measured over every label of a fixture corpus at `samples = 16`: 16.4 seven-card
rows per rollout heads-up, 51.5 at six seats and 83.4 at nine, where ranking all four streets of
every hand paid 98, 294 and 441 — a factor of 5.3 to 6.0 off the term that dominates the cost.

**Where it is wired.** `LockstepDriver(pool, n_actions, runout=RunoutConfig(...))` fills
`HandRecord.baseline_rewards` alongside `rewards`; with `runout=None` the field stays `None` and
nothing changes. Corpus play never gets it — those hands are not averaged, and a single hand's
`baseline_rewards` is not a chip count and does not conserve chips. Both labelling paths do:
`oracle/parallel.py` builds its driver from `OracleConfig.runout_config()`, and the
single-process path in `train/generate.py` borrows the corpus driver *configured*, restoring it
after. `oracle.rollout.hero_values` is the one place that decides which of the two fields an
average reads, so the oracle and G3's split-half column can never disagree about it.

**Cards a forced prefix turned over carry no correction.** A rollout replays the decisions that
led to the labelled one, and the streets dealt during that replay are streets hero had already
seen. They are conditioned on rather than drawn, so there is no luck in them; correcting for them
anyway would subtract a term whose mean is not zero, which is a bias and not a reduction. The
driver skips corrections for decisions before the last forced one.

**A fixed bug underneath it.** The engine's hand evaluator read a full house's kicker off a
descending scan that had already passed the trips, i.e. off the *lowest* qualifying pair, so a
player holding a pocket pair below a board pair was ranked below one holding junk. Measured at
~1 showdown in 10 000 — a bias, not noise, and it was in every label and every hand of pool play.
The batched evaluator was right; they now agree over 123 000 showdowns, including decks
restricted to a few ranks or suits so that full houses and flushes are constant. It had to be
fixed before any of this could be trusted: `b` on a complete board stands in for the settlement,
so a disagreement between the two evaluators would have moved labels rather than quieted them.

**Known, not fixed.** `Judger.get_reward`'s one-live-seat branch does not conserve chips — it
hands the last seat the whole pot including its own contribution. It is unreachable in v8: the
engine settles a fold-out itself in `Table.next_turn`, and the only caller of the other branch is
`env/dealers.py`, which nothing imports. Reported rather than changed.

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

### 2.3a Reading the board — `pool/strength.py`

The degenerate members above never look at the board, which is why a nit among them is a nit
about its *two cards* and nothing else. `PLAN_PROCEDURAL_POOL.md` replaces them with members
that do read it, and this is the layer they read.

**Everything is a `(1326,)` array.** The §7.2 posterior asks every member "what would you have
done holding *this*" for every combo consistent with the board — ~1 225 questions at one
decision — so a quantity is either per decision (pot, stacks, history: identical for all of
them) or per combo (strength, draws: one row each). `ALL_COMBOS` fixes the row order once, a
holding becomes a row through `combo_index`, and `DISJOINT` — 1.7 MB of bool, built at import —
answers "could an opponent hold that while I hold this".

**One board, one table.** `BoardStrength` is one evaluator call over the combos the board has
not blocked, plus a handful of `1326 × 1326` boolean reductions: ~17 ms on the dev box for a
flop, and then read by every member, every decision and every posterior query on that board.
`StrengthCache` keys on the board as a *set*, so two hands that saw the same flop in a different
order share it, and evicts by insertion order past `max_boards` (a table is ~40 KB; the
`1326 × 1326` intermediates are never kept).

**The percentile is exact, with card removal.** `hs[h]` counts only the combos an opponent could
actually hold — not on the board, not sharing a card with `h`. On a monotone board that moves a
hand's percentile by up to four points against the naive rank, and the hands it moves are
exactly the ones that block the strong combos, which is what a betting rule cares about. It is
the same number `env/showdown.py::strength_percentiles` computes one hand at a time on the
river, and a test pins them equal; `strength_percentiles` keeps its job of labelling showdowns.

`hs` and `range_equity(w)` are one formula — equity against a weighted range, `hs` being its
uniform case — computed as the same masked reduction, the uniform case by counting rather than
by a weighted sum only because counts make the river identity exact in floating point.

**What is deliberately crude.** `ehs(n) = hs^n + (1 − hs^n)·p_improve` takes its potential from
an outs count and the rule of four and two, which is what a human regular does at the table and
is systematically generous to combo draws and to dominated ones. Sampling runouts instead is
deferred: the §2.2d control variate already pays 16 runouts per street per sample, and doubling
that for the pool's benefit is a measured label-cost decision rather than a free improvement.
`hand_class` — top pair good kicker, overpair, set — is descriptive only: the cascade thresholds
on `hs`, and mentions a class where a human rule would.

**Preflop is a different object.** There is no board, so there is no table; there is a
`(169, 8)` Monte-Carlo integral of each starting-hand class's pot share against 1–8 random
opponents, 2 M deals and ~21 s on the dev box, seeded and written once to
`/data/v8/tables/preflop_equity_v1.npy` — outside `versions/`, per `CLAUDE.md` §2.
`preflop_rank_pct(table, n)` turns it into "where does this combo sit in the top-x % of
*combos*", weighting each class by its 6 / 4 / 12 combos, so a range written as a fraction means
what published ranges mean and tightens by itself as the table fills up. The repo's existing 169
*ordering* is not usable for either job: it needs `eval7`, which is not the project's evaluator
and has no `aarch64` wheel guarantee, and an ordering is not an equity against *n*.

### 2.3b The other half of a decision — `pool/situation.py`, `pool/stats.py`

§2.3a is per *combo*. A rule also needs the part that is identical for all 1 326 of them — am I
the preflop aggressor, how many players act behind me, what fraction of the pot am I being asked
for — and nothing in the tree produced that: the tokeniser reads the same record, but into
tokens. `situation(ctx)` is one pass over the decisions before the pending one, and the cascade
calls it once per `(record, snap_idx)` group.

**Everything is read off the pot, not off the bets.** A decision that closes a street is stepped
*before* its snapshot is taken, and the engine zeroes `bets` on a street change and pays the pot
into `credits` at the end of a hand — so for exactly the decisions that end something, both read
as nonsense (a call that closes the preflop shows as putting in −10 chips). The pot only ever
grows by what an action adds, on every street and on the last action of a hand alike, so
"this decision put in *x* chips" and "it raised" are both derived from it. This cost a debugging
pass and is the kind of thing that would otherwise show up as a member with a plausible-looking
but wrong preflop read.

**`facing` is the bet as a fraction of the pot it was bet into**, not of the pot the snapshot
carries — those differ by the bet itself, and only the first makes `mdf = 1/(1 + facing)` the
minimum defence frequency the cascade thresholds with. A half-pot bet reads as 0.5.

**It is a function of the prefix**, asserted: a `Situation` built on a finished record equals the
one built on the record truncated at that decision. This is §9's parity in a second costume —
the posterior asks about decisions from the middle of finished hands, so a rule that could see
the rest of the hand would be an oracle rather than a member.

**Ranges are what a regular *assigns*.** Each live opponent's preflop line (unopened / limp /
call / raise / re-raise) and seat map to a weight vector over the 1 326 combos, through the
combo percentile of §2.3a — an open widens from 15 % of combos in the first seat to 45 % on the
button and 75 % heads-up, a re-raise is polar (the top 8 % plus half weight on a band around
30 %), a call excludes the premiums. They are deliberately the same table for every archetype:
how well someone reads ranges is not a style axis in v1, and postflop they are not narrowed at
all — that is the posterior's job and it costs a per-decision update over 1 326 combos per
opponent.

**`hud_stats`** reports the sixteen PokerTracker frequencies over a list of records, each as
`(numerator, denominator)` rather than as a ratio, so a band can be checked against a count when
the denominator is small and a zero denominator is visible instead of being a `nan`. It reads
decisions through the same pass `situation` does, so "this was a bet" cannot come to mean one
thing in a member's rules and another in the report on it. It is the primary evidence for §P4's
realism gate, and the only evidence at all for the table sizes and stack depths where no
benchmark exists.

### 2.3c A regular, as a cascade — `pool/regular.py`

One class with one set of numbers per archetype. It is not a solver and is not trying to be: it
plays *recognisably*, and a pool is a fixed diverse population to best-respond to.

**Intents, then bins.** The rules produce a distribution over five intents — fold, check/call, a
small bet, a big one, all-in — and only then does an intent become an action index. That split
is what lets one rule set serve any raise grid: a size is a target fraction of the pot and the
member plays the nearest *legal* bin to it. Where a size cannot be expressed the mass goes to
all-in if that is legal and to check/call otherwise; folding when checking is free is
redistributed over what is legal before anything is logged. Preflop the conversion runs the
other way — a size stated in big blinds ("open to 2.5") or as a multiple of the raise it faces
("3-bet to 3.2×") becomes a pot fraction against the live pot — so the size a member plays
drifts with the number of limpers, which is what happens at a real table.

**Thresholds are on quantiles, classes are for reading.** Preflop rules threshold on the share
of *combos* better than the hand (§2.3a), so a range written once tightens by itself as the
table fills up; postflop rules threshold on `q = hs^(opponents)`, the probability of holding the
best hand right now, with a draw's potential folded in separately through `ehs`. Hand classes —
top pair good kicker, overpair, set — are never thresholded on. They are what a human rule
*says*; the quantiles are what it does.

**Cost.** A 1 326-row posterior query on a cached board is 4.2 ms and a self-play decision 1–6
ms. Getting there needed §2.3a's `range_equity` rewritten from a masked `1326 × 1326` reduction
into inclusion–exclusion on the two cards a combo holds — the same sums in a different order,
exact, 21 ms → 0.43 ms.

**Three places the design as first written was wrong, and the measurement that showed it.**

* **A "human floor" on equity is a ceiling on style.** "A hand getting the right price never
  folds" reads well, but with the price measured against a *random* hand it makes every
  archetype continue with 93 % of its range against a pot-sized bet — a nit and a calling
  station alike — because most of a preflop range beats a random hand a third of the time. The
  floor is on the **draw's own odds** instead, which is what a player means when they say it,
  and the defence frequency comes out at exactly `defend_factor/(1 + bet/pot)`.
* **Heads-up, the small blind is the button, and a heads-up button is not a six-max button.**
  Interpolating position from a first seat to a button hands the heads-up small blind the
  six-max button's range — 45 % for a TAG — while the same tables credit an *opponent* in that
  seat with 75 %. Scaled to the heads-up norm it opens 74.7 %. This is the table size the
  committed config trains at.
* **The c-bet knob is a bluffing frequency, not a betting frequency.** Value bets at
  `1 − slowplay` whatever it says, so the total bet mass is always above it. The identity that
  holds — and that the tests pin — is that the air bets at exactly `f · bluff_ratio`.
* **Preflop and postflop are not the same seat order.** An opening range interpolated on
  postflop position gives the small blind under-the-gun's range, because postflop it is the
  earliest seat — while preflop it acts second to last with one player behind it. Every
  archetype had it at once, which is why a report about how they *differ* could never have
  caught it. Preflop the small blind now reads as a button, which is also what a regular credits
  an opponent in that seat with.

### 2.3d Ten of them — `pool/archetypes.py`, `gates/pool_realism.py`

Six archetypes lie in one plane — tightness × aggression × bluff share — and share three
regularities an agent could learn once and apply to all six: a bet correlates with strength the
same way, sizes stay between half a pot and a pot, and position bends every range by the same
shape. The other four exist to break one each: `weak_tight` enters as many pots as a
loose-passive and then *folds* instead of calling, `trapper` breaks "a check means weakness",
`polar_reg` breaks the size axis by overbetting, and `stealer` breaks the position curve.

**Jitter is where diversity beyond ten comes from.** A rate is multiplied by a lognormal draw, so
a zero stays zero — an archetype that never bluffs is not jittered into bluffing — while a size
or a stack threshold is *shifted*, because a pot fraction of 0.33 and one of 1.5 want the same
absolute spread and not the same relative one. Every knob is clipped back into its domain.

**The gate answers two questions and gates nothing** (owner decision, 2026-09-03). *Diverse?* —
all ten sit at one table per size and are dealt a fresh seating every hand, and every archetype's stat line
is read off the same hands against the same field. One hand is a data point for every seat at
it, so reading ten stat lines costs what reading one costs, and the numbers are comparable
rather than ten separate experiments. The seating is *drawn* and not rotated by a fixed step:
with ten members at a two-handed table a fixed step is two, the even-indexed archetypes never
leave the button and the odd ones never leave the big blind, and every stat line comes back
positional — which read a maniac as tighter than a TAG. *Better than the degenerate five?* — each archetype is
seated against always-fold / always-call / always-min-raise / maniac / nit and measured in
BB/100 with its standard error. There are no acceptance bands: with two dozen hand-set knobs
there is no realistic path from a measurement back into a fitted pool, so the machinery that
would have guarded against one is not built.

**A `regular` bootstrap entry** seats archetypes in the training pool. Its `n_variants` draws
*parameters* and not styles — the cascade already is the style — so the style defaults to no
modifier, `spread` is what makes variants differ, and several variants at spread zero are
refused rather than silently identical. Every regular in one build shares one board cache and
one preflop table. The labelling workers rebuild them from their parameters (`oracle/parallel.py`
raises on a member kind it cannot mirror, so without this a run with regulars and more than one
worker would die on its first iteration); the board cache is the one thing that does not travel,
because a cache is per process and the worker builds its own.

**It writes as it goes and resumes.** The whole job is twenty-odd minutes of CPU; the report is
written after every cell, each table size is printed the moment it finishes rather than all at
the end, and a re-launch with the same settings replays only the cells that are missing. The run
directory is named and not timestamped, for the same reason the Slumbot run's is: an interrupted
job is resumed, not started again beside itself.

### Progress bars that are not lying — `utils.progress`

`CLAUDE.md` §5 fixes `smoothing=0`, which makes tqdm's rate the plain average
`(n - initial) / elapsed`. Two consequences bit this tree and both are now
handled in one place.

**A resumed phase must pass `initial=`, never `bar.update(start)`.** Advancing the
bar by the labels a *previous* run wrote counts them as having taken this run zero
seconds, and the rate and ETA come out inflated by exactly that ratio — a resumed
run at 4.5 s/label displayed 1.03 s/label, which is how "the workers are 4× faster"
was briefly believed. `_label_sessions` passes `initial=start`;
`tests/test_label_generation.py` pins the construction, since no behaviour can
catch it.

Two bars stand still for long stretches by design, and a bar that does not move is
indistinguishable from a hung run — which is the whole reason §5 asks for one. Both
now say what they are standing still for, in the postfix, without inventing units
or a second bar:

* the **play** bar of `_play_sessions` counts hands, and the §5.5 refit between
  blocks is not a hand: it reports `fit block b: k/N sessions` while it runs;
* the **label** bar releases labels in `todo` order while workers finish out of
  order, so it moves in bursts: it reports how many finished labels are held for
  ordering.

The ETA stays correct through both, because `smoothing=0` averages the stalled
seconds into the elapsed time they actually cost.

### Labelling in parallel — `oracle/parallel.py`, `oracle/transport.py`

A label costs, by G3's profile, roughly 69 % network forward and 31 % Python (v7 event
building, tensor packing, the driver's bookkeeping). One process parallelises neither,
and the label phase is the longest phase of an iteration. `CLAUDE.md` §3 names the layout
the box wants and this is it: **N CPU worker processes and one GPU inference server**, the
server being the parent process, which already owns the CUDA context and every network.
`oracle.n_workers` turns it on; `0` or `1` is the sequential path, unchanged.

**What makes it legitimate.** A label is a pure function of a finished record: every draw
inside `action_values` is keyed by the decision (`default_rng([seed, i, h, d])`,
`_rollout_seed(spec.seed, decision_idx, a, s)`), and the driver samples each hand from its
own generator, so nothing about a label depends on what else is being computed or in what
order.

**Workers hold a weightless mirror of the pool.** `mirror_spec` describes each member
without its weights and `build_mirror` rebuilds it in the worker with the network replaced
by a proxy. Neither `V7NetworkMember` nor `AgentPoolMember` changes: a v7 member asks its
network for `n_actions`, `device_` and `action_logits`, an agent member for `d_emb` and a
call, and a proxy answers exactly those. Styles, legality and the grid transport of
`pool/action_map.py` all run in the worker; only the forward crosses to the parent.

**One request, one forward.** The server answers whichever worker is ready, over exactly
the rows one process would have given the model. So a parallel run's labels equal a
sequential run's **to the bit** — `tests/test_parallel_labels.py` asserts it with a v7
checkpoint in the pool and the agent in hero's seat.

#### Two measurements that shaped this, both on the dev box

*The first design sent tensors through an `mp.Queue`.* `torch` gives every tensor its own
shared-memory segment, created, fd-passed and mapped per send: **8.7–13.9 ms per round
trip** for a ten-tensor payload, comparable to the forward it carried and paid tens of
times per label. `oracle/transport.py` allocates one slab of shared memory per worker at
spawn and sends only offsets over a `Pipe` — the same round trip is **0.062 ms**, a factor
of 140–220. The field layouts (`v7_fields`, `token_fields`) are tables that `pack`/`unpack`
walk, each field landing contiguously so the tensor handed to the model needs no repacking.

*The first design also batched at a barrier* — the server collected one request from every
live worker and answered them in one wide batch, which is reproducible and looks like good
batching. Measured, it was **4.8× slower than one process** and got *worse* from two workers
to four. Workers are never in phase: their requests interleave posterior batches of a
thousand combos with driver steps of a dozen hands, and at a barrier everybody pays the
slowest. Asynchronous service keeps the GPU busy by *overlap* instead, and as a bonus makes
the result bit-identical to sequential.

**Partition by session, strided.** A worker owns whole sessions, reads a disjoint slice of
the records and rebuilds hero's member once per block exactly as the sequential path does.

**The parent keeps the bookkeeping.** Workers return `(position, q, legal, stats)` and
nothing else; the tokens `_ObservedHero` recorded during play, the shard writing and the
progress file stay in the parent, and finished labels are consumed **in `todo` order**, so
the shards are the same shards with the same rows in the same files. That matters beyond
tidiness: `split_heldout` partitions by position.

**What it does not buy.** With every forward serialised through one process the ceiling is
`1 / 0.69 ≈ 1.45×`, reached at a handful of workers; on the dev box, which has no GPU and so
runs the "server" on the same cores as the workers, four workers measured at parity with one
process. The number that explains it is in the server's own log: at toy scale, **946 forwards
for 21 487 rows — 23 rows per forward.** The oracle asks for many small policy batches, and
neither more workers nor a faster wire changes that.

### `oracle.labels_per_batch` — several labels through one `driver.run`

The width of a policy batch is set by how many rollout hands are in flight. One label puts
`samples_per_action × |legal|` hands into `driver.run`, and the lock-step group *drains* as
they finish, so the last rounds of every label are narrow. `action_values_batch` builds `k`
labels' rollouts and plays them together: the driver refills to `batch_hands` throughout, so
the network sees fewer, wider batches for the same work. `train/generate.py::label_chunks`
does the grouping and never lets a chunk cross a `(session, block)` boundary, because hero is
reseated there (§5.5) and a chunk's rollouts are all built before any of them is played.

**It changes what a label costs, not what it is.** Every rollout hand keeps its own deck and
its own `_rollout_seed`, and `env/driver.py` draws each hand's actions from a generator seeded
by that alone, so which other hands shared the batch cannot move an action.
`tests/test_parallel_labels.py` asserts the labels are byte-identical between `k = 1` and
`k = 8` with a network-free pool — the version of "no bias" that can actually be asserted.
With a network in the pool the remaining difference is floating point: a wider batch reduces
in a different order, which on GPU can move a logit in its last bits. Same class as changing
`batch_hands`, and unbiased.

**Two caveats, both measured or arithmetic.** The width is capped by `batch_hands`: with
`samples_per_action = 128` a single label already queues ~1500 hands against a cap of 2048, so
`k` mostly removes the draining tail and the two knobs want raising together. And memory grows
with `k` — `driver.run` holds a `HandRecord` per spec until it returns, so `k = 8` keeps of the
order of 10⁴ records per worker. On the dev box the sweep over `k` measured flat, which is
what §3 predicts and not evidence either way: a 128-row and a 256-row batch cost nearly the
same on a GPU and nearly double on a CPU, so whether widening pays is exactly the question this
box cannot answer.

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

### 2.4c The range head (§5.7) — `oracle/ranges.py`, `nets/range_head.py`

The fourth head, and the only one that is not a leaf: it sits **between** the
trunk's two halves of decoder layers, and what it predicts goes back into the
tokens the remaining layers read. Owner decision 2026-09-02.

**What it predicts.** At every decision token, for every player who is not the
observer and has not folded, a distribution over the 1326 two-card combos — the
observer's belief about that player's holding. One query per (token, live
opponent); the agent's pending token gets one per live opponent too.

**What the target is.** The reach-weighted posterior of §7.2,

```
w(combo) ∝ prior(combo) · Π_t  max(floor, P_i(a_t | combo, history_t))
```

run as a **filter** rather than as a batch computation: the weights are carried
forward, each new action of that opponent multiplies them, each new board card
removes the combos it blocks. `oracle/ranges.py` is that filter and with
`range_prune_threshold = 0` it is bit-identical to calling `opponent_posterior`
at every prefix — `test_range_head.py` pins exactly that, prefix by prefix. One
definition of "opponent range" in the tree, two consumers.

A hard variant was considered first and rejected: keep the combos whose *modal*
action is the one that was played. It is degenerate here. Every pool member emits
finite logits and the style layer mixes in up to 25 % uniform, so no combo ever
has zero reach and the filter's only content would be card removal — which is a
deterministic function of the token's own input. Four of the five degenerate
strategies do not read their cards at all, so their argmax is the same for every
combo and the set comes out either full or empty. The soft weights cost the
**same forwards** — both need `P(a | combo)` for every surviving combo — so
rounding buys nothing and discards the magnitude the belief is for.

**`range_prune_threshold` is the one approximation and it buys forwards.** A
combo below `prune × max weight` leaves the support permanently, so later streets
ask the member about fewer combos. Relative to the maximum and not absolute: a
uniform prior gives every combo `1/C`, so an absolute threshold near that scale
empties the range at the first token and one below it never bites — the knob
would do nothing or everything depending on the size of the support.
`RangeStats.dropped` reports the mass thrown away and `collapsed` the supports a
board card wiped out, so a threshold set past a tail and into a mode shows up as
a number rather than as a quietly different target.

**Where the targets come from, and what they cost.**

| | how | cost |
|---|---|---|
| embedding corpus | `label_ranges` over the played sessions, once, beside `label_showdowns` | ~1225 policy rows per opponent decision against ~1 today. This is the dominant new cost and it is regulated by the corpus size (owner decision 2026-09-02) |
| agent labels | `HandRangeCache` in the label worker | one pass of the hand's opponent likelihoods — what the oracle's own posterior already costs, so ≈ +10 % of a heads-up label (G3: 1225 posterior rows in ~11k) |

The label path recomputes rather than reading the oracle's posteriors out of
`action_values`, whose three-tuple every caller and a dozen tests read. That is
the trade and it is written down rather than assumed. It runs **in the worker**,
never in the parent: the parent is the inference server, and serialising ~1200
policy rows per label behind it is the one thing that must not happen.

**The module is a perceiver decoder, not an MLP on the token.** A range is the
product of that player's likelihoods over every decision they have taken, and
those live in earlier tokens. So queries — "player *s*, at moment *t*" — cross-
attend over the trunk's states across the prefix, `n_range_blocks` times.

Four properties carry it, and each is a way of getting it wrong:

* **The cross-attention is causal.** A query at token *t* sees trunk states at
  *t' ≤ t* and nothing later. Without it the belief at the third decision reads
  the seventh, the loss falls beautifully, and at deployment the head is reading
  actions that have not happened. `test_range_head.py` perturbs the last token
  and asserts the earlier beliefs do not move.
* **The query says who and when.** Position embeddings alone would make all `T`
  queries of one seat the same vector, separated only by their mask. So the query
  is the seat's learnable position embedding, plus that seat's opponent vector,
  plus the trunk state at `t`, with RoPE marking the moment. Seats are the axis
  and not slots, because the seat *is* the poker position — the engine fixes seat
  0 as SB and rotates the players — and identity comes in through the vector.
* **What goes back is the probabilities, detached.** Not the head's hidden state:
  a `d_range`-wide state would let the action loss push arbitrary information
  around the 1326-wide bottleneck and the stop-grad would be closing the wrong
  channel. The token attends over its own active seats' belief vectors, so the
  aggregation is learned and `pos_out` keeps which belief belonged to whom.
* **Blocked combos are dropped, not learned.** A combo holding a board card or
  one of the observer's own gets `-inf` before the softmax, the way an illegal
  action gets an exact zero in a policy target. This is why `own_hole` now rides
  on **every** token: the observer has always known its own hand, the decision
  token only ever showed it on the observer's own rows, and the support of a
  belief is exactly what the board and those two cards leave. The tokeniser does
  not read it, so the §5.1 observation is unchanged.

**`range_weight` is not the ablation.** With the weight at zero the head still
runs and still injects, so the layers above it would consume an untrained head's
output — noise. The ablation is `range_enabled: false`, which removes the module
from the trunk; `test_range_head.py` asserts that switch reproduces the previous
trunk exactly, tensor for tensor, and leaves no `range` parameter behind.

**Read the KL, not the cross-entropy.** The support is 990–1225 combos, so a
perfect head still pays the target's own entropy — ~6.9–7.1 nats. `range_kl` is
`ce − H(target)`, zero at the optimum, and it is the part the head can move.
This is the §11.4 trap in a third place, after §5.6's MSE and §5.1a's.

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

### 2.7a `gates/pool_conditioning.py` — is a past agent stronger reading its tablemates?

The measurement `PLAN_AMORTISED_POOL.md` P2 asks for, and the one that decides whether the
capability P1 built is wired into the phases that seat a past agent (P3, P4). It is **not** an
agent result and says so on its face: every report it writes carries the stamp
`POOL MEMBER CONDITIONING — not an agent result`, and `write_report` refuses to write one that
does not, or one whose numbers carry no standard error.

**The design is the pairing.** One checkpoint sits at slot 0 of the same sessions twice — same
seeds, same tables, same opponents from the bootstrap pool, same rotation — once at `e = 0` and
once conditioned on the `K = 0` vectors of its own view, refreshed every `R` hands over
`embedding_net.pool_agent_window` hands. The hands diverge as soon as the two conditions choose
differently, which is the effect; the *situations* they are dealt into are identical, which is
what makes the difference per session meaningful. A single 400-hand session has a standard error
of tens of BB/100, so the number quoted is the paired difference and its SE over sessions,
grouped also by table size and by stack depth (`CLAUDE.md` §1's no-cliff claim).

The block discipline is hero's (§2.10): the vectors in force during block *b* are computed over
blocks `0 … b−1`, and block 0 is the cold start — seated as the *unconditioned* member in both
halves, because a zero table and no table are bit-identical (§2.8). One reserved pool entry per
session, not per seat, because a member that knows its own slot derives the rotation from the
seat it is asked to act at.

Both checkpoints carry the config they were trained under, and the gate asserts the load-bearing
keys against its own — the action set, the raise grid, the trunk shape — so a checkpoint from
another run fails loudly instead of producing a number nobody can interpret. The report also
carries a timing section (`hand_tokens` per call, `amortised_vectors` per hand of window on the
box it ran on), which is the measurement `PLAN_AMORTISED_POOL.md` §1.1 needs before P4 commits
to a corpus-scale refresh.

The pre-registered decision rule lives in the plan and is *printed* beside the number rather
than encoded: the gate reports, the owner decides. **It has not been run, and it is no longer a
gate on anything.** The owner decided on 2026-09-03 to wire the conditioning in without it: the
pool is strongly exploitable early, reading the opponent is precisely the mechanism that finds
those exploits, and measuring the effect on agents that are weak in exactly that way answers a
different question. The gate stays as a measurement that can be pointed at any two checkpoints —
its own falsifiable use is the prediction that the advantage *shrinks* as the pool converges.

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

* **The vectors are zero by default** — D12 option (a), the plan's recommendation, taken
  2026-08-19. A past agent then plays its *unconditional* policy, the one §6.2's embedding
  dropout trains explicitly, so it is a fixed policy like every other member: no fit nested
  inside a fit, no recursion, no answer needed to "what did agent *k−3* believe about its
  tablemates".
* **One member serves every seat**, because with `e = 0` the slot only selects which zero vector
  is read. That is what `PoolMember` requires and what `AgentPoolMember` cannot give.
* **The observation is rebuilt for the moment being asked about**: the record is truncated at the
  asked-about snapshot and the hypothetical holding is swapped into a copy of the deck. This is
  the same pair of moves `pool/v7_member.py` makes for the same two reasons, and it shares
  `_deck_seen_by` with it rather than repeating it. On the hot path — the driver, acting now,
  real cards — no copy is made at all.

The style layer applies to it exactly as to a v7 checkpoint (§4.2), so the `agent_variants`
members one agent contributes are `with_style` siblings sharing one network by reference (D11).

**Or the member is handed a table** — `(embeddings, own_slot)`, optional and absent by default
(`PLAN_AMORTISED_POOL.md` P1, the revisit of D12). It is then conditioned on its tablemates the
way hero is: `emb = embeddings[slot]` and `seat_emb = embeddings[seat_slot]`, the same two
gathers `AgentPoolMember` makes. Neither of the first two bullets is lost:

* **one member still serves every seat**, because a member that knows *its own* slot derives the
  rotation from the seat it is asked to act at — slot `own_slot` sits at seat `acting_pos` in
  hand `h` iff `h ≡ own_slot − acting_pos (mod n)`, and that fixes every seat's slot. One member
  per (session, slot, block), not one per seat. `test_frozen_agent_vectors.py` asserts the
  derived rotation against `Session.slot_of_seat` at every table size 2–9 and every hand.
* **the sibling pattern is `with_style`'s**: `with_vectors(name, embeddings, own_slot)` is a
  `copy.copy` with the table swapped, so a refresh costs no second network.

A zero table is bit-identical to no table at all, which is what makes the cold start of block 0
the same policy D12 settled on. Nothing in this class decides *which* table a block should hold;
that is the block discipline of the phase that seats the member (§2.10).

**The member also carries the embedding network of its own generation** — `embed_net`, frozen at
the moment that generation entered the pool, and `None` for a member nobody may condition. Never
the loop's current network: nothing anchors the coordinates of the vector space, the loop keeps
training the one network it holds, and a frozen policy handed vectors from a later generation
would be reading a description in a basis that has drifted under it. It would also stop the pool
from being *fixed* — the same member would play differently at iteration 10 and at iteration 20
with nobody having changed it, which makes hero's accumulated per-member results (what PFSP
samples on) results against a moving target, and makes the embedding network describe styles that
move because it moved. `pipeline.py` freezes one snapshot per retrain, hands it to the members
that generation contributes, and rebuilds the same mapping on resume from the per-iteration
checkpoints — one load per generation, about 200 MB each.

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

**Local EV-loss annealing (2026-09-08).** The scale invariance above applies to
legacy scalar T. The default `oracle.temperature` is now
`{"initial_ev_loss_bb": 1.0}`. `ev_loss_budget` returns
`eps_k = initial_ev_loss_bb / (k + 1)` for zero-based outer cycle k;
`decision_temperature` returns `eps_k / (D log A)` for A > 1 legal actions and
the configured BB divisor D. At A = 1, T = 1 is harmless: the sole action loses
zero EV. The exact soft optimum obeys `max Q - <pi_T,Q> <= D T log A = eps_k`.
This bounds entropy smoothing at exact Q and exact optimisation, not oracle
noise, model fitting error or exploitability. The initial budget and harmonic
schedule are choices; no claim of an optimal annealing rate is made.

Thus cycles 1, 10 and 30 allow 1, 0.1 and 1/30 BB per decision. The divisor
cancels in the target under an absolute BB budget but still weights the loss
across states. T is independent of sampled Q, preserving soft_q's linear
gradient. The trainer reads each pending token's pot, facing bet and legal
mask; the loss accepts a vector of temperatures, detaches it and centers Q
before dividing to avoid overflow at small T. Both targets and warm/cold gaps
use the same resolver. A scalar config retains the old fixed-T behaviour.

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
`Session.tokens` gained two, `observer_slot=0` and `window=None`, both of them today's
behaviour: the other slots are the per-observer tokenisation a *non-hero* fit needs, since
pooling by slot over hero's tokens would show that player hero's hole cards (§9 from the pool's
side), and the window is the history such a fit is allowed to look at. The §5.7 range targets
are hero's — their keys name a seat hero has a live opponent at, and `hand_tokens` refuses a key
another observer has no live opponent at — so they are passed for slot 0 alone.
`train/generate.py::amortised_vectors` is that tokenisation around one `amortised_init`, i.e.
`fit_embeddings(steps=0)` with no gradient step, and it returns a `_pad_vectors` table. It is
the `K = 0` conditioning of `PLAN_AMORTISED_POOL.md`, and this phase is the first caller.
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

**`pool_agent_vectors` — the pool's own past agents, conditioned or blind.** `"zero"`, the
default, is D12 option (a): a past agent seated as an opponent reads `e = 0` and plays its
unconditional policy, and every path below is inert. `"amortised"` gives each past agent at each
table the `K = 0` reading of *its own* tablemates, on the same block boundaries as hero's fit,
through the frozen embedding network of its own generation (§2.8). Concretely:

* the phase reserves one play-pool entry per **(session, past-agent slot)** — one per slot and
  not per seat, because a member that knows its own slot derives the rotation from the seat it
  is asked to act at — and points that slot's seat at it in every hand;
* the per-block table grows an observer axis: row 0 is hero's fitted table, unchanged and still
  the only thing a label stores, and row `slot` is that slot's `K = 0` reading. The axis is one
  row wide under `"zero"`, so the default writes exactly the vectors it always wrote;
* the refresh happens where hero's fit happens, from the same records: one tokenisation and one
  trunk pass per conditioned seat, over `pool_agent_window` hands (`null` = the session so far);
* the same seating function runs in the play loop, in the sequential label loop and in every
  worker, and it looks the base member up *through the play pool* — which finds the pool member
  in the parent and that member's weightless mirror in a worker, so the inference server, the
  slab and the runners are untouched. The reserved entries are holes in the mirror, exactly as
  hero's are;
* both keys are in the play signature, because two runs that disagree on them played different
  hands however identical their tables, and their labels must never be spliced.

`_results_by_member` and PFSP are indexed by pool member and never see the reserved entries, so
a past agent is still scored as one member and not as its per-session copies.

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
| D | oracle gap | §8's seven numbers on the held-out slice, twice: conditioned on the fitted vectors and with `e = 0` | `metrics.json` |
| E | close | even iterations update/decay PFSP and apply evaluation feedback; every iteration appends its agent | `state.json` |

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
on-policy. Only iteration 0 starts from random weights and gets its own
`first_iteration_steps`.

**Paired policy updates (2026-10-02).** Iterations 1 and 2 both initialise hero
and the trainable weights from `agent0`; iterations 3 and 4 from `agent2`, and
so on. The even iteration sees the intervening odd checkpoint in the pool.
Weights are restored before collecting hands, so oracle continuations also use
the selected parent. Both `labels.json` and `agent.pt` record `hero_iteration`.
The embedding retrain schedule and pool growth stay as configured.

**Odd checkpoints leave PFSP evidence unchanged.** Their collection results
are not accumulated, `result_decay` is not applied, and evaluation feedback is
not consumed. Evaluation still runs and is saved. The next iteration expands
the sampler for the new pool member, initially unplayed with maximum hardness;
normalisation and clustering can therefore change seat probabilities even
though the old members' result accumulators are unchanged. Sampler RNG state
continues to advance normally. Even iterations, including iteration 0, perform
the existing result update, decay and evaluation feedback.

Resume restores the same even parent for both collection and training. For an
even iteration without a completed `agent.pt`, legacy labels collected using
the immediately preceding odd hero are rebuilt; partial label directories are
kept under `labels.stale*`. Their parent is part of the play signature, so a
partial shard cannot be spliced into the new policy's rollouts. Completed
checkpoints remain readable and are not retrained.

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

**The embedding corpus uses PFSP for every seat**, including the observer, through
the same `PoolSampler` as label collection: PFSP weights, embedding clusters and
the uniform floor from `pool_sampling`. It reads the results and clusters saved
at the previous iteration boundary; new pool members start with maximum PFSP
weight in their own clusters. At iteration 0 there are no results or trained
clusters, so the distribution is uniform. Members may repeat at a table.
The corpus uses a separate RNG seeded by `(seed, iteration, 12)`, leaving the
label sampler unchanged when phase A runs or is skipped on resume. Corpus
self-play results do not update the current hero's PFSP scores. Table sizes,
stacks, hand seeds and seat rotation still come from `env.session.build_sessions`.

`embedding_net.first_corpus_sessions` sets the corpus size at iteration 0;
`corpus_sessions` sets it on later retrains. Omitting the first key uses
`corpus_sessions` for both. The main config uses 300 sessions initially and 30
on later retrains, with 2000 hands per session. The hand-seed layout reserves
the larger count on every iteration so the different sizes cannot overlap.

**Row *i* of the embedding table is pool member *i*.** The table is sized
`len(pool₀) + max_iterations × agent_variants` up front (D9) and the pool grows by exactly
`agent_variants` members per iteration (D11), so the two indices coincide by construction and
iteration *k*'s block starts at `len(pool₀) + k × agent_variants`. A row nobody occupies yet is a
dead parameter at its initialisation, because no token carries its index.

**The oracle gap** (`gap_terms`, `oracle_gap`) is §8's seven numbers — `kl`, `ev_agent`,
`ev_oracle`, `q_best`, `ev_gap_target`, `ev_gap_greedy`, `agreement`, the `GAP_KEYS` tuple — on
the held-out slice, overall and grouped by table size and by stack depth. The two gaps are
differences of the three terms beside them (`ev_gap_target = ev_oracle − ev_agent`,
`ev_gap_greedy = q_best − ev_agent`) and those terms are reported for that reason: a gap that
moved does not say which of its sides moved. One agent forward per held-out decision and **no new rollouts**: the oracle's answer
is already in the shard. What it answers is whether this iteration's training absorbed this
iteration's labels; what it does not is anything about exploitability, and its floor is G3's
Monte-Carlo error rather than zero. `heldout_fraction = 0` reports no gap at all rather than one
measured on the data the optimiser just saw.

**It is measured twice, warm and cold** — §12's distinction moved onto the held-out slice.
`metrics["gap"]` conditions the agent on the vectors §5.5 had fitted when hero acted, which are
the ones the label carries; `metrics["gap_cold"]` pins them to zero and reads the *unconditional*
policy §6.2's embedding dropout trains. Same labels, same oracle `Q`, one extra forward per
label, so `ev_oracle` and `q_best` are identical between the two by construction and only the
agent's side moves. The iteration's last log line is `winrate_line`: `ev_agent` warm, `ev_agent`
cold, and their difference. That difference is what says whether conditioning on the opponent is
paying for itself *at all* — as at §12, a cold number above the warm one is a result to report
rather than a bug to tune away. Two cautions on reading it: `ev_agent` is not a played BB/100 —
phase D plays no hands, this is the agent's policy scored by the oracle's own noisy `Q` on hero's
own state distribution in §6.2's pot-normalised units — and the warm side is scored under vectors
fitted from at most `R` hands of context, so it is the *early* part of §12's warm-up curve rather
than its asymptote.

**Phase D can be re-run after the fact** (`regap.py`). It is a `softmax` over the `q` in the
label shards plus one batched agent forward, so an iteration whose `labels/` and `agent.pt` are
still on disk can be re-measured without replaying a hand — which is how a metric added to
`gap_terms` after a run reaches the iterations that ran before it. The held-out split is
`split_heldout(n, fraction, seed, iteration)` and reads no running RNG, and the checkpoint is the
one that iteration saved, so the recomputation reproduces the numbers the run wrote; every number
already in `metrics.json` is asserted against its recomputed value — `gap` and `gap_cold` alike —
and a mismatch aborts rather than overwriting. An iteration that ran before the cold pass existed
therefore gains a `gap_cold` from `regap.py` without its `gap` moving. It trains nothing and rewrites nothing but `metrics.json` and `report.json`'s
metric list.

**PFSP is fed from the hands the label phase already played.** `train/generate.py`'s manifest now
carries hero's BB and hand count against every member it sat with, and on even
iterations phase E hands them to `PoolSampler.update` before `end_iteration`
ages them (D10). A hand is credited to **every**
opponent at the table in full: hero's chip delta in a multiway hand is not divisible between the
opponents who produced it, and splitting it by table size would make a nine-handed beating look
an eighth as bad as the heads-up one it is being compared against. The cost is that the
nine-handed number carries eight opponents' worth of noise, which is what `result_decay` keeps
from accumulating.

**Resume is per phase, and inside labelling per hand.** `./run.sh --version=v8` after a crash or
a stop continues where it left off: every phase writes its artefact before the next one starts and
skips itself if that artefact is already there, and phase B — the one measured in days — writes a
`labels/progress.json` after every flushed shard as well, so a crash in its middle costs at most
one shard rather than the phase (§2.10).

A mid-labelling resume **plays exactly the hands that are not labelled yet, and refits nothing.**
Once the corpus is played, phase B writes `labels/play.json` and `labels/vectors.npz` beside the
shards: the configuration of every session, hero's result against each member, and the §5.5
embedding table of every `(session, block)`. Those tables are what makes a hand replayable on its
own — the vectors hero acts under in block *b* are fitted over blocks `0 … b−1`, so reading them
back gives exactly the vectors of the interrupted run without the hands that produced them — and
`progress.json` therefore counts whole *hands*, the unit that gets replayed. A shard is flushed at
a hand boundary and never inside one, which is what makes that count a place to start from.

Re-playing the labelled hands, which is what this phase used to do, was the wrong trade. It is
cheap in wall clock — minutes against days — but it assumes the replay reproduces the hands the
shards were labelled from, and on a GPU it does not: neither a forward nor the §5.5 fit reduces in
the same order twice across processes, one flipped action a few hundred hands in gives a different
hand, and the resumed run then either splices two corpora together or dies looking for a decision
that is not there. Playing only the unlabelled tail removes the assumption rather than tightening
it. What still has to hold is the frame: `play.json` names the sessions, and a directory whose
sessions are not this call's is set aside under `<dir>.stale` — never deleted, its shards cost
real time — and the phase starts over rather than splicing or refusing to run.

What makes the resumed run *identical* rather than merely valid, over the hands it does play, is
that no phase reads a running RNG: every stream is seeded from
`(seed, iteration)`, and the one piece of genuinely sequential state — the sampler, whose draws
are consumed in phase B — is written out with phase B's own artefact and restored from it. One
iteration owns one block of a million hand seeds, split in half between the corpus and the
labelled sessions, and both halves are asserted to fit rather than assumed to.

**One temperature specification.** §6.2's scalar T or local-loss schedule lives
once in `oracle.temperature`. Targets, training and metrics resolve it with the
same state scale and outer iteration. Metrics save the specification and the
resolved BB budget; checkpoints already save the config and iteration. A
restart continues at the actual cycle, and `regap.py` uses the checkpoint's
config. Resuming between training and measurement likewise keeps that
checkpoint's target and held-out split even if the next cycle's config changed.

Labels carry `range_model = all_seats_reach_v2` in their manifest and the
played-corpus resume signature. A pre-fix partial label set is moved to
`labels.stale*` before regeneration; old complete labels awaiting training are
regenerated as well. Existing trained checkpoints keep their historical labels
for reproducible metrics; completed cycles are not retrained by a config edit.
New cycles use all-seat conditioning. Start a new experiment if the whole
training history must use the new temperature and oracle.

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

**Who plays is a factory, not a network.** `SlumbotAgent` takes
`(hero_seat, embeddings) -> PoolMember`; the agent's own factory builds an `AgentPoolMember` for
the seat Slumbot dealt it, and a procedural §2.3d archetype arrives through the same door as one
member serving every seat. `evaluation.hero` selects it, absent meaning the agent. Two things
follow and both are said out loud rather than left to fail late: a member with no opponent vector
declares so (`d_emb is None`), which makes a *warm* run against it an error rather than a silent
no-op, so the runner drops the warm mode with a logged reason; and a pool member's run needs no
agent checkpoint and is written under `pool_eval/<archetype>/`, never beside the agent's, because
the two numbers are not the same kind of thing and a directory is the cheapest place to stop them
being read as if they were.

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

### 3.1 A run whose raise grid is not the checkpoint's — `pool/action_map.py`

`CONCEPT.md` §6.1 reuses v7's action set, and until 2026-08-20 that was enforced rather than
implemented: `build_pool` asserted that a v7 entry's action count equalled the pool's and stopped
there. A v8 run is free to choose its own `game.raise_sizes` — a finer preflop grid costs the
evaluation adapter less when it snaps Slumbot's bets to a bin (`evaluation/protocol.py`
`token_to_action_idx`) — and a checkpoint is welded to the grid it was trained on **from both
ends**: `perception.action_proj` is a `Linear(n_actions, d_model)` over the one-hot of the action
just taken, and `action_head` emits one logit per action of that layout.

`RaiseGridMap` is the translation, and the only place that knows the two grids differ. Both
directions are nearest bin by raise fraction, per street — `env/legal.py` reads
`raise_sizes[turn][i]` as a fraction of the effective pot on *every* street, preflop included, so
the two grids' numbers are directly comparable. `fold`, `call` and `all-in` are positional.

* **History (pool → v7).** Each action-bearing snapshot is shallow-copied with its one-hot
  rewritten to v7's layout. The approximation is bounded: what actually went into the pot is
  carried exactly by `pot`, `bets` and `stacks` in the same event, and the one-hot is a redundant
  categorical channel beside them.
* **Policy (v7 → pool).** The member's distribution is softmaxed in v7's layout and each bin's
  whole mass is moved to the single nearest pool bin (owner decision 2026-08-20: nearest bin, not
  split between neighbours, so a member keeps betting the size it meant to). Returning
  log-probabilities rather than logits changes nothing downstream —
  `softmax(log softmax(z) / T) == softmax(z / T)`.

Pool bins that are nobody's nearest therefore receive no mass from v7 members. That hole is
deliberate: the agent's targets are the oracle's Q over every legal action (§ the label pipeline),
not an imitation of the pool, so a size no v7 member plays is still trained — it is simply not
part of the *opponent* distribution. Such a bin is floored to `EMPTY_BIN_PROB` rather than zeroed,
because `StyleParams.apply` masks illegal actions with `-inf` and subtracts the row max: a row in
which every legal action was `-inf` would come out `nan` instead of reaching the driver's "zero
mass on every legal action" assertion.

When the grids match — every gate config — the map is the identity and `build_pool` passes `None`,
so that path is bit-for-bit what it was before. A checkpoint whose config carries no
`game.raise_sizes` (only v7's older `table_bins`) cannot be aligned at all and is refused unless
the action counts already agree.

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

**Nothing in §5.7 has been run at size.** The range head, its target and its injection are
covered behaviourally on CPU at toy scale (`test_range_head.py`), and that is the whole of the
evidence: no corpus has been labelled with ranges at 2M hands, no belief has been trained past a
few gradient steps, and the two numbers that would decide whether it earns its keep — `range_kl`
against its own floor, and the agent's held-out gap with the head on versus `range_enabled:
false` — do not exist. The cost estimates in §2.4c are arithmetic over G3's measured rows, not a
measurement of this path. **The cheapest experiment that discriminates it** needs no new labelling
and no Slumbot run: train the embedding network twice on one corpus, `range_weight` at 0 and at
0.25, and compare held-out action CE, G1's transfer metrics, and — after `warm_start_trunk` — the
agent's held-out gap on labels already on disk.

**The G1 checkpoint no longer loads** (§2.8, D3, and again with §5.7). Extracting the trunk
renamed every parameter under it, and the range head adds parameters inside it, so a `state_dict`
saved before either change no longer matches `OpponentEmbeddingNet`. Nothing on the G1 path needs it — `g1_report.json`, `eval_corpus.pkl`
and `fitted_vectors.npz` are what post-hoc analysis reads — but re-evaluating those weights
would now need a key-remap shim, which does not exist.

**A past agent in the pool plays its unconditional policy *by default*, and that is a decision,
not a detail.** `FrozenAgentMember` seats it at `e = 0` — D12 option (a) — so the pool's own
agents do not *exploit* the tables they sit at while hero always does. `pool_agent_vectors:
"amortised"` (§2.10) removes that asymmetry: each past agent reads the `K = 0` amortised vectors
of its own tablemates, through the frozen embedding network of its own generation, refreshed on
the same block boundaries as hero's fit. The switch is **off in `config.json`** and has never
been run at size; the labels phase is the only phase it touches, because conditioning the
embedding corpus would make a player's style non-stationary within a session, which is what the
single-vector scheme assumes away (owner decision 2026-09-03, `PLAN_AMORTISED_POOL.md` P4).

What is unmeasured: whether a past agent is any *stronger* conditioned. `gates/pool_conditioning.py`
(§2.7a) is built and can answer it on any two checkpoints, and was deliberately not run — the
owner's argument is that an early pool is exploitable by construction, so reading the opponent
must help there, and that the interesting question is instead whether the advantage *decays* as
the pool converges.

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

605 tests, ~317 s on the dev box (CPU-only) — the parallel-evaluation cases spawn processes and
account for most of the increase, and the labelling cases now run the §2.2d control variate. The
30-minute budget from `CLAUDE.md` §4 is comfortably met; it was *not*, at over thirty minutes,
while the baseline settled every board through the engine's pot logic, which is the measurement
that sent it back to the drawing board.

| File | Covers |
|---|---|
| `test_engine_conservation.py` | Chip conservation through the engine (from v7) |
| `test_audit_stage0.py` | Engine invariants: `cumulative_bets` monotonicity, betting/street advance (from v7) |
| `test_solver_value_bet.py` | Solver value-bet pot construction (from v7) |
| `test_driver_lockstep.py` | **Lock-step ≡ sequential** (also with a pinned deck and a forced prefix), chip conservation through the driver, the v7 snapshot convention, the max-actions cap, every table size and stack depth, and the legality rule's corner cases |
| `test_rollout_plumbing.py` | **Replay identity**: a recorded hand replayed from its own deck and action sequence reproduces itself element for element, at every table size and both stack extremes; a forced replay issues zero policy calls; the deck override deals exactly what was asked and is refused if it is not a permutation; a partial prefix is replayed and the rest runs free with chips conserved; an illegal forced action raises naming the seat and the mask; the pending token adds exactly one action-less token, leaves every earlier token bit-identical, shows only the observer's cards, and is refused together with a showdown |
| `test_posterior.py` | The opponent posterior: a hand-computed two-decision example to `1e-12`; one batched policy call per opponent decision; **consecutive hero decisions reuse each opponent action once**, match a fresh posterior through street transitions, and report fewer actual rows; card removal relative to the observer (`C(45, 2)` on the river, the opponent's real holding still in the universe); every prefix length normalised; **a card-independent member leaves the prior exactly alone** and a card-dependent one does not; the posterior through *k* is bit-identical on a record truncated at *k*; an opponent who has not acted is the prior; a zero likelihood warns and falls back instead of returning NaN; `max_combos` caps, renormalises and is seeded, deliberately bypasses the cache, is spent **before** any member is asked (32-row batches, not 1081), draws the same combos under two different posteriors, reproduces the full posterior restricted to its draw, and recovers a functional of the full posterior to 0.02 over 200 seeds; `hole_override` changes the cards and nothing else |
| `test_oracle.py` | The BR oracle, every case exact rather than within a Monte-Carlo tolerance: `q[FOLD]` equals hero's own contribution to `1e-12` at every table size 2–9 and both stack extremes; `Q` equals an enumerated posterior-weighted sum on a fixture where hero's payoff is constant on the range's support and different off it; **hero is never handed a card it could not see** over ~900 rollout queries; every rollout conserves chips; illegal actions carry `nan` and the mask is the recorded one; the same seed gives a bit-identical label; the label is unchanged when the record is truncated at the labelled decision; **the runout is dealt per sample and the visible board is not** — every rollout replays the flop, the turn and river differ between samples, and no opponent is ever handed a card off the visible board or out of hero's hand; eight ranges inside three cards make every joint draw collide by pigeonhole and the label is `nan`; a heads-up river decision cannot collide; dropped samples reduce the divisor instead of counting as zeros; the forward count is a hand count |
| `test_observation_parity.py` | **The fatal invariant**: only the observer's hole cards, board never ahead of the street, no token carries its own action, scalars from the pre-decision snapshot, prefixes independent of what came later |
| `test_embedding_net_masking.py` | Causal within a hand, block-diagonal across hands, hand order irrelevant, the embedding is what changes the prediction, padding inert, **a showdown token cannot reach back into any decision**, the action loss ignores showdown tokens, both showdown heads reach the embedding, zero weights reduce the objective to action CE |
| `test_inference_fit.py` | The joint fit reaches the loss of the vectors that generated the labels, determinism, `K = 0` is the ablation, network weights untouched, cold start, regularisation, **the showdown terms reach the fitted vector** and zero weights reproduce the action-only fit |
| `test_observation_parity.py` (§5.1a part) | Showdown tokens exist exactly for the revealed seats and never among the decisions; **another seat's** revealed cards are the target and appear nowhere in its token, while **the observer's own showdown token carries the observer's own hand** — one rule across every token type, checked on decision and showdown tokens together; the labels match the cards shown, the two masks partition the real tokens, a showdown hand with no labels is refused |
| `test_strength_head.py` | The §5.6 poker prior and the §6.1 warm start: the target is the observer's own percentile on the final board, on the observer's own decision tokens and `-1` everywhere else, matching an independently enumerated value to `1e-12`; §5.1a's showdown labels are that same dict restricted to the revealed seats, so they did not move when the two passes merged; **a hand still in progress carries no target** and neither does a record nobody labelled; **it is a target and not an input** — the same hands tokenised with and without the label differ in `own_strength` alone and the action logits are bit-identical; `collate` masks the padded tail through the sentinel and selects exactly the labelled tokens; the weight shifts the total by exactly its term and 0 removes it while leaving the action CE untouched; a batch with no target is a batch and not an error; the head can actually learn the target, below the variance that is the only baseline it is read against; **the fit never sees the term** — `fit_embeddings` is bit-identical with and without it, and it did move, so that is not two no-ops; `first_retrain_steps` selects the first retrain only, leaves the agent's own key alone, and the trainer runs the count the iteration asks for; and the warm start copies the trunk key for key, leaves the action head at its initialisation, and is an initialisation rather than a tie — one gradient step moves the agent's trunk and not the embedding network's |
| `test_runout.py` | The rollouts' control variate, every case exact rather than within a tolerance: a hand with no decisions left reports the mean over **all 44 rivers**, checked by replaying it once per river through the untouched engine, while its raw result is a whole stack away; both branches of an exactly 50-50 fold, weighted, land on the true expectation to `1e-9` while the raw pair lands nowhere near it; on a complete board with nobody short the baseline **is** `Judger`'s settlement over random multiway states, and the one case where it deliberately is not — a side pot — is pinned as such; it is unchanged by chips beyond the call (call ≡ raise ≡ all-in) and matches the engine's own fold-out arithmetic on a hand the driver played out; cards a forced prefix turned over carry no correction; turning the estimator on moves neither a card, nor an action, nor a chip; the full-house kicker is the highest other rank, and the engine's evaluator and the batched one rank 16 000 showdowns identically on decks restricted to few ranks or few suits; and a street may define fewer raise sizes than another, with the legal set on each street matching that street's own list |
| `test_strength.py` | The per-board table: the combo grid is a bijection and `DISJOINT` matches an explicit check; **the river percentile equals `env/showdown.py`'s** for 50 random (board, holding) pairs — to 1e-6, the rounding being the reference's own float32 arithmetic; card removal moves some hand on a monotone board by more than two points and the moved value is what an explicit loop over the unblocked combos gives; a blocked combo is inert in every array; `range_equity` reproduces `hs` under uniform weights to 1e-12 and against a single-combo range pays 0 to everything it beats and 0.5 where the range is blocked out; hand classes on hand-written boards (top pair by kicker, overpair vs underpair on the same board, set and two pair on a paired board, straight/flush/full house); draws on named hands, no draw surviving the river, outs and `ehs` to the exact number, `ehs(1)` on the river **being** `hs`; texture buckets on the plan's named boards; the preflop integral's combo-weighted mean **is** `1/(n+1)` for every table size, pairs above suited above offsuit, AA between 0.80 and 0.90 against one opponent and falling with more, determinism in the seed and the on-disk cache short-circuiting a second call; `preflop_rank_pct` reproducing a hand-built ordering's cumulative combo mass exactly; and the cache building once per board, sharing a re-ordered board and evicting the oldest |
| `test_situation.py` | Reading a decision into scalars, every hand played along a scripted line so the numbers are arithmetic and not policy: the acting order and position fractions come out of the engine's own seat rules at 6-max and heads-up; a 6-max hand's live seats, preflop aggressor, limper count, players behind and preflop line per seat; a half-pot bet reads as `facing` 0.5 and `pot_odds` 0.25; barrels counted across streets and *not* counted for the preflop raise; a heads-up 3-bet pot's pot, effective stack and SPR to the chip; an all-in facing with its exact pot odds; all six preflop lines recognised on one hand; **the situation is a function of the prefix** — the finished record and the record truncated at the decision give the same dataclass at every decision of a hand; opening ranges widening with position, a 3-bet range polar with the premiums at full weight, a call range excluding aces, a seat yet to act carrying the whole range, and a percentile of the wrong shape refused |
| `test_hud_stats.py` | The stat line, every assertion a `(numerator, denominator)` pair countable by hand from the scripted line above it: an opener's and a 3-bettor's preflop counts, including that an opener never records a 3-bet *opportunity* and a seat folding to two raises does not either; a limp is neither a raise nor a fold and checking the option is not VPIP; a steal is an unopened pot from the cutoff, button or small blind — with no cutoff at three seats and the small blind being the button heads-up — and a limper ahead of the button ends the opportunity rather than making one; a c-bet, two barrels and the hands that faced them, with the aggression factor and aggression percentage separating bets from calls from checks; a check-raise and an overbet in one hand, and a half-pot bet not counting as an overbet; showdown stats counting who got there and who won; counts adding up over hands and over seats to the all-seats report; and an empty record list reporting zeros rather than failing |
| `test_regular.py` | The rule cascade, all of it through `policy()`: a batch equals the concatenation of single calls exactly; legal chip-conserving hands at every table size 2–9 and every depth 10–300 BB, every row a distribution over the legal actions; two members with the same numbers play identically and the style layer still moves them; preflop, aces always raise and 7-2 offsuit always folds, a button opens more combos than a first seat and the heads-up small blind opens over 70 %, a 3-bet range is polar with an empty calling gap between its two bands, a short stack shoves at its table's rate and a deep one never does, a shove is called with the calling range and not the jamming one, a crowded pot is entered less; postflop, a dry board is c-bet more than a wet one and **the air bets at exactly `f · bluff_ratio`** while the total mass does not (and cannot) equal the knob, a second barrel is its own frequency, betting into the aggressor is its own frequency and `donk = 0` means never, six players shrink the c-bet, **the river bluff share is the size's own indifference ratio** at `bluff_ratio = 1` and zero at 0, **the defence frequency is `defend_factor/(1 + bet/pot)` of hero's own range mass** at two sizes and three factors, a draw getting the right price never folds, facing an all-in is a price and nothing else, a low SPR turns value into a shove; and a 1 326-row posterior query on a cached board stays well under the 20 ms budget |
| `test_archetypes.py` | The ten presets and whether they are ten different players: ten distinct presets, a zero-spread draw *is* the preset, a full-spread draw stays inside every domain over five hundred draws, a knob that is zero stays zero (an archetype that never bluffs is not jittered into bluffing), sizes are shifted while rates are scaled with the right spread, the draw is reproducible from its generator, an unknown archetype is refused; then all ten at one nine-handed table for 1 500 hands, rotating through every chair, each read over the same hands against the same field on **the run's own raise grid** — how many pots they enter orders nit / TAG / bluffer / loose-passive / maniac, how they enter orders the passive ones below the aggressive ones with the limp frequency separating them, aggression orders them by ratios rather than levels, and each of the four added archetypes breaks its own regularity (the trapper checks where a TAG bets, the weak-tight folds where a loose-passive calls, the polar reg is the only one overbetting, the stealer gives up to a 3-bet); and **no two of the ten share a stat line** |
| `test_pool_realism.py` | The gate that reports those two things: both sections for every archetype and table size, the hands it says it played were played and the short and deep buckets partition them, a stat with no opportunities behind it comes back `nan` rather than as a fabricated zero — `af` being the case that shows why the numerator alone will not do — the strength section is a winrate with a standard error, the report is written and reloads, and the winrate is the hero seat's own chips over a hand-built set of records |
| `test_pool_style.py` | The five categories partition the action set, 32-scalar round trip, identity style is a masked softmax, position and street gating, temperature, uniform mix, every draw is a valid distribution over legal actions, each degenerate strategy does what it says |
| `test_v7_pool_member.py` | The vendored v7 stack constructs and plays legal hands; the v7 event format is built from the acting seat, masked to the street, and stops at its decision; **a `hole_override` reaches the network** — an override naming the real cards reproduces the plain answer, aces and deuce-trey do not, the record is untouched, and end to end a v7 opponent's posterior leaves the prior |
| `test_g1_gate.py` | The gate end to end: button rotation, uniform 2–9 × 10–300 BB, the four report sections, cold start ≡ `e = 0`, and that the standard error's unit is the session |
| `test_g3_gate.py` | The label-cost sweep end to end: one cell per point of the grid, every column the decision is taken on present and finite, forwards and rollouts monotone in the sample budget, the posterior's and the rollouts' shares adding up to the total, the bar reaching its total when a cell runs short of decisions, the split-half error finite and its gaps **signed on both sides**, the pot-unit error recomputed per label from the stored rows, the hero seat being the configured member and not whoever sat there, the five profile buckets summing to the label's wall clock with the counters reset per label and the wrapping undone on exit, the headline built from the exact-posterior cells only, and a pinned table size that a hand cannot quietly leave |
| `test_agent_net.py` | The agent end to end: one row of logits per hand and every padded position inert; each row answers from **its own** last real token and no hand moves another; through the driver, at every table size 2–9 and both stack extremes, a valid distribution over legal actions and chips conserved; the observation obeys §9 parity along the agent's own call path — only its own cards, board never ahead of the street, no showdown token, the pending token action-less — and the observation does not grow as the record does; with `e = 0` permuting the players is bit-identical and a non-zero vector is not; determinism, weights untouched by a `policy` call, and both parity guards refusing what they are meant to refuse |
| `test_label_generation.py` | Label generation end to end, including **resume**: a run stopped in the middle of its second shard comes back with the same label set shard for shard and byte for byte; **the resumed call deals exactly the unlabelled hands and no others**, deals nothing at all into a finished directory and still hands back the same manifest, and holds when the replay is *not* the same play — hero jams every replayed hand and the labels on disk are still the finished set's prefix, no decision labelled twice, no hand labelled on both sides of the join; a directory written for different sessions, one whose `seed_base` moved, and one whose `play.json` is gone are each set aside under `.stale` and relabelled from scratch rather than spliced into, and a second stale directory does not overwrite the first; a shard is cut at a hand boundary, so no hand's labels span two files; a toy run whose every label is a valid distribution over the environment's own mask with `nan` exactly off it; hero is slot 0 and every hero decision is labelled once; an ordinary pool member works as hero (iteration 0, §7.1); **the embedding of a block ignores every later hand** — hero jams from hand `R` on and the block's vectors come out bit-identical anyway, while block 0 is the zero cold start; the stored prefix stops at the labelled decision, carries only hero's cards and a board never ahead of the street; the same seed writes byte-identical shards and a shard round-trips; the table draw is the exact uniform multiset of a fixed seed; and the session machinery is shared with G1 rather than copied |
| `test_pool_sampling.py` | Pool sampling: a fixed history and seed produce an exact sequence; results accumulate across sessions of different lengths into a mean and the mean into a weight, with a pool of no results and a pool of no spread both flat, and the **magnitude** of a loss — not its sign — moving the weight; a member hero beats the most is reached **only** through the floor — never at `floor_fraction = 0`, every draw uniform at 1; forgetting leaves an unsampled member's estimate exactly where it was and is what lets a member the early agents crushed climb back to the top PFSP weight at all — at `result_decay = 1` it is still winning after forty iterations, at 0.8 it crosses at the twelfth and at 0.5 at the fifth; a member nobody has played is drawn immediately; clustering recovers a hand-built structure and a ten-member blob of near-duplicates does not crowd out a lone style, while a duplicate pair splits one cluster's share; more clusters than members is no clustering; the state round-trips and reproduces the next draw, survives a pool that has since grown by one member and refuses one that has shrunk; and every table size 2–9 gets one member per non-hero seat |
| `test_targets.py` | Targets, loss and the training cycle: a hand-computed softmax to `1e-12`; exact zeros off the mask and what sits under it never read; the two temperature limits reached in float, not approached; one legal action, equal EVs, no legal action, a `nan` under the mask; **the v7 scar** — two situations differing by a factor of 30 give the same target to `1e-12`, and without the divisor one is near-uniform while the other is near-deterministic; the KL is exactly zero on a match, positive off it, blind to illegal logits, and its gradient reaches the logits and not the target; **the linear loss** — the two losses share a minimiser, averaging the gradients at `Q ± ε` reproduces the gradient at `Q` to `1e-12` for `soft_q` and demonstrably not for `kl`, a constant added to every legal EV changes neither value nor gradient, illegal logits are ignored, a label off the mask is refused, and a toy run reaches its target; dropout at `p = 0` and `p = 1`, reproducible from its generator, and per hand per slot rather than per token; a toy run that reduces the loss, is deterministic, leaves the pool and the embeddings untouched and refuses a hand that is not a pending decision; and the cycle — the first iteration runs its own step count, a later one opens from the weights the previous one left, and three cycles in a row keep improving |

| `test_range_head.py` | §5.7 end to end: **the filter is the posterior** — every prefix of a played hand matched combo for combo against `opponent_posterior`, with the one difference (the board visible at the token, one street ahead of a posterior conditioned through the previous decision) applied to the posterior as card removal; **the target is a prefix function** — truncating the record after `t` leaves every belief at or before `t` bit-identical; a player is in the target at the token they fold and never after; no target puts mass on a combo the board or the observer's own hand blocks, and the head's own mask is the support that target lives on; pruning buys forwards, reports the mass it dropped, keeps the mode, and only bites against a member whose play depends on its cards; the target reaches the batch on exactly the rows `act_idx` names and every untargeted row is zero and masked; the head answers one row per active pair with `-inf` on the blocked combos; **the belief at a token cannot read a later one** — perturbing the last token leaves the earlier beliefs unmoved; **`range_enabled: false` is an exact ablation** — the same hidden states tensor for tensor and no `range` parameter left behind; **the injection is detached** — the action loss reaches nothing upstream of the bottleneck; the term is exactly zero when the prediction is the target and the cross-entropy still pays the target's entropy; a batch with no target scores nothing rather than crashing; the weight is what puts the term in the total; the agent reads the same head at its pending decision; and a whole toy iteration runs with the head on — corpus targets, label targets, the sparse target through a shard and back, and both trainers reporting `range_ce` and `range_kl` |
| `test_pipeline.py` | The outer loop end to end at toy scale, including that **a crash in the middle of labelling costs a shard and not the phase** — re-running produces the artefacts an uninterrupted run would have left: two iterations run to completion and write every artefact of every phase; the pool grows by exactly `style.agent_variants` members per iteration, variant 0 unmodified and the rest style draws, with the embedding table reserving `len(pool₀) + max_iterations × agent_variants` rows; **iteration 0 seats the `agent_init` member and iteration 1 seats the agent**, asserted from who was actually asked for an action; the seven §8 gap numbers computed by hand, including that `ev_gap_greedy` is zero exactly when the agent's mass sits on the oracle's best action; the held-out slice reaches the metric and never the optimiser, and `heldout_fraction = 0` reports **no gap** rather than one on training data; the split is a partition, deterministic in `(seed, iteration)` and different between iterations; **a run resumed from a crash in the middle of an iteration reproduces an uninterrupted one** — the same shards byte for byte, the same weights tensor for tensor, differing only in wall clocks; table size and stack depth span 2–9 and 10–300 BB with no weighting; and a past agent in the pool answers `hole_override` (two holdings, two answers, a posterior that moves off the prior) while observing only the moment it was asked about |

| `test_label_generation.py` (the conditioned pool, part 8) | `pool_agent_vectors`: **off is off** — a run with the keys absent and one with them present-but-off write byte-identical shards, the window included, and the stored tables stay one observer wide; with it on, a past agent's table for block *b* equals the `K = 0` reading recomputed from the hands the phase actually played, from *its* seat, through *its own* network, for every session, block and slot, while every other slot's row stays zero and block 0 is cold for everyone; the table of a block **ignores every later hand** — hero jams from hand 3 and blocks 0 and 1 come out bit-identical; two runs differing only in the network the past agent carries produce different tables and different labels while hero's own fit is untouched, which is what says the phase reads the member's generation and not the loop's; a member carrying no generation is never conditioned and its run is the switch-off run; the window really caps (equal at one hand of history, different at two); a directory played under the other setting is set aside; and a conditioned run resumes into the run it would have been, reading the stored tables and replaying only the unlabelled tail |
| `test_parallel_labels.py` (the conditioned pool) | A conditioned past agent labelled sequentially and in two workers produces the same labels and the same shard bytes — the tables are computed once in the parent and shipped, and each worker reseats the member per block exactly as the sequential loop does, which is what the §7.2 posterior requires |
| `test_pipeline.py` (generations, part 7) | Every past agent is given a **frozen copy of the embedding network of its own generation**, weight for weight the one that iteration wrote to disk, with no trainable parameter left; iterations that shared a retrain share one object and iterations that did not are given different ones; a resumed run rebuilds the same mapping from the checkpoints, so the pool does not change under a restart; with the switch off no snapshot is kept at all; and the whole loop runs with the pool conditioned, seating and labelling against a conditioned member from the first iteration that has one |
| `test_pool_conditioning_gate.py` | The P2 gate end to end at toy scale: it runs, stamps its report, writes it, and buckets every session by table size and by stack depth exactly once; **the two conditions play the same tables** — the difference is per session and each row carries one table and one stack; **with the vectors forced to zero the paired difference is exactly 0.0** in every session, which is the pairing with the effect removed, and **a table loud enough to move the policy does move the hands**, which is what stops that from being vacuous (at toy scale the real `K = 0` vectors shift the policy by ~5e-4 and flip nothing); the paired mean and standard error are the hand-computed ones, per condition and for the difference; and a report is refused if it loses the stamp or any standard error, as is a checkpoint trained on another action set or another trunk |
| `test_frozen_agent_vectors.py` | A past agent conditioned on its tablemates (`PLAN_AMORTISED_POOL.md` P1): no table is the member that was always there and a **zero table is bit-identical to it**; a table without a slot, a slot without a table, a mis-shaped table and a slot off the end are each refused; `with_vectors` shares the network and the style by identity, leaves the base's answers and the base's table alone, and its table demonstrably reaches the forward; **the rotation a member derives is the session's own** at every table size 2–9, every slot and every hand, asserted on the slots it labels the seats with; the same tablemates relabelled from another slot are the same table — different tokenisations, identical answers; a row for a slot nobody occupies changes nothing while every seated slot's row changes the answer; the two contracts a past agent already had still hold with a table — `hole_override` gives two holdings two answers and the observation stops at the asked-about decision carrying only the hypothetical holding; slot 0's tokens are byte-for-byte what `Session.tokens` always produced, **every other slot shows that slot's cards and nobody else's**, a window is the last hands and nothing earlier, and the §5.7 targets are refused to every observer but hero; and `amortised_vectors` equals `fit_embeddings(steps=0)` padded, is zero for a slot that has observed nothing and for an empty window, differs between observers, and shrinks with the window |
| `test_slumbot_adapter.py` | The Slumbot seam, entirely off canned action strings — no socket: the mask hero acts under **is `env.legal`'s** on every canned state and reaches the token unchanged, folding is offered facing the blind and refused with nothing to call, and an opponent's all-in leaves no raise; **the chips are Slumbot's and not the abstraction's** — a `b250` that lands on no bin of ours still reads as a 5 BB pot, and every canned state's pot, stack and amount-to-call match the wire to 1e-9; the two seat frames are mirrors, hero holds hero's cards, the first decision of a hand belongs to seat 0 and the first of the flop to seat 1; §9 parity on the built record — only hero's hole cards, a board never ahead of the token's street, no showdown token, the pending decision action-less, and a prefix independent of what came later; **index → wire → index round-trips for every legal action on all four streets with no clamp firing**, while a fold with nothing to call becomes a check, says so in the counter, and is read back as a call; hero's own clamped action is what the next replay sees, and `hero_action_indices` overrides hero's seat only; BB/100 and its standard error against hand-computed values, with Welford's online form agreeing and the zero- and one-hand cases returning zero; a table outside `players_range` / `stack_bb_range` is **refused, not clamped**; and end to end the agent answers every canned state with a legal token, deterministically under its own generator, cold from the zero table and warm from a fitted one, reaching the network through the ordinary `AgentPoolMember` and not a copy of it |

| `test_eval_pipeline.py` | The evaluation runner against a canned Slumbot that speaks the real grammar through `evaluation/protocol.py` — no socket anywhere; the parallel path really spawns processes, which reach the stub by name: shares add up to the hand count for every split, each worker plays its own share into its own file and the aggregate is their sum, a parallel run resumes per worker, and the clamp counters survive a resume because they ride on the hands rather than being tallied; BB/100 and its standard error hand-computed, agreeing with the batch form, with the zero- and one-hand cases returning zero rather than crashing; **a short run is stamped `SCREENING ONLY` and a run at the threshold is not**, and the selection disclosure is refused when absent or incomplete rather than quietly omitted; **cold pins `e = 0` at every decision** — asserted from the vector the member actually held — and never fits anything; warm refreshes exactly on the configured `R`, records per hand how many hands its vector was fitted from, buckets the run by it, and a fitted vector demonstrably reaches the policy; the warm-up hand count is the first bucket that caught cold up and `None` when it never did; **warm worse than cold is reported in §12's own words**; **a resumed run reproduces an uninterrupted one** — the same BB/100, the same standard error, the same per-bucket curve and the same hands byte for byte; a failed hand is counted, keeps its slot in the file so the index does not shift, and does not enter the statistics; and both modes off, or warm with no embedding checkpoint, are refused; **a procedural pool member can play the hands instead of the agent** — no agent checkpoint is loaded or required, the run says out loud which member is playing and that it is not an agent result, the warm run is dropped with a reason rather than failing one process deep (and a warm-only run is then refused outright), a member run resumes like any other, an unknown hero kind is refused, and **omitting the hero section reproduces the agent's own report cell for cell** |

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
      state.json                    sampler state after phase E; odd iterations skip updates/decay
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
