# v8 — Concept

**Status: agreed with the owner 2026-08-16. Gate G1 (§14) is implemented; the agent, the oracle
and the outer loop are not.** All open items (OI-1 … OI-8) are resolved — see §16, where OI-2 is
recorded as revised after implementation showed the original not to be implementable. Nothing in
this document is waiting on a decision. `ARCHITECTURE.md` §4 lists the readings that turning this
document into code required.

This document is the design record for v8. It supersedes nothing in the root `CLAUDE.md`
(project-wide rules) and it will be complemented — not replaced — by `ARCHITECTURE.md`, which
describes what the code actually does once it exists. Where the two disagree, `ARCHITECTURE.md`
is the truth about the code and this file is the record of what was intended and why.
Literature citations in §17 were verified against primary sources on 2026-08-16.

Read `CLAUDE.md` §1 first — the goal, the training distribution, and what is forbidden.
`versions/v7/ARCHITECTURE.md` is read-only reference for the previous architecture.

---

## 1. Summary

v8 drops the 6-phase GTO-supervision + MCTS-self-play structure of v7 entirely. In its place:

> Fix a **pool** of opponent strategies. Against a fixed pool the game stops being an
> imperfect-information game and becomes a stationary single-agent POMDP for hero. In that
> POMDP, compute the exact action values of hero's decisions with a rollout oracle that is
> allowed to read the opponents' strategies, and train the agent on them. Condition the agent
> on a **per-opponent embedding** inferred from observed hand histories, so that one network
> can carry a different response for every opponent. Add the trained agent to the pool and
> repeat.

Five entities:

| # | Entity | Role | Origin |
|---|---|---|---|
| 1 | **Environment** | poker rules, betting, side pots, showdown | reused verbatim from v7 (`env/`) |
| 2 | **Opponent pool** | fixed strategies that can act in any situation | v7 networks + procedural style modifiers + degenerate strategies; later, v8 agents |
| 3 | **Agent v8** | history + opponent embeddings → action | new |
| 4 | **Opponent-embedding network** | hand histories of a player → one vector describing that player | new — the central idea |
| 5 | **BR oracle** | opponents' strategies + game state → EV of each hero action | new |

The agent is trained by supervised regression onto oracle targets. There is no CFR, no MCTS,
and no search at deployment (baseline). One forward per decision.

### The bet

Two bets, and they are separable — one can succeed while the other fails, and the experiments
in §14 are designed to tell them apart.

- **B1 (embedding).** A transformer trained to predict a player's actions, with cross-hand
  information forced through a single fitted vector, produces a vector that (a) actually
  carries style, and (b) **lands somewhere sane for a strategy that was never in the training
  pool**. B1(b) is the load-bearing assumption of the whole project: Slumbot is exactly such a
  strategy.
- **B2 (oracle).** A best-response-to-a-known-pool target is a good enough learning signal that
  iterating it produces a strong general agent, despite never optimising for equilibrium.

---

## 2. Why fixing the pool is the load-bearing simplification

With opponents fixed, hero faces a stationary POMDP. Three consequences, all of them the
reason this architecture is simpler than v7's:

1. **Monte-Carlo evaluation of a hero action is unbiased.** Sample the opponents' hidden cards
   from their true posterior (§7.2), roll the hand out with hero playing its current policy and
   opponents playing theirs, average. No CFR, no counterfactual reach weighting, no unsafe
   subgame solving. The estimator's only error is Monte-Carlo noise plus the approximations
   listed in §7.3.
2. **Training on `softmax(Q/T)` is a sound operator.** It is one step of soft policy
   improvement over the current hero policy (the rollout policy *is* the current agent, §7.1).
   Iterated, this is soft policy iteration against a fixed environment — a convergent scheme,
   unlike best-response iteration in a game.
3. **Multiway is not a special case.** "Best response to a fixed set of others" is well defined
   for any number of opponents. Nothing in this design degrades from 2 players to 9; the
   equilibrium concepts that break in multiplayer poker are simply never invoked.

What we give up is stated honestly in §11.

---

## 3. Entity 1 — Environment

`env/` as inherited from v7 (`Table`, `Judger`, `SimpleDealer`), already present in `v8/`.
Game rules do not change with the agent architecture.

**One required addition, and it is the largest single piece of engineering in v8:** a
**lock-step vectorised driver** over `Table`. The oracle (§7) issues on the order of 10⁴ policy
queries per training label; if hands are simulated one at a time, every query is a batch of
one and the GB10 is idle. The driver must advance N independent hands in lock-step, collect
every pending policy query across all N into one batch, run one forward, and scatter the
results back.

This is a driver *around* `Table`, not a rewrite of it — the engine's step semantics, chip
accounting and side-pot logic stay exactly as they are and stay covered by the existing
conservation tests. Cost of getting this wrong is silent EV bias, so it gets its own
end-to-end equivalence test against the sequential path (§15).

---

## 4. Entity 2 — Opponent pool

A pool member is anything exposing

```
P(action | observation) → distribution over the discrete action set
```

for an arbitrary legal game situation, batchable on GPU.

### 4.1 What is in the pool

**Composition at bootstrap:**

- **v7 networks.** Existing trained checkpoints from `data/v7/`, run through a **vendored frozen
  copy** of v7's agent code (§4.3). Used with `heads={"action"}`, no MCTS and no opponent
  embedding.
- **Procedural style variants.** The same base network with a randomly sampled style modifier
  applied to its output distribution (§4.2). This is the cheapest source of pool diversity by a
  wide margin and it is what keeps the embedding space continuous instead of a 20-way
  classification (§11.3).
- **Degenerate strategies.** always-fold, always-call, always-min-raise, maniac (all-in biased),
  nit (folds without a strong hand). No network, no cost, and they pin the corners of the style
  space that iterated agents will never visit on their own.

**Later:** every trained v8 agent joins the pool. Expected size 500–2000 members at late
iterations, bounded by compute, not by design.

**Backup option (recorded, not baseline):** add v7's CFR solvers (`gto_utils/gpu_solver_v5.py`)
as pool members. Rejected for the baseline because v5 is a CPU numpy sweep costing seconds per
situation — unusable inside rollouts. If solver-quality play is wanted in the pool, the correct
form is to distil a solver into a network **once, offline**, and put the network in the pool.

### 4.2 Style modifiers — live, on the output distribution

**Status: approved by the owner 2026-08-16 (§16, OI-5).** New low-level logic, agreed before
being written per `CLAUDE.md` §5.

v7's `agent/train_scenarios/modifiers.py` rewrites *solver EVs offline in a dataset*
(`modified_ev[i] = ev[i] + |ev[i]|·factor`). v8 needs the opposite: a modifier applied *online*
to a live network's output. The v7 formula does not transfer — pool members emit logits, not
EVs, and `|ev|·factor` has no meaning on a logit. What transfers is v7's **action
categorisation** (`resolve_actions`: fold / call / small raise / big raise / all-in), and that
is reused verbatim.

Proposed form — an additive **logit bias plus temperature plus a uniform mix**:

```
p = (1 − λ) · softmax( (logits + b(s)) / T )  +  λ · uniform_over_legal
```

`b(s) ∈ R^{n_actions}` is assembled from a per-opponent style parameter vector, broadcast from
the 5 action categories to their member actions:

| Block | Size | Gated on | Cost |
|---|---|---|---|
| unconditional | 5 | — | free |
| per position | 5 | acting position bucket | free — position is in the state |
| per street | 5 × 4 | street | free — street is in the state |
| `T`, `λ` | 2 | — | free |

≈32 scalars per pool member, sampled once at pool construction from a config-specified
distribution and frozen. **Zero extra forwards, zero extra parameters, no extra network** — the
whole style space is a bias vector added before the softmax the pool member already computes.
One random draw = one style, so the opponent space becomes continuous and effectively infinite
at zero marginal cost. This is the answer to "how do we expand the opponent space cheaply": not
more checkpoints, more draws.

Additive bias in logit space is multiplicative reweighting of the distribution — well behaved,
sign-symmetric, and composes cleanly with temperature. The uniform mix `λ` is included because
logit bias alone reaches "loose / random" styles only at extreme magnitudes.

**Deliberately dropped from v7: `equity <` / `equity >` conditions.** They require an equity
evaluation (`gto_utils.gpu_solver_v2.gpu_equity_v2`) at every decision, inside the innermost
rollout loop, which would be the single most expensive item in the oracle. Street and position
gating buys most of the same behavioural variety for free. If hand-strength conditioning is
wanted later, the cheap surrogate is a preflop hole-card rank-class bucket — a lookup, not a
computation.

Applies uniformly to any pool member, v7 or v8.

### 4.3 Bootstrap and vendoring (owner decision)

v7's agent architecture code is **copied into v8 as a frozen snapshot** under its own package
(so nothing under `versions/v7/` is touched and no cross-version import is attempted — both
trees define the same top-level package names). The copy is never edited to follow v8 changes.

What must be vendored is not only the model: a v7 checkpoint needs **v7's event format** to be
fed anything. That format is already present in v8 — `evaluation/slumbot_eval.py::_build_events`
constructs it (`hand`, `num_players`, `hero_pos`, `acting_pos`, `big_blind`, `small_blind`,
`stack`, `stacks`, `bets`, `table`, `action`) and is the reference implementation. So a v7 pool
member is: vendored perception + action head, v7 event builder, `skip_opponent_emb=True`,
`heads={"action"}`.

Checkpoint paths are config, pointing into `data/v7/` (owner supplies them). The `bootstrap`
config section carries, per entry: checkpoint path, number of style variants to draw, and
optionally an explicit style parameter vector (used for the hand-designed degenerate styles).
Random draws are the default; explicit lists are the override.

### 4.4 Sampling the pool

Uniform sampling over a growing pool wastes almost all compute on opponents the agent already
crushes, and it is also the configuration in which fictitious play is known to be weakest
(§11.1). Three mechanisms, all config-driven:

- **PFSP** (prioritised fictitious self-play, AlphaStar): sample opponent *i* with probability
  ∝ *f*(loss rate against *i*), so hard opponents dominate the batch.
- **Embedding-space deduplication**: cluster the pool by its own opponent embedding vector
  (entity 4) and sample within clusters rather than over the raw list. Late-iteration
  checkpoints of one lineage are near-duplicates and should not each get a full share.
- **Uniform floor**: a fixed fraction (order 10–20%) sampled uniformly over the entire pool, so
  old styles are never fully evicted.

This is also the answer to "how do we filter the pool retroactively": nothing is deleted,
things are down-weighted.

**Table configuration is sampled independently and uniformly** — 2–9 players, 10–300 BB
effective stacks, per `CLAUDE.md` §1. No weighting toward heads-up or 200 BB anywhere.

---

## 5. Entity 4 — Opponent-embedding network

Described before the agent because the agent consumes its output.

### 5.1 Situation token

One token per **decision** in a hand — every decision by every player, not only the target
player's. Features, concatenated and passed through an MLP to `d_model`:

| Feature | Notes |
|---|---|
| decision index within the hand | ordinal, resets each hand |
| 7 cards (5 board + 2 hole) | `Embedding(53, ·)` per slot, index 52 = unknown. Unrevealed board streets masked. **Opponents' hole cards always masked** in a decision token — the showdown exception lives in §5.1a, where it is a token of its own |
| acting player's stack | in BB |
| pot | including all bets on the current street |
| amount to call | in BB |
| acting player's position | |
| **number of players at the table** | owner decision; the 2–9 axis is not otherwise recoverable from a single token |
| **per-seat stack vector** | owner decision; length `max_players`, in BB, zero-padded. v7 found this necessary — "a per-seat effective-stack signal, without which stack-aware play is unlearnable" (`versions/v7/ARCHITECTURE.md` B.6.2) — and v8 must cover 10–300 BB across 2–9 seats |
| previous action of the hand | the action taken at the previous token |
| **acting player's embedding** | the fitted vector, §5.3 — identical across all tokens of that player |

**The masking rule is not a detail.** At deployment hero does not know an opponent's hole cards.
If the network were trained on true cards it would learn to predict actions from cards, the
fitted vector would carry nothing, and the input distribution at inference would be one the
network never saw. Tokens are built **strictly from the observer's information**; everything
else in the list (stack, pot, call, position, board, actions) is public and is taken from the
acting player's seat, which is legal.

Everything monetary is in BB.

### 5.1a Showdown token — the reveal anchor

**Status: agreed with the owner 2026-08-16 (§16, OI-8), after the rest of this document.**

A showdown is the only place where a player's line is tied to an actual holding, and it is
therefore the sharpest style signal available. With one token per decision it would never be
seen: every decision precedes the reveal, so §5.1's "except in tokens strictly after the
reveal" would have nothing to apply to.

So the reveal gets a token of its own. A hand that reaches showdown appends **one terminal
token per revealed player**, after every decision token of that hand. The token carries the
final board, the revealed player's embedding, the end-of-betting stacks and pot, and a token
type distinguishing it from a decision. It carries **no cards of the revealed hand** — those
are its *targets*:

| Head | Target | Loss | What it is for |
|---|---|---|---|
| strength | exact percentile of the revealed hand on the final board, ties at a half | MSE | board-relative range calibration: how strong was this player's holding on *this* runout |
| class | the 169-way preflop class of the revealed hand | cross-entropy | board-independent style: what this player shows up with |

The percentile is computed by **exact enumeration** over the 990 combos that do not conflict
with the board or the hand itself, not by Monte Carlo — it is cheaper here than v7's 256-sample
estimate and needs no seed. The label is observer-independent: other players' holdings are not
removed from the enumeration even when the observer saw them at the same showdown, because a
label that depended on who was watching would give one hand several labels.

**Why a terminal token and not cards written into the decisions.** By the time an embedding is
fitted, the hand is over and the observer legitimately knows the revealed cards, so filling them
into that hand's decision tokens would not violate §9 as stated. It would still be wrong: it
tells every decision token that this player *reached* showdown — that they were not going to
fold — which is a future leak that lowers the prediction loss while carrying no style at all.
Attention is causal within the hand, so a terminal token attends to every decision and **no
decision attends to it**; the action-prediction task is left bit-identical.

**The showdown terms belong in both objectives, and the second one is easy to miss.** In
training they shape the weights. In the **inference-time fit** (§5.5) they must also be part of
what the vector is optimised against — otherwise the fit optimises action cross-entropy alone
and pulls the vector straight off whatever the showdowns said. v7 could use a training-only
probe (`ShowdownStrengthProbe`, `PLAN_OPPONENT_ADAPTATION` §4) because its embedding was a GRU
state produced by a forward pass; v8's is a latent found by gradient descent at deployment, so
the information has to be in *that* gradient. Both terms are computable at fit time: the hands
being fitted are finished and their showdowns are public.

Weights for the two heads are config, used identically in training and in the fit; setting them
to zero is the ablation that asks whether the anchor earns its keep.

Two properties recorded rather than fixed:

- **Selection bias.** Showdowns are observed only where the player did not fold and was called,
  so the heads learn `P(holding | line, showdown reached)` — a biased slice. It is the same
  slice a human with a HUD sees, and it is all that is available.
- **The signal is only as informative as the pool.** A pool member whose policy ignores its
  cards makes its showdowns unpredictable by construction, and both heads correctly fall to the
  marginal on it. The anchor carries content in proportion to how card-dependent the pool is.

### 5.2 Attention mask — block-diagonal

The history of one player is a concatenation of many hands. The attention mask is
**block-diagonal**: within a hand, a standard causal mask; across hands, zero. Causality within
the hand is mandatory — without it the token at *t* sees the action taken at *t* through
token *t+1*'s "previous action" field, and the prediction task is trivial.

Cross-hand attention is cut deliberately. If it were open, the transformer would infer the
player's style in-context from the prefix and the embedding would receive no gradient pressure
at all — the classic shortcut, and one that hides itself because the prediction loss looks fine.
With the block-diagonal mask **the embedding is the only channel between hands**, which is
exactly its role at deployment.

Consequences, all accepted:

- The network is effectively a **per-hand encoder**; hands are conditionally independent given
  the embedding. Hands are a batch dimension, not a sequence — cost is linear in the number of
  hands, subsampling hands per gradient step is legitimate, and there is no long-context
  problem.
- **No global positional encoding.** The token already carries the decision index within the
  hand; RoPE across hand boundaries would be meaningless.
- The model is a clean latent-variable model, `P(actions of hand | hand, e)` with a factorised
  likelihood over hands. Fitting `e` by gradient is a MAP estimate of the latent; combined with
  the amortised head (§5.4) this is semi-amortised inference (Kim et al. 2018).
- **The model cannot represent a player whose style changes over time.** All hands are
  exchangeable. Accepted: pool members are fixed policies and Slumbot is static.

### 5.3 Per-player embeddings and the forward pass

**One forward per hand covers all players.** Each player's tokens carry that player's own
embedding; the loss is the cross-entropy of the predicted action against the real action, summed
over every player's decision tokens. Each `e_i` receives gradient only from player *i*'s
decisions. This is ~N× cheaper than one forward per target player and it matches deployment,
where hero holds embeddings for all opponents simultaneously.

**Hero has an embedding too**, fitted like everyone else's.

**Consequence:** the embeddings of the players at a table are coupled — predicting player *i*
depends on the others' vectors. At inference they are therefore **fitted jointly**, one gradient
on the concatenated set of vectors (§5.5), not one at a time.

### 5.4 Training

Baseline (owner decision): **no inner loop during training.** Per-player embeddings are an
ordinary trainable table, one vector per pool member, optimised jointly with the transformer
weights by the same optimiser. The gradient-descent fit exists only at inference.

This is the cheapest of the three options (the alternatives — full MAML through the inner loop,
or first-order approximations — are recorded here and not implemented). Its known weakness is a
train/deploy mismatch: at training the vector is fully converged, at deployment it is *K* steps
from its initialisation. Two mitigations, both agreed:

- **Amortised head.** A small network mapping an observed history to a starting vector, trained
  by MSE distillation onto the converged table entry for that player. At inference its output is
  the initialisation and gradient refines it. Without the distillation target the head has no
  training signal at all under the no-inner-loop baseline.
- **Ablation, mandatory:** embedding = mean-pooled hidden states of the transformer over the
  player's tokens, no gradient fit at all. If this matches the fitted vector, the entire
  inference-time optimisation is unnecessary and is removed.

**No embedding dropout when training the embedding network** (owner decision). Dropout applies
only to agent training, where it has a different purpose (§6.2).

**History used to fit a player's embedding: only hands in which the observer was seated.** This
mirrors deployment exactly. In heads-up against Slumbot that is every hand; multiway it is the
observer's own table.

### 5.5 Inference-time fit

Given the histories observed so far:

1. Initialise every player's vector from the amortised head.
2. Run *K* gradient steps on the vectors only (transformer weights frozen), minimising the same
   objective the network was trained on — action-prediction cross-entropy **plus the §5.1a
   showdown terms** — over the observed hands, all players jointly. One implementation of that
   objective, used by both, so the two cannot drift apart.
3. Recompute every *R* hands; between recomputations the vectors are stale. Cold start (no
   history) is the zero vector.

*K*, learning rate, regularisation toward zero, and *R* are config, to be found experimentally
(owner decision — no prior preference). The fit is cheap: it touches only `n_players × d`
parameters, and the block-diagonal mask makes the per-hand forwards a batch.

---

## 6. Entity 3 — Agent v8

### 6.1 Shape

Input: the current hand's history in the **same token format** as §5.1, from hero's own
perspective, with each opponent's fitted embedding attached to that opponent's tokens and hero's
own vector on hero's tokens. Output: a distribution over the discrete action set at the current
decision.

Action set reused from v7: `[fold, call, raise_0 … raise_{bins-1}, all-in]` with per-street
raise fractions from config, `n_actions = len(raise_sizes[street]) + 3`.

**No value head** (owner decision) — the baseline oracle (variant A, §7.1) does not bootstrap, so
none is needed. Note that variant C *does* require one; see §7.4.

**No search at deployment** — one forward per decision. Re-introducing a search phase in the
spirit of v7's phase 6 is recorded as a later experiment, not part of the baseline.

**Relationship to the embedding network (owner decision, §16 OI-4):** the two share the
**tokeniser implementation** — one class, since both consume the §5.1 token and a second copy is
exactly the duplicated low-level logic `CLAUDE.md` §5 warns about. They do **not** share weights:
they optimise different objectives on different retraining cadences, and shared weights would
make every embedding-network retrain silently change the agent's input representation.
Warm-starting the agent's trunk from the embedding network is a config flag, off by default.

**Initialisation: from scratch** (owner decision 2026-08-16, §16 OI-2 revised). An earlier draft
said "from a v7 checkpoint, action head only". That is not implementable and the reason is this
section itself: the agent consumes the §5.1 token through the tokeniser of OI-4, while a v7
checkpoint's weights are shaped for v7's input — seven card vectors per *event* through
`combine: Linear(8·d → d)`, an encoder over N×7 positions, a mean-pool to N, then a decoder.
There is no correspondence between the two parameter sets, so "loading" a v7 checkpoint into a
v8 agent could only mean loading `output_proj`, the last layer of a different trunk, which is
noise. What OI-2 was buying is bought elsewhere — see §7.1.

Config section `agent_init` therefore selects **which pool member occupies hero's seat for
iteration 0**, not which weights the agent starts from.

### 6.2 Target and loss

Target: `softmax(Q_normalised / T)` over legal actions, `Q` from the oracle. Loss: KL.

**Normalisation is mandatory and is where v7 has scars.** Raw EVs in a 300 BB pot and a 10 BB pot
differ by more than an order of magnitude; a single temperature applied to raw EVs produces a
near-deterministic policy in big pots and a near-uniform one in small pots. v7's 97 % fold rate
was a normalisation bug of exactly this family, not an architecture failure
(`versions/v7/ARCHITECTURE.md`, "MCTS value-target normalization"). Baseline: divide by
`pot + facing_bet`, with the divisor itself a config choice.

Illegal and dominated actions are masked before the softmax, using the same legality rule as the
environment and the data generator — one implementation, not two.

**Embedding dropout: yes** (owner decision). During agent training, opponent embeddings are
zeroed with some probability. This forces the agent to learn a usable **unconditional** policy
under `e = 0`, which is the policy it will play against any opponent it has not yet observed —
including the first hands against Slumbot, and including any case where the embedding is
uninformative. Without it, `e = 0` means "population average" at best and "arbitrary" at worst,
and the agent would open against an unknown opponent with a counter-strategy aimed at somebody
else. This is the cheapest available protection for the failure mode described in §11.2.

---

## 7. Entity 5 — BR oracle

### 7.1 Variant A — full rollout (baseline)

At a hero decision point, for each legal action *a*:

1. Sample a joint assignment of hole cards to the opponents from their posterior (§7.2).
2. Apply *a*, then play the hand to its end: opponents act by sampling their own strategies
   conditioned on their (now concrete) cards, hero acts by sampling **its current policy** from
   hero's information only.
3. Record hero's chip delta.

Average over samples → `Q(s, a)`. Averaging is over the posterior-weighted combos, so the
estimate is unbiased for the EV of *a* given the pool's strategies, up to §7.3.

Hero's rollout policy is the **current agent** (owner decision), which makes the whole scheme one
step of soft policy improvement and makes iteration meaningful.

**Cold start (owner decision 2026-08-16, revised).** What has to be competent at label #1 is
**hero's rollout policy and the state distribution it induces**, not the agent's weights — that
is what makes the first policy-improvement step non-myopic and the labelled states worth
labelling. So iteration 0 seats a **v7 pool member in hero's seat** and trains the v8 agent from
scratch on the labels that produces; from iteration 1 hero is the agent, as §8 says.

This delivers exactly what the earlier "initialise the agent from a v7 checkpoint" was for,
without the weight transfer that §6.1 shows to be impossible. It costs on-policy-ness at
iteration 0 — but so did the original, whose "current agent" at iteration 0 was a copy of v7.
Policy improvement is valid from any rollout policy, so this remains a quality choice, not a
correctness one.

`agent_init` is its own config section, separate from the pool bootstrap list, since the
strongest opponent to have in the pool and the best policy to seat as hero at iteration 0 are
different questions.

**Recorded alternative, not baseline: distillation.** Pretrain the v8 agent by KL to a v7 pool
member's output distribution on states drawn from pool play. One forward per state, no oracle,
architecture-independent — it is the only coherent way to transfer v7's *policy* into a v8
network, and it keeps hero on-policy from label #1. It is a training phase rather than an
initialisation, which is why it is not the baseline.

**Also worth revisiting: the embedding-network warm start.** §6.1 has it as a config flag, off by
default. With the v7 warm start gone it becomes the only architecturally coherent one available —
the embedding network is trained first (it is gate G1 and the first pipeline phase), and by the
time the agent exists there is a trained trunk over the *same* tokeniser and the *same* token
format. Whether it should become the default is an open question, not a decision.

No card information leaks to hero: opponents' combos are fixed for the opponents only, and hero's
policy is evaluated on hero's observation. Getting this wrong produces an oracle that
systematically overvalues hero's calls; it gets a dedicated test (§15).

### 7.2 Opponent ranges

"The oracle knows the opponents' strategies, therefore it knows their ranges" is right, but the
range has to be the reach-weighted posterior, not the set of surviving hands:

```
w(combo) ∝ prior(combo) · Π_t  P_i(a_t | combo, history_t)
```

over every action *a_t* that opponent *i* took in this hand, with `P_i` their own strategy.
Combos conflicting with hero's cards or the board get weight zero. Normalise. This is the same
soft-Bayes belief v7 computed in `generation/generate_opponent.py`, including its fixes
(likelihood floor for unobserved combos, card removal relative to the observer).

Inside a rollout no further belief computation is needed — each opponent's combo is already
concrete, and they simply query their own policy.

### 7.3 Declared approximations

These are approximations, not exact computations, and are recorded as such:

- **Joint opponent ranges are approximated by independent marginals with a card-removal
  correction.** The exact joint over 8 opponents is combinatorially impossible. Bias direction
  is unknown and unmeasured.
- **Combo subsampling.** Importance sampling over the posterior weights with a `max_combos` cap
  (v7's `gpu_solver_v5` already has this knob and its semantics).
- **Monte-Carlo noise** in the rollouts, controlled by samples per action.

### 7.4 Recorded alternatives (not baseline)

Both are recorded because §13 makes it likely that variant A does not fit the compute budget at
the label counts we want, and that decision should not be made in a hurry.

- **Variant C — exact root, bootstrapped continuation.** Expand `Q(s, a)` exactly for one hero
  action, play only until hero's *next* decision (opponents' replies are known), then substitute
  `V(s′)` from a value head; terminals still exact. Cost falls by roughly the rollout depth,
  order 10–20×, and the full `Q` vector at the decision point is preserved. **Requires a value
  head**, which the baseline does not have. This is soft policy iteration with bootstrapping —
  biased by the value error, convergent under the usual conditions.
- **Variant B — PPO / actor-critic.** No oracle at all: hero plays against pool members, one
  chip-delta return per hand, clipped surrogate with GAE and a value baseline. Cost per decision
  is ~1 forward instead of ~10⁴, i.e. 3–4 orders of magnitude cheaper, so it buys 10⁸ hands where
  A buys 10⁶. Rejected for the baseline because a single hand's chip delta is dominated by card
  variance, which is the classic reason plain policy gradient is not used in poker. Kept as the
  scaling path of last resort.

---

## 8. The outer loop

```
pool ← v7 networks + procedural style variants + degenerate strategies
repeat:
    ├─ sample table configs (2–9 players, 10–300 BB, uniform) and pool members (PFSP §4.4)
    ├─ play hands  → histories
    ├─ fit / refresh opponent embeddings from those histories        (§5.5)
    ├─ oracle labels Q(s,·) at hero decision points                  (§7)
    ├─ train agent v8 on KL to softmax(Q_norm / T)                   (§6.2)
    ├─ periodically retrain the embedding network on the enlarged history corpus
    └─ pool ← pool ∪ {new agent}
```

The embedding network can be trained **before any v8 agent exists**, on hands played by pool
members among themselves. That is both gate G1 (§14) and the first pipeline phase, and it means
the riskiest bet (B1) is testable first and cheaply.

**The hero seat is occupied by the current agent** (owner decision), so the labelled state
distribution is on-policy — which is what policy iteration assumes. The one exception is
iteration 0, where the agent does not exist yet and a v7 pool member sits in hero's seat (§7.1):
that buys a competent state distribution instead of the random-walk coverage a fresh network
would give, at the cost of one off-policy iteration.

**Each iteration continues the last one** (owner decision 2026-08-18). The agent trained at
iteration *k* is the network iteration *k−1* produced, not a fresh initialisation: the loop is
policy iteration, and restarting from scratch each cycle would discard every best response
already found. Only iteration 0 starts from random weights (§6.1).

**Iteration 0 is the hardest cycle and is budgeted separately** (owner decision 2026-08-18). It
is the only one that starts from a random network, the only one whose labels come from a hero
seat the agent did not occupy, and the only one with no policy to inherit — so it needs the most
gradient steps. That count is therefore its own config key rather than a multiplier on the
per-cycle count: the later cycles are refinements of an existing policy against a slowly
changing pool, and tying their length to the first one's would either waste the compute the
first cycle needs or starve it.

### 8.1 Config sections

Sketch only — names and defaults settle when the code is written. The point of listing them here
is that the experiment surface is explicit (`CLAUDE.md` §5) and that two things the owner asked
to be separable actually are.

| Section | Contents |
|---|---|
| `bootstrap` | list of entries, each: checkpoint path (into `data/v7/`), `n_variants` (how many random style draws to generate from it), optional explicit style vector (for the hand-designed degenerate strategies), optional label |
| `style` | the distribution style draws come from — per-block scale for the §4.2 bias vector, temperature and `λ` ranges |
| `agent_init` | which pool member sits in hero's seat at iteration 0 (§7.1), the agent itself being trained from scratch. **Deliberately separate from `bootstrap`** — the strongest opponent to have in the pool and the best policy to seat as hero are different questions and need not resolve to the same member |
| `embedding_net` | model dims, `K` (inference gradient steps), fit learning rate, regularisation toward zero, `R` (recompute interval), amortised-head weight, ablation switch |
| `oracle` | samples per action, `max_combos`, variant (A / C), EV normalisation divisor, temperature `T` |
| `pool_sampling` | PFSP exponent, uniform floor fraction, dedup cluster count |
| `game` | `raise_sizes` per street, players range (2–9), stack range (10–300 BB) |
| `agent_train` | optimiser, embedding-dropout probability, schedule, and the gradient steps per cycle — with **iteration 0's count a key of its own** (`first_iteration_steps`, owner decision 2026-08-18), see §8 |
| `evaluation` | Slumbot hands, cold/warm switch, frozen-pool-slice screen (§16, OI-7) |

---

## 9. Data flow and observation parity

One rule, stated once because violating it is the most likely way this design fails silently:

> **Every observation the embedding network or the agent ever sees, at training or at inference,
> is constructible from what the observer could have known at that moment.**

Concretely: opponents' hole cards masked in every decision token; a completed hand's showdown
enters only through the terminal token of §5.1a, which no decision token can attend to, and even
there the revealed cards are the target and not an input; the embedding of a player fitted only
from hands the observer sat in; no cross-hand attention (so no accidental future leakage through
the history dimension); causal within the hand.

Note the two different "moments" the rule is applied at, because conflating them is what makes
§5.1a look like a violation when it is not. The **agent** observes a hand in progress and can
never see an unrevealed card. The **embedding network** processes hands that are already over,
and by then the showdown is part of what the observer knows. What §5.1a still refuses is letting
that knowledge flow *backwards* into the decisions of the same hand.

---

## 10. Compliance with `CLAUDE.md` §1

- **Training distribution**: 2–9 players and 10–300 BB sampled uniformly and independently.
  Nothing is weighted toward heads-up or 200 BB.
- **No Slumbot-specific anything.** The pool contains no model of Slumbot; the oracle has no
  Slumbot branch; the observation format assumes neither one opponent nor any stack depth.
- **Adaptation at evaluation time is not specialisation.** Fitting an opponent embedding online
  against Slumbot uses the same generic mechanism the agent uses against every opponent, with no
  Slumbot-specific code. Recorded as such, and reported both cold and warm (§12).
- **Benchmark feedback.** §1 forbids benchmark results from feeding back into **training**.
  Nothing automated does: no screening result enters a loss, a target, a pool weight or a config.
  Candidate selection is done manually by the owner from short Slumbot sessions (§16, OI-7) — a
  human selection step outside the training loop. Its cost is a selection bias on the reported
  figure, which §12 requires to be disclosed rather than removed.

---

## 11. What this design does not guarantee

Recorded so that a later result is not mistaken for a surprise.

### 11.1 Best-response iteration does not converge

Pool-BR-iterate-and-append is fictitious play with a uniform (or PFSP) meta-distribution. In
two-player zero-sum the **average** of the iterates converges (Robinson 1951); the **last**
iterate need not, and it is the last iterate we measure. In multiplayer there is no guarantee of
any kind. Retaining the entire pool damps cycling, and PFSP damps it further — AlphaStar hit
exactly this and answered it with PFSP plus explicit exploiters — but neither is a proof.

### 11.2 "Beats many exploitable opponents" does not imply "not exploitable"

Exploitability is `ε(σ) = max_{π ∈ Π} u(π, σ)` — a maximum over the **whole** strategy space.
Beating a finite pool bounds nothing about that maximum, and there is no continuous transition
between "beats 2000 strategies" and "beats all". Rock-paper-scissors is the minimal counterexample:
"always Rock" beats any number of Scissors-leaning strategies while being maximally exploitable,
and the Nash strategy beats none of them — the criterion "beats everything in the pool" actively
*rejects* Nash there. Poker is not RPS, but its non-transitive component is real and measured
(Balduzzi et al. 2019).

**The conditional architecture weakens the argument further, not less.** The intuition that
beating many opponents forces generality relies on there being *one* strategy. Ours is
conditional: given an opponent embedding it may hold 2000 separate counters, and "beat everyone
in the pool" is then satisfied by a lookup table with no pressure toward generality whatsoever.

**However — for the Slumbot metric our own exploitability is close to irrelevant.** Slumbot is
static and cannot punish us. What decides the number is B1(b): whether an unseen strategy's
embedding lands somewhere sane. The realistic failure is not "we got exploited" but "we applied a
counter-strategy meant for somebody else", which is why embedding dropout (§6.2) and the `e = 0`
fallback matter more here than any equilibrium consideration.

If a measurement of our own exploitability is ever wanted, the tool is **LBR** (Lisý & Bowling
2016/2017) — a local best response giving a computationally cheap *lower bound* on ε. Not
planned. For calibration, their Table 3: LBR wins **4020 ± 115 mbb/h against Slumbot 2016** (fold
/ call / pot / all-in betting, LBR active on turn+river) and 3763 ± 104 with the 56-bet action
set; every bot they tested is exploitable for over 3180 mbb/h at 97.5 % confidence, against 750
mbb/h for folding every hand. That is the scale of exploitability an abstraction-based CFR bot
carries — the owner's scepticism that such bots are "near Nash" is well founded. Note the API
Slumbot is a later, stronger version and no published LBR number for it is known to me.

### 11.3 Pool diversity may be one-dimensional

500–2000 checkpoints of one lineage may span far fewer *styles* than that: neighbouring
iterations are highly correlated, and the embedding space risks collapsing to a single "strength"
axis, on top of which the network learns to identify *which of the known members* this is rather
than a general style space. That failure would leave B1(b) with nothing to generalise along.
Procedural style variants and degenerate strategies (§4) are the mitigation, and they are cheap
enough that there is no reason not to use them from the start.

---

## 12. Evaluation

**Primary metric**: BB/100 against Slumbot (heads-up, 200 BB), **≥ 1 000 000 hands**, always
quoted with its standard error. Shorter runs are screening only and are never reported as
results (`CLAUDE.md` §1).

**Two numbers, always both**:

- **cold** — embedding pinned to zero for the whole run; measures the unconditional policy;
- **warm** — embedding fitted online by the generic mechanism (§5.5), plus the number of hands
  it took to warm up.

If warm is worse than cold, the exploitation mechanism is a net negative. That is a result worth
reporting, not a bug to tune away.

**Selection disclosure.** Candidates are chosen manually from short Slumbot sessions (§16, OI-7),
so every reported result also states how many candidates were screened and over how many hands
each. Without that, the headline number reads as an unbiased measurement when it is the maximum
of several noisy ones. At 50 000 hands a session's standard error is roughly ±2.7 BB/100
(σ ≈ 6 BB/hand), which is the scale of the bias involved.

`evaluation/slumbot_eval.py` is inherited from v7 and does not import yet. Split it as
`ARCHITECTURE.md` already describes: keep the protocol layer (HTTP client, action-string grammar,
token ↔ action mapping, state replay, BB/100 + SE accounting), rewrite the agent adapter against
v8's observation format.

---

## 13. Compute budget

**Every number here is a hypothesis until run on the Spark** (`CLAUDE.md` §3) — the dev box has
no GPU.

Order-of-magnitude for one oracle label under variant A:

- actions ≈ 10, samples per action ≈ 256, remaining decisions per rollout ≈ 15
  → ~4 × 10⁴ policy forwards for the rollouts;
- posterior computation, ~1300 combos × opponents × their decisions, amortised over the hero
  decisions in the hand → same order again.

So ~10⁵ forwards per label as a design figure. At an optimistic sustained 3 × 10⁴ batched
forwards/s that is ~3 s/label; 10⁵ labels ≈ 3–4 days per iteration, 10⁶ labels is out of reach.
Recall that 273 GB/s is roughly an order of magnitude below a datacenter GPU and that memory-bound
kernels will dominate, so large batches are not optional.

Consequences already accepted: the lock-step vectorised driver (§3) is mandatory, not an
optimisation; the pool is networks only (§4); and variant C (§7.4) is the first lever if the
budget does not close. **G3 (§14) measures the real figure before any pipeline is built around
it.**

---

## 14. Gates

Two experiments, both before the full pipeline, each capable of killing the design cheaply.

### G1 — does the embedding carry style, and does it generalise?

Embedding network only. No agent, no oracle. Pool = v7 networks + procedural style variants +
degenerate strategies, playing each other.

Measurements, on **held-out hands of held-out players**:

1. Action-prediction loss vs. number of observed hands, against the `e = 0` baseline. If a few
   hundred hands do not beat `e = 0` clearly, the mechanism does not work and nothing downstream
   is worth building.
2. **Freshly sampled style settings that were never in the embedding network's training
   histories** — the direct test of B1(b). Because styles are procedural, these cost nothing to
   generate and restrict nothing: this is not a held-out opponent pool, it is a fresh draw. Report
   the gap between seen-style and unseen-style prediction loss.
3. **Ablation** (§5.4): mean-pooled hidden states, no gradient fit. If it ties, drop the fit.
4. **The single-vector assumption** (owner: baseline is one vector per player, assumption must be
   checked): break the prediction loss down by table size and by stack depth and look for
   systematic skew. Zero extra work — it is a grouping of the same numbers.

### G3 — what does an oracle label actually cost?

On the Spark, on the lock-step driver, with a realistic pool: measured wall-clock and forwards
per label, and how it scales with samples-per-action, `max_combos`, and table size. Output is the
label budget that §8 can afford, which then fixes the config.

(The former G2 — embedding-space visualisation — was dropped by the owner; its essential content
is folded into G1.2.)

---

## 15. Testing

Per `CLAUDE.md` §4: the whole battery stays under 30 minutes, CPU-only, deterministic, and
prefers end-to-end scenarios over unit tests of helpers. What must be covered:

- **Observation parity** — the property in §9, asserted end-to-end: rebuild an opponent's token
  sequence from the observer's view and assert no unrevealed hole card and no post-decision
  information is ever present. This is the fatal-if-wrong invariant.
- **Attention mask** — block-diagonal + causal-within-hand, tested behaviourally: perturb a token
  in hand *j* and assert the predictions in hand *k ≠ j* are bit-identical; perturb the action at
  token *t* and assert predictions at *t′ ≤ t* are unchanged.
- **Embedding is the only cross-hand channel** — with `e` fixed, shuffling the order of hands
  leaves predictions unchanged.
- **Showdown anchor (§5.1a)** — a terminal token exists exactly for the revealed seats and never
  among the decisions; the revealed cards are the target and appear nowhere in the token; the
  labels match the cards that were shown; perturbing a showdown token leaves every decision-token
  prediction bit-identical (the no-backward-leak property); the action loss ignores showdown
  tokens; zero weights reduce the objective to action cross-entropy exactly; the showdown terms
  reach the *fitted* vector, not only the trained weights.
- **Posterior** — reach weighting against a hand-computed example; zero weight for combos
  conflicting with the board or hero's cards; correct renormalisation; degenerate cases (single
  legal combo, all weights zero).
- **Oracle** — unbiasedness on a tiny hand-solvable game where `Q` can be computed exactly by
  enumeration; no card leak into hero's rollout policy; terminal payoffs match the engine's
  (chips in = chips out).
- **Target construction** — normalisation by `pot + facing_bet`, masking of illegal and dominated
  actions, temperature limits, degenerate distributions.
- **Vectorised driver** — N hands in lock-step produce exactly the same trajectories as N
  sequential hands under the same seeds.
- **Inference-time fit** — deterministic under a fixed seed; the joint fit of several players
  converges on a synthetic case with a known answer.
- Existing engine tests (`test_engine_conservation.py`, `test_audit_stage0.py`) stay.

---

## 16. Open items

- **OI-1 — token completeness. RESOLVED 2026-08-16:** `num_players` and the per-seat stack
  vector are both added to the token (§5.1).
- **OI-2 — oracle cold start. RESOLVED 2026-08-16; REVISED 2026-08-16 by the owner.**
  *Original:* the v8 agent is initialised from a v7 checkpoint (action head only, no MCTS, no
  opponent embedding), so hero's rollout policy is competent from the first label.
  *Revision:* the agent is trained **from scratch**, and a v7 pool member occupies hero's seat
  for iteration 0 instead. The original is not implementable — the agent reads the §5.1 token
  through the OI-4 tokeniser and a v7 checkpoint's weights are shaped for a different input
  representation, so there is nothing to load (§6.1). Seating a v7 member as hero delivers the
  same competent rollout policy and state distribution with no weight transfer at all (§7.1).
  Distillation of a v7 member into the v8 agent is recorded as the alternative; it is a training
  phase, not an initialisation. `agent_init` stays its own config section and now names a pool
  member.
- **OI-3 — state distribution for labelled decisions. RESOLVED 2026-08-16:** the current agent
  occupies the hero seat; labels are on-policy (§8).
- **OI-4 — agent / embedding-network relationship. RESOLVED 2026-08-16.** Three separable
  questions, decided as:
  - *tokeniser implementation* — **one class, used by both.** The tokeniser is the MLP that maps
    the §5.1 feature set to a single `d_model` token vector; both networks consume that token.
    Two copies of that code is exactly the duplicated low-level logic `CLAUDE.md` §5 warns
    about, and any divergence between them would be a silent train/deploy mismatch.
  - *weights* — **separate.** The two networks optimise different things (predict what someone
    else does vs. choose the action that maximises EV) and have different retraining cadences;
    sharing weights would make every embedding-network retrain silently change the agent's input
    representation.
  - *initialisation* — the agent's trunk **may** be warm-started from the embedding network's
    trunk. Config flag, off by default, cheap to try later. With the v7 warm start withdrawn
    (OI-2 revised) this is the only architecturally coherent warm start there is — same
    tokeniser, same token format, and the embedding network is trained first — so whether it
    should become the default is now an open question (§7.1).
- **OI-5 — live style modifiers. APPROVED 2026-08-16.** Design in §4.2: additive logit bias over
  v7's 5 action categories, gated by position and street, plus temperature and a uniform mix;
  ~32 scalars per pool member, drawn randomly at pool construction; equity-gated conditions
  dropped as too expensive inside rollouts.
- **OI-6 — v7 checkpoint access. RESOLVED 2026-08-16:** vendored frozen copy of v7's agent code
  inside v8, checkpoint paths in config pointing into `data/v7/` (§4.3). Note that v7's **event
  builder** must be vendored alongside the model — the reference implementation is already in
  v8 at `evaluation/slumbot_eval.py::_build_events`.
- **OI-8 — showdown information in the embedding. RESOLVED 2026-08-16 by the owner:** a terminal
  reveal token with **both** heads — strength percentile and 169-way preflop class (§5.1a). The
  alternatives were offered and declined: writing the revealed cards into the hand's decision
  tokens (rejected — it leaks "reached showdown", hence "did not fold", backwards into every
  decision), and count-based showdown statistics fed to the amortised head alone (rejected as
  the mechanism — it only moves the starting point, and the fit would pull the vector back off
  it; still available as a cheap complement). The decisive point is that the showdown terms
  belong in the **inference-time fit objective** as well as in training, because v8's embedding
  is a latent found by gradient descent at deployment.
- **OI-7 — candidate screening. RESOLVED 2026-08-16 by the owner: candidates are selected
  manually, by the owner, from short Slumbot sessions.** The benchmark-free alternative (BB/100
  against a frozen slice of the pool) was offered and declined.

  Two consequences follow mechanically and are recorded so the numbers stay interpretable:
  - **Selection bias.** A candidate chosen as the best of *K* on short sessions has an upward
    selection bias; its ≥1 M-hand number will tend to land below its screening number. Every
    reported result therefore states **how many candidates were screened and over how many hands
    each**, so the final figure can be read for what it is.
  - **Scope.** This is a human selection step, not a gradient path: no screening result enters a
    loss, a target, a pool weight, or a config. `CLAUDE.md` §1's prohibition is on benchmark
    results feeding back into *training*, and nothing here does. It remains a channel through
    which the benchmark can influence the project, and it is documented rather than hidden.

---

## 17. Literature

**Verified 2026-08-16** — venue, year and author list checked against the primary source for
every entry below; the LBR figures were read out of the paper's Table 3, not recalled.

| Claim it supports | Reference |
|---|---|
| §7.1 rollout as a policy-improvement operator | Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming*, Athena Scientific 1996 |
| §11.1 fictitious play converges in **average**, not last iterate | Julia Robinson, *An Iterative Method of Solving a Game*, Annals of Mathematics 54(2):296–301, 1951 ([pdf](https://ranger.uta.edu/~weems/NOTES6319/PAPERSONE/robinson.pdf)) |
| §8 the family this outer loop belongs to | Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Pérolat, Silver & Graepel, *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning*, NIPS 2017 — PSRO ([arXiv:1711.00832](https://arxiv.org/abs/1711.00832)) |
| §11.2 non-transitivity is real and measurable in real games | Balduzzi, Garnelo, Bachrach, Czarnecki, Pérolat, Jaderberg & Graepel, *Open-ended Learning in Symmetric Zero-sum Games*, ICML 2019, pp. 434–443 ([pmlr](https://proceedings.mlr.press/v97/balduzzi19a.html)) |
| §4.4 PFSP; uniform league play was found insufficient | Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575, 2019 — league of main / main-exploiter / league-exploiter agents, opponents weighted by win rate |
| §11.2 exploiting without becoming exploitable, if it ever matters | Johanson, Zinkevich & Bowling, *Computing Robust Counter-Strategies*, NIPS 2007 — Restricted Nash Response ([pdf](https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf)); Johanson & Bowling, *Data Biased Robust Counter Strategies*, AISTATS 2009, PMLR v5 ([pmlr](https://proceedings.mlr.press/v5/johanson09a.html)); Ganzfried & Sandholm, *Safe Opponent Exploitation*, EC 2012 / ACM TEAC 3(2), 2015 |
| §11.2 scale of exploitability of abstraction-based CFR bots | Lisý & Bowling, *Equilibrium Approximation Quality of Current No-Limit Poker Bots*, arXiv:1612.07547, AAAI-17 Workshop on Computer Poker — LBR. Table 3: Slumbot 2016 = **4020 ± 115 mbb/h** (fcpa, rounds 3–4) and 3763 ± 104 (56 bets); Act1 2016 = 3302 ± 122; every bot > 3180 mbb/h at 97.5 % confidence; always-fold costs 750 mbb/h ([pdf](https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf)) |
| §5.4 amortised head + gradient refinement | Kim, Wiseman, Miller, Sontag & Rush, *Semi-Amortized Variational Autoencoders*, ICML 2018, PMLR v80 ([arXiv:1802.02550](https://arxiv.org/pdf/1802.02550)) |
| §5.5 closest published analogue of test-time optimisation in a latent space | Rusu, Rao, Sygnowski, Vinyals, Pascanu, Osindero & Hadsell, *Meta-Learning with Latent Embedding Optimization*, ICLR 2019 ([arXiv:1807.05960](https://arxiv.org/abs/1807.05960)) |
| `CLAUDE.md` §1, why the goal is a bet | DeepStack (Moravčík et al., Science 2017), Libratus (Brown & Sandholm, Science 2018), Pluribus (Brown & Sandholm, Science 2019), ReBeL (Brown et al., NeurIPS 2020) — all specialised to a fixed table size, Pluribus to a single stack depth |

**Correction on record:** an earlier draft of this document cited the LBR result from memory as
"of the order of thousands of mbb/g". The real figure for Slumbot 2016 is 4020 ± 115 mbb/h, and
the paper's headline bound is 3180 mbb/h across all bots tested. An intermediate automated
extraction returned 50 / 100 / 1500 mbb/h for Slumbot / Act1 / always-fold; those numbers are
wrong (always-fold is 750 mbb/h by construction) and are recorded here so they do not resurface.
