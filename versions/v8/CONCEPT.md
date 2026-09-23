# v8 — Concept

**Status: agreed with the owner 2026-08-16; amended 2026-08-19 (§6.2's second loss, §7.1's
runout draw, §7.4's withdrawn cost claim, §13's measurement) and 2026-08-20 (§5.6's strength head
and §6.1's trunk warm start, which are one decision; §5.1a's class head weighted to zero on G1's
own numbers; policy distillation from a v7 member declined). Gates G1 and G3 have run; the whole
pipeline is implemented.**
All open items OI-1 … OI-8 are resolved — see §16, where OI-2 is recorded as revised after
implementation showed the original not to be implementable. One later item, **OI-9** (the
accuracy of the `max_combos` subsample), is open and deliberately deferred; nothing else in this
document is waiting on a decision. `ARCHITECTURE.md` §4 lists the readings that turning this
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

  **They are weak, and how weak is load-bearing (owner, 2026-08-19): v7 plays about −90 BB/100
  against Slumbot.** That does not make the oracle wrong — `Q` is by construction the EV against
  *this* pool, whatever the pool is, and the whole point of §2 is that a fixed pool makes that
  exact. What it does mean is that every posterior the oracle conditions on is the range of a
  bad strategy, so anything whose answer depends on the *shape* of that range — how tight it is,
  how concentrated its mass — is being measured on an unrepresentative distribution and does not
  transfer. Two live consequences: the accuracy of the `max_combos` subsample (§7.3) cannot be
  usefully measured on v7 ranges, and iteration 0's hero is a weak rollout policy, so the first
  improvement step starts from further back than the §7.1 cold-start argument assumes.
- **Procedural style variants.** The same base network with a randomly sampled style modifier
  applied to its output distribution (§4.2). This is the cheapest source of pool diversity by a
  wide margin and it is what keeps the embedding space continuous instead of a 20-way
  classification (§11.3). **Every base is also in the pool unmodified** (owner decision
  2026-08-19): the plain network is the strongest member of its family and the reference point
  each draw is a perturbation of, so it is an ingredient in its own right rather than a special
  case of a draw. How many modifications a base contributes stays per-entry config
  (`bootstrap[i].n_variants`), because 8 draws off a v7 checkpoint and 1 off `always_fold` are
  different asks. `PLAN_PIPELINE.md` D13 carries the mechanism.
- **Degenerate strategies.** always-fold, always-call, always-min-raise, maniac (all-in biased),
  nit (folds without a strong hand). No network, no cost, and they pin the corners of the style
  space that iterated agents will never visit on their own.

**Later:** every trained v8 agent joins the pool, as `agent_variants` members rather than one
(owner decision 2026-08-19): the agent itself plus `agent_variants − 1` procedural style draws
off it, exactly as a v7 checkpoint is expanded at bootstrap (§4.2, which already says the
mechanism "applies uniformly to any pool member, v7 or v8"). The reason is §11.3 turned on
ourselves — thirty iterations of one lineage are the *most* correlated set of members the pool
will ever contain, and a style draw costs no forward, no parameter and no checkpoint. `1` is the
no-multiplication setting and is a legitimate value; the number is config because it changes
both the pool's composition and the size of the §5.4 embedding table, and nothing in the loop
should be deciding either.

**A past agent in the pool plays at `e = 0` by default, and can be given the amortised reading of
its tablemates instead** (owner decisions 2026-08-19 and 2026-09-03, `PLAN_PIPELINE.md` D12,
`PLAN_AMORTISED_POOL.md`). At `e = 0` it is a fixed policy like every other member: it plays the
unconditional policy §6.2's embedding dropout trains explicitly, rather than fitting a vector for
whoever it happens to be sitting with. That costs an asymmetry worth naming — hero exploits its
table and the pool's own agents do not, so the agent is trained against a slightly *weaker*
version of its predecessors than the ones that produced the labels.

The alternative D12 weighed was a full §5.5 fit per past agent per table, and it was rejected as
nesting one fit inside another for every rollout. The cheap form it did not consider is the
amortised head with **no gradient step at all** (`K = 0`): one trunk pass over the hands that
agent has seen *from its own seat*, computed once per refresh block at the driver level exactly
as hero's fit is, and only read inside a rollout. Turning it on removes the asymmetry — the
pool's agents then exploit at `K = 0` while hero exploits at `K`.

Two properties make it sound rather than merely cheap. **A past agent reads through the embedding
network of its own generation**, frozen: nothing anchors the coordinates of the vector space, so
a frozen policy handed a later generation's vectors would be reading a description in a basis
that has drifted, and a pool member conditioned by a network that is still training would stop
being a fixed algorithm at all. And **the embedding corpus is never conditioned** (owner decision
2026-09-03): every player in it plays one policy for the whole session, so a member's vector
still means one style. The mismatch that leaves — a network trained to describe static players,
asked at label time to describe adapting ones — is deliberate, and is the same mismatch that
exists against any real opponent, none of which are stationary either.

Expected size 500–2000 members at late iterations, bounded by compute, not by design.

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
fed anything. That format is already present in v8 — v7's inherited `slumbot_eval.py::_build_events`
constructed it (`hand`, `num_players`, `hero_pos`, `acting_pos`, `big_blind`, `small_blind`,
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
type distinguishing it from a decision. For every seat **other than the observer** it carries
**no cards of the revealed hand** — those are its *targets*:

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

**The observer's own showdown token carries the observer's own cards** (owner decision
2026-08-20). Hiding a player's own hand from that player is the one form of masking that buys
nothing: the observer knows it at every moment of the hand, §9 is about what the observer could
have known and never about withholding what it did, and the rule "the observer's own cards
always, everybody else's never" then holds on every token type instead of having an exception.

What the masking is actually for is worth restating, because it is easy to read it as being about
the cards rather than about the gradient. Attention is causal and no decision token attends to a
showdown token, so the *only* channel from a reveal to that player's vector is the gradient of a
loss whose answer is absent from the input. Unmask another seat's hand and the head becomes a
hand evaluator — a duplicate of §5.6 on a third of the hands — and the anchor stops producing any
pressure on the vector at all. That is why the exception stops at the observer.

Its one cost is on the *metric*, not on the model. The observer's own token's strength target is
now a deterministic function of its own input, so the pooled `showdown_strength_mse` mixes a
trivial population with the hard one and **is not comparable to G1's 0.095**. Training is
unaffected — a term the head can already answer stops producing gradient — but the number has to
be read split, or read as the easy half.

Weights for the two heads are config, used identically in training and in the fit; setting them
to zero is the ablation that asks whether the anchor earns its keep. **The 169-way class head is
weighted to zero as of 2026-08-20** (owner decision) on G1's own numbers: held-out 5.75 nats
against a marginal entropy of 5.005, i.e. worse than a constant, after sitting exactly on that
constant for 10 000 steps and then memorising the corpus in the final low-LR phase. The code
stays — zero is the documented ablation switch, and removing the field would change the shard
format for no gain.

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

**The table carries a row for every member the loop will ever produce** (owner decision
2026-08-19). `n_members` is fixed when the table is constructed, and §8 adds `agent_variants`
members per iteration — the agent and its style draws (§4.1) — so the table is sized
`len(pool₀) + max_iterations × agent_variants` up front: iteration *k* owns the block of rows
starting at `len(pool₀) + k × agent_variants`, whose first row is the agent itself and belongs
to it from the moment it is first seated as hero, not from the moment it joins the pool. The
style draws are separate players to identify — that is the entire point of the style layer for
this network — so they are separate rows, not one row shared. A row nobody has occupied yet is never read, because no token carries its
index; it is a dead parameter at its initialisation until its agent exists. Retraining the
network across iterations is therefore a **continuation**: every converged row survives, and the
new agent's row starts where the initialisation put it.

The two alternatives were considered and rejected. Rebuilding the table at each retrain turns
that continuation into a restart and discards every row the previous iterations converged.
Dropping hero's tokens from the corpus is cheaper still, but it means the network never learns to
read the one opponent it is guaranteed to face after §8's last line — the previous agent.

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

### 5.6 The strength head — the poker prior

**Status: agreed with the owner 2026-08-20**, together with the §6.1 warm start; the two are one
decision and neither is worth much alone. The alternative offered at the same time — distilling a
v7 member's policy into the agent as a pretraining phase (§7.1's "recorded alternative") — was
**declined**, on the ground that it would make the agent resemble v7. So the prior transferred
here is about *cards* and never about *actions*.

A third head on the trunk of §5, reading the **observer's own** decision tokens and predicting
the percentile the observer's hand reached on that hand's final board. MSE, weight
`embedding_net.strength_weight`.

**Why it exists.** The agent reads seven card slots through `Embedding(53, ·)` and an MLP, and
nothing else about hand strength: no equity, no percentile, no suit or rank structure. Everything
it knows about which two cards beat which it has to infer from oracle EV labels — whose noise at
200 BB is, by G3's own numbers, larger than the EV differences between the actions they are
meant to rank (§13, §11.4). Learning poker's card evaluation from that signal is the most
expensive possible way to learn it. The same fact is available exactly and for free from played
hands, so it is learned there instead, on a corpus that costs no rollout at all.

**Why this target and not equity.** `env/showdown.py::strength_percentiles` already computes it,
exactly, by enumeration, with no seed and no Monte Carlo — it is the §5.1a label with the
restriction to revealed seats lifted. And scoring on the *final* board rather than the visible
one is what makes the head learn draws: at a flop decision the conditional distribution of the
final percentile given the flop **is** the value of the draw, and regressing to a sample of it
converges on that conditional mean.

**It is a target and never an input**, exactly as §5.1a's are, and §9 is untouched: the token's
cards, board and scalars are bit-identical to what they were before the head existed. Predicting
a quantity the observer will only learn later is what a value target does. A hand still in
progress carries no target, because it has no final board yet.

**Training only — and this is the one place it differs from §5.1a.** The showdown terms belong in
the §5.5 fit because a revealed holding is evidence about the *revealer's* style. The strength of
the observer's own cards is evidence about nobody: its gradient into an opponent's vector is
noise and into the observer's own vector is a channel carrying nothing the vector is for. So the
term shapes the weights and is absent from the fit objective. What carries it into the agent is
§6.1's warm start, not the vectors.

**The MSE floor is not zero, and misreading it is the predictable mistake.** The target is one
runout, not an expectation over runouts, so a perfect head still pays the conditional variance of
the runout: large preflop, zero on the river. The baseline to read the number against is the
**marginal variance of the target on the same corpus**. This is §11.4's trap in a second place,
and it is the same trap the §5.1a heads are already sitting in — G1 measured `showdown_strength`
at a held-out MSE of 0.095 against a target variance of ≈0.085, and the 169-way class head at
5.75 nats against a marginal entropy of 5.005, i.e. both worse than a constant, having sat exactly
on that constant for 10 000 steps before memorising the corpus in the last low-LR phase. The
strength head is a much easier task — it is scored on cards the observer can see — but it is to be
read the same way, and `strength_weight = 0` is the ablation that asks whether it earned its keep.

---

### 5.7 The range head — the belief, made explicit

**Status: agreed with the owner 2026-09-02**, architecture included.

A fourth head, and the only one that is **not a leaf**. It sits between the trunk's two halves of
decoder layers: at every decision token it predicts, for every live opponent, a distribution over
the 1326 two-card combos, and that prediction is fed back into the tokens the remaining layers
read.

**Why a belief at all.** DeepStack and ReBeL both make the range a first-class object and hand it
to the value network as an *input* — in self-play it is computable, so there is nothing to infer.
Ours is not computable: the players are unknown members of a pool with drawn styles, so the range
has to be *inferred* from the line. That puts this closer to the belief-prediction auxiliary tasks
of Hanabi (BAD, SAD, Learned Belief Search) than to the poker systems, and it is the same bet
those make: a representation forced to carry the belief plays better than one left to discover it
from the reward. Here the reward is an oracle EV label whose noise at depth is larger than the
differences it is meant to teach (§13, §11.4), which is the same argument §5.6 makes for the
strength head.

**The target is §7.2's posterior, run as a filter.** Weights carried forward, each new action of
that opponent multiplying them, each new board card removing what it blocks. With no pruning it is
exactly `opponent_posterior` evaluated at every prefix; the two are one estimator with one
implementation, and the agent's labels get theirs from the same object the oracle already builds
to sample the rollouts' holdings.

**Why not a hard range.** The first proposal was a set: keep the combos whose *modal* action is
the one that was played, narrow it street by street. It is degenerate in this pool. Every member
emits finite logits and the style layer mixes in up to 25 % uniform, so no combo ever has zero
reach and the surviving set is exactly the card-removal mask — a deterministic function of the
token's own input. Four of the five degenerate strategies never read their cards, so their argmax
is identical for every combo and the set comes out either full or empty. And the two cost the
**same forwards**: both need `P(a | combo)` for every surviving combo, so rounding to 0/1 buys
nothing and throws away the magnitude, which is the part an EV calculation actually uses. The
support cut that *does* buy forwards is a threshold on the weights, and it is a knob
(`range_prune_threshold`) rather than a definition.

**Its cost is real and is paid in hands.** A corpus hand costs ~1 policy row per decision today
and ~1225 per opponent decision with the target, so the corpus is where this is expensive; the
owner's decision is to regulate it by the number of hands rather than by sampling a subset. A
label pays about 10 % more (G3: 1225 posterior rows in ~11k), because the belief is recomputed
in the label worker rather than threaded out of `action_values`.

**Architecturally it is a perceiver decoder, not an MLP on the token.** A range is the product of
one player's likelihoods over every decision they have taken, and those live in earlier tokens, so
the queries — "player *s*, at moment *t*" — cross-attend over the trunk's states across the whole
prefix. Causally: a query at *t* sees *t' ≤ t* and nothing later, or the belief reads actions that
have not happened and the deployed agent is strictly worse than its loss curve. The query carries
the seat's position embedding, that seat's opponent vector, and the trunk state at *t*.

**What goes back into the trunk is the probability vector, detached.** Not the head's hidden
state: a wide hidden state would let the action loss push arbitrary information around the
1326-wide bottleneck, and the stop-grad would then be closing the wrong channel. With the detach
the head is trained by its own target alone and the layers above consume a belief they cannot
bend. The token attends over its own live opponents' belief vectors, so the aggregation is learned
and the identity of each belief survives it.

**`range_weight = 0` is not the ablation** — with the weight at zero the head still runs and still
injects, so the layers above it read an untrained head's output. `range_enabled: false` is the
ablation, and it removes the module.

**Read the KL, not the cross-entropy.** The support is 990–1225 combos, so a perfect head still
pays the target's own entropy: ~6.9–7.1 nats. The movable part is `ce − H(target)`. This is
§11.4's trap in a third place.

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

**Warm start: the trunk, from the embedding network** (owner decision 2026-08-20). §16 OI-4 left
this as "a config flag, off by default" and §7.1 noted that, with the v7 warm start withdrawn, it
is the only architecturally coherent one available. It is now the baseline and the flag is
`agent_train.warm_start_trunk`. What it copies is `HandEncoder`'s weights, once, at iteration 0,
after the embedding network's first retrain: the two networks read the same §5.1 token through the
same tokeniser, so the parameter sets correspond exactly — which is precisely what OI-4 bought by
sharing the trunk as code and is the thing a v7 checkpoint could not offer (below). The action
head keeps its initialisation, the weights stay separate objects, and the agent's trunk moves
under `soft_q` from its first gradient step: this is an initialisation, not a tie.

What it is *for* is §5.6. Without the strength head the warm start transfers a trunk trained to
predict other people's actions, which is a related but different task; with it, the trunk also
carries hand evaluation, board texture and the value of a draw, learned from free self-play. That
is the half of the job the oracle's labels are worst at teaching.

**Initialisation of the head and of everything else: from scratch** (owner decision 2026-08-16,
§16 OI-2 revised). An earlier draft
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

**Second loss, same optimum: `soft_q` (owner decision 2026-08-19).** `Q` is a Monte-Carlo
estimate, `Q̂ = Q + ε`, and G3 measured `ε` between 0.65 BB (heads-up, 20 BB) and 25.7 BB
(six-handed, 300 BB) at 256 samples. A softmax is not linear, so `E[softmax(Q̂/T)] ≠
softmax(Q/T)`: the gradient the network sees under the KL loss is the *expected* target, and
label noise becomes target **bias** that averaging over labels does not remove. In the
large-noise limit the target degenerates towards whichever action drew the luckiest sample,
whose average over labels is uniform-over-legal — so the policy is pushed towards uniform
exactly where the labels are noisiest, which is deep stacks and multiway, which is the part of
`CLAUDE.md` §1 that must not be sacrificed.

The fix is to make the *loss* linear in `Q̂` rather than the target — no map from EVs into the
simplex can be affine, but the objective can be:

```
L(π) = −⟨π, Q_normalised⟩ − T·H(π)          minimised at  π = softmax(Q_normalised / T)
     = T·KL( π ‖ softmax(Q_normalised / T) ) − T·log Z(Q_normalised)
```

`Q̂` enters linearly, so `E_ε[∇_θ L] = ∇_θ L|_{Q̂=Q}` at every `θ` and the noise stays noise.
Because the oracle returns `Q̂` for *all* legal actions there is no sampled action and no
importance weight, so this is all-action policy gradient and not REINFORCE.

The price, and it is not free: this is the **reverse** KL. Same minimiser, but mode-seeking
rather than mass-covering, so where the network cannot fit the target it drops
small-probability actions instead of smearing over them — and mixing is what §11.2 cares about.
Which loss wins is a measurement, not an argument, which is why `agent_train.loss` selects
between `kl` and `soft_q` rather than one replacing the other. Watch the policy's entropy by
stack depth and table size when reading the answer.

**Normalisation is mandatory and is where v7 has scars.** Raw EVs in a 300 BB pot and a 10 BB pot
differ by more than an order of magnitude; a single temperature applied to raw EVs produces a
near-deterministic policy in big pots and a near-uniform one in small pots. v7's 97 % fold rate
was a normalisation bug of exactly this family, not an architecture failure
(`versions/v7/ARCHITECTURE.md`, "MCTS value-target normalization"). Baseline: divide by
`pot + facing_bet`, with the divisor itself a config choice.

**Annealing by local EV loss (owner decision 2026-09-08).** The default
`oracle.temperature` is now `{"initial_ev_loss_bb": 1.0}`. At zero-based outer
cycle `k`, set `eps_k = initial_ev_loss_bb / (k + 1)`, and for each decision
with `A > 1` legal actions and BB divisor `D`, use `T_k(I) = eps_k / (D log A)`.
A scalar temperature remains supported as a fixed-T historical mode. With
one legal action the loss is zero for any T; the implementation uses T = 1.

For exact Q, entropy optimality against a greedy action gives
`max Q - <pi_T,Q> <= D T H(pi_T) <= D T log A = eps_k`.
This bounds **entropy smoothing alone at the exact local optimum**. It does
not bound noisy-oracle error, neural optimisation error, exploitability or a
whole hand's loss. The harmonic schedule and initial 1 BB budget are explicit
design choices, not an optimal cooling theorem. The first, tenth and thirtieth
cycles budget 1, 0.1 and 1/30 BB per decision; the budget tends to zero.

With a fixed BB budget, D cancels in the soft optimum; the legacy scalar-T
scale-invariance property above intentionally does not apply. Normalised Q
still determines the relative weight of states in the loss. T depends only on
the public pot, facing bet, mask and cycle, preserving the soft_q gradient's
linearity in Q. Both losses, warm/cold metrics and `regap.py` resolve the same
schedule using the outer cycle, including after a restart. This is local soft
policy improvement, not CFR and not a guarantee of equilibrium mixing.

**A concern the G3 numbers raise about the divisor itself, flagged and not decided
(2026-08-19).** `pot + facing_bet` does not scale with the stack, but the payoff does — in NLHE
what can change hands is bounded by the effective stack, not by the pot. G3 measured the label
error in exactly these units and it ran 0.2–3.4 at 256 samples, i.e. **the noise alone is
usually larger than the divisor**, and the per-sample chip-delta deviation came out at 0.5–1.4
of the stack, meaning nearly every rollout is a stack-off. So at depth the normalised target has
a large dynamic range before any noise is added, which is the same family of problem the
paragraph above says v7 was bitten by — only from the other side. `DIVISORS` is a config choice
and `pot` is the only alternative currently implementable from the arguments the function
receives; anything stack-aware would be a third entry and a new decision. Recorded so that a
first training run reads a saturated or a near-uniform policy at 300 BB as this rather than as
an architecture failure.

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

**The runout is dealt, not reused (owner decision 2026-08-19).** Only the board hero can *see* at
the decision is fixed. The streets still to come are dealt afresh for every sample, out of
whatever the assignment left in the deck, so the decomposition is
`p(hands | visible board) · p(runout | hands)` — the posterior conditions on what hero saw, the
runout comes out of what is left, and both halves are exact.

An earlier draft pinned the whole five-card board of the hand the decision came out of, and
argued that re-dealing it would average over runouts the posterior had already conditioned away.
That argument is wrong: the posterior conditions on the *visible* board, and §7.1 defines `Q` as
the EV over everything hero does not know, of which the turn and the river are part. Pinning the
board makes the label an estimate of `E[chips | this exact river]`, whose error
`samples_per_action` cannot reduce at all because every sample shares the one runout — an error
invisible to the split-half column that is supposed to size the sample budget. The draw is per
sample and not per action, so common random numbers still hold.

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
Combos conflicting with hero's cards or the **visible** board get weight zero — a card of a
street still to come is not dead, it is in the deck, which is what §7.1's runout draw says.
Normalise. This is the same soft-Bayes belief v7 computed in
`generation/generate_opponent.py`, including its fixes (likelihood floor for unobserved combos,
card removal relative to the observer).

The exact-posterior implementation reuses that product across consecutive hero decisions of one
hand. A newly revealed board card only removes conflicting combos from the previous support, and
only opponent actions after the previous prefix contribute new likelihood factors. This is the
same Bayes posterior as recomputing the full product; it changes cost, not the oracle. The cache
is bounded to the current hand and is disabled under `max_combos`, where every label intentionally
draws its own random prior subsample.

Inside a rollout no further belief computation is needed — each opponent's combo is already
concrete, and they simply query their own policy.

### 7.3 Declared approximations

**Joint conditioning, corrected 2026-09-08.** For fixed opponent policies and
memory frozen for this hand, let `L_j(c)` be the product of opponent j's action
likelihoods along the observed prefix, including its fold. Then
`p(c_1,...,c_m | I) ∝ 1[all cards disjoint and compatible with I] * Π_j L_j(c_j)`.
The separate ranges are normalised reach factors, not full posterior marginals.
Independent proposals followed by rejection sample this joint exactly when
the factors are exact. Rejection rate measures cost and lost samples, not bias.
The retry cap drops failed samples; conditional on accepting any samples, their
mean has the target expectation. A label with no accepted samples is dropped.

**Folded seats are included.** Their conditioned cards are pinned before
drawing the future board, so both live ranges and runouts retain the information
in folds (card bunching). Their recorded folds remain forced in the rollout;
they never re-enter betting or showdown. The previous implementation gave
folded seats uniformly drawn cards and therefore lost this information.

The remaining range approximations are:
- **Combo subsampling.** Self-normalised importance sampling over the **prior**, with a
  `max_combos` cap. (v7's `gpu_solver_v5` drew from the *posterior* and renormalised, which
  double-counts the mode and saves no policy call; `ARCHITECTURE.md` §2.2b records why the cap
  is applied before the likelihood pass instead.) **Its accuracy has never been measured** —
  G3 sweeps the axis for cost only, and the split-half column cannot see a bias by construction.
  Measured cost: the cap to 128 combos saves about 46% of a label at 32 samples and about 21% at
  128, which is the only lever on the label's fixed cost.

  The right metric, when it is measured, is **not** total variation against the exact posterior:
  a subsample of 128 points cannot be close in TV to a distribution over 1225, yet it can
  estimate expectations perfectly well, which is what it is for. The operative statement is that
  the cap puts a **ceiling of the subsample's effective sample size on the number of distinct
  opponent holdings a label can average over, and that ceiling does not rise with
  `samples_per_action`** — as `n → ∞` the estimate converges to `⟨w_capped, V⟩`, not to
  `⟨w_exact, V⟩`. If `ESS < n`, the extra samples buy nothing. The predicted failure mode is
  specific: the proposal is uniform over the prior, so a tight range — 20–50 combos carrying the
  mass — survives a 128-of-1225 draw with 2–5 combos, and the effective sample size collapses
  exactly where the range is informative. Flat ranges are carried fine.

  **Deferred, 2026-08-19 (owner).** Measuring this on the bootstrap pool would measure it on v7
  ranges, and §4.1 records why the shape of those ranges is not to be trusted. The baseline runs
  the exact posterior, so nothing is blocked by leaving it open.
- **Monte-Carlo noise** in the rollouts, controlled by samples per action. It covers three
  sources — the opponents' hole cards, the runout, and the action draws of hero and the
  opponents — and only their *differences between actions* reach the target, since the softmax
  is shift-invariant and the joint sample is common to all actions (§7.1).

  **Two of the three are now subtracted rather than sampled down (2026-09-01).** A rollout is
  averaged through a control variate: the expected share of the matched pot given the boards
  still possible, subtracted at every card dealt and at every fold/no-fold draw of a player whose
  policy is known. Each term has zero mean over the draw it corrects, so this cannot move what a
  label estimates — it is not a fourth approximation. Where a rollout runs out of decisions the
  terms telescope and what is left is the average over every runout that could have happened,
  which is why "integrate the cards out" is not a separate mechanism. Measured on the dev box
  against the fixture pool: the per-rollout standard deviation falls to 0.41–0.52 of its raw
  value, i.e. four to six times the sample budget, for 20–50% more wall clock.
  `ARCHITECTURE.md` §2.2d carries the design and its declared limit (side pots, and the size of
  a bet). **This makes §13's "quadrupling the samples halves the noise, and nothing else does"
  false as of this date** — it was true of the sample budget alone.

### 7.4 Recorded alternatives (not baseline)

Both are recorded because §13 makes it likely that variant A does not fit the compute budget at
the label counts we want, and that decision should not be made in a hurry.

- **Variant C — exact root, bootstrapped continuation.** Expand `Q(s, a)` exactly for one hero
  action, play only until hero's *next* decision (opponents' replies are known), then substitute
  `V(s′)` from a value head; terminals still exact. The full `Q` vector at the decision point is
  preserved. **Requires a value head**, which the baseline does not have. This is soft policy
  iteration with bootstrapping — biased by the value error, convergent under the usual
  conditions.

  **The cost claim was wrong and is withdrawn (2026-08-19).** This paragraph used to say cost
  falls "by roughly the rollout depth, order 10–20×". That was derived from §13's assumed depth
  of ~15. G3 measured 2.6–8.1, and truncating at hero's *next* decision still plays every
  opponent reply in between — which in multiway is most of them. Netting the value forward
  against the rollout forwards saved leaves roughly 1.3–1.9× on the rollouts, and the posterior
  (5–73% of a label, §13) does not move at all: order 20–40% off the label. **Variant C is a
  variance lever, not a budget lever**, and the variance argument is the stronger one anyway: the
  bootstrap removes the runout and every downstream action draw, which is where the measured
  0.5–1.4-stack-deviations per sample come from. What it cannot touch is the rollouts that end
  before hero acts again — hero calling a shove is a showdown — and those are the highest-variance
  ones.
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

**Every checkpoint is measured against the oracle that taught it** (owner decision 2026-08-19).
Immediately after the agent of iteration *k* is trained, the loop reports how far its play is
from the oracle's on a **held-out slice** of that iteration's own labels — `heldout_fraction` of
them, withheld from training. This costs one agent forward per held-out decision and **no new
rollouts**: the oracle's answer is already in the shard. Seven numbers, all on the same slice:

| | |
|---|---|
| `ev_agent` | `E_{a∼π_agent}[Q_norm]` — the agent's own valuation of the decisions it faced, by the oracle's `Q`. Not a gap and not comparable across iterations whose label distributions differ, but it is the term both gaps are measured *from*, and it is the one that says whether a shrinking gap means the agent improved or the oracle got easier to match |
| `ev_oracle` | `E_{a∼π_oracle}[Q_norm]` — the same sum under the target, i.e. what the agent is being asked to reach |
| `q_best` | `max_a Q_norm` over the legal actions — the ceiling the greedy gap is taken from |
| `kl` | `KL(π_oracle ‖ π_agent)` — the training objective (§6.2) on data the agent did not see. Read against the training loss it is also the memorisation check |
| `ev_gap_target` | `E_{a∼π_oracle}[Q_norm] − E_{a∼π_agent}[Q_norm]` — what the imperfect fit costs, in the §6.2 pot-normalised units, by the oracle's own valuation. Signed on purpose: negative means the agent is *greedier* than the target it was trained on, which is a different failure from being worse |
| `ev_gap_greedy` | `max_a Q_norm − E_{a∼π_agent}[Q_norm]` — the classic policy-improvement gap, non-negative by construction. It cannot reach zero, because the target is a softmax and not an argmax; it is the number to watch across iterations, not to compare against zero |
| `agreement` | fraction of decisions where `argmax π_agent = argmax π_oracle` — coarse, and the one a human reads |

The last four are functions of the first three plus the two distributions, and all seven are
reported because a gap is a difference: one that moved does not say which of its sides moved.
`ev_agent`, `ev_oracle` and `q_best` are what disambiguate it.

Broken down by table size and by stack depth, which is free — it is a grouping of the same
numbers — and is where a failure of the `CLAUDE.md` §1 generalisation bet would show up first.

**What this is not.** It is measured on the state distribution hero itself generated and scored
by the oracle's own `Q`, which carries the Monte-Carlo error G3 measures. So it bounds nothing
about exploitability, it is not comparable to a benchmark, and its floor is the oracle's noise
rather than zero. What it does answer is the question the loop cannot otherwise ask: *did this
iteration's training actually absorb this iteration's labels*, separately from whether the labels
were any good.

### 8.1 Config sections

Sketch only — names and defaults settle when the code is written. The point of listing them here
is that the experiment surface is explicit (`CLAUDE.md` §5) and that two things the owner asked
to be separable actually are.

| Section | Contents |
|---|---|
| `bootstrap` | list of entries, each: checkpoint path (into `data/v7/`), `n_variants` (how many random style draws to generate from it), optional explicit style vector (for the hand-designed degenerate strategies), optional label |
| `style` | the distribution style draws come from — per-block scale for the §4.2 bias vector, temperature and `λ` ranges — and `agent_variants`, how many members each trained agent contributes when it joins the pool (§4.1, owner decision 2026-08-19) |
| `agent_init` | which pool member sits in hero's seat at iteration 0 (§7.1), the agent itself being trained from scratch. **Deliberately separate from `bootstrap`** — the strongest opponent to have in the pool and the best policy to seat as hero are different questions and need not resolve to the same member |
| `embedding_net` | model dims, `K` (inference gradient steps), fit learning rate, regularisation toward zero, `R` (recompute interval), amortised-head weight, the §5.6 `strength_weight`, ablation switch, **`first_retrain_steps` — the first retrain's own step count, `steps` being every later one's** (owner decision 2026-08-20; it is the only retrain that starts from a random network *and* the one whose vectors are stamped onto iteration 0's labels, which is the same argument `first_iteration_steps` rests on and it uses the same helper), and the number of table rows reserved for the members the loop will produce (`max_iterations` × `style.agent_variants`, owner decisions 2026-08-19, see §5.4 and §4.1). Nothing in the loop decides that number — it is config, like every other hyperparameter that changes a result |
| `oracle` | samples per action, `max_combos`, variant (A / C), EV normalisation divisor, temperature `T` |
| `pool_sampling` | PFSP exponent, uniform floor fraction, dedup cluster count, and the per-iteration decay of accumulated results (`result_decay`, owner decision 2026-08-19). The decay is not a tuning knob but a correctness one: hero changes every iteration, so "hero's BB/100 against member *i*" is a property of a pair and a lifetime mean is a mean over every past agent. Without it a member the early agents crushed keeps a positive score for the length of a whole run, the loop never learns it has stopped beating it, and §11.1's cycling diagnostic is smeared away with it. Optional `evaluation_weight` (0.5 in the main config, otherwise 0) blends a separate decayed estimate of the current agent's training/warm evaluation into next iteration's PFSP; see `PLAN_POOL_EVALUATION.md`. Independent session means are normalised per iteration, so the evaluation budget does not multiply their weight. Uniform/cold evaluation and the old candidate remain diagnostic only |
| `game` | `raise_sizes` per street, players range (2–9), stack range (10–300 BB) |
| `agent_train` | optimiser, embedding-dropout probability, `warm_start_trunk` (§6.1), schedule, the gradient steps per cycle — with **iteration 0's count a key of its own** (`first_iteration_steps`, owner decision 2026-08-18), see §8 — and `heldout_fraction`, the share of each iteration's labels withheld from training so the oracle gap can be measured on data the agent did not see (§8) |
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

### 11.4 What the training limit is, and what it is not

Recorded 2026-08-19, after the `soft_q` decision of §6.2, because the question "does this
converge to the right thing" has a precise answer and three of its parts are easy to assume away.

**What the linear loss buys, and it is worth stating first.** `soft_q`'s gradient is linear in
`Q̂`, so `E_ε[∇_θ L] = ∇_θ L|_{Q̂ = E[Q̂]}` and `samples_per_action` **drops out of the limit
entirely** — at `N → ∞` the answer is the same whether a label was built from 32 samples or
1024. Only the gradient variance, i.e. the rate, depends on it. That is why §13 can pick `n`
purely on compute efficiency. Under the KL loss of §6.2 `n` would have been a parameter of the
*fixed point*, which is the failure this decision exists to avoid.

**What the limit actually is.** Even with an unbiased `Q̂`, `N → ∞` gives

```
argmin_θ  E_s [ T·KL( π_θ(·|s) ‖ softmax(Q(s)/T) ) ]
```

— the **reverse-KL projection onto what the network can represent**, weighted by the state
distribution the labels came from. It equals `softmax(Q/T)` pointwise only under realisability,
the network is non-convex so SGD reaches a stationary point rather than the global one, and
changing the PFSP weights or the hero seat changes the distribution and therefore the limit. So:
consistent for a well-defined object, not unbiased for the one §6.2 names.

**Two sources of range bias and a distinction in the oracle's objective.**
Increasing rollout count or training-set size alone cannot remove these.
As corrected in §7.3, collision rejection itself introduces no range bias;
folded-seat conditioning is now retained. Historical collision rates describe
sampling cost, not distance from the true joint distribution.
- **The likelihood floor** (`1e-6`) deliberately gives a combo the member would never play a
  non-zero weight. A fixed smoothing of the range; it does not vanish, and it applies heads-up
  too.
- **`max_combos`** is self-normalised importance sampling, biased at finite `m` and consistent
  only as `m → ∞`. Inactive in the baseline, which uses the exact posterior.
- **It is `Q^{π_rollout}`, not `Q*`.** The oracle evaluates an action assuming hero plays the
  rollout policy afterwards, so iteration 0's labels are `Q^{v7}` and training on them is one
  step of soft improvement over a v7 member — see §4.1 on how weak that member is. Reaching a
  best response is the outer loop's job, and §11.1 already says that iteration is not guaranteed
  to converge.

**A practical trap in reading the loss curve.** Under `soft_q` the logged number does not go to
zero at the perfect solution. At `π = softmax(Q/T)` it logs `T·KL(π ‖ softmax(Q̂/T))`, which for
small `ε` is about `Var_π(ε) / (2T)`. With the measured `SE/pot ≈ 1.4` and `T = 0.5` that floor
is of order 2, and it is label noise rather than underfitting. The floor is set by the error on
the *differences* between actions, so it is the contrast error — not the level error §13
reports — that predicts where the curve flattens.

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

**Done 2026-08-19.** The inherited `evaluation/slumbot_eval.py` was split as described and then
deleted: the protocol layer (HTTP client, action-string grammar, token ↔ action mapping, state
replay, BB/100 + SE accounting) is `evaluation/protocol.py`, bodies verbatim, and the agent
adapter was rewritten against v8's observation format as `evaluation/v8_adapter.py`.
`eval_pipeline.py` plays the hands, runs cold and warm, stamps a short run `SCREENING ONLY` and
refuses to write a report without the selection disclosure above.

**"How many hands it took to warm up" is a derived number** (2026-08-19): no single quantity of
that shape exists in the run, so every warm hand records how many hands its vector had been
fitted from, the run is bucketed by that count, and the reported figure is the first bucket whose
BB/100 reaches the cold run's overall BB/100. *Never catching up* is reported as such — which,
read against §11.2's measured result that the fit is worse than `e = 0` at very short observation,
is a shape to expect rather than a surprise. The fit itself is over the most recent
`fit_window` hands rather than the whole history, because an unbounded fit over a million hands is
not computable; the window is config.

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

**Measured — first run, 2026-08-18** (before the §7.1 runout draw, before the configured hero
seat, before bf16). The design figure above was right on cost and wrong on both of its terms, by
compensating factors:

| | design figure | measured |
|---|---|---|
| forwards per label | ~10⁵ | 4.7k–13.4k |
| sustained forwards/s | 3 × 10⁴ | 1.0k–4.1k |
| seconds per label (256 samples) | ~3 | 9.9 |
| remaining decisions per rollout | ~15 | 2.6–8.1 |

100k labels cost 60 h at 32 samples and 276 h at 256. The wall clock splits 4.9% Python event
building, ~75% the model itself, ~10% tensor packing and ~10% driver and engine — so this is
real compute on a `d_model=384`, 18-layer network at roughly 10 TFLOP/s in fp32, not a Python
bottleneck. The pool member ran with **no autocast at all**; bf16 through `utils.get_amp_config`
is the outstanding lever and is untested on sm_121.

The number that decides the design is not the cost but the noise: split-half `SE(q)` ran 0.65 BB
(heads-up, 20 BB) to 25.7 BB (six-handed, 300 BB) at 256 samples. Since cost is linear in samples
and error is inverse-square-root, reaching 1 BB needs 108 samples at heads-up 20 BB and 169 000
at six-handed 300 BB. **Variant A is affordable and its labels are only clean at short stacks.**
Three things qualified that and are the reason the gate was rerun rather than the design
abandoned: the runout was pinned (§7.1), the error was measured on levels rather than on the
shift-invariant *differences* the target actually depends on, and hero's rollout policy was
whoever the hand seated — 29% of the pool being degenerate.

**Measured — second run, 2026-08-19.** 1728 labels over 216 cells, with the §7.1 runout draw, a
configured v7 hero seat, bf16 in the pool member and the pool composition as a fifth axis. This
is the run the config is fixed from.

*The noise scales as textbook Monte-Carlo, and that is the load-bearing result.* Fitting
`SE ~ n^k` over the four sample budgets gives a median `k` of **−0.51** in BB and **−0.46** in
pot units across the 18 cells, and −0.47 … −0.56 on the aggregates; `SE(32)/SE(256)` is
2.77–2.82 against a theoretical 2.83. **There is no plateau.** That is the direct evidence the
pinned board was the problem: with one runout shared by every sample the curve had to flatten
into an irreducible floor, and it no longer does. Quadrupling the samples halves the noise, and
nothing else does.

*But samples stop being worth buying at 128.* A label costs `a + b·n`, where `a` is the
posterior and does not move with `n`. For a fixed second-budget the total noise variance goes as
`(a/n + b)/C`, so the efficiency of a sample budget is **exactly the share of the label's wall
clock spent on rollouts rather than on the posterior** — measured at 68% (n=32), 77% (64), 85%
(128), 91% (256). The same thing read directly, in samples bought per Spark-hour:

| pool | n=32 | n=64 | n=128 | n=256 |
|---|---|---|---|---|
| all | 61 120 (58%) | 85 248 (81%) | **104 832 (100%)** | 102 912 (98%) |
| v7 | 55 840 (61%) | 75 712 (83%) | **91 136 (100%)** | 88 320 (97%) |

The curve peaks at 128 and turns slightly down after it — seconds per label grow marginally
faster than rollouts on the largest batches. Below 128 the budget is spent re-deriving
posteriors; above it, on nothing. **`oracle.samples_per_action = 128`** is therefore the answer
§14 says this gate exists to produce, and it costs 122 h (pool `all`) to 141 h (pool `v7`) for
100 000 labels.

*The rest of the second run's numbers.* Wall clock splits 6.9% event building, 13.9% tensor
packing, 68.8% model, 0.3% style, 10.3% driver — 584 µs per policy row at 172 rows per call,
85% of rows reaching a network. bf16 bought about **1.5×** on the model, not the 2–4× predicted:
the two runs are not directly comparable (hero, pool mix and hand lengths all changed), so the
honest normaliser is the model's cost per unit of `events`, which is pure Python proportional to
sequence length — it fell from 15.3 to 10.0. The packing share came in at 13.9%, against a
prediction of 2.15× the `events` bucket made from a dev-box calibration; measured 2.02×, so the
calibration held and packing is now the largest remaining engineering item. Collision rates fell
to 0–5% (heads-up), 33–46% (six-handed) and 57–82% (nine-handed), because a card of a street
still to come is no longer dead.

*Pool composition moves both columns, as §4.4 implies it will.* Against the networks-only pool
the error is about 23% lower in pot units (1.05 against 1.36 at 256 samples) and the label is
about 15% more expensive. So the uniform draw over a pool that is 29% degenerate was inflating
the measured variance, and the pipeline's PFSP draw will sit on the quieter side of that.

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
per label, and how it scales with samples-per-action, `max_combos`, table size, stack depth and
**pool composition** — the last because §4.4 samples opponents by PFSP with a uniform floor while
the gate sampled uniformly over a pool that is 29% degenerate, so cost and noise were both taken
against a table the pipeline will not set. Hero's rollout policy is a configured member and not
whoever the hand seated, for the same reason. Output is the label budget that §8 can afford,
which then fixes the config.

**Answered, 2026-08-19: `oracle.samples_per_action = 128`.** Not because the labels are clean
there — they are not, §13 has the numbers — but because that is where the sample budget stops
buying anything: below it the label re-derives a posterior it does not use enough, above it the
curve of samples-bought-per-Spark-hour is flat and then turns down. §13 carries the derivation,
the per-budget table and the noise-scaling result the choice rests on.

Two additions to what this section originally asked for, both because a cost without an accuracy
is not a budget: a split-half standard error of `q`, free because the rollouts are already
played, reported in BB **and** in units of §6.2's `pot + facing_bet` divisor; and a breakdown of
where a label's wall clock goes — event building, tensor packing, model, style, driver — which is
what says whether the answer is an engineering problem or a hardware one. The split-half gaps are
stored **signed**, so the noise on the differences between actions can be recovered offline;
that is the quantity the target depends on, and squaring them first destroys it.

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

- **OI-9 — accuracy of the `max_combos` subsample. OPEN, deferred 2026-08-19 by the owner.**
  The cap is the only lever on a label's fixed cost (§7.3) and its accuracy is unmeasured; G3
  sweeps the axis for cost only. The experiment is cheap — the same decisions, one seed, `Q̂` and
  the posterior's effective sample size at `max_combos ∈ {128, 512, None}` — but it is deferred
  because on the bootstrap pool it would be measuring the shape of a −90 BB/100 strategy's range
  (§4.1). Nothing is blocked: the baseline runs the exact posterior. Revisit once the pool holds
  a v8 agent that is not obviously weak.
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
    **Settled 2026-08-20: it is the default** (`agent_train.warm_start_trunk`), and the open
    question is closed by §5.6 rather than by an experiment — with the strength head in the
    objective the trunk being copied carries a poker prior and not only an action predictor,
    which is what makes the copy worth making. §6.1 carries the mechanism.
- **OI-5 — live style modifiers. APPROVED 2026-08-16.** Design in §4.2: additive logit bias over
  v7's 5 action categories, gated by position and street, plus temperature and a uniform mix;
  ~32 scalars per pool member, drawn randomly at pool construction; equity-gated conditions
  dropped as too expensive inside rollouts.
- **OI-6 — v7 checkpoint access. RESOLVED 2026-08-16:** vendored frozen copy of v7's agent code
  inside v8, checkpoint paths in config pointing into `data/v7/` (§4.3). Note that v7's **event
  builder** must be vendored alongside the model — the reference implementation came from
  v7's `slumbot_eval.py::_build_events` and now lives at `vendor/v7/events.py`.
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
