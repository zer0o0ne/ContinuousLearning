# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

This file is **architecture-independent** and applies to every version of the project.
Anything specific to a version — model architecture, training phases, config schema,
data formats, plans — lives in that version's own documents:

```
versions/<version>/ARCHITECTURE.md   ← how that version works
versions/<version>/PLAN_*.md         ← design notes for that version
```

`.md` files inside a version directory belong to that version **only**. They are never
copied into a new version. A new version starts with its own `ARCHITECTURE.md`.

The whole project was written by Claude Code, so you are responsible for every bug in it.

---

## 1. Goal

### What the agent must be able to do

Train **one** agent that plays no-limit hold'em across the full range of table
configurations:

- **2 to 9 players** at the table
- **10 BB to 300 BB** effective stacks

No table size and no stack depth is privileged during training. All of it is one
distribution the agent is expected to handle.

**Sampling is uniform over both axes.** No weighting, and in particular no weighting
chosen to favour the measured slice. This is not merely a default: it encodes the
expectation that the learned strategy **generalizes smoothly** — that it does not
degrade under a small change in the number of players or in stack depth. That
expectation is falsifiable, and a sharp drop across neighbouring configurations is a
result worth reporting, not a nuisance to tune away.

### What is measured

The headline metric is **BB/100 against Slumbot** — which is a single slice of that
space: **heads-up, 200 BB**.

At this stage, **any positive winrate against Slumbot counts as success.**

Every candidate that looks promising is measured over **at least 1,000,000 hands**.
Shorter runs are screening only: they select what to measure properly and are never
reported as results. Slumbot's BB/100 carries a standard error that shorter runs do
not resolve — quote it alongside the number, always.

### The bet behind this

The training distribution and the measured slice are deliberately different, and that
gap is the point. Nothing in training is specialized to Slumbot, to heads-up, or to
200 BB stacks; the agent arrives at that slice only as one case among many.

The reasoning: if an agent that was never told the benchmark exists still plays it
respectably, that is evidence of *general* strength rather than of fitting to a
benchmark — and it should therefore carry over to the rest of the space (short stacks,
full ring) where no comparable benchmark is available to us.

### What this forbids

Concretely, in every version:

- no training distribution restricted to heads-up or to 200 BB — table size and stack
  depth are sampled across their full ranges
- no Slumbot-specific opponent modelling, exploit, or hardcoded response
- no heads-up-only shortcuts in the observation format, the model, or the search
  (e.g. assuming exactly one opponent)
- no stack-depth assumption baked into the observation format or the action space
- benchmark results must not feed back into training — that is the mechanism by which
  specialization creeps in unnoticed

If a change would raise the Slumbot number by narrowing the agent, it is the wrong
change: say so instead of making it.

Adapting to Slumbot's wire protocol and bet-size grammar at *evaluation* time is not
specialization — that is plumbing, and it belongs in `evaluation/`.

### Fallback — NOT ACTIVE

The generality above is a bet, and it is stronger than anything demonstrated in the
literature: the systems that beat heads-up NLHE benchmarks (DeepStack, Libratus, ReBeL)
were all specialized to heads-up, and Pluribus went 6-max only at a single fixed stack
depth. No published system covers 2–9 players × 10–300 BB *and* competes at heads-up
200 BB.

So there is a fallback: **if results are weak, training is narrowed to the target
situation — heads-up, 200 BB.**

This fallback is **not active and must not be anticipated.** It does not license
partial specialization now, nor "harmless" heads-up shortcuts kept in reserve.
Everything under "What this forbids" holds in full until the owner explicitly decides
to switch. Until then, write code as if the general problem is the only problem.

---

## 2. Active version

```
ACTIVE VERSION: v8
```

**`versions/v8/` is the only directory you may edit.**

Hard rules — no exceptions, no "just a small fix":

1. **Never modify any file under `versions/` other than `versions/v8/`.** Older versions
   (`v0`…`v7`, `template`) are frozen historical baselines. They must stay runnable exactly
   as they are. If you find a bug in `v7`, report it — do not fix it there.
2. **Read-only access to older versions is expected and encouraged.** Use them as reference:
   `versions/v7/ARCHITECTURE.md` documents the previous 6-phase architecture in full.
3. **All data lives outside `versions/`**, under `/data/<version>/…` (gitignored).
   Never write model checkpoints, datasets, or logs into a version directory.
   `v8` must never write into `data/v7/`.
4. When the active version changes, this section is updated first, and only then does work
   on the new version begin.

### Running

```bash
./run.sh      --version=v8   # → cd versions/v8 && python3 pipeline.py
./evaluate.sh --version=v8   # → cd versions/v8 && python3 eval_pipeline.py
```

Every version is a self-contained tree run with its own directory as the working directory;
imports are absolute from the version root (`from env.table import Table`). Consequence:
**two versions can never be imported into the same Python process** — they define the same
top-level package names. Do not attempt cross-version imports; use a subprocess boundary if
two versions genuinely have to interact.

---

## 3. Hardware: written here, executed there

The project is developed and executed on **two different machines with different CPU
architectures**. This shapes everything below.

| | Development box | Execution box |
|---|---|---|
| Role | writing code, tests, review | training, evaluation, benchmarking |
| CPU | x86_64, 8 cores | NVIDIA GB10 Grace Blackwell, 20-core **arm64/aarch64** (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | **none** — no CUDA, no driver | Blackwell, 6144 CUDA cores, compute capability **sm_121** |
| Memory | ordinary host RAM | **128 GB unified LPDDR5X**, ~273 GB/s, shared by CPU and GPU with no static partition |
| OS | Linux | DGX OS (Ubuntu-based) |

### What follows from this

**You cannot execute a single line of GPU code while writing it.** Any statement about GPU
behaviour, kernel availability, dtype support, throughput, or memory footprint is a
**hypothesis until it has been run on the Spark**. Say so explicitly instead of reporting
GPU-path work as verified.

- **Device-agnostic code, always.** No hardcoded `.cuda()`, no assumed CUDA availability.
  Resolve the device once (CUDA → MPS → CPU) and pass it down. Every code path must be
  executable on CPU, if only at toy scale — otherwise it is untestable before deployment.
- **The CPU test battery is the only local correctness gate.** See §4. If logic is not
  covered there, it is unverified until someone runs it on the Spark, which is slow and
  expensive feedback. Push correctness into CPU-testable code.
- **arm64 ≠ x86_64.** Every dependency needs an `aarch64` wheel, and compiled extensions
  (e.g. `eval7`) built on the dev box tell you nothing about the target. Before adding any
  dependency, check that it publishes an aarch64 build; flag it if unsure.
- **sm_121 is new.** It needs a recent CUDA (13.x) and a matching PyTorch build. Prebuilt
  kernels or wheels compiled for older architectures will not run.
- **Unified memory is one pool.** GPU allocations come out of the same 128 GB as host RAM.
  A GPU OOM can take the whole machine down rather than just the CUDA context. When sizing
  multi-process work (N CPU actor processes + a GPU inference server), budget host and
  device memory **together**, not separately.
- **~273 GB/s is modest bandwidth** — roughly an order of magnitude below a datacenter GPU.
  Memory-bound kernels dominate. Prefer large batches and compute-dense formulations, and do
  not extrapolate throughput from published benchmarks run on other hardware.
- **20 CPU cores** is the ceiling for parallel actor processes; leave headroom for the
  inference server and the OS.
- **Linux CUDA requirement**: `vm.max_map_count` must be ≥ 1048576 (the 65530 default is too
  low for `expandable_segments:True`). Check with `sysctl vm.max_map_count`, fix with
  `sudo sysctl -w vm.max_map_count=1048576` (persist in `/etc/sysctl.conf`). Without it the
  inference server can crash with ENOMEM.
- Use the project venv when one is present (`source venv/bin/activate`). The dev box
  currently has no venv and uses the system `python3`.

---

## 4. Testing

**Every version carries its own test battery** under `versions/<version>/tests/`.

```bash
cd versions/v8 && python3 -m pytest tests/ -q
```

**Hard requirement: the full battery must complete in under 30 minutes on the dev box —
CPU-only, 8 cores.** This is a design constraint on the tests, not an aspiration. A test
that needs a GPU, or that needs realistic training scale, does not belong in the battery;
shrink the model, shrink the data, or move the check elsewhere. The battery is run often,
and it is worthless if it is not run.

### What tests MUST cover

**All algorithmic and mathematical logic, including corner cases.** This is not negotiable
and it is where essentially all testing effort goes:

- every loss, normalization, and target-construction formula
- every probability distribution: masking, renormalization, temperature, degenerate cases
- game-rule logic: betting, side pots, all-ins, short calls, uncalled-bet refunds, showdown
- search and tree logic: selection rules, backup, terminal evaluation
- boundaries: empty inputs, single element, all-masked, zero denominators, ties, off-by-one
  at sequence ends, maximum stack/pot, minimum raise
- conservation and invariants (chips in = chips out; probabilities sum to 1; no future leak)

Tests must be **deterministic**: fixed seeds, no probabilistic assertions, no dependence on
test execution order. A flaky test is worse than no test — it trains everyone to ignore red.

Prefer **end-to-end scenario tests over unit tests of internal helpers**: exercise real
behaviour and real outcomes through the public entry points. Internal helpers get refactored;
behaviour is the contract worth pinning.

### What tests should NOT be

**Code-quality tests** — asserting on source text, grepping for forbidden patterns, checking
that a function exists or has a given signature — are allowed only where genuinely necessary
(guarding an invariant that has been broken before and cannot be caught behaviourally).
They are the exception, not the norm. They break on every refactor and prove nothing about
correctness.

---

## 5. Engineering principles

### This is a research project

The purpose of this code is to test hypotheses about how to train a poker agent, and the
architecture and training method are expected to change repeatedly. Optimize for **speed of
the next change**, not for the elegance of the current state:

- prefer small, composable pieces that can be recombined over monolithic paths
- make the experiment surface (config, hyperparameters, on/off switches) explicit
- keep the ability to run a variant without rewriting the pipeline around it
- do not build infrastructure for a generality nobody has asked for yet

### Composition over new code

The codebase must be as **compositional** as possible. Functions and classes should be
reused, extended, and inherited from — not duplicated with variations.

**New low-level logic should appear as rarely as possible, and only after discussing it
with the owner.** Before writing a new primitive:

1. Search for an existing function, class, or module that already does it (or nearly does it).
2. If something nearly does it, prefer extending, parameterizing, or subclassing it.
3. Only if nothing fits, **propose the new primitive and get agreement before implementing it.**

Implementing something through existing code is always the better outcome, even when the
result is slightly less direct. Duplicated low-level logic is how this project accumulates
silent divergence between paths that were supposed to be identical.

### Implement literally

Implement what was asked, as asked. Do not add extra fields, abstractions, config knobs,
fallbacks, or "while I was there" improvements. If something extra seems necessary, say so
and ask — especially in search, training, and numerical code, where an unrequested addition
can silently change results.

---

## 6. Working method: critical thinking and literature

This is a research project, so **the reasoning matters as much as the code**. Implementing a
hypothesis that the literature already refuted is pure waste — both in the hours to build it
and in the training runs to disprove it.

- **Question ideas — mine and yours alike.** When the owner proposes a direction, engage with
  it critically: what would have to be true for it to work, what would falsify it, what is the
  cheapest experiment that discriminates it from the alternative. Agreement by default is not
  helpful here; disagreement stated plainly and early is.
- **Check the literature before building.** Poker/imperfect-information RL is a well-studied
  area (CFR and its variants, DeepStack, Libratus/Pluribus, ReBeL, Player of Games, search in
  imperfect-information games, opponent modelling). Before implementing a non-trivial
  hypothesis, look for whether it has been tried, and what the known failure modes are.
  Web search is available — use it.
- **Say when you don't know.** Distinguish "the literature says X", "this follows from the
  math", "this is my guess", and "this is untested on our hardware". Never present the last
  three as the first.
- **Name the cost.** For an expensive idea, estimate what it costs to test — dev time,
  training time on the Spark — and whether a cheaper proxy experiment exists.

---

## 7. Repository layout

```
run.sh, evaluate.sh          entry points, take --version=<v>
requirements.txt
CLAUDE.md                    this file — architecture-independent
versions/
  template/                  minimal skeleton (frozen)
  v0 … v7/                   frozen historical versions
    ARCHITECTURE.md          how that version works
  v8/                        ACTIVE — the only editable version
    ARCHITECTURE.md
    tests/
data/                        gitignored; all checkpoints, datasets, logs
  <version>/<experiment>/…
```
