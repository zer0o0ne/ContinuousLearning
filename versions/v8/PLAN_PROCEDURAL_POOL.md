# v8 — Implementation plan: procedural regulars in the opponent pool

**Status: P1–P5 built 2026-09-03 — the board-strength layer, the per-decision features and ranges, the stat line, the rule cascade, the ten archetypes with their gate, and a pool member as the hero at Slumbot's table. The Slumbot runs themselves are the owner's, on the Spark. P6 (into the pool, and the label-cost gate) is plan only, and nothing here is wired into training yet.**
**Scope: replace the five degenerate strategies as the procedural half of the pool with
archetype agents that play like recognisable humans, and validate each archetype's
configuration.**

This document is the build order for the "1c" proposal of 2026-09-02: a per-board hand-strength
layer with range-weighted equity, a parametric rule cascade on top of it, and a set of named
archetypes (nit, loose-passive with and without bluffs, maniac, TAG, bluffer, weak-tight
fit-or-fold, trapper, modern polar reg, positional stealer — ten in all; the last four were
added 2026-09-02 at the owner's request, §P4 says what each one adds), each with a
configuration that has been checked against a stat profile in self-play and against Slumbot
over 10 000 hands.

- **Why** the pool exists and what a member is → `CONCEPT.md` §4
- **What is on disk today** → `ARCHITECTURE.md` §2.5 (pool), §2.2b (posterior)
- **In what order to build this, and exactly what each step is** → this file

Project-wide rules are in the root `CLAUDE.md`. Three of them decide the shape of this plan:

- **Composition over new code.** Every new primitive is marked ⚠ and collected in §1 for one
  sign-off conversation.
- **The CPU battery is the only local gate.** Everything below is exercised on CPU at toy
  scale; the Slumbot runs (P5) execute on the Spark or wherever the wire is reachable and
  their numbers are hypotheses until they come back.
- **Benchmark results must not feed back into training** (`CLAUDE.md` §1). P5 measures pool
  *members* against Slumbot. §0.3 records the owner's decision on what that measurement is for
  and what it is not allowed to become.

---

## 0. What was decided, and the two constraints that shape everything

### 0.1 Rejected alternatives

- **Distilling a solver into a small policy** — rejected by the owner 2026-09-02: distilled
  policies lost EV against Slumbot at a scale that made them useless as opponents.
- **E[HS] by sampled runouts inside the strength table** — deferred, not rejected. The control
  variate already pays 16 runouts per street per sample (`env/runout.py`); doubling that for the
  pool's benefit is a decision to take on a measured label cost, not now. v1 gets hand potential
  from an outs count, which is also what a human regular does. Sign-off item ⚠1 keeps the door
  open with one field, `samples = 0`.
- **Range narrowing by postflop action** — out of v1. Ranges are preflop-implied only (§P2).
  A regular does narrow ranges postflop; the cost is a per-decision update over 1326 combos
  per opponent, which is the posterior's job and not the pool's. Revisit after P6's cost gate.

### 0.2 The two constraints

1. **A member answers for ~1225 holdings at once.** `opponent_posterior` asks every pool member
   "what would you have done holding *this*" for every combo consistent with the board, at every
   opponent decision of a labelled hand (`ARCHITECTURE.md` §2.2b). A rule that is written for
   one hand and looped 1225 times is the wrong shape. Every quantity in this plan is therefore
   either **per decision** (history, pot, stacks — identical for all 1225 rows) or **per combo on
   a board** (strength, draws — a table row). The cascade is numpy over rows; there is no
   per-row Python.
2. **Anything expensive is paid once per board.** Measured on the dev box, CPU, 8 threads, with
   the project's own 7-card evaluator:

   | rows scored                         | wall  |
   | ----------------------------------- | ----- |
   | 1 326 — every combo on one board   | 9 ms  |
   | 26 520 — every combo × 20 runouts | 30 ms |

   One board table (P1) is one 1 326-row call, cached by board, and serves every member, every
   decision and every posterior query on that board. The evaluator accepts 5-, 6- and 7-card
   rows (checked 2026-09-02), so flop and turn tables need no runout at all.

### 0.3 What the measurements are for — owner decision 2026-09-03

The owner settled this after the band machinery below was proposed and rejected as
overbuilt: **"нашими с тобой параметрами под сламбота подстроиться невозможно — это требует
колоссального перебора. Наша задача действительно — просто увидеть, что наши агенты
1. Разнообразны 2. Лучше детерминированных стратегий. Больше задач нет."**

So there are exactly two questions, and every measurement in §P4 and §P5 answers one of them:

1. **Are the archetypes diverse?** — the self-play stat profile. Ten archetypes that produce
   ten distinguishable stat lines, at every table size, is the whole claim.
2. **Are they better than the five degenerate strategies?** — a head-to-head in self-play,
   in BB/100. This one needs no external opponent at all.

**Dropped, with the reason.** Pre-registered acceptance bands, a pass/fail gate on BB/100, the
one-re-run rule and the stamped-report ceremony are **not built**. The concern they answered was
a path from the benchmark into the training distribution, and the owner's judgement is that with
two dozen hand-set knobs there is no such path: fitting a pool member to Slumbot would take a
search nobody is going to run by hand. The `CLAUDE.md` §1 prohibition is untouched — it is about
the *agent*, and nothing here feeds an agent.

**What remains of §P5**, if it is run at all: one descriptive run per archetype, its BB/100 and
standard error recorded, ordering the archetypes the way a human would order them. It gates
nothing. The one result that would still cause a change is a member that *beats* Slumbot or
loses several hundred big blinds per hundred — both mean a bug, and a bug is fixed whether or
not a benchmark found it.

---

## 1. Sign-off list — every new primitive in this plan

| #   | Primitive                                                                                                                                                                                                 | Where                  | Why nothing existing does it                                                                                                                                                                                                                      |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⚠1 | `BoardStrength` — per-board table over all 1 326 combos: made-hand score, exact percentile vs a random unblocked combo, hand class (top pair / overpair / …), draw flags, outs, range-weighted equity | `pool/strength.py`   | `env.showdown.strength_percentiles` does the river percentile for *revealed* hands only, one at a time, with a Python loop over hands. It is the model for ⚠1's percentile, and ⚠1 supersedes nothing: `label_showdowns` keeps calling it |
| ⚠2 | `preflop_equity_table` — 169 × 8 equity of each class vs *n* random opponents, Monte Carlo, stored under `/data/v8/tables/`                                                                       | `pool/strength.py`   | the repo has a 169*ordering* (`gto_utils/gto_helper.py::ORDER`) that depends on `eval7`, which is not the project's evaluator and has no `aarch64` guarantee. Multiway ranges need equity *vs n*, which no ordering gives               |
| ⚠3 | `Situation` — per-decision features read off a `HandRecord` prefix: position class, aggressor, bets this street, facing size, SPR, opponents' preflop-implied ranges                                 | `pool/situation.py`  | the tokeniser (`nets/tokeniser.py`) reads the same record for the network, but into tokens, not scalars; nothing exposes "who is the preflop aggressor"                                                                                         |
| ⚠4 | `RegularMember(PoolMember)` + `RegularParams` — the rule cascade                                                                                                                                     | `pool/regular.py`    | the degenerate members are the only procedural policies; none reads a board                                                                                                                                                                       |
| ⚠5 | archetype presets and jitter                                                                                                                                                                              | `pool/archetypes.py` | —                                                                                                                                                                                                                                                |
| ⚠6 | `hud_stats` — VPIP / PFR / 3-bet / c-bet / barrel / AF / WTSD over a list of records, per seat filter                                                                                                  | `pool/stats.py`      | G1's analysis reports losses, not table stats                                                                                                                                                                                                     |
| ⚠7 | edit`pool/build.py` — a `kind: "regular"` bootstrap entry                                                                                                                                            | existing               | —                                                                                                                                                                                                                                                |
| ⚠8 | edit`evaluation/v8_adapter.py::SlumbotAgent` and `eval_pipeline.py` — the hero may be any `PoolMember`, selected by `evaluation.hero`                                                            | existing               | `SlumbotAgent.policy` already builds a `PoolMember` per call and asks it `policy([ctx])`; the edit generalises which member                                                                                                                 |
| ⚠9 | `gates/pool_realism.py` — self-play stat-profile gate per archetype and table size                                                                                                                     | `gates/`             | the G1/G3 gates measure the embedding and the label; nothing measures a member                                                                                                                                                                    |

Not new, reused as-is: `evaluate_hands` (scoring), `hand_class_169` (class index), `PoolMember`
/ `StyleParams` (last mile), `LockstepDriver` + `HandSpec` (self-play), `RaiseGridMap.nearest_bin`
(size → bin), `slumbot_record` / `SlumbotClient` (wire), `utils.progress` (bars),
`stderr_bb_per_100_online` (reporting).

---

## 2. How to use this document

Each section from §P1 on is one Claude Code session:

> Реализуй раздел P<n></n> из `versions/v8/PLAN_PROCEDURAL_POOL.md`.

Fields per session are those of `PLAN_PIPELINE.md` §0 (Depends on / Reads first / Deliverables /
Design / ⚠ Sign-off / Tests / Acceptance / Non-goals), and the close-out protocol is the same:
whole battery green with its wall clock reported (today ~222 s; budget 30 min), `ARCHITECTURE.md`
updated, untested-at-scale items named, no commit unless asked.

Dependency graph:

```
P1 strength ──┐
              ├─→ P3 cascade ─→ P4 archetypes + realism gate ─→ P5 Slumbot ─→ P6 pool integration
P2 situation ─┘                                                  (§0.3: P5 is descriptive)
```

P1 and P2 are independent and may be two sessions in parallel.

---

## P1 — The board strength table

**Depends on:** nothing.
**Reads first:** `env/showdown.py` (whole file), `gto_utils/gpu_solver.py::evaluate_hands`,
`env/runout.py` (the caching pattern), `CLAUDE.md` §4.

**Deliverables:** `pool/strength.py`, `tests/test_strength.py`.

```python
ALL_COMBOS: np.ndarray            # (1326, 2) int64, c0 < c1, fixed order
def combo_index(c0, c1) -> int    # inverse of ALL_COMBOS
COMBO_CLASS: np.ndarray           # (1326,) the 169 class of each combo (hand_class_169)
DISJOINT: np.ndarray              # (1326, 1326) bool, True where two combos share no card

def preflop_equity_table(path, seed=0, n_deals=2_000_000) -> np.ndarray   # (169, 8)
def preflop_rank_pct(table, n_opps) -> np.ndarray                          # (1326,) in [0, 1]

class BoardStrength:
    board: tuple                  # 3, 4 or 5 cards
    live: np.ndarray              # (1326,) bool — combos not blocked by the board
    score: np.ndarray             # (1326,) int64 made-hand score; 0 where not live
    hs: np.ndarray                # (1326,) exact percentile vs a random live opponent combo
    hand_class: np.ndarray        # (1326,) int8, HandClass enum below
    flush_draw, oesd, gutshot, backdoor_flush: np.ndarray   # (1326,) bool
    overcards: np.ndarray         # (1326,) int8 in {0, 1, 2}
    outs: np.ndarray              # (1326,) float
    def ehs(self, n_opps) -> np.ndarray          # (1326,) hs^n + (1 − hs^n)·p_improve
    def range_equity(self, weights) -> np.ndarray   # (1326,) equity vs a weighted range, now
    # board scalars
    paired: bool; monotone: bool; two_tone: bool; high_rank: int; wetness: float

class StrengthCache:
    def __init__(self, max_boards=2048)
    def get(self, board) -> BoardStrength        # board: the visible cards, no −1 entries
```

**Design.**

*The combo grid is fixed and global.* `ALL_COMBOS` is the 1 326 pairs in lexicographic order.
Every per-combo array in this plan is indexed by it; a `DecisionContext`'s holding becomes a row
via `combo_index`. `DISJOINT` is 1.7 MB of bool, built once at import.

*Score.* One evaluator call over `live` combos concatenated with the board — 5, 6 or 7 cards,
the evaluator takes any of them. Blocked combos get score 0 and `hs = 0`.

*Percentile with card removal, exactly.* `hs[h] = (Σ_o live·disjoint(h,o)·[score_o < score_h]

+ ½·[score_o = score_h]) / Σ_o live·disjoint(h,o)`. Implemented as two `(1326 × 1326)`comparisons masked by`DISJOINT`and reduced — under a millisecond, and it reproduces`strength_percentiles` bit for bit on the river (pinned by a test). The naive
  "rank among all combos" would be off by up to ~4 % for hands whose cards block many
  strong combos, which is exactly the hands a regular's rules care about.

*Hand class* (int8 enum, in rank order): `AIR, ACE_HIGH, UNDERPAIR, WEAK_PAIR, MIDDLE_PAIR, TOP_PAIR_WEAK, TOP_PAIR_GOOD, OVERPAIR, TWO_PAIR, TRIPS_SET, STRAIGHT, FLUSH, FULL_PLUS`.
Computed from rank histograms of hole vs board: which hole rank pairs which board rank, and
where that board rank sits in the board's descending order; "good kicker" = other hole card ≥ T
or ≥ the second board rank. Board-paired boards: a hole pair above the board pair is `OVERPAIR`;
a hole card matching the board pair is `TRIPS_SET`. `hs` is the number the cascade thresholds
on; `hand_class` is what its *human-readable* rules mention (a nit "needs top pair good kicker").

*Draws* (flop and turn only; on the river all flags are False):

- `flush_draw`: some suit has exactly 4 cards among hole+board **and at least one hole card is
  in it**; `backdoor_flush`: 3 on the flop, same hole-card condition;
- straights: for each of the 10 five-rank windows (A-5 … T-A), count present ranks among
  hole+board; a window with 4 present is a draw *if a hole card is in it*. `oesd` when the set of
  ranks that complete some window has size ≥ 8 (open-ender or double gutter), `gutshot` when
  it is 4–7 and not `oesd`, neither when the hand already has a straight;
- `overcards`: hole ranks strictly above the top board rank, counted only when the hand has no
  pair.

*Outs.* `outs = max(9·fd, 8·oesd, 4·gut) + 2·[second draw present] + 1.5·overcards + 1·bdfd`,
capped at 15. `p_improve = outs · (4/100)` on the flop, `outs · (2/100)` on the turn, `0` on
the river — the rule of four and two, which is what a regular uses at the table.
`ehs(n) = hs^n + (1 − hs^n) · p_improve`. This is Poki's EHS′ (potential without negative
potential) with a table-talk PPOT; the exact PPOT is what §0.1 deferred.

*Range-weighted equity.* `range_equity(w)[h] = Σ_o w_o·disjoint(h,o)·([s_o < s_h] + ½[s_o = s_h]) / Σ_o w_o·disjoint(h,o)`, same masked matmul, `w` a `(1326,)` non-negative vector (P2 builds it
from a preflop range). Zero denominator (the range is fully blocked) → 0.5.

*Board scalars.* `paired`, `monotone` (3+ of one suit on the flop, 4+ later), `two_tone`,
`high_rank` (top board rank), `wetness` = share of live combos with `flush_draw or oesd or hand_class ≥ STRAIGHT`. Texture buckets for the cascade: `dry` if `wetness < 0.12` and not
`two_tone`, `wet` if `wetness > 0.25` or `monotone`, `mid` otherwise — thresholds pinned by a
test on named boards (K72r dry, 987ss wet, A84 two-tone mid).

*Preflop.* ⚠2: deal `n_deals` random 23-card sequences (9 holdings + board), score all 9 seats
in 200 000-row chunks with a `utils.progress` bar (`unit="deals"`), and for seat 0's class
accumulate its share of the pot against the first *n* opponents, `n = 1..8`, ties split. 2 M
deals ≈ 18 M rows ≈ 20 s on the dev box; ~12 000 deals per class → SE ≈ 0.5 %. Deterministic in
`seed`. Written to `/data/v8/tables/preflop_equity_v1.npy` on first use and loaded after —
**never into `versions/`**. `preflop_rank_pct(table, n)`: the percentile of each combo's class
in the combo-weighted ordering by equity vs *n* (pairs count 6 combos, suited 4, offsuit 12), so
"top 20 %" means 20 % of *combos*, which is what published ranges mean. Preflop `BoardStrength`
does not exist; the cascade reads the class table directly.

*Cache.* `StrengthCache.get(board)` keys on the visible cards as a tuple, evicts oldest past
`max_boards` (a table is ~40 KB, so 2 048 boards ≈ 80 MB per process). One cache per pool
build, shared by every regular member in that process. No randomness anywhere in P1 except the
preflop table's seed.

⚠ **Sign-off:** ⚠1, ⚠2.

**Tests** (`tests/test_strength.py`), all deterministic:

- river `hs` equals `strength_percentiles` for 50 random (board, holding) pairs, to 1e-12;
- card removal: on a board where hero holds the ace of the flush suit, `hs` differs from the
  naive rank by the blocked combos, checked against an explicit loop;
- hand classes on hand-written boards: A♠K♦ on A72r → `TOP_PAIR_GOOD`; A♠3♦ on A72 →
  `TOP_PAIR_WEAK`; QQ on J72 → `OVERPAIR`; 88 on J72 → `UNDERPAIR`; 7x on 772 → `TRIPS_SET`;
- draws: J♠T♠ on 9♠8♦2♣ → oesd and backdoor flush, no flush draw; K♠2♠ on Q♠7♠3♦ → flush
  draw; A♣5♣ on 3♦4♥K♠ → gutshot (wheel); no flags on any river;
- `outs` and `ehs` on the same cases, exact numbers; `ehs(1)` on the river equals `hs`;
- `range_equity` with all weight on the nuts is 0 for every other hand and 0.5 for the nuts
  itself only if it is not blocked; uniform weights reproduce `hs`;
- texture buckets on named boards;
- preflop table: with `n_deals = 20 000` and a fixed seed, AA > KK > … > 72o vs 1 opponent, the
  vs-1 equity of AA ≈ 0.85 ± 0.02, and equity vs *n* is decreasing in *n* for every class;
  `preflop_rank_pct` gives AA the lowest percentile and ~0.0045 (6/1326) of mass below it;
- cache: two `get` calls on the same board return the same object; eviction past `max_boards`.

**Acceptance:** every test above green; a `BoardStrength` for a flop builds in < 30 ms on the
dev box (asserted loosely at 200 ms so the test is not flaky).

**Non-goals:** runout sampling; postflop range narrowing; any per-hand Python loop.

### Outcome — built 2026-09-03

`pool/strength.py` and `tests/test_strength.py` are on disk; ⚠1 and ⚠2 signed off by the owner
on starting the session. Whole battery **523 tests, 264 s** (budget 30 min). Measured on the dev
box: a flop table builds in **17 ms** (the plan's ~30 ms), `range_equity` on a built table costs
**21 ms**, and the 2 M-deal preflop integral takes **21 s** and is now on disk at
`/data/v8/tables/preflop_equity_v1.npy`. Its numbers land on the published ones inside the
sampling error it claims: AA 0.854 vs 1 opponent (true 0.8517), KK 0.828 (0.824), AKs 0.668
(0.670), AKo 0.653 (0.654), JTs 0.578 (0.576), 72o 0.349 (0.349); the combo-weighted mean is
0.5002 / 0.3337 / 0.2501 / 0.2001 / 0.1668 / 0.1428 / 0.1249 / 0.1110 against the exact
1/(n+1).

**Four interpretations the plan left open, decided here.** They are pinned by tests, so changing
one is a visible break rather than a drift:

1. **Draw strength is counted in cards, not in windows.** A five-rank window missing exactly one
   rank has four cards that complete it, so `oesd` is *two or more* completing ranks (eight
   outs, open-ender or double gutter) and `gutshot` is exactly one. This is the only reading
   under which §P1's "size ≥ 8" and the `8·oesd / 4·gut` outs formula agree.
2. **The made-hand category comes from the evaluator; the paired-board rules override it.** A
   pocket pair above every board rank is `OVERPAIR` even where the evaluator sees two pair
   (QQ on 772), and a pocket pair *not* above the whole board is `UNDERPAIR` even where a player
   might say "second pair" — which is what §P1's own case, 88 on J72, asks for. Everything from
   two pair up is the evaluator's own category, so the class can never disagree with the score.
3. **`two_tone` is "a flush draw is live on this board"** — some suit twice over, and not
   monotone — read the same way on every street rather than only on the flop.
4. **`ACE_HIGH` is an ace in *hand* with no pair anywhere**, and `overcards` are what a hand has
   instead of a pair, so a hand that paired anything has none of them.

**Two departures from the plan's test list, both because the test as written cannot pass.**

- The river percentile is asserted equal to `env/showdown.py`'s to **1e-6, not 1e-12**. The
  reference does `0.5 * ties` on an int64 torch tensor, which is a float32, so the ~3e-8 gap is
  the reference's rounding and not this table's. The table is the exact one.
- **"AA > KK > … > 72o" is not resolvable at any battery-affordable sample size.** A class gets
  `n_deals × combos / 1326` samples, so a 20 000-deal table gives a pair class ~90 of them and a
  standard error of ~0.05 — wider than the gap between adjacent classes almost everywhere. What
  the test asserts instead is what *is* resolvable: the combo-weighted pot share equals `1/(n+1)`
  exactly (an invariant, not a statistic), pairs beat suited beat offsuit as groups, AA sits in
  [0.80, 0.90] against one opponent and falls with more, and `preflop_rank_pct` reproduces a
  hand-built ordering's cumulative combo mass to 1e-12. The accuracy evidence for the *shipped*
  table is the comparison with published equities above, which is a run and not a test.

**What is untested:** everything here is CPU-only, and nothing has run on the Spark. The two
numbers P3 will care about are the 21 ms of a `range_equity` call — the cascade needs a few per
decision group, against a 20 ms budget for a whole 1 225-row batch — and the fact that the
`1326 × 1326` reductions are memory-bound, which is the regime the execution box is worst at
(§3 of `CLAUDE.md`: ~273 GB/s). If P3's batch budget is missed, the reduction is the place to
look, and the fix is arithmetic (inclusion–exclusion over the two blocking cards) rather than a
bigger cache.

---

## P2 — Situation features, preflop-implied ranges, HUD stats

**Depends on:** nothing (P1's `preflop_rank_pct` is used only for ranges, and may be stubbed by
a fixed ordering until P1 lands).
**Reads first:** `env/driver.py` (`HandRecord`, `DecisionContext`, snapshot layout),
`env/table.py` (seat order: seat 0 is the small blind, the highest seat is the button;
heads-up seat 0 is both), `pool/style.py::position_bucket`, `evaluation/v8_adapter.py::_build`.

**Deliverables:** `pool/situation.py`, `pool/stats.py`, `tests/test_situation.py`,
`tests/test_hud_stats.py`.

```python
@dataclass
class Situation:
    street: int; n_players: int; n_live: int; n_behind: int
    pos_frac: float               # 0 = first to act postflop … 1 = button; HU: SB=1, BB=0
    is_sb: bool; is_bb: bool
    n_raises_pre: int; pf_aggressor: int; hero_is_pf_aggressor: bool; n_limpers: int
    n_bets_street: int            # bets/raises already made this street
    facing: float                 # to_call / pot, 0 when checked to
    facing_allin: bool; checked_to_hero: bool; hero_bet_prev_street: bool
    hero_barrels: int             # consecutive streets hero has bet, ending on the previous one
    last_aggressor: int           # seat of the last bet/raise on the previous street, −1 if none
    pot_bb: float; eff_stack_bb: float; spr: float; pot_odds: float
    opp_pf_action: dict           # seat → one of FOLD, LIMP, CALL, RAISE, RERAISE, UNOPENED

def situation(ctx) -> Situation
def opponent_ranges(sit, rank_pct_fn) -> dict     # seat → (1326,) weights, live opponents only
def hud_stats(records, seat_filter=None) -> dict
```

**Design.**

*Reading the record.* A `HandRecord` carries `decisions` (`snap_idx, acting_pos, action_idx, legal_mask`) and `snapshots` (`pot, bets, credits, players_state, turn, action`). `situation`
walks the decisions up to `ctx.snap_idx` once — it is per decision, shared by every combo, and
the cascade calls it once per `(record, snap_idx)` group. Preflop action class per seat is the
seat's *last* preflop decision: no voluntary chips → `FOLD`/`UNOPENED` (BB that never acted),
call at the blind level → `LIMP`, call of a raise → `CALL`, first raise → `RAISE`, any raise over
a raise → `RERAISE`. `n_behind` counts live seats that still act after hero on this street.
`eff_stack_bb` is the smaller of hero's stack and the largest live opponent's stack, in big
blinds; `spr = eff_stack_bb / pot_bb` with the pot as of this decision.

*Position.* Postflop order is seat order from the SB, so `pos_frac = seat / (n − 1)` for `n ≥ 3`;
heads-up the button is the SB, `pos_frac = 1` for seat 0 and `0` for seat 1. Preflop the cascade
uses `pos_frac` and the blind flags together (the SB acts last preflop HU, first postflop).

*Preflop-implied ranges.* `opponent_ranges` maps each live opponent's `opp_pf_action` and
position to a weight vector over combos, using `preflop_rank_pct(table, n_players − 1)`:

| action                     | weights                                                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `RAISE`                  | 1 on combos with`pct ≤ open(pos)`, `open(pos)` interpolated from 0.15 (first seat, 9-max) to 0.45 (button), HU SB 0.75 |
| `RERAISE`                | 1 on`pct ≤ 0.08`, plus 0.5 on `0.25 < pct ≤ 0.35` (polar)                                                             |
| `CALL`                   | 1 on`0.05 < pct ≤ 0.30` (caps: no premium, no trash)                                                                     |
| `LIMP`                   | 1 on`pct ≤ 0.55`                                                                                                         |
| `UNOPENED` (BB checking) | 1 on everything live                                                                                                        |

These are the ranges a *regular assigns*, not truths, and they are deliberately the same for
every archetype — how well one reads ranges is not a style axis in v1. Postflop they are not
updated (§0.1).

*HUD stats* (⚠6) per seat filter over records: `vpip`, `pfr`, `threebet`, `fold_to_threebet`,
`limp`, `cbet_flop`, `fold_to_cbet`, `barrel_turn`, `barrel_river`, `af` (bets + raises) /
calls postflop, `wtsd` (saw showdown | saw flop), `wsd` (won at showdown), `agg_pct` (bets +
raises) / (bets + raises + calls + checks), and three that tell the P4 additions apart:
`check_raise` (raises after a check on the same street | checks that later faced a bet),
`steal` (opens from the cutoff, button or small blind | unopened pots in those seats; HU: SB
opens | SB decisions in unopened pots), `overbet_pct` (bets and raises larger than the pot |
bets and raises postflop). Definitions are the PokerTracker ones; each is
`(numerator, denominator)` so bands can be checked with a count, not a ratio, when the
denominator is small.

⚠ **Sign-off:** ⚠3, ⚠6.

**Tests:**

- hand-scripted 6-max records (built with `LockstepDriver` and forced actions): UTG raises, CO
  calls, BB calls → on the flop `pf_aggressor` = UTG, `n_live` = 3, BB's `pos_frac` = 0.2,
  CO's `n_behind` = 0, `facing` = 0 when checked to, `hero_barrels` after a flop bet;
- HU: SB is `pos_frac` 1 and acts first preflop; `eff_stack_bb` and `spr` on a 3-bet pot;
- an all-in facing: `facing_allin` True, `pot_odds` exact;
- `opp_pf_action` for each of the five classes on scripted lines;
- ranges: a button open's weights cover more combos than a UTG open's; a 3-bet range is
  polar (mass in two disjoint percentile bands); a call range excludes AA;
- `hud_stats` on a scripted 20-hand record set with known counts, every stat exact,
  including a check-raise, a button steal and an overbet in the script.

**Acceptance:** tests green; `situation` is a pure function of the record prefix (calling it on a
record truncated after `snap_idx` gives the same dataclass — a test).

**Non-goals:** anything postflop about ranges; per-opponent history across hands.

### Outcome — built 2026-09-03

`pool/situation.py`, `pool/stats.py`, `tests/test_situation.py` and
`tests/test_hud_stats.py` are on disk; ⚠3 and ⚠6 signed off by the owner on continuing the
plan. 23 new tests, all scripted lines rather than sampled play, so every asserted number is
arithmetic. Whole battery **546 tests, 262 s** (budget 30 min).

**One engine fact cost a debugging pass, and it is the kind that would have shipped silently.**
A decision that closes a street is *stepped before its snapshot is taken*, and the engine zeroes
the per-street `bets` on a street change and pays the pot into `credits` at the end of a hand.
So for exactly the decisions that end something — the call that closes the preflop, the last
action of the hand — reading "what did this action put in" off `bets` or off `credits` gives
nonsense: the big blind's call of a raise read as putting in −10 chips, and therefore as
`UNOPENED` rather than `CALL`, and therefore as the whole range instead of a calling range.
Everything is now read off the **pot**, which only ever grows by what an action adds, on every
street and on the last action alike.

**Five interpretations the plan left open, decided here** and pinned by tests:

1. **`facing` is the bet as a fraction of the pot it was bet *into***, not of the pot the
   snapshot carries — those differ by the bet itself. Only the first makes `mdf = 1/(1+facing)`
   the minimum defence frequency §P3 thresholds with, and only the first makes a half-pot bet
   read as 0.5, which is what §P3's own pinned numbers (`0.5·defend_factor` against a pot-sized
   bet) require.
2. **`Situation` carries `hero` and `live`** beyond §P2's field list, because
   `opponent_ranges(sit, …)` cannot name an opponent without them. `opp_pf_action` covers every
   seat, hero included, because §P3 builds hero's *own* range from hero's own preflop line.
3. **A seat that has not acted yet reads `UNOPENED`**, the same as the big blind checking its
   option: in both cases nothing has been learned, and both get the whole range.
4. **`n_behind` is positional** — live, non-all-in seats after hero in the street's acting order.
   It is not "who still owes chips", which is engine state the snapshot does not carry; the
   positional count is what the push/fold correction and the steal rules mean anyway.
   **`hero_barrels` counts postflop streets only**, so a flop c-bet after a preflop raise is
   barrel zero, which is what §P3's `hero_barrels = 0` flop branch needs.
5. **A steal needs an *unentered* pot** — a limper ahead of the button ends the opportunity
   rather than creating one — and the cutoff only exists from four seats up: at three the seat
   below the button is the big blind, and heads-up the small blind *is* the button.

**Two things P3 has to deal with, found here.**

- **The raise grid is pot fractions, and `open_size_bb` is an absolute size.** On the committed
  preflop grid the smallest bin is a pot-sized raise, which from an unopened 6-max pot is a
  raise *to* 2.5 BB and heads-up a raise to 2.0 BB. So `open_size_bb = 2.0` is simply
  unreachable at a full table, and every preflop size knob has to be converted to a pot fraction
  against the live pot before `nearest_bin` sees it — which also means the size a member
  actually uses drifts with the number of limpers. §P3 says "the legal raise bin nearest to
  `target · effective pot`" for postflop and names `open_size_bb` for preflop; the preflop half
  needs the conversion written down before it is implemented.
- **The committed config now trains at `players_range [2, 2]` and `stack_bb_range [200, 200]`**
  — heads-up, 200 BB. That is the owner's decision and nothing here touches it, but it changes
  what §P4's gate is *for*: the 6-max and 9-max stat cells stop being the primary evidence and
  become insurance for a distribution the config is not currently sampling.

**What is untested:** CPU only, nothing on the Spark. Nothing here has been run over a corpus,
so the per-decision cost of `situation` — one Python pass over the decisions of a hand, per
decision — is unmeasured; at a 40-decision hand that is quadratic in the hand's length, and §P3
calls it once per group rather than once per combo, which keeps it off the 1 225-row path but
not off the label path.

---

## P3 — The rule cascade

**Depends on:** P1, P2.
**Reads first:** `pool/base.py`, `pool/style.py`, `pool/degenerate.py`, `pool/action_map.py`
(`nearest_bin`), `env/legal.py` (what a raise bin means: a fraction of the *effective pot*,
per street, and which bins are playable), `oracle/posterior.py` (how `hole_override` batches
arrive).

**Deliverables:** `pool/regular.py`, `tests/test_regular.py`.

```python
@dataclass
class RegularParams:
    # preflop — fractions of combos, in the (n_players − 1)-opponent ordering
    open_early: float; open_late: float; limp_share: float
    call_open: float; threebet_value: float; threebet_bluff: float
    call_threebet: float; fourbet: float
    open_size_bb: float; threebet_mult: float
    push_fold_bb: float           # effective stack at or below which preflop is push/fold
    # postflop — frequencies in [0, 1] unless noted
    value_hs: float               # hs quantile above which a hand bets/raises for value
    cbet_dry: float; cbet_wet: float; oop_factor: float
    size_dry: float; size_wet: float; size_river: float     # pot fractions
    barrel_turn: float; barrel_river: float                 # share of bluff candidates that continue
    bluff_ratio: float            # multiplier on the balanced α = s/(1+s) for polar river bets
    semi_bluff: float             # bet/raise frequency with a strong draw
    defend_factor: float          # multiplier on MDF = 1/(1+s)
    raise_value: float; raise_bluff: float
    slowplay: float; donk: float; overbet: float
    allin_spr: float              # shove for value when spr ≤ this
    multiway_tighten: float       # per extra live opponent, subtracted from continue frequencies

class RegularMember(PoolMember):
    def __init__(self, name, n_actions, params, cache, preflop_table, raise_sizes, style=None)
    def logits(self, contexts) -> np.ndarray     # (B, n_actions)
```

**Design.**

*Batching.* Group `contexts` by `(id(record), snap_idx)`. Per group: one `situation`, one
`BoardStrength` (postflop), one `opponent_ranges`; then the rows of the group are the combos
(`combo_index` of each context's `hole_cards`, override included) and the cascade below runs as
numpy over those rows. A posterior batch (1 225 rows, one group) and a driver batch (many
groups of one row) are the same code.

*Output.* The cascade produces a distribution over five **intents** — `FOLD, CHECK_CALL, BET_SMALL, BET_BIG, ALL_IN` — per row, then maps intents to the action grid: `FOLD → 0`,
`CHECK_CALL → 1`, `ALL_IN → n_actions − 1`, `BET_SMALL/BIG →` the legal raise bin nearest to
`target · effective pot` (`nearest_bin`; `size_*` and the preflop sizes give the targets); when
no raise bin is legal the intent's mass goes to `ALL_IN` if legal, else to `CHECK_CALL`. Logits
are `log(p + 1e-9)`; `StyleParams.apply` then does temperature, bias and legality exactly as
for every other member. An intent that is illegal is redistributed proportionally over the
legal ones *before* the log, so `fold` never receives mass when checking is free.

*Preflop cascade.* `pct = preflop_rank_pct(table, n_players − 1)` for the rows; `open = open_early + (open_late − open_early) · pos_frac`.

- `eff_stack_bb ≤ push_fold_bb`: **push/fold.** Push if `pct ≤ push_range(eff_stack_bb, n_behind)`, call a shove if `pct ≤ call_range(eff_stack_bb)`, else fold. `push_range` is
  linear in the stack between pinned points (§Tests) and divided by `(1 + 0.5·n_behind)`
  for the seats still to act — the crude multiway correction, made explicit.
- Unopened pot (`n_raises_pre = 0`, no limpers): enter with `pct ≤ open`; of the entering
  mass, `limp_share` limps (`CHECK_CALL` at the blind level) and the rest opens `BET_SMALL`
  at `open_size_bb` (+1 BB per limper when there are limpers, and the enter threshold widens by
  `0.1` per limper — the regular attacks limpers).
- Facing one raise: `pct ≤ threebet_value` → 3-bet (`BET_SMALL` at `threebet_mult` × the
  raise, or `ALL_IN` if that is > ⅓ of the effective stack); `call_open` band below it → call;
  a `threebet_bluff` share of the band just below the call range → 3-bet (polar); else fold.
- Facing a re-raise: `pct ≤ fourbet` → raise (or shove per the same ⅓ rule); `pct ≤ call_threebet` → call; else fold.
- Multiway: every enter/continue threshold is multiplied by `(1 − multiway_tighten)^(n_in_pot − 1)`, where `n_in_pot` counts opponents already voluntarily in.
- BB with everyone limped, `facing = 0`: check with the `CHECK_CALL` intent; raise the top
  `open_late · 0.5` of combos.

*Postflop cascade.* Per row: `h = ehs(n_live − 1)` (P1), `q = hs^(n_live − 1)` (made strength,
no potential), `strong_draw = flush_draw | oesd`, `weak_draw = gutshot | backdoor_flush | overcards ≥ 1`, `bucket`:

| bucket     | condition                           |
| ---------- | ----------------------------------- |
| `NUTS`   | `q ≥ 0.97`                       |
| `STRONG` | `q ≥ value_hs`                   |
| `MEDIUM` | `q ≥ 0.55`                       |
| `WEAK`   | `q ≥ 0.35` (showdown value only) |
| `AIR`    | otherwise                           |

and `range_adv` = mean over the aggressor's range weights of `range_equity(w_aggr)` for the
board — one scalar per group saying whose preflop range this board favours (computed for hero
vs each live opponent's range, min over opponents).

1. **Hero is the last aggressor and it is checked to hero (or hero is first to act):**
   - flop, `hero_barrels = 0`: bet frequency `f = cbet_dry` (dry) / `cbet_wet` (wet) / their
     mean (mid), times `oop_factor` when `pos_frac < 0.5`, times `(1 − multiway_tighten)^ (n_live − 2)`, plus `0.15 · (range_adv − 0.5)` clipped to `[0, 1]`. `STRONG+` bets with
     probability `1 − slowplay`; `MEDIUM` bets with `f`; `strong_draw` bets with `semi_bluff`;
     `AIR/WEAK` bets with `f` scaled by `bluff_ratio`; size `size_dry` / `size_wet`.
   - turn, `hero_barrels ≥ 1`: value (`STRONG+`) bets `1 − slowplay`; `strong_draw` bets
     `semi_bluff · barrel_turn`; `AIR/WEAK` bets `barrel_turn · bluff_ratio · f_flop`; scare
     card bonus: `+0.2` to the bluff frequency when the turn card is above every flop card
     (the regular's "overcard barrel"). Size `size_wet` if any draw completed, else
     `size_dry`; an `overbet` share of the *value* bets uses `1.5` pot instead (added with the
     polar-reg archetype, §P4 — a turn overbet is where that style differs from a TAG).
   - river: **polar.** Value = `q ≥ value_hs`; bluff candidates = `AIR` rows that had a draw
     on the turn (missed draws) — flagged by re-reading the turn table's `strong_draw | gutshot` for the same combo. Bluff frequency is set so that `bluffs / (bluffs + value) = bluff_ratio · s/(1+s)` with `s = size_river` *over the aggressor's own range on this
     board* (the P2 range for hero's position and preflop action), computed per group as one
     ratio of range masses; per-row it is a Bernoulli with that frequency, further scaled by
     `barrel_river`. `overbet` share of value bets uses `2.0` pot instead of `size_river`.
     `spr ≤ allin_spr` with value → `ALL_IN`.
2. **Facing a bet of size `s`** (`facing > 0`): `mdf = defend_factor / (1 + s)`, clipped to
   `[0, 1]`. Continue with the top `mdf` of hero's *own range mass* on this board (range from
   P2 for hero's position/action; per row: continue iff the row's `range_equity` percentile
   within that range ≥ `1 − mdf`, else fold). Of the continuing rows: `NUTS/STRONG` raise with
   `raise_value` (size `size_wet`, or `ALL_IN` when `spr ≤ allin_spr`); `strong_draw` raises
   with `raise_bluff · semi_bluff` when `pot_odds < p_improve` fails for a plain call — i.e.
   raise the draws that cannot call profitably; `AIR` raises with `raise_bluff · bluff_ratio · 0.3`; everything else calls. A hand with `h ≥ pot_odds` never folds (the price is right),
   whatever `mdf` says — the human floor.
   `facing_allin`: call iff `h ≥ pot_odds · (1 + 0.1·(1 − defend_factor))`.
3. **Hero is not the aggressor, checked to hero, hero has position or it was checked around:**
   `donk` (OOP into the aggressor) or a stab: `STRONG+` bets `1 − slowplay`; `MEDIUM` bets
   `0.5 · f`; draws bet `semi_bluff · 0.6`; `AIR` bets `f · bluff_ratio · 0.4` (the "float"
   / delayed c-bet). Sizes as in 1.
4. **Multiway:** every bet/raise frequency in 1–3 is multiplied by `(1 − multiway_tighten)^ (n_live − 2)` and `value_hs` is raised by `0.05 · (n_live − 2)`, capped at `0.95`.

All frequencies are clipped to `[0, 1]` after every modifier; every intent vector is
renormalised; every rule is a numpy expression over the group's rows. Where the text says "with
probability p" it means the intent vector carries `p`, not that anything is sampled here — the
driver samples.

⚠ **Sign-off:** ⚠4. Also the following interpretations, so the implementing session does not
re-decide them: quantile buckets (not hand-class buckets) drive the numbers, hand classes are
descriptive only; the river bluff ratio is measured over the P2 range, not over the 1 326
combos; the human floor rule in 2 exists.

**Tests** (`tests/test_regular.py`) — scenario tests through `policy()`, never through internals:

- **vectorisation:** the policy over 1 225 `hole_override` contexts of one decision equals the
  concatenation of 1 225 single-row calls, exactly;
- **legality:** for 2 000 random decisions across 2–9 players and 10–300 BB (seeded
  `LockstepDriver` games), no mass on an illegal action and every row sums to 1;
- **preflop:** with the TAG preset (P4's numbers, copied into the test): 72o folds everywhere;
  AA always raises or shoves; the button opens strictly more combos than the first seat 9-max
  (measured over all 1 326 combos); HU the SB enters ≥ 70 % of combos; a 3-bet range is
  polar; facing a 4-bet with 200 BB, JJ calls and 72o folds;
- **push/fold:** at 10 BB HU the SB's push mass and the BB's call mass match the pinned Nash
  points within ±5 %; at 5 BB the push mass is larger; at 30 BB no push/fold (raise sizes are
  `open_size_bb`). The pinned points are fetched from the HoldemResources HU Nash chart during
  the implementing session and written into the test with the URL and date;
- **c-bet:** as the aggressor HU on K72 rainbow vs 987 two-tone, the total bet mass over the
  aggressor's range is within ±0.03 of `cbet_dry` / `cbet_wet` (TAG preset, `bluff_ratio` 1,
  `slowplay` 0);
- **river balance:** on a blank river after two barrels, the bluff share of the bet mass over
  hero's range is `size_river/(1+size_river)` ± 0.03 with `bluff_ratio = 1`, `≈ 0` with
  `bluff_ratio = 0`, and roughly double with `bluff_ratio = 2`;
- **defence:** facing a pot-sized bet, hero's continuing range mass is `0.5 · defend_factor`
  ± 0.03; facing a third-pot bet it is `0.75 · defend_factor` ± 0.03; a hand with `h ≥ pot_odds` never folds;
- **draws:** a flush draw on the flop facing a small bet calls; the same draw facing a shove
  it cannot afford folds; a strong draw semi-bluffs at the configured rate;
- **multiway:** 6-way, the c-bet mass on K72r is strictly below the HU number; `value_hs`
  rises with players;
- **short stack / SPR:** with `spr ≤ allin_spr` and `NUTS`, the intent is all-in; with 12 BB
  effective the preflop path is push/fold and a raise bin is never used;
- **determinism:** two members built with equal params give identical policies on 500
  decisions;
- **style still applies:** a `StyleParams` with a large fold bias moves mass toward fold.

**Acceptance:** all green; `policy` over a 1 225-row posterior batch on a cached board runs in
< 20 ms on the dev box (asserted at 100 ms).

**Non-goals:** opponent modelling inside the member; reading the *actual* opponent range from
the posterior; any exploitation of table history across hands; ICM.

### Outcome — built 2026-09-03

`pool/regular.py` and `tests/test_regular.py` are on disk; ⚠4 signed off by the owner on
continuing the plan. 22 tests; whole battery at the time **569 tests, 263 s** (budget 30 min). Measured on the dev box: a **1 326-row posterior query on a
cached board costs 4.2 ms** (§P3's acceptance was 20 ms), a self-play decision 1–6 ms, and the
member plays legal chip-conserving hands at every table size 2–9 and every depth 10–300 BB.

**One change to §P1 was needed to get there.** Equity against a weighted range, written as the
masked reduction it is, costs 21 ms, and the cascade needs several per decision. It is now
computed by inclusion–exclusion on the two cards a combo holds — the combos sharing a card with
`h` are those holding `c0`, plus those holding `c1`, minus `h` itself — which is the same sums
in a different order, exact, and **0.43 ms**, a fifty-fold saving. A test pins it against the
masked reduction to 1e-12.

**Three corrections to §P3's own numbers**, each measured rather than argued:

1. **The human floor was a ceiling.** §P3 says a hand with `ehs ≥ pot_odds` never folds. Measured
   against a pot-sized bet: that rule makes **every** archetype continue with 93 % of its range
   — a nit at `defend_factor` 0.55 and a station at 1.35 alike — because most of a preflop range
   beats a *random* hand a third of the time. `defend_factor` stops meaning anything, and it is
   one of the ten style axes. The floor is now on the **draw's own odds** (`p_improve ≥
   pot_odds`), which is what a player says out loud when they invoke it, binds only on real
   draws, and leaves the defence frequency at exactly `defend_factor/(1 + bet/pot)`: measured
   0.275 / 0.503 / 0.677 for the three factors against a pot-sized bet.
2. **Heads-up, the small blind opens the button's range, and a heads-up button is not a six-max
   button.** The positional interpolation gives the heads-up small blind `open_late` — 45 % for
   a TAG — while the same table credits an *opponent* in that seat with 75 %. Every regular
   would open 45 % heads-up and assign its opponent 75 % in the same seat. The button number is
   now scaled to the heads-up norm, and the TAG opens 74.7 %, which is also what §P3's own test
   ("HU the SB enters ≥ 70 %") asks for. This matters more than it looks: the committed config
   trains heads-up.
3. **`threebet_bluff` is a band width, not a share of a band.** Read as a width, the total 3-bet
   frequency is `threebet_value + threebet_bluff`, the range is polar by construction, and the
   units match every other preflop knob. Read as a share, the band it is a share *of* is
   undefined. Pinned by a test: the value band all raises, the calling band none of it, the
   band just below it all raises again, and nothing below that.

**Four interpretations §P3 left open**, also pinned: the multiway factor is applied **once**
(§P3 states it both in the flop rule and in the general multiway rule, and squaring it would
have been silent); "the bettor's range" for the defence equity is the **pooled** live-opponent
range, since nothing carries who raised on the current street and heads-up the pooled range *is*
the single opponent's; `donk` scales the branch where the aggressor still has to act and leaves
a stab at a checked-through pot alone; and a draw is "completed" on the turn when the share of
holdings making a straight or better **rose** from the flop board to the turn board — measured,
so a flush card and a straight card both count and a blank does not.

**Two things §P3 asked for that cannot be true, and are therefore not asserted.**

- **"The c-bet mass is `cbet_dry` ± 0.03."** Value hands bet at `1 − slowplay` whatever the
  c-bet frequency is, so the total mass is always above the knob — measured 0.78 for a knob of
  0.75, and 0.42 for a knob of 0.30. What *is* an identity is the bluffs: the air bets at
  exactly `f · bluff_ratio`, and that is what the test pins, along with dry above wet and the
  mass moving with the knob.
- **"The push/fold mass matches the pinned Nash points within ±5 %."** It does — measured 0.541
  against 0.550 at nine big blinds effective and 0.716 against 0.719 at five — but the points
  *are* the member's own table, so the test is checking interpolation and threshold plumbing,
  not agreement with equilibrium. Said plainly rather than dressed up. The points are the
  heads-up Nash aggregates from https://pailiku.com/push-fold, read 2026-09-03; its 10 BB
  figures (jam 52.9 %, call 34.2 %) agree with the other published tables, and HoldemResources'
  own chart gives per-hand stack thresholds rather than aggregates, so it cannot be used
  directly.

**A known leak, left in deliberately.** Facing an all-in, the price is read through the same
outs-times-four potential term, which is a *flop* heuristic being applied to a spot where both
cards are guaranteed to come. A bare flush draw therefore calls a shove a real player folds.
That is §P1's acknowledged crudeness showing up where it is worst — and calling too wide with a
draw is a recognisable human leak, which is what the pool is for.

**What is untested:** CPU only, nothing on the Spark. Nothing has run at corpus scale, so the
label-path cost of ten regulars in the pool is unmeasured — that is §P6's gate. The archetype
presets do not exist yet; every number above is the single default parameter set.

---

## P4 — Archetypes, jitter, and the self-play realism gate

**Depends on:** P3.
**Reads first:** `pool/build.py`, `pool/sampling.py`, `gates/` (any gate, for the report
shape), `config.json` `bootstrap`.

**Deliverables:** `pool/archetypes.py`, `gates/pool_realism.py`, `tests/test_archetypes.py`,
`tests/test_pool_realism.py`.

```python
ARCHETYPES: dict[str, RegularParams]                  # the ten presets
JITTER: dict[str, float]                              # per-knob relative spread
def draw_params(archetype, rng, spread=1.0) -> RegularParams
BANDS: dict[str, dict[str, dict[str, tuple]]]         # archetype → table size → stat → (lo, hi)
def run_realism_gate(config, out_dir, log) -> dict
```

**Design.**

*Presets* — proposed values; every number here is a design hypothesis written by Claude, not a
measurement, and the owner should read the table as "the shape I mean" before the numbers are
pinned:

| knob                             | `nit`         | `loose_passive` | `loose_passive_bluffy` | `maniac`       | `tag`            | `bluffer`       | `weak_tight`   | `trapper`       | `polar_reg`      | `stealer`        |
| -------------------------------- | --------------- | ----------------- | ------------------------ | ---------------- | ------------------ | ----------------- | ---------------- | ----------------- | ------------------ | ------------------ |
| open_early / open_late           | 0.06 / 0.16     | 0.35 / 0.60       | 0.35 / 0.60              | 0.60 / 0.90      | 0.14 / 0.45        | 0.18 / 0.55       | 0.20 / 0.40      | 0.14 / 0.45       | 0.16 / 0.50        | 0.07 / 0.70        |
| limp_share                       | 0.3             | 0.8               | 0.8                      | 0.0              | 0.0                | 0.0               | 0.3              | 0.1               | 0.0                | 0.0                |
| call_open                        | 0.08            | 0.45              | 0.45                     | 0.10             | 0.15               | 0.15              | 0.30             | 0.18              | 0.12               | 0.10               |
| threebet_value / bluff           | 0.03 / 0        | 0.03 / 0          | 0.03 / 0.02              | 0.25 / 0.30      | 0.06 / 0.04        | 0.06 / 0.10       | 0.04 / 0         | 0.05 / 0.02       | 0.07 / 0.08        | 0.05 / 0.06        |
| call_threebet / fourbet          | 0.03 / 0.015    | 0.20 / 0.02       | 0.20 / 0.02              | 0.30 / 0.20      | 0.08 / 0.025       | 0.10 / 0.04       | 0.04 / 0.02      | 0.10 / 0.02       | 0.09 / 0.035       | 0.04 / 0.02        |
| open_size_bb / threebet_mult     | 3.0 / 3.5       | 2.0 / 3.0         | 2.0 / 3.0                | 4.0 / 4.0        | 2.5 / 3.2          | 2.5 / 3.2         | 2.5 / 3.0        | 2.5 / 3.2         | 2.3 / 3.5          | 2.2 / 3.0          |
| push_fold_bb                     | 8               | 6                 | 6                        | 25               | 12                 | 12                | 10               | 12                | 12                 | 12                 |
| value_hs                         | 0.88            | 0.65              | 0.65                     | 0.50             | 0.75               | 0.72              | 0.80             | 0.75              | 0.74               | 0.75               |
| cbet_dry / cbet_wet              | 0.35 / 0.30     | 0.30 / 0.25       | 0.35 / 0.30              | 0.95 / 0.95      | 0.75 / 0.55        | 0.85 / 0.70       | 0.55 / 0.40      | 0.35 / 0.30       | 0.70 / 0.45        | 0.75 / 0.55        |
| oop_factor                       | 0.7             | 0.9               | 0.9                      | 1.0              | 0.75               | 0.85              | 0.7              | 0.9               | 0.7                | 0.6                |
| size_dry / size_wet / size_river | 0.5 / 0.6 / 0.5 | 0.33 / 0.5 / 0.5  | 0.33 / 0.5 / 0.5         | 1.0 / 1.25 / 1.5 | 0.33 / 0.67 / 0.75 | 0.4 / 0.75 / 1.0  | 0.5 / 0.6 / 0.6  | 0.5 / 0.75 / 0.75 | 0.33 / 0.75 / 1.25 | 0.33 / 0.67 / 0.75 |
| barrel_turn / barrel_river       | 0.1 / 0.0       | 0.1 / 0.05        | 0.3 / 0.4                | 0.9 / 0.9        | 0.55 / 0.45        | 0.75 / 0.70       | 0.2 / 0.1        | 0.4 / 0.35        | 0.6 / 0.5          | 0.5 / 0.35         |
| bluff_ratio                      | 0.1             | 0.1               | 0.9                      | 2.5              | 1.0                | 1.8               | 0.1              | 0.7               | 1.0                | 1.0                |
| semi_bluff                       | 0.15            | 0.10              | 0.25                     | 0.9              | 0.55               | 0.70              | 0.20             | 0.40              | 0.60               | 0.50               |
| defend_factor                    | 0.55            | 1.35              | 1.35                     | 1.3              | 1.0                | 1.05              | 0.60             | 1.0               | 1.0                | 0.85               |
| raise_value / raise_bluff        | 0.9 / 0.0       | 0.3 / 0.0         | 0.3 / 0.15               | 0.9 / 0.6        | 0.6 / 0.2          | 0.55 / 0.4        | 0.8 / 0.0        | 0.9 / 0.3         | 0.5 / 0.3          | 0.6 / 0.15         |
| slowplay / donk / overbet        | 0.3 / 0.0 / 0.0 | 0.2 / 0.3 / 0.0   | 0.2 / 0.3 / 0.0          | 0.0 / 0.5 / 0.5  | 0.1 / 0.05 / 0.1   | 0.05 / 0.1 / 0.25 | 0.05 / 0.1 / 0.0 | 0.6 / 0.05 / 0.1  | 0.1 / 0.05 / 0.6   | 0.1 / 0.0 / 0.1    |
| allin_spr                        | 1.0             | 1.5               | 1.5                      | 4.0              | 1.5                | 2.0               | 1.5              | 1.5               | 2.0                | 1.5                |
| multiway_tighten                 | 0.15            | 0.05              | 0.05                     | 0.0              | 0.12               | 0.10              | 0.12             | 0.12              | 0.12               | 0.12               |

*What the four added archetypes contribute* (owner decision 2026-09-02). The first six lie in
one plane — tightness × aggression, plus a bluff share — and share three regularities: a bet
correlates with strength the same way in all of them, sizes stay in the half-to-pot band, and
position bends every range by the same shape. Each addition breaks one of those:

| archetype      | what it is                                                                                                                                                                                                                | the regularity it breaks                                                                                                                                                                                                      | the stat that isolates it                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `weak_tight` | loose preflop, fit-or-fold after: folds to a c-bet without a pair, folds to 3-bets, raises only two pair and better, never bluffs. The most common mid-stakes population type and the largest exploit surface in the pool | the*loose_passive* answer to a bet is to call; this one folds — same preflop point, opposite postflop reaction                                                                                                             | `fold_to_cbet` high, `wtsd` low       |
| `trapper`    | TAG ranges, low c-bet, high slowplay, check-raises strong hands and some draws                                                                                                                                            | "a check means weakness" — true in the other nine. A balanced opponent (Slumbot) check-raises; an agent that learned the shortcut pays for it there. This is the one addition aimed at the benchmark rather than at the pool | `check_raise` high, `cbet_flop` low   |
| `polar_reg`  | TAG frequencies, modern sizes: third-pot c-bets, 1.5-pot turn and 1.25-pot river bets, more 3-bets                                                                                                                        | the size axis — none of the other nine overbets, and Slumbot's grammar includes pot-plus and all-in sizes the agent would otherwise first meet at the measurement                                                            | `overbet_pct` high                      |
| `stealer`    | extreme position dependence: ~7 % from the first seat, ~70 % from the button, high fold to 3-bet; HU it is an aggressive SB stealer                                                                                       | the shape of the position curve, identical across the others                                                                                                                                                                  | `steal` high, `fold_to_threebet` high |

*Jitter.* `draw_params` multiplies every rate knob by `exp(N(0, JITTER[knob]·spread))`, clips to
its domain, and leaves `push_fold_bb`, sizes and `open_size_bb` on an additive jitter. Default
`JITTER` is 0.15 for frequencies, 0.10 for sizes. This is the diversity source for regulars;
the bootstrap entry `{"kind": "regular", "archetype": "tag", "n_variants": 4}` draws four
parameter sets from one rng stream, and `style` defaults to `"identity"` for regulars — a logit
bias on top of a cascade that already encodes style would double-count it, and the config may
still ask for one explicitly.

*Bands* — the pre-registered stat profile per archetype, per table size (`hu`, `6max`, `9max`),
measured by `hud_stats` in self-play. Proposed (again: hypotheses to confirm before the runs):

| archetype            | table | vpip       | pfr        | cbet_flop  | af       | wtsd       |
| -------------------- | ----- | ---------- | ---------- | ---------- | -------- | ---------- |
| nit                  | 6max  | 0.08–0.14 | 0.06–0.11 | 0.25–0.45 | 1.0–2.5 | 0.15–0.25 |
| loose_passive        | 6max  | 0.40–0.55 | 0.04–0.10 | 0.20–0.40 | 0.3–0.8 | 0.35–0.50 |
| loose_passive_bluffy | 6max  | 0.40–0.55 | 0.04–0.12 | 0.25–0.45 | 0.6–1.2 | 0.30–0.45 |
| maniac               | 6max  | 0.60–0.80 | 0.50–0.75 | 0.85–1.0  | 4–10    | 0.25–0.45 |
| tag                  | 6max  | 0.19–0.26 | 0.15–0.22 | 0.55–0.75 | 2.0–3.5 | 0.24–0.32 |
| bluffer              | 6max  | 0.23–0.32 | 0.18–0.27 | 0.70–0.90 | 2.5–4.5 | 0.22–0.30 |
| weak_tight           | 6max  | 0.26–0.36 | 0.14–0.22 | 0.40–0.60 | 1.2–2.2 | 0.18–0.26 |
| trapper              | 6max  | 0.19–0.27 | 0.14–0.21 | 0.25–0.45 | 1.5–2.8 | 0.26–0.34 |
| polar_reg            | 6max  | 0.20–0.28 | 0.16–0.24 | 0.55–0.75 | 2.2–3.8 | 0.22–0.30 |
| stealer              | 6max  | 0.19–0.28 | 0.16–0.25 | 0.60–0.80 | 2.0–3.5 | 0.20–0.28 |

Four archetype-specific bands on top of the five shared stats, 6-max: `fold_to_cbet (weak_tight) ∈ 0.60–0.75`; `check_raise(trapper) ∈ 0.12–0.25`; `overbet_pct(polar_reg) ∈ 0.25–0.45`; `steal(stealer) ∈ 0.55–0.75` and `fold_to_threebet(stealer) ∈ 0.65–0.85`.

HU bands are the same rows with `vpip`/`pfr` measured from the SB only and shifted up by the
archetype's `open_late` (the implementing session derives them from the presets and writes them
into `BANDS` before running the gate). 9-max bands are 6-max bands scaled by
`(1 − multiway_tighten)^3` on `vpip`/`pfr`.

*The gate* (⚠9). For each archetype and table size: seat one variant at a table with a fixed
mixed field (other archetypes drawn round-robin from the remaining nine, then v7 members to fill), play `hands` hands with
`LockstepDriver` under one `utils.progress` bar over `archetypes × sizes × hands`, compute
`hud_stats` for the archetype's seat, compare with `BANDS`, and write
`/data/v8/pool_realism/<run>/report.json` with every stat, its band, pass/fail, and the seed.
Stack depths are sampled uniformly from the configured range so the profile is over the whole
distribution, and the report also splits every stat by a `short (≤ 25 BB)` / `deep` bucket, so a
push/fold regime that is silently wrong shows up. Default `hands = 20 000` per cell (CPU, no
network; ~minutes per cell on the dev box is the expectation, to be measured).

⚠ **Sign-off:** ⚠5, ⚠9, and the preset and band numbers above.

**Tests:**

- every preset round-trips through `draw_params` with `spread = 0` unchanged, and with
  `spread = 1` every drawn knob stays in its domain (500 draws);
- the gate at toy scale (all ten archetypes, `hu` and `6max`, 1 500 hands, fixed seed) produces a
  report with every stat finite and the pass/fail field present, the bar reaching its total;
- **ordering, not values** (deterministic at 1 500 hands): `vpip(nit) < vpip(tag) < vpip(bluffer) < vpip(loose_passive) < vpip(maniac)`; `af(loose_passive) < af(tag) < af(maniac)`; `pfr(loose_passive) < pfr(nit)`; `wtsd(loose_passive) > wtsd(nit)`;
  `fold_to_cbet(weak_tight) > fold_to_cbet(loose_passive)`; `check_raise(trapper) > check_raise(tag)`; `cbet_flop(trapper) < cbet_flop(tag)`; `overbet_pct(polar_reg) > overbet_pct(tag)`; `steal(stealer) > steal(tag)`; `fold_to_threebet(stealer) > fold_to_threebet(tag)`. These
  inequalities are the archetype definitions; the numeric bands are checked by the gate, not
  the battery.

**Acceptance:** tests green; the full gate has been run once on the dev box at `hands = 20 000`,
its wall clock reported, and every archetype passes its band **or** the report says which
stat missed and by how much — a miss at this stage is a preset to fix, with the change and the
reason recorded in this file's §P4 as an amendment.

**Non-goals:** more than ten archetypes; anything Slumbot.

### Outcome — built 2026-09-03

`pool/archetypes.py`, `gates/pool_realism.py`, `tests/test_archetypes.py` and
`tests/test_pool_realism.py` are on disk; ⚠5 and ⚠9 signed off by the owner on continuing the
plan. The preset table above is transcribed unchanged. Whole battery after §P4 and §P5 and the
early ⚠7: **605 tests, 317 s** (budget 30 min).

**The gate has been run**, 4 000 hands per archetype per table size, 24 minutes on the dev box.
Both questions come back yes. Every archetype produces a distinguishable stat line at every
table size — at six-handed, entry frequencies run 0.10 (nit) / 0.24 (TAG) / 0.26 (trapper,
polar reg) / 0.31 (weak-tight, stealer) / 0.32 (bluffer) / 0.45 (loose-passive) / 0.58 (maniac),
c-bets 0.29 (trapper) to 0.99 (maniac), overbets 0.23 (polar reg) against ≤ 0.04 for everyone
but the maniac, steal attempts 0.67 (stealer) against 0.46 (TAG) and 0.10 (nit). And every one
of the ten beats the five degenerate strategies by many standard errors: heads-up +47 to +279
BB/100, six-handed +357 to +1812, nine-handed +223 to +1985.

**The gate is not the gate §P4 described.** §0.3's decision removed the acceptance bands, so
`BANDS` does not exist and nothing passes or fails. What the gate reports is the two questions
and only those: ten stat lines side by side per table size, and each archetype's BB/100 against
the five degenerate strategies with its standard error.

**Reading ten stat lines costs what reading one costs.** §P4 seated one archetype per cell in a
field of the others — ten cells per table size. But one hand is a data point for *every seat at
it*, so all ten now sit at one table per size and rotate through the chairs: the same hands, the
same field, one run instead of ten, and the numbers become comparable rather than ten separate
experiments. Measured: 151 k hands for the whole job instead of 240 k, and the profile half
dropped from most of the wall clock to a fifth of it.

**Four mistakes worth recording, because all four would have shipped.** The first two are in
the stats, the second two only showed up once the gate was actually run — which is the argument
for running it.

- **`overbet_pct` was measuring how often a member raises.** A raise puts in the call *plus* its
  own size, so a three-quarter-pot raise exceeds the pot that was there before it and counted as
  an overbet — which made the trapper, whose raising frequency is the highest in the pool, the
  biggest overbetter, ahead of the archetype built to overbet. A bet is an overbet when it is
  bigger than the pot; a raise is one when the part *beyond the call* is bigger than the pot the
  call would leave.
- **`steal` was measuring "of the pots you entered, how many did you raise".** Its denominator
  required the seat to enter, so folding an opportunity was not an opportunity and every
  archetype that never limps scored 1.000. The denominator is now the seat's decision in a pot
  nobody has entered yet, which is what an attempt frequency means.

- **The mixed table's seating was positional, not archetypal.** Marching ten archetypes round
  the table by a fixed step looks like it visits every chair and does not: at two seats the step
  is two, so the even-indexed archetypes never left the button and the odd ones never left the
  big blind. The heads-up profile was a report on *seats*, and it read a maniac as tighter than
  a TAG because one was always defending and the other always opening. The seating is now a
  fresh permutation per hand.
- **The small blind was opening under-the-gun's range.** The opening threshold interpolated on
  *postflop* position, where the small blind is the earliest seat — but preflop it acts second
  to last, with one player left behind it. Every archetype had this at once, which is precisely
  why a report about how the archetypes *differ* cannot catch it: a leak they all share is
  invisible in a diversity table. Preflop the small blind now reads as a button, which is also
  what a regular credits an opponent in that seat with.

**The gate writes as it goes and resumes** (owner's request, 2026-09-03, after the third
re-run from scratch): the report is written after every cell, each table size prints as soon as
it finishes, and a re-launch with the same settings replays only the missing cells. The run
directory is named rather than timestamped, so a re-launch is a resume and not a second copy.

**Only the orderings resolvable at the sample size are asserted.** Each archetype gets ~1 350
hands in the battery's own run, so a stat with sixty opportunities behind it has a six-point
standard error. Pinned: the VPIP chain (nit / TAG / bluffer / loose-passive / maniac), the PFR
chain with the limp frequency separating passive from aggressive, the aggression ratios, the
four added archetypes each breaking their own regularity, and — the claim itself — **no two of
the ten sharing a stat line**. Not pinned, and reported instead: `check_raise` (trapper above
TAG by one standard error) and `wtsd` (loose-passive above nit by 1.4 of them). The stealer's
own signature — how often it attacks a pot nobody has entered — needs a **six-handed** table to
be measurable at all: at nine seats the cutoff sees an unopened pot about a dozen times in
fifteen hundred hands, so the battery plays a second, shorter six-max session for it.

**One thing the presets do not survive unchanged.** The archetype tests read the *run's own*
raise grid rather than the three-bin fixture the rest of the battery uses. On a grid whose only
sizes are a half, a whole and a double pot, a 1.25-pot river bet and a 0.75-pot one are the same
player, and the polar reg stops being distinguishable. If the grid changes, these are the
assertions to re-read.

---

## P5 — The Slumbot realism run

**Depends on:** P4; **the owner decision of §0.3.**
**Reads first:** `eval_pipeline.py` (whole file), `evaluation/v8_adapter.py::SlumbotAgent`,
`tests/test_slumbot_adapter.py`, `tests/test_eval_pipeline.py`.

**Deliverables:** edits to `evaluation/v8_adapter.py`, `eval_pipeline.py`; `tests/`
extensions; a report per archetype under `/data/v8/pool_eval/<archetype>/`.

**Design.**

*Hero as a pool member* (⚠8). `SlumbotAgent` takes a `member_factory(hero_seat) → PoolMember`
instead of `net`; the existing behaviour is the factory that builds `AgentPoolMember` with the
embeddings, and `set_embeddings` keeps working through it. A new config section

```json
"evaluation": { "hero": {"kind": "regular", "archetype": "tag", "variant_seed": 0}, ... }
```

selects the factory; absent `hero` means the agent, as today. With a `regular` hero the pipeline
forces `warm: false` (there is nothing to fit), stamps every report and log header
`POOL MEMBER REALISM — not an agent result`, writes under `/data/v8/pool_eval/` and never under
the agent's run directory, and adds the hero's `hud_stats` over the played hands (built from the
`slumbot_history` records the pipeline already keeps for the fit) to the report, so the HU stat
band of P4 is checked on the same hands as the BB/100.

*Protocol.* Per archetype, one run of **10 000 hands**, `n_workers = 8`, the preset with
`spread = 0` (the centre of the archetype, not a jittered variant). The report carries BB/100,
its standard error (≈ ±15 at this length), the stat profile, and the pre-registered band. Then:

| result                           | action                                                          |
| -------------------------------- | --------------------------------------------------------------- |
| BB/100 and stats inside the band | archetype accepted; its config is the preset                    |
| stats inside, BB/100 outside     | the band was wrong or the archetype is; report, owner decides   |
| stats outside                    | preset fixed with a written reason;**one** re-run allowed |

Pre-registered BB/100 bands (Claude's hypotheses; to be confirmed by the owner before any run —
they are in the file precisely so they cannot be written after the number is known):

| archetype            | BB/100 vs Slumbot |
| -------------------- | ----------------- |
| nit                  | −60 … −15      |
| loose_passive        | −120 … −40     |
| loose_passive_bluffy | −110 … −35     |
| maniac               | −250 … −80     |
| tag                  | −45 … −5       |
| bluffer              | −70 … −15      |
| weak_tight           | −80 … −30      |
| trapper              | −50 … −10      |
| polar_reg            | −40 … −5       |
| stealer              | −55 … −15      |

Wall clock is unknown: the pipeline has never played a hand against Slumbot (`ARCHITECTURE.md`
§6). The first run measures it; at a guessed ~1 s per hand per worker, 10 000 hands on 8 workers
is ~20 min, and ten archetypes ~3.5 h.

⚠ **Sign-off:** ⚠8; the bands; the one-re-run rule.

**Tests:**

- `SlumbotAgent` with a `regular` factory returns a legal distribution on the adapter's fixture
  hands, and its `act` maps to a valid wire token (extend `test_slumbot_adapter.py`);
- the pipeline with `hero.kind = regular` against the fake client used by
  `test_eval_pipeline.py`: `warm` forced off, the header stamped, the output directory under
  `pool_eval`, the report carrying `hud_stats` and the band with a pass/fail field, resume
  works;
- absent `hero`: byte-identical report to today on the same fixture.

**Acceptance:** ten reports on disk, each with BB/100 ± SE, the profile, the band and the
verdict; this file amended with the outcome table; nothing in `config.json`'s `bootstrap`
changed by this session.

**Non-goals:** any knob search; runs longer than 10 000 hands (that budget buys ±15, which is
all the ordering needs — a 1 M-hand run is for the agent, never for a pool member).

### Outcome — plumbing built and verified against the live wire, 2026-09-03

⚠8 is done, and §0.3 removed the bands, the one-re-run rule and the stamped-report machinery, so
what remains is a descriptive run. Changes: **who plays is now a factory**, not a network — the
adapter takes `(hero_seat, embeddings) -> PoolMember`, the agent arrives through its own factory
exactly as before, and a §P4 archetype arrives through the same door as one member serving every
seat. `evaluation.hero` selects it and its absence is the agent, byte for byte: a test asserts
the report with no hero section equals the report with the agent named explicitly.

Two consequences are made loud rather than left to fail late. A member that reads no opponent
vector says so, which makes a *warm* run against it an error rather than a silent no-op — the
runner drops the warm mode with a logged reason, and a warm-only run is refused outright. And a
pool member's run loads no agent checkpoint and is written under `pool_eval/<archetype>/`, never
beside the agent's.

**Verified against the real Slumbot**, from the dev box: 40 hands as the TAG archetype, two
workers, **zero failed hands and zero clamps** — every action the cascade chose was expressible
on the wire. Rate **0.77 hands/s per worker**, so eight workers is about six hands a second and
10 000 hands is roughly half an hour per archetype, five hours for all ten. The +120 ± 255
BB/100 from forty hands is not a number, and is not reported as one.

**The runs themselves are the owner's, on the Spark** (their decision, 2026-09-03: the dev box's
CPU is weak). `--hero-archetype` overrides the config's archetype so the ten runs are one config
and a loop; each writes its own directory and each resumes.

---

## P6 — Into the pool, and the label-cost gate

**Depends on:** P4 (P5 if the owner keeps it).
**Reads first:** `pool/build.py`, `pool/sampling.py`, `gates/` (the G3 gate), `config.json`,
`ARCHITECTURE.md` §2.5, §5.

**Deliverables:** `bootstrap` entries in `config.json`; a G3-style label-cost run with regulars
in the pool; `ARCHITECTURE.md` updated.

**Design.**

*Bootstrap.* Add one entry per accepted archetype, `n_variants` 3 each (30 regular members
beside the existing v7 ones), and **drop the five degenerate strategies from the default
config** — keep the classes, they are still the corners the tests use — because a nit and a
maniac that read the board supersede a nit and a maniac that do not. That is a config change,
not a code change, and `CONCEPT.md` §4.1's "every base is also in the pool unmodified" is
satisfied by the `spread = 0` preset being one of the variants.

*Cost gate.* Run the G3 label-cost gate (`gates/`, `test_g3_gate.py`'s machinery at real
settings on the Spark, or at the dev-box scale it already runs at) with the new pool, and
report forwards and wall clock per label against the numbers `ARCHITECTURE.md` §5 holds for
the v7-only pool. Regulars cost no forward; what they cost is `BoardStrength` builds, which
the profile buckets should show as CPU time in the pool's share. If a regular's share is above
a v7 member's, that is the signal §0.1's deferred runout decision was waiting for — in the
other direction.

*Embedding corpus.* Nothing to change: `label_ranges` and the posterior call `policy` on every
member the same way, and the range head's targets are computed from it. The realism report of
P4 is the evidence that the new members' *styles* are distinct enough for the embedding to have
something to separate; G1's gain metric on a corpus with regulars in it is the measurement,
and it is a run, not a session.

**Tests:** `test_pool_style.py` / `test_pool_sampling.py` extended so a `regular` entry builds,
draws `n_variants` distinct parameter sets, and `with_style` shares the cache by reference.

**Acceptance:** the whole battery green with its wall clock; the cost gate's numbers in
`ARCHITECTURE.md` §5; the §2 tree and §7 test table updated; the degenerate strategies' removal
from the default config recorded as an owner decision with the date.

**Non-goals:** retraining anything; changing the sampler.

### Partial outcome — ⚠7 built 2026-09-03, at the owner's request

The owner asked for a Spark test run with **the whole bootstrap pool replaced by the ten
archetypes and no augmentation at all**, so the two pieces of plumbing that needed are done
ahead of the rest of §P6. Nothing else here is built: no cost gate, no `ARCHITECTURE.md` §5
numbers, and the degenerate strategies are gone from *this* config rather than "dropped from the
default" as a recorded decision.

- **A `regular` bootstrap entry.** Its `n_variants` draws *parameters*, not styles — the cascade
  already is the style, and a logit bias on top would count it twice — so `style` defaults to
  no modifier at all, `spread` is what makes variants differ, and asking for several variants at
  spread zero is refused rather than silently building the same member repeatedly. Every regular
  in one build shares one board cache and one preflop table, which is what makes "one evaluator
  call per board" true across members instead of per member.
- **A worker mirror.** `oracle/parallel.py` rebuilds every pool member inside each labelling
  worker without its weights, and it raises on a kind it does not know — so a training run with
  regulars in the pool and more than one worker would have died on its first iteration. A
  regular has no network, so its mirror is its parameters, its raise grid and the 169 × 8
  preflop table (14 kB for two members). What deliberately does **not** travel is the board
  cache: a cache is per process, and the worker builds one and shares it across its own
  regulars.

**The config the owner will run**: ten entries, `style: "identity"`, `n_variants: 1`, spread 0 —
the ten presets exactly, nothing else in the bootstrap pool. The agent still contributes its own
style variants each iteration (`style.agent_variants`), which is the architecture and not an
augmentation of the procedural members.

---

## 3. What this plan does not claim

- That a cascade plays *well*. It plays *recognisably*: the archetypes lose to Slumbot by
  amounts that order them the way a human would order them. A pool of such members is what
  `CONCEPT.md` §4 asks for — a fixed, diverse population to best-respond to — not a benchmark.
- That the potential term is right. Outs × 4/2 is a table heuristic; it is systematically wrong
  for combo draws and for dominated draws, in the direction of overvaluing them, and it is
  applied to the members' *own* hands only. §0.1 says when to revisit.
- That the stat bands and the BB/100 bands are correct. They are written down first so that
  they can be found wrong.
- Anything about the Spark. Every wall-clock figure above is a dev-box measurement or a guess,
  and is marked as such.
