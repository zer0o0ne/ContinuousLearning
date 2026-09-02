"""Lock-step vectorised driver over `Table` (CONCEPT.md §3).

`Table` plays one hand at a time. Every consumer in v8 — the G1 corpus
generator now, the BR oracle later — issues policy queries in bulk, and a query
issued from a single hand is a batch of one, which is the worst possible shape
for the GB10. This driver advances N independent hands together: it collects the
pending decision of every live hand, groups those decisions by which pool member
has to answer them, makes **one** call per member, and scatters the sampled
actions back.

It is a driver *around* the engine. `Table.step`, its chip accounting and its
side-pot logic are untouched, so `test_engine_conservation.py` and
`test_audit_stage0.py` keep covering them. What this file owns is (a) the
lock-step scheduling and (b) the snapshot record each hand leaves behind.

**Determinism.** Batching must not change what happens in a hand — that is the
§15 equivalence requirement, and a silent divergence here would show up much
later as EV bias in the oracle. Two rules give it:

* the deck comes from ``np.random.seed(hand_seed)`` immediately before that
  hand's ``start_table()``, so it depends on the hand and on nothing else;
* actions are sampled by inverse-CDF from a uniform drawn from that hand's own
  ``np.random.Generator``, so the draw does not depend on how many other hands
  happened to be in the same batch.

`run(specs, batch_size=1)` is therefore the sequential path and
`run(specs, batch_size=N)` the lock-step one, and they must agree exactly.

**Snapshot convention** — inherited from v7 (`generation/generate.py`) so that a
vendored v7 checkpoint sees the event stream it was trained on:

* one initial snapshot with ``action=None``;
* per decision, a *pre-decision* snapshot (``active_pos`` = the player on turn,
  ``action=None``) followed by a *post-action* snapshot (``active_pos`` = whoever
  acts next — audit B.2's next-player convention — and ``action`` the one-hot of
  what was just played).

A hand that ends up with several players all-in is run out to the river with the
engine's own no-op steps, so final credits are correct; those steps carry no
decision and are not recorded.

**Rollout plumbing.** A `HandSpec` may additionally pin the deck and force a
prefix of the decisions (`HandSpec.deck`, `HandSpec.forced_actions`). Together
they express "this hand, these cards, this prefix, then free play", which is
what the BR oracle's rollouts are. A forced decision is recorded and stepped by
exactly the same code as a sampled one — only the choice of the action index
differs — and it costs no policy call, which is where the oracle's saving comes
from.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from env.table import Table
from env.legal import legal_action_mask
from env.runout import HandRunout, prime
from env.showdown import showdown_positions
from utils import progress

FOLD = 0
RIVER = 3


@dataclass
class HandSpec:
    """Everything needed to deal and play one hand."""
    num_players: int
    start_credits: list           # per-seat starting stacks, in chips
    seat_members: list            # pool-member index seated at each seat
    seed: int
    big_blind: float
    small_blind: float
    raise_sizes: list             # 4 lists (one per street) of raise fractions
    meta: dict = field(default_factory=dict)
    # Rollout plumbing (PLAN_PIPELINE.md S1). Both are None for an ordinary hand.
    deck: np.ndarray = None       # 52 ints; overrides the dealt deck
    forced_actions: list = None   # action indices, consumed in decision order


@dataclass
class HandRecord:
    """What one played hand leaves behind."""
    spec: HandSpec
    deck: np.ndarray
    snapshots: list               # v7-convention snapshots (see module docstring)
    decisions: list               # dicts: snap_idx, acting_pos, member, action_idx, legal_mask
    rewards: np.ndarray           # per-seat chip delta over the hand
    truncated: bool               # betting hit the max-actions cap
    # Same quantity with the card and fold luck subtracted out, when the driver
    # was given a `runout` config; `None` otherwise. Unbiased for the same
    # expectation, so a *mean* over hands may be read from it — a single hand's
    # entry is not a chip count and does not conserve chips (`env/runout.py`).
    baseline_rewards: np.ndarray = None
    showdown: list = field(default_factory=list)          # seats revealed, §5.1a
    showdown_strength: dict = field(default_factory=dict)  # seat → percentile
    showdown_class: dict = field(default_factory=dict)     # seat → 169-way class
    # §5.6 — every dealt seat's percentile on the final board, revealed or not.
    # `showdown_strength` is this dict restricted to the seats that showed.
    hand_strength: dict = field(default_factory=dict)      # seat → percentile

    @property
    def num_players(self):
        return self.spec.num_players

    def hole_cards(self, pos):
        return [int(c) for c in self.deck[5 + 2 * pos: 7 + 2 * pos]]


class DecisionContext:
    """One pending decision, as handed to a pool member.

    Thin view over `(record, snap_idx)` — it carries no state of its own, so
    building one per decision per hand costs nothing.
    """

    __slots__ = ("record", "snap_idx", "acting_pos", "legal_mask", "turn",
                 "hole_override")

    def __init__(self, record, snap_idx, acting_pos, legal_mask, turn,
                 hole_override=None):
        self.record = record
        self.snap_idx = snap_idx
        self.acting_pos = acting_pos
        self.legal_mask = legal_mask
        self.turn = turn
        # PLAN_PIPELINE.md S2: the posterior asks a member "what would you have
        # done holding *this*", which is the same situation with other cards.
        self.hole_override = hole_override

    @property
    def snapshot(self):
        return self.record.snapshots[self.snap_idx]

    @property
    def num_players(self):
        return self.record.spec.num_players

    @property
    def big_blind(self):
        return self.record.spec.big_blind

    @property
    def pot(self):
        return float(self.snapshot["pot"])

    @property
    def bets(self):
        return np.asarray(self.snapshot["bets"], dtype=np.float64)

    @property
    def credits(self):
        return np.asarray(self.snapshot["credits"], dtype=np.float64)

    @property
    def to_call(self):
        b = self.bets
        return float(max(0.0, b.max() - b[self.acting_pos]))

    @property
    def stack(self):
        return float(self.credits[self.acting_pos])

    @property
    def hole_cards(self):
        """The **acting player's** own hole cards — never anybody else's.

        `hole_override` replaces them with a hypothetical holding; the rest of
        the situation is untouched, so a member answers about the same decision
        under different cards (§7.2).
        """
        if self.hole_override is not None:
            return [int(c) for c in self.hole_override]
        return self.record.hole_cards(self.acting_pos)

    @property
    def board(self):
        """Board as visible on this decision's street; unseen cards are -1."""
        deck, turn = self.record.deck, self.turn
        if turn == 0:
            return [-1] * 5
        if turn == 1:
            return [int(c) for c in deck[:3]] + [-1, -1]
        if turn == 2:
            return [int(c) for c in deck[:4]] + [-1]
        return [int(c) for c in deck[:5]]


def max_actions_for(num_players):
    """v7's B.6.3 cap: high enough not to clip realistic multiway raise-wars."""
    return 6 * num_players + 8


def _suppressed(ctx):
    """True where this decision produces no control-variate correction.

    The same condition `_card_correction` applies, read one step earlier: a
    decision still inside a forced prefix replays a street hero had already
    seen, which is conditioned on rather than drawn, and it is played without a
    policy so there is no fold draw either. Nothing asks for a baseline, so
    nothing is worth ranking.
    """
    forced = ctx.record.spec.forced_actions
    return forced is not None and len(ctx.record.decisions) < len(forced) - 1


def _streets_needed(ctx):
    """The streets one decision can ask a baseline for before the next round.

    Its own street (the fold correction, and the near side of a card
    correction); the next one, which is where the card correction lands if the
    action turns a street over; and the river, where the corrections telescope
    to if the action leaves everybody all-in. Anything else is ranked on demand
    by `HandRunout.scores`, correctly and one call at a time.
    """
    return (ctx.turn, min(ctx.turn + 1, RIVER), RIVER)


class LockstepDriver:
    """Advances many hands together against a fixed pool.

    Args:
        pool: sequence of pool members; `HandSpec.seat_members` indexes it.
        n_actions: size of the discrete action set.
        runout: an `env.runout.RunoutConfig` to also fill `baseline_rewards`
            with the variance-reduced value of each hand, or None to leave that
            field empty. It changes nothing about how a hand is dealt or played
            — the same seeds produce the same hand either way.
    """

    def __init__(self, pool, n_actions, runout=None):
        self.pool = pool
        self.n_actions = n_actions
        self.runout = runout

    def run(self, specs, batch_size=None, desc=None):
        """Play every spec. Returns `HandRecord`s in spec order.

        `batch_size` caps how many hands are in flight at once; `1` is the
        sequential path. The result must not depend on it.

        `desc` labels the progress bar (`CLAUDE.md` §5). The unit is the
        completed hand — the internal loop is over lock-step rounds, which is
        not a quantity anybody wants an ETA in. Omitting `desc` runs silently,
        which is what the tests want.
        """
        if batch_size is None:
            batch_size = len(specs)
        batch_size = max(1, int(batch_size))

        records = [None] * len(specs)
        pending_specs = list(enumerate(specs))
        cursor = 0
        live = []
        # Ranking matrices, shared by every hand in flight that was dealt the
        # same deck (`env/runout.py`) — which, in a label, is every action of
        # one sample. Dropped as the hands that own them finish, so what it
        # holds is bounded by `batch_size` and not by the length of the run.
        scores = {} if self.runout is not None else None
        bar = progress(total=len(specs), desc=desc, unit="hand",
                       disable=desc is None)

        while cursor < len(pending_specs) or live:
            while len(live) < batch_size and cursor < len(pending_specs):
                idx, spec = pending_specs[cursor]
                cursor += 1
                live.append(self._start(idx, spec, scores))

            queries = []
            for state in live:
                ctx = self._advance_to_decision(state)
                if ctx is not None:
                    queries.append((state, ctx))

            finished = [s for s in live if s["done"]]
            for state in finished:
                records[state["idx"]] = self._finish(state)
            bar.update(len(finished))
            live = [s for s in live if not s["done"]]
            if scores is not None and finished:
                held = {s["runout"].key for s in live}
                for key in [k for k in scores if k[0] not in held]:
                    del scores[key]

            if not queries:
                continue

            # A decision whose index is still inside `forced_actions` is played
            # from the spec, not from a policy. Splitting here — before the
            # grouping — is what keeps forced decisions out of every batch.
            forced, free = [], []
            for state, ctx in queries:
                fa = ctx.record.spec.forced_actions
                k = len(ctx.record.decisions)
                (forced if fa is not None and k < len(fa) else free).append(
                    (state, ctx))
            if self.runout is not None:
                # The boards this round's decisions can ask about, ranked in one
                # batch rather than a few hundred rows at a time when a
                # correction asks (`env/runout.py`). A decision inside the
                # suppressed part of a forced prefix asks about none.
                prime([(state["runout"], turn)
                       for state, ctx in forced + free
                       if not _suppressed(ctx)
                       for turn in _streets_needed(ctx)])

            for state, ctx in forced:
                fa = ctx.record.spec.forced_actions
                self._apply(state, ctx, None,
                            ctx.record.spec.seat_members[ctx.acting_pos],
                            action_idx=fa[len(ctx.record.decisions)])

            if not free:
                continue

            by_member = {}
            for state, ctx in free:
                by_member.setdefault(ctx.record.spec.seat_members[ctx.acting_pos],
                                     []).append((state, ctx))

            for member_idx, group in by_member.items():
                member = self.pool[member_idx]
                probs = member.policy([ctx for _s, ctx in group])
                probs = np.asarray(probs, dtype=np.float64)
                assert probs.shape == (len(group), self.n_actions), (
                    f"pool member {member_idx} returned {probs.shape}, "
                    f"expected {(len(group), self.n_actions)}")
                for row, (state, ctx) in enumerate(group):
                    self._apply(state, ctx, probs[row], member_idx)

        bar.close()
        return records

    # ---------------------------------------------------------------- internals

    def _start(self, idx, spec, scores=None):
        table = Table(
            num_players=spec.num_players,
            raise_sizes=spec.raise_sizes,
            start_credits=list(spec.start_credits),
            big_blind=spec.big_blind,
            small_blind=spec.small_blind,
        )
        # The deck must depend on the hand's seed and on nothing else, so that
        # lock-step and sequential runs deal the same cards (§15).
        np.random.seed(spec.seed % (2 ** 32))
        table.start_table()
        if spec.deck is not None:
            deck = np.asarray(spec.deck, dtype=table.deck.dtype)
            assert sorted(deck.tolist()) == list(range(52)), (
                "deck must be a permutation of 0..51")
            table.deck = deck

        record = HandRecord(
            spec=spec,
            deck=np.copy(table.deck),
            snapshots=[_snapshot(table, active_pos=table.active_player, action=None)],
            decisions=[],
            rewards=np.zeros(spec.num_players, dtype=np.float64),
            truncated=False,
        )
        return {
            "idx": idx,
            "table": table,
            "record": record,
            "rng": np.random.default_rng(spec.seed),
            "max_actions": max_actions_for(spec.num_players),
            "done": False,
            "ended": False,
            # Variance reduction (`env/runout.py`). `corr` accumulates the luck
            # this hand happened to get and is subtracted at the end; `freeze`
            # holds the integrated value of a hand that ran out of decisions
            # before it ran out of streets.
            "runout": (HandRunout(record.deck, spec.num_players, self.runout,
                                  scores=scores)
                       if self.runout is not None else None),
            "corr": np.zeros(spec.num_players, dtype=np.float64),
            "run_out_from": None,
        }

    def _advance_to_decision(self, state):
        """Return the pending `DecisionContext`, or None if the hand is over."""
        if state["done"]:
            return None
        table = state["table"]
        record = state["record"]

        if table.several_all_in:
            state["done"] = True
            return None
        if len(record.decisions) >= state["max_actions"]:
            record.truncated = True
            state["done"] = True
            return None
        if table.players_state[table.active_player] != 1:
            # Mirrors v7's guard: nothing to decide, the hand is resolved.
            state["done"] = True
            return None

        acting_pos = int(table.active_player)
        record.snapshots.append(
            _snapshot(table, active_pos=acting_pos, action=None))
        snap_idx = len(record.snapshots) - 1
        mask = legal_action_mask(table, self.n_actions)
        return DecisionContext(record, snap_idx, acting_pos, mask, int(table.turn))

    def _apply(self, state, ctx, probs, member_idx, action_idx=None):
        """Record one decision and step the table.

        Two ways of choosing the action, one way of recording it: `action_idx`
        given plays it verbatim (a forced prefix, no policy call, no draw from
        the hand's generator), `action_idx=None` samples it from `probs`.
        """
        table = state["table"]
        record = state["record"]

        legal = ctx.legal_mask
        p = None
        if action_idx is None:
            p = np.where(legal, probs, 0.0)
            total = p.sum()
            assert total > 0, (
                f"pool member {member_idx} put zero mass on every legal action "
                f"(legal={np.flatnonzero(legal).tolist()})")
            p = p / total

            # Inverse-CDF from this hand's own generator: independent of batching.
            u = float(state["rng"].random())
            action_idx = int(np.searchsorted(np.cumsum(p), u, side="right"))
            action_idx = min(action_idx, self.n_actions - 1)
            assert legal[action_idx], "sampled an illegal action"
        else:
            assert probs is None, "a forced action takes no policy distribution"
            action_idx = int(action_idx)
            assert 0 <= action_idx < self.n_actions and legal[action_idx], (
                f"forced action {action_idx} is illegal for seat "
                f"{ctx.acting_pos} at decision {len(record.decisions)} "
                f"(legal={np.flatnonzero(legal).tolist()})")

        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[action_idx] = 1.0

        decision_idx = len(record.decisions)
        record.decisions.append({
            "snap_idx": ctx.snap_idx,
            "acting_pos": ctx.acting_pos,
            "member": member_idx,
            "action_idx": action_idx,
            "legal_mask": legal.copy(),
        })

        if state["runout"] is not None and probs is not None:
            self._fold_correction(state, ctx, p, action_idx)

        turn_before = int(table.turn)
        end, several_all_in, _state, _bet = table.step(torch.from_numpy(onehot))
        record.snapshots.append(
            _snapshot(table, active_pos=table.active_player,
                      action=onehot.tolist()))
        if state["runout"] is not None:
            self._card_correction(state, decision_idx, turn_before, end)
        if end:
            state["ended"] = True
            state["done"] = True
        elif several_all_in:
            state["done"] = True

    def _fold_correction(self, state, ctx, p, action_idx):
        """Subtract the luck of *this* fold/no-fold draw (`env/runout.py`).

        The baseline is blind to how much a player bet — chips beyond the call
        come back once betting freezes — so the only thing a decision can
        surprise it with is a seat leaving the showdown. Where folding is not
        legal there is nothing to subtract and the term is exactly zero.

        Its expectation over the draw is zero by construction, whatever the
        baseline is worth, so this cannot move what the rollout estimates.
        """
        table, runout = state["table"], state["runout"]
        if not ctx.legal_mask[FOLD]:
            return
        pos = ctx.acting_pos
        live = table.players_state >= 0
        called = np.asarray(table.cumulative_bets, dtype=np.float64).copy()
        called[pos] += min(table.high_bet - table.bets[pos],
                           table.credits[pos])
        still_in = live.copy()
        still_in[pos] = False

        gap = (runout.baseline(table.turn, still_in, table.cumulative_bets)
               - runout.baseline(table.turn, live, called))
        p_fold = float(p[FOLD])
        state["corr"] += ((1.0 - p_fold) if action_idx == FOLD
                          else -p_fold) * gap

    def _card_correction(self, state, decision_idx, turn_before, end):
        """Subtract the luck of the cards this action turned over.

        Two cases, one formula. If decisions remain, the baseline is a
        martingale over the deal, so the surprise of a new street is exactly the
        change in the baseline across it. If none remain — everybody left is
        all-in — every street still to come is one more chance node with nothing
        between them, so their corrections telescope into a single difference;
        `_finish` adds it once the board is complete. Where the pot has no side
        pot that difference cancels the hand's own result and what the rollout
        reports is the average over every runout it could have had.
        """
        table, record = state["table"], state["record"]
        runout = state["runout"]
        # Streets dealt while a forced prefix replays are streets hero had
        # already seen when the labelled decision was taken. They are
        # conditioned on rather than drawn, so there is no luck to remove and
        # removing some anyway would be a bias, not a reduction.
        forced = record.spec.forced_actions
        if forced is not None and decision_idx < len(forced) - 1:
            return
        if end:
            return
        live = table.players_state >= 0
        if table.several_all_in:
            # No decision is left, so every street still to come is one more
            # chance node and their corrections telescope. `_finish` adds the
            # single difference they collapse to, once the board is complete.
            state["run_out_from"] = (turn_before, live.copy(),
                                     np.asarray(table.cumulative_bets,
                                                dtype=np.float64).copy())
        elif int(table.turn) != turn_before:
            state["corr"] += (
                runout.baseline(table.turn, live, table.cumulative_bets)
                - runout.baseline(turn_before, live, table.cumulative_bets))

    def _finish(self, state):
        """Run out any all-in board, then record the per-seat chip delta."""
        table = state["table"]
        record = state["record"]

        if not state["ended"]:
            noop = torch.zeros(self.n_actions, dtype=torch.float32)
            # `several_all_in` makes `step` a pure `next_turn()`, which deals the
            # remaining streets and settles at the river. Bounded by 4 streets.
            for _ in range(8):
                if not table.several_all_in:
                    break
                end, _several, _s, _b = table.step(noop)
                if end:
                    state["ended"] = True
                    break

        record.rewards = (np.asarray(table.credits, dtype=np.float64)
                          - np.asarray(record.spec.start_credits, dtype=np.float64))
        if state["runout"] is not None:
            pending = state["run_out_from"]
            if pending is not None:
                turn_before, live, bets = pending
                runout = state["runout"]
                # `RIVER` and not `table.turn`: the engine's no-op steps may
                # stop early, and what the corrections telescope to is the value
                # on the complete board.
                state["corr"] += (runout.baseline(RIVER, live, bets)
                                  - runout.baseline(turn_before, live, bets))
            record.baseline_rewards = record.rewards - state["corr"]
        # §5.1a: which seats showed their cards. Only the seats — the labels
        # they carry are cards-only and are computed once over the whole corpus
        # by `env.showdown.label_showdowns`, never inside this loop.
        record.showdown = showdown_positions(table.players_state)
        return record


def _snapshot(table, active_pos, action):
    return {
        "pot": float(table.pot),
        "bets": np.copy(table.bets),
        "credits": [float(c) for c in table.credits],
        "turn": int(table.turn),
        "active_pos": int(active_pos),
        "action": action,
    }
