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
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from env.table import Table
from env.legal import legal_action_mask
from env.showdown import showdown_positions


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


@dataclass
class HandRecord:
    """What one played hand leaves behind."""
    spec: HandSpec
    deck: np.ndarray
    snapshots: list               # v7-convention snapshots (see module docstring)
    decisions: list               # dicts: snap_idx, acting_pos, member, action_idx, legal_mask
    rewards: np.ndarray           # per-seat chip delta over the hand
    truncated: bool               # betting hit the max-actions cap
    showdown: list = field(default_factory=list)          # seats revealed, §5.1a
    showdown_strength: dict = field(default_factory=dict)  # seat → percentile
    showdown_class: dict = field(default_factory=dict)     # seat → 169-way class

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

    __slots__ = ("record", "snap_idx", "acting_pos", "legal_mask", "turn")

    def __init__(self, record, snap_idx, acting_pos, legal_mask, turn):
        self.record = record
        self.snap_idx = snap_idx
        self.acting_pos = acting_pos
        self.legal_mask = legal_mask
        self.turn = turn

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
        """The **acting player's** own hole cards — never anybody else's."""
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


class LockstepDriver:
    """Advances many hands together against a fixed pool.

    Args:
        pool: sequence of pool members; `HandSpec.seat_members` indexes it.
        n_actions: size of the discrete action set.
    """

    def __init__(self, pool, n_actions):
        self.pool = pool
        self.n_actions = n_actions

    def run(self, specs, batch_size=None):
        """Play every spec. Returns `HandRecord`s in spec order.

        `batch_size` caps how many hands are in flight at once; `1` is the
        sequential path. The result must not depend on it.
        """
        if batch_size is None:
            batch_size = len(specs)
        batch_size = max(1, int(batch_size))

        records = [None] * len(specs)
        pending_specs = list(enumerate(specs))
        cursor = 0
        live = []

        while cursor < len(pending_specs) or live:
            while len(live) < batch_size and cursor < len(pending_specs):
                idx, spec = pending_specs[cursor]
                cursor += 1
                live.append(self._start(idx, spec))

            queries = []
            for state in live:
                ctx = self._advance_to_decision(state)
                if ctx is not None:
                    queries.append((state, ctx))

            finished = [s for s in live if s["done"]]
            for state in finished:
                records[state["idx"]] = self._finish(state)
            live = [s for s in live if not s["done"]]

            if not queries:
                continue

            by_member = {}
            for state, ctx in queries:
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

        return records

    # ---------------------------------------------------------------- internals

    def _start(self, idx, spec):
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

    def _apply(self, state, ctx, probs, member_idx):
        table = state["table"]
        record = state["record"]

        legal = ctx.legal_mask
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

        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[action_idx] = 1.0

        record.decisions.append({
            "snap_idx": ctx.snap_idx,
            "acting_pos": ctx.acting_pos,
            "member": member_idx,
            "action_idx": action_idx,
            "legal_mask": legal.copy(),
        })

        end, several_all_in, _state, _bet = table.step(torch.from_numpy(onehot))
        record.snapshots.append(
            _snapshot(table, active_pos=table.active_player,
                      action=onehot.tolist()))
        if end:
            state["ended"] = True
            state["done"] = True
        elif several_all_in:
            state["done"] = True

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
