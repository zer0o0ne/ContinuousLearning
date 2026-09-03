"""Sessions — a fixed table playing a fixed number of hands (CONCEPT.md §14, §8).

A session is what deployment looks like: hero sits down at a table, the same
players stay put, the button rotates each hand, and hero observes them over a
run of hands and fits their vectors from the hands hero was in (§5.4). Nothing
in this file is specific to a gate or to a phase — it is the shape of the data
both G1 and label generation (`train/generate.py`) produce, which is exactly why
it lives here rather than in either of them (`PLAN_PIPELINE.md` D4). A copy
would let the two drift, and the drift would show up as a train/deploy mismatch
nobody could see.

Two properties are load-bearing and are the reason this is a class rather than a
loop:

* **The button rotates.** With fixed seating a member would be identifiable by
  its seat, and the network would learn seats instead of styles. `slot_of_seat`
  and `seat_of_slot` are the two directions of that rotation, and they are what
  keeps a player's identity stable while its seat changes.
* **Slot 0 is the observer.** Hero's slot, in G1 and in label generation alike
  (`ARCHITECTURE.md` §4, interpretive decision 6).

**Table configuration is sampled uniformly** over 2–9 players and 10–300 BB
(`CLAUDE.md` §1). Nothing is weighted toward heads-up or 200 BB.

Moved here verbatim from `gates/g1.py`, which imports it back; `test_g1_gate.py`
passing unedited is what says the move changed no behaviour.
"""

from dataclasses import dataclass, field

import numpy as np

from env.driver import HandSpec
from env.showdown import label_showdowns
from nets.features import hand_tokens

STREETS = ("preflop", "flop", "turn", "river")


@dataclass
class Session:
    """A fixed table of members playing a fixed number of hands."""

    idx: int
    num_players: int
    stack_bb: int
    members: list                 # pool-member index per slot; slot 0 observes
    specs: list = field(default_factory=list)
    records: list = field(default_factory=list)
    # §5.7 — one `{(token, seat): (combo_idx, weight)}` per hand, filled by
    # `oracle.ranges.label_ranges` when the range head is on. Empty means the
    # corpus carries no belief target, which is the head's ablation and also
    # every corpus built before §5.7 existed.
    ranges: list = field(default_factory=list)

    def seat_of_slot(self, slot, hand_idx):
        """Slot `slot` sits here in hand `hand_idx` (the button rotates)."""
        return (slot - hand_idx) % self.num_players

    def slot_of_seat(self, hand_idx):
        return [(seat + hand_idx) % self.num_players
                for seat in range(self.num_players)]

    def tokens(self, max_players, n_actions, observer_slot=0, window=None):
        """Token sequences of every hand, from slot `observer_slot`'s view.

        Slot 0 is hero's, and `observer_slot=0, window=None` is what every
        caller that fits *hero's* vectors asks for. The other slots exist for
        the §5.5 fit of a player that is not hero — a past agent seated as an
        opponent conditions on its own tablemates, and doing that from hero's
        tokens would show it hero's hole cards, which is §9's parity broken from
        the pool's side. One observer, one tokenisation.

        `window` keeps only the last `window` hands played, the history the fit
        is allowed to look at; `None` is the whole session so far.

        The §5.7 range targets are hero's: they name a (token, seat) at which
        hero has a live opponent, and `hand_tokens` refuses a key naming a seat
        *this* observer has no live opponent at — the observer's own seat among
        them. They are therefore passed for slot 0 alone; the amortised pass has
        no use for them.
        """
        lo = 0 if window is None else max(0, len(self.records) - int(window))
        out = []
        for h in range(lo, len(self.records)):
            out.append(hand_tokens(
                self.records[h],
                observer_pos=self.seat_of_slot(observer_slot, h),
                slot_of_seat=self.slot_of_seat(h),
                max_players=max_players, n_actions=n_actions,
                ranges=(self.ranges[h]
                        if self.ranges and observer_slot == 0 else None),
            ))
        return out


# The phases of one iteration that deal hands, in the order their seed ranges
# are laid out. Adding a third phase means adding it here and nowhere else.
HAND_PHASES = ("labels", "corpus")


def phase_hands(cfg):
    """How many hands each phase of one iteration deals, from the config.

    Read from the same keys the phases themselves read, so the seed layout
    cannot describe a run different from the one that happens.
    """
    emb = cfg["embedding_net"]
    return {
        "labels": int(cfg["n_sessions"]) * int(cfg["hands_per_session"]),
        "corpus": (int(emb["corpus_sessions"])
                   * int(emb["corpus_hands_per_session"])),
    }


def hand_seed_bases(iteration_seed, hands):
    """`({phase: first seed}, span)` for one iteration.

    Every hand of the run gets a distinct `HandSpec.seed`, and that is the whole
    requirement: the seed fixes the deck and every action draw (`env/driver.py`),
    so two hands sharing one are the *same* hand. Two phases sharing hands would
    make the embedding corpus and the labelled set correlated in a way nothing
    downstream could see.

    The ranges are laid out end to end and each is exactly as long as its phase
    asks for, so the layout scales with the config instead of capping it. An
    earlier version reserved a fixed 500 000 seeds per phase and refused any run
    that wanted more, which is a limit on the experiment imposed by its
    bookkeeping — the wrong way round.

    Iteration `k` takes the whole `span` starting at `iteration_seed · span`, so
    the ranges of consecutive iterations abut and never overlap either.
    """
    span = sum(hands[phase] for phase in HAND_PHASES)
    bases, offset = {}, 0
    for phase in HAND_PHASES:
        bases[phase] = int(iteration_seed) * span + offset
        offset += hands[phase]
    return bases, span


def assert_seeds_stay_distinct(span, n_iterations):
    """The one real constraint on the layout, named instead of guessed at.

    `env/driver.py::_start` deals the deck through `np.random.seed`, which takes
    32 bits, so it sees `spec.seed % 2**32` and two hands whose seeds differ by a
    multiple of `2**32` are dealt the same cards. Every hand of a run lies in one
    contiguous interval of `n_iterations · span` seeds, so "all distinct" is
    exactly that product fitting in 32 bits — which at any plausible size it does
    by orders of magnitude.
    """
    total = int(span) * int(n_iterations)
    assert total <= 2 ** 32, (
        f"{n_iterations} iterations of {span} hands is {total} seeds, past the "
        f"{2 ** 32} the deck draw can tell apart (`np.random.seed` takes 32 "
        f"bits), so some hands would be dealt identical cards")


def raise_sizes_from(game):
    return [list(game["raise_sizes"][s]) for s in STREETS]


def build_sessions(rng, member_ids, game, n_sessions, hands_per_session,
                   seed_base, tag):
    """Uniform over 2–9 players and 10–300 BB, independently (`CLAUDE.md` §1)."""
    lo_p, hi_p = game["players_range"]
    lo_s, hi_s = game["stack_bb_range"]
    bb, sb = game["big_blind"], game["small_blind"]
    raise_sizes = raise_sizes_from(game)

    sessions = []
    for s in range(n_sessions):
        num_players = int(rng.integers(lo_p, hi_p + 1))
        assert num_players <= len(member_ids), (
            f"[{tag}] a {num_players}-handed session needs {num_players} "
            f"distinct pool members, this set has {len(member_ids)}. Table size "
            f"is sampled uniformly over {game['players_range']} and is not "
            f"negotiable (CLAUDE.md §1), so widen the member set instead — more "
            f"`bootstrap` variants, or a larger `corpus.n_unseen_members`.")
        stack_bb = int(rng.integers(lo_s, hi_s + 1))
        members = [int(m) for m in
                   rng.choice(member_ids, size=num_players, replace=False)]
        session = Session(idx=s, num_players=num_players, stack_bb=stack_bb,
                          members=members)
        for h in range(hands_per_session):
            seat_members = [members[(seat + h) % num_players]
                            for seat in range(num_players)]
            session.specs.append(HandSpec(
                num_players=num_players,
                start_credits=[float(stack_bb * bb)] * num_players,
                seat_members=seat_members,
                seed=seed_base + s * hands_per_session + h,
                big_blind=bb, small_blind=sb, raise_sizes=raise_sizes,
                meta={"tag": tag, "session": s, "hand": h},
            ))
        sessions.append(session)
    return sessions


def play(driver, sessions, batch_size, log, tag, bar=True):
    """Play every hand of every session in lock-step, then hand them back.

    `bar=False` suppresses the driver's progress bar, for a caller that plays
    the corpus in several calls and carries its own global bar over the whole
    job — one bar, never one per call (`CLAUDE.md` §5).
    """
    specs = [spec for s in sessions for spec in s.specs]
    log(f"[{tag}] playing {len(specs)} hands over {len(sessions)} sessions")
    records = driver.run(specs, batch_size=batch_size,
                         desc=f"play:{tag}" if bar else None)

    cursor = 0
    n_decisions = 0
    truncated = 0
    for s in sessions:
        s.records = records[cursor:cursor + len(s.specs)]
        cursor += len(s.specs)
        n_decisions += sum(len(r.decisions) for r in s.records)
        truncated += sum(1 for r in s.records if r.truncated)
    # §5.1a: the showdown labels are cards-only, so they are computed once here
    # over the whole set and never again inside a training or fitting loop.
    n_reveals = label_showdowns(records,
                                desc=f"showdown:{tag}" if bar else None)
    n_showdown_hands = sum(1 for r in records if r.showdown)
    log(f"[{tag}] {n_decisions} decisions, {n_reveals} reveals over "
        f"{n_showdown_hands} showdown hands, {truncated} hands hit the "
        f"max-actions cap")
    return n_decisions
