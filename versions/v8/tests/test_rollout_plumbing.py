"""Rollout plumbing in the driver (PLAN_PIPELINE.md S1, CONCEPT.md §3, §15).

A BR-oracle rollout is "this hand, these cards, this prefix of decisions, then
free play". Two optional fields on `HandSpec` express it — `deck` pins what was
dealt, `forced_actions` replays a prefix without a single policy call — and one
optional argument on `hand_tokens` tokenises the moment *before* an action is
chosen, which is the moment the agent observes (§9).

The load-bearing test is the first one: replaying a recorded hand with its own
deck and its own action sequence must reproduce that hand exactly, element for
element. If that holds, the two fields cannot have changed the engine's
semantics, which is what makes editing the driver safe.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest

from env.driver import DecisionContext, LockstepDriver
from nets.features import UNKNOWN_CARD, hand_tokens
from pool.base import PoolMember
from pool.degenerate import AlwaysCall
from pool.style import StyleParams
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, contexts_from, make_pool, make_specs,
    play,
)


# --------------------------------------------------------------------- helpers


def rollout_spec(record, n_forced=None):
    """The spec that replays `record`: its deck, and a prefix of its actions."""
    actions = [d["action_idx"] for d in record.decisions]
    if n_forced is not None:
        actions = actions[:n_forced]
    return replace(record.spec, deck=np.copy(record.deck),
                   forced_actions=actions)


def _snapshots_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y):
            return False
        for k in x:
            if k == "bets":
                if not np.array_equal(np.asarray(x[k]), np.asarray(y[k])):
                    return False
            elif x[k] != y[k]:
                return False
    return True


def _decisions_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x["snap_idx"], x["acting_pos"], x["member"], x["action_idx"]) != \
           (y["snap_idx"], y["acting_pos"], y["member"], y["action_idx"]):
            return False
        if not np.array_equal(x["legal_mask"], y["legal_mask"]):
            return False
    return True


def assert_record_identical(a, b):
    assert np.array_equal(a.deck, b.deck)
    assert _snapshots_equal(a.snapshots, b.snapshots)
    assert _decisions_equal(a.decisions, b.decisions)
    assert np.array_equal(a.rewards, b.rewards)
    assert a.truncated == b.truncated
    assert a.showdown == b.showdown


def _call_only_pool():
    return [AlwaysCall("call", N_ACTIONS, StyleParams.identity())]


class _FreezingSpy(PoolMember):
    """A member that keeps a frozen `DecisionContext` per query it answers.

    The record a live context points at keeps mutating as the hand proceeds, so
    the copy is what a caller of `hand_tokens(..., pending=ctx)` would actually
    be holding: a hand in progress.
    """

    def __init__(self, inner):
        super().__init__("spy", inner.n_actions, inner.style)
        self.inner = inner
        self.frozen = []

    def logits(self, contexts):
        return self.inner.logits(contexts)

    def policy(self, contexts):
        for c in contexts:
            self.frozen.append(DecisionContext(
                copy.deepcopy(c.record), c.snap_idx, c.acting_pos,
                np.copy(c.legal_mask), c.turn))
        return self.inner.policy(contexts)


# ------------------------------------------------------------- 1. replay identity


def test_replaying_a_recorded_hand_reproduces_it_exactly():
    """Deck + full forced prefix ⇒ the same hand, element for element."""
    pool = make_pool()
    records = play(pool, make_specs(seed=11, n_hands=20, n_members=len(pool)))

    replayed = play(pool, [rollout_spec(r) for r in records])
    for original, again in zip(records, replayed):
        assert_record_identical(original, again)
        assert len(again.decisions) == len(original.decisions)


@pytest.mark.parametrize("num_players", list(range(2, 10)))
@pytest.mark.parametrize("stack_bb", [10, 300])
def test_replay_identity_at_every_table_size_and_both_stack_extremes(
        num_players, stack_bb):
    pool = make_pool()
    records = play(pool, make_specs(seed=20 + num_players, n_hands=6,
                                    n_members=len(pool),
                                    num_players=num_players, stack_bb=stack_bb))
    replayed = play(pool, [rollout_spec(r) for r in records])
    for original, again in zip(records, replayed):
        assert_record_identical(original, again)


def test_a_forced_replay_never_calls_a_policy():
    """Where the oracle's saving comes from: a replayed prefix is free."""
    pool = make_pool()
    records = play(pool, make_specs(seed=12, n_hands=8, n_members=len(pool)))

    spies = [_FreezingSpy(m) for m in pool]
    play(spies, [rollout_spec(r) for r in records])
    assert sum(len(s.frozen) for s in spies) == 0
    assert sum(len(r.decisions) for r in records) > 0


# ---------------------------------------------------------- 2. deck override


def test_the_deck_override_deals_exactly_what_was_asked():
    pool = _call_only_pool()
    rng = np.random.default_rng(7)
    deck = rng.permutation(52)

    spec = replace(make_specs(seed=13, n_hands=1, n_members=1,
                              num_players=4, stack_bb=100)[0],
                   deck=deck)
    record = play(pool, [spec])[0]

    assert np.array_equal(record.deck, deck)
    for pos in range(4):
        assert record.hole_cards(pos) == [int(c) for c in deck[5 + 2 * pos:
                                                               7 + 2 * pos]]

    board_of_turn = {
        0: [-1] * 5,
        1: [int(c) for c in deck[:3]] + [-1, -1],
        2: [int(c) for c in deck[:4]] + [-1],
        3: [int(c) for c in deck[:5]],
    }
    turns_seen = set()
    for ctx in contexts_from([record]):
        assert ctx.board == board_of_turn[ctx.turn]
        assert ctx.hole_cards == record.hole_cards(ctx.acting_pos)
        turns_seen.add(ctx.turn)
    assert turns_seen == {0, 1, 2, 3}, (
        "everyone calling should reach the river on every street")


def test_a_deck_that_is_not_a_permutation_is_refused():
    pool = _call_only_pool()
    spec = replace(make_specs(seed=14, n_hands=1, n_members=1,
                              num_players=2)[0],
                   deck=np.arange(52) % 51)
    with pytest.raises(AssertionError, match="permutation"):
        play(pool, [spec])


# --------------------------------------------------------- 3. partial prefix


def test_a_partial_prefix_is_replayed_then_the_hand_runs_free():
    pool = make_pool()
    records = play(pool, make_specs(seed=15, n_hands=40, n_members=len(pool)))
    long_enough = [r for r in records if len(r.decisions) >= 5]
    assert long_enough, "need hands with a prefix to cut"

    k = 3
    resumed = play(pool, [rollout_spec(r, n_forced=k) for r in long_enough])
    for original, again in zip(long_enough, resumed):
        assert np.array_equal(again.deck, original.deck)
        assert len(again.decisions) >= k
        assert _decisions_equal(again.decisions[:k], original.decisions[:k])
        assert abs(float(again.rewards.sum())) < 1e-9
        for d in again.decisions:
            assert d["legal_mask"][d["action_idx"]]
        assert len(again.snapshots) == 1 + 2 * len(again.decisions)


def test_an_empty_forced_prefix_leaves_the_hand_entirely_free():
    pool = make_pool()
    records = play(pool, make_specs(seed=16, n_hands=10, n_members=len(pool)))
    free = play(pool, [replace(r.spec, deck=np.copy(r.deck),
                               forced_actions=[]) for r in records])
    for original, again in zip(records, free):
        assert_record_identical(original, again)


# ------------------------------------------------------ 4. illegal forced action


def test_an_illegal_forced_action_raises_naming_the_seat_and_the_mask():
    pool = make_pool()
    records = play(pool, make_specs(seed=17, n_hands=30, n_members=len(pool)))

    target = None
    for r in records:
        for i, d in enumerate(r.decisions):
            illegal = np.flatnonzero(~np.asarray(d["legal_mask"], dtype=bool))
            if len(illegal):
                target = (r, i, int(illegal[0]))
                break
        if target:
            break
    assert target, "need a decision with at least one illegal action"

    record, i, bad = target
    spec = rollout_spec(record)
    spec.forced_actions[i] = bad
    with pytest.raises(AssertionError,
                       match=rf"forced action {bad} is illegal for seat "
                             rf"{record.decisions[i]['acting_pos']} .*legal="):
        play(pool, [spec])


# ------------------------------------------------------------- 6. pending token


def _hand_with_pending(seed=18, num_players=3, min_decisions=4):
    """Play one hand, keeping a frozen in-progress context per decision."""
    spy = _FreezingSpy(AlwaysCall("call", N_ACTIONS, StyleParams.identity()))
    spec = make_specs(seed=seed, n_hands=1, n_members=1,
                      num_players=num_players, stack_bb=100)[0]
    record = play([spy], [spec])[0]
    assert len(spy.frozen) >= min_decisions
    return record, spy.frozen


def test_a_pending_decision_adds_one_token_that_carries_no_action():
    _record, frozen = _hand_with_pending()
    slots = list(range(MAX_PLAYERS))

    for pending in frozen:
        in_progress = pending.record
        k = len(in_progress.decisions)

        with_pending = hand_tokens(in_progress, 0, slots, MAX_PLAYERS,
                                   N_ACTIONS, pending=pending)
        without = hand_tokens(in_progress, 0, slots, MAX_PLAYERS, N_ACTIONS)

        assert len(with_pending) == k + 1
        assert len(without) == k

        for name in ("cards", "acting_pos", "num_players", "scalars",
                     "seat_stacks", "prev_action", "member", "slot", "action",
                     "legal", "token_type", "sd_strength", "sd_class"):
            a = getattr(with_pending, name)[:k]
            b = getattr(without, name)
            assert np.array_equal(a, b), (
                f"the pending token changed {name} on an earlier token")

        assert with_pending.action[k] == -1
        assert np.array_equal(with_pending.legal[k], pending.legal_mask)
        assert with_pending.token_type[k] == 0
        assert with_pending.acting_pos[k] == pending.acting_pos
        if k > 0:
            assert np.argmax(with_pending.prev_action[k]) == \
                in_progress.decisions[-1]["action_idx"]


def test_the_pending_token_shows_the_observers_cards_and_nobody_elses():
    _record, frozen = _hand_with_pending()
    slots = list(range(MAX_PLAYERS))
    pending = frozen[-1]
    in_progress = pending.record
    k = len(in_progress.decisions)

    for observer in range(in_progress.num_players):
        tokens = hand_tokens(in_progress, observer, slots, MAX_PLAYERS,
                             N_ACTIONS, pending=pending)
        if pending.acting_pos == observer:
            assert tokens.cards[k, 5:].tolist() == \
                in_progress.hole_cards(observer)
        else:
            assert tokens.cards[k, 5:].tolist() == [UNKNOWN_CARD] * 2


def test_a_pending_decision_and_a_showdown_together_are_refused():
    pool = make_pool()
    records = play(pool, make_specs(seed=19, n_hands=40, n_members=len(pool)))
    shown = [r for r in records if r.showdown]
    assert shown, "need a hand that reached showdown"

    record = shown[0]
    ctx = contexts_from([record])[0]
    with pytest.raises(AssertionError, match="two moments"):
        hand_tokens(record, 0, list(range(MAX_PLAYERS)), MAX_PLAYERS,
                    N_ACTIONS, pending=ctx)
