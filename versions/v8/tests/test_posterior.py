"""The reach-weighted opponent posterior (CONCEPT.md §7.2, PLAN_PIPELINE.md S2).

Every test here plays a real hand through the driver and asks the module the
question the oracle will ask it. The two properties that have to hold *exactly*
rather than approximately are the hand-computed example and "a card-independent
member leaves the prior alone" — the second is the behavioural form of "the
likelihood is constant in the combo", and it catches the indexing mistakes a
numeric example happily reproduces.

The hands are pinned with S1's plumbing (`HandSpec.deck`, `forced_actions`), so
each test states which cards are out and which actions were taken instead of
hoping a sampled hand happens to contain them.
"""

import copy
import itertools

import numpy as np
import pytest

from env.driver import DecisionContext, HandSpec, LockstepDriver
from oracle.posterior import (PosteriorCache, combo_universe,
                              opponent_posterior)
from pool.base import PoolMember
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import StyleParams
from tests.g1_fixtures import (BIG_BLIND, N_ACTIONS, RAISE_SIZES, SMALL_BLIND,
                               contexts_from, make_pool, make_specs)

# deck = 0,1,2,3,4 board · 5,6 seat 0 · 7,8 seat 1 · …
PINNED_DECK = np.arange(52)
LOGIT = 4.0
CALL = 1


def _degenerate(name):
    return DEGENERATE_STRATEGIES[name](name, N_ACTIONS, StyleParams.identity())


class PairRaiser(PoolMember):
    """Raises with a pocket pair, calls without one.

    An explicit function of the hole cards and of nothing else, which is what
    makes the posterior computable by hand.
    """

    def logits(self, contexts):
        out = np.zeros((len(contexts), self.n_actions), dtype=np.float64)
        for i, ctx in enumerate(contexts):
            c = ctx.hole_cards
            out[i, 2 if c[0] // 4 == c[1] // 4 else 1] = LOGIT
        return out


class NeverFolds(PoolMember):
    """Puts a hard zero — not a small number — on everything except calling."""

    def logits(self, contexts):
        out = np.full((len(contexts), self.n_actions), -1e9, dtype=np.float64)
        out[:, CALL] = 0.0
        return out


class CountingPool(list):
    """A pool that records the shape of every batch a member is asked."""

    def __init__(self, members):
        super().__init__(_Counting(m, self) for m in members)
        self.calls = []


class _Counting(PoolMember):
    def __init__(self, inner, log):
        super().__init__(inner.name, inner.n_actions, inner.style)
        self.inner = inner
        self.log = log

    def logits(self, contexts):
        return self.inner.logits(contexts)

    def policy(self, contexts):
        self.log.calls.append(len(contexts))
        return self.inner.policy(contexts)


def _play(pool, forced, num_players=2, deck=PINNED_DECK, stack_bb=200, seed=7):
    spec = HandSpec(
        num_players=num_players,
        start_credits=[float(stack_bb * BIG_BLIND)] * num_players,
        seat_members=[0] * num_players,
        seed=seed, big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
        raise_sizes=RAISE_SIZES,
        deck=None if deck is None else np.asarray(deck),
        forced_actions=list(forced),
    )
    return LockstepDriver(pool, N_ACTIONS).run([spec])[0]


def _played_by(record, pos):
    return [t for t, d in enumerate(record.decisions)
            if int(d["acting_pos"]) == pos]


def _explicit_policy(legal, boosted):
    """`PairRaiser`'s distribution, written out independently of `pool.style`."""
    z = np.where(legal, 0.0, -np.inf)
    if legal[boosted]:
        z[boosted] = LOGIT
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------- 1

def test_a_two_decision_posterior_matches_the_hand_computed_weights():
    pool = [PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())]
    record = _play(pool, forced=[CALL, CALL, CALL])
    opp, hero = 1, 0
    acted = _played_by(record, opp)
    assert len(acted) >= 2, "the pinned hand must give the opponent two decisions"
    k = acted[1]

    combos, weights = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                         through_decision=k)

    # By hand: dead = the three flop cards plus hero's two; the likelihood of a
    # combo is the product over the opponent's two decisions of the probability
    # `PairRaiser` gives the action it actually took.
    dead = [0, 1, 2] + record.hole_cards(hero)
    expected_combos = np.asarray(
        list(itertools.combinations([c for c in range(52) if c not in dead], 2)),
        dtype=np.int64)
    assert np.array_equal(combos, expected_combos)

    expected = np.ones(len(combos), dtype=np.float64)
    for t in acted[:2]:
        dec = record.decisions[t]
        legal = np.asarray(dec["legal_mask"], dtype=bool)
        a = int(dec["action_idx"])
        for i, (c0, c1) in enumerate(combos):
            boosted = 2 if c0 // 4 == c1 // 4 else CALL
            expected[i] *= _explicit_policy(legal, boosted)[a]
    expected /= expected.sum()

    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-12)

    # And the shape of the answer: a pair is a different weight from a non-pair,
    # and every combo of a kind carries the same weight.
    is_pair = combos[:, 0] // 4 == combos[:, 1] // 4
    assert is_pair.any() and (~is_pair).any()
    assert weights[is_pair].std() < 1e-18 and weights[~is_pair].std() < 1e-18
    assert weights[is_pair][0] != weights[~is_pair][0]


def test_each_opponent_decision_costs_exactly_one_batched_policy_call():
    pool = CountingPool([PairRaiser("pair_raiser", N_ACTIONS,
                                    StyleParams.identity())])
    record = _play(pool, forced=[CALL, CALL, CALL])
    opp, hero = 1, 0
    acted = _played_by(record, opp)
    k = acted[1]
    pool.calls.clear()

    combos, _w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                    through_decision=k)

    assert pool.calls == [len(combos), len(combos)]


def test_consecutive_hero_decisions_reuse_the_opponents_prefix_likelihoods():
    """The cache asks about every opponent action once, not once per label.

    Fresh and cached answers are compared at every hero decision, including
    decisions after new board cards appear.  The latter pins the less obvious
    part of the optimisation: restricting the cached universe by card removal
    is the same posterior as recomputing every old action on the new street.
    """
    inner = PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())
    record = _play([inner], forced=[CALL] * 7)
    hero, opp = 0, 1
    hero_decisions = _played_by(record, hero)
    assert len(hero_decisions) >= 3, (
        "the pinned hand must give hero several labels to share a cache")

    fresh_pool = CountingPool([
        PairRaiser("fresh", N_ACTIONS, StyleParams.identity())])
    cached_pool = CountingPool([
        PairRaiser("cached", N_ACTIONS, StyleParams.identity())])
    cache = PosteriorCache()
    fresh_rows = cached_rows = 0
    combo_counts = []

    for decision_idx in hero_decisions:
        through = decision_idx - 1
        fresh_c, fresh_w = opponent_posterior(
            record, opp, hero, fresh_pool, N_ACTIONS,
            through_decision=through)
        cached_c, cached_w, rows = cache.posterior(
            record, opp, hero, cached_pool, N_ACTIONS,
            through_decision=through)

        assert np.array_equal(cached_c, fresh_c)
        np.testing.assert_allclose(cached_w, fresh_w, rtol=0.0, atol=1e-15)
        acted = sum(1 for d in record.decisions[:decision_idx]
                    if int(d["acting_pos"]) == opp)
        fresh_rows += len(fresh_c) * acted
        cached_rows += rows
        combo_counts.append(len(cached_c))

    assert len(cached_pool.calls) == sum(
        int(record.decisions[t]["acting_pos"]) == opp
        for t in range(hero_decisions[-1]))
    assert cached_rows < fresh_rows
    assert len(set(combo_counts)) > 1, (
        "the fixture must cross a street so newly visible cards are removed")


def test_a_capped_posterior_deliberately_bypasses_the_cache():
    """Independent per-label prior subsamples must keep their RNG semantics."""
    pool = CountingPool([
        PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())])
    record = _play(pool, forced=[CALL] * 4)
    hero, opp = 0, 1
    decisions = _played_by(record, hero)
    assert len(decisions) >= 2
    cache = PosteriorCache()

    answers = []
    for decision_idx in decisions[:2]:
        answers.append(cache.posterior(
            record, opp, hero, pool, N_ACTIONS,
            through_decision=decision_idx - 1, max_combos=32,
            rng=np.random.default_rng(decision_idx)))

    # Each answer is still independently capped, and `rows` is the full cost
    # of its own prefix rather than an incremental/cache-hit count.
    for decision_idx, (combos, weights, rows) in zip(decisions, answers):
        acted = sum(1 for d in record.decisions[:decision_idx]
                    if int(d["acting_pos"]) == opp)
        assert combos.shape == (32, 2)
        assert weights.shape == (32,)
        assert rows == 32 * acted


# --------------------------------------------------------------------- 2

def test_card_removal_is_relative_to_the_observer():
    pool = [_degenerate("always_call")]
    record = _play(pool, forced=[CALL] * 7)
    hero, opp = 0, 1
    river = [t for t, d in enumerate(record.decisions)
             if record.snapshots[d["snap_idx"]]["turn"] == 3
             and int(d["acting_pos"]) == opp]
    assert river, "the pinned hand must reach a river decision by the opponent"

    combos, _w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                    through_decision=river[0])

    board = [int(c) for c in record.deck[:5]]
    hero_cards = record.hole_cards(hero)
    assert len(combos) == 990                       # C(52 − 5 − 2, 2)
    seen = set(combos.reshape(-1).tolist())
    assert not seen & set(board + hero_cards)
    # The opponent's real holding is *in* the universe — it is what the
    # posterior is over.
    assert tuple(sorted(record.hole_cards(opp))) in {
        tuple(c) for c in combos.tolist()}


def test_the_universe_shrinks_by_exactly_the_dead_cards():
    assert len(combo_universe([])) == 1326
    assert len(combo_universe([0, 1, 2, 3, 4])) == 1081       # C(47, 2)
    assert len(combo_universe(list(range(50)))) == 1          # C(2, 2)
    assert combo_universe(list(range(50))).tolist() == [[50, 51]]
    assert len(combo_universe([7, 7, 7])) == 1275             # C(51, 2)


# --------------------------------------------------------------------- 3

def test_every_prefix_length_returns_a_normalised_posterior():
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL] * 4)
    for k in range(-1, len(record.decisions)):
        for opp in (0, 1):
            combos, w = opponent_posterior(record, opp, 1 - opp, pool,
                                           N_ACTIONS, through_decision=k)
            assert w.shape == (len(combos),)
            assert (w >= 0).all()
            assert abs(float(w.sum()) - 1.0) < 1e-12


# --------------------------------------------------------------------- 4

def test_a_card_independent_member_leaves_the_prior_untouched():
    pool = [_degenerate("always_call")]
    record = _play(pool, forced=[CALL] * 6)
    opp, hero = 1, 0
    acted = _played_by(record, opp)
    assert len(acted) >= 3

    for k in acted:
        combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                       through_decision=k)
        np.testing.assert_allclose(w, np.full(len(combos), 1.0 / len(combos)),
                                   rtol=0.0, atol=1e-12)


def test_a_card_dependent_member_does_not_leave_the_prior_untouched():
    """The mirror of the test above: otherwise a no-op would pass both."""
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL, CALL])
    opp, hero = 1, 0
    k = _played_by(record, opp)[0]
    _combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                    through_decision=k)
    assert w.std() > 1e-6


# --------------------------------------------------------------------- 5

def test_the_posterior_is_a_prefix_function_of_the_hand():
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL] * 6)
    opp, hero = 1, 0

    for k in _played_by(record, opp):
        truncated = copy.deepcopy(record)
        truncated.decisions = truncated.decisions[:k + 1]
        truncated.snapshots = truncated.snapshots[:2 * (k + 1)]
        truncated.showdown = []
        truncated.showdown_strength = {}
        truncated.showdown_class = {}

        full_c, full_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                            through_decision=k)
        cut_c, cut_w = opponent_posterior(truncated, opp, hero, pool, N_ACTIONS,
                                          through_decision=k)
        assert np.array_equal(full_c, cut_c)
        assert np.array_equal(full_w, cut_w)      # bit-identical, not close


# --------------------------------------------------------------------- 6

def test_an_opponent_who_has_not_acted_yet_is_the_prior():
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL, CALL])
    opp, hero = 1, 0
    first = _played_by(record, opp)[0]

    for k in (-1, first - 1):
        combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                       through_decision=k)
        np.testing.assert_allclose(w, np.full(len(combos), 1.0 / len(combos)),
                                   rtol=0.0, atol=1e-12)


def test_the_empty_prefix_sees_no_board_at_all():
    pool = [_degenerate("always_call")]
    record = _play(pool, forced=[CALL] * 7)
    combos, _w = opponent_posterior(record, 1, 0, pool, N_ACTIONS,
                                    through_decision=-1)
    assert len(combos) == 1225                      # C(52 − 2, 2)


def test_a_zero_likelihood_falls_back_to_the_prior_and_says_so():
    pool = [NeverFolds("never_folds", N_ACTIONS, StyleParams.identity())]
    record = _play(pool, forced=[0])               # forced to fold: p = 0 exactly
    assert len(record.decisions) == 1
    opp = int(record.decisions[0]["acting_pos"])
    hero = 1 - opp

    with pytest.warns(RuntimeWarning, match="zero likelihood"):
        combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                       through_decision=0, floor=0.0)

    assert np.isfinite(w).all()
    np.testing.assert_allclose(w, np.full(len(combos), 1.0 / len(combos)),
                               rtol=0.0, atol=1e-12)

    # The floor is what stops this happening for real.
    _c, floored = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                     through_decision=0)
    np.testing.assert_allclose(floored, np.full(len(combos), 1.0 / len(combos)),
                               rtol=0.0, atol=1e-12)


def test_the_observers_own_seat_has_no_posterior():
    pool = [_degenerate("always_call")]
    record = _play(pool, forced=[CALL, CALL])
    with pytest.raises(AssertionError, match="knows its own cards"):
        opponent_posterior(record, 0, 0, pool, N_ACTIONS, through_decision=0)
    with pytest.raises(AssertionError, match="not a decision index"):
        opponent_posterior(record, 1, 0, pool, N_ACTIONS,
                           through_decision=len(record.decisions))


# --------------------------------------------------------------------- 7

def test_max_combos_caps_the_range_normalises_it_and_is_seeded():
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL, CALL])
    opp, hero = 1, 0
    k = _played_by(record, opp)[0]
    kw = dict(through_decision=k, max_combos=64)

    a_c, a_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                  rng=np.random.default_rng(0), **kw)
    b_c, b_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                  rng=np.random.default_rng(0), **kw)

    assert a_c.shape == (64, 2)
    assert np.array_equal(a_c, b_c) and np.array_equal(a_w, b_w)
    assert abs(float(a_w.sum()) - 1.0) < 1e-12
    assert len({tuple(c) for c in a_c.tolist()}) == 64

    # A cap above the universe is a no-op, and needs no generator.
    full_c, full_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                        through_decision=k, max_combos=10_000)
    plain_c, plain_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                          through_decision=k)
    assert np.array_equal(full_c, plain_c) and np.array_equal(full_w, plain_w)

    # One combo asked for is one combo returned, carrying all the weight.
    one_c, one_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                      through_decision=k, max_combos=1,
                                      rng=np.random.default_rng(3))
    assert one_c.shape == (1, 2) and one_w.tolist() == [1.0]

    with pytest.raises(AssertionError, match="np.random.Generator"):
        opponent_posterior(record, opp, hero, pool, N_ACTIONS, **kw)


# --------------------------------------------------- D2: `hole_override`

def test_a_hole_override_changes_the_cards_and_nothing_else():
    pool = make_pool()
    records = LockstepDriver(pool, N_ACTIONS).run(make_specs(n_hands=6,
                                                             n_members=len(pool)))
    for ctx in contexts_from(records)[:40]:
        assert ctx.hole_override is None
        real = ctx.hole_cards
        hypothetical = DecisionContext(ctx.record, ctx.snap_idx, ctx.acting_pos,
                                       ctx.legal_mask, ctx.turn,
                                       hole_override=[51, 50])
        assert hypothetical.hole_cards == [51, 50]
        assert hypothetical.board == ctx.board
        assert hypothetical.pot == ctx.pot
        assert hypothetical.to_call == ctx.to_call
        assert hypothetical.stack == ctx.stack
        assert np.array_equal(hypothetical.credits, ctx.credits)
        # The record is untouched: the override is a view, not an edit.
        assert ctx.hole_cards == real
        assert ctx.record.hole_cards(ctx.acting_pos) == real


def test_a_member_answers_the_override_not_the_real_hand():
    """`nit` folds trash and calls aces — asked about aces, it calls."""
    pool = [_degenerate("nit")]
    record = _play(pool, forced=[CALL, CALL])
    dec = record.decisions[0]
    pos = int(dec["acting_pos"])
    snap_idx, turn = dec["snap_idx"], record.snapshots[dec["snap_idx"]]["turn"]

    def p(hole):
        ctx = DecisionContext(record, snap_idx, pos, dec["legal_mask"],
                              int(turn), hole_override=hole)
        return pool[0].policy([ctx])[0]

    aces = p([48, 49])            # rank 12, 12 — a pair of aces
    trash = p([0, 5])             # deuce–trey offsuit
    assert aces[CALL] > 0.9 > trash[CALL]
    assert trash[0] > 0.9         # folds


# ------------------------------------- 7b: what the cap buys and what it costs

def test_the_cap_is_spent_before_any_member_is_asked():
    """The point of the cap is policy calls, not a shorter answer."""
    pool = CountingPool([_degenerate("nit")])
    record = _play(pool, forced=[CALL, CALL, CALL])
    opp, hero = 1, 0
    k = _played_by(record, opp)[1]
    pool.calls.clear()

    combos, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                   through_decision=k, max_combos=32,
                                   rng=np.random.default_rng(0))

    assert pool.calls == [32, 32]          # not [1081, 1081]
    assert combos.shape == (32, 2) and w.shape == (32,)


def test_the_draw_does_not_depend_on_the_posterior_it_is_drawing_from():
    """Selection independent of the weights is what removes the bias.

    Two members with very different posteriors over the same hand draw the
    *same* combos under the same seed, because the proposal is the prior.
    """
    forced = [CALL, CALL, 2]
    pairs = [PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())]
    flat = [_degenerate("always_call")]
    kw = dict(through_decision=2, max_combos=64)

    a_c, a_w = opponent_posterior(_play(pairs, forced), 1, 0, pairs, N_ACTIONS,
                                  rng=np.random.default_rng(11), **kw)
    b_c, b_w = opponent_posterior(_play(flat, forced), 1, 0, flat, N_ACTIONS,
                                  rng=np.random.default_rng(11), **kw)

    assert np.array_equal(a_c, b_c)
    assert a_w.std() > 1e-6 and b_w.std() < 1e-18


def test_the_subsampled_weights_are_the_full_posterior_restricted_to_the_draw():
    """Self-normalisation: constant prior correction, then the likelihoods."""
    pool = [PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())]
    record = _play(pool, forced=[CALL, CALL, 2])
    opp, hero = 1, 0

    full_c, full_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                        through_decision=2)
    sub_c, sub_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                      through_decision=2, max_combos=128,
                                      rng=np.random.default_rng(5))

    where = {tuple(c): i for i, c in enumerate(full_c.tolist())}
    idx = [where[tuple(c)] for c in sub_c.tolist()]
    expected = full_w[idx] / full_w[idx].sum()
    np.testing.assert_allclose(sub_w, expected, rtol=0.0, atol=1e-12)
    # The draw keeps the universe's order, so `combos` is still sorted.
    assert idx == sorted(idx)


def test_the_subsampled_posterior_is_consistent_with_the_full_one():
    """The estimator converges on the answer instead of on the mode.

    A functional of the posterior — the mass on pocket pairs, which this hand's
    forced raise pushes to ~0.49 — averaged over 200 fixed seeds. Under the
    mode-seeking draw this module used to inherit from `gpu_solver_v5` the same
    average is ~0.89: the selection counted the mode and the weight counted it
    again.
    """
    pool = [PairRaiser("pair_raiser", N_ACTIONS, StyleParams.identity())]
    record = _play(pool, forced=[CALL, CALL, 2])
    opp, hero = 1, 0

    full_c, full_w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                        through_decision=2)
    true = float(full_w[full_c[:, 0] // 4 == full_c[:, 1] // 4].sum())
    assert 0.4 < true < 0.6, "the fixture must not be a degenerate posterior"

    estimates = []
    for seed in range(200):
        c, w = opponent_posterior(record, opp, hero, pool, N_ACTIONS,
                                  through_decision=2, max_combos=128,
                                  rng=np.random.default_rng(seed))
        estimates.append(float(w[c[:, 0] // 4 == c[:, 1] // 4].sum()))

    assert abs(float(np.mean(estimates)) - true) < 0.02
