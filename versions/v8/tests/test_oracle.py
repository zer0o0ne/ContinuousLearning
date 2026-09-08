"""The BR oracle, variant A (PLAN_PIPELINE.md S3, CONCEPT.md §7.1, §15).

`Q(s, a)` is a Monte-Carlo average, so the temptation is to test it with a
tolerance and a large sample. Every test here is exact instead
(`CLAUDE.md` §4): the fold value is a closed form, the enumerated case is built
so that hero's payoff is constant on the *support* of the posterior — and
different off it, which is what makes the test discriminate — and the collision
cases are constructed so the rejection rate is 0 or 1 by hand count rather than
by measurement.
"""

import copy

import numpy as np
import pytest

from env.driver import HandSpec, LockstepDriver
from oracle.posterior import PosteriorCache, opponent_posterior
from oracle.rollout import FOLD, OracleConfig, action_values
from pool.base import PoolMember
from pool.style import StyleParams
from tests.g1_fixtures import (BIG_BLIND, N_ACTIONS, RAISE_SIZES, SMALL_BLIND,
                               make_pool, make_specs, play)

CALL = 1
ALLIN = N_ACTIONS - 1
HARD = 1e9          # softmaxes to an exact one-hot, so a rollout is a function


# ------------------------------------------------------------------- members


class Deterministic(PoolMember):
    """A member with an exactly one-hot played distribution.

    Exactness is the point: with one-hot policies a rollout is a deterministic
    function of the cards, so an enumerated `Q` can be compared to a sampled one
    without a tolerance.
    """

    def action_for(self, ctx):
        raise NotImplementedError

    def logits(self, contexts):
        out = np.zeros((len(contexts), self.n_actions), dtype=np.float64)
        for i, ctx in enumerate(contexts):
            out[i, self.action_for(ctx)] = HARD
        return out


class Caller(Deterministic):
    """Calls or checks, exactly."""

    def action_for(self, ctx):
        return CALL


class ShovesInSet(Deterministic):
    """Shoves with two cards out of `cards`, calls with anything else.

    `street` restricts the shove to one street, so a hand can be walked to the
    river before the range-defining action is taken. That is what makes an
    exactly enumerable case possible now that the runout is dealt rather than
    reused: on the river there is no runout left to average over.
    """

    def __init__(self, name, n_actions, cards, style=None, street=None):
        super().__init__(name, n_actions, style)
        self.cards = set(int(c) for c in cards)
        self.street = street

    def action_for(self, ctx):
        if self.street is not None and ctx.turn != self.street:
            return CALL
        return ALLIN if set(ctx.hole_cards) <= self.cards else CALL


class CallsInSet(Deterministic):
    """Calls with two cards out of `cards`, shoves with anything else.

    The inverse of `ShovesInSet`, and the inverse is what makes a *narrow*
    posterior out of a *non-terminal* action: a seat that called can only hold
    a combo from `cards`, and calling — unlike shoving — leaves the hand alive
    for the seats behind it.
    """

    def __init__(self, name, n_actions, cards, style=None):
        super().__init__(name, n_actions, style)
        self.cards = set(int(c) for c in cards)

    def action_for(self, ctx):
        return CALL if set(ctx.hole_cards) <= self.cards else ALLIN


class FoldsInSet(CallsInSet):
    """A deliberately diagnostic policy: fold only with cards in this set."""

    def action_for(self, ctx):
        return FOLD if set(ctx.hole_cards) <= self.cards else CALL


class Recorder(PoolMember):
    """Wraps a member and keeps every holding it was ever asked about."""

    def __init__(self, inner):
        super().__init__("recorder", inner.n_actions, inner.style)
        self.inner = inner
        self.seen = set()
        self.n_queries = 0

    def logits(self, contexts):
        for ctx in contexts:
            self.seen.add(tuple(sorted(ctx.hole_cards)))
        self.n_queries += len(contexts)
        return self.inner.logits(contexts)


class CapturingDriver(LockstepDriver):
    """A driver that keeps the rollout records the oracle throws away."""

    def __init__(self, pool, n_actions):
        super().__init__(pool, n_actions)
        self.captured = []

    def run(self, specs, batch_size=None, desc=None):
        played = super().run(specs, batch_size=batch_size, desc=desc)
        self.captured.extend(played)
        return played


# ------------------------------------------------------------------- fixtures


def card(rank, suit):
    """Card id — `rank` 0..12 is 2..A, `suit` 0..3 (judger's encoding)."""
    return 4 * rank + suit


def deck_with(board, holes):
    """A 52-card deck: this board, these holdings, the rest in order."""
    deck = [-1] * 52
    deck[:5] = [int(c) for c in board]
    for pos, cards in enumerate(holes):
        deck[5 + 2 * pos: 7 + 2 * pos] = [int(c) for c in cards]
    rest = iter([c for c in range(52) if c not in set(deck)])
    return np.array([c if c >= 0 else next(rest) for c in deck], dtype=np.int64)


def one_hand(pool, seat_members, deck=None, forced=None, stack_bb=10, seed=5):
    n = len(seat_members)
    spec = HandSpec(
        num_players=n, start_credits=[float(stack_bb * BIG_BLIND)] * n,
        seat_members=list(seat_members), seed=seed, big_blind=BIG_BLIND,
        small_blind=SMALL_BLIND, raise_sizes=RAISE_SIZES, deck=deck,
        forced_actions=forced)
    return play(pool, [spec])[0]


# The enumerated case: hero holds A♥A♦ on A♠K♠7♠4♥2♦ — three aces, beaten by
# any two of the four spades the shover's range is made of, and ahead of most
# of the rest of the deck. So `Q(call)` is constant over the posterior's support
# and quite different under the prior.
SPADES = [card(10, 0), card(9, 0), card(7, 0), card(6, 0)]      # Q♠ J♠ 9♠ 8♠
BOARD = [card(12, 0), card(11, 0), card(5, 0), card(2, 1), card(0, 2)]
HERO_ACES = [card(12, 1), card(12, 2)]


def shove_fold_hand():
    """Heads-up, 10 BB: the shover jams, hero must call or fold."""
    pool = [ShovesInSet("shover", N_ACTIONS, SPADES, StyleParams.identity()),
            Caller("hero", N_ACTIONS, StyleParams.identity())]
    deck = deck_with(BOARD, [SPADES[:2], HERO_ACES])
    record = one_hand(pool, [0, 1], deck=deck, stack_bb=10)
    assert [d["action_idx"] for d in record.decisions][:2] == [ALLIN, CALL]
    return pool, record


def river_shove_hand():
    """The same enumerated case, walked to the river before the jam.

    Both members check every street until the river, where the shover jams if
    it holds two of `SPADES`. Hero's decision is therefore taken with all five
    board cards visible, so `_rollout_deck` has no runout to draw and `Q` is an
    exact function of the assignment — which is what lets the test enumerate.
    """
    pool = [ShovesInSet("shover", N_ACTIONS, SPADES, StyleParams.identity(),
                        street=3),
            Caller("hero", N_ACTIONS, StyleParams.identity())]
    deck = deck_with(BOARD, [SPADES[:2], HERO_ACES])
    record = one_hand(pool, [0, 1], deck=deck, stack_bb=10)
    actions = [d["action_idx"] for d in record.decisions]
    idx = actions.index(ALLIN) + 1
    assert int(record.snapshots[record.decisions[idx]["snap_idx"]]["turn"]) == 3
    assert int(record.decisions[idx]["acting_pos"]) == 1
    return pool, record, idx


def contribution_bb(record, decision_idx):
    """What hero has already put in, in BB — the closed-form fold value."""
    dec = record.decisions[decision_idx]
    pos = int(dec["acting_pos"])
    snap = record.snapshots[dec["snap_idx"]]
    start = float(record.spec.start_credits[pos])
    return (start - float(snap["credits"][pos])) / float(record.spec.big_blind)


def truncate_after(record, decision_idx):
    """The same hand as it looked when `decision_idx` was the last decision."""
    cut = copy.deepcopy(record)
    cut.decisions = cut.decisions[:decision_idx + 1]
    cut.snapshots = cut.snapshots[:2 * (decision_idx + 1)]
    return cut


# --------------------------------------------------- 1. fold is a closed form


@pytest.mark.parametrize("num_players", list(range(2, 10)))
@pytest.mark.parametrize("stack_bb", [10, 300])
def test_folding_is_worth_exactly_what_hero_has_already_put_in(
        num_players, stack_bb):
    """No table size and no stack depth is privileged (`CLAUDE.md` §1)."""
    pool = make_pool()
    record = play(pool, make_specs(seed=40 + num_players, n_hands=1,
                                   n_members=len(pool),
                                   num_players=num_players,
                                   stack_bb=stack_bb))[0]
    idx = next(i for i, d in enumerate(record.decisions) if d["legal_mask"][FOLD])

    cfg = OracleConfig(samples_per_action=16, likelihood_floor=1e-6,
                       max_collision_retries=64)
    q, legal, stats = action_values(
        record, idx, LockstepDriver(pool, N_ACTIONS), pool, 0, cfg,
        np.random.default_rng(3))

    assert legal[FOLD] and stats.n_rollouts > 0
    np.testing.assert_allclose(q[FOLD], -contribution_bb(record, idx),
                               rtol=0.0, atol=1e-12)


# ----------------------------------------------------- 2. exact by enumeration


def test_q_is_the_posterior_weighted_sum_of_the_rollout_payoffs():
    """On the river, where the runout is empty, `Q` is exactly enumerable."""
    pool, record, idx = river_shove_hand()
    driver = LockstepDriver(pool, N_ACTIONS)
    cfg = OracleConfig(samples_per_action=16, likelihood_floor=0.0)

    q, legal, _stats = action_values(record, idx, driver, pool, 1, cfg,
                                     np.random.default_rng(11))
    assert np.flatnonzero(legal).tolist() == [FOLD, CALL]

    # The posterior, enumerated. Payoffs come from replaying the hand with each
    # combo pinned at the shover's seat — deterministic, so no seed is involved.
    prefix = [d["action_idx"] for d in record.decisions[:idx]]
    combos, weights = opponent_posterior(record, 0, 1, pool, N_ACTIONS,
                                         through_decision=idx - 1, floor=0.0)
    support = [(c, w) for c, w in zip(combos, weights) if w > 0.0]
    assert len(support) == 6, "the range is the six two-card spade holdings"

    payoff = _payoffs(pool, [c for c, _w in support], CALL, prefix)
    exact = sum(w * r for (_c, w), r in zip(support, payoff)) / BIG_BLIND
    np.testing.assert_allclose(q[CALL], exact, rtol=0.0, atol=1e-12)

    # ...and it is not the prior: hero is ahead of hands the shover never has.
    # ...off the support, where hero's three aces are usually good. Combos
    # holding a board card are skipped: they are unreachable in a rollout.
    others = [c for c, w in zip(combos, weights)
              if w == 0.0 and not set(int(x) for x in c) & set(BOARD)][:20]
    assert any(abs(r / BIG_BLIND - q[CALL]) > 1.0
               for r in _payoffs(pool, others, CALL, prefix))


def _payoffs(pool, combos, action, prefix):
    """Hero's chip delta for each combo in the shover's seat, in chips."""
    specs = []
    for i, combo in enumerate(combos):
        specs.append(HandSpec(
            num_players=2, start_credits=[100.0, 100.0], seat_members=[0, 1],
            seed=1000 + i, big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
            raise_sizes=RAISE_SIZES,
            deck=deck_with(BOARD, [list(combo), HERO_ACES]),
            forced_actions=list(prefix) + [action]))
    return [float(r.rewards[1]) for r in play(pool, specs)]


# --------------------------------------------------------------- 3. card leak


def _leak_setup(samples):
    inner = Caller("call", N_ACTIONS, StyleParams.identity())
    hero = Recorder(inner)
    pool = [inner, hero]
    record = play(pool, make_specs(seed=77, n_hands=1, n_members=1,
                                   num_players=3, stack_bb=200))[0]
    driver = CapturingDriver(pool, N_ACTIONS)
    cfg = OracleConfig(samples_per_action=samples, max_collision_retries=32)
    q, _legal, stats = action_values(record, 0, driver, pool, 1, cfg,
                                     np.random.default_rng(5))
    return record, hero, driver, q, stats


def test_hero_never_sees_a_card_it_could_not_see():
    record, hero, _driver, _q, _stats = _leak_setup(samples=96)
    hero_pos = int(record.decisions[0]["acting_pos"])

    assert hero.n_queries > 800, "hero must act again inside the rollouts"
    assert hero.seen == {tuple(sorted(record.hole_cards(hero_pos)))}


# ------------------------------------------------------- 4. chip conservation


def test_every_rollout_conserves_chips():
    _record, _hero, driver, _q, stats = _leak_setup(samples=16)
    assert driver.captured and len(driver.captured) == stats.n_rollouts
    for played in driver.captured:
        assert abs(float(played.rewards.sum())) < 1e-6


# ------------------------------------------------------------ 5. illegal → nan


def test_illegal_actions_carry_nan_and_the_mask_is_the_recorded_one():
    pool, record = shove_fold_hand()
    q, legal, _stats = action_values(
        record, 1, LockstepDriver(pool, N_ACTIONS), pool, 1,
        OracleConfig(samples_per_action=4, likelihood_floor=0.0),
        np.random.default_rng(2))

    assert np.array_equal(legal, np.asarray(record.decisions[1]["legal_mask"]))
    assert np.isnan(q[~legal]).all()
    assert np.isfinite(q[legal]).all()


# ------------------------------------------------------------ 6. reproducible


def test_the_same_seed_gives_a_bit_identical_label():
    pool = make_pool()
    record = play(pool, make_specs(seed=51, n_hands=1, n_members=len(pool),
                                   num_players=4, stack_bb=100))[0]
    cfg = OracleConfig(samples_per_action=8, max_collision_retries=8)

    def label(seed):
        return action_values(record, 0, LockstepDriver(pool, N_ACTIONS), pool,
                             0, cfg, np.random.default_rng(seed))[0]

    a, b = label(19), label(19)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


# ------------------------------------------------------- 7. prefix independence


def test_the_label_does_not_depend_on_how_the_hand_went_on():
    pool = make_pool()
    record = play(pool, make_specs(seed=52, n_hands=1, n_members=len(pool),
                                   num_players=5, stack_bb=150))[0]
    idx = 1
    assert len(record.decisions) > idx + 1, "need a future to delete"
    cfg = OracleConfig(samples_per_action=8, max_collision_retries=8)

    def label(rec):
        return action_values(rec, idx, LockstepDriver(pool, N_ACTIONS), pool, 0,
                             cfg, np.random.default_rng(23))[0]

    full, cut = label(record), label(truncate_after(record, idx))
    assert np.array_equal(np.isnan(full), np.isnan(cut))
    assert np.array_equal(full[~np.isnan(full)], cut[~np.isnan(cut)])


# ------------------------------------------------------ 8. collision accounting


def test_the_runout_is_dealt_per_sample_but_the_visible_board_is_not():
    """The board splits in two at the decision, and the halves behave oppositely.

    What hero can see is fixed — every rollout has to replay that flop or the
    label is about a different hand. What hero cannot see is drawn again for
    every sample, because §7.1's `Q` averages over everything hero does not
    know and the turn and the river are part of it. Pinning the whole board,
    as an earlier version did, makes the estimate conditional on one runout
    and no sample budget can undo that.

    The dead set is hero's information exactly: no opponent is ever dealt a
    card off the *visible* board or out of hero's hand, and a card of a street
    still to come is fair game — it is in the deck as far as hero knows.
    """
    pool = [Caller("call", N_ACTIONS, StyleParams.identity())]
    record = one_hand(pool, [0, 0], stack_bb=50, seed=31)
    idx = next(i for i, d in enumerate(record.decisions)
               if int(record.snapshots[d["snap_idx"]]["turn"]) == 1)
    hero_pos = int(record.decisions[idx]["acting_pos"])
    opp_pos = 1 - hero_pos

    driver = CapturingDriver(pool, N_ACTIONS)
    q, legal, stats = action_values(
        record, idx, driver, pool, 0,
        OracleConfig(samples_per_action=16, max_collision_retries=32),
        np.random.default_rng(12))

    assert stats.n_rollouts == len(driver.captured) > 0
    assert np.isfinite(q[legal]).all()

    visible = [int(c) for c in record.deck[:3]]
    dead = set(visible) | set(record.hole_cards(hero_pos))
    runouts = set()
    for played in driver.captured:
        assert [int(c) for c in played.deck[:3]] == visible
        assert played.hole_cards(hero_pos) == record.hole_cards(hero_pos)
        assert not set(played.hole_cards(opp_pos)) & dead
        runouts.add(tuple(int(c) for c in played.deck[3:5]))

    assert len(runouts) > 1, "every sample got the same turn and river"


def folded_aces_hand():
    """Seat 2 folds first; its range is exactly the six AA combos."""
    pool = [Caller("hero", N_ACTIONS, StyleParams.identity()),
            Caller("live", N_ACTIONS, StyleParams.identity()),
            FoldsInSet("folder", N_ACTIONS, range(48, 52),
                       StyleParams.identity())]
    deck = deck_with([8, 9, 10, 11, 12], [[4, 5], [6, 7], [48, 49]])
    record = one_hand(pool, [0, 1, 2], deck=deck)
    assert record.decisions[0]["acting_pos"] == 2
    assert record.decisions[0]["action_idx"] == FOLD
    assert record.decisions[1]["acting_pos"] == 0
    return pool, record


@pytest.mark.parametrize("control_variate", [False, True])
def test_fold_likelihood_blocks_live_hands_and_runouts(control_variate):
    """An exact support check, not a frequency estimate: a fold means AA."""
    pool, record = folded_aces_hand()
    pool[0] = Recorder(pool[0])
    driver = CapturingDriver(pool, N_ACTIONS)
    q, legal, stats = action_values(
        record, 1, driver, pool, 0,
        OracleConfig(samples_per_action=16, likelihood_floor=0,
                     control_variate=control_variate), np.random.default_rng(9))
    assert stats.n_rollouts == 16 * legal.sum()
    assert np.isfinite(q[legal]).all()
    assert pool[0].seen == {(4, 5)}, "sampled opponents' cards leaked to hero"
    for played in driver.captured:
        folded = set(played.hole_cards(2))
        assert len(folded) == 2 and folded <= set(range(48, 52))
        assert not folded & set(played.hole_cards(1))
        assert not folded & set(played.deck[:5])
        assert len(set(played.deck)) == 52
        assert played.decisions[0]["action_idx"] == FOLD
        assert played.decisions[0]["acting_pos"] == 2
        assert all(d["acting_pos"] != 2 for d in played.decisions[1:])
    # Every action of a sample is compared on the same complete assignment.
    for offset in range(0, len(driver.captured), int(legal.sum())):
        block = driver.captured[offset:offset + int(legal.sum())]
        assert all(np.array_equal(r.deck, block[0].deck) for r in block)


def test_fold_conditioning_ignores_real_hidden_cards_and_future_history():
    pool, record = folded_aces_hand()
    changed = copy.deepcopy(record)
    unknown = [i for i in range(52) if i not in (5, 6)]  # hero alone is known
    changed.deck[unknown] = changed.deck[unknown][::-1]
    changed.decisions = changed.decisions[:2]
    cfg = OracleConfig(samples_per_action=8, likelihood_floor=0)
    a, b = CapturingDriver(pool, N_ACTIONS), CapturingDriver(pool, N_ACTIONS)
    qa, _, _ = action_values(record, 1, a, pool, 0, cfg, np.random.default_rng(4))
    qb, _, _ = action_values(changed, 1, b, pool, 0, cfg, np.random.default_rng(4))
    np.testing.assert_array_equal(qa, qb)
    assert all(np.array_equal(x.deck, y.deck)
               for x, y in zip(a.captured, b.captured))


def test_cached_fold_range_survives_later_board_reveals():
    pool, record = folded_aces_hand()
    cache = PosteriorCache()
    cfg = OracleConfig(samples_per_action=8, likelihood_floor=1e-6)
    for idx, dec in enumerate(record.decisions):
        if dec["acting_pos"] != 0:
            continue
        a, b = CapturingDriver(pool, N_ACTIONS), CapturingDriver(pool, N_ACTIONS)
        qa, _, sa = action_values(record, idx, a, pool, 0, cfg,
                                  np.random.default_rng(idx))
        qb, _, sb = action_values(record, idx, b, pool, 0, cfg,
                                  np.random.default_rng(idx), cache)
        np.testing.assert_array_equal(qa, qb)
        assert sa.n_rollouts == sb.n_rollouts
        assert all(np.array_equal(x.deck, y.deck)
                   for x, y in zip(a.captured, b.captured))


def test_a_label_whose_every_draw_collides_is_nan():
    """Eight ranges inside three cards: no joint assignment exists at all.

    Sixteen cards are needed and three are available, so the rejection is a
    pigeonhole and not a probability — nothing is rolled out, and `q` is `nan`
    rather than a number nobody computed. `nan` is the contract §6.2's masking
    relies on: a zero here would average silently into a target.
    """
    trio = [card(8, 3), card(7, 3), card(6, 3)]                 # T♣ 9♣ 8♣
    pool = [Caller("hero", N_ACTIONS, StyleParams.identity()),
            CallsInSet("narrow", N_ACTIONS, trio, StyleParams.identity())]
    board = [card(12, 0), card(11, 0), card(5, 0), card(2, 1), card(0, 2)]
    holes = [HERO_ACES] + [[card(r, 1), card(r, 2)] for r in range(3, 11)]
    record = one_hand(pool, [0] + [1] * 8, deck=deck_with(board, holes),
                      forced=[CALL] * 9, stack_bb=100)
    idx = 9
    assert int(record.decisions[idx]["acting_pos"]) == 0
    assert not set(trio) & (set(board[:3]) | set(HERO_ACES))

    q, legal, stats = action_values(
        record, idx, LockstepDriver(pool, N_ACTIONS), pool, 0,
        OracleConfig(samples_per_action=32, likelihood_floor=0.0,
                     max_collision_retries=4),
        np.random.default_rng(4))

    assert stats.collision_rate == 1.0
    assert stats.n_rollouts == 0
    assert np.isnan(q).all() and legal.any()


def test_a_heads_up_river_decision_cannot_collide():
    """One opponent, every board card already dead in the posterior itself.

    The decision *before* hero's has to be on the river too: the posterior is
    conditioned through it, so that is the board it treats as visible, and a
    combo it can still hold is exactly a combo the rollout can deal.
    """
    pool = [Caller("call", N_ACTIONS, StyleParams.identity())]
    record = one_hand(pool, [0, 0], stack_bb=50, seed=31)
    street = [int(record.snapshots[d["snap_idx"]]["turn"])
              for d in record.decisions]
    idx = next(i for i in range(1, len(street))
               if street[i] == 3 and street[i - 1] == 3)

    samples = 24
    q, legal, stats = action_values(
        record, idx, LockstepDriver(pool, N_ACTIONS), pool, 0,
        OracleConfig(samples_per_action=samples, max_collision_retries=0),
        np.random.default_rng(6))

    assert stats.collision_rate == 0.0
    assert stats.n_rollouts == samples * int(legal.sum())
    assert np.isfinite(q[legal]).all()


def test_dropped_samples_reduce_the_divisor_they_are_not_zeros():
    """Nine seats, eight overlapping ranges, and no retries — most draws die.

    What must survive that is the arithmetic: the fold value is the same number
    whatever the opponents hold, so it comes out at its closed form and not
    scaled down by the samples that never ran.
    """
    pool = [Caller("call", N_ACTIONS, StyleParams.identity())]
    record = one_hand(pool, [0] * 9, stack_bb=100, seed=33)
    idx = next(i for i, d in enumerate(record.decisions)
               if int(d["acting_pos"]) == 0)
    assert record.decisions[idx]["legal_mask"][FOLD]

    samples = 512
    q, legal, stats = action_values(
        record, idx, LockstepDriver(pool, N_ACTIONS), pool, 0,
        OracleConfig(samples_per_action=samples, max_collision_retries=0),
        np.random.default_rng(8))

    kept = stats.n_rollouts // int(legal.sum())
    # ~1.5% of draws survive eight independent marginals over one deck, which
    # is the size of the §7.3 approximation at a full ring and the reason
    # `max_collision_retries` is not a formality.
    assert 0 < kept < samples, "the fixture must both keep and drop samples"
    assert stats.collision_rate > 0.9
    assert stats.n_rollouts == kept * int(legal.sum())
    # No retries, so every sample is exactly one attempt.
    assert stats.collision_rate == 1.0 - kept / samples
    np.testing.assert_allclose(q[FOLD], -contribution_bb(record, idx),
                               rtol=0.0, atol=1e-12)


def test_the_forward_count_is_the_posterior_plus_the_free_decisions():
    """`LabelStats` is what G3 reads, so the count is pinned by hand.

    Heads-up against a shove: hero folds or calls all-in, so no rollout has a
    free decision in it and every forward paid for is a posterior row — one
    opponent decision over the 1 225 combos hero cannot rule out.
    """
    pool, record = shove_fold_hand()
    _q, _legal, stats = action_values(
        record, 1, LockstepDriver(pool, N_ACTIONS), pool, 1,
        OracleConfig(samples_per_action=8, likelihood_floor=0.0),
        np.random.default_rng(13))

    assert stats.forwards == 1225
    assert stats.seconds > 0.0
