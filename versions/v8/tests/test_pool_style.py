"""The opponent pool and its live style modifiers (CONCEPT.md §4.1, §4.2).

The style modifier is the whole answer to "how do we expand the opponent space
cheaply" (§4.2) and to the one-dimensional-pool risk (§11.3), so what it does to
a distribution has to be exactly what §4.2 says:

    p = (1 − λ) · softmax( (logits + b(s)) / T )  +  λ · uniform_over_legal

Every member, degenerate or network, goes through the same last mile, so these
tests double as the contract every future pool member must satisfy.
"""

import numpy as np
import pytest

from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import (
    N_STYLE_SCALARS, StyleParams, action_categories, category_matrix,
    position_bucket, sample_style,
)
from tests.g1_fixtures import (
    N_ACTIONS, STYLE_CFG, contexts_from, make_pool, make_specs, play,
)


def _contexts():
    pool = make_pool()
    return contexts_from(play(pool, make_specs(seed=21, n_hands=25,
                                               n_members=len(pool))))


# ------------------------------------------------------------------ categories


def test_the_five_categories_partition_the_action_set():
    cats = action_categories(N_ACTIONS)
    assert len(cats) == 5
    flat = [a for c in cats for a in c]
    assert sorted(flat) == list(range(N_ACTIONS))
    assert len(flat) == len(set(flat)), "categories overlap"
    assert cats[0] == [0] and cats[1] == [1]
    assert cats[4] == [N_ACTIONS - 1], "all-in is its own category"


def test_the_broadcast_matrix_matches_the_categories():
    m = category_matrix(N_ACTIONS)
    assert m.shape == (5, N_ACTIONS)
    assert (m.sum(axis=0) == 1).all(), "every action belongs to exactly one block"


# ---------------------------------------------------------------- style vector


def test_a_style_is_thirty_two_scalars_and_round_trips():
    rng = np.random.default_rng(0)
    style = sample_style(rng, STYLE_CFG)
    flat = style.to_list()
    assert len(flat) == N_STYLE_SCALARS == 32
    again = StyleParams.from_list(flat)
    assert np.allclose(again.uncond, style.uncond)
    assert np.allclose(again.position, style.position)
    assert np.allclose(again.street, style.street)
    assert again.temperature == style.temperature
    assert again.uniform_mix == style.uniform_mix


def test_an_explicit_style_vector_of_the_wrong_length_is_refused():
    with pytest.raises(AssertionError):
        StyleParams.from_list([0.0] * 31)


def test_the_identity_style_is_a_plain_masked_softmax():
    contexts = _contexts()[:32]
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(len(contexts), N_ACTIONS))
    legal = np.stack([c.legal_mask for c in contexts])

    p = StyleParams.identity().apply(logits, legal, contexts)

    z = np.where(legal, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    expected = np.exp(z)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(p, expected)


def test_the_position_block_applies_only_in_the_late_bucket():
    contexts = _contexts()
    early = [c for c in contexts if position_bucket(c.acting_pos,
                                                    c.num_players) == 0]
    late = [c for c in contexts if position_bucket(c.acting_pos,
                                                   c.num_players) == 1]
    assert early and late

    style = StyleParams.identity()
    style.position = np.array([0.0, 0.0, 0.0, 0.0, 5.0])   # shove in position

    for group, expect_shift in ((early, False), (late, True)):
        logits = np.zeros((len(group), N_ACTIONS))
        legal = np.stack([c.legal_mask for c in group])
        p_id = StyleParams.identity().apply(logits, legal, group)
        p_st = style.apply(logits, legal, group)
        shove = legal[:, N_ACTIONS - 1]
        if not shove.any():
            continue
        moved = (p_st[shove, N_ACTIONS - 1] > p_id[shove, N_ACTIONS - 1] + 1e-9)
        assert moved.all() == expect_shift


def test_the_street_block_applies_only_on_its_street():
    contexts = _contexts()
    style = StyleParams.identity()
    style.street[0] = np.array([5.0, 0.0, 0.0, 0.0, 0.0])   # fold preflop

    logits = np.zeros((len(contexts), N_ACTIONS))
    legal = np.stack([c.legal_mask for c in contexts])
    p_id = StyleParams.identity().apply(logits, legal, contexts)
    p_st = style.apply(logits, legal, contexts)

    for i, c in enumerate(contexts):
        if not c.legal_mask[0]:
            continue
        if c.turn == 0:
            assert p_st[i, 0] > p_id[i, 0] + 1e-9
        else:
            assert p_st[i, 0] == pytest.approx(p_id[i, 0])


def test_temperature_sharpens_and_flattens():
    contexts = _contexts()[:32]
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(len(contexts), N_ACTIONS)) * 2.0
    legal = np.stack([c.legal_mask for c in contexts])

    def entropy(style):
        p = style.apply(logits, legal, contexts)
        return float(np.mean([-(row[row > 0] * np.log(row[row > 0])).sum()
                              for row in p]))

    hot, base, cold = (StyleParams.identity() for _ in range(3))
    hot.temperature, cold.temperature = 4.0, 0.25
    assert entropy(cold) < entropy(base) < entropy(hot)


def test_the_uniform_mix_moves_toward_uniform_over_legal_actions():
    contexts = _contexts()[:32]
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(len(contexts), N_ACTIONS)) * 3.0
    legal = np.stack([c.legal_mask for c in contexts])

    style = StyleParams.identity()
    style.uniform_mix = 1.0
    p = style.apply(logits, legal, contexts)

    uniform = legal / legal.sum(axis=1, keepdims=True)
    assert np.allclose(p, uniform)
    assert np.allclose(p[~legal], 0.0)


def test_every_style_draw_yields_a_valid_distribution_over_legal_actions():
    contexts = _contexts()
    legal = np.stack([c.legal_mask for c in contexts])
    rng = np.random.default_rng(4)
    for _ in range(20):
        style = sample_style(rng, STYLE_CFG)
        logits = rng.normal(size=(len(contexts), N_ACTIONS)) * 2.0
        p = style.apply(logits, legal, contexts)
        assert np.all(p >= 0.0)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert np.allclose(p[~legal], 0.0), (
            "a style draw put mass on an illegal action")


# --------------------------------------------------------- degenerate members


def test_always_fold_folds_when_facing_a_bet_and_checks_when_free():
    contexts = _contexts()
    member = DEGENERATE_STRATEGIES["always_fold"]("f", N_ACTIONS)
    p = member.policy(contexts)
    seen_free = seen_facing = False
    for i, c in enumerate(contexts):
        if c.legal_mask[0]:
            seen_facing = True
            assert int(np.argmax(p[i])) == 0
        else:
            seen_free = True
            assert int(np.argmax(p[i])) == 1
    assert seen_free and seen_facing


def test_always_call_always_calls():
    contexts = _contexts()
    p = DEGENERATE_STRATEGIES["always_call"]("c", N_ACTIONS).policy(contexts)
    assert (p.argmax(axis=1) == 1).all()


def test_always_min_raise_takes_the_smallest_playable_sized_raise():
    contexts = _contexts()
    p = DEGENERATE_STRATEGIES["always_min_raise"]("r", N_ACTIONS).policy(contexts)
    seen_raise = False
    for i, c in enumerate(contexts):
        sized = [a for a in np.flatnonzero(c.legal_mask)
                 if 2 <= a < N_ACTIONS - 1]
        if sized:
            seen_raise = True
            assert int(np.argmax(p[i])) == sized[0]
        else:
            assert int(np.argmax(p[i])) == 1
    assert seen_raise


def test_the_maniac_puts_its_mass_on_aggression():
    contexts = _contexts()
    p = DEGENERATE_STRATEGIES["maniac"]("m", N_ACTIONS).policy(contexts)
    for i, c in enumerate(contexts):
        if c.legal_mask[N_ACTIONS - 1]:
            assert int(np.argmax(p[i])) == N_ACTIONS - 1
        assert p[i, 0] < p[i, 1:].sum()


def test_the_nit_folds_without_a_strong_hand():
    contexts = _contexts()
    p = DEGENERATE_STRATEGIES["nit"]("n", N_ACTIONS).policy(contexts)
    seen_strong = seen_weak = False
    for i, c in enumerate(contexts):
        r = sorted(card // 4 for card in c.hole_cards)
        strong = (r[0] == r[1] and r[1] >= 8) or r[0] >= 9
        if strong:
            seen_strong = True
            assert int(np.argmax(p[i])) == 1
        elif c.legal_mask[0]:
            seen_weak = True
            assert int(np.argmax(p[i])) == 0
    assert seen_strong and seen_weak


def test_a_style_sibling_shares_the_base_policy_and_changes_only_the_style():
    contexts = _contexts()[:16]
    base = DEGENERATE_STRATEGIES["maniac"]("m", N_ACTIONS)
    rng = np.random.default_rng(5)
    sibling = base.with_style("m#1", sample_style(rng, STYLE_CFG))

    assert np.allclose(base.logits(contexts), sibling.logits(contexts)), (
        "with_style must not touch the base policy")
    assert not np.allclose(base.policy(contexts), sibling.policy(contexts))
    assert base.name == "m" and base.style.is_identity


def test_the_pool_holds_more_members_than_the_widest_table():
    """A 9-handed session needs 9 distinct members (CLAUDE.md §1: table size is
    sampled over the full 2–9 range and never narrowed to fit the pool)."""
    assert len(make_pool()) >= 9
