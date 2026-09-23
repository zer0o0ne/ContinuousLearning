"""Pool sampling (CONCEPT.md §4.4, `PLAN_PIPELINE.md` S8).

Three mechanisms share every draw and each of them can be checked by what comes
out of `sample_table`, so that is where the assertions are:

* PFSP concentrates the draw on the members hero loses to;
* the uniform floor is the only route by which the member hero beats the most is
  ever seen again;
* clustering by the embedding table stops a blob of near-duplicates from
  crowding out a lone style (§11.3).

Every sequence below is pinned exactly under a fixed seed. Nothing here asserts
a statistical property (`CLAUDE.md` §4).
"""

import numpy as np
import pytest

from pool.sampling import PoolSampler

CFG = {"pfsp_exponent": 2.0, "floor_fraction": 0.2, "n_clusters": 3,
       "result_decay": 1.0}


def test_seat_probabilities_include_cluster_sizes_zero_weights_and_floor():
    s = PoolSampler(4, CFG, np.random.default_rng(0))
    # Three equal hard opponents in one cluster; one easy singleton.
    s.set_vectors(np.array([[0.], [0.], [0.], [10.]]))
    for i, score in enumerate([-10., -10., -10., 10.]):
        s.update(i, score, 100)
    before = s.state_dict()
    np.testing.assert_allclose(s.probabilities(), [.05 + .8/3]*3 + [.05])
    assert s.distribution()["hero_bb_per_100"] == [-10., -10., -10., 10.]
    assert s.state_dict() == before

    fresh = PoolSampler(4, CFG, np.random.default_rng(0))
    fresh.set_vectors(np.array([[0.], [0.], [0.], [10.]]))
    np.testing.assert_allclose(fresh.probabilities(), [.05 + .4/3]*3 + [.45])
    assert fresh.distribution()["hero_bb_per_100"] == [None]*4


def _sampler(cfg=CFG, n_members=6, seed=7):
    """A pool hero loses 5 BB/100 to member 0, 1 BB/100 to member 1, and beats
    every other member by 10 BB/100.

    Member 0's result arrives in two sessions of 100 hands rather than one of
    200, so the accumulation across sessions is exercised by every test below.
    """
    s = PoolSampler(n_members, cfg, np.random.default_rng(seed))
    s.update(0, -5.0, 100)
    s.update(0, -5.0, 100)
    s.update(1, -2.0, 200)
    for m in range(2, n_members):
        s.update(m, 20.0, 200)
    return s


def test_a_fixed_history_and_seed_produce_an_exact_sequence():
    """1 — the whole point of owning the rng: a run is reproducible."""
    drawn = _sampler().sample_table(40)
    assert drawn == [
        1, 0, 2, 0, 0, 0, 0, 0, 4, 0,
        1, 0, 3, 5, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 4, 1, 0,
        0, 0, 0, 5, 0, 0, 5, 2, 0, 0,
    ]
    assert _sampler().sample_table(40) == drawn


def test_results_accumulate_into_a_mean_and_the_mean_into_a_weight():
    """The bridge from reported BB/100 to PFSP's [0, 1] slot, on the numbers.

    Hero is at −5 BB/100 against member 0 (two sessions of 100 hands, so the
    accumulation is doing real work), −1 against member 1 and +10 against the
    rest. Min-max over [−5, +10] puts member 0 at 1, member 1 at 11/15 and every
    member hero beats at exactly 0 — reachable only through the floor.
    """
    s = _sampler()
    assert s.mean_bb_per_100().tolist() == [-5.0, -1.0, 10.0, 10.0, 10.0, 10.0]
    assert s.hardness().tolist() == pytest.approx(
        [1.0, 11.0 / 15.0, 0.0, 0.0, 0.0, 0.0])
    assert s.weights().tolist() == pytest.approx(
        [1.0, (11.0 / 15.0) ** 2, 0.0, 0.0, 0.0, 0.0])


def test_forgetting_leaves_an_unsampled_member_where_it_was():
    """`end_iteration` ages the evidence without moving the estimate.

    Numerator and denominator decay together, so a member nobody played this
    iteration keeps exactly the mean it had — no new information, no new
    estimate. What shrinks is the effective hand count behind it.
    """
    s = _sampler(dict(CFG, result_decay=0.5))
    before = s.mean_bb_per_100().tolist()
    for _ in range(10):
        s.end_iteration()
    assert s.mean_bb_per_100().tolist() == pytest.approx(before)
    assert s.hardness().tolist() == pytest.approx(_sampler().hardness().tolist())


def test_forgetting_is_what_lets_the_loop_notice_it_has_stopped_winning():
    """The recovery path, on the numbers (module docstring).

    A member the early agents crushed over 20 000 hands, against which the
    current agent now loses 10 BB/100 in 330-hand sessions — one iteration's
    worth of what the uniform floor delivers at the scale §13 sketches.

    Without forgetting the estimate is still +7.2 BB/100 after ten iterations
    and reaches zero only around the sixtieth: the loop does not notice inside
    the length of a run. At 0.8 it crosses at the twelfth
    iteration, which is also where the member takes the top PFSP weight; at 0.5
    it crosses at the fifth. The transient is the old evidence decaying to the window's scale, so
    it is set by the decay and not by how much history there was.
    """
    def run(decay, n_iterations):
        s = PoolSampler(2, dict(CFG, result_decay=decay), np.random.default_rng(0))
        s.update(0, 2000.0, 20000)      # +10 BB/100 over 20 000 hands
        s.update(1, 0.0, 20000)
        for _ in range(n_iterations):
            s.end_iteration()
            s.update(0, -33.0, 330)     # −10 BB/100 over one floor session
            s.update(1, 0.0, 330)
        return s

    assert run(1.0, 10).mean_bb_per_100()[0] == pytest.approx(7.167, abs=0.001)
    assert run(1.0, 40).mean_bb_per_100()[0] > 0.0

    assert run(0.8, 10).mean_bb_per_100()[0] == pytest.approx(1.864, abs=0.001)
    assert run(0.8, 11).mean_bb_per_100()[0] > 0.0
    assert run(0.8, 12).mean_bb_per_100()[0] < 0.0
    assert run(0.8, 11).hardness().tolist() == [0.0, 1.0]
    assert run(0.8, 12).hardness().tolist() == [1.0, 0.0]

    assert run(0.5, 4).mean_bb_per_100()[0] > 0.0
    assert run(0.5, 5).mean_bb_per_100()[0] < 0.0


def test_a_pool_with_no_results_and_a_pool_with_no_spread_are_both_flat():
    """The two degenerate histories, both of which must stay samplable."""
    fresh = PoolSampler(4, CFG, np.random.default_rng(0))
    assert fresh.hardness().tolist() == [1.0, 1.0, 1.0, 1.0]

    flat = PoolSampler(4, CFG, np.random.default_rng(0))
    for m in range(4):
        flat.update(m, 3.0, 100)
    assert flat.hardness().tolist() == [1.0, 1.0, 1.0, 1.0]
    assert len(flat.sample_table(4)) == 4


def test_the_magnitude_of_a_loss_is_what_moves_the_weight():
    """Why the mean and not a count of losing sessions (module docstring).

    Two members hero loses to, one by a hair and one catastrophically. Counting
    negative sessions would make them identical; here the catastrophic one
    carries roughly thirty-five times the weight.
    """
    s = PoolSampler(3, CFG, np.random.default_rng(0))
    s.update(0, -0.2, 100)     # −0.2 BB/100
    s.update(1, -50.0, 100)    # −50 BB/100
    s.update(2, 10.0, 100)     # +10 BB/100
    w = s.weights()
    assert w.tolist() == pytest.approx([(10.2 / 60.0) ** 2, 1.0, 0.0])
    assert w[1] / w[0] == pytest.approx(34.6, abs=0.1)


def test_a_member_hero_beats_the_most_is_reached_only_through_the_floor():
    """2 — the floor is what keeps a zero-weight member in the pool at all.

    Members 2–5 carry a PFSP weight of exactly zero, so every appearance of one
    of them is a floor draw. Over 400 draws at `floor_fraction = 0.2` the floor
    fires about 80 times, and 4 in 6 of those land on a zero-weight member; under
    this seed that is 53, pinned exactly.
    """
    drawn = _sampler().sample_table(400)
    n_zero_weight = sum(1 for m in drawn if m >= 2)
    assert n_zero_weight == 53
    assert sorted(set(drawn)) == [0, 1, 2, 3, 4, 5]


def test_without_a_floor_a_beaten_member_is_never_drawn_again():
    """The floor is the whole mechanism, so switching it off must be visible."""
    cfg = dict(CFG, floor_fraction=0.0)
    drawn = _sampler(cfg).sample_table(200)
    assert set(drawn) == {0, 1}


def test_a_floor_of_one_draws_uniformly_over_the_whole_pool():
    """The other end of the floor: every draw ignores PFSP entirely."""
    cfg = dict(CFG, floor_fraction=1.0)
    drawn = _sampler(cfg).sample_table(200)
    rng = np.random.default_rng(7)
    expected = [(rng.random(), int(rng.integers(6)))[1] for _ in range(200)]
    assert drawn == expected


def test_a_member_nobody_has_played_is_drawn_immediately():
    """3 — a freshly appended agent must not wait for a result to exist.

    Member 5 has no results at all, so its hardness is the maximum, 1.0 — the
    same weight as the member hero loses to every time.
    """
    s = PoolSampler(6, CFG, np.random.default_rng(7))
    s.update(0, -5.0, 100)
    s.update(0, -5.0, 100)
    s.update(1, -2.0, 200)
    for m in (2, 3, 4):
        s.update(m, 20.0, 200)
    assert s.hardness().tolist() == [1.0, 11.0 / 15.0, 0.0, 0.0, 0.0, 1.0]
    drawn = s.sample_table(6)
    assert drawn == [5, 0, 2, 1, 0, 1]
    assert 5 in drawn[:6]


def test_clustering_recovers_a_hand_built_structure():
    """4 — the dedup step, on a matrix whose structure is obvious by eye.

    Ten near-duplicates of one lineage, three of another, and one lone style.
    With `n_clusters = 3` the clustering must put the ten together, the three
    together, and leave the lone vector alone.
    """
    vectors = np.zeros((14, 2))
    vectors[:10] = [10.0, 0.0] + np.arange(10)[:, None] * 1e-3
    vectors[10:13] = [0.0, 10.0] + np.arange(3)[:, None] * 1e-3
    vectors[13] = [-10.0, -10.0]

    s = PoolSampler(14, dict(CFG, n_clusters=3), np.random.default_rng(1))
    s.set_vectors(vectors)
    labels = s.state_dict()["cluster_of"]
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:13])) == 1
    assert len({labels[0], labels[10], labels[13]}) == 3


def test_a_blob_of_near_duplicates_does_not_crowd_out_a_lone_style():
    """4 — what the clustering is *for* (§11.3), read off the draws.

    Every member here is unplayed, so PFSP is flat and the only thing shaping
    the draw is the cluster structure: the lone member should get a cluster's
    share, not a member's share. Without clustering it would get 1 in 14.
    """
    vectors = np.zeros((14, 2))
    vectors[:10] = [10.0, 0.0] + np.arange(10)[:, None] * 1e-3
    vectors[10:13] = [0.0, 10.0] + np.arange(3)[:, None] * 1e-3
    vectors[13] = [-10.0, -10.0]

    cfg = dict(CFG, n_clusters=3, floor_fraction=0.0)
    s = PoolSampler(14, cfg, np.random.default_rng(3))
    s.set_vectors(vectors)
    drawn = s.sample_table(600)
    counts = np.bincount(drawn, minlength=14)

    assert int(counts[13]) == 228
    assert int(counts[:10].sum()) == 197
    assert int(counts[10:13].sum()) == 175
    # The lone style outdraws the entire ten-member blob, which is the whole
    # point; flat sampling would have given it 600 / 14 ≈ 43.
    assert counts[13] > counts[:10].max() * 5
    assert int(counts[:10].max()) == 26


def test_two_identical_vectors_share_one_cluster_share():
    """4 — the pathological case: the pool is one style plus a duplicate pair.

    The pair sits in one cluster and therefore splits one cluster's probability
    between them, instead of taking one full share each.
    """
    vectors = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
    cfg = dict(CFG, n_clusters=2, floor_fraction=0.0)
    s = PoolSampler(3, cfg, np.random.default_rng(11))
    s.set_vectors(vectors)
    counts = np.bincount(s.sample_table(400), minlength=3)
    assert counts.tolist() == [217, 94, 89]
    assert int(counts[1] + counts[2]) == 400 - int(counts[0])


def test_more_clusters_than_members_is_no_clustering_at_all():
    """The cold-start shape, and the guard against an empty-cluster crash."""
    vectors = np.arange(8, dtype=np.float64)[:, None]
    s = PoolSampler(4, dict(CFG, n_clusters=99), np.random.default_rng(5))
    s.set_vectors(vectors[:4])
    assert s.state_dict()["cluster_of"] == [0, 1, 2, 3]
    assert len(s.sample_table(4)) == 4


def test_the_state_round_trips_and_reproduces_the_next_draw():
    """5 — a pipeline restart resumes the same stream (§8, resume)."""
    s = _sampler()
    s.sample_table(11)
    state = s.state_dict()
    expected = s.sample_table(9)

    restored = PoolSampler(6, CFG, np.random.default_rng(999))
    restored.load_state_dict(state)
    assert restored.sample_table(9) == expected
    assert restored.state_dict() == s.state_dict()


def test_a_restored_sampler_accepts_a_pool_that_has_since_grown():
    """§8 appends one member per iteration, so the state is always a prefix."""
    s = _sampler()
    s.sample_table(11)
    state = s.state_dict()

    grown = PoolSampler(7, CFG, np.random.default_rng(999))
    grown.load_state_dict(state)
    assert grown.mean_bb_per_100()[:6].tolist() == [-5.0, -1.0, 10.0, 10.0, 10.0, 10.0]
    assert np.isnan(grown.mean_bb_per_100()[6])
    assert grown.hardness()[6] == 1.0
    assert grown.state_dict()["cluster_of"] == [0, 1, 2, 3, 4, 5, 6]
    assert 6 in grown.sample_table(8)

    with pytest.raises(AssertionError):
        PoolSampler(5, CFG, np.random.default_rng(999)).load_state_dict(state)


def test_every_table_size_gets_one_member_per_non_hero_seat():
    """6 — hero is slot 0 and is not drawn from the pool (`env/session.py`)."""
    s = _sampler()
    for num_players in range(2, 10):
        seats = s.sample_table(num_players - 1)
        assert len(seats) == num_players - 1
        assert all(0 <= m < 6 for m in seats)
    assert s.sample_table(0) == []
