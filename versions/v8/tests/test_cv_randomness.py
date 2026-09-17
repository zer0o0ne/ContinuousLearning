"""Regression for frozen auxiliary MC error: exact finite expectation checks."""

from dataclasses import replace

import numpy as np
import pytest

from env.driver import HandSpec, LockstepDriver
from env.runout import HandRunout, RunoutConfig
from oracle.rollout import OracleConfig, _label_plan
from pool.base import PoolMember
from tests.g1_fixtures import N_ACTIONS, RAISE_SIZES


class ExactCaller(PoolMember):
    def logits(self, contexts):
        logits = np.zeros((len(contexts), self.n_actions))
        logits[:, 1] = 1e9
        return logits


def caller():
    return ExactCaller("caller", N_ACTIONS)


@pytest.mark.parametrize("n", [2, 3, 9])
@pytest.mark.parametrize("all_in", [False, True])
def test_exact_auxiliary_average_preserves_ev_including_side_pots(monkeypatch, n, all_in):
    """Enumerate real rivers and a balanced M=16 auxiliary sampler.

    The auxiliary sampler has a finite uniform offset; every entry of its
    cyclic window is uniform over legal rivers. This proves the expectation
    through the production driver without a statistical acceptance interval.
    The 3/9-way all-ins have unequal stacks and real side pots.
    """
    original = HandRunout._job
    def balanced(self, turn):
        known, holes, comps = original(self, turn)
        if int(turn) == 2:
            remaining = np.setdiff1d(np.arange(52), np.concatenate([known, holes.reshape(-1)]))
            comps = remaining[(self.runout_seed + np.arange(self.cfg.samples)) % len(remaining), None]
        return known, holes, comps
    monkeypatch.setattr(HandRunout, "_job", balanced)
    deck = np.random.default_rng(4005).permutation(52)
    credits = [1000.] + [100. + 50*i for i in range(n-1)] if all_in else [1000.]*n
    prefix = [1] * (2*n)
    prefix += [N_ACTIONS-1] + [1]*(n-1) if all_in else [1]
    base = HandSpec(n, credits, [0]*n, 10, 10., 5., RAISE_SIZES,
                    deck=deck, forced_actions=prefix)
    specs = []
    for index in [4] + list(range(5+2*n, 52)):
        variant = deck.copy()
        variant[4], variant[index] = variant[index], variant[4]
        specs.append(replace(base, deck=variant))
    raw = LockstepDriver([caller()], N_ACTIONS).run(specs)
    expected = np.mean([r.rewards for r in raw], axis=0)
    sampled = [replace(s, runout_seed=u) for s in specs for u in range(len(specs))]
    corrected = LockstepDriver([caller()], N_ACTIONS, RunoutConfig(16)).run(sampled, batch_size=512)
    np.testing.assert_allclose(np.mean([r.baseline_rewards for r in corrected], axis=0),
                               expected, rtol=0, atol=1e-8)
    assert all(r.spec.runout_seed is not None for r in corrected)


def test_cache_separates_auxiliary_samples_but_shares_action_alternatives():
    deck = np.random.default_rng(5).permutation(52)
    cache = {}
    a = HandRunout(deck, 2, RunoutConfig(16), cache, runout_seed=11)
    b = HandRunout(deck, 2, RunoutConfig(16), cache, runout_seed=11)
    c = HandRunout(deck, 2, RunoutConfig(16), cache, runout_seed=12)
    for turn in (0, 1, 2):
        assert a.scores(turn) is b.scores(turn)
        assert a.scores(turn) is not c.scores(turn)
        np.testing.assert_array_equal(a._job(turn)[2], b._job(turn)[2])
    assert len(cache) == 6


def test_oracle_refreshes_all_streams_and_shares_auxiliary_seed_within_sample():
    pool = [caller()]
    spec = HandSpec(2, [1000., 1000.], [0, 0], 7, 10., 5., RAISE_SIZES)
    record = LockstepDriver(pool, N_ACTIONS).run([spec])[0]
    cfg = OracleConfig(samples_per_action=16, max_combos=8)
    driver = LockstepDriver(pool, N_ACTIONS, cfg.runout_config())
    one = _label_plan(record, 0, driver, pool, 0, cfg, np.random.default_rng(7))
    again = _label_plan(record, 0, driver, pool, 0, cfg, np.random.default_rng(7))
    other = _label_plan(record, 0, driver, pool, 0, cfg, np.random.default_rng(8))
    width = len(one.legal_idx)
    assert len({s.runout_seed for s in one.specs}) == 16
    for i in range(0, len(one.specs), width):
        assert len({s.runout_seed for s in one.specs[i:i+width]}) == 1
        assert len({s.deck.tobytes() for s in one.specs[i:i+width]}) == 1
    assert [s.runout_seed for s in one.specs] == [s.runout_seed for s in again.specs]
    assert [s.seed for s in one.specs] == [s.seed for s in again.specs]
    assert [s.runout_seed for s in one.specs] != [s.runout_seed for s in other.specs]
    assert [s.seed for s in one.specs] != [s.seed for s in other.specs]


def test_auxiliary_seed_does_not_read_future_board_or_stub():
    deck = np.random.default_rng(9).permutation(52)
    for turn, known in ((0, 0), (1, 3), (2, 4)):
        unseen = list(range(known, 5)) + list(range(9, 52))
        other = deck.copy()
        other[unseen] = other[unseen][::-1]
        a = HandRunout(deck, 2, RunoutConfig(16), runout_seed=55)
        b = HandRunout(other, 2, RunoutConfig(16), runout_seed=55)
        np.testing.assert_array_equal(a._job(turn)[2], b._job(turn)[2])
