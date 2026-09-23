"""Played-policy evaluation invariants, including complete 9-max sessions."""

import copy
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch

from agent.policy import FrozenAgentMember
from evaluation import pool_eval
from evaluation.pool_eval import Candidate, EvaluationSession, evaluate, sample_summary
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import StyleParams
from pool.sampling import PoolSampler
from tests.g1_fixtures import N_ACTIONS, MAX_PLAYERS
from tests.test_pipeline import toy_config


def distribution(cfg, pool):
    sampler = PoolSampler(len(pool), cfg["pool_sampling"], np.random.default_rng(0))
    # Nonuniform PFSP, used by both fresh runs and resume tests.
    sampler.update(0, -10, 100)
    sampler.update(1, 10, 100)
    return sampler.distribution()


def setup(players=2, network=False):
    cfg = copy.deepcopy(toy_config())
    cfg["game"]["players_range"] = [players, players]
    cfg["game"]["stack_bb_range"] = [10, 10]
    cfg["pool_evaluation"] = {"enabled": True, "run": "test",
        "benchmarks": ["current", "training"], "cold_blocks": 3,
        "warm_sessions": 2, "warm_hands_per_session": 3,
        "blocks_per_batch": 2, "batch_hands": 128, "runout_samples": 2}
    pool = [DEGENERATE_STRATEGIES["always_call"](f"caller{i}", N_ACTIONS,
                                               StyleParams.identity()) for i in range(9)]
    desc = [{"name": m.name} for m in pool]
    if network:
        net = AgentNet(cfg["embedding_net"], N_ACTIONS, MAX_PLAYERS)
        embed = OpponentEmbeddingNet(cfg["embedding_net"], N_ACTIONS, MAX_PLAYERS,
                                      n_members=9)
        member = FrozenAgentMember("agent", N_ACTIONS, net, MAX_PLAYERS, "cpu", embed_net=embed)
    else:
        member = pool[0]
    candidates = [Candidate(member, cfg["embedding_net"])] * 2
    return cfg, pool, desc, candidates


@pytest.mark.parametrize("players", [2, 3, 6, 9])
def test_duplicate_covers_every_seat_for_every_hand(players):
    for h in range(5):
        sessions = [EvaluationSession(0, players, 50, [-1]*players, rotation=r)
                    for r in range(players)]
        assert sorted(s.seat_of_slot(0, h) for s in sessions) == list(range(players))
        for s in sessions:
            for slot in range(players):
                assert s.slot_of_seat(h)[s.seat_of_slot(slot, h)] == slot


@pytest.mark.parametrize("players", [2, 6, 9])
def test_identical_policy_has_zero_paired_difference_and_correct_unit_count(tmp_path, players):
    cfg, pool, desc, candidates = setup(players)
    report = evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    assert report["total_hands"] == 2 * (3 + 2*3) * (2*players)
    for modes in report["benchmarks"].values():
        for mode, s in modes.items():
            assert s["units"] == (3 if mode == "cold" else 2)
            assert s["unit"] == ("duplicate_block" if mode == "cold" else "complete_session")
            assert s["hands"] == s["units"]*2*players*(1 if mode == "cold" else 3)
            for kind in ("raw", "cv"):
                assert s[kind]["delta"]["bb_per_100"] == 0
                assert s[kind]["delta"]["stderr_bb_per_100"] == 0
            assert s["by_players"][str(players)]["units"] == s["units"]


def test_standard_error_uses_independent_units_and_pair_covariance():
    rows = [{"raw": [a, b], "cv": None, "hands": 200, "seconds": 1}
            for a, b in [(10., 9.), (-10., -9.), (5., 5.), (-5., -5.)]]
    summary = pool_eval.summarise(rows, .95, 10)
    expected_sd = math.sqrt(2/3)
    assert summary["raw"]["delta"]["stderr_bb_per_100"] == pytest.approx(100*expected_sd/2)
    assert summary["units"] == 4
    assert summary["hands"] == 800
    assert summary["raw"]["delta"]["stderr_bb_per_100"] < summary["raw"]["new"]["stderr_bb_per_100"]
    assert sample_summary([1], .95, 10)["ci_bb_per_100"] is None


def test_resume_reuses_completed_units_and_rejects_different_semantics(tmp_path, monkeypatch):
    cfg, pool, desc, candidates = setup(3)
    first = evaluate(pool, desc, candidates, cfg, tmp_path, 2, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    cfg["pool_evaluation"]["blocks_per_batch"] = 1
    cfg["pool_evaluation"]["batch_hands"] = 16
    monkeypatch.setattr(pool_eval, "_play_units", lambda *a: pytest.fail("replayed a complete unit"))
    second = evaluate(pool, desc, candidates, cfg, tmp_path, 2, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    assert first["benchmarks"] == second["benchmarks"]
    changed = copy.deepcopy(cfg)
    changed["pool_evaluation"]["runout_samples"] = 3
    with pytest.raises(ValueError, match="identity changed"):
        evaluate(pool, desc, candidates, changed, tmp_path, 2, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    changed = copy.deepcopy(cfg)
    changed["game"]["stack_bb_range"] = [11, 11]
    with pytest.raises(ValueError, match="identity changed"):
        evaluate(pool, desc, candidates, changed, tmp_path, 2, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))


def test_interrupted_run_and_different_batch_partition_preserve_results(tmp_path, monkeypatch):
    cfg, pool, desc, candidates = setup(3)
    whole = evaluate(pool, desc, candidates, cfg, tmp_path/"whole", 1, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    real = pool_eval._play_units
    calls = []
    def interrupt(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return real(*args)
    monkeypatch.setattr(pool_eval, "_play_units", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        evaluate(pool, desc, candidates, cfg, tmp_path/"partial", 1, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    monkeypatch.setattr(pool_eval, "_play_units", real)
    cfg["pool_evaluation"]["blocks_per_batch"] = 1
    cfg["pool_evaluation"]["batch_hands"] = 32
    resumed = evaluate(pool, desc, candidates, cfg, tmp_path/"partial", 1, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    for benchmark, modes in whole["benchmarks"].items():
        for mode, s in modes.items():
            for kind in ("raw", "cv"):
                assert s[kind] == resumed["benchmarks"][benchmark][mode][kind]


@pytest.mark.parametrize("players", [2, 9])
def test_real_network_warm_fit_uses_each_lanes_past_and_preserves_weights(tmp_path, players, monkeypatch):
    cfg, pool, desc, candidates = setup(players, network=True)
    cfg["pool_evaluation"].update(benchmarks=["current"], cold_blocks=1,
                                warm_sessions=1, warm_hands_per_session=3)
    original = pool_eval._fit
    observed = []
    def inspect(lane, *args):
        h = args[-1]
        assert len(lane.session.records) == h
        assert all(r.spec.meta["eval_lane"] == lane.session.idx for r in lane.session.records)
        observed.append((lane.candidate, lane.session.rotation, id(lane.session.records)))
        return original(lane, *args)
    monkeypatch.setattr(pool_eval, "_fit", inspect)
    before = {k: v.clone() for k, v in candidates[0].member.net.state_dict().items()}
    embed_before = {k: v.clone() for k, v in candidates[0].member.embed_net.state_dict().items()}
    report = evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    assert len({v[2] for v in observed}) == 2*players
    assert {v[0] for v in observed} == {0, 1}
    for key, value in before.items():
        torch.testing.assert_close(value, candidates[0].member.net.state_dict()[key], rtol=0, atol=0)
    for key, value in embed_before.items():
        torch.testing.assert_close(value, candidates[0].member.embed_net.state_dict()[key], rtol=0, atol=0)
    assert report["benchmarks"]["current"]["warm"]["raw"]["delta"]["bb_per_100"] == 0
    with torch.no_grad():
        next(candidates[0].member.net.parameters()).add_(.01)
    with pytest.raises(ValueError, match="identity changed"):
        evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))


def test_configuration_and_plan_use_uniform_general_table_range():
    cfg, pool, _, _ = setup()
    cfg["game"]["players_range"] = [2, 9]
    c = pool_eval.configuration(cfg)
    units = pool_eval._plan(c, cfg["game"], 2, "current", "cold", 100, list(range(9)))
    assert {u["players"] for u in units} == set(range(2, 10))
    assert all(len(set(u["opponents"])) == u["players"] - 1 for u in units)
    cfg["game"]["players_range"] = [2, 10]
    with pytest.raises(ValueError, match="2–9"):
        pool_eval.configuration(cfg)


def test_past_agents_amortise_from_their_own_session_and_slot(tmp_path, monkeypatch):
    cfg, pool, desc, candidates = setup(3, network=True)
    cfg["embedding_net"]["pool_agent_vectors"] = "amortised"
    cfg["pool_evaluation"].update(benchmarks=["current"], cold_blocks=0,
                                warm_sessions=1, warm_hands_per_session=3)
    pool = [candidates[0].member] * len(pool)
    original = pool_eval.amortised_vectors
    seen = []
    def capture(net, session, slot, *args):
        assert len(session.records) == 2
        assert slot in (1, 2)
        seen.append((session.idx, slot))
        return original(net, session, slot, *args)
    monkeypatch.setattr(pool_eval, "amortised_vectors", capture)
    result = evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8, lambda _: None,
                     training_distribution=distribution(cfg, pool))
    assert len(seen) == 2*3*2
    assert len(set(seen)) == len(seen)
    assert result["benchmarks"]["current"]["warm"]["raw"]["delta"]["bb_per_100"] == 0


def test_training_plan_draws_independent_seats_from_saved_weights():
    cfg, _, _, _ = setup(9)
    settings = pool_eval.configuration(cfg)
    units = pool_eval._plan(settings, cfg["game"], 0, "training", "cold", 5,
                            [0, 1], [0., 1.])
    assert all(u["opponents"] == [1]*8 for u in units)
    with pytest.raises(ValueError, match="probabilities"):
        pool_eval._plan(settings, cfg["game"], 0, "training", "cold", 1, [0, 1])
    cfg["pool_evaluation"]["benchmarks"] = ["anchor"]
    with pytest.raises(ValueError, match="anchor was removed"):
        pool_eval.configuration(cfg)


def test_changed_training_weights_cannot_reuse_completed_units(tmp_path):
    cfg, pool, desc, candidates = setup()
    dist = distribution(cfg, pool)
    evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8,
             training_distribution=dist)
    changed = copy.deepcopy(dist)
    changed["probabilities"] = list(reversed(dist["probabilities"]))
    with pytest.raises(ValueError, match="identity changed"):
        evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8,
                 training_distribution=changed)


@pytest.mark.parametrize("all_hands", [False, True])
def test_truncated_hands_are_discarded_in_pairs_and_resume(tmp_path, monkeypatch, all_hands):
    cfg, pool, desc, candidates = setup(3)
    real_run = pool_eval.LockstepDriver.run

    def truncate(driver, *args, **kwargs):
        records = real_run(driver, *args, **kwargs)
        for r in records:
            # Spoil one rotation of one candidate, on hand zero of each unit.
            if all_hands or (r.spec.meta["eval_lane"] % 6 == 0 and r.spec.meta["hand"] == 0):
                r.truncated = True
                r.rewards[:] = 1e12
                r.baseline_rewards[:] = 1e12
        return records

    monkeypatch.setattr(pool_eval.LockstepDriver, "run", truncate)
    report = evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8,
                      training_distribution=distribution(cfg, pool))
    for modes in report["benchmarks"].values():
        cold, warm = modes["cold"], modes["warm"]
        assert cold["units"] == cold["hands"] == 0
        assert cold["raw"] is cold["cv"] is None
        assert cold["discarded_hands"] == cold["attempted_hands"] == 18
        assert warm["hands"] == (0 if all_hands else 24)
        assert warm["discarded_hands"] == (36 if all_hands else 12)
        if not all_hands:
            assert warm["raw"]["delta"]["bb_per_100"] == 0
    monkeypatch.setattr(pool_eval, "_play_units", lambda *a: pytest.fail("replayed discarded units"))
    resumed = evaluate(pool, desc, candidates, cfg, tmp_path, 0, 8,
                       training_distribution=distribution(cfg, pool))
    assert resumed["benchmarks"] == report["benchmarks"]
