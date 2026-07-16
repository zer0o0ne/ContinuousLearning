"""E2E tests for the phase-6 robustness pool extensions (2026-07-16):

1. `mcts_train.n_random_agents` — weightless uniform-random NON-trainable
   opponents (`random_0..N-1`) joining the free-seat pool alongside past
   snapshots, independent of `past_opponents.enabled`, sequential mode only.
   They act by uniform sampling among legal actions (no ASI, no MCTS tree),
   produce no training examples, and contribute uniform-over-legal chain
   fallback targets for active heroes.
   Source: agent/mcts/collect.py (run_mcts_collection, _play_hands).

2. `mcts_train.n_blank_agents` — randomly-initialized TRAINABLE agents
   (`blank_0..N-1`) joining the cyclic-MCTS pool as equals of the agents
   that passed phases 1-5. Fresh blanks copy event-norm stats from the
   first trained agent; `mcts_value_scale` is NOT copied (bootstraps
   per-agent on the first collection).
   Source: pipeline.py (_blank_agent_cfgs, _copy_event_norm, registry loop).

All tests are fully deterministic: fixed seeds, no probabilistic
assertions, no order dependence.

Run from versions/v7/:
    python -m pytest tests/test_blank_random_agents.py -v
"""

import copy
import inspect
import os
import random
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
for _p in (_PKG_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.agent import ASI
from agent.mcts.collect import run_mcts_collection, _play_hands

MAX_PLAYERS = 3
N_ACTIONS = 2 + 3  # 2 raise bins + fold/call/all-in

_TINY_CONFIG = {
    "architecture": {
        "d_model": 32,
        "n_heads": 2,
        "n_kv_heads": 1,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 128,
        "max_players": MAX_PLAYERS,
        "modelling_dropout": 0.0,
        "memory": {
            "n_levels": 1,
            "max_cluster_size": 4,
            "max_cluster_size_after": 4,
            "beam_width": 2,
        },
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": [0.5, 1.0],
            "flop": [0.5, 1.0],
            "turn": [0.5, 1.0],
            "river": [0.5, 1.0],
        },
        "max_players": MAX_PLAYERS,
        "big_blind": 10,
        "max_stack": 200,
    },
    "solver": {"type": "v1"},
    "mcts": {
        "n_simulations": 8,
        "batch_size": 4,
        "n_equity_iters": 64,
        "temperature": 1.0,
    },
    "mcts_train": {
        "n_workers": 1,
        "min_players": 2,
        "max_players": 2,
        "player_swap_prob": 0.5,
        "min_stack": 100,
        "max_stack": 200,
        "n_terminal_values": 2,
        "value_target_alpha": 0.5,
        "value_target_clip": 5.0,
        "n_random_agents": 0,  # overridden per test
    },
}

_IDENTITY_NORM = {
    "pot_mean": 0.0, "pot_std": 1.0,
    "stack_mean": 0.0, "stack_std": 1.0,
    "bets_mean": 0.0, "bets_std": 1.0,
    "blind_mean": 0.0, "blind_std": 1.0,
    "ev_mean": 0.0, "ev_std": 1.0,
}

# 42 matches the conftest autouse seed; verified to seat random opponents
# whose decisions land in hero chains for the assertions below.
_SEED = 42


def _reseed():
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)


def _make_agent(seed):
    torch.manual_seed(seed)
    agent = ASI(lambda m: None, config=_TINY_CONFIG)
    agent.set_device("cpu")
    agent.eval()
    return agent


def _agents_list(names_seeds):
    return [{"agent": _make_agent(seed), "norm_stats": dict(_IDENTITY_NORM),
             "name": name, "temperature": 1.0}
            for name, seed in names_seeds]


def _collect(n_random_agents, n_hands=4, names_seeds=(("hero", 0),)):
    """Deterministic sequential collection with `n_random_agents` randoms."""
    cfg = copy.deepcopy(_TINY_CONFIG)
    cfg["mcts_train"]["n_random_agents"] = n_random_agents
    _reseed()
    agents = _agents_list(names_seeds)
    _reseed()
    result = run_mcts_collection(
        agents, cfg, "cpu", lambda *a, **k: None, n_hands)
    return result, agents


def _uniform_over_legal_steps(per_agent_examples):
    """Chain steps that carry a uniform-over-STRICT-SUBSET distribution with
    target_valid=True and is_hero=False — the signature of a random
    opponent's fallback target (forced active steps are target_valid=False
    and uniform over ALL actions; tree-backed uniform-over-support targets of
    active opponents exist only alongside n_simulations visit noise)."""
    steps = []
    for exs in per_agent_examples.values():
        for ex in exs:
            for step in ex.chain:
                dist = list(step.target_distribution)
                support = [p for p in dist if p > 0.0]
                if (step.target_valid and not step.is_hero
                        and 0 < len(support) < len(dist)
                        and all(p == pytest.approx(support[0])
                                for p in support)):
                    steps.append(step)
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random agents: e2e collection scenario.
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomAgentsCollection:

    def test_examples_only_for_active_agents(self):
        """Random agents never produce training examples: the result dict has
        exactly the active agents' keys."""
        result, _ = self._cached()
        assert set(result.keys()) == {"hero"}

    def test_random_opponent_decisions_yield_uniform_chain_targets(self):
        """At least one hand seats a random opponent (deterministic under the
        fixed seed) whose decision becomes a valid uniform-over-legal chain
        fallback target for the hero."""
        result, _ = self._cached()
        steps = _uniform_over_legal_steps(result)
        assert len(steps) > 0
        for step in steps:
            support = [p for p in step.target_distribution if p > 0.0]
            assert sum(step.target_distribution) == pytest.approx(1.0)
            assert all(p == pytest.approx(1.0 / len(support))
                       for p in support)

    def test_value_scale_bootstraps_for_active_agent(self):
        """`_finalize_value_targets` bootstraps mcts_value_scale for the
        active agent even with randoms at the table."""
        result, agents = self._cached()
        assert len(result["hero"]) > 0
        assert "mcts_value_scale" in agents[0]["norm_stats"]
        assert agents[0]["norm_stats"]["mcts_value_scale"] > 0.0

    def test_bitwise_deterministic(self):
        """Same seed → identical examples (counts, targets, chains)."""
        res_a, _ = _collect(n_random_agents=6)
        res_b, _ = _collect(n_random_agents=6)
        assert set(res_a.keys()) == set(res_b.keys())
        for name in res_a:
            assert len(res_a[name]) == len(res_b[name])
            for ex_a, ex_b in zip(res_a[name], res_b[name]):
                assert ex_a.value_target == ex_b.value_target
                assert ex_a.action_target == ex_b.action_target
                assert len(ex_a.chain) == len(ex_b.chain)
                for st_a, st_b in zip(ex_a.chain, ex_b.chain):
                    assert st_a.action_taken == st_b.action_taken
                    assert st_a.target_distribution == st_b.target_distribution
                    assert st_a.target_valid == st_b.target_valid
                    assert st_a.value_target == st_b.value_target

    def test_zero_random_agents_is_noop(self):
        """n_random_agents=0 keeps the active-only seating path."""
        result, _ = _collect(n_random_agents=0, n_hands=2)
        assert set(result.keys()) == {"hero"}

    _CACHE = None

    @classmethod
    def _cached(cls):
        # One shared deterministic run for the read-only assertions (the
        # determinism test does its own paired runs).
        if cls._CACHE is None:
            cls._CACHE = _collect(n_random_agents=6)
        return cls._CACHE


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random agents: sequential-only guard + seating plumbing.
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomAgentsPlumbing:

    def test_parallel_mode_guard_present(self):
        """n_random_agents is built only on the sequential path; parallel
        mode logs a warning and disables it (inference-server spec is
        frozen at startup — same restriction as past snapshots)."""
        src = inspect.getsource(run_mcts_collection)
        assert "n_random_agents" in src
        assert "disabled" in src

    def test_play_hands_accepts_random_agent_infos(self):
        sig = inspect.signature(_play_hands)
        assert "random_agent_infos" in sig.parameters
        assert sig.parameters["random_agent_infos"].default is None

    def test_random_infos_join_free_seat_pool_without_materialization(self):
        """Random infos ride the past pool: no ckpt_path → seated directly
        by _reseat_with_past, no _materialize_past call."""
        src = inspect.getsource(_play_hands)
        assert "list(random_agent_infos or [])" in src


# ─────────────────────────────────────────────────────────────────────────────
# 3. Blank agents: config stubs + norm copy + a blank agent plays and trains
#    end to end in collection.
# ─────────────────────────────────────────────────────────────────────────────

class TestBlankAgents:

    def test_blank_agent_cfgs(self):
        import pipeline
        cfgs = pipeline._blank_agent_cfgs(2)
        assert [c["name"] for c in cfgs] == ["blank_0", "blank_1"]
        assert all(c["modifiers"] == [] for c in cfgs)
        assert all(c["is_blank"] is True for c in cfgs)
        assert pipeline._blank_agent_cfgs(0) == []

    def test_copy_event_norm_copies_events_not_value_scale(self):
        import pipeline
        donor = {
            "pot_mean": 30.0, "pot_std": 55.0,
            "stack_mean": 900.0, "stack_std": 400.0,
            "bets_mean": 12.0, "bets_std": 25.0,
            "blind_mean": 7.5, "blind_std": 2.5,
            "ev_mean": 0.1, "ev_std": 1.7,
            "mcts_value_scale": 123.0,
            "mcts_value_scale_n_samples": 999,
            "last_action_loss": 0.5,
        }
        dst = dict(_IDENTITY_NORM)
        pipeline._copy_event_norm(dst, donor)
        for k in pipeline._EVENT_NORM_KEYS:
            assert dst[k] == donor[k]
        # The value axis must bootstrap per-agent — never inherited.
        assert "mcts_value_scale" not in dst
        assert "mcts_value_scale_n_samples" not in dst
        assert "last_action_loss" not in dst

    def test_pipeline_wires_blank_agents_into_cyclic_pool(self):
        """The registry loop iterates regular agents + blank stubs, copies
        event-norm stats into fresh blanks, and warns in single-agent mode."""
        with open(os.path.join(_PKG_ROOT, "pipeline.py")) as f:
            src = f.read()
        assert "_blank_agent_cfgs(n_blank_agents)" in src
        assert 'list(multi_agent["agents"]) + blank_cfgs' in src
        assert "_copy_event_norm(" in src
        assert "only supported in" in src  # single-agent warning

    def test_blank_agent_plays_and_bootstraps_in_collection(self):
        """A fresh randomly-initialized agent with donor event-norm stats is
        a first-class citizen of collection: produces examples and
        bootstraps its own mcts_value_scale."""
        import pipeline
        cfg = copy.deepcopy(_TINY_CONFIG)
        cfg["mcts_train"]["min_players"] = 2
        cfg["mcts_train"]["max_players"] = 2
        _reseed()
        agents = _agents_list([("trained", 0), ("blank_0", 123)])
        # Blank's norm dict receives the donor's event stats (pipeline flow).
        pipeline._copy_event_norm(agents[1]["norm_stats"],
                                  agents[0]["norm_stats"])
        _reseed()
        result = run_mcts_collection(
            agents, cfg, "cpu", lambda *a, **k: None, 4)
        assert set(result.keys()) == {"trained", "blank_0"}
        assert len(result["blank_0"]) > 0
        assert "mcts_value_scale" in agents[1]["norm_stats"]
        # Blank weights genuinely differ from the donor's (random init).
        p_trained = next(iter(agents[0]["agent"].parameters()))
        p_blank = next(iter(agents[1]["agent"].parameters()))
        assert not torch.equal(p_trained, p_blank)
