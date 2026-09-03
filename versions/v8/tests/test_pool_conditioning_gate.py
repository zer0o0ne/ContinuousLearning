"""The P2 gate: is a past agent stronger reading its tablemates? (P2 of
PLAN_AMORTISED_POOL.md)

The gate exists to answer one question with a number and a standard error, and
the ways it could produce a *wrong* number are all about pairing:

* the two conditions must play the same tables — same seeds, same opponents,
  same rotation — or the difference is a difference between two corpora;
* a zero table must reproduce the `e = 0` condition exactly, or "amortised"
  differs from "zero" partly by plumbing;
* the standard error must be the paired one over sessions, because a single
  session of NLHE has a standard error of tens of BB/100 and an unpaired
  comparison at this scale says nothing;
* the report must not be readable as an agent result.

Everything runs end to end on CPU at toy scale (`CLAUDE.md` §4): tiny networks,
four sessions of six hands, the degenerate pool.
"""

import json
import os

import numpy as np
import pytest
import torch

from gates.pool_conditioning import STAMP, run, write_report
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from tests.g1_fixtures import (BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG,
                               RAISE_SIZES, SMALL_BLIND, STYLE_CFG)

N_SESSIONS = 4
N_HANDS = 6
N_MEMBERS = 12


def _config(tmp_path, R=2, window=4):
    return {
        "name": "pc-toy",
        "seed": 3,
        "device": "cpu",
        "out_dir": str(tmp_path),
        "driver_batch_size": 16,
        "game": {
            "n_actions": N_ACTIONS, "max_players": MAX_PLAYERS,
            "big_blind": BIG_BLIND, "small_blind": SMALL_BLIND,
            "players_range": [2, 9], "stack_bb_range": [10, 300],
            "raise_sizes": {street: list(sizes) for street, sizes in
                            zip(("preflop", "flop", "turn", "river"),
                                RAISE_SIZES)},
        },
        "style": STYLE_CFG,
        "bootstrap": [
            {"kind": "degenerate", "strategy": "always_call", "n_variants": 4,
             "label": "call_styles"},
            {"kind": "degenerate", "strategy": "maniac", "n_variants": 4,
             "label": "maniac_styles"},
            {"kind": "degenerate", "strategy": "nit", "n_variants": 4,
             "label": "nit_styles"},
        ],
        "embedding_net": {**NET_CFG, "R": R, "pool_agent_window": window},
        "pool_conditioning": {
            "agent_checkpoint": str(tmp_path / "agent.pt"),
            "embedding_checkpoint": str(tmp_path / "embedding.pt"),
            "agent_name": "agent0",
            "n_sessions": N_SESSIONS,
            "hands_per_session": N_HANDS,
            "seed_base": 500_000,
        },
    }


def _checkpoints(cfg, stored=None):
    """Write the two checkpoints the gate loads, with the config they carry."""
    stored = cfg if stored is None else stored
    torch.manual_seed(0)
    agent = AgentNet(cfg["embedding_net"], N_ACTIONS, MAX_PLAYERS)
    torch.manual_seed(1)
    embed = OpponentEmbeddingNet(cfg["embedding_net"], N_ACTIONS, MAX_PLAYERS,
                                 n_members=N_MEMBERS)
    pc = cfg["pool_conditioning"]
    torch.save({"model_state_dict": agent.state_dict(), "config": stored,
                "iteration": 0}, pc["agent_checkpoint"])
    torch.save({"model_state_dict": embed.state_dict(), "config": stored,
                "iteration": 0}, pc["embedding_checkpoint"])


def _run(tmp_path, cfg=None):
    """One gate run. Returns `(report, payload written to disk)`."""
    cfg = cfg or _config(tmp_path)
    _checkpoints(cfg)
    out_dir = str(tmp_path / "out")
    report = run(cfg, lambda _m: None, out_dir)
    with open(os.path.join(out_dir, "pool_conditioning.json")) as fh:
        return report, json.load(fh)


# ------------------------------------------------------------------ end to end


def test_the_gate_runs_end_to_end_and_writes_a_stamped_report(tmp_path):
    report, payload = _run(tmp_path)

    assert report["stamp"] == STAMP and payload["stamp"] == STAMP
    assert report["n_sessions"] == N_SESSIONS
    assert report["hands_per_condition"] == N_SESSIONS * N_HANDS, (
        "the two conditions did not play the same number of hands")
    for cell in [report["overall"]] + list(report["by_table_size"].values()) \
            + list(report["by_stack_depth"].values()):
        assert set(cell) == {"zero", "amortised", "difference"}
        for stats in cell.values():
            assert not np.isnan(stats["mean"])
    assert len(payload["rows"]) == N_SESSIONS
    assert {r["hands"] for r in payload["rows"]} == {N_HANDS}
    # Every session is in exactly one table-size and one stack bucket.
    assert sum(c["difference"]["n"]
               for c in report["by_table_size"].values()) == N_SESSIONS
    assert sum(c["difference"]["n"]
               for c in report["by_stack_depth"].values()) == N_SESSIONS
    t = report["timings"]
    assert t["refreshes"] > 0 and t["window_hands"] > 0
    assert t["play_seconds"] > 0 and not np.isnan(t["hand_tokens_us"])


def test_the_two_conditions_play_the_same_tables(tmp_path):
    """Paired means paired: same sessions, same seats, same opponents.

    Only what the agent reads for its tablemates may differ between the two
    halves; if the tables differed, the difference would be a difference between
    two corpora and the standard error would be meaningless.
    """
    _report, payload = _run(tmp_path)
    for row in payload["rows"]:
        assert row["difference"] == pytest.approx(
            row["bb_per_100_amortised"] - row["bb_per_100_zero"], abs=1e-12)
    assert {r["session"] for r in payload["rows"]} == set(range(N_SESSIONS))
    # The table configuration is a property of the session, not of the
    # condition: one row per session carries one table size and one stack.
    assert all(2 <= r["num_players"] <= 9 for r in payload["rows"])
    assert all(10 <= r["stack_bb"] <= 300 for r in payload["rows"])


def test_with_the_vectors_forced_to_zero_the_difference_is_exactly_zero(
        tmp_path, monkeypatch):
    """A zero table is `e = 0` (P1), so the two conditions must be one run.

    This is the pairing test with the effect removed: if anything other than the
    vectors differs between the halves — a seed drawn in one and not the other,
    a member rebuilt differently — the hands diverge and the difference is not
    exactly zero.
    """
    import gates.pool_conditioning as gate

    d_emb = NET_CFG["d_emb"]
    monkeypatch.setattr(gate, "amortised_vectors",
                        lambda *a, **k: np.zeros((MAX_PLAYERS, d_emb),
                                                 dtype=np.float32))
    report, payload = _run(tmp_path)

    for row in payload["rows"]:
        assert row["difference"] == 0.0, (
            f"session {row['session']} played differently under a zero table")
    assert report["overall"]["difference"]["mean"] == 0.0
    assert report["overall"]["difference"]["se"] == 0.0
    assert report["overall"]["zero"]["mean"] == pytest.approx(
        report["overall"]["amortised"]["mean"], abs=1e-12)


def test_a_table_that_changes_the_policy_changes_the_hands(tmp_path,
                                                          monkeypatch):
    """The other half of the zero test, and what stops it being vacuous.

    At toy scale an untrained network's policy moves by about 5e-4 under the
    real `K = 0` vectors, and that flips no sampled action over the two dozen
    hands this fixture plays — so the zero test above proves the pairing but
    says nothing about whether the table reaches the forward at all. A table
    large enough to move the policy must move the hands; if it does not, the
    member is being seated without its vectors.
    """
    import gates.pool_conditioning as gate

    d_emb = NET_CFG["d_emb"]
    rng = np.random.default_rng(0)
    loud = (50.0 * rng.normal(size=(MAX_PLAYERS, d_emb))).astype(np.float32)
    monkeypatch.setattr(gate, "amortised_vectors", lambda *a, **k: loud)
    _report, payload = _run(tmp_path)

    assert any(row["difference"] != 0.0 for row in payload["rows"]), (
        "a table that changes the policy changed no hand — the conditioned "
        "member never reached the driver")


def test_the_paired_standard_error_is_the_one_computed_by_hand(tmp_path):
    """Mean and SE of the per-session differences, and nothing else."""
    report, payload = _run(tmp_path)
    diffs = np.array([r["difference"] for r in payload["rows"]])

    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert report["overall"]["difference"]["n"] == len(diffs)
    assert report["overall"]["difference"]["mean"] == pytest.approx(mean,
                                                                    abs=1e-12)
    assert report["overall"]["difference"]["se"] == pytest.approx(se, abs=1e-12)

    for condition in ("zero", "amortised"):
        values = np.array([r[f"bb_per_100_{condition}"]
                           for r in payload["rows"]])
        cell = report["overall"][condition]
        assert cell["mean"] == pytest.approx(values.mean(), abs=1e-12)
        assert cell["se"] == pytest.approx(
            values.std(ddof=1) / np.sqrt(len(values)), abs=1e-12)


def test_the_bb_per_100_is_the_agents_own_chips_over_the_big_blind(tmp_path):
    """The one accounting there is (`train/generate.py::_results_by_member`)."""
    report, payload = _run(tmp_path)
    for row in payload["rows"]:
        # BB/100 is 100 × BB per hand, so it scales with the hand count the row
        # reports and with nothing else.
        assert abs(row["bb_per_100_zero"]) < 100.0 * 300.0, (
            "a session won more than the deepest stack allows")
    total = sum(r["bb_per_100_zero"] for r in payload["rows"]) / N_SESSIONS
    assert report["overall"]["zero"]["mean"] == pytest.approx(total, abs=1e-12)


# ------------------------------------------- what the report refuses to be


def test_a_report_that_could_be_read_as_an_agent_result_is_refused(tmp_path):
    _report, payload = _run(tmp_path)
    path = str(tmp_path / "again.json")

    assert write_report(path, payload) == path

    no_stamp = {**payload, "stamp": "results"}
    with pytest.raises(AssertionError, match="must carry the stamp"):
        write_report(path, no_stamp)

    no_se = json.loads(json.dumps(payload))
    del no_se["overall"]["difference"]["se"]
    with pytest.raises(AssertionError, match="no standard error"):
        write_report(path, no_se)

    no_bucket_se = json.loads(json.dumps(payload))
    bucket = next(iter(no_bucket_se["by_table_size"]))
    del no_bucket_se["by_table_size"][bucket]["zero"]["se"]
    with pytest.raises(AssertionError, match="no standard error"):
        write_report(path, no_bucket_se)


def test_a_checkpoint_from_another_configuration_is_refused(tmp_path):
    """A checkpoint that played a different game would be a silent mismatch."""
    cfg = _config(tmp_path)
    other = json.loads(json.dumps(cfg))
    other["game"]["n_actions"] = N_ACTIONS + 1
    _checkpoints(cfg, stored=other)
    with pytest.raises(AssertionError, match="different game"):
        run(cfg, lambda _m: None, str(tmp_path / "out"))

    other = json.loads(json.dumps(cfg))
    other["embedding_net"]["n_layers"] = NET_CFG["n_layers"] + 1
    _checkpoints(cfg, stored=other)
    with pytest.raises(AssertionError, match="embedding_net.n_layers"):
        run(cfg, lambda _m: None, str(tmp_path / "out"))
