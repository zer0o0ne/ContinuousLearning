"""The Slumbot evaluation runner (CONCEPT.md §12, `PLAN_PIPELINE.md` S11).

Offline: the client is a canned Slumbot that speaks the real grammar through
`evaluation/protocol.py` and never opens a socket, so every case here is exact
and reproducible (`CLAUDE.md` §4). What the battery therefore does **not** cover
is the wire itself — the retry policy, the live grammar and Slumbot's actual
responses are first exercised by a screening run on the Spark.

The properties that matter are the ones a wrong runner gets silently wrong:
the arithmetic of the headline number, the stamp that keeps a screening run
from being read as a result, that *cold* really is cold, that *warm* refreshes
when it says it does, and that a resumed million-hand run ends where an
uninterrupted one would.
"""

import json
import os

import numpy as np
import pytest
import torch

import eval_pipeline
import evaluation.protocol as protocol
from agent.policy import AgentPoolMember
from eval_pipeline import Stats, build_report, run, split_hands, warmup_hands
from evaluation.protocol import SLUMBOT_BIG_BLIND
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG, RAISE_SIZES, SMALL_BLIND,
)
from utils import Logger

GAME = {
    "n_actions": N_ACTIONS,
    "max_players": MAX_PLAYERS,
    "big_blind": BIG_BLIND,
    "small_blind": SMALL_BLIND,
    "players_range": [2, 9],
    "stack_bb_range": [10, 300],
    "raise_sizes": {"preflop": RAISE_SIZES[0], "flop": RAISE_SIZES[1],
                    "turn": RAISE_SIZES[2], "river": RAISE_SIZES[3]},
}
N_RAISE_BINS = N_ACTIONS - 3
DECK = ["2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd",
        "Qh", "Ks", "Ac", "2d", "3h", "4s", "5c", "6d", "7h", "8s"]


class ScriptedSlumbot:
    """A canned Slumbot: the real grammar, a fixed opponent, no socket.

    The state machine is `evaluation/protocol.py`'s own, so the action strings
    this produces are ones the adapter's replay has to be able to parse — a stub
    that invented its own grammar would test the runner against a fiction. The
    opponent calls or checks, always, and the hand's result is scripted: what is
    under test is the bookkeeping, not who won.
    """

    def __init__(self, winnings, seats=(1, 0), reveal=True, start_at=0):
        self.winnings = list(winnings)
        self.seats = list(seats)
        self.reveal = reveal
        # A resumed run rejoins a stream the server has been dealing all along;
        # `start_at` is how a stub says so. Without it a "resume" would be
        # compared against a different set of hands and the test would be about
        # the stub rather than about the runner.
        self.n_hands = int(start_at)
        self.sent = []
        self.state = None

    # ------------------------------------------------------------- internals

    def _visible(self):
        n = {0: 0, 1: 3, 2: 4, 3: 5}[int(self.state["turn"])]
        return DECK[6:6 + n]

    def _apply(self, token):
        protocol.apply_token(self.state, token, RAISE_SIZES, N_RAISE_BINS)
        self.action += token
        if self.state["is_terminal"]:
            return
        protocol.advance_after_action(self.state, token)
        # An all-in runout ends the betting without a token that closes a
        # street; the real server answers such a hand with `winnings`, so the
        # stub does too.
        if (2 in self.state["players_state"]
                and self.state["bets"][0] == self.state["bets"][1]):
            self.state["is_terminal"] = True

    def _opponent_moves(self):
        opp = 1 - self.client_pos
        while (not self.state["is_terminal"]
               and self.state["active_pos"] == opp):
            facing = self.state["high_bet"] > self.state["bets"][opp]
            self._apply("c" if facing else "k")

    def _response(self):
        r = {"client_pos": self.client_pos, "action": self.action,
             "hole_cards": self.hole, "board": self._visible(), "token": "t"}
        if self.state["is_terminal"]:
            r["winnings"] = self.winnings[
                (self.n_hands - 1) % len(self.winnings)]
            r["baseline_winnings"] = 0.0
            if self.reveal:
                r["bot_hole_cards"] = self.bot_hole
        return r

    # ---------------------------------------------------------------- the API

    def new_hand(self):
        i = self.n_hands
        self.n_hands += 1
        self.client_pos = self.seats[i % len(self.seats)]
        self.hole = DECK[0:2]
        self.bot_hole = DECK[2:4]
        self.state = protocol.initial_state()
        self.state["_first_in_street"] = True
        self.action = ""
        self._opponent_moves()
        return self._response()

    def act(self, incr):
        self.sent.append(incr)
        self._apply(incr)
        if not self.state["is_terminal"]:
            self._opponent_moves()
        return self._response()


# ------------------------------------------------------------------ fixtures


def _checkpoints(tmp_path):
    torch.manual_seed(0)
    agent = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS)
    torch.manual_seed(1)
    embed = OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS, n_members=6)
    a_path = str(tmp_path / "agent.pt")
    e_path = str(tmp_path / "embedding.pt")
    torch.save({"model_state_dict": agent.state_dict()}, a_path)
    torch.save({"model_state_dict": embed.state_dict()}, e_path)
    return a_path, e_path


def test_resume_refuses_a_changed_checkpoint_without_appending(tmp_path):
    cfg = _config(tmp_path, evaluation={"hands": 2, "warm": False})
    _report, directory, _lines = _run(tmp_path, cfg)
    path = os.path.join(directory, "cold_w00.jsonl")
    before = open(path, "rb").read()
    checkpoint = torch.load(cfg["evaluation"]["agent_checkpoint"], weights_only=False)
    first = next(iter(checkpoint["model_state_dict"].values()))
    first.add_(.1)
    torch.save(checkpoint, cfg["evaluation"]["agent_checkpoint"])
    with pytest.raises(ValueError, match="identity changed"):
        _run(tmp_path, cfg)
    assert open(path, "rb").read() == before


def _config(tmp_path, **overrides):
    a_path, e_path = _checkpoints(tmp_path)
    cfg = {
        "experiment": "toy",
        "seed": 4,
        "device": "cpu",
        "out_dir": str(tmp_path),
        "game": GAME,
        "embedding_net": dict(NET_CFG, K=2, fit_lr=0.1, fit_reg=0.01, R=3,
                              amortised_weight=1.0,
                              showdown_strength_weight=0.3,
                              showdown_class_weight=0.1),
        "evaluation": {
            "run": "toy",
            "agent_checkpoint": a_path,
            "embedding_checkpoint": e_path,
            "hands": 8,
            "min_reportable_hands": 1000000,
            "cold": True,
            "warm": True,
            "fit_window": 5,
            "warmup_buckets": [0, 1, 2, 5],
            "selection_disclosure": {"candidates_screened": 3,
                                     "screening_hands_each": 5000},
            "log_every": 1000,
        },
    }
    cfg["evaluation"].update(overrides.pop("evaluation", {}))
    cfg.update(overrides)
    return cfg


def _run(tmp_path, cfg, client=None, name="out", lines=None):
    out = str(tmp_path / name)
    log = Logger(str(tmp_path / f"{name}_logs"))
    sink = [] if lines is None else lines

    def logger(msg):
        sink.append(str(msg))
        log(msg)

    try:
        return run(cfg, logger, out, client=client or ScriptedSlumbot(
            [100.0, -50.0, 200.0])), out, sink
    finally:
        log.close()


# ------------------------------------------------- 1: the headline arithmetic


def test_bb_per_100_and_its_standard_error_are_hand_computed():
    chips = [100.0, -50.0, 200.0, 0.0]
    bb = np.asarray(chips) / SLUMBOT_BIG_BLIND

    st = Stats()
    for c in chips:
        st.add(c)
    assert st.n == 4
    assert st.total_bb == pytest.approx(bb.sum(), abs=1e-12)
    assert st.bb_per_100 == pytest.approx(100.0 * bb.sum() / 4, abs=1e-12)
    assert st.bb_per_100 == pytest.approx(62.5, abs=1e-12)
    assert st.stderr == pytest.approx(
        float(bb.std(ddof=1) / np.sqrt(4) * 100.0), abs=1e-12)
    # The online form and the batch form are the same number.
    assert st.stderr == pytest.approx(protocol.stderr_bb_per_100(chips),
                                      abs=1e-12)


def test_the_edge_cases_are_zero_and_not_a_crash():
    empty = Stats()
    assert empty.n == 0 and empty.bb_per_100 == 0.0 and empty.stderr == 0.0
    assert empty.summary() == {"hands": 0, "bb_per_100": 0.0,
                               "stderr_bb_per_100": 0.0, "total_bb": 0.0}

    one = Stats()
    one.add(500.0)
    assert one.n == 1
    assert one.bb_per_100 == pytest.approx(500.0, abs=1e-12)
    assert one.stderr == 0.0, "one hand carries no standard error"


# ------------------------------------------------- 2: the SCREENING ONLY stamp


def test_a_short_run_is_stamped_screening_only(tmp_path):
    report, _out, lines = _run(tmp_path, _config(tmp_path))
    assert report["screening_only"] is True
    assert any("SCREENING ONLY" in ln for ln in lines)
    assert any("candidate(s) screened" in ln for ln in lines)


def test_a_run_at_the_threshold_is_not_stamped(tmp_path):
    cfg = _config(tmp_path, evaluation={"min_reportable_hands": 8})
    report, _out, lines = _run(tmp_path, cfg)
    assert report["screening_only"] is False
    assert not any("SCREENING ONLY" in ln for ln in lines)
    assert any("=== Slumbot result ===" in ln for ln in lines)


def test_the_selection_disclosure_is_required_and_not_a_habit(tmp_path):
    cfg = _config(tmp_path)
    cfg["evaluation"].pop("selection_disclosure")
    with pytest.raises(AssertionError, match="candidates were screened"):
        build_report(cfg, {"cold": Stats().summary()}, log=lambda _m: None)

    cfg["evaluation"]["selection_disclosure"] = {"candidates_screened": 2}
    with pytest.raises(AssertionError, match="candidates were screened"):
        build_report(cfg, {"cold": Stats().summary()}, log=lambda _m: None)


# ------------------------------------------------------------- 3: cold is cold


def test_cold_pins_the_embedding_to_zero_for_every_decision(tmp_path,
                                                            monkeypatch):
    seen = []
    real = AgentPoolMember.logits

    def spy(self, contexts):
        seen.append(float(np.abs(self.embeddings.cpu().numpy()).max()))
        return real(self, contexts)

    monkeypatch.setattr(AgentPoolMember, "logits", spy)
    cfg = _config(tmp_path, evaluation={"warm": False})
    report, _out, _lines = _run(tmp_path, cfg)

    assert seen, "the agent was never asked to act"
    assert max(seen) == 0.0, (
        "cold means `e = 0` at every decision — a fitted vector reached the "
        "policy and the run measured the warm one under the cold name")
    assert "warm" not in report["modes"]
    assert report["modes"]["cold"]["hands"] == 8


def test_cold_never_fits_anything(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(eval_pipeline, "fit_vectors",
                        lambda *a, **k: calls.append(1))
    cfg = _config(tmp_path, evaluation={"warm": False})
    _run(tmp_path, cfg)
    assert calls == []


# --------------------------------------------------------- 4: warm warms up


def test_warm_refreshes_on_the_configured_interval(tmp_path, monkeypatch):
    real = eval_pipeline.fit_vectors
    at = []

    def spy(embed_net, hands, config, device):
        at.append(len(hands))
        return real(embed_net, hands, config, device)

    monkeypatch.setattr(eval_pipeline, "fit_vectors", spy)
    cfg = _config(tmp_path, evaluation={"cold": False, "hands": 10})
    report, out, _lines = _run(tmp_path, cfg)

    # R = 3, and the first refresh cannot happen before there is a hand to fit
    # from: hands 0,1,2 are played cold, then hand 3 is fitted from 3, hand 6
    # from 6 and hand 9 from 9 — capped by `fit_window = 5`.
    assert at == [3, 5, 5], at

    observed = [json.loads(line)["hands_observed"]
                for line in open(os.path.join(out, "warm_w00.jsonl"))]
    assert observed == [0, 0, 0, 3, 3, 3, 5, 5, 5, 5], observed
    assert set(report["modes"]["warm"]["by_hands_observed"]) == {"0", "2", "5"}
    assert sum(cell["hands"] for cell in
               report["modes"]["warm"]["by_hands_observed"].values()) == 10


def test_a_fitted_vector_actually_reaches_the_policy(tmp_path, monkeypatch):
    seen = []
    real = AgentPoolMember.logits

    def spy(self, contexts):
        seen.append(float(np.abs(self.embeddings.cpu().numpy()).max()))
        return real(self, contexts)

    monkeypatch.setattr(AgentPoolMember, "logits", spy)
    cfg = _config(tmp_path, evaluation={"cold": False, "hands": 10})
    _run(tmp_path, cfg)
    assert seen[0] == 0.0, "the cold start is the zero vector (§5.5)"
    assert max(seen) > 0.0, "no fitted vector ever reached a decision"


def test_the_warm_up_hand_count_is_the_first_bucket_that_caught_cold_up():
    edges = [0, 1, 2, 5]
    cold = {"bb_per_100": 10.0}
    warm = {"by_hands_observed": {
        "0": {"bb_per_100": -5.0, "hands": 3},
        "2": {"bb_per_100": 4.0, "hands": 3},
        "5": {"bb_per_100": 25.0, "hands": 4},
    }}
    assert warmup_hands(warm, cold, edges) == 5
    assert warmup_hands(warm, {"bb_per_100": 100.0}, edges) is None, (
        "never catching up is the result, not a missing value")
    assert warmup_hands(warm, {"bb_per_100": -50.0}, edges) == 0
    assert warmup_hands(None, cold, edges) is None


def test_warm_worse_than_cold_is_reported_in_those_words(tmp_path):
    cfg = _config(tmp_path)
    report = build_report(
        cfg, {"cold": {"bb_per_100": 5.0, "hands": 10, "failed_hands": 0,
                       "clamps": {}, "stderr_bb_per_100": 1.0,
                       "by_hands_observed": {}},
              "warm": {"bb_per_100": -3.0, "hands": 10, "failed_hands": 0,
                       "clamps": {}, "stderr_bb_per_100": 1.0,
                       "by_hands_observed": {}}},
        log=lambda _m: None)
    assert report["warm_is_worse_than_cold"] is True
    assert report["warm_minus_cold_bb_per_100"] == pytest.approx(-8.0)

    lines = []
    eval_pipeline.format_report(report, lines.append)
    assert any("the exploitation mechanism is a net negative" in ln
               for ln in lines)
    assert any("not a bug to tune away" in ln for ln in lines)


# ---------------------------------------------------------------- 5: resume


def test_resume_reproduces_an_uninterrupted_runs_statistics(tmp_path):
    winnings = [100.0, -50.0, 200.0, 0.0, -300.0]
    cfg = _config(tmp_path, evaluation={"hands": 10})

    whole, whole_dir, _l = _run(tmp_path, cfg, client=ScriptedSlumbot(winnings),
                                name="whole")

    # A run that got four hands into the cold phase and then died.
    short = _config(tmp_path, evaluation={"hands": 4, "warm": False})
    short["evaluation"]["agent_checkpoint"] = cfg["evaluation"][
        "agent_checkpoint"]
    short["evaluation"]["embedding_checkpoint"] = cfg["evaluation"][
        "embedding_checkpoint"]
    _r, part_dir, _l2 = _run(tmp_path, short,
                             client=ScriptedSlumbot(winnings), name="part")
    assert len(open(os.path.join(part_dir, "cold_w00.jsonl")).readlines()) == 4
    os.remove(os.path.join(part_dir, "slumbot_report.json"))

    resumed, _d, _l3 = _run(tmp_path, cfg,
                            client=ScriptedSlumbot(winnings, start_at=4),
                            name="part")

    for mode in ("cold", "warm"):
        a, b = whole["modes"][mode], resumed["modes"][mode]
        assert a["hands"] == b["hands"] == 10, mode
        assert a["bb_per_100"] == pytest.approx(b["bb_per_100"], abs=1e-12), mode
        assert a["stderr_bb_per_100"] == pytest.approx(
            b["stderr_bb_per_100"], abs=1e-12), mode
        assert a["by_hands_observed"] == b["by_hands_observed"], mode
    assert whole["warmup_hands"] == resumed["warmup_hands"]

    # The hands themselves are the same hands, not merely the same totals.
    for mode in ("cold", "warm"):
        one = open(os.path.join(whole_dir, f"{mode}_w00.jsonl")).read()
        two = open(os.path.join(part_dir, f"{mode}_w00.jsonl")).read()
        assert one == two, mode


def test_a_failed_hand_is_counted_and_does_not_shift_the_index(tmp_path):
    class Flaky(ScriptedSlumbot):
        def new_hand(self):
            r = super().new_hand()
            if self.n_hands == 3:
                raise RuntimeError("Slumbot new_hand ReadTimeout")
            return r

    cfg = _config(tmp_path, evaluation={"warm": False, "hands": 6})
    report, out, lines = _run(tmp_path, cfg, client=Flaky([100.0]),
                              name="flaky")
    assert report["modes"]["cold"]["failed_hands"] == 1
    assert report["modes"]["cold"]["hands"] == 5
    written = [json.loads(ln) for ln in
               open(os.path.join(out, "cold_w00.jsonl"))]
    assert len(written) == 6, "the failed hand keeps its slot in the file"
    assert written[2]["failed"] is True
    assert any("failed:" in ln for ln in lines)


# --------------------------------------------------------- 6: the whole run


def test_the_report_carries_everything_section_12_asks_for(tmp_path):
    report, out, _lines = _run(tmp_path, _config(tmp_path))
    assert set(report["modes"]) == {"cold", "warm"}
    for mode in ("cold", "warm"):
        r = report["modes"][mode]
        assert r["hands"] == 8
        assert set(r) >= {"hands", "bb_per_100", "stderr_bb_per_100",
                          "failed_hands", "clamps", "by_hands_observed"}
    assert report["selection_disclosure"]["candidates_screened"] == 3
    assert "warm_minus_cold_bb_per_100" in report
    on_disk = json.load(open(os.path.join(out, "slumbot_report.json")))
    assert on_disk["report"]["modes"]["cold"]["hands"] == 8


# ------------------------------------------- 4b: a pool member as the hero


def _regular_config(tmp_path, **evaluation):
    """`PLAN_PROCEDURAL_POOL.md` §P5: one archetype plays the hands."""
    from pool.strength import preflop_equity_table

    path = str(tmp_path / "preflop.npy")
    preflop_equity_table(path, seed=0, n_deals=20_000)
    cfg = _config(tmp_path, evaluation={
        "hero": {"kind": "regular", "archetype": "tag", "variant_seed": 0,
                 "spread": 0.0, "preflop_table": path},
        **evaluation})
    # A procedural member is not the agent and does not need its checkpoint.
    cfg["evaluation"]["agent_checkpoint"] = str(tmp_path / "not-here.pt")
    return cfg


def test_a_pool_member_can_play_the_hands_without_an_agent(tmp_path):
    report, _out, lines = _run(tmp_path, _regular_config(tmp_path))
    assert report["modes"]["cold"]["hands"] == 8
    assert "warm" not in report["modes"]
    assert any("no opponent vector" in line for line in lines)
    assert any("procedural pool member 'tag'" in line for line in lines)
    assert any("not an agent result" in line for line in lines)


def test_the_warm_run_is_dropped_rather_than_failing_one_process_deep(tmp_path):
    cfg = _regular_config(tmp_path, cold=False, warm=True)
    with pytest.raises(AssertionError, match="turning both off"):
        _run(tmp_path, cfg)


def test_a_pool_member_run_resumes_like_any_other(tmp_path):
    whole = _regular_config(tmp_path, hands=10)
    short = _regular_config(tmp_path, hands=4)
    full, _out, _lines = _run(tmp_path, whole, name="full")
    _run(tmp_path, short, name="part")
    resumed, _out, _lines = _run(tmp_path, whole, name="part")
    assert (resumed["modes"]["cold"]["bb_per_100"]
            == full["modes"]["cold"]["bb_per_100"])
    assert resumed["modes"]["cold"]["hands"] == full["modes"]["cold"]["hands"]


def test_no_hero_section_is_the_agent_playing_exactly_as_before(tmp_path):
    plain = _config(tmp_path, evaluation={"warm": False})
    named = _config(tmp_path, evaluation={
        "warm": False, "hero": {"kind": "agent"}})
    one, _out, _lines = _run(tmp_path, plain, name="plain")
    two, _out, _lines = _run(tmp_path, named, name="named")
    assert one["modes"]["cold"] == two["modes"]["cold"]


def test_an_unknown_hero_kind_is_refused(tmp_path):
    cfg = _config(tmp_path, evaluation={
        "warm": False, "hero": {"kind": "solver"}})
    with pytest.raises(AssertionError, match="unknown hero kind"):
        _run(tmp_path, cfg)


def test_turning_both_modes_off_is_refused(tmp_path):
    cfg = _config(tmp_path, evaluation={"cold": False, "warm": False})
    with pytest.raises(AssertionError, match="measures nothing"):
        _run(tmp_path, cfg)


def test_warm_without_an_embedding_checkpoint_is_refused(tmp_path):
    cfg = _config(tmp_path)
    cfg["evaluation"]["embedding_checkpoint"] = None
    with pytest.raises(AssertionError, match="embedding_checkpoint"):
        _run(tmp_path, cfg)


# ------------------------------------------------------------ 7: the workers


STUB_WINNINGS = [100.0, -50.0, 200.0, 0.0, -300.0]


def make_stub_client(worker):
    """The canned Slumbot a spawned worker builds instead of an HTTP client.

    Reached by name (`worker_main` resolves `"module:function"`), because a
    spawned process cannot be handed a live object. Each worker gets its own
    session, which is what the parallel path is: `n_workers` independent tables
    against the same opponent, not one table shared.
    """
    return ScriptedSlumbot(STUB_WINNINGS, start_at=0)


def test_the_shares_add_up_to_the_hand_count():
    assert split_hands(10, 1) == [10]
    assert split_hands(10, 3) == [4, 3, 3]
    assert split_hands(2, 4) == [1, 1, 0, 0]
    for n in (0, 1, 7, 1_000_000):
        for w in (1, 3, 8):
            assert sum(split_hands(n, w)) == n


def test_workers_play_their_own_shares_into_their_own_files(tmp_path):
    cfg = _config(tmp_path, evaluation={
        "warm": False, "hands": 6, "n_workers": 2, "min_reportable_hands": 6})
    out = str(tmp_path / "par")
    log = Logger(str(tmp_path / "par_logs"))
    try:
        report = run(cfg, log, out,
                     client_factory="tests.test_eval_pipeline:make_stub_client")
    finally:
        log.close()

    for worker, share in enumerate(split_hands(6, 2)):
        lines = open(os.path.join(out, f"cold_w{worker:02d}.jsonl")).readlines()
        assert len(lines) == share == 3, worker
    assert report["modes"]["cold"]["hands"] == 6
    assert report["screening_only"] is False

    # Two independent sessions of three hands are the same six results as one
    # session of six would have been, because the stub deals the same stream.
    expected = Stats()
    for _ in range(2):
        for chips in STUB_WINNINGS[:3]:
            expected.add(chips)
    assert report["modes"]["cold"]["bb_per_100"] == pytest.approx(
        expected.bb_per_100, abs=1e-12)


def test_a_parallel_run_resumes_per_worker(tmp_path):
    cfg = _config(tmp_path, evaluation={
        "warm": False, "hands": 6, "n_workers": 2, "min_reportable_hands": 6})
    out = str(tmp_path / "par")
    factory = "tests.test_eval_pipeline:make_stub_client"

    short = _config(tmp_path, evaluation={
        "warm": False, "hands": 2, "n_workers": 2, "min_reportable_hands": 6})
    short["evaluation"]["agent_checkpoint"] = cfg["evaluation"][
        "agent_checkpoint"]
    log = Logger(str(tmp_path / "par_logs"))
    try:
        run(short, log, out, client_factory=factory)
        assert all(len(open(os.path.join(out, f"cold_w{w:02d}.jsonl"))
                       .readlines()) == 1 for w in (0, 1))
        report = run(cfg, log, out, client_factory=factory)
    finally:
        log.close()

    assert report["modes"]["cold"]["hands"] == 6
    for worker, share in enumerate(split_hands(6, 2)):
        assert len(open(os.path.join(out, f"cold_w{worker:02d}.jsonl"))
                   .readlines()) == share


def test_the_clamp_counters_survive_a_resume_because_they_ride_on_the_hands(
        tmp_path):
    """Tallied in memory they would restart at zero; on the hands they cannot."""
    cfg = _config(tmp_path, evaluation={"warm": False, "hands": 6})
    _report, out, _lines = _run(tmp_path, cfg, name="clamps")

    path = os.path.join(out, "cold_w00.jsonl")
    hands = [json.loads(ln) for ln in open(path)]
    assert all("clamps" in h for h in hands)
    hands[0]["clamps"] = [1, 0, 2, 0, 0]
    with open(path, "w") as fh:
        for h in hands:
            fh.write(json.dumps(h) + "\n")

    summary = eval_pipeline.aggregate("cold", out, cfg, log=lambda _m: None)
    assert sum(summary["clamps"].values()) == 3
    assert summary["hands"] == 6
