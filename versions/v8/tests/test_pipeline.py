"""The outer loop, end to end (CONCEPT.md §8, `PLAN_PIPELINE.md` S9).

Toy scale — two iterations, three sessions of four hands, two rollout samples
per action — so nothing here says whether the loop *learns* anything. What it
pins is the wiring that is silent when it is wrong:

* who sits in hero's seat at iteration 0 and who sits there afterwards (§7.1),
  asserted by recording who was actually asked for an action;
* that the held-out slice reaches the metric and never the optimiser (§8), and
  that the four gap numbers are the ones §8 defines, warm and cold;
* that the pool grows by exactly `style.agent_variants` members per iteration
  and that the embedding table has a row reserved for each of them (D9, D11);
* that a run resumed from a crash produces the same artefacts as one that was
  never interrupted — the property a multi-day run on the Spark depends on;
* that table size and stack depth are sampled across their full ranges with no
  weighting toward heads-up or 200 BB (`CLAUDE.md` §1).
"""

import json
import os
import shutil

import numpy as np
import pytest
import torch

import pipeline
from env.session import build_sessions
from pipeline import gap_terms, oracle_gap, run, split_heldout, winrate_line
from pool.style import StyleParams
from train.generate import load_shard
from utils import Logger
from tests.g1_fixtures import (
    BIG_BLIND, MAX_PLAYERS, N_ACTIONS, NET_CFG, RAISE_SIZES, SMALL_BLIND,
    STYLE_CFG,
)

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

# Nine members, so a nine-handed table can be seated from the bootstrap pool
# alone (`env/session.py` refuses to shrink one).
BOOTSTRAP = (
    [{"kind": "degenerate", "strategy": s, "style": "identity"}
     for s in ("always_fold", "always_call", "always_min_raise", "maniac",
               "nit")]
    + [{"kind": "degenerate", "strategy": s, "n_variants": 2,
        "label": f"{s}_styles"} for s in ("always_call", "maniac")]
)


def toy_config(**overrides):
    cfg = {
        "experiment": "toy",
        "seed": 5,
        "device": "cpu",
        "out_dir": None,                      # set by the caller
        "n_iterations": 2,
        "n_sessions": 3,
        "hands_per_session": 4,
        "labels_per_shard": 8,
        "driver_batch_size": 32,
        "game": GAME,
        "style": dict(STYLE_CFG, agent_variants=2),
        "bootstrap": BOOTSTRAP,
        "agent_init": {"kind": "degenerate", "strategy": "nit",
                       "style": "identity", "label": "agent_init"},
        "embedding_net": dict(
            NET_CFG, K=2, fit_lr=0.1, fit_reg=0.01, R=2, max_iterations=2,
            retrain_every=1, corpus_sessions=2, corpus_hands_per_session=4,
            steps=3, batch_hands=4, lr=1e-3, amortised_weight=1.0,
            showdown_strength_weight=0.3, showdown_class_weight=0.1,
            log_every=100),
        "oracle": {"samples_per_action": 2, "max_combos": 4, "batch_hands": 64,
                   "temperature": 0.5, "divisor": "pot_plus_bet"},
        "pool_sampling": {"pfsp_exponent": 2.0, "floor_fraction": 0.25,
                          "n_clusters": 4, "result_decay": 0.8},
        "agent_train": {"loss": "soft_q", "steps": 3, "first_iteration_steps": 4,
                        "batch_hands": 4, "lr": 1e-3, "embedding_dropout": 0.1,
                        "heldout_fraction": 0.25, "log_every": 100},
        "evaluation": {"hands": 10, "min_reportable_hands": 1000000},
    }
    cfg.update(overrides)
    return cfg


def _run(tmp_path, cfg=None, name="exp"):
    cfg = cfg or toy_config()
    base = str(tmp_path / name)
    cfg["out_dir"] = base
    log = Logger(base)
    try:
        metrics = run(cfg, lambda _m: None, os.path.join(base, "toy"))
    finally:
        log.close()
    return metrics, os.path.join(base, "toy")


def _labels_of(exp_dir, iteration):
    manifest = json.load(open(os.path.join(
        exp_dir, f"iter_{iteration:04d}", "labels.json")))["manifest"]
    return [lab for path in manifest["shards"] for lab in load_shard(path)]


# ---------------------------------------------------------- 1: it runs at all


def test_the_loop_runs_and_writes_every_artefact(tmp_path):
    metrics, exp_dir = _run(tmp_path)

    assert len(metrics) == 2
    for k in range(2):
        it = os.path.join(exp_dir, f"iter_{k:04d}")
        for name in ("labels.json", "agent.pt", "metrics.json", "state.json",
                     "embedding.pt"):
            assert os.path.exists(os.path.join(it, name)), f"iter {k}: {name}"
        assert os.listdir(os.path.join(it, "labels")), "no shards written"
    assert os.path.exists(os.path.join(exp_dir, "report.json"))

    for k, m in enumerate(metrics):
        assert m["iteration"] == k
        assert m["n_labels"] > 0
        assert m["gap"]["n_heldout"] > 0
        assert set(m["gap"]["overall"]) == {"n", *pipeline.GAP_KEYS}
        assert m["gap"]["by_table_size"] and m["gap"]["by_stack_bb"]
        assert 0.0 <= m["gap"]["overall"]["agreement"] <= 1.0
        assert m["gap"]["overall"]["ev_gap_greedy"] >= -1e-12
        assert m["gap_cold"]["n_heldout"] == m["gap"]["n_heldout"]
        assert set(m["gap_cold"]["overall"]) == {"n", *pipeline.GAP_KEYS}
        assert m["gap_cold"]["by_table_size"] and m["gap_cold"]["by_stack_bb"]
        assert m["gap_cold"]["overall"]["ev_gap_greedy"] >= -1e-12


# ------------------------------------------------- 2 / 3c: the pool and rows


def test_the_pool_grows_by_agent_variants_with_a_row_reserved_for_each(tmp_path):
    """D9 and D11: `agent_variants` members per iteration, row *i* is member *i*."""
    cfg = toy_config()
    metrics, exp_dir = _run(tmp_path, cfg)
    variants = cfg["style"]["agent_variants"]

    n_pool0 = metrics[0]["n_pool"]
    assert metrics[1]["n_pool"] == n_pool0 + variants
    final = json.load(open(os.path.join(exp_dir, "iter_0001", "state.json")))
    assert final["n_pool"] == n_pool0 + 2 * variants

    report = json.load(open(os.path.join(exp_dir, "report.json")))
    assert len(report["pool"]) == n_pool0 + 2 * variants
    appended = report["pool"][n_pool0:]
    assert [d["base"] for d in appended] == (
        ["agent0"] * variants + ["agent1"] * variants)
    # Variant 0 of each block is the agent unmodified (D13's rule, applied to
    # the agent); the rest are style draws and must differ from it.
    identity = StyleParams.identity().to_list()
    for lo in (0, variants):
        assert appended[lo]["style"] == identity
        for v in range(1, variants):
            assert appended[lo + v]["style"] != appended[lo]["style"]

    state = torch.load(os.path.join(exp_dir, "iter_0001", "embedding.pt"),
                       map_location="cpu", weights_only=False)
    rows = state["model_state_dict"]["embeddings.weight"].shape[0]
    assert rows == n_pool0 + cfg["embedding_net"]["max_iterations"] * variants


# --------------------------------------------------- 3: who sits in hero's seat


def test_iteration_zero_seats_agent_init_and_iteration_one_seats_the_agent(
        tmp_path, monkeypatch):
    """§7.1, asserted from who was actually asked to act — not from config."""
    seen = {}
    real = pipeline.hero_factory

    def spy(iteration, agent_net, init_member, game, device):
        make = real(iteration, agent_net, init_member, game, device)

        def wrapped(observer_pos, slot_of_seat, embeddings):
            member = make(observer_pos, slot_of_seat, embeddings)
            inner_policy = member.policy

            def policy(contexts):
                seen.setdefault(iteration, set()).add(type(member).__name__)
                return inner_policy(contexts)

            member.policy = policy
            return member
        return wrapped

    monkeypatch.setattr(pipeline, "hero_factory", spy)
    _run(tmp_path)

    assert seen[0] == {"Nit"}, seen
    assert seen[1] == {"AgentPoolMember"}, seen


# --------------------------------------- 3b: the held-out slice and the gap


def test_the_gap_numbers_are_the_ones_section_8_defines():
    """Hand-computed, exactly — the arithmetic is the whole contract."""
    legal = np.array([True, True, True, False])
    q_norm = np.array([1.0, 0.0, -1.0, 0.0])
    pi_oracle = np.array([0.5, 0.3, 0.2, 0.0])
    pi_agent = np.array([0.25, 0.25, 0.5, 0.0])
    log_pi = np.where(legal, np.log(np.where(legal, pi_agent, 1.0)), -np.inf)


    terms = gap_terms(q_norm, pi_oracle, log_pi, legal)
    expect_kl = sum(p * np.log(p / a) for p, a in
                    zip(pi_oracle[:3], pi_agent[:3]))
    assert terms["kl"] == pytest.approx(expect_kl, abs=1e-12)
    # ⟨π_o, Q⟩ = 0.5 − 0.2 = 0.3 ; ⟨π_a, Q⟩ = 0.25 − 0.5 = −0.25
    assert terms["ev_agent"] == pytest.approx(-0.25, abs=1e-12)
    assert terms["ev_oracle"] == pytest.approx(0.3, abs=1e-12)
    # the best legal action, and not the illegal one that ties it at 0.0
    assert terms["q_best"] == pytest.approx(1.0, abs=1e-12)
    assert terms["ev_gap_target"] == pytest.approx(0.55, abs=1e-12)
    assert terms["ev_gap_greedy"] == pytest.approx(1.25, abs=1e-12)
    # the two gaps are differences of the three terms reported beside them
    assert (terms["ev_gap_target"]
            == pytest.approx(terms["ev_oracle"] - terms["ev_agent"], abs=1e-12))
    assert (terms["ev_gap_greedy"]
            == pytest.approx(terms["q_best"] - terms["ev_agent"], abs=1e-12))
    assert terms["agreement"] == 0.0

    # `ev_gap_greedy` is zero exactly when the agent puts all its mass on the
    # oracle's best action. A masked log-softmax is finite on every legal
    # action, so "all the mass" is reached in the limit and not exactly — the
    # residual here is 1e-300, which is what a converged greedy policy looks
    # like in float.
    with np.errstate(divide="ignore"):
        greedy = np.log(np.array([1.0, 1e-300, 1e-300, 0.0]))
        spread = np.log(np.array([0.9, 0.1, 1e-300, 0.0]))
    terms = gap_terms(q_norm, pi_oracle, greedy, legal)
    assert terms["ev_gap_greedy"] == pytest.approx(0.0, abs=1e-12)
    assert terms["agreement"] == 1.0
    # ...and strictly positive as soon as any mass sits anywhere else.
    assert gap_terms(q_norm, pi_oracle, spread, legal)["ev_gap_greedy"] > 0.0


def test_the_heldout_labels_reach_the_metric_and_never_the_optimiser(tmp_path):
    trained = []
    cfg = toy_config()

    real = pipeline.train_agent

    def spy(net, hands, targets, embeddings, *args, **kwargs):
        trained.append(len(hands))
        return real(net, hands, targets, embeddings, *args, **kwargs)

    import pipeline as mod
    mod.train_agent = spy
    try:
        metrics, exp_dir = _run(tmp_path, cfg)
    finally:
        mod.train_agent = real

    for k, m in enumerate(metrics):
        n = m["n_labels"]
        _train_idx, held_idx = split_heldout(n, 0.25, cfg["seed"], k)
        assert m["n_train"] == trained[k] == n - len(held_idx)
        assert m["gap"]["n_heldout"] == len(held_idx) > 0
        assert m["gap"]["overall"]["n"] == len(held_idx)


def test_no_heldout_means_no_gap_rather_than_a_gap_on_training_data(tmp_path):
    cfg = toy_config(n_iterations=1)
    cfg["agent_train"] = dict(cfg["agent_train"], heldout_fraction=0.0)
    metrics, _ = _run(tmp_path, cfg)

    assert metrics[0]["gap"] == metrics[0]["gap_cold"] == {"n_heldout": 0}
    assert metrics[0]["n_train"] == metrics[0]["n_labels"]
    assert "not measured" in winrate_line(metrics[0]["gap"],
                                          metrics[0]["gap_cold"])


def test_phase_d_scores_the_agent_with_the_fitted_vectors_and_with_zero(tmp_path):
    """§12's cold/warm pair, on the held-out slice instead of on Slumbot.

    Phase D reports `ev_agent` twice: conditioned on the vectors §5.5 had
    fitted when hero acted, and with those vectors pinned to zero. The cold
    number has to be *the same measurement* on the same labels, which is pinned
    here by rebuilding it along a path that shares no code with `cold=True` —
    labels whose stored tables are already zero — and by checking that the
    oracle's own side of it (`ev_oracle`, `q_best`) does not move at all, since
    nothing about the agent enters it.
    """
    cfg = toy_config()
    metrics, exp_dir = _run(tmp_path, cfg)

    for m in metrics:
        for key in ("ev_oracle", "q_best"):
            assert m["gap_cold"]["overall"][key] == pytest.approx(
                m["gap"]["overall"][key], abs=1e-12)

    k = 0
    labels = _labels_of(exp_dir, k)
    _train_idx, held_idx = split_heldout(len(labels), 0.25, cfg["seed"], k)
    held = [labels[i] for i in held_idx]
    assert any(np.any(np.asarray(lab["embeddings"]) != 0.0) for lab in held), (
        "the held-out labels carry no fitted vectors, so this run cannot tell "
        "a cold measurement from a warm one")

    state = torch.load(os.path.join(exp_dir, f"iter_{k:04d}", "agent.pt"),
                       map_location="cpu", weights_only=False)
    net = pipeline.frozen_agent_net(state["model_state_dict"], cfg, GAME, "cpu")
    rest = (cfg["oracle"]["temperature"], cfg["oracle"]["divisor"],
            cfg["agent_train"]["batch_hands"], "cpu", lambda _m: None)
    zeroed = [dict(lab, embeddings=np.zeros_like(lab["embeddings"]))
              for lab in held]

    warm = oracle_gap(net, held, *rest)["overall"]
    cold = oracle_gap(net, held, *rest, cold=True)["overall"]
    by_hand = oracle_gap(net, zeroed, *rest)["overall"]

    assert cold["ev_agent"] == pytest.approx(by_hand["ev_agent"], abs=1e-12)
    assert cold["kl"] == pytest.approx(by_hand["kl"], abs=1e-12)
    assert cold["ev_agent"] != warm["ev_agent"]
    assert warm["ev_agent"] == pytest.approx(
        metrics[k]["gap"]["overall"]["ev_agent"], abs=1e-12)
    assert cold["ev_agent"] == pytest.approx(
        metrics[k]["gap_cold"]["overall"]["ev_agent"], abs=1e-12)

    line = winrate_line(metrics[k]["gap"], metrics[k]["gap_cold"])
    assert f"{warm['ev_agent']:+.4f}" in line
    assert f"{cold['ev_agent']:+.4f}" in line


def test_the_heldout_split_is_a_partition(tmp_path):
    train_idx, held_idx = split_heldout(37, 0.25, seed=5, iteration=1)
    assert sorted(train_idx.tolist() + held_idx.tolist()) == list(range(37))
    assert len(held_idx) == 9
    again = split_heldout(37, 0.25, seed=5, iteration=1)
    assert np.array_equal(again[1], held_idx)
    assert not np.array_equal(
        split_heldout(37, 0.25, seed=5, iteration=2)[1], held_idx)


# ------------------------------------------------------------------ 4: resume


def _strip_clocks(obj):
    """Every wall clock out of a JSON artefact.

    A resumed run is expected to produce the same *results*, not to have taken
    the same time to produce them, and `metrics.json` records both. Timings are
    what the owner reads against G3's prediction, so they belong in the file;
    they are simply not part of what "identical" means here.
    """
    if isinstance(obj, dict):
        return {k: _strip_clocks(v) for k, v in obj.items()
                if k not in ("timings", "seconds")}
    if isinstance(obj, list):
        return [_strip_clocks(v) for v in obj]
    return obj


def _artefacts(exp_dir):
    """Every artefact of a run, in a form two runs can be compared on."""
    out = {}
    for root, _dirs, files in os.walk(exp_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, exp_dir)
            if name.endswith(".pt"):
                # `torch.save`'s container is compared through its tensors: the
                # claim being tested is that the weights are the same, and a
                # byte comparison of a zip would also be testing the serialiser.
                state = torch.load(path, map_location="cpu",
                                   weights_only=False)["model_state_dict"]
                out[rel] = {k: v.clone() for k, v in state.items()}
            elif name.endswith(".json"):
                blob = _strip_clocks(json.load(open(path)))
                if "config" in blob:
                    blob["config"]["out_dir"] = ""
                # Shard paths and the config's `out_dir` say where the run was
                # written, and the two runs being compared are in different
                # directories by construction.
                out[rel] = json.dumps(blob, sort_keys=True).replace(
                    exp_dir, "<exp>").encode()
            else:
                out[rel] = open(path, "rb").read()
    return out


def test_resume_from_an_interrupted_phase_reproduces_an_uninterrupted_run(
        tmp_path):
    """A crash during agent training must cost the training, not the labels."""
    _metrics, whole = _run(tmp_path, name="whole")

    # A run that got as far as iteration 0's labels and then died.
    _metrics, part = _run(tmp_path, toy_config(n_iterations=1), name="part")
    it0 = os.path.join(part, "iter_0000")
    for name in ("state.json", "metrics.json", "agent.pt"):
        os.remove(os.path.join(it0, name))
    shutil.rmtree(os.path.join(part, "iter_0001"), ignore_errors=True)
    os.remove(os.path.join(part, "report.json"))

    cfg = toy_config()
    cfg["out_dir"] = str(tmp_path / "part")
    log = Logger(cfg["out_dir"])
    try:
        run(cfg, lambda _m: None, part)
    finally:
        log.close()

    a, b = _artefacts(whole), _artefacts(part)
    assert sorted(a) == sorted(b)
    for name in a:
        if name.endswith(".pt"):
            assert sorted(a[name]) == sorted(b[name]), name
            for key in a[name]:
                assert torch.equal(a[name][key], b[name][key]), f"{name}:{key}"
        else:
            assert a[name] == b[name], name


# -------------------------------------------------- 5: the CLAUDE.md §1 rule


def test_table_size_and_stack_depth_span_their_full_ranges_unweighted(tmp_path):
    """The exact multiset, from the sessions the loop says it played."""
    cfg = toy_config(n_iterations=2, n_sessions=40)
    metrics, exp_dir = _run(tmp_path, cfg)

    for k in range(2):
        labels = _labels_of(exp_dir, k)
        sessions = build_sessions(
            np.random.default_rng(pipeline._iteration_seed(cfg["seed"], k)),
            list(range(metrics[k]["n_pool"])), GAME, cfg["n_sessions"],
            cfg["hands_per_session"],
            seed_base=pipeline._iteration_seed(cfg["seed"], k) * 1_000_000,
            tag="labels")
        for lab in labels:
            s = sessions[lab["session"]]
            assert lab["num_players"] == s.num_players
            assert lab["stack_bb"] == s.stack_bb
        sizes = {s.num_players for s in sessions}
        stacks = [s.stack_bb for s in sessions]
        assert sizes == set(range(2, 10)), sizes
        assert min(stacks) < 60 and max(stacks) > 250


# ------------------------------- 6: a past agent, seated as an ordinary member


def _played_hand():
    """One nine-handed hand, played out and showdown-labelled."""
    from tests.g1_fixtures import make_pool, make_specs, play as play_hands
    pool = make_pool(seed=1)
    specs = make_specs(seed=2, n_hands=1, n_members=len(pool), num_players=9,
                       stack_bb=100)
    return pool, play_hands(pool, specs)[0]


def _frozen_member():
    from agent.policy import FrozenAgentMember
    from nets.agent_net import AgentNet
    torch.manual_seed(0)
    net = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS).eval()
    return FrozenAgentMember("agent0", N_ACTIONS, net, MAX_PLAYERS, "cpu")


def test_a_past_agent_answers_the_question_the_posterior_asks_of_a_member():
    """D12: the §7.2 posterior asks every member "holding *this*, what then?".

    A member that ignores the override answers about the real hand, every combo
    gets the same likelihood, and the posterior of every seat it occupies
    silently stays at the prior — the exact failure `pool/v7_member.py` records.
    """
    from env.driver import DecisionContext
    from oracle.posterior import opponent_posterior

    _pool, record = _played_hand()
    member = _frozen_member()
    dec = record.decisions[0]
    seat = int(dec["acting_pos"])

    def ctx_with(cards):
        return DecisionContext(record, dec["snap_idx"], seat,
                               dec["legal_mask"], 0, hole_override=cards)

    logits = member.logits([ctx_with([0, 1]), ctx_with([50, 51]),
                            ctx_with([0, 1])])
    assert np.allclose(logits[0], logits[2]), "the same holding, twice"
    assert not np.allclose(logits[0], logits[1]), (
        "two different holdings produced the same answer — the override was "
        "not read")

    pool = [member] * (max(d["member"] for d in record.decisions) + 1)
    combos, weights = opponent_posterior(
        record, opp_pos=seat, observer_pos=(seat + 1) % 9, pool=pool,
        n_actions=N_ACTIONS, through_decision=0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert weights.std() > 0.0, "the posterior never moved off the prior"


def test_a_past_agent_observes_the_moment_it_is_asked_about_and_no_later_one():
    """§9, for a member handed a decision from the middle of a finished hand."""
    from env.driver import DecisionContext
    from nets.features import TOKEN_DECISION, UNKNOWN_CARD

    _pool, record = _played_hand()
    assert record.showdown, "this fixture is meant to reach a showdown"
    member = _frozen_member()

    for t, dec in enumerate(record.decisions):
        seat = int(dec["acting_pos"])
        ctx = DecisionContext(record, dec["snap_idx"], seat, dec["legal_mask"],
                              0, hole_override=[0, 1])
        tokens = member._observation(ctx)

        assert (tokens.token_type == TOKEN_DECISION).all(), (
            "a hand still in progress has no showdown token")
        assert len(tokens) == t + 1, "the observation ran past the decision"
        assert int(tokens.action[-1]) == -1, "the pending decision has no action"
        assert list(tokens.action[:-1]) == [
            int(d["action_idx"]) for d in record.decisions[:t]]
        # Hero's own (overridden) cards and nobody else's.
        assert list(tokens.cards[-1, 5:]) == [0, 1]
        for u in range(t):
            if int(tokens.acting_pos[u]) != seat:
                assert list(tokens.cards[u, 5:]) == [UNKNOWN_CARD] * 2


def test_a_crash_in_the_middle_of_labelling_costs_a_shard_and_not_the_phase(
        tmp_path, monkeypatch):
    """`./run.sh --version=v8` again, and the run carries on where it stopped.

    Labelling is the phase measured in days (`CONCEPT.md` §13), so this is the
    crash that matters. The artefacts a resumed run leaves must be the ones an
    uninterrupted run would have left — not merely valid ones.
    """
    import train.generate

    # Small shards, so the crash lands after at least one of them is safely on
    # disk — a shard is the unit a crash can cost, and this test is about what
    # it must *not* cost.
    _metrics, whole = _run(tmp_path, toy_config(labels_per_shard=3),
                           name="whole")

    calls = {"n": 0}
    real = train.generate.action_values_batch

    def crash_after(requests, *args, **kwargs):
        calls["n"] += len(requests)
        if calls["n"] > 7:
            raise RuntimeError("the box went away")
        return real(requests, *args, **kwargs)

    cfg = toy_config(labels_per_shard=3)
    cfg["out_dir"] = str(tmp_path / "part")
    part = os.path.join(cfg["out_dir"], "toy")
    log = Logger(cfg["out_dir"])
    monkeypatch.setattr(train.generate, "action_values_batch",
                        crash_after)
    try:
        with pytest.raises(RuntimeError, match="the box went away"):
            run(cfg, lambda _m: None, part)
    finally:
        log.close()
    monkeypatch.setattr(train.generate, "action_values_batch", real)

    it0 = os.path.join(part, "iter_0000", "labels")
    assert os.path.exists(os.path.join(it0, "progress.json"))
    assert not os.path.exists(os.path.join(part, "iter_0000", "labels.json")), (
        "the phase never finished, so its manifest must not exist")

    log = Logger(cfg["out_dir"])
    try:
        run(cfg, lambda _m: None, part)
    finally:
        log.close()

    a, b = _artefacts(whole), _artefacts(part)
    assert sorted(a) == sorted(b)
    for name in a:
        if name.endswith(".pt"):
            for key in a[name]:
                assert torch.equal(a[name][key], b[name][key]), f"{name}:{key}"
        elif not name.endswith("progress.json"):
            assert a[name] == b[name], name


# ------------- 7: a past agent carries the network of its own generation


"""Whose vectors a past agent reads (`PLAN_AMORTISED_POOL.md` §0.2, ⚠11).

Nothing anchors the coordinates of the embedding space, and the loop keeps
training the one network it holds. A past agent frozen at iteration *j* was
trained to read the vectors *that* network produced, so it carries a frozen copy
of it and reads through that copy for the rest of the run. Handing it the live
network instead would be two silent failures at once: a frozen policy reading a
basis that drifts under it, and a pool whose members play differently at
iteration 10 and at iteration 20 with nobody having changed them.
"""


def _conditioned_config(**overrides):
    cfg = toy_config(**overrides)
    cfg["embedding_net"]["pool_agent_vectors"] = "amortised"
    cfg["embedding_net"]["pool_agent_window"] = None
    return cfg


def _generations(monkeypatch):
    """`{iteration: the embedding network its members were given}`."""
    import pipeline as pipe
    seen = {}
    real = pipe.agent_variant_members

    def spy(net, iteration, config, game, device, seed, embed_net=None):
        seen[int(iteration)] = embed_net
        return real(net, iteration, config, game, device, seed,
                    embed_net=embed_net)

    monkeypatch.setattr(pipe, "agent_variant_members", spy)
    return seen


def test_every_past_agent_carries_a_frozen_copy_of_its_own_generation(
        tmp_path, monkeypatch):
    """Two iterations, one retrain each, so the two agents differ by generation."""
    import torch as _torch

    seen = _generations(monkeypatch)
    cfg = _conditioned_config(n_iterations=2)
    _metrics, exp_dir = _run(tmp_path, cfg)

    assert set(seen) == {0, 1}
    for k, net in seen.items():
        assert net is not None, f"iteration {k} was given no generation"
        assert not any(p.requires_grad for p in net.parameters()), (
            "the snapshot is trainable, so a later phase could move it")
        on_disk = _torch.load(os.path.join(exp_dir, f"iter_{k:04d}",
                                           "embedding.pt"),
                              map_location="cpu",
                              weights_only=False)["model_state_dict"]
        for name, tensor in net.state_dict().items():
            assert _torch.equal(tensor, on_disk[name]), (
                f"iteration {k}'s snapshot is not the network of that "
                f"iteration: {name} differs")
    assert seen[0] is not seen[1], (
        "both agents were given one object, so the earlier one is reading a "
        "network that was trained further after it was frozen")


def test_iterations_that_share_a_retrain_share_one_generation(
        tmp_path, monkeypatch):
    """The network is retrained every `retrain_every` iterations, so the agents
    in between were trained against the same one and must share it — one object,
    one inference runner when they are mirrored into a label worker."""
    seen = _generations(monkeypatch)
    cfg = _conditioned_config(n_iterations=2)
    cfg["embedding_net"]["retrain_every"] = 2
    _metrics, _exp = _run(tmp_path, cfg)

    assert seen[0] is seen[1] and seen[0] is not None


def test_with_the_switch_off_no_generation_is_kept(tmp_path, monkeypatch):
    """Freezing a 50M-parameter copy per generation to condition nobody is
    minutes and gigabytes spent on nothing."""
    seen = _generations(monkeypatch)
    _metrics, _exp = _run(tmp_path, toy_config(n_iterations=2))
    assert set(seen) == {0, 1}
    assert all(net is None for net in seen.values())


def test_a_resumed_run_gives_every_past_agent_the_same_generation(
        tmp_path, monkeypatch):
    """The mapping is rebuilt from the checkpoints, not guessed at.

    A resumed run reconstructs the pool from disk; if it handed those members
    the *current* network, the pool would change under a restart — the same
    corpus, resumed, would be played against different opponents.
    """
    import pipeline as pipe
    import torch as _torch

    cfg = _conditioned_config(n_iterations=2)
    _metrics, exp_dir = _run(tmp_path, cfg)

    # A second call over the finished directory resumes: every iteration is
    # already on disk, so the whole pool is rebuilt and nothing is trained.
    seen = _generations(monkeypatch)
    vintages = {}
    real = pipe.embedding_vintages

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        vintages.update(out)
        return out

    monkeypatch.setattr(pipe, "embedding_vintages", spy)
    cfg2 = _conditioned_config(n_iterations=2)
    cfg2["out_dir"] = os.path.dirname(exp_dir)
    log = Logger(cfg2["out_dir"])
    try:
        run(cfg2, lambda _m: None, exp_dir)
    finally:
        log.close()

    assert set(vintages) == {0, 1}, "the pool was not rebuilt from disk"
    for k, net in vintages.items():
        assert net is not None
        on_disk = _torch.load(os.path.join(exp_dir, f"iter_{k:04d}",
                                           "embedding.pt"),
                              map_location="cpu",
                              weights_only=False)["model_state_dict"]
        for name, tensor in net.state_dict().items():
            assert _torch.equal(tensor, on_disk[name]), (
                f"the resumed run gave iteration {k} another generation")


def test_the_whole_loop_runs_with_the_pool_reading_its_tablemates(tmp_path):
    """End to end with the switch on: three iterations, every artefact written.

    Iteration 0 has no past agent in the pool at all, iteration 1 has one and
    iteration 2 has two — so this is also the first run in which a conditioned
    member is seated, refreshed and labelled against.
    """
    cfg = _conditioned_config(n_iterations=3)
    cfg["embedding_net"]["max_iterations"] = 3
    metrics, exp_dir = _run(tmp_path, cfg)

    assert len(metrics) == 3
    for k in range(3):
        it_dir = os.path.join(exp_dir, f"iter_{k:04d}")
        for name in ("agent.pt", "metrics.json", "state.json"):
            assert os.path.exists(os.path.join(it_dir, name)), (k, name)
        labels = _labels_of(exp_dir, k)
        assert labels, f"iteration {k} labelled nothing"
    # The pool grew by `agent_variants` members an iteration, as it always does.
    assert metrics[2]["n_pool"] - metrics[0]["n_pool"] == \
        2 * int(cfg["style"]["agent_variants"])

    # ... and a past agent really was seated and conditioned, or the run above
    # exercised the switch and nothing else. The phase widens its table of
    # vectors to one row per slot exactly when it has a conditioned seat to
    # fill, so the shape of what it stored is the evidence.
    widened = []
    for k in range(3):
        with np.load(os.path.join(exp_dir, f"iter_{k:04d}", "labels",
                                  "vectors.npz")) as z:
            widened.append(int(z["vectors"].shape[2]))
    assert widened[0] == 1, "iteration 0 has no past agent to condition"
    assert max(widened[1:]) > 1, (
        "no past agent was ever seated, so the loop ran the switch on and "
        "conditioned nobody")
