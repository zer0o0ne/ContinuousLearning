"""The G1 gate end to end (CONCEPT.md §14).

Toy scale — a few dozen hands, a two-layer network, a handful of gradient
steps — so nothing here says anything about whether the embedding works. What it
pins is that the experiment is the experiment §14 asks for: sessions that rotate
the button, uniform sampling over table size and stack depth, a fresh style set
that was never trained on, and the four measurements coming out with the shapes
they are supposed to have.
"""

import json
import math
import pickle

import numpy as np
import pytest

from gates.g1 import (
    Session, _decisions_by_slot, aggregate, build_sessions, fit_variants,
    metrics_of, run, split_bases,
)
from tests.g1_fixtures import RAISE_SIZES

GAME = {
    "n_actions": 6,
    "max_players": 9,
    "big_blind": 10,
    "small_blind": 5,
    "players_range": [2, 9],
    "stack_bb_range": [10, 300],
    "raise_sizes": {"preflop": RAISE_SIZES[0], "flop": RAISE_SIZES[1],
                    "turn": RAISE_SIZES[2], "river": RAISE_SIZES[3]},
}


def _config():
    return {
        "seed": 5,
        "device": "cpu",
        "game": GAME,
        "style": {"uncond_scale": 0.8, "position_scale": 0.5,
                  "street_scale": 0.5, "log_temperature_range": [-0.4, 0.4],
                  "uniform_mix_range": [0.0, 0.2]},
        "bootstrap": [
            {"kind": "degenerate", "strategy": s, "style": "identity"}
            for s in ("always_fold", "always_call", "always_min_raise",
                      "maniac", "nit")
        ] + [
            {"kind": "degenerate", "strategy": s, "n_variants": 3,
             "label": f"{s}_styles"}
            for s in ("always_call", "maniac", "nit")
        ],
        "corpus": {
            "train_sessions": 10, "train_hands_per_session": 6,
            "eval_sessions": 3, "eval_hands_per_session": 4,
            "observed_hand_counts": [0, 2, 4],
            "n_unseen_members": 10, "driver_batch_size": 32,
        },
        "embedding_net": {
            "d_model": 32, "d_emb": 8, "n_heads": 4, "n_kv_heads": 2,
            "n_layers": 1, "d_ff": 64, "d_card": 8, "d_index": 8,
            "max_decisions": 64, "K": 4, "fit_lr": 0.1, "fit_reg": 0.01,
        },
        "train": {"steps": 5, "batch_hands": 4, "lr": 0.001,
                  "amortised_weight": 1.0, "log_every": 100},
    }


# ------------------------------------------------------------------- sessions


def test_a_session_rotates_the_button():
    """Without rotation a member would be identifiable by its seat, and the
    network would learn seats instead of styles."""
    rng = np.random.default_rng(0)
    sessions = build_sessions(rng, list(range(12)), GAME, 8, 6,
                              seed_base=1, tag="t")
    rotated = 0
    for s in sessions:
        seats = {m: set() for m in s.members}
        for h, spec in enumerate(s.specs):
            for seat, member in enumerate(spec.seat_members):
                seats[member].add(seat)
            assert sorted(spec.seat_members) == sorted(s.members), (
                "every hand of a session seats exactly the same members")
        if all(len(v) > 1 for v in seats.values()):
            rotated += 1
    assert rotated == len(sessions)


def test_the_observer_seat_tracks_slot_zero_through_the_rotation():
    session = Session(idx=0, num_players=5, stack_bb=100,
                      members=[0, 1, 2, 3, 4])
    for h in range(12):
        seat = session.seat_of_slot(0, h)
        assert session.slot_of_seat(h)[seat] == 0
        assert [session.slot_of_seat(h)[session.seat_of_slot(k, h)]
                for k in range(5)] == list(range(5))


def test_table_size_and_stack_depth_span_their_full_ranges():
    """`CLAUDE.md` §1: uniform over 2–9 players and 10–300 BB, no weighting
    toward heads-up or 200 BB."""
    rng = np.random.default_rng(1)
    sessions = build_sessions(rng, list(range(12)), GAME, 400, 2,
                              seed_base=1, tag="t")
    sizes = sorted({s.num_players for s in sessions})
    stacks = [s.stack_bb for s in sessions]
    assert sizes == list(range(2, 10))
    assert min(stacks) < 40 and max(stacks) > 270
    counts = np.bincount([s.num_players for s in sessions], minlength=10)[2:10]
    assert counts.min() > 0.5 * counts.mean(), "table size is not near-uniform"


def test_a_session_wider_than_the_member_set_is_refused_loudly():
    rng = np.random.default_rng(2)
    with pytest.raises(AssertionError, match="distinct pool members"):
        build_sessions(rng, [0, 1], GAME, 200, 2, seed_base=1, tag="t")


# ---------------------------------------------------------------- aggregation


def test_the_report_carries_every_measurement_section(tmp_path):
    report = run(_config(), lambda _m: None, str(tmp_path))

    assert set(report["curves"]) == {"seen", "unseen"}
    assert "style_generalisation_gap" in report          # §14.2
    assert set(report["ablation"]) == {"seen", "unseen"}  # §14.3
    assert report["by_table_size"] and report["by_stack_depth"]  # §14.4

    for curve in report["curves"].values():
        assert sorted(curve) == [0, 2, 4]
        for point in curve.values():
            for metric in ("ce_zero", "ce_ablation", "ce_fit", "gain_fit"):
                assert set(point[metric]) == {"n", "mean", "se"}

    payload = json.loads((tmp_path / "g1_report.json").read_text())
    assert payload["fresh_style_draws"], "the B1(b) set must be recorded"
    assert len(payload["pool"][0]["style"]) == 32
    assert payload["rows"]


def test_with_no_observed_hands_the_fit_is_exactly_the_zero_baseline(tmp_path):
    """§5.5 cold start: no history, so the vector is zero and nothing may
    quietly fall back on the training-time embedding table."""
    report = run(_config(), lambda _m: None, str(tmp_path))
    for curve in report["curves"].values():
        p = curve[0]
        assert p["ce_fit"]["mean"] == pytest.approx(p["ce_zero"]["mean"])
        assert p["ce_ablation"]["mean"] == pytest.approx(p["ce_zero"]["mean"])
        assert p["gain_fit"]["mean"] == pytest.approx(0.0, abs=1e-9)


def test_the_gain_metric_is_the_improvement_over_the_zero_baseline():
    rows = [
        {"set": "seen", "session": 0, "num_players": 2, "stack_bb": 100,
         "observed_hands": 4, "ce_zero": 1.0, "ce_ablation": 0.9,
         "ce_fit": 0.7},
        {"set": "seen", "session": 1, "num_players": 9, "stack_bb": 20,
         "observed_hands": 4, "ce_zero": 2.0, "ce_ablation": 2.0,
         "ce_fit": 1.6},
    ]
    report = aggregate(rows)
    point = report["curves"]["seen"][4]
    assert point["gain_fit"]["mean"] == pytest.approx((0.3 + 0.4) / 2)
    assert point["ce_fit"]["mean"] == pytest.approx(1.15)
    assert report["ablation"]["seen"][4]["fit_minus_ablation"] == \
        pytest.approx(1.15 - 1.45)
    assert set(report["by_table_size"]["seen"]) == {"2", "9"}
    assert set(report["by_stack_depth"]["seen"]) == {"41-100", "10-40"}


def test_rows_of_one_session_are_averaged_before_the_standard_error():
    """Rows from one session share their evaluation hands, so they are not
    independent samples and must not be counted as such."""
    rows = [
        {"set": "seen", "session": 0, "num_players": 4, "stack_bb": 100,
         "observed_hands": 1, "ce_zero": 1.0, "ce_ablation": 1.0, "ce_fit": c}
        for c in (0.6, 0.8, 1.0)
    ] + [
        {"set": "seen", "session": 1, "num_players": 4, "stack_bb": 100,
         "observed_hands": 1, "ce_zero": 1.0, "ce_ablation": 1.0, "ce_fit": 1.2}
    ]
    point = aggregate(rows)["curves"]["seen"][1]
    assert point["ce_fit"]["n"] == 2, "the unit of the SE is the session"
    assert point["ce_fit"]["mean"] == pytest.approx((0.8 + 1.2) / 2)
    assert point["ce_fit"]["se"] == pytest.approx(0.2)


def test_a_single_session_reports_no_standard_error_rather_than_zero():
    rows = [{"set": "seen", "session": 0, "num_players": 4, "stack_bb": 100,
             "observed_hands": 1, "ce_zero": 1.0, "ce_ablation": 1.0,
             "ce_fit": 0.9}]
    point = aggregate(rows)["curves"]["seen"][1]
    assert point["ce_fit"]["n"] == 1
    assert math.isnan(point["ce_fit"]["se"])


def test_the_unseen_set_draws_from_every_base_not_just_the_first_few():
    """§14.2 compares fresh style draws against the trained ones, so the two
    sets have to be built from the same bases.

    `fresh_style_variants` used to cycle over *members*. A base that expanded
    into many style variants occupies many consecutive member slots, so the
    later bases were never reached — in the first pilot the unseen set ended up
    with no network member at all, and §14.2 was comparing a mixed set against
    an all-degenerate one.
    """
    from collections import Counter

    import numpy as np

    from pool.build import build_pool, fresh_style_variants

    config = _config()
    rng = np.random.default_rng(0)
    members, descriptors = build_pool(config, rng)
    bases = {d["base"] for d in descriptors}
    assert len(bases) < len(members), "the fixture must have expanded bases"

    _fresh, fresh_desc = fresh_style_variants(
        members, descriptors, len(bases), rng, config["style"], tag="fresh")
    drawn = Counter(d["base"] for d in fresh_desc)
    assert set(drawn) == bases, (
        f"bases missing from the unseen set: {sorted(bases - set(drawn))}")
    assert set(drawn.values()) == {1}, "one draw per base when n == n_bases"


def test_every_fresh_draw_is_a_new_style_on_a_shared_base():
    import numpy as np

    from pool.build import build_pool, fresh_style_variants

    config = _config()
    rng = np.random.default_rng(1)
    members, descriptors = build_pool(config, rng)
    fresh, fresh_desc = fresh_style_variants(
        members, descriptors, 12, rng, config["style"], tag="fresh")

    trained_styles = {tuple(d["style"]) for d in descriptors}
    for d in fresh_desc:
        assert tuple(d["style"]) not in trained_styles, (
            "a fresh draw reproduced a style the network was trained on")
    assert len({tuple(d["style"]) for d in fresh_desc}) == len(fresh)
    assert len(fresh) == 12


# ---------------------------------------------------- C1: held-out base sets


HOLDOUT = ["nit", "nit_styles"]


def _c1_config():
    """The §14 fixture plus a base partition and every extra condition on.

    The bases are widened first: `split_bases` refuses a partition whose either
    half cannot seat a 9-handed table, and `CLAUDE.md` §1 does not allow
    narrowing the table-size range to get around that.
    """
    config = _config()
    config["bootstrap"] = [
        {"kind": "degenerate", "strategy": s, "style": "identity"}
        for s in ("always_fold", "always_call", "always_min_raise", "maniac",
                  "nit")
    ] + [
        {"kind": "degenerate", "strategy": "always_call", "n_variants": 6,
         "label": "call_styles"},
        {"kind": "degenerate", "strategy": "maniac", "n_variants": 6,
         "label": "maniac_styles"},
        {"kind": "degenerate", "strategy": "nit", "n_variants": 10,
         "label": "nit_styles"},
    ]
    config["corpus"].update({"holdout_bases": list(HOLDOUT),
                             "save_eval_corpus": True})
    config["eval_conditions"] = {
        "oracle_embedding": True, "zero_init_fit": True,
        "no_showdown_fit": True, "fit_steps_sweep": [1, 8],
        "showdown_holdout": True, "save_fitted_vectors": True,
    }
    config["train"].update({"showdown_strength_weight": 0.3,
                            "showdown_class_weight": 0.1})
    return config


@pytest.fixture(scope="module")
def c1_run(tmp_path_factory):
    """One toy run with C1 and every B condition on, shared by the tests below.

    Toy scale, so no number here means anything — what is pinned is that the
    partition holds, that every switched-on condition reaches the report, and
    that the side files are written.
    """
    out = tmp_path_factory.mktemp("c1")
    report = run(_c1_config(), lambda _m: None, str(out))
    payload = json.loads((out / "g1_report.json").read_text())
    return report, payload, out


def test_a_held_out_base_is_partitioned_out_of_the_trainable_members():
    from pool.build import build_pool

    config = _c1_config()
    members, descriptors = build_pool(config, np.random.default_rng(0))
    train_ids, holdout_ids = split_bases(descriptors, HOLDOUT, 9,
                                         lambda _m: None)

    assert set(train_ids) & set(holdout_ids) == set()
    assert sorted(train_ids + holdout_ids) == list(range(len(members)))
    assert {descriptors[i]["base"] for i in holdout_ids} == set(HOLDOUT)
    assert not {descriptors[i]["base"] for i in train_ids} & set(HOLDOUT)


def test_a_partition_that_cannot_seat_a_full_table_is_refused_loudly():
    from pool.build import build_pool

    config = _c1_config()
    _members, descriptors = build_pool(config, np.random.default_rng(0))
    # Everything but the five identity degenerates: five members cannot seat a
    # 9-handed session, and narrowing the table range is not an option.
    everything = ["call_styles", "maniac_styles", "nit_styles"]
    with pytest.raises(AssertionError, match="distinct ones"):
        split_bases(descriptors, everything, 9, lambda _m: None)


def test_a_holdout_base_that_is_not_a_base_is_refused_loudly():
    from pool.build import build_pool

    _members, descriptors = build_pool(_c1_config(), np.random.default_rng(0))
    with pytest.raises(AssertionError, match="not bases of this pool"):
        split_bases(descriptors, ["v7_typo"], 9, lambda _m: None)


def test_the_held_out_set_is_scored_and_is_made_only_of_held_out_members(c1_run):
    report, payload, _out = c1_run
    assert set(report["curves"]) == {"seen", "unseen", "heldout"}
    assert payload["holdout_bases"] == HOLDOUT

    desc = payload["pool"] + payload["fresh_style_draws"]
    by_set = {}
    for row in payload["rows"]:
        by_set.setdefault(row["set"], set()).add(desc[row["member"]]["base"])
    assert by_set["heldout"] <= set(HOLDOUT)
    assert not by_set["seen"] & set(HOLDOUT)
    assert not by_set["unseen"] & set(HOLDOUT), (
        "a fresh style draw was taken off a held-out base, which confounds "
        "§14.2 with C1")


def test_every_other_set_is_compared_against_seen(c1_run):
    report, _payload, _out = c1_run
    gaps = report["style_generalisation_gap"]
    assert set(gaps) == {"unseen", "heldout"}
    for gap in gaps.values():
        for point in gap.values():
            assert {"ce_fit", "gain_fit"} <= set(point)


# -------------------------------------------------- B: the extra conditions


def test_the_metric_list_follows_the_rows_not_a_constant():
    rows = [{"set": "seen", "session": 0, "num_players": 2, "stack_bb": 100,
             "observed_hands": 1, "ce_zero": 2.0, "ce_ablation": 1.8,
             "ce_fit": 1.5, "ce_oracle": 1.0}]
    assert metrics_of(rows) == (
        "ce_zero", "ce_ablation", "ce_fit", "ce_oracle",
        "gain_ablation", "gain_fit", "gain_oracle")
    point = aggregate(rows)["curves"]["seen"][1]
    assert point["gain_oracle"]["mean"] == pytest.approx(1.0)
    assert point["gain_fit"]["mean"] == pytest.approx(0.5)


def test_the_named_fit_variants_follow_the_switches():
    assert fit_variants({"K": 50}) == ["fit"]
    cfg = {"K": 50, "eval_conditions": {
        "fit_steps_sweep": [10, 50, 200], "zero_init_fit": True,
        "no_showdown_fit": True}}
    # 50 is the baseline `K` and is already measured as `fit`; it must not be
    # measured a second time under another name.
    assert fit_variants(cfg) == ["fit", "fit_k10", "fit_k200",
                                 "fit_zero_init", "fit_no_showdown"]


def test_every_switched_on_condition_reaches_the_report(c1_run):
    report, payload, _out = c1_run
    expected = {"ce_zero", "ce_ablation", "ce_fit", "ce_fit_k1", "ce_fit_k8",
                "ce_fit_zero_init", "ce_fit_no_showdown", "ce_oracle"}
    assert expected <= set(payload["rows"][0])
    assert expected <= set(report["metrics"])
    for curve in report["curves"].values():
        for point in curve.values():
            for metric in expected:
                assert set(point[metric]) == {"n", "mean", "se"}


def test_the_extra_conditions_are_confined_to_their_windows(tmp_path):
    """Each extra condition is another fit at every window, and that is the
    whole cost of the gate. Outside `condition_windows` only the §5.5 baseline
    runs, and the report has to say "not measured" rather than quietly average
    over the windows where it was."""
    config = _c1_config()
    config["corpus"]["save_eval_corpus"] = False
    config["eval_conditions"]["condition_windows"] = [4]
    report = run(config, lambda _m: None, str(tmp_path))
    payload = json.loads((tmp_path / "g1_report.json").read_text())

    for row in payload["rows"]:
        assert ("ce_fit_k8" in row) == (row["observed_hands"] == 4)
        assert "ce_fit" in row, "the baseline fit runs at every window"
        assert "ce_oracle" in row, "the oracle is free of the window"

    curve = report["curves"]["seen"]
    assert curve[2]["gain_fit_k8"]["n"] == 0
    assert curve[2]["gain_fit"]["n"] > 0
    assert curve[4]["gain_fit_k8"]["n"] > 0


def test_the_oracle_condition_does_not_move_with_the_observation_window(c1_run):
    """It is the trained table row, which no amount of observation changes. If
    it ever varied with `n`, the fit would be leaking into it."""
    report, _payload, _out = c1_run
    for curve in report["curves"].values():
        means = {round(point["ce_oracle"]["mean"], 10) for point in
                 curve.values()}
        assert len(means) == 1


def test_the_cold_start_leaves_every_fitted_condition_at_the_baseline(c1_run):
    """§5.5: with no history there is no vector, so every fit — whatever its
    step count or its initialisation — must be exactly `e = 0`."""
    report, _payload, _out = c1_run
    for curve in report["curves"].values():
        point = curve[0]
        for metric in ("ce_ablation", "ce_fit", "ce_fit_k1", "ce_fit_k8",
                       "ce_fit_zero_init", "ce_fit_no_showdown"):
            assert point[metric]["mean"] == pytest.approx(
                point["ce_zero"]["mean"])


def test_the_per_slot_observation_budget_counts_only_that_slots_decisions():
    """§14.4's table-size breakdown is confounded by how much of each opponent
    the fit saw, and the table-wide count cannot say."""
    from nets.features import TOKEN_DECISION, TOKEN_SHOWDOWN

    class _Tokens:
        def __init__(self, slots, kinds):
            self.slot = np.array(slots, dtype=np.int64)
            self.token_type = np.array(kinds, dtype=np.int64)

    observed = [
        _Tokens([0, 1, 1, 2], [TOKEN_DECISION] * 4),
        _Tokens([1, 2, 2], [TOKEN_DECISION, TOKEN_DECISION, TOKEN_SHOWDOWN]),
    ]
    assert _decisions_by_slot(observed, 3) == [1, 3, 2]
    assert _decisions_by_slot([], 3) == [0, 0, 0]


def test_the_showdown_heads_are_scored_on_held_out_hands(c1_run):
    """The training curve is printed on the corpus being fitted, where a final
    board is very nearly a unique key; the held-out number is what says whether
    the head found a signal or a lookup."""
    report, _payload, _out = c1_run
    holdout = report["showdown_holdout"]
    assert set(holdout) == {"seen", "unseen", "heldout"}
    for v in holdout.values():
        assert v["n_tokens"] > 0 and v["sessions"] > 0
        assert v["showdown_class_ce"] > 0.0
        assert v["showdown_strength_mse"] >= 0.0


def test_the_fitted_vectors_are_written_for_every_seat(c1_run):
    """Hero included: hero's vector is the control for "does a fitted vector
    say who this is", and the descriptors carry every member's true style."""
    _report, payload, out = c1_run
    saved = np.load(out / "fitted_vectors.npz")
    assert saved["vector"].shape[1] == _c1_config()["embedding_net"]["d_emb"]
    assert set(np.unique(saved["set"])) == {"seen", "unseen", "heldout"}
    assert 0 in set(np.unique(saved["slot"])), "hero's vector is not saved"
    assert set(np.unique(saved["observed_hands"])) == \
        set(payload["config"]["corpus"]["observed_hand_counts"])


def test_the_saved_eval_corpus_is_enough_to_re_evaluate_without_replay(c1_run):
    """Everything measured after training is a function of these tokens and the
    checkpoint, so a changed `eval_conditions` switch must not need the pool."""
    _report, payload, out = c1_run
    with open(out / "eval_corpus.pkl", "rb") as fh:
        corpus = pickle.load(fh)

    corpus_cfg = payload["config"]["corpus"]
    n_hands = max(corpus_cfg["observed_hand_counts"]) + \
        corpus_cfg["eval_hands_per_session"]
    assert set(corpus) == {"seen", "unseen", "heldout"}
    for tag, sessions in corpus.items():
        assert len(sessions) == corpus_cfg["eval_sessions"]
        for s in sessions:
            assert len(s["tokens"]) == n_hands
            assert len(s["members"]) == s["num_players"]
            seen_members = {int(m) for t in s["tokens"] for m in t.member}
            assert seen_members <= set(s["members"])


def test_the_timings_cover_every_phase_of_the_run(c1_run):
    """Every cost figure for the Spark is a hypothesis until it has been run
    there (`CLAUDE.md` §3), and the run is where the measurement is free."""
    report, _payload, _out = c1_run
    assert set(report["timings"]) == {
        "play:train", "play:eval", "train",
        "eval:seen", "eval:unseen", "eval:heldout"}
    assert all(v >= 0.0 for v in report["timings"].values())
