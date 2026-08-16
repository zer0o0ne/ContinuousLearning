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

import numpy as np
import pytest

from gates.g1 import Session, aggregate, build_sessions, run
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
