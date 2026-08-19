"""The G3 gate end to end (`CONCEPT.md` §14, `PLAN_PIPELINE.md` S4).

Toy scale — a handful of hands, two samples per action, a degenerate pool — so
nothing here says anything about what a label costs; that number only exists
once the gate has run on the Spark. What is pinned is that the experiment is the
one §14 asks for: a cell per point of the sweep, a cost that grows with the
sample budget, a forwards count that separates the posterior from the rollouts
without losing any, and a bar that reaches its total even when a cell runs short
of decisions. Whether the labels themselves are right is `test_oracle.py`'s job
and is deliberately not re-tested here.
"""

import json

import numpy as np
import pytest

from gates.g3 import (
    PROFILE, MeasuringDriver, build_hands, choose_decisions, headline,
    measure_label, profile_groups, profiling, run,
)
from oracle.rollout import OracleConfig
from tests.g1_fixtures import RAISE_SIZES, make_pool, make_specs

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

SAMPLES = [2, 8]
COMBOS = [None, 8]
TABLES = [2, 3]
STACKS = [10, 20]
LABELS_PER_CELL = 2
N_CELLS = len(SAMPLES) * len(COMBOS) * len(TABLES) * len(STACKS)


def _config():
    return {
        "seed": 3,
        "device": "cpu",
        "iteration_labels": 1000,
        "game": GAME,
        "style": {"uncond_scale": 0.8, "position_scale": 0.5,
                  "street_scale": 0.5, "log_temperature_range": [-0.4, 0.4],
                  "uniform_mix_range": [0.0, 0.2]},
        "bootstrap": [
            {"kind": "degenerate", "strategy": s, "style": "identity"}
            for s in ("always_call", "always_min_raise", "maniac", "nit")
        ] + [
            {"kind": "degenerate", "strategy": "maniac", "n_variants": 3,
             "label": "maniac_styles"},
        ],
        "sweep": {
            "samples_per_action": SAMPLES,
            "max_combos": COMBOS,
            "table_sizes": TABLES,
            "stack_bb": STACKS,
            "hands_per_table": 3,
            "labels_per_cell": LABELS_PER_CELL,
            "driver_batch_size": 16,
        },
        "oracle": {"likelihood_floor": 1e-6, "batch_hands": 64,
                   "max_collision_retries": 32},
    }


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    out = tmp_path_factory.mktemp("g3")
    report = run(_config(), lambda _m: None, str(out))
    payload = json.loads((out / "g3_report.json").read_text())
    return report, payload


# ------------------------------------------------------------------ the sweep


def test_the_sweep_has_exactly_one_cell_per_point_of_the_grid(gate):
    report, _ = gate
    cells = report["cells"]
    assert len(cells) == N_CELLS
    keys = {(c["samples_per_action"], c["max_combos"], c["players"],
             c["stack_bb"]) for c in cells}
    assert len(keys) == N_CELLS, "a cell was labelled twice"
    assert keys == {(s, c, p, b) for s in SAMPLES for c in COMBOS
                    for p in TABLES for b in STACKS}
    for c in cells:
        assert c["n_labels"] == LABELS_PER_CELL
        assert 0.0 <= c["collision_rate"] <= 1.0


def test_every_cell_reports_the_columns_the_decision_is_taken_on(gate):
    """§13's question is answered by these six numbers and nothing else."""
    report, _ = gate
    for c in report["cells"]:
        for key in ("seconds_per_label", "forwards_per_label", "depth",
                    "collision_rate", "labels_per_hour", "se_q"):
            assert key in c, key
            assert not np.isnan(c[key]), f"{key} is nan in cell {c}"
        assert c["labels_per_hour"] > 0
        assert c["iteration_hours"] == pytest.approx(
            1000 / c["labels_per_hour"])


def test_the_forward_count_grows_with_the_sample_budget(gate):
    """The sweep is worthless if the cost does not move with the knob it
    sweeps: every cell labels the *same* decisions, so more samples can only
    mean more rollouts and more forwards."""
    report, _ = gate
    by_group = {}
    for c in report["cells"]:
        key = (c["max_combos"], c["players"], c["stack_bb"])
        by_group.setdefault(key, {})[c["samples_per_action"]] = c
    assert by_group
    for key, cells in by_group.items():
        seq = [cells[s] for s in SAMPLES]
        rollouts = [c["rollouts_per_label"] for c in seq]
        forwards = [c["forwards_per_label"] for c in seq]
        assert rollouts == sorted(rollouts), f"{key}: {rollouts}"
        assert forwards == sorted(forwards), f"{key}: {forwards}"
        assert forwards[-1] > forwards[0], f"{key} did not move at all"


def test_the_posterior_and_the_rollouts_split_the_forwards_between_them(gate):
    """Both halves are needed separately: the posterior's share is amortised
    over the hero decisions of a hand (§13) and the rollouts' share is not, so a
    single total cannot say which one to attack."""
    _, payload = gate
    rows = payload["rows"]
    assert len(rows) == N_CELLS * LABELS_PER_CELL
    for r in rows:
        assert r["posterior_forwards"] + r["rollout_forwards"] == r["forwards"]
        # Zero at the very first decision of a hand: no opponent has acted, so
        # there is no likelihood to evaluate and the posterior is the prior.
        assert r["posterior_forwards"] >= 0
        assert r["rollout_forwards"] > 0
        assert r["depth"] == pytest.approx(
            r["rollout_forwards"] / r["n_rollouts"])
    assert any(r["posterior_forwards"] > 0 for r in rows), (
        "no label in the whole sweep paid for a posterior")


def test_the_bar_reaches_its_total(gate):
    """A cell that ran short of decisions still owes the bar its units
    (`CLAUDE.md` §5), so planned is always done plus skipped."""
    report, _ = gate
    n = report["n_labels"]
    assert n["planned"] == N_CELLS * LABELS_PER_CELL
    assert n["done"] + n["skipped"] == n["planned"]
    assert n["skipped"] == 0, "this toy config has decisions to spare"


def test_a_cell_short_of_decisions_is_skipped_not_faked(tmp_path):
    """The bar's total is fixed before the hands are played, so the one thing
    that must not happen is a silent shortfall (`CLAUDE.md` §5)."""
    config = _config()
    config["sweep"]["hands_per_table"] = 1
    config["sweep"]["labels_per_cell"] = 40
    report = run(config, lambda _m: None, str(tmp_path))
    n = report["n_labels"]
    assert n["skipped"] > 0
    assert n["done"] + n["skipped"] == n["planned"]
    assert all(c["n_labels"] < 40 for c in report["cells"])


# ---------------------------------------------------------- standard error


def test_the_split_half_error_is_reported_in_big_blinds(gate):
    """The ⚠ addition of S4: the sweep says what samples cost, this column says
    how many are needed. It is free — the rollouts were already played."""
    report, _ = gate
    for c in report["cells"]:
        assert c["se_q"] >= 0.0
        assert np.isfinite(c["se_q"])
    for h in report["headline"]:
        assert np.isfinite(h["se_q"])


def test_the_headline_is_one_line_per_sample_budget_at_the_exact_posterior():
    """`max_combos = None` is what §7.2 describes; the capped rows are the
    fallback and must not be averaged into the sentence §13 asks for."""
    cells = [
        {"samples_per_action": 32, "max_combos": None, "seconds_per_label": 2.0,
         "forwards_per_label": 100.0, "se_q": 0.4},
        {"samples_per_action": 32, "max_combos": 128, "seconds_per_label": 1.0,
         "forwards_per_label": 50.0, "se_q": 0.4},
        {"samples_per_action": 64, "max_combos": None, "seconds_per_label": 4.0,
         "forwards_per_label": 200.0, "se_q": 0.2},
    ]
    rows = headline(cells, iteration_labels=3600)
    assert [r["samples_per_action"] for r in rows] == [32, 64]
    assert rows[0]["seconds_per_label"] == 2.0, "the capped cell leaked in"
    assert rows[0]["labels_per_hour"] == pytest.approx(1800.0)
    assert rows[0]["iteration_hours"] == pytest.approx(2.0)
    assert rows[1]["iteration_hours"] == pytest.approx(4.0)


# ------------------------------------------------------------------ the hands


def test_the_labelled_decisions_are_the_same_in_every_cell():
    """Cells differ in the sample budget and nothing else; if they differed in
    which decisions they labelled, the columns could not be compared."""
    rng = np.random.default_rng(4)
    records = _play(rng, num_players=3, stack_bb=20, n_hands=4)
    a = choose_decisions(np.random.default_rng([3, 0]), records, 5)
    b = choose_decisions(np.random.default_rng([3, 0]), records, 5)
    assert a == b and len(a) == 5


def test_the_table_is_the_size_the_cell_says_it_is():
    """G3 pins table size and stack depth — that is the axis it measures — and
    a hand that quietly came out 2-handed would put its cost in the wrong row."""
    rng = np.random.default_rng(0)
    specs = build_hands(rng, list(range(9)), GAME, 6, 100, 5, seed_base=1)
    assert len(specs) == 5
    for spec in specs:
        assert spec.num_players == 6
        assert len(set(spec.seat_members)) == 6
        assert spec.start_credits == [1000.0] * 6
    assert len({s.seed for s in specs}) == 5


def test_a_table_wider_than_the_pool_is_refused():
    with pytest.raises(AssertionError, match="distinct pool members"):
        build_hands(np.random.default_rng(0), [0, 1, 2], GAME, 6, 100, 1,
                    seed_base=1)


# ------------------------------------------------------------------- helpers


def _play(rng, num_players, stack_bb, n_hands):
    from tests.g1_fixtures import make_pool
    from gates.g3 import MeasuringDriver
    pool = make_pool()
    specs = build_hands(rng, list(range(len(pool))), GAME, num_players,
                        stack_bb, n_hands, seed_base=7)
    return MeasuringDriver(pool, GAME["n_actions"]).run(specs)


# ---------------------------------------------------------------- the profile


def test_the_four_buckets_account_for_every_second_of_a_label(gate):
    """The profile is a decomposition, not a sample: whatever the label spent
    that the wrapped layers did not see is `driver`, and the four add back up.
    A bucket that silently dropped time would make the shares argue for
    attacking the wrong thing."""
    _, payload = gate
    for r in payload["rows"]:
        total = r["t_events"] + r["t_forward"] + r["t_style"] + r["t_driver"]
        assert total == pytest.approx(r["seconds"], rel=1e-9, abs=1e-9)
        for key in ("t_events", "t_forward", "t_style", "t_driver"):
            assert r[key] >= 0.0, f"{key} is negative: {r}"


def test_a_pool_with_no_network_spends_nothing_on_events_or_forwards(gate):
    """This config's pool is degenerate only, so no query reaches
    `build_v7_events` or a model — and the profile has to say so rather than
    attribute the time to a layer that never ran."""
    _, payload = gate
    for r in payload["rows"]:
        assert r["event_rows"] == 0
        assert r["t_events"] == 0.0
        assert r["t_forward"] == 0.0
        assert r["policy_rows"] > 0
        assert r["policy_calls"] > 0


def test_the_counters_are_per_label_and_not_cumulative(gate):
    """Every cell labels the same decisions, so a label's row count has to
    grow with the sample budget and with nothing else. If `reset` were missed
    the counters would grow monotonically down the sweep instead."""
    _, payload = gate
    by_key = {}
    for r in payload["rows"]:
        by_key.setdefault(
            (r["max_combos"], r["players"], r["stack_bb"]), {}
        ).setdefault(r["samples_per_action"], []).append(r["policy_rows"])
    assert by_key
    for key, by_samples in by_key.items():
        totals = [sum(by_samples[s]) for s in SAMPLES]
        assert totals == sorted(totals), f"{key}: {totals}"
        assert totals[-1] > totals[0], f"{key} did not move at all"


def test_a_network_member_puts_its_time_in_events_and_forward(tmp_path):
    """The bucket that decides the next step is `events` against `forward`, and
    the degenerate pool above can never exercise it. A tiny randomly
    initialised v7 network can: nothing here is about how it plays, only that a
    query through it lands in the two buckets it passes through."""
    from tests.test_v7_pool_member import _member

    pool = [_member()] + make_pool()
    specs = make_specs(seed=7, n_hands=4, n_members=len(pool), num_players=3,
                       stack_bb=20)
    for spec in specs:
        spec.seat_members = [0, 1, 2]      # seat 0 is the network
    driver = MeasuringDriver(pool, pool[0].n_actions)
    records = driver.run(specs, batch_size=4)
    record = next(r for r in records if len(r.decisions) > 1)

    cfg = OracleConfig(samples_per_action=2, max_combos=8,
                       batch_hands=32)
    row = measure_label(record, 1, driver, pool, cfg,
                        np.random.default_rng(0))

    assert row["event_rows"] > 0, "the network was never asked anything"
    assert row["t_events"] > 0.0
    assert row["t_forward"] > 0.0
    assert 0 < row["event_rows"] <= row["policy_rows"]
    total = row["t_events"] + row["t_forward"] + row["t_style"] + row["t_driver"]
    assert total == pytest.approx(row["seconds"], rel=1e-9, abs=1e-9)


def test_the_wrapping_is_undone_when_the_block_exits():
    """`profiling` patches production classes. Leaving them patched would make
    every later test measure a wrapped `policy`, which is exactly the kind of
    order-dependence `CLAUDE.md` §4 rules out."""
    import pool.base
    import pool.v7_member

    before = (pool.base.PoolMember.policy,
              pool.v7_member.V7NetworkMember.logits,
              pool.v7_member.build_v7_events)
    with profiling() as prof:
        assert prof is PROFILE
        assert pool.base.PoolMember.policy is not before[0]
        assert pool.v7_member.V7NetworkMember.logits is not before[1]
        assert pool.v7_member.build_v7_events is not before[2]
    assert (pool.base.PoolMember.policy,
            pool.v7_member.V7NetworkMember.logits,
            pool.v7_member.build_v7_events) == before


def test_the_grouped_profile_sums_the_labels_it_groups():
    """Seconds are summed, not averaged, so the shares are shares of the wall
    clock the run actually spent (`profile_groups`)."""
    rows = [
        {"players": 2, "samples_per_action": 32, "seconds": 1.0,
         "t_events": 0.5, "t_forward": 0.2, "t_style": 0.1, "t_driver": 0.2,
         "policy_rows": 100, "policy_calls": 10, "event_rows": 100},
        {"players": 6, "samples_per_action": 32, "seconds": 9.0,
         "t_events": 1.0, "t_forward": 6.0, "t_style": 1.0, "t_driver": 1.0,
         "policy_rows": 900, "policy_calls": 20, "event_rows": 450},
    ]
    groups = {g["group"]: g for g in profile_groups(rows)}
    assert set(groups) == {"plr=2", "plr=6", "smp=32", "all"}

    total = groups["all"]
    assert total["seconds"] == pytest.approx(10.0)
    assert total["t_events"] == pytest.approx(1.5)
    assert total["rows_per_call"] == pytest.approx(1000 / 30)
    assert total["network_share"] == pytest.approx(0.55)
    # (10.0 - 1.2) seconds inside `policy`, over 1000 rows.
    assert total["us_per_row"] == pytest.approx(8800.0)

    assert groups["smp=32"]["seconds"] == pytest.approx(10.0)
    assert groups["plr=2"]["n_labels"] == 1
    assert groups["plr=6"]["us_per_row"] == pytest.approx(1e6 * 8.0 / 900)

