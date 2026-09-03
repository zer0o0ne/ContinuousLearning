"""The gate that reports the two things asked of the procedural pool
(`PLAN_PROCEDURAL_POOL.md` §P4, §0.3).

Whether the archetypes really are ten different players is asserted in
`test_archetypes.py`, over a sample big enough to say so. What is checked here
is the *gate*: that it runs at every table size, that every cell it claims to
have played was played, that the two sections mean what their names say, and
that a stat with no opportunities behind it comes back as an honest `nan`
instead of a fabricated zero.

There are no bands to check. §0.3 removed them.
"""

import json
import os

import numpy as np
import pytest

from gates.pool_realism import SHORT_BB, SIZES, ratio, run, winrate
from pool.stats import STAT_NAMES
from pool.strength import preflop_equity_table
from tests.test_archetypes import GAME

NAMES = ["nit", "maniac", "tag"]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    table_dir = tmp_path_factory.mktemp("tables")
    path = table_dir / "preflop_equity.npy"
    preflop_equity_table(str(path), seed=0, n_deals=50_000)
    out = tmp_path_factory.mktemp("out")
    config = {"game": GAME, "pool_realism": {
        "archetypes": NAMES, "sizes": ["hu", "6max"], "hands": 60,
        "stack_bb_range": [10, 300], "seed": 4242, "spread": 0.0,
        "batch_size": 16, "preflop_table": str(path)}}
    lines = []
    result = run(config, lines.append, str(out))
    return result, lines, str(out)


def test_the_gate_reports_both_sections_for_every_cell(report):
    result, _lines, _out = report
    assert set(result["profile"]) == set(NAMES)
    assert set(result["strength"]) == set(NAMES)
    for name in NAMES:
        assert set(result["profile"][name]) == {"hu", "6max"}
        for label in ("hu", "6max"):
            cell = result["profile"][name][label]
            assert set(cell) == {"all", "short", "deep"}
            assert set(cell["all"]) == set(STAT_NAMES)
            assert cell["all"]["vpip"][1] > 0


def test_every_hand_it_says_it_played_was_played(report):
    result, _lines, _out = report
    hands = result["settings"]["hands"]
    for name in NAMES:
        for label, n_players in (("hu", 2), ("6max", 6)):
            cell = result["profile"][name][label]
            # The short and deep buckets partition the hands, and the hero is
            # dealt into every one of them — though not asked to act in every
            # one, since a hand can end before the action reaches its seat.
            assert (cell["short"]["vpip"][1] + cell["deep"]["vpip"][1]
                    == cell["all"]["vpip"][1] <= hands)
            assert cell["all"]["vpip"][1] > hands / 2
            assert result["strength"][name][label]["hands"] == hands


def test_the_stack_buckets_split_where_they_say_they_do(report):
    result, _lines, _out = report
    assert SHORT_BB == 25.0
    for name in NAMES:
        short = result["profile"][name]["6max"]["short"]["vpip"][1]
        deep = result["profile"][name]["6max"]["deep"]["vpip"][1]
        # Stacks are uniform on 10–300 BB, so about a twentieth are short.
        assert 0 < short < deep


def test_a_stat_with_no_opportunities_is_not_invented(report):
    result, _lines, _out = report
    empty = [(name, stat)
             for name in NAMES
             for stat, counts in result["profile"][name]["hu"]["short"].items()
             if counts[1] == 0]
    assert empty, "sixty short-stacked hands should leave some stat empty"
    for name, stat in empty:
        # `nan`, not a zero: nobody bet into this player, so there is no
        # frequency — and `af` shows why the numerator alone will not do, since
        # its two halves count different things (bets over calls).
        assert np.isnan(ratio(result["profile"][name]["hu"]["short"][stat]))


def test_the_strength_section_is_a_winrate_with_an_error(report):
    result, _lines, _out = report
    for name in NAMES:
        for label in ("hu", "6max"):
            cell = result["strength"][name][label]
            assert np.isfinite(cell["bb_per_100"]) and np.isfinite(cell["se"])
            assert cell["se"] > 0


def test_the_report_is_written_and_reloads(report):
    result, lines, out = report
    path = os.path.join(out, "pool_realism.json")
    assert os.path.exists(path)
    with open(path) as fh:
        stored = json.load(fh)
    assert stored["settings"] == result["settings"]
    assert stored["config"]["pool_realism"]["hands"] == 60
    assert any("stat profile" in line for line in lines)
    assert any("against the degenerate five" in line for line in lines)


def test_an_interrupted_run_keeps_what_it_paid_for(tmp_path_factory):
    """The report is written after every cell and resumed from.

    A whole job is twenty-odd minutes; losing it to an interrupted terminal —
    and then re-running it — is the thing this stops.
    """
    from gates.pool_realism import _already_done, _load, run

    table_dir = tmp_path_factory.mktemp("resume_tables")
    path = table_dir / "preflop_equity.npy"
    preflop_equity_table(str(path), seed=0, n_deals=20_000)
    out = str(tmp_path_factory.mktemp("resume_out"))
    config = {"game": GAME, "pool_realism": {
        "archetypes": ["nit", "tag"], "sizes": ["hu"], "hands": 40,
        "stack_bb_range": [10, 300], "seed": 11, "spread": 0.0,
        "batch_size": 16, "preflop_table": str(path)}}

    whole = run(config, lambda _m: None, out)
    report_path = os.path.join(out, "pool_realism.json")

    # Throw one cell away, as an interrupted run would have left it, and let
    # the next call notice and replay exactly that cell.
    with open(report_path) as fh:
        stored = json.load(fh)
    del stored["strength"]["tag"]["hu"]
    with open(report_path, "w") as fh:
        json.dump(stored, fh)

    lines = []
    resumed = run(config, lines.append, out)
    assert any("resuming" in line for line in lines)
    # Through JSON, because a resumed cell comes back from disk as a list where
    # a freshly played one is still a tuple — the numbers are what is at stake.
    def plain(value):
        return json.loads(json.dumps(value))

    assert plain(resumed["profile"]) == plain(whole["profile"])
    assert plain(resumed["strength"]) == plain(whole["strength"])

    # Settings that differ are a different experiment, not something to splice.
    other = {"game": GAME, "pool_realism": dict(config["pool_realism"],
                                                hands=20)}
    assert _load(report_path, dict(other["pool_realism"]), lambda _m: None) \
        is None


def test_the_resume_arithmetic_counts_finished_cells_only():
    from gates.pool_realism import _already_done

    names, sizes = ["a", "b"], {"hu": 2, "6max": 6}
    mixed = {"hu": 100, "6max": 40}
    report = {"profile": {"a": {"hu": {}}, "b": {}},
              "strength": {"a": {"hu": {}}, "b": {"hu": {}, "6max": {}}}}
    # The mixed table counts only when *every* archetype was read off it.
    assert _already_done(report, names, sizes, mixed, hands=10) == 30
    report["profile"]["b"]["hu"] = {}
    assert _already_done(report, names, sizes, mixed, hands=10) == 130


def test_a_winrate_is_the_hero_seats_own_chips():
    class Spec:
        def __init__(self, seat):
            self.meta = {"hero_seat": seat}

    class Record:
        def __init__(self, seat, rewards):
            self.spec = Spec(seat)
            self.rewards = np.array(rewards, dtype=np.float64)

    records = [Record(0, [10.0, -10.0]), Record(1, [-30.0, 30.0]),
               Record(0, [20.0, -20.0])]
    got = winrate(records, big_blind=10.0)
    assert got["hands"] == 3
    assert got["bb_per_100"] == pytest.approx(100.0 * (1.0 + 3.0 + 2.0) / 3.0)
    assert got["se"] > 0


def test_the_table_sizes_are_the_three_the_pool_is_read_at():
    assert SIZES == {"hu": 2, "6max": 6, "9max": 9}
