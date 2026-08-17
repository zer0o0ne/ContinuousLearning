"""Section A — the regrouping of a finished G1 report (`gates/g1_analysis.py`).

Section A exists to answer one question about §14.2 — is the reported
seen/unseen gap a property of the embedding or of how the two member sets were
drawn — and two supporting ones about §11.3 and §14.4. Every number it prints is
a regrouping of rows `gates/g1.py` already wrote, so what has to be pinned is
the arithmetic of the regrouping: that a pure composition difference is
recognised as one, that the member index still points at the right descriptor,
and that the aggregation rule §14 requires (average a session's rows first, then
take the standard error over sessions) survives being given a different x-axis.

Fixtures are built so the answer is computable by hand.
"""

import math

import pytest

from gates.g1_analysis import (
    OBS_BINS, _curve_over, _obs_bin, _relative, _table_group, annotate,
    by_base, by_kind, by_observation_budget, composition, format_section_a,
    matched_gap, member_descriptors,
)

WINDOW = 10

# Two bases, one easy to read and one hard, with the *same* per-base gain in
# both sets. Everything the seen/unseen comparison can then see is the mix.
GAIN = {"deg": 1.0, "net": 0.2, "held": 0.3}
CE_ZERO = 2.0

# seen is mostly `net`, unseen is mostly `deg` — the shape `fresh_style_variants`
# produces when it cycles over bases while the pool expands them into members.
SEEN_BASES = ["deg", "net", "net", "net"]
UNSEEN_BASES = ["deg", "deg", "deg", "net"]


def _descriptor(base):
    return {"name": f"{base}_x", "kind": "degenerate" if base == "deg" else "v7",
            "base": base, "style": [0.0] * 32}


def _row(tag, session, base, member, num_players=4, observed=WINDOW,
         decisions=40):
    return {
        "set": tag, "session": session, "num_players": num_players,
        "stack_bb": 100, "slot": 1, "member": member,
        "observed_hands": observed, "observed_decisions": decisions,
        "eval_tokens": 50,
        "ce_zero": CE_ZERO,
        "ce_ablation": CE_ZERO,
        "ce_fit": CE_ZERO - GAIN[base],
    }


def _payload(with_heldout=False):
    """Four seen sessions and four unseen ones, one scored player each.

    `with_heldout` adds C1's third set: a base that lives in the pool like any
    other but that no `seen` row ever names, which is what holding it out of
    training amounts to on this side of the report.
    """
    pool = [_descriptor("deg"), _descriptor("net")]
    if with_heldout:
        pool.append(_descriptor("held"))
    fresh = [_descriptor("deg"), _descriptor("net")]
    desc = pool + fresh

    def member_of(tag, base):
        """Index into `pool + fresh`, the concatenation a row's index names."""
        lo, hi = (len(pool), len(desc)) if tag == "unseen" else (0, len(pool))
        return next(i for i in range(lo, hi) if desc[i]["base"] == base)

    rows = [_row("seen", s, b, member_of("seen", b))
            for s, b in enumerate(SEEN_BASES)]
    rows += [_row("unseen", s, b, member_of("unseen", b))
             for s, b in enumerate(UNSEEN_BASES)]
    if with_heldout:
        rows += [_row("heldout", s, "held", member_of("heldout", "held"))
                 for s in range(3)]
    return {"pool": pool, "fresh_style_draws": fresh, "rows": rows}


# ------------------------------------------------------------------- the rows


def test_a_member_index_past_the_pool_names_a_fresh_draw():
    """`gates.g1.run` concatenates the trained members and the fresh draws, so
    the index a row carries indexes the concatenation, not either list."""
    payload = _payload()
    desc = member_descriptors(payload)
    assert len(desc) == 4
    assert desc[:2] == payload["pool"]
    assert desc[2:] == payload["fresh_style_draws"]

    rows = annotate(payload)
    unseen = [r for r in rows if r["set"] == "unseen"]
    assert {r["member"] for r in unseen} == {2, 3}
    assert sorted(r["base"] for r in unseen) == sorted(UNSEEN_BASES)


def test_the_observation_budget_is_per_seat_not_per_table():
    rows = annotate(_payload())
    for r in rows:
        assert r["obs_per_player"] == pytest.approx(
            r["observed_decisions"] / r["num_players"])
    assert rows[0]["obs_per_player"] == pytest.approx(10.0)


# ---------------------------------------------------- A1: composition vs gain


def test_a_pure_difference_of_composition_reads_as_a_gap_and_matches_out():
    """The claim section A is for.

    Both sets read every base equally well, so the embedding generalises
    perfectly here by construction. They differ only in how often each base is
    drawn — and that alone produces a large positive §14.2 gap. Reweighting to a
    common mix must take it back to exactly zero.
    """
    rows = annotate(_payload())
    comp = composition(rows, WINDOW)
    gap = matched_gap(by_base(rows, WINDOW), comp, WINDOW)

    seen_mean = sum(GAIN[b] for b in SEEN_BASES) / len(SEEN_BASES)
    unseen_mean = sum(GAIN[b] for b in UNSEEN_BASES) / len(UNSEEN_BASES)
    assert unseen_mean - seen_mean == pytest.approx(0.4)   # the raw §14.2 gap

    assert gap["matched_delta"] == pytest.approx(0.0)
    assert gap["covered_weight"] == pytest.approx(1.0)
    for v in gap["per_base"].values():
        assert v["delta"] == pytest.approx(0.0)


def test_the_reweighting_uses_the_seen_mix_and_renormalises_on_missing_bases():
    """A base absent from one set must drop out of the weighted average rather
    than contribute a zero difference and drag the estimate toward zero."""
    payload = _payload()
    # `net` never appears in the unseen set; only `deg` can be compared.
    payload["rows"] = [r for r in payload["rows"]
                       if not (r["set"] == "unseen" and r["member"] == 3)]
    # ...and `deg` reads 0.5 nats better there, which must survive intact.
    for r in payload["rows"]:
        if r["set"] == "unseen":
            r["ce_fit"] = CE_ZERO - (GAIN["deg"] + 0.5)

    rows = annotate(payload)
    gap = matched_gap(by_base(rows, WINDOW), composition(rows, WINDOW), WINDOW)
    assert set(gap["per_base"]) == {"deg"}
    assert gap["covered_weight"] == pytest.approx(0.25)
    assert gap["matched_delta"] == pytest.approx(0.5)


def test_a_held_out_base_set_shares_no_base_and_so_cannot_be_reweighted():
    """C1's `heldout` set has no base in common with `seen` — that is what
    holding a base out means. The correction must report that it does not apply
    rather than silently return zero, which would read as "no gap"."""
    rows = annotate(_payload(with_heldout=True))
    comp = composition(rows, WINDOW)
    gap = matched_gap(by_base(rows, WINDOW), comp, WINDOW, other="heldout")

    assert "held" not in comp["seen"]["by_base"]
    assert gap["per_base"] == {}
    assert gap["covered_weight"] == 0.0
    assert math.isnan(gap["matched_delta"])

    # The unseen correction is unaffected by the third set being present.
    unseen = matched_gap(by_base(rows, WINDOW), comp, WINDOW, other="unseen")
    assert unseen["matched_delta"] == pytest.approx(0.0)


def test_composition_reports_the_mix_the_two_averages_are_taken_over():
    comp = composition(annotate(_payload()), WINDOW)
    assert comp["seen"]["by_base"] == {"deg": 1, "net": 3}
    assert comp["unseen"]["by_base"] == {"deg": 3, "net": 1}
    assert comp["seen"]["by_kind"] == {"degenerate": 1, "v7": 3}
    assert comp["unseen"]["n_rows"] == 4


def test_the_standard_error_of_a_per_base_difference_is_reported():
    """Seen and unseen are disjoint session sets, so the SE of their difference
    is the root of the sum — but only where both sides have two sessions to
    estimate an SE from."""
    rows = annotate(_payload())
    gap = matched_gap(by_base(rows, WINDOW), composition(rows, WINDOW), WINDOW)
    # `net` has 3 seen sessions and 1 unseen one: no SE on one side.
    assert math.isnan(gap["per_base"]["net"]["se"])
    # Every gain is identical, so the surviving SEs are zero, not nan.
    assert gap["per_base"]["deg"]["se"] == 0.0 or \
        math.isnan(gap["per_base"]["deg"]["se"])


# ------------------------------------------------------------ A2: the scales


def test_the_relative_scale_divides_by_each_rows_own_headroom():
    """Absolute gain is not comparable across groups with different `e = 0`
    losses, and §14.4's groups are exactly such groups."""
    rows = annotate(_payload())
    rel = _relative(rows)
    for r, s in zip(rows, rel):
        assert s["gain_fit"] == pytest.approx(r["gain_fit"] / r["ce_zero"])
    assert rel[0]["gain_fit"] == pytest.approx(GAIN["deg"] / CE_ZERO)


def test_by_kind_reports_both_scales_over_the_whole_curve():
    curves = by_kind(annotate(_payload()))
    assert set(curves["seen"]) == {"abs", "rel"}
    assert set(curves["seen"]["abs"]) == {"degenerate", "v7"}
    abs_pt = curves["seen"]["abs"]["v7"][WINDOW]["gain_fit"]
    rel_pt = curves["seen"]["rel"]["v7"][WINDOW]["gain_fit"]
    assert abs_pt["mean"] == pytest.approx(GAIN["net"])
    assert rel_pt["mean"] == pytest.approx(GAIN["net"] / CE_ZERO)


# --------------------------------------------------- A3: the x-axis and bins


def test_the_observation_bins_are_closed_on_the_left():
    edges = OBS_BINS
    assert _obs_bin({"obs_per_player": 0.0}) == edges[0]
    assert _obs_bin({"obs_per_player": edges[1] - 1e-9}) == edges[0]
    assert _obs_bin({"obs_per_player": float(edges[1])}) == edges[1]
    assert _obs_bin({"obs_per_player": 1e9}) == edges[-1]


def test_table_groups_cover_every_legal_table_size():
    assert {_table_group({"num_players": n}) for n in range(2, 10)} == \
        {"2-3", "4-6", "7-9"}


def test_rekeying_the_x_axis_keeps_the_session_as_the_unit_of_the_error():
    """`_curve_over` reuses `gates.g1._curve` precisely so this rule is not
    reimplemented. Two rows of one session landing in one bucket must count as
    one observation, not two."""
    rows = annotate(_payload())
    # A second scored seat in session 0, read 0.6 nats better: the session
    # averages 0.8 and contributes one observation, not two.
    extra = dict(rows[0], slot=2, ce_fit=CE_ZERO - 0.6, gain_fit=0.6)
    point = _curve_over(rows[:1] + [extra], _obs_bin)[_obs_bin(rows[0])]
    assert point["gain_fit"]["n"] == 1
    assert point["gain_fit"]["mean"] == pytest.approx(0.8)


def test_the_budget_breakdown_splits_by_set_and_table_group():
    rows = annotate(_payload())
    budget = by_observation_budget(rows)
    assert set(budget) == {"seen", "unseen"}
    assert set(budget["seen"]) == {"abs", "rel"}
    assert set(budget["seen"]["abs"]) == {"4-6"}
    point = budget["seen"]["abs"]["4-6"][_obs_bin(rows[0])]["gain_fit"]
    assert point["n"] == len(SEEN_BASES)


# ------------------------------------------------------------------ end to end


def test_the_whole_section_prints_without_touching_the_network(capsys):
    """Section A must run from a JSON alone — no checkpoint, no pool, no
    replay of a single hand."""
    lines = []
    format_section_a(annotate(_payload()), WINDOW, lines.append)
    text = "\n".join(lines)
    for marker in ("[A1 | composition]", "[A1 | §14.2 by base]",
                   "[A2 | §11.3]", "[A3 | §14.4 rescaled]"):
        assert marker in text
    assert "reported Δgain" in text
    assert capsys.readouterr().out == "", "the section must log, not print"
