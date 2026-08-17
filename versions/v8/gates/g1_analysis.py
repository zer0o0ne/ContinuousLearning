"""Section A — regrouping a finished G1 report, no extra compute.

`gates/g1.py` writes every measurement row it took into `g1_report.json`, and
three questions the printed report leaves open are answerable from those rows
alone. Nothing here runs the network, plays a hand, or fits a vector; this
module reads one JSON and prints three tables.

The three questions, in the order they matter:

**A1 — is the §14.2 result a difference of composition?** The printed §14.2 gap
compares two sets whose member draws are built differently. The seen set draws
from the whole pool, where a base that expanded into many style variants
occupies many member slots; the unseen set is `fresh_style_variants`, which
cycles over *bases*, so every base contributes about equally. A pool with 7
network bases at 8 variants each and 8 degenerate bases at 1–6 variants each is
therefore mostly networks when drawn as members and mostly degenerates when
drawn as bases. Since a degenerate strategy is far easier to identify than a
network, that alone can move the gap. `gain_fit` per base, and the gap
recomputed with the seen set's base mix, separates the two readings.

**A2 — does the embedding read style, or does it read "degenerate vs network"?**
`CONCEPT.md` §11.3 is the risk that the pool spans far fewer styles than
members. If essentially all of the gain sits on the degenerate bases and the
network bases barely move, then "the embedding carries style" means "the
embedding tells a calling station from a nit", which is not what B1(b) needs to
hold for Slumbot.

**A3 — is the §14.4 skew by table size a skew, or a data budget?** The printed
breakdown fixes the number of observed *hands*. At a fixed hand count a 9-handed
table gives each individual opponent several times fewer decisions than a
heads-up one, and the joint fit has 32 free parameters per seat to spend them
on. Re-plotting the same rows against observed tokens *per player* asks whether
the table-size spread survives once the observation budget is matched.

Run::

    cd versions/v8 && python3 -m gates.g1_analysis --report path/to/g1_report.json
"""

import argparse
import json
import math

from gates.g1 import _curve, _with_gains

# A3's x-axis. Log-spaced because the observation windows themselves are, and
# open-ended at the top so no row falls outside the grid.
OBS_BINS = [0, 4, 16, 64, 256]

# A3's table-size grouping. Per-seat-count is what §14.4 already prints and it
# is too thin to cross with a second axis — 12 to 25 sessions per cell before
# the split, single digits after it.
TABLE_GROUPS = ((2, 3), (4, 6), (7, 9))


# ------------------------------------------------------------------- the rows


def member_descriptors(payload):
    """One descriptor per member index, trained members then fresh draws.

    `gates.g1.run` builds the pool as ``all_members = members + fresh`` and
    stores the two descriptor lists separately, so the member index a row
    carries indexes their concatenation.
    """
    return list(payload["pool"]) + list(payload["fresh_style_draws"])


def annotate(payload):
    """Measurement rows with the fields section A groups by.

    ``base`` / ``kind`` come from the member descriptor; ``obs_per_player`` is
    the observation budget the fit had per seat, which is the quantity A3 needs
    and the report only carries table-wide.

    ``observed_decisions`` counts every token of the observed window — decisions
    plus the terminal showdown tokens of §5.1a — so it is an observation budget
    rather than a decision count, and `gates.g1` records it once per session.
    Dividing by the number of seats is therefore a per-opponent average, not a
    per-slot count; B4 of the follow-up list is what makes it exact.
    """
    desc = member_descriptors(payload)
    out = []
    for row in payload["rows"]:
        d = desc[row["member"]]
        out.append(_with_gains({
            **row,
            "base": d["base"],
            "kind": d["kind"],
            "obs_per_player": row["observed_decisions"] / row["num_players"],
        }))
    return out


def _curve_over(rows, value_of):
    """`gates.g1._curve` with a different x-axis.

    `_curve` is the one place that knows the aggregation rule §14 requires —
    average the rows of a session first, then take the standard error over
    sessions, because rows of one session share their evaluation hands — and it
    reads its x-axis off ``observed_hands``. Rather than keep a second copy of
    that rule here, the rows are re-keyed: ``observed_hands`` is overwritten
    with whatever the caller wants on the x-axis. Nothing downstream reads the
    field again, and the returned keys must be sortable.
    """
    return _curve([{**r, "observed_hands": value_of(r)} for r in rows])


def _relative(rows):
    """Rows with `gain_fit` replaced by the gain as a fraction of the row's own
    `e = 0` loss.

    Same re-keying device as `_curve_over` and for the same reason: `_curve` is
    the single implementation of §14's aggregation rule and reads a fixed metric
    list, so a derived metric arrives by overwriting the field it reads rather
    than through a second aggregator. Read the result out of ``gain_fit`` and
    remember it is a ratio.

    Absolute gain is not comparable across groups whose `e = 0` loss differs,
    and §14.4's groups are exactly such groups: a 9-handed table starts at about
    1.9 nats and a heads-up one at about 2.9, so the same fraction of the
    available headroom shows up as a much smaller number of nats at the big
    table. Every comparison across table sizes has to be made on this scale as
    well as on the absolute one.
    """
    return [{**r, "gain_fit": r["gain_fit"] / r["ce_zero"]}
            for r in rows if r["ce_zero"] > 0]


def _by_set(rows):
    """Split by measurement set. Session indices restart in each set, so rows
    of different sets may never be aggregated together."""
    out = {}
    for r in rows:
        out.setdefault(r["set"], []).append(r)
    return out


def _point(rows, tag, window, metric="gain_fit"):
    """One aggregated point: set `tag`, window `window`, metric `metric`."""
    sel = [r for r in rows if r["set"] == tag]
    return _curve(sel).get(window, {}).get(metric)


def _se_of_difference(a, b):
    """SE of ``a − b`` for two independently measured means.

    Exact here: the seen and the unseen set are disjoint collections of
    sessions, played by disjoint sets of members.
    """
    if not (a and b and a["n"] and b["n"]):
        return float("nan")
    if math.isnan(a["se"]) or math.isnan(b["se"]):
        return float("nan")
    return math.sqrt(a["se"] ** 2 + b["se"] ** 2)


# --------------------------------------------------------------- A1 — by base


def composition(rows, window):
    """Share of scored rows each base and each kind holds, per set.

    This is the confound itself, stated before it is corrected for: it is the
    distribution §14.2's two averages are taken over.
    """
    out = {}
    for tag, group in _by_set(rows).items():
        sel = [r for r in group if r["observed_hands"] == window]
        by_base, by_kind = {}, {}
        for r in sel:
            by_base[r["base"]] = by_base.get(r["base"], 0) + 1
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        out[tag] = {"n_rows": len(sel), "by_base": by_base, "by_kind": by_kind}
    return out


def by_base(rows, window):
    """`gain_fit` per base per set at one observation window.

    The unit of the standard error stays the session, exactly as in the printed
    report: filtering to one base leaves at most a few rows of any session, and
    `_curve` averages them before it counts.
    """
    return {tag: {b: _curve([r for r in group
                             if r["observed_hands"] == window
                             and r["base"] == b])
                  for b in sorted({r["base"] for r in group})}
            for tag, group in _by_set(rows).items()}


def matched_gap(base_curves, comp, window, other="unseen", metric="gain_fit"):
    """§14.2's gap against `seen` recomputed with the seen set's base mix.

    The printed gap is ``mean_other − mean_seen``, and each mean is taken over
    that set's own distribution of bases. Reweighting the per-base differences
    by the *seen* set's base shares answers the question §14.2 meant to ask:
    what would the gap be if the two sets were made of the same things.

    A `heldout` set shares no base with `seen` by construction — that is what
    holding a base out means — so nothing is left to reweight and the result is
    empty. That is the honest answer: for C1 the composition correction does not
    apply, and the raw gap is the number, read against the per-kind breakdown of
    A2 rather than against a common base.

    The reweighted standard error is **approximate** and is marked as such
    wherever it is printed: bases within one set share sessions and evaluation
    hands, so their per-base means are correlated and summing their variances
    ignores those covariances. The per-base differences themselves carry exact
    standard errors — the sets are disjoint collections of sessions.
    """
    seen = base_curves.get("seen", {})
    unseen = base_curves.get(other, {})
    shares = comp["seen"]["by_base"]
    total = sum(shares.values())

    per_base, weighted, var = {}, 0.0, 0.0
    for b, count in shares.items():
        s = seen.get(b, {}).get(window, {}).get(metric)
        u = unseen.get(b, {}).get(window, {}).get(metric)
        if not (s and u and s["n"] and u["n"]):
            continue
        w = count / total
        se = _se_of_difference(u, s)
        per_base[b] = {"weight": w, "seen": s, "unseen": u,
                       "delta": u["mean"] - s["mean"], "se": se}
        weighted += w * (u["mean"] - s["mean"])
        if not math.isnan(se):
            var += (w * se) ** 2

    covered = sum(v["weight"] for v in per_base.values())
    return {
        "per_base": per_base,
        # Renormalised by the covered weight, so a base missing from one set
        # cannot silently shrink the estimate toward zero.
        "matched_delta": weighted / covered if covered else float("nan"),
        "matched_se_approx": (math.sqrt(var) / covered if covered and var
                              else float("nan")),
        "covered_weight": covered,
    }


# --------------------------------------------------------------- A2 — by kind


def by_kind(rows):
    """The full §14.1 curve, split into degenerate bases and network bases.

    The network bases are the only ones that condition on cards, so their curve
    is the one that says whether the embedding found a style space or a
    two-point classification (`CONCEPT.md` §11.3).

    Both scales are returned: ``abs`` in nats and ``rel`` as a fraction of each
    row's own `e = 0` loss, because a difference between the two kinds could
    otherwise be nothing but a difference in how predictable they are to begin
    with.
    """
    out = {}
    for tag, group in _by_set(rows).items():
        kinds = sorted({r["kind"] for r in group})
        out[tag] = {
            "abs": {k: _curve([r for r in group if r["kind"] == k])
                    for k in kinds},
            "rel": {k: _curve(_relative([r for r in group if r["kind"] == k]))
                    for k in kinds},
        }
    return out


# ------------------------------------------- A3 — by observation budget/player


def _obs_bin(row):
    """Lower edge of the `obs_per_player` bucket. Sortable, so it doubles as
    the x-axis key."""
    edge = OBS_BINS[0]
    for lo in OBS_BINS:
        if row["obs_per_player"] >= lo:
            edge = lo
    return edge


def _table_group(row):
    for lo, hi in TABLE_GROUPS:
        if lo <= row["num_players"] <= hi:
            return f"{lo}-{hi}"
    return "other"


def by_observation_budget(rows):
    """`gain_fit` against observation budget per player, crossed with table size.

    §14.4 holds the number of observed hands fixed and finds the gain falling
    with table size. Holding the *budget per player* fixed instead asks how much
    of that is the budget and how much is the table. Both scales are returned,
    for the reason `_relative` gives: the groups differ in headroom as well as
    in size.
    """
    out = {}
    for tag, group in _by_set(rows).items():
        groups = sorted({_table_group(r) for r in group})
        out[tag] = {
            "abs": {g: _curve_over([r for r in group if _table_group(r) == g],
                                   _obs_bin)
                    for g in groups},
            "rel": {g: _curve_over(_relative([r for r in group
                                              if _table_group(r) == g]),
                                   _obs_bin)
                    for g in groups},
        }
    return out


# ------------------------------------------------------------------ reporting


def _cell(st):
    if not st or not st["n"]:
        return "—"
    return f"{st['mean']:.4f}±{st['se']:.4f}"


def _bin_label(edge):
    idx = OBS_BINS.index(edge)
    if idx + 1 < len(OBS_BINS):
        return f"{edge}-{OBS_BINS[idx + 1]}"
    return f"{edge}+"


def _format_a1(rows, window, comp, base_curves, sets, log):
    log(f"\n[A1 | composition] share of scored rows by base kind, "
        f"window n={window}")
    log(f"  {'set':>8} {'degenerate':>12} {'v7':>12} {'rows':>7}")
    for tag in sets:
        c, n = comp[tag]["by_kind"], comp[tag]["n_rows"]
        log(f"  {tag:>8} {c.get('degenerate', 0) / n:>11.1%} "
            f"{c.get('v7', 0) / n:>11.1%} {n:>7}")
    log("  The sets §14.2 compares are not made of the same things: seen")
    log("  members are drawn from the pool, fresh draws are cycled over bases.")

    seen_pt = _point(rows, "seen", window)
    for other in [t for t in sets if t != "seen"]:
        gap = matched_gap(base_curves, comp, window, other=other)
        log(f"\n[A1 | §14.2 by base] {other} vs seen, gain of fit at "
            f"n={window}, mean ± SE over sessions")
        if gap["per_base"]:
            log(f"  {'base':>18} {'seen gain':>16} {other + ' gain':>16} "
                f"{'Δ ' + other + '−seen':>18} {'weight':>7}")
            for b, v in sorted(gap["per_base"].items(),
                               key=lambda kv: -kv[1]["weight"]):
                se = "" if math.isnan(v["se"]) else f"±{v['se']:.4f}"
                log(f"  {b:>18} {_cell(v['seen']):>16} "
                    f"{_cell(v['unseen']):>16} "
                    f"{v['delta']:>+11.4f}{se:>7} {v['weight']:>6.1%}")
            log("  weight = that base's share of the seen set's scored rows.")
        else:
            log("  no base is in both sets — nothing to reweight, which is what")
            log("  holding whole bases out means. Read the raw gap against A2.")

        other_pt = _point(rows, other, window)
        raw_delta = other_pt["mean"] - seen_pt["mean"]
        log(f"\n  reported Δgain (§14.2)           {raw_delta:>+8.4f} "
            f"± {_se_of_difference(other_pt, seen_pt):.4f}")
        if gap["per_base"]:
            log(f"  Δgain at the seen set's base mix {gap['matched_delta']:>+8.4f} "
                f"± {gap['matched_se_approx']:.4f}  (SE approximate — bases "
                f"within a set share sessions)")
            log(f"  bases covered by the reweighting: "
                f"{gap['covered_weight']:.1%} of the seen set")
            log("  If the reweighted Δ is near zero the reported gap is")
            log("  composition, and B1(b) reads as 'no measurable penalty'.")


def _format_a2(kind_curves, sets, log):
    log("\n[A2 | §11.3] gain of fit by base kind — style, or "
        "degenerate-vs-network?")
    for tag in sets:
        curves = kind_curves[tag]
        kinds = sorted(curves["abs"])
        windows = sorted(next(iter(curves["abs"].values())))
        zero = {k: curves["abs"][k][windows[-1]]["ce_zero"] for k in kinds}
        log(f"  [{tag}]  e=0 baseline: "
            + ", ".join(f"{k} {_cell(zero[k])}" for k in kinds))
        header = " ".join(f"{k + ' gain':>18} {'rel':>13}" for k in kinds)
        log(f"    {'hands':>6} {header}")
        for n in windows:
            line = " ".join(
                f"{_cell(curves['abs'][k].get(n, {}).get('gain_fit')):>18} "
                f"{_cell(curves['rel'][k].get(n, {}).get('gain_fit')):>13}"
                for k in kinds)
            log(f"    {n:>6} {line}")
    log("  rel = gain as a fraction of that row's own e=0 loss. The v7 bases are")
    log("  the only ones that condition on cards; if their gain is small at")
    log("  matched headroom, the embedding classifies families, not style.")


def _format_a3(budget, sets, log):
    log("\n[A3 | §14.4 rescaled] gain of fit against observed tokens per player")
    for scale, title in (("abs", "nats"), ("rel", "fraction of e=0")):
        log(f"\n  -- gain in {title} --")
        for tag in sets:
            groups = sorted(budget[tag][scale])
            log(f"  [{tag}]")
            log(f"    {'tok/player':>11} "
                + " ".join(f"{g + ' players':>22}" for g in groups))
            for edge in OBS_BINS:
                cells = []
                for g in groups:
                    pt = budget[tag][scale][g].get(edge, {}).get("gain_fit")
                    text = f"{_cell(pt)} ({pt['n']})" if pt and pt["n"] else "—"
                    cells.append(f"{text:>22}")
                log(f"    {_bin_label(edge):>11} " + " ".join(cells))
    log("  (n) is the number of sessions behind the cell. §14.4 holds hands")
    log("  fixed; this holds the fit's observation budget per seat fixed. The")
    log("  second table removes the headroom difference between table sizes.")


def format_section_a(rows, window, log):
    comp = composition(rows, window)
    base_curves = by_base(rows, window)
    # `seen` first, then every set measured against it.
    sets = ["seen"] + sorted(set(comp) - {"seen"}) if "seen" in comp \
        else sorted(comp)

    log("=" * 78)
    log("G1 — section A: regrouping the report, no extra compute")
    log("=" * 78)
    _format_a1(rows, window, comp, base_curves, sets, log)
    _format_a2(by_kind(rows), sets, log)
    _format_a3(by_observation_budget(rows), sets, log)
    log("")


def main():
    parser = argparse.ArgumentParser(
        description="G1 section A — regroup an existing g1_report.json")
    parser.add_argument("--report", default="g1_report.json")
    parser.add_argument("--window", type=int, default=None,
                        help="observation window for A1 (default: the longest)")
    args = parser.parse_args()

    with open(args.report) as fh:
        payload = json.load(fh)
    rows = annotate(payload)
    window = args.window if args.window is not None \
        else max(r["observed_hands"] for r in rows)
    format_section_a(rows, window, print)


if __name__ == "__main__":
    main()
