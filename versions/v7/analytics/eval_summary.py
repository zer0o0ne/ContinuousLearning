"""Consolidated summary for an eval pipeline run.

Reads `<run_dir>/internal/*.json` and `<run_dir>/slumbot/*.json` produced by
eval_pipeline.py and renders one human-readable report (printed to stdout
and optionally written to `<run_dir>/SUMMARY.md`).

Usage:
    # Summarize the latest eval run for v5
    python analytics/eval_summary.py

    # Summarize a specific run
    python analytics/eval_summary.py \
        --run_dir /Users/.../data/v5/evaluation/2026_05_03_10_30_00

    # Don't write SUMMARY.md (just print)
    python analytics/eval_summary.py --no_save

The summary covers, per agent:
  internal section: BB/100, stderr, fold/all-in rate, decisions made,
                    use_opponent_embedding, use_mcts, action_temperature.
  slumbot section:  BB/100 raw + baseline-corrected, stderr, hands_failed,
                    clamps, decisions, settings.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# ----------------------------- Discovery --------------------------------

def _project_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    version = os.path.basename(os.path.abspath(os.path.join(here, "..")))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return version, project_root


def find_latest_eval_run(version: Optional[str] = None) -> Optional[str]:
    """Return latest `data/<version>/evaluation/<datetime>/` dir, or None."""
    if version is None:
        version, project_root = _project_paths()
    else:
        _, project_root = _project_paths()
    base = os.path.join(project_root, "data", version, "evaluation")
    if not os.path.isdir(base):
        return None
    subdirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )
    if not subdirs:
        return None
    return os.path.join(base, subdirs[-1])


def _load_section_json(section_dir: str) -> Optional[dict]:
    """Find the most recent JSON file inside a section dir."""
    if not os.path.isdir(section_dir):
        return None
    json_files = sorted(
        f for f in os.listdir(section_dir)
        if f.endswith(".json") and "config" not in f.lower()
    )
    if not json_files:
        return None
    with open(os.path.join(section_dir, json_files[-1])) as f:
        return json.load(f)


# ----------------------------- Formatting -------------------------------

def _fmt_num(x, fmt="{:+.2f}"):
    if x is None:
        return "-"
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return str(x)


def _action_index_summary(action_dist):
    """Return short string summarising fold/call/raise share."""
    if not action_dist:
        return "-"
    n = len(action_dist)
    fold = action_dist[0]
    call = action_dist[1] if n > 1 else 0
    allin = action_dist[-1] if n > 2 else 0
    raises = sum(action_dist[2:n - 1])
    return f"f={fold:.2f} c={call:.2f} r={raises:.2f} a={allin:.2f}"


# ---------------------- Internal evaluation --------------------

def _format_internal(internal: dict) -> list[str]:
    lines = ["## Internal evaluation"]
    lines.append("")
    if not internal:
        lines.append("_No internal evaluation results found._")
        return lines

    n_hands = internal.get("n_hands", "?")
    num_players = internal.get("num_players", "?")
    big_blind = internal.get("big_blind", "?")
    n_tables = internal.get("n_tables", "?")
    lines.append(
        f"`hands_dealt={n_hands}, num_players={num_players}, "
        f"big_blind={big_blind}, n_tables={n_tables}`"
    )
    lines.append("")
    agents = internal.get("agents", {})
    if not agents:
        lines.append("_No per-agent records._")
        return lines

    rows = []
    for name, a in agents.items():
        rows.append({
            "name": name,
            "hands": a.get("hands_played", 0),
            "bb100": a.get("bb_per_100"),
            "bb100_raw": a.get("bb_per_100_raw"),
            "stderr": a.get("stderr_bb_per_100"),
            "dec": a.get("decisions_made", 0),
            "fold": a.get("fold_rate"),
            "allin": a.get("allin_rate"),
            "temp": a.get("temperature"),
            "opp_emb": a.get("use_opponent_embedding", False),
            "mcts": a.get("use_mcts", False),
            "dist": a.get("action_distribution", []),
            "dist_by_street": a.get("action_distribution_by_street", []),
        })
    rows.sort(key=lambda r: r["bb100"] if r["bb100"] is not None else -float("inf"),
              reverse=True)

    header = ["Agent", "Hands", "BB/100", "BB/100 raw", "+-stderr",
              "Decisions", "Fold", "All-in", "Temp", "OppEmb", "MCTS"]
    table_rows = [header]
    for r in rows:
        table_rows.append([
            str(r["name"]),
            str(r["hands"]),
            _fmt_num(r["bb100"]),
            _fmt_num(r["bb100_raw"]),
            _fmt_num(r["stderr"], "{:.2f}"),
            str(r["dec"]),
            _fmt_num(r["fold"], "{:.3f}"),
            _fmt_num(r["allin"], "{:.3f}"),
            _fmt_num(r["temp"], "{:.2f}"),
            "Y" if r["opp_emb"] else " ",
            "Y" if r["mcts"] else " ",
        ])
    lines.extend(_render_table(table_rows))
    lines.append("")
    lines.append(
        "_BB/100 = leak-corrected (per-hand zero-sum). "
        "BB/100 raw = uncorrected; large gap signals env chip-leakage._"
    )
    lines.append("")

    # Per-agent action distributions
    street_names = ["preflop", "flop", "turn", "river"]
    has_streets = any(r["dist_by_street"] for r in rows)
    if has_streets:
        lines.append("### Action distributions")
        lines.append("")
        for r in rows:
            lines.append(f"**{r['name']}**: {_action_index_summary(r['dist'])}")
            if r["dist_by_street"]:
                for si, srow in enumerate(r["dist_by_street"]):
                    sname = street_names[si] if si < len(street_names) else f"street{si}"
                    lines.append(f"  {sname:>8}: {_action_index_summary(srow)}")
            lines.append("")
    return lines


# ---------------------- Slumbot evaluation --------------------

def _format_slumbot(slumbot: dict) -> list[str]:
    lines = ["## Slumbot evaluation"]
    lines.append("")
    if not slumbot:
        lines.append("_No slumbot evaluation results found._")
        return lines

    n_hands = slumbot.get("n_hands_per_agent", "?")
    chip_scale = slumbot.get("chip_scale", "?")
    bb_internal = slumbot.get("big_blind_internal", "?")
    lines.append(
        f"`hands_per_agent={n_hands}, chip_scale={chip_scale}, "
        f"big_blind_internal={bb_internal}`"
    )
    lines.append("")
    agents = slumbot.get("agents", {})
    if not agents:
        lines.append("_No per-agent records._")
        return lines

    rows = []
    for name, a in agents.items():
        rows.append({
            "name": name,
            "hands": a.get("hands_played", 0),
            "failed": a.get("hands_failed", 0),
            "raw": a.get("bb_per_100_raw"),
            "base": a.get("bb_per_100_baseline_corrected"),
            "residual": a.get("bb_per_100_aivat_residual"),
            "stderr": a.get("stderr_bb_per_100"),
            "stderr_base": a.get("stderr_bb_per_100_baseline_corrected"),
            "dec": a.get("decisions_made", 0),
            "fold": a.get("fold_rate"),
            "allin": a.get("allin_rate"),
            "temp": a.get("temperature"),
            "opp_emb": a.get("use_opponent_embedding", False),
            "mcts": a.get("use_mcts", False),
            "agent_type": a.get("type", "model"),
            "clamps": a.get("clamps", {}),
            "dist": a.get("action_distribution", []),
            "dist_by_street": a.get("action_distribution_by_street", []),
        })
    rows.sort(
        key=lambda r: r["raw"] if r["raw"] is not None else -float("inf"),
        reverse=True,
    )

    header = ["Agent", "Type", "Hands", "Fail", "BB/100 raw", "+-stderr",
              "BB/100 base", "+-stderr(b)", "Residual",
              "Fold", "All-in", "Temp", "OppEmb", "MCTS", "Clamps"]
    table_rows = [header]
    for r in rows:
        clamp_str = ",".join(
            f"{k[:1]}{v}" for k, v in (r["clamps"] or {}).items()) or "-"
        table_rows.append([
            str(r["name"]),
            str(r["agent_type"]),
            str(r["hands"]),
            str(r["failed"]),
            _fmt_num(r["raw"]),
            _fmt_num(r["stderr"], "{:.2f}"),
            _fmt_num(r["base"]),
            _fmt_num(r["stderr_base"], "{:.2f}"),
            _fmt_num(r["residual"]),
            _fmt_num(r["fold"], "{:.3f}"),
            _fmt_num(r["allin"], "{:.3f}"),
            _fmt_num(r["temp"], "{:.2f}"),
            "Y" if r["opp_emb"] else " ",
            "Y" if r["mcts"] else " ",
            clamp_str,
        ])
    lines.extend(_render_table(table_rows))
    lines.append("")

    # Per-agent action mix + per-street breakdown
    lines.append("### Action distributions")
    lines.append("")
    street_names = ["preflop", "flop", "turn", "river"]
    for r in rows:
        lines.append(f"**{r['name']}**: {_action_index_summary(r['dist'])}")
        if r["dist_by_street"]:
            for si, srow in enumerate(r["dist_by_street"]):
                sname = street_names[si] if si < len(street_names) else f"street{si}"
                lines.append(f"  {sname:>8}: {_action_index_summary(srow)}")
        lines.append("")
    return lines


# ---------------------- Markdown table renderer --------------------

def _render_table(rows: list[list[str]]) -> list[str]:
    """Render a list of rows (first row = header) as a Markdown pipe table."""
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    widths = [max(len(r[c]) for r in rows) for c in range(n_cols)]
    out = []
    out.append("| " + " | ".join(rows[0][c].ljust(widths[c]) for c in range(n_cols)) + " |")
    out.append("|" + "|".join("-" * (widths[c] + 2) for c in range(n_cols)) + "|")
    for r in rows[1:]:
        out.append("| " + " | ".join(r[c].ljust(widths[c]) for c in range(n_cols)) + " |")
    return out


# ---------------------- Top-level --------------------

def build_summary(run_dir: str) -> str:
    internal = _load_section_json(os.path.join(run_dir, "internal"))
    slumbot = _load_section_json(os.path.join(run_dir, "slumbot"))

    lines = []
    lines.append(f"# Eval summary — {os.path.basename(run_dir)}")
    lines.append("")
    lines.append(f"`run_dir = {run_dir}`")
    lines.append("")

    cfg_path = os.path.join(run_dir, "eval_config.json")
    if os.path.isfile(cfg_path):
        lines.append(f"Config snapshot: `{cfg_path}`")
        lines.append("")

    lines.extend(_format_internal(internal))
    lines.append("")
    lines.extend(_format_slumbot(slumbot))
    lines.append("")
    return "\n".join(lines)


def write_summary(run_dir: str, log=print) -> Optional[str]:
    """Build summary and write to <run_dir>/SUMMARY.md. Returns path or None."""
    text = build_summary(run_dir)
    out = os.path.join(run_dir, "SUMMARY.md")
    with open(out, "w") as f:
        f.write(text)
    log(f"Eval summary written to {out}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default=None,
                   help="Eval run directory (data/<v>/evaluation/<datetime>/). "
                        "If omitted, picks latest for current version.")
    p.add_argument("--no_save", action="store_true",
                   help="Don't write SUMMARY.md to disk (only print to stdout).")
    return p.parse_args()


def main():
    args = _parse_args()
    run_dir = args.run_dir or find_latest_eval_run()
    if run_dir is None:
        print("ERROR: no eval run directory found.")
        return 1
    text = build_summary(run_dir)
    print(text)
    if not args.no_save:
        out = os.path.join(run_dir, "SUMMARY.md")
        with open(out, "w") as f:
            f.write(text)
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
