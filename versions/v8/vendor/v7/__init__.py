"""Frozen snapshot of v7's agent code (CONCEPT.md §4.3, §16 OI-6).

Copied verbatim from `versions/v7/agent/` on 2026-08-16, with the sole change
that absolute imports `agent.…` were repointed to `vendor.v7.…` — two version
trees cannot be imported into one process (`CLAUDE.md` §2), so a v7 checkpoint
can only be run from a copy living inside v8.

**This package is never edited to follow v8 changes.** It exists to make v7
checkpoints usable as opponent-pool members and nothing else. If v8's own
architecture moves, this copy stays where v7 left it.

Contents:
  attn_utils.py            v7/agent/attn_utils.py
  perception/*             v7/agent/perception/*
  action/*                 v7/agent/action/*
  agent.py                 NOT a copy — the perception+action-head subset of
                           v7's `ASI`, which is all a pool member needs
                           (`heads={"action"}`, `skip_opponent_emb=True`).
  events.py                NOT a copy — the v7 event format, reconstructed from
                           the reference implementation that came into v8 with
                           `evaluation/slumbot_eval.py::_build_events`.
"""
