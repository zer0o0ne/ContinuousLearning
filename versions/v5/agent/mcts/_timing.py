"""
Opt-in timing instrumentation for the parallel inference path.

Activated by env var `MCTS_TIMING=1`. Outputs jsonl records to a file given
by `MCTS_TIMING_PATH` (default `/tmp/mcts_timing_<pid>.jsonl`). Each process
(server + each actor) writes its own file via the pid suffix. Disabled mode
adds one `is_enabled` truthy check per call site — negligible.

Use:
    from agent.mcts import _timing
    if _timing.ENABLED:
        with _timing.span("server_batch", n_reqs=len(bucket), rtype=rtype):
            ...

Or for one-shot events:
    _timing.event("rpc_put", worker_id=wid, agent=name, payload_bytes=n)

A run is one process; records are written in append mode and flushed on
process exit. Cross-process aggregation happens in `bench_parallel.py`.
"""

import json
import os
import sys
import time
import atexit


ENABLED = os.environ.get("MCTS_TIMING", "0") == "1"

_buf = []
_BUF_LIMIT = 4096  # auto-flush when buffer reaches this many records
_path = None


def _resolve_path():
    base = os.environ.get("MCTS_TIMING_PATH", "/tmp/mcts_timing.jsonl")
    root, ext = os.path.splitext(base)
    return f"{root}_{os.getpid()}{ext or '.jsonl'}"


def _flush():
    global _buf
    if not _buf:
        return
    p = _path or _resolve_path()
    try:
        with open(p, "a") as f:
            for rec in _buf:
                f.write(json.dumps(rec, default=str))
                f.write("\n")
    except Exception as e:
        sys.stderr.write(f"[_timing] flush failed: {e}\n")
    _buf = []


if ENABLED:
    _path = _resolve_path()
    atexit.register(_flush)


def event(name, **fields):
    """Record a one-shot timestamped event."""
    if not ENABLED:
        return
    rec = {"t": time.monotonic(), "name": name, **fields}
    _buf.append(rec)
    if len(_buf) >= _BUF_LIMIT:
        _flush()


class span:
    """Context manager that records (start, end, duration_ms) for `name`."""

    __slots__ = ("name", "fields", "t0")

    def __init__(self, name, **fields):
        self.name = name
        self.fields = fields
        self.t0 = 0.0

    def __enter__(self):
        if ENABLED:
            self.t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not ENABLED:
            return False
        t1 = time.monotonic()
        rec = {"t": self.t0, "name": self.name,
               "duration_ms": (t1 - self.t0) * 1000.0, **self.fields}
        _buf.append(rec)
        if len(_buf) >= _BUF_LIMIT:
            _flush()
        return False
