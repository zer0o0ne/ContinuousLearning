"""Tests for 2026-06-30 fixes:
  1. perception.py: numpy array stacks in boolean context (slumbot ValueError)
  2. mcts_predict/train.py: checkpoint cleanup respects save_every_cycles

Run (from versions/v6):
    python -m pytest tests/test_fixes_june30.py -v
    # or:
    python -m tests.test_fixes_june30
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.perception.perception import extract_event_tensors, EventSequenceEmbedder
from agent.train_scenarios.mcts_predict.train import _cleanup_old_snapshots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ACTIONS = 6
MAX_PLAYERS = 2


def _make_event(stacks_type="numpy"):
    """Build a minimal event dict.

    stacks_type controls the type of the 'stacks' field:
      "numpy"  — np.ndarray (slumbot_eval produces this)
      "list"   — Python list (training data loader produces this)
      "none"   — key absent (legacy events without B.6.2)
    """
    action = torch.zeros(N_ACTIONS, dtype=torch.float32)
    action[1] = 1.0
    event = {
        "table": [0, 1, 2, 3, 4],
        "hand": [10, 11],
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": MAX_PLAYERS,
        "pot": 100.0,
        "stack": 500.0,
        "bets": np.array([5.0, 10.0], dtype=np.float32),
        "action": action,
    }
    if stacks_type == "numpy":
        event["stacks"] = np.array([500.0, 490.0], dtype=np.float32)
    elif stacks_type == "list":
        event["stacks"] = [500.0, 490.0]
    # "none" — no stacks key
    return event


# ---------------------------------------------------------------------------
# 1. Numpy stacks in perception — must not raise ValueError
# ---------------------------------------------------------------------------

def test_extract_event_tensors_numpy_stacks():
    """extract_event_tensors must handle numpy array stacks without error."""
    events = [_make_event("numpy")]
    result = extract_event_tensors([events], MAX_PLAYERS)
    assert result is not None
    assert result["stacks"].shape == (1, MAX_PLAYERS)
    np.testing.assert_allclose(
        result["stacks"].numpy(), [[500.0, 490.0]], atol=1e-5)


def test_extract_event_tensors_list_stacks():
    """extract_event_tensors must still work with Python list stacks."""
    events = [_make_event("list")]
    result = extract_event_tensors([events], MAX_PLAYERS)
    assert result is not None
    np.testing.assert_allclose(
        result["stacks"].numpy(), [[500.0, 490.0]], atol=1e-5)


def test_extract_event_tensors_missing_stacks():
    """extract_event_tensors must handle missing stacks (zeros)."""
    events = [_make_event("none")]
    result = extract_event_tensors([events], MAX_PLAYERS)
    assert result is not None
    np.testing.assert_allclose(
        result["stacks"].numpy(), [[0.0, 0.0]], atol=1e-5)


def test_build_batch_tensors_numpy_stacks():
    """_build_batch_tensors must handle numpy array stacks."""
    emb = EventSequenceEmbedder(16, N_ACTIONS, MAX_PLAYERS)
    events = [[_make_event("numpy")]]
    # This used to raise ValueError: truth value of array is ambiguous
    bt = emb._build_batch_tensors(events, device="cpu")
    assert bt is not None
    assert bt["stacks_emb"].shape[0] == 1


def test_embed_event_numpy_stacks():
    """embed_event must handle numpy array stacks."""
    emb = EventSequenceEmbedder(16, N_ACTIONS, MAX_PLAYERS)
    event = _make_event("numpy")
    # This used to raise ValueError: truth value of array is ambiguous
    out = emb.embed_event(event, device="cpu")
    assert out.shape == (7, 16)


# ---------------------------------------------------------------------------
# 2. Checkpoint cleanup respects save_every_cycles
# ---------------------------------------------------------------------------

def _populate_snapshots(tmpdir, cycle_ids):
    """Create dummy cycle_NNNN.pt files and return the directory."""
    for cid in cycle_ids:
        path = os.path.join(tmpdir, f"cycle_{cid:04d}.pt")
        with open(path, "w") as f:
            f.write("x")
    return tmpdir


def _surviving(tmpdir):
    """Return sorted list of cycle ids that still exist."""
    ids = []
    for f in os.listdir(tmpdir):
        if f.startswith("cycle_") and f.endswith(".pt"):
            ids.append(int(f.replace("cycle_", "").replace(".pt", "")))
    return sorted(ids)


def test_cleanup_keep_every_matches_save_every():
    """With keep_every=save_every_cycles, all written snapshots survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # save_every_cycles=5, 50 cycles → snapshots at 0,5,10,...,45,49
        saved = list(range(0, 50, 5)) + [49]
        _populate_snapshots(tmpdir, saved)
        _cleanup_old_snapshots(tmpdir, keep_last=3, keep_every=5)
        remaining = _surviving(tmpdir)
        # All multiples of 5 kept by keep_every, 49 kept by keep_last
        assert remaining == sorted(saved), (
            f"expected {sorted(saved)}, got {remaining}")


def test_cleanup_keep_every_1():
    """With save_every_cycles=1 (keep_every=1), all snapshots survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = list(range(20))
        _populate_snapshots(tmpdir, saved)
        _cleanup_old_snapshots(tmpdir, keep_last=3, keep_every=1)
        remaining = _surviving(tmpdir)
        assert remaining == saved, f"expected all kept, got {remaining}"


def test_cleanup_hardcoded_10_deletes_intermediates():
    """Old behavior (keep_every=10) would delete intermediate snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = list(range(0, 50, 5)) + [49]
        _populate_snapshots(tmpdir, saved)
        _cleanup_old_snapshots(tmpdir, keep_last=3, keep_every=10)
        remaining = _surviving(tmpdir)
        # With keep_every=10: keeps 0,10,20,30,40 + last 3 (40,45,49)
        # So 5,15,25,35 would be deleted
        assert 5 not in remaining, "keep_every=10 should delete cycle 5"
        assert 15 not in remaining, "keep_every=10 should delete cycle 15"


def test_cleanup_few_files_no_delete():
    """Fewer than keep_last files → nothing deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = [0, 5]
        _populate_snapshots(tmpdir, saved)
        _cleanup_old_snapshots(tmpdir, keep_last=3, keep_every=5)
        assert _surviving(tmpdir) == saved


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
