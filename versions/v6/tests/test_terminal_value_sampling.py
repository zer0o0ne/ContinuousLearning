"""Tests for random terminal-value target sampling (n_terminal_values).

Replaced the K_worst/K_best tails-only selection: uniform random sampling
covers the whole outcome range, so the value head is not trained only on
very winning / very losing situations.

All tests deterministic: module `random` is explicitly seeded.

Run (from versions/v6):
    python -m tests.test_terminal_value_sampling
"""

import random

from agent.mcts.collect import _select_terminal_targets


class _Node:
    def __init__(self, parent=None, action_idx=None, q=0.0, terminal=False):
        self.parent = parent
        self.action_idx = action_idx
        self.Q = q
        self.is_terminal = terminal
        self.children = {}


def _make_tree(qs):
    """Root with one terminal child per Q value (action_idx = index)."""
    root = _Node()
    for i, q in enumerate(qs):
        root.children[i] = _Node(parent=root, action_idx=i, q=q, terminal=True)
    return root


def test_selector_samples_n_random_terminals():
    root = _make_tree([float(q) for q in range(10)])

    random.seed(123)
    out1 = _select_terminal_targets(root, 4)
    random.seed(123)
    out2 = _select_terminal_targets(root, 4)
    assert out1 == out2, "not reproducible under a fixed seed"
    assert len(out1) == 4
    qs = sorted(q for _, q in out1)
    # With seed 123 the draw includes non-tail (middling) values — the whole
    # point of the change. Exact draw pinned for determinism.
    assert qs == [3.0, 5.0, 8.0, 9.0], qs
    assert qs != [0.0, 1.0, 8.0, 9.0], "degenerated to tails-only selection"
    # Paths are the action indices of the sampled terminals
    for path, q in out1:
        assert path == [int(q)]
    print("test_selector_samples_n_random_terminals: OK")


def test_selector_n_zero_and_small_tree():
    root = _make_tree([1.0, 2.0, 3.0])
    assert _select_terminal_targets(root, 0) == []
    assert _select_terminal_targets(root, -3) == []
    # Fewer terminals than requested → all of them, no sampling
    random.seed(7)
    out = _select_terminal_targets(root, 8)
    assert sorted(q for _, q in out) == [1.0, 2.0, 3.0]
    # Empty tree
    assert _select_terminal_targets(_Node(), 4) == []
    print("test_selector_n_zero_and_small_tree: OK")


def test_selector_clip():
    root = _make_tree([-10.0, 0.5, 10.0])
    random.seed(1)
    out = _select_terminal_targets(root, 8, clip_val=5.0)
    qs = sorted(q for _, q in out)
    assert qs == [-5.0, 0.5, 5.0], qs
    print("test_selector_clip: OK")


if __name__ == "__main__":
    test_selector_samples_n_random_terminals()
    test_selector_n_zero_and_small_tree()
    test_selector_clip()
    print("\nALL TERMINAL-VALUE SAMPLING TESTS PASSED")
