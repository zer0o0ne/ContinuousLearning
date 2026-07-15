"""
Tests for MCTS PUCT formula, action selection, and visit distribution.

Covers:
  - PUCT formula correctness
  - Hero Q = W/N backup (visit-weighted average)
  - Opponent Q = W/N backup
  - Opponent pessimism blend
  - Dirichlet noise at root
  - Temperature-based action selection (_best_action)
  - get_n_distribution with and without label smoothing
  - Virtual loss apply / resolve
  - Strange traversal (inverse-N sampling)
  - re_backup_terminals

All tests operate directly on MCTSNode objects and MCTS internals; no
neural-network forwards are performed.
"""

import math
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Make sure the project root is on the path when running from
# versions/v7/tests/ or from versions/v7/ directly.
sys.path.insert(0, "/home/dev/ContinuousLearning/versions/v7")

from agent.mcts.mcts import (
    MCTS,
    MCTSNode,
    get_n_distribution,
    re_backup_terminals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mcts(c_puct=1.5, opp_pessimism_alpha=0.5, opp_prior_smoothing=0.0,
               strange_p=0.0, temperature=1.0, batch_size=1, virtual_loss=1.0,
               dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
               dirichlet_alpha_inner=0.3, dirichlet_epsilon_inner=0.05,
               search_scale=1.0):
    """Return an MCTS instance with a stub evaluator – no real agent needed."""
    mock_evaluator = MagicMock()
    mock_evaluator.n_actions = 5

    cfg = {
        "n_simulations": 100,
        "c_puct": c_puct,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_epsilon": dirichlet_epsilon,
        "dirichlet_alpha_inner": dirichlet_alpha_inner,
        "dirichlet_epsilon_inner": dirichlet_epsilon_inner,
        "temperature": temperature,
        "batch_size": batch_size,
        "virtual_loss": virtual_loss,
        "opp_prior_smoothing": opp_prior_smoothing,
        "opp_pessimism_alpha": opp_pessimism_alpha,
    }
    mcts = MCTS.__new__(MCTS)
    mcts.agent = None
    mcts.device = torch.device("cpu")
    mcts.search_scale = float(search_scale)
    mcts._root_credits = None
    mcts.evaluator = mock_evaluator
    mcts.n_simulations = cfg["n_simulations"]
    mcts.c_puct = c_puct
    mcts.n_actions = mock_evaluator.n_actions
    mcts.dirichlet_alpha = dirichlet_alpha
    mcts.dirichlet_epsilon = dirichlet_epsilon
    mcts.dirichlet_alpha_inner = dirichlet_alpha_inner
    mcts.dirichlet_epsilon_inner = dirichlet_epsilon_inner
    mcts.temperature = temperature
    mcts.batch_size = batch_size
    mcts.virtual_loss = virtual_loss
    mcts.opponent_emb_table = None
    mcts.opp_prior_smoothing = opp_prior_smoothing
    mcts.opp_pessimism_alpha = opp_pessimism_alpha
    mcts.strange_p = strange_p
    return mcts


def _make_hero_root_with_children(priors, child_Qs, child_Ns, c_puct=1.5,
                                   opp_pessimism_alpha=0.5):
    """Build a hero root node with given children stats and return (mcts, root)."""
    mcts = _make_mcts(c_puct=c_puct, opp_pessimism_alpha=opp_pessimism_alpha)
    root = MCTSNode(is_hero=True)
    root.N = sum(child_Ns)

    for i, (p, q, n) in enumerate(zip(priors, child_Qs, child_Ns)):
        child = MCTSNode(action_idx=i, parent=root, is_hero=False, P=p)
        child.N = n
        child.W = q * n  # consistent with Q = W/N
        child.Q = q
        root.children[i] = child

    root.W = sum(c.W for c in root.children.values())
    root.Q = root.W / root.N if root.N > 0 else 0.0
    return mcts, root


# ---------------------------------------------------------------------------
# 1. PUCT formula correctness
# ---------------------------------------------------------------------------

class TestPUCTFormula(unittest.TestCase):
    """UCB(a) = child.Q + c_puct * child.P * sqrt(N_parent) / (1 + child.N)"""

    def test_ucb_formula_single_child(self):
        """Manually verify the UCB score matches the formula for each child."""
        priors = [0.5, 0.3, 0.2]
        child_Qs = [0.1, 0.4, -0.2]
        child_Ns = [10, 2, 1]
        c_puct = 1.5

        mcts, root = _make_hero_root_with_children(priors, child_Qs, child_Ns,
                                                    c_puct=c_puct)
        sqrt_N = math.sqrt(root.N)

        for i, (p, q, n) in enumerate(zip(priors, child_Qs, child_Ns)):
            expected_ucb = q + c_puct * p * sqrt_N / (1 + n)
            child = root.children[i]
            actual_ucb = child.Q + c_puct * child.P * sqrt_N / (1 + child.N)
            self.assertAlmostEqual(actual_ucb, expected_ucb, places=10,
                                   msg=f"UCB mismatch for action {i}")

    def test_puct_selects_highest_ucb(self):
        """_select_child at a hero node picks the action with the highest UCB."""
        priors = [0.5, 0.3, 0.2]
        child_Qs = [0.1, 0.4, -0.2]
        child_Ns = [10, 2, 1]
        c_puct = 1.5

        mcts, root = _make_hero_root_with_children(priors, child_Qs, child_Ns,
                                                    c_puct=c_puct)
        sqrt_N = math.sqrt(root.N)
        ucbs = {i: child_Qs[i] + c_puct * priors[i] * sqrt_N / (1 + child_Ns[i])
                for i in range(3)}
        expected_action = max(ucbs, key=ucbs.get)

        selected = mcts._select_child(root, strange=False)
        self.assertEqual(selected, expected_action)

    def test_puct_exploration_term_decreases_with_visits(self):
        """The exploration bonus shrinks as a child accumulates visits."""
        c_puct = 2.0
        P = 0.5
        sqrt_N = math.sqrt(100)

        bonus_low_visits = c_puct * P * sqrt_N / (1 + 1)
        bonus_high_visits = c_puct * P * sqrt_N / (1 + 50)
        self.assertGreater(bonus_low_visits, bonus_high_visits)

    def test_puct_selects_unvisited_child_over_visited_bad_child(self):
        """An unvisited child (N=0) gets a large exploration bonus; should be
        preferred over a visited child with a strongly negative Q."""
        c_puct = 1.5
        # Action 0: visited many times, strongly negative Q
        # Action 1: unvisited
        priors = [0.5, 0.5]
        child_Qs = [-1.0, 0.0]
        child_Ns = [50, 0]

        mcts, root = _make_hero_root_with_children(priors, child_Qs, child_Ns,
                                                    c_puct=c_puct)
        sqrt_N = math.sqrt(root.N)
        ucb_0 = child_Qs[0] + c_puct * priors[0] * sqrt_N / (1 + child_Ns[0])
        ucb_1 = child_Qs[1] + c_puct * priors[1] * sqrt_N / (1 + child_Ns[1])
        self.assertGreater(ucb_1, ucb_0)

        selected = mcts._select_child(root, strange=False)
        self.assertEqual(selected, 1)

    def test_puct_c_puct_zero_selects_best_q(self):
        """With c_puct=0 the exploration term vanishes; argmax Q wins."""
        priors = [0.1, 0.8, 0.1]
        child_Qs = [0.5, 0.2, 0.8]  # action 2 has highest Q
        child_Ns = [5, 5, 5]

        mcts, root = _make_hero_root_with_children(priors, child_Qs, child_Ns,
                                                    c_puct=0.0)
        selected = mcts._select_child(root, strange=False)
        self.assertEqual(selected, 2)


# ---------------------------------------------------------------------------
# 2. Hero Q = W/N (visit-weighted average)
# ---------------------------------------------------------------------------

class TestHeroQBackup(unittest.TestCase):
    """Hero node Q should equal W/N (visit-weighted average)."""

    def test_hero_q_is_visit_weighted_avg(self):
        mcts = _make_mcts()
        root = MCTSNode(is_hero=True)
        root.N = 3
        root.W = 0.0

        child_qs = [0.1, 0.7, 0.3]
        for i, q in enumerate(child_qs):
            c = MCTSNode(action_idx=i, parent=root, is_hero=False, P=1/3)
            c.N = 1
            c.W = q
            c.Q = q
            root.W += q
            root.children[i] = c

        mcts._recompute_node_after_backup(root)
        expected = sum(child_qs) / 3
        self.assertAlmostEqual(root.Q, expected)

    def test_hero_q_is_w_over_n(self):
        """Hero Q = W/N regardless of children distribution."""
        mcts = _make_mcts()
        root = MCTSNode(is_hero=True)
        root.N = 2
        root.W = -0.4

        c0 = MCTSNode(action_idx=0, parent=root, is_hero=False, P=0.5)
        c0.N = 2
        c0.W = -0.4
        c0.Q = -0.2
        root.children[0] = c0

        c1 = MCTSNode(action_idx=1, parent=root, is_hero=False, P=0.5)
        c1.N = 0
        c1.Q = 1.0
        root.children[1] = c1

        mcts._recompute_node_after_backup(root)
        self.assertAlmostEqual(root.Q, -0.2)

    def test_hero_q_falls_back_to_w_over_n_when_no_visited_children(self):
        """If no children have been visited yet, fall back to W/N."""
        mcts = _make_mcts()
        node = MCTSNode(is_hero=True)
        node.N = 1
        node.W = 0.42

        # Expand with unvisited children
        for i in range(3):
            c = MCTSNode(action_idx=i, parent=node, is_hero=False, P=1/3)
            c.N = 0
            node.children[i] = c

        mcts._recompute_node_after_backup(node, leaf_value=0.42)
        self.assertAlmostEqual(node.Q, 0.42)

    def test_hero_q_updates_after_new_child_visit(self):
        """Hero Q = W/N reflects visit-weighted average after backup."""
        mcts = _make_mcts()
        root = MCTSNode(is_hero=True)
        root.N = 2
        root.W = 1.1  # 0.3 + 0.8

        c0 = MCTSNode(action_idx=0, parent=root, is_hero=False, P=0.5)
        c0.N = 1
        c0.W = 0.3
        c0.Q = 0.3
        root.children[0] = c0

        c1 = MCTSNode(action_idx=1, parent=root, is_hero=False, P=0.5)
        c1.N = 1
        c1.W = 0.8
        c1.Q = 0.8
        root.children[1] = c1

        mcts._recompute_node_after_backup(root)
        self.assertAlmostEqual(root.Q, 1.1 / 2)


# ---------------------------------------------------------------------------
# 3. Opponent Q = W/N
# ---------------------------------------------------------------------------

class TestOpponentQBackup(unittest.TestCase):
    """Opponent (non-hero) nodes use Q = W/N (then pessimism blend)."""

    def _make_opp_node_no_pessimism(self, W, N):
        mcts = _make_mcts(opp_pessimism_alpha=1.0)  # disable pessimism
        node = MCTSNode(is_hero=False)
        node.N = N
        node.W = W
        node.Q = 0.0  # will be computed
        return mcts, node

    def test_opp_q_equals_w_over_n(self):
        mcts, node = self._make_opp_node_no_pessimism(W=1.5, N=3)
        mcts._recompute_node_after_backup(node)
        self.assertAlmostEqual(node.Q, 1.5 / 3)

    def test_opp_q_with_negative_W(self):
        mcts, node = self._make_opp_node_no_pessimism(W=-2.0, N=4)
        mcts._recompute_node_after_backup(node)
        self.assertAlmostEqual(node.Q, -0.5)

    def test_opp_q_zero_when_no_visits(self):
        mcts, node = self._make_opp_node_no_pessimism(W=0.0, N=0)
        mcts._recompute_node_after_backup(node)
        self.assertAlmostEqual(node.Q, 0.0)

    def test_terminal_node_uses_w_over_n_regardless_of_is_hero(self):
        """Terminal node always uses Q = W/N, ignoring is_hero."""
        mcts = _make_mcts()
        node = MCTSNode(is_hero=True, is_terminal=True)
        node.N = 5
        node.W = 2.5
        mcts._recompute_node_after_backup(node)
        self.assertAlmostEqual(node.Q, 0.5)


# ---------------------------------------------------------------------------
# 4. Opponent pessimism blend
# ---------------------------------------------------------------------------

class TestOpponentPessimismBlend(unittest.TestCase):
    """Q_opp = alpha * E_P[Q_children] + (1-alpha) * min(Q_children)"""

    def _build_opp_node_with_children(self, child_priors, child_qs, child_ns,
                                       alpha):
        mcts = _make_mcts(opp_pessimism_alpha=alpha)
        node = MCTSNode(is_hero=False)
        total_n = sum(child_ns)
        node.N = total_n
        node.W = sum(q * n for q, n in zip(child_qs, child_ns))

        for i, (p, q, n) in enumerate(zip(child_priors, child_qs, child_ns)):
            c = MCTSNode(action_idx=i, parent=node, is_hero=True, P=p)
            c.N = n
            c.W = q * n
            c.Q = q
            node.children[i] = c

        return mcts, node

    def test_pessimism_blend_alpha_one_is_noop(self):
        """alpha=1 → _refresh_opp_q is a no-op (guard: opp_pessimism_alpha >= 1.0).

        Q stays at whatever W/N produced before the call — the legacy
        sample-mean path. The caller (recompute_node_after_backup) already
        set Q = W/N before calling _refresh_opp_q, so the sentinel survives.
        """
        priors = [0.6, 0.4]
        qs = [0.8, 0.2]
        ns = [5, 5]
        mcts, node = self._build_opp_node_with_children(priors, qs, ns, alpha=1.0)
        # Set a sentinel Q value (mirrors what _recompute_node_after_backup
        # sets via W/N before calling _refresh_opp_q).
        sentinel = node.W / node.N  # = W/N
        node.Q = sentinel
        mcts._refresh_opp_q(node)
        # With alpha >= 1.0, the function returns early — Q is unchanged.
        self.assertAlmostEqual(node.Q, sentinel, places=8)

    def test_pessimism_blend_alpha_zero_is_min_q(self):
        """alpha=0 → Q = min(child.Q) over visited children."""
        priors = [0.6, 0.4]
        qs = [0.8, 0.2]
        ns = [5, 5]
        mcts, node = self._build_opp_node_with_children(priors, qs, ns, alpha=0.0)
        mcts._refresh_opp_q(node)
        self.assertAlmostEqual(node.Q, min(qs), places=8)

    def test_pessimism_blend_intermediate_alpha(self):
        """Blend uses exact formula: alpha * E_P[Q] + (1-alpha) * min_Q."""
        priors = [0.7, 0.3]
        qs = [0.9, 0.1]
        ns = [3, 3]
        alpha = 0.5
        mcts, node = self._build_opp_node_with_children(priors, qs, ns, alpha=alpha)
        mcts._refresh_opp_q(node)

        total_p = sum(priors)
        expected_q = sum(p * q for p, q in zip(priors, qs)) / total_p
        min_q = min(qs)
        blend = alpha * expected_q + (1.0 - alpha) * min_q
        self.assertAlmostEqual(node.Q, blend, places=8)

    def test_pessimism_no_op_when_no_visited_children(self):
        """If no child is visited, _refresh_opp_q is a no-op."""
        mcts = _make_mcts(opp_pessimism_alpha=0.3)
        node = MCTSNode(is_hero=False)
        node.N = 0
        node.W = 0.0
        node.Q = 0.77  # arbitrary sentinel
        for i in range(3):
            c = MCTSNode(action_idx=i, parent=node, P=1/3)
            c.N = 0
            node.children[i] = c
        mcts._refresh_opp_q(node)
        self.assertAlmostEqual(node.Q, 0.77)  # untouched

    def test_pessimism_only_visited_children_contribute(self):
        """Unvisited children are excluded from E_P and min computation."""
        priors = [0.6, 0.4]
        qs = [0.5, 0.0]    # child 1 never visited
        ns = [4, 0]
        alpha = 0.5
        mcts, node = self._build_opp_node_with_children(priors, qs, ns, alpha=alpha)
        mcts._refresh_opp_q(node)
        # Only child 0 contributes → E_P = 0.5, min = 0.5 → blend = 0.5
        self.assertAlmostEqual(node.Q, 0.5, places=8)

    def test_pessimism_no_op_for_hero_node(self):
        """_refresh_opp_q is a no-op when called on a hero node."""
        mcts = _make_mcts(opp_pessimism_alpha=0.3)
        node = MCTSNode(is_hero=True)
        node.Q = 1.23  # sentinel
        mcts._refresh_opp_q(node)
        self.assertAlmostEqual(node.Q, 1.23)

    def test_recompute_opp_node_applies_pessimism(self):
        """_recompute_node_after_backup triggers _refresh_opp_q for opp nodes."""
        priors = [0.5, 0.5]
        qs = [0.8, 0.0]
        ns = [3, 3]
        alpha = 0.5
        mcts, node = self._build_opp_node_with_children(priors, qs, ns, alpha=alpha)
        # Set W/N baseline
        node.Q = sum(q * n for q, n in zip(qs, ns)) / sum(ns)
        mcts._recompute_node_after_backup(node)

        total_p = sum(priors)
        expected_q = sum(p * q for p, q in zip(priors, qs)) / total_p
        min_q = min(qs)
        blend = alpha * expected_q + (1.0 - alpha) * min_q
        self.assertAlmostEqual(node.Q, blend, places=8)


# ---------------------------------------------------------------------------
# 5. Dirichlet noise at root
# ---------------------------------------------------------------------------

class TestDirichletNoise(unittest.TestCase):
    """P'(a) = (1-eps) * P(a) + eps * Dir(alpha)"""

    def test_priors_sum_to_one_after_noise(self):
        mcts = _make_mcts(dirichlet_epsilon=0.25, dirichlet_alpha=0.3)
        root = MCTSNode(is_hero=True)
        priors = [0.5, 0.3, 0.2]
        for i, p in enumerate(priors):
            c = MCTSNode(action_idx=i, parent=root, P=p)
            root.children[i] = c

        mcts._add_dirichlet_noise(root)
        total = sum(root.children[a].P for a in root.children)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_noise_blends_by_epsilon(self):
        """After adding noise, each new prior is in [0, 1] and sums to 1."""
        eps = 0.25
        mcts = _make_mcts(dirichlet_epsilon=eps, dirichlet_alpha=0.3)
        root = MCTSNode(is_hero=True)
        original_priors = [0.6, 0.3, 0.1]
        for i, p in enumerate(original_priors):
            c = MCTSNode(action_idx=i, parent=root, P=p)
            root.children[i] = c

        np.random.seed(42)
        mcts._add_dirichlet_noise(root)

        for a, c in root.children.items():
            self.assertGreaterEqual(c.P, 0.0)
            self.assertLessEqual(c.P, 1.0)
        total = sum(c.P for c in root.children.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_noise_no_op_when_epsilon_zero(self):
        """With epsilon=0, priors are unchanged."""
        mcts = _make_mcts(dirichlet_epsilon=0.0)
        root = MCTSNode(is_hero=True)
        original_priors = [0.5, 0.3, 0.2]
        for i, p in enumerate(original_priors):
            c = MCTSNode(action_idx=i, parent=root, P=p)
            root.children[i] = c

        mcts._add_dirichlet_noise(root)
        for i, orig in enumerate(original_priors):
            self.assertAlmostEqual(root.children[i].P, orig)

    def test_noise_no_op_when_no_children(self):
        """_add_dirichlet_noise is a no-op on a leaf (no children)."""
        mcts = _make_mcts(dirichlet_epsilon=0.25)
        root = MCTSNode(is_hero=True)
        mcts._add_dirichlet_noise(root)  # should not raise

    def test_noise_formula_mean_effect(self):
        """After many samples, the noised prior mean converges toward the blend."""
        eps = 0.25
        alpha = 0.3
        n_trials = 2000
        n_actions = 4
        original_priors = [0.4, 0.3, 0.2, 0.1]
        noised_priors_sum = [0.0] * n_actions

        for _ in range(n_trials):
            mcts = _make_mcts(dirichlet_epsilon=eps, dirichlet_alpha=alpha)
            root = MCTSNode(is_hero=True)
            for i, p in enumerate(original_priors):
                c = MCTSNode(action_idx=i, parent=root, P=p)
                root.children[i] = c
            mcts._add_dirichlet_noise(root)
            for i in range(n_actions):
                noised_priors_sum[i] += root.children[i].P

        # E[P'(a)] = (1-eps)*P(a) + eps*(1/n_actions)
        for i in range(n_actions):
            empirical_mean = noised_priors_sum[i] / n_trials
            expected_mean = (1.0 - eps) * original_priors[i] + eps * (1.0 / n_actions)
            self.assertAlmostEqual(empirical_mean, expected_mean, delta=0.02,
                                   msg=f"Prior {i} mean mismatch")


# ---------------------------------------------------------------------------
# 6. Temperature-based action selection
# ---------------------------------------------------------------------------

class TestBestAction(unittest.TestCase):
    """_best_action: temp<=0 → argmax N, temp>0 → sample ∝ N^(1/temp)"""

    def _make_root_with_visit_counts(self, counts):
        root = MCTSNode(is_hero=True)
        root.N = sum(counts)
        for i, n in enumerate(counts):
            c = MCTSNode(action_idx=i, parent=root, P=1/len(counts))
            c.N = n
            root.children[i] = c
        return root

    def test_temp_zero_selects_argmax_n(self):
        mcts = _make_mcts(temperature=0.0)
        counts = [5, 20, 8, 2]
        root = self._make_root_with_visit_counts(counts)
        action = mcts._best_action(root)
        self.assertEqual(action, 1)  # action 1 has N=20

    def test_temp_negative_selects_argmax_n(self):
        mcts = _make_mcts(temperature=-1.0)
        counts = [3, 1, 15, 7]
        root = self._make_root_with_visit_counts(counts)
        action = mcts._best_action(root)
        self.assertEqual(action, 2)  # action 2 has N=15

    def test_temp_positive_is_stochastic(self):
        """With temperature=1, multiple actions should appear over many draws."""
        mcts = _make_mcts(temperature=1.0)
        counts = [50, 30, 20]
        root = self._make_root_with_visit_counts(counts)
        selected = set()
        for _ in range(500):
            selected.add(mcts._best_action(root))
        # All three actions should be selected at least once.
        self.assertEqual(len(selected), 3)

    def test_temp_high_flattens_distribution(self):
        """High temperature → near-uniform; low temperature → concentrates on max."""
        counts = [100, 10, 5]
        n_trials = 5000

        # High temperature: should pick actions roughly evenly weighted
        mcts_hot = _make_mcts(temperature=10.0)
        root_hot = self._make_root_with_visit_counts(counts)
        counts_hot = {i: 0 for i in range(3)}
        for _ in range(n_trials):
            counts_hot[mcts_hot._best_action(root_hot)] += 1

        # Low temperature: action 0 (max N) should dominate
        mcts_cold = _make_mcts(temperature=0.1)
        root_cold = self._make_root_with_visit_counts(counts)
        counts_cold = {i: 0 for i in range(3)}
        for _ in range(n_trials):
            counts_cold[mcts_cold._best_action(root_cold)] += 1

        # At high temp: action 0 should NOT monopolize (< 80%)
        self.assertLess(counts_hot[0] / n_trials, 0.80)
        # At low temp: action 0 should dominate (> 95%)
        self.assertGreater(counts_cold[0] / n_trials, 0.95)

    def test_temp_one_selects_proportional_to_n(self):
        """With temperature=1, P(a) ∝ N(a). Check empirical frequencies."""
        mcts = _make_mcts(temperature=1.0)
        counts = [60, 30, 10]
        root = self._make_root_with_visit_counts(counts)
        total = sum(counts)
        expected_probs = [c / total for c in counts]

        n_trials = 10000
        empirical = {i: 0 for i in range(3)}
        for _ in range(n_trials):
            empirical[mcts._best_action(root)] += 1

        for i in range(3):
            self.assertAlmostEqual(empirical[i] / n_trials, expected_probs[i],
                                   delta=0.02)


# ---------------------------------------------------------------------------
# 7. get_n_distribution
# ---------------------------------------------------------------------------

class TestGetNDistribution(unittest.TestCase):

    def _make_root(self, n_actions, child_visits):
        """child_visits: {action_idx: N}"""
        root = MCTSNode(is_hero=True)
        root.N = sum(child_visits.values())
        for a, n in child_visits.items():
            c = MCTSNode(action_idx=a, parent=root, P=1 / len(child_visits))
            c.N = n
            root.children[a] = c
        return root

    def test_basic_distribution_sums_to_one(self):
        root = self._make_root(5, {0: 10, 1: 5, 2: 15})
        dist = get_n_distribution(root, n_actions=5)
        self.assertAlmostEqual(sum(dist), 1.0, places=8)

    def test_illegal_actions_have_zero_mass(self):
        root = self._make_root(5, {1: 20, 3: 10})
        dist = get_n_distribution(root, n_actions=5)
        self.assertAlmostEqual(dist[0], 0.0)
        self.assertAlmostEqual(dist[2], 0.0)
        self.assertAlmostEqual(dist[4], 0.0)

    def test_proportional_to_visit_counts(self):
        root = self._make_root(4, {0: 6, 1: 4})
        dist = get_n_distribution(root, n_actions=4)
        self.assertAlmostEqual(dist[0], 6 / 10, places=8)
        self.assertAlmostEqual(dist[1], 4 / 10, places=8)

    def test_no_label_smoothing_sums_to_one(self):
        root = self._make_root(5, {0: 100, 2: 50, 4: 25})
        dist = get_n_distribution(root, n_actions=5, label_smoothing=0.0)
        self.assertAlmostEqual(sum(dist), 1.0, places=8)

    def test_label_smoothing_sums_to_one(self):
        root = self._make_root(5, {0: 100, 2: 50, 4: 25})
        dist = get_n_distribution(root, n_actions=5, label_smoothing=0.05)
        self.assertAlmostEqual(sum(dist), 1.0, places=6)

    def test_label_smoothing_illegal_actions_stay_zero(self):
        root = self._make_root(5, {0: 50, 1: 50})
        dist = get_n_distribution(root, n_actions=5, label_smoothing=0.1)
        # Actions 2, 3, 4 are illegal — must remain 0
        self.assertAlmostEqual(dist[2], 0.0)
        self.assertAlmostEqual(dist[3], 0.0)
        self.assertAlmostEqual(dist[4], 0.0)

    def test_label_smoothing_formula(self):
        """target[a] = (1-eps)*N(a)/sum(N) + eps/n_legal for legal actions."""
        child_visits = {0: 60, 1: 40}
        root = self._make_root(4, child_visits)
        eps = 0.1
        n_legal = 2
        total = 100
        dist = get_n_distribution(root, n_actions=4, label_smoothing=eps)

        for a, n in child_visits.items():
            expected = (1.0 - eps) * (n / total) + eps / n_legal
            self.assertAlmostEqual(dist[a], expected, places=8,
                                   msg=f"Smoothed dist mismatch for action {a}")

    def test_zero_total_visits_uniform_over_legal(self):
        """If no child has any visits, return uniform over legal actions."""
        root = MCTSNode(is_hero=True)
        for i in range(3):
            c = MCTSNode(action_idx=i, parent=root, P=1/3)
            c.N = 0
            root.children[i] = c
        dist = get_n_distribution(root, n_actions=5)
        for i in range(3):
            self.assertAlmostEqual(dist[i], 1/3, places=8)
        for i in range(3, 5):
            self.assertAlmostEqual(dist[i], 0.0)

    def test_no_legal_children_uniform_over_all_actions(self):
        """Pathological case: no children → uniform over full action set."""
        root = MCTSNode(is_hero=True)
        dist = get_n_distribution(root, n_actions=4)
        for p in dist:
            self.assertAlmostEqual(p, 0.25, places=8)


# ---------------------------------------------------------------------------
# 8. Virtual loss
# ---------------------------------------------------------------------------

class TestVirtualLoss(unittest.TestCase):
    """Virtual loss: N increases, W decreases along path; resolved by +vl+leaf."""

    def _make_chain(self, length=3):
        """Create a chain of nodes: root → c1 → c2 → ... with N=0, W=0."""
        nodes = [MCTSNode(is_hero=(i % 2 == 0)) for i in range(length)]
        for i in range(1, length):
            nodes[i].parent = nodes[i - 1]
            nodes[i - 1].children[0] = nodes[i]
            nodes[i].N = 0
            nodes[i].W = 0.0
        return nodes

    def test_apply_virtual_loss_increments_n(self):
        mcts = _make_mcts(virtual_loss=1.0)
        path = self._make_chain(3)
        mcts._apply_virtual_loss(path)
        for n in path:
            self.assertEqual(n.N, 1)

    def test_apply_virtual_loss_decrements_w(self):
        vl = 1.5
        mcts = _make_mcts(virtual_loss=vl)
        path = self._make_chain(3)
        mcts._apply_virtual_loss(path)
        for n in path:
            self.assertAlmostEqual(n.W, -vl)

    def test_virtual_loss_then_backup_net_effect(self):
        """After applying virtual loss then resolving (+vl + leaf_value),
        the net effect on W is exactly +leaf_value."""
        vl = 1.0
        leaf_value = 0.6
        mcts = _make_mcts(virtual_loss=vl)
        path = self._make_chain(3)

        initial_W = [n.W for n in path]
        initial_N = [n.N for n in path]

        # Apply virtual loss
        mcts._apply_virtual_loss(path)

        # Resolve: +vl + leaf_value
        for n in path:
            n.W += vl + leaf_value

        for i, n in enumerate(path):
            self.assertEqual(n.N, initial_N[i] + 1)
            self.assertAlmostEqual(n.W, initial_W[i] + leaf_value)

    def test_virtual_loss_two_paths_steer_away(self):
        """A second selection after virtual loss applied to one path should
        avoid the in-flight branch (PUCT score drops due to W decrease)."""
        mcts = _make_mcts(c_puct=1.5, virtual_loss=1.0, opp_pessimism_alpha=1.0)
        root = MCTSNode(is_hero=True)
        root.N = 10

        # Two children with identical initial UCB potential
        c0 = MCTSNode(action_idx=0, parent=root, is_hero=False, P=0.5)
        c0.N = 5
        c0.W = 0.5
        c0.Q = 0.1

        c1 = MCTSNode(action_idx=1, parent=root, is_hero=False, P=0.5)
        c1.N = 5
        c1.W = 0.5
        c1.Q = 0.1

        root.children = {0: c0, 1: c1}

        # Apply virtual loss to path through c0
        path_0 = [root, c0]
        mcts._apply_virtual_loss(path_0)

        # With virtual loss on c0, PUCT should now prefer c1
        # (c0.N increased and c0.W decreased → Q drops)
        selected = mcts._select_child(root, strange=False)
        self.assertEqual(selected, 1)


# ---------------------------------------------------------------------------
# 9. Label smoothing in visit distribution (already partially in section 7,
#    this section adds edge-case coverage)
# ---------------------------------------------------------------------------

class TestLabelSmoothingEdgeCases(unittest.TestCase):

    def test_single_legal_action_with_smoothing(self):
        """With one legal action and smoothing, its mass stays at 1.0."""
        root = MCTSNode(is_hero=True)
        c = MCTSNode(action_idx=2, parent=root, P=1.0)
        c.N = 10
        root.children[2] = c
        dist = get_n_distribution(root, n_actions=5, label_smoothing=0.05)
        self.assertAlmostEqual(dist[2], 1.0, places=8)

    def test_eps_zero_matches_raw_visit_distribution(self):
        """label_smoothing=0 must give the same result as calling without it."""
        root = MCTSNode(is_hero=True)
        for a, n in [(0, 70), (1, 20), (2, 10)]:
            c = MCTSNode(action_idx=a, parent=root, P=1/3)
            c.N = n
            root.children[a] = c

        dist_no_smooth = get_n_distribution(root, n_actions=5)
        dist_zero_eps = get_n_distribution(root, n_actions=5, label_smoothing=0.0)
        for i in range(5):
            self.assertAlmostEqual(dist_no_smooth[i], dist_zero_eps[i], places=12)

    def test_smoothed_minimum_mass_per_legal_action(self):
        """Every legal action gets at least eps/n_legal mass after smoothing."""
        root = MCTSNode(is_hero=True)
        for a, n in [(0, 1000), (1, 0), (2, 0)]:
            c = MCTSNode(action_idx=a, parent=root, P=1/3)
            c.N = n
            root.children[a] = c

        eps = 0.05
        n_legal = 3
        dist = get_n_distribution(root, n_actions=5, label_smoothing=eps)
        floor = eps / n_legal
        for a in [0, 1, 2]:
            self.assertGreaterEqual(dist[a], floor - 1e-12)


# ---------------------------------------------------------------------------
# 10. Strange traversal (inverse-N sampling)
# ---------------------------------------------------------------------------

class TestStrangeTraversal(unittest.TestCase):
    """strange=True: hero selects with P(a) ∝ 1/(N(a)+1); opp unchanged."""

    def test_strange_samples_from_all_children(self):
        """Over many trials, inverse-N sampling visits all children."""
        mcts = _make_mcts(strange_p=1.0)
        root = MCTSNode(is_hero=True)
        root.N = 100
        # One very heavily visited action and two lightly visited
        for a, n in [(0, 90), (1, 5), (2, 5)]:
            c = MCTSNode(action_idx=a, parent=root, P=1/3)
            c.N = n
            root.children[a] = c

        selected_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(500):
            selected_counts[mcts._select_child(root, strange=True)] += 1

        # All three should be selected
        for a in [0, 1, 2]:
            self.assertGreater(selected_counts[a], 0)

    def test_strange_prefers_least_visited(self):
        """Inverse-N sampling should prefer the least-visited action."""
        mcts = _make_mcts(strange_p=1.0)
        root = MCTSNode(is_hero=True)
        root.N = 55
        # action 0: very visited; action 1: unvisited
        for a, n in [(0, 50), (1, 0)]:
            c = MCTSNode(action_idx=a, parent=root, P=0.5)
            c.N = n
            root.children[a] = c

        # weights: action 0 → 1/51, action 1 → 1/1 = 1.0
        # P(action 1) ≈ 1.0/(1.0+1/51) ≈ 0.98
        selected_counts = {0: 0, 1: 0}
        for _ in range(1000):
            selected_counts[mcts._select_child(root, strange=True)] += 1

        # action 1 should be chosen much more often
        self.assertGreater(selected_counts[1], selected_counts[0] * 5)

    def test_strange_weights_formula(self):
        """Verify P(a) ∝ 1/(N(a)+1) matches empirical frequencies."""
        mcts = _make_mcts(strange_p=1.0)
        root = MCTSNode(is_hero=True)
        root.N = 16
        ns = {0: 5, 1: 10, 2: 1}
        for a, n in ns.items():
            c = MCTSNode(action_idx=a, parent=root, P=1/3)
            c.N = n
            root.children[a] = c

        weights = {a: 1.0 / (n + 1) for a, n in ns.items()}
        total_w = sum(weights.values())
        expected_probs = {a: w / total_w for a, w in weights.items()}

        n_trials = 10000
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(n_trials):
            counts[mcts._select_child(root, strange=True)] += 1

        for a in [0, 1, 2]:
            self.assertAlmostEqual(counts[a] / n_trials, expected_probs[a],
                                   delta=0.03)

    def test_strange_false_uses_puct_not_inverse_n(self):
        """With strange=False on a hero node, PUCT is used (deterministic)."""
        mcts = _make_mcts(c_puct=2.0, strange_p=0.0)
        root = MCTSNode(is_hero=True)
        root.N = 20
        # action 1 has much higher Q, should win PUCT
        for a, (q, n, p) in enumerate([(0.0, 10, 0.5), (1.0, 10, 0.5)]):
            c = MCTSNode(action_idx=a, parent=root, P=p)
            c.N = n
            c.Q = q
            root.children[a] = c

        selected = mcts._select_child(root, strange=False)
        self.assertEqual(selected, 1)

    def test_strange_opp_node_uses_prior_sampling_unchanged(self):
        """At opp nodes, strange=True still uses prior-based sampling."""
        mcts = _make_mcts(strange_p=1.0)
        node = MCTSNode(is_hero=False)
        node.N = 50
        priors = {0: 0.9, 1: 0.1}
        for a, p in priors.items():
            c = MCTSNode(action_idx=a, parent=node, P=p)
            c.N = 20  # equal visits, so inverse-N would pick uniformly
            node.children[a] = c

        counts = {0: 0, 1: 0}
        for _ in range(2000):
            counts[mcts._select_child(node, strange=True)] += 1

        # Prior-based: action 0 (P=0.9) should dominate
        self.assertGreater(counts[0], counts[1] * 3)


# ---------------------------------------------------------------------------
# 11. re_backup_terminals
# ---------------------------------------------------------------------------

class TestReBackupTerminals(unittest.TestCase):
    """re_backup_terminals propagates delta=(new_Q*N - old_W) up to root."""

    def _make_tree_with_terminal(self, initial_term_q, new_term_q, n_visits,
                                  opp_pessimism_alpha=1.0):
        """
        root (hero) → child (opp) → terminal

        root.N = child.N = terminal.N = n_visits
        initial: terminal.Q = initial_term_q, terminal.W = initial_term_q * n_visits
        """
        root = MCTSNode(is_hero=True)
        child = MCTSNode(action_idx=0, parent=root, is_hero=False, P=1.0)
        terminal = MCTSNode(action_idx=0, parent=child, is_hero=False,
                            is_terminal=True, P=1.0)

        terminal.N = n_visits
        terminal.W = initial_term_q * n_visits
        terminal.Q = initial_term_q
        terminal._term_value = initial_term_q  # cached from search

        child.N = n_visits
        child.W = initial_term_q * n_visits
        child.Q = initial_term_q

        root.N = n_visits
        root.W = initial_term_q * n_visits
        root.Q = initial_term_q

        child.children[0] = terminal
        root.children[0] = child

        # Override terminal Q to the equity-based value
        terminal.Q = new_term_q

        return root, child, terminal

    def test_terminal_w_updated(self):
        root, child, terminal = self._make_tree_with_terminal(
            initial_term_q=0.1, new_term_q=0.5, n_visits=5)
        re_backup_terminals(root, opp_pessimism_alpha=1.0)
        self.assertAlmostEqual(terminal.W, 0.5 * 5, places=8)

    def test_ancestor_w_updated_by_delta(self):
        initial_q, new_q, n = 0.1, 0.5, 5
        root, child, terminal = self._make_tree_with_terminal(
            initial_term_q=initial_q, new_term_q=new_q, n_visits=n)
        delta = (new_q - initial_q) * n

        root_w_before = root.W
        child_w_before = child.W

        re_backup_terminals(root, opp_pessimism_alpha=1.0)

        self.assertAlmostEqual(child.W, child_w_before + delta, places=8)
        self.assertAlmostEqual(root.W, root_w_before + delta, places=8)

    def test_hero_q_updated_to_w_over_n(self):
        """After backup, hero root.Q = W/N."""
        initial_q, new_q, n = 0.2, 0.9, 4
        root, child, terminal = self._make_tree_with_terminal(
            initial_term_q=initial_q, new_term_q=new_q, n_visits=n)
        re_backup_terminals(root, opp_pessimism_alpha=1.0)
        # child.Q = new_q (W/N after delta propagation), root has one child
        # root.Q = W/N = child.Q (single child, same W/N)
        self.assertAlmostEqual(root.Q, new_q, places=8)

    def test_idempotent(self):
        """Running re_backup_terminals twice produces the same result."""
        initial_q, new_q, n = 0.3, 0.7, 6
        root, child, terminal = self._make_tree_with_terminal(
            initial_term_q=initial_q, new_term_q=new_q, n_visits=n)

        re_backup_terminals(root, opp_pessimism_alpha=1.0)
        root_q_first = root.Q
        root_w_first = root.W

        re_backup_terminals(root, opp_pessimism_alpha=1.0)
        self.assertAlmostEqual(root.Q, root_q_first, places=10)
        self.assertAlmostEqual(root.W, root_w_first, places=10)

    def test_zero_visit_terminal_skipped(self):
        """Terminal with N=0 is skipped (no delta)."""
        root = MCTSNode(is_hero=True)
        child = MCTSNode(action_idx=0, parent=root, is_hero=False,
                         is_terminal=True, P=1.0)
        child.N = 0
        child.W = 0.0
        child.Q = 0.99  # will change Q but N=0 → skipped
        root.N = 0
        root.W = 0.0
        root.Q = 0.0
        root.children[0] = child

        re_backup_terminals(root, opp_pessimism_alpha=1.0)
        self.assertAlmostEqual(root.W, 0.0)  # no change
        self.assertAlmostEqual(child.W, 0.0)  # no change


# ---------------------------------------------------------------------------
# 12. _recompute_node_after_backup — integrated correctness
# ---------------------------------------------------------------------------

class TestRecomputeNodeIntegrated(unittest.TestCase):
    """Multi-level tree: verify bottom-up Q propagation correctness."""

    def test_two_level_hero_opp_backup(self):
        """
        root (hero)
          action 0 → opp_node
              action 0 → leaf (terminal, Q=0.8)
              action 1 → leaf (terminal, Q=0.3)
        opp node gets W/N (no pessimism).
        root.Q = W/N = opp_node.Q (single child).
        """
        mcts = _make_mcts(opp_pessimism_alpha=1.0)

        root = MCTSNode(is_hero=True)
        opp_node = MCTSNode(action_idx=0, parent=root, is_hero=False, P=1.0)
        leaf0 = MCTSNode(action_idx=0, parent=opp_node, is_hero=False,
                         is_terminal=True, P=0.6)
        leaf1 = MCTSNode(action_idx=1, parent=opp_node, is_hero=False,
                         is_terminal=True, P=0.4)

        # Simulate backup of leaf0 (value=0.8) × 2, leaf1 (value=0.3) × 1
        leaf0.N, leaf0.W, leaf0.Q = 2, 1.6, 0.8
        leaf1.N, leaf1.W, leaf1.Q = 1, 0.3, 0.3
        opp_node.children = {0: leaf0, 1: leaf1}
        root.children = {0: opp_node}

        # opp_node: W = 1.6+0.3=1.9, N=3
        opp_node.N = 3
        opp_node.W = 1.9

        # root: W = 1.9, N = 3
        root.N = 3
        root.W = 1.9

        # Recompute bottom-up
        for n in [leaf0, leaf1, opp_node, root]:
            mcts._recompute_node_after_backup(n)

        self.assertAlmostEqual(opp_node.Q, 1.9 / 3, places=8)
        self.assertAlmostEqual(root.Q, opp_node.Q, places=8)

    def test_hero_q_is_visit_weighted_with_strange_path(self):
        """Hero Q = W/N: bad children contribute proportionally to their visits."""
        mcts = _make_mcts()
        root = MCTSNode(is_hero=True)
        root.N = 12

        good_child = MCTSNode(action_idx=0, parent=root, P=0.7)
        good_child.N = 10
        good_child.W = 9.0
        good_child.Q = 0.9

        bad_child = MCTSNode(action_idx=1, parent=root, P=0.3)
        bad_child.N = 2
        bad_child.W = -1.0
        bad_child.Q = -0.5

        root.children = {0: good_child, 1: bad_child}
        root.W = good_child.W + bad_child.W  # 8.0

        mcts._recompute_node_after_backup(root)
        # Hero Q = W/N = 8.0 / 12
        self.assertAlmostEqual(root.Q, 8.0 / 12, places=8)


# ---------------------------------------------------------------------------
# 13. MCTSNode slots and initialization
# ---------------------------------------------------------------------------

class TestMCTSNodeInit(unittest.TestCase):

    def test_default_init(self):
        node = MCTSNode()
        self.assertIsNone(node.action_idx)
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, {})
        self.assertTrue(node.is_hero)
        self.assertFalse(node.is_terminal)
        self.assertEqual(node.N, 0)
        self.assertAlmostEqual(node.W, 0.0)
        self.assertAlmostEqual(node.Q, 0.0)
        self.assertAlmostEqual(node.P, 0.0)
        self.assertIsNone(node.action_embedding)
        self.assertIsNone(node._term_value)

    def test_custom_init(self):
        parent = MCTSNode()
        emb = torch.zeros(16)
        node = MCTSNode(action_idx=3, parent=parent, is_hero=False,
                        is_terminal=True, P=0.42, action_embedding=emb)
        self.assertEqual(node.action_idx, 3)
        self.assertIs(node.parent, parent)
        self.assertFalse(node.is_hero)
        self.assertTrue(node.is_terminal)
        self.assertAlmostEqual(node.P, 0.42)
        self.assertIs(node.action_embedding, emb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
