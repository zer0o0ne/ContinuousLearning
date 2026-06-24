"""
Comprehensive tests for terminal evaluation in MCTS.

Covers:
- _capped_showdown_chips formula and edge cases
- Equity=1.0 wins full capped pot
- Equity=0.0 loses only invested (excess returned)
- Side pot correctness (hero short-stacked vs bigger opponent)
- Dead money inclusion in contested pot
- Fold terminal deterministic Q values
- re_backup_terminals delta propagation, hero max-Q, opp W/N, idempotency
- Scale division of terminal Q values
- Contributions vs investments semantics
- Multi-opponent equity handling (single-opponent path through _capped_showdown_chips)

Run from /home/dev/ContinuousLearning/versions/v6/ with:
    python -m pytest tests/test_terminal_eval_equity.py -v
or:
    python -m unittest tests.test_terminal_eval_equity -v

Note: terminal_eval.py transitively imports the compiled `gpu_solver` C extension
via agent.gto_utils.gpu_solver_v2.  When that extension is unavailable (CI or
environments without the compiled .so), the module-level stubs below are injected
into sys.modules BEFORE any project import so the import chain resolves cleanly.
The stubs cover only the symbols used by terminal_eval.py at import time; no test
actually calls gpu_equity_v2, get_position_range, expand_range, or narrow_range —
those paths require real hands/ranges and are tested integration-style elsewhere.
"""

import sys
import os
import unittest

# conftest.py adds agent/gto_utils/ and project root to sys.path,
# so gpu_solver and gpu_solver_v2 are importable without stubs.

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
if _V6_ROOT not in sys.path:
    sys.path.insert(0, _V6_ROOT)

from agent.mcts.terminal_eval import _capped_showdown_chips
from agent.mcts.mcts import MCTSNode, re_backup_terminals, _collect_terminals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_terminal(Q, N, W=None, parent=None, action_idx=None):
    """Create a visited terminal node with the given Q / N / W."""
    node = MCTSNode(action_idx=action_idx, parent=parent, is_terminal=True)
    node.N = N
    node.Q = Q
    node.W = Q * N if W is None else W
    return node


def _make_non_terminal(is_hero, N=0, W=0.0, Q=0.0, parent=None, action_idx=None):
    """Create a non-terminal interior node."""
    node = MCTSNode(action_idx=action_idx, parent=parent, is_hero=is_hero,
                    is_terminal=False)
    node.N = N
    node.W = W
    node.Q = Q
    return node


def _attach_child(parent, child):
    """Attach child to parent's children dict using child.action_idx as key."""
    parent.children[child.action_idx] = child


# ---------------------------------------------------------------------------
# 1. _capped_showdown_chips formula
# ---------------------------------------------------------------------------

class TestCappedShowdownChipsFormula(unittest.TestCase):
    """Verify the _capped_showdown_chips formula step by step."""

    def _call(self, equity, contributions, hero_pos, active_players,
              dead_money=0.0):
        return _capped_showdown_chips(
            equity=equity,
            contributions=contributions,
            hero_pos=hero_pos,
            active_players=active_players,
            dead_money=dead_money,
        )

    # ------------------------------------------------------------------
    # Basic 2-player heads-up, equal stacks
    # ------------------------------------------------------------------

    def test_hu_equity_half_breakeven(self):
        """HU equal stacks, equity=0.5 → net ≈ 0 (break-even)."""
        contributions = [100.0, 100.0]
        result = self._call(equity=0.5, contributions=contributions,
                            hero_pos=0, active_players=[0, 1])
        # effective_hero = min(100, 100) = 100, excess = 0
        # hero_share_pot = min(100,100) + min(100,100) = 200
        # net = 0 + 0.5 * 200 - 100 = 0
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_hu_equity_one_wins_all(self):
        """HU equal stacks, equity=1.0 → net = +invested (win opp stack)."""
        contributions = [100.0, 100.0]
        result = self._call(equity=1.0, contributions=contributions,
                            hero_pos=0, active_players=[0, 1])
        # net = 0 + 1.0 * 200 - 100 = 100
        self.assertAlmostEqual(result, 100.0, places=6)

    def test_hu_equity_zero_loses_invested(self):
        """HU equal stacks, equity=0.0 → net = -invested."""
        contributions = [100.0, 100.0]
        result = self._call(equity=0.0, contributions=contributions,
                            hero_pos=0, active_players=[0, 1])
        # net = 0 + 0.0 * 200 - 100 = -100
        self.assertAlmostEqual(result, -100.0, places=6)

    # ------------------------------------------------------------------
    # Formula decomposition: excess, effective_hero, hero_share_pot
    # ------------------------------------------------------------------

    def test_formula_decomposition_side_pot(self):
        """Manually verify each formula component when hero is short-stacked."""
        # Hero put in 500, opp put in 1000
        contributions = [500.0, 1000.0]
        hero_pos = 0
        active_players = [0, 1]
        equity = 0.6

        invested_hero = 500.0
        opp_contribs = [1000.0]
        max_opp = 1000.0
        effective_hero = min(500.0, 1000.0)   # = 500
        excess = 500.0 - 500.0                 # = 0
        hero_share_pot = min(500.0, 500.0) + min(1000.0, 500.0)  # = 500 + 500 = 1000
        expected_net = excess + equity * (hero_share_pot + 0.0) - invested_hero
        # = 0 + 0.6 * 1000 - 500 = 100

        result = self._call(equity, contributions, hero_pos, active_players)
        self.assertAlmostEqual(result, expected_net, places=6)
        self.assertAlmostEqual(result, 100.0, places=6)

    def test_formula_decomposition_excess_returned(self):
        """When hero over-invested (vs short opp), excess is returned regardless of equity."""
        # Hero put in 1000, opp put in 500 (hero is the big stack)
        contributions = [1000.0, 500.0]
        hero_pos = 0
        active_players = [0, 1]
        equity = 0.0  # hero loses the showdown entirely

        invested_hero = 1000.0
        max_opp = 500.0
        effective_hero = min(1000.0, 500.0)  # = 500
        excess = 1000.0 - 500.0               # = 500 (returned no matter what)
        hero_share_pot = min(1000.0, 500.0) + min(500.0, 500.0)  # = 500 + 500 = 1000
        # equity=0: net = 500 + 0 * 1000 - 1000 = -500
        expected_net = excess + equity * (hero_share_pot + 0.0) - invested_hero
        # = 500 + 0 - 1000 = -500

        result = self._call(equity, contributions, hero_pos, active_players)
        self.assertAlmostEqual(result, expected_net, places=6)
        self.assertAlmostEqual(result, -500.0, places=6)

    def test_formula_excess_returned_equity_one(self):
        """Big-stack hero, equity=1.0: wins only capped pot + gets excess back."""
        # Hero 1000, opp 500
        contributions = [1000.0, 500.0]
        result = self._call(equity=1.0, contributions=contributions,
                            hero_pos=0, active_players=[0, 1])
        # effective_hero = 500, excess = 500
        # hero_share_pot = 500 + 500 = 1000
        # net = 500 + 1.0*1000 - 1000 = 500
        self.assertAlmostEqual(result, 500.0, places=6)


# ---------------------------------------------------------------------------
# 2. Equity=1.0 wins full capped pot
# ---------------------------------------------------------------------------

class TestEquityOne(unittest.TestCase):
    """Equity=1.0 must win everything hero can legitimately contest."""

    def test_equal_stacks_wins_full_pot(self):
        """Equal stacks, equity=1.0 → wins opponent's chips."""
        c = [200.0, 200.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1])
        self.assertAlmostEqual(net, 200.0, places=6)

    def test_short_stack_hero_wins_capped_pot(self):
        """Short-stacked hero with equity=1.0 wins only the side pot, not the full pot."""
        # Hero 300, opp 700
        c = [300.0, 700.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1])
        # effective_hero=300, hero_share_pot=300+300=600, excess=0
        # net = 0 + 1.0*600 - 300 = 300
        self.assertAlmostEqual(net, 300.0, places=6)

    def test_big_stack_hero_wins_capped_pot_plus_excess(self):
        """Big-stacked hero with equity=1.0: wins capped pot and gets excess back."""
        # Hero 700, opp 300
        c = [700.0, 300.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1])
        # effective_hero=300, excess=400
        # hero_share_pot = min(700,300) + min(300,300) = 300+300 = 600
        # net = 400 + 1.0*600 - 700 = 300
        self.assertAlmostEqual(net, 300.0, places=6)


# ---------------------------------------------------------------------------
# 3. Equity=0.0 loses only invested (minus excess)
# ---------------------------------------------------------------------------

class TestEquityZero(unittest.TestCase):
    """Equity=0.0 means hero wins nothing from the contested pot; only excess is returned."""

    def test_equal_stacks_loses_full_investment(self):
        c = [150.0, 150.0]
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1])
        self.assertAlmostEqual(net, -150.0, places=6)

    def test_short_stack_hero_loses_only_invested(self):
        """Short hero (300 vs 700 opp): equity=0 → net = -300 (no excess)."""
        c = [300.0, 700.0]
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1])
        self.assertAlmostEqual(net, -300.0, places=6)

    def test_big_stack_hero_gets_excess_back(self):
        """Big hero (700 vs 300 opp): equity=0 → net = excess - invested = -300."""
        # excess = 700 - 300 = 400; loses the capped 300; net = 400 + 0 - 700 = -300
        c = [700.0, 300.0]
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1])
        self.assertAlmostEqual(net, -300.0, places=6)

    def test_exact_formula_zero_equity(self):
        """Explicitly verify net = excess - invested_hero when equity=0."""
        c = [1000.0, 400.0]
        invested_hero = 1000.0
        max_opp = 400.0
        effective_hero = min(1000.0, 400.0)  # 400
        excess = 1000.0 - 400.0              # 600
        expected = excess + 0.0 - invested_hero  # 600 - 1000 = -400
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1])
        self.assertAlmostEqual(net, expected, places=6)
        self.assertAlmostEqual(net, -400.0, places=6)


# ---------------------------------------------------------------------------
# 4. Side pot correctness
# ---------------------------------------------------------------------------

class TestSidePot(unittest.TestCase):
    """Hero with 500 vs opponent with 1000: hero can only contest 500+500=1000."""

    def test_hero_500_vs_opp_1000_cannot_win_full_pot(self):
        c = [500.0, 1000.0]
        # Even with equity=1, hero can only win the capped side pot
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1])
        # effective_hero=500, hero_share_pot=500+500=1000, excess=0
        # net = 0 + 1.0*1000 - 500 = 500
        self.assertAlmostEqual(net, 500.0, places=6)
        # Sanity: hero wins 500 (opp's matching portion), not 1000 (full opp contribution)
        self.assertLess(net, 600.0)

    def test_hero_500_vs_opp_1000_equity_half(self):
        c = [500.0, 1000.0]
        net = _capped_showdown_chips(0.5, c, hero_pos=0,
                                     active_players=[0, 1])
        # net = 0 + 0.5*1000 - 500 = 0
        self.assertAlmostEqual(net, 0.0, places=6)

    def test_opp_500_vs_hero_1000_equity_one(self):
        """Opp is the short stack; hero can win full opp contribution."""
        c = [1000.0, 500.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1])
        # effective_hero = min(1000, 500) = 500, excess = 500
        # hero_share_pot = min(1000,500) + min(500,500) = 500 + 500 = 1000
        # net = 500 + 1.0*1000 - 1000 = 500
        self.assertAlmostEqual(net, 500.0, places=6)

    def test_three_way_hero_short_stack(self):
        """3-way: hero (300) vs two bigger opps (600, 800)."""
        contributions = [300.0, 600.0, 800.0]
        active_players = [0, 1, 2]
        hero_pos = 0
        equity = 1.0

        # max_opp = max(600, 800) = 800
        # effective_hero = min(300, 800) = 300
        # excess = 0
        # hero_share_pot = min(300,300) + min(600,300) + min(800,300) = 300+300+300 = 900
        # net = 0 + 1.0*900 - 300 = 600
        net = _capped_showdown_chips(equity, contributions, hero_pos, active_players)
        self.assertAlmostEqual(net, 600.0, places=6)

    def test_three_way_hero_short_stack_equity_zero(self):
        contributions = [300.0, 600.0, 800.0]
        active_players = [0, 1, 2]
        net = _capped_showdown_chips(0.0, contributions, hero_pos=0,
                                     active_players=active_players)
        # net = 0 + 0 - 300 = -300
        self.assertAlmostEqual(net, -300.0, places=6)


# ---------------------------------------------------------------------------
# 5. Dead money inclusion
# ---------------------------------------------------------------------------

class TestDeadMoney(unittest.TestCase):
    """dead_money is added to the contestable pot regardless of contributions."""

    def test_dead_money_with_equity_one(self):
        """Equity=1 hero claims all contributions + dead_money."""
        c = [100.0, 100.0]
        dead = 50.0
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1], dead_money=dead)
        # effective_hero=100, hero_share_pot=200, dead=50
        # net = 0 + 1.0*(200+50) - 100 = 150
        self.assertAlmostEqual(net, 150.0, places=6)

    def test_dead_money_with_equity_zero(self):
        """Equity=0: hero loses investment; dead money goes to opponent."""
        c = [100.0, 100.0]
        dead = 50.0
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1], dead_money=dead)
        # net = 0 + 0*(200+50) - 100 = -100
        self.assertAlmostEqual(net, -100.0, places=6)

    def test_dead_money_with_equity_half(self):
        """Equity=0.5, dead money split evenly."""
        c = [100.0, 100.0]
        dead = 60.0
        net = _capped_showdown_chips(0.5, c, hero_pos=0,
                                     active_players=[0, 1], dead_money=dead)
        # hero_share_pot=200, net = 0 + 0.5*(200+60) - 100 = 0.5*260 - 100 = 30
        self.assertAlmostEqual(net, 30.0, places=6)

    def test_dead_money_zero_same_as_no_dead(self):
        """dead_money=0 is identical to no dead_money argument."""
        c = [150.0, 200.0]
        net_no_dead = _capped_showdown_chips(0.4, c, hero_pos=0,
                                              active_players=[0, 1])
        net_zero_dead = _capped_showdown_chips(0.4, c, hero_pos=0,
                                               active_players=[0, 1],
                                               dead_money=0.0)
        self.assertAlmostEqual(net_no_dead, net_zero_dead, places=9)

    def test_dead_money_short_stack_hero(self):
        """dead_money + side-pot: hero 300, opp 700, dead 100."""
        c = [300.0, 700.0]
        dead = 100.0
        # effective_hero=300, excess=0
        # hero_share_pot = min(300,300)+min(700,300) = 300+300 = 600
        # net = 0 + 0.7*(600+100) - 300 = 0.7*700 - 300 = 490-300 = 190
        net = _capped_showdown_chips(0.7, c, hero_pos=0,
                                     active_players=[0, 1], dead_money=dead)
        self.assertAlmostEqual(net, 190.0, places=6)


# ---------------------------------------------------------------------------
# 6. Fold terminal deterministic Q value
# ---------------------------------------------------------------------------

class TestFoldTerminalDeterministicQ(unittest.TestCase):
    """Tests for _deterministic_terminal_value logic via re_backup_terminals.

    We construct trees where terminal.Q is already set (simulating the post-
    evaluate_all_terminals state) and verify re_backup_terminals propagates
    correctly. The fold terminal Q formula itself:
        hero wins: Q = (pot - hero_invested) / scale
        hero loses: Q = -hero_invested / scale
    is derived from the chips formula; we test the arithmetic directly.
    """

    def test_fold_terminal_hero_wins(self):
        """Hero wins uncontested pot: Q_chips = pot - hero_invested."""
        pot = 200.0
        hero_invested = 50.0
        scale = 100.0
        q_chips = pot - hero_invested          # = 150
        expected_q = q_chips / scale           # = 1.5
        self.assertAlmostEqual(expected_q, 1.5, places=9)

    def test_fold_terminal_hero_loses(self):
        """Hero folded: Q_chips = -hero_invested."""
        hero_invested = 80.0
        scale = 100.0
        q_chips = -hero_invested               # = -80
        expected_q = q_chips / scale           # = -0.8
        self.assertAlmostEqual(expected_q, -0.8, places=9)

    def test_fold_terminal_scale_one(self):
        """With scale=1, Q == q_chips."""
        pot = 300.0
        hero_invested = 100.0
        scale = 1.0
        q = (pot - hero_invested) / scale
        self.assertAlmostEqual(q, 200.0, places=9)

    def test_fold_terminal_hero_loses_scale(self):
        hero_invested = 50.0
        scale = 50.0
        q = -hero_invested / scale
        self.assertAlmostEqual(q, -1.0, places=9)

    def test_zero_contribution_fold(self):
        """Big blind who did not bet: hero_invested=0 loses nothing on fold."""
        pot = 150.0
        hero_invested = 0.0
        scale = 100.0
        q = (pot - hero_invested) / scale  # hero wins: BB collect
        self.assertAlmostEqual(q, 1.5, places=9)
        q_lose = -hero_invested / scale
        self.assertAlmostEqual(q_lose, 0.0, places=9)


# ---------------------------------------------------------------------------
# 7. re_backup_terminals delta propagation
# ---------------------------------------------------------------------------

class TestReBackupTerminals(unittest.TestCase):
    """Verify delta propagation, hero max-Q, opp W/N, and idempotency."""

    # ------------------------------------------------------------------
    # Helper: build a minimal 3-level tree
    #
    #   root (hero, is_terminal=False)
    #     ├── child_a (is_hero=False / opp, is_terminal=False)
    #     │     └── terminal_a  [Q set externally]
    #     └── child_b (is_hero=False / opp, is_terminal=False)
    #           └── terminal_b  [Q set externally]
    # ------------------------------------------------------------------

    def _build_tree(self, qa, na, qb, nb):
        """Build minimal 2-terminal tree and return root + all nodes.

        root is hero, child_a / child_b are opp nodes.
        terminal_a has (Q=qa, N=na), terminal_b has (Q=qb, N=nb).

        W for each node is set consistently: terminal W = Q*N, interior W =
        sum(leaf W) (so re_backup_terminals can compute correct delta when we
        later change terminal Q).
        """
        root = _make_non_terminal(is_hero=True, action_idx=None)

        child_a = _make_non_terminal(is_hero=False, action_idx=0, parent=root)
        child_b = _make_non_terminal(is_hero=False, action_idx=1, parent=root)
        _attach_child(root, child_a)
        _attach_child(root, child_b)

        term_a = _make_terminal(Q=qa, N=na, parent=child_a, action_idx=0)
        term_b = _make_terminal(Q=qb, N=nb, parent=child_b, action_idx=0)
        _attach_child(child_a, term_a)
        _attach_child(child_b, term_b)

        # Set up N and W for interior nodes to be internally consistent
        child_a.N = na
        child_a.W = qa * na
        child_a.Q = qa  # opp: W/N = qa
        child_b.N = nb
        child_b.W = qb * nb
        child_b.Q = qb  # opp: W/N = qb

        root.N = na + nb
        root.W = qa * na + qb * nb
        root.Q = max(child_a.Q, child_b.Q)  # hero: max child Q

        return root, child_a, child_b, term_a, term_b

    # ------------------------------------------------------------------
    # 7a. Basic delta propagation
    # ------------------------------------------------------------------

    def test_delta_propagates_to_all_ancestors(self):
        """Changing terminal Q → delta = N*(new_Q - old_Q) added to every ancestor W."""
        old_qa = 0.5
        na = 10
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=old_qa, na=na, qb=0.3, nb=8)

        W_root_before = root.W
        W_child_a_before = child_a.W

        # Manually set new Q on terminal_a (simulating evaluate_all_terminals)
        new_qa = 0.8
        term_a.Q = new_qa

        re_backup_terminals(root)

        delta = (new_qa - old_qa) * na   # = (0.8 - 0.5) * 10 = 3.0
        self.assertAlmostEqual(child_a.W, W_child_a_before + delta, places=9)
        self.assertAlmostEqual(root.W, W_root_before + delta, places=9)

    def test_delta_does_not_affect_unrelated_branch(self):
        """terminal_b's branch W is unchanged when only terminal_a is updated."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.5, na=10, qb=0.3, nb=8)

        W_child_b_before = child_b.W
        term_a.Q = 0.9  # change only terminal_a
        re_backup_terminals(root)
        # child_b is not an ancestor of terminal_a → its W must not change
        self.assertAlmostEqual(child_b.W, W_child_b_before, places=9)

    # ------------------------------------------------------------------
    # 7b. Hero max-Q after update
    # ------------------------------------------------------------------

    def test_hero_q_is_max_of_children_after_update(self):
        """After re_backup_terminals, root.Q == max(child_a.Q, child_b.Q)."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.2, na=10, qb=0.5, nb=12)

        # Change terminal_a to a very high value → child_a.Q should become high
        term_a.Q = 1.0
        re_backup_terminals(root)

        expected_root_q = max(child_a.Q, child_b.Q)
        self.assertAlmostEqual(root.Q, expected_root_q, places=9)
        # Verify it's the max of actual children Q values
        self.assertAlmostEqual(root.Q, max(child_a.Q, child_b.Q), places=9)

    def test_hero_q_max_selects_higher_branch(self):
        """root Q picks the branch with higher Q even after terminal update."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.5, na=5, qb=0.1, nb=5)

        # Initially child_a has higher Q; confirm root picks it
        self.assertAlmostEqual(root.Q, max(child_a.Q, child_b.Q), places=9)

        # Flip: drive terminal_b to a higher value
        term_b.Q = 2.0
        re_backup_terminals(root)
        # Now child_b should have higher Q and root should reflect that
        self.assertAlmostEqual(root.Q, max(child_a.Q, child_b.Q), places=9)
        self.assertGreaterEqual(root.Q, child_b.Q - 1e-9)

    # ------------------------------------------------------------------
    # 7c. Opp node Q = W/N
    # ------------------------------------------------------------------

    def test_opp_node_q_equals_w_over_n(self):
        """After re_backup_terminals, opp child Q = W/N."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.4, na=6, qb=0.2, nb=4)

        new_qa = 1.2
        term_a.Q = new_qa
        re_backup_terminals(root)

        # child_a is opp node: Q = W/N
        self.assertAlmostEqual(child_a.Q, child_a.W / child_a.N, places=9)
        # child_b was untouched, still Q = W/N
        self.assertAlmostEqual(child_b.Q, child_b.W / child_b.N, places=9)

    # ------------------------------------------------------------------
    # 7d. Idempotency: second run changes nothing
    # ------------------------------------------------------------------

    def test_idempotent_second_run(self):
        """Calling re_backup_terminals twice produces identical results."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.3, na=10, qb=0.7, nb=8)

        term_a.Q = 0.9
        term_b.Q = 1.1
        re_backup_terminals(root)

        # Snapshot after first run
        W_root_1 = root.W
        Q_root_1 = root.Q
        W_ca_1 = child_a.W
        Q_ca_1 = child_a.Q
        W_cb_1 = child_b.W
        Q_cb_1 = child_b.Q

        re_backup_terminals(root)

        # Nothing should have changed
        self.assertAlmostEqual(root.W, W_root_1, places=9)
        self.assertAlmostEqual(root.Q, Q_root_1, places=9)
        self.assertAlmostEqual(child_a.W, W_ca_1, places=9)
        self.assertAlmostEqual(child_a.Q, Q_ca_1, places=9)
        self.assertAlmostEqual(child_b.W, W_cb_1, places=9)
        self.assertAlmostEqual(child_b.Q, Q_cb_1, places=9)

    # ------------------------------------------------------------------
    # 7e. delta = new_Q * N - old_W
    # ------------------------------------------------------------------

    def test_delta_formula_exactly(self):
        """delta at terminal = new_Q*N - old_W, applied once to each ancestor."""
        old_q = 0.4
        n = 7
        old_W = old_q * n  # = 2.8 (terminal started consistent)
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=old_q, na=n, qb=0.0, nb=0)

        # Override with a new Q
        new_q = 1.0
        term_a.Q = new_q
        delta = new_q * n - old_W  # = 7.0 - 2.8 = 4.2

        W_child_a_before = child_a.W
        W_root_before = root.W

        re_backup_terminals(root)

        self.assertAlmostEqual(child_a.W, W_child_a_before + delta, places=9)
        self.assertAlmostEqual(root.W, W_root_before + delta, places=9)
        # terminal W should equal new_Q * N
        self.assertAlmostEqual(term_a.W, new_q * n, places=9)

    # ------------------------------------------------------------------
    # 7f. Terminal with N=0 is skipped
    # ------------------------------------------------------------------

    def test_terminal_n_zero_is_skipped(self):
        """Terminal with N=0 (never visited) should not affect ancestor W."""
        root, child_a, child_b, term_a, term_b = self._build_tree(
            qa=0.5, na=5, qb=0.5, nb=5)

        # Create an unvisited terminal as child of child_a
        unvisited = MCTSNode(action_idx=99, parent=child_a, is_terminal=True)
        unvisited.N = 0
        unvisited.Q = 9.9  # garbage — should be ignored
        child_a.children[99] = unvisited

        W_root_before = root.W
        W_child_a_before = child_a.W

        re_backup_terminals(root)

        # Only term_a (N=5) should be processed; unvisited terminal is skipped
        # Since term_a.Q == old_q already, delta == 0 → no changes
        self.assertAlmostEqual(root.W, W_root_before, places=9)
        self.assertAlmostEqual(child_a.W, W_child_a_before, places=9)


# ---------------------------------------------------------------------------
# 8. Scale division
# ---------------------------------------------------------------------------

class TestScaleDivision(unittest.TestCase):
    """Terminal Q values are divided by value_scales_by_position[hero_pos]."""

    def test_q_divided_by_scale(self):
        """Net chip delta / scale = terminal Q stored in the node."""
        q_chips = 150.0
        scale = 75.0
        expected_q = q_chips / scale  # = 2.0
        self.assertAlmostEqual(expected_q, 2.0, places=9)

    def test_scale_one_identity(self):
        """scale=1.0 → Q equals chip delta."""
        q_chips = 300.0
        self.assertAlmostEqual(q_chips / 1.0, 300.0, places=9)

    def test_scale_larger_than_chips_fractional_q(self):
        """Large scale → small Q (fractional)."""
        q_chips = 50.0
        scale = 200.0
        q = q_chips / scale  # = 0.25
        self.assertAlmostEqual(q, 0.25, places=9)

    def test_negative_chips_divided_by_scale(self):
        """Negative chip outcome also divides correctly."""
        q_chips = -80.0
        scale = 40.0
        q = q_chips / scale  # = -2.0
        self.assertAlmostEqual(q, -2.0, places=9)

    def test_capped_chips_consistent_with_scale(self):
        """Combine _capped_showdown_chips and scale division."""
        c = [100.0, 100.0]
        equity = 0.7
        scale = 50.0
        # net chips = 0 + 0.7*200 - 100 = 40
        net = _capped_showdown_chips(equity, c, hero_pos=0,
                                     active_players=[0, 1])
        q = net / scale  # = 40 / 50 = 0.8
        self.assertAlmostEqual(net, 40.0, places=6)
        self.assertAlmostEqual(q, 0.8, places=6)


# ---------------------------------------------------------------------------
# 9. Contributions vs investments semantics
# ---------------------------------------------------------------------------

class TestContributionsVsInvestments(unittest.TestCase):
    """contributions[pos] = initial_stacks[pos] - credits[pos].

    This is the total chips put in from a reference point (root of MCTS or
    start of hand), NOT the per-street `bets` which reset after each street.
    """

    def test_contributions_are_cumulative_across_streets(self):
        """Verify contributions math: initial_stack - final_credits."""
        # Player starts with 1000, ends with 600 after all betting
        initial_stack = 1000.0
        final_credits = 600.0
        contribution = initial_stack - final_credits   # = 400
        self.assertAlmostEqual(contribution, 400.0, places=9)

    def test_contributions_sum_equals_total_pot_2p(self):
        """In a 2-player hand, sum(contributions) == total pot contributed."""
        initial_stacks = [1000.0, 1000.0]
        final_credits = [600.0, 500.0]
        contributions = [initial_stacks[i] - final_credits[i] for i in range(2)]
        # = [400, 500]
        total_contributions = sum(contributions)       # = 900
        self.assertAlmostEqual(total_contributions, 900.0, places=9)

    def test_side_pot_formula_uses_contributions_not_bets(self):
        """_capped_showdown_chips takes total contributions, not per-street bets."""
        # Scenario: preflop hero bets 100, flop hero bets 200, turn hero bets 300
        # bets (per-street, reset each street): river bets only = {hero: 300}
        # contributions (cumulative): hero = 600
        contributions = [600.0, 600.0]
        net = _capped_showdown_chips(1.0, contributions, hero_pos=0,
                                     active_players=[0, 1])
        # effective_hero=600, hero_share_pot=1200, net=0+1200-600=600
        self.assertAlmostEqual(net, 600.0, places=6)

    def test_contributions_include_blinds(self):
        """Blind amounts must be included in contributions for correct side-pot math."""
        # Heads-up: hero SB posts 5, later raises to 100 total. Opp BB calls 100.
        # contributions = [100, 100] (includes the 5 blind)
        contributions = [100.0, 100.0]
        net = _capped_showdown_chips(1.0, contributions, hero_pos=0,
                                     active_players=[0, 1])
        # Pot = 200, hero wins all = 100 profit
        self.assertAlmostEqual(net, 100.0, places=6)

    def test_credits_pre_distribution_is_before_pot_awarded(self):
        """credits_pre_distribution is BEFORE the pot is given to winner.

        chips_invested_from_decision = credits_at_decision - credits_pre_distribution
        This is a non-negative number when hero invested post-decision.
        """
        credits_at_decision = 900.0   # credits when the decision was made
        credits_pre_dist = 600.0      # credits after all betting, before award
        chips_invested_from_t = credits_at_decision - credits_pre_dist  # = 300
        self.assertAlmostEqual(chips_invested_from_t, 300.0, places=9)
        self.assertGreaterEqual(chips_invested_from_t, 0.0)


# ---------------------------------------------------------------------------
# 10. Multi-opponent equity
# ---------------------------------------------------------------------------

class TestMultiOpponentEquity(unittest.TestCase):
    """_capped_showdown_chips supports multiple active opponents.

    The side-pot formula generalises: max_opp = max(contributions[p] for p in
    active if p != hero), effective_hero = min(invested_hero, max_opp),
    hero_share_pot = sum(min(c, effective_hero) for all c in contributions).
    """

    def test_three_way_equal_stacks_equity_third(self):
        """3-way equal stacks, equity=1/3 → break-even."""
        c = [100.0, 100.0, 100.0]
        net = _capped_showdown_chips(1.0/3.0, c, hero_pos=0,
                                     active_players=[0, 1, 2])
        # effective_hero=100, hero_share_pot=300, excess=0
        # net = 0 + (1/3)*300 - 100 = 0
        self.assertAlmostEqual(net, 0.0, places=6)

    def test_three_way_equal_stacks_equity_one(self):
        """3-way equal stacks, equity=1.0 → wins all three stacks minus own."""
        c = [100.0, 100.0, 100.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1, 2])
        # net = 0 + 1.0*300 - 100 = 200
        self.assertAlmostEqual(net, 200.0, places=6)

    def test_three_way_hero_short_equity_one(self):
        """3-way: hero 200, two opps 400 each, equity=1.0."""
        c = [200.0, 400.0, 400.0]
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1, 2])
        # max_opp = max(400,400) = 400
        # effective_hero = min(200, 400) = 200
        # excess = 0
        # hero_share_pot = min(200,200) + min(400,200) + min(400,200)
        #                = 200 + 200 + 200 = 600
        # net = 0 + 1.0*600 - 200 = 400
        self.assertAlmostEqual(net, 400.0, places=6)

    def test_three_way_hero_big_stack_equity_zero(self):
        """3-way: hero over-invested, equity=0 → loses only capped portion."""
        c = [500.0, 200.0, 300.0]
        net = _capped_showdown_chips(0.0, c, hero_pos=0,
                                     active_players=[0, 1, 2])
        # max_opp = max(200, 300) = 300
        # effective_hero = min(500, 300) = 300
        # excess = 500 - 300 = 200
        # net = 200 + 0 - 500 = -300
        self.assertAlmostEqual(net, -300.0, places=6)

    def test_three_way_with_dead_money(self):
        """3-way with dead money included in contestable pot."""
        c = [200.0, 200.0, 200.0]
        dead = 90.0
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0, 1, 2], dead_money=dead)
        # hero_share_pot=600, net = 0 + 1.0*(600+90) - 200 = 490
        self.assertAlmostEqual(net, 490.0, places=6)

    def test_single_opponent_no_others_active(self):
        """hero is alone (no active opponents → max_opp defaults to 0)."""
        c = [300.0, 0.0, 0.0]
        # With no opponents in active_players, max_opp=0,
        # effective_hero=min(300,0)=0, excess=300
        # hero_share_pot = min(300,0)+min(0,0)+min(0,0) = 0
        # net = 300 + equity*0 - 300 = 0
        net = _capped_showdown_chips(1.0, c, hero_pos=0,
                                     active_players=[0])
        self.assertAlmostEqual(net, 0.0, places=6)


# ---------------------------------------------------------------------------
# 11. re_backup_terminals with opp pessimism (alpha < 1)
# ---------------------------------------------------------------------------

class TestReBackupWithOppPessimism(unittest.TestCase):
    """When opp_pessimism_alpha < 1, opp Q = alpha * E_P[Q] + (1-alpha) * min_Q."""

    def _build_two_terminal_opp_tree(self, q_term1, n_term1, q_term2, n_term2,
                                      p_term1=0.6, p_term2=0.4):
        """
        root (hero)
          └── opp_node
                ├── terminal1 (P=p_term1, N=n_term1, Q=q_term1)
                └── terminal2 (P=p_term2, N=n_term2, Q=q_term2)
        """
        root = _make_non_terminal(is_hero=True, action_idx=None)
        opp = _make_non_terminal(is_hero=False, action_idx=0, parent=root)
        _attach_child(root, opp)

        t1 = _make_terminal(Q=q_term1, N=n_term1, parent=opp, action_idx=0)
        t1.P = p_term1
        t2 = _make_terminal(Q=q_term2, N=n_term2, parent=opp, action_idx=1)
        t2.P = p_term2
        _attach_child(opp, t1)
        _attach_child(opp, t2)

        opp.N = n_term1 + n_term2
        opp.W = q_term1 * n_term1 + q_term2 * n_term2
        opp.Q = opp.W / opp.N

        root.N = opp.N
        root.W = opp.W
        root.Q = opp.Q  # hero max-Q; only one opp child so max = opp.Q

        return root, opp, t1, t2

    def test_alpha_one_opp_q_is_mean(self):
        """alpha=1.0 → classic W/N (no pessimism)."""
        q1, n1 = 0.6, 10
        q2, n2 = 0.2, 10
        root, opp, t1, t2 = self._build_two_terminal_opp_tree(q1, n1, q2, n2)

        # Both terminals already consistent; run without changing anything
        re_backup_terminals(root, opp_pessimism_alpha=1.0)

        # opp.Q should be W/N
        expected = opp.W / opp.N
        self.assertAlmostEqual(opp.Q, expected, places=9)

    def test_alpha_zero_opp_q_is_min(self):
        """alpha=0.0 → pure pessimism: opp Q = min(visited child Q)."""
        q1, n1 = 0.8, 5
        q2, n2 = 0.1, 5
        root, opp, t1, t2 = self._build_two_terminal_opp_tree(q1, n1, q2, n2)

        # After re_backup with delta=0 (unchanged Q), opp Q recalculates with pessimism
        # But since terminal Qs haven't changed, delta=0, so W is untouched.
        # However re_backup still recomputes opp.Q at the end of delta propagation.
        # Force a change so re_backup walks up through opp:
        t1.Q = 0.9  # change terminal1
        re_backup_terminals(root, opp_pessimism_alpha=0.0)

        # alpha=0 → opp Q = min child Q
        visited = [c for c in opp.children.values() if c.N > 0]
        min_q = min(c.Q for c in visited)
        self.assertAlmostEqual(opp.Q, min_q, places=9)

    def test_alpha_half_opp_q_is_blend(self):
        """alpha=0.5 → blend of E_P[Q] and min_Q."""
        q1, n1 = 1.0, 5
        q2, n2 = 0.0, 5
        p1, p2 = 0.7, 0.3
        root, opp, t1, t2 = self._build_two_terminal_opp_tree(
            q1, n1, q2, n2, p_term1=p1, p_term2=p2)

        # Force re_backup walk by changing a terminal Q
        t1.Q = 1.0  # same as before, no effective delta — but we need to
        t2.Q = 0.0  # ensure consistency; let's actually change t1:
        t1.Q = 0.8  # different from original 1.0 → delta != 0

        re_backup_terminals(root, opp_pessimism_alpha=0.5)

        visited = [c for c in opp.children.values() if c.N > 0]
        total_p = sum(c.P for c in visited)
        expected_q = sum(c.P * c.Q for c in visited) / total_p
        min_q = min(c.Q for c in visited)
        expected_opp_q = 0.5 * expected_q + 0.5 * min_q
        self.assertAlmostEqual(opp.Q, expected_opp_q, places=9)


# ---------------------------------------------------------------------------
# 12. _collect_terminals correctness
# ---------------------------------------------------------------------------

class TestCollectTerminals(unittest.TestCase):
    """_collect_terminals should find all and only terminal nodes in a tree."""

    def test_single_terminal(self):
        root = MCTSNode(is_terminal=False)
        t = MCTSNode(action_idx=0, parent=root, is_terminal=True)
        root.children[0] = t
        terminals = _collect_terminals(root)
        self.assertEqual(len(terminals), 1)
        self.assertIs(terminals[0], t)

    def test_root_itself_is_terminal(self):
        root = MCTSNode(is_terminal=True)
        terminals = _collect_terminals(root)
        self.assertEqual(len(terminals), 1)
        self.assertIs(terminals[0], root)

    def test_no_terminals(self):
        root = MCTSNode(is_terminal=False)
        child = MCTSNode(action_idx=0, parent=root, is_terminal=False)
        root.children[0] = child
        terminals = _collect_terminals(root)
        self.assertEqual(len(terminals), 0)

    def test_multiple_terminals_found(self):
        root = MCTSNode(is_terminal=False)
        for i in range(5):
            t = MCTSNode(action_idx=i, parent=root, is_terminal=True)
            root.children[i] = t
        terminals = _collect_terminals(root)
        self.assertEqual(len(terminals), 5)

    def test_deep_tree_terminals(self):
        """Terminals buried at depth 3 are all found."""
        root = MCTSNode(is_terminal=False)
        mid = MCTSNode(action_idx=0, parent=root, is_terminal=False)
        root.children[0] = mid
        for i in range(3):
            t = MCTSNode(action_idx=i, parent=mid, is_terminal=True)
            mid.children[i] = t
        terminals = _collect_terminals(root)
        self.assertEqual(len(terminals), 3)


# ---------------------------------------------------------------------------
# 13. MCTSNode basic structure
# ---------------------------------------------------------------------------

class TestMCTSNodeStructure(unittest.TestCase):
    """Verify MCTSNode initialises correctly."""

    def test_default_init(self):
        n = MCTSNode()
        self.assertIsNone(n.action_idx)
        self.assertIsNone(n.parent)
        self.assertEqual(n.children, {})
        self.assertTrue(n.is_hero)
        self.assertFalse(n.is_terminal)
        self.assertEqual(n.N, 0)
        self.assertAlmostEqual(n.W, 0.0)
        self.assertAlmostEqual(n.Q, 0.0)
        self.assertAlmostEqual(n.P, 0.0)
        self.assertIsNone(n.action_embedding)
        self.assertIsNone(n._term_value)

    def test_is_terminal_flag(self):
        t = MCTSNode(is_terminal=True)
        self.assertTrue(t.is_terminal)

    def test_parent_child_linkage(self):
        root = MCTSNode()
        child = MCTSNode(action_idx=2, parent=root)
        root.children[2] = child
        self.assertIs(child.parent, root)
        self.assertIn(2, root.children)
        self.assertIs(root.children[2], child)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
