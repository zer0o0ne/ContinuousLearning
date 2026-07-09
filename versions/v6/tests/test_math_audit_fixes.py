"""
E2E scenario tests for the 2026-07 math-audit fixes (phases 4/5/6).

Covers, through real code paths with fixed seeds and exact/tight assertions:

1. Side-pot pro-call bias (terminal_eval): a facing-bet MCTS root whose call
   line reaches showdown gets Q = eq*full_pot - call, NOT eq*root_pot (the old
   code refunded the call as "uncalled excess").
2. Folded-player layers: chips a folded player contributed into hero's
   uncontested layers are won outright, not dropped.
3. equity_cache keying: identical narrowed ranges on different streets must
   not collide (gpu_equity_v2 called once per street).
4. Terminal-rollout padding (phase 6 train): terminal value predictions for a
   short example inside a padded batch match the same example alone (B=1).
5. Root LM pair dedup (phase 6 train): same-hand examples sharing event
   objects contribute each transition once per batch.
6. Sharded opponent stack norm-stats: per-value denominator (mean/std match
   numpy over all per-position values).
7. GRU observer-copy rewind (A.4.5): with gru_sample_groups, N observer
   copies of one scenario advance the shared table exactly once.
8. opp_pessimism_alpha defaults agree across mcts.py / terminal_eval.py.

All tests are deterministic: fixed seeds (tests/conftest.py), exact or
tight-tolerance assertions, CPU only.

Run from versions/v6/:  python -m pytest tests/test_math_audit_fixes.py -v
"""

import inspect
import os
import sys

import numpy as np
import torch
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
if _V6_ROOT not in sys.path:
    sys.path.insert(0, _V6_ROOT)
_GTO = os.path.join(_V6_ROOT, "agent", "gto_utils")
if _GTO not in sys.path:
    sys.path.insert(0, _GTO)

from agent.mcts import terminal_eval as te
from agent.mcts.terminal_eval import (
    _capped_showdown_chips, evaluate_all_terminals,
)
from agent.mcts.mcts import MCTSNode, re_backup_terminals
from agent.mcts.game_state import GameState
from agent.agent import ASI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAISE_SIZES = [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0], [0.5, 1.0]]
N_RAISE_BINS = 2
N_ACTIONS = N_RAISE_BINS + 3  # fold, call, 2 raises, all-in
MAX_PLAYERS = 2


def _log(msg):
    pass


_TINY_CONFIG = {
    "architecture": {
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "d_ff": 64,
        "max_seq_len": 84,  # 84 // 7 = 12 max events
        "max_players": MAX_PLAYERS,
        "modelling_dropout": 0.0,
        "memory": {
            "n_levels": 1,
            "max_cluster_size": 4,
            "max_cluster_size_after": 4,
            "beam_width": 2,
        },
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": [0.5, 1.0],
            "flop": [0.5, 1.0],
            "turn": [0.5, 1.0],
            "river": [0.5, 1.0],
        },
        "max_players": MAX_PLAYERS,
        "big_blind": 10,
        "max_stack": 100,
    },
}


def _tiny_agent(opponent_embedding=False):
    import copy
    cfg = copy.deepcopy(_TINY_CONFIG)
    cfg["architecture"]["opponent_embedding"]["enabled"] = opponent_embedding
    agent = ASI(_log, config=cfg)
    agent.eval()
    return agent


def _hu_facing_bet_root(opp_bet=150.0, hero_in=50.0, stack=1000.0, turn=1):
    """Heads-up GameState at hero's decision facing a bet.

    Whole-hand history: hero (pos 0) put in `hero_in`, opp (pos 1) put in
    `opp_bet` — all already inside `pot` (Table semantics: pot += bet at bet
    time). Hero to act.
    """
    return GameState(
        num_players=2,
        hero_pos=0,
        active_player=0,
        players_state=[0, 0],
        credits=[stack - hero_in, stack - opp_bet],
        bets=[hero_in, opp_bet],
        pot=hero_in + opp_bet,
        high_bet=opp_bet,
        turn=turn,
        raise_sizes=RAISE_SIZES,
        n_raise_bins=N_RAISE_BINS,
        big_blind=10.0,
    )


def _terminal_after_call(root_gs):
    """root -> hero calls (action 1) -> both check down -> terminal.

    Builds the minimal MCTS node chain whose replay in
    evaluate_all_terminals reaches a showdown terminal.
    """
    gs = root_gs.clone()
    actions = [1]
    gs.step(1)
    guard = 0
    while not gs.is_terminal and guard < 20:
        gs.step(1)
        actions.append(1)
        guard += 1
    assert gs.is_terminal, "hand did not terminate by checking down"
    assert sum(1 for s in gs.players_state if s >= 0) == 2, \
        "expected showdown (both active)"

    root = MCTSNode(action_idx=None, is_hero=True)
    root.N = 10
    node = root
    for a in actions:
        child = MCTSNode(action_idx=a, parent=node, is_hero=True)
        child.N = 5
        node.children[a] = child
        node = child
    node.is_terminal = True
    return root, node


class _EqCallCounter:
    """Stub for gpu_equity_v2 returning a fixed equity, counting calls."""

    def __init__(self, equity):
        self.equity = equity
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.equity


@pytest.fixture()
def _stub_equity(monkeypatch):
    """Stub the heavy equity/range machinery inside terminal_eval.

    Range narrowing becomes identity, equity is a fixed 0.3. Real
    get_position_range/expand_range still run (cheap, CPU).
    """
    counter = _EqCallCounter(0.3)
    monkeypatch.setattr(te, "gpu_equity_v2", counter)
    monkeypatch.setattr(
        te, "_narrow_by_real_actions",
        lambda range_types, *a, **k: range_types)
    monkeypatch.setattr(
        te, "_narrow_by_simulated_actions",
        lambda range_types, *a, **k: range_types)
    return counter


def _hand_record_for(root_gs, decisions, start_stacks, num_players=2):
    deck = np.arange(52, dtype=np.int64)  # deterministic board 0..4
    return {
        "decisions": decisions,
        "deck": deck,
        "hero_hands": {0: [50, 51], 1: [48, 49]},
        "num_players": num_players,
        "big_blind": 10.0,
        "final_pot": float(root_gs.pot),
        "final_active_positions": [0, 1],
        "credits_pre_distribution": list(root_gs.credits),
        "initial_credits": list(root_gs.credits),
        "start_stacks": list(start_stacks),
    }


# ---------------------------------------------------------------------------
# 1+2. Side-pot math through evaluate_all_terminals
# ---------------------------------------------------------------------------

class TestSidePotFacingBet:
    def test_facing_bet_call_showdown_q(self, _stub_equity):
        """Audit trace 1: root pot 200 (opp 150 / hero 50), hero calls 100.

        Correct Q = eq*300 - 100 = -10 at eq=0.3. The pre-fix code returned
        eq*200 = +60 (the call refunded as uncalled excess) — the sign of
        the call/fold decision flipped.
        """
        stack = 1000.0
        root_gs = _hu_facing_bet_root(opp_bet=150.0, hero_in=50.0, stack=stack)
        root, term = _terminal_after_call(root_gs)
        decisions = [{
            "player_pos": 0,
            "mcts_root": root,
            "game_state_at_root": root_gs,
        }]
        hand_record = _hand_record_for(root_gs, decisions,
                                       start_stacks=[stack, stack])
        evaluate_all_terminals(hand_record, {0: None, 1: None},
                               device="cpu", config={},
                               combo_probs_cache={})
        assert term.Q == pytest.approx(0.3 * 300.0 - 100.0, abs=1e-6)
        assert term.Q < 0.0  # calling at eq=0.3 must read as losing

    def test_fold_terminal_unchanged(self, _stub_equity):
        """Fold-family terminals keep the from-root convention (Q=0 for an
        immediate root fold)."""
        stack = 1000.0
        root_gs = _hu_facing_bet_root(opp_bet=150.0, hero_in=50.0, stack=stack)
        root = MCTSNode(action_idx=None, is_hero=True)
        root.N = 10
        fold_child = MCTSNode(action_idx=0, parent=root, is_hero=True)
        fold_child.N = 5
        fold_child.is_terminal = True
        root.children[0] = fold_child
        decisions = [{
            "player_pos": 0,
            "mcts_root": root,
            "game_state_at_root": root_gs,
        }]
        hand_record = _hand_record_for(root_gs, decisions,
                                       start_stacks=[stack, stack])
        evaluate_all_terminals(hand_record, {0: None, 1: None},
                               device="cpu", config={},
                               combo_probs_cache={})
        assert fold_child.Q == pytest.approx(0.0, abs=1e-9)


class TestCappedShowdownChips:
    """Direct traces of the corrected gross-return helper."""

    def test_audit_trace_facing_bet(self):
        gross = _capped_showdown_chips(0.3, [150.0, 150.0], 0, [0, 1])
        assert gross - 100.0 == pytest.approx(-10.0)

    def test_folded_layer_won_outright(self):
        """hero 500, folded 300, active opp 200: the folded player's 100 in
        layers 200-300 is won outright, not dropped."""
        gross = _capped_showdown_chips(0.4, [500.0, 300.0, 200.0], 0, [0, 2])
        # det = (500-200) own + (300-200) folded = 400; contested = 600
        assert gross == pytest.approx(400.0 + 0.4 * 600.0)

    def test_hero_covered(self):
        """hero all-in 200 vs opp 500: hero contests only 400 total."""
        gross = _capped_showdown_chips(1.0, [200.0, 500.0], 0, [0, 1])
        assert gross == pytest.approx(400.0)

    def test_overbet_uncalled_layer_returned(self):
        """hero 500 vs active opp 200, no folded money: 300 returns
        deterministically even at eq=0."""
        gross = _capped_showdown_chips(0.0, [500.0, 200.0], 0, [0, 1])
        assert gross == pytest.approx(300.0)

    def test_fully_matched_reduces_to_eq_times_pot(self):
        gross = _capped_showdown_chips(0.7, [250.0, 250.0], 0, [0, 1])
        assert gross == pytest.approx(0.7 * 500.0)


# ---------------------------------------------------------------------------
# 3. equity_cache keyed by street
# ---------------------------------------------------------------------------

class TestEquityCacheStreetKey:
    def _decision_at_turn(self, turn, stack=1000.0):
        gs = _hu_facing_bet_root(opp_bet=150.0, hero_in=50.0, stack=stack,
                                 turn=turn)
        root, _ = _terminal_after_call(gs)
        return {
            "player_pos": 0,
            "mcts_root": root,
            "game_state_at_root": gs,
        }

    def test_different_streets_no_collision(self, _stub_equity):
        """Identical narrowed ranges on different boards must trigger two
        equity computations (pre-fix: the second street reused the first
        street's cached equity)."""
        stack = 1000.0
        d_flop = self._decision_at_turn(1, stack)
        d_turn = self._decision_at_turn(2, stack)
        gs0 = d_flop["game_state_at_root"]
        hand_record = _hand_record_for(gs0, [d_flop, d_turn],
                                       start_stacks=[stack, stack])
        evaluate_all_terminals(hand_record, {0: None, 1: None},
                               device="cpu", config={},
                               combo_probs_cache={})
        assert _stub_equity.calls == 2

    def test_same_street_still_cached(self, _stub_equity):
        """Two identical-street decisions with identical ranges reuse the
        cache (one equity call)."""
        stack = 1000.0
        d1 = self._decision_at_turn(1, stack)
        d2 = self._decision_at_turn(1, stack)
        gs0 = d1["game_state_at_root"]
        hand_record = _hand_record_for(gs0, [d1, d2],
                                       start_stacks=[stack, stack])
        evaluate_all_terminals(hand_record, {0: None, 1: None},
                               device="cpu", config={},
                               combo_probs_cache={})
        assert _stub_equity.calls == 1


# ---------------------------------------------------------------------------
# 4+5. Phase-6 forward: terminal-rollout padding + LM pair dedup
# ---------------------------------------------------------------------------

def _mk_event(action=None, acting_pos=1, hero_pos=0):
    act = [0.0] * N_ACTIONS
    if action is not None:
        act[action] = 1.0
    return {
        "hand": [50, 51],
        "num_players": 2,
        "hero_pos": hero_pos,
        "acting_pos": acting_pos,
        "big_blind": 0.1,
        "small_blind": 0.05,
        "stack": 1.0,
        "stacks": [1.0, 1.0],
        "table": [0, 1, 2, -1, -1],
        "pot": 0.2,
        "bets": [0.1, 0.1],
        "action": act,
    }


def _hand_event_stream(n_decisions):
    """[init, (pre, post)*k, final pre] shared-object stream for one hand."""
    events = [_mk_event()]
    for k in range(n_decisions):
        events.append(_mk_event())                     # pre-decision snapshot
        events.append(_mk_event(action=1 + (k % 2)))   # post-action
    events.append(_mk_event())                         # final pre-decision
    return events


class TestTerminalRolloutPadding:
    def test_padded_batch_matches_unpadded(self):
        """Terminal rollout preds for the SHORT example of a mixed-length
        batch must equal the same example alone (pre-fix: from the second
        rollout step on, the modelling state was gathered from a pad slot
        and RoPE positions were shifted by the pad width)."""
        from agent.train_scenarios.mcts_predict.train import _mcts_forward
        agent = _tiny_agent()
        short_events = _hand_event_stream(1)   # 4 events
        long_events = _hand_event_stream(4)    # 10 events
        terminals_short = [([1, 2], -0.5), ([0], 0.25)]

        with torch.no_grad():
            out_batched = _mcts_forward(
                agent, [short_events, long_events], [[], []], "cpu",
                p_tf=0.0,
                examples_per_batch_terminals=[terminals_short, []])
            out_solo = _mcts_forward(
                agent, [short_events], [[]], "cpu",
                p_tf=0.0,
                examples_per_batch_terminals=[terminals_short])

        preds_batched = torch.stack(out_batched["terminal_value_preds"][0])
        preds_solo = torch.stack(out_solo["terminal_value_preds"][0])
        assert torch.allclose(preds_batched, preds_solo, atol=1e-5), (
            f"padded {preds_batched.tolist()} != solo {preds_solo.tolist()}")


class TestRootLmPairDedup:
    def test_same_hand_examples_dedup(self):
        """Two examples of one hand share event OBJECTS; their common
        transitions must enter the pooled LM batch once."""
        from agent.train_scenarios.mcts_predict.train import _mcts_forward
        from agent.modelling.modelling import build_lm_pairs
        agent = _tiny_agent()
        full = _hand_event_stream(3)   # 8 events
        ex_a = full[:6]   # shares prefix objects with ex_b
        ex_b = full[:8]

        n_a = build_lm_pairs([ex_a])[0].numel()
        n_b = build_lm_pairs([ex_b])[0].numel()
        assert n_a >= 1 and n_b > n_a  # sanity: real duplication exists

        with torch.no_grad():
            out = _mcts_forward(agent, [ex_a, ex_b], [[], []], "cpu",
                                p_tf=0.0)
        # ex_b contains every transition of ex_a → unique count = n_b.
        assert out["lm_pred"].shape[0] == n_b

    def test_different_hands_not_deduped(self):
        """Distinct hands (distinct event objects) keep all their pairs."""
        from agent.train_scenarios.mcts_predict.train import _mcts_forward
        from agent.modelling.modelling import build_lm_pairs
        agent = _tiny_agent()
        h1 = _hand_event_stream(2)
        h2 = _hand_event_stream(2)
        n1 = build_lm_pairs([h1])[0].numel()
        n2 = build_lm_pairs([h2])[0].numel()
        with torch.no_grad():
            out = _mcts_forward(agent, [h1, h2], [[], []], "cpu", p_tf=0.0)
        assert out["lm_pred"].shape[0] == n1 + n2


# ---------------------------------------------------------------------------
# 6. Sharded opponent stack norm-stats
# ---------------------------------------------------------------------------

class TestShardedStackStats:
    def test_per_value_denominator(self, tmp_path):
        from agent.train_scenarios.sharded import (
            compute_opponent_norm_stats_from_shards,
        )
        rng = np.random.RandomState(42)
        n_players = 6
        events = []
        for _ in range(200):
            events.append({
                "pot": float(rng.uniform(10, 500)),
                "big_blind": 10.0,
                "bets": [float(x) for x in rng.uniform(0, 50, n_players)],
                "stacks": [float(x) for x in rng.uniform(100, 3000, n_players)],
            })
        scenarios = [{"events": events}]
        shard_path = tmp_path / "shard_000.pt"
        torch.save(scenarios, shard_path)

        class _FakeShards:
            shard_info = [(str(shard_path), 1)]

        stats = compute_opponent_norm_stats_from_shards(_FakeShards())
        all_stacks = np.array([s for e in events for s in e["stacks"]])
        assert stats["stack_mean"] == pytest.approx(all_stacks.mean(),
                                                    rel=1e-6)
        assert stats["stack_std"] == pytest.approx(all_stacks.std(),
                                                   rel=1e-5)
        # Regression guard: the old bug inflated the mean ~n_players x and
        # collapsed std to the 1.0 fallback.
        assert stats["stack_std"] > 100.0


# ---------------------------------------------------------------------------
# 7. GRU observer-copy rewind
# ---------------------------------------------------------------------------

class TestGruSampleGroups:
    def _observer_copy(self, hero_pos):
        ev = []
        for k in range(3):
            e = _mk_event(action=1 if k else None, acting_pos=1,
                          hero_pos=hero_pos)
            e["opponent_id"] = "villain"
            ev.append(e)
        return ev

    def test_copies_advance_table_once(self):
        """Three observer copies of one scenario sharing a group id must
        leave the table exactly one scenario-advance from its start state —
        identical to processing only the last copy."""
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        agent = _tiny_agent(opponent_embedding=True)
        copies = [self._observer_copy(hero_pos=p) for p in (0, 1, 0)]

        with torch.no_grad():
            table_grouped = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                copies, device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=table_grouped,
                gru_sample_groups=[7, 7, 7])

            table_single = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                [copies[-1]], device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=table_single)

        h_grouped = table_grouped.embeddings["villain"]
        h_single = table_single.embeddings["villain"]
        assert torch.allclose(h_grouped, h_single, atol=1e-6), (
            "grouped copies must persist exactly one scenario-advance "
            "(the last copy's), matching a single-copy pass")

    def test_no_groups_preserves_legacy_accumulation(self):
        """Without groups, copies accumulate (legacy behavior) — the end
        state differs from a single-copy pass."""
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        agent = _tiny_agent(opponent_embedding=True)
        copies = [self._observer_copy(hero_pos=p) for p in (0, 1, 0)]

        with torch.no_grad():
            table_legacy = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                copies, device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=table_legacy)

            table_single = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                [copies[-1]], device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=table_single)

        h_legacy = table_legacy.embeddings["villain"]
        h_single = table_single.embeddings["villain"]
        assert not torch.allclose(h_legacy, h_single, atol=1e-6)

    def test_distinct_groups_match_legacy(self):
        """Different group ids behave exactly like the no-groups path."""
        from agent.perception.opponent_embeddings import OpponentEmbeddingTable
        agent = _tiny_agent(opponent_embedding=True)
        copies = [self._observer_copy(hero_pos=p) for p in (0, 1, 0)]

        with torch.no_grad():
            t_groups = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                copies, device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=t_groups,
                gru_sample_groups=[1, 2, 3])

            t_legacy = OpponentEmbeddingTable(agent.perception.d_model)
            agent.perception.forward_batch(
                copies, device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=t_legacy)

        assert torch.allclose(t_groups.embeddings["villain"],
                              t_legacy.embeddings["villain"], atol=1e-7)


# ---------------------------------------------------------------------------
# 8. opp_pessimism_alpha defaults agree
# ---------------------------------------------------------------------------

class TestPessimismDefaults:
    def test_re_backup_default_matches_mcts(self):
        sig = inspect.signature(re_backup_terminals)
        assert sig.parameters["opp_pessimism_alpha"].default == 0.5

    def test_terminal_eval_source_default(self):
        src = inspect.getsource(te.evaluate_all_terminals)
        assert 'cfg.get("opp_pessimism_alpha", 0.5)' in src
