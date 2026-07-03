"""Opponent-embedding tables persist across MCTS training cycles.

Previously `run_mcts_collection` and `train_mcts` each created fresh
OpponentEmbeddingTables per call (= per cycle), so accumulated context about
frequently-seen players was dropped every cycle. The pipeline's cyclic loop
now owns persistent tables and passes them in:
  - `run_mcts_collection(..., opp_tables=...)` — shared dict keyed by agent
    name, reused across cycles (sequential path).
  - `train_mcts(..., opponent_emb_table=...)` — per-agent table stored in
    `agent_info["opp_emb_table"]`.
Defaults (None) keep the old per-call behavior for legacy call sites.

All tests are fully deterministic: fixed seeds, no probabilistic
assertions, no order dependence.

Run from versions/v6/:
    python3 -m pytest tests/test_opp_table_persistence.py -v
"""

import copy
import inspect
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
for _p in (_PKG_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.agent import ASI
from agent.perception.opponent_embeddings import OpponentEmbeddingTable
from agent.train_scenarios.mcts_predict.train import _mcts_forward, train_mcts
from agent.mcts.collect import run_mcts_collection

from test_mcts_lm_loss import _TINY_CONFIG, _event

_OPP_CONFIG = copy.deepcopy(_TINY_CONFIG)
_OPP_CONFIG["architecture"]["opponent_embedding"] = {"enabled": True}


def _make_agent(seed=0):
    torch.manual_seed(seed)
    agent = ASI(lambda m: None, config=_OPP_CONFIG)
    agent.set_device("cpu")
    return agent


def _seq_with_opponent(opp_id, pot=20.0):
    """Two-decision sequence whose events carry an opponent_id (the acting
    player's identity — what the GRU table is keyed by)."""
    seq = [_event(None, pot=pot), _event(1, pot=pot),
           _event(None, pot=2 * pot)]
    for e in seq:
        e["opponent_id"] = opp_id
    return seq


def test_mcts_forward_accumulates_into_passed_table_across_calls():
    """The same table object passed to consecutive forwards (= consecutive
    cycles) keeps its entries and keeps evolving them — no reset."""
    agent = _make_agent()
    table = OpponentEmbeddingTable(agent.perception.d_model)

    with torch.no_grad():
        _mcts_forward(agent, [_seq_with_opponent("villain_A")], [[]], "cpu",
                      opponent_emb_table=table)
    assert "villain_A" in table.embeddings
    state_after_first = table.embeddings["villain_A"].detach().clone()
    assert not torch.equal(state_after_first,
                           torch.zeros_like(state_after_first))

    with torch.no_grad():
        _mcts_forward(agent, [_seq_with_opponent("villain_B")], [[]], "cpu",
                      opponent_emb_table=table)
    # Old entry survived the second call; new one was added alongside.
    assert "villain_A" in table.embeddings
    assert "villain_B" in table.embeddings
    assert torch.equal(table.embeddings["villain_A"].detach(),
                       state_after_first)

    with torch.no_grad():
        _mcts_forward(agent, [_seq_with_opponent("villain_A")], [[]], "cpu",
                      opponent_emb_table=table)
    # Re-seeing villain_A ADVANCES the persisted state (seeded from it),
    # rather than restarting from zeros: a fresh table fed the same events
    # ends in a different state than the twice-fed persistent one.
    fresh = OpponentEmbeddingTable(agent.perception.d_model)
    with torch.no_grad():
        _mcts_forward(agent, [_seq_with_opponent("villain_A")], [[]], "cpu",
                      opponent_emb_table=fresh)
    assert not torch.equal(table.embeddings["villain_A"].detach(),
                           fresh.embeddings["villain_A"].detach())


def test_train_mcts_accepts_and_uses_external_table():
    """train_mcts(opponent_emb_table=...) must use the caller's table
    (persistent across cycles) instead of creating a fresh one."""
    sig = inspect.signature(train_mcts)
    assert "opponent_emb_table" in sig.parameters
    assert sig.parameters["opponent_emb_table"].default is None
    src = inspect.getsource(train_mcts)
    assert "opp_table = opponent_emb_table" in src


def test_run_mcts_collection_accepts_and_fills_external_dict():
    """run_mcts_collection(opp_tables=...) must reuse the caller's dict and
    only create tables for agents missing from it."""
    sig = inspect.signature(run_mcts_collection)
    assert "opp_tables" in sig.parameters
    assert sig.parameters["opp_tables"].default is None
    src = inspect.getsource(run_mcts_collection)
    assert "if opp_tables is None" in src
    assert 'a["name"] not in opp_tables' in src


def test_pipeline_passes_persistent_tables():
    """The cyclic loop owns the persistent tables and threads them into both
    collection and training."""
    pipeline_path = os.path.join(_PKG_ROOT, "pipeline.py")
    with open(pipeline_path) as f:
        src = f.read()
    assert "mcts_collection_opp_tables = {}" in src
    assert "opp_tables=mcts_collection_opp_tables" in src
    assert 'opponent_emb_table=agent_info.get("opp_emb_table")' in src
