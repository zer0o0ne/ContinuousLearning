"""Tests for Этап-0 fix 0.5: opponent-embedding keying in evaluation.

Verifies the invariants the fix relies on (composite "{table_uid}:{agent_name}"
keys), without standing up a full model forward:

  * the SAME opponent agent seated at two different tables produces disjoint
    embedding keys, so their embeddings are independent (no cross-table clobber
    inside a shared per-agent table);
  * the matchup-boundary reset deletes only the finished table's namespaced
    entries, leaving other tables' state intact.

Run (from versions/v6):
    python -m tests.test_eval_opp_emb_keys
"""

import torch

from agent.perception.opponent_embeddings import OpponentEmbeddingTable


def _opp_keys(table_uid, seated_names):
    """Mirror the composite-key construction in evaluation/evaluate.py (0.5)."""
    return [f"{table_uid}:{nm}" for nm in seated_names]


def test_composite_keys_independent_across_tables():
    seated = ["alice", "bob"]
    k0 = _opp_keys(0, seated)
    k1 = _opp_keys(1, seated)
    # Same agents, different table_uid -> disjoint key sets.
    assert set(k0).isdisjoint(set(k1)), (k0, k1)

    # One per-agent table shared across both tables (the eval reality): the same
    # opponent agent 'alice' at two parallel tables lands in independent slots.
    tbl = OpponentEmbeddingTable(d_model=4)
    tbl.embeddings["0:alice"] = torch.ones(4)
    tbl.embeddings["1:alice"] = torch.full((4,), 7.0)
    assert torch.equal(tbl.get("0:alice", "cpu"), torch.ones(4))
    assert torch.equal(tbl.get("1:alice", "cpu"), torch.full((4,), 7.0))
    print(f"test_composite_keys_independent_across_tables: OK keys={k0} vs {k1}")


def test_reset_is_table_scoped():
    tbl = OpponentEmbeddingTable(d_model=4)
    tbl.embeddings["0:alice"] = torch.ones(4)
    tbl.embeddings["0:bob"] = torch.ones(4)
    tbl.embeddings["1:alice"] = torch.ones(4)

    # Reset table 0 — mirror the evaluate.py finalize block.
    prefix = "0:"
    for key in [k for k in tbl.embeddings if k.startswith(prefix)]:
        del tbl.embeddings[key]

    assert set(tbl.embeddings.keys()) == {"1:alice"}, set(tbl.embeddings.keys())
    print("test_reset_is_table_scoped: OK (table 0 cleared, table 1 intact)")


if __name__ == "__main__":
    test_composite_keys_independent_across_tables()
    test_reset_is_table_scoped()
    print("\nALL OPP-EMB KEYING TESTS PASSED")
