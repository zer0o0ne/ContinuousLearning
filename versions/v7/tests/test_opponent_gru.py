"""Этап-A A.4 opponent-GRU regression tests.

A.4.1 — the GRU update signal is taken from the TABLE-card slots (0-4) only;
        perturbing the observer's hole cards (slots 5,6) must NOT change the
        injected opponent embedding, while perturbing the board must.
A.4.2 — per-event causal injection: each of an opponent's events gets its own
        running embedding (not one shared pooled value), and an event's
        embedding never depends on LATER events (here: events in a later sample
        of the same batch) — no cross-sample future leak.
A.4.3 — OpponentEmbeddingTable.clone() is an independent deep copy, so a
        validation pass on a clone cannot mutate the training table.

Run (from versions/v6):
    python -m tests.test_opponent_gru
"""

import copy

import torch

from agent.perception.perception import Perception
from agent.perception.opponent_embeddings import OpponentEmbeddingTable

N_ACTIONS = 5
MAX_PLAYERS = 4
CONFIG = {
    "d_model": 16,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "d_ff": 32,
    "max_seq_len": 64,
    "max_players": MAX_PLAYERS,
    "memory": {"n_levels": 1, "max_cluster_size": 4,
               "max_cluster_size_after": 4, "beam_width": 2},
    "opponent_embedding": {"enabled": True},
}


def _event(opp_id, table, hand):
    return {
        "table": list(table), "hand": list(hand),
        "num_players": 2, "hero_pos": 0, "acting_pos": 1,
        "big_blind": 0.0, "small_blind": 0.0, "stack": 0.0, "pot": 0.0,
        "bets": [0.0] * MAX_PLAYERS, "action": [0.0] * N_ACTIONS,
        "opponent_id": opp_id,
    }


def _capture_injected(perception, event_sequences):
    """Run forward_batch (fresh table) and return the per-event embeddings
    passed to the injector (list of length T, entries (d_model,) or None)."""
    cap = {}
    orig = perception.embedder._apply_post_inject

    def spy(out_pre, meta, opp_embs, device="cpu"):
        cap["embs"] = opp_embs
        return orig(out_pre, meta, opp_embs, device=device)

    perception.embedder._apply_post_inject = spy
    try:
        table = OpponentEmbeddingTable(CONFIG["d_model"])
        with torch.no_grad():
            perception.forward_batch(event_sequences, device="cpu",
                                     skip_memory=True, skip_opponent_emb=False,
                                     opponent_emb_table=table, gru_window=1)
    finally:
        perception.embedder._apply_post_inject = orig
    return cap["embs"]


def _build_perception():
    torch.manual_seed(0)
    p = Perception(CONFIG, N_ACTIONS)
    p.eval()
    return p


def test_signal_uses_table_slots_only():
    p = _build_perception()
    # Sample 0: opp X acts on a flop board with hole cards [10, 11].
    base = [[_event("X", [0, 4, 8, -1, -1], [10, 11])]]
    embs_base = _capture_injected(p, base)
    # Perturb only the OBSERVER hole cards (slots 5,6) -> embedding unchanged.
    hole = [[_event("X", [0, 4, 8, -1, -1], [20, 21])]]
    embs_hole = _capture_injected(p, hole)
    assert torch.allclose(embs_base[0], embs_hole[0], atol=1e-6), (
        "A.4.1: hole-card change altered the opponent embedding (signal leaked slots 5/6)")
    # Perturb the BOARD (table slots 0-4) -> embedding must change.
    board = [[_event("X", [1, 5, 9, -1, -1], [10, 11])]]
    embs_board = _capture_injected(p, board)
    assert not torch.allclose(embs_base[0], embs_board[0], atol=1e-6), (
        "A.4.1: board change had no effect — signal not derived from table slots")
    print("test_signal_uses_table_slots_only: OK")


def test_per_event_causal_no_cross_sample_leak():
    p = _build_perception()
    # Batch of two samples, both featuring opp X. Flat order: s0e0, s0e1, s1e0.
    seqs = [
        [_event("X", [0, 4, 8, -1, -1], [10, 11]),
         _event("X", [0, 4, 8, 12, -1], [10, 11])],
        [_event("X", [0, 4, 8, 12, 16], [30, 31])],
    ]
    embs = _capture_injected(p, seqs)
    assert len(embs) == 3 and all(e is not None for e in embs)
    # Per-event: the two events of X in sample 0 get DISTINCT running states.
    assert not torch.allclose(embs[0], embs[1], atol=1e-6), (
        "A.4.2: both events share one pooled embedding — not per-event causal")
    # No future leak: change sample 1's (later) event; sample 0's embeddings
    # must be identical (they precede it in flat order).
    seqs2 = copy.deepcopy(seqs)
    seqs2[1][0]["table"] = [2, 6, 10, 14, 18]
    embs2 = _capture_injected(p, seqs2)
    assert torch.allclose(embs[0], embs2[0], atol=1e-6) and \
            torch.allclose(embs[1], embs2[1], atol=1e-6), (
        "A.4.2: a later sample's event changed an earlier sample's embedding (cross-sample leak)")
    print("test_per_event_causal_no_cross_sample_leak: OK")


def test_clone_isolates_validation():
    table = OpponentEmbeddingTable(8)
    table.embeddings["a"] = torch.ones(8)
    snapshot = {k: v.clone() for k, v in table.embeddings.items()}
    val_copy = table.clone()
    # Simulate a validation forward advancing the clone's table.
    val_copy.embeddings["a"] = val_copy.embeddings["a"] + 5.0
    val_copy.embeddings["b"] = torch.full((8,), 9.0)
    # Original training table untouched.
    assert set(table.embeddings.keys()) == {"a"}, table.embeddings.keys()
    assert torch.equal(table.embeddings["a"], snapshot["a"]), "clone shares storage with original"
    print("test_clone_isolates_validation: OK")


if __name__ == "__main__":
    test_clone_isolates_validation()
    test_signal_uses_table_slots_only()
    test_per_event_causal_no_cross_sample_leak()
    print("\nALL OPPONENT-GRU (A.4) TESTS PASSED")
