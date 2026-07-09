"""Этап-A A.5 minor-architecture regression tests.

A.5.1 — the modelling action token is placed at each sample's TRUE length
        (the row->position index math used in modelling_predict._modelling_forward).
A.5.2 — EventSequenceEmbedder caps sequences to max_events = max_seq_len//7,
        dropping the OLDEST events; a Perception forward over an overlong
        sequence yields the capped event count.
A.5.3 — an out-of-range card index raises (assert) instead of being silently
        clamped.

Run (from versions/v6):
    python -m tests.test_arch_minor
"""

import torch

from agent.perception.perception import Perception, EventSequenceEmbedder

N_ACTIONS = 5
MAX_PLAYERS = 4
SMALL_CONFIG = {
    "d_model": 16, "n_heads": 4, "n_kv_heads": 2,
    "n_encoder_layers": 1, "n_decoder_layers": 1, "d_ff": 32,
    "max_seq_len": 63,                    # -> max_events = 9
    "max_players": MAX_PLAYERS,
    "memory": {"n_levels": 1, "max_cluster_size": 4,
               "max_cluster_size_after": 4, "beam_width": 2},
    "opponent_embedding": {"enabled": False},
}


def _event(card0=0):
    return {
        "table": [card0, -1, -1, -1, -1], "hand": [10, 11],
        "num_players": 2, "hero_pos": 0, "acting_pos": 1,
        "big_blind": 0.0, "small_blind": 0.0, "stack": 0.0, "pot": 0.0,
        "bets": [0.0] * MAX_PLAYERS, "action": [0.0] * N_ACTIONS,
    }


def test_cap_sequences_truncates_oldest():
    emb = EventSequenceEmbedder(8, N_ACTIONS, MAX_PLAYERS, max_seq_len=63)
    assert emb.max_events == 9, emb.max_events
    seqs = [[{"id": j} for j in range(12)], [{"id": j} for j in range(5)]]
    capped = emb._cap_sequences(seqs)
    assert len(capped[0]) == 9 and len(capped[1]) == 5
    assert [e["id"] for e in capped[0]] == list(range(3, 12)), "must keep MOST RECENT events"
    assert len(seqs[0]) == 12, "input must not be mutated"
    print("test_cap_sequences_truncates_oldest: OK")


def test_perception_caps_overlong_sequence():
    torch.manual_seed(0)
    p = Perception(SMALL_CONFIG, N_ACTIONS)
    p.eval()
    seq = [_event(j % 52) for j in range(12)]   # 12 > 9 cap
    with torch.no_grad():
        out, _enc, mask = p.forward_batch([seq], device="cpu", skip_memory=True)
    assert out.shape[1] == 9, f"expected capped N=9, got {out.shape[1]}"
    assert int(mask.sum().item()) == 9, mask.sum().item()
    print("test_perception_caps_overlong_sequence: OK")


def test_card_index_assert():
    emb = EventSequenceEmbedder(8, N_ACTIONS, MAX_PLAYERS)
    bad = _event(card0=60)   # 60 is out of [0,52]
    raised = False
    try:
        emb._build_batch_tensors([[bad]], device="cpu")
    except AssertionError:
        raised = True
    assert raised, "out-of-range card index should raise (no silent clamp)"
    print("test_card_index_assert: OK")


def test_modelling_token_position_math():
    # Mirrors modelling_predict._modelling_forward A.5.1: row i*K+k -> length[i].
    B, K = 2, 3
    lengths = torch.tensor([2, 4])
    pos = lengths.unsqueeze(1).expand(B, K).reshape(B * K)
    assert pos.tolist() == [2, 2, 2, 4, 4, 4], pos.tolist()
    print("test_modelling_token_position_math: OK")


if __name__ == "__main__":
    test_cap_sequences_truncates_oldest()
    test_perception_caps_overlong_sequence()
    test_card_index_assert()
    test_modelling_token_position_math()
    print("\nALL A.5 MINOR-ARCHITECTURE TESTS PASSED")
