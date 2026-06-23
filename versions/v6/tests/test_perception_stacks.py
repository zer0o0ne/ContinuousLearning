"""Per-position stacks event-schema tests (Audit B.6.2 + B.6.1).

B.6.2 adds a per-position stacks VECTOR to every event and a dedicated
projection in EventSequenceEmbedder (combine becomes Linear(8d, d)). Without an
effective-stack signal across seats the agent cannot learn stack-aware play.

B.6.1 samples independent per-seat starting stacks, so generated events carry
asymmetric stacks.

Run (from versions/v6):
    python -m tests.test_perception_stacks
"""

import random
import numpy as np
import torch

from agent.perception.perception import EventSequenceEmbedder

D_MODEL = 16
N_ACTIONS = 6
MAX_PLAYERS = 4


def _event(stacks, num_players=4, hero_pos=0, acting_pos=1):
    action = [0.0] * N_ACTIONS
    action[1] = 1.0
    return {
        "hand": [12, 25], "table": [-1, -1, -1, -1, -1],
        "num_players": num_players, "hero_pos": hero_pos, "acting_pos": acting_pos,
        "big_blind": 10.0, "small_blind": 5.0,
        "stack": float(stacks[hero_pos]), "stacks": list(stacks),
        "pot": 30.0, "bets": [5.0, 10.0, 0.0, 0.0], "action": action,
    }


def test_embedder_shapes():
    emb = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
    assert tuple(emb.combine.weight.shape) == (D_MODEL, D_MODEL * 8), (
        f"combine must be Linear(8d, d): {tuple(emb.combine.weight.shape)}")
    assert hasattr(emb, "stacks_proj"), "stacks_proj missing"
    assert tuple(emb.stacks_proj.weight.shape) == (D_MODEL, MAX_PLAYERS)
    print("test_embedder_shapes: OK (combine 8d, stacks_proj present)")


def test_stacks_consumed_batched():
    torch.manual_seed(0)
    emb = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS).eval()
    e1 = _event([200.0, 150.0, 300.0, 120.0])
    e2 = _event([40.0, 150.0, 300.0, 120.0])  # only hero's stack differs
    with torch.no_grad():
        o1, _ = emb.forward_batch([[e1]], device="cpu")
        o2, _ = emb.forward_batch([[e2]], device="cpu")
    assert not torch.allclose(o1, o2), "stacks vector not consumed (batched path)"
    print("test_stacks_consumed_batched: OK")


def test_stacks_consumed_single():
    torch.manual_seed(0)
    emb = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS).eval()
    with torch.no_grad():
        a = emb.embed_event(_event([200.0, 150.0, 300.0, 120.0]))
        b = emb.embed_event(_event([200.0, 150.0, 40.0, 120.0]))  # seat-2 stack differs
    assert not torch.allclose(a, b), "stacks vector not consumed (single path)"
    print("test_stacks_consumed_single: OK")


def test_missing_stacks_defaults_zero():
    torch.manual_seed(0)
    emb = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS).eval()
    e_missing = _event([0.0, 0.0, 0.0, 0.0])
    del e_missing["stacks"]
    e_zeros = _event([0.0, 0.0, 0.0, 0.0])
    with torch.no_grad():
        om, _ = emb.forward_batch([[e_missing]], device="cpu")   # must not crash
        oz, _ = emb.forward_batch([[e_zeros]], device="cpu")
    assert torch.allclose(om, oz), "missing stacks must behave as a zero vector"
    print("test_missing_stacks_defaults_zero: OK")


def test_generation_emits_asymmetric_stacks():
    """B.6.1: generated events carry independent per-seat (asymmetric) stacks."""
    from agent.train_scenarios.generation.generate import generate_scenario
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    cfg = {
        "mc_iterations": 120, "big_blind": 10, "max_stack": 400, "max_players": 4,
        "gto_temperature": 0.2, "solver": "v3",
        "raise_sizes": {s: [0.5, 1.0, 2.0] for s in ("preflop", "flop", "turn", "river")},
        "eqr_enabled": True, "combo_response_iters": 6, "reraise_threshold": 0.72,
        "weighted_sampling": False,
        "threshold_smoothing": {"enabled": True, "beta_fold": 0.07, "beta_reraise": 0.07},
    }
    saw_stacks = False
    saw_asymmetric = False
    n_scen = 0
    for _ in range(10):
        res = generate_scenario(cfg, device="cpu")
        if not res:
            continue
        for s in res:
            if s.get("scenario_type") == "modelling":
                continue
            n_scen += 1
            for e in s["events"]:
                st = e.get("stacks")
                assert st is not None, "event missing per-position stacks (B.6.2)"
                assert len(st) == e["num_players"], "stacks length must be num_players"
                saw_stacks = True
                # initial (preflop) starting stacks differ across seats a.s.
                if len({round(float(x), 3) for x in st}) > 1:
                    saw_asymmetric = True
    assert n_scen > 10 and saw_stacks
    assert saw_asymmetric, "per-seat stacks never differed — B.6.1 not effective"
    print(f"test_generation_emits_asymmetric_stacks: OK (scenarios={n_scen})")


if __name__ == "__main__":
    test_embedder_shapes()
    test_stacks_consumed_batched()
    test_stacks_consumed_single()
    test_missing_stacks_defaults_zero()
    test_generation_emits_asymmetric_stacks()
    print("\nALL B.6 PERCEPTION-STACKS TESTS PASSED")
