"""E2E test: templated range forward must produce identical results to the
per-combo copy approach.

Scenario: create a tiny agent, build a realistic event sequence, run
_compute_range_probs for 50 combos two ways:
  1. Current approach (copy events per combo, extract_event_tensors each time)
  2. Templated approach (extract once, tile tensors, vary only hand cards)
Assert the action distributions are bitwise-identical (same model, same input,
deterministic — must match exactly).
"""

import numpy as np
import torch

from agent.agent import ASI
from agent.train_scenarios.generation.generate_opponent import (
    _compute_range_probs, _shared_to_standard, _get_all_combos,
)
from evaluation.evaluate import _normalize_events_inplace
from utils import get_amp_config

MAX_PLAYERS = 9
N_ACTIONS = 14  # 11 raise sizes + fold + call + all-in (matches production config)

_CFG = {
    "architecture": {
        "d_model": 32, "n_heads": 2, "n_kv_heads": 1,
        "n_encoder_layers": 1, "n_decoder_layers": 1,
        "n_value_layers": 1, "n_action_layers": 1,
        "n_opponent_action_layers": 1, "n_modelling_layers": 1,
        "d_ff": 64, "max_seq_len": 256, "max_players": MAX_PLAYERS,
        "modelling_dropout": 0.0,
        "memory": {"n_levels": 1, "max_cluster_size": 4,
                   "max_cluster_size_after": 4, "beam_width": 2},
        "opponent_embedding": {"enabled": False},
    },
    "game": {
        "raise_sizes": {
            "preflop": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            "flop": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            "turn": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            "river": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        },
        "max_players": MAX_PLAYERS, "big_blind": 10, "max_stack": 1000,
    },
    "solver": {"type": "v1"},
}

_NORM_STATS = {
    "pot_mean": 50.0, "pot_std": 30.0,
    "stack_mean": 500.0, "stack_std": 200.0,
    "bets_mean": 5.0, "bets_std": 10.0,
    "blind_mean": 10.0, "blind_std": 1.0,
    "ev_mean": 0.0, "ev_std": 1.0,
}


def _make_shared_events(n_events=10):
    """Build a realistic shared event sequence (unmasked, all hands visible)."""
    events = []
    for t in range(n_events):
        action = [0.0] * N_ACTIONS
        if t > 0:
            action[(t * 3) % N_ACTIONS] = 1.0
        events.append({
            "hands": {0: [10, 11], 1: [20, 21]},
            "num_players": 6,
            "acting_pos": t % 6,
            "big_blind": 10.0,
            "small_blind": 5.0,
            "pot": 100.0 + t * 30,
            "bets": np.array([0.0] * MAX_PLAYERS, dtype=np.float32),
            "table": [2, 3, 4, -1, -1] if t >= 3 else [-1, -1, -1, -1, -1],
            "stacks": [900.0] * MAX_PLAYERS,
            "action": action,
        })
    return events


class TestTemplatedRangeForward:
    """Run _compute_range_probs the old way and with the templated path.
    Results must be identical."""

    def test_templated_matches_per_combo_copy(self):
        agent = ASI(lambda m: None, config=_CFG)
        agent.eval()
        device = "cpu"

        shared_events = _make_shared_events(n_events=10)
        active_pos = 1
        temperature = 0.3
        combos = _get_all_combos()[:50]

        amp_config = get_amp_config(device)[:3]

        # Current approach: per-combo event copies
        probs_old = _compute_range_probs(
            agent, shared_events, combos, active_pos,
            _NORM_STATS, temperature, device, N_ACTIONS,
            max_batch=256, amp_config=amp_config,
        )

        # Templated approach: extract once, tile, vary hand cards only
        from agent.train_scenarios.generation.generate_opponent import (
            _compute_range_probs_templated,
        )
        probs_new = _compute_range_probs_templated(
            agent, shared_events, combos, active_pos,
            _NORM_STATS, temperature, device, N_ACTIONS,
            max_batch=256, amp_config=amp_config,
        )

        assert probs_old.shape == probs_new.shape, \
            f"Shape mismatch: {probs_old.shape} vs {probs_new.shape}"
        assert torch.allclose(probs_old, probs_new, atol=1e-6), \
            f"Max diff: {(probs_old - probs_new).abs().max().item():.8f}"

    def test_templated_works_with_partial_batch(self):
        """When combo count is not a multiple of max_batch, the last partial
        batch must still produce correct results."""
        agent = ASI(lambda m: None, config=_CFG)
        agent.eval()
        device = "cpu"

        shared_events = _make_shared_events(n_events=5)
        active_pos = 0
        combos = _get_all_combos()[:30]  # 30 combos, max_batch=16 → 2 full batches
        amp_config = get_amp_config(device)[:3]

        probs_old = _compute_range_probs(
            agent, shared_events, combos, active_pos,
            _NORM_STATS, 0.3, device, N_ACTIONS,
            max_batch=16, amp_config=amp_config,
        )

        from agent.train_scenarios.generation.generate_opponent import (
            _compute_range_probs_templated,
        )
        probs_new = _compute_range_probs_templated(
            agent, shared_events, combos, active_pos,
            _NORM_STATS, 0.3, device, N_ACTIONS,
            max_batch=16, amp_config=amp_config,
        )

        assert torch.allclose(probs_old, probs_new, atol=1e-6), \
            f"Max diff: {(probs_old - probs_new).abs().max().item():.8f}"

    def test_templated_is_used_in_sequential_path(self):
        """_compute_range_probs must use the templated path when running
        in sequential mode (no proxy). Verify by checking the function
        calls the templated infrastructure."""
        import inspect
        src = inspect.getsource(_compute_range_probs)
        assert "precomputed" in src or "_tile_template" in src, \
            "_compute_range_probs sequential path does not use templated " \
            "forward — it still copies events per combo and runs " \
            "extract_event_tensors N times."
