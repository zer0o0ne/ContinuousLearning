"""Opponent-adaptation upgrade tests (PLAN_OPPONENT_ADAPTATION.md).

Covers:
  P0   — pool-ID <-> agent binding in opponent data generation (§1)
  §2   — style probe: canonical style vector, opp-state collection, MSE loss
  §3   — count-based opponent stats vector (HUD)
  §4   — showdown strength anchor

All tests are deterministic: fixed seeds, exact assertions, CPU only.

Run (from versions/v6):
    python -m tests.test_opponent_adaptation
"""

import json
import os
import random
import tempfile

import numpy as np
import torch

from agent.train_scenarios.generation.generate_opponent import (
    generate_opponent_hand, generate_opponent_dataset, load_opponent_shards,
)
from agent.train_scenarios.generation.generate import _read_meta

N_ACTIONS = 6  # 3 raise bins + fold/call/allin
_RS = [0.5, 1.0, 2.0]


class _StubAgent:
    """Uniform action logits — exercises generation without checkpoints."""

    def forward_batch(self, batch_events, skip_memory=True, heads=None,
                      precomputed=None):
        n = precomputed["B"] if precomputed is not None else len(batch_events)
        return {"action_logits": torch.zeros(n, N_ACTIONS)}


def _dummy_norm_stats():
    return {
        "pot_mean": 0.0, "pot_std": 1.0,
        "stack_mean": 0.0, "stack_std": 1.0,
        "bets_mean": 0.0, "bets_std": 1.0,
        "blind_mean": 0.0, "blind_std": 1.0,
    }


def _stub_agents(names):
    return [{
        "agent": _StubAgent(), "name": n,
        "norm_stats": _dummy_norm_stats(), "temperature": 1.0,
    } for n in names]


def _gen_config(n_hands, n_player_pool=6, swap_prob=0.3):
    return {
        "game": {
            "raise_sizes": {s: list(_RS)
                            for s in ("preflop", "flop", "turn", "river")},
            "max_players": 4,
            "big_blind": 10,
            "max_stack": 300,
        },
        "opponent_data": {
            "n_hands": n_hands,
            "n_workers": 1,
            "min_stack": 120,
            "max_batch_combos": 256,
            "save_every_hands": 10,
            "player_swap_prob": swap_prob,
            "n_player_pool": n_player_pool,
            "action_temperature": 1.0,
            "bayes": {"enabled": True, "tau_belief": 2.0,
                      "ess_truncation_mass": 0.9},
        },
    }


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# P0 — pool binding (§1)
# ---------------------------------------------------------------------------

def test_pool_binding_consistency():
    """Test 1: sequential generation — every persistent ID is always played
    by its round-robin-bound agent; labels + meta binding persisted."""
    _seed_all(7)
    names = ["agent_b", "agent_a"]  # deliberately unsorted
    agents = _stub_agents(names)
    sorted_names = sorted(names)

    cfg = _gen_config(n_hands=60)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = generate_opponent_dataset(cfg, tmpdir, "cpu", lambda m: None,
                                        agents_override=agents)
        assert out == tmpdir
        scenarios = load_opponent_shards(tmpdir)
        assert len(scenarios) > 30, f"too few scenarios: {len(scenarios)}"

        meta = _read_meta(tmpdir)
        binding = meta.get("pool_binding")
        assert binding is not None, "meta.json missing pool_binding"
        # Round-robin over agents_list order (as provided, not sorted)
        pool = cfg["opponent_data"]["n_player_pool"]
        expected = {f"p_{i}": names[i % len(names)] for i in range(pool)}
        assert binding == expected, f"binding {binding} != round-robin {expected}"

        seen_by_id = {}
        for s in scenarios:
            acting = s["acting_pos"]
            opp_id = s["opponent_ids"][acting]
            # Scenario label matches the binding for the acting persistent ID
            assert s["acting_agent"] == binding[opp_id], (
                f"scenario acting_agent {s['acting_agent']} != "
                f"binding[{opp_id}]={binding[opp_id]}")
            # Index matches the sorted-name convention
            assert s["acting_agent_idx"] == sorted_names.index(s["acting_agent"])
            seen_by_id.setdefault(opp_id, set()).add(s["acting_agent"])

        # Every persistent ID maps to exactly ONE agent across all hands
        multi = {k: v for k, v in seen_by_id.items() if len(v) != 1}
        assert not multi, f"IDs played by multiple agents: {multi}"
        # The test actually exercised repeats (same ID in many scenarios)
        assert any(len(v) == 1 for v in seen_by_id.values())
        assert len(seen_by_id) >= 2, "vacuous: fewer than 2 IDs observed"
    print(f"test_pool_binding_consistency: OK "
          f"({len(scenarios)} scenarios, {len(seen_by_id)} IDs)")


def test_pool_binding_hand_level():
    """Test 1b: direct generate_opponent_hand honors an explicit binding
    (the exact call shape the parallel actors use)."""
    _seed_all(11)
    names = ["x", "y", "z"]
    agents = _stub_agents(names)
    amp = (False, "cpu", torch.float32)
    gen_cfg = {}
    gen_cfg.update(_gen_config(1)["game"])
    gen_cfg.update(_gen_config(1)["opponent_data"])

    pool = [f"p_{i}" for i in range(6)]
    binding = {pool[i]: names[i % 3] for i in range(6)}
    n_checked = 0
    for _ in range(30):
        roster = random.sample(pool, gen_cfg["max_players"])
        res = generate_opponent_hand(gen_cfg, agents, "cpu", amp,
                                     player_ids=roster, pool_binding=binding)
        if not res:
            continue
        for s in res:
            opp_id = s["opponent_ids"][s["acting_pos"]]
            assert s["acting_agent"] == binding[opp_id]
            n_checked += 1
    assert n_checked > 20, f"vacuous: only {n_checked} scenarios"
    print(f"test_pool_binding_hand_level: OK ({n_checked} scenarios)")


def test_pool_binding_legacy_fallback():
    """Without a binding, generation still works (legacy random seating) and
    scenarios still carry acting_agent labels."""
    _seed_all(13)
    agents = _stub_agents(["only"])
    amp = (False, "cpu", torch.float32)
    gen_cfg = {}
    gen_cfg.update(_gen_config(1)["game"])
    gen_cfg.update(_gen_config(1)["opponent_data"])
    res = None
    for _ in range(10):
        res = generate_opponent_hand(gen_cfg, agents, "cpu", amp)
        if res:
            break
    assert res, "no scenarios generated on the legacy path"
    for s in res:
        assert s["acting_agent"] == "only"
        assert s["acting_agent_idx"] == 0
    print("test_pool_binding_legacy_fallback: OK")


# ---------------------------------------------------------------------------
# §3 — count-based opponent stats (HUD vector)
# ---------------------------------------------------------------------------

from agent.perception.perception import Perception
from agent.perception.opponent_embeddings import (
    OpponentEmbeddingTable, stats_features, action_category,
    zero_stat_counts, N_STAT_FEATURES,
)

MAX_PLAYERS = 4
_PERC_CONFIG = {
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
    "opponent_embedding": {"enabled": True, "stats_enabled": True},
}


def _perc_event(opp_id, acting_pos, action_idx=None, table=(-1,) * 5,
                bets=(0.0,) * MAX_PLAYERS, hand=(10, 11)):
    action = [0.0] * N_ACTIONS
    if action_idx is not None:
        action[action_idx] = 1.0
    return {
        "table": list(table), "hand": list(hand),
        "num_players": 2, "hero_pos": 0, "acting_pos": acting_pos,
        "big_blind": 0.0, "small_blind": 0.0, "stack": 0.0, "pot": 0.0,
        "bets": list(bets), "action": action,
        "opponent_id": opp_id,
    }


def _build_perception(stats=True, seed=0):
    torch.manual_seed(seed)
    cfg = {k: v for k, v in _PERC_CONFIG.items()}
    cfg["opponent_embedding"] = {"enabled": True}
    if stats is not None:
        cfg["opponent_embedding"]["stats_enabled"] = stats
    p = Perception(cfg, N_ACTIONS)
    p.eval()
    return p


def test_stats_features_pure():
    """Test 3: exact feature values from hand-built counts."""
    z = stats_features(zero_stat_counts())
    assert z.shape == (N_STAT_FEATURES,) and z.dtype == np.float32
    np.testing.assert_allclose(z[:40], 0.2, rtol=0, atol=1e-7)   # (0+1)/(0+5)
    np.testing.assert_allclose(z[40:], 0.0, rtol=0, atol=1e-7)

    counts = zero_stat_counts()
    counts[0] = [2, 1, 0, 0, 0]      # bucket 0: 3 observations
    f = stats_features(counts)
    np.testing.assert_allclose(
        f[0:5], [3 / 8, 2 / 8, 1 / 8, 1 / 8, 1 / 8], rtol=0, atol=1e-7)
    np.testing.assert_allclose(f[5:40], 0.2, rtol=0, atol=1e-7)
    expected_conf = np.log1p(3.0) / 5.0
    np.testing.assert_allclose(f[40], expected_conf, rtol=0, atol=1e-7)
    np.testing.assert_allclose(f[41:48], 0.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(f[48], expected_conf, rtol=0, atol=1e-7)

    # Category mapping: layout [fold, call, r0, r1, r2, allin], bins=3, mid=1
    assert [action_category(i, N_ACTIONS) for i in range(N_ACTIONS)] == \
        [0, 1, 2, 3, 3, 4]
    print("test_stats_features_pure: OK")


def test_stats_update_counts():
    """Test 4: attribution — an event's action is counted for the PREVIOUS
    event's acting player in the previous event's street/facing bucket;
    zero-action events count nothing."""
    p = _build_perception()
    table = OpponentEmbeddingTable(_PERC_CONFIG["d_model"])
    seq = [
        # A to act preflop facing a bet (bets: A=1, max=2) → bucket 0*2+1=1
        _perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
        # A called (idx 1). Now B to act, not facing (2 vs 2) → bucket 0
        _perc_event("B", acting_pos=1, action_idx=1, bets=(2.0, 2.0, 0.0, 0.0)),
        # B big-raised (idx 3). Back to A.
        _perc_event("A", acting_pos=0, action_idx=3, bets=(2.0, 6.0, 0.0, 0.0)),
    ]
    with torch.no_grad():
        p.forward_batch([seq], device="cpu", skip_memory=True,
                        skip_opponent_emb=False, opponent_emb_table=table)

    exp_a = zero_stat_counts()
    exp_a[1, 1] = 1.0     # A: call, preflop facing
    exp_b = zero_stat_counts()
    exp_b[0, 3] = 1.0     # B: big raise, preflop not facing
    np.testing.assert_array_equal(table.stats["A"], exp_a)
    np.testing.assert_array_equal(table.stats["B"], exp_b)
    print("test_stats_update_counts: OK")


def test_stats_group_rewind():
    """Test 5: observer copies of one scenario (same group id) advance the
    stats ONCE; ungrouped copies advance twice."""
    seq = [
        _perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
        _perc_event("B", acting_pos=1, action_idx=1, bets=(2.0, 2.0, 0.0, 0.0)),
    ]
    for groups, expected in (([7, 7], 1.0), (None, 2.0)):
        p = _build_perception()
        table = OpponentEmbeddingTable(_PERC_CONFIG["d_model"])
        with torch.no_grad():
            p.forward_batch([seq, [dict(e) for e in seq]], device="cpu",
                            skip_memory=True, skip_opponent_emb=False,
                            opponent_emb_table=table,
                            gru_sample_groups=groups)
        assert table.stats["A"][1, 1] == expected, (
            f"groups={groups}: count {table.stats['A'][1, 1]} != {expected}")
    print("test_stats_group_rewind: OK")


def test_stats_injection_offset():
    """Test 4b: the stats projection is added at the injection point — with
    proj weight=0 / bias=c, every opp event's injected vector shifts by
    exactly c vs bias=0."""
    p = _build_perception()
    with torch.no_grad():
        p.opp_stats_proj.weight.zero_()
        p.opp_stats_proj.bias.zero_()
    seq = [
        _perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
        _perc_event("B", acting_pos=1, action_idx=1, bets=(2.0, 2.0, 0.0, 0.0)),
    ]

    captured = {}
    orig = p.embedder._apply_post_inject

    def spy(out_pre, meta, opp_embs, device="cpu"):
        captured["embs"] = [e.clone() if e is not None else None
                            for e in opp_embs]
        return orig(out_pre, meta, opp_embs, device=device)

    p.embedder._apply_post_inject = spy
    try:
        with torch.no_grad():
            p.forward_batch([seq], device="cpu", skip_memory=True,
                            skip_opponent_emb=False,
                            opponent_emb_table=OpponentEmbeddingTable(16))
        base = captured["embs"]
        with torch.no_grad():
            p.opp_stats_proj.bias.fill_(0.5)
            p.forward_batch([seq], device="cpu", skip_memory=True,
                            skip_opponent_emb=False,
                            opponent_emb_table=OpponentEmbeddingTable(16))
        shifted = captured["embs"]
    finally:
        p.embedder._apply_post_inject = orig

    assert len(base) == 2 and all(e is not None for e in base)
    for b, s in zip(base, shifted):
        torch.testing.assert_close(s, b + 0.5, rtol=0, atol=1e-6)
    print("test_stats_injection_offset: OK")


def test_stats_disabled_bitwise_legacy():
    """Test 6: stats_enabled=false is bitwise-identical to a config without
    the key (same seed → same params → same forward)."""
    seq = [
        _perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
        _perc_event("B", acting_pos=1, action_idx=1, bets=(2.0, 2.0, 0.0, 0.0)),
    ]
    p_off = _build_perception(stats=False, seed=3)
    p_absent = _build_perception(stats=None, seed=3)
    outs = []
    for p in (p_off, p_absent):
        table = OpponentEmbeddingTable(_PERC_CONFIG["d_model"])
        with torch.no_grad():
            out, _, _ = p.forward_batch([seq], device="cpu", skip_memory=True,
                                        skip_opponent_emb=False,
                                        opponent_emb_table=table)
        outs.append(out)
        assert not hasattr(p, "opp_stats_proj")
        assert len(table.stats) == 0 or all(
            v.sum() == 0 for v in table.stats.values())
    assert torch.equal(outs[0], outs[1]), "stats_enabled=false != absent key"
    print("test_stats_disabled_bitwise_legacy: OK")


def test_stats_state_dict_roundtrip():
    """Test 7: state_dict round-trip preserves stats; legacy states load."""
    t = OpponentEmbeddingTable(8)
    t.get("A", "cpu")
    t.stats["A"][2, 1] = 4.0
    t.embeddings["A"] = torch.arange(8, dtype=torch.float32)
    sd = t.state_dict()

    t2 = OpponentEmbeddingTable(8)
    t2.load_state_dict(sd)
    assert torch.equal(t2.embeddings["A"], t.embeddings["A"])
    np.testing.assert_array_equal(t2.stats["A"], t.stats["A"])
    # Mutating the copy must not touch the original (deep copy)
    t2.stats["A"][0, 0] = 99.0
    assert t.stats["A"][0, 0] == 0.0

    # Legacy state (embeddings only) — empty stats, no crash
    t3 = OpponentEmbeddingTable(8)
    t3.load_state_dict({"A": torch.zeros(8)})
    assert t3.stats == {}
    # ...and get() lazily creates fresh counts
    t3.get("A", "cpu")
    np.testing.assert_array_equal(t3.stats["A"], zero_stat_counts())

    # clone() deep-copies stats
    c = t.clone()
    c.stats["A"][0, 0] = 77.0
    assert t.stats["A"][0, 0] == 0.0
    print("test_stats_state_dict_roundtrip: OK")


# ---------------------------------------------------------------------------
# §2 — style probe
# ---------------------------------------------------------------------------

import math

from agent.agent import ASI
from agent.train_scenarios.modifiers import build_style_vector, STYLE_DIMS
from agent.train_scenarios.opponent_action_predict.train import (
    train_opponent_action, _build_probe_ctx, _probe_losses,
)

_ASI_CONFIG = {
    "architecture": {
        "d_model": 16,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 32,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_value_layers": 1,
        "n_action_layers": 1,
        "n_opponent_action_layers": 1,
        "n_modelling_layers": 1,
        "max_seq_len": 64,
        "max_players": MAX_PLAYERS,
        "memory": {"n_levels": 1, "max_cluster_size": 4,
                   "max_cluster_size_after": 4, "beam_width": 2},
        "opponent_embedding": {"enabled": True, "stats_enabled": True,
                               "style_probe": True, "showdown_probe": True},
    },
    "game": {
        "raise_sizes": {s: list(_RS)
                        for s in ("preflop", "flop", "turn", "river")},
        "max_players": MAX_PLAYERS,
        "big_blind": 10,
    },
}


def test_build_style_vector():
    """Test 8: exact hand-computed encodings (n_actions=14 — config layout)."""
    n_actions = 14  # 11 raise bins: small = idx 2..6 (5), big = 7..12 (6)
    bt = 0.2

    # Empty modifiers → zeros + log(base_temperature)
    v = build_style_vector([], n_actions, bt)
    assert len(v) == STYLE_DIMS
    np.testing.assert_allclose(v[:15], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v[15], math.log(bt), rtol=0, atol=1e-12)

    # lag_bluffer from config.json — hand-computed
    mods = [
        {"type": "action_bias", "actions": "aggressive", "factor": 0.75},
        {"type": "action_bias", "actions": "fold", "factor": -0.25},
        {"type": "conditional_bias", "condition": "equity < 0.3",
         "actions": "raises", "factor": 0.5},
        {"type": "temperature", "value": 0.3},
    ]
    v = build_style_vector(mods, n_actions, bt)
    exp = [0.0] * STYLE_DIMS
    exp[0] = -0.25                       # fold bias
    exp[2] = exp[3] = exp[4] = 0.75      # aggressive covers raises + allin
    exp[7] = exp[8] = 0.5 * 0.3          # equity<0.3 raises, region 0.3
    exp[15] = math.log(0.3)
    np.testing.assert_allclose(v, exp, rtol=0, atol=1e-12)

    # Coverage weighting: "4:10" hits 3/5 small, 3/6 big; explicit indices
    mods = [
        {"type": "action_bias", "actions": "4:10", "factor": 0.7},
        {"type": "action_bias", "actions": [0, 1, 13], "factor": -0.15},
    ]
    v = build_style_vector(mods, n_actions, bt)
    exp = [0.0] * STYLE_DIMS
    exp[0] = exp[1] = exp[4] = -0.15
    exp[2] = 0.7 * 3 / 5
    exp[3] = 0.7 * 3 / 6
    exp[15] = math.log(bt)
    np.testing.assert_allclose(v, exp, rtol=0, atol=1e-12)

    # high-equity condition, region 1 - t
    mods = [{"type": "conditional_bias", "condition": "equity > 0.6",
             "actions": "fold", "factor": 0.5}]
    v = build_style_vector(mods, n_actions, bt)
    assert abs(v[10] - 0.5 * 0.4) < 1e-12
    # Determinism
    assert v == build_style_vector(mods, n_actions, bt)
    print("test_build_style_vector: OK")


def test_collect_opp_states_alignment():
    """Test 9: opp_last_states[i] == the injected h at sample i's last event;
    mask is 0 for samples without opponent ids."""
    p = _build_perception(stats=False, seed=5)
    seq_a = [
        _perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
        _perc_event("B", acting_pos=1, action_idx=1, bets=(2.0, 2.0, 0.0, 0.0)),
    ]
    seq_b = [_perc_event("C", acting_pos=0)]
    no_id = [dict(_perc_event(None, acting_pos=0))]
    no_id[0].pop("opponent_id")

    captured = {}
    orig = p.embedder._apply_post_inject

    def spy(out_pre, meta, opp_embs, device="cpu"):
        captured["embs"] = opp_embs
        return orig(out_pre, meta, opp_embs, device=device)

    p.embedder._apply_post_inject = spy
    try:
        table = OpponentEmbeddingTable(16)
        with torch.no_grad():
            _, _, _, (states, smask) = p.forward_batch(
                [seq_a, no_id, seq_b], device="cpu", skip_memory=True,
                skip_opponent_emb=False, opponent_emb_table=table,
                collect_opp_states=True)
    finally:
        p.embedder._apply_post_inject = orig

    embs = captured["embs"]
    # Flat order: a0, a1, noid0, b0 — last opp event per sample: 1, None, 3
    assert torch.equal(states[0], embs[1])
    assert torch.equal(states[2], embs[3])
    assert torch.equal(states[1], torch.zeros(16))
    assert smask.tolist() == [1.0, 0.0, 1.0]
    print("test_collect_opp_states_alignment: OK")


def _style_scenario(hand_id, style, opp_id, n_events=6):
    """Synthetic shared-format scenario whose event stream reflects a style:
    'folder' hands carry fold one-hots in small pots, 'raiser' — big-raise
    one-hots in inflated pots (as a real aggressor's hands would look)."""
    a_idx = 0 if style == "folder" else 3
    scale = 1.0 if style == "folder" else 8.0
    events = []
    for j in range(n_events):
        action = [0.0] * N_ACTIONS
        if 0 < j < n_events - 1:      # initial + final decision carry no action
            action[a_idx] = 1.0
        events.append({
            "hands": {},
            "num_players": 2,
            "acting_pos": 0,
            "big_blind": 10.0, "small_blind": 5.0,
            "pot": (15.0 + j) * scale, "bets": np.array([5.0, 10.0]) * scale,
            "table": [-1] * 5,
            "action": action,
            "stacks": [200.0, 195.0],
            "opponent_ids": {0: opp_id, 1: "hero_id"},
        })
    target = [0.0] * N_ACTIONS
    target[a_idx] = 1.0
    return {
        "events": events,
        "opponent_action_probs": target,
        "forward_combos": None,
        "acting_pos": 0,
        "hero_positions": [1],
        "num_players": 2,
        "pot": 15.0, "facing_bet": 5.0,
        "n_events": n_events,
        "hand_id": hand_id,
        "opponent_ids": {0: opp_id, 1: "hero_id"},
        "acting_agent": style,
    }


def _make_style_dataset(n_per_style=40):
    scenarios = []
    for h in range(n_per_style):
        scenarios.append(_style_scenario(2 * h, "folder", f"f_{h % 4}"))
        scenarios.append(_style_scenario(2 * h + 1, "raiser", f"r_{h % 4}"))
    return scenarios


def _style_targets():
    n_actions = N_ACTIONS
    return {
        "folder": build_style_vector(
            [{"type": "action_bias", "actions": "fold", "factor": 0.4},
             {"type": "temperature", "value": 0.5}], n_actions, 0.2),
        "raiser": build_style_vector(
            [{"type": "action_bias", "actions": "aggressive", "factor": 0.7}],
            n_actions, 0.2),
    }


def test_style_loss_e2e_learns_styles():
    """Test 10: real phase-5 training on two synthetic styles — style MSE
    falls and nearest-neighbor identification is perfect on validation."""
    _seed_all(21)
    agent = ASI(lambda m: None, config=_ASI_CONFIG)
    scenarios = _make_style_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        class _L:
            def __call__(self, m):
                pass

            def run_dir(self, phase):
                d = os.path.join(tmpdir, phase)
                os.makedirs(d, exist_ok=True)
                return d

        train_cfg = {
            "lr": 5e-3, "batch_size": 8, "epochs": 12, "val_split": 0.15,
            "log_every": 1000, "gru_window": 8,
            "style_probe_weight": 1.0,
            "showdown_probe_weight": 0.0,
            "style_targets": _style_targets(),
        }
        history, _ = train_opponent_action(
            agent, train_cfg, "cpu", _L(), scenarios_override=scenarios)

    style_hist = history["style_loss"]
    assert len(style_hist) > 5, "style loss was never computed"
    first = style_hist[0][1]
    last = style_hist[-1][1]
    assert last < first, f"style MSE did not fall: {first:.4f} -> {last:.4f}"
    nn_hist = history["val_style_nn_acc"]
    assert len(nn_hist) >= 1
    assert nn_hist[-1][1] == 1.0, f"final style_nn_acc {nn_hist[-1][1]} != 1.0"
    print(f"test_style_loss_e2e_learns_styles: OK "
          f"(MSE {first:.4f} -> {last:.4f}, nn_acc={nn_hist[-1][1]})")


def test_style_target_absent_is_masked():
    """Test 11: unknown/missing acting_agent and all-NaN showdown targets →
    probe losses contribute exactly 0; constant dims are excluded."""
    _seed_all(31)
    agent = ASI(lambda m: None, config=_ASI_CONFIG)
    logs = []
    ctx = _build_probe_ctx(
        agent,
        {"style_probe_weight": 1.0, "showdown_probe_weight": 1.0,
         "style_targets": _style_targets()},
        use_opp_emb=True, log=logs.append)
    assert ctx is not None and ctx["use_style"] and ctx["use_showdown"]
    # log-T dims differ, fold/raise dims differ → mask has informative dims,
    # and every pool-constant dim is excluded
    t = torch.tensor([_style_targets()["folder"], _style_targets()["raiser"]])
    const_dims = (t.std(dim=0, unbiased=False) < 1e-8)
    assert torch.equal(ctx["dim_mask"], ~const_dims)

    states = torch.randn(3, 16)
    out = {"opp_last_states": states, "opp_states_mask": torch.ones(3)}
    aux = [{"acting_agent": None, "showdown_target": float("nan")},
           {"acting_agent": "unknown_agent", "showdown_target": float("nan")},
           {"acting_agent": None, "showdown_target": float("nan")}]
    loss, style_l, sd_l, nn_c, nn_t = _probe_losses(agent, out, aux, ctx, "cpu")
    assert float(loss) == 0.0 and style_l is None and sd_l is None
    assert nn_c == 0 and nn_t == 0

    # Valid style label + NaN showdown → only the style part contributes
    aux[0]["acting_agent"] = "folder"
    loss2, style_l2, sd_l2, _, _ = _probe_losses(agent, out, aux, ctx, "cpu")
    assert style_l2 is not None and sd_l2 is None
    assert abs(loss2.detach().item() - ctx["style_w"] * style_l2) < 1e-6
    print("test_style_target_absent_is_masked: OK")


# ---------------------------------------------------------------------------
# §4 — showdown strength anchor
# ---------------------------------------------------------------------------

from agent.train_scenarios.generation.generate_opponent import (
    _showdown_strength, _label_showdown_strengths,
)


def test_showdown_strength_labels():
    """Test 12: strength values are ordered (nuts > air), bounded, and the
    generated labels are deterministic and sit on each actor's LAST scenario."""
    # Direct MC: aces vs 2-3 offsuit on a dry board (card = rank*4 + suit)
    board = [16, 25, 38, 47, 30]          # ranks 4, 6, 9, 11, 7 — no draws
    s_nuts = _showdown_strength(board, (48, 49), seed="1:0")    # A♠ A♥
    s_air = _showdown_strength(board, (0, 5), seed="1:1")       # 2, 3 offsuit
    assert 0.0 <= s_air < s_nuts <= 1.0, (s_air, s_nuts)
    # Determinism of the MC itself
    assert s_nuts == _showdown_strength(board, (48, 49), seed="1:0")

    # Generated data: labels only on the actor's last scenario, in [0,1],
    # and bitwise-identical across two same-seeded runs.
    def _run():
        _seed_all(17)
        agents = _stub_agents(["a", "b"])
        amp = (False, "cpu", torch.float32)
        gen_cfg = {}
        gen_cfg.update(_gen_config(1)["game"])
        gen_cfg.update(_gen_config(1)["opponent_data"])
        out = []
        for h in range(120):
            res = generate_opponent_hand(gen_cfg, agents, "cpu", amp,
                                         hand_seed=h)
            if res:
                out.append(res)
        return out

    hands_1 = _run()
    hands_2 = _run()
    n_labeled = 0
    labels_1, labels_2 = [], []
    for hand in hands_1:
        last_of_actor = {}
        for i, s in enumerate(hand):
            last_of_actor[s["acting_pos"]] = i
        for i, s in enumerate(hand):
            sd = s.get("actor_showdown_strength")
            if sd is not None:
                assert 0.0 <= sd <= 1.0
                assert last_of_actor[s["acting_pos"]] == i, (
                    "label not on the actor's LAST scenario")
                n_labeled += 1
                labels_1.append(sd)
    for hand in hands_2:
        for s in hand:
            sd = s.get("actor_showdown_strength")
            if sd is not None:
                labels_2.append(sd)
    assert n_labeled > 10, f"vacuous: only {n_labeled} labeled scenarios"
    assert labels_1 == labels_2, "labels not reproducible across seeded runs"
    print(f"test_showdown_strength_labels: OK "
          f"(nuts={s_nuts:.3f} > air={s_air:.3f}, {n_labeled} labels)")


def test_showdown_label_skipped_on_fold_and_collision():
    """Test 13: fold-ended hands get no labels; a board-colliding fixed hand
    is skipped while a clean one is labeled."""
    # Fold-ended: an agent that always folds → every hand ends by fold-out.
    class _Folder:
        def forward_batch(self, batch_events, skip_memory=True, heads=None,
                          precomputed=None):
            n = (precomputed["B"] if precomputed is not None
                 else len(batch_events))
            logits = torch.full((n, N_ACTIONS), -30.0)
            logits[:, 0] = 30.0
            return {"action_logits": logits}

    _seed_all(19)
    agents = [{"agent": _Folder(), "name": "folder",
               "norm_stats": _dummy_norm_stats(), "temperature": 1.0}]
    amp = (False, "cpu", torch.float32)
    gen_cfg = {}
    gen_cfg.update(_gen_config(1)["game"])
    gen_cfg.update(_gen_config(1)["opponent_data"])
    n_scen = 0
    for h in range(60):
        res = generate_opponent_hand(gen_cfg, agents, "cpu", amp, hand_seed=h)
        for s in res or []:
            assert s.get("actor_showdown_strength") is None, (
                "fold-ended hand got a showdown label")
            n_scen += 1
    assert n_scen > 20, f"vacuous: {n_scen} scenarios"

    # Collision skip: pos 0's fixed hand shares a card with the board.
    scen = [{"acting_pos": 0}, {"acting_pos": 1}]
    deck = np.array([16, 25, 38, 47, 30] + list(range(0, 10)))
    _label_showdown_strengths(
        scen, deck, players_state=np.array([1, 1]),
        fixed_hands={0: (16, 51), 1: (48, 49)},   # 16 is on the board
        num_players=2, hand_seed=5)
    assert "actor_showdown_strength" not in scen[0], "colliding hand labeled"
    assert 0.0 <= scen[1]["actor_showdown_strength"] <= 1.0
    # <2 live players → nothing labeled
    scen2 = [{"acting_pos": 0}]
    _label_showdown_strengths(
        scen2, deck, players_state=np.array([1, -1]),
        fixed_hands={0: (48, 49)}, num_players=2, hand_seed=5)
    assert "actor_showdown_strength" not in scen2[0]
    print("test_showdown_label_skipped_on_fold_and_collision: OK")


# ---------------------------------------------------------------------------
# Integration / compat
# ---------------------------------------------------------------------------

def _make_labeled_style_dataset(n_per_style=24):
    """Style dataset with showdown labels: folders reveal weak hands (0.2),
    raisers strong ones (0.8)."""
    scenarios = []
    for h in range(n_per_style):
        f = _style_scenario(2 * h, "folder", f"f_{h % 4}")
        f["actor_showdown_strength"] = 0.2
        r = _style_scenario(2 * h + 1, "raiser", f"r_{h % 4}")
        r["actor_showdown_strength"] = 0.8
        scenarios.extend([f, r])
    return scenarios


def _run_phase5_e2e(tmpdir, seed):
    _seed_all(seed)
    agent = ASI(lambda m: None, config=_ASI_CONFIG)
    scenarios = _make_labeled_style_dataset()

    class _L:
        def __call__(self, m):
            pass

        def run_dir(self, phase):
            d = os.path.join(tmpdir, phase)
            os.makedirs(d, exist_ok=True)
            return d

    train_cfg = {
        "lr": 3e-3, "batch_size": 8, "epochs": 2, "val_split": 0.15,
        "log_every": 1000, "gru_window": 8,
        "style_probe_weight": 0.5,
        "showdown_probe_weight": 0.5,
        "style_targets": _style_targets(),
    }
    history, run_dir = train_opponent_action(
        agent, train_cfg, "cpu", _L(), scenarios_override=scenarios)
    return agent, history, run_dir


def test_phase5_e2e_all_losses():
    """Test 15: full phase-5 run with stats + both probes — completes, saves
    best.pt with the new modules, reloads, and is seed-reproducible."""
    with tempfile.TemporaryDirectory() as t1:
        agent, history, run_dir = _run_phase5_e2e(t1, seed=42)
        best = os.path.join(run_dir, "best.pt")
        assert os.path.exists(best), "best.pt not saved"
        assert len(history["style_loss"]) > 0
        assert len(history["showdown_loss"]) > 0
        assert len(history["val_style_nn_acc"]) > 0

        ckpt = torch.load(best, weights_only=False)
        sd = ckpt["model_state_dict"]
        assert any(k.startswith("style_probe.") for k in sd)
        assert any(k.startswith("showdown_probe.") for k in sd)
        assert any(k.startswith("perception.opp_stats_proj.") for k in sd)

        # Reload into a fresh ASI and run a forward
        fresh = ASI(lambda m: None, config=_ASI_CONFIG)
        fresh.load_checkpoint(best)
        table = OpponentEmbeddingTable(16)
        seq = [_perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0))]
        out = fresh.forward_batch([seq], skip_memory=True,
                                  heads={"opponent_action"},
                                  skip_opponent_emb=False,
                                  opponent_emb_table=table,
                                  collect_opp_states=True)
        assert out["opponent_action_logits"].shape == (1, N_ACTIONS)
        assert out["opp_last_states"].shape == (1, 16)
        best_val_1 = ckpt["val_loss"]

    with tempfile.TemporaryDirectory() as t2:
        _, _, run_dir2 = _run_phase5_e2e(t2, seed=42)
        ckpt2 = torch.load(os.path.join(run_dir2, "best.pt"),
                           weights_only=False)
        assert ckpt2["val_loss"] == best_val_1, (
            f"not seed-reproducible: {ckpt2['val_loss']} != {best_val_1}")
    print(f"test_phase5_e2e_all_losses: OK (best val {best_val_1:.6f})")


def test_old_checkpoint_loads():
    """Test 16: a checkpoint WITHOUT the new modules loads into the new code
    (strict=False), preserving old weights and fresh-initializing the rest."""
    _seed_all(50)
    donor = ASI(lambda m: None, config=_ASI_CONFIG)
    sd = {k: v.clone() for k, v in donor.state_dict().items()
          if not k.startswith(("style_probe.", "showdown_probe.",
                               "perception.opp_stats_proj."))}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "best.pt")
        torch.save({"model_state_dict": sd, "norm_stats": None}, path)
        fresh = ASI(lambda m: None, config=_ASI_CONFIG)
        fresh.load_checkpoint(path)
    # Old weights preserved exactly
    assert torch.equal(next(fresh.opponent_action_head.parameters()),
                       next(donor.opponent_action_head.parameters()))
    # New modules exist and forward runs
    table = OpponentEmbeddingTable(16)
    seq = [_perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0))]
    out = fresh.forward_batch([seq], skip_memory=True,
                              heads={"opponent_action"},
                              skip_opponent_emb=False,
                              opponent_emb_table=table)
    assert out["opponent_action_logits"].shape == (1, N_ACTIONS)
    print("test_old_checkpoint_loads: OK")


def test_phase6_and_eval_paths_unaffected():
    """Test 17: with probes registered, deployment-style forwards (all heads,
    opp table with stats) run and are seed-reproducible; probes stay unused."""
    outs = []
    for _ in range(2):
        _seed_all(60)
        agent = ASI(lambda m: None, config=_ASI_CONFIG)
        agent.eval()
        table = OpponentEmbeddingTable(16)
        seqs = [
            [_perc_event("A", acting_pos=0, bets=(1.0, 2.0, 0.0, 0.0)),
             _perc_event("B", acting_pos=1, action_idx=1,
                         bets=(2.0, 2.0, 0.0, 0.0))],
            [_perc_event("C", acting_pos=0)],
        ]
        with torch.no_grad():
            # phase-6 collection-style: all heads, opponent table active
            out_all = agent.forward_batch(seqs, skip_memory=True, heads=None,
                                          skip_opponent_emb=False,
                                          opponent_emb_table=table,
                                          gru_window=8)
            # eval-style: batched action head, no opponent table
            out_eval = agent.forward_batch(seqs, skip_memory=True,
                                           heads={"action"})
        outs.append((out_all, out_eval))
    a, b = outs
    for key in ("action_logits", "opponent_action_logits", "value",
                "action_embeddings"):
        assert torch.equal(a[0][key], b[0][key]), f"{key} not reproducible"
    assert torch.equal(a[1]["action_logits"], b[1]["action_logits"])
    assert "opp_last_states" not in a[0], "opp states leaked into normal path"
    print("test_phase6_and_eval_paths_unaffected: OK")


if __name__ == "__main__":
    test_pool_binding_consistency()
    test_pool_binding_hand_level()
    test_pool_binding_legacy_fallback()
    test_stats_features_pure()
    test_stats_update_counts()
    test_stats_group_rewind()
    test_stats_injection_offset()
    test_stats_disabled_bitwise_legacy()
    test_stats_state_dict_roundtrip()
    test_build_style_vector()
    test_collect_opp_states_alignment()
    test_style_loss_e2e_learns_styles()
    test_style_target_absent_is_masked()
    test_showdown_strength_labels()
    test_showdown_label_skipped_on_fold_and_collision()
    test_phase5_e2e_all_losses()
    test_old_checkpoint_loads()
    test_phase6_and_eval_paths_unaffected()
    print("\nALL OPPONENT-ADAPTATION TESTS PASSED")
