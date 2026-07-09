"""Architecture invariant tests.

These tests codify critical design decisions that MUST NOT change. Each test
documents WHY the invariant matters (not just that it exists), so a future
engineer knows the cost of breaking it.

Run from versions/v6/:
    python -m tests.test_architecture_invariants
    # or
    python -m unittest tests.test_architecture_invariants
"""

import sys
import unittest
import inspect

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Small model parameters — fast, still exercises all code paths
# ---------------------------------------------------------------------------
D_MODEL = 32
N_HEADS = 4
N_KV_HEADS = 2          # GQA: fewer kv heads than query heads
N_LAYERS = 1
D_FF = 64
N_ACTIONS = 6
MAX_PLAYERS = 6
PERC_MAX_SEQ = 128       # perception max_seq_len
HEAD_MAX_SEQ = 192       # heads get extra room for modelling embeddings (> PERC_MAX_SEQ)
B = 2
N_EVENTS = 4             # events per sample in forward pass tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(n_actions=N_ACTIONS, max_players=MAX_PLAYERS):
    """Minimal valid event dict with all required fields."""
    return {
        "table": [0, 1, 2, -1, -1],   # 3 real cards, 2 unknowns (-1 → 52)
        "hand":  [3, 4],
        "hero_pos": 0,
        "acting_pos": 1,
        "num_players": 3,
        "pot": 1.0,
        "stack": 10.0,
        "bets": [0.5, 0.5, 0.0],
        "stacks": [10.0, 9.5, 10.0],
        "action": [1.0 / n_actions] * n_actions,
    }


def _make_event_sequences(batch_size=B, n_events=N_EVENTS,
                           n_actions=N_ACTIONS, max_players=MAX_PLAYERS):
    return [[_make_event(n_actions, max_players) for _ in range(n_events)]
            for _ in range(batch_size)]


def _make_perception_config():
    return {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "n_encoder_layers": N_LAYERS,
        "n_decoder_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_seq_len": PERC_MAX_SEQ,
        "max_players": MAX_PLAYERS,
        "memory": {
            "n_levels": 2,
            "max_cluster_size": 4,
            "max_cluster_size_after": 2,
            "beam_width": 4,
        },
        "opponent_embedding": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Invariant 1 — CARDS_PER_EVENT = 7
# ---------------------------------------------------------------------------

class TestCardsPerEvent(unittest.TestCase):
    """CARDS_PER_EVENT = 7 is the fundamental slot layout: 5 table + 2 hand.
    Every downstream dimension (encoder token count, mean-pool window, mask
    stride) is derived from this constant. Changing it breaks all of them."""

    def test_constant_exists_and_equals_7(self):
        from agent.perception.perception import EventSequenceEmbedder
        self.assertTrue(
            hasattr(EventSequenceEmbedder, "CARDS_PER_EVENT"),
            "EventSequenceEmbedder must define the class attribute CARDS_PER_EVENT",
        )
        self.assertEqual(
            EventSequenceEmbedder.CARDS_PER_EVENT, 7,
            "CARDS_PER_EVENT must equal 7 (5 table + 2 hand)",
        )

    def test_embed_event_returns_7_vectors(self):
        """embed_event must return shape (7, d_model) — one vector per card slot."""
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        embedder.eval()
        event = _make_event()
        with torch.no_grad():
            out = embedder.embed_event(event)
        self.assertEqual(
            out.shape,
            (7, D_MODEL),
            f"embed_event must return (7, d_model); got {tuple(out.shape)}",
        )

    def test_forward_batch_token_count_multiple_of_7(self):
        """forward_batch must produce N*7 tokens (S divisible by 7)."""
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        embedder.eval()
        seqs = _make_event_sequences()
        with torch.no_grad():
            embeddings, mask = embedder.forward_batch(seqs)
        S = embeddings.shape[1]
        self.assertEqual(
            S % 7, 0,
            f"Embedding sequence length {S} must be divisible by 7",
        )


# ---------------------------------------------------------------------------
# Invariant 2 — Mean pool window = 7
# ---------------------------------------------------------------------------

class TestMeanPoolWindow7(unittest.TestCase):
    """After the encoder, a non-overlapping mean pool with window=7 collapses
    the N*7 per-card tokens into N per-event vectors. The mask is strided by
    the same factor (mask[:, ::7]). This is load-bearing: the modelling head
    appends per-event embeddings into this reduced space, and the MCTS context
    extension relies on event-level granularity."""

    def _get_perception_output(self):
        from agent.perception.perception import Perception
        torch.manual_seed(42)
        perc = Perception(_make_perception_config(), N_ACTIONS)
        perc.eval()
        seqs = _make_event_sequences()
        with torch.no_grad():
            output, encoded, mask = perc.forward_batch(seqs, skip_memory=True)
        return output, encoded, mask, len(seqs[0])

    def test_encoded_shape_collapses_to_n_events(self):
        """encoded tensor returned from perception must be (B, N_events, d_model)."""
        _, encoded, _, n_events = self._get_perception_output()
        self.assertEqual(
            tuple(encoded.shape),
            (B, n_events, D_MODEL),
            f"encoded must be (B, N_events, d_model); got {tuple(encoded.shape)}",
        )

    def test_mask_stride_equals_7(self):
        """Mask after mean pool must have shape (B, N_events) — one entry per event.
        This is computed as mask[:, ::7] from the per-card mask."""
        _, _, mask, n_events = self._get_perception_output()
        self.assertEqual(
            tuple(mask.shape),
            (B, n_events),
            f"mask must be (B, N_events) after stride-7; got {tuple(mask.shape)}",
        )

    def test_mean_pool_is_correct_operation(self):
        """Manually verify the mean-pool formula view(B, S//7, 7, D).mean(dim=2)
        against the encoder output before pooling."""
        from agent.perception.perception import EventSequenceEmbedder, Perception
        from agent.perception.encoder import Encoder

        torch.manual_seed(7)
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        encoder = Encoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, PERC_MAX_SEQ)
        embedder.eval(); encoder.eval()

        seqs = _make_event_sequences(batch_size=1, n_events=3)
        with torch.no_grad():
            embedded, mask = embedder.forward_batch(seqs)
            enc_out = encoder(embedded, mask=mask)   # (1, 3*7, D)
            B_, S, D = enc_out.shape
            C = 7
            manual_pool = enc_out.view(B_, S // C, C, D).mean(dim=2)  # (1, 3, D)
        self.assertEqual(tuple(manual_pool.shape), (1, 3, D_MODEL))
        # Result is finite (no NaN/Inf)
        self.assertTrue(torch.isfinite(manual_pool).all())


# ---------------------------------------------------------------------------
# Invariant 3 — Embedder combine input = 8 * d_model
# ---------------------------------------------------------------------------

class TestEmbedderCombineInput(unittest.TestCase):
    """The combine Linear maps 8*d_model → d_model: one card embedding + 7
    context embeddings (hero_pos, acting_pos, num_players, scalar, bets,
    action, stacks). Adding or removing a context component requires changing
    this projection — breaking that silently would produce wrong shapes."""

    def test_combine_weight_shape(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        w = embedder.combine.weight
        # Linear weight is (out_features, in_features)
        self.assertEqual(
            tuple(w.shape),
            (D_MODEL, 8 * D_MODEL),
            f"combine weight must be (d_model, 8*d_model); got {tuple(w.shape)}",
        )

    def test_combine_in_features(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.combine.in_features,
            8 * D_MODEL,
            "combine.in_features must be 8 * d_model",
        )

    def test_combine_out_features(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.combine.out_features,
            D_MODEL,
            "combine.out_features must equal d_model",
        )

    def test_context_embedding_count(self):
        """There are exactly 7 context embedding projections beyond card_embed."""
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        context_projections = [
            embedder.hero_pos_embed,
            embedder.acting_pos_embed,
            embedder.num_players_embed,
            embedder.scalar_proj,
            embedder.bet_proj,
            embedder.action_proj,
            embedder.stacks_proj,
        ]
        self.assertEqual(len(context_projections), 7)


# ---------------------------------------------------------------------------
# Invariant 4 — Card embedding size = 53
# ---------------------------------------------------------------------------

class TestCardEmbeddingSize(unittest.TestCase):
    """card_embed has 53 entries: indices 0-51 are the 52 playing cards, index
    52 is the no-card sentinel used for unknown/absent cards. Any change to
    this breaks saved checkpoints and data that uses -1 → 52 mapping."""

    def test_card_embed_num_embeddings(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.card_embed.num_embeddings,
            53,
            "card_embed must have 53 entries (0-51 = cards, 52 = no-card)",
        )

    def test_card_embed_embedding_dim(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.card_embed.embedding_dim,
            D_MODEL,
            "card_embed embedding_dim must equal d_model",
        )

    def test_no_card_sentinel_index_accessible(self):
        """Index 52 (the no-card sentinel) must be a valid lookup."""
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        embedder.eval()
        with torch.no_grad():
            out = embedder.card_embed(torch.tensor([52]))
        self.assertEqual(out.shape, (1, D_MODEL))

    def test_negative_card_maps_to_52(self):
        """embed_event converts negative card indices to 52 (no-card sentinel)."""
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        embedder.eval()
        # Event with all unknown cards (-1)
        event_unknown = {
            "table": [-1, -1, -1, -1, -1],
            "hand": [-1, -1],
            "hero_pos": 0, "acting_pos": 0, "num_players": 2,
            "pot": 1.0, "stack": 10.0,
            "bets": [0.0, 0.0],
            "stacks": [10.0, 10.0],
            "action": [1.0 / N_ACTIONS] * N_ACTIONS,
        }
        # Should not raise; -1 cards are mapped to 52
        with torch.no_grad():
            out = embedder.embed_event(event_unknown)
        self.assertEqual(out.shape, (7, D_MODEL))


# ---------------------------------------------------------------------------
# Invariant 5 — Source embedding size = 2
# ---------------------------------------------------------------------------

class TestSourceEmbeddingSize(unittest.TestCase):
    """source_embed distinguishes table cards (0) from hand cards (1). There
    are exactly 2 categories. This encodes the observability boundary: hand
    cards are hero-private, table cards are public."""

    def test_source_embed_num_embeddings(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.source_embed.num_embeddings,
            2,
            "source_embed must have exactly 2 entries (0=table, 1=hand)",
        )

    def test_source_embed_embedding_dim(self):
        from agent.perception.perception import EventSequenceEmbedder
        embedder = EventSequenceEmbedder(D_MODEL, N_ACTIONS, MAX_PLAYERS)
        self.assertEqual(
            embedder.source_embed.embedding_dim,
            D_MODEL,
        )

    def test_table_hand_ids_are_0_and_1(self):
        """The source_ids used in embed_event must be [0,0,0,0,0,1,1]."""
        from agent.perception.perception import EventSequenceEmbedder
        # Inspect the source code to verify the literal ids used
        src = inspect.getsource(EventSequenceEmbedder.embed_event)
        self.assertIn("[0, 0, 0, 0, 0, 1, 1]", src,
                      "embed_event must use source_ids [0,0,0,0,0,1,1]")


# ---------------------------------------------------------------------------
# Invariant 6 — Causal attention in encoder/decoder/value/action/opponent_action
# ---------------------------------------------------------------------------

class TestCausalAttention(unittest.TestCase):
    """Encoder, decoder, value, action, and opponent_action heads use causal
    attention via build_causal_padding_mask. This is essential: without
    causality the model can cheat by reading future events during encoding,
    and the modelling-head reconstruction target (which relies on causal
    state representations) becomes meaningless."""

    def _assert_causal(self, module, name, seq_len=10, d=D_MODEL, b=2):
        module.eval()
        torch.manual_seed(100)
        x = torch.randn(b, seq_len, d)
        mask = torch.ones(b, seq_len)
        t = seq_len // 2  # perturb this position

        # Capture per-position outputs via the final RMSNorm hook
        captured = {}
        handle = module.norm.register_forward_hook(
            lambda m, inp, out: captured.__setitem__("out", out.detach().clone())
        )
        try:
            with torch.no_grad():
                module(x, mask=mask)
                states_a = captured["out"].clone()

                x2 = x.clone()
                x2[:, t] += 5.0
                module(x2, mask=mask)
                states_b = captured["out"].clone()
        finally:
            handle.remove()

        # Positions strictly before t must be unchanged
        diff_before = (states_a[:, :t] - states_b[:, :t]).abs().max().item()
        self.assertLess(
            diff_before, 1e-4,
            f"{name}: positions < {t} changed when token {t} was perturbed "
            f"(max diff={diff_before:.2e}) — attention is not causal",
        )
        # Position t itself must change (sanity check the test is live)
        diff_at = (states_a[:, t] - states_b[:, t]).abs().max().item()
        self.assertGreater(
            diff_at, 1e-4,
            f"{name}: perturbation at position {t} had no effect — "
            f"test is not exercising real computation",
        )

    def test_encoder_is_causal(self):
        from agent.perception.encoder import Encoder
        enc = Encoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, PERC_MAX_SEQ)
        self._assert_causal(enc, "Encoder")

    def test_decoder_is_causal(self):
        from agent.perception.decoder import Decoder
        dec = Decoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, PERC_MAX_SEQ)
        self._assert_causal(dec, "Decoder")

    def test_value_head_is_causal(self):
        from agent.value.value import ValueHead
        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self._assert_causal(vh, "ValueHead")

    def test_action_head_is_causal(self):
        from agent.action.action import ActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self._assert_causal(ah, "ActionHead")

    def test_opponent_action_head_is_causal(self):
        from agent.opponent_action.opponent_action import OpponentActionHead
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self._assert_causal(oah, "OpponentActionHead")

    def test_encoder_uses_build_causal_padding_mask(self):
        """Encoder.forward must call build_causal_padding_mask."""
        from agent.perception.encoder import Encoder
        src = inspect.getsource(Encoder.forward)
        self.assertIn(
            "build_causal_padding_mask", src,
            "Encoder.forward must call build_causal_padding_mask",
        )

    def test_decoder_uses_build_causal_padding_mask(self):
        from agent.perception.decoder import Decoder
        src = inspect.getsource(Decoder.forward)
        self.assertIn("build_causal_padding_mask", src)

    def test_value_head_uses_build_causal_padding_mask(self):
        from agent.value.value import ValueHead
        src = inspect.getsource(ValueHead.forward)
        self.assertIn("build_causal_padding_mask", src)

    def test_action_head_uses_build_causal_padding_mask(self):
        from agent.action.action import ActionHead
        src = inspect.getsource(ActionHead.forward)
        self.assertIn("build_causal_padding_mask", src)

    def test_opponent_action_head_uses_build_causal_padding_mask(self):
        from agent.opponent_action.opponent_action import OpponentActionHead
        src = inspect.getsource(OpponentActionHead.forward)
        self.assertIn("build_causal_padding_mask", src)


# ---------------------------------------------------------------------------
# Invariant 7 — Causal attention in ModellingHead self-attn stack
# ---------------------------------------------------------------------------

class TestModellingHeadCausalSelfAttn(unittest.TestCase):
    """The redesigned ModellingHead (PLAN_MODELLING_HEAD_REDESIGN.md §2) runs a
    CAUSAL Qwen3 self-attn stack over the decoder context: s_t must see only
    positions <= t, because h(t, a) is trained to predict the NEXT decision
    state from the state at t — access to future tokens would let the head
    cheat during teacher-forced LM training. Causality comes from the shared
    build_causal_padding_mask (an explicit mask disables Qwen3's is_causal)."""

    def _make_head(self):
        from agent.modelling.modelling import ModellingHead
        return ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                             N_LAYERS, D_FF, HEAD_MAX_SEQ)

    def test_encode_uses_build_causal_padding_mask(self):
        """ModellingHead._encode must build its mask via build_causal_padding_mask
        (the single source of truth for causality + padding)."""
        from agent.modelling.modelling import ModellingHead
        src = inspect.getsource(ModellingHead._encode)
        self.assertIn(
            "build_causal_padding_mask", src,
            "ModellingHead._encode must call build_causal_padding_mask",
        )

    def test_self_attn_stack_is_causal(self):
        """Perturbing context tokens strictly AFTER position t must leave the
        head's output at t exactly unchanged; perturbing token t itself must
        change it (sanity check that the test is live)."""
        mh = self._make_head()
        mh.eval()
        torch.manual_seed(3)
        seq_len = 8
        t = 4
        context = torch.randn(1, seq_len, D_MODEL)
        mask = torch.ones(1, seq_len)
        batch_idx = torch.zeros(N_ACTIONS, dtype=torch.long)
        positions = torch.full((N_ACTIONS,), t, dtype=torch.long)
        actions = torch.arange(N_ACTIONS)

        with torch.no_grad():
            base = mh.forward_positions(context, mask, batch_idx, positions, actions)

            future = context.clone()
            future[0, t + 1:] += 5.0  # only tokens AFTER t
            out_future = mh.forward_positions(future, mask, batch_idx,
                                              positions, actions)

            at_t = context.clone()
            at_t[0, t] += 5.0
            out_at_t = mh.forward_positions(at_t, mask, batch_idx,
                                            positions, actions)

        self.assertTrue(
            torch.equal(base, out_future),
            "Output at position t changed when tokens after t were perturbed "
            "— ModellingHead self-attn stack is not causal",
        )
        self.assertFalse(
            torch.allclose(base, out_at_t, atol=1e-5),
            "Perturbing token t had no effect at t — test is not exercising "
            "real computation",
        )

    def test_perturbing_one_action_does_not_affect_another(self):
        """Action conditioning is a per-action MLP on cat(s_t, e_a): perturbing
        action b's embedding must leave action a's output EXACTLY unchanged
        (the old symmetric cross-action attention is gone), while changing
        action b's own output."""
        mh = self._make_head()
        mh.eval()
        torch.manual_seed(3)
        context = torch.randn(1, 6, D_MODEL)
        mask = torch.ones(1, 6)

        a, b = 0, N_ACTIONS - 1

        with torch.no_grad():
            base = mh(context, mask=mask)
            w_orig = mh.action_embeddings.weight.data.clone()

            mh.action_embeddings.weight.data[b] += 4.0
            out_b = mh(context, mask=mask)
            mh.action_embeddings.weight.data.copy_(w_orig)

        effect_a_from_b = (out_b[0, a] - base[0, a]).abs().max().item()
        effect_b_from_b = (out_b[0, b] - base[0, b]).abs().max().item()

        self.assertEqual(
            effect_a_from_b, 0.0,
            f"Perturbing action {b} must NOT affect action {a} "
            f"(got {effect_a_from_b}) — conditioning must be per-action",
        )
        self.assertGreater(
            effect_b_from_b, 1e-4,
            f"Perturbing action {b} must affect its own output "
            f"(got {effect_b_from_b})",
        )


# ---------------------------------------------------------------------------
# Invariant 8 — No cross-attention in ModellingHead
# ---------------------------------------------------------------------------

class TestModellingHeadNoCrossAttention(unittest.TestCase):
    """The cross-attention path (Qwen3CrossAttention + cross_norms/cross_attns
    with learnable action queries) was REMOVED by the redesign
    (PLAN_MODELLING_HEAD_REDESIGN.md §2): action conditioning is now a pure MLP
    h(t, a) = mlp_out(GELU(mlp_in(cat(s_t, e_a)))). Reintroducing
    cross-attention would resurrect the collapse dynamics proven in
    analytics/modelling_head_collapse_analysis.pdf."""

    def test_no_cross_attention_class_in_module(self):
        """Qwen3CrossAttention must no longer exist in the modelling module."""
        import agent.modelling.modelling as modelling_module
        self.assertFalse(
            hasattr(modelling_module, "Qwen3CrossAttention"),
            "Qwen3CrossAttention must be removed from agent.modelling.modelling",
        )

    def test_head_has_no_cross_attn_submodules(self):
        """ModellingHead must have no cross_attns/cross_norms and no
        cross-attention keys in its state_dict."""
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertFalse(hasattr(mh, "cross_attns"),
                         "ModellingHead must not have cross_attns")
        self.assertFalse(hasattr(mh, "cross_norms"),
                         "ModellingHead must not have cross_norms")
        cross_keys = [k for k in mh.state_dict() if "cross" in k]
        self.assertEqual(
            cross_keys, [],
            f"ModellingHead state_dict must have no cross-attention keys: {cross_keys}",
        )

    def test_action_conditioning_mlp_shapes(self):
        """The replacement conditioning MLP maps cat(s_t, e_a): 2*d_model →
        d_ff → d_model."""
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(mh.mlp_in.in_features, 2 * D_MODEL,
                         "mlp_in.in_features must be 2 * d_model (cat(s, e))")
        self.assertEqual(mh.mlp_in.out_features, D_FF)
        self.assertEqual(mh.mlp_out.in_features, D_FF)
        self.assertEqual(mh.mlp_out.out_features, D_MODEL,
                         "mlp_out must project back to d_model")

    def test_modelling_head_self_attn_uses_rope(self):
        """The causal self-attention stack DOES use RoPE (position_ids =
        arange(N), mirroring perception/decoder.py conventions)."""
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        # ModellingHead has a .rope attribute for the self-attention path
        self.assertTrue(
            hasattr(mh, "rope"),
            "ModellingHead must have a .rope for self-attention layers",
        )


# ---------------------------------------------------------------------------
# Invariant 9 — Masked mean pool in heads
# ---------------------------------------------------------------------------

class TestMaskedMeanPoolInHeads(unittest.TestCase):
    """ValueHead and ActionHead (and OpponentActionHead) reduce the sequence
    dimension via masked mean pool: sum(x * mask) / clamp(sum(mask), min=1).
    The clamp(min=1) prevents division-by-zero on all-padding sequences.
    This must NOT be replaced by e.g. last-token selection or an unmasked mean."""

    def test_value_head_uses_masked_mean_pool(self):
        from agent.value.value import ValueHead
        src = inspect.getsource(ValueHead.forward)
        # Must have mask-weighted sum divided by clamped count
        self.assertIn("mask.sum(", src,
                      "ValueHead must use masked mean pool")
        self.assertIn("clamp(min=1)", src,
                      "ValueHead must clamp denominator at min=1 to avoid div-by-zero")

    def test_action_head_uses_masked_mean_pool(self):
        from agent.action.action import ActionHead
        src = inspect.getsource(ActionHead.forward)
        self.assertIn("mask.sum(", src)
        self.assertIn("clamp(min=1)", src)

    def test_opponent_action_head_uses_masked_mean_pool(self):
        from agent.opponent_action.opponent_action import OpponentActionHead
        src = inspect.getsource(OpponentActionHead.forward)
        self.assertIn("mask.sum(", src)
        self.assertIn("clamp(min=1)", src)

    def test_clamp_prevents_div_by_zero(self):
        """With an all-zeros mask the denominator clamps to 1, producing finite output."""
        from agent.value.value import ValueHead
        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        vh.eval()
        x = torch.randn(2, 5, D_MODEL)
        all_zero_mask = torch.zeros(2, 5)
        with torch.no_grad():
            out = vh(x, mask=all_zero_mask)
        self.assertTrue(
            torch.isfinite(out).all(),
            "ValueHead must produce finite output even with all-zero mask",
        )

    def test_padding_tokens_do_not_affect_output(self):
        """Changing values at padded positions must not change the pooled output."""
        from agent.action.action import ActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        ah.eval()
        torch.manual_seed(10)
        seq_len = 6
        real_len = 4
        x = torch.randn(1, seq_len, D_MODEL)
        mask = torch.zeros(1, seq_len)
        mask[0, :real_len] = 1.0

        with torch.no_grad():
            out1 = ah(x, mask=mask)
            x2 = x.clone()
            # Perturb only the padded positions
            x2[0, real_len:] += 100.0
            out2 = ah(x2, mask=mask)

        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-4),
            "ActionHead output must not change when padded positions are perturbed",
        )


# ---------------------------------------------------------------------------
# Invariant 10 — ValueHead output shape = (B, 1)
# ---------------------------------------------------------------------------

class TestValueHeadOutputShape(unittest.TestCase):
    """ValueHead must output a single scalar per sample: shape (B, 1). The
    training loss (SmoothL1) and all downstream value usage assume this shape."""

    def test_output_shape(self):
        from agent.value.value import ValueHead
        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        vh.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = vh(x, mask=mask)
        self.assertEqual(
            tuple(out.shape), (B, 1),
            f"ValueHead output must be (B, 1); got {tuple(out.shape)}",
        )

    def test_linear_head_out_features_is_1(self):
        from agent.value.value import ValueHead
        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(
            vh.head.out_features, 1,
            "ValueHead.head must have out_features=1",
        )

    def test_linear_head_attribute_exists(self):
        from agent.value.value import ValueHead
        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertTrue(
            hasattr(vh, "head") and isinstance(vh.head, nn.Linear),
            "ValueHead must have a .head nn.Linear attribute",
        )


# ---------------------------------------------------------------------------
# Invariant 11 — ActionHead output shape = (B, n_actions)
# ---------------------------------------------------------------------------

class TestActionHeadOutputShape(unittest.TestCase):
    """ActionHead must output one logit per action: shape (B, n_actions). The
    softmax / KL loss in GTO probability training and MCTS policy target
    training both require this exact shape."""

    def test_output_shape(self):
        from agent.action.action import ActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        ah.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = ah(x, mask=mask)
        self.assertEqual(
            tuple(out.shape), (B, N_ACTIONS),
            f"ActionHead output must be (B, n_actions); got {tuple(out.shape)}",
        )

    def test_output_proj_out_features(self):
        from agent.action.action import ActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(ah.output_proj.out_features, N_ACTIONS)

    def test_output_proj_attribute_exists(self):
        from agent.action.action import ActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertTrue(
            hasattr(ah, "output_proj") and isinstance(ah.output_proj, nn.Linear),
            "ActionHead must have an .output_proj nn.Linear attribute",
        )


# ---------------------------------------------------------------------------
# Invariant 12 — ModellingHead output shape = (B, n_actions, d_model)
# ---------------------------------------------------------------------------

class TestModellingHeadOutputShape(unittest.TestCase):
    """ModellingHead outputs one d_model embedding vector per action:
    (B, n_actions, d_model). MCTS context extension appends these embeddings to
    the perception sequence; the exact shape is expected by that code path."""

    def test_output_shape(self):
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        mh.eval()
        context = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = mh(context, mask=mask)
        self.assertEqual(
            tuple(out.shape), (B, N_ACTIONS, D_MODEL),
            f"ModellingHead output must be (B, n_actions, d_model); got {tuple(out.shape)}",
        )

    def test_action_embeddings_shape(self):
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(mh.action_embeddings.num_embeddings, N_ACTIONS)
        self.assertEqual(mh.action_embeddings.embedding_dim, D_MODEL)

    def test_n_actions_stored_on_module(self):
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(mh.n_actions, N_ACTIONS)


# ---------------------------------------------------------------------------
# Invariant 13 — OpponentActionHead architecture identical to ActionHead
# ---------------------------------------------------------------------------

class TestOpponentActionHeadMatchesActionHead(unittest.TestCase):
    """OpponentActionHead must mirror ActionHead exactly: same layer types,
    same forward structure, same output shape. Any divergence would invalidate
    the 'identical architecture' invariant noted in CLAUDE.md and complicate
    weight-transfer reasoning."""

    def test_output_shape_matches_action_head(self):
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        ah.eval(); oah.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out_ah = ah(x, mask=mask)
            out_oah = oah(x, mask=mask)
        self.assertEqual(out_ah.shape, out_oah.shape)

    def test_same_layer_count(self):
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(len(ah.layers), len(oah.layers))

    def test_same_output_proj_shape(self):
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(
            tuple(ah.output_proj.weight.shape),
            tuple(oah.output_proj.weight.shape),
        )

    def test_both_have_layers_norm_rope_output_proj(self):
        """Both must have the same set of key attributes."""
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        for attr in ("layers", "norm", "rope", "output_proj"):
            self.assertTrue(hasattr(ah, attr), f"ActionHead missing {attr}")
            self.assertTrue(hasattr(oah, attr), f"OpponentActionHead missing {attr}")

    def test_both_use_causal_mask(self):
        """Both must call build_causal_padding_mask in their forward methods."""
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah_src = inspect.getsource(ActionHead.forward)
        oah_src = inspect.getsource(OpponentActionHead.forward)
        self.assertIn("build_causal_padding_mask", ah_src)
        self.assertIn("build_causal_padding_mask", oah_src)

    def test_parameter_count_matches(self):
        """Both heads must have the same number of trainable parameters."""
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        oah = OpponentActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        ah_params = sum(p.numel() for p in ah.parameters())
        oah_params = sum(p.numel() for p in oah.parameters())
        self.assertEqual(
            ah_params, oah_params,
            f"ActionHead ({ah_params}) and OpponentActionHead ({oah_params}) "
            f"must have identical parameter counts",
        )


# ---------------------------------------------------------------------------
# Invariant 14 — GQA: n_kv_heads < n_heads supported
# ---------------------------------------------------------------------------

class TestGQASupport(unittest.TestCase):
    """All attention modules support Grouped Query Attention (n_kv_heads <
    n_heads). This reduces memory bandwidth at inference and allows KV-cache
    compression. The relationship n_heads % n_kv_heads == 0 must hold so heads
    can be evenly grouped."""

    def _check_gqa(self, cls, name, *extra_args):
        """Instantiate cls with N_KV_HEADS < N_HEADS and run a forward pass."""
        self.assertLess(N_KV_HEADS, N_HEADS,
                        "Test config must have n_kv_heads < n_heads")
        module = cls(*extra_args, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        module.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = module(x, mask=mask)
        return out

    def test_encoder_gqa(self):
        from agent.perception.encoder import Encoder
        out = self._check_gqa(Encoder, "Encoder", D_MODEL)
        self.assertEqual(out.shape, (B, N_EVENTS, D_MODEL))

    def test_decoder_gqa(self):
        from agent.perception.decoder import Decoder
        out = self._check_gqa(Decoder, "Decoder", D_MODEL)
        self.assertEqual(out.shape, (B, N_EVENTS, D_MODEL))

    def test_value_head_gqa(self):
        from agent.value.value import ValueHead
        module = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        module.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = module(x, mask=mask)
        self.assertEqual(out.shape, (B, 1))

    def test_action_head_gqa(self):
        from agent.action.action import ActionHead
        module = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        module.eval()
        x = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = module(x, mask=mask)
        self.assertEqual(out.shape, (B, N_ACTIONS))

    def test_modelling_head_gqa(self):
        from agent.modelling.modelling import ModellingHead
        module = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                               N_LAYERS, D_FF, HEAD_MAX_SEQ)
        module.eval()
        context = torch.randn(B, N_EVENTS, D_MODEL)
        mask = torch.ones(B, N_EVENTS)
        with torch.no_grad():
            out = module(context, mask=mask)
        self.assertEqual(out.shape, (B, N_ACTIONS, D_MODEL))

    def test_kv_heads_config_stored(self):
        """The Qwen3Config inside each module must reflect n_kv_heads."""
        from agent.perception.encoder import Encoder
        enc = Encoder(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(enc.config.num_key_value_heads, N_KV_HEADS)

    def test_modelling_self_attn_gqa(self):
        """The ModellingHead causal self-attn stack must support
        n_kv_heads < n_heads: config reflects it and the k_proj weight is
        GQA-shaped (n_kv_heads * head_dim, d_model) vs the full q_proj."""
        from agent.modelling.modelling import ModellingHead
        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(mh.config.num_key_value_heads, N_KV_HEADS)
        head_dim = D_MODEL // N_HEADS
        attn = mh.self_attn_layers[0].self_attn
        self.assertEqual(
            tuple(attn.q_proj.weight.shape), (D_MODEL, D_MODEL),
            "modelling self-attn q_proj must be (d_model, d_model)",
        )
        self.assertEqual(
            tuple(attn.k_proj.weight.shape),
            (N_KV_HEADS * head_dim, D_MODEL),
            "modelling self-attn k_proj must be GQA-shaped "
            "(n_kv_heads * head_dim, d_model)",
        )


# ---------------------------------------------------------------------------
# Invariant 15 — Head max_seq_len > perception max_seq_len
# ---------------------------------------------------------------------------

class TestHeadMaxSeqLen(unittest.TestCase):
    """Heads receive extra room beyond the perception max_seq_len to accommodate
    modelling embeddings appended during MCTS context extension. Without this
    headroom the RoPE would extrapolate beyond its trained range when the context
    grows during MCTS search."""

    def test_heads_accept_longer_sequences_than_perception(self):
        """Heads must handle input sequences longer than PERC_MAX_SEQ."""
        from agent.value.value import ValueHead
        from agent.action.action import ActionHead
        # Set HEAD_MAX_SEQ > PERC_MAX_SEQ at instantiation time
        self.assertGreater(HEAD_MAX_SEQ, PERC_MAX_SEQ)
        long_len = PERC_MAX_SEQ + 10  # deliberately longer than perception max
        # HEAD_MAX_SEQ must cover this
        self.assertLessEqual(long_len, HEAD_MAX_SEQ)

        vh = ValueHead(D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        ah = ActionHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
        vh.eval(); ah.eval()

        x = torch.randn(B, long_len, D_MODEL)
        mask = torch.ones(B, long_len)
        with torch.no_grad():
            v_out = vh(x, mask=mask)
            a_out = ah(x, mask=mask)

        self.assertEqual(v_out.shape, (B, 1))
        self.assertEqual(a_out.shape, (B, N_ACTIONS))

    def test_head_max_seq_stored_in_config(self):
        """The max_position_embeddings in each head's Qwen3Config must equal
        the max_seq_len passed at construction."""
        from agent.value.value import ValueHead
        from agent.action.action import ActionHead
        from agent.opponent_action.opponent_action import OpponentActionHead
        from agent.modelling.modelling import ModellingHead

        for cls, extra in [
            (ValueHead, []),
            (ActionHead, [N_ACTIONS]),
            (OpponentActionHead, [N_ACTIONS]),
        ]:
            m = cls(D_MODEL, *extra, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, HEAD_MAX_SEQ)
            self.assertEqual(
                m.config.max_position_embeddings, HEAD_MAX_SEQ,
                f"{cls.__name__}.config.max_position_embeddings must equal HEAD_MAX_SEQ",
            )

        mh = ModellingHead(D_MODEL, N_ACTIONS, N_HEADS, N_KV_HEADS,
                           N_LAYERS, D_FF, HEAD_MAX_SEQ)
        self.assertEqual(mh.config.max_position_embeddings, HEAD_MAX_SEQ)

    def test_head_max_seq_greater_than_perception_in_real_config(self):
        """In the real config.json the head max_seq_len must exceed perception's."""
        import json, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.json"
        )
        if not os.path.exists(config_path):
            self.skipTest("config.json not found next to versions/v6/")
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architecture", {})
        perc_max = arch.get("max_seq_len")
        head_max = arch.get("head_max_seq_len")
        if perc_max is None or head_max is None:
            self.skipTest(
                "config.json does not have max_seq_len / head_max_seq_len"
            )
        self.assertGreater(
            head_max, perc_max,
            f"head_max_seq_len ({head_max}) must be > max_seq_len ({perc_max})",
        )


# ---------------------------------------------------------------------------
# Bonus: build_causal_padding_mask contract
# ---------------------------------------------------------------------------

class TestBuildCausalPaddingMask(unittest.TestCase):
    """build_causal_padding_mask is the single source of truth for combining
    causality and padding. Its contract must hold precisely."""

    def setUp(self):
        from agent.attn_utils import build_causal_padding_mask
        self.fn = build_causal_padding_mask

    def test_output_shape_with_padding_mask(self):
        b, s = 3, 8
        pm = torch.ones(b, s)
        out = self.fn(pm, s, torch.float32, torch.device("cpu"))
        self.assertEqual(tuple(out.shape), (b, 1, s, s))

    def test_output_shape_without_padding_mask(self):
        s = 8
        out = self.fn(None, s, torch.float32, torch.device("cpu"))
        self.assertEqual(tuple(out.shape), (1, 1, s, s))

    def test_causal_upper_triangle_is_neg_inf(self):
        """Upper triangle (future positions) must be filled with dtype.min."""
        s = 6
        out = self.fn(None, s, torch.float32, torch.device("cpu"))[0, 0]
        min_val = torch.finfo(torch.float32).min
        for i in range(s):
            for j in range(i + 1, s):
                self.assertEqual(out[i, j].item(), min_val,
                                 f"position ({i},{j}) should be masked")

    def test_causal_lower_triangle_is_zero_without_padding(self):
        """Lower triangle (allowed positions) must be 0.0 when no padding."""
        s = 6
        out = self.fn(None, s, torch.float32, torch.device("cpu"))[0, 0]
        for i in range(s):
            for j in range(0, i + 1):
                self.assertEqual(out[i, j].item(), 0.0,
                                 f"position ({i},{j}) should be allowed")

    def test_padded_keys_are_masked(self):
        """Padding positions (mask=0) must be blocked for all query positions."""
        b, s = 1, 5
        pad_from = 3
        pm = torch.ones(b, s)
        pm[0, pad_from:] = 0.0
        out = self.fn(pm, s, torch.float32, torch.device("cpu"))[0, 0]
        min_val = torch.finfo(torch.float32).min
        for q in range(s):
            for k in range(pad_from, s):
                # Both causal future AND padding key cases get negative min
                # (they add, but here padding positions exceed causal boundary
                # only when q < k, otherwise the causal term is 0).
                # Key check: attending to a padded key from a REAL position
                # that is causally allowed (q >= k) must still be blocked.
                if q >= k:
                    self.assertLess(
                        out[q, k].item(), 0.0,
                        f"padded key at ({q},{k}) should be blocked",
                    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with verbose output for the standalone invocation
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
