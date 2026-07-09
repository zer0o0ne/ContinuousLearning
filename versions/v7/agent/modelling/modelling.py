import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm,
)

from agent.attn_utils import build_causal_padding_mask


def build_lm_pairs(event_sequences):
    """LM-pair construction shared by phase 4 and phase 6
    (PLAN_MODELLING_HEAD_REDESIGN.md §3).

    Event convention (collect.py `_play_hands`): pre-decision snapshot
    (action=None → zeros vector) at p, post-action snapshot (one-hot) at
    p+1, next decision's pre-decision at p+2. For every event index q whose
    `action` one-hot has max ≥ 0.5:
      source = q−1, action = argmax(action_q), target = q+1 (skipped when
      q+1 does not exist — last decision of the sequence has no target).

    Returns four 1-D long tensors (batch_idx, src_positions, actions,
    tgt_positions), all length M (possibly 0).
    """
    batch_idx, src_pos, actions, tgt_pos = [], [], [], []
    for bi, seq in enumerate(event_sequences):
        for q in range(1, len(seq) - 1):
            action = seq[q].get("action")
            if action is None:
                continue
            action_t = torch.as_tensor(action, dtype=torch.float32)
            if action_t.numel() == 0 or action_t.max().item() < 0.5:
                continue
            batch_idx.append(bi)
            src_pos.append(q - 1)
            actions.append(int(action_t.argmax().item()))
            tgt_pos.append(q + 1)
    return (
        torch.tensor(batch_idx, dtype=torch.long),
        torch.tensor(src_pos, dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(tgt_pos, dtype=torch.long),
    )


def lm_loss(pred, target, infonce_weight=0.0, infonce_temperature=0.1):
    """MSE + InfoNCE over in-batch negatives
    (PLAN_MODELLING_HEAD_REDESIGN.md §4).

    pred (M, D); target (M, D) — caller detaches (stop-grad, owner
    decision 4). InfoNCE: cosine-similarity logits pred_i · target_j / τ,
    positives on the diagonal; skipped when M < 2.

    Returns (loss, components) where components = {"mse": float,
    "infonce": float} for logging.
    """
    mse = nn.functional.mse_loss(pred, target)
    components = {"mse": float(mse.detach().item()), "infonce": 0.0}
    loss = mse
    if infonce_weight > 0.0 and pred.shape[0] >= 2:
        pred_n = nn.functional.normalize(pred, dim=-1)
        tgt_n = nn.functional.normalize(target, dim=-1)
        logits = pred_n @ tgt_n.t() / infonce_temperature
        labels = torch.arange(pred.shape[0], device=pred.device)
        nce = nn.functional.cross_entropy(logits, labels)
        components["infonce"] = float(nce.detach().item())
        loss = loss + infonce_weight * nce
    return loss, components


class ModellingHead(nn.Module):
    """
    Autoregressive action-conditioned next-decision-state predictor
    (PLAN_MODELLING_HEAD_REDESIGN.md §2).

    context (B, N, D) = decoder output (perception_out)
      → causal Qwen3 self-attn stack (RoPE position_ids = arange(N),
        causal + padding mask — mirrors perception/decoder.py)   → s (B, N, D)
      → action conditioning: e_a = Embedding(n_actions, D)
        h(t, a) = mlp_out( GELU( mlp_in( cat(s_t, e_a) ) ) )
      → final Qwen3RMSNorm on h

    Causality is inherent: s_t sees only positions ≤ t. The full B×N×A×D
    tensor is never materialized — h is computed only at requested
    (position, action) pairs.
    """

    def __init__(self, d_model, n_actions, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_actions = n_actions
        self.d_model = d_model

        # Learnable action embeddings — conditioning vectors (name kept for
        # state_dict continuity with the old head)
        self.action_embeddings = nn.Embedding(n_actions, d_model)

        self.config = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=d_model // n_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_seq_len,
        )
        self.config._attn_implementation = "sdpa"

        self.rope = Qwen3RotaryEmbedding(config=self.config)
        self.self_attn_layers = nn.ModuleList(
            [Qwen3DecoderLayer(self.config, layer_idx=i) for i in range(n_layers)]
        )

        # Action-conditioning MLP: cat(s_t, e_a) → d_ff → d_model
        self.mlp_in = nn.Linear(2 * d_model, d_ff)
        self.mlp_out = nn.Linear(d_ff, d_model)

        self.norm = Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False

    def _encode(self, context, mask=None):
        """Causal Qwen3 self-attn stack over the context.

        Args:
            context: (batch, seq_len, d_model) — decoder output
            mask: (batch, seq_len) float — 1 for real, 0 for padding
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = context.shape
        position_ids = torch.arange(seq_len, device=context.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(context, position_ids)

        # A.1 convention: explicit combined causal + padding mask. An explicit
        # mask disables Qwen3's built-in is_causal, so causality is encoded here.
        attn_mask = build_causal_padding_mask(mask, seq_len, context.dtype, context.device)

        x = context
        use_ckpt = (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled())
        for layer in self.self_attn_layers:
            if use_ckpt:
                layer_out = ckpt.checkpoint(
                    layer, x,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask,
                    use_reentrant=False)
            else:
                layer_out = layer(x, position_ids=position_ids,
                                  position_embeddings=position_embeddings,
                                  attention_mask=attn_mask)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return x

    def _condition(self, s, e):
        """h = norm( mlp_out( GELU( mlp_in( cat(s, e) ) ) ) ).

        Args:
            s: (..., d_model) — self-attn states
            e: (..., d_model) — action embeddings
        Returns: (..., d_model)
        """
        h = self.mlp_in(torch.cat([s, e], dim=-1))
        h = self.mlp_out(self.resid_dropout(nn.functional.gelu(h)))
        return self.norm(h)

    def forward(self, context, mask=None):
        """h at each example's LAST true position, for ALL actions.

        Args:
            context: (batch, seq_len, d_model) — decoder output
            mask: (batch, seq_len) float — 1 for real, 0 for padding.
                None ⇒ full lengths.
        Returns: (batch, n_actions, d_model) — action embedding vectors
        """
        batch_size, seq_len, _ = context.shape
        s = self._encode(context, mask=mask)

        if mask is not None:
            last_pos = mask.sum(dim=1).long().clamp(min=1) - 1
        else:
            last_pos = torch.full((batch_size,), seq_len - 1,
                                  dtype=torch.long, device=context.device)
        s_last = s[torch.arange(batch_size, device=context.device), last_pos]  # (B, D)

        e = self.action_embeddings.weight.to(s_last.dtype)  # (A, D)
        s_exp = s_last.unsqueeze(1).expand(batch_size, self.n_actions, self.d_model)
        e_exp = e.unsqueeze(0).expand(batch_size, self.n_actions, self.d_model)
        return self._condition(s_exp, e_exp)

    def forward_positions(self, context, mask, batch_idx, positions, actions):
        """Training mode: h at requested (batch, position, action) triples.

        Computes the self-attn stack ONCE over the full sequence, then gathers
        s[batch_idx, positions] before the action MLP.

        Args:
            context: (batch, seq_len, d_model) — decoder output
            mask: (batch, seq_len) float — 1 for real, 0 for padding
            batch_idx: (M,) long — sample index per pair
            positions: (M,) long — source position per pair
            actions: (M,) long — action index per pair
        Returns: (M, d_model)
        """
        device = context.device
        batch_idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)

        s = self._encode(context, mask=mask)
        s_g = s[batch_idx, positions]  # (M, D)
        e = self.action_embeddings(actions).to(s_g.dtype)  # (M, D)
        return self._condition(s_g, e)
