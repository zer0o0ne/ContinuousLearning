import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm, repeat_kv,
    apply_rotary_pos_emb,
)


class Qwen3CrossAttention(nn.Module):
    """
    Cross-attention built from Qwen3 components: GQA, QK-norm, same projections.
    Q from one source, K/V from another. No RoPE (action queries are orderless,
    context keys already have positional info from the decoder).
    """

    def __init__(self, config: Qwen3Config, dropout=0.0):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None, q_position_embeddings=None, kv_position_embeddings=None):
        """
        Args:
            query:     (B, Nq, d_model) — action embeddings
            key_value: (B, Nkv, d_model) — decoder output (context)
            mask:      (B, Nkv) float — 1 for real, 0 for padding
            q_position_embeddings: (cos, sin) tuple for RoPE on queries
            kv_position_embeddings: (cos, sin) tuple for RoPE on keys
        Returns: (B, Nq, d_model)
        """
        B, Nq, _ = query.shape
        Nkv = key_value.shape[1]

        q = self.q_norm(self.q_proj(query).view(B, Nq, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(key_value).view(B, Nkv, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(key_value).view(B, Nkv, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on queries — gives each action a distinct positional signature
        if q_position_embeddings is not None:
            cos, sin = q_position_embeddings
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)

        # RoPE on keys — positional encoding for context tokens
        if kv_position_embeddings is not None:
            cos, sin = kv_position_embeddings
            _, k = apply_rotary_pos_emb(k, k, cos, sin)

        # GQA: repeat K, V heads to match Q heads
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        if mask is not None:
            # (B, 1, 1, Nkv) — broadcast over heads and query positions
            attn_mask = (1.0 - mask[:, None, None, :]) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # (B, n_heads, Nq, head_dim)

        out = out.transpose(1, 2).contiguous().reshape(B, Nq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class ModellingHead(nn.Module):
    """
    Modelling head with learnable action embeddings and cross-attention.
    Action embeddings (queries) attend to decoder output (keys/values) via Qwen3-style
    cross-attention (GQA + QK-norm), then refine via Qwen3 self-attention + FFN.
    Output: (batch, n_actions, d_model) — one embedding vector per action.
    """

    def __init__(self, d_model, n_actions, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.d_model = d_model

        # Learnable action embeddings — queries for cross-attention
        self.action_embeddings = nn.Embedding(n_actions, d_model)

        self.config = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_seq_len,
        )
        self.config._attn_implementation = "sdpa"

        # Each block: pre-norm cross-attention + Qwen3 self-attention with FFN
        self.cross_norms = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()

        for i in range(n_layers):
            self.cross_norms.append(Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps))
            self.cross_attns.append(Qwen3CrossAttention(self.config, dropout=dropout))
            self.self_attn_layers.append(Qwen3DecoderLayer(self.config, layer_idx=i))

        self.rope = Qwen3RotaryEmbedding(config=self.config)
        self.norm = Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False

    def forward(self, context, mask=None):
        """
        Args:
            context: (batch, seq_len, d_model) — decoder output
            mask: (batch, seq_len) float — 1 for real, 0 for padding
        Returns: (batch, n_actions, d_model) — action embedding vectors
        """
        batch_size, seq_len = context.shape[0], context.shape[1]

        # Initialize action queries from learnable embeddings
        action_ids = torch.arange(self.n_actions, device=context.device)
        x = self.action_embeddings(action_ids).unsqueeze(0).expand(batch_size, -1, -1)

        # RoPE for action queries (self-attention + cross-attention Q)
        position_ids = torch.arange(self.n_actions, device=context.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(x, position_ids)

        # RoPE for context keys (cross-attention K)
        kv_position_ids = torch.arange(seq_len, device=context.device).unsqueeze(0).expand(batch_size, -1)
        kv_position_embeddings = self.rope(context, kv_position_ids)

        use_ckpt = (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled())

        def _block(x, cross_norm, cross_attn, self_attn, context, mask):
            residual = x
            x = residual + self.resid_dropout(cross_attn(
                cross_norm(x), context, mask=mask,
                q_position_embeddings=position_embeddings,
                kv_position_embeddings=kv_position_embeddings))
            layer_out = self_attn(x, position_ids=position_ids,
                                  position_embeddings=position_embeddings)
            return layer_out[0] if isinstance(layer_out, tuple) else layer_out

        for cross_norm, cross_attn, self_attn in zip(
            self.cross_norms, self.cross_attns, self.self_attn_layers
        ):
            if use_ckpt:
                x = ckpt.checkpoint(
                    _block, x, cross_norm, cross_attn, self_attn,
                    context, mask, use_reentrant=False)
            else:
                x = _block(x, cross_norm, cross_attn, self_attn, context, mask)

        return self.norm(x)
