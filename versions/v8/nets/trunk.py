"""The hand encoder shared by both networks (CONCEPT.md §5.1, §5.2, §16 OI-4).

Tokeniser → RoPE → Qwen3 decoder layers → final norm. One causal pass over each
hand independently: within a hand attention is causal, across hands it is zero,
so hands are the batch dimension and not a sequence (§5.2).

OI-4 settled that the opponent-embedding network and the agent share this as
**code** and not as weights. They read the same §5.1 situation token and run the
same trunk over it, so a change to the observation or to the attention structure
lands in both at once; but they optimise different objectives on different
retraining cadences, so each owns its own parameters. Two copies of this file
would be exactly the silent divergence `CLAUDE.md` §5 warns about — and the
divergence would be invisible, because both copies would keep training happily.
"""

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm,
)

from attn_utils import build_causal_padding_mask
from nets.tokeniser import SituationTokeniser


class HandEncoder(nn.Module):
    """(B, T, d_model) hidden states for a batch of tokenised hands."""

    def __init__(self, cfg, n_actions, max_players):
        super().__init__()
        d_model = cfg["d_model"]
        d_emb = cfg["d_emb"]
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg.get("n_kv_heads", max(1, n_heads // 2))
        n_layers = cfg["n_layers"]
        d_ff = cfg["d_ff"]
        max_decisions = cfg.get("max_decisions", 64)

        self.d_model = d_model
        self.d_emb = d_emb
        self.n_actions = n_actions
        self.max_players = max_players

        self.tokeniser = SituationTokeniser(
            d_model=d_model, d_emb=d_emb, n_actions=n_actions,
            max_players=max_players,
            d_card=cfg.get("d_card", 32), d_index=cfg.get("d_index", 32),
            max_decisions=max_decisions,
        )

        qwen = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=d_model // n_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_decisions,
        )
        qwen._attn_implementation = "sdpa"
        self.qwen_config = qwen
        self.rope = Qwen3RotaryEmbedding(config=qwen)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(qwen, layer_idx=i) for i in range(n_layers)])
        self.norm = Qwen3RMSNorm(d_model, eps=qwen.rms_norm_eps)

    def forward(self, batch, emb):
        """(B, T, d_model) — one causal pass over each hand independently."""
        x = self.tokeniser(batch, emb)
        B, T, _ = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rope(x, position_ids)
        attn_mask = build_causal_padding_mask(batch["mask"], T, x.dtype, x.device)

        for layer in self.layers:
            out = layer(x, position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        attention_mask=attn_mask)
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)
