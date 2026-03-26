import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm


class ActionHead(nn.Module):
    """
    Action generation module with learnable action embeddings.

    Current implementation: mean pool perception output + linear projection (placeholder).

    Future autoregressive generation:
        1. Start with a learned start token as first decoder input
        2. Use Qwen3 layers with perception output as context
        3. At each step:
           a. Decode current sequence -> logits over n_actions
           b. Sample or argmax to get action index
           c. Look up action_embeddings[index], append to decoder input
           d. Repeat for max_gen_steps
        4. Final step's logits = action distribution
    """

    def __init__(self, d_model, n_actions, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len, max_gen_steps):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_gen_steps = max_gen_steps

        self.action_embeddings = nn.Embedding(n_actions, d_model)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.config = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_seq_len,
        )
        self.config._attn_implementation = "eager"

        self.rope = Qwen3RotaryEmbedding(config=self.config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(self.config, layer_idx=i) for i in range(n_layers)]
        )
        self.norm = Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.output_proj = nn.Linear(d_model, n_actions, bias=False)
        self.context_pool = nn.Linear(d_model, d_model, bias=False)

    def forward(self, context):
        """
        Placeholder forward: mean pool context + linear projection.

        Args: context (batch, seq_len, d_model) -- perception output
        Returns: (batch, n_actions) action logits
        """
        pooled = context.mean(dim=1)
        pooled = self.context_pool(pooled)
        return self.output_proj(pooled)

    def generate(self, context):
        """
        STUB: Full autoregressive generation.

        Args: context (batch, seq_len, d_model) -- perception output as context
        Returns: (batch, n_actions) action logits from last generation step

        When implemented:
            batch = context.size(0)
            tgt = self.start_token.expand(batch, -1, -1)

            for step in range(self.max_gen_steps):
                seq_len = tgt.size(1)
                pos_ids = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch, -1)
                pos_emb = self.rope(tgt, pos_ids)

                x = tgt
                for layer in self.layers:
                    x = layer(x, position_ids=pos_ids, position_embeddings=pos_emb)
                x = self.norm(x)

                logits = self.output_proj(x[:, -1:])
                action_idx = logits.argmax(dim=-1)
                next_emb = self.action_embeddings(action_idx)
                tgt = torch.cat([tgt, next_emb], dim=1)

            return self.output_proj(self.norm(x)[:, -1])
        """
        return self.forward(context)
