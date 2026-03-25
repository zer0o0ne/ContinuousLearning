import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """
    Action generation module with learnable action embeddings.

    Current implementation: mean pool perception output + linear projection (placeholder).

    Future autoregressive generation:
        1. Start with a learned start token as first decoder input
        2. Use TransformerDecoder with perception output as cross-attention memory
        3. At each step:
           a. Decode current sequence → logits over n_actions
           b. Sample or argmax to get action index
           c. Look up action_embeddings[index], append to decoder input
           d. Repeat for max_gen_steps
        4. Final step's logits = action distribution
    """

    def __init__(self, d_model, n_actions, n_heads, n_layers, d_ff, dropout, max_gen_steps):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_gen_steps = max_gen_steps

        self.action_embeddings = nn.Embedding(n_actions, d_model)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_actions)
        self.context_pool = nn.Linear(d_model, d_model)

    def forward(self, context):
        """
        Placeholder forward: mean pool context + linear projection.

        Args: context (batch, seq_len, d_model) — perception output
        Returns: (batch, n_actions) action logits
        """
        pooled = context.mean(dim=1)
        pooled = self.context_pool(pooled)
        return self.output_proj(pooled)

    def generate(self, context):
        """
        STUB: Full autoregressive generation.

        Args: context (batch, seq_len, d_model) — perception output as cross-attention memory
        Returns: (batch, n_actions) action logits from last generation step

        When implemented:
            batch = context.size(0)
            tgt = self.start_token.expand(batch, -1, -1)

            for step in range(self.max_gen_steps):
                decoded = self.decoder(tgt, context)
                logits = self.output_proj(decoded[:, -1:])
                action_idx = logits.argmax(dim=-1)
                next_emb = self.action_embeddings(action_idx)
                tgt = torch.cat([tgt, next_emb], dim=1)

            return self.output_proj(decoded[:, -1])
        """
        return self.forward(context)
