import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Transformer-based value estimator.
    Takes a sequence from perception → transformer self-attention → mean pool → scalar.
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args: x (batch, seq_len, d_model) — perception output sequence
        Returns: (batch, 1)
        """
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
