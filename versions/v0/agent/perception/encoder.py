import torch
import torch.nn as nn
import math


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer("pos_encoding", self._sinusoidal_encoding(max_seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args: x (batch, seq_len, d_model)
        Returns: (batch, d_model)
        """
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x[:, 0])

    @staticmethod
    def _sinusoidal_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
