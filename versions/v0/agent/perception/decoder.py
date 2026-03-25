import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    """
    Self-attention decoder over concatenated memory + encoder vectors.
    Input and output have the same sequence length.
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pos_encoding", self._sinusoidal_encoding(max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence):
        """
        Args: sequence (batch, seq_len, d_model) — concat of [memory_vectors, encoder_vector]
        Returns: (batch, seq_len, d_model)
        """
        x = sequence + self.pos_encoding[:, :sequence.size(1), :]
        x = self.layers(x)
        return self.norm(x)

    @staticmethod
    def _sinusoidal_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
