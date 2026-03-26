import torch.nn as nn


class EncoderBlock(nn.Module):
    """Single encoder block with residual connection and post-norm."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.ff(x) + x)


class Encoder(nn.Module):
    """MLP encoder with per-layer residual connections."""

    def __init__(self, d_model, n_layers, d_ff):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, d_ff) for _ in range(n_layers)]
        )

    def forward(self, x):
        """
        Args: x (batch, d_model)
        Returns: (batch, d_model)
        """
        for block in self.blocks:
            x = block(x)
        return x
