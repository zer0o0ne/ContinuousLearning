import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.config = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_seq_len + 1,
        )
        self.config._attn_implementation = "eager"

        self.rope = Qwen3RotaryEmbedding(config=self.config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(self.config, layer_idx=i) for i in range(n_layers)]
        )
        self.norm = Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps)

    def forward(self, x):
        """
        Args: x (batch, seq_len, d_model)
        Returns: (batch, d_model)
        """
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(x, position_ids)

        for layer in self.layers:
            x = layer(x, position_ids=position_ids, position_embeddings=position_embeddings)

        x = self.norm(x)
        return x[:, 0]
