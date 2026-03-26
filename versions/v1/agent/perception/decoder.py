import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm


class Decoder(nn.Module):
    """
    Self-attention decoder over concatenated memory + encoder vectors.
    Input and output have the same sequence length.
    """

    def __init__(self, d_model, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
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

    def forward(self, sequence):
        """
        Args: sequence (batch, seq_len, d_model) — concat of [memory_vectors, encoder_vector]
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = sequence.shape
        position_ids = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(sequence, position_ids)

        x = sequence
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, position_embeddings=position_embeddings)

        return self.norm(x)
