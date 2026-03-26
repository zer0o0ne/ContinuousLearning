import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm


class ActionHead(nn.Module):
    """
    Classifier-based action head.
    Takes a sequence from perception -> Qwen3 self-attention -> mean pool -> action logits.
    """

    def __init__(self, d_model, n_actions, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len):
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
        self.output_proj = nn.Linear(d_model, n_actions, bias=False)

    def forward(self, context):
        """
        Args: context (batch, seq_len, d_model) -- perception output
        Returns: (batch, n_actions) action logits
        """
        batch_size, seq_len, _ = context.shape
        position_ids = torch.arange(seq_len, device=context.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(context, position_ids)

        x = context
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, position_embeddings=position_embeddings)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.output_proj(x)
