import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm

from agent.attn_utils import build_causal_padding_mask


class Decoder(nn.Module):
    """
    Self-attention decoder over concatenated memory + encoder vectors.
    Input and output have the same sequence length.
    """

    def __init__(self, d_model, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.config = Qwen3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=d_model // n_heads,
            intermediate_size=d_ff,
            num_hidden_layers=n_layers,
            max_position_embeddings=max_seq_len,
        )
        self.config._attn_implementation = "sdpa"

        self.rope = Qwen3RotaryEmbedding(config=self.config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(self.config, layer_idx=i) for i in range(n_layers)]
        )
        self.norm = Qwen3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, sequence, mask=None):
        """
        Args:
            sequence: (batch, seq_len, d_model) — concat of [memory_vectors, encoder_vector]
            mask: (batch, seq_len) float — 1 for real, 0 for padding (optional)
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = sequence.shape
        position_ids = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(sequence, position_ids)

        # A.1: explicit combined causal + padding mask. An explicit mask
        # disables Qwen3's built-in is_causal, so causality is encoded here.
        attn_mask = build_causal_padding_mask(mask, seq_len, sequence.dtype, sequence.device)

        x = sequence
        use_ckpt = (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled())
        for layer in self.layers:
            if use_ckpt:
                layer_out = ckpt.checkpoint(
                    layer, x,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask,
                    use_reentrant=False)
            else:
                layer_out = layer(x, position_ids=position_ids,
                                  position_embeddings=position_embeddings,
                                  attention_mask=attn_mask)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return self.norm(x)
