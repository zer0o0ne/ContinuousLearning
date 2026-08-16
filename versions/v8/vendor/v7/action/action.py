import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm

from vendor.v7.attn_utils import build_causal_padding_mask


class ActionHead(nn.Module):
    """
    Classifier-based action head.
    Takes a sequence from perception -> Qwen3 self-attention -> mean pool -> action logits.
    """

    def __init__(self, d_model, n_actions, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len):
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
        self.output_proj = nn.Linear(d_model, n_actions, bias=False)
        self.gradient_checkpointing = False

    def forward(self, context, mask=None):
        """
        Args:
            context: (batch, seq_len, d_model) -- perception output
            mask: (batch, seq_len) float — 1 for real, 0 for padding (optional)
        Returns: (batch, n_actions) action logits
        """
        batch_size, seq_len, _ = context.shape
        position_ids = torch.arange(seq_len, device=context.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rope(context, position_ids)

        # A.1: explicit combined causal + padding mask. An explicit mask
        # disables Qwen3's built-in is_causal, so causality is encoded here.
        attn_mask = build_causal_padding_mask(mask, seq_len, context.dtype, context.device)

        x = context
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

        x = self.norm(x)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.output_proj(x)
