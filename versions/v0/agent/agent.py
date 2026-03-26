import torch
import torch.nn as nn
import numpy as np

from agent.perception.perception import Perception
from agent.value.value import ValueHead
from agent.action.action import ActionHead


class ASI(nn.Module):
    def __init__(self, log, config=None):
        super().__init__()
        self.log = log
        self.config = config or {}

        arch = config.get("architecture", config)

        d_model = arch.get("d_model", 128)
        n_actions = arch.get("n_actions", 13)
        max_players = arch.get("max_players", 6)
        n_heads = arch.get("n_heads", 4)
        n_kv_heads = arch.get("n_kv_heads", n_heads // 2)
        mem_cfg = arch.get("memory", {})
        act_cfg = arch.get("action", {})
        max_seq_len = arch.get("max_seq_len", 256)

        self.perception = Perception(arch)
        self.value_head = ValueHead(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config.get("n_value_layers", 2),
            d_ff=config.get("d_ff", 512),
            max_seq_len=max_seq_len + mem_cfg.get("beam_width", 4) + 64,
        )
        self.action_head = ActionHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config.get("n_action_layers", 2),
            d_ff=config.get("d_ff", 512),
            max_seq_len=act_cfg.get("max_gen_steps", 4) + 1,
            max_gen_steps=act_cfg.get("max_gen_steps", 4),
        )

        self.device_ = "cpu"
        self.n_actions = n_actions
        self.optimizer = None
        self.loss_buffer = []

    def forward_batch(self, event_sequences, skip_memory=True):
        """
        Batch-parallel forward pass over event sequences.

        Args:
            event_sequences: list of lists of event dicts
            skip_memory: bypass memory retrieval (True for gto_ev_predict)
        Returns: {"action_logits": (B, n_actions), "value": (B, 1)}
        """
        perception_out, encoded = self.perception.forward_batch(
            event_sequences, device=self.device_, skip_memory=skip_memory
        )
        value = self.value_head(perception_out)
        action_logits = self.action_head(perception_out)

        return {"action_logits": action_logits, "value": value}

    def set_device(self, device):
        self.device_ = device
        self.to(device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.get("lr", 1e-4)
            )
