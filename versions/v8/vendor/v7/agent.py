"""The perception + action-head subset of v7's `ASI` (CONCEPT.md §4.3).

A v7 pool member needs exactly one thing from v7's network: `P(action | events)`.
That is `perception.forward_batch(..., skip_memory=True, skip_opponent_emb=True)`
followed by `action_head`. The value / modelling / opponent-action heads of the
full `ASI` are never queried, so they are not constructed here — their weights
simply land in `unexpected` when a full v7 checkpoint is loaded (`strict=False`,
exactly as v7's own `ASI.load_checkpoint` does it).

Construction and checkpoint loading follow `versions/v7/agent/agent.py` so that
a checkpoint written by v7 loads with no missing keys under `perception.` and
`action_head.`.
"""

import os

import torch
import torch.nn as nn

from vendor.v7.perception.perception import Perception
from vendor.v7.action.action import ActionHead


def n_actions_from_config(config):
    """Number of discrete actions implied by a v7 config.

    Layout `[fold, call, raise_0 … raise_{bins-1}, all-in]` — the same one v8
    uses (CONCEPT.md §6.1), which is why v7 checkpoints drop into the v8 pool
    without an action remapping.
    """
    game = config.get("game", {})
    raise_sizes = game.get("raise_sizes")
    if raise_sizes:
        n_raise_bins = len(next(iter(raise_sizes.values())))
    else:
        n_raise_bins = game.get("table_bins", 10)
    return n_raise_bins + 3


class V7Agent(nn.Module):
    """v7 perception + action head, frozen, evaluated in inference mode."""

    def __init__(self, config, log=print):
        super().__init__()
        self.log = log
        self.config = config or {}

        arch = self.config.get("architecture", self.config)
        d_model = arch.get("d_model", 128)
        n_heads = arch.get("n_heads", 4)
        n_kv_heads = arch.get("n_kv_heads", n_heads // 2)
        d_ff = arch.get("d_ff", 512)
        max_seq_len = arch.get("max_seq_len", 256)
        mem_cfg = arch.get("memory", {})
        head_max_seq_len = max_seq_len + mem_cfg.get("beam_width", 4) + 64

        n_actions = n_actions_from_config(self.config)

        self.perception = Perception(arch, n_actions)
        self.action_head = ActionHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=arch.get("n_action_layers", 2),
            d_ff=d_ff,
            max_seq_len=head_max_seq_len,
        )

        self.n_actions = n_actions
        self.max_players = arch.get("max_players", 6)
        self.device_ = "cpu"

    @torch.no_grad()
    def action_logits(self, event_sequences):
        """(B, n_actions) logits for a batch of v7 event sequences."""
        perception_out, _encoded, mask = self.perception.forward_batch(
            event_sequences,
            device=self.device_,
            skip_memory=True,
            skip_opponent_emb=True,
        )
        return self.action_head(perception_out, mask=mask)

    def set_device(self, device):
        self.device_ = device
        self.to(device)
        return self

    def load_checkpoint(self, path):
        """Load v7 weights. Copied from `ASI.load_checkpoint` (v7 agent.py)."""
        ckpt_path = path
        if os.path.isdir(path):
            ckpt_path = os.path.join(path, "best.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = self._find_best_checkpoint(path)

        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"v7 checkpoint {path!r} not found. A pool member is only "
                f"meaningful with its trained weights — a randomly initialised "
                f"v7 network is not the strategy the pool is supposed to hold."
            )

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        missing = [k for k in missing]
        if missing:
            modules = sorted(set(k.split(".")[0] for k in missing))
            self.log(
                f"WARNING: {ckpt_path}: {len(missing)} parameters of the "
                f"perception/action subset were NOT in the checkpoint and stay "
                f"randomly initialised (modules: {modules})"
            )
        else:
            self.log(f"Loaded v7 pool member from {ckpt_path}")
        return self

    @staticmethod
    def _find_best_checkpoint(agent_dir):
        """Search v7 scenario subdirectories. Copied from `ASI`."""
        for scenario in ("mcts_predict", "opponent_action_predict",
                         "modelling_predict", "gto_predict",
                         "gto_probs_predict", "gto_ev_predict"):
            scenario_dir = os.path.join(agent_dir, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            subdirs = sorted(
                [d for d in os.listdir(scenario_dir)
                 if os.path.isdir(os.path.join(scenario_dir, d))],
                reverse=True,
            )
            for subdir in subdirs:
                ckpt_path = os.path.join(scenario_dir, subdir, "best.pt")
                if os.path.exists(ckpt_path):
                    return ckpt_path
        return None
