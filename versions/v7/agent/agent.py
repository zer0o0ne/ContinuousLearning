import torch
import torch.nn as nn
import numpy as np

from agent.perception.perception import Perception
from agent.value.value import ValueHead
from agent.action.action import ActionHead
from agent.modelling.modelling import ModellingHead
from agent.opponent_action.opponent_action import OpponentActionHead


class ASI(nn.Module):
    def __init__(self, log, config=None):
        super().__init__()
        self.log = log
        self.config = config or {}

        arch = config.get("architecture", config)
        game = config.get("game", {})

        d_model = arch.get("d_model", 128)
        n_heads = arch.get("n_heads", 4)
        n_kv_heads = arch.get("n_kv_heads", n_heads // 2)
        d_ff = arch.get("d_ff", 512)
        max_seq_len = arch.get("max_seq_len", 256)
        mem_cfg = arch.get("memory", {})

        raise_sizes = game.get("raise_sizes")
        if raise_sizes:
            n_raise_bins = len(next(iter(raise_sizes.values())))
        else:
            n_raise_bins = game.get("table_bins", 10)
        n_actions = n_raise_bins + 3  # fold + call + raises + all-in

        head_max_seq_len = max_seq_len + mem_cfg.get("beam_width", 4) + 64

        self.perception = Perception(arch, n_actions)
        self.value_head = ValueHead(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=arch.get("n_value_layers", 2),
            d_ff=d_ff,
            max_seq_len=head_max_seq_len,
        )
        self.action_head = ActionHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=arch.get("n_action_layers", 2),
            d_ff=d_ff,
            max_seq_len=head_max_seq_len,
        )
        self.opponent_action_head = OpponentActionHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=arch.get("n_opponent_action_layers", 2),
            d_ff=d_ff,
            max_seq_len=head_max_seq_len,
        )
        self.modelling_head = ModellingHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=arch.get("n_modelling_layers", 4),
            d_ff=d_ff,
            max_seq_len=head_max_seq_len,
            dropout=arch.get("modelling_dropout", 0.1),
            spectral_clamp=arch.get("modelling_spectral_clamp"),
        )

        # Phase-5 training-only probes on the opponent GRU state
        # (PLAN_OPPONENT_ADAPTATION §2/§4). None when disabled — old configs
        # and checkpoints are unaffected (strict=False loading).
        opp_emb_cfg = arch.get("opponent_embedding", {}) or {}
        self.style_probe = None
        self.showdown_probe = None
        if opp_emb_cfg.get("style_probe", False):
            from agent.opponent_action.probes import StyleProbe
            self.style_probe = StyleProbe(d_model)
        if opp_emb_cfg.get("showdown_probe", False):
            from agent.opponent_action.probes import ShowdownStrengthProbe
            self.showdown_probe = ShowdownStrengthProbe(d_model)

        self.device_ = "cpu"
        self.n_actions = n_actions
        self.optimizer = None
        self.loss_buffer = []
        self._checkpoint_norm_stats = None
        self._checkpoint_temperature = None

    def forward_batch(self, event_sequences, skip_memory=True, heads=None,
                      skip_opponent_emb=True, opponent_emb_table=None,
                      gru_window=1, precomputed=None, gru_sample_groups=None,
                      collect_opp_states=False):
        """
        Batch-parallel forward pass over event sequences.

        Args:
            event_sequences: list of lists of event dicts
            skip_memory: bypass memory retrieval (True for gto_ev_predict)
            heads: optional set of head names to compute, e.g. {"action"}.
            skip_opponent_emb: if True, skip opponent GRU embedding injection
            opponent_emb_table: OpponentEmbeddingTable instance
            gru_window: BPTT window length for opponent GRU updates
            precomputed: dict from extract_event_tensors() — skips dict extraction
            gru_sample_groups: optional per-sample group ids for the GRU
                observer-copy rewind (A.4.5, see Perception.forward_batch)
            collect_opp_states: §2 — also return per-sample acting-opponent
                GRU states ("opp_last_states" (B, d), "opp_states_mask" (B,))
                for the phase-5 probes.
        Returns: dict with computed head outputs
        """
        perception_frozen = not any(p.requires_grad for p in self.perception.parameters())
        # NOTE: the phase-5 probes need gradient THROUGH the GRU states even
        # though perception's encoder/decoder are frozen — the opponent_gru's
        # own requires_grad decides. Keep no_grad only when NOTHING in
        # perception (incl. the GRU / stats proj) is trainable.
        if perception_frozen:
            with torch.no_grad():
                p_out = self.perception.forward_batch(
                    event_sequences, device=self.device_, skip_memory=skip_memory,
                    skip_opponent_emb=skip_opponent_emb,
                    opponent_emb_table=opponent_emb_table,
                    gru_window=gru_window, precomputed=precomputed,
                    gru_sample_groups=gru_sample_groups,
                    collect_opp_states=collect_opp_states,
                )
            p_out = list(p_out)
            p_out[0] = p_out[0].detach()
        else:
            p_out = self.perception.forward_batch(
                event_sequences, device=self.device_, skip_memory=skip_memory,
                skip_opponent_emb=skip_opponent_emb,
                opponent_emb_table=opponent_emb_table,
                gru_window=gru_window, precomputed=precomputed,
                gru_sample_groups=gru_sample_groups,
                collect_opp_states=collect_opp_states,
            )
        if collect_opp_states:
            perception_out, encoded, mask, (opp_states, opp_states_mask) = p_out
        else:
            perception_out, encoded, mask = p_out

        result = {}
        if collect_opp_states:
            result["opp_last_states"] = opp_states
            result["opp_states_mask"] = opp_states_mask
        if heads is None or "action" in heads:
            result["action_logits"] = self.action_head(perception_out, mask=mask)
        if heads is None or "opponent_action" in heads:
            result["opponent_action_logits"] = self.opponent_action_head(perception_out, mask=mask)
        if heads is None or "value" in heads:
            result["value"] = self.value_head(perception_out, mask=mask)
        if heads is None or "modelling" in heads:
            result["action_embeddings"] = self.modelling_head(perception_out, mask=mask)

        return result

    def load_checkpoint(self, path):
        """Load model weights from a checkpoint file or directory.

        Supports partial loading: missing parameters stay randomly initialized.
        Logs warnings for missing/unexpected keys grouped by module.

        Args:
            path: path to .pt file, or directory containing best.pt
        """
        import os
        ckpt_path = path
        if os.path.isdir(path):
            ckpt_path = os.path.join(path, "best.pt")
            if not os.path.exists(ckpt_path):
                # Search in scenario subdirectories (gto_predict/<timestamp>/best.pt, etc.)
                ckpt_path = self._find_best_checkpoint(path)

        if ckpt_path is None or not os.path.exists(ckpt_path):
            self.log(f"WARNING: checkpoint {path} not found, agent initialized randomly")
            return

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Preserve norm_stats so downstream training (e.g. opponent_action)
        # can reuse the distribution perception was trained on.
        if ckpt.get("norm_stats") is not None:
            self._checkpoint_norm_stats = ckpt["norm_stats"]
        if ckpt.get("temperature") is not None:
            self._checkpoint_temperature = ckpt["temperature"]

        if missing:
            self.log(f"WARNING: Loaded checkpoint from {ckpt_path}, but {len(missing)} parameters "
                     f"were NOT found and initialized randomly:")
            modules = sorted(set(k.split(".")[0] for k in missing))
            for mod in modules:
                mod_keys = [k for k in missing if k.startswith(mod + ".")]
                self.log(f"  - {mod}: {len(mod_keys)} params")
        if unexpected:
            self.log(f"WARNING: {len(unexpected)} unexpected keys in checkpoint (ignored)")
        if not missing and not unexpected:
            self.log(f"Loaded agent from {ckpt_path} (all parameters matched)")
        elif not missing:
            self.log(f"Loaded agent from {ckpt_path}")

    @staticmethod
    def _find_best_checkpoint(agent_dir):
        """Search scenario subdirectories for the best checkpoint."""
        import os
        for scenario in ("mcts_predict", "opponent_action_predict", "modelling_predict",
                          "gto_predict", "gto_probs_predict", "gto_ev_predict"):
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

    def set_device(self, device):
        self.device_ = device
        self.to(device)

    def set_gradient_checkpointing(self, enabled: bool):
        self.perception.set_gradient_checkpointing(enabled)
        for head in (self.value_head, self.action_head,
                     self.opponent_action_head, self.modelling_head):
            head.gradient_checkpointing = bool(enabled)
