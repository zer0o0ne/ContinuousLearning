import torch
import torch.nn as nn
import numpy as np

from perception.encoder import Encoder
from perception.decoder import Decoder
from perception.memory import HierarchicalMemory


class StateEmbedder(nn.Module):
    """Converts raw env_state dict into a sequence of token embeddings."""

    def __init__(self, d_model, n_actions, max_players):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_players = max_players

        self.card_embed = nn.Embedding(53, d_model)       # 0-51 = cards, 52 = no-card
        self.pos_embed = nn.Embedding(max_players, d_model)
        self.scalar_proj = nn.Linear(2, d_model)           # pot, bank
        self.bet_proj = nn.Linear(max_players, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        self.combine = nn.Linear(d_model * 5, d_model)     # cards, pos, scalars, bets, action

    def forward(self, env_state, device="cpu"):
        """
        Args: env_state dict with "table_state" (history) and "now" (current state)
        Returns: (1, seq_len, d_model)
        """
        tokens = []

        for step in env_state["table_state"]:
            token = self._embed_step(step, device)
            tokens.append(token)

        now = env_state["now"]
        now_token = self._embed_now(now, device)
        tokens.append(now_token)

        if len(tokens) == 0:
            return torch.zeros(1, 1, self.d_model, device=device)

        return torch.stack(tokens, dim=1)

    def _embed_step(self, step, device):
        """Embed a single history step: {pos, pot, action, table}."""
        table_cards = self._cards_to_tensor(step["table"], device)
        card_emb = self.card_embed(table_cards).mean(dim=0)

        pos_emb = self.pos_embed(torch.tensor(step["pos"], device=device))

        pot_val = step["pot"] if isinstance(step["pot"], (int, float)) else step["pot"]
        scalars = torch.tensor([float(pot_val), 0.0], dtype=torch.float, device=device)
        scalar_emb = self.scalar_proj(scalars)

        bets = torch.zeros(self.max_players, dtype=torch.float, device=device)
        bet_emb = self.bet_proj(bets)

        action = step["action"]
        if isinstance(action, torch.Tensor):
            action_vec = action.float().to(device)
        else:
            action_vec = torch.zeros(self.n_actions, dtype=torch.float, device=device)
        if action_vec.dim() > 1:
            action_vec = action_vec.squeeze(0)
        action_emb = self.action_proj(action_vec)

        combined = torch.cat([card_emb, pos_emb, scalar_emb, bet_emb, action_emb])
        return self.combine(combined).unsqueeze(0)

    def _embed_now(self, now, device):
        """Embed the current state: {pos, pot, bank, hand, table, bets, active_positions}."""
        table_cards = self._cards_to_tensor(now["table"], device)
        hand_cards = self._cards_to_tensor(now["hand"], device)
        all_cards = torch.cat([table_cards, hand_cards])
        card_emb = self.card_embed(all_cards).mean(dim=0)

        pos_emb = self.pos_embed(torch.tensor(int(now["pos"]), device=device))

        scalars = torch.tensor([float(now["pot"]), float(now["bank"])], dtype=torch.float, device=device)
        scalar_emb = self.scalar_proj(scalars)

        bets = torch.zeros(self.max_players, dtype=torch.float, device=device)
        raw_bets = now["bets"]
        if isinstance(raw_bets, np.ndarray):
            raw_bets = raw_bets.tolist()
        for i, b in enumerate(raw_bets):
            if i < self.max_players:
                bets[i] = float(b)
        bet_emb = self.bet_proj(bets)

        action_emb = self.action_proj(torch.zeros(self.n_actions, dtype=torch.float, device=device))

        combined = torch.cat([card_emb, pos_emb, scalar_emb, bet_emb, action_emb])
        return self.combine(combined).unsqueeze(0)

    def _cards_to_tensor(self, cards, device):
        """Convert card list to tensor, mapping -1 to 52 (no-card token)."""
        result = []
        for c in cards:
            c_int = int(c)
            result.append(c_int if c_int >= 0 else 52)
        return torch.tensor(result, dtype=torch.long, device=device)


class Perception(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        n_actions = config["n_actions"]
        max_players = config.get("max_players", 6)
        mem_cfg = config["memory"]

        self.embedder = StateEmbedder(d_model, n_actions, max_players)
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=config["n_heads"],
            n_layers=config["n_encoder_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_seq_len=config["max_seq_len"],
        )
        self.memory = HierarchicalMemory(
            n_levels=mem_cfg["n_levels"],
            cluster_size=mem_cfg["cluster_size"],
            beam_width=mem_cfg["beam_width"],
            n_results=mem_cfg["n_results"],
            d_model=d_model,
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_heads=config["n_heads"],
            n_layers=config["n_decoder_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_seq_len=mem_cfg["n_results"] + 1 + 64,
        )

    def forward(self, env_state, device="cpu"):
        """
        Full perception pipeline.

        Args: env_state dict from dealer
        Returns: (1, n_mem+1, d_model)
        """
        embedded = self.embedder(env_state, device=device)
        encoded = self.encoder(embedded)

        mem_result = self.memory.search(encoded.squeeze(0))
        mem_vectors = mem_result["vectors"].unsqueeze(0).to(encoded.device)

        decoder_input = torch.cat([mem_vectors, encoded.unsqueeze(1)], dim=1)
        output = self.decoder(decoder_input)
        return output

    def store(self, encoded_vector):
        """Store encoder output in memory for future retrieval."""
        self.memory.insert(encoded_vector)
