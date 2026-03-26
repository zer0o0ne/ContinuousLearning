import torch
import torch.nn as nn
import numpy as np

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.perception.memory import HierarchicalMemory


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
        Args: env_state dict with "now" (current state)
        Returns: (1, d_model)
        """
        now = env_state["now"]
        return self._embed_now(now, device)

    def forward_batch(self, env_states, device="cpu"):
        """Batch-parallel embedding of multiple env_states.

        Args:
            env_states: list of env_state dicts
            device: torch device

        Returns: (B, d_model)
        """
        B = len(env_states)

        # --- Cards: pad to max length, embed, masked mean-pool ---
        card_seqs = []
        card_lengths = []
        for es in env_states:
            now = es["now"]
            table_cards = [int(c) if int(c) >= 0 else 52 for c in now["table"]]
            hand_cards = [int(c) if int(c) >= 0 else 52 for c in now["hand"]]
            seq = table_cards + hand_cards
            card_seqs.append(seq)
            card_lengths.append(len(seq))

        max_len = max(card_lengths)
        padded = torch.full((B, max_len), 52, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.float, device=device)
        for i, seq in enumerate(card_seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1.0

        card_embs = self.card_embed(padded)  # (B, max_len, d_model)
        # Masked mean: zero out padding, sum, divide by actual length
        card_embs = card_embs * mask.unsqueeze(-1)
        card_emb = card_embs.sum(dim=1) / torch.tensor(
            card_lengths, dtype=torch.float, device=device
        ).unsqueeze(-1)  # (B, d_model)

        # --- Positions ---
        positions = torch.tensor(
            [int(es["now"]["pos"]) for es in env_states],
            dtype=torch.long, device=device
        )
        pos_emb = self.pos_embed(positions)  # (B, d_model)

        # --- Scalars (pot, bank) ---
        scalars = torch.tensor(
            [[float(es["now"]["pot"]), float(es["now"]["bank"])] for es in env_states],
            dtype=torch.float, device=device
        )
        scalar_emb = self.scalar_proj(scalars)  # (B, d_model)

        # --- Bets ---
        bets = torch.zeros(B, self.max_players, dtype=torch.float, device=device)
        for i, es in enumerate(env_states):
            raw_bets = es["now"]["bets"]
            if isinstance(raw_bets, np.ndarray):
                raw_bets = raw_bets.tolist()
            for j, b in enumerate(raw_bets):
                if j < self.max_players:
                    bets[i, j] = float(b)
        bet_emb = self.bet_proj(bets)  # (B, d_model)

        # --- Action (placeholder zeros) ---
        action_emb = self.action_proj(
            torch.zeros(B, self.n_actions, dtype=torch.float, device=device)
        )  # (B, d_model)

        # --- Combine ---
        combined = torch.cat([card_emb, pos_emb, scalar_emb, bet_emb, action_emb], dim=-1)
        return self.combine(combined)  # (B, d_model)

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

        n_heads = config["n_heads"]
        n_kv_heads = config.get("n_kv_heads", n_heads // 2)

        self.embedder = StateEmbedder(d_model, n_actions, max_players)
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=config["n_encoder_layers"],
            d_ff=config["d_ff"],
        )
        self.memory = HierarchicalMemory(
            n_levels=mem_cfg["n_levels"],
            max_cluster_size=mem_cfg["max_cluster_size"],
            max_cluster_size_after=mem_cfg["max_cluster_size_after"],
            beam_width=mem_cfg["beam_width"],
            d_model=d_model,
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config["n_decoder_layers"],
            d_ff=config["d_ff"],
            max_seq_len=mem_cfg["beam_width"] + 1 + 64,
        )

    def forward(self, env_state, device="cpu", skip_memory=False):
        """
        Single-sample forward. Kept for inference compatibility.

        Returns: tuple (output, encoded)
            output: (1, seq_len, d_model)
            encoded: (1, d_model)
        """
        embedded = self.embedder(env_state, device=device)
        encoded = self.encoder(embedded)

        if skip_memory:
            decoder_input = encoded.unsqueeze(1)
        else:
            mem_result = self.memory.search(encoded.squeeze(0))
            mem_vectors = mem_result["vectors"].unsqueeze(0).to(encoded.device)
            decoder_input = torch.cat([mem_vectors, encoded.unsqueeze(1)], dim=1)

        output = self.decoder(decoder_input)
        return output, encoded

    def forward_batch(self, env_states, device="cpu", skip_memory=False):
        """
        Batch-parallel forward.

        Args:
            env_states: list of env_state dicts
            device: torch device
            skip_memory: bypass memory retrieval

        Returns: tuple (output, encoded)
            output: (B, seq_len, d_model)
            encoded: (B, d_model)
        """
        embedded = self.embedder.forward_batch(env_states, device=device)  # (B, d_model)
        encoded = self.encoder(embedded)  # (B, d_model)

        if skip_memory:
            decoder_input = encoded.unsqueeze(1)  # (B, 1, d_model)
        else:
            mem_vectors = self.memory.search_batch(encoded)  # (B, beam_width, d_model)
            mem_vectors = mem_vectors.to(encoded.device)
            decoder_input = torch.cat([mem_vectors, encoded.unsqueeze(1)], dim=1)

        output = self.decoder(decoder_input)
        return output, encoded
