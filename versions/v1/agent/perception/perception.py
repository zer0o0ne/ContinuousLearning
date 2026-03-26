import torch
import torch.nn as nn
import numpy as np

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.perception.memory import HierarchicalMemory


class EventSequenceEmbedder(nn.Module):
    """Embeds a sequence of poker events into (seq_len, d_model) vectors.

    Each event contains: hand cards, board cards, hero_pos, acting_pos,
    num_players, blinds, stack, pot, bets, action.
    """

    def __init__(self, d_model, n_actions, max_players):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_players = max_players

        self.card_embed = nn.Embedding(53, d_model)        # 0-51 = cards, 52 = no-card
        self.hero_pos_embed = nn.Embedding(max_players, d_model)
        self.acting_pos_embed = nn.Embedding(max_players, d_model)
        self.num_players_embed = nn.Embedding(max_players + 1, d_model)  # index by num_players
        self.scalar_proj = nn.Linear(2, d_model)            # pot, stack
        self.blind_proj = nn.Linear(2, d_model)             # small_blind, big_blind
        self.bet_proj = nn.Linear(max_players, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        self.combine = nn.Linear(d_model * 8, d_model)

    def embed_event(self, event, device="cpu"):
        """Embed a single event dict into (d_model,) vector."""
        # Cards: hand + table, map -1 to 52
        table_cards = [int(c) if int(c) >= 0 else 52 for c in event["table"]]
        hand_cards = [int(c) if int(c) >= 0 else 52 for c in event["hand"]]
        all_cards = torch.tensor(table_cards + hand_cards, dtype=torch.long, device=device)
        card_emb = self.card_embed(all_cards).mean(dim=0)  # (d_model,)

        hero_pos_emb = self.hero_pos_embed(
            torch.tensor(int(event["hero_pos"]), device=device)
        )
        acting_pos_emb = self.acting_pos_embed(
            torch.tensor(int(event["acting_pos"]), device=device)
        )
        num_players_emb = self.num_players_embed(
            torch.tensor(int(event["num_players"]), device=device)
        )

        scalars = torch.tensor(
            [float(event["pot"]), float(event["stack"])],
            dtype=torch.float, device=device
        )
        scalar_emb = self.scalar_proj(scalars)

        blinds = torch.tensor(
            [float(event["big_blind"]), float(event.get("small_blind", event["big_blind"] * 0.5))],
            dtype=torch.float, device=device
        )
        blind_emb = self.blind_proj(blinds)

        bets = torch.zeros(self.max_players, dtype=torch.float, device=device)
        raw_bets = event["bets"]
        if isinstance(raw_bets, np.ndarray):
            raw_bets = raw_bets.tolist()
        for i, b in enumerate(raw_bets):
            if i < self.max_players:
                bets[i] = float(b)
        bet_emb = self.bet_proj(bets)

        action = event["action"]
        if isinstance(action, torch.Tensor):
            action_t = action.float().to(device)
        else:
            action_t = torch.tensor(action, dtype=torch.float, device=device)
        action_emb = self.action_proj(action_t)

        combined = torch.cat([
            card_emb, hero_pos_emb, acting_pos_emb, scalar_emb,
            bet_emb, action_emb, num_players_emb, blind_emb
        ])
        return self.combine(combined)  # (d_model,)

    def forward_batch(self, event_sequences, device="cpu"):
        """Embed a batch of event sequences.

        Args:
            event_sequences: list of lists of event dicts (B samples, variable lengths)
            device: torch device

        Returns:
            embeddings: (B, max_seq_len, d_model)
            mask: (B, max_seq_len) float — 1 for real events, 0 for padding
        """
        B = len(event_sequences)
        seq_lengths = [len(seq) for seq in event_sequences]
        max_len = max(seq_lengths)

        # Pre-allocate
        embeddings = torch.zeros(B, max_len, self.d_model, dtype=torch.float, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.float, device=device)

        # Embed each event in each sequence
        for i, seq in enumerate(event_sequences):
            for j, event in enumerate(seq):
                embeddings[i, j] = self.embed_event(event, device=device)
            mask[i, :len(seq)] = 1.0

        return embeddings, mask


class Perception(nn.Module):
    def __init__(self, config, n_actions):
        super().__init__()
        d_model = config["d_model"]
        max_players = config.get("max_players", 6)
        mem_cfg = config["memory"]

        n_heads = config["n_heads"]
        n_kv_heads = config.get("n_kv_heads", n_heads // 2)
        max_seq_len = config.get("max_seq_len", 256)

        self.embedder = EventSequenceEmbedder(d_model, n_actions, max_players)
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config["n_encoder_layers"],
            d_ff=config["d_ff"],
            max_seq_len=max_seq_len,
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
            max_seq_len=max_seq_len + mem_cfg["beam_width"] + 64,
        )

    def forward_batch(self, event_sequences, device="cpu", skip_memory=True):
        """
        Batch-parallel forward over event sequences.

        Args:
            event_sequences: list of lists of event dicts
            device: torch device
            skip_memory: if True, encoder output goes directly to decoder

        Returns: tuple (output, encoded)
            output: (B, seq_len, d_model)
            encoded: (B, seq_len, d_model)
        """
        embedded, mask = self.embedder.forward_batch(event_sequences, device=device)
        encoded = self.encoder(embedded)  # (B, seq_len, d_model)

        if skip_memory:
            decoder_input = encoded
        else:
            # Future: use last token for memory lookup
            last_token = encoded[:, -1, :]  # (B, d_model)
            mem_vectors = self.memory.search_batch(last_token)
            mem_vectors = mem_vectors.to(encoded.device)
            decoder_input = torch.cat([mem_vectors, encoded], dim=1)

        output = self.decoder(decoder_input)
        return output, encoded
