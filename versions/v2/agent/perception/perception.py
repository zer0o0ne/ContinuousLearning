import torch
import torch.nn as nn
import numpy as np

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.perception.memory import HierarchicalMemory


class EventSequenceEmbedder(nn.Module):
    """Embeds a sequence of poker events into per-card vectors.

    Each event produces 7 vectors (5 table cards + 2 hand cards, fixed order).
    Each vector combines the card embedding with full game context
    (positions, pot, stack, bets, action, num_players).
    A learned source embedding distinguishes table cards from hand cards.
    """

    CARDS_PER_EVENT = 7  # 5 table + 2 hand

    def __init__(self, d_model, n_actions, max_players):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_players = max_players

        self.card_embed = nn.Embedding(53, d_model)        # 0-51 = cards, 52 = no-card
        self.source_embed = nn.Embedding(2, d_model)       # 0 = table card, 1 = hand card
        self.hero_pos_embed = nn.Embedding(max_players, d_model)
        self.acting_pos_embed = nn.Embedding(max_players, d_model)
        self.num_players_embed = nn.Embedding(max_players + 1, d_model)
        self.scalar_proj = nn.Linear(2, d_model)            # pot, stack
        self.bet_proj = nn.Linear(max_players, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        # card_emb + 6 context components = 7 * d_model
        self.combine = nn.Linear(d_model * 7, d_model)
        self.post_embed_norm = nn.LayerNorm(d_model)

    def embed_event(self, event, device="cpu"):
        """Embed a single event dict into (7, d_model) — one vector per card.

        Card order: [table_0, table_1, table_2, table_3, table_4, hand_0, hand_1]
        """
        # 7 card embeddings in fixed order: 5 table + 2 hand
        # Card indices: 0-51 = real cards, -1 (or any negative) → 52 = no-card token
        table_cards = [int(c) if int(c) >= 0 else 52 for c in event["table"]]
        hand_cards = [int(c) if int(c) >= 0 else 52 for c in event["hand"]]
        card_ids = torch.tensor(table_cards + hand_cards, dtype=torch.long, device=device)
        card_ids = card_ids.clamp(0, 52)  # safety: ensure valid embedding indices
        card_embs = self.card_embed(card_ids)  # (7, d_model)

        # Source embedding: 0=table (first 5), 1=hand (last 2)
        source_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.long, device=device)
        source_embs = self.source_embed(source_ids)  # (7, d_model)

        # Context: shared across all 7 cards
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

        # Context: cat 6 embeddings, broadcast to all 7 cards
        context = torch.cat([
            hero_pos_emb, acting_pos_emb, num_players_emb,
            scalar_emb, bet_emb, action_emb
        ])  # (6 * d_model,)
        context = context.unsqueeze(0).expand(7, -1)  # (7, 6 * d_model)

        # Per-card: cat(card_emb, context) → Linear(7d → d) → + source → LayerNorm
        combined = torch.cat([card_embs, context], dim=-1)  # (7, 7 * d_model)
        out = self.combine(combined) + source_embs           # (7, d_model)
        return self.post_embed_norm(out)                     # (7, d_model)

    def forward_batch(self, event_sequences, device="cpu"):
        """Embed a batch of event sequences into per-card vectors.

        Vectorized: collects all events into batch tensors, transfers to device
        once, runs all embedding layers in batch.

        Args:
            event_sequences: list of lists of event dicts (B samples, variable lengths)
            device: torch device

        Returns:
            embeddings: (B, max_events * 7, d_model)
            mask: (B, max_events * 7) float — 1 for real, 0 for padding
        """
        C = self.CARDS_PER_EVENT
        B = len(event_sequences)
        seq_lengths = [len(seq) for seq in event_sequences]
        max_events = max(seq_lengths)

        # --- Collect raw values from all events into Python lists ---
        all_card_ids = []      # each: list of 7 ints
        all_hero_pos = []
        all_acting_pos = []
        all_num_players = []
        all_scalars = []       # each: [pot, stack]
        all_bets = []          # each: list of max_players floats
        all_actions = []       # each: list of n_actions floats
        batch_idx_list = []    # which sample i
        event_idx_list = []    # which event j

        for i, seq in enumerate(event_sequences):
            for j, event in enumerate(seq):
                # Card ids: negative → 52 (no-card)
                table_cards = [int(c) if int(c) >= 0 else 52 for c in event["table"]]
                hand_cards = [int(c) if int(c) >= 0 else 52 for c in event["hand"]]
                cards = table_cards + hand_cards
                cards = [max(0, min(c, 52)) for c in cards]  # clamp safety
                all_card_ids.append(cards)

                all_hero_pos.append(int(event["hero_pos"]))
                all_acting_pos.append(int(event["acting_pos"]))
                all_num_players.append(int(event["num_players"]))
                all_scalars.append([float(event["pot"]), float(event["stack"])])

                # Bets: variable length → pad to max_players
                raw_bets = event["bets"]
                if isinstance(raw_bets, np.ndarray):
                    raw_bets = raw_bets.tolist()
                padded_bets = [0.0] * self.max_players
                for k, b in enumerate(raw_bets):
                    if k < self.max_players:
                        padded_bets[k] = float(b)
                all_bets.append(padded_bets)

                # Action: Tensor or list
                action = event["action"]
                if isinstance(action, torch.Tensor):
                    all_actions.append(action.float().tolist())
                else:
                    all_actions.append([float(a) for a in action])

                batch_idx_list.append(i)
                event_idx_list.append(j)

        T = len(all_card_ids)  # total events across batch

        if T == 0:
            embeddings = torch.zeros(B, max_events * C, self.d_model,
                                     dtype=torch.float, device=device)
            mask = torch.zeros(B, max_events * C, dtype=torch.float, device=device)
            return embeddings, mask

        # --- Build batch tensors on CPU, transfer to device once ---
        card_ids = torch.tensor(all_card_ids, dtype=torch.long, device=device)       # (T, 7)
        hero_pos = torch.tensor(all_hero_pos, dtype=torch.long, device=device)       # (T,)
        acting_pos = torch.tensor(all_acting_pos, dtype=torch.long, device=device)   # (T,)
        num_players = torch.tensor(all_num_players, dtype=torch.long, device=device) # (T,)
        scalars = torch.tensor(all_scalars, dtype=torch.float, device=device)        # (T, 2)
        bets = torch.tensor(all_bets, dtype=torch.float, device=device)              # (T, max_players)
        actions = torch.tensor(all_actions, dtype=torch.float, device=device)        # (T, n_actions)

        # --- Batch embedding lookups + projections ---
        card_embs = self.card_embed(card_ids)                    # (T, 7, d_model)
        hero_pos_emb = self.hero_pos_embed(hero_pos)             # (T, d_model)
        acting_pos_emb = self.acting_pos_embed(acting_pos)       # (T, d_model)
        num_players_emb = self.num_players_embed(num_players)    # (T, d_model)
        scalar_emb = self.scalar_proj(scalars)                   # (T, d_model)
        bet_emb = self.bet_proj(bets)                            # (T, d_model)
        action_emb = self.action_proj(actions)                   # (T, d_model)

        # Context: cat 6 embeddings, broadcast to 7 cards
        context = torch.cat([
            hero_pos_emb, acting_pos_emb, num_players_emb,
            scalar_emb, bet_emb, action_emb
        ], dim=-1)                                               # (T, 6*d_model)
        context = context.unsqueeze(1).expand(-1, 7, -1)         # (T, 7, 6*d_model)

        # Combine: cat(card_emb, context) → Linear → + source → LayerNorm
        combined = torch.cat([card_embs, context], dim=-1)       # (T, 7, 7*d_model)
        combined = combined.reshape(T * 7, self.d_model * 7)     # (T*7, 7*d_model)
        out = self.combine(combined)                              # (T*7, d_model)
        out = out.view(T, 7, self.d_model)                       # (T, 7, d_model)

        # Source embedding: constant [0,0,0,0,0,1,1], same for all events
        source_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.long, device=device)
        source_embs = self.source_embed(source_ids)              # (7, d_model)
        out = out + source_embs.unsqueeze(0)                     # (T, 7, d_model) broadcast

        out = self.post_embed_norm(out)                          # (T, 7, d_model)

        # --- Scatter into (B, max_events*7, d_model) output ---
        embeddings = torch.zeros(B, max_events * C, self.d_model,
                                 dtype=out.dtype, device=device)
        mask = torch.zeros(B, max_events * C, dtype=torch.float, device=device)

        bi = torch.tensor(batch_idx_list, dtype=torch.long, device=device)  # (T,)
        ei = torch.tensor(event_idx_list, dtype=torch.long, device=device)  # (T,)
        # Expand indices for 7 cards per event
        bi_exp = bi.unsqueeze(1).expand(-1, 7).reshape(-1)            # (T*7,)
        offsets = torch.arange(7, device=device).unsqueeze(0).expand(T, -1)  # (T, 7)
        col_idx = (ei.unsqueeze(1) * 7 + offsets).reshape(-1)         # (T*7,)

        embeddings[bi_exp, col_idx] = out.reshape(T * 7, self.d_model)

        for i, sl in enumerate(seq_lengths):
            mask[i, :sl * C] = 1.0

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
        encoded = self.encoder(embedded, mask=mask)  # (B, N*7, d_model)

        # Mean pool window=7: compress each event's 7 card vectors into 1
        C = EventSequenceEmbedder.CARDS_PER_EVENT
        B, S, D = encoded.shape
        encoded = encoded.view(B, S // C, C, D).mean(dim=2)  # (B, N, d_model)
        mask = mask[:, ::C]  # (B, N) — all 7 positions per event share same mask value

        if skip_memory:
            decoder_input = encoded
            decoder_mask = mask
        else:
            # Future: use last token for memory lookup
            last_token = encoded[:, -1, :]  # (B, d_model)
            mem_vectors = self.memory.search_batch(last_token)
            mem_vectors = mem_vectors.to(encoded.device)
            decoder_input = torch.cat([mem_vectors, encoded], dim=1)
            # Memory vectors are always real — prepend 1s to mask
            mem_mask = torch.ones(mask.shape[0], mem_vectors.shape[1],
                                  dtype=mask.dtype, device=mask.device)
            decoder_mask = torch.cat([mem_mask, mask], dim=1)

        output = self.decoder(decoder_input, mask=decoder_mask)
        return output, encoded, mask
