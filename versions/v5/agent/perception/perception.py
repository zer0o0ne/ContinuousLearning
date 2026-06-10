import torch
import torch.nn as nn
import numpy as np

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.perception.memory import HierarchicalMemory
from agent.perception.opponent_embeddings import OpponentGRUUpdater


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

    def _build_batch_tensors(self, event_sequences, device="cpu", mask_hand=False):
        """Collect raw event data into batch tensors and compute embedding lookups.

        Shared helper for forward_batch (and subclasses).

        Args:
            event_sequences: list of lists of event dicts
            device: torch device
            mask_hand: if True, replace hand card ids with 52 (no-card token)

        Returns:
            None if no events, else dict with:
                B, T, max_events, seq_lengths,
                batch_idx, event_idx (long tensors on device),
                card_embs (T,7,D), hero_pos_emb, acting_pos_emb, num_players_emb,
                scalar_emb, bet_emb, action_emb (each T,D)
        """
        B = len(event_sequences)
        seq_lengths = [len(seq) for seq in event_sequences]
        max_events = max(seq_lengths) if seq_lengths else 0

        all_card_ids = []
        all_hero_pos = []
        all_acting_pos = []
        all_num_players = []
        all_scalars = []
        all_bets = []
        all_actions = []
        batch_idx_list = []
        event_idx_list = []

        for i, seq in enumerate(event_sequences):
            for j, event in enumerate(seq):
                table_cards = [int(c) if int(c) >= 0 else 52 for c in event["table"]]
                if mask_hand:
                    hand_cards = [52, 52]
                else:
                    hand_cards = [int(c) if int(c) >= 0 else 52 for c in event["hand"]]
                cards = table_cards + hand_cards
                cards = [max(0, min(c, 52)) for c in cards]
                all_card_ids.append(cards)

                all_hero_pos.append(int(event["hero_pos"]))
                all_acting_pos.append(int(event["acting_pos"]))
                all_num_players.append(int(event["num_players"]))
                all_scalars.append([float(event["pot"]), float(event["stack"])])

                raw_bets = event["bets"]
                if isinstance(raw_bets, np.ndarray):
                    raw_bets = raw_bets.tolist()
                padded_bets = [0.0] * self.max_players
                for k, b in enumerate(raw_bets):
                    if k < self.max_players:
                        padded_bets[k] = float(b)
                all_bets.append(padded_bets)

                action = event["action"]
                if isinstance(action, torch.Tensor):
                    all_actions.append(action.float().tolist())
                else:
                    all_actions.append([float(a) for a in action])

                batch_idx_list.append(i)
                event_idx_list.append(j)

        T = len(all_card_ids)
        if T == 0:
            return None

        card_ids = torch.tensor(all_card_ids, dtype=torch.long, device=device)
        hero_pos = torch.tensor(all_hero_pos, dtype=torch.long, device=device)
        acting_pos = torch.tensor(all_acting_pos, dtype=torch.long, device=device)
        num_players = torch.tensor(all_num_players, dtype=torch.long, device=device)
        scalars = torch.tensor(all_scalars, dtype=torch.float, device=device)
        bets = torch.tensor(all_bets, dtype=torch.float, device=device)
        actions = torch.tensor(all_actions, dtype=torch.float, device=device)

        return {
            "B": B, "T": T, "max_events": max_events, "seq_lengths": seq_lengths,
            "batch_idx": torch.tensor(batch_idx_list, dtype=torch.long, device=device),
            "event_idx": torch.tensor(event_idx_list, dtype=torch.long, device=device),
            "card_embs": self.card_embed(card_ids),
            "hero_pos_emb": self.hero_pos_embed(hero_pos),
            "acting_pos_emb": self.acting_pos_embed(acting_pos),
            "num_players_emb": self.num_players_embed(num_players),
            "scalar_emb": self.scalar_proj(scalars),
            "bet_emb": self.bet_proj(bets),
            "action_emb": self.action_proj(actions),
        }

    def _compute_pre_inject(self, event_sequences, device="cpu"):
        """Compute per-event features up to (but excluding) opp_emb injection.

        Output is post-`combine`, post-`source_embed`, pre-LayerNorm.
        Lets the caller compute an opp_emb update signal from these features
        and re-enter the pipeline via `_apply_post_inject` with fresh embeddings.

        Returns:
            out_pre: (T, 7, d_model) — None if batch is empty
            meta: dict with B, T, max_events, seq_lengths, batch_idx, event_idx
        """
        bt = self._build_batch_tensors(event_sequences, device=device, mask_hand=False)
        B = len(event_sequences)
        seq_lengths = [len(seq) for seq in event_sequences]
        max_events = max(seq_lengths) if seq_lengths else 0

        meta = {
            "B": B, "max_events": max_events, "seq_lengths": seq_lengths,
            "T": 0, "batch_idx": None, "event_idx": None,
        }
        if bt is None:
            return None, meta

        T = bt["T"]
        meta.update(T=T, batch_idx=bt["batch_idx"], event_idx=bt["event_idx"])

        context = torch.cat([
            bt["hero_pos_emb"], bt["acting_pos_emb"], bt["num_players_emb"],
            bt["scalar_emb"], bt["bet_emb"], bt["action_emb"],
        ], dim=-1)                                               # (T, 6*d_model)
        context = context.unsqueeze(1).expand(-1, 7, -1)         # (T, 7, 6*d_model)

        combined = torch.cat([bt["card_embs"], context], dim=-1) # (T, 7, 7*d_model)
        combined = combined.reshape(T * 7, self.d_model * 7)
        out = self.combine(combined).view(T, 7, self.d_model)

        source_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.long, device=device)
        source_embs = self.source_embed(source_ids)              # (7, d_model)
        out = out + source_embs.unsqueeze(0)                     # (T, 7, d_model)
        return out, meta

    def _apply_post_inject(self, out_pre, meta, opponent_embs_per_event, device="cpu"):
        """Inject opp_emb into hand slots, LayerNorm, scatter to (B, M*7, d_model).

        out_pre may be None when batch had no events; returns zero tensors.
        opponent_embs_per_event: None or list of length T (parallel to flat
        event order), each entry a (d_model,) tensor or None.
        """
        C = self.CARDS_PER_EVENT
        B = meta["B"]
        max_events = meta["max_events"]
        seq_lengths = meta["seq_lengths"]

        if out_pre is None:
            embeddings = torch.zeros(B, max_events * C, self.d_model,
                                     dtype=torch.float, device=device)
            mask = torch.zeros(B, max_events * C, dtype=torch.float, device=device)
            return embeddings, mask

        T = meta["T"]
        out = out_pre

        if opponent_embs_per_event is not None:
            indices = [i for i, e in enumerate(opponent_embs_per_event) if e is not None]
            if indices:
                stacked = torch.stack([opponent_embs_per_event[i] for i in indices])  # (K, d_model)
                idx_t = torch.tensor(indices, dtype=torch.long, device=device)
                out = out.clone()
                out[idx_t, 5, :] = out[idx_t, 5, :] + stacked
                out[idx_t, 6, :] = out[idx_t, 6, :] + stacked

        out = self.post_embed_norm(out)                          # (T, 7, d_model)

        embeddings = torch.zeros(B, max_events * C, self.d_model,
                                 dtype=out.dtype, device=device)
        mask = torch.zeros(B, max_events * C, dtype=torch.float, device=device)

        bi = meta["batch_idx"]
        ei = meta["event_idx"]
        bi_exp = bi.unsqueeze(1).expand(-1, 7).reshape(-1)
        offsets = torch.arange(7, device=device).unsqueeze(0).expand(T, -1)
        col_idx = (ei.unsqueeze(1) * 7 + offsets).reshape(-1)

        embeddings[bi_exp, col_idx] = out.reshape(T * 7, self.d_model)
        for i, sl in enumerate(seq_lengths):
            mask[i, :sl * C] = 1.0
        return embeddings, mask

    def forward_batch(self, event_sequences, device="cpu", opponent_embs_per_event=None):
        """Embed a batch of event sequences into per-card vectors.

        Thin wrapper over `_compute_pre_inject` + `_apply_post_inject`.
        Returns (embeddings: (B, M*7, d_model), mask: (B, M*7)).
        """
        out_pre, meta = self._compute_pre_inject(event_sequences, device=device)
        return self._apply_post_inject(out_pre, meta, opponent_embs_per_event,
                                       device=device)


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

        opp_cfg = config.get("opponent_embedding", {})
        self.opp_emb_enabled = opp_cfg.get("enabled", False)
        if self.opp_emb_enabled:
            self.opponent_gru = OpponentGRUUpdater(d_model)
        self.d_model = d_model

    def set_gradient_checkpointing(self, enabled: bool):
        self.encoder.gradient_checkpointing = bool(enabled)
        self.decoder.gradient_checkpointing = bool(enabled)

    def forward_batch(self, event_sequences, device="cpu", skip_memory=True,
                      skip_opponent_emb=True, opponent_emb_table=None):
        """
        Batch-parallel forward over event sequences.

        Args:
            event_sequences: list of lists of event dicts
            device: torch device
            skip_memory: if True, encoder output goes directly to decoder
            skip_opponent_emb: if True, skip opponent GRU embedding injection
            opponent_emb_table: optional OpponentEmbeddingTable instance.
                Only used when skip_opponent_emb=False. When provided and
                opp_emb_enabled, each opponent's embedding is updated via GRU
                BEFORE the encoder forward (signal = embedder pre-injection
                features of that opponent's LAST event in the batch), and the
                fresh embedding is then injected into hand-slot tokens so the
                encoder sees the updated value. This keeps GRU parameters on
                the gradient path from loss back through the encoder.
                The table is mutated in-place.

        Returns: tuple (output, encoded, mask)
            output: (B, seq_len, d_model)
            encoded: (B, seq_len, d_model)
            mask: (B, seq_len)
        """
        C = EventSequenceEmbedder.CARDS_PER_EVENT
        use_opp_emb = (not skip_opponent_emb and self.opp_emb_enabled
                       and opponent_emb_table is not None)

        if use_opp_emb:
            # Flat parallel list of opponent_ids (one entry per event, None if
            # event has no opponent_id). Must match flat order used by
            # _build_batch_tensors (sample-major, event-major).
            opp_event_map = []
            for seq in event_sequences:
                for event in seq:
                    opp_event_map.append(event.get("opponent_id"))

            # Stage 1: embedder pre-injection features for every event.
            # out_pre: (T, 7, d_model). Used both as GRU signal source AND as
            # input to the post-injection stage (no double work).
            out_pre, meta = self.embedder._compute_pre_inject(
                event_sequences, device=device,
            )

            new_opp_embs = {}
            if out_pre is not None:
                # Per opp_id, take the LAST event where this opponent acted.
                # Signal = mean over its 7 card vectors of out_pre at that
                # event. GRU sees only one signal per opp per forward.
                last_flat_idx = {}
                for flat_idx, opp_id in enumerate(opp_event_map):
                    if opp_id is not None:
                        last_flat_idx[opp_id] = flat_idx

                for opp_id, flat_idx in last_flat_idx.items():
                    signal = out_pre[flat_idx].mean(dim=0)            # (d_model,)
                    old_emb = opponent_emb_table.get(opp_id, device)  # detached / zeros
                    new_emb = self.opponent_gru(signal, old_emb)      # in graph → GRU params
                    new_opp_embs[opp_id] = new_emb
                    opponent_emb_table.embeddings[opp_id] = new_emb

            # Build per-event opp_emb list using the FRESH embedding so the
            # encoder forward depends on new_emb → gradient reaches GRU.
            opponent_embs_per_event = [
                new_opp_embs[oid] if oid is not None else None
                for oid in opp_event_map
            ]

            embedded, mask = self.embedder._apply_post_inject(
                out_pre, meta, opponent_embs_per_event, device=device,
            )
        else:
            embedded, mask = self.embedder.forward_batch(
                event_sequences, device=device,
            )

        encoded = self.encoder(embedded, mask=mask)  # (B, N*7, d_model)

        # Mean pool window=7: compress each event's 7 card vectors into 1
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
