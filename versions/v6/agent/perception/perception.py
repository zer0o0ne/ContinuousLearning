import warnings

import torch
import torch.nn as nn
import numpy as np

from agent.perception.encoder import Encoder
from agent.perception.decoder import Decoder
from agent.perception.memory import HierarchicalMemory
from agent.perception.opponent_embeddings import OpponentGRUUpdater

# A.4.5: "no group" marker for the GRU sample-group rewind in forward_batch.
# A unique object so ANY caller-provided group id (including None-able ints)
# can never collide with it.
_GROUP_SENTINEL = object()


def extract_event_tensors(event_sequences, max_players):
    """Extract raw numeric fields from event dicts into CPU tensors.

    Moves the CPU-bound Python loop out of the GPU forward path so it can
    run in DataLoader collate (overlapped with the previous batch's GPU work).

    Returns None if no events, else a dict of CPU tensors ready for
    _build_batch_tensors(precomputed=...).
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
    all_stacks = []
    all_actions = []
    batch_idx_list = []
    event_idx_list = []

    for i, seq in enumerate(event_sequences):
        for j, event in enumerate(seq):
            table_cards = [int(c) if int(c) >= 0 else 52 for c in event["table"]]
            hand_cards = [int(c) if int(c) >= 0 else 52 for c in event["hand"]]
            cards = table_cards + hand_cards
            assert all(0 <= c <= 52 for c in cards), (
                f"card index out of [0,52]: {cards}")
            all_card_ids.append(cards)

            all_hero_pos.append(int(event["hero_pos"]))
            all_acting_pos.append(int(event["acting_pos"]))
            all_num_players.append(int(event["num_players"]))
            all_scalars.append([float(event["pot"]), float(event["stack"])])

            raw_bets = event["bets"]
            if isinstance(raw_bets, np.ndarray):
                raw_bets = raw_bets.tolist()
            padded_bets = [0.0] * max_players
            for k, b in enumerate(raw_bets):
                if k < max_players:
                    padded_bets[k] = float(b)
            all_bets.append(padded_bets)

            raw_stacks = event.get("stacks")
            if raw_stacks is None:
                raw_stacks = []
            elif isinstance(raw_stacks, np.ndarray):
                raw_stacks = raw_stacks.tolist()
            padded_stacks = [0.0] * max_players
            for k, sv in enumerate(raw_stacks):
                if k < max_players:
                    padded_stacks[k] = float(sv)
            all_stacks.append(padded_stacks)

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

    return {
        "card_ids": torch.tensor(all_card_ids, dtype=torch.long),
        "hero_pos": torch.tensor(all_hero_pos, dtype=torch.long),
        "acting_pos": torch.tensor(all_acting_pos, dtype=torch.long),
        "num_players": torch.tensor(all_num_players, dtype=torch.long),
        "scalars": torch.tensor(all_scalars, dtype=torch.float),
        "bets": torch.tensor(all_bets, dtype=torch.float),
        "stacks": torch.tensor(all_stacks, dtype=torch.float),
        "actions": torch.tensor(all_actions, dtype=torch.float),
        "batch_idx": torch.tensor(batch_idx_list, dtype=torch.long),
        "event_idx": torch.tensor(event_idx_list, dtype=torch.long),
        "seq_lengths": seq_lengths,
        "max_events": max_events,
        "B": B,
    }


class EventSequenceEmbedder(nn.Module):
    """Embeds a sequence of poker events into per-card vectors.

    Each event produces 7 vectors (5 table cards + 2 hand cards, fixed order).
    Each vector combines the card embedding with full game context
    (positions, pot, stack, bets, action, num_players).
    A learned source embedding distinguishes table cards from hand cards.
    """

    CARDS_PER_EVENT = 7  # 5 table + 2 hand

    def __init__(self, d_model, n_actions, max_players, max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_players = max_players
        # A.5.2: cap on events so the encoder's N*7 token count never exceeds
        # max_seq_len (RoPE would otherwise silently extrapolate). None = no cap.
        self.max_seq_len = max_seq_len
        self.max_events = (max_seq_len // self.CARDS_PER_EVENT) if max_seq_len else None
        self._warned_cap = False

        self.card_embed = nn.Embedding(53, d_model)        # 0-51 = cards, 52 = no-card
        self.source_embed = nn.Embedding(2, d_model)       # 0 = table card, 1 = hand card
        self.hero_pos_embed = nn.Embedding(max_players, d_model)
        self.acting_pos_embed = nn.Embedding(max_players, d_model)
        self.num_players_embed = nn.Embedding(max_players + 1, d_model)
        self.scalar_proj = nn.Linear(2, d_model)            # pot, hero stack
        self.bet_proj = nn.Linear(max_players, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        # B.6.2: per-position stacks vector → its own projection (mirrors
        # bet_proj). Without a per-seat effective-stack signal the agent cannot
        # learn stack-aware play. Changes the event schema — events must now
        # carry a "stacks" list; missing → treated as zeros.
        self.stacks_proj = nn.Linear(max_players, d_model)
        # card_emb + 7 context components = 8 * d_model
        self.combine = nn.Linear(d_model * 8, d_model)
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
        # A.5.3: assert (stripped under python -O) instead of a silent clamp,
        # which would hide upstream card-encoding bugs.
        assert int(card_ids.min()) >= 0 and int(card_ids.max()) <= 52, (
            f"card index out of [0,52]: {card_ids.tolist()}")
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

        # B.6.2: per-position stacks vector (mirrors bets). Missing → zeros.
        stacks = torch.zeros(self.max_players, dtype=torch.float, device=device)
        raw_stacks = event.get("stacks")
        if raw_stacks is None:
            raw_stacks = []
        elif isinstance(raw_stacks, np.ndarray):
            raw_stacks = raw_stacks.tolist()
        for i, sv in enumerate(raw_stacks):
            if i < self.max_players:
                stacks[i] = float(sv)
        stacks_emb = self.stacks_proj(stacks)

        action = event["action"]
        if isinstance(action, torch.Tensor):
            action_t = action.float().to(device)
        else:
            action_t = torch.tensor(action, dtype=torch.float, device=device)
        action_emb = self.action_proj(action_t)

        # Context: cat 7 embeddings, broadcast to all 7 cards
        context = torch.cat([
            hero_pos_emb, acting_pos_emb, num_players_emb,
            scalar_emb, bet_emb, action_emb, stacks_emb
        ])  # (7 * d_model,)
        context = context.unsqueeze(0).expand(7, -1)  # (7, 7 * d_model)

        # Per-card: cat(card_emb, context) → Linear(8d → d) → + source → LayerNorm
        combined = torch.cat([card_embs, context], dim=-1)  # (7, 8 * d_model)
        out = self.combine(combined) + source_embs           # (7, d_model)
        return self.post_embed_norm(out)                     # (7, d_model)

    def _cap_sequences(self, event_sequences):
        """A.5.2: cap each sequence to the most recent `max_events` events.

        Keeps N*7 <= max_seq_len so the encoder/decoder RoPE never extrapolates.
        Drops the OLDEST events (recent betting is most relevant) and warns once.
        No-op when no cap is set or nothing overflows; never mutates the input.
        """
        cap = self.max_events
        if not cap:
            return event_sequences
        capped = None
        for i, seq in enumerate(event_sequences):
            if len(seq) > cap:
                if capped is None:
                    capped = list(event_sequences)
                if not self._warned_cap:
                    warnings.warn(
                        f"event sequence(s) exceed max_events={cap} "
                        f"(max_seq_len={self.max_seq_len}); truncating oldest "
                        f"events. Raise architecture.max_seq_len for real runs.",
                        stacklevel=2,
                    )
                    self._warned_cap = True
                capped[i] = seq[-cap:]
        return capped if capped is not None else event_sequences

    def _build_batch_tensors(self, event_sequences, device="cpu", mask_hand=False,
                             precomputed=None):
        """Collect raw event data into batch tensors and compute embedding lookups.

        Args:
            event_sequences: list of lists of event dicts (ignored when precomputed)
            device: torch device
            mask_hand: if True, replace hand card ids with 52 (no-card token)
            precomputed: dict from extract_event_tensors() — skips dict extraction

        Returns:
            None if no events, else dict with:
                B, T, max_events, seq_lengths,
                batch_idx, event_idx (long tensors on device),
                card_embs (T,7,D), hero_pos_emb, acting_pos_emb, num_players_emb,
                scalar_emb, bet_emb, action_emb (each T,D)
        """
        if precomputed is not None:
            if len(precomputed.get("card_ids", [])) == 0:
                return None
            card_ids = precomputed["card_ids"].to(device)
            if mask_hand:
                card_ids = card_ids.clone()
                card_ids[:, 5:] = 52
            return {
                "B": precomputed["B"], "T": len(card_ids),
                "max_events": precomputed["max_events"],
                "seq_lengths": precomputed["seq_lengths"],
                "batch_idx": precomputed["batch_idx"].to(device),
                "event_idx": precomputed["event_idx"].to(device),
                "card_embs": self.card_embed(card_ids),
                "hero_pos_emb": self.hero_pos_embed(precomputed["hero_pos"].to(device)),
                "acting_pos_emb": self.acting_pos_embed(precomputed["acting_pos"].to(device)),
                "num_players_emb": self.num_players_embed(precomputed["num_players"].to(device)),
                "scalar_emb": self.scalar_proj(precomputed["scalars"].to(device)),
                "bet_emb": self.bet_proj(precomputed["bets"].to(device)),
                "stacks_emb": self.stacks_proj(precomputed["stacks"].to(device)),
                "action_emb": self.action_proj(precomputed["actions"].to(device)),
            }

        B = len(event_sequences)
        seq_lengths = [len(seq) for seq in event_sequences]
        max_events = max(seq_lengths) if seq_lengths else 0

        all_card_ids = []
        all_hero_pos = []
        all_acting_pos = []
        all_num_players = []
        all_scalars = []
        all_bets = []
        all_stacks = []
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
                assert all(0 <= c <= 52 for c in cards), (
                    f"card index out of [0,52]: {cards}")
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

                raw_stacks = event.get("stacks")
                if raw_stacks is None:
                    raw_stacks = []
                elif isinstance(raw_stacks, np.ndarray):
                    raw_stacks = raw_stacks.tolist()
                padded_stacks = [0.0] * self.max_players
                for k, sv in enumerate(raw_stacks):
                    if k < self.max_players:
                        padded_stacks[k] = float(sv)
                all_stacks.append(padded_stacks)

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
        stacks = torch.tensor(all_stacks, dtype=torch.float, device=device)
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
            "stacks_emb": self.stacks_proj(stacks),
            "action_emb": self.action_proj(actions),
        }

    def _compute_pre_inject(self, event_sequences, device="cpu", precomputed=None):
        """Compute per-event features up to (but excluding) opp_emb injection.

        Output is post-`combine`, post-`source_embed`, pre-LayerNorm.
        Lets the caller compute an opp_emb update signal from these features
        and re-enter the pipeline via `_apply_post_inject` with fresh embeddings.

        Returns:
            out_pre: (T, 7, d_model) — None if batch is empty
            meta: dict with B, T, max_events, seq_lengths, batch_idx, event_idx
        """
        bt = self._build_batch_tensors(event_sequences, device=device, mask_hand=False,
                                       precomputed=precomputed)
        if precomputed is not None:
            B = precomputed["B"]
            seq_lengths = precomputed["seq_lengths"]
            max_events = precomputed["max_events"]
        else:
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
            bt["scalar_emb"], bt["bet_emb"], bt["action_emb"], bt["stacks_emb"],
        ], dim=-1)                                               # (T, 7*d_model)
        context = context.unsqueeze(1).expand(-1, 7, -1)         # (T, 7, 7*d_model)

        combined = torch.cat([bt["card_embs"], context], dim=-1) # (T, 7, 8*d_model)
        combined = combined.reshape(T * 7, self.d_model * 8)
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

    def forward_batch(self, event_sequences, device="cpu", opponent_embs_per_event=None,
                      precomputed=None):
        """Embed a batch of event sequences into per-card vectors.

        Thin wrapper over `_compute_pre_inject` + `_apply_post_inject`.
        Returns (embeddings: (B, M*7, d_model), mask: (B, M*7)).
        """
        if precomputed is None:
            event_sequences = self._cap_sequences(event_sequences)  # A.5.2
        out_pre, meta = self._compute_pre_inject(event_sequences, device=device,
                                                  precomputed=precomputed)
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

        self.embedder = EventSequenceEmbedder(d_model, n_actions, max_players,
                                              max_seq_len=max_seq_len)
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
                      skip_opponent_emb=True, opponent_emb_table=None,
                      gru_window=1, precomputed=None, gru_sample_groups=None):
        """
        Batch-parallel forward over event sequences.

        Args:
            event_sequences: list of lists of event dicts
            device: torch device
            skip_memory: if True, encoder output goes directly to decoder
            skip_opponent_emb: if True, skip opponent GRU embedding injection
            opponent_emb_table: optional OpponentEmbeddingTable instance.
            gru_window: truncated-BPTT depth (A.4).
            precomputed: dict from extract_event_tensors() — skips dict extraction
            gru_sample_groups: optional list (len B) of hashable group ids,
                aligned with event_sequences. Consecutive samples with the
                same id are treated as observer copies of ONE scenario: each
                copy's GRU pass rewinds to the state as of the group start so
                the shared table advances once per scenario (A.4.5). None →
                every sample advances (legacy behavior).

        Returns: tuple (output, encoded, mask)
            output: (B, seq_len, d_model)
            encoded: (B, seq_len, d_model)
            mask: (B, seq_len)
        """
        C = EventSequenceEmbedder.CARDS_PER_EVENT
        if precomputed is None:
            event_sequences = self.embedder._cap_sequences(event_sequences)
        use_opp_emb = (not skip_opponent_emb and self.opp_emb_enabled
                       and opponent_emb_table is not None)

        if use_opp_emb:
            # Flat parallel list of opponent_ids (one entry per event, None if
            # event has no opponent_id). Must match flat order used by
            # _build_batch_tensors (sample-major, event-major).
            opp_event_map = []
            flat_sample_of = []
            for b_i, seq in enumerate(event_sequences):
                for event in seq:
                    opp_event_map.append(event.get("opponent_id"))
                    flat_sample_of.append(b_i)
            if gru_sample_groups is not None:
                assert len(gru_sample_groups) == len(event_sequences), (
                    f"gru_sample_groups length {len(gru_sample_groups)} != "
                    f"batch size {len(event_sequences)}")

            # Stage 1: embedder pre-injection features for every event.
            # out_pre: (T, 7, d_model). Used both as GRU signal source AND as
            # input to the post-injection stage (no double work).
            out_pre, meta = self.embedder._compute_pre_inject(
                event_sequences, device=device, precomputed=precomputed,
            )

            k = max(1, int(gru_window))
            opponent_embs_per_event = [None] * len(opp_event_map)
            if out_pre is not None:
                # A.4.2: a single causal pass over the flat event list. Flat
                # order is sample-major then event-major (see
                # _build_batch_tensors), so this visits each sample's events in
                # chronological order with NO cross-sample interleaving; the
                # per-opponent running state and the table advance sequentially
                # in that (hand) order. At each event we inject the running
                # state AS OF that event — never an embedding derived from later
                # events of the same sample (no future leak).
                #
                # gru_window=k bounds truncated BPTT: the running value always
                # carries forward, but the graph is detached every k GRU steps
                # (k=1 → depth-1 BPTT, matching the legacy default).
                running = {}              # opp_id -> hidden state (value carries)
                steps_since_detach = {}   # opp_id -> GRU steps in current graph
                # A.4.5: consecutive samples sharing a group id are observer
                # copies of the SAME scenario (phase 5 expands each decision
                # into per-observer samples). Without a rewind, the shared
                # table advances once per copy — O(observers)× more GRU steps
                # per decision than deployment ever performs, so the head
                # trains against systematically over-saturated states. Rewind
                # each duplicate copy to the state as of the group start; the
                # table then advances once per scenario (the last copy's end
                # state persists).
                prev_sample = None
                prev_group = _GROUP_SENTINEL
                group_start_running = {}
                group_start_steps = {}
                group_start_table = None
                for flat_idx, opp_id in enumerate(opp_event_map):
                    b_i = flat_sample_of[flat_idx]
                    if b_i != prev_sample:
                        prev_sample = b_i
                        g = (gru_sample_groups[b_i]
                             if gru_sample_groups is not None
                             else _GROUP_SENTINEL)
                        if (g is not _GROUP_SENTINEL and g == prev_group):
                            # Rewind the running state AND the table: the
                            # per-event write-back below advanced the table
                            # during the previous copy, and a fresh opp_id
                            # falls back to the table.
                            running = dict(group_start_running)
                            steps_since_detach = dict(group_start_steps)
                            opponent_emb_table.embeddings = dict(
                                group_start_table)
                        else:
                            group_start_running = dict(running)
                            group_start_steps = dict(steps_since_detach)
                            if gru_sample_groups is not None:
                                group_start_table = dict(
                                    opponent_emb_table.embeddings)
                            prev_group = g
                    if opp_id is None:
                        continue
                    h = running.get(opp_id)
                    if h is None:
                        h = opponent_emb_table.get(opp_id, device)   # detached / zeros
                        steps_since_detach[opp_id] = 0
                    elif steps_since_detach[opp_id] >= k:
                        h = h.detach()
                        steps_since_detach[opp_id] = 0
                    # A.4.1: GRU signal from the table-card slots (0-4) only —
                    # the hand slots (5,6) are the OBSERVER's hole cards and
                    # would contaminate the shared opponent embedding.
                    signal = out_pre[flat_idx][:5].mean(dim=0)       # (d_model,)
                    h = self.opponent_gru(signal, h)                 # in graph → GRU params
                    steps_since_detach[opp_id] += 1
                    running[opp_id] = h
                    opponent_embs_per_event[flat_idx] = h
                    opponent_emb_table.embeddings[opp_id] = h

            embedded, mask = self.embedder._apply_post_inject(
                out_pre, meta, opponent_embs_per_event, device=device,
            )
        else:
            embedded, mask = self.embedder.forward_batch(
                event_sequences, device=device, precomputed=precomputed,
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
