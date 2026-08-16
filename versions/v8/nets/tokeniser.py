"""The situation tokeniser (CONCEPT.md §5.1, §16 OI-4).

One class, used by both the opponent-embedding network and — when it lands —
the agent. OI-4 settled that deliberately: the tokeniser is the MLP mapping the
§5.1 feature set to a single `d_model` token vector, both networks consume that
token, and a second copy of this code is exactly the duplicated low-level logic
`CLAUDE.md` §5 warns about. Any divergence between two copies would be a silent
train/deploy mismatch.

Weights are **not** shared (OI-4 again) — the two networks optimise different
objectives on different retraining cadences. Only the code is.
"""

import torch
import torch.nn as nn

from nets.features import N_TOKEN_TYPES

N_CARDS_PER_TOKEN = 7  # 5 board + 2 hole
N_SCALARS = 3          # acting stack, pot, amount to call — all in BB


class SituationTokeniser(nn.Module):
    """§5.1 features → one `d_model` vector per decision.

    The acting player's embedding is one of the features, concatenated with the
    rest before the MLP, so its width `d_emb` is free.
    """

    def __init__(self, d_model, d_emb, n_actions, max_players, d_card=32,
                 d_index=32, max_decisions=64):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.n_actions = n_actions
        self.max_players = max_players
        self.max_decisions = max_decisions

        self.card_embed = nn.Embedding(53, d_card)               # 52 = unknown
        self.decision_idx_embed = nn.Embedding(max_decisions, d_index)
        self.acting_pos_embed = nn.Embedding(max_players, d_index)
        self.num_players_embed = nn.Embedding(max_players + 1, d_index)
        # §5.1a: a decision token and a showdown token carry different things
        # in the same slots, so the type has to be readable.
        self.token_type_embed = nn.Embedding(N_TOKEN_TYPES, d_index)

        d_in = (N_CARDS_PER_TOKEN * d_card + 4 * d_index + N_SCALARS
                + max_players + n_actions + d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch, emb):
        """(B, T, d_model).

        Args:
            batch: the dict produced by `nets.features.collate`.
            emb: (B, T, d_emb) — the *acting* player's embedding at each token,
                identical across all tokens of that player within a hand (§5.1).
        """
        cards = self.card_embed(batch["cards"])                   # (B,T,7,d_card)
        B, T = cards.shape[0], cards.shape[1]
        cards = cards.reshape(B, T, -1)

        idx = batch["decision_idx"].clamp(max=self.max_decisions - 1)
        parts = [
            cards,
            self.decision_idx_embed(idx),
            self.acting_pos_embed(batch["acting_pos"]),
            self.num_players_embed(batch["num_players"]),
            self.token_type_embed(batch["token_type"]),
            batch["scalars"],
            batch["seat_stacks"],
            batch["prev_action"],
            emb,
        ]
        x = torch.cat(parts, dim=-1)
        return self.norm(self.mlp(x)) * batch["mask"].unsqueeze(-1)
