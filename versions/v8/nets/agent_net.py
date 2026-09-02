"""Entity 3 — the agent (CONCEPT.md §6.1).

The same §5.1 observation the opponent-embedding network reads, the same trunk
(`nets/trunk.py`, OI-4), and one linear head to the action set. Nothing else:

* **no value head** (§6.1, plan D5). §7.4's variant C would need one, and it is
  the first lever if the label budget does not close — but that is a decision
  G3 has to inform, and building the head before then would be an addition
  nobody asked for and a term nobody could train.
* **no search.** The agent is a policy; the search that produces its targets is
  the BR oracle (`oracle/`), and it runs offline over played hands, not inside
  this forward.

**The logits come from the last real token.** The observation of a decision that
has not been taken yet is the hand up to now plus one pending token (§9, and
`nets.features.hand_tokens(pending=...)`), and that pending token is the last
one in the row. Rows in a batch have different lengths, so the read is a gather
at each row's own length, never at a fixed position — reading the padded tail
would answer about a token that does not exist.

**The logits are unmasked.** Legality is applied in exactly one place per
consumer — `PoolMember.policy` for a played action, the target construction for
a trained one — and both take the mask carried on the token, which is the mask
the driver sampled with. A second masking site inside the network is how the
two drift.
"""

import torch
import torch.nn as nn

from nets.trunk import HandEncoder


class AgentNet(nn.Module):
    """(B, n_actions) logits at each hand's last real token."""

    def __init__(self, cfg, n_actions, max_players):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.d_emb = cfg["d_emb"]
        self.n_actions = n_actions
        self.max_players = max_players

        self.encoder = HandEncoder(cfg, n_actions, max_players)
        self.action_out = nn.Linear(self.d_model, n_actions)

    def forward(self, batch, emb, seat_emb=None):
        """(B, n_actions) logits. See `logits_and_range` for the §5.7 head."""
        return self.logits_and_range(batch, emb, seat_emb)[0]

    def logits_and_range(self, batch, emb, seat_emb=None):
        """`(action logits, range logits)` — the second is `None` with §5.7 off.

        The agent carries no head of its own for the belief: the head is in the
        trunk (`nets/range_head.py`), so `warm_start_trunk` hands it over
        already trained and this network inherits it with the rest of the
        encoder rather than starting a second one from scratch.
        """
        hidden, range_logits = self.encoder(batch, emb, seat_emb)
        mask = batch["mask"]
        lengths = mask.sum(dim=1).long()
        assert int(lengths.min()) > 0, (
            "a row of the batch has no real token at all — `collate` drops "
            "empty hands, so this is a hand-building bug")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.action_out(hidden[rows, lengths - 1]), range_logits
