"""The range head and its injection back into the trunk (CONCEPT.md §5.7).

The third thing the trunk does, between its two halves of decoder layers: at
every decision token it predicts, for every live opponent, a distribution over
the 1326 two-card combos — the observer's belief about that player's holding —
and feeds that belief back into the tokens the remaining layers see.

**Why a belief at all.** DeepStack and ReBeL make the range a first-class object
and hand it to the value network as an *input*, because in self-play it is
computable. Ours is not: the players are unknown members of a pool with drawn
styles, so the range has to be *inferred*. This module is that inference, made
explicit and supervised, rather than something the trunk may or may not learn on
its own from an EV label whose noise is larger than the differences it teaches
(§11.4, §13).

**Why a perceiver decoder and not an MLP on the token.** A range is not readable
off one situation vector: it is the product of that player's likelihoods over
every decision they have taken, and those live in earlier tokens. So the head is
a small stack of cross-attention blocks whose queries are "player *s*, at moment
*t*" and whose keys are the trunk's states over the whole prefix. One query per
(token, live opponent).

Three properties carry the design, and each is a way of getting it wrong:

* **The cross-attention is causal.** A query at token *t* attends to trunk
  states at *t' ≤ t* only. Without that the belief at the third decision reads
  the seventh, the loss falls beautifully, and at deployment the head is reading
  actions that have not happened. It is the same rule as the trunk's own
  attention and it is enforced by the same kind of mask.
* **The query says *who* and *when*.** Position embeddings alone would make all
  `T` queries of one seat the same vector, distinguishable only by their mask —
  formally sufficient, practically a query that cannot ask about *now*. So the
  query is the seat's learnable position embedding plus the trunk state at `t`
  plus that seat's opponent vector, and RoPE marks the moment.
* **What goes back into the trunk is the probabilities, and it is detached.**
  Not the head's hidden state: a `d_range`-wide state would let the action loss
  push arbitrary information around the 1326-wide bottleneck, and the stop-grad
  would then be closing the wrong channel. With the detach the head is trained
  by its own target and by nothing else, and the layers above it consume a
  belief they cannot bend (owner decision 2026-09-02).

**The blocked combos are dropped, not learned.** A combo holding a card that is
on the board or in the observer's own hand gets `-inf` before the softmax, the
same way an illegal action gets an exact zero in a policy target. Making the
head rediscover card removal would spend its capacity on arithmetic that is a
deterministic function of its input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding, apply_rotary_pos_emb,
)

from nets.features import UNKNOWN_CARD
from oracle.ranges import COMBO_CARDS, N_COMBOS


def combo_block_mask(batch):
    """(B, T, 1326) bool — True where a combo is still possible.

    Dead is the board as of the token plus the observer's own two cards
    (`own_hole`, which is on every token for exactly this). `UNKNOWN_CARD` is
    the 53rd row of the scatter and is dropped, so a street that has not come
    blocks nothing.
    """
    B, T = batch["mask"].shape
    dead = torch.cat([batch["cards"][:, :, :5], batch["own_hole"]], dim=-1)
    seen = torch.zeros((B, T, UNKNOWN_CARD + 1), dtype=torch.bool,
                       device=dead.device)
    seen.scatter_(2, dead, True)
    seen = seen[:, :, :UNKNOWN_CARD]
    a, b = _combo_cards(dead.device)
    return ~(seen[:, :, a] | seen[:, :, b])


_COMBO_CACHE = {}


def _combo_cards(device):
    key = str(device)
    if key not in _COMBO_CACHE:
        t = torch.as_tensor(COMBO_CARDS, dtype=torch.long, device=device)
        _COMBO_CACHE[key] = (t[:, 0].contiguous(), t[:, 1].contiguous())
    return _COMBO_CACHE[key]


class _CrossBlock(nn.Module):
    """Pre-norm cross-attention + FFN, one block of the head."""

    def __init__(self, d_q, d_kv, n_heads, d_ff):
        super().__init__()
        assert d_q % n_heads == 0, (
            f"d_range {d_q} must divide into {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = d_q // n_heads
        self.norm_q = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_kv)
        self.q_proj = nn.Linear(d_q, d_q, bias=False)
        self.k_proj = nn.Linear(d_kv, d_q, bias=False)
        self.v_proj = nn.Linear(d_kv, d_q, bias=False)
        self.o_proj = nn.Linear(d_q, d_q, bias=False)
        self.norm_ff = nn.LayerNorm(d_q)
        self.ff = nn.Sequential(nn.Linear(d_q, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_q))

    def _heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, q, kv, attn_mask, rope_q=None, rope_kv=None):
        B, Lq, _ = q.shape
        h = self.norm_q(q)
        kvn = self.norm_kv(kv)
        qh = self._heads(self.q_proj(h))
        kh = self._heads(self.k_proj(kvn))
        vh = self._heads(self.v_proj(kvn))
        if rope_q is not None:
            cos_q, sin_q = rope_q
            cos_k, sin_k = rope_kv
            qh, _ = apply_rotary_pos_emb(qh, qh, cos_q, sin_q)
            kh, _ = apply_rotary_pos_emb(kh, kh, cos_k, sin_k)
        out = F.scaled_dot_product_attention(qh, kh, vh,
                                             attn_mask=attn_mask.unsqueeze(1))
        out = out.transpose(1, 2).reshape(B, Lq, -1)
        q = q + self.o_proj(out)
        return q + self.ff(self.norm_ff(q))


class RangeHead(nn.Module):
    """§5.7 — beliefs at every (token, live opponent), and their injection.

    `forward` returns `(logits, hidden)`: `logits` is `(A, 1326)` over the
    batch's active (hand, token, seat) rows in `act_idx` order, already `-inf`
    on blocked combos; `hidden` is the trunk states with the belief injected.
    """

    def __init__(self, cfg, d_model, d_emb, max_players):
        super().__init__()
        d_range = cfg.get("d_range", d_model)
        n_heads = cfg.get("range_heads", cfg["n_heads"])
        d_ff = cfg.get("d_range_ff", d_range * 2)
        n_blocks = int(cfg.get("n_range_blocks", 2))
        assert n_blocks >= 1, "a range head with no block predicts nothing"

        self.d_range = d_range
        self.max_players = max_players
        self.pos_embed = nn.Embedding(max_players, d_range)
        self.q_in = nn.Linear(d_model + d_emb, d_range)
        self.blocks = nn.ModuleList(
            [_CrossBlock(d_range, d_model, n_heads, d_ff)
             for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(d_range)
        self.combo_out = nn.Linear(d_range, N_COMBOS)

        # The way back in: the probability vector becomes one key/value per
        # active seat, and the token attends over its own seats.
        self.value_in = nn.Linear(N_COMBOS, d_model)
        self.pos_out = nn.Embedding(max_players, d_model)
        self.inject = _CrossBlock(d_model, d_model, n_heads, d_ff)

        # Its own rotary: `d_range` need not equal `d_model`, so the trunk's
        # head dimension is not this module's.
        self.rope = Qwen3RotaryEmbedding(config=Qwen3Config(
            hidden_size=d_range, num_attention_heads=n_heads,
            num_key_value_heads=n_heads, head_dim=d_range // n_heads,
            max_position_embeddings=cfg.get("max_decisions", 64)))

    def forward(self, hidden, batch, seat_emb):
        """`(logits, hidden')` — the belief, and the trunk states it went into.

        `logits` has one row per entry of `batch["act_idx"]`, in that order.
        """
        act = batch["act_idx"]
        B, T, d_model = hidden.shape
        P = self.max_players
        if act.numel() == 0:
            # No live opponent anywhere in the batch — a corpus of walkovers is
            # a legitimate batch, not an error. Nothing to predict, and the
            # identity to inject.
            return hidden.new_zeros((0, N_COMBOS)), hidden
        b, t, s = act[:, 0], act[:, 1], act[:, 2]

        # ---- queries: who (the seat's position embedding and that seat's
        # opponent vector) and when (the trunk state at `t`, and RoPE at `t`).
        q = hidden.new_zeros((B, T, P, self.d_range))
        q[b, t, s] = (self.pos_embed(s)
                      + self.q_in(torch.cat([hidden[b, t], seat_emb[b, t, s]],
                                            dim=-1)))
        q = q.reshape(B, T * P, self.d_range)

        # ---- the causal mask. Query `(t, s)` sees its own hand's real tokens
        # up to and including `t`; every other pair is off. A row for an
        # inactive seat is opened onto token 0 and its output discarded, because
        # a fully masked softmax is NaN and not zero.
        pos = torch.arange(T, device=hidden.device)
        causal = (pos.unsqueeze(1) >= pos.unsqueeze(0))            # (T, T)
        allow = (causal.unsqueeze(1)                                # (T,1,T)
                 & (batch["mask"] > 0).unsqueeze(1).unsqueeze(1))   # (B,1,1,T)
        allow = allow & batch["active"].unsqueeze(-1)               # (B,T,P,T)
        allow = allow.reshape(B, T * P, T).clone()
        allow[:, :, 0] |= ~allow.any(dim=-1)

        pos_q = pos.repeat_interleave(P).unsqueeze(0).expand(B, -1)
        cos_q, sin_q = self.rope(q, pos_q)
        kv_pos = pos.unsqueeze(0).expand(B, -1)
        cos_k, sin_k = self.rope(hidden, kv_pos)
        for blk in self.blocks:
            q = blk(q, hidden, allow, rope_q=(cos_q, sin_q),
                    rope_kv=(cos_k, sin_k))

        q = q.reshape(B, T, P, self.d_range)[b, t, s]
        logits = self.combo_out(self.norm_out(q))
        blocked = combo_block_mask(batch)[b, t]
        logits = logits.masked_fill(~blocked, float("-inf"))

        # ---- back into the trunk. The value is the probability vector,
        # detached: the layers above consume a belief they cannot bend, and the
        # head is trained by its own target alone.
        probs = torch.softmax(logits, dim=-1).detach()
        kv = hidden.new_zeros((B, T, P, d_model))
        kv[b, t, s] = self.value_in(probs) + self.pos_out(s)
        seats = batch["active"].reshape(B * T, P).clone()
        seats[:, 0] |= ~seats.any(dim=-1)
        injected = self.inject(hidden.reshape(B * T, 1, d_model),
                               kv.reshape(B * T, P, d_model),
                               seats.reshape(B * T, 1, P))
        injected = injected.reshape(B, T, d_model)
        keep = batch["active"].any(dim=-1, keepdim=True)
        return logits, torch.where(keep, injected, hidden)


def range_loss(logits, batch):
    """Cross-entropy of the §5.7 belief, and the KL that is readable next to it.

    Returns `(loss, ce, kl)` or `None` when the batch carries no target.

    **The absolute cross-entropy is almost all floor.** The support is 990–1225
    combos, so a perfect head still pays the target's own entropy, ~6.9–7.1
    nats. Read the KL — `ce − H(target)` — which is zero at the optimum and is
    the part the head can actually move. This is the §11.4 trap in a third
    place, after the strength head's MSE and the showdown head's.
    """
    if "range_target" not in batch or not bool(batch["range_mask"].any()):
        return None
    sel = batch["range_mask"]
    target = batch["range_target"][sel]
    log_p = torch.log_softmax(logits[sel].float(), dim=-1)
    # A blocked combo carries `-inf` here and exactly zero target mass, and
    # `0 · -inf` is NaN — so the term is dropped rather than multiplied out.
    log_p = torch.where(target > 0, log_p, torch.zeros_like(log_p))
    ce = -(target * log_p).sum(dim=-1).mean()
    entropy = -(target * torch.log(target.clamp_min(1e-30))).sum(dim=-1).mean()
    return ce, float(ce.detach()), float((ce - entropy).detach())
