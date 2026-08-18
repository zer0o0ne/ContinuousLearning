"""Entity 4 — the opponent-embedding network (CONCEPT.md §5).

A transformer that predicts what each player did, with all cross-hand
information forced through a single per-player vector. That vector is the whole
point: at deployment it is the only thing hero carries from one hand to the
next about an opponent.

**The attention structure (§5.2).** Within a hand, causal. Across hands, zero.
The second half is not implemented as a mask over a concatenated sequence — with
cross-hand attention cut, hands are conditionally independent given the
embedding, so they are simply the batch dimension. That is the same object as a
block-diagonal mask and is cheaper; §5.2 says as much ("hands are a batch
dimension, not a sequence"). Everything §5.2 lists as a consequence follows:
cost is linear in the number of hands, subsampling hands per gradient step is
legitimate, and the embedding is the only channel between hands.

Causality within the hand is what stops the task from being trivial: token *t*
carries the action taken at token *t−1*, so a non-causal token *t* would read
its own answer out of token *t+1*.

**Training (§5.4).** No inner loop. Per-player embeddings are an ordinary
trainable table, one row per pool member, optimised jointly with the transformer
by the same optimiser. The gradient fit exists only at inference (§5.5), which
leaves a train/deploy mismatch — at training the vector is converged, at
deployment it is *K* steps from its initialisation. The amortised head is the
mitigation: a small network from an observed history to a starting vector,
trained by MSE distillation onto the converged table row. Without that target it
would have no training signal at all under the no-inner-loop baseline.

The head reads a forward pass taken with **e = 0**, because that is the only
thing available at inference before any vector exists. That costs a second
forward per training step, and it is the reason `loss_terms` runs the trunk
twice.

**Showdowns (§5.1a).** Two extra heads read the terminal tokens and predict what
a revealed player showed: a board-relative strength percentile (MSE) and the
board-independent 169-way preflop class (cross-entropy). They matter twice over,
and the second time is the one that is easy to miss:

* in **training**, they shape the weights — the only objective in the model that
  ties a line to an actual holding;
* in the **inference-time fit**, they are part of the objective the vector is
  fitted against. Without that, `fit_embeddings` would optimise action
  cross-entropy alone and pull the vector straight off whatever the showdowns
  said. v7 could get away with a training-only probe because its embedding was a
  GRU state produced by a forward pass; v8's is a latent found by gradient
  descent at deployment, so the information has to be in *that* gradient.

Both terms are computable at fit time from what hero saw: the hands being fitted
are finished, and their showdowns are public.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.showdown import N_HAND_CLASSES
from nets.trunk import HandEncoder


def pool_by_key(hidden, key, mask, n_keys):
    """Mean-pool `hidden` over the tokens sharing each key.

    Returns ``(pooled (n_keys, d), counts (n_keys,))``. Keys with no tokens get
    a zero vector and a zero count, which is the honest "nothing observed"
    answer and matches the cold start of §5.5.
    """
    d = hidden.shape[-1]
    flat_h = hidden.reshape(-1, d)
    flat_key = key.reshape(-1)
    flat_m = mask.reshape(-1).to(flat_h.dtype)

    sums = torch.zeros(n_keys, d, dtype=flat_h.dtype, device=flat_h.device)
    sums = sums.index_add(0, flat_key, flat_h * flat_m.unsqueeze(-1))
    counts = torch.zeros(n_keys, dtype=flat_h.dtype, device=flat_h.device)
    counts = counts.index_add(0, flat_key, flat_m)
    return sums / counts.clamp(min=1.0).unsqueeze(-1), counts


class OpponentEmbeddingNet(nn.Module):
    """Per-hand encoder + per-player embedding table + amortised head."""

    def __init__(self, cfg, n_actions, max_players, n_members):
        super().__init__()
        d_model = cfg["d_model"]
        d_emb = cfg["d_emb"]

        self.d_model = d_model
        self.d_emb = d_emb
        self.n_actions = n_actions
        self.max_players = max_players
        self.n_members = n_members

        # §5.1/§5.2 trunk, shared as code with the agent (OI-4, `nets/trunk.py`).
        self.encoder = HandEncoder(cfg, n_actions, max_players)
        self.action_out = nn.Linear(d_model, n_actions)
        # §5.1a — the two showdown heads. Read only on terminal tokens.
        self.showdown_strength_out = nn.Linear(d_model, 1)
        self.showdown_class_out = nn.Linear(d_model, N_HAND_CLASSES)

        # §5.4: an ordinary trainable table, one vector per pool member. Hero
        # has one too — every seated player is just a member here.
        self.embeddings = nn.Embedding(n_members, d_emb)
        nn.init.normal_(self.embeddings.weight, std=cfg.get("emb_init_std", 0.02))

        d_amort = cfg.get("d_amortised_hidden", d_model)
        self.amortised = nn.Sequential(
            nn.Linear(d_model, d_amort), nn.GELU(), nn.Linear(d_amort, d_emb))

    # ------------------------------------------------------------------ views

    def member_emb(self, batch):
        """Embeddings from the trainable table (training, §5.4)."""
        return self.embeddings(batch["member"])

    def slot_emb(self, batch, vectors):
        """Embeddings from a free parameter set (inference fit, §5.5)."""
        return vectors[batch["slot"]]

    def zero_emb(self, batch):
        """The `e = 0` baseline, and the cold start of §5.5."""
        B, T = batch["mask"].shape
        return torch.zeros(B, T, self.d_emb, device=batch["mask"].device,
                           dtype=self.embeddings.weight.dtype)

    # --------------------------------------------------------------- forwards

    def hidden(self, batch, emb):
        """(B, T, d_model) — one causal pass over each hand independently."""
        return self.encoder(batch, emb)

    def forward(self, batch, emb):
        """(B, T, n_actions) — the action predicted at each decision token."""
        return self.action_out(self.hidden(batch, emb))

    # ------------------------------------------------------------------ losses

    def action_ce(self, logits, batch, per_token=False):
        """Cross-entropy of the predicted action against the real one.

        Scored on **decision tokens only** — a showdown token took no action.

        Illegal actions are masked out before the softmax, using the same
        legality rule the driver sampled with (`env.legal`) — predicting mass on
        an action the player could not take is not an error the network should
        be asked to make.
        """
        mask = batch["decision_mask"]
        off = (mask == 0).unsqueeze(-1)
        legal = batch["legal"] | off          # unscored tokens: allow everything,
        masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        ce = F.cross_entropy(
            masked.reshape(-1, self.n_actions),
            batch["action"].reshape(-1),
            reduction="none",
        ).reshape(mask.shape)
        if per_token:
            return ce * mask
        return (ce * mask).sum() / mask.sum().clamp(min=1.0)

    def showdown_losses(self, hidden, batch):
        """(strength MSE, class CE) over the terminal tokens (§5.1a).

        Returns ``(None, None)`` when the batch contains no showdown at all —
        a corpus of fold-outs is a legitimate batch, not an error.
        """
        sel = batch["showdown_mask"] > 0
        if not bool(sel.any()):
            return None, None
        h = hidden[sel]
        strength = F.mse_loss(self.showdown_strength_out(h).squeeze(-1),
                              batch["sd_strength"][sel])
        cls = F.cross_entropy(self.showdown_class_out(h),
                              batch["sd_class"][sel])
        return strength, cls

    def objective(self, batch, emb, weights, hidden=None):
        """The loss the embedding is fitted against — training and inference.

        One function so the two cannot drift apart. §5.5's inference fit
        optimises exactly what §5.4's training optimises, restricted to the
        vectors; if the showdown terms appeared in only one of them, the fitted
        vector would be pulled off whatever the showdowns said.

        Returns ``(total, parts)``.
        """
        if hidden is None:
            hidden = self.hidden(batch, emb)
        ce = self.action_ce(self.action_out(hidden), batch)
        total = ce
        parts = {"action_ce": float(ce.detach())}

        strength, cls = self.showdown_losses(hidden, batch)
        if strength is not None:
            w_s = weights.get("showdown_strength", 0.0)
            w_c = weights.get("showdown_class", 0.0)
            parts["showdown_strength_mse"] = float(strength.detach())
            parts["showdown_class_ce"] = float(cls.detach())
            total = total + w_s * strength + w_c * cls
        return total, parts

    def loss_terms(self, batch, weights):
        """Training losses for one batch of hands.

        Returns ``(total, parts)``. Two trunk passes: one with the fitted table
        rows for the prediction and showdown losses, one with `e = 0` whose
        pooled hidden states are the amortised head's input (see the module
        docstring).
        """
        total, parts = self.objective(batch, self.member_emb(batch), weights)

        amortised_weight = weights.get("amortised", 0.0)
        if amortised_weight > 0.0:
            hidden0 = self.hidden(batch, self.zero_emb(batch))
            pooled, counts = pool_by_key(hidden0, batch["member"], batch["mask"],
                                         self.n_members)
            seen = counts > 0
            if seen.any():
                pred = self.amortised(pooled[seen])
                target = self.embeddings.weight[seen].detach()
                amort = F.mse_loss(pred, target)
                parts["amortised_mse"] = float(amort.detach())
                total = total + amortised_weight * amort
        return total, parts

    # ---------------------------------------------------------- inference fit

    def amortised_init(self, batch, n_slots):
        """(n_slots, d_emb) starting vectors from the observed history (§5.5.1).

        The pooling key is the *slot*, not the member: at inference a player is a
        seat at the observed table and its pool-member identity is unknown.
        """
        with torch.no_grad():
            hidden0 = self.hidden(batch, self.zero_emb(batch))
            pooled, counts = pool_by_key(hidden0, batch["slot"], batch["mask"],
                                         n_slots)
            init = self.amortised(pooled)
            # Cold start (§5.5): a slot with no observed decision gets zero.
            return init * (counts > 0).unsqueeze(-1).to(init.dtype)


def loss_weights(cfg):
    """The §5.1a / §5.4 auxiliary-loss weights, from one place.

    `showdown_strength` and `showdown_class` weight the two showdown heads and
    are used **both** in training and in the inference-time fit — setting either
    to 0 is the ablation that answers "does the showdown anchor earn its keep".
    """
    return {
        "amortised": cfg.get("amortised_weight", 1.0),
        "showdown_strength": cfg.get("showdown_strength_weight", 0.0),
        "showdown_class": cfg.get("showdown_class_weight", 0.0),
    }


def fit_embeddings(net, batch, n_slots, steps, lr, reg, init=None,
                   weights=None):
    """*K* gradient steps on the vectors only (CONCEPT.md §5.5).

    Transformer weights are frozen; all slots are fitted **jointly** in one
    optimisation, because §5.3 leaves the players' embeddings coupled — a
    prediction for player *i* depends on the others' vectors.

    Deterministic: full-batch gradient descent with no sampling anywhere, so the
    same inputs give the same vectors.

    Args:
        steps: *K*. ``0`` is the §5.4 ablation — the amortised head's output
            used as-is, with no gradient fit at all.
        reg: coefficient of the pull toward zero.
        init: (n_slots, d_emb) starting point; the amortised head's output when
            omitted.
        weights: the showdown-term weights (§5.1a). Omitted means action
            cross-entropy alone, which is the pre-showdown behaviour.
    """
    device = batch["mask"].device
    if init is None:
        init = net.amortised_init(batch, n_slots)
    vectors = init.detach().clone().to(device)
    if steps <= 0:
        return vectors

    weights = weights or {}
    was_trainable = [p.requires_grad for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        vectors.requires_grad_(True)
        opt = torch.optim.Adam([vectors], lr=lr)
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss, _parts = net.objective(
                batch, net.slot_emb(batch, vectors), weights)
            loss = loss + reg * vectors.pow(2).sum(-1).mean()
            loss.backward()
            opt.step()
    finally:
        for p, flag in zip(net.parameters(), was_trainable):
            p.requires_grad_(flag)
    return vectors.detach()


@torch.no_grad()
def evaluate_ce_by_member(net, batch, emb, members):
    """Mean action-prediction CE in nats for each of `members`, in one forward.

    `evaluate_ce` runs the trunk once per member it is asked about. Scoring
    every seat of a 9-handed table under every measured condition then costs
    nine identical passes over the same batch, and the evaluation loop is where
    G1 spends most of its time. The per-token losses do not depend on which
    member is being scored, so one forward answers for all of them and only the
    masked average is repeated.

    Returns ``{member: (ce, n_tokens)}``; a member with no scored token gets
    ``(nan, 0)``, exactly as `evaluate_ce` does.
    """
    per_token = net.action_ce(net(batch, emb), batch, per_token=True)
    mask = batch["decision_mask"]
    out = {}
    for m in members:
        weight = mask * (batch["member"] == m).to(mask.dtype)
        total = weight.sum()
        if float(total) == 0.0:
            out[m] = (float("nan"), 0)
        else:
            out[m] = (float((per_token * weight).sum() / total), int(total))
    return out


@torch.no_grad()
def evaluate_ce(net, batch, emb, member_filter=None):
    """Mean action-prediction CE in nats, optionally over one player's tokens.

    Args:
        emb: (B, T, d_emb) embeddings to condition on.
        member_filter: pool-member index; when given, only that player's
            decision tokens count.
    """
    logits = net(batch, emb)
    per_token = net.action_ce(logits, batch, per_token=True)
    weight = batch["decision_mask"]
    if member_filter is not None:
        weight = weight * (batch["member"] == member_filter).to(weight.dtype)
    total = weight.sum()
    if float(total) == 0.0:
        return float("nan"), 0
    return float((per_token * weight).sum() / total), int(total)
