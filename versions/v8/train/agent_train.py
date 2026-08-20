"""Training the agent on oracle labels (CONCEPT.md §6.2, §8).

One gradient step is: sample hands from the label set, tokenise-and-pad them
into a batch, read the logits at each hand's pending decision, and take the KL
to that decision's target. The observation is built by
`nets.features.hand_tokens` — the same builder the agent uses at deployment and
the embedding network uses at training — so there is no second construction path
that could drift (§9).

**Embedding dropout (§6.2).** With probability `p` a slot's vector is replaced
by zeros for a whole hand. This is what makes `e = 0` a *usable unconditional
policy* — the policy hero plays against an opponent it has not observed yet,
including the first hands of a Slumbot session — rather than "population average
at best, arbitrary at worst". The draw is per hand per slot and not per token:
a slot the agent is blind to has to be blind for every token of that hand.
§5.4 forbids the same dropout when training the embedding network, where it
would only destroy the signal the vector exists to carry.

**Cycles (owner decision 2026-08-18).** The outer loop of §8 is a cycle: the
agent trains, joins the pool, and the next agent trains against the enlarged
pool. Two consequences are the caller's contract and are written down here
because getting either wrong is invisible in the loss curve:

* **Iteration 0 is the hardest cycle and gets its own step count.** It starts
  from a random network, it is the only cycle whose labels come from a hero seat
  the agent did not occupy (§7.1), and it has no policy to inherit — so it needs
  the most gradient steps, and `first_iteration_steps` is a config key of its
  own rather than a multiplier applied to `steps`.
* **Every later cycle continues the previous one.** `train_agent` trains the
  module it is handed, in place; the caller passes the *same* agent from one
  iteration to the next, and the network it hands to iteration *k* is the one
  iteration *k−1* returned. Re-initialising between cycles would throw away
  every earlier best response and turn policy iteration into a sequence of
  unrelated fits. The optimiser and the cosine schedule are fresh per cycle —
  each cycle is a training run against a different label set, and carrying a
  decayed learning rate into it would leave the last cycles unable to move.
"""

import numpy as np
import torch

from nets.features import collate
from train.targets import kl_loss, soft_q_loss
from utils import progress


def steps_for_iteration(cfg, iteration, first_key="first_iteration_steps"):
    """Gradient steps for one training cycle.

    `first_key` on iteration 0, `steps` on every cycle after it. Omitting the
    key means the first cycle is trained like the rest.

    `first_key` is a parameter because the embedding network needs exactly the
    same rule under its own name (`first_retrain_steps`, §5.4): its first
    retrain is the one that starts from a random network *and* produces the
    vectors iteration 0's labels are stamped with, so it needs the most steps
    for the same reason iteration 0 of the agent does. Two copies of a
    three-line rule is the duplication `CLAUDE.md` §5 warns about.
    """
    steps = int(cfg["steps"])
    if int(iteration) == 0:
        return int(cfg.get(first_key, steps))
    return steps


def embedding_dropout(tables, p, generator=None):
    """Zero a slot's vector for a whole hand with probability `p` (§6.2).

    Args:
        tables: (B, n_slots, d_emb) — one slot → vector table per hand. The
            draw is taken here, before the tables are read per token, which is
            what makes it per hand per slot.
        p: dropout probability. `0` returns the tables untouched.
        generator: `torch.Generator` on the tables' device, so a run is
            reproducible.
    """
    if p <= 0.0:
        return tables
    keep = torch.rand(tables.shape[:2], generator=generator,
                      device=tables.device) >= p
    return tables * keep.unsqueeze(-1).to(tables.dtype)


def token_embeddings(tables, slot, d_emb):
    """(B, T, d_emb) — the acting player's vector on each token of each hand."""
    return tables.gather(1, slot.unsqueeze(-1).expand(-1, -1, d_emb))


def train_agent(net, hands, targets, embeddings, cfg, device, log, seed,
                iteration=0):
    """One training cycle of the agent (§6.2). Trains `net` in place.

    Args:
        net: the `AgentNet` being trained — the one the previous cycle
            returned, not a fresh one (see the module docstring).
        hands: list of `HandTokens`, each ending in the pending token of the
            decision that was labelled (`action[-1] == -1`).
        targets: (N, n_actions), one row per hand. What a row *is* depends on
            `cfg["loss"]`: a distribution from `train.targets.policy_target`
            under `kl`, and a normalised EV vector from
            `train.targets.normalised_q` under `soft_q` (§6.2). Both carry
            exact zeros off the decision's legal mask.
        embeddings: sequence of N tables of shape (max_players, d_emb) — the
            vectors the hand's seats were conditioned on. Frozen here: they are
            fitted by the embedding network (§5.5), not by this loss.
        cfg: the `agent_train` config section — `steps`,
            `first_iteration_steps`, `batch_hands`, `lr`, and optionally
            `loss` (`kl`, the default, or `soft_q`), `temperature` (required by
            `soft_q` and unused by `kl`, which has already spent it building
            the target), `weight_decay`, `eta_min`, `grad_clip`,
            `embedding_dropout`, `log_every`.
        iteration: which cycle of §8's loop this is. Selects the step count and
            nothing else.

    Returns the per-step loss history.
    """
    n = len(hands)
    assert n > 0, "cannot train on an empty label set"
    targets = np.asarray(targets, dtype=np.float64)
    assert targets.shape == (n, net.n_actions), (
        f"expected one target of {net.n_actions} actions per hand, got "
        f"{targets.shape} for {n} hands")
    assert len(embeddings) == n, (
        f"{len(embeddings)} embedding tables for {n} hands")
    assert np.asarray(embeddings[0]).shape == (net.max_players, net.d_emb), (
        f"each table is (max_players, d_emb) = "
        f"({net.max_players}, {net.d_emb}); got "
        f"{np.asarray(embeddings[0]).shape}")

    legal_last = np.stack([h.legal[-1] for h in hands])
    assert all(int(h.action[-1]) == -1 for h in hands), (
        "a labelled hand must end in the pending token of the decision the "
        "label is about (§9) — a hand whose last token carries an action is "
        "one the agent has already been told the answer to")
    loss_name = cfg.get("loss", "kl")
    assert loss_name in ("kl", "soft_q"), (
        f"unknown loss {loss_name!r}; choices are 'kl' and 'soft_q' (§6.2)")
    if loss_name == "kl":
        assert np.allclose(targets.sum(axis=1), 1.0, atol=1e-9), (
            "every target must be a distribution")
    else:
        assert np.isfinite(targets).all(), (
            "a normalised EV must be finite — `nan` marks illegal actions and "
            "`train.targets.normalised_q` zeroes them")
        temperature = float(cfg["temperature"])
    assert not (targets * ~legal_last).any(), (
        "a target carries a value on an action the environment called illegal")

    steps = steps_for_iteration(cfg, iteration)
    batch_hands = min(int(cfg["batch_hands"]), n)
    p_drop = float(cfg.get("embedding_dropout", 0.0))
    log(f"[agent] iteration {iteration}: {steps} steps over {n} labels, "
        f"batch {batch_hands} hands, loss {loss_name}, "
        f"embedding dropout {p_drop}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=cfg.get("eta_min", 0.0))
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    net.train()
    history = []
    for step in progress(range(1, steps + 1),
                         desc=f"agent it{iteration}", unit="step"):
        pick = rng.choice(n, size=batch_hands, replace=False)
        batch = collate([hands[i] for i in pick], device=device)
        tables = torch.as_tensor(
            np.stack([np.asarray(embeddings[i], dtype=np.float32)
                      for i in pick]), device=device)
        tables = embedding_dropout(tables, p_drop, generator=gen)
        logits = net(batch, token_embeddings(tables, batch["slot"], net.d_emb))

        rows = torch.arange(len(pick), device=logits.device)
        last = batch["mask"].sum(dim=1).long() - 1
        legal = batch["legal"][rows, last]
        target = torch.as_tensor(targets[pick], dtype=logits.dtype,
                                 device=logits.device)
        if loss_name == "kl":
            loss = kl_loss(logits, target, legal)
        else:
            loss = soft_q_loss(logits, target, legal, temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        # Both losses are a KL and both are zero at the optimum, so one key
        # carries either: `kl` reports `KL(target ‖ π)` and `soft_q` reports
        # `T·KL(π ‖ target)`.
        history.append({"step": step, "kl": float(loss.detach())})
        if step % cfg.get("log_every", 100) == 0 or step == 1:
            log(f"[agent] step {step}/{steps} "
                f"{loss_name}={history[-1]['kl']:.4f}")
    net.eval()
    return history
