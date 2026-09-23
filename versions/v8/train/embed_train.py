"""§5.4 training of the opponent-embedding network (CONCEPT.md §5.4, §8).

The baseline as §5.4 states it: **no inner loop.** The per-player embeddings are
an ordinary trainable table, one row per pool member, optimised jointly with the
transformer by the same optimiser; the gradient-descent fit of §5.5 exists only
at inference. Everything the objective contains — action cross-entropy, the two
§5.1a showdown terms, the amortised head's distillation — lives in
`nets/embedding_net.py::loss_terms`, so this file is the loop around it and
nothing more.

**One row per member, and the rows of the agents that do not exist yet.**
`n_members` is fixed when the table is constructed and §8 adds one member per
iteration, so the table is sized `len(pool₀) + max_iterations` up front and the
agent of iteration *k* owns row `len(pool₀) + k` (owner decision 2026-08-19,
`CONCEPT.md` §5.4). `max_iterations` is a config key of the `embedding_net`
section (§8.1) — this loop never decides how many rows there are, it is handed a
network that already has them. Retraining across iterations is therefore a
continuation: converged rows survive and an unoccupied row is a dead parameter
at its initialisation, because no token carries its index.

**No embedding dropout here** (§5.4, owner decision). Dropout applies to agent
training only, where it has a different purpose (§6.2).

**The first retrain gets its own step count** (`first_retrain_steps`), by the
same rule and the same helper as the agent's `first_iteration_steps`
(`train.agent_train.steps_for_iteration`). It is the only retrain that starts
from a random network, and it is the one whose vectors are stamped onto
iteration 0's labels — a network still near its initialisation would attach
noise to every one of them. Later retrains are continuations against a slowly
growing pool and are much shorter.

This was `gates/g1.py::train_embedding_net`, moved so that the pipeline depends
on nothing but the v7 checkpoints the first cycle needs. G1 imports it back and
`test_g1_gate.py` passing unedited is what says the move changed no behaviour.
"""

import numpy as np
import torch

from nets.embedding_net import loss_weights
from nets.features import TOKEN_DECISION, TOKEN_SHOWDOWN, collate
from train.agent_train import steps_for_iteration
from utils import progress


def train_embedding_net(net, sessions, cfg, game, device, log, seed,
                        iteration=0):
    """§5.4 baseline: no inner loop, table and transformer trained jointly.

    `iteration` selects the step count and nothing else: `first_retrain_steps`
    on the first retrain, `steps` on every one after it (see the module
    docstring). It defaults to 0 so a caller with a single training run — G1 —
    is unchanged.
    """
    max_players = game["max_players"]
    n_actions = game["n_actions"]

    corpus = []
    for s in sessions:
        corpus.extend(t for t in s.tokens(max_players, n_actions) if len(t) > 0)
    n_decision = sum(int((t.token_type == TOKEN_DECISION).sum()) for t in corpus)
    n_showdown = sum(int((t.token_type == TOKEN_SHOWDOWN).sum()) for t in corpus)
    # §5.6: a corpus whose records were never labelled carries no strength
    # target at all, and the head would then train on nothing while the loss
    # curve looked fine. The count says so out loud instead.
    n_strength = sum(int((t.own_strength >= 0).sum()) for t in corpus)
    log(f"[train] corpus: {len(corpus)} hands, {n_decision} decision tokens, "
        f"{n_showdown} showdown tokens, {n_strength} strength targets")
    if not corpus:
        log("[train] no complete corpus hands; skipping embedding update")
        net.eval()
        return []

    steps = steps_for_iteration(cfg, iteration, "first_retrain_steps")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=cfg.get("eta_min", 0.0))
    rng = np.random.default_rng(seed)
    batch_hands = min(cfg["batch_hands"], len(corpus))
    weights = loss_weights(cfg)

    net.train()
    history = []
    for step in progress(range(1, steps + 1), desc="train", unit="step"):
        pick = rng.choice(len(corpus), size=batch_hands, replace=False)
        batch = collate([corpus[i] for i in pick], device=device)
        total, parts = net.loss_terms(batch, weights)

        opt.zero_grad(set_to_none=True)
        total.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        if step % cfg.get("log_every", 100) == 0 or step == 1:
            log(f"[train] step {step}/{steps} "
                + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            history.append({"step": step, **parts})
    net.eval()
    return history
