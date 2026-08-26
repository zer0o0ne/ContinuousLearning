"""The outer loop (CONCEPT.md §8, `PLAN_PIPELINE.md` S9).

Everything before this file produces one artefact each; this is the file that
turns them into a training run:

```
iteration k:
  A  retrain the embedding network on pool self-play          (§5.4, every `retrain_every`)
     ↳ at k = 0 only, warm-start the agent's trunk from it    (§6.1 OI-4, §5.6)
  B  play, fit vectors, label hero's decisions                (§8 lines 1–4, train/generate.py)
  C  train the agent on all but `heldout_fraction` of them    (§6.2)
  D  measure the oracle gap on the held-out slice             (§8)
  E  age the PFSP results and append the agent to the pool    (§4.4, §4.1)
```

Only the last two lines are new logic; the rest is composition, and deliberately
so — a phase that reimplements what a module already does is how the state
distribution the agent trains on and the one anything else measures drift apart.

**Iteration 0 differs in exactly one way** (§7.1, §8): the agent is trained from
scratch, and a v7 pool member — named by the `agent_init` section, which is
deliberately *not* part of `bootstrap` — sits in hero's seat, so the labelled
states come from a competent policy instead of a random walk. From iteration 1
hero is the agent and the labels are on-policy. Everything else about iteration 0
is ordinary, including that its training continues into iteration 1's rather than
being restarted (§8, `train/agent_train.py`).

**The embedding network is the first phase, not a later one** (§8: "can be
trained before any v8 agent exists, on hands played by pool members among
themselves. That is both gate G1 and the first pipeline phase"). It is therefore
run at the *start* of an iteration and not at the end of one — the labels of
iteration 0 carry the vectors it fitted, and a network still at its
initialisation would attach noise to every one of them. That ordering is also
what makes the §6.1 warm start possible at all: by the time the agent is built
there is a trained trunk over the *same* tokeniser and the *same* token format,
carrying the §5.6 poker prior, and `agent_train.warm_start_trunk` copies it into
the agent once. Its corpus is pool
self-play, replayed at each retrain over the pool *as it stands*, which is what
grows: iteration k's corpus contains the agents of iterations 0 … k−1 as
opponents, which is the case §5.4 cares about ("the network never learns to read
the one opponent it is guaranteed to face"). Replaying rather than accumulating
is what makes a resumed run byte-identical to an uninterrupted one without
carrying every hand ever played on disk.

**Row *i* of the embedding table is pool member *i*.** The table is sized
`len(pool₀) + max_iterations × agent_variants` up front (D9, §5.4) and the pool
grows by exactly `agent_variants` members per iteration (D11), so the two indices
coincide by construction and iteration k's block starts at
`len(pool₀) + k × agent_variants`. A row nobody occupies yet is a dead parameter
at its initialisation, because no token carries its index.

**Resume is per phase, not per iteration.** Every phase writes its artefact
before the next one starts and skips itself if that artefact is already there; a
run on the Spark is measured in days (§13) and a crash in phase C must not cost
phase B. What makes the resumed run *identical* rather than merely valid is that
no phase reads a running RNG: every stream is seeded from `(seed, iteration)`,
and the one piece of genuinely sequential state — the sampler, whose draws are
consumed in phase B — is written out with phase B's own artefact.

Run::

    ./run.sh --version=v8            # → cd versions/v8 && python3 pipeline.py
"""

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from agent.policy import AgentPoolMember, FrozenAgentMember
from env.driver import LockstepDriver
from env.session import (assert_seeds_stay_distinct, build_sessions,
                         hand_seed_bases, phase_hands, play)
from gates.g1 import _stack_bucket
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from nets.features import collate
from pool.build import build_pool
from pool.sampling import PoolSampler
from pool.style import StyleParams, sample_style
from train.agent_train import token_embeddings, train_agent
from train.embed_train import train_embedding_net
from train.generate import generate_labels, load_shard
from train.targets import normalised_q, policy_target
from utils import Logger, progress, resolve_device

TAG = "loop"

def _iter_dir(exp_dir, iteration):
    return os.path.join(exp_dir, f"iter_{int(iteration):04d}")


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)


def _iteration_seed(seed, iteration):
    """The seed every stream of one iteration is derived from.

    Derived and not carried: a resumed run re-enters iteration k with the same
    number here regardless of which phases it re-ran.
    """
    return int(seed) * 1_000 + int(iteration)


# --------------------------------------------------------------- the oracle gap

#: What `gap_terms` reports and `oracle_gap` averages, in the order §8 lists it.
GAP_KEYS = ("kl", "ev_agent", "ev_oracle", "q_best",
            "ev_gap_target", "ev_gap_greedy", "agreement")


def gap_terms(q_norm, pi_oracle, log_pi_agent, legal):
    """The four §8 numbers for one decision.

    Args:
        q_norm: (n_actions,) `Q` in §6.2's pot-normalised units, exact zero off
            `legal` (`train.targets.normalised_q`).
        pi_oracle: (n_actions,) the target the agent was trained toward —
            `softmax(Q_norm / T)`, exact zero off `legal`.
        log_pi_agent: (n_actions,) the agent's masked log-probabilities. `-inf`
            off `legal`, finite on it.
        legal: (n_actions,) bool.

    Returns a dict with `kl`, `ev_agent`, `ev_oracle`, `q_best`,
    `ev_gap_target`, `ev_gap_greedy`, `agreement`. The two gaps are differences
    of the three `ev_*`/`q_best` terms and those terms are reported alongside
    them, because a gap that moved does not say which of its two sides moved.
    §8 defines all four and, importantly, what they are not: they are scored by
    the oracle's own noisy `Q` on hero's own state distribution, so none of them
    bounds anything about exploitability and the floor of the two gaps is G3's
    Monte-Carlo error rather than zero.
    """
    legal = np.asarray(legal, dtype=bool)
    q_norm = np.asarray(q_norm, dtype=np.float64)
    pi_o = np.asarray(pi_oracle, dtype=np.float64)
    logp_a = np.asarray(log_pi_agent, dtype=np.float64)
    assert legal.any(), "a decision with no legal action is not a decision"
    assert np.isfinite(logp_a[legal]).all(), (
        "the agent's log-probability is not finite on a legal action")

    pi_a = np.zeros_like(pi_o)
    pi_a[legal] = np.exp(logp_a[legal])

    nz = pi_o > 0.0
    kl = float((pi_o[nz] * (np.log(pi_o[nz]) - logp_a[nz])).sum())
    ev_agent = float((pi_a * q_norm).sum())
    ev_oracle = float((pi_o * q_norm).sum())
    best = float(q_norm[legal].max())
    return {
        "kl": kl,
        "ev_agent": ev_agent,
        "ev_oracle": ev_oracle,
        "q_best": best,
        "ev_gap_target": ev_oracle - ev_agent,
        "ev_gap_greedy": best - ev_agent,
        "agreement": float(int(np.argmax(pi_a)) == int(np.argmax(pi_o))),
    }


@torch.no_grad()
def oracle_gap(net, labels, temperature, divisor, batch_hands, device, log):
    """§8's held-out measurement of one checkpoint against its own oracle.

    One agent forward per held-out decision and **no new rollouts** — the
    oracle's answer is already in the shard, so this is a softmax over stored
    `q` and a batched forward, not a second labelling pass.

    Returns `{n_heldout, overall, by_table_size, by_stack_bb}`; a run with no
    held-out labels reports `n_heldout = 0` and no numbers at all, rather than a
    gap measured on the data the optimiser just saw.
    """
    if not labels:
        log(f"[{TAG}] no held-out labels — the oracle gap is not measured")
        return {"n_heldout": 0}

    net.eval()
    rows = []
    bar = progress(total=len(labels), desc="gap", unit="label")
    for lo in range(0, len(labels), int(batch_hands)):
        chunk = labels[lo:lo + int(batch_hands)]
        batch = collate([lab["tokens"] for lab in chunk], device=device)
        tables = torch.as_tensor(
            np.stack([np.asarray(lab["embeddings"], dtype=np.float32)
                      for lab in chunk]), device=device)
        logits = net(batch, token_embeddings(tables, batch["slot"], net.d_emb))
        idx = torch.arange(len(chunk), device=logits.device)
        last = batch["mask"].sum(dim=1).long() - 1
        legal = batch["legal"][idx, last]
        logp = torch.log_softmax(
            logits.masked_fill(~legal, float("-inf")), dim=-1)
        logp = logp.double().cpu().numpy()

        for j, lab in enumerate(chunk):
            mask = np.asarray(lab["legal"], dtype=bool)
            args = (lab["q"], mask, lab["pot_bb"], lab["facing_bet_bb"])
            terms = gap_terms(
                normalised_q(*args, divisor=divisor),
                policy_target(*args, temperature=temperature, divisor=divisor),
                logp[j], mask)
            rows.append({**terms, "num_players": int(lab["num_players"]),
                         "stack_bb": int(lab["stack_bb"])})
        bar.update(len(chunk))
    bar.close()

    def summarise(subset):
        return {"n": len(subset),
                **{k: float(np.mean([r[k] for r in subset]))
                   for k in GAP_KEYS}}

    def grouped(key):
        out = {}
        for value in sorted({r[key] for r in rows}):
            out[str(value)] = summarise([r for r in rows if r[key] == value])
        return out

    report = {
        "n_heldout": len(rows),
        "overall": summarise(rows),
        "by_table_size": grouped("num_players"),
        "by_stack_bb": {b: summarise([r for r in rows
                                      if _stack_bucket(r["stack_bb"]) == b])
                        for b in sorted({_stack_bucket(r["stack_bb"])
                                         for r in rows})},
    }
    o = report["overall"]
    log(f"[{TAG}] oracle gap over {len(rows)} held-out labels: "
        f"kl={o['kl']:.4f} ev_agent={o['ev_agent']:+.4f} "
        f"ev_oracle={o['ev_oracle']:+.4f} q_best={o['q_best']:+.4f} "
        f"ev_gap_target={o['ev_gap_target']:+.4f} "
        f"ev_gap_greedy={o['ev_gap_greedy']:.4f} "
        f"agreement={o['agreement']:.3f}")
    return report


# ------------------------------------------------------------------ the phases


def agent_variant_members(net, iteration, config, game, device, seed):
    """The `style.agent_variants` members one trained agent contributes (D11).

    Variant 0 is the agent unmodified — the reference point every style draw is
    a perturbation of, and the same rule D13 applies to every v7 base. The rest
    are `with_style` siblings sharing this one network by reference, so they cost
    no forward and no parameter (§4.2).
    """
    style_cfg = config.get("style", {})
    n_variants = int(style_cfg["agent_variants"])
    base = FrozenAgentMember(f"agent{iteration}", int(game["n_actions"]), net,
                             int(game["max_players"]), device,
                             StyleParams.identity())
    members = [base]
    rng = np.random.default_rng([int(seed), int(iteration), 991])
    for v in range(1, n_variants):
        members.append(base.with_style(f"agent{iteration}#{v}",
                                       sample_style(rng, style_cfg)))
    descriptors = [{"name": m.name, "kind": "agent", "base": f"agent{iteration}",
                    "style": m.style.to_list()} for m in members]
    return members, descriptors


def warm_start_trunk(agent_net, embed_net, log):
    """Copy the embedding network's trunk into the agent's (§6.1, OI-4, §5.6).

    Both networks are built from the same `embedding_net` config section over
    the same tokeniser and the same §5.1 token, so `HandEncoder` state dicts
    correspond exactly — that is what OI-4 bought by sharing the trunk as code.
    They keep separate weights, so this is a one-time initialisation and not a
    tie: the agent's trunk moves under `soft_q` from the first gradient step and
    the two diverge immediately.

    What it carries is the §5.6 poker prior — hand evaluation, board texture,
    the value of a draw — learned on a corpus of free self-play. Without it the
    agent has to learn all of that from oracle labels whose noise at depth is
    larger than the EV differences they are meant to teach.
    """
    agent_net.encoder.load_state_dict(embed_net.encoder.state_dict())
    log(f"[{TAG}] agent trunk warm-started from the embedding network "
        f"({sum(p.numel() for p in agent_net.encoder.parameters())/1e6:.2f}M "
        f"parameters); the action head stays at its initialisation")


def frozen_agent_net(state_dict, config, game, device):
    """A past agent's network: loaded, frozen, and never trained again."""
    net = AgentNet(config["embedding_net"], int(game["n_actions"]),
                   int(game["max_players"])).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def hero_factory(iteration, agent_net, init_member, game, device):
    """Who sits in hero's seat while this iteration's labels are generated.

    Iteration 0 seats the `agent_init` pool member (§7.1); every later iteration
    seats the agent itself, which is what makes the labelled state distribution
    on-policy. The factory shape is `train/generate.py`'s: one member per seat,
    rebuilt whenever the fitted vectors are refreshed.
    """
    if int(iteration) == 0:
        return lambda observer_pos, slot_of_seat, embeddings: init_member
    return lambda observer_pos, slot_of_seat, embeddings: AgentPoolMember(
        agent_net, embeddings, slot_of_seat, int(game["max_players"]),
        int(game["n_actions"]), observer_pos, device)


def embedding_phase(embed_net, pool, config, game, device, log, seed, iteration):
    """§5.4 training over a fresh corpus of pool self-play (phase A)."""
    emb_cfg = config["embedding_net"]
    it_seed = _iteration_seed(seed, iteration)
    sessions = build_sessions(
        np.random.default_rng([int(seed), int(iteration), 11]),
        list(range(len(pool))), game, int(emb_cfg["corpus_sessions"]),
        int(emb_cfg["corpus_hands_per_session"]),
        seed_base=hand_seed_bases(it_seed, phase_hands(config))[0]["corpus"],
        tag="corpus")
    driver = LockstepDriver(pool, int(game["n_actions"]))
    play(driver, sessions, int(config["driver_batch_size"]), log, "corpus")
    torch.manual_seed(it_seed + 500_000)
    return train_embedding_net(embed_net, sessions, emb_cfg, game, device, log,
                               it_seed + 500_000, iteration=iteration)


def build_targets(labels, loss, temperature, divisor):
    """One target row per label — a distribution under `kl`, an EV under `soft_q`.

    §6.2's two losses consume different payloads and `train/targets.py` builds
    both out of one normalisation, so the branch is here and nowhere else.
    """
    out = []
    for lab in labels:
        args = (lab["q"], np.asarray(lab["legal"], dtype=bool), lab["pot_bb"],
                lab["facing_bet_bb"])
        if loss == "kl":
            out.append(policy_target(*args, temperature=temperature,
                                     divisor=divisor))
        else:
            out.append(normalised_q(*args, divisor=divisor))
    return np.stack(out) if out else np.zeros((0, 0))


def split_heldout(n, fraction, seed, iteration):
    """(train, held-out) index arrays. Deterministic in `(seed, iteration)`."""
    rng = np.random.default_rng([int(seed), int(iteration), 23])
    perm = rng.permutation(int(n))
    n_held = int(round(float(fraction) * int(n)))
    assert n_held < n, (
        f"heldout_fraction {fraction} would withhold every one of {n} labels — "
        f"there would be nothing left to train on")
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


# -------------------------------------------------------------------- the loop


def run(config, log, exp_dir):
    game = config["game"]
    seed = int(config.get("seed", 0))
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")
    os.makedirs(exp_dir, exist_ok=True)

    emb_cfg = config["embedding_net"]
    style_cfg = config.get("style", {})
    train_cfg = dict(config["agent_train"])
    oracle_cfg = config["oracle"]
    temperature = float(oracle_cfg["temperature"])
    divisor = oracle_cfg.get("divisor", "pot_plus_bet")
    # §6.2's temperature is one number: the one that builds the `kl` target is
    # the one the `soft_q` loss is written in. It lives in the `oracle` section
    # (§8.1) and is handed to the trainer rather than configured twice.
    train_cfg["temperature"] = temperature
    loss = train_cfg.get("loss", "kl")

    n_iterations = int(config["n_iterations"])
    agent_variants = int(style_cfg["agent_variants"])
    max_iterations = int(emb_cfg["max_iterations"])
    retrain_every = int(emb_cfg["retrain_every"])
    assert n_iterations <= max_iterations, (
        f"the embedding table reserves rows for {max_iterations} iterations "
        f"(§5.4, D9) and the run asks for {n_iterations}")
    hands = phase_hands(config)
    _bases, span = hand_seed_bases(0, hands)
    assert_seeds_stay_distinct(span, n_iterations)
    log(f"[{TAG}] hand seeds: {span} per iteration "
        + ", ".join(f"{p} {hands[p]}" for p in hands))

    rng = np.random.default_rng(seed)
    pool, descriptors = build_pool(config, rng, device=device, log=log)
    n_pool0 = len(pool)
    n_members = n_pool0 + max_iterations * agent_variants
    log(f"[{TAG}] pool: {n_pool0} bootstrap members, embedding table reserves "
        f"{n_members} rows ({max_iterations} × {agent_variants} for the agents)")

    # §7.1: hero's seat at iteration 0. Built through `build_pool` so it is an
    # ordinary member and there is one construction path, not two.
    init_members, init_desc = build_pool(
        {"bootstrap": [config["agent_init"]], "game": game, "style": style_cfg},
        np.random.default_rng(seed + 1), device=device, log=log)
    assert len(init_members) == 1, (
        f"`agent_init` names the single member that sits in hero's seat at "
        f"iteration 0 (§7.1); this entry produced {len(init_members)}")
    init_member = init_members[0]
    log(f"[{TAG}] agent_init: {init_member.name!r} (not a pool member)")

    torch.manual_seed(seed)
    embed_net = OpponentEmbeddingNet(emb_cfg, int(game["n_actions"]),
                                     int(game["max_players"]),
                                     n_members=n_members).to(device)
    torch.manual_seed(seed + 1)
    agent_net = AgentNet(emb_cfg, int(game["n_actions"]),
                         int(game["max_players"])).to(device)
    log(f"[{TAG}] embedding network "
        f"{sum(p.numel() for p in embed_net.parameters())/1e6:.2f}M, agent "
        f"{sum(p.numel() for p in agent_net.parameters())/1e6:.2f}M parameters")

    # ------------------------------------------------------------- resume
    start = 0
    while (start < n_iterations
           and os.path.exists(os.path.join(_iter_dir(exp_dir, start),
                                           "state.json"))):
        start += 1
    for k in range(start):
        it_dir = _iter_dir(exp_dir, k)
        state = torch.load(os.path.join(it_dir, "agent.pt"), map_location=device,
                           weights_only=False)["model_state_dict"]
        members, desc = agent_variant_members(
            frozen_agent_net(state, config, game, device), k, config, game,
            device, seed)
        pool += members
        descriptors += desc
    if start:
        agent_net.load_state_dict(torch.load(
            os.path.join(_iter_dir(exp_dir, start - 1), "agent.pt"),
            map_location=device, weights_only=False)["model_state_dict"])
    latest_emb = max((j for j in range(start + 1)
                      if os.path.exists(os.path.join(_iter_dir(exp_dir, j),
                                                     "embedding.pt"))),
                     default=None)
    if latest_emb is not None:
        embed_net.load_state_dict(torch.load(
            os.path.join(_iter_dir(exp_dir, latest_emb), "embedding.pt"),
            map_location=device, weights_only=False)["model_state_dict"])
    if start:
        log(f"[{TAG}] resuming at iteration {start}: pool is {len(pool)} "
            f"members, embedding network from iteration {latest_emb}")

    metrics_all = []
    for k in range(start, n_iterations):
        it_dir = _iter_dir(exp_dir, k)
        os.makedirs(it_dir, exist_ok=True)
        log(f"[{TAG}] ===== iteration {k}: pool {len(pool)} members =====")
        timings = {}

        sampler = PoolSampler(len(pool), config["pool_sampling"],
                              np.random.default_rng([seed, k, 7]))
        if k:
            sampler.load_state_dict(
                _read_json(os.path.join(_iter_dir(exp_dir, k - 1),
                                        "state.json"))["sampler"])

        # ------------------------------------------------- A: embedding network
        emb_path = os.path.join(it_dir, "embedding.pt")
        if k % retrain_every == 0:
            if os.path.exists(emb_path):
                embed_net.load_state_dict(torch.load(
                    emb_path, map_location=device,
                    weights_only=False)["model_state_dict"])
                log(f"[{TAG}] embedding network restored from {emb_path}")
            else:
                t0 = time.perf_counter()
                history = embedding_phase(embed_net, pool, config, game, device,
                                          log, seed, k)
                timings["embedding"] = time.perf_counter() - t0
                torch.save({"model_state_dict": embed_net.state_dict(),
                            "config": config, "iteration": k,
                            "history": history}, emb_path)

        # §6.1 / §5.6: the agent starts from the trunk phase A just trained.
        # Here and not before the loop, because the embedding network has to be
        # trained first; only at iteration 0, because every later iteration
        # continues the agent §8 already produced; and only when this iteration
        # has no agent on disk, because a resumed run loads that one instead.
        agent_path = os.path.join(it_dir, "agent.pt")
        if (k == 0 and train_cfg.get("warm_start_trunk", False)
                and not os.path.exists(agent_path)):
            warm_start_trunk(agent_net, embed_net, log)

        # ------------------------------------------------------- B: the labels
        labels_path = os.path.join(it_dir, "labels.json")
        if os.path.exists(labels_path):
            blob = _read_json(labels_path)
            sampler.load_state_dict(blob["sampler"])
            manifest = blob["manifest"]
            timings.update(blob.get("timings", {}))
            log(f"[{TAG}] {manifest['n_labels']} labels restored from "
                f"{labels_path}")
        else:
            # The pool is clustered in the space the agent conditions on, which
            # is what makes the §11.3 dedup mean anything. Reserved-but-empty
            # rows are excluded: nobody has ever occupied them.
            sampler.set_vectors(
                embed_net.embeddings.weight[:len(pool)].detach().cpu().numpy())
            t0 = time.perf_counter()
            manifest = generate_labels(
                LockstepDriver(pool, int(game["n_actions"])), pool, sampler,
                embed_net, hero_factory(k, agent_net, init_member, game, device),
                {"seed": _iteration_seed(seed, k),
                 "n_sessions": int(config["n_sessions"]),
                 "hands_per_session": int(config["hands_per_session"]),
                 "driver_batch_size": int(config["driver_batch_size"]),
                 "labels_per_shard": int(config["labels_per_shard"]),
                 "game": game, "embedding_net": emb_cfg,
                 "oracle": oracle_cfg},
                os.path.join(it_dir, "labels"), log)
            timings["labels"] = time.perf_counter() - t0
            manifest = {**manifest, "stats": asdict(manifest["stats"])}
            _write_json(labels_path, {"manifest": manifest,
                                      "sampler": sampler.state_dict(),
                                      "timings": timings})

        labels = [lab for path in manifest["shards"] for lab in load_shard(path)]
        assert len(labels) == manifest["n_labels"], (
            f"{len(labels)} labels on disk, manifest says "
            f"{manifest['n_labels']}")
        train_idx, held_idx = split_heldout(
            len(labels), train_cfg.get("heldout_fraction", 0.0), seed, k)

        # ------------------------------------------------------ C: the agent
        if os.path.exists(agent_path):
            agent_net.load_state_dict(torch.load(
                agent_path, map_location=device,
                weights_only=False)["model_state_dict"])
            log(f"[{TAG}] agent restored from {agent_path}")
            history = None
        else:
            t0 = time.perf_counter()
            torch.manual_seed(_iteration_seed(seed, k))
            picked = [labels[i] for i in train_idx]
            history = train_agent(
                agent_net, [lab["tokens"] for lab in picked],
                build_targets(picked, loss, temperature, divisor),
                [lab["embeddings"] for lab in picked], train_cfg, device, log,
                seed=_iteration_seed(seed, k), iteration=k)
            timings["agent"] = time.perf_counter() - t0
            torch.save({"model_state_dict": agent_net.state_dict(),
                        "config": config, "iteration": k, "history": history},
                       agent_path)

        # ------------------------------------------------- D: the oracle gap
        metrics_path = os.path.join(it_dir, "metrics.json")
        if os.path.exists(metrics_path):
            metrics = _read_json(metrics_path)
        else:
            t0 = time.perf_counter()
            metrics = {
                "iteration": k,
                "n_pool": len(pool),
                "n_labels": len(labels),
                "n_train": int(len(train_idx)),
                "loss": loss,
                "label_stats": manifest["stats"],
                "gap": oracle_gap(agent_net, [labels[i] for i in held_idx],
                                  temperature, divisor,
                                  int(train_cfg["batch_hands"]), device, log),
            }
            timings["gap"] = time.perf_counter() - t0
            metrics["timings"] = timings
            _write_json(metrics_path, metrics)
        metrics_all.append(metrics)

        # ----------------------------------------- E: results, decay, the pool
        for member, result in manifest["results"].items():
            sampler.update(int(member), float(result["hero_bb"]),
                           int(result["n_hands"]))
        sampler.end_iteration()
        new_members, new_desc = agent_variant_members(
            frozen_agent_net(agent_net.state_dict(), config, game, device), k,
            config, game, device, seed)
        pool += new_members
        descriptors += new_desc
        _write_json(os.path.join(it_dir, "state.json"),
                    {"sampler": sampler.state_dict(), "n_pool": len(pool),
                     "iteration": k})
        log(f"[{TAG}] iteration {k} done in "
            f"{sum(timings.values()):.1f}s ({timings}); pool is now "
            f"{len(pool)} members")

    _write_json(os.path.join(exp_dir, "report.json"),
                {"config": config, "pool": descriptors,
                 "agent_init": init_desc, "metrics": metrics_all})
    log(f"[{TAG}] wrote {os.path.join(exp_dir, 'report.json')}")
    return metrics_all


def main():
    parser = argparse.ArgumentParser(description="CONCEPT.md §8 outer loop")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    base_dir = config.get("out_dir", "../../data/v8")
    log = Logger(base_dir)
    # Named and not timestamped: resume has to find the run it is resuming.
    exp_dir = os.path.join(base_dir, config["experiment"])
    try:
        run(config, log, exp_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
