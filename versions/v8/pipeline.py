"""The outer loop (CONCEPT.md §8, `PLAN_PIPELINE.md` S9).

Everything before this file produces one artefact each; this is the file that
turns them into a training run:

```
iteration k:
  A  retrain the embedding network on pool self-play          (§5.4, every `retrain_every`)
     ↳ at k = 0 only, warm-start the agent's trunk from it    (§6.1 OI-4, §5.6)
  B  play, fit vectors, label hero's decisions                (§8 lines 1–4, train/generate.py)
  C  train the agent on all but `heldout_fraction` of them    (§6.2)
  D  measure the oracle gap on the held-out slice, warm and cold  (§8)
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
from evaluation.identity import atomic_json
from gates.g1 import _stack_bucket
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet
from nets.features import collate
from pool.build import build_pool
from pool.sampling import PoolSampler
from pool.style import StyleParams, sample_style
from train.agent_train import (seat_embeddings, token_embeddings,
                               train_agent)
from oracle.ranges import label_ranges
from oracle.rollout import RANGE_MODEL, estimator_signature
from train.embed_train import train_embedding_net
from train.generate import generate_labels, load_shard
from train.targets import ev_loss_budget, normalised_q, policy_target
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
def oracle_gap(net, labels, temperature, divisor, batch_hands, device, log,
               cold=False, iteration=0):
    """§8's held-out measurement of one checkpoint against its own oracle.

    One agent forward per held-out decision and **no new rollouts** — the
    oracle's answer is already in the shard, so this is a softmax over stored
    `q` and a batched forward, not a second labelling pass.

    `cold` is §12's distinction, on the held-out slice instead of on Slumbot:
    **warm** (the default) conditions the agent on the vectors the label carries
    — the ones §5.5 had fitted when hero acted — and **cold** pins them to zero
    and reads the *unconditional* policy §6.2's embedding dropout trains. The
    same labels, the same oracle `Q`, one forward each, so `ev_agent` measured
    both ways says what conditioning on the opponent is worth on this slice.

    Returns `{n_heldout, overall, by_table_size, by_stack_bb}`; a run with no
    held-out labels reports `n_heldout = 0` and no numbers at all, rather than a
    gap measured on the data the optimiser just saw.
    """
    mode = "cold" if cold else "warm"
    if not labels:
        log(f"[{TAG}] no held-out labels — the {mode} oracle gap is not "
            f"measured")
        return {"n_heldout": 0}

    net.eval()
    rows = []
    bar = progress(total=len(labels), desc=f"gap:{mode}", unit="label")
    for lo in range(0, len(labels), int(batch_hands)):
        chunk = labels[lo:lo + int(batch_hands)]
        batch = collate([lab["tokens"] for lab in chunk], device=device)
        tables = torch.as_tensor(
            np.stack([np.asarray(lab["embeddings"], dtype=np.float32)
                      for lab in chunk]), device=device)
        if cold:
            tables = torch.zeros_like(tables)
        logits, _range = net.logits_and_range(
            batch, token_embeddings(tables, batch["slot"], net.d_emb),
            seat_embeddings(tables, batch["seat_slot"]))
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
                policy_target(*args, temperature=temperature, divisor=divisor,
                              iteration=iteration),
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
    log(f"[{TAG}] {mode} oracle gap over {len(rows)} held-out labels: "
        f"kl={o['kl']:.4f} ev_agent={o['ev_agent']:+.4f} "
        f"ev_oracle={o['ev_oracle']:+.4f} q_best={o['q_best']:+.4f} "
        f"ev_gap_target={o['ev_gap_target']:+.4f} "
        f"ev_gap_greedy={o['ev_gap_greedy']:.4f} "
        f"agreement={o['agreement']:.3f}")
    return report


def winrate_line(gap, gap_cold):
    """Phase D's headline: the agent's held-out `ev_agent`, warm and cold.

    Two numbers and not one, for the reason §12 reports two against Slumbot: the
    warm number is the agent conditioned on the vectors §5.5 fitted for the
    table it was at, the cold one is the same agent with `e = 0`, and only the
    pair says whether conditioning on the opponent is paying for itself. As at
    §12, a cold number above the warm one is a result to report rather than a
    bug to tune away.

    It is `ev_agent` and not a played BB/100: phase D plays no hands. This is
    the agent's policy scored by the oracle's own `Q` on the held-out labels.
    Its distribution and continuation policy change between iterations; it is
    a dimensionless diagnostic, not a played-policy winrate.
    """
    warm, cold = gap.get("overall"), gap_cold.get("overall")
    if not warm or not cold:
        return "held-out ev_agent: not measured (no held-out labels)"
    return (f"held-out ev_agent (normalised Q, dimensionless) over {warm['n']} labels: "
            f"warm (fitted vectors) {warm['ev_agent']:+.4f}, "
            f"cold (e = 0) {cold['ev_agent']:+.4f}, "
            f"warm − cold {warm['ev_agent'] - cold['ev_agent']:+.4f}")


# ------------------------------------------------------------------ the phases


def agent_variant_members(net, iteration, config, game, device, seed,
                          embed_net=None):
    """The `style.agent_variants` members one trained agent contributes (D11).

    Variant 0 is the agent unmodified — the reference point every style draw is
    a perturbation of, and the same rule D13 applies to every v7 base. The rest
    are `with_style` siblings sharing this one network by reference, so they cost
    no forward and no parameter (§4.2).

    `embed_net` is the **frozen embedding network of this generation** — the one
    that produced the vectors this agent was trained to read. It travels with
    the member for the whole run, so a phase that lets a past agent read its
    tablemates reads them through the network the agent understands, and not
    through whatever the loop has retrained since. `None` when nothing may
    condition it. See `PLAN_AMORTISED_POOL.md` §0.2.
    """
    style_cfg = config.get("style", {})
    n_variants = int(style_cfg["agent_variants"])
    base = FrozenAgentMember(f"agent{iteration}", int(game["n_actions"]), net,
                             int(game["max_players"]), device,
                             StyleParams.identity(), embed_net=embed_net)
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


def frozen_embedding_net(state_dict, config, game, device):
    """A snapshot of the embedding network, frozen — one generation's reader.

    The loop holds exactly one embedding network and keeps training it, so a
    reference to it is a reference to a moving object. A past agent needs the
    weights as they were when it was trained, which is what this copy is. The
    member table's height is the checkpoint's own: nothing here reads a trained
    row, but a state dict does not load into a network of another height.
    """
    net = OpponentEmbeddingNet(
        config["embedding_net"], int(game["n_actions"]),
        int(game["max_players"]),
        n_members=int(state_dict["embeddings.weight"].shape[0])).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def embedding_vintages(exp_dir, upto, config, game, device, enabled, log):
    """`{iteration: the embedding network that generation reads}`, on resume.

    The agent of iteration *k* was trained against the network of the most
    recent retrain at or before *k*, so iterations that shared a retrain share
    one snapshot object — which is also what makes them share one inference
    runner when they are mirrored into a label worker.

    `enabled` is the pool-conditioning switch: with it off nothing reads a past
    agent's vectors, and loading a network per generation would be minutes and
    gigabytes spent on nothing.
    """
    if not enabled:
        return {k: None for k in range(upto)}
    out, cache, latest = {}, {}, None
    for k in range(upto):
        path = os.path.join(_iter_dir(exp_dir, k), "embedding.pt")
        if os.path.exists(path):
            latest = path
        if latest is None:
            out[k] = None                # trained before any network existed
            continue
        if latest not in cache:
            cache[latest] = frozen_embedding_net(
                torch.load(latest, map_location=device,
                           weights_only=False)["model_state_dict"],
                config, game, device)
        out[k] = cache[latest]
    if cache:
        log(f"[{TAG}] {len(cache)} frozen embedding generations for the "
            f"{upto} past agents in the pool")
    return out


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
    # §5.7 — the belief targets, once over the corpus and never inside the
    # training loop, exactly where `label_showdowns` sits for the same reason.
    if emb_cfg.get("range_enabled", False):
        rstats = label_ranges(
            sessions, pool, int(game["n_actions"]),
            floor=float(config.get("oracle", {}).get("likelihood_floor", 1e-6)),
            prune=float(emb_cfg.get("range_prune_threshold", 0.0)),
            desc="ranges")
        log(f"[{TAG}] §5.7 ranges: {rstats.emitted} targets, "
            f"{rstats.forwards} forwards, {rstats.dropped:.1f} total mass "
            f"pruned, {rstats.collapsed} supports collapsed")
    torch.manual_seed(it_seed + 500_000)
    return train_embedding_net(embed_net, sessions, emb_cfg, game, device, log,
                               it_seed + 500_000, iteration=iteration)


def build_targets(labels, loss, temperature, divisor, iteration=0):
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
                                     divisor=divisor, iteration=iteration))
        else:
            out.append(normalised_q(*args, divisor=divisor))
    return np.stack(out) if out else np.zeros((0, 0))


def split_heldout(n, fraction, seed, iteration):
    """(train, held-out) index arrays. Deterministic in `(seed, iteration)`."""
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng([int(seed), int(iteration), 23])
    perm = rng.permutation(int(n))
    n_held = int(round(float(fraction) * int(n)))
    assert n_held < n, (
        f"heldout_fraction {fraction} would withhold every one of {n} labels — "
        f"there would be nothing left to train on")
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


def load_training_distribution(config, labels_path, n_members):
    """Read the pre-update mixture, including for checkpoints predating snapshots."""
    saved = _read_json(labels_path)
    if "pool_distribution" in saved:
        return saved["pool_distribution"]
    sampler = PoolSampler(n_members, config["pool_sampling"], np.random.default_rng(0))
    sampler.load_state_dict(saved["sampler"])
    return sampler.distribution()


def log_pool_distribution(pool, distribution, iteration, log):
    log(f"[{TAG}] iteration {iteration} training pool weights "
        "(p = seat probability after PFSP, clustering and uniform floor; hero = PFSP score):")
    evaluation_values = distribution.get("evaluation_bb_per_100", [None] * len(pool))
    evaluation_shares = distribution.get("evaluation_mix", [0.0] * len(pool))
    for i, member in enumerate(pool):
        p = distribution["probabilities"][i]
        w = distribution["pfsp_weights"][i]
        cluster = distribution["cluster_of"][i]
        mean = distribution["hero_bb_per_100"][i]
        score = "unplayed" if mean is None else f"{mean:+.2f} BB/100"
        evaluation, share = evaluation_values[i], evaluation_shares[i]
        extra = "" if evaluation is None else f", eval={evaluation:+.2f} BB/100, eval_mix={share:.3f}"
        log(f"[{TAG}] pool[{i}] {member.name}: p={p:.8f} ({100*p:.4f}%), "
            f"pfsp={w:.6f}, cluster={cluster}, hero={score}{extra}")


def apply_evaluation_feedback(sampler, report, iteration, log):
    """Called only by the training loop after evaluation; standalone is read-only."""
    from evaluation.pool_eval import training_warm_feedback

    if not sampler.evaluation_weight or report is None:
        return None
    if report["iteration"] != iteration or report["candidates"][0] != f"agent{iteration}":
        raise ValueError("PFSP evaluation feedback must describe the current iteration's agent")
    feedback = training_warm_feedback(report)
    if feedback is None:
        log(f"[{TAG}] PFSP feedback skipped: no training/warm evaluation")
        return None
    applied = sampler.update_evaluation(iteration, feedback["by_member"])
    log(f"[{TAG}] PFSP training/warm feedback: current agent, {feedback['estimator']}, "
        f"{feedback['units']} independent sessions, {len(feedback['by_member'])} opponents, "
        f"evaluation_weight={sampler.evaluation_weight:g}, applied={applied}")
    return {**feedback, "iteration": iteration, "run": report["run"],
            "evaluation_weight": sampler.evaluation_weight, "applied": applied}


def pool_evaluation(pool, descriptors, agent_net, embed_net, measured_config,
                    config, exp_dir, iteration, bootstrap_size, device, log,
                    reference_iteration=None, training_distribution=None):
    """Compare the completed checkpoint with the preceding policy, no feedback."""
    from evaluation.pool_eval import Candidate, configuration, evaluate

    cfg = configuration(config)
    if cfg is None:
        return None
    game = config["game"]
    reference = iteration - 1 if reference_iteration is None else reference_iteration
    if not -1 <= reference < iteration:
        raise ValueError("reference_iteration must precede the evaluated iteration (-1 means agent_init)")
    new = FrozenAgentMember(f"agent{iteration}", game["n_actions"], agent_net,
                             game["max_players"], device, embed_net=embed_net)
    if reference >= 0:
        variants = int(config["style"]["agent_variants"])
        base = pool[bootstrap_size + reference * variants]
        previous = torch.load(os.path.join(_iter_dir(exp_dir, reference), "agent.pt"),
                              map_location="cpu", weights_only=False)
        old_config = previous.get("config") or config
        old_embedding = base.embed_net
        if cfg["warm_sessions"] and old_embedding is None:
            paths = [os.path.join(_iter_dir(exp_dir, k), "embedding.pt")
                     for k in range(reference + 1)]
            path = next((p for p in reversed(paths) if os.path.exists(p)), None)
            if path is None:
                raise ValueError("Previous checkpoint has no embedding generation for warm evaluation")
            old_embedding = frozen_embedding_net(
                torch.load(path, map_location=device, weights_only=False)["model_state_dict"],
                old_config, game, device)
        old = FrozenAgentMember(f"agent{reference}", game["n_actions"], base.net,
                                 game["max_players"], device, embed_net=old_embedding)
    else:
        members, _ = build_pool(
            {"bootstrap": [config["agent_init"]], "game": game, "style": config["style"]},
            np.random.default_rng(int(config.get("seed", 0)) + 1), device=device, log=log)
        old, old_config = members[0], config
    candidates = [Candidate(new, measured_config["embedding_net"]),
                  Candidate(old, old_config["embedding_net"])]
    if "training" in cfg["benchmarks"] and training_distribution is None:
        training_distribution = load_training_distribution(
            measured_config, os.path.join(_iter_dir(exp_dir, iteration), "labels.json"), len(pool))
        log_pool_distribution(pool, training_distribution, iteration, log)
    return evaluate(pool, descriptors, candidates, config,
                    os.path.join(_iter_dir(exp_dir, iteration), "pool_evaluation"),
                    iteration, bootstrap_size, log, training_distribution=training_distribution)


# -------------------------------------------------------------------- the loop


def run(config, log, exp_dir):
    from evaluation.pool_eval import configuration as evaluation_configuration
    evaluation_configuration(config)  # validate before the expensive phases
    game = config["game"]
    seed = int(config.get("seed", 0))
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")
    os.makedirs(exp_dir, exist_ok=True)

    emb_cfg = config["embedding_net"]
    style_cfg = config.get("style", {})
    train_cfg = dict(config["agent_train"])
    oracle_cfg = config["oracle"]
    temperature = oracle_cfg["temperature"]
    ev_loss_budget(temperature)  # validate before expensive corpus generation
    divisor = oracle_cfg.get("divisor", "pot_plus_bet")
    # One specification and resolver for targets, training and evaluation.
    train_cfg["temperature"] = temperature
    train_cfg["divisor"] = divisor
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
    conditioned = str(emb_cfg.get("pool_agent_vectors", "zero")) == "amortised"
    vintages = embedding_vintages(exp_dir, start, config, game, device,
                                  conditioned, log)
    for k in range(start):
        it_dir = _iter_dir(exp_dir, k)
        state = torch.load(os.path.join(it_dir, "agent.pt"), map_location=device,
                           weights_only=False)["model_state_dict"]
        members, desc = agent_variant_members(
            frozen_agent_net(state, config, game, device), k, config, game,
            device, seed, embed_net=vintages[k])
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
    # The generation an agent appended *now* would belong to: on a fresh run
    # nothing has been trained yet and iteration 0 retrains before it is used;
    # on a resume it is the generation the restored network belongs to, which is
    # the one the last past agent already carries.
    vintage = vintages.get(start - 1) if start else None

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
            # The generation every agent trained from here on belongs to. Taken
            # after the retrain and kept frozen: the live network keeps moving,
            # and a past agent must read the weights it was trained against.
            if conditioned:
                vintage = frozen_embedding_net(embed_net.state_dict(), config,
                                               game, device)

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
        blob = _read_json(labels_path) if os.path.exists(labels_path) else None
        if (blob is not None
                and (blob["manifest"].get("range_model") != RANGE_MODEL
                     or blob["manifest"].get("q_estimator") !=
                     estimator_signature(oracle_cfg, emb_cfg))
                and not os.path.exists(agent_path)):
            log(f"[{TAG}] labels use an incompatible Q estimator; rebuilding "
                "this unfinished iteration and keeping the old shards aside")
            blob = None
        if blob is not None:
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
            log_pool_distribution(pool, sampler.distribution(), k, log)
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
                                      "pool_distribution": sampler.distribution(),
                                      "timings": timings})

        labels = [lab for path in manifest["shards"] for lab in load_shard(path)]
        assert len(labels) == manifest["n_labels"], (
            f"{len(labels)} labels on disk, manifest says "
            f"{manifest['n_labels']}")
        train_idx, held_idx = split_heldout(
            len(labels), train_cfg.get("heldout_fraction", 0.0), seed, k)

        # ------------------------------------------------------ C: the agent
        measured_config = config
        if os.path.exists(agent_path):
            checkpoint = torch.load(agent_path, map_location=device,
                                    weights_only=False)
            agent_net.load_state_dict(checkpoint["model_state_dict"])
            measured_config = checkpoint.get("config") or config
            log(f"[{TAG}] agent restored from {agent_path}")
            history = None
        else:
            t0 = time.perf_counter()
            torch.manual_seed(_iteration_seed(seed, k))
            picked = [labels[i] for i in train_idx]
            if picked:
                history = train_agent(
                    agent_net, [lab["tokens"] for lab in picked],
                    build_targets(picked, loss, temperature, divisor, iteration=k),
                    [lab["embeddings"] for lab in picked], train_cfg, device, log,
                    seed=_iteration_seed(seed, k), iteration=k)
            else:
                history = []
                log(f"[{TAG}] no valid training labels; keeping the agent weights for iteration {k}")
            timings["agent"] = time.perf_counter() - t0
            torch.save({"model_state_dict": agent_net.state_dict(),
                        "config": config, "iteration": k, "history": history,
                        "q_estimator": manifest.get("q_estimator", "legacy")},
                       agent_path)

        training_distribution = load_training_distribution(measured_config, labels_path, len(pool))
        if blob is not None:
            log_pool_distribution(pool, training_distribution, k, log)

        # ------------------------------------------------- D: the oracle gap
        metrics_path = os.path.join(it_dir, "metrics.json")
        if os.path.exists(metrics_path):
            metrics = _read_json(metrics_path)
        else:
            t0 = time.perf_counter()
            # A checkpoint already trained before a restart keeps its own
            # target and held-out split, even if the next cycle's config changed.
            measured_train = measured_config["agent_train"]
            measured_oracle = measured_config["oracle"]
            train_idx, held_idx = split_heldout(
                len(labels), measured_train.get("heldout_fraction", 0.0),
                int(measured_config.get("seed", 0)), k)
            held = [labels[i] for i in held_idx]
            gap_args = (agent_net, held, measured_oracle["temperature"],
                        measured_oracle.get("divisor", "pot_plus_bet"),
                        int(measured_train["batch_hands"]), device, log)
            metrics = {
                "iteration": k,
                "n_pool": len(pool),
                "n_labels": len(labels),
                "n_train": int(len(train_idx)),
                "loss": measured_train.get("loss", "kl"),
                "temperature": measured_oracle["temperature"],
                "entropy_ev_loss_budget_bb": ev_loss_budget(
                    measured_oracle["temperature"], k),
                "range_model": manifest.get("range_model", "legacy_live_seats"),
                "q_estimator": manifest.get("q_estimator", "legacy"),
                "gap_units": "dimensionless_normalised_oracle_q",
                "label_stats": manifest["stats"],
                "gap": oracle_gap(*gap_args, iteration=k),
                "gap_cold": oracle_gap(*gap_args, cold=True, iteration=k),
            }
            timings["gap"] = time.perf_counter() - t0
            metrics["timings"] = timings
            _write_json(metrics_path, metrics)
        log(f"[{TAG}] " + winrate_line(metrics.get("gap", {}),
                                       metrics.get("gap_cold", {})))

        # The mixture stays fixed throughout evaluation. Only after completion
        # can training/warm results feed the next iteration's PFSP state.
        evaluation = None
        if config.get("pool_evaluation", {}).get("enabled", False):
            t0 = time.perf_counter()
            evaluation = pool_evaluation(
                pool, descriptors, agent_net, embed_net, measured_config,
                config, exp_dir, k, n_pool0, device, log,
                training_distribution=training_distribution)
            timings["pool_evaluation"] = time.perf_counter() - t0
            metrics["pool_evaluation"] = evaluation
            metrics["timings"] = timings
            _write_json(metrics_path, metrics)
        metrics_all.append(metrics)

        # ----------------------------------------- E: results, decay, the pool
        for member, result in manifest["results"].items():
            sampler.update(int(member), float(result["hero_bb"]),
                           int(result["n_hands"]))
        sampler.end_iteration()
        feedback = apply_evaluation_feedback(sampler, evaluation, k, log)
        new_members, new_desc = agent_variant_members(
            frozen_agent_net(agent_net.state_dict(), config, game, device), k,
            config, game, device, seed,
            embed_net=vintage if conditioned else None)
        pool += new_members
        descriptors += new_desc
        # Commit the feedback and its iteration marker together. A crash before
        # this atomic commit resumes the original pre-update sampler instead.
        atomic_json(os.path.join(it_dir, "state.json"),
                    {"sampler": sampler.state_dict(), "n_pool": len(pool),
                     "iteration": k, "evaluation_feedback": feedback})
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
