"""Evaluation against Slumbot (CONCEPT.md §12, `PLAN_PIPELINE.md` S11).

The only number in this project measured against something that is not ours.
`evaluation/protocol.py` speaks the wire, `evaluation/v8_adapter.py` seats the
agent at it, and this file plays the hands, keeps the books and writes the
report.

**Two numbers, always both** (§12):

* **cold** — the embedding pinned to zero for the whole run. This is the
  *unconditional* policy, the one §6.2's embedding dropout trains explicitly and
  the one that cannot be a lookup table (§11.2, R7).
* **warm** — the vector fitted online by the generic §5.5 mechanism, plus the
  curve of BB/100 against how many hands had been observed when each hand was
  played, which is where "how many hands did it take to warm up" is read off.

If warm is worse than cold, **the exploitation mechanism is a net negative**.
That is a result to report, not a bug to tune away, and the report says so in
those words. §11.2's R5 already measured the shape of that risk on G1's data:
the fit was *worse* than `e = 0` for members observed for two decisions or
fewer, so a warm number that only overtakes cold after some hundreds of hands is
the expected shape rather than a surprise.

**Reporting discipline, enforced in code rather than in a habit.** Every report
carries the hand count, BB/100, its standard error, and §12's selection
disclosure — how many candidates were screened and over how many hands each.
Without that last one the headline reads as an unbiased measurement when it is
the maximum of several noisy ones; at 50 000 hands a session's standard error is
about ±2.7 BB/100, which is the scale of the bias involved. A run below
`min_reportable_hands` is stamped **SCREENING ONLY** in the report and in the
printed header, because `CLAUDE.md` §1 says shorter runs select what to measure
and are never themselves a result.

**Resumable, because a million hands against a remote API will be interrupted.**
Every completed hand is appended to `<mode>_w<k>.jsonl` before the next one
starts, and a restart replays that file into the accumulators, rebuilds the fit
window from its tail and carries on. A hand the wire lost is written too, marked
failed, so the hand index — which seeds the agent's own draws — does not shift
underneath a resume.

**Parallel, because a million hands is a million HTTP round trips.** A hand is
dominated by network latency, not by the forward, so `evaluation.n_workers`
processes each play their own share against their own Slumbot session. Processes
and not threads: the GIL and a single CUDA stream make threaded inference on
batch-of-one forwards slower than the serial path, which is the same reason v7's
evaluation was multiprocess. Each worker owns a *session* in §5.5's sense — it
sits down at its own table, observes the same opponent from scratch and fits its
own vector — so the warm-up curve is per session, which is exactly the quantity
§12 asks about, and there is no shared state to make the fit irreproducible.
Each worker also owns its own file, so a resume is per worker and needs no
coordination. The parent holds no network at all when workers are running; it
drains their queue, moves the one bar, and adds the files up at the end.

**No feedback, ever** (`CLAUDE.md` §1). Nothing here writes into a config, a
loss, a target or a pool weight, and candidate selection is a manual owner step
(§16, OI-7). This file measures.

Run::

    ./evaluate.sh --version=v8    # → cd versions/v8 && python3 eval_pipeline.py
"""

import argparse
import importlib
import json
import os
import queue as queue_mod
import sys
import traceback
from collections import deque

import numpy as np
import torch

from evaluation.protocol import (
    SLUMBOT_BIG_BLIND, SlumbotClient, board_to_ints, card_to_int,
    clamp_counters, stderr_bb_per_100_online,
)
from evaluation.v8_adapter import AgentMemberFactory, MemberFactory, \
    HERO_SLOT, OPP_SLOT, N_SEATS, SlumbotAgent, \
    _flip, slumbot_history
from env.session import raise_sizes_from
from evaluation.identity import check_identity, file_digest
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet, fit_embeddings, loss_weights
from nets.features import collate, hand_tokens
from train.generate import _pad_vectors
from utils import Logger, progress, resolve_device

TAG = "eval"
MODES = ("cold", "warm")
CLAMP_KEYS = tuple(sorted(clamp_counters()))


# ------------------------------------------------------------------ the books


class Stats:
    """BB/100 and its standard error, accumulated one hand at a time.

    Welford's form, because a million-hand run keeps no list of hands in memory
    — and the standard-error formula stays in `evaluation/protocol.py`, where
    the batch version is, so the two cannot disagree.
    """

    def __init__(self):
        self.n = 0
        self.total_bb = 0.0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, chips):
        bb = float(chips) / SLUMBOT_BIG_BLIND
        self.n += 1
        self.total_bb += bb
        delta = bb - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (bb - self._mean)

    @property
    def bb_per_100(self):
        return 100.0 * self.total_bb / self.n if self.n else 0.0

    @property
    def stderr(self):
        return stderr_bb_per_100_online(self.n, self._m2)

    def summary(self):
        return {"hands": self.n, "bb_per_100": self.bb_per_100,
                "stderr_bb_per_100": self.stderr, "total_bb": self.total_bb}


def _bucket(hands_observed, edges):
    """The observation-count bucket a hand belongs to (§12's warm-up curve).

    The bucket is named by its lower edge, so a report reads "after 100 observed
    hands the agent was winning at X" rather than by an index nobody can map
    back to a number of hands.
    """
    lo = 0
    for edge in edges:
        if hands_observed >= int(edge):
            lo = int(edge)
    return lo


# ------------------------------------------------------------- the checkpoints


def hero_factory(config, device, agent_net, log):
    """What plays the hands: the agent, or one procedural pool member.

    `evaluation.hero` absent means the agent, which is what every run before
    `PLAN_PROCEDURAL_POOL.md` §P5 did and what the file is named for. A
    `regular` hero is one archetype of §P4 — not an agent result, and nothing
    about it feeds training; it is here because the only external check on
    whether the archetypes are ordered the way a human would order them is a
    real opponent.
    """
    game = config["game"]
    spec = config["evaluation"].get("hero") or {"kind": "agent"}
    kind = spec.get("kind", "agent")
    if kind == "agent":
        return AgentMemberFactory(agent_net, game, device)

    assert kind == "regular", f"unknown hero kind {kind!r}"
    from pool.archetypes import draw_params
    from pool.regular import RegularMember
    from pool.strength import (DEFAULT_TABLE_PATH, StrengthCache,
                               preflop_equity_table)

    archetype = spec["archetype"]
    seed = int(spec.get("variant_seed", 0))
    spread = float(spec.get("spread", 0.0))
    params = draw_params(archetype, np.random.default_rng(seed), spread)
    member = RegularMember(
        archetype, int(game["n_actions"]), params,
        # A worker plays one hand at a time against Slumbot, so three boards
        # are live at once and a big cache is only memory held per process.
        StrengthCache(int(spec.get("max_boards", 256))),
        preflop_equity_table(spec.get("preflop_table",
                                      DEFAULT_TABLE_PATH)),
        raise_sizes_from(game))
    log(f"[{TAG}] hero is the procedural pool member {archetype!r} "
        f"(spread {spread}, variant seed {seed}) — not an agent result")
    return MemberFactory(member)


def hero_is_the_agent(config):
    return (config["evaluation"].get("hero") or {}).get("kind", "agent") == "agent"


def load_networks(config, device, log, need_embedding):
    """The agent, and the embedding network the warm run fits against.

    Both are `pipeline.py`'s artefacts and both are read by path — nothing here
    searches a directory for a "best" checkpoint, because §16's OI-7 makes
    candidate selection a manual owner step and a runner that picks its own
    candidate is exactly the feedback `CLAUDE.md` §1 forbids.
    """
    game = config["game"]
    ev = config["evaluation"]
    agent_net = None
    if hero_is_the_agent(config):
        agent_state = torch.load(ev["agent_checkpoint"], map_location=device,
                                 weights_only=False)["model_state_dict"]
        agent_net = AgentNet(config["embedding_net"], int(game["n_actions"]),
                             int(game["max_players"])).to(device)
        agent_net.load_state_dict(agent_state)
        agent_net.eval()
        log(f"[{TAG}] agent from {ev['agent_checkpoint']}")

    embed_net = None
    if need_embedding:
        assert ev.get("embedding_checkpoint"), (
            "the warm run fits an opponent vector (§5.5) and needs the "
            "embedding network that defines the objective; set "
            "`evaluation.embedding_checkpoint` or turn `evaluation.warm` off")
        state = torch.load(ev["embedding_checkpoint"], map_location=device,
                           weights_only=False)["model_state_dict"]
        embed_net = OpponentEmbeddingNet(
            config["embedding_net"], int(game["n_actions"]),
            int(game["max_players"]),
            n_members=int(state["embeddings.weight"].shape[0])).to(device)
        embed_net.load_state_dict(state)
        embed_net.eval()
        log(f"[{TAG}] embedding network from {ev['embedding_checkpoint']}")
    return agent_net, embed_net


# ---------------------------------------------------------------- one hand


def play_hand(client, agent, counters):
    """Play one hand to its end. Returns everything the run needs to keep.

    The loop is v7's shape and the two indices are §2.12's: hero appends the
    **effective** action — what the wire token says it did once
    `action_idx_to_incr` has clamped — and never the index the agent asked for,
    or every later observation in the hand carries an action hero did not take.
    """
    r = client.new_hand()
    client_pos = int(r["client_pos"])
    hole_cards = [card_to_int(c) for c in r["hole_cards"]]
    board = board_to_ints(r.get("board") or [])
    hero_action_indices = []

    while True:
        if r.get("board"):
            board = board_to_ints(r["board"])
        action_str = r.get("action") or ""
        if r.get("winnings") is not None:
            bot = r.get("bot_hole_cards") or None
            return {
                "client_pos": client_pos,
                "hole_cards": hole_cards,
                "board": board,
                "action": action_str,
                "hero_action_indices": hero_action_indices,
                "bot_hole_cards": ([card_to_int(c) for c in bot] if bot
                                   else None),
                "winnings": float(r["winnings"]),
                "baseline_winnings": float(r.get("baseline_winnings") or 0.0),
            }
        incr, effective, _chosen = agent.act(
            action_str, client_pos, hole_cards, board,
            hero_action_indices=hero_action_indices, counters=counters)
        hero_action_indices.append(effective)
        r = client.act(incr)


# ------------------------------------------------------------------- the fit


def fit_vectors(embed_net, hands, config, device):
    """§5.5's fit over the hands observed so far, at Slumbot's table.

    The same objective, the same optimiser and the same config keys the agent
    uses against every other opponent — `CONCEPT.md` §10 records that as
    adaptation and not specialisation, and it is why this is four lines of
    composition rather than a mechanism of its own.

    Hero is slot 0 and Slumbot is slot 1, as in every session the agent was
    trained on; hero's seat rotates hand to hand exactly as the button does
    there, so `slot_of_seat` is rebuilt per hand and the two players keep their
    identities across the rotation.
    """
    game = config["game"]
    emb_cfg = config["embedding_net"]
    tokens = []
    for hand in hands:
        record = slumbot_history(
            hand["action"], hand["client_pos"], hand["hole_cards"],
            hand["board"], game, hand["winnings"],
            hero_action_indices=hand["hero_action_indices"],
            bot_hole_cards=hand.get("bot_hole_cards"))
        hero_seat = _flip(hand["client_pos"])
        tokens.append(hand_tokens(
            record, observer_pos=hero_seat,
            slot_of_seat=[HERO_SLOT if seat == hero_seat else OPP_SLOT
                          for seat in range(N_SEATS)],
            max_players=int(game["max_players"]),
            n_actions=int(game["n_actions"])))
    tokens = [t for t in tokens if len(t)]
    if not tokens:
        return None

    batch = collate(tokens, device=device)
    fitted = fit_embeddings(
        embed_net, batch, N_SEATS, steps=int(emb_cfg["K"]),
        lr=emb_cfg["fit_lr"], reg=emb_cfg["fit_reg"],
        init=embed_net.amortised_init(batch, N_SEATS),
        weights=loss_weights(emb_cfg))
    return _pad_vectors(fitted.detach().cpu().numpy(),
                        int(game["max_players"]), embed_net.d_emb)


# ------------------------------------------------------------------ one mode


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def shard_path(out_dir, mode, worker):
    return os.path.join(out_dir, f"{mode}_w{int(worker):02d}.jsonl")


def split_hands(n_hands, n_workers):
    """`n_hands` divided into one share per worker, the remainder at the front."""
    n, w = int(n_hands), int(n_workers)
    return [n // w + (1 if i < n % w else 0) for i in range(w)]


def build_client(config, log):
    ev = config["evaluation"]
    return SlumbotClient(
        username=ev.get("username", ""), password=ev.get("password", ""),
        timeout=ev.get("timeout", 10), retries=ev.get("retries", 4),
        backoff=ev.get("backoff", 1.0), log=log)


def play_shard(mode, worker, n_hands, agent_net, embed_net, config, device,
               out_dir, log, client, on_hand=None):
    """One worker's share of one mode, resumable at the hand boundary.

    Everything this produces is in `<mode>_w<k>.jsonl` and nothing is held only
    in memory — the clamp counters ride along on each hand rather than being
    tallied, so a resumed run's totals are the totals and not what happened
    since the restart.

    The worker's hands are one **session** in §5.5's sense: it fits its own
    vector from its own window and nobody else's, which is what deployment
    looks like and what makes the warm-up curve mean anything.
    """
    game = config["game"]
    ev = config["evaluation"]
    emb_cfg = config["embedding_net"]
    warm = mode == "warm"
    R = int(emb_cfg["R"])
    window = deque(maxlen=int(ev["fit_window"]))
    seed = int(config.get("seed", 0))

    agent = SlumbotAgent(hero_factory(config, device, agent_net, log), game,
                         device)
    path = shard_path(out_dir, mode, worker)
    replayed = _read_jsonl(path)
    # Kept for the periodic log line only. The reported numbers are added up
    # from disk by `aggregate`, so this cannot disagree with them by drifting —
    # it can only be behind, which is what a progress line is.
    running = Stats()
    for hand in replayed:
        if not hand.get("failed"):
            window.append(hand)
            running.add(hand["winnings"])
    if replayed:
        log(f"[{TAG}:{mode}] worker {worker} resuming after {len(replayed)} "
            f"hands")

    fitted_from = 0
    if warm and window:
        vectors = fit_vectors(embed_net, list(window), config, device)
        if vectors is not None:
            agent.set_embeddings(vectors)
            fitted_from = len(window)

    fh = open(path, "a", encoding="utf-8")
    try:
        for idx in range(len(replayed), int(n_hands)):
            if warm and idx % R == 0 and window:
                vectors = fit_vectors(embed_net, list(window), config, device)
                if vectors is not None:
                    agent.set_embeddings(vectors)
                    fitted_from = len(window)
            # Seeded per hand rather than carried, so a resumed run draws what
            # an uninterrupted one would have drawn.
            agent.rng = np.random.default_rng(
                [seed, MODES.index(mode), int(worker), idx])
            counters = clamp_counters()
            try:
                hand = play_hand(client, agent, counters)
            except Exception as exc:            # noqa: BLE001 — see below
                # One hand lost, no desync, no cascade — the same policy
                # `SlumbotClient` applies to a timeout. The hand is written so
                # the index does not shift under a resume, and counted so a run
                # that is quietly failing cannot look like a clean one.
                fh.write(json.dumps({"failed": True,
                                     "error": f"{type(exc).__name__}: {exc}"})
                         + "\n")
                fh.flush()
                log(f"[{TAG}:{mode}] worker {worker} hand {idx} failed: "
                    f"{type(exc).__name__}: {exc}")
                if on_hand is not None:
                    on_hand()
                continue

            hand["hands_observed"] = fitted_from
            hand["clamps"] = [int(counters[k]) for k in CLAMP_KEYS]
            fh.write(json.dumps(hand) + "\n")
            fh.flush()
            window.append(hand)
            running.add(hand["winnings"])
            if on_hand is not None:
                on_hand()
            if (idx + 1) % int(ev.get("log_every", 1000)) == 0:
                log(f"[{TAG}:{mode}] worker {worker}: {idx + 1} hands, "
                    f"{running.bb_per_100:+.2f} ± {running.stderr:.2f} BB/100")
    finally:
        fh.close()


def aggregate(mode, out_dir, config, log):
    """Every worker's file, added up into the mode's summary.

    Read off disk rather than accumulated in memory, so the number a resumed run
    reports is the number the whole run earned and a worker that died still
    contributes everything it wrote.
    """
    ev = config["evaluation"]
    edges = [int(e) for e in ev["warmup_buckets"]]
    stats, by_observed = Stats(), {}
    clamps = {k: 0 for k in CLAMP_KEYS}
    failed = 0
    n_workers = int(ev.get("n_workers", 1))
    for worker in range(n_workers):
        for hand in _read_jsonl(shard_path(out_dir, mode, worker)):
            if hand.get("failed"):
                failed += 1
                continue
            stats.add(hand["winnings"])
            by_observed.setdefault(_bucket(hand["hands_observed"], edges),
                                   Stats()).add(hand["winnings"])
            for k, v in zip(CLAMP_KEYS, hand.get("clamps", ())):
                clamps[k] += int(v)

    summary = stats.summary()
    summary.update({
        "mode": mode,
        "failed_hands": failed,
        "clamps": clamps,
        "by_hands_observed": {str(k): v.summary()
                              for k, v in sorted(by_observed.items())},
    })
    log(f"[{TAG}:{mode}] {summary['hands']} hands: "
        f"{summary['bb_per_100']:+.2f} ± {summary['stderr_bb_per_100']:.2f} "
        f"BB/100, {failed} failed, clamps {summary['clamps']}")
    return summary


# ------------------------------------------------------------- the workers


def _resolve(dotted):
    """`"package.module:name"` → the object. The offline-test seam only."""
    module, _, name = dotted.partition(":")
    return getattr(importlib.import_module(module), name)


def worker_main(config, mode, worker, n_hands, out_dir, result_q,
                client_factory):
    """A worker process: its own nets, its own session, its own file.

    Everything crosses the process boundary as data — the checkpoints are loaded
    from their paths here rather than pickled — so the parent never has to touch
    a GPU while workers are running, and there is no shared model to serialise
    around.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    def log(text):
        result_q.put(("log", str(text)))

    try:
        device = resolve_device(config.get("device", "auto"))
        agent_net, embed_net = load_networks(config, device, log,
                                             need_embedding=mode == "warm")
        client = (_resolve(client_factory)(worker) if client_factory
                  else build_client(config, log))
        play_shard(mode, worker, n_hands, agent_net, embed_net, config, device,
                   out_dir, log, client,
                   on_hand=lambda: result_q.put(("hand", worker)))
        result_q.put(("done", worker, None))
    except BaseException:                       # noqa: BLE001
        result_q.put(("done", worker, traceback.format_exc()))


def run_mode(mode, config, device, out_dir, log, bar, agent_net, embed_net,
             client, client_factory):
    """`evaluation.hands` hands in one condition, across `n_workers` processes."""
    ev = config["evaluation"]
    n_workers = int(ev.get("n_workers", 1))
    assert n_workers >= 1, f"n_workers is a process count, got {n_workers}"
    shares = split_hands(int(ev["hands"]), n_workers)

    if n_workers == 1:
        play_shard(mode, 0, shares[0], agent_net, embed_net, config, device,
                   out_dir, log,
                   client if client is not None else build_client(config, log),
                   on_hand=lambda: bar.update(1))
    else:
        _run_workers(mode, config, out_dir, log, bar, shares, client_factory)
    return aggregate(mode, out_dir, config, log)


def _run_workers(mode, config, out_dir, log, bar, shares, client_factory):
    """Spawn one process per share and drain their queue until they are done.

    `spawn` and not `fork`: a forked child inherits a CUDA context it may not
    use, and the parent here is the process that would have initialised one.
    The parent loads no network at all when there is more than one worker, for
    the same reason — the GPU belongs to the workers.

    The timeout on the queue is what stops a worker that dies without a word
    from hanging the run: if nothing is alive and nothing is queued, the loop
    ends and `aggregate` reports on whatever reached disk.
    """
    import torch.multiprocessing as tmp

    ctx = tmp.get_context("spawn")
    result_q = ctx.Queue(maxsize=max(256, 8 * len(shares)))
    procs = []
    for worker, n_hands in enumerate(shares):
        p = ctx.Process(target=worker_main,
                        args=(config, mode, worker, n_hands, out_dir,
                              result_q, client_factory),
                        daemon=False)
        p.start()
        procs.append(p)
    log(f"[{TAG}:{mode}] {len(procs)} workers, shares {shares}")

    pending = len(procs)
    while pending:
        try:
            msg = result_q.get(timeout=1.0)
        except queue_mod.Empty:
            if not any(p.is_alive() for p in procs) and result_q.empty():
                log(f"[{TAG}:{mode}] {pending} worker(s) vanished without "
                    f"reporting; the run continues on what reached disk")
                break
            continue
        if msg[0] == "hand":
            bar.update(1)
        elif msg[0] == "log":
            log(msg[1])
        elif msg[0] == "done":
            pending -= 1
            if msg[2]:
                log(f"[{TAG}:{mode}] worker {msg[1]} died:\n{msg[2]}")
    for p in procs:
        p.join()


# -------------------------------------------------------------- the report


def warmup_hands(warm, cold, edges):
    """The first observation-count bucket at which warm caught cold up.

    §12 asks for "how many hands it took to warm up" and this is the only form
    of that number the run can produce: the warm curve is bucketed by how many
    hands the vector had been fitted from, and this is the first bucket whose
    BB/100 reaches the cold run's overall BB/100. `None` means it never did over
    this run — which, read against §11.2's R5, is the result and not a missing
    value.
    """
    if warm is None or cold is None:
        return None
    target = cold["bb_per_100"]
    for edge in sorted({0, *(int(e) for e in edges)}):
        cell = warm["by_hands_observed"].get(str(edge))
        if cell and cell["bb_per_100"] >= target:
            return edge
    return None


def build_report(config, results, log):
    ev = config["evaluation"]
    threshold = int(ev["min_reportable_hands"])
    disclosure = ev.get("selection_disclosure")
    assert isinstance(disclosure, dict) and {
        "candidates_screened", "screening_hands_each"} <= set(disclosure), (
        "§12: every reported result states how many candidates were screened "
        "and over how many hands each, or the headline reads as an unbiased "
        "measurement when it is the maximum of several noisy ones. Set "
        "`evaluation.selection_disclosure` with `candidates_screened` and "
        "`screening_hands_each`.")

    hands = min((r["hands"] for r in results.values()), default=0)
    cold, warm = results.get("cold"), results.get("warm")
    report = {
        "screening_only": hands < threshold,
        "min_reportable_hands": threshold,
        "selection_disclosure": disclosure,
        "modes": results,
        "warmup_hands": warmup_hands(warm, cold, ev["warmup_buckets"]),
    }
    if cold and warm:
        delta = warm["bb_per_100"] - cold["bb_per_100"]
        report["warm_minus_cold_bb_per_100"] = delta
        report["warm_is_worse_than_cold"] = delta < 0.0
    return report


def format_report(report, log):
    log("")
    if report["screening_only"]:
        log(f"=== SCREENING ONLY — under {report['min_reportable_hands']} "
            f"hands, this selects what to measure and is not a result "
            f"(CLAUDE.md §1) ===")
    else:
        log("=== Slumbot result ===")
    d = report["selection_disclosure"]
    log(f"selection disclosure: {d['candidates_screened']} candidate(s) "
        f"screened over {d['screening_hands_each']} hands each")

    for mode in MODES:
        r = report["modes"].get(mode)
        if r is None:
            continue
        log(f"{mode:>5}: {r['bb_per_100']:+8.2f} ± {r['stderr_bb_per_100']:.2f} "
            f"BB/100 over {r['hands']} hands "
            f"({r['failed_hands']} failed, clamps {sum(r['clamps'].values())})")
        for edge, cell in r["by_hands_observed"].items():
            log(f"        after {edge:>6} observed hands: "
                f"{cell['bb_per_100']:+8.2f} ± {cell['stderr_bb_per_100']:.2f} "
                f"BB/100 over {cell['hands']} hands")

    if "warm_minus_cold_bb_per_100" in report:
        delta = report["warm_minus_cold_bb_per_100"]
        log(f"warm − cold: {delta:+.2f} BB/100; warm caught cold up after "
            f"{report['warmup_hands']} observed hands"
            if report["warmup_hands"] is not None else
            f"warm − cold: {delta:+.2f} BB/100; warm never caught cold up over "
            f"this run")
        if report["warm_is_worse_than_cold"]:
            log("warm is worse than cold: the exploitation mechanism is a net "
                "negative. That is a result to report, not a bug to tune away "
                "(CONCEPT.md §12).")


# ---------------------------------------------------------------- the runner


def run(config, log, out_dir, client=None, client_factory=None):
    """The whole evaluation. `client` and `client_factory` are the offline seam.

    `client` replaces the HTTP client on the single-process path and
    `client_factory` — a `"module:function"` string, called with the worker
    index — replaces it inside each spawned worker, which cannot be handed a
    live object. Both are `None` in production and exist so that
    `tests/test_eval_pipeline.py` can exercise this file without a socket.
    """
    ev = config["evaluation"]
    modes = [m for m in MODES if ev.get(m, True)]
    if not hero_is_the_agent(config) and "warm" in modes:
        # A procedural member reads no opponent vector, so there is no warm
        # condition to run — said once, out loud, rather than left to fail one
        # process deep.
        log(f"[{TAG}] hero reads no opponent vector: the warm run is skipped")
        modes = [m for m in modes if m != "warm"]
    assert modes, "§12 reports cold and warm; turning both off measures nothing"
    n_workers = int(ev.get("n_workers", 1))
    os.makedirs(out_dir, exist_ok=True)
    # Hand budgets can grow on resume, but a different checkpoint or session
    # protocol must never append to an existing run's winrate.
    identity = {"version": "slumbot_identity_v1", "game": config["game"],
                "seed": config.get("seed", 0),
                "workers": n_workers, "hero": ev.get("hero") or {"kind": "agent"},
                "fit_window": ev.get("fit_window"),
                "embedding_config": config["embedding_net"]}
    if hero_is_the_agent(config):
        identity["agent"] = file_digest(ev["agent_checkpoint"])
    if "warm" in modes:
        assert ev.get("embedding_checkpoint"), "warm requires evaluation.embedding_checkpoint"
    for mode in modes:
        mode_identity = {**identity, "mode": mode}
        if mode == "warm":
            mode_identity["embedding"] = file_digest(ev["embedding_checkpoint"])
        directory = os.path.join(out_dir, "identities", mode)
        if not os.path.exists(os.path.join(directory, "identity.json")):
            import glob
            if glob.glob(os.path.join(out_dir, f"{mode}_w*.jsonl")):
                raise ValueError("Unversioned Slumbot results; use a new evaluation.run name")
        check_identity(directory, mode_identity)

    # With workers, the parent holds no network and touches no GPU: the
    # checkpoints are loaded inside each worker from their paths.
    device = agent_net = embed_net = None
    if n_workers == 1:
        device = resolve_device(config.get("device", "auto"))
        log(f"device: {device}")
        agent_net, embed_net = load_networks(config, device, log,
                                             need_embedding="warm" in modes)
    else:
        # The refusals `load_networks` makes are the run's, not a worker's, and
        # a run that dies one process deep is a worse way to learn about them.
        assert not hero_is_the_agent(config) or os.path.exists(
            ev["agent_checkpoint"]), (
            f"no agent checkpoint at {ev.get('agent_checkpoint')!r}")
        assert "warm" not in modes or ev.get("embedding_checkpoint"), (
            "the warm run fits an opponent vector (§5.5) and needs the "
            "embedding network that defines the objective; set "
            "`evaluation.embedding_checkpoint` or turn `evaluation.warm` off")

    # One bar over every hand of the whole job, never one per mode and never
    # one per worker (`CLAUDE.md` §5).
    bar = progress(total=len(modes) * int(ev["hands"]), desc="slumbot",
                   unit="hand")
    results = {}
    for mode in modes:
        results[mode] = run_mode(mode, config, device, out_dir, log, bar,
                                 agent_net, embed_net, client, client_factory)
    bar.close()

    report = build_report(config, results, log)
    format_report(report, log)
    path = os.path.join(out_dir, "slumbot_report.json")
    with open(path, "w") as fh:
        json.dump({"config": config, "report": report}, fh, indent=1,
                  default=float)
    log(f"wrote {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="CONCEPT.md §12 — Slumbot")
    parser.add_argument("--config", default="config.json")
    # Ten archetypes are ten runs of the same config into ten directories
    # (`PLAN_PROCEDURAL_POOL.md` §P5); this saves editing the file between them
    # and changes nothing else.
    parser.add_argument("--hero-archetype", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)
    if args.hero_archetype:
        hero = config["evaluation"].get("hero") or {}
        assert hero.get("kind") == "regular", (
            "--hero-archetype names one of the procedural archetypes; the "
            "config's `evaluation.hero.kind` must already be \"regular\"")
        hero["archetype"] = args.hero_archetype
        config["evaluation"]["hero"] = hero

    base_dir = config.get("out_dir", "../../data/v8")
    log = Logger(base_dir)
    # Named and not timestamped: a million-hand run is resumed, not restarted.
    # A pool member's run lives under `pool_eval/` and never beside the agent's,
    # because the two numbers are not the same kind of thing and a directory is
    # the cheapest place to keep them from being read as if they were.
    hero = config["evaluation"].get("hero") or {"kind": "agent"}
    if hero.get("kind", "agent") == "agent":
        out_dir = os.path.join(base_dir, config["experiment"], "slumbot",
                               config["evaluation"].get("run", "run0"))
    else:
        out_dir = os.path.join(base_dir, "pool_eval", hero["archetype"],
                               config["evaluation"].get("run", "run0"))
    try:
        run(config, log, out_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
