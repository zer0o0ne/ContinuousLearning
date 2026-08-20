"""Data-parallel label generation: N CPU workers, one GPU inference server.

`CLAUDE.md` §3 names the shape this has to take on the Spark — "N CPU actor
processes + a GPU inference server" — and §13/G3 says why: a label's wall clock
is roughly 69 % the network forward and 31 % Python (v7 event building, tensor
packing, the driver's own bookkeeping). One process can parallelise neither.

**What is parallel and what is not.** Every hero decision is an independent
label: `action_values` reads a finished record and nothing else, and its
randomness is keyed by the decision (`np.random.default_rng([seed, i, h, d])`,
`_rollout_seed(spec.seed, decision_idx, a, s)`), never by a shared stream. So
the labels may be computed in any order, by any number of processes, and the
answer is the same. That is the property this module rests on; it was true
before this module existed.

**The layout.**

* The **parent** owns the single CUDA context and every network. It never runs
  a driver: during this phase it is the inference server. It also keeps the
  labels' bookkeeping — the tokens `_ObservedHero` recorded during play, the
  shard writing, the progress file — so the shards it writes are byte-for-byte
  the ones the sequential path writes, in the same order.
* Each **worker** is a pure-CPU process holding a *weightless mirror* of the
  pool: the same member objects with the same styles and grid maps, but with
  the network swapped for a proxy that packs its inputs and asks the parent.
  Workers own whole sessions, so they touch a disjoint slice of the records.

**Why the server batches at a barrier.** Each worker has at most one request in
flight, and the server waits until *every live worker* has one before running
anything. That is bulk-synchronous, and it buys two things at once: the forward
is one wide batch instead of N narrow ones, and the batch's composition depends
only on which workers are live — not on who happened to arrive first. An
opportunistic server that coalesced whatever was in the queue would be faster
by a straggler and would make a run unreproducible, which is not a trade this
project makes (`CLAUDE.md` §4).

Batch composition still decides floating-point reduction order, so on GPU a
different `n_workers` gives bitwise-different logits, exactly as a different
`oracle.batch_hands` does today. Fixed `n_workers` reproduces.

**What this does not do.** It does not change a single number about *what* a
label is: the oracle, the posterior, the rollouts, the styles and the shard
format are untouched. It changes who runs them.
"""

import os
import queue as queue_mod

import numpy as np
import torch

from agent.policy import AgentPoolMember, FrozenAgentMember
from env.driver import LockstepDriver
from nets.features import empty_batch
from oracle.rollout import action_values
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.v7_member import V7NetworkMember
from vendor.v7.agent import V7Agent
from vendor.v7.perception.perception import extract_event_tensors

# The server blocks on the request queue; if nothing arrives for this long it
# checks that its workers are still alive rather than waiting forever on a
# process that died with its request unsent.
POLL_SECONDS = 5.0

DONE = "done"
REQUEST = "request"

V7 = "v7"
AGENT = "agent"


# ------------------------------------------------------------ worker-side nets


class _V7Proxy:
    """Stands in for a `vendor.v7.agent.V7Agent` inside a worker.

    `V7NetworkMember` asks its network for three things — `n_actions`,
    `device_` and `action_logits` — so a proxy that answers those needs no
    change to the member at all. `device_` is `"cpu"`, which is also what makes
    `get_amp_config` disable autocast in the worker: the forward does not happen
    here, and the parent applies its own autocast when it does.
    """

    def __init__(self, key, n_actions, max_players, max_events, client):
        self.key = key
        self.n_actions = int(n_actions)
        self.max_players = int(max_players)
        self.max_events = int(max_events) if max_events else 0
        self.client = client
        self.device_ = "cpu"

    def action_logits(self, event_sequences):
        if self.max_events:
            # A.5.2, `EventSequenceEmbedder._cap_sequences`: the cap is applied
            # there only on the non-precomputed path, and this proxy takes the
            # precomputed one. With `max_actions_for(9) = 62` decisions plus at
            # most nine terminal events a v8 sequence is far short of it, so
            # this is a guard and not a code path anybody expects to run.
            event_sequences = [seq[-self.max_events:] if len(seq) > self.max_events
                               else seq for seq in event_sequences]
        payload = extract_event_tensors(event_sequences, self.max_players)
        assert payload is not None, "a v7 query with no events at all"
        return self.client.request(self.key, V7, payload, len(event_sequences))


class _AgentNetProxy:
    """Stands in for `nets.agent_net.AgentNet` inside a worker.

    `AgentPoolMember` and `FrozenAgentMember` use `net.d_emb` and call
    `net(batch, emb)`; nothing else of the network reaches them.
    """

    def __init__(self, key, d_emb, client):
        self.key = key
        self.d_emb = int(d_emb)
        self.client = client

    def __call__(self, batch, emb):
        return self.client.request(self.key, AGENT, (batch, emb),
                                   int(batch["mask"].shape[0]))


class ForwardClient:
    """A worker's end of the barrier: one request, one blocking wait."""

    def __init__(self, worker, request_q, reply_q):
        self.worker = int(worker)
        self.request_q = request_q
        self.reply_q = reply_q

    def request(self, key, kind, payload, rows):
        self.request_q.put((REQUEST, self.worker, key, kind, payload, rows))
        out = self.reply_q.get()
        assert out.shape[0] == rows, (
            f"asked for {rows} rows and the server returned {out.shape[0]}")
        return out


# ------------------------------------------------------- mirroring the pool


def mirror_spec(member, key_of_net):
    """A picklable description of `member` with its weights left behind.

    `key_of_net` maps a loaded network to the key the server knows it by;
    several members share one network (`with_style` siblings, D11) and must
    share one key, so identity is what it is keyed on.
    """
    if member is None:
        return None
    if isinstance(member, V7NetworkMember):
        agent = member.agent
        return (V7, member.name, member.n_actions, member.style,
                member.action_map, key_of_net(agent),
                agent.max_players, agent.perception.embedder.max_events)
    if isinstance(member, FrozenAgentMember):
        return ("frozen", member.name, member.n_actions, member.style,
                key_of_net(member.net), member.net.d_emb, member.max_players)
    if isinstance(member, AgentPoolMember):
        return ("hero", member.n_actions, key_of_net(member.net),
                member.net.d_emb, member.max_players)
    if type(member) in set(DEGENERATE_STRATEGIES.values()):
        return ("degenerate", type(member), member.name, member.n_actions,
                member.style)
    raise TypeError(
        f"{type(member).__name__} has no mirror: `oracle/parallel.py` rebuilds "
        f"every pool member inside the worker without its weights, and a "
        f"member kind it does not know about cannot be rebuilt")


def build_mirror(spec, client):
    """The worker's weightless copy of one member."""
    if spec is None:
        return None
    kind = spec[0]
    if kind == V7:
        _k, name, n_actions, style, action_map, key, max_players, max_events = spec
        proxy = _V7Proxy(key, _v7_actions(action_map, n_actions), max_players,
                         max_events, client)
        return V7NetworkMember(name, n_actions, proxy, style,
                               action_map=action_map)
    if kind == "frozen":
        _k, name, n_actions, style, key, d_emb, max_players = spec
        return FrozenAgentMember(name, n_actions,
                                 _AgentNetProxy(key, d_emb, client),
                                 max_players, "cpu", style)
    if kind == "degenerate":
        _k, cls, name, n_actions, style = spec
        return cls(name, n_actions, style)
    raise ValueError(f"unknown mirror kind {kind!r}")


def _v7_actions(action_map, n_actions):
    """How many actions the *checkpoint* has, which is not the pool's count
    when a `RaiseGridMap` is transporting between two raise grids."""
    return n_actions if action_map is None else action_map.n_src


def hero_mirror(spec, client):
    """Rebuild `hero_factory` (pipeline.py) inside the worker.

    Iteration 0 seats a fixed pool member and the factory ignores its arguments;
    every later iteration builds one `AgentPoolMember` per seat and per refresh
    of the fitted vectors, which is what the arguments are for.
    """
    if spec[0] == "hero":
        _k, n_actions, key, d_emb, max_players = spec
        net = _AgentNetProxy(key, d_emb, client)
        return lambda observer_pos, slot_of_seat, embeddings: AgentPoolMember(
            net, embeddings, slot_of_seat, max_players, n_actions,
            observer_pos, "cpu")
    member = build_mirror(spec, client)
    return lambda observer_pos, slot_of_seat, embeddings: member


# ------------------------------------------------------- server-side batching


def merge_v7(payloads):
    """Concatenate `extract_event_tensors` outputs into one wider batch.

    Every per-event tensor is `(T_i, …)` and stacks along 0; `batch_idx` is the
    only field that has to be renumbered, because it points at the sequence an
    event belongs to. The result is what `extract_event_tensors` would have
    returned for the concatenated list of sequences, which is the property
    `tests/test_parallel_labels.py` pins against running them one at a time.
    """
    per_event = ("card_ids", "hero_pos", "acting_pos", "num_players",
                 "scalars", "bets", "stacks", "actions", "event_idx")
    out = {k: torch.cat([p[k] for p in payloads], dim=0) for k in per_event}
    idx, offset = [], 0
    lengths = []
    for p in payloads:
        idx.append(p["batch_idx"] + offset)
        offset += int(p["B"])
        lengths.extend(p["seq_lengths"])
    out["batch_idx"] = torch.cat(idx, dim=0)
    out["seq_lengths"] = lengths
    out["max_events"] = max(int(p["max_events"]) for p in payloads)
    out["B"] = offset
    return out


def merge_tokens(payloads):
    """Concatenate collated token batches, padding the short ones.

    `collate` pads to the longest hand of its own batch, so batches from
    different workers disagree on `T`. `nets.features.empty_batch` is where the
    pad values live and this pads with it rather than restating them.
    """
    batches = [b for b, _e in payloads]
    embs = [e for _b, e in payloads]
    B = sum(int(b["mask"].shape[0]) for b in batches)
    T = max(int(b["mask"].shape[1]) for b in batches)
    n_actions = int(batches[0]["prev_action"].shape[2])
    max_players = int(batches[0]["seat_stacks"].shape[2])
    d_emb = int(embs[0].shape[2])

    out = empty_batch(B, T, n_actions, max_players)
    emb = torch.zeros((B, T, d_emb), dtype=embs[0].dtype)
    row = 0
    for batch, e in zip(batches, embs):
        b, t = int(batch["mask"].shape[0]), int(batch["mask"].shape[1])
        for key, value in out.items():
            value[row:row + b, :t] = batch[key]
        emb[row:row + b, :t] = e
        row += b
    # The three derived masks are functions of what was just copied, and
    # `collate` is where they are defined; recomputing them here would be a
    # second statement of the same rule.
    from nets.features import TOKEN_DECISION, TOKEN_SHOWDOWN
    out["decision_mask"] = out["mask"] * (out["token_type"] == TOKEN_DECISION)
    out["showdown_mask"] = out["mask"] * (out["token_type"] == TOKEN_SHOWDOWN)
    out["strength_mask"] = out["mask"] * (out["own_strength"] >= 0)
    return out, emb


class V7Runner:
    """The parent's end of a v7 query: one autocast forward over the batch."""

    def __init__(self, agent, amp):
        self.agent = agent
        self.amp_enabled, self.amp_device_type, self.amp_dtype = amp

    def run(self, payloads):
        merged = merge_v7(payloads)
        device = self.agent.device_
        with torch.no_grad(), torch.autocast(
                device_type=self.amp_device_type, dtype=self.amp_dtype,
                enabled=self.amp_enabled):
            perception_out, _enc, mask = self.agent.perception.forward_batch(
                None, device=device, skip_memory=True, skip_opponent_emb=True,
                precomputed=merged)
            logits = self.agent.action_head(perception_out, mask=mask)
        return logits.float().cpu()


class AgentRunner:
    """The parent's end of an agent-network query."""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def run(self, payloads):
        batch, emb = merge_tokens(payloads)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.net(batch, emb.to(self.device))
        return out.float().cpu()


class ForwardServer:
    """The barrier. Owns every network and answers every worker in lock-step."""

    def __init__(self, runners, request_q, reply_qs, procs, log):
        self.runners = runners
        self.request_q = request_q
        self.reply_qs = reply_qs
        self.procs = procs
        self.log = log
        self.active = set(range(len(reply_qs)))
        self.rounds = 0
        self.rows = 0

    def round(self, drain):
        """Collect one request from every live worker, answer them all.

        `drain` is called whenever the server is waiting, so the parent can
        take finished labels off the result queue without a second thread.
        Returns False once every worker has said it is done.
        """
        pending = {}
        while self.active and len(pending) < len(self.active):
            drain()
            try:
                item = self.request_q.get(timeout=POLL_SECONDS)
            except queue_mod.Empty:
                self._check_alive()
                continue
            if item[0] == DONE:
                self.active.discard(int(item[1]))
                continue
            _tag, worker, key, kind, payload, rows = item
            assert worker not in pending, (
                f"worker {worker} has two requests in flight; the barrier "
                f"assumes one, and batches would stop being reproducible")
            pending[worker] = (key, kind, payload, rows)
        if not pending:
            return bool(self.active)

        by_key = {}
        for worker, (key, kind, payload, rows) in sorted(pending.items()):
            by_key.setdefault((key, kind), []).append((worker, payload, rows))
        for (key, _kind), group in by_key.items():
            out = self.runners[key].run([p for _w, p, _r in group])
            self.rows += int(out.shape[0])
            row = 0
            for worker, _payload, rows in group:
                self.reply_qs[worker].put(out[row:row + rows].clone())
                row += rows
            assert row == out.shape[0], (
                f"the forward returned {out.shape[0]} rows for {row} asked for")
        self.rounds += 1
        return True

    def _check_alive(self):
        dead = [i for i in sorted(self.active) if not self.procs[i].is_alive()]
        assert not dead, (
            f"label worker(s) {dead} died without finishing; their labels are "
            f"missing and the phase cannot be completed. Their traceback is "
            f"above, in the parent's stderr.")


# --------------------------------------------------------------- the worker


def _worker_main(worker, todo, session_of, block_vectors, pool_spec, hero_spec,
                 hero_plain, hero_rec, n_actions, ocfg, seed, R,
                 request_q, reply_q, result_q):
    """Label every decision in `todo`. Runs in its own process, CPU only."""
    torch.set_num_threads(1)
    client = ForwardClient(worker, request_q, reply_q)
    play_pool = [build_mirror(spec, client) for spec in pool_spec]
    hero_factory = hero_mirror(hero_spec, client)
    driver = LockstepDriver(play_pool, n_actions)

    from train.generate import HERO_SLOT, _slot_of_seat_at

    current = (None, None)
    try:
        for pos, i, h, d in todo:
            s = session_of[i]
            block = h // R
            if (i, block) != current:
                current = (i, block)
                for seat in range(s.num_players):
                    member = hero_factory(seat, _slot_of_seat_at(s.num_players, seat),
                                          block_vectors[i][block])
                    play_pool[hero_plain[i][seat]] = member
                    play_pool[hero_rec[i][seat]] = member
            record = s.records[h]
            hero_seat = s.seat_of_slot(HERO_SLOT, h)
            q, legal, stats = action_values(
                record, d, driver, play_pool, hero_plain[i][hero_seat], ocfg,
                np.random.default_rng([int(seed), i, h, d]))
            result_q.put((pos, q, legal, stats))
    finally:
        request_q.put((DONE, worker))


# ------------------------------------------------------------- orchestration


def runner_table(play_pool, hero_prototype, device, log):
    """`{key: runner}` for every distinct network in the pool, plus its keys.

    Returns `(runners, key_of_net)`. Keying is by object identity so that the
    style siblings that share one loaded checkpoint share one runner — and one
    batch — instead of queueing behind each other.
    """
    runners, keys = {}, {}

    def key_of_net(net):
        if id(net) in keys:
            return keys[id(net)]
        key = f"net{len(keys)}"
        keys[id(net)] = key
        if isinstance(net, V7Agent):
            runners[key] = V7Runner(net, _amp_of(net))
        else:
            runners[key] = AgentRunner(net, device)
        return key

    specs = [mirror_spec(m, key_of_net) for m in play_pool]
    hero = mirror_spec(hero_prototype, key_of_net)
    log(f"[labels] inference server: {len(runners)} networks, "
        f"{sum(1 for s in specs if s is not None)} mirrored members")
    return runners, specs, hero


def _amp_of(agent):
    from utils import get_amp_config
    enabled, device_type, dtype, _scaler = get_amp_config(agent.device_)
    return enabled, device_type, dtype


def spawn_workers(n_workers, todo, sessions, block_vectors, pool_spec,
                  hero_spec, hero_plain, hero_rec, n_actions, ocfg, seed, R,
                  log):
    """Fork off `n_workers` label workers over a session-strided partition.

    Sessions rather than decisions, because a worker then reads only its own
    slice of the records and rebuilds hero's member once per block exactly as
    the sequential path does. Strided rather than contiguous, because at a
    barrier an idle worker costs everyone: striding mixes long and short
    sessions into every slice.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    reply_qs = [ctx.Queue() for _ in range(n_workers)]
    result_q = ctx.Queue()

    owners = {i: i % n_workers for i in range(len(sessions))}
    procs = []
    for w in range(n_workers):
        mine = [t for t in todo if owners[t[1]] == w]
        session_of = {i: sessions[i] for i in range(len(sessions))
                      if owners[i] == w}
        vectors = {i: block_vectors[i] for i in session_of}
        p = ctx.Process(
            target=_worker_main,
            args=(w, mine, session_of, vectors, pool_spec, hero_spec,
                  hero_plain, hero_rec, n_actions, ocfg, seed, R,
                  request_q, reply_qs[w], result_q),
            daemon=True)
        p.start()
        procs.append(p)
        log(f"[labels] worker {w}: {len(mine)} labels over "
            f"{len(session_of)} sessions (pid {p.pid})")
    return procs, request_q, reply_qs, result_q


def collect(server, result_q, procs, todo, start, consume, log):
    """Drive the barrier, hand finished labels to `consume` **in todo order**.

    Workers own strided sessions, so labels come back out of order; the parent
    holds them until their turn. Consuming in order is what makes the shards
    this phase writes the same shards, in the same order, that one process
    would have written — which matters because `split_heldout` partitions by
    position, so a reordered corpus is a different held-out set.
    """
    results, next_pos = {}, int(start)

    def drain():
        drain_results(result_q, results)

    while True:
        alive = server.round(drain)
        drain()
        next_pos = _consume_prefix(results, next_pos, todo, consume)
        if not alive:
            break

    # The workers have all said they are done, but `mp.Queue` is asynchronous:
    # their last results may still be in flight, and a process cannot be joined
    # until what it queued has been taken off.
    while next_pos < len(todo):
        try:
            pos, q, legal, stats = result_q.get(timeout=POLL_SECONDS)
        except queue_mod.Empty:
            _check_exited(procs)
            continue
        results[int(pos)] = (q, legal, stats)
        next_pos = _consume_prefix(results, next_pos, todo, consume)
    assert not results, (
        f"{len(results)} labels came back for positions nobody asked for")
    log(f"[labels] server: {server.rounds} batched rounds, {server.rows} rows")
    return next_pos


def _consume_prefix(results, next_pos, todo, consume):
    while next_pos in results:
        q, legal, stats = results.pop(next_pos)
        i, h, d = todo[next_pos]
        consume(next_pos, i, h, d, q, legal, stats)
        next_pos += 1
    return next_pos


def _check_exited(procs):
    bad = [w for w, p in enumerate(procs)
           if p.exitcode is not None and p.exitcode != 0]
    assert not bad, (
        f"label worker(s) {bad} exited badly with labels still outstanding")


def drain_results(result_q, into):
    """Move whatever finished labels are waiting into `into`. Never blocks."""
    n = 0
    while True:
        try:
            pos, q, legal, stats = result_q.get_nowait()
        except queue_mod.Empty:
            return n
        into[int(pos)] = (q, legal, stats)
        n += 1


def join_workers(procs, log):
    """Reap the workers. Never raises — it runs in a `finally`.

    A worker still alive here is one blocked on a reply that will not come,
    because the parent is unwinding out of the serving loop. Terminating it is
    the only way out: it cannot be joined while it waits, and the real failure
    is the exception already on its way up. `collect` is what raises when labels
    are actually missing.
    """
    stragglers, bad = [], []
    for w, p in enumerate(procs):
        p.join(timeout=POLL_SECONDS)
        if p.is_alive():
            stragglers.append(w)
            p.terminate()
            p.join(timeout=POLL_SECONDS)
        elif p.exitcode != 0:
            bad.append((w, p.exitcode))
    if stragglers:
        log(f"[labels] WARNING: terminated worker(s) {stragglers} still "
            f"waiting on the server")
    if bad:
        log(f"[labels] WARNING: worker(s) exited badly: {bad}")
    log(f"[labels] {len(procs)} workers joined")


def worker_count(cfg):
    """`oracle.n_workers`, clamped to what the box has.

    `0` (the default) is the sequential path — one process, no server, the code
    that ran before this module existed.
    """
    n = int(cfg.get("oracle", {}).get("n_workers", 0))
    if n <= 1:
        return 0
    # One core for the parent, which is doing the GPU work for everybody.
    return min(n, max(1, (os.cpu_count() or 2) - 1))
