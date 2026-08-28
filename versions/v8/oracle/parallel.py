"""Data-parallel label generation: N CPU workers, one GPU inference server.

`CLAUDE.md` §3 names the shape this has to take on the Spark — "N CPU actor
processes + a GPU inference server" — and §13/G3 says why: a label's wall clock
is roughly 69 % the network forward and 31 % Python (v7 event building, tensor
packing, the driver's own bookkeeping). One process parallelises neither.

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
  the network swapped for a proxy that packs its inputs into shared memory and
  asks the parent. Workers own whole sessions, so they touch a disjoint slice
  of the records.

**One request, one forward — and why there is no barrier.** The server answers
each request on its own, over exactly the rows one process would have handed
the model. Two things follow, and the second was learned the hard way:

* the labels a parallel run produces are the labels a sequential run produces,
  down to the bit, because no batch anywhere is composed differently;
* nobody waits for anybody. The first version of this module was
  bulk-synchronous — the server collected one request from *every* live worker
  and answered them in one wide batch, which is reproducible and batches well
  on paper. Measured, it was **4.8× slower than one process** at toy scale and
  got *worse* going from two workers to four, because workers are never in
  phase: their requests interleave posterior batches of a thousand combos with
  driver steps of a dozen hands, and at a barrier everybody pays the slowest.
  Asynchronous service keeps the GPU busy by *overlap* instead — while the
  parent runs one worker's forward, the others are in their Python.

The other half of that measurement is `oracle/transport.py`: sending tensors
through an `mp.Queue` cost 8.7–13.9 ms per round trip and dominated everything.
The slabs it allocates bring the same round trip to 0.062 ms.

**The ceiling is Amdahl's.** With every forward serialised through one process,
the speedup cannot exceed `1 / 0.69 ≈ 1.45×`, and it is reached at a handful of
workers. Moving the 69 % itself means handing the model wider batches, which
means labelling several hero decisions in one `driver.run` — a change to the
oracle, not to this module.
"""

import os
import queue as queue_mod
from multiprocessing.connection import wait

import numpy as np
import torch

from agent.policy import AgentPoolMember, FrozenAgentMember
from env.driver import LockstepDriver
from nets.features import derived_masks
from oracle.rollout import action_values_batch
from oracle.posterior import PosteriorCache
from oracle.transport import Slab, slab_rows, token_fields, v7_fields
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.v7_member import V7NetworkMember
from vendor.v7.agent import V7Agent
from vendor.v7.perception.perception import extract_event_tensors

# How long the server waits on idle pipes before checking that its workers are
# still alive, rather than blocking forever on one that died mid-request.
POLL_SECONDS = 5.0

V7 = "v7"
AGENT = "agent"

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ------------------------------------------------------------ worker-side nets


class ForwardClient:
    """A worker's end of the wire: pack into the slab, ring, read the answer."""

    def __init__(self, slab, conn):
        self.slab = slab
        self.conn = conn

    def request(self, key, kind, fields, tensors, cells, shape, extra, out_cols):
        self.slab.pack(fields, tensors, cells)
        self.conn.send((key, kind, cells, shape, extra))
        self.conn.recv()
        return self.slab.out[:shape[0], :out_cols].clone()

    def done(self):
        self.conn.send(None)


class _V7Proxy:
    """Stands in for a `vendor.v7.agent.V7Agent` inside a worker.

    `V7NetworkMember` asks its network for three things — `n_actions`,
    `device_` and `action_logits` — so a proxy that answers those needs no
    change to the member at all. `device_` is `"cpu"`, which is also what makes
    `get_amp_config` disable autocast in the worker: the forward does not happen
    here, and the parent applies its own when it does.
    """

    def __init__(self, key, n_actions, max_players, max_events, client):
        self.key = key
        self.n_actions = int(n_actions)
        self.max_players = int(max_players)
        self.max_events = int(max_events) if max_events else 0
        self.client = client
        self.device_ = "cpu"
        self.fields = v7_fields(self.max_players, self.n_actions)

    def action_logits(self, event_sequences):
        if self.max_events:
            # A.5.2, `EventSequenceEmbedder._cap_sequences`: applied there only
            # on the non-precomputed path, and this proxy takes the precomputed
            # one. A v8 sequence is far short of the cap, so this is a guard.
            event_sequences = [seq[-self.max_events:] if len(seq) > self.max_events
                               else seq for seq in event_sequences]
        payload = extract_event_tensors(event_sequences, self.max_players)
        assert payload is not None, "a v7 query with no events at all"
        cells = int(payload["card_ids"].shape[0])
        b = int(payload["B"])
        self.client.slab.meta[:b].copy_(
            torch.as_tensor(payload["seq_lengths"], dtype=torch.int64))
        return self.client.request(
            self.key, V7, self.fields, payload, cells, (b,),
            int(payload["max_events"]), self.n_actions)


class _AgentNetProxy:
    """Stands in for `nets.agent_net.AgentNet` inside a worker.

    `AgentPoolMember` and `FrozenAgentMember` use `net.d_emb` and call
    `net(batch, emb)`; nothing else of the network reaches them.
    """

    def __init__(self, key, d_emb, max_players, n_actions, client):
        self.key = key
        self.d_emb = int(d_emb)
        self.n_actions = int(n_actions)
        self.client = client
        self.fields = token_fields(int(max_players), self.n_actions, self.d_emb)

    def __call__(self, batch, emb):
        b, t = (int(x) for x in batch["mask"].shape)
        tensors = dict(batch)
        tensors["emb"] = emb
        return self.client.request(self.key, AGENT, self.fields, tensors,
                                   b * t, (b, t), t, self.n_actions)


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
        net = _AgentNetProxy(key, d_emb, max_players, n_actions, client)
        return FrozenAgentMember(name, n_actions, net, max_players, "cpu", style)
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
        net = _AgentNetProxy(key, d_emb, max_players, n_actions, client)
        return lambda observer_pos, slot_of_seat, embeddings: AgentPoolMember(
            net, embeddings, slot_of_seat, max_players, n_actions,
            observer_pos, "cpu")
    member = build_mirror(spec, client)
    return lambda observer_pos, slot_of_seat, embeddings: member


# --------------------------------------------------------- server-side models


class V7Runner:
    """The parent's end of a v7 query: one autocast forward over the slab."""

    def __init__(self, agent, amp):
        self.agent = agent
        self.fields = v7_fields(int(agent.max_players), int(agent.n_actions))
        self.amp_enabled, self.amp_device_type, self.amp_dtype = amp

    def run(self, slab, cells, shape, extra):
        b = int(shape[0])
        payload = slab.unpack(self.fields, cells, (cells,))
        payload["seq_lengths"] = slab.meta[:b].tolist()
        payload["max_events"] = int(extra)
        payload["B"] = b
        with torch.no_grad(), torch.autocast(
                device_type=self.amp_device_type, dtype=self.amp_dtype,
                enabled=self.amp_enabled):
            out, _enc, mask = self.agent.perception.forward_batch(
                None, device=self.agent.device_, skip_memory=True,
                skip_opponent_emb=True, precomputed=payload)
            logits = self.agent.action_head(out, mask=mask)
        return logits.float().cpu()


class AgentRunner:
    """The parent's end of an agent-network query."""

    def __init__(self, net, max_players, n_actions, device):
        self.net = net
        self.device = device
        self.fields = token_fields(int(max_players), int(n_actions),
                                   int(net.d_emb))

    def run(self, slab, cells, shape, extra):
        views = slab.unpack(self.fields, cells, tuple(shape))
        emb = views.pop("emb")
        batch = {k: v.to(self.device) for k, v in views.items()}
        batch.update(derived_masks(batch))
        with torch.no_grad():
            out = self.net(batch, emb.to(self.device))
        return out.float().cpu()


class ForwardServer:
    """Answers whichever worker is ready, one request at a time."""

    def __init__(self, runners, slabs, conns, procs, log):
        self.runners = runners
        self.slabs = slabs
        self.conns = conns
        self.procs = procs
        self.log = log
        self.by_conn = {c: w for w, c in enumerate(conns)}
        self.active = set(range(len(conns)))
        self.requests = 0
        self.rows = 0

    def poll(self, drain):
        """Serve whatever is ready. Returns False once every worker is done.

        `drain` runs on every idle wake-up so the parent can take finished
        labels off the result queue without a second thread.
        """
        if not self.active:
            return False
        ready = wait([self.conns[w] for w in sorted(self.active)],
                     timeout=POLL_SECONDS)
        if not ready:
            drain()
            self._check_alive()
            return True
        for conn in ready:
            worker = self.by_conn[conn]
            message = conn.recv()
            if message is None:
                self.active.discard(worker)
                continue
            key, _kind, cells, shape, extra = message
            slab = self.slabs[worker]
            logits = self.runners[key].run(slab, cells, shape, extra)
            rows, cols = logits.shape
            slab.out[:rows, :cols].copy_(logits)
            self.requests += 1
            self.rows += int(rows)
            conn.send(True)
        return True

    def _check_alive(self):
        dead = [w for w in sorted(self.active) if not self.procs[w].is_alive()]
        assert not dead, (
            f"label worker(s) {dead} died without finishing; their labels are "
            f"missing and the phase cannot be completed. Their traceback is "
            f"above, in the parent's stderr.")


# --------------------------------------------------------------- the worker


def _worker_main(worker, todo, session_of, block_vectors, pool_spec, hero_spec,
                 hero_plain, hero_rec, n_actions, ocfg, seed, R, slab, conn,
                 result_q):
    """Label every decision in `todo`. Runs in its own process, CPU only."""
    torch.set_num_threads(1)
    client = ForwardClient(slab, conn)
    play_pool = [build_mirror(spec, client) for spec in pool_spec]
    hero_factory = hero_mirror(hero_spec, client)
    driver = LockstepDriver(play_pool, n_actions)

    from train.generate import (_label_requests, _seat_hero, label_chunks)

    current = (None, None)
    posterior_cache = PosteriorCache()
    try:
        for chunk in label_chunks(todo, R, ocfg.labels_per_batch):
            i, h = chunk[0][1], chunk[0][2]
            block = h // R
            if (i, block) != current:
                current = (i, block)
                _seat_hero(play_pool, (hero_plain[i], hero_rec[i]),
                           session_of[i], block_vectors[i], block, hero_factory)
            answers = action_values_batch(
                _label_requests(chunk, session_of, hero_plain, seed),
                driver, play_pool, ocfg,
                posterior_cache=posterior_cache)
            for (pos, _i, _h, _d), (q, legal, stats) in zip(chunk, answers):
                result_q.put((pos, q, legal, stats))
    finally:
        client.done()


# ------------------------------------------------------------- orchestration


def runner_table(play_pool, hero_prototype, game, device, log):
    """`{key: runner}` for every distinct network in the pool, and the mirrors.

    Keying is by object identity so that the style siblings sharing one loaded
    checkpoint share one runner instead of one per member.
    """
    runners, keys = {}, {}
    max_players = int(game["max_players"])
    n_actions = int(game["n_actions"])

    def key_of_net(net):
        if id(net) in keys:
            return keys[id(net)]
        key = f"net{len(keys)}"
        keys[id(net)] = key
        if isinstance(net, V7Agent):
            runners[key] = V7Runner(net, _amp_of(net))
        else:
            runners[key] = AgentRunner(net, max_players, n_actions, device)
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
                  hero_spec, hero_plain, hero_rec, runners, game, ocfg, seed,
                  R, log):
    """Start `n_workers` label workers over a session-strided partition.

    Sessions rather than decisions, because a worker then reads only its own
    slice of the records and rebuilds hero's member once per block exactly as
    the sequential path does. Strided rather than contiguous, so a slice is a
    mix of table sizes and stack depths rather than a run of neighbours.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    # A worker is one core's worth of Python. numpy's BLAS does not read
    # `torch.set_num_threads`, and it picks its thread count up from the
    # environment at import — which under `spawn` is this process's environment
    # as of `start()`. Left alone, N workers each open a pool the width of the
    # machine and 20 cores get several hundred threads to schedule.
    inherited = {k: os.environ.get(k) for k in _BLAS_VARS}
    os.environ.update({k: "1" for k in _BLAS_VARS})
    rows = slab_rows(ocfg.batch_hands, ocfg.max_combos)
    specs = [v7_fields(int(game["max_players"]), int(game["n_actions"]))]
    specs += [r.fields for r in runners.values()]
    n_out = max([int(game["n_actions"])]
                + [int(r.agent.n_actions) for r in runners.values()
                   if isinstance(r, V7Runner)])

    owners = {i: i % n_workers for i in range(len(sessions))}
    procs, conns, slabs = [], [], []
    for w in range(n_workers):
        mine = [t for t in todo if owners[t[1]] == w]
        session_of = {i: sessions[i] for i in range(len(sessions))
                      if owners[i] == w}
        vectors = {i: block_vectors[i] for i in session_of}
        slab = Slab(rows, specs, n_out)
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_worker_main,
            args=(w, mine, session_of, vectors, pool_spec, hero_spec,
                  hero_plain, hero_rec, int(game["n_actions"]), ocfg, seed, R,
                  slab, child_conn, result_q),
            daemon=True)
        p.start()
        child_conn.close()
        procs.append(p)
        conns.append(parent_conn)
        slabs.append(slab)
        log(f"[labels] worker {w}: {len(mine)} labels over "
            f"{len(session_of)} sessions (pid {p.pid})")
    for key, value in inherited.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    log(f"[labels] {n_workers} slabs of {slabs[0].nbytes() / 1e6:.0f} MB "
        f"shared memory, {rows} rows each")
    return procs, conns, slabs, result_q


def collect(server, result_q, procs, todo, start, consume, log, results=None):
    """Serve the workers and hand finished labels to `consume` in todo order.

    Workers own strided sessions, so labels come back out of order; the parent
    holds them until their turn. Consuming in order is what makes the shards
    this phase writes the same shards, in the same order, that one process
    would have written — which matters because `split_heldout` partitions by
    position, so a reordered corpus is a different held-out set.
    """
    # The caller may pass the reorder buffer in so it can report its depth on
    # the progress bar; it is this function's own dict either way.
    results = {} if results is None else results
    next_pos = int(start)

    def drain():
        drain_results(result_q, results)

    while True:
        alive = server.poll(drain)
        drain()
        next_pos = _consume_prefix(results, next_pos, todo, consume)
        if not alive:
            break

    # Every worker has said it is done, but `mp.Queue` is asynchronous: their
    # last results may still be in flight, and a process cannot be joined until
    # what it queued has been taken off.
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
    log(f"[labels] server: {server.requests} forwards, {server.rows} rows")
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
