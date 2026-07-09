import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""
GPU inference server for parallel MCTS collection.

Runs in a single dedicated process holding every agent's model on the GPU.
CPU actor processes (see `agent/mcts/collect.py`) run the tree search and game
logic and offload all neural-net forwards here via multiprocessing queues. The
server coalesces requests arriving from many actors into large GPU batches —
this is the whole point of the actor/server split: each actor keeps its own
within-tree `batch_size` (16) unchanged, and cross-actor batching provides GPU
efficiency on top.

Request types (tags defined in `evaluator.py`):
  ROOT          — perception + 4 heads on one event sequence (one decision root).
                  Uses a per-(worker, agent) opponent-embedding table (variant B)
                  so the stateful GRU accumulates within each actor's hand stream
                  without cross-actor races. ROOT is rare (1/decision) and is run
                  per request (not cross-actor batched) because each worker has
                  its own table.
  LEAF          — value (+ action/opponent/modelling when expanding) on already-
                  assembled context tensors. Batched ACROSS actors: contexts are
                  padded to a common length and stacked into one big forward.
  FORWARD_BATCH — action-head logits on event batches, for terminal_eval range
                  narrowing. Batched across actors by agent.

Protocol:
  request  = (req_id, worker_id, agent_name, req_type, payload)   on shared req_q
  response = (req_id, "OK"|"ERROR", result)                       on resp_qs[worker_id]
  SENTINEL = None on req_q triggers shutdown.
"""

import time
import traceback
from collections import defaultdict

import torch

from agent.agent import ASI
from agent.mcts.evaluator import (
    REQ_ROOT, REQ_LEAF, REQ_LEAF_CACHED, REQ_FORWARD, REQ_FORWARD_TEMPLATED,
)
from agent.mcts import _timing


def _server_log(*args, **kwargs):
    """No-op logger for ASI construction inside the server process."""
    pass


def _build_agents(spec, device):
    """Build and load every agent on `device` in eval mode.

    spec: list of dicts {name, config, state_dict, norm_stats}. state_dict holds
    CPU tensors; set_device moves the model onto `device`.
    """
    agents = {}
    for s in spec:
        agent = ASI(_server_log, s["config"])
        agent.load_state_dict(s["state_dict"], strict=False)
        agent.set_device(device)
        agent.eval()
        agents[s["name"]] = agent
    return agents


def _server_pad_and_stack(ctxs, masks, device):
    """Pad a list of (B_i, L_i, d) contexts + (B_i, L_i) masks to a common
    L_max and stack along the batch dim into (ΣB_i, L_max, d) / (ΣB_i, L_max).

    Cross-request analogue of `mcts._pad_and_stack`. Padding rows are masked
    out by the heads' attention mask, so they never contaminate real rows.
    Contexts arrive as float16 (IPC); cast back to float32 for the model.
    """
    L_max = max(c.shape[1] for c in ctxs)
    d = ctxs[0].shape[2]
    total = sum(c.shape[0] for c in ctxs)
    big = torch.zeros(total, L_max, d, dtype=torch.float32)
    big_mask = torch.zeros(total, L_max, dtype=masks[0].dtype)
    off = 0
    for c, m in zip(ctxs, masks):
        B_i, L_i = c.shape[0], c.shape[1]
        big[off:off + B_i, :L_i] = c.float()
        big_mask[off:off + B_i, :L_i] = m
        off += B_i
    return big.to(device), big_mask.to(device)


def _run_root(agent, worker_id, agent_name, event_sequences, device, get_table,
              root_cache):
    """Perception + all 4 heads on one event sequence (batch size 1).

    Mirrors `LocalEvaluator.evaluate_root`: `skip_opponent_emb` is False exactly
    when the agent has opponent embedding enabled (then a per-(worker, agent)
    table is used and mutated by the GRU).

    Side effect: stores the GPU-side `(p_out, mask)` in `root_cache` keyed by
    `(worker_id, agent_name)`. Subsequent `REQ_LEAF_CACHED` requests from that
    pair rebuild full LEAF context server-side using this cache, so the actor
    never re-ships root_ctx. Each ROOT overwrites the cached value; one
    decision corresponds to exactly one ROOT, so freshness is automatic."""
    if agent.perception.opp_emb_enabled:
        table = get_table(worker_id, agent_name)
        skip = False
    else:
        table = None
        skip = True
    p_out, encoded, mask = agent.perception.forward_batch(
        event_sequences, device=device, skip_memory=True,
        skip_opponent_emb=skip, opponent_emb_table=table)
    value = agent.value_head(p_out, mask=mask)
    act_logits = agent.action_head(p_out, mask=mask)
    opp_logits = agent.opponent_action_head(p_out, mask=mask)
    act_embs = agent.modelling_head(p_out, mask=mask)
    # Cache the GPU-resident root_ctx + root_mask for downstream LEAF_CACHED.
    # `.detach()` ensures we don't hold autograd state — torch.set_grad_enabled
    # is already False at server start, but explicit is safer.
    root_cache[(worker_id, agent_name)] = (p_out.detach(), mask.detach())
    root_len = int(p_out.shape[1])
    return (root_len, value.half().cpu(),
            act_logits.half().cpu(), opp_logits.half().cpu(),
            act_embs.half().cpu())


def _run_root_batch(agent, reqs, device, get_table, root_cache, resp_qs):
    """Batched ROOT: perception + 4 heads on N event sequences at once.

    Opponent-embedding must be handled per-worker (each worker mutates its own
    GRU table). When opp_emb is enabled, we fall back to serial execution;
    when disabled, we batch all event sequences into one forward.
    """
    if agent.perception.opp_emb_enabled:
        for (rid, wid, name, _rtype, payload) in reqs:
            res = _run_root(agent, wid, name, payload, device, get_table,
                            root_cache)
            resp_qs[wid].put((rid, "OK", res))
        return

    all_events = []
    for (_rid, _wid, _name, _rtype, payload) in reqs:
        all_events.extend(payload)

    p_out, encoded, mask = agent.perception.forward_batch(
        all_events, device=device, skip_memory=True,
        skip_opponent_emb=True, opponent_emb_table=None)
    values = agent.value_head(p_out, mask=mask)
    act_logits = agent.action_head(p_out, mask=mask)
    opp_logits = agent.opponent_action_head(p_out, mask=mask)
    act_embs = agent.modelling_head(p_out, mask=mask)

    off = 0
    for (rid, wid, name, _rtype, payload) in reqs:
        n = len(payload)
        po = p_out[off:off + n]
        mk = mask[off:off + n]
        root_cache[(wid, name)] = (po.detach(), mk.detach())
        root_len = int(po.shape[1])
        res = (root_len,
               values[off:off + n].half().cpu(),
               act_logits[off:off + n].half().cpu(),
               opp_logits[off:off + n].half().cpu(),
               act_embs[off:off + n].half().cpu())
        resp_qs[wid].put((rid, "OK", res))
        off += n


def _run_leaf_batch(agent, reqs, device):
    """Value (+ expansion heads) on the stacked contexts of all LEAF requests.

    Returns a list aligned with `reqs`; each entry is
    (values_i, act_logits_i|None, opp_logits_i|None, act_embs_i|None) sliced to
    that request's own batch size. Mirrors `LocalEvaluator.evaluate_leaves`:
    expansion heads are returned only for requests whose own batch needs
    expansion (None otherwise), exactly as the in-process path."""
    ctxs, masks, sizes, needs = [], [], [], []
    for (_rid, _wid, _name, _rtype, payload) in reqs:
        ctx, msk, ne = payload
        ctxs.append(ctx)
        masks.append(msk)
        sizes.append(ctx.shape[0])
        needs.append(bool(ne))

    big_ctx, big_mask = _server_pad_and_stack(ctxs, masks, device)
    values = agent.value_head(big_ctx, mask=big_mask)
    any_exp = any(needs)
    act = opp = embs = None
    if any_exp:
        act = agent.action_head(big_ctx, mask=big_mask)
        opp = agent.opponent_action_head(big_ctx, mask=big_mask)
        embs = agent.modelling_head(big_ctx, mask=big_mask)

    out = []
    off = 0
    for i, n in enumerate(sizes):
        v = values[off:off + n].half().cpu()
        if needs[i]:
            out.append((v, act[off:off + n].half().cpu(),
                        opp[off:off + n].half().cpu(),
                        embs[off:off + n].half().cpu()))
        else:
            out.append((v, None, None, None))
        off += n
    return out


def _run_leaf_cached_batch(agent, reqs, device, root_cache):
    """Like `_run_leaf_batch` but the actor shipped only the action-embedding
    delta — the root_ctx prefix lives on the server in `root_cache`.

    For each request, look up the cached `(root_ctx, root_mask)` by
    `(worker_id, agent_name)`, concatenate with the delta, then aggregate
    via `_server_pad_and_stack` exactly as the legacy path. The cache is
    populated by `_run_root` and guaranteed present because MCTS always
    calls `evaluate_root` before any `evaluate_leaves` on the same actor."""
    ctxs, masks, sizes, needs = [], [], [], []
    for (_rid, wid, name, _rtype, payload) in reqs:
        delta_ctx, delta_mask, ne = payload
        cached = root_cache.get((wid, name))
        if cached is None:
            raise RuntimeError(
                f"LEAF_CACHED received for (worker={wid}, agent={name}) "
                f"but no ROOT has populated the cache. "
                f"Did the actor call evaluate_root before evaluate_leaves?")
        root_ctx, root_mask = cached  # both on `device`, fp32
        B_i = delta_ctx.shape[0]
        # delta_ctx may have no delta rows (depth=0) → shape (B_i, 0, d). cat
        # handles that as a no-op. delta_ctx arrives in fp16; cast to fp32 to
        # match the cached root_ctx dtype before concat.
        delta_ctx_gpu = delta_ctx.to(device=device, dtype=torch.float32,
                                     non_blocking=True)
        delta_mask_gpu = delta_mask.to(device=device, non_blocking=True)
        root_ctx_exp = root_ctx.expand(B_i, -1, -1)
        root_mask_exp = root_mask.expand(B_i, -1)
        ctx_full = torch.cat([root_ctx_exp, delta_ctx_gpu], dim=1)
        mask_full = torch.cat([root_mask_exp, delta_mask_gpu], dim=1)
        ctxs.append(ctx_full)
        masks.append(mask_full)
        sizes.append(B_i)
        needs.append(bool(ne))

    # Pad to max L across requests and concat along batch dim. We're already
    # on `device` (unlike _server_pad_and_stack which expects CPU input), so
    # inline the pad-and-stack here to avoid pointless GPU→CPU→GPU round-trip.
    L_max = max(c.shape[1] for c in ctxs)
    d = ctxs[0].shape[2]
    total = sum(sizes)
    big_ctx = torch.zeros(total, L_max, d, device=device, dtype=torch.float32)
    big_mask = torch.zeros(total, L_max, device=device, dtype=masks[0].dtype)
    off = 0
    for c, m, n in zip(ctxs, masks, sizes):
        L_i = c.shape[1]
        big_ctx[off:off + n, :L_i] = c
        big_mask[off:off + n, :L_i] = m
        off += n

    values = agent.value_head(big_ctx, mask=big_mask)
    any_exp = any(needs)
    act = opp = embs = None
    if any_exp:
        act = agent.action_head(big_ctx, mask=big_mask)
        opp = agent.opponent_action_head(big_ctx, mask=big_mask)
        embs = agent.modelling_head(big_ctx, mask=big_mask)

    out = []
    off = 0
    for i, n in enumerate(sizes):
        v = values[off:off + n].half().cpu()
        if needs[i]:
            out.append((v, act[off:off + n].half().cpu(),
                        opp[off:off + n].half().cpu(),
                        embs[off:off + n].half().cpu()))
        else:
            out.append((v, None, None, None))
        off += n
    return out


def _run_forward_batch(agent, reqs, device):
    """action-head logits on the concatenated event batches of all FORWARD
    requests. Returns a list aligned with `reqs` (each its own action_logits)."""
    all_events, lens = [], []
    heads = ("action",)
    for (_rid, _wid, _name, _rtype, payload) in reqs:
        evts, hds = payload
        all_events.extend(evts)
        lens.append(len(evts))
        heads = hds
    out = agent.forward_batch(all_events, skip_memory=True, heads=set(heads))
    logits = out["action_logits"]
    res, off = [], 0
    for n in lens:
        res.append(logits[off:off + n].cpu())
        off += n
    return res


def _run_forward_templated_batch(agent, reqs, device):
    """Templated counterpart to `_run_forward_batch` for opponent_data range
    inference. Each request carries (template_events, hand_pairs, heads) where
    `template_events` is sent ONCE; the server replicates it per hand pair and
    swaps in the hand. This avoids the actor pickling N (=combo count) full
    copies of the event sequence per range call — that path used to dominate
    `_compute_range_probs` IPC time for typical 100-event histories × 256
    combos × pickled-dict overhead.

    Cross-request: we still batch all requests' (combo × events) into one
    `agent.forward_batch` to amortize per-call setup."""
    all_events, lens = [], []
    heads = ("action",)
    for (_rid, _wid, _name, _rtype, payload) in reqs:
        template, hand_pairs, hds = payload
        heads = hds
        # Per-request expansion: one combo = one shallow-copied event list
        # with the `hand` slot overwritten. `dict(e)` matches the actor's old
        # behaviour (`[dict(e) for e in template]` in _compute_range_probs).
        # `hand` lives in every event because perception's per-event hand_emb
        # consumes it.
        for c1, c2 in hand_pairs:
            events = [dict(e) for e in template]
            for e in events:
                e["hand"] = [c1, c2]
            all_events.append(events)
        lens.append(len(hand_pairs))
    out = agent.forward_batch(all_events, skip_memory=True, heads=set(heads))
    logits = out["action_logits"]
    res, off = [], 0
    for n in lens:
        res.append(logits[off:off + n].cpu())
        off += n
    return res


def _run_group(agent, rtype, reqs, device, get_table, resp_qs, root_cache):
    """Run one (agent, req_type) group and scatter responses to the actors."""
    n_leaves = 0
    if rtype == REQ_LEAF:
        for r in reqs:
            n_leaves += r[4][0].shape[0]  # ctx.shape[0] per request
    elif rtype == REQ_LEAF_CACHED:
        for r in reqs:
            n_leaves += r[4][0].shape[0]  # delta_ctx.shape[0]
    elif rtype == REQ_FORWARD:
        for r in reqs:
            n_leaves += len(r[4][0])  # len(events)
    elif rtype == REQ_FORWARD_TEMPLATED:
        for r in reqs:
            n_leaves += len(r[4][1])  # len(hand_pairs)

    with _timing.span("server_run_group", rtype=rtype, n_reqs=len(reqs),
                      n_items=n_leaves):
        if rtype == REQ_ROOT:
            _run_root_batch(agent, reqs, device, get_table, root_cache, resp_qs)
        elif rtype == REQ_LEAF:
            results = _run_leaf_batch(agent, reqs, device)
            for req, res in zip(reqs, results):
                resp_qs[req[1]].put((req[0], "OK", res))
        elif rtype == REQ_LEAF_CACHED:
            results = _run_leaf_cached_batch(agent, reqs, device, root_cache)
            for req, res in zip(reqs, results):
                resp_qs[req[1]].put((req[0], "OK", res))
        elif rtype == REQ_FORWARD:
            results = _run_forward_batch(agent, reqs, device)
            for req, res in zip(reqs, results):
                resp_qs[req[1]].put((req[0], "OK", res))
        elif rtype == REQ_FORWARD_TEMPLATED:
            results = _run_forward_templated_batch(agent, reqs, device)
            for req, res in zip(reqs, results):
                resp_qs[req[1]].put((req[0], "OK", res))
        if _timing.ENABLED and isinstance(device, str) and device.startswith("cuda"):
            # Sync so the span captures actual GPU time, not just launch time.
            torch.cuda.synchronize()


def server_main(spec, req_q, resp_qs, ready_event, stop_event, server_cfg):
    """Inference server entry point (runs in its own process).

    spec: list of {name, config, state_dict, norm_stats}.
    req_q: shared request queue (all actors put here).
    resp_qs: list of per-worker response queues (indexed by worker_id).
    ready_event: set once all models are loaded (parent waits on this).
    stop_event: set by the parent to request shutdown.
    server_cfg: {device, server_max_batch, server_linger_ms}.
    """
    import sys
    torch.set_grad_enabled(False)
    device = server_cfg.get("device", "cuda")
    max_batch = int(server_cfg.get("server_max_batch", 256))
    n_workers = int(server_cfg.get("n_workers", 1))
    base_linger = float(server_cfg.get("server_linger_ms", 2))
    # E.2.3: with few workers, linger longer to accumulate a batch; with many,
    # the queue fills fast so linger can be shorter. Scale inversely so the
    # effective batch size stays ~constant regardless of worker count.
    if n_workers >= 4:
        linger = base_linger * max(0.25, 2.0 / n_workers) / 1000.0
    else:
        linger = base_linger / 1000.0

    sys.stderr.write(f"[server] starting on device={device} max_batch={max_batch} "
                     f"n_workers={n_workers} linger={linger*1000:.1f}ms\n")
    sys.stderr.flush()

    opp_tables = {}
    # Server-side root_ctx cache: per-(worker, agent) GPU tensors populated by
    # `_run_root` and consumed by `_run_leaf_cached_batch`. Mirrors the actor's
    # LocalEvaluator behaviour where root_ctx is computed once per decision
    # and reused for every LEAF flush. Memory cost: O(n_workers × n_agents),
    # ~25–50 KB per entry — negligible. Each new ROOT overwrites freely (one
    # decision = one root), so no LRU bound is needed.
    root_cache = {}

    def get_table(worker_id, agent_name):
        key = (worker_id, agent_name)
        t = opp_tables.get(key)
        if t is None:
            from agent.perception.opponent_embeddings import OpponentEmbeddingTable
            t = OpponentEmbeddingTable(agents[agent_name].perception.d_model)
            opp_tables[key] = t
        return t

    fatal_reason = None
    try:
        agents = _build_agents(spec, device)
        ready_event.set()
        sys.stderr.write(f"[server] ready, {len(agents)} agent(s) loaded\n")
        sys.stderr.flush()

        while not stop_event.is_set():
            try:
                first = req_q.get(timeout=0.1)
            except Exception:
                continue  # queue.Empty — re-check stop_event
            if first is None:  # SENTINEL
                break

            # Per-iteration outer guard: ANY BaseException raised below
            # (bucket forming, group iteration, _run_group, agents lookup,
            # torch internals) is caught, broadcast to actors, and the server
            # continues to the next request. An actor's death must not kill
            # the server — neither directly nor via a downstream CUDA fault.
            try:
                # Form a batch: drain whatever is queued, up to max_batch,
                # within a short linger window. With N blocked actors the
                # queue naturally holds up to N requests, so linger mainly
                # smooths arrival jitter.
                bucket = [first]
                deadline = time.monotonic() + linger
                bucket_t0 = time.monotonic() if _timing.ENABLED else 0.0
                while len(bucket) < max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        r = req_q.get(timeout=remaining)
                    except Exception:
                        break
                    if r is None:  # SENTINEL mid-drain
                        stop_event.set()
                        break
                    bucket.append(r)
                if _timing.ENABLED:
                    _timing.event(
                        "server_bucket_formed",
                        n_reqs=len(bucket),
                        linger_used_ms=(time.monotonic() - bucket_t0) * 1000.0,
                        hit_max=(len(bucket) >= max_batch),
                    )

                # Group by (req_type, agent). ROOT requests are batched across
                # workers when opp_emb is disabled; _run_root_batch falls back
                # to serial when opp_emb is on (each worker mutates its own
                # GRU table). LEAF_CACHED looks up per-(worker, agent) cache
                # entries internally, so cross-worker batching is safe.
                groups = defaultdict(list)
                for req in bucket:
                    _rid, wid, name, rtype, _payload = req
                    groups[(rtype, name, None)].append(req)

                for (rtype, name, _wid), reqs in groups.items():
                    try:
                        _run_group(agents[name], rtype, reqs, device,
                                   get_table, resp_qs, root_cache)
                    except BaseException:
                        # BaseException so SystemExit/KeyboardInterrupt from
                        # torch don't kill the server silently — surface them
                        # to the actors involved in this group.
                        tb = traceback.format_exc()
                        sys.stderr.write(
                            f"[server] _run_group failed "
                            f"(rtype={rtype} agent={name} n_reqs={len(reqs)}):"
                            f"\n{tb}\n")
                        sys.stderr.flush()
                        for req in reqs:
                            try:
                                resp_qs[req[1]].put((req[0], "ERROR", tb))
                            except Exception:
                                pass
            except BaseException:
                # Last-resort guard: error outside any inner try (e.g. bad
                # req unpack, KeyError on agents[name] before _run_group,
                # something deeper). Log it, notify ALL actors so any blocked
                # _rpc gets unstuck, and keep the server alive.
                tb = traceback.format_exc()
                sys.stderr.write(
                    f"[server] per-iter error (server stays alive):\n{tb}\n")
                sys.stderr.flush()
                for q in resp_qs:
                    try:
                        q.put((None, "ERROR", tb))
                    except Exception:
                        pass
                continue
    except BaseException:
        # Model build / fatal loop error: surface to anyone waiting.
        tb = traceback.format_exc()
        fatal_reason = tb
        sys.stderr.write(f"[server] FATAL in main loop:\n{tb}\n")
        sys.stderr.flush()
        for q in resp_qs:
            try:
                q.put((None, "ERROR", tb))
            except Exception:
                pass
        # Don't re-raise — let finally run cleanly and the process exit 0 so
        # the parent's death-detector reads a clean state. The traceback is
        # already on stderr and on every resp_q.
    finally:
        if fatal_reason is None:
            sys.stderr.write("[server] shutting down (clean)\n")
        else:
            sys.stderr.write("[server] shutting down after fatal error\n")
        sys.stderr.flush()
        # Unblock any actor still waiting on a response so it can't hang.
        for q in resp_qs:
            try:
                q.put((None, "ERROR", "inference server shut down"))
            except Exception:
                pass
