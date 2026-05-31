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
from agent.mcts.evaluator import REQ_ROOT, REQ_LEAF, REQ_FORWARD


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


def _run_root(agent, worker_id, agent_name, event_sequences, device, get_table):
    """Perception + all 4 heads on one event sequence (batch size 1).

    Mirrors `LocalEvaluator.evaluate_root`: `skip_opponent_emb` is False exactly
    when the agent has opponent embedding enabled (then a per-(worker, agent)
    table is used and mutated by the GRU)."""
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
    return (p_out.cpu(), mask.cpu(), value.cpu(),
            act_logits.cpu(), opp_logits.cpu(), act_embs.cpu())


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
        v = values[off:off + n].cpu()
        if needs[i]:
            out.append((v, act[off:off + n].cpu(),
                        opp[off:off + n].cpu(), embs[off:off + n].cpu()))
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


def _run_group(agent, rtype, reqs, device, get_table, resp_qs):
    """Run one (agent, req_type) group and scatter responses to the actors."""
    if rtype == REQ_ROOT:
        for (rid, wid, name, _rtype, payload) in reqs:
            res = _run_root(agent, wid, name, payload, device, get_table)
            resp_qs[wid].put((rid, "OK", res))
    elif rtype == REQ_LEAF:
        results = _run_leaf_batch(agent, reqs, device)
        for req, res in zip(reqs, results):
            resp_qs[req[1]].put((req[0], "OK", res))
    elif rtype == REQ_FORWARD:
        results = _run_forward_batch(agent, reqs, device)
        for req, res in zip(reqs, results):
            resp_qs[req[1]].put((req[0], "OK", res))


def server_main(spec, req_q, resp_qs, ready_event, stop_event, server_cfg):
    """Inference server entry point (runs in its own process).

    spec: list of {name, config, state_dict, norm_stats}.
    req_q: shared request queue (all actors put here).
    resp_qs: list of per-worker response queues (indexed by worker_id).
    ready_event: set once all models are loaded (parent waits on this).
    stop_event: set by the parent to request shutdown.
    server_cfg: {device, server_max_batch, server_linger_ms}.
    """
    torch.set_grad_enabled(False)
    device = server_cfg.get("device", "cuda")
    max_batch = int(server_cfg.get("server_max_batch", 256))
    linger = float(server_cfg.get("server_linger_ms", 2)) / 1000.0

    opp_tables = {}

    def get_table(worker_id, agent_name):
        key = (worker_id, agent_name)
        t = opp_tables.get(key)
        if t is None:
            from agent.perception.opponent_embeddings import OpponentEmbeddingTable
            t = OpponentEmbeddingTable(agents[agent_name].perception.d_model)
            opp_tables[key] = t
        return t

    try:
        agents = _build_agents(spec, device)
        ready_event.set()

        while not stop_event.is_set():
            try:
                first = req_q.get(timeout=0.1)
            except Exception:
                continue  # queue.Empty — re-check stop_event
            if first is None:  # SENTINEL
                break

            # Form a batch: drain whatever is queued, up to max_batch, within a
            # short linger window. With N blocked actors the queue naturally
            # holds up to N requests, so linger mainly smooths arrival jitter.
            bucket = [first]
            deadline = time.monotonic() + linger
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

            # Group by (req_type, agent, worker_for_ROOT). ROOT is keyed by
            # worker too because each worker has its own opp-emb table.
            groups = defaultdict(list)
            for req in bucket:
                _rid, wid, name, rtype, _payload = req
                key = (rtype, name, wid if rtype == REQ_ROOT else None)
                groups[key].append(req)

            for (rtype, name, _wid), reqs in groups.items():
                try:
                    _run_group(agents[name], rtype, reqs, device, get_table, resp_qs)
                except Exception:
                    tb = traceback.format_exc()
                    for req in reqs:
                        resp_qs[req[1]].put((req[0], "ERROR", tb))
    except Exception:
        # Model build / fatal loop error: surface to anyone waiting.
        tb = traceback.format_exc()
        for q in resp_qs:
            try:
                q.put((None, "ERROR", tb))
            except Exception:
                pass
        raise
    finally:
        # Unblock any actor still waiting on a response so it can't hang.
        for q in resp_qs:
            try:
                q.put((None, "ERROR", "inference server shut down"))
            except Exception:
                pass
