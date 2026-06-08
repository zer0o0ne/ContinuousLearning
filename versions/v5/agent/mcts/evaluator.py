"""
Evaluator boundary for MCTS neural-net access.

`MCTS` no longer calls `self.agent.*_head` / `self.agent.perception` directly:
it goes through an evaluator object that exposes exactly two calls plus
`n_actions`. This decouples the tree search (CPU, in actor processes) from the
neural-net forwards (GPU, in a central inference server) — see
`agent/mcts/inference_server.py` and the parallel path in
`agent/mcts/collect.py`.

Two implementations:

- `LocalEvaluator` wraps a live `ASI` and runs the forwards in-process. Its
  bodies are a verbatim move of the previous `mcts._evaluate_root` /
  `mcts._flush_pending` head calls, so the sequential collection path
  (`mcts_train.n_workers <= 1`) is bit-for-bit identical to before this
  refactor.

- `RemoteEvaluator` marshals the two calls to the inference server over
  multiprocessing queues and blocks for the response. Context tensors are sent
  in float16 to halve IPC volume (heads run in fp32 on the GPU; the server
  casts back). One request is in flight per actor at a time (each call blocks),
  so a worker's response queue only ever holds its own replies.

`EvalProxy` is the analogous remote handle for `terminal_eval`'s only model
call (`agent.forward_batch(..., heads={"action"})` for range narrowing).
"""

import os
import time

import torch

from agent.mcts import _timing


# Optimization switch shared with `inference_server.py`. When 1 (default), the
# parallel path uses:
#   - REQ_LEAF_CACHED: actor ships only the action-embedding deltas; the server
#     reconstructs full context from a cached root_ctx per (worker, agent).
#     Eliminates re-shipping the (large) root_ctx in every LEAF flush (~625×
#     per decision per actor in a typical 10k-sim run).
#   - REQ_FORWARD_TEMPLATED: opponent_data range inference ships the event
#     template ONCE plus N hand pairs, server replicates server-side — saves
#     ~256× the dict-spine pickling cost per range call.
# Set to 0 to fall back to the original REQ_LEAF / REQ_FORWARD paths for A/B
# benchmarking. The flag is consulted only by RemoteEvaluator / EvalProxy /
# the server's request handler — LocalEvaluator is unaffected.
PARALLEL_OPT_ENABLED = os.environ.get("MCTS_PARALLEL_OPT", "1") == "1"


class RemoteServerError(RuntimeError):
    """Raised in an actor when the inference server reported an error for a
    request (the server-side traceback is carried as the message)."""


class LocalEvaluator:
    """In-process evaluator wrapping a live ASI. Bit-for-bit equivalent to the
    pre-refactor direct `self.agent.*` calls in `mcts.py`."""

    def __init__(self, agent, device, opponent_emb_table=None):
        self.agent = agent
        self.device = device
        self.opponent_emb_table = opponent_emb_table
        self.n_actions = agent.n_actions

    def evaluate_root(self, event_sequences):
        # Verbatim move of the old `mcts._evaluate_root` body.
        skip_opp = self.opponent_emb_table is None
        p_out, encoded, mask = self.agent.perception.forward_batch(
            event_sequences, device=self.device, skip_memory=True,
            skip_opponent_emb=skip_opp,
            opponent_emb_table=self.opponent_emb_table,
        )
        value = self.agent.value_head(p_out, mask=mask)
        act_logits = self.agent.action_head(p_out, mask=mask)
        opp_logits = self.agent.opponent_action_head(p_out, mask=mask)
        act_embs = self.agent.modelling_head(p_out, mask=mask)
        return p_out, mask, value, act_logits, opp_logits, act_embs

    def evaluate_leaves(self, batch_ctx, batch_mask, needs_expansion):
        # Verbatim move of the old `mcts._flush_pending` head block.
        values = self.agent.value_head(batch_ctx, mask=batch_mask)
        if needs_expansion:
            act_logits = self.agent.action_head(batch_ctx, mask=batch_mask)
            opp_logits = self.agent.opponent_action_head(batch_ctx, mask=batch_mask)
            act_embs = self.agent.modelling_head(batch_ctx, mask=batch_mask)
            return values, act_logits, opp_logits, act_embs
        return values, None, None, None


# Request type tags (shared with inference_server.py).
REQ_ROOT = "ROOT"
REQ_LEAF = "LEAF"
REQ_LEAF_CACHED = "LEAF_CACHED"     # uses server-side root_ctx cache
REQ_FORWARD = "FORWARD_BATCH"
REQ_FORWARD_TEMPLATED = "FORWARD_TEMPLATED"  # template + hand pairs


class _RemoteBase:
    """Shared RPC plumbing: one blocking request at a time over (req_q, resp_q).

    Each actor has its own `resp_q` and only ever has a single request in
    flight (every call blocks until the reply arrives), so the next item on
    `resp_q` is always this call's response.
    """

    def __init__(self, worker_id, req_q, resp_q):
        self.worker_id = worker_id
        self.req_q = req_q
        self.resp_q = resp_q
        self._req_id = 0

    def _rpc(self, agent_name, req_type, payload):
        self._req_id += 1
        rid = (self.worker_id, self._req_id)
        if _timing.ENABLED:
            t0 = time.monotonic()
            self.req_q.put((rid, self.worker_id, agent_name, req_type, payload))
            t1 = time.monotonic()
            out_rid, status, result = self.resp_q.get()
            t2 = time.monotonic()
            _timing.event(
                "actor_rpc",
                worker_id=self.worker_id,
                agent=agent_name,
                rtype=req_type,
                put_ms=(t1 - t0) * 1000.0,
                wait_ms=(t2 - t1) * 1000.0,
            )
        else:
            self.req_q.put((rid, self.worker_id, agent_name, req_type, payload))
            out_rid, status, result = self.resp_q.get()
        if status == "ERROR":
            raise RemoteServerError(result)
        return result


class RemoteEvaluator(_RemoteBase):
    """Evaluator that offloads root/leaf forwards to the inference server.

    With `PARALLEL_OPT_ENABLED` (default), LEAF requests ship only the action-
    embedding deltas — the part of the context that's NOT the cached root_ctx.
    The server holds a per-(worker, agent) root_ctx cache populated on every
    ROOT response, so each LEAF flush avoids re-shipping the (1, L_root, d)
    perception output (the dominant IPC payload at high simulation counts).

    `self._root_len` is set after every `evaluate_root` to the length of the
    cached prefix; LEAF slicing uses it to extract just the delta.
    """

    def __init__(self, worker_id, agent_name, req_q, resp_q, n_actions):
        super().__init__(worker_id, req_q, resp_q)
        self.agent_name = agent_name
        self.n_actions = n_actions
        self._root_len = 0

    def evaluate_root(self, event_sequences):
        # event_sequences are plain dicts (np bets serialize fine via pickle).
        result = self._rpc(self.agent_name, REQ_ROOT, event_sequences)
        # Cache the root prefix length locally so subsequent LEAF flushes can
        # strip it off before sending. Server-side cache is keyed by the same
        # (worker_id, agent_name) and is populated by `_run_root`.
        # result = (p_out, mask, value, act_logits, opp_logits, act_embs)
        p_out = result[0]
        self._root_len = int(p_out.shape[1])
        return result

    def evaluate_leaves(self, batch_ctx, batch_mask, needs_expansion):
        if PARALLEL_OPT_ENABLED and self._root_len > 0:
            # Strip the cached root prefix and ship only the deltas. The server
            # rebuilds full context by concatenating its cached root_ctx.
            r = self._root_len
            # If a path has no delta (depth 0 → leaf is root, only possible for
            # degenerate trees), batch_ctx.shape[1] == r and delta is empty.
            delta_ctx = batch_ctx[:, r:, :].detach().to(torch.float16).cpu()
            delta_mask = batch_mask[:, r:].detach().cpu()
            payload = (delta_ctx, delta_mask, bool(needs_expansion))
            return self._rpc(self.agent_name, REQ_LEAF_CACHED, payload)
        # Legacy path: full batch_ctx over the wire.
        payload = (
            batch_ctx.detach().to(torch.float16).cpu(),
            batch_mask.detach().cpu(),
            bool(needs_expansion),
        )
        return self._rpc(self.agent_name, REQ_LEAF, payload)


class EvalProxy(_RemoteBase):
    """Remote handle for terminal_eval range narrowing AND opponent_data range
    inference.

    Two transports:

    - `forward_batch(agent_name, batch_events, heads)` ships the full
      `batch_events` over the wire (one list per combo). Kept for terminal_eval
      and as the legacy path. With `PARALLEL_OPT_ENABLED`, opponent_data
      switches to `forward_batch_templated` (below) to amortize the dict-spine
      pickling cost across combos.

    - `forward_batch_templated(agent_name, template_events, hand_pairs, heads)`
      ships the template ONCE plus a list of `(c1, c2)` hand pairs. The server
      replicates the template per pair and overwrites the `hand` field. Returns
      the action-head logits with shape `(len(hand_pairs), n_actions)`.
    """

    def forward_batch(self, agent_name, batch_events, heads=("action",)):
        return self._rpc(agent_name, REQ_FORWARD, (batch_events, tuple(heads)))

    def forward_batch_templated(self, agent_name, template_events, hand_pairs,
                                heads=("action",)):
        payload = (template_events, list(hand_pairs), tuple(heads))
        return self._rpc(agent_name, REQ_FORWARD_TEMPLATED, payload)
