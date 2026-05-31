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

import torch


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
REQ_FORWARD = "FORWARD_BATCH"


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
        self.req_q.put((rid, self.worker_id, agent_name, req_type, payload))
        out_rid, status, result = self.resp_q.get()
        if status == "ERROR":
            raise RemoteServerError(result)
        return result


class RemoteEvaluator(_RemoteBase):
    """Evaluator that offloads root/leaf forwards to the inference server."""

    def __init__(self, worker_id, agent_name, req_q, resp_q, n_actions):
        super().__init__(worker_id, req_q, resp_q)
        self.agent_name = agent_name
        self.n_actions = n_actions

    def evaluate_root(self, event_sequences):
        # event_sequences are plain dicts (np bets serialize fine via pickle).
        return self._rpc(self.agent_name, REQ_ROOT, event_sequences)

    def evaluate_leaves(self, batch_ctx, batch_mask, needs_expansion):
        payload = (
            batch_ctx.detach().to(torch.float16).cpu(),
            batch_mask.detach().cpu(),
            bool(needs_expansion),
        )
        return self._rpc(self.agent_name, REQ_LEAF, payload)


class EvalProxy(_RemoteBase):
    """Remote handle for terminal_eval range narrowing.

    `forward_batch(agent_name, batch_events, heads)` returns the action-head
    logits tensor (CPU) directly — terminal_eval only ever needs
    `heads={"action"}`.
    """

    def forward_batch(self, agent_name, batch_events, heads=("action",)):
        return self._rpc(agent_name, REQ_FORWARD, (batch_events, tuple(heads)))
