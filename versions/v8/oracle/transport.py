"""Shared-memory transport between a label worker and the inference server.

**Why this exists, measured.** The first version of `oracle/parallel.py` sent the
model's inputs through an `mp.Queue` as a dict of tensors. `torch`'s IPC gives
every tensor its own shared-memory segment, created, fd-passed and mapped per
send, so a request cost **8.7–13.9 ms** on the dev box (ten tensors each way) —
comparable to the forward it was carrying, tens of times per label. Labelling
got *slower* with workers, which is the opposite of the point.

Allocating the buffers **once** and sending only offsets brings the same round
trip to **0.062 ms** — a factor of 140–220. That is what this module is: one
slab of shared memory per worker, plus a `Pipe` carrying a small tuple.

    worker                                  server (the parent)
    ------                                  -------------------
    pack fields into the slab               unpack views over the same slab
    conn.send((key, kind, rows, …))  ─────► run the model
    conn.recv()                      ◄───── write logits into out, send back
    read out[:B]

**The layout is a table, not a format.** `v7_fields` and `token_fields` list
what each kind of request carries — name, width, dtype — and `pack`/`unpack`
walk that list. Each field lands in a contiguous region so the tensor the server
hands the model is contiguous and its `.to(device)` is a straight copy.

Nothing here decides *what* is computed. A payload packed and unpacked is the
same tensor that would have been passed in-process, so a forward taken through
this transport is bit-for-bit the forward taken without it.
"""

import torch

INT, FLOAT, BOOL = "i", "f", "b"

# A v8 hand cannot produce more events than this: `max_actions_for(9) = 62`
# decisions plus one terminal event per seat. It bounds the slab, and `pack`
# refuses loudly rather than writing past it.
MAX_EVENTS_PER_HAND = 80


def v7_fields(max_players, n_actions):
    """What `vendor.v7.perception.extract_event_tensors` produces, per event."""
    return (
        ("card_ids", 7, INT),
        ("hero_pos", 1, INT),
        ("acting_pos", 1, INT),
        ("num_players", 1, INT),
        ("batch_idx", 1, INT),
        ("event_idx", 1, INT),
        ("scalars", 2, FLOAT),
        ("bets", max_players, FLOAT),
        ("stacks", max_players, FLOAT),
        ("actions", n_actions, FLOAT),
    )


def token_fields(max_players, n_actions, d_emb):
    """What `nets.features.collate` produces, per token, plus the embeddings.

    The three derived masks `collate` appends are not carried: they are
    functions of `mask`, `token_type` and `own_strength`, and `collate` is where
    that rule is written. The server recomputes them through it.
    """
    return (
        ("cards", 7, INT),
        ("decision_idx", 1, INT),
        ("acting_pos", 1, INT),
        ("num_players", 1, INT),
        ("member", 1, INT),
        ("slot", 1, INT),
        ("action", 1, INT),
        ("token_type", 1, INT),
        ("sd_class", 1, INT),
        ("legal", n_actions, BOOL),
        ("scalars", 3, FLOAT),
        ("seat_stacks", max_players, FLOAT),
        ("prev_action", n_actions, FLOAT),
        ("sd_strength", 1, FLOAT),
        ("own_strength", 1, FLOAT),
        ("mask", 1, FLOAT),
        ("emb", d_emb, FLOAT),
    )


def _widths(fields):
    out = {INT: 0, FLOAT: 0, BOOL: 0}
    for _name, cols, dtype in fields:
        out[dtype] += cols
    return out


class Slab:
    """One worker's buffers, allocated once and never sent again.

    Sized for the largest request either kind can make: a policy query covers at
    most `max_rows` situations (a driver batch, or a posterior's combos) and a
    v7 sequence at most `MAX_EVENTS_PER_HAND` events. The memory is a shared
    mapping, so pages the run never touches are never backed.
    """

    def __init__(self, max_rows, specs, n_actions_out):
        self.max_rows = int(max_rows)
        self.max_cells = self.max_rows * MAX_EVENTS_PER_HAND
        widths = [_widths(f) for f in specs]
        self.ints = torch.zeros(self.max_cells * max(w[INT] for w in widths),
                                dtype=torch.int64)
        self.floats = torch.zeros(self.max_cells * max(w[FLOAT] for w in widths),
                                  dtype=torch.float32)
        self.bools = torch.zeros(self.max_cells * max(w[BOOL] for w in widths),
                                 dtype=torch.uint8)
        self.meta = torch.zeros(self.max_rows, dtype=torch.int64)
        self.out = torch.zeros((self.max_rows, int(n_actions_out)),
                               dtype=torch.float32)
        for t in (self.ints, self.floats, self.bools, self.meta, self.out):
            t.share_memory_()

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.ints, self.floats, self.bools, self.meta, self.out))

    def _blocks(self):
        return {INT: self.ints, FLOAT: self.floats, BOOL: self.bools}

    def pack(self, fields, tensors, cells):
        """Copy every field of one request into the slab.

        `cells` is the number of rows *of the flattened payload* — events for a
        v7 query, tokens for an agent one.
        """
        assert cells <= self.max_cells, (
            f"a request of {cells} rows does not fit a slab of "
            f"{self.max_cells}; raise `oracle.batch_hands` awareness in "
            f"`slab_rows` or lower the batch")
        cursor = {INT: 0, FLOAT: 0, BOOL: 0}
        blocks = self._blocks()
        for name, cols, dtype in fields:
            src = tensors[name]
            off = cursor[dtype]
            width = cells * cols
            block = blocks[dtype]
            view = block[off:off + width]
            if dtype == BOOL:
                view.copy_(src.reshape(-1).to(torch.uint8))
            else:
                view.copy_(src.reshape(-1))
            cursor[dtype] = off + width

    def unpack(self, fields, cells, shape):
        """Views over the slab, one per field, shaped as the model expects."""
        cursor = {INT: 0, FLOAT: 0, BOOL: 0}
        blocks = self._blocks()
        out = {}
        for name, cols, dtype in fields:
            off = cursor[dtype]
            width = cells * cols
            view = blocks[dtype][off:off + width]
            if dtype == BOOL:
                view = view.view(torch.bool)
            out[name] = view.view(*shape, cols) if cols > 1 else view.view(*shape)
            cursor[dtype] = off + width
        return out


def slab_rows(batch_hands, max_combos):
    """The widest policy query the oracle can make.

    The driver asks about at most one situation per hand in flight, and the
    posterior about one per combo it kept — `max_combos = None` is the whole
    universe of two-card combos on a five-card board.
    """
    return int(max(int(batch_hands), int(max_combos or 1081)))
