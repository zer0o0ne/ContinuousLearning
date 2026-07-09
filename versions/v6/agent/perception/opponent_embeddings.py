import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Count-based opponent stats (HUD vector) — PLAN_OPPONENT_ADAPTATION §3.
#
# Raw storage per opponent: counts (8, 5) float64 — bucket × action category.
# Buckets: street (preflop/flop/turn/river) × facing (no bet to call / facing
# a bet). Categories: fold, call, small raise, big raise, all-in (same split
# as modifiers.resolve_actions).
# ---------------------------------------------------------------------------

N_STAT_STREETS = 4
N_STAT_FACING = 2
N_STAT_BUCKETS = N_STAT_STREETS * N_STAT_FACING          # 8
N_STAT_CATEGORIES = 5
# 40 smoothed in-bucket frequencies + 8 per-bucket confidences + 1 global
N_STAT_FEATURES = N_STAT_BUCKETS * N_STAT_CATEGORIES + N_STAT_BUCKETS + 1  # 49

# Reserved key for stats inside state_dict() — cannot collide with real
# opponent ids in practice (documented, not enforced).
_STATS_STATE_KEY = "__stats__"


def zero_stat_counts():
    """Fresh raw-count array for one opponent."""
    return np.zeros((N_STAT_BUCKETS, N_STAT_CATEGORIES), dtype=np.float64)


def stats_features(counts):
    """Derive the 49-dim bounded feature vector from raw counts.

    Args:
        counts: (8, 5) float64 raw counts (bucket × category)

    Returns:
        np.ndarray (49,) float32:
          [0:40]  Laplace-smoothed in-bucket frequencies
                  (count + 1) / (bucket_total + 5), bucket-major
          [40:48] per-bucket confidence log1p(bucket_total) / 5.0
          [48]    global confidence log1p(total) / 5.0
    """
    counts = np.asarray(counts, dtype=np.float64)
    bucket_totals = counts.sum(axis=1)                       # (8,)
    freqs = (counts + 1.0) / (bucket_totals + N_STAT_CATEGORIES)[:, None]
    feats = np.empty(N_STAT_FEATURES, dtype=np.float32)
    feats[:N_STAT_BUCKETS * N_STAT_CATEGORIES] = freqs.reshape(-1)
    feats[N_STAT_BUCKETS * N_STAT_CATEGORIES:
          N_STAT_BUCKETS * N_STAT_CATEGORIES + N_STAT_BUCKETS] = (
        np.log1p(bucket_totals) / 5.0)
    feats[-1] = np.log1p(bucket_totals.sum()) / 5.0
    return feats


def action_category(action_idx, n_actions):
    """Map an action index to its stat category (0..4).

    Layout [fold, call, raise_0 .. raise_(bins-1), all-in]; small/big raise
    split at bins // 2 — same convention as modifiers.resolve_actions.
    """
    if action_idx == 0:
        return 0                                  # fold
    if action_idx == 1:
        return 1                                  # call/check
    if action_idx == n_actions - 1:
        return 4                                  # all-in
    bins = n_actions - 3
    mid = bins // 2
    return 2 if (action_idx - 2) < mid else 3     # small / big raise


class OpponentEmbeddingTable:
    """Dynamic table of per-opponent embedding vectors.

    Not an nn.Module — size varies between sessions. The GRU updater
    (which has fixed parameters) lives in Perception as a registered submodule.

    Each entry holds the latest GRU hidden state for an opponent. These are
    plain tensors, NEVER nn.Parameters: optimizer.step() does not touch them;
    they are advanced only by replacement with fresh GRU output (A.4 detach
    semantics). Within a forward the stored tensor stays in the autograd graph
    so gradients reach the GRU (truncated BPTT); `detach_all()` is called
    between training steps to cut the graph — the embedding VALUE carries
    forward across steps, the gradient history does not.
    """

    DEFAULT_MAX_SIZE = 10000

    def __init__(self, d_model, max_size=None):
        self.d_model = d_model
        self.max_size = max_size if max_size is not None else self.DEFAULT_MAX_SIZE
        self.embeddings = {}  # str -> Tensor(d_model)
        # §3: raw HUD counts per opponent, (8, 5) float64. Plain numpy — never
        # in the autograd graph; snapshots stay valid because updates REPLACE
        # the array (copy-on-write in Perception.forward_batch), not mutate it.
        self.stats = {}       # str -> np.ndarray(8, 5)
        self._access_order = []  # LRU tracking

    def get(self, opponent_id, device):
        """Return the stored embedding for opponent_id (detached zero if new).

        Never an optimizer parameter — see the class docstring for the detach
        semantics.
        """
        if opponent_id not in self.embeddings:
            if self.max_size and len(self.embeddings) >= self.max_size:
                self._evict_oldest()
            self.embeddings[opponent_id] = torch.zeros(
                self.d_model, device=device,
            )
        if opponent_id not in self.stats:
            self.stats[opponent_id] = zero_stat_counts()
        if opponent_id in self._access_order:
            self._access_order.remove(opponent_id)
        self._access_order.append(opponent_id)
        emb = self.embeddings[opponent_id]
        # A.4.5: compare device TYPE, not the full device. `torch.device("cuda")`
        # has index None while a tensor lives on "cuda:0", so a plain `!=` was
        # always true and re-moved/re-detached the embedding on every access.
        if emb.device.type != torch.device(device).type:
            self.embeddings[opponent_id] = emb.to(device).detach()
        return self.embeddings[opponent_id]

    def get_stats(self, opponent_id):
        """Raw HUD counts for opponent_id (fresh zeros if unseen)."""
        counts = self.stats.get(opponent_id)
        if counts is None:
            counts = zero_stat_counts()
            self.stats[opponent_id] = counts
        return counts

    def _evict_oldest(self):
        """Remove the least recently used entry."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            self.embeddings.pop(oldest, None)
            self.stats.pop(oldest, None)

    def detach_all(self):
        """Detach all embeddings from computation graph (truncated BPTT)."""
        for key in self.embeddings:
            self.embeddings[key] = self.embeddings[key].detach()

    def clone(self):
        """Deep copy with detached, cloned embeddings.

        A.4.3: validation runs forward passes that mutate the table; pass a
        clone so the live training table is never advanced by the val set.
        Detaching makes this safe even if entries are still in the graph.
        """
        new = OpponentEmbeddingTable(self.d_model, max_size=self.max_size)
        new.embeddings = {k: v.detach().clone() for k, v in self.embeddings.items()}
        new.stats = {k: v.copy() for k, v in self.stats.items()}
        new._access_order = list(self._access_order)
        return new

    def __len__(self):
        return len(self.embeddings)

    def state_dict(self):
        sd = {k: v.detach().cpu() for k, v in self.embeddings.items()}
        if self.stats:
            sd[_STATS_STATE_KEY] = {k: v.copy() for k, v in self.stats.items()}
        return sd

    def load_state_dict(self, state, device="cpu"):
        state = dict(state)
        # D4 backward-compat: legacy states have no stats sub-dict.
        stats = state.pop(_STATS_STATE_KEY, None)
        self.embeddings = {
            k: v.to(device) for k, v in state.items()
        }
        self.stats = ({k: np.asarray(v, dtype=np.float64).copy()
                       for k, v in stats.items()} if stats else {})


class OpponentGRUUpdater(nn.Module):
    """Single-step GRU for updating opponent embeddings.

    Takes a signal vector (summarizing opponent behavior in a hand)
    and the current opponent embedding, returns the updated embedding.
    Stays in the computation graph for end-to-end training.
    """

    def __init__(self, d_model):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_model, d_model)

    def forward(self, signal, hidden):
        """One GRU step.

        Args:
            signal: (batch, d_model) or (d_model,) — input from encoder output
            hidden: (batch, d_model) or (d_model,) — current opponent embedding

        Returns:
            new_hidden: same shape as hidden — updated embedding
        """
        squeeze = signal.dim() == 1
        if squeeze:
            signal = signal.unsqueeze(0)
            hidden = hidden.unsqueeze(0)
        out = self.gru_cell(signal, hidden)
        if squeeze:
            out = out.squeeze(0)
        return out
