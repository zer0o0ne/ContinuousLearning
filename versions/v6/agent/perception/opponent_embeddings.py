import torch
import torch.nn as nn


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

    def __init__(self, d_model):
        self.d_model = d_model
        self.embeddings = {}  # str -> Tensor(d_model)

    def get(self, opponent_id, device):
        """Return the stored embedding for opponent_id (detached zero if new).

        Never an optimizer parameter — see the class docstring for the detach
        semantics.
        """
        if opponent_id not in self.embeddings:
            self.embeddings[opponent_id] = torch.zeros(
                self.d_model, device=device,
            )
        emb = self.embeddings[opponent_id]
        # A.4.5: compare device TYPE, not the full device. `torch.device("cuda")`
        # has index None while a tensor lives on "cuda:0", so a plain `!=` was
        # always true and re-moved/re-detached the embedding on every access.
        if emb.device.type != torch.device(device).type:
            self.embeddings[opponent_id] = emb.to(device).detach()
        return self.embeddings[opponent_id]

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
        new = OpponentEmbeddingTable(self.d_model)
        new.embeddings = {k: v.detach().clone() for k, v in self.embeddings.items()}
        return new

    def __len__(self):
        return len(self.embeddings)

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.embeddings.items()}

    def load_state_dict(self, state, device="cpu"):
        self.embeddings = {
            k: v.to(device) for k, v in state.items()
        }


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
