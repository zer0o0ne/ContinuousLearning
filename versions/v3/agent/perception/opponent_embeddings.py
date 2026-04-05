import torch
import torch.nn as nn


class OpponentEmbeddingTable:
    """Dynamic table of per-opponent embedding vectors.

    Not an nn.Module — size varies between sessions. The GRU updater
    (which has fixed parameters) lives in Perception as a registered submodule.

    Embeddings are plain tensors with requires_grad=True so they participate
    in the computation graph for BPTT across hands in a session.
    """

    def __init__(self, d_model):
        self.d_model = d_model
        self.embeddings = {}  # str -> Tensor(d_model)

    def get(self, opponent_id, device):
        """Return embedding for opponent_id, creating zero-init if new.

        Embeddings do NOT require grad — they are updated only by GRU forward
        output replacement, never by optimizer.step().
        """
        if opponent_id not in self.embeddings:
            self.embeddings[opponent_id] = torch.zeros(
                self.d_model, device=device,
            )
        emb = self.embeddings[opponent_id]
        if emb.device != torch.device(device):
            self.embeddings[opponent_id] = emb.to(device).detach()
        return self.embeddings[opponent_id]

    def detach_all(self):
        """Detach all embeddings from computation graph (truncated BPTT)."""
        for key in self.embeddings:
            self.embeddings[key] = self.embeddings[key].detach()

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
