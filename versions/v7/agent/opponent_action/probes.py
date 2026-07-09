"""Training-only probes on the opponent GRU state (PLAN_OPPONENT_ADAPTATION).

Both probes consume the acting opponent's GRU hidden state at a decision
point (`opp_last_states` from Perception.forward_batch) and exist ONLY to
give the opponent embedding a direct supervised objective in phase 5:

- StyleProbe (§2): regress the canonical 16-dim style vector of the agent
  that generated the opponent's actions (modifiers.build_style_vector).
- ShowdownStrengthProbe (§4): regress the strength percentile of the
  opponent's revealed hand on showdown-terminated hands.

Neither probe is used at deployment (MCTS, evaluation); the learned
knowledge lives in the GRU weights. They are registered on ASI so they
persist in checkpoints, but no other phase's optimizer includes them.
"""

import torch.nn as nn

from agent.train_scenarios.modifiers import STYLE_DIMS


class StyleProbe(nn.Module):
    """d_model → STYLE_DIMS style-vector regression head."""

    def __init__(self, d_model, n_style_dims=STYLE_DIMS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_style_dims),
        )

    def forward(self, x):
        return self.net(x)


class ShowdownStrengthProbe(nn.Module):
    """d_model → scalar strength-percentile regression head (target ∈ [0,1])."""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
