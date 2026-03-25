import torch
import torch.nn as nn
import numpy as np

from perception.perception import Perception
from value.value import ValueHead
from action.action import ActionHead


class AgentMemory:
    """Simple per-player story buffer for dealers.py compatibility."""

    def __init__(self, max_players):
        self.stories = [[] for _ in range(max_players)]

    def reset(self, max_players=None):
        if max_players is not None:
            self.stories = [[] for _ in range(max_players)]
        else:
            for s in self.stories:
                s.clear()


class ASI(nn.Module):
    def __init__(self, log, config=None):
        super().__init__()
        self.log = log
        self.config = config or {}

        d_model = config.get("d_model", 128)
        n_actions = config.get("n_actions", 13)
        max_players = config.get("max_players", 6)
        n_heads = config.get("n_heads", 4)
        n_kv_heads = config.get("n_kv_heads", n_heads // 2)
        mem_cfg = config.get("memory", {})
        act_cfg = config.get("action", {})

        self.perception = Perception(config)
        self.value_head = ValueHead(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config.get("n_value_layers", 2),
            d_ff=config.get("d_ff", 512),
            max_seq_len=mem_cfg.get("n_results", 8) + 1 + 64,
        )
        self.action_head = ActionHead(
            d_model=d_model,
            n_actions=n_actions,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=config.get("n_action_layers", 2),
            d_ff=config.get("d_ff", 512),
            max_seq_len=act_cfg.get("max_gen_steps", 4) + 1,
            max_gen_steps=act_cfg.get("max_gen_steps", 4),
        )

        self.memory = AgentMemory(max_players)
        self.players = []
        self.human_pos = None
        self.device_ = "cpu"
        self.n_players = 0
        self.n_actions = n_actions
        self.optimizer = None
        self.loss_buffer = []

    def forward(self, env_state):
        """
        Full forward pass through all components.

        Args: env_state dict from dealer
        Returns: {"action_logits": (1, n_actions), "value": (1, 1)}
        """
        perception_out = self.perception(env_state, device=self.device_)
        value = self.value_head(perception_out)
        action_logits = self.action_head(perception_out)
        return {"action_logits": action_logits, "value": value}

    def sit(self, n_players, with_human=False):
        self.n_players = n_players
        self.players = list(range(n_players))
        if with_human:
            self.human_pos = np.random.randint(n_players)

    def set_device(self, device):
        self.device_ = device
        self.to(device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.get("lr", 1e-4)
            )

    def step(self, active_player, env_state):
        """
        Inference step for dealer.

        Returns: {"action": Tensor(n_actions,)}
        """
        with torch.no_grad():
            result = self.forward(env_state)
            action_logits = result["action_logits"].squeeze(0)
            action_probs = torch.softmax(action_logits, dim=-1)

            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            action = torch.zeros(self.n_actions, device=self.device_)
            action[action_idx] = 1.0

            self.memory.stories[active_player].append(env_state)

        return {"action": action}

    def save_loss(self, actions, reward):
        """
        Buffer losses from a completed game. Loss formula is a placeholder.

        Args:
            actions: list[list[dict]] — per-player action history
            reward: dict with "rewards" array
        Returns: list[float] — loss per player
        """
        losses = []
        for player_idx in range(self.n_players):
            player_reward = float(reward["rewards"][player_idx])
            loss = torch.tensor(-player_reward, dtype=torch.float, device=self.device_)
            losses.append(loss.item())
            self.loss_buffer.append(loss)
        return losses

    def optimize(self):
        if not self.loss_buffer or self.optimizer is None:
            return
        total_loss = sum(self.loss_buffer) / len(self.loss_buffer)
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()
        self.loss_buffer = []

    def rotate(self):
        if len(self.players) > 1:
            self.players = [self.players[-1]] + self.players[:-1]

    def init_history__(self, env_state, action, n_players):
        for i in range(n_players):
            if i < len(self.memory.stories) and len(self.memory.stories[i]) == 0:
                self.memory.stories[i].append(env_state)
