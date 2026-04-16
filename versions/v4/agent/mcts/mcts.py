"""
Monte Carlo Tree Search using the agent's neural network.

Each node stores a game state (betting logic) and a latent embedding.
The search builds context by concatenating root perception output with
action embeddings collected along the path from root to leaf.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F


class MCTSNode:
    """Single node in the MCTS tree."""
    __slots__ = [
        "action_idx", "parent", "children", "is_hero", "is_terminal",
        "N", "W", "Q", "P", "action_embedding",
    ]

    def __init__(self, action_idx=None, parent=None, is_hero=True,
                 is_terminal=False, P=0.0, action_embedding=None):
        self.action_idx = action_idx       # action that led here (None for root)
        self.parent = parent
        self.children = {}                 # action_idx -> MCTSNode
        self.is_hero = is_hero             # whose turn at this node
        self.is_terminal = is_terminal
        self.N = 0                         # visit count
        self.W = 0.0                       # total backed-up value (hero perspective)
        self.Q = 0.0                       # mean value = W / N
        self.P = P                         # prior probability
        self.action_embedding = action_embedding  # (d_model,) tensor


class MCTS:
    """Monte Carlo Tree Search for poker decision-making.

    Uses the agent's perception, value, action, opponent_action, and modelling
    heads to evaluate positions and explore the game tree in latent space.
    """

    def __init__(self, agent, device, mcts_config=None):
        self.agent = agent
        self.device = device
        cfg = mcts_config or {}
        self.n_simulations = cfg.get("n_simulations", 1000)
        self.c_puct = cfg.get("c_puct", 1.5)
        self.n_actions = agent.n_actions
        self.dirichlet_alpha = cfg.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = cfg.get("dirichlet_epsilon", 0.25)
        self.temperature = cfg.get("temperature", 1.0)

    @torch.no_grad()
    def search(self, event_sequences, game_state):
        """Run MCTS from current position. Returns best action index.

        Args:
            event_sequences: list of lists of event dicts (normalized, batch size 1)
            game_state: GameState at the decision point

        Returns:
            int — best action index
        """
        legal = game_state.get_legal_actions()
        if len(legal) == 1:
            return legal[0]

        # Evaluate root: full perception forward
        root_ctx, root_mask, value, act_logits, opp_logits, act_embs = \
            self._evaluate_root(event_sequences)

        # Create and expand root
        root = MCTSNode(is_hero=game_state.is_hero_turn())
        self._expand_node(root, game_state, act_logits, opp_logits, act_embs)
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.n_simulations):
            self._simulate(root, root_ctx, root_mask, game_state)

        return self._best_action(root)

    def _evaluate_root(self, event_sequences):
        """Run perception + all heads on the real event sequences."""
        p_out, encoded, mask = self.agent.perception.forward_batch(
            event_sequences, device=self.device, skip_memory=True,
        )
        value = self.agent.value_head(p_out, mask=mask)
        act_logits = self.agent.action_head(p_out, mask=mask)
        opp_logits = self.agent.opponent_action_head(p_out, mask=mask)
        act_embs = self.agent.modelling_head(p_out, mask=mask)
        return p_out, mask, value, act_logits, opp_logits, act_embs

    def _simulate(self, root, root_ctx, root_mask, root_gs):
        """One MCTS simulation: select → expand → backup."""
        # SELECT: descend from root to leaf
        node = root
        path = [node]

        while node.children and not node.is_terminal:
            action = self._select_child(node)
            node = node.children[action]
            path.append(node)

        # TERMINAL: increment N only
        if node.is_terminal:
            for n in path:
                n.N += 1
            return

        # BUILD CONTEXT: root_ctx + action embeddings along path
        context, mask = self._build_context(root_ctx, root_mask, path)

        # GET GAME STATE at this leaf
        gs = self._replay_game_state(root_gs, path)

        # EXPAND leaf
        if node is root:
            # Root already expanded, but still back up its value
            leaf_value = self.agent.value_head(context, mask=mask).item()
        else:
            leaf_value, act_logits, opp_logits, act_embs = \
                self._evaluate_node(context, mask)
            self._expand_node(node, gs, act_logits, opp_logits, act_embs)

        # BACKUP
        for n in path:
            n.N += 1
            n.W += leaf_value
            n.Q = n.W / n.N

    def _select_child(self, node):
        """Select child action. PUCT for hero nodes, prior sampling for opponents."""
        if node.is_hero:
            best_action = None
            best_ucb = -float("inf")
            sqrt_parent = math.sqrt(node.N)
            for action, child in node.children.items():
                ucb = child.Q + self.c_puct * child.P * sqrt_parent / (1 + child.N)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = action
            return best_action
        else:
            # Opponent: sample proportional to prior (trained opponent model)
            actions = list(node.children.keys())
            priors = [node.children[a].P for a in actions]
            return random.choices(actions, weights=priors, k=1)[0]

    def _build_context(self, root_ctx, root_mask, path):
        """Build context tensor by appending action embeddings along path."""
        if len(path) <= 1:
            return root_ctx, root_mask

        # Collect embeddings from path (skip root which has no embedding)
        embeddings = []
        for node in path[1:]:
            if node.action_embedding is not None:
                embeddings.append(node.action_embedding)

        if not embeddings:
            return root_ctx, root_mask

        emb_stack = torch.stack(embeddings).unsqueeze(0)  # (1, depth, d_model)
        context = torch.cat([root_ctx, emb_stack], dim=1)
        extra_mask = torch.ones(1, len(embeddings), dtype=root_mask.dtype,
                                device=self.device)
        mask = torch.cat([root_mask, extra_mask], dim=1)
        return context, mask

    def _replay_game_state(self, root_gs, path):
        """Replay actions along path to get GameState at leaf."""
        gs = root_gs.clone()
        for node in path[1:]:
            if node.action_idx is not None:
                gs.step(node.action_idx)
        return gs

    def _evaluate_node(self, context, mask):
        """Run all heads on the given context."""
        value = self.agent.value_head(context, mask=mask).item()
        act_logits = self.agent.action_head(context, mask=mask)
        opp_logits = self.agent.opponent_action_head(context, mask=mask)
        act_embs = self.agent.modelling_head(context, mask=mask)
        return value, act_logits, opp_logits, act_embs

    def _expand_node(self, node, game_state, act_logits, opp_logits, act_embs):
        """Create children for each legal action."""
        legal = game_state.get_legal_actions()
        logits = act_logits if node.is_hero else opp_logits
        logits = logits[0]  # (n_actions,)

        # Mask illegal actions and compute priors
        mask = torch.full_like(logits, float("-inf"))
        for a in legal:
            mask[a] = 0.0
        priors = F.softmax(logits + mask, dim=0)

        for a in legal:
            child_gs = game_state.clone()
            child_gs.step(a)

            child = MCTSNode(
                action_idx=a,
                parent=node,
                is_hero=child_gs.is_hero_turn() if not child_gs.is_terminal else True,
                is_terminal=child_gs.is_terminal,
                P=priors[a].item(),
                action_embedding=act_embs[0, a].detach(),
            )
            node.children[a] = child

    def _add_dirichlet_noise(self, root):
        """Mix root children priors with Dirichlet noise for exploration."""
        if not root.children or self.dirichlet_epsilon <= 0:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for a, n in zip(actions, noise):
            root.children[a].P = (1 - eps) * root.children[a].P + eps * n

    def _best_action(self, root):
        """Select action from root using temperature-scaled visit counts."""
        if self.temperature <= 0:
            return max(root.children, key=lambda a: root.children[a].N)
        actions = list(root.children.keys())
        counts = np.array([root.children[a].N for a in actions], dtype=np.float64)
        counts = counts ** (1.0 / self.temperature)
        total = counts.sum()
        if total == 0:
            return random.choice(actions)
        probs = counts / total
        return actions[np.random.choice(len(actions), p=probs)]


def _collect_terminals(node):
    """DFS to collect all terminal nodes in the tree."""
    if node.is_terminal:
        return [node]
    terminals = []
    for child in node.children.values():
        terminals.extend(_collect_terminals(child))
    return terminals


def re_backup_terminals(root):
    """After terminal Q values are set externally, propagate to all ancestors.

    During MCTS search, terminal paths only incremented N without adding to W.
    This function adds each terminal's Q * N contribution to all ancestors' W,
    then recalculates Q = W / N for every affected node.
    """
    for terminal in _collect_terminals(root):
        if terminal.N == 0:
            continue
        terminal.W = terminal.Q * terminal.N
        contribution = terminal.W
        node = terminal.parent
        while node is not None:
            node.W += contribution
            node.Q = node.W / node.N if node.N > 0 else 0.0
            node = node.parent


def get_n_distribution(root, n_actions):
    """Extract visit-count distribution from root's children.

    Returns list of length n_actions with normalized visit counts.
    """
    total = sum(child.N for child in root.children.values())
    if total == 0:
        return [1.0 / n_actions] * n_actions
    dist = [0.0] * n_actions
    for action_idx, child in root.children.items():
        dist[action_idx] = child.N / total
    return dist
