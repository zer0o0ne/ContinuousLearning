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
        "terminal_Q",
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
        # In-search terminal value, cached on first detection so subsequent
        # visits back up the same real game-theoretic Q (not 0). For fold
        # terminals it is deterministic; for showdowns we use a fair-share-
        # of-pot heuristic (1/n_active). None means "not yet evaluated".
        self.terminal_Q = None


class MCTS:
    """Monte Carlo Tree Search for poker decision-making.

    Uses the agent's perception, value, action, opponent_action, and modelling
    heads to evaluate positions and explore the game tree in latent space.
    """

    def __init__(self, agent, device, mcts_config=None, opponent_emb_table=None,
                 terminal_evaluator=None):
        self.agent = agent
        self.device = device
        cfg = mcts_config or {}
        self.n_simulations = cfg.get("n_simulations", 1000)
        self.c_puct = cfg.get("c_puct", 1.5)
        self.n_actions = agent.n_actions
        self.dirichlet_alpha = cfg.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = cfg.get("dirichlet_epsilon", 0.25)
        self.temperature = cfg.get("temperature", 1.0)
        # Batched search: collect up to `batch_size` leaves, mark in-flight
        # paths with `virtual_loss` so concurrent simulators avoid them.
        self.batch_size = cfg.get("batch_size", 1)
        self.virtual_loss = cfg.get("virtual_loss", 1.0)
        # Opponent embedding is only injected at the root (inner-tree nodes
        # work in latent space without re-running perception).
        self.opponent_emb_table = opponent_emb_table
        # Callable(GameState) -> float, used to compute Q at terminal leaves
        # during search. Without it, terminals back up Q=0, which biases
        # visit counts (action_target) toward the priors and the value head
        # bootstrap, eliminating the policy-improvement signal. When None,
        # terminal nodes default to Q=0 (legacy behaviour).
        self.terminal_evaluator = terminal_evaluator

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

        # Pending queue: in-flight (path, gs) waiting for batched NN evaluation.
        # Their nodes already carry virtual loss so subsequent _select_to_leaf
        # calls steer away from them.
        pending = []
        sim = 0
        while sim < self.n_simulations:
            while len(pending) < self.batch_size and sim < self.n_simulations:
                path, gs = self._select_to_leaf(root, game_state)
                sim += 1
                if gs is None or path[-1].is_terminal:
                    self._backup_terminal(path)
                    continue
                self._apply_virtual_loss(path)
                pending.append((path, gs))

            if pending:
                self._flush_pending(pending, root_ctx, root_mask)
                pending.clear()

        self.last_root = root
        return self._best_action(root)

    def _backup_terminal(self, path):
        """Back up the cached terminal Q at path[-1] through every ancestor.

        Mirrors a normal backup but uses the precomputed game-theoretic Q
        instead of a value-head estimate. If terminal_Q is None (no
        evaluator configured), falls back to Q=0 — the legacy behaviour
        where terminal paths only bumped N.
        """
        leaf_q = path[-1].terminal_Q
        if leaf_q is None:
            for n in path:
                n.N += 1
            return
        for n in path:
            n.N += 1
            n.W += leaf_q
            n.Q = n.W / n.N

    def _apply_virtual_loss(self, path):
        """Tentatively mark a path as visited with a negative value bias.

        Increments N and decreases W by `virtual_loss` along the path so that
        subsequent selections see this path as less attractive. Reversed in
        _flush_pending after the real leaf value arrives.
        """
        vl = self.virtual_loss
        for n in path:
            n.N += 1
            n.W -= vl
            n.Q = n.W / n.N

    def _pad_and_stack(self, contexts, masks):
        """Pad list of (1, L_i, d) contexts and (1, L_i) masks to common L_max
        and stack into (B, L_max, d) / (B, L_max).

        At B=1 returns the inputs unchanged so head forwards are bit-for-bit
        identical to the sequential implementation.
        """
        if len(contexts) == 1:
            return contexts[0], masks[0]
        L_max = max(c.shape[1] for c in contexts)
        B = len(contexts)
        d = contexts[0].shape[2]
        device = contexts[0].device

        batch_ctx = torch.zeros(B, L_max, d, device=device, dtype=contexts[0].dtype)
        batch_mask = torch.zeros(B, L_max, device=device, dtype=masks[0].dtype)
        for i, (c, m) in enumerate(zip(contexts, masks)):
            L_i = c.shape[1]
            batch_ctx[i, :L_i] = c[0]
            batch_mask[i, :L_i] = m[0]
        return batch_ctx, batch_mask

    def _flush_pending(self, pending, root_ctx, root_mask):
        """Evaluate all queued leaves in ONE batched NN forward, then expand
        and resolve virtual loss for each.

        At batch_size=1 this is bit-for-bit equivalent to the previous
        sequential code: _pad_and_stack returns the single context unchanged
        and head outputs are identical.
        """
        vl = self.virtual_loss

        contexts, masks = [], []
        for path, gs in pending:
            ctx, msk = self._build_context(root_ctx, root_mask, path)
            contexts.append(ctx)
            masks.append(msk)

        batch_ctx, batch_mask = self._pad_and_stack(contexts, masks)

        # ONE forward per head over the whole batch
        values     = self.agent.value_head(batch_ctx, mask=batch_mask)            # (B, 1)
        act_logits = self.agent.action_head(batch_ctx, mask=batch_mask)           # (B, n_actions)
        opp_logits = self.agent.opponent_action_head(batch_ctx, mask=batch_mask)  # (B, n_actions)
        act_embs   = self.agent.modelling_head(batch_ctx, mask=batch_mask)        # (B, n_actions, d)

        for i, (path, gs) in enumerate(pending):
            leaf = path[-1]
            leaf_value = values[i].item()
            self._expand_node(leaf, gs,
                              act_logits[i:i+1], opp_logits[i:i+1], act_embs[i:i+1])
            for n in path:
                n.W += vl + leaf_value
                n.Q = n.W / n.N

    def _evaluate_root(self, event_sequences):
        """Run perception + all heads on the real event sequences."""
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

    def _select_to_leaf(self, root, root_gs):
        """Descend from root to a leaf, replay GameState, set lazy flags.

        On first visit to a node, sets node.is_terminal / node.is_hero from
        the replayed GameState (deferred from _expand_node).

        Returns:
            path: list[MCTSNode] from root to leaf (length >= 1)
            gs:   GameState at the leaf, or None if the leaf is an
                  already-known terminal (no replay needed)
        """
        node = root
        path = [node]
        while node.children and not node.is_terminal:
            action = self._select_child(node)
            node = node.children[action]
            path.append(node)

        if node.is_terminal:
            return path, None

        gs = self._replay_game_state(root_gs, path)
        if node is not root:
            node.is_terminal = gs.is_terminal
            node.is_hero = gs.is_hero_turn()
            if node.is_terminal and self.terminal_evaluator is not None \
                    and node.terminal_Q is None:
                node.terminal_Q = self.terminal_evaluator(gs)
        return path, gs

    def _simulate(self, root, root_ctx, root_mask, root_gs):
        """One MCTS simulation: select → expand → backup."""
        path, gs = self._select_to_leaf(root, root_gs)
        node = path[-1]

        # Terminal — known from prior visit (gs is None) or freshly detected
        if gs is None or node.is_terminal:
            self._backup_terminal(path)
            return

        # BUILD CONTEXT: root_ctx + action embeddings along path
        context, mask = self._build_context(root_ctx, root_mask, path)

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
        """Create children for each legal action (lazy — no GameState clones).

        is_hero / is_terminal are set on first visit in _simulate via
        _replay_game_state, so we only allocate the prior + action embedding
        here. Saves up to n_legal GameState clones per expansion.
        """
        legal = game_state.get_legal_actions()
        logits = act_logits if node.is_hero else opp_logits
        logits = logits[0]  # (n_actions,)

        # Mask illegal actions and compute priors
        mask = torch.full_like(logits, float("-inf"))
        for a in legal:
            mask[a] = 0.0
        priors = F.softmax(logits + mask, dim=0).tolist()  # one host transfer

        for a in legal:
            child = MCTSNode(
                action_idx=a,
                parent=node,
                is_hero=False,        # filled lazily on first visit
                is_terminal=False,    # filled lazily on first visit
                P=priors[a],
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
    """Re-propagate terminal Q values to ancestors after an external override.

    Two situations need to be handled correctly without double-counting:
      (a) Terminals were Q=0 during search (no `terminal_evaluator`):
          `terminal.W == 0` going in. Delta = new_W - 0 = new_W. Behaviour
          equivalent to the previous "add Q*N" implementation.
      (b) Terminals had a search-time Q from `terminal_evaluator` (e.g. the
          fold/fair-share heuristic in `_make_terminal_evaluator`): each visit
          during search already added that heuristic Q to every ancestor's W.
          We now overwrite `terminal.Q` with the equity-based value; the
          ancestors must receive only the *difference* `(new_Q − old_Q) * N`,
          else the heuristic contribution would be double-counted in W.

    Operates by reading the existing `terminal.W` as the cumulative old-Q
    contribution and writing the difference to ancestors. Final `terminal.W`
    is consistent with `terminal.Q * terminal.N` regardless of starting state.
    """
    for terminal in _collect_terminals(root):
        if terminal.N == 0:
            continue
        old_W = terminal.W
        new_W = terminal.Q * terminal.N
        delta = new_W - old_W
        terminal.W = new_W
        if delta == 0.0:
            continue
        node = terminal.parent
        while node is not None:
            node.W += delta
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
