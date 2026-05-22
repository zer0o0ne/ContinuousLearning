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
        "N", "W", "Q", "P", "action_embedding", "_term_value",
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
        # Cached value_head output for terminal nodes. Set on the FIRST visit
        # (in `_flush_pending`); subsequent visits skip NN and back up this
        # cached scalar sum-style. None for non-terminal nodes and for
        # terminals that haven't been evaluated yet.
        self._term_value = None


class MCTS:
    """Monte Carlo Tree Search for poker decision-making.

    Uses the agent's perception, value, action, opponent_action, and modelling
    heads to evaluate positions and explore the game tree in latent space.
    """

    def __init__(self, agent, device, mcts_config=None, opponent_emb_table=None,
                 strange_p=0.0):
        self.agent = agent
        self.device = device
        cfg = mcts_config or {}
        self.n_simulations = cfg.get("n_simulations", 1000)
        self.c_puct = cfg.get("c_puct", 1.5)
        self.n_actions = agent.n_actions
        self.dirichlet_alpha = cfg.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = cfg.get("dirichlet_epsilon", 0.25)
        # Probability per simulation of doing a "strange" traversal: at every
        # hero node along the descent we sample action with weight 1/(N+1)
        # (inverse visit count) instead of PUCT. Opp selection stays the same.
        # Combined with hero max-backup (W = max child.W), bad strange paths
        # do not poison hero ancestors' W. Set externally by `collect.py` as
        # `C(cycle) * exp(-last_action_loss)` so search broadens when the
        # action head has already converged. Default 0 disables it.
        self.strange_p = float(strange_p)
        # Small Dirichlet noise mixed into hero priors at **inner** nodes (i.e.
        # any hero expansion that is not the root). Root has its own, stronger
        # noise via `_add_dirichlet_noise`. Inner noise keeps deep hero
        # subtrees from collapsing onto a 1–2-action sharp prior when
        # `action_head` is overconfident.
        self.dirichlet_alpha_inner = cfg.get("dirichlet_alpha_inner", 0.3)
        self.dirichlet_epsilon_inner = cfg.get("dirichlet_epsilon_inner", 0.05)
        self.temperature = cfg.get("temperature", 1.0)
        # Batched search: collect up to `batch_size` leaves, mark in-flight
        # paths with `virtual_loss` so concurrent simulators avoid them.
        self.batch_size = cfg.get("batch_size", 1)
        self.virtual_loss = cfg.get("virtual_loss", 1.0)
        # Opponent embedding is only injected at the root (inner-tree nodes
        # work in latent space without re-running perception).
        self.opponent_emb_table = opponent_emb_table
        # ── Opp-node robustness (counters opp_action_head self-pool bias) ──
        # Prior smoothing: at expansion time, opp priors are mixed with
        # uniform-over-legal `(1 − λ)·P_model + λ/n_legal`. Stops a
        # confidently-wrong opp model from collapsing exploration of
        # off-distribution opponent replies (Slumbot ≠ pool of self-play
        # agents the opp head was trained on).
        self.opp_prior_smoothing = float(cfg.get("opp_prior_smoothing", 0.15))
        # Pessimistic Q at opp nodes: after each backup, an opp node's Q is
        # overridden by `α·E_P[Q] + (1−α)·min Q` over visited children.
        # α=1 keeps the legacy sample-mean (full trust in opp model);
        # α=0 is pure worst-case. With α<1, hero PUCT at the opp's parent
        # sees a lower Q whenever a strong opp reply exists → search
        # discounts branches whose value depends on opp making a low-prob
        # mistake. Only affects opp-node reads in PUCT and read-time
        # consumers; never modifies W (zero-sum semantics preserved).
        self.opp_pessimism_alpha = float(cfg.get("opp_pessimism_alpha", 0.5))

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

        # Pending queue: in-flight (path, gs, is_terminal_flag) waiting for
        # batched NN evaluation. Their nodes already carry virtual loss so
        # subsequent _select_to_leaf calls steer away from them.
        #   is_terminal_flag=True → first visit to a terminal: needs ONE
        #     value_head forward to cache V_pred, then sum-style backup.
        #     gs is unused (terminal has no children to expand).
        #   is_terminal_flag=False → regular non-terminal leaf: full
        #     expansion via _expand_node.
        pending = []
        sim = 0
        while sim < self.n_simulations:
            while len(pending) < self.batch_size and sim < self.n_simulations:
                strange = (self.strange_p > 0.0
                           and random.random() < self.strange_p)
                path, gs = self._select_to_leaf(root, game_state, strange=strange)
                sim += 1
                leaf = path[-1]
                if leaf.is_terminal:
                    if leaf._term_value is None:
                        # First terminal visit — queue for batched value_head.
                        self._apply_virtual_loss(path)
                        pending.append((path, None, True))
                    else:
                        # Repeat visit: sum-style backup with cached V_pred.
                        self._backup_cached_terminal(path, leaf._term_value)
                    continue
                self._apply_virtual_loss(path)
                pending.append((path, gs, False))

            if pending:
                self._flush_pending(pending, root_ctx, root_mask)
                pending.clear()

        self.last_root = root
        return self._best_action(root)

    def _backup_cached_terminal(self, path, V_pred):
        """Sum-style backup of a cached terminal V_pred (repeat visit).

        No NN evaluation needed (`V_pred` was cached on first visit in
        `_flush_pending`); no virtual loss applied (nothing is in flight).
        Treats the terminal as a deterministic leaf with value V_pred:
        each visit adds V_pred to W of every "sum-style" node along the
        path (opp ancestors + the terminal itself), so W and N scale
        together and hero max-backup compares apples to apples between
        terminal and non-terminal children. Hero non-terminal ancestors
        re-derive their W from `max(child.W)` in the bottom-up pass.
        """
        for n in path:
            n.N += 1
            if (not n.is_hero) or n.is_terminal:
                n.W += V_pred
        for n in reversed(path):
            self._recompute_node_after_backup(n, leaf_value=V_pred)

    def _apply_virtual_loss(self, path):
        """Tentatively mark a path as visited with a negative value bias.

        Under **mixed backup** semantics:
          - All nodes: `N += 1` (PUCT exploration term shrinks for this
            child in concurrent sims).
          - **Sum-style nodes** (opp ancestors + the terminal leaf, regardless
            of its `is_hero` flag): `W -= vl`; Q = W/N drops sample-mean.
          - **Hero non-terminal nodes**: W is NOT touched directly (it lives
            in `max(child.W)`, which hasn't changed yet — no child's W moved
            during this VL phase). Q = W/N still drops a bit thanks to N bump.
            Opp pessimism is re-applied inline so any hero still further up
            sees the right Q via children comparison in PUCT.
        """
        vl = self.virtual_loss
        for n in path:
            n.N += 1
            if (not n.is_hero) or n.is_terminal:
                n.W -= vl
        # Bottom-up Q (and hero W) recompute so PUCT reads are consistent.
        for n in reversed(path):
            self._recompute_node_after_backup(n)

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
        for path, _gs, _is_term in pending:
            ctx, msk = self._build_context(root_ctx, root_mask, path)
            contexts.append(ctx)
            masks.append(msk)

        batch_ctx, batch_mask = self._pad_and_stack(contexts, masks)

        # value_head is needed by all (both leaves and first-visit terminals).
        # action/opp/modelling are needed only for non-terminal leaves; we skip
        # them when the batch is entirely terminals, but if even one
        # non-terminal is present we just run all heads (cheap to batch
        # together, avoids extra branching).
        values     = self.agent.value_head(batch_ctx, mask=batch_mask)            # (B, 1)
        needs_expansion = any(not is_term for _, _, is_term in pending)
        if needs_expansion:
            act_logits = self.agent.action_head(batch_ctx, mask=batch_mask)           # (B, n_actions)
            opp_logits = self.agent.opponent_action_head(batch_ctx, mask=batch_mask)  # (B, n_actions)
            act_embs   = self.agent.modelling_head(batch_ctx, mask=batch_mask)        # (B, n_actions, d)

        for i, (path, gs, is_term) in enumerate(pending):
            leaf = path[-1]
            leaf_value = values[i].item()
            if is_term:
                # Cache V_pred on the terminal node so all future visits reuse
                # it without another NN call.
                leaf._term_value = float(leaf_value)
            else:
                self._expand_node(leaf, gs,
                                  act_logits[i:i+1], opp_logits[i:i+1], act_embs[i:i+1])
            # Bookkeeping pass: sum-style nodes (opp ancestors + the terminal
            # leaf if `is_term`) get `+vl + leaf_value` → net `+leaf_value`.
            # Hero non-terminal nodes are skipped here and have their W
            # rederived from `max(child.W)` in the bottom-up pass below.
            for n in path:
                if (not n.is_hero) or n.is_terminal:
                    n.W += vl + leaf_value
            # Bottom-up: hero W ← max(child.W) [or leaf_value at first visit],
            # opp Q ← W/N (then pessimism if any), terminal Q ← W/N.
            for n in reversed(path):
                self._recompute_node_after_backup(n, leaf_value=leaf_value)

    def _recompute_node_after_backup(self, n, leaf_value=None):
        """Recompute `n.W` (hero only) and `n.Q` under mixed backup rules.

        Called bottom-up after `N`/`W` bookkeeping along a path:
          - **terminal** (`is_terminal=True`, regardless of `is_hero`): pure
            sum-style. `W` was already updated additively in the caller; we
            just refresh Q = W/N. `is_hero` at a terminal is semantically
            void (no acting player at hand end) — max-backup doesn't apply.
          - **hero non-terminal**: `W = max(c.W for c.N > 0)`. If no visited
            children (only happens at a hero leaf that was just expanded),
            fall back to `leaf_value` (value-head estimate is the only signal
            available for this fresh node).
          - **opp non-terminal**: `W` was already updated additively in the
            caller; compute Q = W/N and apply opp pessimism if any.
        """
        if n.is_terminal:
            n.Q = n.W / n.N if n.N > 0 else 0.0
            return
        if n.is_hero:
            visited = [c for c in n.children.values() if c.N > 0]
            if visited:
                n.W = max(c.W for c in visited)
            elif leaf_value is not None:
                n.W = float(leaf_value)
            # else: keep W untouched (e.g. VL phase before eval — no children
            # have changed, no fresh leaf_value to assign).
            n.Q = n.W / n.N if n.N > 0 else 0.0
        else:
            n.Q = n.W / n.N if n.N > 0 else 0.0
            if self.opp_pessimism_alpha < 1.0:
                self._refresh_opp_q(n)

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

    def _select_to_leaf(self, root, root_gs, strange=False):
        """Descend from root to a leaf, replay GameState, set lazy flags.

        On first visit to a node, sets node.is_terminal / node.is_hero from
        the replayed GameState (deferred from _expand_node).

        `strange`: when True, hero selection uses inverse-N sampling instead
        of PUCT (opp selection is unchanged). Triggered by `self.strange_p`
        in `search()` to broaden exploration once the action head has mostly
        converged. Under hero max-W backup, a strange path that lands in a
        worse leaf does not depress hero ancestors' W (max ignores it), so
        these explorations cost only the budget, not the value estimate.

        Returns:
            path: list[MCTSNode] from root to leaf (length >= 1)
            gs:   GameState at the leaf, or None if the leaf is an
                  already-known terminal (no replay needed)
        """
        node = root
        path = [node]
        while node.children and not node.is_terminal:
            action = self._select_child(node, strange=strange)
            node = node.children[action]
            path.append(node)

        if node.is_terminal:
            return path, None

        gs = self._replay_game_state(root_gs, path)
        if node is not root:
            node.is_terminal = gs.is_terminal
            node.is_hero = gs.is_hero_turn()
        return path, gs

    def _simulate(self, root, root_ctx, root_mask, root_gs):
        """One MCTS simulation: select → expand → backup. (Legacy
        single-sim entry point, not used by the batched `search()`.)"""
        path, gs = self._select_to_leaf(root, root_gs)
        node = path[-1]

        if node.is_terminal:
            if node._term_value is None:
                # First terminal visit — evaluate value_head once and cache.
                context, mask = self._build_context(root_ctx, root_mask, path)
                leaf_value = self.agent.value_head(context, mask=mask).item()
                node._term_value = float(leaf_value)
            self._backup_cached_terminal(path, node._term_value)
            return

        # BUILD CONTEXT: root_ctx + action embeddings along path
        context, mask = self._build_context(root_ctx, root_mask, path)

        # EXPAND leaf
        if node is root:
            leaf_value = self.agent.value_head(context, mask=mask).item()
        else:
            leaf_value, act_logits, opp_logits, act_embs = \
                self._evaluate_node(context, mask)
            self._expand_node(node, gs, act_logits, opp_logits, act_embs)

        # BACKUP (mixed: hero non-terminal = max child.W, opp/terminal = sum).
        for n in path:
            n.N += 1
            if (not n.is_hero) or n.is_terminal:
                n.W += leaf_value
        for n in reversed(path):
            self._recompute_node_after_backup(n, leaf_value=leaf_value)

    def _select_child(self, node, strange=False):
        """Select child action.

        Regular: PUCT at hero, prior-weighted sampling at opp.
        Strange (`strange=True`): inverse-N sampling at hero
            (`P(a) ∝ 1/(N(a)+1)`), opp unchanged. The opp branch is identical
            in both modes — opponents don't know we're exploring, so their
            response distribution stays realistic.
        """
        if node.is_hero:
            if strange:
                actions = list(node.children.keys())
                weights = [1.0 / (node.children[a].N + 1.0) for a in actions]
                return random.choices(actions, weights=weights, k=1)[0]
            # Regular PUCT
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

        For opp nodes, `self.opp_prior_smoothing` mixes the model prior with
        uniform-over-legal to keep search exploring off-model opp replies
        — critical when the opp's true policy (e.g. Slumbot) differs from
        the agent pool that trained `opponent_action_head`. Hero priors are
        untouched (they get root-only Dirichlet noise via `_add_dirichlet_noise`).
        """
        legal = game_state.get_legal_actions()
        logits = act_logits if node.is_hero else opp_logits
        logits = logits[0]  # (n_actions,)

        # Mask illegal actions and compute priors
        mask = torch.full_like(logits, float("-inf"))
        for a in legal:
            mask[a] = 0.0
        priors = F.softmax(logits + mask, dim=0).tolist()  # one host transfer

        # Smooth opp priors with uniform over legal actions to keep mass on
        # off-model responses. Applied at expansion (so it affects both
        # selection-time sampling and prior-based Q blending downstream).
        if (not node.is_hero) and self.opp_prior_smoothing > 0.0 and legal:
            lam = self.opp_prior_smoothing
            u = 1.0 / float(len(legal))
            for a in legal:
                priors[a] = (1.0 - lam) * priors[a] + lam * u

        # Light Dirichlet noise on **inner** hero nodes (root has its own,
        # stronger noise via `_add_dirichlet_noise`). Without this, when
        # `action_head` is sharp, PUCT explores at most 1–2 hero actions
        # below the root and the deep subtree is effectively single-path.
        # Skipped at root (parent is None) — there `_add_dirichlet_noise`
        # already mixes a larger ε.
        if (node.is_hero and node.parent is not None
                and self.dirichlet_epsilon_inner > 0.0 and len(legal) > 0):
            noise = np.random.dirichlet(
                [self.dirichlet_alpha_inner] * len(legal))
            eps = self.dirichlet_epsilon_inner
            for i, a in enumerate(legal):
                priors[a] = (1.0 - eps) * priors[a] + eps * float(noise[i])

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

    def _refresh_opp_q(self, node):
        """Override opp node Q with `α·E_P[Q] + (1−α)·min Q` over visited children.

        Default sample-mean (``W/N``) treats the opp model's prior as ground
        truth: when an opp samples a fold-heavy response, the empirical Q
        at the parent's opp child climbs toward "hero gets the pot",
        regardless of whether off-model opp lines would punish that hero
        action. Blending in ``min_a Q(child_a)`` makes the value the hero
        sees through the opp ancestor pessimistic — branches where any
        decent opp reply is bad for hero get discounted.

        ``E_P[Q]`` uses the (smoothed) priors as weights, not visit
        proportions, so it's robust to under-explored children (whose
        ``W/N`` is high-variance). Falls back to a plain visited-average
        if all visited priors are ~0.

        No-op when ``opp_pessimism_alpha >= 1.0`` (full trust in opp
        model, legacy behaviour) or when no child has been visited yet.
        Writes only ``node.Q``; never touches ``W`` or ``N`` (those stay
        the zero-sum-preserving sample of leaf values).
        """
        if node.is_hero or self.opp_pessimism_alpha >= 1.0:
            return
        visited = [c for c in node.children.values() if c.N > 0]
        if not visited:
            return
        total_p = sum(c.P for c in visited)
        if total_p < 1e-12:
            expected_q = sum(c.Q for c in visited) / float(len(visited))
        else:
            expected_q = sum(c.P * c.Q for c in visited) / total_p
        min_q = min(c.Q for c in visited)
        alpha = self.opp_pessimism_alpha
        node.Q = alpha * expected_q + (1.0 - alpha) * min_q

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


def action_path_from_root(node):
    """List of `action_idx` from root → `node` (root excluded)."""
    path = []
    n = node
    while n.parent is not None:
        if n.action_idx is not None:
            path.append(n.action_idx)
        n = n.parent
    path.reverse()
    return path


def re_backup_terminals(root):
    """Propagate equity-based terminal Q values up to ancestors after search.

    `_backup_terminal` leaves `terminal.W == 0` during the search (only bumps
    `N`), so this is the sole path by which terminal information enters
    ancestors' `W` / `Q`. `evaluate_all_terminals` first writes the
    equity-based value into `terminal.Q`; here we set
    `terminal.W = terminal.Q * terminal.N` and propagate the change upward
    respecting the **mixed backup semantics**:

      - **opp ancestor**: `W += delta` (additive sum continues across all
        terminals sharing this ancestor — order-independent).
      - **hero ancestor**: `W = max(child.W for c in n.children if c.N > 0)`.
        The new max is computed from already-updated children below; if max
        didn't change, propagation can stop early because no ancestor above
        will see a different value through this branch (the change carried
        by `delta` becomes 0).

    Multiple terminals are processed sequentially; opp accumulators handle
    that natively, and hero `max(...)` recomputation is order-independent.
    Idempotent: re-running on the same tree adds 0 everywhere.
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
            if node.is_hero:
                visited = [c for c in node.children.values() if c.N > 0]
                if not visited:
                    break
                new_node_W = max(c.W for c in visited)
                delta = new_node_W - node.W
                node.W = new_node_W
                node.Q = node.W / node.N if node.N > 0 else 0.0
                if delta == 0.0:
                    break  # max unchanged → no further upward change
            else:
                node.W += delta
                node.Q = node.W / node.N if node.N > 0 else 0.0
            node = node.parent


def get_n_distribution(root, n_actions, label_smoothing=0.0):
    """Extract visit-count distribution from root's children.

    Returns a list of length ``n_actions`` with normalized visit counts.
    Illegal actions (no child node) remain 0.

    ``label_smoothing`` ε mixes the visit distribution with uniform mass
    over LEGAL actions:

        ``target[a] = (1 − ε) · N(a)/ΣN  +  ε / n_legal``  for a in legal
        ``target[a] = 0``                                  otherwise

    so that legal actions with zero or near-zero visits keep a small mass
    on the KL target. This breaks the cycle-over-cycle policy sharpening
    pathology: with ε=0 and ``n_simulations=3000`` against a 14-action
    space, postflop trees routinely leave half the legal actions at N=0;
    KL training then concentrates all action_head mass on the visited
    subset, the next cycle's MCTS sees an even sharper prior, fewer
    actions get explored, etc. A tiny ε (~0.03–0.05) is enough to hold
    the floor without distorting the visit-based policy signal.
    """
    legal_actions = list(root.children.keys())
    n_legal = len(legal_actions)
    total = sum(root.children[a].N for a in legal_actions)
    dist = [0.0] * n_actions

    if n_legal == 0:
        # Pathological: no legal children; uniform over the full action set.
        return [1.0 / n_actions] * n_actions

    if total == 0:
        # No visits — fall back to uniform over legal actions (illegal stay 0).
        u = 1.0 / float(n_legal)
        for a in legal_actions:
            dist[a] = u
        return dist

    if label_smoothing > 0.0:
        eps = float(label_smoothing)
        uniform = eps / float(n_legal)
        for a in legal_actions:
            child = root.children[a]
            dist[a] = (1.0 - eps) * (child.N / total) + uniform
    else:
        for a in legal_actions:
            child = root.children[a]
            dist[a] = child.N / total
    return dist
