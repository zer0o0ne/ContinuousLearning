"""The v8 agent as a Slumbot player (CONCEPT.md §12, `PLAN_PIPELINE.md` S10).

`evaluation/protocol.py` turns Slumbot's wire into a betting state; this file
turns that state into the observation v8 was trained on and asks the agent what
to do. It is the half of v7's `slumbot_eval.py` that had to be rewritten, and
nothing else did.

**One observation builder, one legality rule, one last mile.** The agent is
reached through `AgentPoolMember`, exactly as the driver reaches it during
self-play and the oracle reaches it inside a rollout — so the tokens come from
`nets.features.hand_tokens`, the mask comes from `env.legal.legal_action_mask`,
and logits become a played distribution in `PoolMember.policy`. None of the
three has a Slumbot branch. What this file adds is a `HandRecord` built from a
replay rather than from the engine, and that is the *only* second construction
path it introduces; §9's parity is a property of `hand_tokens` and holds here
for the same reason it holds there.

**Why the record is built rather than replayed through the engine.** The engine
could replay the hand from a pinned deck and a forced prefix (S1), and it would
be one path fewer — but a forced action is an *index*, and the engine would then
size the opponent's bets from our own raise bins. Slumbot does not bet in our
bins. The pot and the stacks the agent reads would be the abstraction's, not the
table's, and the agent would misjudge the pot by whatever the nearest bin
happened to miss by. The abstraction is unavoidable in what hero can *say*
(`action_idx_to_incr`); it must not leak into what hero *sees*. So the chips in
this record are Slumbot's, to the chip, and only the action indices on the
tokens are abstracted.

**The two frames meet here and nowhere else.** Slumbot numbers seats
`pos 0 = BB`, `pos 1 = SB`; v8's engine posts the small blind at seat 0 and has
seat 1 act first postflop heads-up. The conventions are exact mirrors, so
`v8_seat = 1 - slumbot_pos` on every street, and it is applied once, in
`_flip`.

**Hero is slot 0.** As in every session the agent was trained on
(`env/session.py`, `ARCHITECTURE.md` §4 interpretive decision 6), so the vector
at slot 0 is hero's and the one at slot 1 is the opponent's. Cold start is a
table of zeros (§5.5); §12's *warm* run replaces it through `set_embeddings`
every `R` hands, with the same generic fit used against every other opponent —
which `CONCEPT.md` §10 records as adaptation and not specialisation.

**What this file refuses.** A table configuration outside the ranges the agent
was trained over (`game.players_range`, `game.stack_bb_range`) is an error, not
something to clamp: Slumbot deals heads-up 200 BB, and if that ever falls
outside the training distribution the honest outcome is a stopped run rather
than a number nobody can interpret.
"""

import numpy as np

from agent.policy import AgentPoolMember
from env.driver import DecisionContext, HandRecord, HandSpec
from env.legal import legal_action_mask
from env.session import raise_sizes_from
from env.showdown import label_showdowns, showdown_positions
from env.table import Table
from evaluation.protocol import (
    SLUMBOT_BIG_BLIND, SLUMBOT_STACK_SIZE, action_idx_to_incr, clamp_counters,
    effective_action_idx, replay_action_string,
)

N_SEATS = 2
HERO_SLOT = 0
OPP_SLOT = 1
N_CARDS = 52


def _flip(slumbot_pos):
    """Slumbot's seat → v8's seat. The mirror, applied once (module docstring)."""
    return N_SEATS - 1 - int(slumbot_pos)


def _reorder(values):
    """A per-seat array in Slumbot's frame, in v8's."""
    return [values[_flip(seat)] for seat in range(N_SEATS)]


def check_table_is_in_range(game):
    """Slumbot's table against the ranges the agent was trained over.

    `CLAUDE.md` §1 samples 2–9 players and 10–300 BB; Slumbot is one point in
    that space and the whole bet is that the agent arrives there as one case
    among many. If a config ever puts that point outside the training
    distribution, the run stops — a clamped table would report a number about a
    situation the agent never saw, under the name of one it did.
    """
    lo_p, hi_p = game["players_range"]
    lo_s, hi_s = game["stack_bb_range"]
    stack_bb = SLUMBOT_STACK_SIZE / SLUMBOT_BIG_BLIND
    assert int(lo_p) <= N_SEATS <= int(hi_p), (
        f"Slumbot deals a {N_SEATS}-handed table and `game.players_range` is "
        f"{[lo_p, hi_p]} — the agent was not trained on this table size")
    assert float(lo_s) <= stack_bb <= float(hi_s), (
        f"Slumbot deals {stack_bb:g} BB and `game.stack_bb_range` is "
        f"{[lo_s, hi_s]} — the agent was not trained on this stack depth")
    return N_SEATS, stack_bb


def _table_view(state, game, scale):
    """A real `env.table.Table` holding one replayed Slumbot state.

    Assigned rather than played, because the state came off the wire and not out
    of the engine. It exists for exactly one reason: `env.legal` is the one
    legality rule in v8 (`CONCEPT.md` §6.2) and it reads a `Table`, so this is
    how the mask hero acts under is the same mask hero was trained under, rather
    than a second implementation of the same paragraph.

    Two fields have no direct counterpart on the wire:

    * `last_raise_size` is the minimum legal raise *increment*, which Slumbot
      expresses as `max(big blind, last_bet_size)` in its own min-raise rule —
      the same quantity, so it is mapped straight across;
    * `_last_full_raise_level` guards v8's "a short all-in did not reopen the
      betting" case, which cannot arise heads-up: a short all-in makes every
      other live player all-in, and `legal_actions` already drops every raise on
      that branch. Setting it to the current high bet makes the guard inert, so
      the reopen rule is decided by the branch that actually applies.
    """
    table = Table(num_players=N_SEATS, raise_sizes=raise_sizes_from(game),
                  start_credits=SLUMBOT_STACK_SIZE * scale,
                  big_blind=float(game["big_blind"]),
                  small_blind=float(game["small_blind"]))
    table.turn = int(state["turn"])
    table.pot = float(state["pot"]) * scale
    table.bets = np.asarray([b * scale for b in _reorder(state["bets"])],
                            dtype=float)
    table.credits = [float(c) * scale for c in _reorder(state["credits"])]
    table.high_bet = float(state["high_bet"]) * scale
    table.players_state = np.asarray(_reorder(state["players_state"]),
                                     dtype=float)
    table.active_player = _flip(state["active_pos"])
    table.several_all_in = False
    table.last_raise_size = max(float(game["big_blind"]),
                                float(state["last_bet_size"]) * scale)
    table._last_full_raise_level = table.high_bet
    return table


def _deck(hole_cards, board, hero_seat, bot_hole_cards=None):
    """A 52-card permutation carrying what hero can see, and filler elsewhere.

    `hand_tokens` reads the board through `_board_as_of`, which masks by the
    snapshot's street, and the hole cards only at the observer's own seat — so
    the filler is unreachable by construction and the observation is hero's
    information exactly. It is a permutation rather than a padded array because
    that is what a `HandRecord`'s deck is everywhere else in v8.

    `bot_hole_cards` is the reveal Slumbot publishes when a hand goes to
    showdown, and it is filled in only for a *finished* hand (§5.1a's terminal
    token, which no decision token can attend to). During the hand the
    opponent's two slots hold filler, because during the hand hero has not seen
    them.
    """
    deck = np.full(N_CARDS, -1, dtype=np.int64)
    revealed = [int(c) for c in board if int(c) >= 0]
    deck[:len(revealed)] = revealed
    deck[5 + 2 * hero_seat: 7 + 2 * hero_seat] = [int(c) for c in hole_cards]
    if bot_hole_cards:
        opp = N_SEATS - 1 - hero_seat
        deck[5 + 2 * opp: 7 + 2 * opp] = [int(c) for c in bot_hole_cards]
    used = np.zeros(N_CARDS, dtype=bool)
    used[deck[deck >= 0]] = True
    deck[np.flatnonzero(deck < 0)] = np.flatnonzero(~used)
    return deck


def _snapshot(state, scale, action):
    """One snapshot in the driver's own format (`env/driver.py::_snapshot`)."""
    return {
        "pot": float(state["pot"]) * scale,
        "bets": np.asarray([b * scale for b in _reorder(state["bets"])],
                           dtype=float),
        "credits": [float(c) * scale for c in _reorder(state["credits"])],
        "turn": int(state["turn"]),
        "active_pos": _flip(state["active_pos"]),
        "action": action,
    }


def _replay(action_str, client_pos, game, hero_action_indices):
    n_actions = int(game["n_actions"])
    return replay_action_string(
        action_str, raise_sizes_from(game), n_actions - 3,
        hero_pos=int(client_pos), hero_action_indices=hero_action_indices)


def _build(state, steps, client_pos, deck, game):
    """The `HandRecord` of everything that has happened, and nothing else.

    The snapshot sequence is the driver's: one initial snapshot, then a
    pre-decision and a post-action snapshot per action. The duplicate that opens
    the list (initial and the first pre-decision describe the same table) is the
    driver's shape too — the agent reads only the snapshots its decisions point
    at, and matching the shape is what lets anything that reasons about
    `snap_idx` work here unchanged.
    """
    n_actions = int(game["n_actions"])
    scale = float(game["big_blind"]) / SLUMBOT_BIG_BLIND
    hero_seat = _flip(client_pos)
    raise_sizes = raise_sizes_from(game)

    spec = HandSpec(
        num_players=N_SEATS,
        start_credits=[SLUMBOT_STACK_SIZE * scale] * N_SEATS,
        seat_members=[HERO_SLOT if seat == hero_seat else OPP_SLOT
                      for seat in range(N_SEATS)],
        seed=0,
        big_blind=float(game["big_blind"]),
        small_blind=float(game["small_blind"]),
        raise_sizes=raise_sizes,
        meta={"tag": "slumbot", "client_pos": int(client_pos)},
    )

    opening = steps[0]["state"] if steps else state
    snapshots = [_snapshot(opening, scale, None)]
    decisions = []
    for i, step in enumerate(steps):
        pre = step["state"]
        snapshots.append(_snapshot(pre, scale, None))
        action_idx = int(step["action_idx"])
        onehot = np.zeros(n_actions, dtype=np.float32)
        onehot[action_idx] = 1.0
        decisions.append({
            "snap_idx": len(snapshots) - 1,
            "acting_pos": _flip(step["acting_pos"]),
            "member": spec.seat_members[_flip(step["acting_pos"])],
            "action_idx": action_idx,
            "legal_mask": legal_action_mask(_table_view(pre, game, scale),
                                            n_actions),
        })
        post = steps[i + 1]["state"] if i + 1 < len(steps) else state
        snapshots.append(_snapshot(post, scale, onehot.tolist()))

    return HandRecord(
        spec=spec, deck=deck, snapshots=snapshots, decisions=decisions,
        rewards=np.zeros(N_SEATS, dtype=np.float64), truncated=False)


def slumbot_record(action_str, client_pos, hole_cards, board, game,
                   hero_action_indices=()):
    """The v8 observation of the moment Slumbot is asking hero about.

    Returns `(record, ctx, state)` — a `HandRecord` of everything that has
    happened, the `DecisionContext` of the pending decision, and the replayed
    Slumbot-frame state the wire token will be built from.
    """
    n_actions = int(game["n_actions"])
    scale = float(game["big_blind"]) / SLUMBOT_BIG_BLIND
    hero_seat = _flip(client_pos)

    state, steps = _replay(action_str, client_pos, game, hero_action_indices)
    assert not state["is_terminal"], (
        f"Slumbot is asking for an action but the replayed hand is over "
        f"(action={action_str!r})")
    assert int(state["active_pos"]) == int(client_pos), (
        f"Slumbot is asking hero to act but the replay says seat "
        f"{state['active_pos']} is to act (action={action_str!r})")

    record = _build(state, steps, client_pos,
                    _deck(hole_cards, board, hero_seat), game)
    record.snapshots.append(_snapshot(state, scale, None))
    ctx = DecisionContext(
        record, len(record.snapshots) - 1, hero_seat,
        legal_action_mask(_table_view(state, game, scale), n_actions),
        int(state["turn"]))
    return record, ctx, state


def slumbot_history(action_str, client_pos, hole_cards, board, game,
                    winnings, hero_action_indices=(), bot_hole_cards=None):
    """A **finished** hand as a v8 `HandRecord` — the corpus the §5.5 fit reads.

    The difference from `slumbot_record` is the one §9 insists on: this is the
    other moment. A finished hand has no pending decision, and by the time it is
    finished the showdown is part of what the observer knows — so the reveal
    Slumbot publishes goes into the deck and `env.showdown` labels it, exactly
    as `env/session.py::play` does for a hand the engine dealt. What §5.1a still
    refuses is that knowledge flowing *backwards* into the decisions of the same
    hand, and that is a property of the attention mask, not of this file.

    A hand Slumbot settles without a reveal — a fold, or a showdown whose
    `bot_hole_cards` the response did not carry — contributes its decision
    tokens and no terminal token. `showdown_positions` refusing to name a seat
    whose cards nobody has is the honest outcome, and `objective` treats a batch
    with no showdown in it as a legitimate batch rather than an error.
    """
    scale = float(game["big_blind"]) / SLUMBOT_BIG_BLIND
    hero_seat = _flip(client_pos)

    # The hand is over because the server said so — `winnings` came back — and
    # the replay is not asked to agree. It cannot always: an all-in runout ends
    # the betting without a token that closes a street, so the inherited parser
    # walks on to the next street and never marks itself terminal. What the
    # replay is for here is the *decisions*, and those it has either way.
    state, steps = _replay(action_str, client_pos, game, hero_action_indices)

    record = _build(state, steps, client_pos,
                    _deck(hole_cards, board, hero_seat, bot_hole_cards), game)
    record.rewards[hero_seat] = float(winnings) * scale
    record.rewards[N_SEATS - 1 - hero_seat] = -float(winnings) * scale
    if bot_hole_cards:
        record.showdown = showdown_positions(_reorder(state["players_state"]))
        label_showdowns([record])
    return record


class SlumbotAgent:
    """The v8 agent seated at Slumbot's table.

    Args:
        net: a trained `AgentNet`.
        game: the `game` config section — the action set, the raise grid and the
            ranges the table is checked against.
        device: where the forward runs.
        embeddings: `(max_players, d_emb)`, slot 0 hero and slot 1 the opponent.
            Zeros — the §5.5 cold start, and §12's *cold* run — when omitted.
        rng: `np.random.Generator`. The action is **sampled** from the agent's
            own distribution, because that distribution is the policy; the
            generator is owned here so a resumed evaluation replays the same
            draws (§12).
    """

    def __init__(self, net, game, device, embeddings=None, rng=None):
        check_table_is_in_range(game)
        self.net = net
        self.game = game
        self.device = device
        self.n_actions = int(game["n_actions"])
        self.n_raise_bins = self.n_actions - 3
        self.max_players = int(game["max_players"])
        self.raise_sizes = raise_sizes_from(game)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.embeddings = np.zeros((self.max_players, net.d_emb),
                                   dtype=np.float32)
        if embeddings is not None:
            self.set_embeddings(embeddings)

    def set_embeddings(self, vectors):
        """Install the fitted vectors — §12's *warm* run, every `R` hands."""
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.shape == (self.max_players, self.net.d_emb), (
            f"embeddings are (max_players, d_emb) = "
            f"({self.max_players}, {self.net.d_emb}); got {vectors.shape}")
        self.embeddings = vectors

    def policy(self, action_str, client_pos, hole_cards, board,
               hero_action_indices=()):
        """The agent's distribution over the actions legal right now.

        Returns `(probs, record, ctx, state)`. Split out from `act` so §12 can
        report what the policy *was*, and so a test can read it without a wire.
        """
        record, ctx, state = slumbot_record(
            action_str, client_pos, hole_cards, board, self.game,
            hero_action_indices=hero_action_indices)
        hero_seat = _flip(client_pos)
        slot_of_seat = [HERO_SLOT if seat == hero_seat else OPP_SLOT
                        for seat in range(N_SEATS)]
        member = AgentPoolMember(self.net, self.embeddings, slot_of_seat,
                                 self.max_players, self.n_actions, hero_seat,
                                 self.device)
        probs = np.asarray(member.policy([ctx])[0], dtype=np.float64)
        return probs, record, ctx, state

    def act(self, action_str, client_pos, hole_cards, board,
            hero_action_indices=(), counters=None):
        """One decision. Returns `(incr, effective_idx, chosen_idx)`.

        `incr` is what goes on the wire, `effective_idx` is what hero can be
        said to have played once `action_idx_to_incr` has had its say, and
        `chosen_idx` is what the agent asked for. The two differ exactly when a
        clamp fired, and the caller appends `effective_idx` — never
        `chosen_idx` — to `hero_action_indices`, or every later observation in
        the hand carries an action hero did not take.
        """
        probs, _record, _ctx, state = self.policy(
            action_str, client_pos, hole_cards, board, hero_action_indices)
        chosen = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        street = self.raise_sizes[int(state["turn"])]
        incr = action_idx_to_incr(
            state, chosen, street, self.n_raise_bins,
            hero_slumbot_pos=int(client_pos),
            clamp_counters=counters if counters is not None
            else clamp_counters())
        return incr, effective_action_idx(state, incr, street,
                                          self.n_raise_bins), chosen
