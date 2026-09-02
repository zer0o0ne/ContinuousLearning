"""Building the §5.1 situation token features from played hands.

This is where observation parity lives (CONCEPT.md §9):

    Every observation the embedding network or the agent ever sees, at training
    or at inference, is constructible from what the observer could have known at
    that moment.

Concretely, in this file:

* one token per **decision** — every decision by every player, not only the
  observer's;
* the board is the board as of that decision's street, never the final board;
* hole cards are the **observer's own**; every other player's two card slots are
  the unknown token (52). This rule holds on **every** token type, showdown
  tokens included: the observer never has to infer its own hand, and no token
  ever shows anybody else's.

* everything else in the token (stack, pot, amount to call, position, number of
  players, per-seat stacks, the previous action) is public;
* everything monetary is in big blinds.

**Showdown tokens (§5.1a).** A hand that reaches showdown gets one extra token
per revealed player, appended after every decision token. §5.1's exception —
opponents' cards may appear "in tokens strictly after the reveal" — is what
these are: they are the only tokens after the reveal, because the reveal happens
when the hand is already over.

**Another seat's** revealed cards are the **target** of those tokens, not an
input: their card slots stay masked and the head predicts what was shown
(`env/showdown.py`). That masking is the whole mechanism — attention is causal
and no decision token attends to a showdown token, so the only channel from a
reveal to that player's vector is the gradient of a loss whose answer is *not*
in the input. Put another seat's cards in and the anchor stops carrying style
altogether.

The **observer's own** showdown token is the exception, and it is not one of
substance: the observer knows its own hand, so its slots carry it (owner
decision 2026-08-20) and the rule above becomes uniform across token types. The
price is on the metric rather than on the model — see `nets/embedding_net.py`.

Because attention is causal within the hand, a showdown token attends to every
decision token and **no decision token attends to it**, so the action-prediction
task is untouched and nothing about the outcome flows backwards.

This is why the cards are not simply written into the decision tokens of a
showdown hand, even though a completed hand's cards are legitimately part of
what the observer knows by the time the embedding is fitted: doing that would
tell every decision token that this player *reached* showdown — i.e. that they
were not going to fold — which is a future leak that improves the prediction
loss while carrying no style at all.

**The strength target (§5.6).** Every decision token of the *observer* carries
`own_strength`, the percentile of the observer's own hand on that hand's final
board. Like the showdown labels it is a **target and never an input**: the card
slots and the board of the token are untouched, so the observation is bit-for-bit
what it was before this field existed, and predicting a quantity the observer
will only learn later is what a value target does rather than a leak. It is
`-1` — outside the [0, 1] a percentile lives in, hence its own mask — on every
other token, on every token of a hand still in progress (no final board exists
yet), and on every token of a record nobody labelled.

**The range target (§5.7).** The observer's belief about every *other* live
player, at every token: one weight vector over the 1326 two-card combos per
(token, seat), produced by `oracle/ranges.py`. Like the two above it is a
**target and never an input** — nothing about anybody else's cards enters the
token, and the belief is a function of the history the observer has already
seen, so it carries no future. It is stored sparsely (`RangeTargets`) because a
concentrated range is a handful of combos and a dense 1326-vector per token per
seat would be the largest thing in a corpus by an order of magnitude.

`active_opp` says which seats that belief is *about* — seats that are not the
observer and have not folded yet. It is public information (a fold is public),
it is needed on every batch and not only on a labelled one, because §5.7's head
runs in the forward whether or not a target exists, and it excludes showdown
tokens: the hand is over there and there is no belief left to hold.

`own_hole` carries the observer's own two cards on **every** token, not only on
the ones where the observer acts. That is not new information — the observer has
always known its own hand — but the decision token only ever showed it on the
observer's own rows, and the range head needs it on every row: the support of a
belief is exactly "the combos the board and the observer's own cards leave", and
blockers are half of what a range is about. Nothing else reads it; the tokeniser
does not, so the §5.1 observation is bit-for-bit what it was.

`seat_slot` and `seat_member` are the same naming as `slot`/`member` but for
**every seat at once** rather than the acting one, because the range head is
asked about a player who is not the one acting and needs that player's vector.

The token's `member` and `slot` fields are the two ways a player is named. At
training a player *is* a pool member and its vector is a row of the trainable
table (§5.4); at inference a player is a seat at the observed table and its
vector is one of the free parameters being fitted (§5.5). Both indices are
carried so one token batch serves both.
"""

from dataclasses import dataclass

import numpy as np
import torch

from oracle.ranges import N_COMBOS

UNKNOWN_CARD = 52
TOKEN_DECISION = 0
TOKEN_SHOWDOWN = 1
N_TOKEN_TYPES = 2


@dataclass
class RangeTargets:
    """§5.7 targets of one hand, flat and sparse. `K` entries in total."""

    token: np.ndarray          # (K,) int32 — token index within the hand
    seat: np.ndarray           # (K,) int16 — the seat the belief is about
    combo: np.ndarray          # (K,) int16 — canonical index, 0..1325
    weight: np.ndarray         # (K,) float32 — sums to 1 per (token, seat)


@dataclass
class HandTokens:
    """Token features of one hand, from one observer's view. Length T."""

    cards: np.ndarray          # (T, 7) int64 — 5 board + 2 hole, 52 = unknown
    decision_idx: np.ndarray   # (T,) int64
    acting_pos: np.ndarray     # (T,) int64
    num_players: np.ndarray    # (T,) int64
    scalars: np.ndarray        # (T, 3) float32 — stack, pot, to_call (BB)
    seat_stacks: np.ndarray    # (T, max_players) float32 — BB, zero-padded
    prev_action: np.ndarray    # (T, n_actions) float32
    member: np.ndarray         # (T,) int64 — pool member acting
    slot: np.ndarray           # (T,) int64 — seat-independent player slot
    action: np.ndarray         # (T,) int64 — the action actually taken
    legal: np.ndarray          # (T, n_actions) bool
    token_type: np.ndarray     # (T,) int64 — TOKEN_DECISION / TOKEN_SHOWDOWN
    sd_strength: np.ndarray    # (T,) float32 — revealed-hand percentile
    sd_class: np.ndarray       # (T,) int64 — revealed-hand 169-way class
    # §5.6 — the observer's own hand's percentile on the final board, on the
    # observer's own decision tokens. `-1` everywhere else, which is the mask:
    # a percentile is in [0, 1], so the sentinel cannot collide with a label.
    own_strength: np.ndarray   # (T,) float32
    # §5.7 — the observer's own two cards on every token, for the range head's
    # combo mask and for nothing else.
    own_hole: np.ndarray       # (T, 2) int64
    # §5.7 — every seat's slot and member, not only the acting one, and which
    # seats the range head is asked about at this token.
    seat_slot: np.ndarray      # (T, max_players) int64
    seat_member: np.ndarray    # (T, max_players) int64
    active_opp: np.ndarray     # (T, max_players) bool
    # §5.7 — the sparse targets. `None` on a hand nobody tracked ranges for,
    # which is every hand of an evaluation replay and of a corpus built with
    # the head switched off.
    ranges: RangeTargets = None

    def __len__(self):
        return len(self.decision_idx)


# The fields that are one row per token, in the order `collate` and the shard
# writer walk them. `ranges` is deliberately not here: it is `K` rows and not
# `T`, so it is padded, sharded and concatenated by its own code.
TOKEN_FIELDS = tuple(
    f for f in (
        "cards", "decision_idx", "acting_pos", "num_players", "scalars",
        "seat_stacks", "prev_action", "member", "slot", "action", "legal",
        "token_type", "sd_strength", "sd_class", "own_strength",
        "own_hole", "seat_slot", "seat_member", "active_opp",
    ))


def _board_as_of(deck, turn):
    if turn == 0:
        return [UNKNOWN_CARD] * 5
    if turn == 1:
        return [int(c) for c in deck[:3]] + [UNKNOWN_CARD] * 2
    if turn == 2:
        return [int(c) for c in deck[:4]] + [UNKNOWN_CARD]
    return [int(c) for c in deck[:5]]


def hand_tokens(record, observer_pos, slot_of_seat, max_players, n_actions,
                pending=None, ranges=None):
    """Tokenise one `HandRecord` from `observer_pos`'s point of view.

    Args:
        record: a `env.driver.HandRecord`.
        observer_pos: the seat whose information defines the observation. Must
            be a seat at this table — the observer has to have been sitting
            there (§5.4).
        slot_of_seat: seat → player slot, stable across the hands of a session
            even as the button rotates.
        max_players: width of the per-seat stack vector.
        n_actions: size of the action set.
        pending: a `DecisionContext` for a decision that has not been taken yet.
            Appends one extra decision token with `action = -1` and `legal` from
            the context — the moment the *agent* observes (§9), as opposed to
            the completed hand the embedding network observes. A hand with a
            pending decision has no showdown, and passing both is refused.
        ranges: `{(token, seat): (combo_idx, weight)}` from
            `oracle.ranges.hand_ranges`, the §5.7 target. Keys naming a
            (token, seat) this observer has no live opponent at are refused
            rather than dropped: the two are derived from the same fold rule and
            a disagreement means one of them is wrong.
    """
    assert 0 <= observer_pos < record.num_players, (
        f"observer seat {observer_pos} is not at this {record.num_players}-handed "
        f"table — an observation may only be built for a seated observer")

    bb = float(record.spec.big_blind)
    n = record.num_players
    decisions = record.decisions
    n_dec = len(decisions)

    revealed = sorted(record.showdown)
    if revealed:
        assert record.showdown_strength and record.showdown_class, (
            "this hand reached showdown but carries no labels — call "
            "`env.showdown.label_showdowns` over the corpus first (§5.1a)")
    assert pending is None or not revealed, (
        "a hand with a pending decision has not reached showdown — passing "
        "both means the caller has confused the two moments of §9")
    T = n_dec + len(revealed) + (1 if pending is not None else 0)

    cards = np.full((T, 7), UNKNOWN_CARD, dtype=np.int64)
    decision_idx = np.arange(T, dtype=np.int64)
    acting_pos = np.zeros(T, dtype=np.int64)
    num_players = np.full(T, n, dtype=np.int64)
    scalars = np.zeros((T, 3), dtype=np.float32)
    seat_stacks = np.zeros((T, max_players), dtype=np.float32)
    prev_action = np.zeros((T, n_actions), dtype=np.float32)
    member = np.zeros(T, dtype=np.int64)
    slot = np.zeros(T, dtype=np.int64)
    action = np.zeros(T, dtype=np.int64)
    legal = np.zeros((T, n_actions), dtype=bool)
    token_type = np.full(T, TOKEN_DECISION, dtype=np.int64)
    sd_strength = np.zeros(T, dtype=np.float32)
    sd_class = np.zeros(T, dtype=np.int64)
    own_strength = np.full(T, -1.0, dtype=np.float32)
    own_hole = np.tile(np.asarray(record.hole_cards(observer_pos),
                                  dtype=np.int64), (T, 1))
    seat_slot = np.zeros((T, max_players), dtype=np.int64)
    seat_member = np.zeros((T, max_players), dtype=np.int64)
    active_opp = np.zeros((T, max_players), dtype=bool)

    # §5.7 — every seat's naming, constant over the hand, and who the belief is
    # about at each token. A player who folds at token `t` was still holding
    # cards while that decision was taken, so they are active *at* `t` and gone
    # from `t + 1`; dropping them for the whole hand would condition the target
    # on the future.
    seat_slot[:, :n] = np.asarray(slot_of_seat, dtype=np.int64)[None, :n]
    seat_member[:, :n] = np.asarray(record.spec.seat_members,
                                    dtype=np.int64)[None, :n]
    folded = set()
    for t in range(n_dec + (1 if pending is not None else 0)):
        for seat in range(n):
            active_opp[t, seat] = seat != observer_pos and seat not in folded
        if t < n_dec and int(decisions[t]["action_idx"]) == 0:
            folded.add(int(decisions[t]["acting_pos"]))

    def _fill_decision(t, snap_idx, pos):
        """The public part of a decision token — identical for a decision that
        was taken and one that is pending."""
        snap = record.snapshots[snap_idx]
        bets = np.asarray(snap["bets"], dtype=np.float64)
        credits = np.asarray(snap["credits"], dtype=np.float64)

        cards[t, :5] = _board_as_of(record.deck, int(snap["turn"]))
        if pos == observer_pos:
            cards[t, 5:] = record.hole_cards(pos)

        acting_pos[t] = pos
        scalars[t] = (credits[pos] / bb,
                      float(snap["pot"]) / bb,
                      max(0.0, bets.max() - bets[pos]) / bb)
        seat_stacks[t, :n] = credits / bb
        if t > 0:
            prev_action[t, decisions[t - 1]["action_idx"]] = 1.0
        member[t] = record.spec.seat_members[pos]
        slot[t] = slot_of_seat[pos]

    for t, dec in enumerate(decisions):
        _fill_decision(t, dec["snap_idx"], int(dec["acting_pos"]))
        action[t] = dec["action_idx"]
        legal[t] = dec["legal_mask"]

    # The moment before an action is chosen: same token, no action yet.
    if pending is not None:
        t = n_dec
        _fill_decision(t, pending.snap_idx, int(pending.acting_pos))
        action[t] = -1
        legal[t] = pending.legal_mask

    # §5.1a — one terminal token per revealed player, after every decision.
    # The revealed cards are the target, so the hole-card slots stay masked;
    # the final board is public once the hand is over.
    last = record.snapshots[-1]
    final_credits = np.asarray(last["credits"], dtype=np.float64)
    for k, pos in enumerate(revealed):
        t = n_dec + k
        cards[t, :5] = [int(c) for c in record.deck[:5]]
        # The observer's own hand is not a thing the observer has to infer, so
        # its slots carry it here exactly as they do on a decision token
        # (owner decision 2026-08-20). One rule for every token type: the
        # observer's own cards always, everybody else's never. Another seat's
        # showdown token stays masked — those cards are the target, and it is
        # the only place the §5.1a anchor gets any gradient into the vector.
        if pos == observer_pos:
            cards[t, 5:] = record.hole_cards(pos)
        acting_pos[t] = pos
        scalars[t] = (final_credits[pos] / bb, float(last["pot"]) / bb, 0.0)
        seat_stacks[t, :n] = final_credits / bb
        if k == 0 and n_dec > 0:
            prev_action[t, decisions[-1]["action_idx"]] = 1.0
        member[t] = record.spec.seat_members[pos]
        slot[t] = slot_of_seat[pos]
        legal[t] = True          # never scored for an action; keeps rows finite
        token_type[t] = TOKEN_SHOWDOWN
        sd_strength[t] = record.showdown_strength[pos]
        sd_class[t] = record.showdown_class[pos]

    # §5.6 — the strength-head target, on the observer's own decision tokens.
    # A hand still in progress has no final board to score, so a tokenisation
    # with a pending decision carries no target at all; the same is true of a
    # record nobody labelled (an evaluation replay, where the unseen cards are
    # filler and a percentile computed from them would be a fiction).
    own = record.hand_strength.get(observer_pos)
    if pending is None and own is not None:
        own_strength[(acting_pos == observer_pos)
                     & (token_type == TOKEN_DECISION)] = own

    return HandTokens(cards, decision_idx, acting_pos, num_players, scalars,
                      seat_stacks, prev_action, member, slot, action, legal,
                      token_type, sd_strength, sd_class, own_strength,
                      own_hole, seat_slot, seat_member, active_opp,
                      range_targets(ranges, active_opp))


def range_targets(ranges, active_opp):
    """`{(token, seat): (idx, w)}` → one flat `RangeTargets`. `None` stays `None`."""
    if ranges is None:
        return None
    keys = sorted(ranges)
    token, seat, combo, weight = [], [], [], []
    for t, s in keys:
        assert active_opp[t, s], (
            f"a range target was produced for seat {s} at token {t}, where "
            f"that seat is not a live opponent — `oracle.ranges` and "
            f"`hand_tokens` disagree about who folded when")
        idx, w = ranges[(t, s)]
        token.append(np.full(len(idx), t, dtype=np.int32))
        seat.append(np.full(len(idx), s, dtype=np.int16))
        combo.append(np.asarray(idx, dtype=np.int16))
        weight.append(np.asarray(w, dtype=np.float32))
    if not keys:
        empty = lambda d: np.zeros(0, dtype=d)
        return RangeTargets(empty(np.int32), empty(np.int16), empty(np.int16),
                            empty(np.float32))
    return RangeTargets(np.concatenate(token), np.concatenate(seat),
                        np.concatenate(combo), np.concatenate(weight))


def empty_batch(B, T, n_actions, max_players):
    """A padded, all-sentinel batch of `B` hands of `T` tokens.

    Split out of `collate` because the padding *values* are load-bearing and
    there must be exactly one statement of them: `oracle/parallel.py` merges
    already-collated batches from several worker processes into one wider batch
    and has to pad the short ones the same way `collate` would have.
    """
    return {
        "cards": torch.full((B, T, 7), UNKNOWN_CARD, dtype=torch.long),
        "decision_idx": torch.zeros((B, T), dtype=torch.long),
        "acting_pos": torch.zeros((B, T), dtype=torch.long),
        "num_players": torch.zeros((B, T), dtype=torch.long),
        "scalars": torch.zeros((B, T, 3), dtype=torch.float32),
        "seat_stacks": torch.zeros((B, T, max_players), dtype=torch.float32),
        "prev_action": torch.zeros((B, T, n_actions), dtype=torch.float32),
        "member": torch.zeros((B, T), dtype=torch.long),
        "slot": torch.zeros((B, T), dtype=torch.long),
        "action": torch.zeros((B, T), dtype=torch.long),
        "legal": torch.zeros((B, T, n_actions), dtype=torch.bool),
        "token_type": torch.zeros((B, T), dtype=torch.long),
        "sd_strength": torch.zeros((B, T), dtype=torch.float32),
        "sd_class": torch.zeros((B, T), dtype=torch.long),
        # §5.6: the pad has to be the sentinel and not zero — zero is a legal
        # percentile, and a padded tail scored as "the worst hand possible"
        # would be a target nobody produced.
        "own_strength": torch.full((B, T), -1.0, dtype=torch.float32),
        # §5.7 — padding with the unknown card leaves the padded tail's combo
        # mask blocking nothing, which is what an all-masked row wants.
        "own_hole": torch.full((B, T, 2), UNKNOWN_CARD, dtype=torch.long),
        # §5.7 — every seat's naming and the seats the belief is about. Seat 0
        # is a safe index for a seat that does not exist at this table; the
        # active mask is what keeps it out of every computation.
        "seat_slot": torch.zeros((B, T, max_players), dtype=torch.long),
        "seat_member": torch.zeros((B, T, max_players), dtype=torch.long),
        "active_opp": torch.zeros((B, T, max_players), dtype=torch.bool),
        "mask": torch.zeros((B, T), dtype=torch.float32),
    }


def collate(hands, device="cpu"):
    """Pad a list of `HandTokens` into one batch of tensors.

    Hands are the batch dimension, not a sequence — that is what the
    block-diagonal attention mask of §5.2 amounts to once cross-hand attention
    is cut, and it is why the corpus can be subsampled per gradient step.
    """
    hands = [h for h in hands if len(h) > 0]
    assert hands, "cannot collate an empty batch of hands"
    B = len(hands)
    T = max(len(h) for h in hands)
    n_actions = hands[0].prev_action.shape[1]
    max_players = hands[0].seat_stacks.shape[1]

    out = empty_batch(B, T, n_actions, max_players)
    for b, h in enumerate(hands):
        t = len(h)
        for name in TOKEN_FIELDS:
            out[name][b, :t] = torch.from_numpy(getattr(h, name))
        out["mask"][b, :t] = 1.0

    out.update(derived_masks(out))
    out.update(range_batch(hands, out))
    return {k: v.to(device) for k, v in out.items()}


def range_batch(hands, out):
    """The §5.7 target, densified onto the batch's active (hand, token, seat) rows.

    Sparse on disk and in the corpus, dense here: `act_idx` is at most
    `B · T · max_players` rows and in practice a few hundred, so a
    `(A, 1326)` block is megabytes where a `(B, T, max_players, 1326)` one would
    be gigabytes. The rows line up with `act_idx` by construction, so the head
    and the loss index the same way and no third alignment has to be trusted.

    Returns nothing at all when no hand in the batch carries a target — an
    inference batch, an evaluation replay, or a corpus built with §5.7 off.
    """
    if all(h.ranges is None for h in hands):
        return {}
    act = out["act_idx"]
    A = act.shape[0]
    row_of = {(int(b), int(t), int(s)): i
              for i, (b, t, s) in enumerate(act.tolist())}
    target = torch.zeros((A, N_COMBOS), dtype=torch.float32)
    has = torch.zeros(A, dtype=torch.bool)
    for b, h in enumerate(hands):
        if h.ranges is None:
            continue
        rows = np.fromiter(
            (row_of[(b, int(t), int(s))]
             for t, s in zip(h.ranges.token, h.ranges.seat)),
            dtype=np.int64, count=len(h.ranges.token))
        target[torch.from_numpy(rows),
               torch.from_numpy(h.ranges.combo.astype(np.int64))] = \
            torch.from_numpy(h.ranges.weight)
        has[torch.from_numpy(np.unique(rows))] = True
    return {"range_target": target, "range_mask": has}


def derived_masks(batch):
    """The three masks every consumer of a batch needs, derived once.

    An action target only exists on decision tokens, a revealed hand only on
    showdown tokens, and the §5.6 strength target only where `hand_tokens` left
    a percentile instead of the sentinel. `oracle/transport.py` does not carry
    them between processes — it rebuilds them here, so there is one statement of
    what they are.
    """
    decision_mask = batch["mask"] * (batch["token_type"] == TOKEN_DECISION)
    # §5.7 — the (hand, token, seat) triples the range head answers about, in
    # row-major order. It is derived and not carried, for the same reason the
    # three masks are: it is a function of `active_opp` and `mask`, and one
    # statement of that rule is the point.
    active = batch["active_opp"] & (decision_mask > 0).unsqueeze(-1)
    return {
        "decision_mask": decision_mask,
        "showdown_mask": batch["mask"] * (batch["token_type"] == TOKEN_SHOWDOWN),
        "strength_mask": batch["mask"] * (batch["own_strength"] >= 0),
        "active": active,
        "act_idx": active.nonzero(),
    }
