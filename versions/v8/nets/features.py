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

The token's `member` and `slot` fields are the two ways a player is named. At
training a player *is* a pool member and its vector is a row of the trainable
table (§5.4); at inference a player is a seat at the observed table and its
vector is one of the free parameters being fitted (§5.5). Both indices are
carried so one token batch serves both.
"""

from dataclasses import dataclass

import numpy as np
import torch

UNKNOWN_CARD = 52
TOKEN_DECISION = 0
TOKEN_SHOWDOWN = 1
N_TOKEN_TYPES = 2


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

    def __len__(self):
        return len(self.decision_idx)


def _board_as_of(deck, turn):
    if turn == 0:
        return [UNKNOWN_CARD] * 5
    if turn == 1:
        return [int(c) for c in deck[:3]] + [UNKNOWN_CARD] * 2
    if turn == 2:
        return [int(c) for c in deck[:4]] + [UNKNOWN_CARD]
    return [int(c) for c in deck[:5]]


def hand_tokens(record, observer_pos, slot_of_seat, max_players, n_actions,
                pending=None):
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
                      token_type, sd_strength, sd_class, own_strength)


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
        out["token_type"][b, :t] = torch.from_numpy(h.token_type)
        out["sd_strength"][b, :t] = torch.from_numpy(h.sd_strength)
        out["sd_class"][b, :t] = torch.from_numpy(h.sd_class)
        out["own_strength"][b, :t] = torch.from_numpy(h.own_strength)
        out["cards"][b, :t] = torch.from_numpy(h.cards)
        out["decision_idx"][b, :t] = torch.from_numpy(h.decision_idx)
        out["acting_pos"][b, :t] = torch.from_numpy(h.acting_pos)
        out["num_players"][b, :t] = torch.from_numpy(h.num_players)
        out["scalars"][b, :t] = torch.from_numpy(h.scalars)
        out["seat_stacks"][b, :t] = torch.from_numpy(h.seat_stacks)
        out["prev_action"][b, :t] = torch.from_numpy(h.prev_action)
        out["member"][b, :t] = torch.from_numpy(h.member)
        out["slot"][b, :t] = torch.from_numpy(h.slot)
        out["action"][b, :t] = torch.from_numpy(h.action)
        out["legal"][b, :t] = torch.from_numpy(h.legal)
        out["mask"][b, :t] = 1.0

    out.update(derived_masks(out))
    return {k: v.to(device) for k, v in out.items()}


def derived_masks(batch):
    """The three masks every consumer of a batch needs, derived once.

    An action target only exists on decision tokens, a revealed hand only on
    showdown tokens, and the §5.6 strength target only where `hand_tokens` left
    a percentile instead of the sentinel. `oracle/transport.py` does not carry
    them between processes — it rebuilds them here, so there is one statement of
    what they are.
    """
    return {
        "decision_mask": batch["mask"] * (batch["token_type"] == TOKEN_DECISION),
        "showdown_mask": batch["mask"] * (batch["token_type"] == TOKEN_SHOWDOWN),
        "strength_mask": batch["mask"] * (batch["own_strength"] >= 0),
    }
