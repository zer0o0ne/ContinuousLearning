"""Observation parity — the fatal-if-wrong invariant (CONCEPT.md §9, §15).

    Every observation the embedding network or the agent ever sees, at training
    or at inference, is constructible from what the observer could have known at
    that moment.

If this breaks, nothing else in v8 matters: a network trained on cards it will
not have at deployment learns to predict actions from cards, the fitted vector
carries nothing, and the input distribution at inference is one the network
never saw. The failure is completely silent — the prediction loss looks fine.

Asserted end-to-end on real played hands, against the snapshots the tokens were
built from, not against a re-implementation of the tokeniser.
"""

import numpy as np
import pytest

from env.showdown import hand_class_169
from nets.features import (
    TOKEN_DECISION, TOKEN_SHOWDOWN, UNKNOWN_CARD, collate, hand_tokens,
)
from tests.g1_fixtures import MAX_PLAYERS, N_ACTIONS, make_pool, make_specs, play


def _records():
    pool = make_pool()
    return play(pool, make_specs(seed=11, n_hands=40, n_members=len(pool)))


def _tokens_for(record, observer):
    return hand_tokens(record, observer,
                       slot_of_seat=list(range(record.num_players)),
                       max_players=MAX_PLAYERS, n_actions=N_ACTIONS)


def test_only_the_observers_hole_cards_are_ever_present():
    """One rule, every token type — decision tokens and showdown tokens alike.

    The observer's own hand is never something the observer has to infer, so it
    is present wherever the observer is the acting seat; nobody else's ever is
    (owner decision 2026-08-20). The showdown tokens are deliberately *not*
    skipped here — that they obey the same rule is the property.
    """
    checked_masked = checked_own = checked_showdown = 0
    for record in _records():
        for observer in range(record.num_players):
            tok = _tokens_for(record, observer)
            own = record.hole_cards(observer)
            for t in range(len(tok)):
                acting = int(tok.acting_pos[t])
                slots = tok.cards[t, 5:].tolist()
                if acting == observer:
                    assert slots == own
                    checked_own += 1
                    checked_showdown += int(
                        tok.token_type[t] == TOKEN_SHOWDOWN)
                else:
                    assert slots == [UNKNOWN_CARD, UNKNOWN_CARD], (
                        f"seat {acting}'s hole cards leaked into a token built "
                        f"for observer {observer}")
                    checked_masked += 1
    assert checked_masked > 0 and checked_own > 0
    assert checked_showdown > 0, (
        "no observer reached a showdown, so the showdown half is untested")


def test_no_other_seats_cards_appear_anywhere_in_a_decision_token():
    """Not just the hole-card slots — nowhere in the token at all."""
    for record in _records():
        for observer in range(record.num_players):
            tok = _tokens_for(record, observer)
            tok.cards[tok.token_type != TOKEN_DECISION] = UNKNOWN_CARD
            forbidden = set()
            for pos in range(record.num_players):
                if pos != observer:
                    forbidden.update(record.hole_cards(pos))
            forbidden -= set(int(c) for c in record.deck[:5])  # board is public
            forbidden -= set(record.hole_cards(observer))
            present = set(int(c) for c in tok.cards.reshape(-1))
            assert not (present & forbidden), (
                f"cards {sorted(present & forbidden)} belong to another seat")


def test_the_board_is_never_ahead_of_a_decisions_street():
    for record in _records():
        tok = _tokens_for(record, 0)
        for t, dec in enumerate(record.decisions):
            turn = int(record.snapshots[dec["snap_idx"]]["turn"])
            revealed = int((tok.cards[t, :5] != UNKNOWN_CARD).sum())
            expected = {0: 0, 1: 3, 2: 4, 3: 5}[turn]
            assert revealed == expected, (
                f"street {turn} token shows {revealed} board cards")
            assert tok.cards[t, :expected].tolist() == \
                [int(c) for c in record.deck[:expected]]


def test_a_token_never_carries_its_own_action():
    """`prev_action` is the action at the *previous* token. A token that knew
    its own answer would make the whole prediction task trivial."""
    for record in _records():
        tok = _tokens_for(record, 0)
        assert not tok.prev_action[0].any(), (
            "the first decision of a hand has no previous action")
        for t in range(1, len(record.decisions)):
            assert int(np.argmax(tok.prev_action[t])) == \
                record.decisions[t - 1]["action_idx"]
            assert tok.prev_action[t].sum() == 1.0


def test_scalars_come_from_the_pre_decision_snapshot():
    """No post-decision information: pot, stacks and the amount to call are the
    state the player faced, not the state after acting."""
    for record in _records():
        tok = _tokens_for(record, 0)
        bb = record.spec.big_blind
        for t, dec in enumerate(record.decisions):
            snap = record.snapshots[dec["snap_idx"]]
            pos = int(dec["acting_pos"])
            bets = np.asarray(snap["bets"], dtype=float)
            credits = np.asarray(snap["credits"], dtype=float)
            assert tok.scalars[t] == pytest.approx(
                [credits[pos] / bb, snap["pot"] / bb,
                 max(0.0, bets.max() - bets[pos]) / bb], abs=1e-5)
            assert tok.seat_stacks[t, :record.num_players] == pytest.approx(
                credits / bb, abs=1e-5)
            assert not tok.seat_stacks[t, record.num_players:].any()


def test_tokens_of_a_prefix_do_not_depend_on_what_happened_later():
    """Truncating a hand's decisions leaves the surviving tokens untouched."""
    for record in _records():
        if len(record.decisions) < 3:
            continue
        full = _tokens_for(record, 0)
        cut = len(record.decisions) - 1
        record.decisions, tail = record.decisions[:cut], record.decisions[cut:]
        try:
            prefix = _tokens_for(record, 0)
        finally:
            record.decisions = record.decisions + tail
        assert np.array_equal(prefix.cards[:cut], full.cards[:cut])
        assert np.array_equal(prefix.scalars[:cut], full.scalars[:cut])
        assert np.array_equal(prefix.prev_action[:cut], full.prev_action[:cut])
        assert np.array_equal(prefix.action[:cut], full.action[:cut])


def test_an_observer_who_was_not_seated_is_refused():
    """§5.4: a player's embedding is fitted only from hands the observer sat
    in. Building an observation for an absent observer is a programming error,
    not something to paper over."""
    record = _records()[0]
    with pytest.raises(AssertionError):
        _tokens_for(record, record.num_players)


# ------------------------------------------------------ showdown tokens (§5.1a)


def test_showdown_tokens_appear_exactly_for_the_revealed_seats():
    seen_showdown = seen_foldout = False
    for record in _records():
        tok = _tokens_for(record, 0)
        n_dec = len(record.decisions)
        assert (tok.token_type[:n_dec] == TOKEN_DECISION).all(), (
            "showdown tokens must come after every decision, never among them")
        assert (tok.token_type[n_dec:] == TOKEN_SHOWDOWN).all()
        assert tok.acting_pos[n_dec:].tolist() == sorted(record.showdown)
        if record.showdown:
            seen_showdown = True
            assert len(record.showdown) >= 2, "a showdown needs two players"
        else:
            seen_foldout = True
            assert len(tok) == n_dec
    assert seen_showdown and seen_foldout


def test_another_seats_revealed_cards_are_the_target_and_never_an_input():
    """The whole point of the terminal token, for every seat but the observer.

    A showdown token is the only channel from a reveal to that player's vector,
    and it is a channel only while the answer is absent from the input. So
    another seat's hand must not be readable from its token — not in the hole
    slots and not anywhere else in it. The observer's own token is the declared
    exception and is checked by the test below.
    """
    checked = 0
    for record in _records():
        observer = 0
        tok = _tokens_for(record, observer)
        for t in range(len(record.decisions), len(tok)):
            pos = int(tok.acting_pos[t])
            if pos == observer:
                continue
            assert tok.cards[t, 5:].tolist() == [UNKNOWN_CARD, UNKNOWN_CARD]
            revealed = set(record.hole_cards(pos))
            present = set(int(c) for c in tok.cards[t])
            board = set(int(c) for c in record.deck[:5])
            assert not ((present & revealed) - board)
            checked += 1
    assert checked > 0


def test_the_observers_own_showdown_token_carries_the_observers_own_hand():
    """The 2026-08-20 exception, and the reason it is not one of substance.

    The observer knows its own cards at every moment of the hand, so a token
    that hid them from the observer alone would be hiding public information
    from the only player entitled to it. Its strength target then becomes
    readable from its own input — which is a fact about the *metric*
    (`OpponentEmbeddingNet.showdown_losses`) and not about §9.
    """
    checked = 0
    for record in _records():
        for observer in range(record.num_players):
            if observer not in record.showdown:
                continue
            tok = _tokens_for(record, observer)
            rows = [t for t in range(len(record.decisions), len(tok))
                    if int(tok.acting_pos[t]) == observer]
            assert len(rows) == 1, "one showdown token per revealed seat"
            assert tok.cards[rows[0], 5:].tolist() == \
                record.hole_cards(observer)
            checked += 1
    assert checked > 0


def test_a_showdown_token_shows_the_full_final_board():
    for record in _records():
        if not record.showdown:
            continue
        tok = _tokens_for(record, 0)
        for t in range(len(record.decisions), len(tok)):
            assert tok.cards[t, :5].tolist() == \
                [int(c) for c in record.deck[:5]]


def test_the_showdown_targets_match_the_cards_that_were_shown():
    checked = 0
    for record in _records():
        tok = _tokens_for(record, 0)
        for t in range(len(record.decisions), len(tok)):
            pos = int(tok.acting_pos[t])
            a, b = record.hole_cards(pos)
            assert int(tok.sd_class[t]) == hand_class_169(a, b)
            assert 0.0 <= float(tok.sd_strength[t]) <= 1.0
            assert float(tok.sd_strength[t]) == \
                pytest.approx(record.showdown_strength[pos])
            checked += 1
    assert checked > 0


def test_the_two_masks_partition_the_real_tokens():
    records = _records()
    batch = collate([_tokens_for(r, 0) for r in records])
    total = batch["decision_mask"] + batch["showdown_mask"]
    assert (total == batch["mask"]).all()
    assert (batch["decision_mask"] * batch["showdown_mask"]).sum() == 0
    assert batch["showdown_mask"].sum() > 0


def test_a_hand_reaching_showdown_but_missing_its_labels_is_refused():
    """Silently dropping the terminal token would make the showdown anchor
    disappear from a corpus with no error at all."""
    record = next(r for r in _records() if r.showdown)
    record.showdown_strength = {}
    with pytest.raises(AssertionError, match="label_showdowns"):
        _tokens_for(record, 0)


def test_every_token_carries_the_table_size_and_the_acting_seat():
    """§5.1 / OI-1: the 2–9 axis is not recoverable from a token otherwise."""
    for record in _records():
        tok = _tokens_for(record, 0)
        assert (tok.num_players == record.num_players).all()
        for t, dec in enumerate(record.decisions):
            assert int(tok.acting_pos[t]) == dec["acting_pos"]
            assert int(tok.decision_idx[t]) == t
