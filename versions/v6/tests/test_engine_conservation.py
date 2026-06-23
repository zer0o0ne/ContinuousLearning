"""Chip-conservation tests for the poker engine (env/table.py).

Verifies Этап-0 fix 0.1: at showdown the engine must pay out the entire pot
(including chips bet on earlier streets), so chips are conserved across every
hand:  sum(credits_after) == sum(credits_before).

Covers: random multi-street play, heads-up all-in runouts (the original bug
case), multi-way all-in side pots, and split pots. Run:

    source venv/bin/activate
    python -m tests.test_engine_conservation        # from versions/v6
"""

import numpy as np
import torch

from env.table import Table

BIG_BLIND = 10
SMALL_BLIND = 5
# 4 streets, 11 raise bins each (mirrors config.json shape); n_actions = 14.
RAISE_SIZES = [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]] * 4
N_ACTIONS = len(RAISE_SIZES[0]) + 3
TOL = 1e-6


def _onehot(idx):
    a = torch.zeros(N_ACTIONS, dtype=torch.float32)
    a[idx] = 1.0
    return a


def _new_table(credits, deck=None):
    table = Table(
        num_players=len(credits),
        raise_sizes=RAISE_SIZES,
        start_credits=int(max(credits)),
        big_blind=BIG_BLIND,
        small_blind=SMALL_BLIND,
    )
    table.credits = [float(c) for c in credits]
    table.start_table()
    if deck is not None:
        table.deck = np.asarray(deck)
    return table


def _run_to_end(table, action_fn, max_steps=2000):
    """Step the table until the hand ends. action_fn(table) -> one-hot tensor.
    During an all-in runout the table ignores the action, so a dummy is fine."""
    end = False
    steps = 0
    while not end and steps < max_steps:
        if table.several_all_in:
            action = _onehot(0)
        else:
            action = action_fn(table)
        end, _several_all_in, _state, _bet = table.step(action)
        steps += 1
    assert end, "hand did not terminate within max_steps"
    return table


def _assert_conserved(before, table, label):
    after = sum(table.credits)
    assert abs(after - before) < TOL, (
        f"{label}: chip leak {before - after:+.6f} "
        f"(before={before}, after={after}, credits={table.credits})"
    )
    assert all(c >= -TOL for c in table.credits), (
        f"{label}: negative credit produced: {table.credits}"
    )


def test_random_play():
    """Many random hands, 2-6 players, asymmetric stacks → conservation."""
    rng = np.random.RandomState(12345)
    n_hands = 4000
    for h in range(n_hands):
        num_players = int(rng.randint(2, 7))
        credits = [int(rng.randint(BIG_BLIND, 600)) for _ in range(num_players)]
        before = sum(credits)
        table = _new_table(credits)

        def random_action(_table):
            return _onehot(int(rng.randint(0, N_ACTIONS)))

        _run_to_end(table, random_action)
        _assert_conserved(before, table, f"random hand {h} ({num_players}p)")
    print(f"test_random_play: OK ({n_hands} hands)")


def test_hu_allin_showdown():
    """The original bug: HU all-in preflop 500/500 reaching showdown.

    Rig the deck so seat 0 (pocket aces) beats seat 1; verify conservation AND
    that the winner actually collects the pot (the bug paid out (0, 0))."""
    # Ragged board (no flush/straight): ranks 0,2,5,8,10 with mixed suits.
    # seat0 = AA (48,49), seat1 = KK (44,45) -> seat0 wins outright.
    deck = [0, 9, 22, 35, 40, 48, 49, 44, 45]
    before = 1000.0
    table = _new_table([500, 500], deck=deck)
    _run_to_end(table, lambda _t: _onehot(N_ACTIONS - 1))  # both shove
    _assert_conserved(before, table, "HU all-in showdown")
    assert table.credits[0] > table.credits[1], (
        f"winner (seat 0, AA) should collect: credits={table.credits}"
    )
    assert abs(table.credits[0] - 1000.0) < TOL, (
        f"winner should hold the whole 1000 pot: {table.credits}"
    )
    print(f"test_hu_allin_showdown: OK (credits={table.credits})")


def test_multiway_allin_sidepot():
    """3-way all-in with asymmetric stacks → side pots; conservation holds."""
    # Short stack (seat 0) has the best hand and is covered → wins main pot only.
    # Ragged board; seat0 = AA, seat1 = KK, seat2 = JJ.
    deck = [0, 9, 22, 35, 40, 48, 49, 44, 45, 36, 37]
    credits = [100, 500, 500]
    before = float(sum(credits))
    table = _new_table(credits, deck=deck)
    _run_to_end(table, lambda _t: _onehot(N_ACTIONS - 1))  # everyone shoves
    _assert_conserved(before, table, "3-way all-in side pot")
    # Main pot (300 = 100*3) → seat0 (AA, covered short stack). Side pot
    # (800 = 400+400 from seat1/seat2) → seat1 (KK) over seat2 (JJ).
    assert table.credits[0] == 300.0, f"short stack main pot: {table.credits}"
    assert table.credits[1] == 800.0, f"side pot to KK: {table.credits}"
    assert table.credits[2] == 0.0, f"JJ busts: {table.credits}"
    print(f"test_multiway_allin_sidepot: OK (credits={table.credits})")


def test_split_pot():
    """HU all-in with identical hands → split; conservation + equal payout."""
    # Ragged board; both seats hold pocket aces (different suits) → genuine tie.
    deck = [0, 9, 22, 35, 40, 48, 49, 50, 51]
    before = 1000.0
    table = _new_table([500, 500], deck=deck)
    _run_to_end(table, lambda _t: _onehot(N_ACTIONS - 1))
    _assert_conserved(before, table, "HU split pot")
    assert abs(table.credits[0] - 500.0) < TOL and abs(table.credits[1] - 500.0) < TOL, (
        f"split should return 500 each: {table.credits}"
    )
    print(f"test_split_pot: OK (credits={table.credits})")


def test_fold_win():
    """All-but-one fold preflop → fold-win path; conservation holds."""
    rng = np.random.RandomState(7)
    for h in range(500):
        num_players = int(rng.randint(2, 7))
        credits = [int(rng.randint(BIG_BLIND, 600)) for _ in range(num_players)]
        before = sum(credits)
        table = _new_table(credits)
        # Everyone folds except via blinds defense: force folds.
        _run_to_end(table, lambda _t: _onehot(0))
        _assert_conserved(before, table, f"fold-win hand {h}")
    print("test_fold_win: OK (500 hands)")


if __name__ == "__main__":
    test_random_play()
    test_hu_allin_showdown()
    test_multiway_allin_sidepot()
    test_split_pot()
    test_fold_win()
    print("\nALL CHIP-CONSERVATION TESTS PASSED")
