"""Shared fixtures for the G1 tests.

Deliberately tiny: a 3-bin action set and a handful of degenerate members with
fixed style draws. `CLAUDE.md` §4 caps the whole battery at 30 minutes on a
CPU-only dev box, and none of the properties under test need scale.
"""

import numpy as np

from env.driver import HandSpec, LockstepDriver
from env.showdown import label_showdowns
from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import StyleParams, sample_style

RAISE_SIZES = [[0.5, 1.0, 2.0]] * 4
N_ACTIONS = len(RAISE_SIZES[0]) + 3
MAX_PLAYERS = 9
BIG_BLIND = 10.0
SMALL_BLIND = 5.0

STYLE_CFG = {
    "uncond_scale": 0.8,
    "position_scale": 0.5,
    "street_scale": 0.5,
    "log_temperature_range": [-0.4, 0.4],
    "uniform_mix_range": [0.0, 0.2],
}

NET_CFG = {
    "d_model": 48, "d_emb": 8, "n_heads": 4, "n_kv_heads": 2,
    "n_layers": 2, "d_ff": 96, "d_card": 8, "d_index": 8,
    "max_decisions": 80,
}

# §5.1a — both showdown heads on, so the tests exercise the real objective.
LOSS_WEIGHTS = {"amortised": 1.0, "showdown_strength": 0.3,
                "showdown_class": 0.1}


def make_pool(seed=0, n_styled=6):
    """Identity-style degenerates plus a few style draws. At least 9 members."""
    rng = np.random.default_rng(seed)
    members = [cls(name, N_ACTIONS, StyleParams.identity())
               for name, cls in DEGENERATE_STRATEGIES.items()]
    names = list(DEGENERATE_STRATEGIES)
    for i in range(n_styled):
        name = names[i % len(names)]
        members.append(DEGENERATE_STRATEGIES[name](
            f"{name}#{i}", N_ACTIONS, sample_style(rng, STYLE_CFG)))
    return members


def make_specs(seed=0, n_hands=16, n_members=1, num_players=None,
               stack_bb=None, raise_sizes=None):
    """Hands with uniformly sampled table size and stack depth.

    `raise_sizes` overrides the fixture grid for the tests that need the table
    to run on a grid other than the one a v7 checkpoint was trained on.
    """
    rng = np.random.default_rng(seed + 1)
    specs = []
    for h in range(n_hands):
        n = num_players if num_players else int(rng.integers(2, 10))
        s = stack_bb if stack_bb else int(rng.integers(10, 301))
        specs.append(HandSpec(
            num_players=n,
            start_credits=[float(s * BIG_BLIND)] * n,
            seat_members=[int(rng.integers(0, n_members)) for _ in range(n)],
            seed=90_000 + seed * 1000 + h,
            big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
            raise_sizes=raise_sizes if raise_sizes else RAISE_SIZES,
            meta={"hand": h},
        ))
    return specs


def play(pool, specs, batch_size=None, n_actions=N_ACTIONS):
    """Play the specs and label the showdowns, as the corpus builder does."""
    records = LockstepDriver(pool, n_actions).run(specs, batch_size=batch_size)
    label_showdowns(records)
    return records


def contexts_from(records):
    """Re-materialise the `DecisionContext` of every recorded decision.

    Lets a test hand real, varied situations to a pool member without having to
    hand-build a table state.
    """
    from env.driver import DecisionContext
    out = []
    for record in records:
        for dec in record.decisions:
            snap = record.snapshots[dec["snap_idx"]]
            out.append(DecisionContext(record, dec["snap_idx"],
                                       dec["acting_pos"], dec["legal_mask"],
                                       int(snap["turn"])))
    return out


def session_specs(members, num_players, n_hands, stack_bb=100, seed=0):
    """One rotating-button session: seat s in hand h holds slot (s + h) % n."""
    specs = []
    for h in range(n_hands):
        specs.append(HandSpec(
            num_players=num_players,
            start_credits=[float(stack_bb * BIG_BLIND)] * num_players,
            seat_members=[members[(s + h) % num_players]
                          for s in range(num_players)],
            seed=70_000 + seed * 1000 + h,
            big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
            raise_sizes=RAISE_SIZES, meta={"hand": h},
        ))
    return specs


def slot_of_seat(num_players, hand_idx):
    return [(seat + hand_idx) % num_players for seat in range(num_players)]


def observer_seat(num_players, hand_idx, slot=0):
    return (slot - hand_idx) % num_players
