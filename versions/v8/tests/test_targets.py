"""Agent targets, the KL loss and the training cycle (CONCEPT.md §6.2, §8).

The assertion this file exists for is `test_the_target_is_invariant_to_the_scale
_of_the_situation`: it is the regression test for v7's 97 % fold rate, which was
a normalisation bug and not an architecture failure. Everything else here is the
supporting cast — exact zeros off the mask, the two temperature limits, a loss
that reads as a distance, a dropout that is per hand and not per token, and the
cycle contract: the first iteration gets its own step count and every later one
continues the network the previous one produced.
"""

import copy
import math

import numpy as np
import pytest
import torch

from tests.g1_fixtures import MAX_PLAYERS, N_ACTIONS, NET_CFG, make_pool
from tests.test_agent_net import _agent_at_seat_zero, _net
from train.agent_train import (
    embedding_dropout, steps_for_iteration, token_embeddings, train_agent,
)
from train.targets import kl_loss, policy_target


def _legal(*idx):
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for i in idx:
        mask[i] = True
    return mask


def _q(values):
    """EVs with `nan` everywhere the caller did not name an action."""
    q = np.full(N_ACTIONS, np.nan)
    for i, v in values.items():
        q[i] = v
    return q


# ------------------------------------------------------------------ 1: the shape of the target


def test_the_target_matches_the_softmax_computed_by_hand():
    """Hand-computed, through `math.exp` rather than through this module."""
    legal = _legal(0, 1, 2)
    q = _q({0: 0.0, 1: 2.0, 2: -1.0})
    got = policy_target(q, legal, pot_bb=3.0, facing_bet_bb=1.0,
                        temperature=0.5)

    scale = 3.0 + 1.0
    raw = [math.exp((v / scale) / 0.5) for v in (0.0, 2.0, -1.0)]
    total = sum(raw)
    for a, expected in enumerate(r / total for r in raw):
        assert got[a] == pytest.approx(expected, abs=1e-12)
    assert (got[3:] == 0.0).all()


def test_illegal_actions_get_exact_zero_and_the_legal_mass_is_one():
    legal = _legal(0, 1, 4, 5)
    q = _q({0: 1.5, 1: -3.0, 4: 0.25, 5: 7.0})
    t = policy_target(q, legal, pot_bb=12.0, facing_bet_bb=4.0,
                      temperature=0.3)

    # Exact, not "small": an illegal action must be unplayable, not unlikely.
    assert (t[~legal] == 0.0).all()
    assert (t[legal] > 0.0).all()
    assert abs(t.sum() - 1.0) <= 1e-15


def test_what_sits_under_the_mask_is_never_read():
    """`nan` at illegal actions is a convention, not an input."""
    legal = _legal(1, 2, 3)
    q = _q({1: 0.4, 2: -0.1, 3: 1.2})
    poisoned = np.where(legal, q, 1e9)
    a = policy_target(q, legal, 6.0, 2.0, 0.7)
    b = policy_target(poisoned, legal, 6.0, 2.0, 0.7)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ 3: temperature


def test_a_cold_temperature_is_the_argmax_and_a_hot_one_is_uniform():
    legal = _legal(0, 1, 2, 4)
    q = _q({0: 0.0, 1: 2.0, 2: -1.0, 4: 0.5})

    cold = policy_target(q, legal, 3.0, 1.0, temperature=1e-300)
    assert np.array_equal(cold, np.eye(N_ACTIONS)[1])

    hot = policy_target(q, legal, 3.0, 1.0, temperature=1e300)
    assert np.array_equal(hot, legal.astype(np.float64) / 4.0)


def test_a_temperature_that_is_not_positive_and_finite_is_refused():
    legal = _legal(0, 1)
    q = _q({0: 1.0, 1: 2.0})
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(AssertionError, match="temperature"):
            policy_target(q, legal, 3.0, 1.0, temperature=bad)


# ------------------------------------------------------------------ 4: degenerate inputs


def test_one_legal_action_is_a_one_hot_at_any_temperature():
    legal = _legal(4)
    q = _q({4: -42.0})
    for T in (1e-300, 0.1, 1.0, 1e300):
        assert np.array_equal(policy_target(q, legal, 5.0, 0.0, T),
                              np.eye(N_ACTIONS)[4])


def test_equal_evs_are_uniform_over_the_legal_actions():
    legal = _legal(0, 2, 5)
    q = _q({0: 1.25, 2: 1.25, 5: 1.25})
    t = policy_target(q, legal, 8.0, 3.0, temperature=0.4)
    assert np.array_equal(t, legal.astype(np.float64) / 3.0)


def test_a_decision_with_no_legal_action_is_refused():
    with pytest.raises(AssertionError, match="no legal action"):
        policy_target(np.full(N_ACTIONS, np.nan), np.zeros(N_ACTIONS, bool),
                      3.0, 1.0, 0.5)


def test_a_nan_under_the_mask_is_refused():
    legal = _legal(0, 1)
    with pytest.raises(AssertionError, match="finite EV"):
        policy_target(_q({0: 1.0}), legal, 3.0, 1.0, 0.5)


# ------------------------------------------------------------------ 5: the v7 scar


def test_the_target_is_invariant_to_the_scale_of_the_situation():
    """The regression test for v7's 97 % fold rate.

    Two situations identical up to a factor of 30 on the pot, the bet faced and
    every EV. Under `pot + facing_bet` normalisation they are the same decision
    and must produce the same target; under raw EVs the second would be
    near-deterministic and the first near-uniform, which is exactly the failure
    `versions/v7/ARCHITECTURE.md` records.
    """
    legal = _legal(0, 1, 2, 3, 5)
    evs = {0: 0.0, 1: 0.6, 2: -0.4, 3: 0.15, 5: -1.1}
    small = _q(evs)
    big = _q({a: 30.0 * v for a, v in evs.items()})

    a = policy_target(small, legal, pot_bb=2.5, facing_bet_bb=1.0,
                      temperature=0.25)
    b = policy_target(big, legal, pot_bb=75.0, facing_bet_bb=30.0,
                      temperature=0.25)
    assert np.abs(a - b).max() <= 1e-12

    # And the same numbers with the normalisation removed are the v7 failure
    # itself: one temperature over raw EVs is near-uniform in the small pot and
    # near-deterministic in the large one.
    flat_a = policy_target(small, legal, 1.0, 0.0, 0.25)
    flat_b = policy_target(big, legal, 1.0, 0.0, 0.25)
    assert flat_a.max() < 0.9 and flat_b.max() > 0.99


def test_the_divisor_is_a_choice_and_an_unknown_one_is_refused():
    legal = _legal(0, 1, 2)
    q = _q({0: 0.0, 1: 1.0, 2: -1.0})
    pot_only = policy_target(q, legal, 4.0, 4.0, 0.5, divisor="pot")
    both = policy_target(q, legal, 4.0, 4.0, 0.5, divisor="pot_plus_bet")
    # Halving the divisor sharpens the same ordering.
    assert np.argmax(pot_only) == np.argmax(both) == 1
    assert pot_only[1] > both[1]

    with pytest.raises(AssertionError, match="unknown divisor"):
        policy_target(q, legal, 4.0, 4.0, 0.5, divisor="stack")


# ------------------------------------------------------------------ 6: the loss


def _logits_and_target(dtype=torch.float64):
    legal = torch.tensor([[True, True, True, True, False, False]])
    logits = torch.zeros((1, 6), dtype=dtype)
    target = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0]], dtype=dtype)
    return logits, target, legal


def test_the_loss_is_exactly_zero_when_the_prediction_is_the_target():
    logits, target, legal = _logits_and_target()
    assert float(kl_loss(logits, target, legal)) == 0.0


def test_the_loss_is_positive_as_soon_as_the_prediction_moves():
    logits, target, legal = _logits_and_target()
    for eps in (1e-3, 0.1, 5.0):
        moved = logits.clone()
        moved[0, 0] += eps
        assert float(kl_loss(moved, target, legal)) > 0.0


def test_the_loss_ignores_the_logits_of_illegal_actions():
    logits, target, legal = _logits_and_target()
    poisoned = logits.clone()
    poisoned[0, 4:] = 100.0
    assert float(kl_loss(poisoned, target, legal)) == float(
        kl_loss(logits, target, legal))


def test_the_gradient_reaches_the_logits_and_not_the_target():
    logits, target, legal = _logits_and_target()
    logits = logits.clone().requires_grad_(True)
    target = target.clone().requires_grad_(True)
    moved = logits + torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                  dtype=logits.dtype)
    kl_loss(moved, target, legal).backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().max()) > 0.0
    # Illegal actions are not part of the computation at all.
    assert float(logits.grad[0, 4:].abs().max()) == 0.0
    assert target.grad is None


def test_a_target_that_disagrees_with_the_mask_is_refused():
    logits, target, legal = _logits_and_target()
    bad = target.clone()
    bad[0, 4] = 0.1
    with pytest.raises(AssertionError, match="illegal action"):
        kl_loss(logits, bad, legal)


# ------------------------------------------------------------------ 7: embedding dropout


def _tables(n_hands=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n_hands, MAX_PLAYERS, NET_CFG["d_emb"]), generator=g) + 1.0


def test_dropout_at_zero_leaves_every_vector_intact():
    tables = _tables()
    g = torch.Generator().manual_seed(3)
    assert torch.equal(embedding_dropout(tables, 0.0, generator=g), tables)


def test_dropout_at_one_zeroes_every_vector():
    tables = _tables()
    g = torch.Generator().manual_seed(3)
    assert not embedding_dropout(tables, 1.0, generator=g).any()


def test_dropout_is_reproducible_from_its_generator():
    tables = _tables()
    a = embedding_dropout(tables, 0.5, torch.Generator().manual_seed(11))
    b = embedding_dropout(tables, 0.5, torch.Generator().manual_seed(11))
    c = embedding_dropout(tables, 0.5, torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_dropout_is_per_slot_per_hand_and_not_per_token():
    """A slot the agent is blind to is blind for the whole hand."""
    tables = _tables(n_hands=8, seed=1)
    dropped = embedding_dropout(tables, 0.5, torch.Generator().manual_seed(5))

    # Every hand reads all nine slots, several times each.
    slot = torch.arange(MAX_PLAYERS).repeat(3).unsqueeze(0).expand(8, -1)
    tokens = token_embeddings(dropped, slot, NET_CFG["d_emb"])

    alive = (dropped.abs().sum(-1) > 0)
    assert alive.any() and not alive.all(), "the draw has to bite both ways"
    for b in range(8):
        for s in range(MAX_PLAYERS):
            rows = tokens[b, slot[b] == s]
            expected = tables[b, s] if alive[b, s] else torch.zeros_like(tables[b, s])
            assert torch.equal(rows, expected.expand_as(rows))
        # ... and the same slot is not decided once for the whole batch.
    assert not torch.equal(alive[0], alive.all(dim=0))


# ------------------------------------------------------------------ 8: a toy training run


TRAIN_CFG = {
    "steps": 12,
    "first_iteration_steps": 40,
    "batch_hands": 8,
    "lr": 3e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "embedding_dropout": 0.0,
    "log_every": 1000,
}


def _labels(num_players=3, n_hands=10, seed=1):
    """Real observations from a played session, with synthetic oracle EVs.

    The hands are the agent's own — collected through the driver, so they carry
    the pending token of a decision it actually faced (§9) — and only the `q`
    behind the target is invented, which is what S6 is allowed to invent (S7
    owns real labels).
    """
    agent, _records = _agent_at_seat_zero(num_players, 100, n_hands, seed)
    rng = np.random.default_rng(seed)
    hands, targets, tables = [], [], []
    for _record, _n_dec, tok in agent.seen:
        legal = tok.legal[-1]
        q = np.full(N_ACTIONS, np.nan)
        q[legal] = rng.normal(0.0, 4.0, size=int(legal.sum()))
        _stack, pot_bb, to_call_bb = (float(x) for x in tok.scalars[-1])
        hands.append(tok)
        targets.append(policy_target(q, legal, pot_bb, to_call_bb,
                                     temperature=0.5))
        tables.append(rng.normal(0.0, 0.3, size=(MAX_PLAYERS,
                                                 NET_CFG["d_emb"])))
    assert len(hands) >= TRAIN_CFG["batch_hands"]
    return hands, np.stack(targets), tables


def _silent(_message):
    pass


def test_a_toy_training_run_reduces_the_loss():
    hands, targets, tables = _labels()
    net = _net(0)
    history = train_agent(net, hands, targets, tables, TRAIN_CFG, "cpu",
                          _silent, seed=0, iteration=0)

    assert len(history) == TRAIN_CFG["first_iteration_steps"]
    assert all(np.isfinite(h["kl"]) and h["kl"] >= 0.0 for h in history)
    first = np.mean([h["kl"] for h in history[:10]])
    last = np.mean([h["kl"] for h in history[-10:]])
    assert last < first


def test_training_leaves_the_pool_and_the_embeddings_untouched():
    """Nothing but the agent's own weights may move.

    The pool is not an argument of the loop and the embeddings are an input to
    it: they are fitted by the embedding network (§5.5), and a loop that
    updated them here would be optimising the opponent model against the
    agent's own loss without anybody asking for it.
    """
    hands, targets, tables = _labels(seed=2)
    pool = make_pool(seed=2)
    pool_before = [m.style.to_list() for m in pool]
    tables_before = [np.array(t, copy=True) for t in tables]
    targets_before = targets.copy()

    net = _net(0)
    train_agent(net, hands, targets, tables, TRAIN_CFG, "cpu", _silent,
                seed=0, iteration=1)

    assert [m.style.to_list() for m in pool] == pool_before
    assert all(np.array_equal(a, b) for a, b in zip(tables, tables_before))
    assert np.array_equal(targets, targets_before)


def test_a_training_run_is_deterministic():
    hands, targets, tables = _labels(seed=3)
    a = train_agent(_net(0), hands, targets, tables, TRAIN_CFG, "cpu",
                    _silent, seed=7, iteration=1)
    b = train_agent(_net(0), hands, targets, tables, TRAIN_CFG, "cpu",
                    _silent, seed=7, iteration=1)
    assert [h["kl"] for h in a] == [h["kl"] for h in b]


def test_a_hand_that_is_not_a_pending_decision_is_refused():
    hands, targets, tables = _labels(seed=4)
    broken = copy.deepcopy(hands[0])
    broken.action[-1] = 0
    with pytest.raises(AssertionError, match="pending token"):
        train_agent(_net(0), [broken] + hands[1:], targets, tables, TRAIN_CFG,
                    "cpu", _silent, seed=0, iteration=1)


# ------------------------------------------------------------------ 9: the training cycle


def test_the_first_cycle_gets_its_own_step_count():
    cfg = {"steps": 10, "first_iteration_steps": 40}
    assert steps_for_iteration(cfg, 0) == 40
    assert steps_for_iteration(cfg, 1) == 10
    assert steps_for_iteration(cfg, 9) == 10
    # Omitting the key means the first cycle is trained like the rest.
    assert steps_for_iteration({"steps": 10}, 0) == 10


def test_the_first_cycle_actually_runs_more_steps_than_a_later_one():
    hands, targets, tables = _labels(seed=5)
    first = train_agent(_net(0), hands, targets, tables, TRAIN_CFG, "cpu",
                        _silent, seed=0, iteration=0)
    later = train_agent(_net(0), hands, targets, tables, TRAIN_CFG, "cpu",
                        _silent, seed=0, iteration=1)
    assert len(first) == 40 and len(later) == 12
    # The two cycles start from the same weights and draw the same first batch,
    # so the only difference is how long each of them runs.
    assert first[0]["kl"] == later[0]["kl"]


def test_a_later_cycle_continues_the_network_the_previous_one_produced():
    """The agent at iteration k is the result of training at iteration k−1."""
    hands, targets, tables = _labels(seed=6)

    fresh = _net(0)
    cold = train_agent(fresh, hands, targets, tables, TRAIN_CFG, "cpu",
                       _silent, seed=0, iteration=1)

    warm = _net(0)
    before = copy.deepcopy(warm.state_dict())
    cycle0 = train_agent(warm, hands, targets, tables, TRAIN_CFG, "cpu",
                         _silent, seed=0, iteration=0)
    cycle1 = train_agent(warm, hands, targets, tables, TRAIN_CFG, "cpu",
                         _silent, seed=0, iteration=1)

    # Same seed, same data: cycle 0 and the cold run open on the same batch
    # from the same weights, so their first losses are the same number.
    assert cycle0[0]["kl"] == cold[0]["kl"]
    # Cycle 1 opens on that same batch — but from the weights cycle 0 left.
    assert cycle1[0]["kl"] < cycle0[0]["kl"]
    assert cycle1[0]["kl"] == pytest.approx(cycle0[-1]["kl"], rel=0.5)

    after = warm.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in after)


def test_the_cycles_are_a_chain_and_not_a_sequence_of_restarts():
    """Three cycles in a row keep improving on the same labels."""
    hands, targets, tables = _labels(seed=8)
    net = _net(0)
    opens = []
    for iteration in range(3):
        history = train_agent(net, hands, targets, tables, TRAIN_CFG, "cpu",
                              _silent, seed=0, iteration=iteration)
        opens.append(history[0]["kl"])
    assert opens[0] > opens[1] > opens[2]
