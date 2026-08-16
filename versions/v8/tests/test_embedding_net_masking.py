"""The attention structure of the embedding network (CONCEPT.md §5.2, §15).

Two properties, both tested behaviourally rather than by inspecting a mask:

* **causal within a hand** — perturbing token *t* leaves the predictions at
  *t′ ≤ t* bit-identical. Without this the token at *t* sees the action taken at
  *t* through token *t+1*'s `prev_action` field and the prediction task is
  trivial;
* **block-diagonal across hands** — perturbing a token in hand *j* leaves hand
  *k ≠ j* bit-identical, and reordering the hands permutes the predictions and
  changes nothing else. With cross-hand attention open the transformer would
  infer style in-context from the prefix and the embedding would receive no
  gradient pressure at all — the shortcut §5.2 exists to close, and one that
  hides itself because the prediction loss looks fine.

Together these are "the embedding is the only cross-hand channel".
"""

import numpy as np
import pytest
import torch

from nets.embedding_net import OpponentEmbeddingNet
from nets.features import collate, hand_tokens
from tests.g1_fixtures import (
    MAX_PLAYERS, N_ACTIONS, NET_CFG, make_pool, observer_seat, play,
    session_specs, slot_of_seat,
)


def _setup(n_hands=8, num_players=4, seed=0):
    pool = make_pool()
    members = [0, 1, 2, 3][:num_players]
    records = play(pool, session_specs(members, num_players, n_hands, seed=seed))
    hands = [
        hand_tokens(r, observer_seat(num_players, h),
                    slot_of_seat(num_players, h), MAX_PLAYERS, N_ACTIONS)
        for h, r in enumerate(records)
    ]
    hands = [h for h in hands if len(h) >= 3]
    assert len(hands) >= 4
    batch = collate(hands)
    torch.manual_seed(0)
    net = OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS,
                               n_members=len(pool)).eval()
    return net, batch, hands


@torch.no_grad()
def _logits(net, batch):
    return net(batch, net.member_emb(batch))


def test_predictions_are_causal_within_a_hand():
    net, batch, hands = _setup()
    base = _logits(net, batch)

    lengths = batch["mask"].sum(dim=1).long()
    row = int(torch.argmax(lengths))
    t = int(lengths[row]) - 2
    assert t >= 1

    perturbed = {k: v.clone() for k, v in batch.items()}
    # Rewrite everything token t carries about the action just taken.
    perturbed["prev_action"][row, t] = 0.0
    perturbed["prev_action"][row, t, (int(batch["action"][row, t - 1]) + 1)
                             % N_ACTIONS] = 1.0
    perturbed["scalars"][row, t] += 3.0
    perturbed["cards"][row, t, 5:] = 0

    after = _logits(net, perturbed)
    assert torch.equal(base[row, :t], after[row, :t]), (
        "a change at token t moved a prediction at t' < t — attention is not "
        "causal within the hand")
    assert not torch.equal(base[row, t], after[row, t]), (
        "the perturbation had no effect at all; the test is not testing "
        "anything")


def test_hands_do_not_see_each_other():
    net, batch, hands = _setup()
    base = _logits(net, batch)

    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["scalars"][0] += 5.0
    perturbed["cards"][0] = 0
    perturbed["prev_action"][0] = 0.0
    after = _logits(net, perturbed)

    assert not torch.equal(base[0], after[0])
    assert torch.equal(base[1:], after[1:]), (
        "perturbing one hand changed another — cross-hand attention is open")


def test_shuffling_the_order_of_hands_changes_nothing():
    """With `e` fixed, hands are exchangeable (§5.2). The corpus may therefore
    be subsampled per gradient step without changing what is learned."""
    net, batch, hands = _setup()
    base = _logits(net, batch)

    order = np.array([3, 1, 0, 2] + list(range(4, len(hands))))
    shuffled = {k: v[order] for k, v in batch.items()}
    after = _logits(net, shuffled)

    for new_row, old_row in enumerate(order):
        t = int(batch["mask"][old_row].sum())
        assert torch.equal(base[old_row, :t], after[new_row, :t])


def test_the_embedding_is_what_changes_the_prediction():
    """The channel is not merely closed — it is the embedding that is open."""
    net, batch, _ = _setup()
    with torch.no_grad():
        zero = net(batch, net.zero_emb(batch))
        vectors = torch.randn(8, net.d_emb)
        conditioned = net(batch, net.slot_emb(batch, vectors))
    assert not torch.allclose(zero, conditioned), (
        "the embedding had no effect on the prediction at all")


def test_a_showdown_token_cannot_reach_back_into_any_decision():
    """§5.1a's safety property, and the reason the reveal is a terminal token
    rather than cards written into the decisions: the outcome of the hand must
    not flow backwards into the prediction of the actions that led to it."""
    net, batch, hands = _setup()
    rows = (batch["showdown_mask"].sum(dim=1) > 0).nonzero().flatten()
    assert len(rows) > 0, "no showdown in the fixture — the test is vacuous"

    base = _logits(net, batch)
    perturbed = {k: v.clone() for k, v in batch.items()}
    sd = batch["showdown_mask"] > 0
    perturbed["cards"][sd] = 0
    perturbed["scalars"][sd] += 7.0
    perturbed["prev_action"][sd] = 1.0
    after = _logits(net, perturbed)

    dec = batch["decision_mask"] > 0
    assert torch.equal(base[dec], after[dec]), (
        "perturbing a showdown token moved a decision-token prediction — the "
        "reveal is leaking backwards")
    assert not torch.equal(base[sd], after[sd]), (
        "the perturbation did nothing at all; the test is not testing anything")


def test_the_action_loss_ignores_showdown_tokens():
    net, batch, _ = _setup()
    with torch.no_grad():
        logits = net(batch, net.member_emb(batch))
        ce_real = net.action_ce(logits, batch)

        wrecked = {k: v.clone() for k, v in batch.items()}
        sd = batch["showdown_mask"] > 0
        wrecked["action"][sd] = 0
        wrecked["legal"][sd] = False
        wrecked["legal"][sd, 0] = True
        ce_wrecked = net.action_ce(logits, wrecked)
    assert torch.equal(ce_real, ce_wrecked), (
        "a showdown token contributed to the action cross-entropy")


def test_both_showdown_heads_produce_a_loss_and_reach_the_embedding():
    net, batch, _ = _setup()
    weights = {"showdown_strength": 0.3, "showdown_class": 0.1}
    total, parts = net.objective(batch, net.member_emb(batch), weights)
    assert "showdown_strength_mse" in parts and "showdown_class_ce" in parts

    net.zero_grad(set_to_none=True)
    total.backward()
    assert float(net.embeddings.weight.grad.abs().sum()) > 0
    assert float(net.showdown_strength_out.weight.grad.abs().sum()) > 0
    assert float(net.showdown_class_out.weight.grad.abs().sum()) > 0


def test_zero_weights_reduce_the_objective_to_action_cross_entropy():
    """The §5.1a ablation has to be exactly "as if the heads were not there"."""
    net, batch, _ = _setup()
    total, parts = net.objective(batch, net.member_emb(batch), {})
    assert float(total.detach()) == pytest.approx(parts["action_ce"])


def test_padding_never_reaches_a_real_token():
    """Appending an extra padded hand must not move any prediction."""
    net, batch, hands = _setup()
    base = _logits(net, batch)

    padded = collate(hands + [hands[0]])
    padded_out = _logits(net, padded)
    for row in range(len(hands)):
        t = int(batch["mask"][row].sum())
        assert torch.equal(base[row, :t], padded_out[row, :t])
