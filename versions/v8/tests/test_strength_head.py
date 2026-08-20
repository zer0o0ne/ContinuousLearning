"""The strength head and the agent's warm start (CONCEPT.md §5.6, §6.1).

Three properties, and each of them is a way the pretrain could be wrong without
any loss curve saying so:

* **the target is right and it is a target.** `own_strength` is the percentile
  of the *observer's own* hand on that hand's final board, it sits on the
  observer's own decision tokens and nowhere else, and it changes no input the
  network reads. A hand still in progress carries none, because it has no final
  board — the case that would otherwise turn a value target into a leak.
* **it shapes the trunk and not the vectors.** The term is in training only;
  §5.5's fit must be bit-identical with and without it, because the strength of
  the observer's own cards says nothing about anybody's style.
* **the warm start copies the trunk and ties nothing.** After it the agent's
  encoder *is* the embedding network's, its action head is untouched, and one
  gradient step moves one and not the other.

Everything is end-to-end through played hands (`CLAUDE.md` §4): the records come
out of the driver, the labels out of `label_showdowns`, the tokens out of
`hand_tokens`.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from env.driver import DecisionContext, LockstepDriver
from env.showdown import label_showdowns, strength_percentiles
from env.session import Session
from nets.agent_net import AgentNet
from nets.embedding_net import OpponentEmbeddingNet, fit_embeddings
from nets.features import TOKEN_DECISION, collate, hand_tokens
from pipeline import warm_start_trunk
from train.agent_train import steps_for_iteration
from train.embed_train import train_embedding_net
from tests.g1_fixtures import (
    MAX_PLAYERS, N_ACTIONS, NET_CFG, make_pool, observer_seat, session_specs,
    slot_of_seat,
)

NUM_PLAYERS = 3
N_HANDS = 10
WEIGHTS = {"amortised": 1.0, "showdown_strength": 0.3, "showdown_class": 0.1,
           "strength": 1.5}


def _records(seed=0, labelled=True):
    """A played session, labelled or not.

    `labelled=False` drops the reveals along with the labels, because that is
    the only shape an unlabelled record has in the wild: the evaluation replay
    sets `showdown` and calls `label_showdowns` in the same branch, and §5.1a
    already refuses a reveal whose labels nobody computed.
    """
    pool = make_pool()
    members = list(range(NUM_PLAYERS))
    specs = session_specs(members, NUM_PLAYERS, N_HANDS, seed=seed)
    records = LockstepDriver(pool, N_ACTIONS).run(specs)
    if labelled:
        label_showdowns(records)
    else:
        for record in records:
            record.showdown = []
    return pool, records


def _tokens(records):
    """One tokenisation per hand, from slot 0's rotating seat."""
    out = []
    for h, record in enumerate(records):
        tokens = hand_tokens(record, observer_seat(NUM_PLAYERS, h),
                             slot_of_seat(NUM_PLAYERS, h), MAX_PLAYERS,
                             N_ACTIONS)
        if len(tokens):
            out.append((h, record, tokens))
    return out


def _net(pool, seed=0):
    torch.manual_seed(seed)
    return OpponentEmbeddingNet(NET_CFG, N_ACTIONS, MAX_PLAYERS,
                               n_members=len(pool)).eval()


# ------------------------------------------------------------------ the target


def test_target_is_the_observers_own_percentile_on_the_final_board():
    _pool, records = _records()
    seen = 0
    for h, record, tokens in _tokens(records):
        observer = observer_seat(NUM_PLAYERS, h)
        expected = float(strength_percentiles(
            record.deck[:5], [record.hole_cards(observer)])[0])

        own = tokens.own_strength
        mine = (tokens.acting_pos == observer) & (tokens.token_type
                                                  == TOKEN_DECISION)
        assert np.allclose(own[mine], expected, atol=1e-12)
        assert np.all(own[~mine] == -1.0)
        seen += int(mine.sum())
    assert seen > 0, "the fixture produced no decision by the observer"


def test_showdown_labels_are_the_same_dict_restricted_to_the_revealed_seats():
    """§5.1a's labels must not have moved when §5.6 joined the same pass."""
    _pool, records = _records()
    checked = 0
    for record in records:
        assert set(record.hand_strength) == set(range(record.num_players))
        for pos in record.showdown:
            expected = float(strength_percentiles(
                record.deck[:5], [record.hole_cards(pos)])[0])
            assert record.showdown_strength[pos] == pytest.approx(expected,
                                                                  abs=1e-12)
            assert record.hand_strength[pos] == record.showdown_strength[pos]
            checked += 1
    assert checked > 0, "the fixture produced no showdown"


def test_a_hand_in_progress_carries_no_target():
    """No final board exists yet, so there is nothing to score (§5.6)."""
    _pool, records = _records()
    asked = 0
    for h, record, _tok in _tokens(records):
        observer = observer_seat(NUM_PLAYERS, h)
        for d, dec in enumerate(record.decisions):
            if int(dec["acting_pos"]) != observer:
                continue
            snap = record.snapshots[dec["snap_idx"]]
            # Truncate to the moment of the decision, exactly as the live path
            # does: the agent is asked before the hand has finished.
            live = type(record)(
                spec=record.spec, deck=record.deck,
                snapshots=record.snapshots[:dec["snap_idx"] + 1],
                decisions=record.decisions[:d], rewards=record.rewards,
                truncated=record.truncated)
            live.hand_strength = record.hand_strength
            ctx = DecisionContext(live, dec["snap_idx"], observer,
                                  dec["legal_mask"], int(snap["turn"]))
            tokens = hand_tokens(live, observer, slot_of_seat(NUM_PLAYERS, h),
                                 MAX_PLAYERS, N_ACTIONS, pending=ctx)
            assert np.all(tokens.own_strength == -1.0)
            asked += 1
    assert asked > 0


def test_an_unlabelled_record_carries_no_target_and_no_crash():
    """The evaluation replay builds records nobody labels (§5.6)."""
    _pool, records = _records(labelled=False)
    for _h, _record, tokens in _tokens(records):
        assert np.all(tokens.own_strength == -1.0)


def test_the_target_changes_no_input_the_network_reads():
    """It is a target, not a feature.

    The same hands tokenised twice, differing in the §5.6 label alone — the
    showdown tokens stay, so the two batches have the same shape and the
    comparison is of the label and of nothing else.
    """
    pool, labelled = _records(seed=1)
    stripped = [replace(r, hand_strength={}) for r in labelled]
    net = _net(pool)

    a = collate([t for _h, _r, t in _tokens(labelled)])
    b = collate([t for _h, _r, t in _tokens(stripped)])
    assert bool((a["own_strength"] >= 0).any())
    assert bool((b["own_strength"] == -1.0).all())
    for key in a:
        if key in ("own_strength", "strength_mask"):
            continue
        assert torch.equal(a[key], b[key]), key

    with torch.no_grad():
        assert torch.equal(net(a, net.member_emb(a)), net(b, net.member_emb(b)))


def test_collate_masks_the_padding_and_only_the_labelled_tokens():
    _pool, records = _records()
    hands = [t for _h, _r, t in _tokens(records)]
    assert len({len(h) for h in hands}) > 1, "need ragged hands to test padding"
    batch = collate(hands)

    mask = batch["strength_mask"] > 0
    assert torch.equal(mask, batch["own_strength"] >= 0)
    assert bool((batch["own_strength"][mask] >= 0).all())
    assert bool((batch["own_strength"][mask] <= 1).all())
    # The padded tail is never scored, and it is the sentinel that says so.
    pad = batch["mask"] == 0
    assert bool((batch["own_strength"][pad] == -1.0).all())
    for b, hand in enumerate(hands):
        assert int(mask[b].sum()) == int((hand.own_strength >= 0).sum())


# ------------------------------------------------------------------- the loss


def test_the_weight_shifts_the_total_by_exactly_its_term():
    pool, records = _records()
    net = _net(pool)
    batch = collate([t for _h, _r, t in _tokens(records)])

    off = dict(WEIGHTS, strength=0.0)
    total_off, parts_off = net.loss_terms(batch, off)
    total_on, parts_on = net.loss_terms(batch, WEIGHTS)

    assert "own_strength_mse" in parts_on
    mse = parts_on["own_strength_mse"]
    assert parts_off["own_strength_mse"] == pytest.approx(mse, rel=1e-9)
    assert float(total_on.detach()) == pytest.approx(
        float(total_off.detach()) + WEIGHTS["strength"] * mse, rel=1e-5)
    assert parts_off["action_ce"] == pytest.approx(parts_on["action_ce"],
                                                   rel=1e-9)


def test_a_batch_with_no_target_is_a_batch_and_not_an_error():
    pool, records = _records(labelled=False)
    net = _net(pool)
    batch = collate([t for _h, _r, t in _tokens(records)])
    assert net.strength_loss(net.hidden(batch, net.member_emb(batch)),
                             batch) is None
    total, parts = net.loss_terms(batch, WEIGHTS)
    assert "own_strength_mse" not in parts
    assert torch.isfinite(total)


def test_the_head_can_learn_the_target():
    """A trunk given the observer's own cards can predict its own strength.

    Not a quality bar — a wiring check: if the gradient reached the head at all,
    a few hundred steps on ten hands drive the MSE below the variance of the
    target, which is the only baseline the number is ever read against (§5.6).
    """
    pool, records = _records()
    net = _net(pool)
    batch = collate([t for _h, _r, t in _tokens(records)])
    target = batch["own_strength"][batch["strength_mask"] > 0]
    baseline = float(target.var(unbiased=False))

    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    net.train()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = net.strength_loss(net.hidden(batch, net.member_emb(batch)),
                                 batch)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        final = float(net.strength_loss(
            net.hidden(batch, net.member_emb(batch)), batch))
    assert final < baseline


def test_the_fit_never_sees_the_strength_term():
    """§5.5 optimises the vectors; the observer's own cards cannot inform them."""
    pool, records = _records()
    net = _net(pool)
    batch = collate([t for _h, _r, t in _tokens(records)])
    init = torch.zeros(NUM_PLAYERS, net.d_emb)

    with_term = fit_embeddings(net, batch, NUM_PLAYERS, steps=5, lr=0.05,
                               reg=0.01, init=init, weights=WEIGHTS)
    without = fit_embeddings(net, batch, NUM_PLAYERS, steps=5, lr=0.05,
                             reg=0.01, init=init,
                             weights=dict(WEIGHTS, strength=0.0))
    assert torch.equal(with_term, without)
    # …and the fit did move, so the equality above is not two no-ops.
    assert not torch.equal(with_term, init)


# ------------------------------------------------------------- the step counts


def test_first_retrain_steps_selects_only_the_first_retrain():
    cfg = {"steps": 7, "first_retrain_steps": 31}
    assert steps_for_iteration(cfg, 0, "first_retrain_steps") == 31
    assert steps_for_iteration(cfg, 1, "first_retrain_steps") == 7
    assert steps_for_iteration(cfg, 2, "first_retrain_steps") == 7
    # An absent key means the first retrain is trained like the rest.
    assert steps_for_iteration({"steps": 7}, 0, "first_retrain_steps") == 7
    # The agent's key is untouched by the generalisation.
    assert steps_for_iteration({"steps": 7, "first_iteration_steps": 9}, 0) == 9


def test_the_trainer_runs_the_step_count_the_iteration_asks_for():
    pool, records = _records()
    session = Session(idx=0, num_players=NUM_PLAYERS, stack_bb=100,
                      members=list(range(NUM_PLAYERS)),
                      specs=[r.spec for r in records], records=list(records))
    cfg = {"steps": 3, "first_retrain_steps": 8, "batch_hands": 4, "lr": 1e-4,
           "eta_min": 1e-5, "weight_decay": 0.0, "grad_clip": 1.0,
           "log_every": 1, "amortised_weight": 1.0,
           "showdown_strength_weight": 0.3, "showdown_class_weight": 0.0,
           "strength_weight": 1.0}
    game = {"max_players": MAX_PLAYERS, "n_actions": N_ACTIONS}

    lines = []
    first = train_embedding_net(_net(pool), [session], cfg, game, "cpu",
                                lines.append, seed=0, iteration=0)
    later = train_embedding_net(_net(pool), [session], cfg, game, "cpu",
                                lines.append, seed=0, iteration=1)
    assert [h["step"] for h in first] == list(range(1, 9))
    assert [h["step"] for h in later] == list(range(1, 4))
    assert any("strength targets" in line for line in lines)
    assert all("own_strength_mse" in h for h in first)


# ------------------------------------------------------------- the warm start


def test_warm_start_copies_the_trunk_and_leaves_the_head_alone():
    pool, _records_ = _records()
    embed = _net(pool, seed=0)
    torch.manual_seed(1)
    agent = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS)
    before = agent.action_out.weight.detach().clone()

    warm_start_trunk(agent, embed, lambda *_: None)

    for (na, pa), (ne, pe) in zip(agent.encoder.state_dict().items(),
                                  embed.encoder.state_dict().items()):
        assert na == ne
        assert torch.equal(pa, pe)
    assert torch.equal(agent.action_out.weight, before)


def test_warm_start_is_an_initialisation_and_not_a_tie():
    pool, records = _records()
    embed = _net(pool, seed=0)
    torch.manual_seed(1)
    agent = AgentNet(NET_CFG, N_ACTIONS, MAX_PLAYERS)
    warm_start_trunk(agent, embed, lambda *_: None)

    batch = collate([t for _h, _r, t in _tokens(records)])
    emb = torch.zeros(*batch["mask"].shape, agent.d_emb)
    opt = torch.optim.SGD(agent.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)
    agent(batch, emb).square().mean().backward()
    opt.step()

    tuned = agent.encoder.state_dict()
    frozen = embed.encoder.state_dict()
    assert any(not torch.equal(tuned[k], frozen[k]) for k in frozen)
