"""§5.7 — the opponent-range target, the head that predicts it, and the
injection of that belief back into the trunk.

Behavioural throughout (`CLAUDE.md` §4): the target is checked against the
estimator it is supposed to be, the head is checked for the leak it could have,
and the switch is checked for being an exact ablation.
"""

import numpy as np
import pytest
import torch

from env.driver import LockstepDriver
from nets.embedding_net import OpponentEmbeddingNet, loss_weights
from nets.agent_net import AgentNet
from nets.features import collate, hand_tokens, range_targets
from nets.range_head import combo_block_mask, range_loss
from oracle.posterior import combo_universe, opponent_posterior
from oracle.ranges import (N_COMBOS, RangeTracker, combo_index, hand_ranges,
                           label_ranges)
from tests.g1_fixtures import (MAX_PLAYERS, N_ACTIONS, NET_CFG, make_pool,
                               make_specs, play)

RANGE_CFG = dict(NET_CFG, range_enabled=True, range_layer=1, n_range_blocks=2,
                 d_range=32, range_heads=4)


def a_hand(seed=5, num_players=3, min_decisions=4):
    """One played record with enough decisions to condition on."""
    pool = make_pool()
    specs = make_specs(seed=seed, n_hands=24, n_members=len(pool),
                       num_players=num_players)
    for record in play(pool, specs):
        if len(record.decisions) >= min_decisions:
            return pool, record
    pytest.fail("the fixture produced no hand long enough")


# ------------------------------------------------------------------ the target


def test_the_target_is_the_posterior_at_every_prefix_when_nothing_is_pruned():
    """§5.7's whole claim: the filter *is* `opponent_posterior`, run forward.

    Not "close to" — the same numbers, prefix by prefix, up to the one thing
    that genuinely differs: the filter's dead set is the board visible at the
    token, where a posterior conditioned through the previous decision is one
    street behind. That difference is applied to the posterior here, which is
    exactly card removal followed by renormalisation.
    """
    pool, record = a_hand()
    observer = 0
    targets, stats = hand_ranges(record, observer, pool, N_ACTIONS, prune=0.0)
    assert stats.dropped == 0.0 and stats.collapsed == 0
    assert targets, "a multiway hand must produce at least one target"

    for (t, seat), (idx, w) in targets.items():
        combos, weights = opponent_posterior(
            record, seat, observer, pool, N_ACTIONS, through_decision=t - 1)
        turn = int(record.snapshots[record.decisions[t]["snap_idx"]]["turn"])
        board = [int(c) for c in record.deck[:(0, 3, 4, 5)[turn]]]
        keep = ~np.isin(combos, np.asarray(board, dtype=np.int64)).any(axis=1)
        combos, weights = combos[keep], weights[keep] / weights[keep].sum()

        assert np.array_equal(np.sort(idx), np.sort(combo_index(combos)))
        order = np.argsort(idx)
        assert np.allclose(np.asarray(w)[order],
                           weights[np.argsort(combo_index(combos))], atol=1e-12)


def test_the_target_is_a_prefix_function_and_carries_no_future():
    """The belief at token `t` may not move when the hand is played on.

    Truncating the record after `t` and re-running the filter has to give the
    same answer — that is §9's rule stated for the range, and the way it fails
    is a target that quietly encodes what the opponent did later.
    """
    import copy

    pool, record = a_hand(min_decisions=5)
    full, _ = hand_ranges(record, 0, pool, N_ACTIONS)

    cut = 3
    short = copy.copy(record)
    short.decisions = record.decisions[:cut + 1]
    part, _ = hand_ranges(short, 0, pool, N_ACTIONS)

    for t in range(cut + 1):
        for seat in range(record.num_players):
            if (t, seat) in full or (t, seat) in part:
                assert (t, seat) in full and (t, seat) in part, (t, seat)
                assert np.array_equal(full[(t, seat)][0], part[(t, seat)][0])
                assert np.allclose(full[(t, seat)][1], part[(t, seat)][1])


def test_a_player_is_in_the_target_at_the_token_they_fold_and_never_after():
    pool, record = a_hand(num_players=4, min_decisions=5)
    folds = [(t, int(d["acting_pos"])) for t, d in enumerate(record.decisions)
             if int(d["action_idx"]) == 0]
    if not folds:
        pytest.skip("this fixture hand has no fold")
    t_fold, seat = folds[0]
    targets, _ = hand_ranges(record, 0, pool, N_ACTIONS)
    if seat == 0:
        pytest.skip("the folder is the observer, which has no target at all")
    assert (t_fold, seat) in targets, (
        "a player still holds cards while the decision to fold is being taken")
    assert all(key[1] != seat for key in targets if key[0] > t_fold)


def test_no_target_puts_mass_on_a_blocked_combo():
    pool, record = a_hand(min_decisions=6)
    targets, _ = hand_ranges(record, 0, pool, N_ACTIONS)
    hole = set(record.hole_cards(0))
    for (t, _seat), (idx, w) in targets.items():
        turn = int(record.snapshots[record.decisions[t]["snap_idx"]]["turn"])
        dead = hole | {int(c) for c in record.deck[:(0, 3, 4, 5)[turn]]}
        alive = combo_index(combo_universe(sorted(dead)))
        assert set(int(i) for i in idx).issubset(set(int(i) for i in alive))
        assert abs(float(np.sum(w)) - 1.0) < 1e-5   # stored as float32


def test_pruning_drops_a_tail_and_says_how_much_it_dropped():
    """The threshold only bites against a member whose play depends on cards.

    A degenerate member that plays the same way with every holding leaves the
    range uniform, and a uniform range has nothing to prune — which is itself
    the point of §5.7 and the reason the fixture here is a nit facing a caller
    rather than the general pool.
    """
    from env.driver import HandSpec
    from pool.degenerate import AlwaysCall, Nit
    from pool.style import StyleParams
    from tests.g1_fixtures import BIG_BLIND, RAISE_SIZES, SMALL_BLIND

    # Seat 0 observes and is the caller; the nit is the opponent, which is the
    # only arrangement where the observer has a range to narrow at all.
    pool = [Nit("nit", N_ACTIONS, StyleParams.identity()),
            AlwaysCall("caller", N_ACTIONS, StyleParams.identity())]
    specs = [HandSpec(num_players=2, start_credits=[2000.0, 2000.0],
                      seat_members=[1, 0], seed=1000 + i,
                      big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
                      raise_sizes=RAISE_SIZES)
             for i in range(64)]
    record = next((r for r in play(pool, specs)
                   if sum(1 for d in r.decisions
                          if int(d["acting_pos"]) == 1) >= 2), None)
    assert record is not None, "the fixture produced no nit with two decisions"

    wide, wide_stats = hand_ranges(record, 0, pool, N_ACTIONS, prune=0.0)
    tight, tight_stats = hand_ranges(record, 0, pool, N_ACTIONS, prune=0.05)

    assert tight_stats.forwards < wide_stats.forwards, (
        "a pruned support must buy forwards — that is the only reason it exists")
    assert tight_stats.dropped > 0.0
    assert all(len(tight[k][0]) <= len(wide[k][0]) for k in tight)
    # What it threw away is a tail: the heaviest combo is still there.
    for key, (idx, _w) in tight.items():
        best = int(np.asarray(wide[key][0])[int(np.argmax(wide[key][1]))])
        assert best in set(int(i) for i in idx)


def test_the_target_reaches_the_batch_on_the_rows_act_idx_names():
    pool, record = a_hand(min_decisions=4)
    targets, _ = hand_ranges(record, 0, pool, N_ACTIONS)
    tokens = hand_tokens(record, observer_pos=0,
                         slot_of_seat=list(range(record.num_players)),
                         max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                         ranges=targets)
    batch = collate([tokens])

    assert batch["range_target"].shape == (batch["act_idx"].shape[0], N_COMBOS)
    rows = {(int(b), int(t), int(s)): i
            for i, (b, t, s) in enumerate(batch["act_idx"].tolist())}
    for (t, seat), (idx, w) in targets.items():
        row = rows[(0, t, seat)]
        assert bool(batch["range_mask"][row])
        got = batch["range_target"][row].numpy()
        assert np.allclose(got[np.asarray(idx, dtype=np.int64)], w, atol=1e-7)
        assert abs(float(got.sum()) - 1.0) < 1e-5
    # Every row without a target is exactly zero and is masked out.
    assert not batch["range_target"][~batch["range_mask"]].any()


def test_the_combo_mask_the_head_uses_is_the_support_the_target_lives_on():
    pool, record = a_hand(min_decisions=4)
    targets, _ = hand_ranges(record, 0, pool, N_ACTIONS)
    tokens = hand_tokens(record, observer_pos=0,
                         slot_of_seat=list(range(record.num_players)),
                         max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                         ranges=targets)
    batch = collate([tokens])
    allowed = combo_block_mask(batch)
    for (t, _seat), (idx, _w) in targets.items():
        assert allowed[0, t][np.asarray(idx, dtype=np.int64)].all(), (
            "the head masks off a combo the target puts mass on")


# -------------------------------------------------------------------- the head


def _nets(cfg=RANGE_CFG, n_members=12):
    return (OpponentEmbeddingNet(cfg, N_ACTIONS, MAX_PLAYERS, n_members),
            AgentNet(cfg, N_ACTIONS, MAX_PLAYERS))


def _batch(seed=5, n_hands=6, ranges=True):
    pool = make_pool()
    records = play(pool, make_specs(seed=seed, n_hands=n_hands,
                                   n_members=len(pool), num_players=3))
    hands = []
    for r in records:
        if not r.decisions:
            continue
        targets = (hand_ranges(r, 0, pool, N_ACTIONS)[0] if ranges else None)
        hands.append(hand_tokens(
            r, observer_pos=0, slot_of_seat=[0, 1, 2],
            max_players=MAX_PLAYERS, n_actions=N_ACTIONS, ranges=targets))
    return collate([h for h in hands if len(h) > 0])


def test_the_head_answers_one_row_per_active_pair_and_masks_the_impossible():
    net, _agent = _nets()
    batch = _batch()
    hidden, logits = net.hidden_and_range(
        batch, net.member_emb(batch), net.member_seat_emb(batch))

    assert logits.shape == (batch["act_idx"].shape[0], N_COMBOS)
    assert hidden.shape[:2] == batch["mask"].shape
    b, t = batch["act_idx"][:, 0], batch["act_idx"][:, 1]
    blocked = ~combo_block_mask(batch)[b, t]
    assert torch.isinf(logits[blocked]).all() and (logits[blocked] < 0).all()
    assert torch.isfinite(logits[~blocked]).all()


def test_the_belief_at_a_token_cannot_read_a_later_token():
    """The cross-attention is causal, and this is how that is checked.

    Perturbing the *last* token's features may not move the belief at the
    first. Without the causal mask the head reads the whole hand and this test
    is the only thing between that and a loss curve that looks better than the
    deployed agent ever will.
    """
    net, _agent = _nets()
    batch = _batch(n_hands=4)
    lengths = batch["mask"].sum(dim=1).long()
    assert int(lengths.max()) >= 3

    def logits_of(bt):
        with torch.no_grad():
            return net.hidden_and_range(bt, net.member_emb(bt),
                                        net.member_seat_emb(bt))[1]

    base = logits_of(batch)
    moved = {k: (v.clone() if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    row = int(torch.argmax(lengths))
    last = int(lengths[row]) - 1
    moved["scalars"][row, last] += 7.5
    after = logits_of(moved)

    early = ((batch["act_idx"][:, 0] == row) & (batch["act_idx"][:, 1] < last))
    assert bool(early.any()), "the fixture must have an earlier active pair"
    assert torch.allclose(base[early], after[early], atol=1e-6)


def test_switching_the_head_off_reproduces_the_trunk_it_replaced():
    """`range_enabled: false` is an exact ablation, not a small one."""
    plain = dict(NET_CFG)
    torch.manual_seed(3)
    net = OpponentEmbeddingNet(plain, N_ACTIONS, MAX_PLAYERS, 12)
    batch = _batch(ranges=False)
    with torch.no_grad():
        hidden, logits = net.hidden_and_range(batch, net.member_emb(batch),
                                              net.member_seat_emb(batch))
        direct = net.hidden(batch, net.member_emb(batch))
    assert logits is None
    assert torch.equal(hidden, direct)
    assert not any("range" in name for name, _ in net.named_parameters())


def test_the_injected_belief_is_detached_from_the_action_loss():
    """Stop-grad: the action head's gradient may not reach the belief head.

    The whole point of the bottleneck is that the layers above consume a belief
    they cannot bend; without the detach the 1326 outputs become a free feature
    the action loss reshapes, and the range term is then fighting it.
    """
    net, _agent = _nets()
    batch = _batch()
    hidden, _logits = net.hidden_and_range(
        batch, net.member_emb(batch), net.member_seat_emb(batch))
    net.action_ce(net.action_out(hidden), batch).backward()

    head = net.encoder.range_head
    for name, p in head.named_parameters():
        if name.startswith(("value_in", "pos_out", "inject")):
            continue                       # these are downstream of the detach
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name


def test_the_belief_term_is_zero_exactly_when_the_prediction_is_the_target():
    batch = _batch()
    target = batch["range_target"]
    logits = torch.log(target.clamp_min(1e-30))
    logits = logits.masked_fill(target == 0, float("-inf"))
    loss, ce, kl = range_loss(logits, batch)
    assert abs(kl) < 1e-5
    assert ce > 0.0, "the cross-entropy still pays the target's own entropy"

    flat = torch.zeros_like(logits)
    _loss2, _ce2, kl2 = range_loss(flat, batch)
    assert kl2 > kl


def test_a_batch_with_no_target_scores_nothing_rather_than_crashing():
    batch = _batch(ranges=False)
    net, _agent = _nets()
    _hidden, logits = net.hidden_and_range(
        batch, net.member_emb(batch), net.member_seat_emb(batch))
    assert range_loss(logits, batch) is None


def test_the_weight_is_what_puts_the_term_in_the_total():
    net, _agent = _nets()
    batch = _batch()
    off = net.loss_terms(batch, loss_weights({"range_weight": 0.0}))
    on = net.loss_terms(batch, loss_weights({"range_weight": 1.0}))
    assert "range_ce" in off[1] and "range_kl" in off[1]
    assert float(on[0].detach()) > float(off[0].detach())
    assert abs(off[1]["range_ce"] - on[1]["range_ce"]) < 1e-4


def test_the_agent_reads_the_same_head_at_its_pending_decision():
    _net, agent = _nets()
    pool = make_pool()
    records = play(pool, make_specs(seed=9, n_hands=6, n_members=len(pool),
                                    num_players=3))
    record = next(r for r in records if len(r.decisions) >= 3)
    targets, _ = hand_ranges(record, 0, pool, N_ACTIONS, emit_at={2})
    tokens = hand_tokens(record, observer_pos=0, slot_of_seat=[0, 1, 2],
                         max_players=MAX_PLAYERS, n_actions=N_ACTIONS,
                         ranges=targets)
    batch = collate([tokens])
    emb = torch.zeros(*batch["mask"].shape, agent.d_emb)
    seat_emb = torch.zeros(*batch["seat_slot"].shape, agent.d_emb)
    logits, rng_logits = agent.logits_and_range(batch, emb, seat_emb)
    assert logits.shape == (1, N_ACTIONS)
    assert rng_logits.shape[0] == batch["act_idx"].shape[0]
    assert bool(batch["range_mask"].any())


# ------------------------------------------------------ end to end, §5.7 on


def test_a_whole_iteration_runs_with_the_range_head_on(tmp_path):
    """The loop with §5.7 enabled: corpus targets, label targets, both losses.

    This is the test that says the plumbing closes — `label_ranges` over the
    corpus, `label_range_target` on every label, the sparse target through a
    shard and back, and the belief term in both trainers. Everything else in
    this file pins one property; this one pins that they compose.
    """
    from tests.test_pipeline import _labels_of, _run, toy_config

    cfg = toy_config()
    cfg["embedding_net"] = dict(
        cfg["embedding_net"], range_enabled=True, range_layer=1,
        n_range_blocks=1, d_range=32, range_heads=4, range_weight=0.5,
        range_prune_threshold=0.0)
    cfg["agent_train"] = dict(cfg["agent_train"], range_weight=0.5)

    metrics, exp_dir = _run(tmp_path, cfg=cfg, name="range")
    assert len(metrics) == 2

    labels = _labels_of(exp_dir, 0)
    assert labels, "the run wrote no labels"
    with_target = [lab for lab in labels if lab["tokens"].ranges is not None
                   and len(lab["tokens"].ranges.token)]
    assert with_target, "no label carried a §5.7 target through its shard"

    for lab in with_target:
        tokens = lab["tokens"]
        pending = len(tokens) - 1
        assert set(int(t) for t in tokens.ranges.token) == {pending}, (
            "a label's belief belongs to its own pending decision and to no "
            "other token")
        for seat in set(int(s) for s in tokens.ranges.seat):
            sel = tokens.ranges.seat == seat
            assert abs(float(tokens.ranges.weight[sel].sum()) - 1.0) < 1e-4
            assert bool(tokens.active_opp[pending, seat])


def test_the_two_trainers_report_the_belief_term(tmp_path):
    """Both losses have to *say* the term is there, or nobody can read it.

    A weight nobody logs is a weight nobody can size, and §5.7's own number is
    the KL and not the cross-entropy — so both have to be in the history.
    """
    import os

    import torch

    from tests.test_pipeline import _run, toy_config

    cfg = toy_config()
    cfg["embedding_net"] = dict(
        cfg["embedding_net"], range_enabled=True, range_layer=1,
        n_range_blocks=1, d_range=32, range_heads=4, range_weight=0.5,
        range_prune_threshold=0.0)
    cfg["agent_train"] = dict(cfg["agent_train"], range_weight=0.5)
    _metrics, exp_dir = _run(tmp_path, cfg=cfg, name="range_hist")

    agent = torch.load(os.path.join(exp_dir, "iter_0000", "agent.pt"),
                       map_location="cpu", weights_only=False)
    assert any("range_kl" in row and "range_ce" in row
               for row in agent["history"]), (
        "the agent trained with the head on and reported no belief term")

    embed = torch.load(os.path.join(exp_dir, "iter_0000", "embedding.pt"),
                       map_location="cpu", weights_only=False)
    assert any("range_kl" in row and "range_ce" in row
               for row in embed["history"]), (
        "the embedding network trained with the head on and reported none")
