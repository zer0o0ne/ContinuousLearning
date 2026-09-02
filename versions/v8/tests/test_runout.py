"""Variance reduction in the rollouts (`env/runout.py`, CONCEPT.md §7.3).

Every test here is exact (`CLAUDE.md` §4). The estimator's whole claim is that
it changes the noise on a rollout and not what the rollout estimates, and that
claim is an identity, not a tendency — so it is checked against an
independently computed expectation to machine precision rather than against a
tolerance over a large sample.

The identities that matter, and the tests that pin them:

* a hand with no decisions left is worth the average over **every** board that
  could still come — checked against replaying that hand once per possible
  river through the untouched engine;
* the fold/no-fold correction has zero mean over the draw it corrects, so the
  reduced values of the two branches average to the same number the raw ones
  are *supposed* to average to — and, in the case built here, visibly do not;
* the baseline cannot see chips beyond the call, which is what makes the
  correction computable without replaying every raise size;
* turning the estimator on does not move a card or an action of the hand it
  measures.

The engine's own showdown is the reference throughout: `env/runout.py` ranks
many boards at once but settles every one of them through `Judger`.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from env.driver import HandSpec, LockstepDriver
from env.judger import Judger
from env.legal import legal_actions
from env.runout import (RunoutConfig, board_completions, equity_baseline,
                        seat_scores)
from env.table import Table
from gto_utils.gpu_solver import evaluate_hands
from pool.base import PoolMember
from pool.style import StyleParams
from oracle.rollout import OracleConfig, action_values
from tests.g1_fixtures import (BIG_BLIND, N_ACTIONS, RAISE_SIZES, SMALL_BLIND,
                               make_pool, make_specs, play)

FOLD, CALL = 0, 1
ALLIN = N_ACTIONS - 1
#: Walk a heads-up hand to the turn and shove: call, check, check, check, all-in.
TO_TURN_SHOVE = [CALL, CALL, CALL, CALL, ALLIN]


class Coin(PoolMember):
    """Equal logits everywhere.

    Facing an all-in only fold and call are legal, so this member's draw there
    is exactly 50-50 — which is what makes the fold correction's weight
    checkable without measuring a frequency.
    """

    def logits(self, contexts):
        return np.zeros((len(contexts), self.n_actions), dtype=np.float64)


def _spec(seed, deck=None, forced=TO_TURN_SHOVE, credits=1000.0):
    return HandSpec(num_players=2, start_credits=[credits, credits],
                    seat_members=[0, 0], seed=seed,
                    big_blind=BIG_BLIND, small_blind=SMALL_BLIND,
                    raise_sizes=RAISE_SIZES, meta={}, deck=deck,
                    forced_actions=list(forced))


def _every_river(deck):
    """The 44 decks that differ from `deck` only in the river card.

    Slots from `5 + 2 * players` on hold cards nobody was dealt, so swapping the
    river with each of them — and keeping the deck itself — enumerates every
    river consistent with what the hand has already seen.
    """
    deck = np.asarray(deck)
    out = []
    for j in [4] + list(range(9, 52)):
        variant = deck.copy()
        variant[4], variant[j] = variant[j], variant[4]
        out.append(variant)
    return out


# ------------------------------------------------------- the engine's showdown


def test_full_house_reads_the_highest_pair_as_its_kicker():
    """Both hands play the board's full house, so neither can win.

    The engine used to read the kicker off a descending scan that had already
    passed the trips, i.e. off the *lowest* qualifying pair, so a player holding
    a pocket pair below a board pair was ranked below one holding junk. It cost
    about one showdown in ten thousand, always in the same direction.
    """
    judger = Judger()
    # Board: Q♦ 3♠ Q♥ 3♣ 3♥ — threes full of queens, on the board.
    board = [45, 7, 46, 4, 6]
    with_low_pair = np.array(board + [1, 2])      # pocket deuces
    with_junk = np.array(board + [22, 27])        # seven-eight
    assert judger.compare_hands(with_low_pair, with_junk) == (1, 1)

    # Board: J♣ J♠ A♦ A♠ J♥ — jacks full of aces, again on the board.
    board = [37, 39, 48, 51, 38]
    assert judger.compare_hands(np.array(board + [16, 19]),
                                np.array(board + [15, 18])) == (1, 1)


@pytest.mark.parametrize("deck_pool, seed", [
    (np.arange(52), 11),                       # ordinary hands
    (np.arange(20), 12),                       # five ranks: full houses everywhere
    (np.arange(12), 13),                       # three ranks: quads and full houses
    (np.concatenate([np.arange(0, 52, 4),
                     np.arange(1, 52, 4)]), 14),   # two suits: flushes
])
def test_engine_and_batched_evaluator_rank_hands_identically(deck_pool, seed):
    """One ordering, two implementations, and they have to be the same one.

    `env/runout.py` ranks a board with the batched evaluator and then settles
    through `Judger`; if the two disagreed anywhere, integrating the board out
    would not reduce the noise on a label, it would move the label. The
    restricted decks are here because a uniform deck almost never produces the
    hands the two evaluators can differ on.
    """
    rng = np.random.default_rng(seed)
    n = 4000
    hands = [rng.permutation(deck_pool)[:9] for _ in range(n)]
    left = torch.as_tensor(np.array([np.concatenate([h[:5], h[5:7]]) for h in hands]))
    right = torch.as_tensor(np.array([np.concatenate([h[:5], h[7:9]]) for h in hands]))
    ls, rs = evaluate_hands(left).numpy(), evaluate_hands(right).numpy()

    judger = Judger()
    for k in range(n):
        expected = ((1, 0) if ls[k] > rs[k] else
                    (0, 1) if ls[k] < rs[k] else (1, 1))
        assert judger.compare_hands(left[k].numpy(), right[k].numpy()) == expected


# ------------------------------------------------------------- the baseline


def test_on_a_complete_board_the_baseline_is_the_settlement():
    """With nobody short, `b` on a finished board *is* what the engine pays out.

    That is what makes the run-out corrections cancel exactly: the estimator
    reports `R − b(final) + b(all-in state)`, and where the two first terms are
    the same number what is left is the average over every runout. Random
    multiway states, every live seat matched, checked against `Judger` itself.
    """
    judger = Judger()
    rng = np.random.default_rng(5)
    for _ in range(60):
        n = int(rng.integers(2, 6))
        deck = rng.permutation(52)
        players_state = np.where(rng.random(n) < 0.3, -1.0, 2.0)
        players_state[rng.permutation(n)[:2]] = 2.0   # two seats always show
        live = players_state >= 0
        # Everybody who is still in has matched everybody else, which is the
        # case with no side pot; folded seats may have put in anything.
        bets = np.round(rng.random(n) * 40.0, 2)
        bets[live] = 100.0

        holes = deck[5:5 + 2 * n].reshape(n, 2)
        scores = seat_scores(deck[:5], holes, np.zeros((1, 0), dtype=np.int64))
        got = equity_baseline(scores, live, bets)
        want = judger.get_reward(deck, players_state, bets)
        assert np.allclose(got, want, rtol=0, atol=1e-9)


def test_the_baseline_is_not_a_settlement_when_somebody_is_short():
    """The declared limit: a side pot is priced as if the short seat could win it.

    Recorded as a test because it is the one case where "integrate the cards
    out" stops being exact — the reduction is partial there, and only the
    reduction. Nothing about it can move a label: the correction is zero-mean
    whatever the baseline says.
    """
    judger = Judger()
    rng = np.random.default_rng(9)
    deck = rng.permutation(52)
    n = 3
    live = np.array([True, True, True])
    bets = np.array([100.0, 20.0, 100.0])       # seat 1 is all-in for less
    holes = deck[5:5 + 2 * n].reshape(n, 2)
    scores = seat_scores(deck[:5], holes, np.zeros((1, 0), dtype=np.int64))

    baseline = equity_baseline(scores, live, bets)
    settled = judger.get_reward(deck, np.array([2.0, 2.0, 2.0]), bets)
    assert np.isclose(baseline.sum(), 0.0)      # it still conserves chips
    if int(np.argmax(scores[0])) == 1:          # the short seat wins the pot
        assert not np.allclose(baseline, settled, rtol=0, atol=1e-9)


def test_the_baseline_cannot_see_chips_beyond_the_call():
    """A raise, an all-in and a call are worth the same once betting freezes.

    Unmatched chips come back, so they cannot change anybody's expectation —
    which is what lets the correction at a decision be computed from the call
    amount alone instead of from every raise size on the grid. Multiway, with
    unequal contributions, so the side-pot path is the one being asked.
    """
    judger = Judger()
    rng = np.random.default_rng(6)
    deck = rng.permutation(52)
    n = 3
    holes = deck[5:5 + 2 * n].reshape(n, 2)
    scores = seat_scores(deck[:4], holes, board_completions(
        deck[:4], holes.reshape(-1), 1, rng, RunoutConfig(samples=64)))
    live = np.array([True, True, True])

    base = np.array([100.0, 60.0, 100.0])
    called = base.copy()
    called[1] += 40.0                      # seat 1 calls to the high bet
    raised = called.copy()
    raised[1] += 250.0                     # ...or raises, unmatched by anybody
    shoved = called.copy()
    shoved[1] += 900.0

    want = equity_baseline(scores, live, called)
    for variant in (raised, shoved):
        assert np.allclose(equity_baseline(scores, live, variant), want,
                           rtol=0, atol=1e-9)


def test_a_fold_out_is_worth_the_pot_and_needs_no_board():
    """One seat left is a closed form, not an average — and the engine agrees.

    The engine settles a fold-out itself rather than through `Judger`: the last
    seat collects the pot, so its delta is what everybody else put in. That is
    the convention the baseline has to share, and it is checked here against a
    hand the driver actually played to a fold-out.
    """
    bets = np.array([30.0, 30.0, 12.0])
    got = equity_baseline(None, np.array([False, True, False]), bets)
    assert np.allclose(got, [-30.0, 42.0, -12.0])
    assert np.isclose(got.sum(), 0.0)

    pool = [Coin("coin", N_ACTIONS, StyleParams.identity())]
    played = LockstepDriver(pool, N_ACTIONS).run([_spec(77, forced=[FOLD])])[0]
    live = np.array([False, True])
    contributed = np.array([SMALL_BLIND, BIG_BLIND])
    assert np.allclose(equity_baseline(None, live, contributed),
                       played.rewards, rtol=0, atol=1e-9)


# ----------------------------------------------------- the estimator, end to end


def test_a_hand_with_no_decisions_left_is_worth_its_average_runout():
    """The exact identity the whole mechanism rests on.

    Both seats are all-in on the turn, so the river is the only thing left
    undecided. The reduced value has to be the mean over all 44 of them — and
    the raw one is a whole stack away from it, which is the noise being removed.
    """
    pool = [Coin("coin", N_ACTIONS, StyleParams.identity())]
    plain = LockstepDriver(pool, N_ACTIONS)
    reduced = LockstepDriver(pool, N_ACTIONS, runout=RunoutConfig(samples=64))

    # 4005 is a hand whose river decides it: the raw result is a full stack
    # either way, so an estimator that merely tracked the outcome would pass
    # nothing here.
    record = reduced.run([_spec(4005, forced=TO_TURN_SHOVE + [CALL])])[0]
    assert record.baseline_rewards is not None

    want = np.mean([plain.run([_spec(4005, deck=d,
                                     forced=TO_TURN_SHOVE + [CALL])])[0].rewards
                    for d in _every_river(record.deck)], axis=0)
    assert np.allclose(record.baseline_rewards, want, rtol=0, atol=1e-9)
    assert not np.allclose(record.rewards, want, rtol=0, atol=1.0)
    assert np.isclose(record.rewards.sum(), 0.0)


def test_the_fold_draw_is_subtracted_without_moving_the_mean():
    """Both branches of a 50-50 fold, weighted, hit the exact expectation.

    The raw pair does not: one branch wins the blinds and the other wins a
    stack, and their mean is nowhere near what the decision is actually worth.
    The reduced pair lands on it exactly, which is the zero-mean property of
    the correction and its sign at the same time.
    """
    pool = [Coin("coin", N_ACTIONS, StyleParams.identity())]
    plain = LockstepDriver(pool, N_ACTIONS)
    reduced = LockstepDriver(pool, N_ACTIONS, runout=RunoutConfig(samples=64))

    deck = reduced.run([_spec(4005)])[0].deck
    branches = {}
    for seed in range(1, 400):
        record = reduced.run([_spec(seed, deck=deck)])[0]
        branches.setdefault(record.decisions[-1]["action_idx"], record)
        if len(branches) == 2:
            break
    assert set(branches) == {FOLD, CALL}, "the 50-50 draw never went both ways"

    folded = plain.run([_spec(1, deck=deck,
                              forced=TO_TURN_SHOVE + [FOLD])])[0].rewards
    called = np.mean(
        [plain.run([_spec(1, deck=d, forced=TO_TURN_SHOVE + [CALL])])[0].rewards
         for d in _every_river(deck)], axis=0)
    want = 0.5 * folded + 0.5 * called

    got = 0.5 * (branches[FOLD].baseline_rewards
                 + branches[CALL].baseline_rewards)
    assert np.allclose(got, want, rtol=0, atol=1e-9)

    raw = 0.5 * (branches[FOLD].rewards + branches[CALL].rewards)
    assert not np.allclose(raw, want, rtol=0, atol=1.0)


def test_cards_the_forced_prefix_turned_over_carry_no_correction():
    """A rollout's prefix replays what hero already saw, so it holds no luck.

    Correcting for the flop a labelled decision was *taken on* would subtract a
    number whose mean is not zero, which is a bias rather than a reduction. The
    check is that hero's value is untouched by how the prefix got there: two
    labelled decisions on the same turn card, reached through prefixes that
    differ only in cards already visible, must both report the average runout.
    """
    pool = [Coin("coin", N_ACTIONS, StyleParams.identity())]
    plain = LockstepDriver(pool, N_ACTIONS)
    reduced = LockstepDriver(pool, N_ACTIONS, runout=RunoutConfig(samples=64))

    for seed in (4000, 4004, 4005):
        record = reduced.run([_spec(seed, forced=TO_TURN_SHOVE + [CALL])])[0]
        want = np.mean(
            [plain.run([_spec(seed, deck=d,
                              forced=TO_TURN_SHOVE + [CALL])])[0].rewards
             for d in _every_river(record.deck)], axis=0)
        assert np.allclose(record.baseline_rewards, want, rtol=0, atol=1e-9)


def test_the_estimator_does_not_change_the_hand():
    """Same deck, same decisions, same chips — only an extra field."""
    pool = make_pool(0)
    specs = [HandSpec(num_players=int(2 + h % 4),
                      start_credits=[1000.0] * int(2 + h % 4),
                      seat_members=[(h + s) % len(pool)
                                    for s in range(int(2 + h % 4))],
                      seed=31_000 + h, big_blind=BIG_BLIND,
                      small_blind=SMALL_BLIND, raise_sizes=RAISE_SIZES,
                      meta={"hand": h})
             for h in range(24)]

    plain = LockstepDriver(pool, N_ACTIONS).run(specs)
    reduced = LockstepDriver(pool, N_ACTIONS,
                             runout=RunoutConfig(samples=32)).run(specs)
    for a, b in zip(plain, reduced):
        assert np.array_equal(a.deck, b.deck)
        assert np.allclose(a.rewards, b.rewards, rtol=0, atol=0)
        assert ([d["action_idx"] for d in a.decisions]
                == [d["action_idx"] for d in b.decisions])
        assert a.baseline_rewards is None
        assert b.baseline_rewards is not None
        assert np.isclose(b.rewards.sum(), 0.0)


@pytest.mark.parametrize("num_players", [2, 6])
def test_a_label_does_not_depend_on_how_its_rollouts_were_batched(num_players):
    """The rankings are shared between hands dealt the same cards, and one
    label is exactly that case: every legal action is rolled out on the same
    deck, an action changing the forced prefix and nothing else.

    That sharing is the one place the batch could leak into the answer. If the
    boards a street averages over were chosen by whichever hand happened to
    rank it first, `q` would move with `batch_hands` — so this is the test that
    the completions come from the cards that street has already shown and from
    nothing else (`env/runout.py`, §15).
    """
    pool = make_pool(0)
    records = play(pool, make_specs(seed=17, n_hands=6, n_members=len(pool),
                                    num_players=num_players, stack_bb=120))
    record = next(r for r in records if len(r.decisions) >= 2)

    reference = None
    for batch_hands in (1, 7, 4096):
        cfg = OracleConfig(samples_per_action=24, batch_hands=batch_hands,
                           control_variate=True, runout_samples=8)
        driver = LockstepDriver(pool, N_ACTIONS, runout=cfg.runout_config())
        q, _legal, _stats = action_values(record, 1, driver, pool, 0, cfg,
                                          np.random.default_rng(5))
        if reference is None:
            reference = q
        np.testing.assert_array_equal(q, reference)


def test_folding_hero_keeps_its_exact_closed_form():
    """`q[FOLD]` is minus what hero put in — and stays bit-exact with this on.

    Hero's chip delta after folding carries no variance to remove, so every
    correction taken after hero is out of the hand must land on hero's seat as
    exactly zero. It does, for a reason worth stating: once hero has folded,
    hero's own matched contribution can no longer move — every later chip
    belongs to somebody still in — so the baseline's hero entry is a constant
    and every difference of it vanishes. Checked through the oracle's own entry
    point at three table sizes, against the closed form and not a tolerance.
    """
    pool = make_pool(0)
    cfg = OracleConfig(samples_per_action=8, likelihood_floor=1e-6)
    checked = 0
    for n in (2, 4, 9):
        for record in play(pool, make_specs(seed=5, n_hands=8,
                                            n_members=len(pool),
                                            num_players=n, stack_bb=100)):
            for d, decision in enumerate(record.decisions):
                if not decision["legal_mask"][FOLD]:
                    continue
                hero = int(decision["acting_pos"])
                snapshot = record.snapshots[decision["snap_idx"]]
                invested = (record.spec.start_credits[hero]
                            - snapshot["credits"][hero])
                driver = LockstepDriver(pool, N_ACTIONS,
                                        runout=RunoutConfig(samples=16))
                q, _legal, _stats = action_values(
                    record, d, driver, pool, 0, cfg,
                    np.random.default_rng([1, d, n]))
                assert q[FOLD] == -invested / record.spec.big_blind
                checked += 1
                break
    assert checked >= 12


# --------------------------------------------------- per-street raise-size grids


def test_a_street_may_define_fewer_raise_sizes_than_another():
    """Streets carry their own grids; the widest one fixes the action layout.

    A shorter street leaves its trailing bins illegal, which is how every other
    unavailable action is already expressed — so nothing downstream of the mask
    has to know that the grids differ.
    """
    raise_sizes = {0: [1.0, 2.0, 3.0, 4.0], 1: [0.5, 1.0], 2: [0.5], 3: [1.0]}
    table = Table(num_players=2, raise_sizes=raise_sizes,
                  start_credits=[1000.0, 1000.0], big_blind=BIG_BLIND,
                  small_blind=SMALL_BLIND)
    np.random.seed(0)
    table.start_table()
    assert table.n_raise_bins == 4

    n_actions = table.n_raise_bins + 3
    def step(idx):
        onehot = np.zeros(n_actions, dtype=np.float32)
        onehot[idx] = 1.0
        return table.step(torch.from_numpy(onehot))

    # Preflop offers all four bins plus the all-in slot; the flop offers two.
    assert legal_actions(table) == [0, 1, 2, 3, 4, 5, 6]
    step(CALL)
    step(CALL)
    assert table.turn == 1
    assert legal_actions(table) == [1, 2, 3, 6]
    step(CALL)
    step(CALL)
    assert table.turn == 2
    assert legal_actions(table) == [1, 2, 6]
