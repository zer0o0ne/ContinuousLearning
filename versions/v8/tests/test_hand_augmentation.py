"""Hand-owned policies must agree in play, posterior, rollouts and resume."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from env.driver import DecisionContext, LockstepDriver
from env.session import Session
from pool.augmentation import (augment_sessions, augmentation_probability,
                               augmentation_signature)
from pool.degenerate import AlwaysCall
from pool.style import StyleParams, sample_style
from tests.g1_fixtures import N_ACTIONS, STYLE_CFG, make_specs


STYLE = dict(STYLE_CFG, augmentation_zero_iteration=10)


@pytest.mark.parametrize("iteration,expected", [
    (0, 1.0), (2, 0.8), (5, 0.5), (8, 0.2),
    (10, 0.0), (11, 0.0), (100, 0.0),
])
def test_linear_probability(iteration, expected):
    assert augmentation_probability(STYLE, iteration) == pytest.approx(expected)


@pytest.mark.parametrize("end", [-1, 2.5, True, "10", None])
def test_invalid_endpoint_is_rejected(end):
    with pytest.raises(ValueError, match="augmentation_zero_iteration"):
        augmentation_probability(dict(STYLE, augmentation_zero_iteration=end), 0)


def _sessions():
    # Every seat is the same member: draws must still be independent.
    return [Session(0, 3, 80, [0, 0, 0], specs=make_specs(
        n_hands=12, num_players=3, stack_bb=80))]


def test_draws_are_per_hand_and_seat_reproducible_and_keep_rotating_hero_plain():
    sessions = _sessions()
    original = copy.deepcopy(sessions)
    augment_sessions(sessions, STYLE, 0, exclude_slot=0)
    again = copy.deepcopy(original)
    augment_sessions(again, STYLE, 0, exclude_slot=0)
    identity = StyleParams.identity().to_list()
    draws = []
    for h, spec in enumerate(sessions[0].specs):
        np.testing.assert_array_equal(spec.style_augmentations,
                                      again[0].specs[h].style_augmentations)
        assert spec.seed == original[0].specs[h].seed
        assert spec.seat_members == original[0].specs[h].seat_members
        for seat in range(3):
            row = spec.style_augmentations[seat]
            if seat == sessions[0].seat_of_slot(0, h):
                np.testing.assert_array_equal(row, identity)
            else:
                assert not np.array_equal(row, identity)
                draws.append(tuple(row))
    assert len(set(draws)) == len(draws)
    augment_sessions(again, STYLE, 0)  # corpus: even the observer qualifies
    assert all(not np.array_equal(row, identity)
               for spec in again[0].specs for row in spec.style_augmentations)


def test_mid_schedule_independently_selects_seats_and_keeps_selected_draws():
    sessions = _sessions()
    full = copy.deepcopy(sessions)
    augment_sessions(full, STYLE, 0)
    augment_sessions(sessions, STYLE, 5)
    selected = []
    identity = StyleParams.identity().to_list()
    for spec, all_spec in zip(sessions[0].specs, full[0].specs):
        for seat in range(3):
            expected = np.random.default_rng([spec.seed, seat, 7351]).random() < 0.5
            row = (identity if spec.style_augmentations is None
                   else spec.style_augmentations[seat])
            np.testing.assert_array_equal(
                row, all_spec.style_augmentations[seat] if expected else identity)
            selected.append(expected)
    assert any(selected) and not all(selected)


def test_disabled_and_finished_schedule_preserve_the_original_hands():
    sessions = _sessions()
    driver = LockstepDriver([AlwaysCall("call", N_ACTIONS)], N_ACTIONS)
    before = driver.run(sessions[0].specs, batch_size=4)
    for cfg, iteration in [({}, 0), (dict(STYLE, augmentation_zero_iteration=0), 0),
                           (STYLE, 10), (STYLE, 100)]:
        augment_sessions(sessions, cfg, iteration)
        assert augmentation_signature(cfg, iteration) is None
        assert all(spec.style_augmentations is None for spec in sessions[0].specs)
        after = driver.run(sessions[0].specs, batch_size=9)
        assert [[d["action_idx"] for d in r.decisions] for r in after] == [
            [d["action_idx"] for d in r.decisions] for r in before]


def test_augmentation_composes_with_permanent_style_in_one_batched_forward():
    base = sample_style(np.random.default_rng(3), STYLE)
    aug = sample_style(np.random.default_rng(8), STYLE)
    member = AlwaysCall("call", N_ACTIONS, base)
    contexts = [SimpleNamespace(
        legal_mask=np.array([False] + [True] * (N_ACTIONS - 1)),
        acting_pos=seat, num_players=3, turn=street,
        style_augmentation=np.array(aug.to_list()) if selected else None)
        for selected in (False, True) for street in range(4) for seat in range(3)]
    composed = StyleParams(
        base.uncond + aug.uncond, base.position + aug.position,
        base.street + aug.street, base.temperature * aug.temperature,
        base.uniform_mix + (1 - base.uniform_mix) * aug.uniform_mix)
    clean = [SimpleNamespace(**{k: v for k, v in vars(c).items()
                                if k != "style_augmentation"}) for c in contexts]
    logits = member.logits(clean)
    legal = np.stack([c.legal_mask for c in clean])
    expected = np.concatenate([
        base.apply(logits[:12], legal[:12], clean[:12]),
        composed.apply(logits[12:], legal[12:], clean[12:])])
    np.testing.assert_allclose(member.policy(contexts), expected, atol=1e-15)
    assert member.style is base
    np.testing.assert_array_equal(base.to_list(), sample_style(
        np.random.default_rng(3), STYLE).to_list())


def test_real_play_and_posterior_contexts_keep_the_same_style_in_rollouts():
    from oracle.rollout import OracleConfig, _label_plan

    sessions = _sessions()
    augment_sessions(sessions, STYLE, 0)
    member = AlwaysCall("call", N_ACTIONS)
    driver = LockstepDriver([member], N_ACTIONS)
    specs = sessions[0].specs
    one = driver.run(specs, batch_size=1)
    many = driver.run(specs, batch_size=12)
    for a, b in zip(one, many):
        assert [d["action_idx"] for d in a.decisions] == [d["action_idx"] for d in b.decisions]
        np.testing.assert_array_equal(a.rewards, b.rewards)
    record = one[0]
    plan = _label_plan(record, 0, driver, [member], 0,
                       OracleConfig(samples_per_action=1, max_combos=2),
                       np.random.default_rng(7))
    assert plan.specs
    assert all(spec.style_augmentations is record.spec.style_augmentations
               for spec in plan.specs)
    dec = record.decisions[0]
    def context(rec, cards=None):
        return DecisionContext(rec, dec["snap_idx"], dec["acting_pos"],
                               dec["legal_mask"], 0, hole_override=cards)
    # Changed rollout RNG and hypothetical cards cannot change the style.
    original = context(record)
    hypothetical = context(record, [1, 2])
    rollout = context(replace(record, spec=plan.specs[0]))
    np.testing.assert_array_equal(member.policy([original]), member.policy([hypothetical]))
    np.testing.assert_array_equal(member.policy([original]), member.policy([rollout]))


def test_augmented_labels_match_in_workers_and_resume(tmp_path):
    from tests.test_label_generation import _cfg, _run
    from tests.test_parallel_labels import _scripted_hero

    cfg = _cfg(n_sessions=2, hands=4, R=2)
    cfg.update(style=STYLE, iteration=5)
    seq, labels = _run(tmp_path / "seq", cfg=cfg, hero=_scripted_hero())
    assert labels
    cfg["oracle"]["n_workers"] = 2
    par, _ = _run(tmp_path / "par", cfg=cfg, hero=_scripted_hero())
    def contents(manifest):
        return [open(path, "rb").read() for path in manifest["shards"]]
    assert contents(seq) == contents(par)
    resumed, _ = _run(tmp_path / "par", cfg=cfg, hero=_scripted_hero())
    assert contents(seq) == contents(resumed)
    assert not (tmp_path / "par.stale").exists()
    assert par["style_augmentation"] == augmentation_signature(STYLE, 5)
    # Turning it off must invalidate the partial/full play cache too.
    cfg["iteration"] = 10
    plain, _ = _run(tmp_path / "par", cfg=cfg, hero=_scripted_hero())
    assert (tmp_path / "par.stale" / "play.json").exists()
    assert plain.get("style_augmentation") is None
    play = json.loads((tmp_path / "par" / "play.json").read_text())
    assert play["style_augmentation"] is None


def test_pipeline_uses_both_schedules_and_disjoint_seeds(tmp_path, monkeypatch):
    import pipeline
    import train.generate
    from env.session import phase_hands
    from tests.test_pipeline import _run, toy_config

    cfg = toy_config(n_sessions=1, n_sessions_first_iteration=3)
    cfg["style"]["augmentation_zero_iteration"] = 1
    seen = {}
    real = augment_sessions

    def inspect(sessions, style, iteration, **kwargs):
        real(sessions, style, iteration, **kwargs)
        phase = "labels" if kwargs.get("exclude_slot") == 0 else "corpus"
        specs = [spec for session in sessions for spec in session.specs]
        seen[phase, iteration] = {spec.seed for spec in specs}
        for session in sessions:
            for h, spec in enumerate(session.specs):
                if iteration == 1:
                    assert spec.style_augmentations is None
                else:
                    assert spec.style_augmentations is not None
                    if phase == "labels":
                        np.testing.assert_array_equal(
                            spec.style_augmentations[session.seat_of_slot(0, h)],
                            StyleParams.identity().to_list())

    monkeypatch.setattr(pipeline, "augment_sessions", inspect)
    monkeypatch.setattr(train.generate, "augment_sessions", inspect)
    _, directory = _run(tmp_path, cfg)
    from pathlib import Path
    for iteration, count in enumerate((3, 1)):
        manifest = json.loads((Path(directory) / f"iter_{iteration:04d}"
                               / "labels.json").read_text())["manifest"]
        assert manifest["n_sessions"] == count
        assert manifest["n_hands"] == count * cfg["hands_per_session"]
        assert manifest.get("style_augmentation") == augmentation_signature(
            cfg["style"], iteration)
    assert len(seen) == 4
    assert len(set().union(*seen.values())) == sum(map(len, seen.values()))
    assert phase_hands(cfg)["labels"] == 3 * cfg["hands_per_session"]
    cfg["n_sessions_first_iteration"] = 1
    cfg["n_sessions"] = 4
    assert phase_hands(cfg)["labels"] == 4 * cfg["hands_per_session"]


def test_changed_augmentation_and_first_session_count_rebuild_untrained_labels(tmp_path):
    from pathlib import Path
    import pipeline
    from tests.test_pipeline import _run, toy_config

    cfg = toy_config(n_iterations=1, n_sessions=1)
    _, directory = _run(tmp_path, cfg)
    iteration = Path(directory) / "iter_0000"
    for name in ("state.json", "metrics.json", "agent.pt"):
        (iteration / name).unlink()
    cfg["style"]["augmentation_zero_iteration"] = 10
    cfg["n_sessions_first_iteration"] = 3
    pipeline.run(cfg, lambda _: None, directory)
    manifest = json.loads((iteration / "labels.json").read_text())["manifest"]
    assert manifest["n_sessions"] == 3
    assert manifest["style_augmentation"] == augmentation_signature(cfg["style"], 0)
    assert (iteration / "labels.stale" / "progress.json").exists()


def test_partial_augmented_labels_resume_byte_for_byte(tmp_path, monkeypatch):
    from pathlib import Path
    import train.generate
    from tests.test_label_generation import _cfg, _run
    from tests.test_parallel_labels import _scripted_hero

    cfg = _cfg(n_sessions=2, hands=4, R=2)
    cfg.update(style=STYLE, iteration=3, labels_per_shard=1)
    whole, _ = _run(tmp_path / "whole", cfg=cfg, hero=_scripted_hero())
    assert len(whole["shards"]) >= 2
    real_write = train.generate._write_npz
    def crash(path, arrays):
        if Path(path).name == "shard_0001.npz":
            raise RuntimeError("interrupted")
        return real_write(path, arrays)
    monkeypatch.setattr(train.generate, "_write_npz", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(tmp_path / "partial", cfg=cfg, hero=_scripted_hero())
    monkeypatch.setattr(train.generate, "_write_npz", real_write)
    resumed, _ = _run(tmp_path / "partial", cfg=cfg, hero=_scripted_hero())
    assert [Path(p).read_bytes() for p in resumed["shards"]] == [
        Path(p).read_bytes() for p in whole["shards"]]
