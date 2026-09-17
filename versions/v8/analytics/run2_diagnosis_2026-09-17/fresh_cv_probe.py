"""CPU prototype: fresh auxiliary runout randomness per outer sample.

Production modules are not edited or monkey-patched. The subclasses are an
experiment seam; a production fix should carry a dedicated runout seed in
HandSpec and include it in the existing cache key.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce import (ROOT, Caller, DiagnosticOpponent, every_river,
                       independent_payoff)
import numpy as np
from env.driver import HandSpec, LockstepDriver
from env.runout import HandRunout, RunoutConfig
from oracle.rollout import OracleConfig, _label_plan, _label_q


class FreshRunout(HandRunout):
    def __init__(self, deck, num_players, cfg, auxiliary_seed, scores=None):
        super().__init__(deck, num_players, cfg, scores=scores)
        self.auxiliary_seed = int(auxiliary_seed)
        self.key = (*self.key, self.auxiliary_seed)

    def _rng(self, known, holes):
        digest = hashlib.blake2b(
            np.asarray(known, dtype=np.int64).tobytes() + holes.tobytes()
            + bytes([self.num_players]), digest_size=8).digest()
        return np.random.default_rng([
            self.auxiliary_seed, int.from_bytes(digest, "big"), 0x5EED])


class FreshDriver(LockstepDriver):
    runout_class = FreshRunout

    def _start(self, idx, spec, scores=None):
        state = super()._start(idx, spec, scores=scores)
        if self.runout is not None:
            state["runout"] = self.runout_class(
                state["record"].deck, spec.num_players, self.runout,
                spec.meta["cv_seed"], scores=scores)
        return state


class EnumeratedRunout(FreshRunout):
    """Finite exact check of the auxiliary expectation at M=16.

    Uniformly choosing one of 44 offsets gives each of 16 entries a uniform
    marginal over the 44 legal rivers. Enumerating those offsets checks the
    expectation exactly, rather than asserting a statistical tolerance.
    """

    def _job(self, turn):
        known, holes, completions = super()._job(turn)
        if turn == 2:
            dead = set(known) | set(holes.reshape(-1))
            rivers = np.array([c for c in range(52) if c not in dead])
            offsets = (self.auxiliary_seed + np.arange(self.cfg.samples)) % len(rivers)
            completions = rivers[offsets, None]
        return known, holes, completions


class EnumeratedDriver(FreshDriver):
    runout_class = EnumeratedRunout


def fixture():
    game = json.loads((ROOT / "config.json").read_text())["game"]
    grid = [game["raise_sizes"][s] for s in ("preflop", "flop", "turn", "river")]
    pool = [Caller("hero", 17), DiagnosticOpponent("diagnostic", 17)]
    spec = HandSpec(
        num_players=2, start_credits=[2000., 2000.], seat_members=[0, 1],
        seed=4005, big_blind=10., small_blind=5., raise_sizes=grid,
        deck=np.random.RandomState(4005).permutation(52),
        forced_actions=[15, 6, 1, 11, 7, 1, 16, 1])
    record = LockstepDriver(pool, 17).run([spec])[0]
    return pool, spec, record


def label(pool, record, samples, seed, fresh=True, batch_size=2048):
    # Two independent PRNG streams: actual rollout randomness and auxiliary
    # equity estimation. Repeating an independent label changes both streams.
    child_seeds = np.random.SeedSequence(seed).spawn(2)
    game_rng, auxiliary_rng = [np.random.default_rng(s) for s in child_seeds]
    cfg = OracleConfig(samples_per_action=samples, likelihood_floor=0.,
                       control_variate=True, runout_samples=16,
                       batch_hands=batch_size)
    driver = (FreshDriver if fresh else LockstepDriver)(
        pool, 17, runout=cfg.runout_config())
    plan = _label_plan(record, 7, driver, pool, 0, cfg, game_rng)
    count = len(plan.legal_idx)
    seeds = auxiliary_rng.integers(0, 2**63, size=plan.n_samples)
    plan.specs = [replace(spec, meta={**spec.meta, "cv_seed": int(seeds[i // count])})
                  for i, spec in enumerate(plan.specs)]
    played = driver.run(plan.specs, batch_size=batch_size)
    q, legal, _ = _label_q(plan, played, 0.)
    return q[legal].tolist()


def exact_auxiliary_checks(pool, spec, record):
    out = {}
    for mode in ("all_in", "ordinary_turn_to_river"):
        base = spec if mode == "all_in" else replace(
            spec, seat_members=[0, 0], forced_actions=[15, 6, 1, 11, 7, 1, 1, 1])
        decks = list(every_river(record.deck))
        raw = LockstepDriver(pool, 17).run([replace(base, deck=d) for d in decks])
        reference = float(np.mean([r.rewards[0] for r in raw]) / 10.)
        specs = [replace(base, deck=d, meta={"cv_seed": offset})
                 for d in decks for offset in range(44)]
        played = EnumeratedDriver(pool, 17, runout=RunoutConfig(samples=16)).run(
            specs, batch_size=256)
        values = np.array([r.baseline_rewards[0] / 10. for r in played]).reshape(44, 44)
        # Conditional on any actual river, averaging the independent auxiliary
        # draws equals the exact river-integrated result in these checkdowns.
        np.testing.assert_allclose(values.mean(axis=1), reference, atol=1e-10)
        out[mode] = {"exact_ev_bb": reference, "corrected_mean_bb": float(values.mean()),
                     "actual_rivers": 44, "auxiliary_offsets": 44,
                     "M": 16, "max_error_after_auxiliary_average_bb":
                     float(np.max(np.abs(values.mean(axis=1) - reference)))}
    return out


if __name__ == "__main__":
    pool, spec, record = fixture()
    exact = float(np.mean([independent_payoff(d) for d in every_river(record.deck)]))
    result = {"exact_call_bb": exact, "exact_auxiliary_checks":
              exact_auxiliary_checks(pool, spec, record), "label_examples": [],
              "independent_replicates": {}}
    for samples in (128, 1024):
        result["label_examples"].append({"S": samples,
            "old_q_fold_call": label(pool, record, samples, 7, fresh=False),
            "fresh_q_fold_call": label(pool, record, samples, 7)})
    for samples, repetitions in ((128, 32), (1024, 8)):
        values = np.array([label(pool, record, samples, 1000 + i)[1]
                           for i in range(repetitions)])
        p = 8 / 44
        result["independent_replicates"][str(samples)] = {
            "repetitions": repetitions, "mean_call_bb": float(values.mean()),
            "sample_sd_call_bb": float(values.std(ddof=1)),
            "theoretical_sd_call_bb": float(400 * np.sqrt(p * (1-p) / (samples * 16))),
            "values_bb": values.tolist()}
    batches = [label(pool, record, 16, 123, batch_size=b) for b in (1, 7, 4096)]
    np.testing.assert_array_equal(batches, np.repeat([batches[0]], 3, axis=0))
    result["batch_invariance"] = {"batches": [1, 7, 4096], "q": batches[0], "passed": True}
    print(json.dumps(result, indent=2))
