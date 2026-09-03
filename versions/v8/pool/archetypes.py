"""Ten recognisable players, as ten sets of numbers
(`PLAN_PROCEDURAL_POOL.md` §P4).

`pool/regular.py` is one cascade; this file is the only reason there are ten of
them rather than one. Every number below is a *design hypothesis* — nobody
measured them — and the gate beside this file is what turns them into evidence.

**Six of them lie in one plane** and the other four are there to break it. The
first six differ by tightness, aggression and how much of their betting is
bluff, and they share three regularities that an agent could learn once and
apply to all of them: a bet correlates with strength the same way, sizes stay
between half a pot and a pot, and position bends every range by the same shape.

* `weak_tight` breaks the first — it enters as many pots as a loose-passive and
  then folds instead of calling, which is the most common live population type
  and the largest exploit surface in the pool;
* `trapper` breaks "a check means weakness", which is true in the other nine
  and false against anything balanced;
* `polar_reg` breaks the size axis: it is the only one that overbets, and an
  agent that never met a pot-and-a-half bet in training meets its first one at
  the benchmark;
* `stealer` breaks the position curve, opening 7 % from the first seat and 70 %
  from the button.

**Jitter is where the diversity beyond ten comes from.** A rate is multiplied by
a lognormal draw, so a zero stays zero — an archetype that never bluffs is not
jittered into bluffing — and a size or a stack threshold is shifted additively,
because a size of 0.33 and a size of 1.5 want the same absolute spread, not the
same relative one. Everything is clipped back into its domain afterwards.
"""

from dataclasses import fields, replace

import numpy as np

from pool.regular import RegularParams

#: The ten presets. Read a column as "the shape I mean", not as a measurement.
ARCHETYPES = {
    "nit": RegularParams(
        open_early=0.06, open_late=0.16, limp_share=0.3, call_open=0.08,
        threebet_value=0.03, threebet_bluff=0.0, call_threebet=0.03,
        fourbet=0.015, open_size_bb=3.0, threebet_mult=3.5, push_fold_bb=8,
        value_hs=0.88, cbet_dry=0.35, cbet_wet=0.30, oop_factor=0.7,
        size_dry=0.5, size_wet=0.6, size_river=0.5,
        barrel_turn=0.1, barrel_river=0.0, bluff_ratio=0.1, semi_bluff=0.15,
        defend_factor=0.55, raise_value=0.9, raise_bluff=0.0,
        slowplay=0.3, donk=0.0, overbet=0.0, allin_spr=1.0,
        multiway_tighten=0.15),
    "loose_passive": RegularParams(
        open_early=0.35, open_late=0.60, limp_share=0.8, call_open=0.45,
        threebet_value=0.03, threebet_bluff=0.0, call_threebet=0.20,
        fourbet=0.02, open_size_bb=2.0, threebet_mult=3.0, push_fold_bb=6,
        value_hs=0.65, cbet_dry=0.30, cbet_wet=0.25, oop_factor=0.9,
        size_dry=0.33, size_wet=0.5, size_river=0.5,
        barrel_turn=0.1, barrel_river=0.05, bluff_ratio=0.1, semi_bluff=0.10,
        defend_factor=1.35, raise_value=0.3, raise_bluff=0.0,
        slowplay=0.2, donk=0.3, overbet=0.0, allin_spr=1.5,
        multiway_tighten=0.05),
    "loose_passive_bluffy": RegularParams(
        open_early=0.35, open_late=0.60, limp_share=0.8, call_open=0.45,
        threebet_value=0.03, threebet_bluff=0.02, call_threebet=0.20,
        fourbet=0.02, open_size_bb=2.0, threebet_mult=3.0, push_fold_bb=6,
        value_hs=0.65, cbet_dry=0.35, cbet_wet=0.30, oop_factor=0.9,
        size_dry=0.33, size_wet=0.5, size_river=0.5,
        barrel_turn=0.3, barrel_river=0.4, bluff_ratio=0.9, semi_bluff=0.25,
        defend_factor=1.35, raise_value=0.3, raise_bluff=0.15,
        slowplay=0.2, donk=0.3, overbet=0.0, allin_spr=1.5,
        multiway_tighten=0.05),
    "maniac": RegularParams(
        open_early=0.60, open_late=0.90, limp_share=0.0, call_open=0.10,
        threebet_value=0.25, threebet_bluff=0.30, call_threebet=0.30,
        fourbet=0.20, open_size_bb=4.0, threebet_mult=4.0, push_fold_bb=25,
        value_hs=0.50, cbet_dry=0.95, cbet_wet=0.95, oop_factor=1.0,
        size_dry=1.0, size_wet=1.25, size_river=1.5,
        barrel_turn=0.9, barrel_river=0.9, bluff_ratio=2.5, semi_bluff=0.9,
        defend_factor=1.3, raise_value=0.9, raise_bluff=0.6,
        slowplay=0.0, donk=0.5, overbet=0.5, allin_spr=4.0,
        multiway_tighten=0.0),
    "tag": RegularParams(
        open_early=0.14, open_late=0.45, limp_share=0.0, call_open=0.15,
        threebet_value=0.06, threebet_bluff=0.04, call_threebet=0.08,
        fourbet=0.025, open_size_bb=2.5, threebet_mult=3.2, push_fold_bb=12,
        value_hs=0.75, cbet_dry=0.75, cbet_wet=0.55, oop_factor=0.75,
        size_dry=0.33, size_wet=0.67, size_river=0.75,
        barrel_turn=0.55, barrel_river=0.45, bluff_ratio=1.0, semi_bluff=0.55,
        defend_factor=1.0, raise_value=0.6, raise_bluff=0.2,
        slowplay=0.1, donk=0.05, overbet=0.1, allin_spr=1.5,
        multiway_tighten=0.12),
    "bluffer": RegularParams(
        open_early=0.18, open_late=0.55, limp_share=0.0, call_open=0.15,
        threebet_value=0.06, threebet_bluff=0.10, call_threebet=0.10,
        fourbet=0.04, open_size_bb=2.5, threebet_mult=3.2, push_fold_bb=12,
        value_hs=0.72, cbet_dry=0.85, cbet_wet=0.70, oop_factor=0.85,
        size_dry=0.4, size_wet=0.75, size_river=1.0,
        barrel_turn=0.75, barrel_river=0.70, bluff_ratio=1.8, semi_bluff=0.70,
        defend_factor=1.05, raise_value=0.55, raise_bluff=0.4,
        slowplay=0.05, donk=0.1, overbet=0.25, allin_spr=2.0,
        multiway_tighten=0.10),
    "weak_tight": RegularParams(
        open_early=0.20, open_late=0.40, limp_share=0.3, call_open=0.30,
        threebet_value=0.04, threebet_bluff=0.0, call_threebet=0.04,
        fourbet=0.02, open_size_bb=2.5, threebet_mult=3.0, push_fold_bb=10,
        value_hs=0.80, cbet_dry=0.55, cbet_wet=0.40, oop_factor=0.7,
        size_dry=0.5, size_wet=0.6, size_river=0.6,
        barrel_turn=0.2, barrel_river=0.1, bluff_ratio=0.1, semi_bluff=0.20,
        defend_factor=0.60, raise_value=0.8, raise_bluff=0.0,
        slowplay=0.05, donk=0.1, overbet=0.0, allin_spr=1.5,
        multiway_tighten=0.12),
    "trapper": RegularParams(
        open_early=0.14, open_late=0.45, limp_share=0.1, call_open=0.18,
        threebet_value=0.05, threebet_bluff=0.02, call_threebet=0.10,
        fourbet=0.02, open_size_bb=2.5, threebet_mult=3.2, push_fold_bb=12,
        value_hs=0.75, cbet_dry=0.35, cbet_wet=0.30, oop_factor=0.9,
        size_dry=0.5, size_wet=0.75, size_river=0.75,
        barrel_turn=0.4, barrel_river=0.35, bluff_ratio=0.7, semi_bluff=0.40,
        defend_factor=1.0, raise_value=0.9, raise_bluff=0.3,
        slowplay=0.6, donk=0.05, overbet=0.1, allin_spr=1.5,
        multiway_tighten=0.12),
    "polar_reg": RegularParams(
        open_early=0.16, open_late=0.50, limp_share=0.0, call_open=0.12,
        threebet_value=0.07, threebet_bluff=0.08, call_threebet=0.09,
        fourbet=0.035, open_size_bb=2.3, threebet_mult=3.5, push_fold_bb=12,
        value_hs=0.74, cbet_dry=0.70, cbet_wet=0.45, oop_factor=0.7,
        size_dry=0.33, size_wet=0.75, size_river=1.25,
        barrel_turn=0.6, barrel_river=0.5, bluff_ratio=1.0, semi_bluff=0.60,
        defend_factor=1.0, raise_value=0.5, raise_bluff=0.3,
        slowplay=0.1, donk=0.05, overbet=0.6, allin_spr=2.0,
        multiway_tighten=0.12),
    "stealer": RegularParams(
        open_early=0.07, open_late=0.70, limp_share=0.0, call_open=0.10,
        threebet_value=0.05, threebet_bluff=0.06, call_threebet=0.04,
        fourbet=0.02, open_size_bb=2.2, threebet_mult=3.0, push_fold_bb=12,
        value_hs=0.75, cbet_dry=0.75, cbet_wet=0.55, oop_factor=0.6,
        size_dry=0.33, size_wet=0.67, size_river=0.75,
        barrel_turn=0.5, barrel_river=0.35, bluff_ratio=1.0, semi_bluff=0.50,
        defend_factor=0.85, raise_value=0.6, raise_bluff=0.15,
        slowplay=0.1, donk=0.0, overbet=0.1, allin_spr=1.5,
        multiway_tighten=0.12),
}

#: Knobs that are shifted rather than scaled: a pot fraction of 0.33 and one of
#: 1.5 want the same absolute spread, not the same relative one, and so do a
#: raise size in big blinds and a stack threshold. The sigma is in the knob's
#: own units.
ADDITIVE = {
    "size_dry": 0.10, "size_wet": 0.10, "size_river": 0.10,
    "open_size_bb": 0.25, "threebet_mult": 0.25, "push_fold_bb": 1.5,
}
#: Everything else is a rate, scaled by `exp(N(0, sigma))` — so a zero stays
#: zero, which is what makes "never bluffs" survive being jittered.
RATE_SIGMA = 0.15

#: What each knob may be, after jitter. Frequencies and combo fractions live in
#: [0, 1]; the rest are bounded by what the engine and the raise grid can
#: express at all.
DOMAIN = {
    "size_dry": (0.05, 4.0), "size_wet": (0.05, 4.0),
    "size_river": (0.05, 4.0),
    "open_size_bb": (1.5, 8.0), "threebet_mult": (2.0, 6.0),
    "push_fold_bb": (0.0, 30.0),
    "value_hs": (0.3, 0.99), "oop_factor": (0.0, 1.5),
    "bluff_ratio": (0.0, 5.0), "defend_factor": (0.0, 3.0),
    "allin_spr": (0.0, 10.0), "multiway_tighten": (0.0, 0.5),
}
DEFAULT_DOMAIN = (0.0, 1.0)

#: Per-knob spread, so a single number in the config can widen or narrow the
#: whole draw without changing the shape of it.
JITTER = {f.name: ADDITIVE.get(f.name, RATE_SIGMA)
          for f in fields(RegularParams)}


def draw_params(archetype, rng, spread=1.0):
    """One jittered variant of a preset. `spread = 0` is the preset itself."""
    assert archetype in ARCHETYPES, (
        f"{archetype!r} is not one of {sorted(ARCHETYPES)}")
    base = ARCHETYPES[archetype]
    if spread == 0:
        return replace(base)

    drawn = {}
    for name, sigma in JITTER.items():
        value = float(getattr(base, name))
        scale = float(sigma) * float(spread)
        if name in ADDITIVE:
            value += float(rng.normal(0.0, scale))
        else:
            value *= float(np.exp(rng.normal(0.0, scale)))
        lo, hi = DOMAIN.get(name, DEFAULT_DOMAIN)
        drawn[name] = float(np.clip(value, lo, hi))
    return RegularParams(**drawn)
