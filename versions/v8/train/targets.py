"""Agent targets and loss (CONCEPT.md §6.2).

The target at a hero decision is `softmax(Q_normalised / T)` over the legal
actions, and the loss is the KL from it to the agent's masked policy.

**The normalisation is the whole of this module, and it is a scar.** Raw EVs in
a 300 BB pot and a 10 BB pot differ by more than an order of magnitude, so one
temperature over raw EVs produces a near-deterministic target in big pots and a
near-uniform one in small ones. v7's 97 % fold rate was a normalisation bug of
exactly this family — `versions/v7/ARCHITECTURE.md`, "MCTS value-target
normalization" — and not an architecture failure. Dividing by `pot + facing_bet`
makes the target invariant to the scale of the situation, which is the property
`test_targets.py` pins to 1e-12 and the reason this function exists at all
rather than a `softmax` at the call site.

The divisor is a config choice (§6.2), so both readings of "the scale of this
situation" are available: `pot_plus_bet` (baseline) and `pot`, which is the same
quantity before the bet hero is facing is counted. Nothing else is implementable
from this signature, and inventing a third from quantities it does not receive
would be a knob nobody asked for.

**Masking.** Illegal actions receive *exact* zero, not a small number: they are
dropped before the softmax, not suppressed inside it. The mask is the one the
environment produced (`env.legal.legal_action_mask`, carried on the token), and
§6.2's "dominated" is already part of it — `env/legal.py` drops dominated raise
bins when it builds the mask, so there is one legality rule and not two.
"""

import numpy as np
import torch

DIVISORS = {
    "pot_plus_bet": lambda pot_bb, facing_bet_bb: pot_bb + facing_bet_bb,
    "pot": lambda pot_bb, facing_bet_bb: pot_bb,
}


def policy_target(q, legal, pot_bb, facing_bet_bb, temperature,
                  divisor="pot_plus_bet"):
    """softmax(Q_normalised / T) over legal actions (§6.2).

    `q` in BB with `nan` at illegal actions. Returns (n_actions,) summing to 1
    with exact zeros off `legal`.

    Args:
        q: (n_actions,) oracle EVs in big blinds. Entries off `legal` are never
            read and are expected to be `nan`.
        legal: (n_actions,) bool — the environment's mask for this decision.
        pot_bb: pot before the decision, in big blinds.
        facing_bet_bb: what hero has to call, in big blinds.
        temperature: `T`. Positive and finite; the two limits are reached by
            passing a very small or very large `T`, not by passing 0 or `inf`.
        divisor: a key of `DIVISORS`.
    """
    q = np.asarray(q, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    assert q.ndim == 1 and q.shape == legal.shape, (
        f"q {q.shape} and legal {legal.shape} must be one decision's action set")
    assert legal.any(), "a decision with no legal action is not a decision"
    assert np.isfinite(q[legal]).all(), (
        "every legal action needs a finite EV — `nan` marks the illegal ones, "
        "and a `nan` under the mask means the oracle skipped an action it was "
        "asked about")
    assert np.isfinite(temperature) and temperature > 0, (
        f"temperature must be finite and positive, got {temperature}")
    assert divisor in DIVISORS, (
        f"unknown divisor {divisor!r}; choices are {sorted(DIVISORS)}")

    scale = float(DIVISORS[divisor](float(pot_bb), float(facing_bet_bb)))
    assert scale > 0.0, (
        f"the {divisor} normaliser is {scale}, so the target would not be "
        "defined — every decision has blinds in the pot behind it")

    out = np.zeros_like(q)
    with np.errstate(over="ignore", under="ignore"):
        z = q[legal] / scale
        z = (z - z.max()) / temperature
        out[legal] = np.exp(z)

    total = out.sum()
    assert total > 0.0, "the softmax underflowed to zero everywhere"
    return out / total


def kl_loss(logits, target, legal):
    """KL(target ‖ softmax(masked logits)), averaged over the batch.

    Zero exactly when the masked prediction is the target, so the number is
    readable as a distance and not only as something to minimise. `target` is
    detached: it is data, and a gradient into it would be a gradient into the
    oracle.

    Args:
        logits: (B, n_actions) — the agent's raw output, unmasked (§6.1).
        target: (B, n_actions) — distributions with exact zeros off `legal`.
        legal: (B, n_actions) bool.
    """
    assert logits.shape == target.shape == legal.shape, (
        f"logits {tuple(logits.shape)}, target {tuple(target.shape)} and legal "
        f"{tuple(legal.shape)} must agree")
    assert bool(legal.any(dim=-1).all()), (
        "a row with no legal action cannot be scored")
    target = target.detach().to(logits.dtype)
    assert bool((target >= 0).all()), "a target with a negative mass is not one"
    assert not bool((target * ~legal).any()), (
        "the target puts mass on an illegal action — it was built with a "
        "different mask than the one being scored (§6.2 says there is one)")

    logp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    # Off `legal` the target is zero, so the term is zero; the `where` is what
    # keeps `0 · -inf` from turning it into `nan` on the way there.
    logp = torch.where(legal, logp, torch.zeros_like(logp))
    tiny = torch.finfo(target.dtype).tiny
    terms = target * (torch.log(target.clamp_min(tiny)) - logp)
    return terms.sum(dim=-1).mean()
