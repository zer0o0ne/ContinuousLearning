"""Agent targets and losses (CONCEPT.md §6.2).

Two losses with the same optimum and different behaviour under label noise;
`agent_train`'s `loss` key picks one.

* **`kl`** — the baseline §6.2 describes. Build `softmax(Q_normalised / T)` and
  take the KL from it to the agent's masked policy.
* **`soft_q`** — the same optimum reached without ever forming that
  distribution: `−⟨π_θ, Q_normalised⟩ − T·H(π_θ)`, whose minimiser over the
  simplex is exactly `softmax(Q_normalised / T)`.

**Why the second one exists.** `Q` is a Monte-Carlo estimate, `Q̂ = Q + ε`, and
G3 measured `ε` at 0.65 BB heads-up on 20 BB stacks and 25.7 BB six-handed on
300 BB. A softmax is not linear, so `E[softmax(Q̂/T)] ≠ softmax(Q/T)`: under the
`kl` loss the *gradient* the network sees is the expected target, and label
noise turns into target **bias** that averaging over labels cannot remove. In
the large-noise limit the target degenerates towards whichever action drew the
luckiest sample, whose average is uniform-over-legal — that is, the policy is
pushed towards uniform exactly where the labels are noisiest, which is deep
stacks and multiway.

`soft_q` is linear in `Q̂`, so its gradient is unbiased at every `θ` and the
noise stays noise. The price is that it is the *reverse* KL — mode-seeking
rather than mass-covering — so where the network cannot fit the target it
drops small-probability actions rather than smearing over them, and mixing is
the thing §11.2 cares about. Which one wins is a measurement, which is why
both are here rather than one.

**The normalisation is the whole of this module, and it is a scar.** Raw EVs in
a 300 BB pot and a 10 BB pot differ by more than an order of magnitude, so one
temperature over raw EVs produces a near-deterministic target in big pots and a
near-uniform one in small ones. v7's 97 % fold rate was a normalisation bug of
exactly this family — `versions/v7/ARCHITECTURE.md`, "MCTS value-target
normalization" — and not an architecture failure. Dividing by `pot + facing_bet`
makes the target invariant to the scale of the situation, which is the property
`test_targets.py` pins to 1e-12 and the reason this function exists at all
rather than a `softmax` at the call site.

That scale invariance describes the legacy **scalar** temperature. The default
now specifies a decaying absolute BB loss budget: `T = eps / (scale * log A)`.
The scale then cancels in the target, intentionally: the same BB mistake has
the same budget in small and large pots. Normalising Q still weights the loss
across training states. `decision_temperature` is shared by both losses and
the held-out metric; it never derives T from noisy action values.

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


def ev_scale(pot_bb, facing_bet_bb, divisor="pot_plus_bet"):
    """The positive, finite BB scale shared by EVs and their temperature."""
    assert divisor in DIVISORS, (
        f"unknown divisor {divisor!r}; choices are {sorted(DIVISORS)}")
    scale = float(DIVISORS[divisor](float(pot_bb), float(facing_bet_bb)))
    assert np.isfinite(scale) and scale > 0.0, (
        f"the {divisor} normaliser must be finite and positive, got {scale}")
    return scale


def ev_loss_budget(temperature, iteration=0):
    """Harmonic local EV budget in BB, or None for a legacy scalar T.

    `iteration` is the zero-based outer policy-improvement cycle, including on
    resume. The initial budget is a design choice, not a convergence theorem.
    """
    assert isinstance(iteration, (int, np.integer)) and iteration >= 0, (
        f"iteration must be a non-negative integer, got {iteration}")
    if not isinstance(temperature, dict):
        assert np.isfinite(temperature) and temperature > 0, (
            f"temperature must be finite and positive, got {temperature}")
        return None
    assert set(temperature) == {"initial_ev_loss_bb"}, (
        "temperature schedule must contain only initial_ev_loss_bb")
    initial = float(temperature["initial_ev_loss_bb"])
    assert np.isfinite(initial) and initial > 0, (
        f"initial_ev_loss_bb must be finite and positive, got {initial}")
    return initial / (int(iteration) + 1)


def decision_temperature(temperature, legal, pot_bb, facing_bet_bb,
                         divisor="pot_plus_bet", iteration=0):
    """Resolve a scalar T or a local-loss schedule for one decision.

    For A legal actions, scale D and exact Q, the soft optimum satisfies
    max Q - <softmax(Q/(D*T)), Q> <= D*T*log(A). Thus T = eps/(D*log(A))
    budgets entropy smoothing alone; it does not bound oracle or fitting error.
    T depends on the state, never on sampled Q, preserving soft_q's linear
    gradient in noisy labels. A forced action loses zero EV at any positive T.
    """
    budget = ev_loss_budget(temperature, iteration)
    if budget is None:
        return float(temperature)
    legal = np.asarray(legal, dtype=bool)
    assert legal.ndim == 1 and legal.any(), "a decision needs a legal action"
    scale = ev_scale(pot_bb, facing_bet_bb, divisor)
    n_legal = int(legal.sum())
    t = 1.0 if n_legal == 1 else budget / (scale * np.log(n_legal))
    assert np.isfinite(t) and t > 0, "scheduled temperature is not representable"
    return float(t)


def normalised_q(q, legal, pot_bb, facing_bet_bb, divisor="pot_plus_bet"):
    """`Q / scale` on the legal actions, exact zero off them (§6.2).

    The payload the `soft_q` loss consumes, and the first half of what
    `policy_target` does — one normalisation, not two.

    Args:
        q: (n_actions,) oracle EVs in big blinds. Entries off `legal` are never
            read and are expected to be `nan`.
        legal: (n_actions,) bool — the environment's mask for this decision.
        pot_bb: pot before the decision, in big blinds.
        facing_bet_bb: what hero has to call, in big blinds.
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
    scale = ev_scale(pot_bb, facing_bet_bb, divisor)
    out = np.zeros_like(q)
    out[legal] = q[legal] / scale
    return out


def policy_target(q, legal, pot_bb, facing_bet_bb, temperature,
                  divisor="pot_plus_bet", iteration=0):
    """softmax(Q_normalised / T) over legal actions (§6.2), for the `kl` loss.

    `q` in BB with `nan` at illegal actions. Returns (n_actions,) summing to 1
    with exact zeros off `legal`.

    Args:
        q, legal, pot_bb, facing_bet_bb, divisor: see `normalised_q`.
        temperature: positive scalar T, or {"initial_ev_loss_bb": positive BB}.
        iteration: zero-based outer cycle for the harmonic EV-loss budget.
    """
    legal = np.asarray(legal, dtype=bool)
    temperature = decision_temperature(temperature, legal, pot_bb,
                                       facing_bet_bb, divisor, iteration)
    qn = normalised_q(q, legal, pot_bb, facing_bet_bb, divisor)

    out = np.zeros_like(qn)
    with np.errstate(over="ignore", under="ignore"):
        z = qn[legal]
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


def soft_q_loss(logits, q_norm, legal, temperature):
    """`T·KL(π_θ ‖ softmax(Q_normalised / T))`, averaged over the batch.

    Written out, and this is the whole point of the function:

        L = −⟨π_θ, Q̂ₙ⟩  +  T·Σ π_θ log π_θ  +  T·log Σ exp(Q̂ₙ/T)

    The first two terms are the objective — expected value plus an entropy
    bonus — and `Q̂ₙ` enters **linearly**, so `E_ε[∇_θ L] = ∇_θ L|_{Q̂=Q}` and
    label noise never becomes target bias. The third term does not involve `θ`
    at all; it is added so the number printed in the log is a KL, that is
    non-negative and exactly zero when the masked policy is the target, the
    way `kl_loss` is readable. It shifts the value and not the gradient.

    Note the direction: this is `KL(π ‖ target)`, the reverse of `kl_loss`.
    Same minimiser, different behaviour when the network cannot reach it — see
    the module docstring.

    Args:
        logits: (B, n_actions) — the agent's raw output, unmasked (§6.1).
        q_norm: (B, n_actions) — `normalised_q` per row, exact zeros off
            `legal`. Data, so no gradient flows into it.
        legal: (B, n_actions) bool.
        temperature: positive finite scalar T or a (B,) tensor of row-wise T.
            Detached data, resolved from the state and iteration, not from Q.
    """
    assert logits.shape == q_norm.shape == legal.shape, (
        f"logits {tuple(logits.shape)}, q_norm {tuple(q_norm.shape)} and legal "
        f"{tuple(legal.shape)} must agree")
    assert bool(legal.any(dim=-1).all()), (
        "a row with no legal action cannot be scored")
    t = torch.as_tensor(temperature, dtype=logits.dtype,
                        device=logits.device).detach()
    assert t.ndim == 0 or t.shape == logits.shape[:-1], (
        "temperature must be scalar or have one value per decision")
    assert bool((torch.isfinite(t) & (t > 0)).all()), (
        "temperature must be finite and positive in the logits dtype")
    q_norm = q_norm.detach().to(logits.dtype)
    assert not bool((q_norm * ~legal).any()), (
        "q_norm carries a value on an illegal action — it was built with a "
        "different mask than the one being scored (§6.2 says there is one)")
    assert bool(torch.isfinite(q_norm).all()), (
        "a non-finite normalised EV would make the objective meaningless; "
        "`nan` marks illegal actions and must have been zeroed by "
        "`normalised_q`")

    logp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    p = logp.exp()
    # Off `legal` the policy is exactly zero, so the entropy term is zero
    # there; the `where` is what keeps `0 · -inf` from turning it into `nan`.
    plogp = p * torch.where(legal, logp, torch.zeros_like(logp))

    # Center before division: cooling must not overflow the common EV offset.
    q_centered = q_norm - q_norm.masked_fill(~legal, float("-inf")).amax(
        dim=-1, keepdim=True)
    q_centered = q_centered.masked_fill(~legal, 0.0)
    shift = t * torch.logsumexp(
        (q_centered / t.unsqueeze(-1)).masked_fill(~legal, float("-inf")),
        dim=-1)
    per_row = -(p * q_centered).sum(dim=-1) + t * plogp.sum(dim=-1) + shift
    return per_row.mean()
