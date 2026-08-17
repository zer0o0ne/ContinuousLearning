"""G1 — does the embedding carry style, and does it generalise? (CONCEPT.md §14)

Embedding network only. No agent, no oracle. The pool plays itself, the network
is trained to predict what each player did, and the four measurements §14 asks
for are read off held-out hands of held-out players:

1. **action-prediction loss vs. number of observed hands**, against the `e = 0`
   baseline. "If a few hundred hands do not beat `e = 0` clearly, the mechanism
   does not work and nothing downstream is worth building."
2. **freshly sampled style settings never in the training histories** — the
   direct test of B1(b), the load-bearing assumption of the whole project. The
   reported number is the gap between the seen-style and unseen-style curves.
3. **the ablation** (§5.4): amortised head only, no gradient fit (`K = 0`). If
   it ties, the inference-time optimisation is unnecessary and comes out.
4. **the single-vector assumption**: the same numbers broken down by table size
   and by stack depth, looking for systematic skew. Zero extra work — it is a
   grouping of the numbers already computed.

**Sessions.** A session is a fixed set of pool members sitting down together for
a fixed number of hands at a fixed stack depth, with the button rotating each
hand. Everything is a session because that is what deployment looks like: hero
sits at a table, observes the players there, and fits their vectors from hands
hero was in (§5.4). Rotation matters — with fixed seating a member would be
identifiable by its seat, and the network would learn seats instead of styles.

Within an evaluation session the first `max(observed_hand_counts)` hands are the
observation window and the rest are held out for the measurement, so a curve
over "number of observed hands" is a curve over prefixes of one session.

**Table configuration is sampled uniformly** over 2–9 players and 10–300 BB
(`CLAUDE.md` §1). Nothing is weighted toward heads-up or 200 BB.

**Evaluation sets.** `seen` are members the network trained on, `unseen` are
fresh style draws over the same bases (measurement 2), and `heldout` — present
when `corpus.holdout_bases` names any — are members of bases withheld from
training entirely. The three are a ladder of how far from the training
distribution a strategy sits, and only the last one asks the question B1(b) is
about: measurement 2 varies 32 style scalars over a base the network knows,
which is interpolation inside one parametric family, and Slumbot is not a point
in that family. See `split_bases`.

**Extra conditions.** `eval_conditions` switches on measurements that share the
same fits and cost nothing to keep aligned: the trained table row as the fit's
ceiling, the fit at other step counts, the fit from a zero start, the fit with
the §5.1a terms off, the showdown heads scored on held-out hands, and the fitted
vectors themselves written out. Each of them exists to answer a question the
baseline report leaves open; see `evaluate_sessions` and `fit_variants`.

Run::

    cd versions/v8 && python3 -m gates.g1 --config config_g1.json
"""

import argparse
import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from env.driver import HandSpec, LockstepDriver
from env.showdown import N_HAND_CLASSES, label_showdowns
from nets.embedding_net import (
    OpponentEmbeddingNet, evaluate_ce_by_member, fit_embeddings,
)
from nets.features import TOKEN_DECISION, TOKEN_SHOWDOWN, collate, hand_tokens
from pool.build import build_pool, fresh_style_variants
from utils import Logger, progress, resolve_device

STREETS = ("preflop", "flop", "turn", "river")


@dataclass
class Session:
    """A fixed table of members playing a fixed number of hands."""

    idx: int
    num_players: int
    stack_bb: int
    members: list                 # pool-member index per slot; slot 0 observes
    specs: list = field(default_factory=list)
    records: list = field(default_factory=list)

    def seat_of_slot(self, slot, hand_idx):
        """Slot `slot` sits here in hand `hand_idx` (the button rotates)."""
        return (slot - hand_idx) % self.num_players

    def slot_of_seat(self, hand_idx):
        return [(seat + hand_idx) % self.num_players
                for seat in range(self.num_players)]

    def tokens(self, max_players, n_actions):
        """Token sequences of every hand, from the observer's (slot 0) view."""
        out = []
        for h, record in enumerate(self.records):
            out.append(hand_tokens(
                record, observer_pos=self.seat_of_slot(0, h),
                slot_of_seat=self.slot_of_seat(h),
                max_players=max_players, n_actions=n_actions,
            ))
        return out


def raise_sizes_from(game):
    return [list(game["raise_sizes"][s]) for s in STREETS]


def build_sessions(rng, member_ids, game, n_sessions, hands_per_session,
                   seed_base, tag):
    """Uniform over 2–9 players and 10–300 BB, independently (`CLAUDE.md` §1)."""
    lo_p, hi_p = game["players_range"]
    lo_s, hi_s = game["stack_bb_range"]
    bb, sb = game["big_blind"], game["small_blind"]
    raise_sizes = raise_sizes_from(game)

    sessions = []
    for s in range(n_sessions):
        num_players = int(rng.integers(lo_p, hi_p + 1))
        assert num_players <= len(member_ids), (
            f"[{tag}] a {num_players}-handed session needs {num_players} "
            f"distinct pool members, this set has {len(member_ids)}. Table size "
            f"is sampled uniformly over {game['players_range']} and is not "
            f"negotiable (CLAUDE.md §1), so widen the member set instead — more "
            f"`bootstrap` variants, or a larger `corpus.n_unseen_members`.")
        stack_bb = int(rng.integers(lo_s, hi_s + 1))
        members = [int(m) for m in
                   rng.choice(member_ids, size=num_players, replace=False)]
        session = Session(idx=s, num_players=num_players, stack_bb=stack_bb,
                          members=members)
        for h in range(hands_per_session):
            seat_members = [members[(seat + h) % num_players]
                            for seat in range(num_players)]
            session.specs.append(HandSpec(
                num_players=num_players,
                start_credits=[float(stack_bb * bb)] * num_players,
                seat_members=seat_members,
                seed=seed_base + s * hands_per_session + h,
                big_blind=bb, small_blind=sb, raise_sizes=raise_sizes,
                meta={"tag": tag, "session": s, "hand": h},
            ))
        sessions.append(session)
    return sessions


def play(driver, sessions, batch_size, log, tag):
    """Play every hand of every session in lock-step, then hand them back."""
    specs = [spec for s in sessions for spec in s.specs]
    log(f"[{tag}] playing {len(specs)} hands over {len(sessions)} sessions")
    records = driver.run(specs, batch_size=batch_size, desc=f"play:{tag}")

    cursor = 0
    n_decisions = 0
    truncated = 0
    for s in sessions:
        s.records = records[cursor:cursor + len(s.specs)]
        cursor += len(s.specs)
        n_decisions += sum(len(r.decisions) for r in s.records)
        truncated += sum(1 for r in s.records if r.truncated)
    # §5.1a: the showdown labels are cards-only, so they are computed once here
    # over the whole set and never again inside a training or fitting loop.
    n_reveals = label_showdowns(records, desc=f"showdown:{tag}")
    n_showdown_hands = sum(1 for r in records if r.showdown)
    log(f"[{tag}] {n_decisions} decisions, {n_reveals} reveals over "
        f"{n_showdown_hands} showdown hands, {truncated} hands hit the "
        f"max-actions cap")
    return n_decisions


# ---------------------------------------------------------------------- training


def loss_weights(cfg):
    """The §5.1a / §5.4 auxiliary-loss weights, from one place.

    `showdown_strength` and `showdown_class` weight the two showdown heads and
    are used **both** in training and in the inference-time fit — setting either
    to 0 is the ablation that answers "does the showdown anchor earn its keep".
    """
    return {
        "amortised": cfg.get("amortised_weight", 1.0),
        "showdown_strength": cfg.get("showdown_strength_weight", 0.0),
        "showdown_class": cfg.get("showdown_class_weight", 0.0),
    }


def train_embedding_net(net, sessions, cfg, game, device, log, seed):
    """§5.4 baseline: no inner loop, table and transformer trained jointly."""
    max_players = game["max_players"]
    n_actions = game["n_actions"]

    corpus = []
    for s in sessions:
        corpus.extend(t for t in s.tokens(max_players, n_actions) if len(t) > 0)
    n_decision = sum(int((t.token_type == TOKEN_DECISION).sum()) for t in corpus)
    n_showdown = sum(int((t.token_type == TOKEN_SHOWDOWN).sum()) for t in corpus)
    log(f"[train] corpus: {len(corpus)} hands, {n_decision} decision tokens, "
        f"{n_showdown} showdown tokens")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["steps"], eta_min=cfg.get("eta_min", 0.0))
    rng = np.random.default_rng(seed)
    batch_hands = min(cfg["batch_hands"], len(corpus))
    weights = loss_weights(cfg)

    net.train()
    history = []
    for step in progress(range(1, cfg["steps"] + 1), desc="train", unit="step"):
        pick = rng.choice(len(corpus), size=batch_hands, replace=False)
        batch = collate([corpus[i] for i in pick], device=device)
        total, parts = net.loss_terms(batch, weights)

        opt.zero_grad(set_to_none=True)
        total.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        if step % cfg.get("log_every", 100) == 0 or step == 1:
            log(f"[train] step {step}/{cfg['steps']} "
                + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            history.append({"step": step, **parts})
    net.eval()
    return history


# -------------------------------------------------------------------- evaluation


def _decisions_by_slot(observed, n_slots):
    """Decision tokens each slot contributed to the observed window.

    `observed_decisions` is the whole table's window and is the same number for
    every seat, so it cannot say how much the fit actually saw of any one
    opponent — which is the quantity §14.4's table-size breakdown is confounded
    by. Showdown tokens are excluded: they are a reveal, not a decision.
    """
    counts = np.zeros(n_slots, dtype=np.int64)
    for t in observed:
        sel = t.token_type == TOKEN_DECISION
        np.add.at(counts, t.slot[sel], 1)
    return [int(c) for c in counts]


@torch.no_grad()
def showdown_holdout(net, batch):
    """§5.1a's two heads scored on hands the network was never trained on.

    Training prints these losses on the corpus it is fitting, where a hand's
    final board is five specific cards and therefore very nearly a unique key: a
    head with enough capacity can answer from the board alone, and the printed
    curve then measures memorisation rather than any line-to-holding signal. The
    same two numbers on held-out hands is what tells the two apart — and it
    matters beyond bookkeeping, because these terms are also in the objective
    `fit_embeddings` optimises at inference (§5.5), where every board is new.

    `e = 0` is used so the number is a property of the heads rather than of
    whatever vector happened to be fitted. Returns ``None`` for a batch with no
    showdown in it.
    """
    hidden = net.hidden(batch, net.zero_emb(batch))
    strength, cls = net.showdown_losses(hidden, batch)
    if strength is None:
        return None
    return {
        "showdown_strength_mse": float(strength),
        "showdown_class_ce": float(cls),
        "n_tokens": int(batch["showdown_mask"].sum()),
    }


def _variant_steps(name, K):
    """Gradient steps a fitted condition runs. `fit_k<n>` says so in its name."""
    return int(name[5:]) if name.startswith("fit_k") else K


def fit_variants(cfg):
    """Names of the fitted conditions this config measures, in report order.

    ``fit`` is the §5.5 baseline and is always present; everything after it is
    switched on from the `eval_conditions` section, and every one of them is a
    question about the fit itself rather than about the embedding:

    ``fit_k<K>``
        the same fit run for a different number of gradient steps. `K`, `fit_lr`
        and `fit_reg` have never been measured, and §14.1 shows the fit *losing*
        to the ablation on a one-hand window — the regime in which hero sits
        down at a new table.
    ``fit_zero_init``
        the fit started from zero instead of from the amortised head. §14.3 asks
        whether the fit beats the head; this asks the reverse, and if it ties,
        the head comes out and takes the second trunk pass of `loss_terms` with
        it.
    ``fit_no_showdown``
        the fit with the §5.1a terms weighted to zero. Their heads may have
        memorised boards (`showdown_holdout`), in which case they contribute a
        gradient on new hands that is noise, at roughly a third the size of the
        action term.

    All of them except ``fit`` are restricted to `condition_windows`, because
    each is another fit at every observation window and that is where the whole
    cost of the gate sits.
    """
    conditions = cfg.get("eval_conditions", {})
    names = ["fit"]
    names += [f"fit_k{k}" for k in conditions.get("fit_steps_sweep", [])
              if k != cfg["K"]]
    if conditions.get("zero_init_fit"):
        names.append("fit_zero_init")
    if conditions.get("no_showdown_fit"):
        names.append("fit_no_showdown")
    return names


def evaluate_sessions(net, sessions, cfg, game, device, log, tag):
    """Measurements 1, 3 and 4 for one set of sessions.

    For every session and every prefix length *n* in `observed_hand_counts`, the
    players' vectors are fitted **jointly** (§5.3, §5.5) on the first *n* hands
    and scored on the held-out tail. The conditions per point:

    ``zero``
        `e = 0` — the baseline of §14.1, and also the policy the agent will play
        against an opponent it has not observed (§6.2).
    ``ablation``
        the amortised head's output with no gradient fit (`K = 0`, §5.4).
    ``fit``
        *K* gradient steps from that initialisation (§5.5).
    ``oracle``
        the member's own row of the trained embedding table. Not a deployable
        condition — the row only exists for a member the network was trained on
        — but it is the ceiling the fit is trying to reach, and without it the
        distance between "the fit is weak" and "the trunk cannot do better" is
        not measurable. On a set of members that were never trained the row is
        its initialisation, so the condition doubles as a leak check: it must
        come out at `e = 0`.
    the entries of `fit_variants`
        the fit run differently; see there.

    Only **non-observer** players are scored: the observer is hero, and hero's
    own actions are not what the mechanism is for. Hero's vector is still fitted
    (§5.3 couples them) and still recorded when vectors are being saved.
    """
    max_players = game["max_players"]
    n_actions = game["n_actions"]
    counts = list(cfg["observed_hand_counts"])
    K = cfg["K"]
    weights = loss_weights(cfg)
    conditions = cfg.get("eval_conditions", {})
    variants = fit_variants(cfg)
    no_showdown = {**weights, "showdown_strength": 0.0, "showdown_class": 0.0}

    # Every extra condition is another full fit at every window, and a fit's
    # cost is linear in the window — so measuring all of them everywhere costs
    # roughly `sum(steps) / K` times the baseline evaluation. `condition_windows`
    # is where that is traded off: outside it only the §5.5 baseline fit runs,
    # and the extra conditions report a dash at those windows.
    extra_windows = set(conditions.get("condition_windows", counts))
    fits_at = {n: len(variants) if n in extra_windows else 1 for n in counts}
    steps_at = {n: sum(_variant_steps(name, K) for name in variants
                       if n in extra_windows or name == "fit")
                for n in counts}
    log(f"[eval:{tag}] {sum(fits_at.values())} fits and "
        f"{sum(steps_at[n] * n for n in counts)} hand-steps of fitting per "
        f"session, over {len(sessions)} sessions")

    rows, vectors, showdown = [], [], []
    # One global bar over sessions × observation windows × fitted conditions,
    # not a bar per session (`CLAUDE.md` §5). A fit is the unit because it is
    # the expensive step, and its cost varies by two orders of magnitude across
    # the window lengths — which is exactly why the ETA has to average over
    # everything done so far rather than over the last few fits.
    bar = progress(total=len(sessions) * sum(fits_at.values()),
                   desc=f"eval:{tag}", unit="fit")
    for s in sessions:
        tokens = s.tokens(max_players, n_actions)
        window = tokens[:max(counts)]
        tail = [t for t in tokens[max(counts):] if len(t) > 0]
        if not tail:
            # Skipped sessions still advance the bar by the fits they would
            # have contributed, so it reaches its total.
            bar.update(sum(fits_at.values()))
            continue
        assert len(tail) >= 1
        eval_batch = collate(tail, device=device)

        targets = [(slot, m) for slot, m in enumerate(s.members) if slot != 0]
        members = [m for _slot, m in targets]
        base = evaluate_ce_by_member(
            net, eval_batch, net.zero_emb(eval_batch), members)

        oracle = None
        if conditions.get("oracle_embedding"):
            table = net.embeddings.weight.detach()[
                torch.as_tensor(s.members, device=device)]
            oracle = evaluate_ce_by_member(
                net, eval_batch, net.slot_emb(eval_batch, table), members)

        if conditions.get("showdown_holdout"):
            held = showdown_holdout(net, eval_batch)
            if held is not None:
                showdown.append({"set": tag, "session": s.idx, **held})

        for n in counts:
            observed = [t for t in window[:n] if len(t) > 0]
            active = variants if n in extra_windows else ["fit"]
            if observed:
                fit_batch = collate(observed, device=device)
                init = net.amortised_init(fit_batch, s.num_players)
                zero_init = torch.zeros(s.num_players, net.d_emb,
                                        device=device)
                fitted = {}
                for name in active:
                    fitted[name] = fit_embeddings(
                        net, fit_batch, s.num_players,
                        steps=_variant_steps(name, K),
                        lr=cfg["fit_lr"], reg=cfg["fit_reg"],
                        init=zero_init if name == "fit_zero_init" else init,
                        weights=(no_showdown if name == "fit_no_showdown"
                                 else weights))
                    bar.update(1)
            else:
                # Cold start (§5.5): no history, so the vector is zero and the
                # ablation has nothing to pool over either.
                init = torch.zeros(s.num_players, net.d_emb, device=device)
                fitted = {name: init for name in active}
                bar.update(len(active))

            scored = {"ablation": evaluate_ce_by_member(
                net, eval_batch, net.slot_emb(eval_batch, init), members)}
            for name, vec in fitted.items():
                scored[name] = evaluate_ce_by_member(
                    net, eval_batch, net.slot_emb(eval_batch, vec), members)

            per_slot = _decisions_by_slot(observed, s.num_players)
            for slot, member in targets:
                ce_zero, n_tok = base[member]
                if n_tok == 0:
                    continue
                row = {
                    "set": tag,
                    "session": s.idx,
                    "num_players": s.num_players,
                    "stack_bb": s.stack_bb,
                    "slot": slot,
                    "member": member,
                    "observed_hands": n,
                    "observed_decisions": sum(len(t) for t in observed),
                    "observed_decisions_slot": per_slot[slot],
                    "eval_tokens": n_tok,
                    "ce_zero": ce_zero,
                    "ce_ablation": scored["ablation"][member][0],
                }
                for name in active:
                    row[f"ce_{name}"] = scored[name][member][0]
                if oracle is not None:
                    row["ce_oracle"] = oracle[member][0]
                rows.append(row)

            if conditions.get("save_fitted_vectors"):
                # Every slot, hero included: hero's vector is the control for
                # "does a fitted vector say who this is" (§11.2), and the
                # descriptors give the true 32-scalar style of every member, so
                # the saved vectors are what a style-decoding probe reads.
                for slot, member in enumerate(s.members):
                    vectors.append({
                        "set": tag, "session": s.idx, "observed_hands": n,
                        "slot": slot, "member": member,
                        "vector": [float(x) for x in fitted["fit"][slot]],
                    })
    bar.close()
    log(f"[eval:{tag}] {len(rows)} measurement rows over {len(sessions)} "
        f"sessions, conditions: {', '.join(variants)}"
        + (", oracle" if conditions.get("oracle_embedding") else ""))
    return rows, vectors, showdown


# ----------------------------------------------------------------- aggregation


def _stats(values):
    """Mean and standard error of the mean. Never quote one without the other."""
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    m = sum(v) / len(v)
    if len(v) < 2:
        return {"n": len(v), "mean": m, "se": float("nan")}
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return {"n": len(v), "mean": m, "se": math.sqrt(var / len(v))}


BASELINE = "ce_zero"


def metrics_of(rows):
    """Every condition measured in `rows`, and its gain over `e = 0`.

    The set of conditions is not fixed: `eval_conditions` switches some of them
    on, so the report is built from whatever the rows actually carry rather than
    from a constant that has to be edited in step. A condition is any ``ce_*``
    key; `e = 0` is the baseline every gain is taken against and so is listed
    first and never gains against itself.
    """
    ce = sorted({k for r in rows for k in r if k.startswith("ce_")}
                - {BASELINE})
    return (BASELINE,) + tuple(ce) + tuple(f"gain_{k[3:]}" for k in ce)


def _with_gains(row):
    """`gain_*` = how much the embedding beat `e = 0`. Positive is better.

    The measurement §14.1 asks for is "against the `e = 0` baseline", and the
    difference is the part that is comparable across sets: two sets of sessions
    have different tables, different styles and therefore different intrinsic
    predictability, so their raw losses are not on the same scale — but the
    improvement each gets from conditioning on an embedding is.
    """
    gains = {f"gain_{k[3:]}": row[BASELINE] - row[k]
             for k in row if k.startswith("ce_") and k != BASELINE}
    return {**row, **gains}


def _curve(rows):
    """Per-metric mean at each observed-hand count, aggregated by session.

    Rows from one session share their evaluation hands and their table, so they
    are not independent. Averaging within a session first makes the session the
    unit, which is what the quoted standard error is then the standard error of.
    """
    metrics = metrics_of(rows)
    out = {}
    for n in sorted({r["observed_hands"] for r in rows}):
        sel = [r for r in rows if r["observed_hands"] == n]
        out[n] = {}
        for metric in metrics:
            # A condition may be measured at only some of the windows
            # (`eval_conditions.condition_windows`), so each metric is
            # aggregated over the rows that actually carry it. Where none do,
            # `_stats` reports n = 0 and the report prints a dash.
            by_session = {}
            for r in sel:
                if metric in r:
                    by_session.setdefault(r["session"], []).append(r)
            out[n][metric] = _stats([
                sum(r[metric] for r in group) / len(group)
                for group in by_session.values()
            ])
    return out


STACK_BUCKETS = [(10, 40), (41, 100), (101, 200), (201, 300)]


def _stack_bucket(stack_bb):
    for lo, hi in STACK_BUCKETS:
        if lo <= stack_bb <= hi:
            return f"{lo}-{hi}"
    return "other"


def aggregate(rows, showdown=None):
    """Everything §14 asks to be reported, from the same measurement rows."""
    rows = [_with_gains(r) for r in rows]
    sets = sorted({r["set"] for r in rows})
    metrics = metrics_of(rows)
    curves = {s: _curve([r for r in rows if r["set"] == s]) for s in sets}
    report = {"curves": curves, "metrics": list(metrics)}  # §14.1

    # §14.2 — B1(b). Every other set is compared against `seen`, because `seen`
    # is the only one whose members the network was trained on and so the only
    # sensible reference. Two readings, both reported:
    #   `ce_fit`   the raw loss gap §14.2 asks for — confounded by the sets
    #              having different tables and different intrinsic difficulty;
    #   `gain_fit` the gap in improvement over each set's own e=0 baseline —
    #              the confound cancels, so this is the one that answers "does
    #              the embedding still work on a strategy never trained on".
    if "seen" in sets:
        report["style_generalisation_gap"] = {
            other: {n: {metric: (curves[other][n][metric]["mean"]
                                 - curves["seen"][n][metric]["mean"])
                        for metric in metrics}
                    for n in sorted(set(curves[other]) & set(curves["seen"]))}
            for other in sets if other != "seen"
        }

    # §14.3 — does the K-step fit earn its keep over the amortised head alone?
    report["ablation"] = {
        s: {n: {"fit_minus_ablation": (curves[s][n]["ce_fit"]["mean"]
                                       - curves[s][n]["ce_ablation"]["mean"])}
            for n in curves[s]}
        for s in sets
    }

    # §14.4 — the single-vector assumption, by table size and stack depth.
    report["by_table_size"] = {
        s: {str(n_p): _curve([r for r in rows
                              if r["set"] == s and r["num_players"] == n_p])
            for n_p in sorted({r["num_players"] for r in rows if r["set"] == s})}
        for s in sets
    }
    report["by_stack_depth"] = {
        s: {b: _curve([r for r in rows if r["set"] == s
                       and _stack_bucket(r["stack_bb"]) == b])
            for b in sorted({_stack_bucket(r["stack_bb"])
                             for r in rows if r["set"] == s})}
        for s in sets
    }

    # §5.1a's two heads on held-out hands. Token-weighted, because a session's
    # contribution to the training loss is its showdown tokens, not its
    # sessionhood, and the comparison being made is against that loss.
    if showdown:
        by_set = {}
        for rec in showdown:
            by_set.setdefault(rec["set"], []).append(rec)
        report["showdown_holdout"] = {
            s: {
                "showdown_strength_mse": sum(
                    r["showdown_strength_mse"] * r["n_tokens"] for r in v)
                / sum(r["n_tokens"] for r in v),
                "showdown_class_ce": sum(
                    r["showdown_class_ce"] * r["n_tokens"] for r in v)
                / sum(r["n_tokens"] for r in v),
                "n_tokens": sum(r["n_tokens"] for r in v),
                "sessions": len(v),
            }
            for s, v in by_set.items()
        }
    return report


def format_report(report, log):
    log("")
    log("=" * 78)
    log("G1 — opponent embedding: does it carry style, and does it generalise?")
    log("=" * 78)

    def cell(st):
        return f"{st['mean']:.4f}±{st['se']:.4f}" if st["n"] else "—"

    for s, curve in report["curves"].items():
        log(f"\n[§14.1 | {s} set] action-prediction CE in nats, "
            f"mean ± SE over sessions")
        log(f"  {'hands':>6} {'e=0':>16} {'ablation K=0':>16} {'fit K>0':>16}"
            f" {'gain of fit':>16} {'sess':>5}")
        for n, m in curve.items():
            log(f"  {n:>6} {cell(m['ce_zero']):>16} "
                f"{cell(m['ce_ablation']):>16} {cell(m['ce_fit']):>16} "
                f"{cell(m['gain_fit']):>16} {m['ce_fit']['n']:>5}")
        log("  gain of fit = e=0 CE minus fitted CE; positive means the "
            "embedding helped.")

    extra = [m for m in report.get("metrics", [])
             if m.startswith("gain_") and m not in ("gain_ablation",
                                                    "gain_fit")]
    if extra:
        log("\n[extra conditions] gain over e=0, mean ± SE over sessions "
            "(see `fit_variants`)")
        for s, curve in report["curves"].items():
            log(f"  [{s}]")
            log(f"    {'hands':>6} "
                + " ".join(f"{m[5:]:>17}" for m in extra))
            for n, m in curve.items():
                log(f"    {n:>6} "
                    + " ".join(f"{cell(m[k]):>17}" for k in extra))
        log("  oracle is the trained table row: the ceiling the fit is aiming")
        log("  at on `seen`, and a leak check elsewhere — a set whose members")
        log("  were never trained has no row, so its oracle gain must read ~0.")

    for other, gap in report.get("style_generalisation_gap", {}).items():
        log(f"\n[§14.2 | B1(b)] {other} minus seen")
        log(f"  {'hands':>6} {'Δ CE(fit)':>12} {'Δ gain(fit)':>13}")
        for n, g in gap.items():
            log(f"  {n:>6} {g['ce_fit']:>12.4f} {g['gain_fit']:>13.4f}")
        log("  Δ CE is the raw gap §14.2 asks for; it is confounded by the two "
            "sets having")
        log("  different tables. Δ gain compares each set against its own e=0 "
            "baseline, so a")
        log("  Δ gain near zero is the evidence B1(b) needs: the held-out draws "
            "are read as")
        log("  well as trained ones. A large negative Δ gain is B1(b) failing.")
        log("  Both readings are confounded by set composition — "
            "`gates.g1_analysis` corrects it.")

    log("\n[§14.3 | ablation] fitted CE minus ablation CE, nats "
        "(negative = the K-step fit earns its keep)")
    for s, per_n in report["ablation"].items():
        log(f"  {s:>6}: " + "  ".join(
            f"n={n}:{v['fit_minus_ablation']:+.4f}" for n, v in per_n.items()))

    for name, title in (("by_table_size", "table size (players)"),
                        ("by_stack_depth", "stack depth (BB)")):
        log(f"\n[§14.4 | single-vector assumption] by {title}, at the longest "
            f"observation window")
        for s, groups in report[name].items():
            for g, curve in groups.items():
                if not curve:
                    continue
                last = max(curve)
                m = curve[last]
                log(f"  {s:>6} {g:>8}: n={last:<4} "
                    f"e=0={cell(m['ce_zero'])}  fit={cell(m['ce_fit'])}  "
                    f"gain={cell(m['gain_fit'])}  sess={m['ce_fit']['n']}")

    if "showdown_holdout" in report:
        log("\n[§5.1a | showdown heads on held-out hands, e=0]")
        log(f"  {'set':>8} {'strength MSE':>14} {'class CE':>12} "
            f"{'tokens':>10} {'sess':>5}")
        for s, v in report["showdown_holdout"].items():
            log(f"  {s:>8} {v['showdown_strength_mse']:>14.4f} "
                f"{v['showdown_class_ce']:>12.4f} {v['n_tokens']:>10} "
                f"{v['sessions']:>5}")
        log(f"  Compare against the training losses above. ln({N_HAND_CLASSES})"
            f" = {math.log(N_HAND_CLASSES):.4f} is the class CE of a uniform")
        log("  guess; a training loss far below a held-out loss near it means")
        log("  the head answered from the board, which is a key, not a signal.")

    if "timings" in report:
        log("\n[timings] wall clock, seconds")
        for phase, seconds in report["timings"].items():
            log(f"  {phase:>18} {seconds:>10.1f}")
    log("")


# ----------------------------------------------------------------------- driver


def split_bases(descriptors, holdout_bases, min_members, log):
    """Member indices that may be trained on, and those held out (§14 C1).

    §14.2's fresh style draws are a re-draw of the 32 style scalars over a base
    the network *was* trained on, so they test interpolation inside one
    parametric family. Slumbot is not a point in that family. Holding whole
    bases out of training — the pool member, all of its style variants, and any
    sibling checkpoint of the same lineage — makes the third evaluation set the
    one that asks the question B1(b) is actually about: a strategy of a kind the
    network has never seen.

    Held-out members stay in the member list, and therefore in the embedding
    table, exactly as the fresh draws do: their rows never receive a gradient
    because no training session seats them.
    """
    known = {d["base"] for d in descriptors}
    unknown = [b for b in holdout_bases if b not in known]
    assert not unknown, (
        f"holdout_bases names {unknown} which are not bases of this pool. The "
        f"bases are {sorted(known)} — a base is an entry's `label`, or its "
        f"strategy/checkpoint name when the entry has no label.")

    train_ids = [i for i, d in enumerate(descriptors)
                 if d["base"] not in holdout_bases]
    holdout_ids = [i for i, d in enumerate(descriptors)
                   if d["base"] in holdout_bases]
    for name, ids in (("trainable", train_ids), ("held-out", holdout_ids)):
        assert not ids or len(ids) >= min_members, (
            f"the {name} member set has {len(ids)} members, and a session is "
            f"sampled with up to {min_members} distinct ones (`CLAUDE.md` §1 "
            f"forbids narrowing the table-size range). Hold out fewer bases, or "
            f"give the ones you hold out more `n_variants`.")
    if holdout_ids:
        log(f"held-out bases (never in training): {sorted(holdout_bases)} "
            f"— {len(holdout_ids)} members, {len(train_ids)} remain trainable")
    return train_ids, holdout_ids


def save_eval_corpus(path, sessions_by_tag):
    """The tokenised evaluation sessions, so a later eval needs no replay.

    Everything measured after training is a function of these tokens and the
    checkpoint. Without them, changing an `eval_conditions` switch means
    replaying every evaluation hand — and replay is only *probably* exact, since
    the pool's network members are sampled through GPU forwards whose bitwise
    reproducibility nobody has promised. Roughly 3 KB per hand.
    """
    payload = {
        tag: [{"idx": s.idx, "num_players": s.num_players,
               "stack_bb": s.stack_bb, "members": s.members,
               "tokens": tokens}
              for s, tokens in pairs]
        for tag, pairs in sessions_by_tag.items()
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def run(config, log, out_dir):
    game = config["game"]
    seed = config.get("seed", 0)
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")
    timings = {}

    rng = np.random.default_rng(seed)
    members, descriptors = build_pool(config, rng, device=device, log=log)
    log(f"pool: {len(members)} members "
        f"({len({d['base'] for d in descriptors})} bases)")

    corpus_cfg = config["corpus"]
    train_ids, holdout_ids = split_bases(
        descriptors, list(corpus_cfg.get("holdout_bases", [])),
        game["players_range"][1], log)

    # The fresh draws are style re-draws of a *trainable* base: a fresh style on
    # a held-out base would confound §14.2 with C1.
    fresh, fresh_desc = fresh_style_variants(
        [members[i] for i in train_ids], [descriptors[i] for i in train_ids],
        corpus_cfg["n_unseen_members"], rng, config.get("style", {}),
        tag="fresh")
    log(f"fresh style draws (never in training): {len(fresh)}")

    all_members = members + fresh
    unseen_ids = list(range(len(members), len(all_members)))

    driver = LockstepDriver(all_members, game["n_actions"])
    batch_size = corpus_cfg.get("driver_batch_size", 256)

    t0 = time.perf_counter()
    train_sessions = build_sessions(
        rng, train_ids, game, corpus_cfg["train_sessions"],
        corpus_cfg["train_hands_per_session"], seed_base=10_000_000, tag="train")
    play(driver, train_sessions, batch_size, log, "train")
    timings["play:train"] = time.perf_counter() - t0

    eval_hands = max(corpus_cfg["observed_hand_counts"]) + \
        corpus_cfg["eval_hands_per_session"]
    # seed_base is per set and must stay distinct, or two sets would deal
    # identical hands and their comparison would not be independent.
    eval_specs = [("seen", train_ids, 20_000_000),
                  ("unseen", unseen_ids, 30_000_000)]
    if holdout_ids:
        eval_specs.append(("heldout", holdout_ids, 40_000_000))

    t0 = time.perf_counter()
    eval_sessions = {}
    for tag, ids, seed_base in eval_specs:
        eval_sessions[tag] = build_sessions(
            rng, ids, game, corpus_cfg["eval_sessions"], eval_hands,
            seed_base=seed_base, tag=tag)
        play(driver, eval_sessions[tag], batch_size, log, tag)
    timings["play:eval"] = time.perf_counter() - t0

    net = OpponentEmbeddingNet(
        config["embedding_net"], game["n_actions"], game["max_players"],
        n_members=len(all_members),
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"embedding network: {n_params/1e6:.2f}M parameters")

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    history = train_embedding_net(net, train_sessions, config["train"], game,
                                  device, log, seed)
    timings["train"] = time.perf_counter() - t0

    eval_cfg = {**config["embedding_net"], **corpus_cfg, **config["train"],
                "eval_conditions": config.get("eval_conditions", {})}
    rows, vectors, showdown = [], [], []
    for tag, sessions in eval_sessions.items():
        t0 = time.perf_counter()
        r, v, sd = evaluate_sessions(net, sessions, eval_cfg, game, device,
                                     log, tag)
        timings[f"eval:{tag}"] = time.perf_counter() - t0
        rows += r
        vectors += v
        showdown += sd

    report = aggregate(rows, showdown=showdown)
    report["timings"] = timings
    format_report(report, log)

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config": config,
        "pool": descriptors,
        "fresh_style_draws": fresh_desc,
        "holdout_bases": list(corpus_cfg.get("holdout_bases", [])),
        "train_history": history,
        "rows": rows,
        "report": report,
    }
    path = os.path.join(out_dir, "g1_report.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    log(f"wrote {path}")

    if vectors:
        # Kept out of the report: 32 floats per row would multiply its size,
        # and every use of them is array-shaped anyway.
        v_path = os.path.join(out_dir, "fitted_vectors.npz")
        np.savez_compressed(
            v_path,
            set=np.array([v["set"] for v in vectors]),
            session=np.array([v["session"] for v in vectors], dtype=np.int64),
            observed_hands=np.array([v["observed_hands"] for v in vectors],
                                    dtype=np.int64),
            slot=np.array([v["slot"] for v in vectors], dtype=np.int64),
            member=np.array([v["member"] for v in vectors], dtype=np.int64),
            vector=np.array([v["vector"] for v in vectors], dtype=np.float32),
        )
        log(f"wrote {v_path} ({len(vectors)} fitted vectors)")

    if corpus_cfg.get("save_eval_corpus"):
        t0 = time.perf_counter()
        c_path = save_eval_corpus(
            os.path.join(out_dir, "eval_corpus.pkl"),
            {tag: [(s, s.tokens(game["max_players"], game["n_actions"]))
                   for s in sessions]
             for tag, sessions in eval_sessions.items()})
        log(f"wrote {c_path} in {time.perf_counter() - t0:.1f}s")

    torch.save({"model_state_dict": net.state_dict(), "config": config},
               os.path.join(out_dir, "embedding_net.pt"))
    return report


def main():
    parser = argparse.ArgumentParser(description="CONCEPT.md §14 gate G1")
    parser.add_argument("--config", default="config_g1.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    base_dir = config.get("out_dir", "../../data/v8/g1")
    log = Logger(base_dir)
    out_dir = log.run_dir("g1")
    try:
        run(config, log, out_dir)
    finally:
        log.close()


if __name__ == "__main__":
    main()
