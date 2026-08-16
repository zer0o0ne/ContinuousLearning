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

Run::

    cd versions/v8 && python3 -m gates.g1 --config config_g1.json
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from env.driver import HandSpec, LockstepDriver
from env.showdown import label_showdowns
from nets.embedding_net import (
    OpponentEmbeddingNet, evaluate_ce, fit_embeddings,
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


def evaluate_sessions(net, sessions, cfg, game, device, log, tag):
    """Measurements 1, 3 and 4 for one set of sessions.

    For every session and every prefix length *n* in `observed_hand_counts`, the
    players' vectors are fitted **jointly** (§5.3, §5.5) on the first *n* hands
    and scored on the held-out tail. Three conditions per point:

    ``zero``
        `e = 0` — the baseline of §14.1, and also the policy the agent will play
        against an opponent it has not observed (§6.2).
    ``ablation``
        the amortised head's output with no gradient fit (`K = 0`, §5.4).
    ``fit``
        *K* gradient steps from that initialisation (§5.5).

    Only **non-observer** players are scored: the observer is hero, and hero's
    own actions are not what the mechanism is for.
    """
    max_players = game["max_players"]
    n_actions = game["n_actions"]
    counts = list(cfg["observed_hand_counts"])
    n_eval_hands = cfg["eval_hands_per_session"]
    K = cfg["K"]
    weights = loss_weights(cfg)

    rows = []
    # One global bar over sessions × observation windows, not a bar per session
    # (`CLAUDE.md` §5). A fit is the unit because it is the expensive step, and
    # its cost varies by two orders of magnitude across the window lengths —
    # which is exactly why the ETA has to average over everything done so far
    # rather than over the last few fits.
    bar = progress(total=len(sessions) * len(counts), desc=f"eval:{tag}",
                   unit="fit")
    for s in sessions:
        tokens = s.tokens(max_players, n_actions)
        window = tokens[:max(counts)]
        tail = [t for t in tokens[max(counts):] if len(t) > 0]
        if not tail:
            # Skipped sessions still advance the bar by the fits they would
            # have contributed, so it reaches its total.
            bar.update(len(counts))
            continue
        assert len(tail) >= 1
        eval_batch = collate(tail, device=device)

        targets = [(slot, m) for slot, m in enumerate(s.members) if slot != 0]
        base = {}
        for slot, member in targets:
            ce, n_tok = evaluate_ce(net, eval_batch, net.zero_emb(eval_batch),
                                    member_filter=member)
            base[slot] = (ce, n_tok)

        for n in counts:
            observed = [t for t in window[:n] if len(t) > 0]
            if observed:
                fit_batch = collate(observed, device=device)
                init = net.amortised_init(fit_batch, s.num_players)
                fitted = fit_embeddings(
                    net, fit_batch, s.num_players, steps=K,
                    lr=cfg["fit_lr"], reg=cfg["fit_reg"], init=init,
                    weights=weights)
            else:
                # Cold start (§5.5): no history, so the vector is zero and the
                # ablation has nothing to pool over either.
                init = torch.zeros(s.num_players, net.d_emb, device=device)
                fitted = init

            emb_ablation = net.slot_emb(eval_batch, init)
            emb_fit = net.slot_emb(eval_batch, fitted)
            for slot, member in targets:
                ce_zero, n_tok = base[slot]
                if n_tok == 0:
                    continue
                ce_abl, _ = evaluate_ce(net, eval_batch, emb_ablation,
                                        member_filter=member)
                ce_fit, _ = evaluate_ce(net, eval_batch, emb_fit,
                                        member_filter=member)
                rows.append({
                    "set": tag,
                    "session": s.idx,
                    "num_players": s.num_players,
                    "stack_bb": s.stack_bb,
                    "slot": slot,
                    "member": member,
                    "observed_hands": n,
                    "observed_decisions": sum(len(t) for t in observed),
                    "eval_tokens": n_tok,
                    "ce_zero": ce_zero,
                    "ce_ablation": ce_abl,
                    "ce_fit": ce_fit,
                })
            bar.update(1)
    bar.close()
    log(f"[eval:{tag}] {len(rows)} measurement rows over {len(sessions)} sessions")
    return rows


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


METRICS = ("ce_zero", "ce_ablation", "ce_fit", "gain_ablation", "gain_fit")


def _with_gains(row):
    """`gain_*` = how much the embedding beat `e = 0`. Positive is better.

    The measurement §14.1 asks for is "against the `e = 0` baseline", and the
    difference is the part that is comparable across sets: two sets of sessions
    have different tables, different styles and therefore different intrinsic
    predictability, so their raw losses are not on the same scale — but the
    improvement each gets from conditioning on an embedding is.
    """
    return {
        **row,
        "gain_ablation": row["ce_zero"] - row["ce_ablation"],
        "gain_fit": row["ce_zero"] - row["ce_fit"],
    }


def _curve(rows):
    """Per-metric mean at each observed-hand count, aggregated by session.

    Rows from one session share their evaluation hands and their table, so they
    are not independent. Averaging within a session first makes the session the
    unit, which is what the quoted standard error is then the standard error of.
    """
    out = {}
    for n in sorted({r["observed_hands"] for r in rows}):
        sel = [r for r in rows if r["observed_hands"] == n]
        by_session = {}
        for r in sel:
            by_session.setdefault(r["session"], []).append(r)
        out[n] = {
            metric: _stats([
                sum(r[metric] for r in group) / len(group)
                for group in by_session.values()
            ])
            for metric in METRICS
        }
    return out


STACK_BUCKETS = [(10, 40), (41, 100), (101, 200), (201, 300)]


def _stack_bucket(stack_bb):
    for lo, hi in STACK_BUCKETS:
        if lo <= stack_bb <= hi:
            return f"{lo}-{hi}"
    return "other"


def aggregate(rows):
    """Everything §14 asks to be reported, from the same measurement rows."""
    rows = [_with_gains(r) for r in rows]
    sets = sorted({r["set"] for r in rows})
    curves = {s: _curve([r for r in rows if r["set"] == s]) for s in sets}
    report = {"curves": curves}  # §14.1

    # §14.2 — B1(b). Two readings, both reported:
    #   `ce_fit`   the raw loss gap §14.2 asks for — confounded by the two sets
    #              having different tables and different intrinsic difficulty;
    #   `gain_fit` the gap in improvement over each set's own e=0 baseline —
    #              the confound cancels, so this is the one that answers "does
    #              the embedding still work on a style never trained on".
    if "seen" in sets and "unseen" in sets:
        gap = {}
        for n in sorted(set(curves["seen"]) & set(curves["unseen"])):
            gap[n] = {
                metric: (curves["unseen"][n][metric]["mean"]
                         - curves["seen"][n][metric]["mean"])
                for metric in METRICS
            }
        report["style_generalisation_gap"] = gap

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
    return report


def format_report(report, log):
    log("")
    log("=" * 78)
    log("G1 — opponent embedding: does it carry style, and does it generalise?")
    log("=" * 78)

    def cell(st):
        return f"{st['mean']:.4f}±{st['se']:.4f}" if st["n"] else "—"

    for s, curve in report["curves"].items():
        log(f"\n[§14.1 | {s} styles] action-prediction CE in nats, "
            f"mean ± SE over sessions")
        log(f"  {'hands':>6} {'e=0':>16} {'ablation K=0':>16} {'fit K>0':>16}"
            f" {'gain of fit':>16} {'sess':>5}")
        for n, m in curve.items():
            log(f"  {n:>6} {cell(m['ce_zero']):>16} "
                f"{cell(m['ce_ablation']):>16} {cell(m['ce_fit']):>16} "
                f"{cell(m['gain_fit']):>16} {m['ce_fit']['n']:>5}")
        log("  gain of fit = e=0 CE minus fitted CE; positive means the "
            "embedding helped.")

    if "style_generalisation_gap" in report:
        log("\n[§14.2 | B1(b)] unseen-style minus seen-style")
        log(f"  {'hands':>6} {'Δ CE(fit)':>12} {'Δ gain(fit)':>13}")
        for n, g in report["style_generalisation_gap"].items():
            log(f"  {n:>6} {g['ce_fit']:>12.4f} {g['gain_fit']:>13.4f}")
        log("  Δ CE is the raw gap §14.2 asks for; it is confounded by the two "
            "sets having")
        log("  different tables. Δ gain compares each set against its own e=0 "
            "baseline, so a")
        log("  Δ gain near zero is the evidence B1(b) needs: fresh style draws "
            "are read as well")
        log("  as trained ones. A large negative Δ gain is B1(b) failing.")

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
    log("")


# ----------------------------------------------------------------------- driver


def run(config, log, out_dir):
    game = config["game"]
    seed = config.get("seed", 0)
    device = resolve_device(config.get("device", "auto"))
    log(f"device: {device}")

    rng = np.random.default_rng(seed)
    members, descriptors = build_pool(config, rng, device=device, log=log)
    log(f"pool: {len(members)} members "
        f"({len({d['base'] for d in descriptors})} bases)")

    fresh, fresh_desc = fresh_style_variants(
        members, descriptors, config["corpus"]["n_unseen_members"],
        rng, config.get("style", {}), tag="fresh")
    log(f"fresh style draws (never in training): {len(fresh)}")

    all_members = members + fresh
    seen_ids = list(range(len(members)))
    unseen_ids = list(range(len(members), len(all_members)))

    driver = LockstepDriver(all_members, game["n_actions"])
    corpus_cfg = config["corpus"]
    batch_size = corpus_cfg.get("driver_batch_size", 256)

    train_sessions = build_sessions(
        rng, seen_ids, game, corpus_cfg["train_sessions"],
        corpus_cfg["train_hands_per_session"], seed_base=10_000_000, tag="train")
    play(driver, train_sessions, batch_size, log, "train")

    eval_hands = max(corpus_cfg["observed_hand_counts"]) + \
        corpus_cfg["eval_hands_per_session"]
    seen_sessions = build_sessions(
        rng, seen_ids, game, corpus_cfg["eval_sessions"], eval_hands,
        seed_base=20_000_000, tag="seen")
    unseen_sessions = build_sessions(
        rng, unseen_ids, game, corpus_cfg["eval_sessions"], eval_hands,
        seed_base=30_000_000, tag="unseen")
    play(driver, seen_sessions, batch_size, log, "seen")
    play(driver, unseen_sessions, batch_size, log, "unseen")

    net = OpponentEmbeddingNet(
        config["embedding_net"], game["n_actions"], game["max_players"],
        n_members=len(all_members),
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"embedding network: {n_params/1e6:.2f}M parameters")

    torch.manual_seed(seed)
    history = train_embedding_net(net, train_sessions, config["train"], game,
                                  device, log, seed)

    eval_cfg = {**config["embedding_net"], **corpus_cfg, **config["train"]}
    rows = (evaluate_sessions(net, seen_sessions, eval_cfg, game, device, log,
                              "seen")
            + evaluate_sessions(net, unseen_sessions, eval_cfg, game, device,
                                log, "unseen"))
    report = aggregate(rows)
    format_report(report, log)

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config": config,
        "pool": descriptors,
        "fresh_style_draws": fresh_desc,
        "train_history": history,
        "rows": rows,
        "report": report,
    }
    path = os.path.join(out_dir, "g1_report.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    log(f"wrote {path}")
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
