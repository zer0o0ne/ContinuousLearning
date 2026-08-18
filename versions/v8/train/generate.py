"""Label generation, end to end (CONCEPT.md §8, `PLAN_PIPELINE.md` S7).

One turn of the outer loop's middle three lines:

    play hands → fit / refresh opponent embeddings (§5.5) → oracle labels at
    hero's decisions (§7) → shards on disk

and nothing else: no agent training (S9 orchestrates that), no pool sampling
policy (S8 owns it, and it arrives here as `sampler`).

**Hero is slot 0** of every session, as in G1 (`ARCHITECTURE.md` §4,
interpretive decision 6), and the sessions themselves come from `env/session.py`
— the same button rotation, the same uniform 2–9 × 10–300 BB draw that G1 used
(D4). That sharing is the point: the state distribution the agent is trained on
and the one the embedding network was measured on have to be the same object,
not two implementations of the same paragraph.

**Order is the whole difficulty.** Three things have to line up at every label:

1. the vectors hero *acted* with,
2. the vectors *attached* to the label, and
3. the vectors hero would have had at deployment — fitted from the hands
   already played at that point, never from the hand being labelled.

They line up here by construction rather than by care: the session is played in
blocks of `R` hands, hero's member for block *b* is built from the vectors
fitted over blocks `0 … b−1`, and the label of a decision in block *b* stores
that same table. Between refreshes the vectors are stale, deliberately (§5.5).
A vector fitted over the whole session and attached afterwards would be a future
leak that no loss curve would ever show.

**The observation is kept, not rebuilt.** Hero's member is wrapped in a recorder
that stores `hand_tokens(..., pending=ctx)` for every decision it is asked
about. That is the observation hero actually acted on, so the stored prefix is
§9-parity-correct for the same reason the live one is — as opposed to
re-tokenising a truncated copy of the finished record afterwards, which is a
second construction path for observations and therefore a place for the two to
drift.

**One member is one seat at one table** (`agent/policy.py`), and hero's seat
rotates, so hero is not one member but one per seat of each session, rebuilt at
every refresh. Hero's play forwards therefore do not batch across sessions. That
is deliberate and cheap: a label costs `|A| × samples_per_action` rollout hands
against a *single* hero member, so the rollouts — which are essentially the
whole cost — batch exactly as before.

**The shards are `.npz` written without timestamps**, so the same seed produces
byte-identical files. `np.savez` stamps every zip entry with the wall clock,
which would make reproducibility unverifiable by comparison.
"""

import io
import os
import zipfile
from dataclasses import fields

import numpy as np

from env.session import Session, build_sessions, play
from nets.embedding_net import fit_embeddings, loss_weights
from nets.features import HandTokens, collate, hand_tokens
from oracle.rollout import LabelStats, OracleConfig, action_values
from pool.base import PoolMember
from utils import progress

TAG = "labels"
HERO_SLOT = 0


class _ObservedHero(PoolMember):
    """Hero's member, keeping the observation it was actually shown (§9).

    Delegates the whole last mile — `policy`, style and all — to the member it
    wraps, so wrapping cannot change a single action. What it adds is the
    tokenised prefix of every decision it was asked about, keyed by the hand and
    the decision index the driver is about to write.
    """

    def __init__(self, inner, slot_of_seat, max_players, n_actions,
                 observer_pos):
        super().__init__(f"hero@seat{observer_pos}", n_actions)
        self.inner = inner
        self.slot_of_seat = list(slot_of_seat)
        self.max_players = max_players
        self.observer_pos = int(observer_pos)
        self.seen = {}

    def logits(self, contexts):
        return self.inner.logits(contexts)

    def policy(self, contexts):
        for ctx in contexts:
            meta = ctx.record.spec.meta
            key = (int(meta["session"]), int(meta["hand"]),
                   len(ctx.record.decisions))
            self.seen[key] = hand_tokens(
                ctx.record, observer_pos=int(ctx.acting_pos),
                slot_of_seat=self.slot_of_seat, max_players=self.max_players,
                n_actions=self.n_actions, pending=ctx)
        return self.inner.policy(contexts)


def _slot_of_seat_at(num_players, hero_seat):
    """The rotation in which hero (slot 0) sits at `hero_seat`.

    `Session.seat_of_slot(0, h) = (−h) mod n`, so the seat fixes the hand index
    modulo the table size, and that in turn fixes every seat's slot.
    """
    hand_idx = (-int(hero_seat)) % int(num_players)
    return [(seat + hand_idx) % num_players for seat in range(num_players)]


def _pad_vectors(fitted, max_players, d_emb):
    """A (max_players, d_emb) table from a (num_players, d_emb) fit.

    Slots the session does not have stay zero — the cold-start value (§5.5), and
    the value the agent reads for a seat that is not at the table.
    """
    table = np.zeros((max_players, d_emb), dtype=np.float32)
    table[:len(fitted)] = np.asarray(fitted, dtype=np.float32)
    return table


# ------------------------------------------------------------------- the shards


def _write_npz(path, arrays):
    """`np.savez` with the timestamps taken out.

    Every entry of a zip carries a modification time, and `np.savez` fills it
    from the wall clock — so two runs of the same seed produce different bytes
    and "the run is reproducible" stops being a checkable claim. Fixing the
    stamp is the whole difference; the file is an ordinary `.npz` that
    `np.load` reads.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zf.writestr(info, buf.getvalue())


def _shard_arrays(labels):
    """One shard's worth of labels as flat arrays.

    Token sequences are ragged, so they are concatenated with an offset index
    rather than padded — padding a corpus to its longest hand is memory spent on
    nothing. The embedding tables are stored **once per table** and referenced:
    every label of a block shares one, and one row per label would be the
    dominant cost of the file for no information at all.
    """
    tokens = [lab["tokens"] for lab in labels]
    lengths = [len(t) for t in tokens]
    out = {"tok_offsets": np.concatenate(
        [[0], np.cumsum(lengths)]).astype(np.int64)}
    for f in fields(HandTokens):
        out[f"tok_{f.name}"] = np.concatenate(
            [getattr(t, f.name) for t in tokens], axis=0)

    tables, index = [], []
    for lab in labels:
        table = lab["embeddings"]
        for k, seen in enumerate(tables):
            if seen is table:
                index.append(k)
                break
        else:
            index.append(len(tables))
            tables.append(table)
    out["emb_index"] = np.asarray(index, dtype=np.int64)
    out["emb_tables"] = np.stack(tables).astype(np.float32)

    out["q"] = np.stack([lab["q"] for lab in labels]).astype(np.float64)
    out["legal"] = np.stack([lab["legal"] for lab in labels]).astype(bool)
    for key in ("pot_bb", "facing_bet_bb"):
        out[key] = np.asarray([lab[key] for lab in labels], dtype=np.float64)
    for key in ("session", "hand", "decision", "num_players", "stack_bb",
                "hero_seat"):
        out[key] = np.asarray([lab[key] for lab in labels], dtype=np.int64)
    return out


def load_shard(path):
    """The labels of one shard, in the order they were written.

    Each entry is the dict `generate_labels` wrote: `tokens` (a `HandTokens`
    whose last token is the pending decision, S1), `q` in BB with `nan` off
    `legal`, `pot_bb` and `facing_bet_bb` for the §6.2 normalisation, the
    `embeddings` table in force when hero acted, and the table metadata.
    """
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    offsets = data["tok_offsets"]
    tables = data["emb_tables"]
    labels = []
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        tokens = HandTokens(**{f.name: data[f"tok_{f.name}"][lo:hi]
                               for f in fields(HandTokens)})
        labels.append({
            "tokens": tokens,
            "q": data["q"][i],
            "legal": data["legal"][i],
            "pot_bb": float(data["pot_bb"][i]),
            "facing_bet_bb": float(data["facing_bet_bb"][i]),
            "embeddings": tables[int(data["emb_index"][i])],
            **{k: int(data[k][i]) for k in
               ("session", "hand", "decision", "num_players", "stack_bb",
                "hero_seat")},
        })
    return labels


# -------------------------------------------------------------------- the phase


def generate_labels(driver, pool, sampler, embed_net, agent_member, cfg,
                    out_dir, log):
    """Play sessions, fit opponent embeddings, label hero's decisions, write shards.

    Returns a manifest dict: shard paths, label count, aggregate `LabelStats`.

    Args:
        driver: a `LockstepDriver` over `pool`. Its `pool` attribute is extended
            with hero's per-seat members for the duration of the call and
            restored afterwards.
        pool: the opponent members. Hero is **not** one of them — it is built by
            `agent_member` and seated on top.
        sampler: `sample_table(k) -> k member indices into pool` (§4.4, S8).
            Asked for hero's opponents only, since hero occupies slot 0 and is
            not drawn from the pool.
        embed_net: the trained `OpponentEmbeddingNet`, used for the §5.5 fit.
        agent_member: builds hero's member for one seat —
            `agent_member(observer_pos, slot_of_seat, embeddings) -> PoolMember`.
            A factory rather than a member because "one member is one seat at
            one table" (`agent/policy.py`) and because the vectors it conditions
            on change every `R` hands. At iteration 0 the factory ignores both
            arguments and returns the pool member of §7.1; from iteration 1 it
            returns an `AgentPoolMember` over the current network.
        cfg: `seed`, `n_sessions`, `hands_per_session`, `driver_batch_size`,
            `labels_per_shard`, and the `game`, `embedding_net` and `oracle`
            sections of §8.1.
        out_dir: directory the shards are written into.
        log: a `Logger`.
    """
    game = cfg["game"]
    emb_cfg = cfg["embedding_net"]
    # The §8.1 `oracle` section also carries the §6.2 EV divisor and
    # temperature. Those belong to the *target*, which is built from `q` where
    # the label is read, so they are not the rollout's business and are left
    # alone here rather than silently accepted and ignored.
    rollout_keys = {f.name for f in fields(OracleConfig)}
    ocfg = OracleConfig(**{k: v for k, v in cfg.get("oracle", {}).items()
                           if k in rollout_keys})
    max_players = int(game["max_players"])
    n_actions = int(game["n_actions"])
    hands_per_session = int(cfg["hands_per_session"])
    R = int(emb_cfg["R"])
    assert R > 0, "R is the refresh interval in hands (§5.5) and must be ≥ 1"
    device = next(embed_net.parameters()).device
    weights = loss_weights(emb_cfg)
    rng = np.random.default_rng(cfg["seed"])

    sessions = build_sessions(rng, list(range(len(pool))), game,
                              int(cfg["n_sessions"]), hands_per_session,
                              seed_base=int(cfg["seed"]) * 1_000_000, tag=TAG)

    # `build_sessions` draws its members uniformly and those draws are dropped:
    # who sits at the table is the sampler's decision (§4.4), and the table
    # *configuration* is the part being reused. The dropped draw is deliberate —
    # keeping it leaves the uniform 2–9 × 10–300 BB stream identical to G1's,
    # which is the whole reason the two share this function (D4).
    #
    # Hero occupies slot 0, so the sampler is asked for the opponents only;
    # `members[0]` is hero and is not a pool index at all.
    for s in sessions:
        opponents = list(sampler.sample_table(s.num_players - 1))
        assert len(opponents) == s.num_players - 1, (
            f"the sampler returned {len(opponents)} members for a "
            f"{s.num_players}-handed table's {s.num_players - 1} opponents")
        s.members = [-1] + [int(m) for m in opponents]

    # Hero's members: one per (session, seat), twice over. The recorded copy
    # plays the sessions and keeps the observations; the plain copy plays hero
    # inside the oracle's rollouts, where recording tens of thousands of
    # throw-away hands would be a memory leak with nothing reading it.
    play_pool = list(pool)
    hero_plain, hero_rec = [], []
    for s in sessions:
        hero_plain.append([len(play_pool) + i for i in range(s.num_players)])
        play_pool += [None] * s.num_players
        hero_rec.append([len(play_pool) + i for i in range(s.num_players)])
        play_pool += [None] * s.num_players

    for s, rec in zip(sessions, hero_rec):
        for h, spec in enumerate(s.specs):
            sos = s.slot_of_seat(h)
            spec.seat_members = [
                rec[seat] if sos[seat] == HERO_SLOT else s.members[sos[seat]]
                for seat in range(s.num_players)]

    saved_pool = driver.pool
    driver.pool = play_pool
    try:
        block_vectors = _play_sessions(
            driver, play_pool, sessions, hero_rec, agent_member, embed_net,
            emb_cfg, weights, max_players, n_actions, R, device, cfg, log)
        manifest = _label_sessions(
            driver, play_pool, sessions, hero_plain, hero_rec, block_vectors,
            agent_member, ocfg, R, max_players, cfg, out_dir, log)
    finally:
        driver.pool = saved_pool
    return manifest


def _play_sessions(driver, play_pool, sessions, hero_rec, agent_member,
                   embed_net, emb_cfg, weights, max_players, n_actions, R,
                   device, cfg, log):
    """Play every session in blocks of `R` hands, refitting between blocks.

    Returns, per session, the embedding table in force during each block. The
    table of block *b* is fitted over the hands of blocks `0 … b−1` and over
    nothing else, which is the property `tests/test_label_generation.py` pins.
    """
    hands_per_session = int(cfg["hands_per_session"])
    n_blocks = (hands_per_session + R - 1) // R
    d_emb = embed_net.d_emb
    block_vectors = [[_pad_vectors(np.zeros((0, d_emb)), max_players, d_emb)]
                     for _ in sessions]

    # The recorders are built once and keep their observations for the whole
    # job; a refresh swaps the member inside them, so nothing hero saw in an
    # earlier block is thrown away with the vectors that produced it.
    for s, rec in zip(sessions, hero_rec):
        for seat in range(s.num_players):
            play_pool[rec[seat]] = _ObservedHero(
                None, _slot_of_seat_at(s.num_players, seat), max_players,
                n_actions, seat)

    bar = progress(total=len(sessions) * hands_per_session, desc="play",
                   unit="hand")
    for b in range(n_blocks):
        lo, hi = b * R, min((b + 1) * R, hands_per_session)
        for s, rec, vectors in zip(sessions, hero_rec, block_vectors):
            for seat in range(s.num_players):
                hero = play_pool[rec[seat]]
                hero.inner = agent_member(seat, hero.slot_of_seat, vectors[b])

        blocks = [Session(idx=s.idx, num_players=s.num_players,
                          stack_bb=s.stack_bb, members=s.members,
                          specs=s.specs[lo:hi]) for s in sessions]
        play(driver, blocks, int(cfg["driver_batch_size"]), log,
             f"{TAG}:block{b}", bar=False)
        for s, blk in zip(sessions, blocks):
            s.records.extend(blk.records)
        bar.update(sum(len(blk.specs) for blk in blocks))

        if b + 1 == n_blocks:
            break
        for s, vectors in zip(sessions, block_vectors):
            observed = [t for t in s.tokens(max_players, n_actions) if len(t)]
            if not observed:
                vectors.append(vectors[-1])
                continue
            batch = collate(observed, device=device)
            fitted = fit_embeddings(
                embed_net, batch, s.num_players, steps=int(emb_cfg["K"]),
                lr=emb_cfg["fit_lr"], reg=emb_cfg["fit_reg"],
                init=embed_net.amortised_init(batch, s.num_players),
                weights=weights)
            vectors.append(_pad_vectors(fitted.cpu().numpy(), max_players,
                                        d_emb))
    bar.close()
    log(f"[{TAG}] played {sum(len(s.records) for s in sessions)} hands over "
        f"{len(sessions)} sessions in {n_blocks} blocks of {R}")
    return block_vectors


def _hero_decisions(sessions, hero_rec):
    """Every decision hero took, as `(session, hand, decision_idx)`."""
    out = []
    for i, (s, rec) in enumerate(zip(sessions, hero_rec)):
        for h, record in enumerate(s.records):
            hero_seat = s.seat_of_slot(HERO_SLOT, h)
            for d, dec in enumerate(record.decisions):
                if int(dec["acting_pos"]) == hero_seat:
                    out.append((i, h, d))
    return out


def _label_sessions(driver, play_pool, sessions, hero_plain, hero_rec,
                    block_vectors, agent_member, ocfg, R, max_players, cfg,
                    out_dir, log):
    """One oracle label per hero decision, sharded to disk.

    The bar counts **hero decisions across the whole job** (`CLAUDE.md` §5):
    that is what the phase costs, and a bar per session would answer a question
    nobody is asking. Playing the hands, above, has its own bar because it
    happens first and in a different unit; nothing is nested.
    """
    os.makedirs(out_dir, exist_ok=True)
    todo = _hero_decisions(sessions, hero_rec)
    log(f"[{TAG}] {len(todo)} hero decisions to label")

    per_shard = int(cfg["labels_per_shard"])
    shards, buffer = [], []
    n_labels = 0
    forwards = seconds = rollouts = 0
    collisions, dropped = [], 0

    def flush():
        if not buffer:
            return
        path = os.path.join(out_dir, f"shard_{len(shards):04d}.npz")
        _write_npz(path, _shard_arrays(buffer))
        shards.append(path)
        buffer.clear()

    bar = progress(total=len(todo), desc="label", unit="label")
    current = (None, None)
    for i, h, d in todo:
        s = sessions[i]
        block = h // R
        if (i, block) != current:
            # Hero plays its own rollouts with the vectors it acted under; the
            # member is rebuilt per block because the pool slot holds whichever
            # block was played last.
            current = (i, block)
            for seat in range(s.num_players):
                sos = _slot_of_seat_at(s.num_players, seat)
                play_pool[hero_plain[i][seat]] = agent_member(
                    seat, sos, block_vectors[i][block])

        record = s.records[h]
        hero_seat = s.seat_of_slot(HERO_SLOT, h)
        label_rng = np.random.default_rng([int(cfg["seed"]), i, h, d])
        q, legal, stats = action_values(
            record, d, driver, play_pool, hero_plain[i][hero_seat], ocfg,
            label_rng)
        forwards += stats.forwards
        seconds += stats.seconds
        rollouts += stats.n_rollouts
        collisions.append(stats.collision_rate)
        bar.update(1)

        if not np.isfinite(q[legal]).all():
            # Every joint draw collided, so there is no EV to build a target
            # from (`oracle/rollout.py`). Dropping it is the honest outcome; the
            # count is in the manifest because a large one is a broken run.
            dropped += 1
            continue

        tokens = play_pool[hero_rec[i][hero_seat]].seen[(s.idx, h, d)]
        _stack_bb, pot_bb, to_call_bb = (float(x) for x in tokens.scalars[-1])
        buffer.append({
            "tokens": tokens, "q": q, "legal": legal,
            "pot_bb": pot_bb, "facing_bet_bb": to_call_bb,
            "embeddings": block_vectors[i][block],
            "session": s.idx, "hand": h, "decision": d,
            "num_players": s.num_players, "stack_bb": s.stack_bb,
            "hero_seat": hero_seat,
        })
        n_labels += 1
        if len(buffer) >= per_shard:
            flush()
    flush()
    bar.close()

    stats = LabelStats(
        forwards=forwards, seconds=seconds,
        collision_rate=float(np.mean(collisions)) if collisions else 0.0,
        n_rollouts=rollouts)
    log(f"[{TAG}] {n_labels} labels in {len(shards)} shards, {dropped} dropped, "
        f"{stats.forwards} forwards, {stats.seconds:.1f}s of labelling, "
        f"collision rate {100 * stats.collision_rate:.1f}%")
    return {"shards": shards, "n_labels": n_labels, "n_dropped": dropped,
            "n_hands": sum(len(s.records) for s in sessions),
            "n_sessions": len(sessions), "stats": stats}
