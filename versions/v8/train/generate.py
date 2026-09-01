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

**Labelling resumes at the hand boundary, and re-plays only what it has to.**
This is the longest-running thing in the project — `CONCEPT.md` §13 budgets an
iteration of it in days — so losing it whole to a crash in its middle is not an
acceptable failure mode. Three files make a resume possible:

* `play.json` and `vectors.npz`, written once the corpus has been played: the
  configuration of every session, hero's result against each member, and the
  per-block embedding tables of §5.5. The tables are the whole reason a hand can
  be re-played on its own — the vectors hero acts under in block *b* are the
  ones fitted over blocks `0 … b−1`, and reading them off disk gives exactly
  those vectors back without the hands that produced them.
* `progress.json`, rewritten after every flushed shard: how many leading *hands*
  are completely labelled. Hands and not decisions, because the hand is the unit
  that gets re-played.

A second call with the same `out_dir` therefore plays exactly the hands that are
not labelled yet, and refits nothing at all. That is not an optimisation. A
re-played hand is re-derived from the network and from the fits, and neither is
bit-reproducible across processes on a GPU: one action flipped a few hundred
hands in gives a *different* hand, and the shards already on disk then belong to
a corpus that no longer exists — which is how a resume ends up splicing two
label sets together, or falling over when the decision it is resuming at is not
there any more. Playing only the unlabelled tail removes the question: a
labelled hand is read back from its shard and never recomputed.

What still has to agree is the frame around those hands. The sessions are a
function of the seed and of the sampler; `play.json` records what they were, and
a directory whose sessions are not this call's sessions is set aside rather than
resumed into (`_set_aside`). Within a hand, every label draws from its own
`(seed, session, hand, decision)` generator, so a label computed after a resume
is bit-identical to the one that would have been computed without one.
"""

import io
import json
import os
import zipfile
from dataclasses import fields

import numpy as np

from env.session import (Session, build_sessions, hand_seed_bases,
                         phase_hands, play)
from nets.embedding_net import fit_embeddings, loss_weights
from oracle.parallel import (ForwardServer, collect, join_workers,
                             runner_table, spawn_workers, worker_count)
from oracle.posterior import PosteriorCache
from nets.features import HandTokens, collate, hand_tokens
from oracle.rollout import (LabelStats, OracleConfig,
                            action_values_batch)
from pool.base import PoolMember
from utils import progress

TAG = "labels"
HERO_SLOT = 0
PROGRESS = "progress.json"
PLAY = "play.json"
VECTORS = "vectors.npz"


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

    Returns a manifest dict: shard paths, label count, aggregate `LabelStats`,
    and `results` — hero's BB and hand count against each member it sat with,
    which is what §4.4's PFSP is driven by (`_results_by_member`).

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

    # The same layout `pipeline.py` uses for the corpus, asked the same way, so
    # the two phases cannot disagree about who owns which seeds (`env/session`).
    seed_base = hand_seed_bases(int(cfg["seed"]), phase_hands(cfg))[0]["labels"]
    sessions = build_sessions(rng, list(range(len(pool))), game,
                              int(cfg["n_sessions"]), hands_per_session,
                              seed_base=seed_base, tag=TAG)

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

    # What an earlier call to this phase left behind, read before a single hand
    # is played: the corpus it played (`played`) and how far into the hands it
    # got labelling (`done`). Both are `None` for a fresh directory, and for one
    # that cannot be resumed into — which is set aside here, while setting it
    # aside is still free.
    os.makedirs(out_dir, exist_ok=True)
    signature = _play_signature(sessions, seed_base, R, hands_per_session)
    played, done = _resume_state(out_dir, signature, log)
    first_hand = _first_hands(done, len(sessions), hands_per_session)

    saved_pool = driver.pool
    driver.pool = play_pool
    try:
        if played is None:
            block_vectors = _play_sessions(
                driver, play_pool, sessions, hero_rec, agent_member, embed_net,
                emb_cfg, weights, max_players, n_actions, R, device, cfg, log)
            results = _results_by_member(sessions)
            n_hands = sum(len(s.records) for s in sessions)
            _write_play(out_dir, signature, block_vectors, results, n_hands)
        else:
            # The hands already labelled are not re-played, so neither hero's
            # results nor the hand count can be recomputed from the records
            # this call holds — they are read back from the call that did play
            # them, over the whole corpus (`_write_play`).
            block_vectors, results, n_hands = played
            _play_sessions(
                driver, play_pool, sessions, hero_rec, agent_member, embed_net,
                emb_cfg, weights, max_players, n_actions, R, device, cfg, log,
                first_hand=first_hand, vectors=block_vectors)
        manifest = _label_sessions(
            driver, play_pool, sessions, hero_plain, hero_rec, block_vectors,
            agent_member, ocfg, R, max_players, cfg, out_dir, done, log)
    finally:
        driver.pool = saved_pool
    manifest["results"] = results
    manifest["n_hands"] = n_hands
    return manifest


def _results_by_member(sessions):
    """Hero's result against each member it sat with — `PoolSampler.update`'s input.

    §4.4's PFSP is driven by how hero *does* against a member, and the hands
    that answer that are the ones this phase already played. Returning them here
    rather than recomputing them anywhere else is the only way the loop can have
    them at all: the records live inside this call.

    **A hand is credited to every opponent at the table, in full.** Hero's chip
    delta in a multiway hand is not divisible between the opponents who produced
    it — hero played that hand against all of them — and splitting it by the
    table size would make a nine-handed beating look an eighth as bad as the
    heads-up one it is being compared against, which is the opposite of what
    §4.4 samples on. The consequence to keep in mind is the one that follows
    directly: the number is "hero's BB/100 while member *i* was at the table",
    so at nine-handed it carries eight opponents' worth of noise. `result_decay`
    (D10) is what stops that noise from accumulating forever.
    """
    out = {}
    for s in sessions:
        for h, record in enumerate(s.records):
            hero_seat = s.seat_of_slot(HERO_SLOT, h)
            bb = float(record.rewards[hero_seat]) / float(record.spec.big_blind)
            for slot in range(1, s.num_players):
                entry = out.setdefault(int(s.members[slot]),
                                       {"hero_bb": 0.0, "n_hands": 0})
                entry["hero_bb"] += bb
                entry["n_hands"] += 1
    return out


def _play_sessions(driver, play_pool, sessions, hero_rec, agent_member,
                   embed_net, emb_cfg, weights, max_players, n_actions, R,
                   device, cfg, log, first_hand=None, vectors=None):
    """Play every session in blocks of `R` hands, refitting between blocks.

    Returns, per session, the embedding table in force during each block. The
    table of block *b* is fitted over the hands of blocks `0 … b−1` and over
    nothing else, which is the property `tests/test_label_generation.py` pins.

    A resumed call passes both `first_hand` — the first hand of each session
    that still needs labelling — and `vectors`, the tables the earlier call
    fitted. It then plays that tail and nothing else, and fits nothing: the
    tables of block *b* are a function of the hands before it, so re-fitting
    them from a corpus whose head is missing would be a different fit, and
    re-playing the head to avoid that is the thing being avoided (module
    docstring). `session.records` is padded with `None` up to `first_hand` so
    that a hand keeps the index it was labelled under.
    """
    hands_per_session = int(cfg["hands_per_session"])
    n_blocks = (hands_per_session + R - 1) // R
    d_emb = embed_net.d_emb
    first_hand = ([0] * len(sessions) if first_hand is None
                  else [int(f) for f in first_hand])
    fitting = vectors is None
    block_vectors = ([[_pad_vectors(np.zeros((0, d_emb)), max_players, d_emb)]
                      for _ in sessions] if fitting else vectors)

    # The recorders are built once and keep their observations for the whole
    # job; a refresh swaps the member inside them, so nothing hero saw in an
    # earlier block is thrown away with the vectors that produced it.
    for s, rec, start in zip(sessions, hero_rec, first_hand):
        s.records = [None] * start
        for seat in range(s.num_players):
            play_pool[rec[seat]] = _ObservedHero(
                None, _slot_of_seat_at(s.num_players, seat), max_players,
                n_actions, seat)

    # One bar over every hand of the job. The refit between blocks is not a
    # hand and does not advance it — but it is minutes of silence, and a bar
    # that does not move is indistinguishable from a hung run, which is the
    # whole reason §5 asks for one. So the fit reports itself in the postfix:
    # the bar stands still and *says* what it is standing still for. The ETA
    # stays honest either way, because `smoothing=0` averages the fits into the
    # elapsed time they actually cost.
    bar = progress(total=sum(hands_per_session - f for f in first_hand),
                   desc="play", unit="hand")
    for b in range(n_blocks):
        lo, hi = b * R, min((b + 1) * R, hands_per_session)
        blocks, played = [], []
        for s, rec, start, vecs in zip(sessions, hero_rec, first_hand,
                                       block_vectors):
            if max(lo, start) >= hi:      # every hand of it is already labelled
                continue
            for seat in range(s.num_players):
                hero = play_pool[rec[seat]]
                hero.inner = agent_member(seat, hero.slot_of_seat, vecs[b])
            blocks.append(Session(idx=s.idx, num_players=s.num_players,
                                  stack_bb=s.stack_bb, members=s.members,
                                  specs=s.specs[max(lo, start):hi]))
            played.append(s)

        if blocks:
            play(driver, blocks, int(cfg["driver_batch_size"]), log,
                 f"{TAG}:block{b}", bar=False)
            for s, blk in zip(played, blocks):
                s.records.extend(blk.records)
            bar.update(sum(len(blk.specs) for blk in blocks))

        if b + 1 == n_blocks or not fitting:
            continue
        every = max(1, len(sessions) // 50)   # ≤ 50 refreshes per refit round
        for fitted_n, (s, vectors_of) in enumerate(zip(sessions, block_vectors), 1):
            if fitted_n % every == 0 or fitted_n == len(sessions):
                bar.set_postfix_str(
                    f"fit block {b + 1}: {fitted_n}/{len(sessions)} sessions",
                    refresh=True)
            observed = [t for t in s.tokens(max_players, n_actions) if len(t)]
            if not observed:
                vectors_of.append(vectors_of[-1])
                continue
            batch = collate(observed, device=device)
            fitted = fit_embeddings(
                embed_net, batch, s.num_players, steps=int(emb_cfg["K"]),
                lr=emb_cfg["fit_lr"], reg=emb_cfg["fit_reg"],
                init=embed_net.amortised_init(batch, s.num_players),
                weights=weights)
            vectors_of.append(_pad_vectors(fitted.cpu().numpy(), max_players,
                                           d_emb))
        bar.set_postfix_str("", refresh=True)
    bar.close()
    n_played = sum(hands_per_session - f for f in first_hand)
    log(f"[{TAG}] played {n_played} hands over {len(sessions)} sessions in "
        f"{n_blocks} blocks of {R}"
        + ("" if fitting else " (resumed: the labelled hands were not "
                              "re-played and nothing was refitted)"))
    return block_vectors


def _hero_decisions(sessions, hero_rec):
    """Every decision hero took, as `(session, hand, decision_idx)`.

    Over the hands this call *played*: a resumed call holds `None` where an
    already-labelled hand would be, and those are not decisions it has to label
    again.
    """
    out = []
    for i, (s, rec) in enumerate(zip(sessions, hero_rec)):
        for h, record in enumerate(s.records):
            if record is None:
                continue
            hero_seat = s.seat_of_slot(HERO_SLOT, h)
            for d, dec in enumerate(record.decisions):
                if int(dec["acting_pos"]) == hero_seat:
                    out.append((i, h, d))
    return out


def _play_signature(sessions, seed_base, R, hands_per_session):
    """What a played corpus has to agree with for its vectors to be reusable.

    Every table this phase sits hero at, and the two numbers that decide which
    hands those tables play: the base their seeds are dealt from
    (`env/session.hand_seed_bases` — it moves when a phase changes size) and the
    refresh interval that cuts them into blocks. Two calls that agree on all of
    it are playing the same sessions; a call that disagrees anywhere is playing
    different hands, and its labels cannot be spliced onto the ones on disk.
    """
    return {
        "seed_base": int(seed_base), "R": int(R),
        "hands_per_session": int(hands_per_session),
        "sessions": [{"idx": int(s.idx), "num_players": int(s.num_players),
                      "stack_bb": int(s.stack_bb),
                      "members": [int(m) for m in s.members]}
                     for s in sessions],
    }


def _write_play(out_dir, signature, block_vectors, results, n_hands):
    """The corpus a later call needs, once the hands have been played.

    The vectors go to their own `.npz` because they are an array and the rest is
    not; both are written before the first label, because a call that crashes in
    the middle of labelling is exactly the call this is for.
    """
    _write_npz(os.path.join(out_dir, VECTORS),
               {"vectors": np.stack([np.stack(v) for v in block_vectors])})
    with open(os.path.join(out_dir, PLAY), "w") as fh:
        json.dump({**signature, "n_hands": int(n_hands),
                   "results": {str(k): v for k, v in results.items()}}, fh)


def _set_aside(out_dir, why, log):
    """Move a labels directory that cannot be resumed out of the way.

    Refusing to splice two label sets is right; killing the run over it is not.
    The stale directory is *renamed*, never deleted: its shards are real labels
    that cost real time, and it is not this function's call to destroy them. The
    name is derived, not timestamped, so the same failure lands in the same place
    on a rerun.
    """
    for n in range(1000):
        aside = f"{out_dir}.stale" + (f".{n}" if n else "")
        if not os.path.exists(aside):
            break
    else:                                       # pragma: no cover - 1000 stales
        raise AssertionError(f"{out_dir}: too many set-aside directories")
    os.rename(out_dir, aside)
    os.makedirs(out_dir, exist_ok=True)
    log(f"[{TAG}] WARNING: cannot resume — {why}. The old directory is kept at "
        f"{aside} and this phase starts from the first hand. Delete it once you "
        f"are sure you do not want those labels.")


def _read_json(path):
    """`json.load`, or `None` if the file is missing or is not readable JSON."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def _stale_reason(out_dir, signature, before, done, has_progress):
    """Why this directory cannot be resumed into, or `None` if it can.

    Everything here is a way for the labels on disk to belong to hands this call
    is not going to play — which is the one failure a resume must never walk
    into, because nothing downstream can see a corpus spliced out of two runs.
    """
    play_path = os.path.join(out_dir, PLAY)
    if before is None:
        return (f"{play_path} is not there or cannot be read, so the corpus "
                f"those shards were labelled from is gone — its hands cannot "
                f"be replayed and its embedding fits cannot be recovered")
    for key, value in signature.items():
        if before.get(key) != value:
            return (f"{play_path} was written for a different phase: its "
                    f"`{key}` is not this call's, so these are other sessions "
                    f"playing other hands")
    if not os.path.exists(os.path.join(out_dir, VECTORS)):
        return (f"{play_path} is there but {VECTORS} is not, so the embedding "
                f"tables hero acted under are gone")
    if done is None:
        if has_progress:
            return (f"{os.path.join(out_dir, PROGRESS)} is there but cannot be "
                    f"read, so how much of this corpus is labelled is unknown")
        return None                    # played but never labelled: no shards yet
    missing = [q for q in done.get("shards", []) if not os.path.exists(q)]
    if missing:
        return (f"{os.path.join(out_dir, PROGRESS)} names shards that are not "
                f"on disk: {missing}")
    n_hands = int(signature["hands_per_session"]) * len(signature["sessions"])
    if not 0 <= int(done.get("hands_done", -1)) <= n_hands:
        return (f"{os.path.join(out_dir, PROGRESS)} says "
                f"{done.get('hands_done')} of {n_hands} hands are labelled, "
                f"which is not a place this phase can resume at")
    return None


def _resume_state(out_dir, signature, log):
    """`(played, done)` — the corpus of an earlier call and its label progress.

    `played` is `(block_vectors, results, n_hands)` when the hands of an earlier
    call can be picked up, and `None` when they cannot; `done` is the parsed
    `progress.json` when there are labels to keep, and `None` when there are
    none. A directory that cannot be resumed into is set aside and both come
    back `None` — the one thing that must never happen is resuming into it, and
    the one thing that need not happen is the run dying over it.
    """
    prog_path = os.path.join(out_dir, PROGRESS)
    before = _read_json(os.path.join(out_dir, PLAY))
    done = _read_json(prog_path)
    if before is None and done is None and not os.path.exists(prog_path):
        return None, None                                # a fresh directory

    why = _stale_reason(out_dir, signature, before, done,
                        os.path.exists(prog_path))
    if why is not None:
        _set_aside(out_dir, why, log)
        return None, None

    with np.load(os.path.join(out_dir, VECTORS)) as z:
        vectors = z["vectors"]
    n_blocks = -(-int(signature["hands_per_session"]) // int(signature["R"]))
    assert vectors.shape[:2] == (len(signature["sessions"]), n_blocks), (
        f"{os.path.join(out_dir, VECTORS)} holds {vectors.shape[:2]} embedding "
        f"tables and this phase has {(len(signature['sessions']), n_blocks)} "
        f"(session, block) pairs")
    played = ([list(per_session) for per_session in vectors],
              {int(k): v for k, v in before["results"].items()},
              int(before["n_hands"]))
    if done is None:
        log(f"[{TAG}] the corpus of an earlier call is on disk and no hand of "
            f"it is labelled yet: replaying every hand, refitting nothing")
        return played, None
    log(f"[{TAG}] resuming after {done['hands_done']}/"
        f"{int(signature['hands_per_session']) * len(signature['sessions'])} "
        f"hands ({done['n_labels']} labels in {len(done['shards'])} shards); "
        f"the labelled hands are not replayed")
    return played, done


def _first_hands(done, n_sessions, hands_per_session):
    """The first hand of each session that still needs labelling.

    `hands_done` is a count of leading hands over the whole job, laid out
    session by session in the order the labels were written, so it splits into
    per-session cursors by division. Sessions before the cursor are labelled
    whole and are not played at all.
    """
    frontier = int(done["hands_done"]) if done else 0
    return [min(max(frontier - i * hands_per_session, 0), hands_per_session)
            for i in range(n_sessions)]


def label_chunks(todo, R, size):
    """Consecutive labels of one `(session, block)`, at most `size` at a time.

    `size` is `oracle.labels_per_batch`: the labels of a chunk are built and
    then played through one `driver.run`, which is what keeps the lock-step
    group refilled to `batch_hands` instead of draining away as one label's
    rollouts finish (`oracle/rollout.py::action_values_batch`).

    A chunk never crosses a block boundary. Hero's member is reseated there
    (§5.5), and every label of a chunk has its rollouts built *before* any of
    them is played, so they must all be built against the same seated member.
    """
    chunk, key = [], None
    for item in todo:
        _pos, i, h, _d = item
        here = (i, h // R)
        if chunk and (here != key or len(chunk) >= int(size)):
            yield chunk
            chunk = []
        key = here
        chunk.append(item)
    if chunk:
        yield chunk


def _seat_hero(play_pool, hero_slots, session, block_vectors, block,
               agent_member):
    """Put hero's member in its pool slots for one `(session, block)`."""
    for seat in range(session.num_players):
        sos = _slot_of_seat_at(session.num_players, seat)
        member = agent_member(seat, sos, block_vectors[block])
        for slots in hero_slots:
            play_pool[slots[seat]] = member


def _label_requests(chunk, sessions, hero_plain, seed):
    """`action_values_batch`'s input for one chunk, in todo order."""
    out = []
    for _pos, i, h, d in chunk:
        s = sessions[i]
        hero_seat = s.seat_of_slot(HERO_SLOT, h)
        out.append((s.records[h], d, hero_plain[i][hero_seat],
                    np.random.default_rng([int(seed), i, h, d])))
    return out


def _label_in_parallel(n_workers, todo, consume, sessions, play_pool,
                       hero_plain, hero_rec, block_vectors, agent_member, ocfg,
                       R, cfg, driver, log, results=None):
    """The §3 layout: `n_workers` CPU processes, this process as the server.

    Everything about *what* a label is stays where it was — the workers call
    the same `action_values` over the same records with the same per-decision
    seeds, and `consume` is the same `consume`. What moves is the network: the
    workers hold weightless mirrors of the pool and this process runs every
    forward, batching across workers at a barrier. See `oracle/parallel.py`.
    """
    # Hero's slots are rebuilt inside each worker from the block vectors it
    # owns, so they are mirrored as holes rather than as members.
    mirror_pool = list(play_pool)
    for seats in list(hero_plain) + list(hero_rec):
        for idx in seats:
            mirror_pool[idx] = None

    prototype = agent_member(0, _slot_of_seat_at(sessions[0].num_players, 0),
                             block_vectors[0][0])
    runners, pool_spec, hero_spec = runner_table(
        mirror_pool, prototype, cfg["game"], _device_of(play_pool), log)

    procs, conns, slabs, result_q = spawn_workers(
        n_workers, [(pos, i, h, d) for pos, (i, h, d) in enumerate(todo)],
        sessions, block_vectors, pool_spec, hero_spec, hero_plain, hero_rec,
        runners, cfg["game"], ocfg, int(cfg["seed"]), R, log)
    server = ForwardServer(runners, slabs, conns, procs, log)
    try:
        done = collect(server, result_q, procs, todo, 0, consume, log,
                       results=results)
        assert done == len(todo), (
            f"the workers returned {done} of {len(todo)} labels")
    finally:
        join_workers(procs, log)


def _device_of(play_pool):
    """The device the pool's networks are on — where the server will run."""
    for member in play_pool:
        net = getattr(member, "net", None)
        if net is not None:
            return next(net.parameters()).device
        agent = getattr(member, "agent", None)
        if agent is not None:
            return agent.device_
    return "cpu"


def _label_sessions(driver, play_pool, sessions, hero_plain, hero_rec,
                    block_vectors, agent_member, ocfg, R, max_players, cfg,
                    out_dir, done, log):
    """One oracle label per hero decision, sharded to disk.

    `done` is the `progress.json` of an earlier call (`_resume_state`) or
    `None`. Its hands are not in `sessions` — they were not played — so this
    call labels the decisions it has and adds its counts to the ones it
    inherited.

    The bar counts **hero decisions across the whole job** (`CLAUDE.md` §5):
    that is what the phase costs, and a bar per session would answer a question
    nobody is asking. A resumed call starts the bar at what it is skipping, so
    the bar still reaches its total and its ETA still means what it says.
    Playing the hands, above, has its own bar because it happens first and in a
    different unit; nothing is nested.
    """
    os.makedirs(out_dir, exist_ok=True)
    hands_per_session = int(cfg["hands_per_session"])
    n_hands = len(sessions) * hands_per_session
    todo = _hero_decisions(sessions, hero_rec)
    log(f"[{TAG}] {len(todo)} hero decisions to label")

    per_shard = int(cfg["labels_per_shard"])
    shards = list(done["shards"]) if done else []
    decisions_done = int(done["decisions_done"]) if done else 0
    n_labels = int(done["n_labels"]) if done else 0
    dropped = int(done["n_dropped"]) if done else 0
    forwards = int(done["forwards"]) if done else 0
    rollouts = int(done["n_rollouts"]) if done else 0
    seconds = float(done["seconds"]) if done else 0.0
    collision_sum = float(done["collision_sum"]) if done else 0.0
    collision_n = int(done["collision_n"]) if done else 0
    buffer = []

    # Where each label sits in the job's hands, and which of them are the last
    # decision of theirs. A shard is flushed **at a hand boundary and never
    # inside one**: `progress.json` counts whole hands, because the hand is what
    # a later call re-plays, and a shard cut in the middle of one would leave
    # that hand's remaining decisions unlabelled with nothing on disk saying so.
    hand_of = [i * hands_per_session + h for i, h, _d in todo]
    ends_hand = [p + 1 == len(todo) or hand_of[p + 1] != hand_of[p]
                 for p in range(len(todo))]
    # The first hand that is *not* labelled once position `p` is written: the
    # next label's hand, and the whole job when there is no next label. Hands in
    # between hold no decision of hero's, so there is nothing there to label.
    next_hand = [hand_of[p + 1] if p + 1 < len(todo) else n_hands
                 for p in range(len(todo))]

    def flush(hands, decisions):
        """A shard, then the note that says the shard is safely on disk.

        In this order and never the other: a progress file naming a shard that
        was not written would make the next run skip hands nobody labelled.
        """
        if buffer:
            path = os.path.join(out_dir, f"shard_{len(shards):04d}.npz")
            _write_npz(path, _shard_arrays(buffer))
            shards.append(path)
            buffer.clear()
        with open(os.path.join(out_dir, PROGRESS), "w") as fh:
            json.dump({"n_hands": int(n_hands), "hands_done": int(hands),
                       "decisions_done": int(decisions),
                       "shards": shards, "n_labels": n_labels,
                       "n_dropped": dropped, "forwards": forwards,
                       "n_rollouts": rollouts, "seconds": seconds,
                       "collision_sum": collision_sum,
                       "collision_n": collision_n}, fh)

    # `initial` and not `bar.update(start)`: the skipped labels were done by an
    # earlier run and must not be counted as having taken this run's zero
    # seconds — see `utils.progress`.
    bar = progress(total=decisions_done + len(todo), desc="label",
                   unit="label", initial=decisions_done)

    buffered = None

    def consume(pos, i, h, d, q, legal, stats):
        """One finished label, whoever computed it.

        Both paths — the sequential loop below and the workers of
        `oracle/parallel.py` — end here, so a label is turned into a shard row
        by one piece of code and the two paths cannot drift apart in what they
        write.
        """
        nonlocal n_labels, dropped, forwards, seconds, rollouts
        nonlocal collision_sum, collision_n
        s = sessions[i]
        forwards += stats.forwards
        seconds += stats.seconds
        rollouts += stats.n_rollouts
        collision_sum += stats.collision_rate
        collision_n += 1
        bar.update(1)
        if buffered is not None:
            # Workers finish out of order and the parent releases labels only
            # in `todo` order, so the bar moves in bursts: it can sit still
            # while several hundred labels are already computed and waiting for
            # an earlier one. Saying how many are held is the difference
            # between "stalled" and "reordering", and the two look identical
            # otherwise.
            bar.set_postfix_str(f"{buffered()} held for ordering",
                                refresh=False)

        if np.isfinite(q[legal]).all():
            hero_seat = s.seat_of_slot(HERO_SLOT, h)
            tokens = play_pool[hero_rec[i][hero_seat]].seen[(s.idx, h, d)]
            _stack_bb, pot_bb, to_call_bb = (float(x) for x in
                                             tokens.scalars[-1])
            buffer.append({
                "tokens": tokens, "q": q, "legal": legal,
                "pot_bb": pot_bb, "facing_bet_bb": to_call_bb,
                "embeddings": block_vectors[i][h // R],
                "session": s.idx, "hand": h, "decision": d,
                "num_players": s.num_players, "stack_bb": s.stack_bb,
                "hero_seat": hero_seat,
            })
            n_labels += 1
        else:
            # Every joint draw collided, so there is no EV to build a target
            # from (`oracle/rollout.py`). Dropping it is the honest outcome; the
            # count is in the manifest because a large one is a broken run.
            dropped += 1
        if len(buffer) >= per_shard and ends_hand[pos]:
            flush(next_hand[pos], decisions_done + pos + 1)

    # Rollouts get averaged and corpus hands do not, so the variance-reduced
    # value belongs to the labelling driver and to nothing else. The parallel
    # path builds its own driver from the same config (`oracle/parallel.py`);
    # this path borrows the corpus driver, so it borrows it configured.
    saved_runout = driver.runout
    driver.runout = ocfg.runout_config()
    try:
        n_workers = worker_count(cfg)
        if n_workers and todo:
            held = {}
            buffered = held.__len__
            _label_in_parallel(n_workers, todo, consume, sessions,
                               play_pool, hero_plain, hero_rec, block_vectors,
                               agent_member, ocfg, R, cfg, driver, log,
                               results=held)
        else:
            current = (None, None)
            posterior_cache = PosteriorCache()
            pending = [(pos, i, h, d) for pos, (i, h, d) in enumerate(todo)]
            for chunk in label_chunks(pending, R, ocfg.labels_per_batch):
                i, h = chunk[0][1], chunk[0][2]
                block = h // R
                if (i, block) != current:
                    # Hero plays its own rollouts with the vectors it acted under;
                    # the member is rebuilt per block because the pool slot holds
                    # whichever block was played last.
                    current = (i, block)
                    _seat_hero(play_pool, (hero_plain[i],), sessions[i],
                               block_vectors[i], block, agent_member)
                answers = action_values_batch(
                    _label_requests(chunk, sessions, hero_plain, cfg["seed"]),
                    driver, play_pool, ocfg, posterior_cache=posterior_cache)
                for (pos, i, h, d), (q, legal, stats) in zip(chunk, answers):
                    consume(pos, i, h, d, q, legal, stats)
    finally:
        driver.runout = saved_runout
    flush(n_hands, decisions_done + len(todo))
    bar.close()

    stats = LabelStats(
        forwards=forwards, seconds=seconds,
        collision_rate=(collision_sum / collision_n) if collision_n else 0.0,
        n_rollouts=rollouts)
    log(f"[{TAG}] {n_labels} labels in {len(shards)} shards, {dropped} dropped, "
        f"{stats.forwards} forwards, {stats.seconds:.1f}s of labelling, "
        f"collision rate {100 * stats.collision_rate:.1f}%")
    return {"shards": shards, "n_labels": n_labels, "n_dropped": dropped,
            "n_sessions": len(sessions), "stats": stats}
