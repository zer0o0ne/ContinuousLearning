"""Shard-based lazy-loading dataset for memory-efficient training.

Scenarios are stored as numbered shard files (shard_000000.pt, ...).
Only one shard is in memory at a time via an LRU-1 cache.
ShardBatchSampler ensures batches come from the same shard so
the cache never thrashes.
"""

import bisect
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from agent.train_scenarios.generation.generate import _read_meta


# ---------------------------------------------------------------------------
# Shard container
# ---------------------------------------------------------------------------

class ShardedScenarios:
    """Lazy-loading container that serves scenarios from numbered shard files.

    Only one shard is in memory at a time (LRU cache, size 1).
    Use with ShardBatchSampler to avoid thrashing.
    """

    def __init__(self, save_dir, shard_subdir="dataset_shards"):
        self.save_dir = save_dir
        self._shard_dir = os.path.join(save_dir, shard_subdir)
        self._shards = []       # [(path, count), ...]
        self._offsets = []      # cumulative start offset per shard
        self._total = 0

        self._cache_path = None
        self._cache_data = None

        meta = _read_meta(save_dir)
        shard_paths = self._list_shard_paths()

        legacy_path = os.path.join(save_dir, "dataset.pt")
        storage = (meta or {}).get("storage", "monolithic")

        if storage != "sharded" and os.path.exists(legacy_path) and not shard_paths:
            self._shards = [(legacy_path, None)]
        else:
            shard_counts = (meta or {}).get("shard_counts")
            if shard_counts and len(shard_counts) == len(shard_paths):
                self._shards = list(zip(shard_paths, shard_counts))
            elif shard_paths:
                self._shards = self._count_by_loading(shard_paths)
            if storage != "sharded" and os.path.exists(legacy_path):
                self._shards.insert(0, (legacy_path, None))

        self._resolve_counts()

    def _list_shard_paths(self):
        import glob
        if not os.path.isdir(self._shard_dir):
            return []
        return sorted(glob.glob(os.path.join(self._shard_dir, "shard_*.pt")))

    @staticmethod
    def _count_by_loading(paths):
        result = []
        for p in paths:
            data = torch.load(p, weights_only=False)
            result.append((p, len(data)))
            del data
        return result

    def _resolve_counts(self):
        resolved = []
        for path, count in self._shards:
            if count is None:
                data = torch.load(path, weights_only=False)
                count = len(data)
                del data
            resolved.append((path, count))
        self._shards = resolved
        offset = 0
        self._offsets = []
        for _, count in self._shards:
            self._offsets.append(offset)
            offset += count
        self._total = offset

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        si = bisect.bisect_right(self._offsets, idx) - 1
        path = self._shards[si][0]
        local = idx - self._offsets[si]
        if self._cache_path != path:
            self._cache_data = torch.load(path, weights_only=False)
            self._cache_path = path
        return self._cache_data[local]

    @property
    def shard_info(self):
        return list(self._shards)

    @property
    def shard_sizes(self):
        return [c for _, c in self._shards]

    @property
    def shard_offsets(self):
        return list(self._offsets)

    @property
    def n_shards(self):
        return len(self._shards)

    def clear_cache(self):
        self._cache_data = None
        self._cache_path = None


# ---------------------------------------------------------------------------
# Metadata scanner
# ---------------------------------------------------------------------------

def scan_shard_metadata(shards):
    """Stream through shards extracting lightweight per-item metadata.

    Returns (hand_ids, n_events) — each a list[int] of length len(shards).
    Peak memory: one shard at a time.
    """
    hand_ids = []
    n_events = []
    gi = 0
    for path, _ in shards.shard_info:
        data = torch.load(path, weights_only=False)
        for s in data:
            hand_ids.append(s.get("hand_id", gi))
            n_events.append(s.get("n_events", len(s.get("events", []))))
            gi += 1
        del data
    return hand_ids, n_events


def scan_opponent_metadata(shards):
    """Stream through opponent shards extracting per-item + expansion metadata.

    Returns (hand_ids, n_events, expanded_indices) where expanded_indices is
    a list of (scenario_global_idx, hero_pos) tuples.
    """
    hand_ids = []
    n_events = []
    expanded = []
    gi = 0
    for path, _ in shards.shard_info:
        data = torch.load(path, weights_only=False)
        for s in data:
            hand_ids.append(s.get("hand_id", gi))
            n_events.append(len(s.get("events", [])))
            for hp in s["hero_positions"]:
                expanded.append((gi, hp))
            gi += 1
        del data
    return hand_ids, n_events, expanded


# ---------------------------------------------------------------------------
# Norm stats (streaming)
# ---------------------------------------------------------------------------

def compute_norm_stats_from_shards(shards, modifiers=None, mod_params=None):
    """Compute z-score normalization stats by streaming through shards.

    Peak memory: one shard at a time.
    ``modifiers`` / ``mod_params`` are applied per-scenario so the stats
    reflect the agent-specific modified EV targets.
    """
    ev_s, ev_sq = 0.0, 0.0
    pot_s, pot_sq = 0.0, 0.0
    stk_s, stk_sq = 0.0, 0.0
    bets_s, bets_sq = 0.0, 0.0
    bld_s, bld_sq = 0.0, 0.0
    n_ev = 0
    n_evt = 0
    n_bets = 0

    for path, _ in shards.shard_info:
        data = torch.load(path, weights_only=False)
        for sc in data:
            if modifiers:
                sc = apply_modifier_single(sc, modifiers, *mod_params)
            denom = max(sc.get("pot", 0) + sc.get("facing_bet", 0),
                        sc["events"][-1]["big_blind"])
            ev = sc["ev_target"] / denom
            ev_s += ev
            ev_sq += ev * ev
            n_ev += 1
            for event in sc["events"]:
                pot_s += event["pot"]
                pot_sq += event["pot"] ** 2
                stk_s += event["stack"]
                stk_sq += event["stack"] ** 2
                bld_s += event["big_blind"]
                bld_sq += event["big_blind"] ** 2
                n_evt += 1
                raw = event["bets"]
                if isinstance(raw, np.ndarray):
                    raw = raw.tolist()
                for b in raw:
                    fb = float(b)
                    bets_s += fb
                    bets_sq += fb * fb
                    n_bets += 1
        del data

    def _st(s, sq, n):
        if n == 0:
            return 0.0, 1.0
        m = s / n
        std = max(sq / n - m * m, 0.0) ** 0.5
        return m, (std if std > 1e-8 else 1.0)

    return {
        "ev_mean": _st(ev_s, ev_sq, n_ev)[0], "ev_std": _st(ev_s, ev_sq, n_ev)[1],
        "pot_mean": _st(pot_s, pot_sq, n_evt)[0], "pot_std": _st(pot_s, pot_sq, n_evt)[1],
        "stack_mean": _st(stk_s, stk_sq, n_evt)[0], "stack_std": _st(stk_s, stk_sq, n_evt)[1],
        "bets_mean": _st(bets_s, bets_sq, n_bets)[0], "bets_std": _st(bets_s, bets_sq, n_bets)[1],
        "blind_mean": _st(bld_s, bld_sq, n_evt)[0], "blind_std": _st(bld_s, bld_sq, n_evt)[1],
    }


def compute_opponent_norm_stats_from_shards(shards):
    """Compute norm stats for opponent data by streaming (events only)."""
    pot_s, pot_sq = 0.0, 0.0
    stk_s, stk_sq = 0.0, 0.0
    bets_s, bets_sq = 0.0, 0.0
    bld_s, bld_sq = 0.0, 0.0
    n_evt = 0
    n_bets = 0

    for path, _ in shards.shard_info:
        data = torch.load(path, weights_only=False)
        for sc in data:
            for event in sc["events"]:
                pot_s += event["pot"]
                pot_sq += event["pot"] ** 2
                n_evt += 1
                for pos_stacks in (event.get("stacks") or []):
                    stk_s += float(pos_stacks)
                    stk_sq += float(pos_stacks) ** 2
                bld_s += event["big_blind"]
                bld_sq += event["big_blind"] ** 2
                raw = event["bets"]
                if isinstance(raw, np.ndarray):
                    raw = raw.tolist()
                for b in raw:
                    fb = float(b)
                    bets_s += fb
                    bets_sq += fb * fb
                    n_bets += 1
        del data

    def _st(s, sq, n):
        if n == 0:
            return 0.0, 1.0
        m = s / n
        std = max(sq / n - m * m, 0.0) ** 0.5
        return m, (std if std > 1e-8 else 1.0)

    n_stk = n_evt  # one stack per event per position, but we summed per-position
    return {
        "pot_mean": _st(pot_s, pot_sq, n_evt)[0], "pot_std": _st(pot_s, pot_sq, n_evt)[1],
        "stack_mean": _st(stk_s, stk_sq, n_stk)[0], "stack_std": _st(stk_s, stk_sq, n_stk)[1],
        "bets_mean": _st(bets_s, bets_sq, n_bets)[0], "bets_std": _st(bets_s, bets_sq, n_bets)[1],
        "blind_mean": _st(bld_s, bld_sq, n_evt)[0], "blind_std": _st(bld_s, bld_sq, n_evt)[1],
    }


# ---------------------------------------------------------------------------
# Per-scenario modifier + normalize helpers
# ---------------------------------------------------------------------------

def apply_modifier_single(scenario, modifiers, n_actions, big_blind, temperature):
    """Apply modifiers to a single scenario, returning a modified shallow copy.

    Mirrors ``modifiers.apply_modifiers`` but for one item.
    """
    if not modifiers:
        return scenario

    s = {**scenario,
         "action_evs": list(scenario["action_evs"]),
         "action_probs": list(scenario["action_probs"])}

    temp = temperature
    bias_mods = []
    for mod in modifiers:
        if mod["type"] == "temperature":
            temp = mod["value"]
        else:
            bias_mods.append(mod)

    from agent.train_scenarios.modifiers import resolve_actions, _parse_condition, _check_condition
    parsed = []
    for mod in bias_mods:
        entry = {
            "type": mod["type"],
            "actions": resolve_actions(mod["actions"], n_actions),
            "factor": mod["factor"],
        }
        if mod["type"] == "conditional_bias":
            field, op, thresh = _parse_condition(mod["condition"])
            entry["cond"] = (field, op, thresh)
        parsed.append(entry)

    normalizer = max(s["pot"] + s["facing_bet"], big_blind) * temp
    evs = [float(e) for e in s["action_evs"]]
    total_factor = [0.0] * len(evs)

    for mod in parsed:
        if mod["type"] == "conditional_bias":
            field, op, thresh = mod["cond"]
            if not _check_condition(s, field, op, thresh):
                continue
        for idx in mod["actions"]:
            if idx < len(evs):
                total_factor[idx] += mod["factor"]

    for i in range(len(evs)):
        if total_factor[i] != 0.0:
            evs[i] = evs[i] + abs(evs[i]) * total_factor[i]

    evs_t = torch.tensor(evs, dtype=torch.float32)
    s["action_evs"] = evs
    s["ev_target"] = float(evs_t.max().item())
    s["action_probs"] = F.softmax(evs_t / normalizer, dim=0).tolist()
    return s


def normalize_single(scenario, norm_stats):
    """Shallow-copy and normalize a single scenario (events + ev_target)."""
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    pot_m, pot_s = norm_stats["pot_mean"], norm_stats["pot_std"]
    stk_m, stk_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bets_m, bets_s = norm_stats["bets_mean"], norm_stats["bets_std"]
    bld_m, bld_s = norm_stats["blind_mean"], norm_stats["blind_std"]

    s = {**scenario}
    denom = max(s.get("pot", 0) + s.get("facing_bet", 0),
                s["events"][-1]["big_blind"])
    s["ev_target"] = (s["ev_target"] / denom - ev_m) / ev_s

    events = []
    for e in s["events"]:
        ne = {**e}
        ne["pot"] = (ne["pot"] - pot_m) / pot_s
        ne["stack"] = (ne["stack"] - stk_m) / stk_s
        ne["big_blind"] = (ne["big_blind"] - bld_m) / bld_s
        ne["small_blind"] = (ne["small_blind"] - bld_m) / bld_s
        if isinstance(ne["bets"], np.ndarray):
            ne["bets"] = (ne["bets"] - bets_m) / bets_s
        else:
            ne["bets"] = [(b - bets_m) / bets_s for b in ne["bets"]]
        if "stacks" in ne:
            if isinstance(ne["stacks"], np.ndarray):
                ne["stacks"] = (ne["stacks"] - stk_m) / stk_s
            else:
                ne["stacks"] = [(c - stk_m) / stk_s for c in ne["stacks"]]
        events.append(ne)
    s["events"] = events
    return s


def normalize_action_evs_single(scenario, norm_stats):
    """Normalize action_evs for a single scenario (modelling phase)."""
    ev_m, ev_s = norm_stats["ev_mean"], norm_stats["ev_std"]
    s = {**scenario}
    denom = max(s.get("pot", 0) + s.get("facing_bet", 0),
                s["events"][-1]["big_blind"])
    evs = s["action_evs"]
    if isinstance(evs, np.ndarray):
        s["action_evs"] = (evs / denom - ev_m) / ev_s
    else:
        s["action_evs"] = [(ev / denom - ev_m) / ev_s for ev in evs]
    return s


# ---------------------------------------------------------------------------
# Sharded GTO Dataset
# ---------------------------------------------------------------------------

class ShardedGTODataset(Dataset):
    """Lazy-loading dataset for GTO training phases.

    Pulls scenarios from a ShardedScenarios container, optionally applying
    modifiers and normalization per-item.  The ``phase`` parameter selects
    which target fields to return:

    - ``"ev"``:        (events, ev_target_tensor)
    - ``"probs"``:     (events, action_probs_tensor)
    - ``"combined"``:  (events, ev_target_tensor, action_probs_tensor)
    - ``"modelling"``: (events, action_evs_tensor)
    """

    PHASE_FIELDS = {
        "ev", "probs", "combined", "modelling",
    }

    def __init__(self, shards, norm_stats, phase,
                 modifiers=None, mod_params=None, indices=None):
        if phase not in self.PHASE_FIELDS:
            raise ValueError(f"Unknown phase {phase!r}")
        self.shards = shards
        self.norm_stats = norm_stats
        self.phase = phase
        self.modifiers = modifiers
        self.mod_params = mod_params
        self.indices = indices  # subset (train / val)

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.shards)

    def __getitem__(self, idx):
        real = self.indices[idx] if self.indices is not None else idx
        s = self.shards[real]

        if self.modifiers:
            s = apply_modifier_single(s, self.modifiers, *self.mod_params)

        if self.phase == "modelling" and self.norm_stats:
            s = normalize_action_evs_single(s, self.norm_stats)

        if self.norm_stats:
            s = normalize_single(s, self.norm_stats)

        if self.phase == "ev":
            return s["events"], torch.tensor(s["ev_target"], dtype=torch.float32)
        elif self.phase == "probs":
            return s["events"], torch.tensor(s["action_probs"], dtype=torch.float32)
        elif self.phase == "combined":
            return (s["events"],
                    torch.tensor(s["ev_target"], dtype=torch.float32),
                    torch.tensor(s["action_probs"], dtype=torch.float32))
        else:  # modelling
            return s["events"], torch.tensor(s["action_evs"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sharded Opponent Action Dataset
# ---------------------------------------------------------------------------

class ShardedOpponentDataset(Dataset):
    """Lazy-loading dataset for opponent action training.

    Mirrors ``OpponentActionDataset`` but loads scenarios from shards.
    ``expanded_indices`` is a pre-built list of ``(scenario_global_idx, hero_pos)``
    produced by ``scan_opponent_metadata``.
    """

    def __init__(self, shards, expanded_indices, norm_stats=None, subset_indices=None):
        self.shards = shards
        self.expanded = expanded_indices
        self.norm_stats = norm_stats
        self.subset = subset_indices

    def __len__(self):
        return len(self.subset) if self.subset is not None else len(self.expanded)

    def __getitem__(self, idx):
        real = self.subset[idx] if self.subset is not None else idx
        s_idx, hero_pos = self.expanded[real]
        scenario = self.shards[s_idx]

        from agent.train_scenarios.opponent_action_predict.dataset import (
            OpponentActionDataset,
        )
        dummy = OpponentActionDataset.__new__(OpponentActionDataset)
        hero_hand = dummy._resolve_hero_hand(scenario["events"], hero_pos, seed=idx)
        events = dummy._to_standard(scenario["events"], hero_pos, hero_hand)
        if self.norm_stats is not None:
            from agent.train_scenarios.opponent_action_predict.dataset import (
                _normalize_events_inplace,
            )
            _normalize_events_inplace(events, self.norm_stats)
        target = dummy._observer_target(scenario, hero_hand)
        return events, target


# ---------------------------------------------------------------------------
# ShardBatchSampler
# ---------------------------------------------------------------------------

class ShardBatchSampler(Sampler):
    """Yields batches where all items come from the same shard.

    Shuffles shard order + items within each shard every epoch.
    Items are grouped by sequence length within each shard for padding
    efficiency when ``n_events`` is provided.
    """

    def __init__(self, shard_sizes, shard_offsets, batch_size,
                 indices=None, n_events=None, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        n_shards = len(shard_sizes)
        self._shard_groups = [[] for _ in range(n_shards)]

        all_indices = indices if indices is not None else list(range(sum(shard_sizes)))
        self._all_indices = all_indices
        for local_pos, global_idx in enumerate(all_indices):
            si = bisect.bisect_right(shard_offsets, global_idx) - 1
            self._shard_groups[si].append(local_pos)

        self._shard_groups = [g for g in self._shard_groups if g]
        self._n_events = n_events

        self._n_batches = 0
        for g in self._shard_groups:
            n = len(g) // batch_size
            if not drop_last and len(g) % batch_size:
                n += 1
            self._n_batches += n

    def __iter__(self):
        order = list(range(len(self._shard_groups)))
        random.shuffle(order)
        for si in order:
            items = list(self._shard_groups[si])
            random.shuffle(items)
            if self._n_events is not None:
                items.sort(key=lambda i: self._n_events[self._all_indices[i]])
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        return self._n_batches


# ---------------------------------------------------------------------------
# Shard-aware split
# ---------------------------------------------------------------------------

def shard_aware_split(hand_ids, val_split, seed=42):
    """Split global indices by hand_id.

    Returns (train_indices, val_indices) as lists of global indices.
    """
    from collections import defaultdict
    hand_to_indices = defaultdict(list)
    for idx, hid in enumerate(hand_ids):
        hand_to_indices[hid].append(idx)

    unique = list(hand_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_split))
    val_hands = set(unique[:n_val])

    train_idx = []
    val_idx = []
    for hid, idxs in hand_to_indices.items():
        if hid in val_hands:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx


def shard_aware_split_expanded(hand_ids, expanded_indices, val_split, seed=42):
    """Split expanded (opponent) indices by hand_id of the parent scenario."""
    from collections import defaultdict
    hand_to_scenarios = defaultdict(set)
    for s_idx, hid in enumerate(hand_ids):
        hand_to_scenarios[hid].add(s_idx)

    unique = list(hand_to_scenarios.keys())
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_split))
    val_hands = set(unique[:n_val])
    val_scenarios = set()
    for hid in val_hands:
        val_scenarios.update(hand_to_scenarios[hid])

    train_idx = []
    val_idx = []
    for exp_idx, (s_idx, _) in enumerate(expanded_indices):
        if s_idx in val_scenarios:
            val_idx.append(exp_idx)
        else:
            train_idx.append(exp_idx)
    return train_idx, val_idx
