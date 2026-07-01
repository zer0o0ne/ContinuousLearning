"""Per-agent head-distribution analytics.

For each agent in a multi-agent training run, loads the latest MCTS checkpoint
(`mcts_predict/<timestamp>/best.pt`), runs inference on a fixed stratified pick
of game situations from the raw GTO dataset (the one referenced by
`config.dataset.dataset_dir`), and reports adequacy + cross-agent divergence
of every head:

  * action_head            — fold/call/raise/all-in mass, entropy, KL vs GTO
  * opponent_action_head   — same bucketed view; expected to be similar across
                             agents (opponent head trained on shared data)
  * value_head             — output in z-space vs GTO-implied z-target derived
                             from `ev_target / max(pot+facing_bet, big_blind)`
                             and the agent's own `ev_mean`/`ev_std`
  * modelling_head         — mean pairwise cosine similarity of per-action
                             embeddings (low ≈ distinct, high ≈ collapsed)

Outputs:
  * head_distributions.pt  — raw inference for every (agent, situation)
  * head_distributions.txt — formatted tables identical to the printout

Usage:
    # Default: takes config.json next to versions/v5, walks every agent under
    # config.multi_agent.save_dir, writes the report under <save_dir>/analysis.
    python analytics/head_distributions.py

    # Custom run
    python analytics/head_distributions.py \\
        --config versions/v5/config.json \\
        --agents_dir data/v5/5_final_agents \\
        --n_per_bucket 3 --seed 42 --device cpu

Reproducibility:
  * Picks are deterministic for a given (dataset, seed, n_per_bucket).
  * Inference runs under torch.no_grad() and ASI.eval(); on CPU it is bitwise
    reproducible across runs of the same checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path bootstrap so this file works both as `python -m analytics.head_distributions`
# and as `python versions/v5/analytics/head_distributions.py`.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _VERSION_DIR not in sys.path:
    sys.path.insert(0, _VERSION_DIR)

from agent.agent import ASI  # noqa: E402
from agent.train_scenarios.generation.generate import load_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Stratified picking
# ---------------------------------------------------------------------------

# Buckets cover early/late streets and weak/strong equity. n_events is a proxy
# for street depth (preflop ~ 2-3 events, river ~ 14+).
_BUCKET_DEFS = [
    ("preflop_low_eq",   lambda s: s["n_events"] <= 3 and s["equity"] < 0.4),
    ("preflop_high_eq",  lambda s: s["n_events"] <= 3 and s["equity"] > 0.6),
    ("flop_low_eq",      lambda s: 4 <= s["n_events"] <= 7 and s["equity"] < 0.4),
    ("flop_high_eq",     lambda s: 4 <= s["n_events"] <= 7 and s["equity"] > 0.6),
    ("turn",             lambda s: 8 <= s["n_events"] <= 12),
    ("river",            lambda s: s["n_events"] >= 13),
]


def pick_situations(dataset: List[dict], n_per_bucket: int, seed: int):
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {name: [] for name, _ in _BUCKET_DEFS}
    for i, s in enumerate(dataset):
        for name, pred in _BUCKET_DEFS:
            if pred(s):
                buckets[name].append(i)
                break
    picks: List[int] = []
    bucket_of: Dict[int, str] = {}
    for name, _ in _BUCKET_DEFS:
        pool = buckets[name]
        if not pool:
            continue
        chosen = rng.sample(pool, min(n_per_bucket, len(pool)))
        for idx in chosen:
            picks.append(idx)
            bucket_of[idx] = name
    return picks, bucket_of


# ---------------------------------------------------------------------------
# Per-agent normalization (mirrors generation/generate._normalize_scenarios but
# operates on a fresh copy of the events so the raw dataset stays untouched).
# ---------------------------------------------------------------------------

def normalize_events(events, norm_stats):
    pot_m, pot_s = norm_stats["pot_mean"],   norm_stats["pot_std"]
    stk_m, stk_s = norm_stats["stack_mean"], norm_stats["stack_std"]
    bet_m, bet_s = norm_stats["bets_mean"],  norm_stats["bets_std"]
    bl_m,  bl_s  = norm_stats["blind_mean"], norm_stats["blind_std"]

    out = []
    for e in events:
        ne = dict(e)
        ne["pot"]         = (e["pot"] - pot_m) / pot_s
        ne["stack"]       = (e["stack"] - stk_m) / stk_s
        ne["big_blind"]   = (e["big_blind"] - bl_m) / bl_s
        ne["small_blind"] = (e["small_blind"] - bl_m) / bl_s
        raw = e["bets"]
        if isinstance(raw, np.ndarray):
            ne["bets"] = (raw - bet_m) / bet_s
        else:
            ne["bets"] = np.array(
                [(b - bet_m) / bet_s for b in raw], dtype=np.float64)
        if "stacks" in e:
            raw_stacks = e["stacks"]
            if isinstance(raw_stacks, np.ndarray):
                ne["stacks"] = (raw_stacks - stk_m) / stk_s
            else:
                ne["stacks"] = np.array(
                    [(s - stk_m) / stk_s for s in raw_stacks], dtype=np.float64)
        out.append(ne)
    return out


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def find_agents(agents_dir: str) -> List[str]:
    """Return list of agent subdirectories that have an mcts_predict/best.pt."""
    out = []
    for name in sorted(os.listdir(agents_dir)):
        sub = os.path.join(agents_dir, name)
        if not os.path.isdir(sub):
            continue
        mcts_dir = os.path.join(sub, "mcts_predict")
        if not os.path.isdir(mcts_dir):
            continue
        ts_dirs = sorted([d for d in os.listdir(mcts_dir)
                          if os.path.isdir(os.path.join(mcts_dir, d))],
                         reverse=True)
        for ts in ts_dirs:
            if os.path.exists(os.path.join(mcts_dir, ts, "best.pt")):
                out.append(name)
                break
    return out


def latest_mcts_ckpt(agents_dir: str, agent_name: str) -> str:
    mcts_dir = os.path.join(agents_dir, agent_name, "mcts_predict")
    ts_dirs = sorted([d for d in os.listdir(mcts_dir)
                      if os.path.isdir(os.path.join(mcts_dir, d))],
                     reverse=True)
    for ts in ts_dirs:
        p = os.path.join(mcts_dir, ts, "best.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no best.pt under {mcts_dir}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _silent_log(*a, **k):
    pass


def run_inference(config: dict, ckpt_path: str, samples: List[dict],
                  device: str, chunk_size: int) -> dict:
    """Load `ckpt_path` into a fresh ASI, run all four heads on `samples`."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    norm_stats = ckpt["norm_stats"]

    agent = ASI(_silent_log, config)
    agent.set_device(device)
    missing, unexpected = agent.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    agent.eval()

    evt_seqs = [normalize_events(s["events"], norm_stats) for s in samples]

    action_logits_all = []
    opp_logits_all    = []
    value_all         = []
    mod_emb_all       = []
    with torch.no_grad():
        for i in range(0, len(evt_seqs), chunk_size):
            chunk = evt_seqs[i:i+chunk_size]
            out = agent.forward_batch(
                chunk, skip_memory=True,
                heads={"action", "value", "opponent_action", "modelling"},
                skip_opponent_emb=True,
            )
            action_logits_all.append(out["action_logits"].detach().cpu())
            opp_logits_all.append(out["opponent_action_logits"].detach().cpu())
            value_all.append(out["value"].detach().cpu())
            mod_emb_all.append(out["action_embeddings"].detach().cpu())

    action_logits = torch.cat(action_logits_all, dim=0)   # (N, n_actions)
    opp_logits    = torch.cat(opp_logits_all,    dim=0)   # (N, n_actions)
    value         = torch.cat(value_all,         dim=0)   # (N,) or (N,1)
    mod_emb       = torch.cat(mod_emb_all,       dim=0)   # (N, n_actions, d)

    n_actions = action_logits.shape[-1]
    emb_norms = mod_emb.norm(dim=-1)                       # (N, n_actions)
    mod_emb_n = mod_emb / (mod_emb.norm(dim=-1, keepdim=True) + 1e-9)
    cos = torch.einsum("nad,nbd->nab", mod_emb_n, mod_emb_n)
    eye = torch.eye(n_actions).unsqueeze(0)
    cos_off = cos * (1 - eye)
    mean_cos = cos_off.sum(dim=(1, 2)) / (n_actions * (n_actions - 1))

    del agent, ckpt
    gc.collect()

    return {
        "ckpt_path": ckpt_path,
        "norm_stats": norm_stats,
        "action_probs":     F.softmax(action_logits, dim=-1),     # (N, n_a)
        "opp_action_probs": F.softmax(opp_logits,    dim=-1),     # (N, n_a)
        "value":            value.squeeze(-1)
                              if value.dim() == 2 else value,     # (N,)
        "mod_emb_norms":    emb_norms,                            # (N, n_a)
        "mod_emb_mean_cos": mean_cos,                             # (N,)
        "missing_keys":     missing,
        "unexpected_keys":  unexpected,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _bucketize_probs(p):
    """fold / call / raise (any sized) / all-in masses."""
    arr = np.asarray(p, dtype=np.float64)
    return float(arr[0]), float(arr[1]), float(arr[2:-1].sum()), float(arr[-1])


def _kl(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def _entropy(p):
    p = np.asarray(p, dtype=np.float64) + 1e-12
    return float(-(p * np.log(p)).sum())


def make_report(picks: List[int], bucket_of: Dict[int, str],
                samples: List[dict], results: Dict[str, dict],
                big_blind_fallback: float) -> str:
    agent_names = list(results.keys())
    n = len(samples)
    lines: List[str] = []
    P = lines.append

    P("=" * 100)
    P("Head-distribution analytics")
    P(f"Agents: {', '.join(agent_names)}")
    P(f"Situations: {n}  (n_per_bucket × {len(_BUCKET_DEFS)} buckets)")
    P("=" * 100)

    for a in agent_names:
        r = results[a]
        P(f"  {a:<24}  ckpt={r['ckpt_path']}")
        if r["missing_keys"]:
            P(f"    WARN: {len(r['missing_keys'])} missing weights, first="
              f"{r['missing_keys'][:3]}")
        if r["unexpected_keys"]:
            P(f"    WARN: {len(r['unexpected_keys'])} unexpected weights, first="
              f"{r['unexpected_keys'][:3]}")
    P("")

    # ----- 1. Action-head: per-situation bucketed mass + GTO baseline -----
    P("[1] Action-head: fold / call / raise / all-in mass — per situation")
    header = f'{"#":>3} {"bucket":<17} {"n_ev":>4} {"eq":>5} {"facing":>6} {"np":>2}  '
    header += f'{"GTO":<22}'
    for a in agent_names:
        header += f'{a[:16]:<22}'
    P(header)

    for i, idx in enumerate(picks):
        s = samples[i]
        gp = np.asarray(s["action_probs"])
        gf, gc, gr, ga = _bucketize_probs(gp)
        row = (f'{i:>3} {bucket_of[idx]:<17} {s["n_events"]:>4} '
               f'{s["equity"]:>5.2f} {s["facing_bet"]:>6.0f} {s["num_players"]:>2}  '
               f'f{gf:.2f} c{gc:.2f} r{gr:.2f} a{ga:.2f}  ')
        for a in agent_names:
            p = results[a]["action_probs"][i].numpy()
            f, c, r, al = _bucketize_probs(p)
            row += f'f{f:.2f} c{c:.2f} r{r:.2f} a{al:.2f}  '
        P(row)
    P("")

    # ----- 2. Action-head: aggregate mass + entropy + KL vs GTO ----------
    P("[2] Action-head: aggregate over all situations (mean per agent)")
    P(f'{"agent":<24}{"fold":>7}{"call":>7}{"raise":>7}{"allin":>7}{"H":>9}'
      f'{"KL(GTO||·) mean":>20}{"  median":>10}{"  max":>8}')
    gto_means = np.zeros(4)
    for i in range(n):
        gto_means += np.array(_bucketize_probs(samples[i]["action_probs"]))
    gto_means /= n
    for a in agent_names:
        probs = results[a]["action_probs"].numpy()  # (n, 14)
        masses = np.array([_bucketize_probs(probs[i]) for i in range(n)])
        ent = np.array([_entropy(probs[i]) for i in range(n)])
        kls = np.array([_kl(samples[i]["action_probs"], probs[i])
                        for i in range(n)])
        P(f'{a:<24}{masses[:,0].mean():>7.3f}{masses[:,1].mean():>7.3f}'
          f'{masses[:,2].mean():>7.3f}{masses[:,3].mean():>7.3f}'
          f'{ent.mean():>9.3f}'
          f'{kls.mean():>20.3f}{np.median(kls):>10.3f}{kls.max():>8.2f}')
    P(f'{"GTO target":<24}{gto_means[0]:>7.3f}{gto_means[1]:>7.3f}'
      f'{gto_means[2]:>7.3f}{gto_means[3]:>7.3f}')
    P("")

    # ----- 3. Cross-agent symmetric KL on action head --------------------
    P("[3] Cross-agent symmetric KL on action-head (mean over situations)")
    P(f'{"":<24}' + ''.join(f'{a[:14]:>16}' for a in agent_names))
    for a1 in agent_names:
        row = f'{a1:<24}'
        for a2 in agent_names:
            vals = []
            for i in range(n):
                p1 = results[a1]["action_probs"][i].numpy()
                p2 = results[a2]["action_probs"][i].numpy()
                vals.append(0.5 * (_kl(p1, p2) + _kl(p2, p1)))
            row += f'{np.mean(vals):>16.3f}'
        P(row)
    P("")

    # ----- 4. Value-head: output vs GTO-implied z-target ----------------
    P("[4] Value-head: agent z-output vs GTO-implied z-target "
      "(delta = agent - target)")
    header = f'{"#":>3} {"eq":>5} {"facing":>6} {"gto_ev":>10} {"gto_ratio":>10}   '
    for a in agent_names:
        header += f'{a[:16]:>18}'
    P(header)
    for i, idx in enumerate(picks):
        s = samples[i]
        denom = max(s["pot"] + s["facing_bet"], big_blind_fallback)
        gto_ratio = s["ev_target"] / denom
        row = (f'{i:>3} {s["equity"]:>5.2f} {s["facing_bet"]:>6.0f} '
               f'{s["ev_target"]:>10.2f} {gto_ratio:>10.3f}   ')
        for a in agent_names:
            v = float(results[a]["value"][i])
            em = results[a]["norm_stats"]["ev_mean"]
            es = results[a]["norm_stats"]["ev_std"]
            tgt_z = (gto_ratio - em) / es
            row += f' v={v:>+5.2f} d={v-tgt_z:>+5.2f}'
        P(row)
    P("")

    # ----- 5. Opponent-action-head: aggregate mass + entropy ------------
    P("[5] Opponent-action-head: aggregate mass (mean per agent)")
    P(f'{"agent":<24}{"fold":>7}{"call":>7}{"raise":>7}{"allin":>7}{"H":>9}')
    for a in agent_names:
        probs = results[a]["opp_action_probs"].numpy()
        masses = np.array([_bucketize_probs(probs[i]) for i in range(n)])
        ent = np.array([_entropy(probs[i]) for i in range(n)])
        P(f'{a:<24}{masses[:,0].mean():>7.3f}{masses[:,1].mean():>7.3f}'
          f'{masses[:,2].mean():>7.3f}{masses[:,3].mean():>7.3f}'
          f'{ent.mean():>9.3f}')
    P("")

    # ----- 6. Modelling-head: cosine similarity between action embeddings -
    P("[6] Modelling-head: mean off-diagonal cosine similarity between "
      "per-action embeddings")
    P("    (low = distinct action representations; high = collapsed)")
    P(f'{"agent":<24}{"mean":>8}{"std":>8}{"min":>8}{"max":>8}')
    for a in agent_names:
        cos = results[a]["mod_emb_mean_cos"].numpy()
        P(f'{a:<24}{cos.mean():>8.3f}{cos.std():>8.3f}'
          f'{cos.min():>8.3f}{cos.max():>8.3f}')
    P("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _project_root():
    return os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def _resolve_data_path(config, rel_path):
    """Resolve a relative path against data/<version>/ like the pipeline does."""
    if rel_path and os.path.isabs(rel_path):
        return rel_path
    version = os.path.basename(_VERSION_DIR)
    project_root = _project_root()
    name = config.get("name", "experiment")
    base = os.path.join(project_root, "data", version)
    if rel_path:
        return os.path.join(base, rel_path)
    return os.path.join(base, name)


def analyze(config_path: str, agents_dir: Optional[str], dataset_path: Optional[str],
            out_dir: Optional[str], n_per_bucket: int, seed: int,
            device: Optional[str], chunk_size: int, agents_filter: Optional[List[str]]):
    config = json.load(open(config_path))

    if agents_dir is None:
        save_dir_cfg = config.get("multi_agent", {}).get("save_dir")
        if not save_dir_cfg:
            raise ValueError("agents_dir not given and config has no "
                             "multi_agent.save_dir")
        agents_dir = _resolve_data_path(config, save_dir_cfg)
    agents_dir = os.path.abspath(agents_dir)

    device = _detect_device(device)

    if dataset_path is not None:
        if os.path.isdir(dataset_path):
            print(f"Loading dataset from directory {dataset_path} ...", flush=True)
            dataset = load_dataset(dataset_path, log=print)
        else:
            print(f"Loading dataset from {dataset_path} ...", flush=True)
            dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
    else:
        dataset_dir = config["dataset"].get("dataset_dir", "")
        if dataset_dir:
            dataset_dir = _resolve_data_path(config, dataset_dir) if not os.path.isabs(dataset_dir) else dataset_dir
        else:
            dataset_dir = os.path.join(_resolve_data_path(config, ""), "dataset")
        print(f"Loading dataset from {dataset_dir} ...", flush=True)
        dataset = load_dataset(dataset_dir, log=print)

    if dataset is None:
        raise FileNotFoundError(
            f"No dataset found. Tried shards and legacy dataset.pt. "
            f"Pass --dataset <path> explicitly.")
    print(f"  {len(dataset)} samples loaded", flush=True)

    picks, bucket_of = pick_situations(dataset, n_per_bucket, seed)
    samples = [dataset[i] for i in picks]
    print(f"Picked {len(samples)} situations: "
          f"{ {b: sum(1 for i in picks if bucket_of[i] == b) for b, _ in _BUCKET_DEFS} }",
          flush=True)

    big_blind = float(config.get("game", {}).get("big_blind", 10.0))

    all_agents = find_agents(agents_dir)
    if agents_filter:
        all_agents = [a for a in all_agents if a in set(agents_filter)]
    if not all_agents:
        raise RuntimeError(f"no agents with mcts_predict/best.pt found under "
                           f"{agents_dir}")
    print(f"Agents under analysis ({len(all_agents)}): "
          f"{', '.join(all_agents)}")
    print(f"Device: {device}")
    print()

    results: Dict[str, dict] = {}
    for name in all_agents:
        ckpt = latest_mcts_ckpt(agents_dir, name)
        print(f"  [{name}] -> {ckpt}", flush=True)
        t0 = time.time()
        results[name] = run_inference(config, ckpt, samples, device, chunk_size)
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    report_text = make_report(picks, bucket_of, samples, results, big_blind)
    print()
    print(report_text)

    if out_dir is None:
        ts = time.strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = os.path.join(agents_dir, "analysis", "head_distributions", ts)
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "head_distributions.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)

    pt_path = os.path.join(out_dir, "head_distributions.pt")
    serializable = {
        "config_path": config_path,
        "agents_dir": agents_dir,
        "dataset_path": dataset_path,
        "n_per_bucket": n_per_bucket,
        "seed": seed,
        "device": device,
        "picks": picks,
        "bucket_of": bucket_of,
        "big_blind": big_blind,
        "gto": [
            {
                "idx": picks[i],
                "bucket": bucket_of[picks[i]],
                "gto_probs":    list(samples[i]["action_probs"]),
                "gto_evs":      list(samples[i]["action_evs"]),
                "ev_target":    float(samples[i]["ev_target"]),
                "equity":       float(samples[i]["equity"]),
                "pot":          float(samples[i]["pot"]),
                "facing_bet":   float(samples[i]["facing_bet"]),
                "stack":        float(samples[i]["stack"]),
                "num_players":  int(samples[i]["num_players"]),
                "n_events":     int(samples[i]["n_events"]),
            }
            for i in range(len(samples))
        ],
        "agents": {
            name: {
                "ckpt_path":         r["ckpt_path"],
                "norm_stats":        r["norm_stats"],
                "action_probs":      r["action_probs"],
                "opp_action_probs":  r["opp_action_probs"],
                "value":             r["value"],
                "mod_emb_norms":     r["mod_emb_norms"],
                "mod_emb_mean_cos":  r["mod_emb_mean_cos"],
            }
            for name, r in results.items()
        },
    }
    torch.save(serializable, pt_path)

    print(f"Saved: {txt_path}")
    print(f"Saved: {pt_path}")
    return txt_path, pt_path


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_cfg = os.path.join(_VERSION_DIR, "config.json")
    p.add_argument("--config", default=default_cfg,
                   help=f"Path to config.json (default: {default_cfg})")
    p.add_argument("--agents_dir", default=None,
                   help="Directory containing per-agent folders. Defaults to "
                        "config.multi_agent.save_dir.")
    p.add_argument("--dataset", default=None,
                   help="Path to dataset directory (with shards) or legacy "
                        "dataset.pt file. Defaults to auto-resolve from config.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write outputs (default: "
                        "<agents_dir>/analysis/head_distributions/<ts>).")
    p.add_argument("--n_per_bucket", type=int, default=3,
                   help="Situations per stratification bucket (default: 3 → 18 total).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps. Default: auto-detect.")
    p.add_argument("--chunk_size", type=int, default=6,
                   help="Batch size for inference forward passes (default: 6).")
    p.add_argument("--agents", nargs="*", default=None,
                   help="Restrict to these agent names (default: all found).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        config_path=args.config,
        agents_dir=args.agents_dir,
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        n_per_bucket=args.n_per_bucket,
        seed=args.seed,
        device=args.device,
        chunk_size=args.chunk_size,
        agents_filter=args.agents,
    )
