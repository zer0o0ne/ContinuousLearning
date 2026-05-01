"""Profile and validate MCTS implementations.

Run from versions/v5/ with the venv active:

    source ../../venv/bin/activate
    cd versions/v5

    # Time breakdown (InstrumentedMCTS):
    python mcts_profile.py --mode profile

    # Capture reference trace (action + root visit distribution per decision):
    python mcts_profile.py --mode baseline

    # Verify current MCTS matches the saved baseline:
    python mcts_profile.py --mode verify

The baseline trace is saved under data/v5/ (kept out of git).
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from agent.agent import ASI
from agent.mcts.mcts import MCTS
from agent.mcts.game_state import GameState
from agent.train_scenarios.generation.generate import _get_raise_sizes
from env.table import Table
from evaluation.evaluate import _rebuild_events, _normalize_events_inplace
from utils import Logger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "v5", "mcts_baseline.pt")


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

class Timings:
    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, key, dt):
        self.total[key] += dt
        self.count[key] += 1

    def report(self):
        return [(k, self.total[k], self.count[k],
                 self.total[k] / max(self.count[k], 1))
                for k in sorted(self.total, key=lambda x: -self.total[x])]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Instrumented MCTS — mirrors current production _simulate
# ---------------------------------------------------------------------------

class InstrumentedMCTS(MCTS):
    def __init__(self, *args, timings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = timings or Timings()

    def _evaluate_root(self, event_sequences):
        t0 = time.perf_counter()
        p_out, encoded, mask = self.agent.perception.forward_batch(
            event_sequences, device=self.device, skip_memory=True,
        )
        sync(self.device)
        self.timings.add("root_perception", time.perf_counter() - t0)

        for name, head in (("root_value_head", self.agent.value_head),
                           ("root_action_head", self.agent.action_head),
                           ("root_opp_head", self.agent.opponent_action_head),
                           ("root_modelling_head", self.agent.modelling_head)):
            t0 = time.perf_counter()
            out = head(p_out, mask=mask)
            sync(self.device)
            self.timings.add(name, time.perf_counter() - t0)
            if name == "root_value_head":
                value = out
            elif name == "root_action_head":
                act_logits = out
            elif name == "root_opp_head":
                opp_logits = out
            else:
                act_embs = out

        return p_out, mask, value, act_logits, opp_logits, act_embs

    def _simulate(self, root, root_ctx, root_mask, root_gs):
        # SELECT
        t0 = time.perf_counter()
        node = root
        path = [node]
        while node.children and not node.is_terminal:
            action = self._select_child(node)
            node = node.children[action]
            path.append(node)
        self.timings.add("select", time.perf_counter() - t0)

        if node.is_terminal:
            for n in path:
                n.N += 1
            self.timings.add("terminal_known", 0.0)
            return

        t0 = time.perf_counter()
        gs = self._replay_game_state(root_gs, path)
        if node is not root:
            node.is_terminal = gs.is_terminal
            node.is_hero = gs.is_hero_turn()
        self.timings.add("replay_game_state", time.perf_counter() - t0)

        if node.is_terminal:
            for n in path:
                n.N += 1
            self.timings.add("terminal_first", 0.0)
            return

        t0 = time.perf_counter()
        context, mask = self._build_context(root_ctx, root_mask, path)
        sync(self.device)
        self.timings.add("build_context", time.perf_counter() - t0)

        if node is root:
            t0 = time.perf_counter()
            leaf_value = self.agent.value_head(context, mask=mask).item()
            sync(self.device)
            self.timings.add("leaf_value_only", time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            leaf_value = self.agent.value_head(context, mask=mask).item()
            sync(self.device)
            self.timings.add("leaf_value_head", time.perf_counter() - t0)

            t0 = time.perf_counter()
            act_logits = self.agent.action_head(context, mask=mask)
            sync(self.device)
            self.timings.add("leaf_action_head", time.perf_counter() - t0)

            t0 = time.perf_counter()
            opp_logits = self.agent.opponent_action_head(context, mask=mask)
            sync(self.device)
            self.timings.add("leaf_opp_head", time.perf_counter() - t0)

            t0 = time.perf_counter()
            act_embs = self.agent.modelling_head(context, mask=mask)
            sync(self.device)
            self.timings.add("leaf_modelling_head", time.perf_counter() - t0)

            t0 = time.perf_counter()
            self._expand_node(node, gs, act_logits, opp_logits, act_embs)
            self.timings.add("expand", time.perf_counter() - t0)

        t0 = time.perf_counter()
        for n in path:
            n.N += 1
            n.W += leaf_value
            n.Q = n.W / n.N
        self.timings.add("backup", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Hand runner
# ---------------------------------------------------------------------------

def setup(seed=42, batch_size=None, virtual_loss=None):
    """Load config, agent, return everything needed for run_hand."""
    config_path = "config.json"
    with open(config_path) as f:
        config = json.load(f)

    mcts_cfg = config.setdefault("mcts", {})
    if batch_size is not None:
        mcts_cfg["batch_size"] = batch_size
    if virtual_loss is not None:
        mcts_cfg["virtual_loss"] = virtual_loss

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    agent_base = os.path.join(PROJECT_ROOT, "data", "v5",
                              "5_final_agents", "gto_pure")
    log_dir = os.path.join(PROJECT_ROOT, "data", "v5",
                           "first_big_train", "profile_logs")
    os.makedirs(log_dir, exist_ok=True)
    log = Logger(log_dir)

    agent = ASI(log, config)
    agent.set_device(device)
    agent.load_checkpoint(agent_base)
    agent.eval()

    ckpt_path = ASI._find_best_checkpoint(agent_base)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    norm_stats = ckpt.get("norm_stats") or {
        "pot_mean": 0, "pot_std": 1, "stack_mean": 0, "stack_std": 1,
        "bets_mean": 0, "bets_std": 1, "blind_mean": 0, "blind_std": 1,
    }
    return config, device, agent, norm_stats


def run_hand(mcts_factory, config, device, agent, norm_stats, seed=42):
    """Play one hand with a given MCTS factory and capture a trace per decision.

    Returns:
        trace: list of dicts — one per decision — with
               action_idx, visit_dist (sorted list of (a, N)), root_Q, leaf_count
        decision_times: list of wall times per decision
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game_cfg = config.get("game", {})
    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3
    big_blind = game_cfg.get("big_blind", 10)
    small_blind = big_blind // 2

    num_players = 4
    table = Table(
        num_players=num_players, raise_sizes=raise_sizes,
        start_credits=1000, big_blind=big_blind, small_blind=small_blind,
    )
    table.start_table()

    snapshots = [{
        "pot": table.pot, "bets": np.copy(table.bets),
        "credits": list(table.credits), "turn": table.turn,
        "active_pos": table.active_player, "action": None,
    }]

    dummy_action = torch.zeros(n_actions, dtype=torch.float32)
    trace = []
    decision_times = []
    hand_done = False

    while not hand_done and len(decision_times) < 50:
        while table.several_all_in:
            end, _, _, _ = table.step(dummy_action)
            if end:
                hand_done = True
                break
        if hand_done:
            break

        active_pos = table.active_player
        if table.players_state[active_pos] != 1:
            break

        snapshots.append({
            "pot": table.pot, "bets": np.copy(table.bets),
            "credits": list(table.credits), "turn": table.turn,
            "active_pos": active_pos, "action": None,
        })

        events = _rebuild_events(
            snapshots, table.deck, active_pos,
            num_players, big_blind, small_blind, n_actions,
            up_to=len(snapshots) - 1,
        )
        norm_events = [dict(e) for e in events]
        for e in norm_events:
            if isinstance(e["bets"], np.ndarray):
                e["bets"] = np.copy(e["bets"])
        _normalize_events_inplace(norm_events, norm_stats)

        gs = GameState.from_table(table, active_pos)
        mcts = mcts_factory(agent, device, config.get("mcts", {}))

        dt0 = time.perf_counter()
        action_idx = mcts.search([norm_events], gs)
        sync(device)
        decision_times.append(time.perf_counter() - dt0)

        root = mcts.last_root
        visit_dist = sorted((a, c.N) for a, c in root.children.items())
        trace.append({
            "action_idx": action_idx,
            "visit_dist": visit_dist,
            "root_Q": float(root.Q),
            "n_children": len(root.children),
        })

        action_vec = torch.zeros(n_actions, dtype=torch.float32)
        action_vec[action_idx] = 1.0
        end, _, _, _ = table.step(action_vec)

        snapshots.append({
            "pot": table.pot, "bets": np.copy(table.bets),
            "credits": list(table.credits), "turn": table.turn,
            "active_pos": table.active_player, "action": action_vec,
        })

        if end:
            hand_done = True

    return trace, decision_times


# ---------------------------------------------------------------------------
# Trace comparison
# ---------------------------------------------------------------------------

def compare_traces(ref, cur, q_tol=1e-5):
    """Return list of human-readable diffs (empty list = identical)."""
    diffs = []
    if len(ref) != len(cur):
        diffs.append(f"decision count: ref={len(ref)} cur={len(cur)}")
        return diffs
    for i, (r, c) in enumerate(zip(ref, cur)):
        if r["action_idx"] != c["action_idx"]:
            diffs.append(f"decision {i}: action {r['action_idx']} vs {c['action_idx']}")
        if r["visit_dist"] != c["visit_dist"]:
            diffs.append(f"decision {i}: visit_dist mismatch\n"
                         f"  ref: {r['visit_dist']}\n"
                         f"  cur: {c['visit_dist']}")
        if abs(r["root_Q"] - c["root_Q"]) > q_tol:
            diffs.append(f"decision {i}: root_Q {r['root_Q']:.6f} vs "
                         f"{c['root_Q']:.6f} (|Δ|={abs(r['root_Q']-c['root_Q']):.2e})")
    return diffs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_profile(args):
    config, device, agent, norm_stats = setup(args.seed, args.batch_size, args.virtual_loss)
    mcts_cfg = config.get("mcts", {})
    print(f"Device: {device}, n_simulations={mcts_cfg.get('n_simulations')}, "
          f"batch_size={mcts_cfg.get('batch_size', 1)}, "
          f"virtual_loss={mcts_cfg.get('virtual_loss', 1.0)}")

    factory = lambda agent, device, cfg: MCTS(agent, device, cfg)

    t0 = time.perf_counter()
    trace, decision_times = run_hand(factory, config, device, agent, norm_stats, args.seed)
    hand_total = time.perf_counter() - t0

    print()
    print("=" * 78)
    print(f"HAND TOTAL: {hand_total:.2f}s, {len(trace)} decisions "
          f"(avg {hand_total/max(len(trace),1):.2f}s/decision)")
    print("=" * 78)
    print()
    print("Per-decision wall times:")
    for i, dt in enumerate(decision_times):
        print(f"  decision {i}: {dt:.3f}s  → action {trace[i]['action_idx']}  "
              f"Q={trace[i]['root_Q']:+.4f}")


def cmd_baseline(args):
    config, device, agent, norm_stats = setup(args.seed, args.batch_size, args.virtual_loss)
    print(f"Device: {device}, capturing baseline trace...")

    factory = lambda agent, device, cfg: MCTS(agent, device, cfg)
    trace, decision_times = run_hand(factory, config, device, agent, norm_stats, args.seed)

    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    torch.save({"trace": trace, "decision_times": decision_times,
                "seed": args.seed}, BASELINE_PATH)
    print(f"Saved {len(trace)} decisions → {BASELINE_PATH}")
    for i, d in enumerate(trace):
        print(f"  decision {i}: action={d['action_idx']:>3}  "
              f"Q={d['root_Q']:+.4f}  "
              f"top_visits={sorted(d['visit_dist'], key=lambda x: -x[1])[:3]}")


def cmd_verify(args):
    if not os.path.exists(BASELINE_PATH):
        print(f"No baseline at {BASELINE_PATH} — run --mode baseline first.")
        return 1

    ref = torch.load(BASELINE_PATH, weights_only=False)
    config, device, agent, norm_stats = setup(args.seed, args.batch_size, args.virtual_loss)
    print(f"Device: {device}, verifying against {BASELINE_PATH}")

    factory = lambda agent, device, cfg: MCTS(agent, device, cfg)
    trace, decision_times = run_hand(factory, config, device, agent, norm_stats, args.seed)

    diffs = compare_traces(ref["trace"], trace, q_tol=args.q_tol)
    if not diffs:
        print(f"OK: {len(trace)} decisions match baseline bit-for-bit "
              f"(q_tol={args.q_tol})")
        return 0
    print(f"MISMATCH ({len(diffs)} diff(s)):")
    for d in diffs:
        print(f"  {d}")
    return 2


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "baseline", "verify"],
                        default="profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-tol", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override mcts.batch_size for this run.")
    parser.add_argument("--virtual-loss", type=float, default=None,
                        help="Override mcts.virtual_loss for this run.")
    args = parser.parse_args()

    if args.mode == "profile":
        return cmd_profile(args)
    if args.mode == "baseline":
        return cmd_baseline(args)
    if args.mode == "verify":
        return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
