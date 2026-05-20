"""
Slumbot HU NLHE evaluation module.

Plays a list of trained agents heads-up against the public Slumbot HTTP API
(slumbot.com/slumbot/api). Reports raw and baseline-corrected BB/100.

Per-agent options (config.slumbot_eval.agents[*]):
  - path: directory containing a checkpoint (file or scenario subdirs)
  - use_opponent_embedding: bool — feed events through OpponentEmbeddingTable
  - use_mcts: bool — pick actions via MCTS.search instead of action-head sampling.
                     If use_opponent_embedding is also true, the opp_emb_table
                     is injected at the root perception call (inner-tree nodes
                     don't re-run perception).
  - action_temperature: float (optional) — overrides checkpoint temperature

Standalone:
    python -m evaluation.slumbot_eval --config config.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import requests
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from agent.agent import ASI
from agent.mcts.game_state import GameState
from agent.mcts.mcts import MCTS
from agent.perception.opponent_embeddings import OpponentEmbeddingTable
from agent.train_scenarios.generation.generate import _get_raise_sizes
from evaluation.evaluate import (
    _find_best_checkpoint,
    _normalize_events_inplace,
    _resolve_checkpoint_path,
)
from utils import get_amp_config


# Slumbot fixed parameters (from sample_api.py)
SLUMBOT_HOST = "slumbot.com"
SLUMBOT_NUM_STREETS = 4
SLUMBOT_SMALL_BLIND = 50
SLUMBOT_BIG_BLIND = 100
SLUMBOT_STACK_SIZE = 20000

_RANK_TO_IDX = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT_TO_IDX = {"c": 0, "d": 1, "h": 2, "s": 3}


# ============================================================================
# Card translation
# ============================================================================

def _card_to_int(card_str):
    """Slumbot card "Ac"/"Td"/etc → int 0..51 with rank*4 + suit encoding."""
    if len(card_str) != 2:
        raise ValueError(f"Invalid card '{card_str}'")
    rank, suit = card_str[0], card_str[1].lower()
    if rank not in _RANK_TO_IDX:
        raise ValueError(f"Invalid rank in '{card_str}'")
    if suit not in _SUIT_TO_IDX:
        raise ValueError(f"Invalid suit in '{card_str}'")
    return _RANK_TO_IDX[rank] * 4 + _SUIT_TO_IDX[suit]


def _board_to_ints(board_strs):
    """5-card board (-1 padded). board_strs may be empty / 3 / 4 / 5 long."""
    ints = [_card_to_int(c) for c in board_strs]
    while len(ints) < 5:
        ints.append(-1)
    return ints


# ============================================================================
# Action <-> discrete index translation
# ============================================================================

def _token_to_action_idx(state_pre, token, raise_sizes_for_street, n_raise_bins):
    """Encode a Slumbot action token into our discrete action index.

    state_pre: dict with 'bets', 'credits', 'pot', 'high_bet', 'active_pos'
               in Slumbot chip units (Slumbot frame: pos 0=BB, pos 1=SB).
    """
    if token == "f":
        return 0
    if token == "k" or token == "c":
        return 1
    assert token.startswith("b"), f"unknown token {token!r}"
    new_total = int(token[1:])
    pos = state_pre["active_pos"]
    bets_pos = state_pre["bets"][pos]
    credits_pos = state_pre["credits"][pos]
    added = new_total - bets_pos
    if added >= credits_pos - 1e-9:
        return n_raise_bins + 2  # all-in
    call_amount = state_pre["high_bet"] - bets_pos
    effective_pot = state_pre["pot"] - bets_pos
    raise_pct = max(0.0, (added - call_amount) / max(effective_pot, 1.0))
    diffs = [abs(raise_pct - rs) for rs in raise_sizes_for_street]
    return 2 + int(np.argmin(diffs))


def _action_idx_to_incr(state_pre, action_idx, raise_sizes_for_street,
                        n_raise_bins, hero_slumbot_pos, clamp_counters):
    """Translate our discrete action_idx into a legal Slumbot 'incr' string.

    Applies fallback clamping ("аккуратно"):
      - fold without facing a bet → check
      - raise below min legal → bump to min legal
      - raise above stack → cap at all-in
      - raise that doesn't exceed call → emit call/check
    Each clamp increments the corresponding counter.
    """
    bets = state_pre["bets"]
    credits = state_pre["credits"]
    high_bet = state_pre["high_bet"]
    last_bet_size = state_pre["last_bet_size"]
    bets_hero = bets[hero_slumbot_pos]
    cap = bets_hero + credits[hero_slumbot_pos]
    facing_bet = high_bet > bets_hero + 1e-9

    if action_idx == 0:
        if facing_bet:
            return "f"
        clamp_counters["fold_to_check"] += 1
        return "k"

    if action_idx == 1:
        return "c" if facing_bet else "k"

    if action_idx == n_raise_bins + 2:
        return f"b{int(round(cap))}"

    raise_pct = raise_sizes_for_street[action_idx - 2]
    call_amount = high_bet - bets_hero
    effective_pot = state_pre["pot"] - bets_hero
    added = round(call_amount + raise_pct * effective_pot)
    new_total = bets_hero + added

    # Slumbot min raise rule (sample_api.py:188-204)
    min_legal_total = high_bet + max(SLUMBOT_BIG_BLIND, last_bet_size)
    if min_legal_total > cap:
        min_legal_total = cap

    # If the computed "raise" is actually a call/check (added <= call_amount),
    # gracefully degrade to 'c'/'k' rather than emit an illegal bet.
    if added <= call_amount + 1e-9:
        clamp_counters["raise_to_call"] += 1
        return "c" if facing_bet else "k"

    if new_total < min_legal_total:
        clamp_counters["raise_bumped"] += 1
        new_total = min_legal_total
    if new_total > cap:
        clamp_counters["raise_to_allin"] += 1
        new_total = cap

    return f"b{int(round(new_total))}"


# ============================================================================
# Action-string replay
# ============================================================================

def _initial_state(hole_cards_int, board_ints):
    """Initial Slumbot-frame state before any action (blinds posted).

    Slumbot frame: pos 0 = BB, pos 1 = SB. SB acts first preflop.
    """
    bets = [SLUMBOT_BIG_BLIND, SLUMBOT_SMALL_BLIND]
    credits = [SLUMBOT_STACK_SIZE - SLUMBOT_BIG_BLIND,
               SLUMBOT_STACK_SIZE - SLUMBOT_SMALL_BLIND]
    return {
        "pot": SLUMBOT_BIG_BLIND + SLUMBOT_SMALL_BLIND,
        "bets": bets,
        "credits": credits,
        "high_bet": SLUMBOT_BIG_BLIND,
        "last_bet_size": SLUMBOT_BIG_BLIND - SLUMBOT_SMALL_BLIND,
        "turn": 0,
        "active_pos": 1,            # Slumbot frame — SB first preflop
        "players_state": [1, 1],    # both active and "moving"
        "is_terminal": False,
        "hole_cards": hole_cards_int,
        "board": board_ints,
    }


def _make_snapshot(state, n_actions, action_idx, hero_slumbot_pos):
    """Build a snapshot dict in evaluate.py-compatible format.

    Stores active_pos in USER frame (1 - slumbot_pos). Chip values stay in
    Slumbot units; events are scaled at build time.
    """
    if action_idx is None:
        action_tensor = None
    else:
        action_tensor = torch.zeros(n_actions, dtype=torch.float32)
        action_tensor[action_idx] = 1.0
    return {
        "pot": float(state["pot"]),
        "bets": np.array(state["bets"], dtype=np.float32),
        "credits": list(state["credits"]),
        "turn": int(state["turn"]),
        "active_pos": 1 - state["active_pos"],
        "action": action_tensor,
    }


def _apply_token(state, token, raise_sizes, n_raise_bins):
    """Apply a single Slumbot token (k/c/f/b<N>) to state.
    Returns the action_idx that this token represents.
    """
    pos = state["active_pos"]
    raise_sizes_for_street = raise_sizes[state["turn"]]

    # Encode action_idx BEFORE mutating state (state_pre semantics)
    action_idx = _token_to_action_idx(
        state, token, raise_sizes_for_street, n_raise_bins
    )

    if token == "f":
        state["players_state"][pos] = -1
        state["is_terminal"] = True
        return action_idx

    if token == "k":
        # check (no chip movement)
        pass
    elif token == "c":
        # call: match high_bet
        amount = state["high_bet"] - state["bets"][pos]
        amount = min(amount, state["credits"][pos])
        state["bets"][pos] += amount
        state["credits"][pos] -= amount
        state["pot"] += amount
        if state["credits"][pos] <= 0:
            state["players_state"][pos] = 2  # all-in
        state["last_bet_size"] = 0
    elif token.startswith("b"):
        new_total = int(token[1:])
        added = new_total - state["bets"][pos]
        added = min(added, state["credits"][pos])
        state["bets"][pos] += added
        state["credits"][pos] -= added
        state["pot"] += added
        new_last_bet_size = state["bets"][pos] - state["high_bet"]
        if new_last_bet_size > 0:
            state["last_bet_size"] = new_last_bet_size
        state["high_bet"] = max(state["high_bet"], state["bets"][pos])
        if state["credits"][pos] <= 0:
            state["players_state"][pos] = 2
    else:
        raise ValueError(f"Unknown token: {token!r}")

    return action_idx


def _advance_after_action(state, token):
    """After applying a non-fold action, decide whether the street ends or
    play continues. Mirrors sample_api.py ParseAction transitions."""
    pos = state["active_pos"]
    other = 1 - pos
    # Did this action close the street?
    if token == "k":
        # check by SB preflop never closes (BB still to act); BB check
        # postflop closes after both checked. We track via "both checked".
        # Simpler: check ends street if the other player has already acted
        # this street (i.e., the other was last to act and it was a check).
        # State: we infer using last_bet_size (0 means no outstanding bet).
        # If high_bet equals current bets, opponent has matched (or both 0):
        opp_done = (state["bets"][other] == state["high_bet"]
                    and (state["turn"] > 0 or state["bets"][other] == SLUMBOT_BIG_BLIND))
        # Special: postflop opening check from BB (first to act) does NOT
        # close — opponent hasn't acted yet. We detect "first action of street"
        # via a flag passed in via state['_first_in_street'].
        if state.pop("_first_in_street", False):
            opp_done = False
        if opp_done:
            _street_advance_or_terminal(state)
            return
        # otherwise continue: pass turn
        state["active_pos"] = other
        return

    if token == "c":
        # Call closes the street UNLESS preflop BB option (SB limps then BB
        # still has option). sample_api: after a call, next is opponent's
        # turn but `check_or_call_ends_street=True` — meaning if the OPPONENT
        # next acts and check/calls, the street is closed. Practically, a
        # call always closes the street EXCEPT preflop SB-limp (SB calls BB
        # for 50→100; BB still has option). We handle via _first_in_street.
        if state.pop("_first_in_street", False):
            # SB limp preflop — BB still gets option
            state["active_pos"] = other
            return
        _street_advance_or_terminal(state)
        return

    # bet/raise: opponent must respond
    state.pop("_first_in_street", None)
    state["active_pos"] = other
    state["players_state"][pos] = 0  # already acted (waiting for response)
    state["players_state"][other] = 1
    return


def _street_advance_or_terminal(state):
    """Move to next street or mark terminal at showdown."""
    if state["turn"] >= SLUMBOT_NUM_STREETS - 1:
        state["is_terminal"] = True
        return
    state["turn"] += 1
    state["bets"] = [0, 0]
    state["high_bet"] = 0
    state["last_bet_size"] = 0
    # Postflop: BB (Slumbot pos 0) acts first
    state["active_pos"] = 0
    state["players_state"] = [1, 1]
    state["_first_in_street"] = True


def _replay_action_string(action_str, hole_cards_int, board_ints,
                          client_pos, raise_sizes, n_raise_bins, n_actions,
                          hero_action_indices):
    """Re-parse the entire Slumbot action string from scratch, building
    SlumbotState + snapshot list.

    The snapshot pattern matches evaluate.py training distribution:
    initial pre-decision snap, then for each action (pre-decision, post-action)
    pairs. Pre-decision snap mirrors the prior post-action snap with action=None.

    Returns: (state, snapshots, hero_moves_seen)
    """
    state = _initial_state(hole_cards_int, board_ints)
    state["_first_in_street"] = True
    # Initial snap also serves as pre-decision for the first action
    snapshots = [_make_snapshot(state, n_actions, None, client_pos)]
    hero_moves_seen = 0

    if not action_str:
        return state, snapshots, hero_moves_seen

    i = 0
    sz = len(action_str)
    is_first_token = True
    while i < sz:
        c = action_str[i]
        if c == "/":
            # Tolerate stray slashes (e.g., "b20000c///" — all-in runout).
            i += 1
            continue

        if c in ("k", "c", "f"):
            token = c
            i += 1
        elif c == "b":
            j = i + 1
            while j < sz and action_str[j].isdigit():
                j += 1
            token = action_str[i:j]
            i = j
        else:
            raise ValueError(f"Unknown char {c!r} at offset {i} in {action_str!r}")

        # Pre-decision snap (skip for first token — initial snap covers it)
        if not is_first_token:
            snapshots.append(_make_snapshot(state, n_actions, None, client_pos))
        is_first_token = False

        is_hero = (state["active_pos"] == client_pos)
        if is_hero and hero_moves_seen < len(hero_action_indices):
            action_idx = hero_action_indices[hero_moves_seen]
            hero_moves_seen += 1
            _apply_token(state, token, raise_sizes, n_raise_bins)
        else:
            action_idx = _apply_token(state, token, raise_sizes, n_raise_bins)
            if is_hero:
                hero_moves_seen += 1

        if state["is_terminal"]:
            snapshots.append(_make_snapshot(state, n_actions, action_idx, client_pos))
            break

        _advance_after_action(state, token)
        snapshots.append(_make_snapshot(state, n_actions, action_idx, client_pos))

        if state["is_terminal"]:
            break

    return state, snapshots, hero_moves_seen


# ============================================================================
# Event building
# ============================================================================

def _build_events(snapshots, hole_cards_int, board_ints, hero_user_pos,
                  client_pos, num_players, big_blind_internal,
                  small_blind_internal, chip_scale, n_actions,
                  hero_id="hero", opp_id="slumbot"):
    """Convert snapshots → list of evaluate.py-compatible event dicts.

    Chip values are divided by chip_scale to land in training-time units.
    """
    events = []
    inv_scale = 1.0 / chip_scale
    # Always include all 5 board slots (-1 for unrevealed)
    table = list(board_ints)

    for snap in snapshots:
        action = snap["action"]
        if action is None:
            action = torch.zeros(n_actions, dtype=torch.float32)
        # active_pos in snap is already in user frame (1 - slumbot_pos)
        acting_pos_user = snap["active_pos"]
        opponent_id = hero_id if acting_pos_user == hero_user_pos else opp_id
        # Reorder bets to user frame: user_pos = 1 - slumbot_pos
        bets_user = np.zeros(num_players, dtype=np.float32)
        for slumbot_pos in range(2):
            user_pos = 1 - slumbot_pos
            bets_user[user_pos] = float(snap["bets"][slumbot_pos]) * inv_scale
        stack_user_hero = float(snap["credits"][1 - hero_user_pos]) * inv_scale
        events.append({
            "hand": list(hole_cards_int),
            "num_players": num_players,
            "hero_pos": hero_user_pos,
            "acting_pos": acting_pos_user,
            "big_blind": float(big_blind_internal),
            "small_blind": float(small_blind_internal),
            "stack": stack_user_hero,
            "table": table,
            "pot": float(snap["pot"]) * inv_scale,
            "bets": bets_user,
            "action": action,
            "opponent_id": opponent_id,
        })
    return events


def _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins, chip_scale):
    """Construct a GameState (in training chip units) from SlumbotState for MCTS."""
    inv_scale = 1.0 / chip_scale
    # Reorder to user frame
    bets_user = [0.0, 0.0]
    credits_user = [0.0, 0.0]
    players_state_user = [0, 0]
    for slumbot_pos in range(2):
        user_pos = 1 - slumbot_pos
        bets_user[user_pos] = float(state["bets"][slumbot_pos]) * inv_scale
        credits_user[user_pos] = float(state["credits"][slumbot_pos]) * inv_scale
        players_state_user[user_pos] = int(state["players_state"][slumbot_pos])

    return GameState(
        num_players=2,
        hero_pos=hero_user_pos,
        active_player=1 - state["active_pos"],
        players_state=players_state_user,
        credits=credits_user,
        bets=bets_user,
        pot=float(state["pot"]) * inv_scale,
        high_bet=float(state["high_bet"]) * inv_scale,
        turn=int(state["turn"]),
        raise_sizes=raise_sizes,
        n_raise_bins=n_raise_bins,
        is_terminal=False,
        several_all_in=False,
    )


# ============================================================================
# HTTP client
# ============================================================================

class SlumbotClient:
    """Thin wrapper over Slumbot's HTTP API.

    Retries transient network failures (ConnectTimeout / ReadTimeout /
    ConnectionError) with exponential backoff. Uses a persistent
    requests.Session so TCP+TLS handshakes are reused across requests
    (significant speedup on long evals).
    """

    def __init__(self, host=SLUMBOT_HOST, username="", password="",
                 timeout=30, retries=4, backoff=1.0, log=None):
        self.host = host
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self.log = log
        self.session = requests.Session()
        self.token = None
        if username and password:
            self.token = self._login(username, password)

    def _post(self, endpoint, data):
        import time
        url = f"https://{self.host}/slumbot/api/{endpoint}"
        # Retry on transient network errors (timeout / connection reset).
        # Other errors (HTTP 4xx/5xx, error_msg in body) propagate immediately.
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(url, json=data, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Slumbot {endpoint} HTTP {r.status_code}: {r.text}")
                body = r.json()
                if "error_msg" in body:
                    raise RuntimeError(
                        f"Slumbot {endpoint} error: {body['error_msg']}")
                new_tok = body.get("token")
                if new_tok:
                    self.token = new_tok
                return body
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt >= self.retries:
                    break
                wait = self.backoff * (2 ** attempt)
                if self.log is not None:
                    self.log(
                        f"  Slumbot {endpoint} {type(e).__name__}, "
                        f"retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.retries})")
                time.sleep(wait)
        raise RuntimeError(
            f"Slumbot {endpoint} failed after {self.retries + 1} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def _login(self, username, password):
        body = self._post("login", {"username": username, "password": password})
        tok = body.get("token")
        if not tok:
            raise RuntimeError("Slumbot login: no token in response")
        return tok

    def new_hand(self):
        data = {}
        if self.token:
            data["token"] = self.token
        return self._post("new_hand", data)

    def act(self, incr):
        if not self.token:
            raise RuntimeError("act() before token established (call new_hand first)")
        return self._post("act", {"token": self.token, "incr": incr})


# ============================================================================
# Agent loading
# ============================================================================

def _resolve_agent_path(path, project_root, version):
    """Resolve a relative agent path against data/<version>/."""
    if not path:
        raise ValueError("agent entry missing 'path'")
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, "data", version, path)


def _load_one_agent(agent_entry, config, device, project_root, version,
                    fallback_temperature, log):
    """Load one agent from a path. Returns dict bundle or None on failure."""
    path = _resolve_agent_path(agent_entry["path"], project_root, version)
    name = agent_entry.get("name") or os.path.basename(path.rstrip("/"))

    ckpt_path = _resolve_checkpoint_path(path)
    if ckpt_path is None:
        log(f"WARNING: no checkpoint for agent '{name}' at {path}, skipping")
        return None

    asi = ASI(log, config)
    asi.set_device(device)
    asi.load_checkpoint(ckpt_path)
    asi.eval()

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        log(f"WARNING: no norm_stats in '{name}', using identity normalization")
        norm_stats = {
            "pot_mean": 0.0, "pot_std": 1.0,
            "stack_mean": 0.0, "stack_std": 1.0,
            "bets_mean": 0.0, "bets_std": 1.0,
            "blind_mean": 0.0, "blind_std": 1.0,
        }
    # Temperature precedence: checkpoint > per-agent override > section fallback.
    # Checkpoint wins by default; entry override only kicks in if checkpoint
    # didn't store one.
    ckpt_temp = ckpt.get("temperature")
    if ckpt_temp is not None:
        temperature = float(ckpt_temp)
    else:
        entry_temp = agent_entry.get("action_temperature")
        temperature = float(entry_temp) if entry_temp is not None else fallback_temperature

    use_opp_emb = bool(agent_entry.get("use_opponent_embedding", False))
    use_mcts = bool(agent_entry.get("use_mcts", False))

    opp_table = None
    if use_opp_emb and asi.perception.opp_emb_enabled:
        opp_table = OpponentEmbeddingTable(asi.perception.d_model)

    mcts = None
    if use_mcts:
        mcts_cfg = config.get("mcts", {})
        mcts = MCTS(asi, device, mcts_cfg, opponent_emb_table=opp_table)

    log(f"Loaded '{name}' from {ckpt_path} "
        f"(temp={temperature}, mcts={use_mcts}, opp_emb={opp_table is not None})")

    return {
        "name": name,
        "agent": asi,
        "norm_stats": norm_stats,
        "temperature": float(temperature),
        "opp_table": opp_table,
        "mcts": mcts,
        "use_opp_emb": use_opp_emb,
    }


# ============================================================================
# Decision making
# ============================================================================

def _choose_action(bundle, events, state, hero_user_pos, raise_sizes,
                   n_raise_bins, n_actions, chip_scale,
                   amp_enabled, device_type, amp_dtype):
    """Run the agent (or MCTS) on the events and return the chosen action_idx."""
    asi = bundle["agent"]
    norm_stats = bundle["norm_stats"]
    # Normalize a copy in place (events are local to this hand)
    _normalize_events_inplace(events, norm_stats)

    gs = _build_game_state(state, hero_user_pos, raise_sizes, n_raise_bins, chip_scale)

    if bundle["mcts"] is not None:
        return int(bundle["mcts"].search([events], gs))

    opp_table = bundle["opp_table"]
    skip_opp = (opp_table is None)
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            out = asi.forward_batch(
                [events], skip_memory=True,
                skip_opponent_emb=skip_opp,
                opponent_emb_table=opp_table,
            )
    logits = out["action_logits"][0]
    legal_mask = torch.tensor(
        gs.get_legal_action_mask(n_actions), dtype=torch.bool, device=logits.device,
    )
    logits = logits.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(logits / max(bundle["temperature"], 1e-3), dim=0)
    return int(torch.multinomial(probs, 1).item())


# ============================================================================
# Hand loop
# ============================================================================

def _play_one_hand(client, bundle, config, raise_sizes, n_raise_bins, n_actions,
                   chip_scale, big_blind_internal, small_blind_internal,
                   amp_enabled, device_type, amp_dtype, clamp_counters, log,
                   action_hist=None, action_hist_by_street=None):
    """Play one Slumbot hand. Returns (winnings, baseline_winnings).

    Optional analytics buffers (mutated in place):
      action_hist: np.ndarray (n_actions,) — counts per chosen idx (post-clamp)
      action_hist_by_street: np.ndarray (4, n_actions) — same, split by street
    """
    r = client.new_hand()
    client_pos = r["client_pos"]
    hero_user_pos = 1 - client_pos
    hole_cards_int = [_card_to_int(c) for c in r["hole_cards"]]
    board_ints = _board_to_ints(r.get("board") or [])
    hero_action_indices = []

    while True:
        # Update board if Slumbot revealed more cards
        if r.get("board"):
            board_ints = _board_to_ints(r["board"])
        action_str = r.get("action") or ""

        # Check for hand end
        if "winnings" in r and r["winnings"] is not None:
            return float(r["winnings"]), float(r.get("baseline_winnings") or 0.0)

        state, snapshots, _ = _replay_action_string(
            action_str, hole_cards_int, board_ints, client_pos,
            raise_sizes, n_raise_bins, n_actions, hero_action_indices,
        )

        if state["is_terminal"]:
            # Server should respond with winnings on next /act, but we shouldn't
            # have an action to send. This branch can only be hit if our local
            # parser disagrees with server — break to safety.
            log(f"  WARN: local state terminal but no winnings in response, action={action_str!r}")
            return 0.0, 0.0

        # Whose turn?
        if state["active_pos"] != client_pos:
            # Slumbot's turn but no winnings: shouldn't happen (server would have
            # taken its action before responding). Defensive: re-fetch by acting
            # as a check/call placeholder is illegal. Just break.
            log(f"  WARN: opponent's turn in response, action={action_str!r}")
            return 0.0, 0.0

        # Pre-decision snapshot (action=None) — match training distribution
        snapshots_with_pre = list(snapshots)
        # Append a "we're about to act" snapshot mirroring evaluate.py:371-378
        pre_snap = _make_snapshot(state, n_actions, None, client_pos)
        snapshots_with_pre.append(pre_snap)

        events = _build_events(
            snapshots_with_pre, hole_cards_int, board_ints, hero_user_pos,
            client_pos, num_players=2,
            big_blind_internal=big_blind_internal,
            small_blind_internal=small_blind_internal,
            chip_scale=chip_scale, n_actions=n_actions,
        )

        action_idx = _choose_action(
            bundle, events, state, hero_user_pos, raise_sizes,
            n_raise_bins, n_actions, chip_scale,
            amp_enabled, device_type, amp_dtype,
        )

        incr = _action_idx_to_incr(
            state, action_idx, raise_sizes[state["turn"]], n_raise_bins,
            hero_slumbot_pos=client_pos, clamp_counters=clamp_counters,
        )

        # If the clamp degraded a raise to call/check, _token_to_action_idx on
        # the resulting token will give a different idx than what we chose.
        # Cache the EFFECTIVE idx so the replay produces consistent events.
        if incr == "f":
            effective_idx = 0
        elif incr in ("c", "k"):
            effective_idx = 1
        elif incr.startswith("b"):
            effective_idx = _token_to_action_idx(
                state, incr, raise_sizes[state["turn"]], n_raise_bins,
            )
        else:
            effective_idx = action_idx
        hero_action_indices.append(effective_idx)

        # Analytics: record action distribution per street
        if action_hist is not None:
            action_hist[effective_idx] += 1
        if action_hist_by_street is not None:
            action_hist_by_street[int(state["turn"]), effective_idx] += 1

        r = client.act(incr)


# ============================================================================
# Main runner
# ============================================================================

def _resolve_paths(config):
    here = os.path.dirname(os.path.abspath(__file__))
    version = os.path.basename(os.path.abspath(os.path.join(here, "..")))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return version, project_root


def _bb_per_100(chip_winnings_sum, n_hands):
    if n_hands <= 0:
        return 0.0
    return (chip_winnings_sum / SLUMBOT_BIG_BLIND) / (n_hands / 100.0)


def _stderr_bb_per_100(per_hand_chips):
    if len(per_hand_chips) < 2:
        return 0.0
    arr = np.asarray(per_hand_chips, dtype=np.float64) / SLUMBOT_BIG_BLIND
    return float(arr.std(ddof=1) / np.sqrt(len(arr)) * 100.0)


def run_slumbot_evaluation(config, device, log, results_dir_override=None):
    cfg = config.get("slumbot_eval", {})
    if not cfg:
        log("slumbot_eval: no config section, skipping")
        return

    agents_cfg = cfg.get("agents", [])
    if not agents_cfg:
        log("slumbot_eval: agents list empty, skipping")
        return

    n_hands = int(cfg.get("n_hands", 1000))
    log_every = int(cfg.get("log_every", 50))
    fallback_temperature = float(cfg.get("action_temperature", 0.5))
    host = cfg.get("host", SLUMBOT_HOST)
    username = cfg.get("username", "") or ""
    password = cfg.get("password", "") or ""
    timeout = int(cfg.get("request_timeout", 30))
    retries = int(cfg.get("retries", 4))
    backoff = float(cfg.get("backoff", 1.0))

    game_cfg = config.get("game", {})
    big_blind_internal = float(game_cfg.get("big_blind", 10))
    small_blind_internal = big_blind_internal / 2.0
    chip_scale = SLUMBOT_BIG_BLIND / big_blind_internal

    raise_sizes = _get_raise_sizes(game_cfg)
    n_raise_bins = len(raise_sizes[0])
    n_actions = n_raise_bins + 3

    amp_enabled, device_type, amp_dtype, _ = get_amp_config(device)

    version, project_root = _resolve_paths(config)
    if results_dir_override:
        results_dir = results_dir_override
    else:
        exp_name = config.get("name", "default")
        results_dir = os.path.join(project_root, "data", version, exp_name, "slumbot_eval")
    os.makedirs(results_dir, exist_ok=True)

    log("=== Slumbot evaluation ===")
    log(f"host={host}, n_hands={n_hands}/agent, "
        f"chip_scale={chip_scale} (BB internal={big_blind_internal})")

    all_results = {}

    for agent_entry in agents_cfg:
        bundle = _load_one_agent(
            agent_entry, config, device, project_root, version,
            fallback_temperature, log,
        )
        if bundle is None:
            continue

        name = bundle["name"]
        client = SlumbotClient(host=host, username=username,
                               password=password, timeout=timeout,
                               retries=retries, backoff=backoff, log=log)
        per_hand_chips = []
        per_hand_baseline = []
        clamp_counters = defaultdict(int)
        action_hist = np.zeros(n_actions, dtype=np.int64)
        action_hist_by_street = np.zeros((4, n_actions), dtype=np.int64)
        history = {"bb100_raw": [], "bb100_baseline": []}
        hands_failed = 0

        log(f"\n--- Playing {name} for {n_hands} hands ---")
        pbar = tqdm(total=n_hands, desc=f"Slumbot/{name}", unit="hand")
        for hand_idx in range(n_hands):
            try:
                w, b = _play_one_hand(
                    client, bundle, config, raise_sizes, n_raise_bins, n_actions,
                    chip_scale, big_blind_internal, small_blind_internal,
                    amp_enabled, device_type, amp_dtype, clamp_counters, log,
                    action_hist=action_hist,
                    action_hist_by_street=action_hist_by_street,
                )
            except Exception as e:
                hands_failed += 1
                log(f"  hand {hand_idx + 1} failed: {type(e).__name__}: {e}")
                pbar.update(1)
                # Try to recover by waiting briefly and starting a new hand
                continue

            per_hand_chips.append(w)
            per_hand_baseline.append(b)

            done = len(per_hand_chips)
            # `baseline_winnings` from Slumbot is its AIVAT-style low-variance
            # estimator of the SAME quantity as `winnings` (expected hero
            # chip P&L), with chance and known-strategy variance removed.
            # The right "baseline-corrected" winrate is therefore the mean
            # of baseline_winnings — NOT the difference (which is the AIVAT
            # correction term and should be ≈ 0 in expectation).
            running_bcorr = _bb_per_100(sum(per_hand_baseline), done)
            running_residual = _bb_per_100(
                sum(w_i - b_i for w_i, b_i in zip(per_hand_chips, per_hand_baseline)),
                done,
            )
            pbar.set_postfix({
                "BB/100 base": f"{running_bcorr:+.1f}",
                "failed": hands_failed,
            }, refresh=False)
            pbar.update(1)
            if done > 0 and done % log_every == 0:
                raw = _bb_per_100(sum(per_hand_chips), done)
                stderr_raw = _stderr_bb_per_100(per_hand_chips)
                stderr_bcorr = _stderr_bb_per_100(per_hand_baseline)
                history["bb100_raw"].append((done, raw))
                history["bb100_baseline"].append((done, running_bcorr))
                log(f"  [{name}] {done}/{n_hands}: "
                    f"raw={raw:+.2f} BB/100 (stderr={stderr_raw:.2f}), "
                    f"baseline_corrected={running_bcorr:+.2f} BB/100 (stderr={stderr_bcorr:.2f}), "
                    f"aivat_residual={running_residual:+.2f} BB/100, "
                    f"clamps={dict(clamp_counters)}")
        pbar.close()

        n_played = len(per_hand_chips)
        total_chips = float(sum(per_hand_chips))
        total_baseline = float(sum(per_hand_baseline))
        bb100_raw = _bb_per_100(total_chips, n_played)
        bb100_bcorr = _bb_per_100(total_baseline, n_played)
        bb100_residual = _bb_per_100(total_chips - total_baseline, n_played)
        stderr_bb100 = _stderr_bb_per_100(per_hand_chips)
        stderr_bb100_bcorr = _stderr_bb_per_100(per_hand_baseline)
        mbb_per_hand_raw = bb100_raw * 10.0  # 1000 mbb / 100 hands

        # Action distribution analytics
        total_actions = int(action_hist.sum())
        action_dist = (action_hist / max(total_actions, 1)).tolist()
        action_dist_by_street = (
            action_hist_by_street
            / np.maximum(action_hist_by_street.sum(axis=1, keepdims=True), 1)
        ).tolist()

        agent_result = {
            "hands_played": n_played,
            "hands_failed": hands_failed,
            "session_total_chips": total_chips,
            "session_baseline_total_chips": total_baseline,
            "bb_per_100_raw": round(bb100_raw, 4),
            "bb_per_100_baseline_corrected": round(bb100_bcorr, 4),
            "bb_per_100_aivat_residual": round(bb100_residual, 4),
            "mbb_per_hand_raw": round(mbb_per_hand_raw, 4),
            "stderr_bb_per_100": round(stderr_bb100, 4),
            "stderr_bb_per_100_baseline_corrected": round(stderr_bb100_bcorr, 4),
            "clamps": dict(clamp_counters),
            "use_mcts": bundle["mcts"] is not None,
            "use_opponent_embedding": bundle["opp_table"] is not None,
            "temperature": bundle["temperature"],
            "decisions_made": total_actions,
            "fold_rate": round(action_dist[0], 4) if total_actions else 0.0,
            "allin_rate": round(action_dist[n_actions - 1], 4) if total_actions else 0.0,
            "action_distribution": [round(p, 4) for p in action_dist],
            "action_distribution_by_street": [
                [round(p, 4) for p in row] for row in action_dist_by_street
            ],
            "action_counts_total": [int(c) for c in action_hist],
            "action_counts_by_street": [
                [int(c) for c in row] for row in action_hist_by_street
            ],
        }
        all_results[name] = agent_result

        # Save per-agent history (raw per-hand outcomes for variance analysis)
        hist_path = os.path.join(results_dir, f"{log.init_time}_{name}.history.pt")
        torch.save({
            "per_hand_chips": per_hand_chips,
            "per_hand_baseline": per_hand_baseline,
            "history": history,
            "action_hist": action_hist.tolist(),
            "action_hist_by_street": action_hist_by_street.tolist(),
        }, hist_path)

        log(f"\n[{name}] FINAL: raw={bb100_raw:+.2f} BB/100 (stderr={stderr_bb100:.2f}), "
            f"baseline_corrected={bb100_bcorr:+.2f} BB/100 (stderr={stderr_bb100_bcorr:.2f}), "
            f"aivat_residual={bb100_residual:+.2f} BB/100 "
            f"({n_played} hands, {hands_failed} failed)")
        log(f"[{name}] Clamps: {dict(clamp_counters)}")

    # Save aggregate JSON
    out_json = os.path.join(results_dir, f"{log.init_time}.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_hands_per_agent": n_hands,
            "chip_scale": chip_scale,
            "big_blind_internal": big_blind_internal,
            "agents": all_results,
            "config": cfg,
        }, f, indent=4)
    log(f"\nResults saved to {out_json}")


# ============================================================================
# CLI
# ============================================================================

def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent against Slumbot")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = _pick_device()

    from utils import Logger
    version, project_root = _resolve_paths(config)
    name = config.get("name", "default")
    base_dir = os.path.join(project_root, "data", version, name)
    log = Logger(base_dir)
    log(f"Slumbot eval — version={version}, experiment={name}, device={device}")

    run_slumbot_evaluation(config, device, log)


if __name__ == "__main__":
    main()
