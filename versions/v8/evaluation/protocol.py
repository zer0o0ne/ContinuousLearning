"""Slumbot's wire protocol (CONCEPT.md §12, `PLAN_PIPELINE.md` S10).

This is v7's evaluation code, kept. Everything here maps between Slumbot's HTTP
API and a poker state and back — the HTTP client and its retry policy, the
action-string grammar, the token ↔ action-index translation, the replay of an
action string into a betting state, and the BB/100 accounting. None of it knows
what a v8 agent is, and none of it changed when the agent did, which is exactly
why it survived the rewrite: `evaluation/slumbot_eval.py` was deleted and its
v7-specific half — the event builder, the action chooser, the MCTS and solver
paths — went with it, while this file is the half that was architecture-
independent all along.

**This is plumbing, not specialisation** (`CLAUDE.md` §1, `CONCEPT.md` §10).
Adapting to Slumbot's wire format and bet-size grammar at *evaluation* time is
allowed and belongs here. What must never appear in this tree is a
Slumbot-specific policy branch, an opponent model keyed on "this is Slumbot", or
an assumption that a table is heads-up or a stack is 200 BB — the adapter takes
both from config like any other parameter (`evaluation/v8_adapter.py`).

**Two frames, and confusing them is the classic bug.** Slumbot numbers its seats
`pos 0 = BB`, `pos 1 = SB`, and the SB acts first preflop. v8's engine numbers
them the other way — `env/table.py::start_table` posts the small blind at seat 0
and the big blind at seat 1, seat 0 acts first preflop and, heads-up, seat 1
acts first postflop (`next_turn`'s `start_pos = 1 if num_players == 2`). So the
two conventions are exactly each other's mirror and `v8_seat = 1 - slumbot_pos`
holds on every street. Everything in *this* file is in Slumbot's frame; the flip
happens once, in the adapter.

**What changed from v7's file.** The bodies are verbatim; three things are not:

* the leading underscores are gone from the names that cross a module boundary,
  since this is now a module with a surface rather than a section of one file;
* `replay_action_string` returns a **neutral trace** — one entry per action,
  carrying the state as it was *before* that action — instead of v7-format
  snapshots. Building an observation out of the trace is the adapter's job, and
  v7's snapshot format was the one genuinely architecture-specific thing in the
  replay;
* the state no longer carries the cards. They are the caller's, they were only
  ever there for v7's event builder, and the betting state is the whole of what
  a replay can know.
"""

import copy
import time

import numpy as np
import requests

# Slumbot's fixed parameters (from its own `sample_api.py`). They are the
# table Slumbot deals, not a configuration of ours — the adapter checks them
# against `game.players_range` / `game.stack_bb_range` and refuses rather than
# clamps if they fall outside what the agent was trained over.
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

def card_to_int(card_str):
    """Slumbot card "Ac"/"Td"/etc → int 0..51 with rank*4 + suit encoding."""
    if len(card_str) != 2:
        raise ValueError(f"Invalid card '{card_str}'")
    rank, suit = card_str[0], card_str[1].lower()
    if rank not in _RANK_TO_IDX:
        raise ValueError(f"Invalid rank in '{card_str}'")
    if suit not in _SUIT_TO_IDX:
        raise ValueError(f"Invalid suit in '{card_str}'")
    return _RANK_TO_IDX[rank] * 4 + _SUIT_TO_IDX[suit]


def board_to_ints(board_strs):
    """5-card board (-1 padded). board_strs may be empty / 3 / 4 / 5 long."""
    ints = [card_to_int(c) for c in board_strs]
    while len(ints) < 5:
        ints.append(-1)
    return ints


# ============================================================================
# Action <-> discrete index translation
# ============================================================================

def token_to_action_idx(state_pre, token, raise_sizes_for_street, n_raise_bins):
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


def action_idx_to_incr(state_pre, action_idx, raise_sizes_for_street,
                       n_raise_bins, hero_slumbot_pos, clamp_counters):
    """Translate our discrete action_idx into a legal Slumbot 'incr' string.

    Applies fallback clamping:
      - fold without facing a bet → check
      - raise below min legal → bump to min legal
      - raise above stack → cap at all-in
      - raise that doesn't exceed call → emit call/check
    Each clamp increments the corresponding counter.

    The counters are not decoration: a clamp is the abstraction gap showing
    itself, and a run in which they fire often is a run whose reported BB/100
    belongs to a policy slightly different from the one that was trained.
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
        call_total = high_bet
        dist_to_call = new_total - call_total
        dist_to_min = min_legal_total - new_total
        if dist_to_call <= dist_to_min:
            clamp_counters["raise_rounded_to_call"] += 1
            return "c" if facing_bet else "k"
        clamp_counters["raise_bumped"] += 1
        new_total = min_legal_total
    if new_total > cap:
        clamp_counters["raise_to_allin"] += 1
        new_total = cap

    return f"b{int(round(new_total))}"


def effective_action_idx(state_pre, incr, raise_sizes_for_street, n_raise_bins):
    """The index hero *actually* played, recovered from the token it sent.

    `action_idx_to_incr` clamps, so a raise can leave as a call and a fold as a
    check. The next replay of the action string has to agree with what the
    server saw, and the server saw the token — so this is the index that goes
    into `hero_action_indices`, not the one the agent chose. Getting this wrong
    is silent: the observation of every later decision in the hand carries an
    action hero did not take.
    """
    if incr == "f":
        return 0
    if incr in ("c", "k"):
        return 1
    return token_to_action_idx(state_pre, incr, raise_sizes_for_street,
                               n_raise_bins)


def clamp_counters():
    """A fresh set of the counters `action_idx_to_incr` increments."""
    return {"fold_to_check": 0, "raise_to_call": 0, "raise_rounded_to_call": 0,
            "raise_bumped": 0, "raise_to_allin": 0}


# ============================================================================
# Action-string replay
# ============================================================================

def initial_state():
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
    }


def apply_token(state, token, raise_sizes, n_raise_bins):
    """Apply a single Slumbot token (k/c/f/b<N>) to state.
    Returns the action_idx that this token represents.
    """
    pos = state["active_pos"]
    raise_sizes_for_street = raise_sizes[state["turn"]]

    # Encode action_idx BEFORE mutating state (state_pre semantics)
    action_idx = token_to_action_idx(
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


def advance_after_action(state, token):
    """After applying a non-fold action, decide whether the street ends or
    play continues. Mirrors sample_api.py ParseAction transitions."""
    pos = state["active_pos"]
    other = 1 - pos
    if token == "k":
        # A check ends the street only if the opponent has already acted on it.
        # "Already acted" is read off the bets: they match the high bet, and
        # preflop that means the BB is in for a full big blind rather than the
        # posted one. The first action of a street is flagged explicitly,
        # because a postflop opening check from the BB looks identical.
        opp_done = (state["bets"][other] == state["high_bet"]
                    and (state["turn"] > 0
                         or state["bets"][other] == SLUMBOT_BIG_BLIND))
        if state.pop("_first_in_street", False):
            opp_done = False
        if opp_done:
            street_advance_or_terminal(state)
            return
        state["active_pos"] = other
        return

    if token == "c":
        # A call closes the street except for the preflop SB limp, after which
        # the BB still has its option.
        if state.pop("_first_in_street", False):
            state["active_pos"] = other
            return
        street_advance_or_terminal(state)
        return

    # bet/raise: opponent must respond
    state.pop("_first_in_street", None)
    state["active_pos"] = other
    state["players_state"][pos] = 0  # already acted (waiting for response)
    state["players_state"][other] = 1
    return


def street_advance_or_terminal(state):
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


def replay_action_string(action_str, raise_sizes, n_raise_bins,
                         hero_pos=None, hero_action_indices=()):
    """Re-parse a whole Slumbot action string from scratch.

    Returns `(state, steps)` — the betting state after the last token, and one
    entry per action taken:

    ``{"state": <the state as it was before this action>, "acting_pos": <Slumbot
    seat>, "token": <wire token>, "action_idx": <our discrete index>}``

    The state in an entry is a **copy**, so the trace is a history and not a
    view of one mutating dict. That is what lets the adapter build one
    observation per past decision without replaying anything twice.

    `hero_action_indices` exists because the index hero *chose* and the index
    recoverable from the token it produced can differ: `action_idx_to_incr`
    clamps, and a clamped raise comes back off the wire as a call. The chosen
    index is the truth about what hero did, so the caller passes what it
    played, in order, and it overrides the recovered one for hero's seat.
    """
    state = initial_state()
    state["_first_in_street"] = True
    steps = []
    hero_moves_seen = 0

    if not action_str:
        return state, steps

    i = 0
    sz = len(action_str)
    while i < sz:
        c = action_str[i]
        if c == "/":
            # Tolerate stray slashes (e.g. "b20000c///" — an all-in runout).
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
            raise ValueError(
                f"Unknown char {c!r} at offset {i} in {action_str!r}")

        pre = copy.deepcopy(state)
        pre.pop("_first_in_street", None)
        acting_pos = state["active_pos"]
        is_hero = hero_pos is not None and acting_pos == hero_pos

        recovered = apply_token(state, token, raise_sizes, n_raise_bins)
        if is_hero and hero_moves_seen < len(hero_action_indices):
            action_idx = int(hero_action_indices[hero_moves_seen])
        else:
            action_idx = recovered
        if is_hero:
            hero_moves_seen += 1

        steps.append({"state": pre, "acting_pos": acting_pos, "token": token,
                      "action_idx": action_idx})

        if state["is_terminal"]:
            break
        advance_after_action(state, token)
        if state["is_terminal"]:
            break

    return state, steps


# ============================================================================
# HTTP client
# ============================================================================

class SlumbotClient:
    """Thin wrapper over Slumbot's HTTP API.

    Retry policy is per-endpoint, because `act` is NOT idempotent:

    - `login` / `new_hand` are idempotent (a replay just makes a fresh
      session/hand), so they retry on ConnectTimeout / ReadTimeout /
      ConnectionError with exponential backoff.
    - `act` retries ONLY on ConnectTimeout, i.e. the connection was never
      established and the request provably never reached the server. A
      ReadTimeout means the request WAS sent and the response didn't arrive
      in time — Slumbot has most likely already applied the action, so
      replaying the same `incr` lands on an advanced state and comes back as
      "Illegal call" / "Unexpected action", desyncing the hand. Those are
      surfaced as a failed hand instead (one lost hand per timeout, no
      desync, no failure cascade).

    Connections are NOT kept alive (`Connection: close`): hero can think for
    seconds between two `act` calls, long enough for the server or a NAT box to
    drop an idle keep-alive socket silently — the next request then vanishes and
    only surfaces as a full-`timeout` ReadTimeout. One TLS handshake per request
    (~100 ms) is negligible next to think time.
    """

    def __init__(self, host=SLUMBOT_HOST, username="", password="",
                 timeout=10, retries=4, backoff=1.0, log=None):
        self.host = host
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self.log = log
        self.session = requests.Session()
        self.session.headers["Connection"] = "close"
        self.token = None
        if username and password:
            self.token = self._login(username, password)

    def _post(self, endpoint, data, idempotent=True):
        url = f"https://{self.host}/slumbot/api/{endpoint}"
        # Retry on transient network errors (timeout / connection reset).
        # Other errors (HTTP 4xx/5xx, error_msg in body) propagate immediately.
        # For non-idempotent endpoints only ConnectTimeout is retried (see the
        # class docstring): every other failure mode may have been applied
        # server-side already.
        if idempotent:
            retryable = (requests.exceptions.ConnectTimeout,
                         requests.exceptions.ReadTimeout,
                         requests.exceptions.ConnectionError)
        else:
            retryable = (requests.exceptions.ConnectTimeout,)
        last_exc = None
        attempts = 0
        for attempt in range(self.retries + 1):
            attempts += 1
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
            except retryable as e:
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
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                # Only reachable for non-idempotent endpoints. The request may
                # have been applied server-side, so we must NOT replay it.
                raise RuntimeError(
                    f"Slumbot {endpoint} {type(e).__name__} after the request "
                    f"was sent — not retried (non-idempotent), hand abandoned: "
                    f"{e}"
                ) from e
        raise RuntimeError(
            f"Slumbot {endpoint} failed after {attempts} attempts: "
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
            raise RuntimeError(
                "act() before token established (call new_hand first)")
        # Non-idempotent: a replayed action desyncs the hand (class docstring).
        return self._post("act", {"token": self.token, "incr": incr},
                          idempotent=False)


# ============================================================================
# Result accounting
# ============================================================================

def bb_per_100(chip_winnings_sum, n_hands):
    if n_hands <= 0:
        return 0.0
    return (chip_winnings_sum / SLUMBOT_BIG_BLIND) / (n_hands / 100.0)


def stderr_bb_per_100(per_hand_chips):
    if len(per_hand_chips) < 2:
        return 0.0
    arr = np.asarray(per_hand_chips, dtype=np.float64) / SLUMBOT_BIG_BLIND
    return float(arr.std(ddof=1) / np.sqrt(len(arr)) * 100.0)


def stderr_bb_per_100_online(welford_n, welford_M2):
    """O(1) stderr of BB/100 from Welford's online variance state.

    welford_n:  number of samples incorporated so far
    welford_M2: running sum of squared deviations (already in BB units)
    Returns the same value as `stderr_bb_per_100` but without iterating the
    list — a million-hand run does not keep one.
    """
    if welford_n < 2:
        return 0.0
    variance = welford_M2 / (welford_n - 1)  # sample variance (ddof=1)
    return float(np.sqrt(variance / welford_n) * 100.0)
