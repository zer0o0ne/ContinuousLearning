"""E2E tests for the SlumbotClient retry policy (2026-08 fix).

Scenario reproduced: during a 4-worker Slumbot run, hero thinks for seconds
between two `act` calls (MCTS). A request reaches Slumbot and IS applied, but
the response doesn't come back in time → ReadTimeout. The old client replayed
the same `incr` with the same token, which landed on an already-advanced state
and came back as "Illegal call" / "Unexpected action", killing the hand.

Covered behaviour (evaluation/slumbot_eval.py, SlumbotClient):
  1. `act` is NOT replayed after a ReadTimeout — the server sees exactly one
     application, the client raises, the hand is abandoned cleanly.
  2. `act` IS retried on ConnectTimeout (connection never established, so the
     request provably never reached the server).
  3. `new_hand` / `login` stay idempotent: they retry on ReadTimeout and
     succeed.
  4. Connections are not kept alive (`Connection: close`), so a long think
     between requests cannot leave a stale socket behind.

All tests are deterministic: a scripted in-process fake server, no network,
no sleeps (backoff=0), no randomness.

Run (from versions/v7):
    python -m pytest tests/test_slumbot_client_retry.py -v
"""

import pytest
import requests

from evaluation import slumbot_eval
from evaluation.slumbot_eval import SlumbotClient


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.text = str(body)

    def json(self):
        return self._body


class FakeSlumbotSession:
    """Scripted stand-in for requests.Session.

    `failures` maps endpoint name → list of exceptions to raise on successive
    calls (None = respond normally). A raised exception still counts as a
    server-side application of the request — that's the whole point: the
    request arrived, only the response was lost.
    """

    def __init__(self, failures=None, act_replies=None):
        self.headers = {}
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.act_replies = list(act_replies or [])
        self.calls = []          # (endpoint, payload) in arrival order
        self.hand_actions = []   # `incr` values the server actually applied

    def post(self, url, json=None, timeout=None):
        endpoint = url.rstrip("/").split("/")[-1]
        self.calls.append((endpoint, json))
        if endpoint == "act":
            incr = json["incr"]
            if self.hand_actions:
                # A second action on a hand that only ever had one legal
                # action left — exactly how Slumbot answers a replay.
                self.hand_actions.append(incr)
                body = {"error_msg": "Illegal call"}
            else:
                self.hand_actions.append(incr)
                body = (self.act_replies.pop(0) if self.act_replies
                        else {"token": "tok-after-act", "action": incr})
        elif endpoint == "new_hand":
            body = {"token": "tok-new-hand", "action": "", "hole_cards": ["Ac", "Kd"]}
        elif endpoint == "login":
            body = {"token": "tok-login"}
        else:
            body = {}

        queue = self.failures.get(endpoint)
        if queue:
            exc = queue.pop(0)
            if exc is not None:
                raise exc
        return FakeResponse(body)


@pytest.fixture
def fake_session(monkeypatch):
    """Installs a FakeSlumbotSession factory; returns a setter for the script."""
    holder = {}

    def _install(**kwargs):
        session = FakeSlumbotSession(**kwargs)
        monkeypatch.setattr(slumbot_eval.requests, "Session", lambda: session)
        holder["session"] = session
        return session

    return _install


def _client(log_lines=None):
    return SlumbotClient(host="slumbot.test", timeout=1, retries=4,
                         backoff=0.0,
                         log=(log_lines.append if log_lines is not None else None))


class TestActNotReplayed:
    def test_read_timeout_on_act_does_not_replay(self, fake_session):
        """The exact production failure: response lost after the server applied
        the action. Client must raise, not re-send."""
        session = fake_session(
            failures={"act": [requests.exceptions.ReadTimeout("read timed out")]})
        client = _client()
        client.new_hand()

        with pytest.raises(RuntimeError) as excinfo:
            client.act("c")

        # Server saw the action exactly once — no replay, hence no
        # "Illegal call" / "Unexpected action" desync.
        assert session.hand_actions == ["c"]
        assert sum(1 for ep, _ in session.calls if ep == "act") == 1
        msg = str(excinfo.value)
        assert "not retried" in msg
        assert "ReadTimeout" in msg

    def test_replay_would_have_desynced(self, fake_session):
        """Guard on the fake: a second act on the same hand is what Slumbot
        rejects. Proves test 1 asserts something real."""
        session = fake_session()
        client = _client()
        client.new_hand()
        client.act("c")

        with pytest.raises(RuntimeError, match="Illegal call"):
            client.act("c")
        assert session.hand_actions == ["c", "c"]

    def test_connection_error_on_act_does_not_replay(self, fake_session):
        session = fake_session(
            failures={"act": [requests.exceptions.ConnectionError("reset by peer")]})
        client = _client()
        client.new_hand()

        with pytest.raises(RuntimeError, match="not retried"):
            client.act("f")
        assert session.hand_actions == ["f"]

    def test_connect_timeout_on_act_is_retried(self, fake_session):
        """Connection never established → the request provably never arrived,
        so replaying it is safe (and necessary to survive a flaky link)."""
        session = fake_session(
            failures={"act": [requests.exceptions.ConnectTimeout("connect timed out")]})
        client = SlumbotClient(host="slumbot.test", timeout=1, retries=4,
                              backoff=0.0)
        client.new_hand()
        # The fake applies the action before raising, so the retry hits the
        # "already acted" branch; what matters here is that a retry happened.
        with pytest.raises(RuntimeError, match="Illegal call"):
            client.act("c")
        assert sum(1 for ep, _ in session.calls if ep == "act") == 2


class TestIdempotentEndpointsStillRetry:
    def test_new_hand_retries_on_read_timeout(self, fake_session):
        session = fake_session(
            failures={"new_hand": [requests.exceptions.ReadTimeout("read timed out"),
                                   None]})
        log_lines = []
        client = _client(log_lines)

        body = client.new_hand()

        assert body["token"] == "tok-new-hand"
        assert client.token == "tok-new-hand"
        assert sum(1 for ep, _ in session.calls if ep == "new_hand") == 2
        assert any("retrying" in line for line in log_lines)

    def test_new_hand_gives_up_after_configured_retries(self, fake_session):
        timeouts = [requests.exceptions.ReadTimeout("read timed out")] * 3
        session = fake_session(failures={"new_hand": timeouts})
        client = SlumbotClient(host="slumbot.test", timeout=1, retries=2,
                              backoff=0.0)

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.new_hand()
        assert sum(1 for ep, _ in session.calls if ep == "new_hand") == 3


class TestNoKeepAlive:
    def test_connection_close_header(self, fake_session):
        session = fake_session()
        client = _client()
        assert client.session is session
        assert session.headers.get("Connection") == "close"
