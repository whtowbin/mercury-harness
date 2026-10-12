"""The four defects found in the 2026-09-16 cpx_barometer trace.

That run took 441 seconds, of which 117 was the model and 324 was pytest, and
it ended by reporting a step as failed that had in fact succeeded. Each test
here pins one of the specific behaviours that produced it:

1. `_subsumed_by_suite` / deferred step verifies — the gate ran each step's
   own pytest selection and then the whole suite that contains it, twice over,
   for 46 of those seconds.
2. `_rescued_by_verify` — a step that ran out of iterations was declared
   failed without its verify command ever being consulted, and that command
   passed on the first try ninety seconds later at the gate.
3. `_salvage` preferring a tool-free turn — the step's reported result was a
   mid-sentence narration of an edit it was about to make.
4. `is_assent` / `pending_suggestion` — the goal "yes" produced three
   clarifying questions, one of which offered, as a clickable option, the
   suggestion the harness had just read out of its own memory file.
"""

from __future__ import annotations

import contextlib
import time

import pytest
import requests

from agent.phases.clarify import is_assent
from agent.phases.finalcheck import _subsumed_by_suite

# -- 1. a step verify a green suite already proves -------------------------


def test_the_two_commands_from_the_traced_run_are_subsumed():
    """Both step verifies in the traced run were pytest selections inside the
    same suite that ran moments later, at 22s each, twice over."""
    assert _subsumed_by_suite("uv run pytest test_eval_and_compare.py -v -k permutation_importance")
    assert _subsumed_by_suite("uv run pytest test_eval_and_compare.py::TestMain -v")


def test_plain_and_module_form_pytest_are_subsumed():
    assert _subsumed_by_suite("pytest tests/test_x.py")
    assert _subsumed_by_suite("python -m pytest tests/test_x.py::test_y")
    assert _subsumed_by_suite("uv run python -m pytest -k slow")


def test_a_non_pytest_verify_is_never_subsumed():
    """A prose or lint check has no relationship to the test suite at all —
    this is what keeps the pandoc/ruff verify commands running."""
    assert not _subsumed_by_suite("pandoc -f gfm -t plain -o /dev/null README.md")
    assert not _subsumed_by_suite("uv run ruff check eval_and_compare.py")
    assert not _subsumed_by_suite("./scripts/check.sh")
    assert not _subsumed_by_suite("")


def test_pytest_asking_a_different_question_is_not_subsumed():
    """Changing plugins, rootdir, config or collection means the suite's green
    result does not cover this command, so it must still run."""
    assert not _subsumed_by_suite("pytest -p no:cacheprovider tests/")
    assert not _subsumed_by_suite("pytest --rootdir=/elsewhere tests/")
    assert not _subsumed_by_suite("pytest --doctest-modules src/")
    assert not _subsumed_by_suite("pytest -c other.ini tests/")
    # An inline env assignment runs under settings the suite did not use.
    assert not _subsumed_by_suite("PYTHONPATH=src pytest tests/")


def test_an_unparseable_command_is_not_subsumed():
    """shlex raises on an unbalanced quote; the safe answer is "run it"."""
    assert not _subsumed_by_suite('pytest "unclosed')


# -- 4. "yes" is an assent, "yes but..." is not ----------------------------


def test_the_traced_goal_is_recognised_as_an_assent():
    assert is_assent("yes")
    assert is_assent("Yes")
    assert is_assent("  yes.  ")
    assert is_assent("yes please")
    assert is_assent("go ahead")
    assert is_assent("sure")


def test_an_assent_carrying_its_own_content_is_not_one():
    """ "yes" and "yes, but use a dict" say opposite things about whether the
    previous suggestion is still the plan, and only the first can be replaced
    by it."""
    assert not is_assent("yes, but use a dict instead")
    assert not is_assent("yes lets address those")
    assert not is_assent("no")
    assert not is_assent("implement the k-fold cross validation")
    assert not is_assent("")


# -- a single chat() call is bounded including its retries ----------------


def test_retries_stop_at_the_overall_deadline(monkeypatch):
    """Seven attempts at 180s plus six backoffs of up to 90s is 1,800 seconds
    for one chat() call. A traced benchmark run spent 1,954s between two tool
    calls with nothing in the trace to show for it, then reported the call's
    latency as 1.17s, because only the final successful attempt is timed.
    max_call_s bounds the whole sequence."""
    import agent.llm as llm

    slept: list[float] = []
    now = [1000.0]
    monkeypatch.setattr(llm.time, "time", lambda: now[0])
    monkeypatch.setattr(
        llm.time, "sleep", lambda s: (slept.append(s), now.__setitem__(0, now[0] + s))[0]
    )

    client = llm.OpenRouterClient("k", max_retries=6, max_call_s=30.0)
    attempts = [0]

    def always_failing(_budget):
        attempts[0] += 1
        raise llm._Retryable("rate limited")

    with pytest.raises(llm.LLMError):
        client._request(always_failing)
    assert sum(slept) <= 30.0, f"slept {sum(slept)}s against a 30s budget"
    assert attempts[0] < 7, "it should give up on the deadline, not exhaust every attempt"


def test_a_call_that_succeeds_first_time_never_sleeps(monkeypatch):
    """The deadline must not cost anything on the ordinary path."""
    import agent.llm as llm

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client = llm.OpenRouterClient("k")
    sentinel = llm.Reply(content="done")
    assert client._request(lambda _budget: sentinel) is sentinel
    assert slept == []
    assert sentinel.attempts == 1, "a first-try reply is stamped as one attempt"
    assert sentinel.retry_error == ""


def test_retry_after_is_still_honoured_within_the_budget(monkeypatch):
    """A provider's own Retry-After is the best information available, and
    bounding the total must not throw it away when there is room for it."""
    import agent.llm as llm

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client = llm.OpenRouterClient("k", max_call_s=600.0)
    assert client._sleep(0, "45", deadline=llm.time.time() + 600.0) is True
    assert slept == [45.0]


def test_a_backoff_longer_than_the_time_left_is_not_taken(monkeypatch):
    """Sleeping past the budget buys an attempt that can never be made."""
    import agent.llm as llm

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client = llm.OpenRouterClient("k")
    assert client._sleep(0, "90", deadline=llm.time.time() + 5.0) is False
    assert slept == [], "no point waiting 90s for a deadline 5s away"


def test_a_retried_call_is_visible_in_the_trace(tmp_path):
    """The reason the stalls took a sweep and a three-way trace comparison to
    diagnose: a call that spent 1,954s being retried recorded itself as
    latency 1.17s, because only the successful attempt is timed. Nothing in
    the trace distinguished it from a fast call preceded by a mystery gap."""
    import json

    from agent.llm import Reply, Usage
    from agent.trace import Trace

    trace = Trace(str(tmp_path / "t"), echo=False)
    reply = Reply(
        content="ok",
        usage=Usage(10, 5, 0.0, 1.17),
        attempts=4,
        total_s=903.2,
        retry_error="_Retryable: rate limited",
    )
    trace.llm_call("review:1", "m", reply.usage, "stop", reply=reply)
    trace.close()

    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    call = next(e for e in events if e["kind"] == "llm_call")
    assert call["attempts"] == 4
    assert call["total_s"] == 903.2
    retried = next(e for e in events if e["kind"] == "llm_retried")
    assert retried["waited_s"] > 900, "the time lost to backoff is the whole point"
    assert "rate limited" in retried["error"]


def test_an_ordinary_call_records_no_retry_event(tmp_path):
    """One event per call on the normal path; llm_retried is a real signal,
    not a line on every call."""
    import json

    from agent.llm import Reply, Usage
    from agent.trace import Trace

    trace = Trace(str(tmp_path / "t"), echo=False)
    reply = Reply(content="ok", usage=Usage(10, 5, 0.0, 1.2), attempts=1, total_s=1.2)
    trace.llm_call("plan", "m", reply.usage, "stop", reply=reply)
    trace.close()

    kinds = [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]
    assert "llm_retried" not in kinds


# -- a streamed call reuses its connection --------------------------------


def _sse_server(chunked: bool):
    """An OpenRouter-shaped SSE endpoint that counts TCP connections."""
    import http.server
    import json as _json
    import socketserver
    import threading

    seen: list[int] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            seen.append(id(self.connection))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            done = {
                "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            body = [f"data: {_json.dumps(done)}\n\n", "data: [DONE]\n\n"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in body:
                    b = part.encode()
                    self.wfile.write(b"%X\r\n" % len(b) + b + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                payload = "".join(body).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1], seen


def test_a_chunked_stream_reuses_one_connection():
    """urllib3 returns a connection to its pool only once the body is read to
    the end. Returning at `[DONE]` left a chunked response's terminal chunk
    unread, so resp.close() discarded -- and reset -- the socket, and the next
    call paid a fresh TCP and TLS handshake. Measured before the fix: five
    sequential calls opened four connections. Against a median TTFT that is
    78% of all model latency, that handshake is not noise."""
    from agent.llm import OpenRouterClient

    srv, port, seen = _sse_server(chunked=True)
    try:
        client = OpenRouterClient("k", base_url=f"http://127.0.0.1:{port}", stream=True)
        for _ in range(5):
            client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert len(set(seen)) == 1, f"5 calls opened {len(set(seen))} connections"


def test_draining_is_bounded_and_cannot_hang(monkeypatch):
    """A provider that never closes must not turn the drain into its own
    stall -- it is capped by both a line count and the stream's deadline."""
    import agent.llm as llm

    pulled = [0]

    def endless():
        while True:
            pulled[0] += 1
            yield b""

    class _NoSocket:
        raw = None

    llm._drain(endless(), llm.time.time() + 60, _NoSocket())
    assert pulled[0] <= llm._DRAIN_LINES

    pulled[0] = 0
    llm._drain(endless(), llm.time.time() - 1, _NoSocket())
    assert pulled[0] == 0, "an expired deadline drains nothing"


# -- what the code review caught -------------------------------------------


def test_a_step_that_never_reported_at_all_is_not_rescued(tmp_path):
    """`_rescued_by_verify` is for the ran-out-of-iterations case only.
    `_apply_result` also reports a problem when the executor produced no
    terminal call and nothing salvageable -- there is then no evidence it did
    anything. A step whose verify command was already green before the run
    ("add a case to test_x.py", where test_x.py exists and passes) would
    otherwise be cleared on the strength of a test it never touched, marked
    verified, and returned before the retry that would have done the work."""
    from agent.config import Settings
    from agent.phases.execute import _rescued_by_verify
    from agent.phases.plan import Step

    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest test_x.py")
    settings = Settings(workspace=str(tmp_path), goal="g")
    # result=None is "never called finish_step", not "ran out of iterations".
    assert _rescued_by_verify(None, None, step, settings, None) is False
    assert _rescued_by_verify(None, None, step, settings, {"summary": "x"}) is False


def test_a_shell_chain_is_never_settled_by_the_suite():
    """run_verify uses shell=True and shlex.split keeps `&&` as an ordinary
    token, so `pytest x && ruff check .` matched the pytest head and was
    declared settled by a green suite that never ran ruff."""
    assert not _subsumed_by_suite("uv run pytest test_x.py && uv run ruff check .")
    assert not _subsumed_by_suite("pytest test_x.py || echo failed")


def test_a_path_outside_the_suite_s_root_is_not_settled():
    """The suite runs `pytest .` from the workspace root, so a verify naming
    a sibling directory, an absolute path, or a directory pytest's own
    norecursedirs skips is not a subset of what it collected."""
    assert not _subsumed_by_suite("pytest ../sibling/test_x.py")
    assert not _subsumed_by_suite("pytest /elsewhere/test_x.py")
    assert not _subsumed_by_suite("pytest build/test_x.py")


def test_python_dash_m_pytest_is_still_subsumed():
    """`python -m pytest` spells the interpreter's module flag `-m`, which is
    also pytest's marker flag; reading it as the latter rejected every
    `python -m pytest` command outright."""
    assert _subsumed_by_suite("python -m pytest tests/test_x.py::test_y")
    assert _subsumed_by_suite("uv run python -m pytest -k slow")


def test_a_replayable_call_still_hits_the_hard_repeat_limit(ctx_factory):
    """The replay cache used to answer before the hard-limit refusal, making
    it unreachable for run_tests -- 434 of 766 traced repeats -- so a stuck
    model got unlimited cheap replays and spun to max_iters instead of being
    forced onto a different path."""
    import json as _json

    from agent.llm import Reply, Usage
    from agent.runner import REPEAT_HARD_LIMIT, run_tool_loop
    from agent.tools import execution_registry

    ctx = ctx_factory()
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    same = {
        "id": "c1",
        "function": {"name": "run_tests", "arguments": _json.dumps({"path": "test_x.py"})},
    }

    class Client:
        calls = 0

        def chat(self, **kw):
            Client.calls += 1
            return Reply(content="", tool_calls=[same], usage=Usage(1, 1, 0.0, 0.01))

    _, messages = run_tool_loop(
        client=Client(),
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=REPEAT_HARD_LIMIT + 2,
    )
    refusals = [
        m
        for m in messages
        if m.get("role") == "tool" and "refusing to run" in str(m.get("content", ""))
    ]
    assert refusals, "the hard limit must still fire for a replayable tool"


@pytest.fixture
def ctx_factory(tmp_path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    def make():
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        return ToolContext(
            workspace=Workspace(ws),
            approver=Approver(auto_approve=True),
            trace=Trace(str(tmp_path / "t"), echo=False),
        )

    return make


# -- redundancy against an unstable link ----------------------------------


def test_socket_level_errors_are_told_apart_from_provider_errors():
    """A dropped connection and a rate limit both fail a request, and they
    want opposite responses: one needs a new connection, the other needs a
    wait. ReadTimeout is deliberately on the "wait" side -- the server took
    the request and went quiet, so the socket is probably fine and an
    instant retry would hammer something already struggling."""
    import agent.llm as llm

    assert llm._is_socket_level(requests.ConnectionError("reset"))
    assert llm._is_socket_level(requests.exceptions.ChunkedEncodingError("cut"))
    assert not llm._is_socket_level(requests.exceptions.ReadTimeout("quiet"))
    assert not llm._is_socket_level(llm._Retryable("HTTP 429"))


def test_a_dropped_connection_retries_at_once_and_clears_the_pool(monkeypatch):
    """The pooled connection went dead while idle. Waiting does not repair
    that -- only a new connection does -- and the pool must be cleared first
    or the retry can draw a second dead socket out of it."""
    import agent.llm as llm

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client = llm.OpenRouterClient("k")
    cleared: list[int] = []
    monkeypatch.setattr(client, "_reset_pool", lambda: cleared.append(1))
    reconnects: list[int] = []
    client.on_reconnect = lambda attempt, error: reconnects.append(attempt)

    calls = [0]

    def flaky(_budget):
        calls[0] += 1
        if calls[0] == 1:
            raise requests.ConnectionError("connection reset by peer")
        return llm.Reply(content="recovered")

    assert client._request(flaky).content == "recovered"
    assert slept == [], "a dead socket is not repaired by waiting"
    assert cleared == [1], "the pool must be cleared before reconnecting"
    assert reconnects == [1]


def test_instant_reconnects_are_capped_then_fall_back_to_backoff(monkeypatch):
    """A host that is genuinely unreachable must not be spun on. Past the
    cap the ordinary exponential backoff takes over."""
    import agent.llm as llm

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client = llm.OpenRouterClient("k", max_retries=6)
    monkeypatch.setattr(client, "_reset_pool", lambda: None)

    def always_down(_budget):
        raise requests.ConnectionError("network is unreachable")

    with pytest.raises(llm.LLMError):
        client._request(always_down)
    assert len(slept) > 0, "past the reconnect cap it must back off, not spin"


def test_the_streamed_request_waits_for_headers_not_a_whole_generation():
    """With stream=True, post() returns once headers arrive, so its read
    timeout is the wait for the provider's first word. Giving it the full
    180s is why a half-open connection cost seven attempts of 180s apiece:
    the read that timed out was that one, before any check of ours could
    run. TTFT across 10,390 traced calls: median 1.14s, p90 5.46s."""
    import agent.llm as llm

    client = llm.OpenRouterClient("k", timeout=180.0)
    connect, read = client._first_byte_timeout(None)
    assert read == llm.FIRST_BYTE_TIMEOUT_S
    assert read < client.timeout
    assert connect == llm.CONNECT_TIMEOUT_S
    # Never past what is left of the call's own budget.
    _, tight = client._first_byte_timeout(5.0)
    assert tight <= 5.0


def test_a_diffusion_model_gets_a_wider_first_byte_bound():
    """Mercury/Inception spends most of a call's time on prefill and
    parallel refinement before committing to any output -- measured across
    84 traced `inception/` calls: median TTFT 15.2s, p90 25.0s, p99 51.1s,
    already past the all-model FIRST_BYTE_TIMEOUT_S of 45s. Without this, a
    perfectly healthy slow prefill at the tail gets misread as a half-open
    connection and retried."""
    import agent.llm as llm

    client = llm.OpenRouterClient("k", timeout=180.0)
    _, fast_model_read = client._first_byte_timeout(None, "openai/gpt-4o")
    _, diffusion_read = client._first_byte_timeout(None, "inception/mercury-2.5")
    assert fast_model_read == llm.FIRST_BYTE_TIMEOUT_S
    assert diffusion_read == llm.DIFFUSION_FIRST_BYTE_TIMEOUT_S
    assert diffusion_read > llm.FIRST_BYTE_TIMEOUT_S
    # Still bounded by what's left of the call's own budget either way.
    _, tight = client._first_byte_timeout(5.0, "inception/mercury-2.5")
    assert tight <= 5.0


def test_connections_carry_tcp_keepalive():
    """The characteristic failure of an unstable link is silence, not a
    reset, and the OS default only starts probing after two hours."""
    import socket as _socket

    import agent.llm as llm

    options = dict(((level, opt), value) for level, opt, value in llm._KEEPALIVE_OPTIONS)
    assert options[(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE)] == 1
    assert len(llm._KEEPALIVE_OPTIONS) > 1, "keepalive without tuning probes after hours"


def test_the_client_releases_its_pool_and_is_a_context_manager():
    """Every prompt in a TUI session builds a fresh client, and nothing
    closed them -- one set of idle sockets per prompt, never reused."""
    import agent.llm as llm

    closed: list[int] = []
    with llm.OpenRouterClient("k") as client:
        client.session.close = lambda: closed.append(1)
    assert closed == [1]
    client.close()  # idempotent
    assert closed == [1, 1]


# -- end to end against the two shapes an unstable link actually takes ----


def _flaky_server(mode: str, drop_first: int):
    """A server whose first `drop_first` connections fail, then works.

    `mode="reset"` sends a TCP reset; `mode="silence"` accepts and never
    answers, which is the half-open connection a WiFi drop or NAT timeout
    leaves behind. The two need opposite handling and are easy to confuse:
    an earlier version of this probe closed `self.connection` -- a buffered
    wrapper -- rather than `self.request`, so no reset was ever sent and the
    reset case masqueraded as silence, which made the reconnect path look
    broken when it was the test that was wrong.
    """
    import http.server
    import json as _json
    import socket as _socket
    import socketserver
    import struct
    import threading
    import time as _time

    state = {"n": 0}
    done = {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
    body = [f"data: {_json.dumps(done)}\n\n", "data: [DONE]\n\n"]

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            state["n"] += 1
            if state["n"] <= drop_first:
                if mode == "reset":
                    # SO_LINGER 0 on the *raw* socket makes close() a reset.
                    self.request.setsockopt(
                        _socket.SOL_SOCKET, _socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    self.request.close()
                    self.close_connection = True
                else:
                    _time.sleep(300)
                return
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in body:
                raw = part.encode()
                self.wfile.write(b"%X\r\n" % len(raw) + raw + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_a_link_that_resets_recovers_without_waiting():
    """A reset says the connection is dead, not that the provider is busy.
    Reconnecting costs nothing and waiting buys nothing, so two dropped
    connections should cost approximately zero rather than two backoffs."""
    from agent.llm import OpenRouterClient

    srv, port = _flaky_server("reset", drop_first=2)
    reconnects: list[int] = []
    try:
        client = OpenRouterClient(
            "k",
            base_url=f"http://127.0.0.1:{port}",
            stream=True,
            max_call_s=60.0,
            on_reconnect=lambda a, e: reconnects.append(a),
        )
        started = time.monotonic()
        reply = client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        client.close()
    finally:
        srv.shutdown()
    assert reply.content == "hi"
    assert reply.attempts == 3, "two dead connections, then a live one"
    assert reconnects == [1, 2], "each drop should reconnect rather than back off"
    assert elapsed < 5.0, f"a reset should not cost a backoff; took {elapsed:.1f}s"


@pytest.mark.slow
def test_a_link_that_goes_silent_recovers_at_the_first_byte_bound(monkeypatch):
    """The half-open case: the peer accepts and never speaks again. Nothing
    below the read timeout notices, so the whole question is how long we wait
    for a first byte. At the old 180s that was seven attempts of three
    minutes apiece and a whole call budget spent discovering a connection
    that had been dead throughout.

    `timeout` is squeezed here so the same arithmetic runs in a fraction of a
    second: the first-byte bound is min(FIRST_BYTE_TIMEOUT_S, this attempt's
    timeout), so a 0.6s timeout makes each silent attempt cost ~0.6s instead
    of 45 -- the socket wait is real, not mocked, so this is the only way to
    keep the test both fast and honest about what it exercises. Not pushed
    any lower than this: at 0.3s the test became flaky under a loaded
    machine (a full suite run, other tests contending for the scheduler),
    timing out before the connection accept/drop cycle could complete.

    `slow`, same reasoning as test_bench.py's kill-mechanism test: under
    `pytest -n auto`, several workers' worth of real socket/subprocess work
    contends for the same CPU at once, which is a much heavier version of
    the "loaded machine" scenario above -- measured actually flaking there
    (576s instead of under a second) even though this test alone, or the
    plain sequential suite, is reliably fast. Excluded from the fast
    parallel loop (`-m "not slow"`), included in the plain default run.
    """
    import agent.llm as llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # skip the backoffs
    srv, port = _flaky_server("silence", drop_first=2)
    try:
        client = llm.OpenRouterClient(
            "k", base_url=f"http://127.0.0.1:{port}", stream=True, timeout=0.6, max_call_s=30.0
        )
        started = time.monotonic()
        reply = client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        client.close()
    finally:
        srv.shutdown()
    assert reply.content == "hi"
    assert reply.attempts == 3, "two silent connections, then a live one"
    # Two attempts bounded by the first-byte timeout, not by `timeout` alone
    # and certainly not by the call budget.
    assert elapsed < 4.0, f"silent connections should cost ~0.6s each; took {elapsed:.1f}s"


def test_a_keepalive_trickle_cannot_hold_a_stream_past_its_budget():
    """The stall that started all of this, finally reproduced.

    OpenRouter sends `: OPENROUTER PROCESSING` comments to hold a slow
    generation open. `_iter_sse_chunks` checks its deadline when a line
    *arrives* — and `iter_lines` buffers 512 bytes before yielding one, so a
    provider trickling 25-byte comments has to send twenty before the check
    can run at all. Its docstring claimed the deadline was "the actual
    backstop ... keep-alives or not"; it was not, because nothing inside the
    read loop can bound a read that is waiting to fill a buffer.

    Measured against a server that sends keepalives and never any content: a
    60s budget took 173s to give up before the watchdog, and 60s after.
    """
    import http.server
    import socketserver
    import threading
    import time as _time

    import agent.llm as llm

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            with contextlib.suppress(Exception):
                while True:
                    part = b": OPENROUTER PROCESSING\n\n"
                    self.wfile.write(b"%X\r\n" % len(part) + part + b"\r\n")
                    self.wfile.flush()
                    _time.sleep(0.02)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = llm.OpenRouterClient(
            "k",
            base_url=f"http://127.0.0.1:{srv.server_address[1]}",
            stream=True,
            timeout=180.0,
            max_call_s=0.5,
            max_retries=0,
        )
        started = time.monotonic()
        with pytest.raises(llm.LLMError):
            client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        client.close()
    finally:
        srv.shutdown()
    # Without the watchdog this ran to the 512-byte buffer's pace, many times
    # the budget. The bound is the budget, not the buffer -- the exact ratio
    # doesn't matter, only that it stays close to max_call_s rather than to
    # `timeout` (180s) or the keepalive interval times the buffer size.
    assert elapsed < 3.0, f"budget was 0.5s; took {elapsed:.1f}s"


# -- cache_prompts says when it cannot do anything ------------------------


def _cache_events(tmp_path, **overrides):
    import json

    from agent.config import Settings
    from agent.loop import _note_inert_cache_flag
    from agent.trace import Trace

    trace = Trace(str(tmp_path), echo=False)
    _note_inert_cache_flag(trace, Settings(workspace=".", goal="g", **overrides))
    trace.close()
    return [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]


def test_the_cache_flag_reports_itself_inert_on_the_default_models(tmp_path):
    """`cache_prompts` sends an explicit cache breakpoint, and only Anthropic
    models accept one -- a provider that does not understand the
    array-of-blocks content form rejects the request outright, which is why
    it cannot be sent blindly. On the all-Mercury defaults the flag therefore
    does nothing in either position, and someone passing --no-cache to cut
    cost deserves to know that rather than conclude caching does not help."""
    assert "cache_control_inert" in _cache_events(tmp_path)


def test_the_cache_flag_stays_quiet_when_it_is_live(tmp_path):
    """One Anthropic model in the run is enough for the breakpoint to matter."""
    assert "cache_control_inert" not in _cache_events(
        tmp_path, planner_model="anthropic/claude-sonnet-5"
    )


def test_turning_the_cache_flag_off_says_nothing(tmp_path):
    """The note is about a flag that is on and powerless, not about a flag
    somebody deliberately switched off."""
    assert "cache_control_inert" not in _cache_events(tmp_path, cache_prompts=False)
