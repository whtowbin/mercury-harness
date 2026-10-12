"""The stream watchdog against a real socket, not a mocked post().

The existing keep-alive test patches `post()`, so it never exercised the one
thing that bounds a stalled stream: shutting the real socket down from another
thread. Under urllib3 2.x that shutdown targeted a socket that was None, and a
live sweep lost 929s per stalled call against a 180s deadline.
"""

import http.server
import threading
import time

import pytest

from agent import llm


@pytest.fixture
def keepalive_server():
    """Headers at once, then only SSE keep-alive comments, slowly, for 30s."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(60):
                    self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                    self.wfile.flush()
                    time.sleep(0.5)
            except OSError:
                pass

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_the_watchdog_can_reach_the_socket_of_a_streamed_response(keepalive_server):
    import requests

    resp = requests.post(f"{keepalive_server}/x", json={}, stream=True, timeout=(5, 5))
    try:
        assert llm._response_sockets(resp), "no socket found: the watchdog would be a no-op"
    finally:
        resp.close()


def test_a_keepalive_only_stream_is_cut_off_at_its_deadline(keepalive_server):
    client = llm.OpenRouterClient(
        api_key="x", base_url=keepalive_server, timeout=1.0, max_call_s=1.5, stream=True
    )
    started = time.time()
    with pytest.raises(llm.LLMError, match="deadline"):
        client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    # Before the fix this ran until 20 keep-alives had filled iter_lines' buffer
    # (~10s here, ~15 min against a real provider). Generous for CI contention.
    assert time.time() - started < 5.0


@pytest.fixture
def late_content_server():
    """Keep-alives for about a second, then real content and a finished reply."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(3):
                self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                self.wfile.flush()
                time.sleep(0.3)
            chunk = '{"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}'
            self.wfile.write(f"data: {chunk}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_a_mercury_stream_of_only_keepalives_is_cut_at_the_content_deadline(
    keepalive_server, monkeypatch
):
    """Keep-alives satisfy the first-byte check, so before the content bound an
    attempt ran to the provider's own 504 at ~120s (DESIGN-LOG §69)."""
    monkeypatch.setattr(llm, "DIFFUSION_FIRST_CONTENT_TIMEOUT_S", 0.5)
    client = llm.OpenRouterClient(
        api_key="x", base_url=keepalive_server, timeout=20.0, max_call_s=2.0, stream=True
    )
    started = time.time()
    with pytest.raises(llm.LLMError, match="no content within"):
        client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert time.time() - started < 6.0  # against a 20s attempt deadline


def test_content_after_a_few_keepalives_is_not_cut(late_content_server, monkeypatch):
    monkeypatch.setattr(llm, "DIFFUSION_FIRST_CONTENT_TIMEOUT_S", 3.0)
    client = llm.OpenRouterClient(
        api_key="x", base_url=late_content_server, timeout=20.0, max_call_s=20.0, stream=True
    )
    reply = client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert reply.content == "hello"


def test_the_content_bound_is_mercury_only():
    assert llm._first_content_timeout_for("inception/mercury-2.5") == 30.0
    assert llm._first_content_timeout_for("z-ai/glm-5.3-flash") is None


def _content_then_silence(silence_s: float):
    """One content delta at once, then only keep-alives for `silence_s`, then a tool
    call and a finished reply: the shape of a Mercury runaway (DESIGN-LOG §95).

    HTTP/1.1 with chunked transfer encoding, as real providers stream: urllib3 hands
    each chunk over as it arrives. A plain HTTP/1.0 body would sit in iter_lines'
    512-byte read until enough keep-alives arrived, which delays the first delta and
    no real provider does."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                self._chunk(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
                end = time.time() + silence_s
                while time.time() < end:
                    self._chunk(b": OPENROUTER PROCESSING\n\n")
                    time.sleep(0.1)
                call = (
                    '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
                    '"function":{"name":"finish_step","arguments":"{}"}}]},'
                    '"finish_reason":"tool_calls"}]}'
                )
                self._chunk(f"data: {call}\n\ndata: [DONE]\n\n".encode())
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client cut the stream, which is what the bound is for

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def short_silence_server():
    server = _content_then_silence(1.0)
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def long_silence_server():
    server = _content_then_silence(30.0)
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_the_longest_silence_after_first_content_is_recorded(short_silence_server):
    client = llm.OpenRouterClient(
        api_key="x", base_url=short_silence_server, timeout=20.0, max_call_s=20.0, stream=True
    )
    reply = client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert reply.tool_calls and reply.tool_calls[0]["function"]["name"] == "finish_step"
    assert 0.8 <= reply.usage.max_gap_s < 5.0
    assert reply.usage.ttft_s < 0.8  # the first delta came at once; the gap is after it


def test_a_mercury_stream_silent_after_its_first_delta_is_cut_and_says_where(
    long_silence_server,
):
    failures: list[str] = []
    client = llm.OpenRouterClient(
        api_key="x",
        base_url=long_silence_server,
        timeout=20.0,
        max_call_s=2.5,
        stream=True,
        content_stall_s=0.5,
        on_attempt_failed=lambda attempt, seconds, error: failures.append(error),
    )
    started = time.time()
    with pytest.raises(llm.LLMError, match="silent for"):
        client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert time.time() - started < 8.0  # against 30s of silence and a 20s read timeout
    assert failures and "silent for" in failures[0]
    assert "[first content at " in failures[0] and "then silent" in failures[0]


def test_a_short_silence_under_the_bound_is_not_cut(short_silence_server):
    client = llm.OpenRouterClient(
        api_key="x",
        base_url=short_silence_server,
        timeout=20.0,
        max_call_s=20.0,
        stream=True,
        content_stall_s=3.0,
    )
    reply = client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert reply.tool_calls


def test_the_stall_bound_is_mercury_only(short_silence_server):
    client = llm.OpenRouterClient(
        api_key="x",
        base_url=short_silence_server,
        timeout=20.0,
        max_call_s=20.0,
        stream=True,
        content_stall_s=0.3,
    )
    reply = client.chat(model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}])
    assert reply.tool_calls  # a 1s silence, three times the bound, and not cut


def test_a_cut_before_any_content_says_no_content(keepalive_server, monkeypatch):
    monkeypatch.setattr(llm, "DIFFUSION_FIRST_CONTENT_TIMEOUT_S", 0.5)
    failures: list[str] = []
    client = llm.OpenRouterClient(
        api_key="x",
        base_url=keepalive_server,
        timeout=20.0,
        max_call_s=1.5,
        stream=True,
        on_attempt_failed=lambda attempt, seconds, error: failures.append(error),
    )
    with pytest.raises(llm.LLMError):
        client.chat(model="inception/mercury", messages=[{"role": "user", "content": "hi"}])
    assert failures and failures[0].endswith("[no content]")
