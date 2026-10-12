"""A 402 for OpenRouter's in-flight cap is retried; any other 402 is final.

Against a real local server, not a mocked post (the user's rule for I/O). The body is the
shape OpenRouter sent when two SciCode arms died at a $0.09 balance (DESIGN-LOG §81.1).
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import llm

IN_FLIGHT = {
    "error": {
        "message": "Your in-flight budget is exhausted. Add credits.",
        "code": 402,
        "metadata": {
            "reason": "in_flight_budget_exhausted",
            "limit_source": "openrouter_in_flight_budget",
            "headers": {"Retry-After": "1"},
        },
    }
}
NO_CREDIT = {"error": {"message": "Insufficient credits.", "code": 402}}
OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def _serve(replies: list[tuple[int, dict, dict]]):
    """A local server answering POSTs from `replies` in order; returns (base url, calls, stop)."""
    calls: list[int] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, body, headers = replies[min(len(calls), len(replies) - 1)]
            calls.append(status)
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", calls, server.shutdown


def _chat(base: str) -> str:
    client = llm.OpenRouterClient(api_key="x", base_url=base, stream=False, max_retries=2)
    return client.chat(model="m", messages=[{"role": "user", "content": "hi"}]).content


def test_the_in_flight_cap_is_waited_out_and_the_call_succeeds():
    base, calls, stop = _serve([(402, IN_FLIGHT, {"Retry-After": "1"}), (200, OK, {})])
    try:
        assert _chat(base) == "ok"
    finally:
        stop()
    assert calls == [402, 200]


def test_the_retry_after_inside_the_body_is_used_when_the_header_is_missing():
    base, calls, stop = _serve([(402, IN_FLIGHT, {}), (200, OK, {})])
    try:
        assert _chat(base) == "ok"
    finally:
        stop()
    assert calls == [402, 200]


def test_a_402_for_no_credit_is_final_and_not_retried():
    base, calls, stop = _serve([(402, NO_CREDIT, {}), (200, OK, {})])
    try:
        with pytest.raises(llm.LLMError, match="402"):
            _chat(base)
    finally:
        stop()
    assert calls == [402]


def test_the_in_flight_cap_inside_a_200_error_envelope_is_retryable_too():
    assert isinstance(llm._envelope_error(IN_FLIGHT["error"]), llm._Retryable)
    assert llm._envelope_error(IN_FLIGHT["error"]).retry_after == "1"
    assert not isinstance(llm._envelope_error(NO_CREDIT["error"]), llm._Retryable)
