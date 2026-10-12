"""A dead search backend is seen, traced and refused before a sweep (DESIGN-LOG §89.1).

The local SearXNG answered 200 with zero results for two days because every upstream
engine was blocking this machine's IP; the traces said only `results: 0`. These tests
stand up a real local HTTP server speaking SearXNG's JSON, not a mocked urlopen.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent.approval import Approver
from agent.tools import ToolContext, web
from agent.trace import Trace
from agent.workspace import Workspace
from bench import cli, diagnostics

BLOCKED = {
    "results": [],
    "unresponsive_engines": [
        ["brave", "Suspended: too many requests"],
        ["duckduckgo", "CAPTCHA"],
    ],
}
WORKING = {
    "results": [{"title": "Python", "url": "https://www.python.org/", "content": "The language"}],
    "unresponsive_engines": [],
}


@pytest.fixture
def searxng(monkeypatch):
    """A local server answering /search with `state["body"]`; returns that state dict."""
    state: dict = {"body": BLOCKED, "queries": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            state["queries"].append(self.path)
            data = json.dumps(state["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()


def test_a_blocked_backend_fails_the_canary_and_names_the_engines(searxng):
    ok, detail = web.search_health()
    assert ok is False
    assert "brave" in detail and "duckduckgo" in detail
    assert len(searxng["queries"]) == len(web._CANARY_QUERIES)  # tried each canary


def test_a_working_backend_passes_after_one_query(searxng):
    searxng["body"] = WORKING
    assert web.search_health() == (True, "searxng")
    assert len(searxng["queries"]) == 1


def test_no_backend_is_neither_healthy_nor_failed(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    ok, detail = web.search_health()
    assert ok is None and "SEARXNG_URL" in detail


def test_the_trace_records_which_engines_failed(searxng, tmp_path):
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    web.web_search(ctx, "anything")
    with open(ctx.trace.path) as f:
        events = [json.loads(line) for line in f]
    [search] = [e for e in events if e["kind"] == "web_search"]
    assert search["results"] == 0
    assert search["failed_engines"] == ["brave", "duckduckgo"]


def test_diagnostics_count_searches_and_empty_ones():
    events = [
        {"kind": "web_search", "results": 0, "failed_engines": ["brave"]},
        {"kind": "web_search", "results": 4, "failed_engines": []},
        {"kind": "web_search", "results": 0},
    ]
    d = diagnostics.from_events(events)
    assert (d.web_searches, d.empty_searches) == (3, 2)


def test_a_sweep_refuses_to_start_on_dead_search_unless_allowed(searxng, capsys):
    assert cli._search_ready(allow_dead=False) is False
    assert "--allow-dead-search" in capsys.readouterr().err
    assert cli._search_ready(allow_dead=True) is True
    searxng["body"] = WORKING
    assert cli._search_ready(allow_dead=False) is True


def test_run_and_scout_both_accept_the_opt_out():
    parser = cli.build_parser()
    for command in (["run"], ["scout", "--configs", "mercury-only"]):
        assert parser.parse_args([*command, "--allow-dead-search"]).allow_dead_search is True
        assert parser.parse_args(command).allow_dead_search is False
