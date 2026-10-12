"""The final GAIA pass searches with Brave, chosen explicitly, with the key kept out of sight
(DESIGN-LOG §115). The canary measures search quality, not liveness (§112.2).

The canary is tested end to end: a real local HTTP server speaking SearXNG's JSON, and the
harness's own interpreter run as a real subprocess, as the driver runs it.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path

import pytest

from agent.tools import web

_spec = importlib.util.spec_from_file_location(
    "gaia", Path(__file__).resolve().parents[1] / "scripts" / "gaia" / "gaia.py"
)
assert _spec and _spec.loader
gaia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaia)

KEY = "brv-test-key-0123456789"


@pytest.fixture
def clean(monkeypatch):
    for name in ("SEARCH_BACKEND", "SEARXNG_URL", "BRAVE_API_KEY", "GAIA_SEARCH"):
        monkeypatch.delenv(name, raising=False)


def test_an_explicit_backend_beats_a_searxng_url(clean, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    assert web.search_backend() == "searxng"
    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    assert web.search_backend() == "brave"
    monkeypatch.setenv("SEARCH_BACKEND", "searxng")
    monkeypatch.setenv("BRAVE_API_KEY", KEY)
    assert web.search_backend() == "searxng"


def test_without_the_switch_the_order_is_unchanged(clean, monkeypatch):
    assert web.search_backend() is None
    monkeypatch.setenv("BRAVE_API_KEY", KEY)
    assert web.search_backend() == "brave"


def test_the_keychain_is_read_only_when_brave_is_named(clean, monkeypatch):
    from agent import config

    def keychain(service, account):
        raise AssertionError("the Keychain was consulted without SEARCH_BACKEND=brave")

    monkeypatch.setattr(config, "_get_keychain_secret", keychain)
    assert web._brave_key() == ""
    monkeypatch.setattr(config, "_get_keychain_secret", lambda service, account: KEY)
    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    assert web._brave_key() == KEY


def test_brave_named_without_a_key_fails_loudly(clean, monkeypatch):
    from agent import config

    monkeypatch.setattr(config, "_get_keychain_secret", lambda service, account: None)
    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    found = web._brave("anything", 3)
    assert isinstance(found, str) and found.startswith("error:") and "no key" in found


def test_the_key_goes_in_a_header_never_the_url(clean, monkeypatch):
    seen: dict = {}

    def fake_get(url, timeout, headers=None):
        seen.update(url=url, headers=headers or {})
        return json.dumps({"web": {"results": []}}).encode()

    monkeypatch.setattr(web, "_get", fake_get)
    monkeypatch.setenv("BRAVE_API_KEY", KEY)
    web._brave("python asyncio", 3)
    assert KEY not in seen["url"] and seen["headers"]["X-Subscription-Token"] == KEY


def test_gaia_brave_mode_drops_searxng_and_passes_the_key_in_the_environment(clean, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("GAIA_SEARCH", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", KEY)
    env = gaia.run_env()
    assert "SEARXNG_URL" not in env
    assert env["SEARCH_BACKEND"] == "brave" and env["BRAVE_API_KEY"] == KEY


def test_without_gaia_search_the_runs_keep_searxng(clean, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    env = gaia.run_env()
    assert env["SEARXNG_URL"] == "http://127.0.0.1:8888" and "SEARCH_BACKEND" not in env


def test_run_start_records_the_search_backend(clean, monkeypatch, tmp_path):
    from agent.config import Settings
    from agent.loop import _narrate_run_start
    from agent.trace import Trace
    from agent.workspace import Workspace

    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    trace = Trace(str(tmp_path / "t"), echo=False)
    _narrate_run_start(trace, Workspace(tmp_path), None, Settings(workspace=".", goal="g"))
    start = next(
        json.loads(line)
        for line in trace.path.read_text().splitlines()
        if json.loads(line)["kind"] == "run_start"
    )
    assert start["search_backend"] == "brave"
    assert KEY not in trace.path.read_text()


# --- the canary, end to end ----------------------------------------------------------------


@pytest.fixture
def searxng(monkeypatch):
    """A local SearXNG stand-in: the answer appears only for the query naming the diamond."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hit = "diamond" in self.path.lower()
            body = {
                "results": [
                    {
                        # A gold only in the title, or past the trace's 160 characters, must
                        # not count: the reference round's recorded hits have neither.
                        "title": "900000 in the title only",
                        "url": "https://example.org/a",
                        "content": ("made of diamond" if hit else "x" * 200 + " 900000"),
                    }
                ]
            }
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _reference_round(root: Path) -> None:
    """Two questions; the reference recorded the gold in its hits for the first only."""
    for tid, query, snippet in [
        ("q1", "photonic crystal diamond 2012", "a diamond slab"),
        ("q2", "Pearl City house sale 2022", "an unrelated page"),
    ]:
        trace = root / "traces" / tid / "run-1.jsonl"
        trace.parent.mkdir(parents=True)
        event = {
            "kind": "web_search",
            "query": query,
            "hits": [{"url": "https://example.org", "snippet": snippet}],
        }
        trace.write_text(json.dumps(event) + "\n")


def test_the_canary_scores_gold_in_results_through_the_harness_interpreter(
    tmp_path, monkeypatch, searxng
):
    _reference_round(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin",
        "SEARXNG_URL": searxng,
        "BRAVE_API_KEY": KEY,  # present but unused: must never surface in the output
        "HOME": str(tmp_path),
    }
    result = gaia.canary_score(
        tmp_path, ["q1", "q2"], {"q1": "Diamond", "q2": "900000"}, env, per=2
    )
    assert result == {
        "backend": "searxng",
        "questions": 2,
        "queries": 2,
        "errors": 0,
        "now": 1,
        "reference": 1,
    }
    assert KEY not in json.dumps(result)
