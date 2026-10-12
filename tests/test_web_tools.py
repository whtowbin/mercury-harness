"""fetch_url and web_search: HTML-to-text extraction, and the SearXNG- and
Brave-backed search, with the network stubbed — no key, instance or live
network needed."""

from __future__ import annotations

import json
import urllib.error
from email.message import Message

import pytest

from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.web import fetch_url, web_search
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stub_urlopen(monkeypatch, body: bytes, status: int = 200):
    def fake(*_a, **_k):
        return FakeResponse(body, status)

    class FakeOpener:
        def open(self, *args, **kwargs):
            return fake(*args, **kwargs)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: FakeOpener())


# -- fetch_url --------------------------------------------------------------


def test_fetch_url_strips_html_to_readable_text(ctx, monkeypatch):
    html = b"""
    <html><head><style>.x{color:red}</style></head>
    <body>
      <h1>Title</h1>
      <p>First paragraph.</p>
      <script>evil()</script>
      <p>Second <b>paragraph</b> with inline markup.</p>
    </body></html>
    """
    stub_urlopen(monkeypatch, html)
    result = fetch_url(ctx, "https://example.com/doc")
    assert "Title" in result
    assert "First paragraph." in result
    assert "Second paragraph with inline markup." in result
    assert "evil()" not in result
    assert "color:red" not in result


def test_fetch_url_passes_through_plain_text_unchanged(ctx, monkeypatch):
    stub_urlopen(monkeypatch, b"just plain text, no markup here")
    result = fetch_url(ctx, "https://example.com/readme.txt")
    assert result == "just plain text, no markup here"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "https://example.com:8443/",
    ],
)
def test_fetch_url_rejects_non_public_destinations(ctx, url):
    result = fetch_url(ctx, url)
    assert result.startswith("error:")


def test_fetch_url_rejects_non_http_schemes(ctx):
    result = fetch_url(ctx, "file:///etc/passwd")
    assert result.startswith("error:")


def test_fetch_url_truncates_long_content(ctx, monkeypatch):
    stub_urlopen(monkeypatch, b"x" * 20_000)
    result = fetch_url(ctx, "https://example.com", max_chars=100)
    assert len(result) < 300  # the body plus the truncation note
    assert "truncated" in result


def test_fetch_url_reports_an_http_error_cleanly(ctx, monkeypatch):
    def fake(*_a, **_k):
        raise urllib.error.HTTPError("https://example.com", 404, "Not Found", Message(), None)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = fetch_url(ctx, "https://example.com/missing")
    assert result.startswith("error:")
    assert "404" in result


def test_fetch_url_reports_an_unreachable_host_cleanly(ctx, monkeypatch):
    def fake(*_a, **_k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = fetch_url(ctx, "https://nowhere.example")
    assert result.startswith("error:")


# -- connect-time SSRF validation ---------------------------------------------
# The tests above stub urllib.request.build_opener entirely, so they never
# exercise _get's actual connection classes — only that fetch_url still
# calls build_opener and copes with what comes back. These run a real local
# server through the real opener, proving the custom connect() overrides
# used to close the DNS-rebinding gap do not just avoid crashing but still
# fetch real bytes end to end, and that a host resolving to a non-public
# address is rejected at connect time even when the initial resolve check
# is bypassed (standing in for a resolver that answers differently a
# moment later than it did for that first check).


def test_a_real_local_server_is_still_fetchable_through_the_new_connection_classes(
    ctx, monkeypatch
):
    import http.server
    import threading

    from agent.tools import web as web_module

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"hello from a real socket"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # quiet the test output
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 127.0.0.1 is exactly what _public_url/_resolve_public_address exist
        # to reject, so the pre-flight check is bypassed here on purpose —
        # this test's job is the connection classes, not that gate.
        monkeypatch.setattr(web_module, "_public_url", lambda url: None)
        monkeypatch.setattr(web_module, "_resolve_public_address", lambda host, port: "127.0.0.1")
        result = fetch_url(ctx, f"http://127.0.0.1:{server.server_port}/")
        assert result == "hello from a real socket"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_resolve_public_address_rejects_loopback():
    from agent.tools.web import _resolve_public_address, _SSRFBlocked

    with pytest.raises(_SSRFBlocked):
        _resolve_public_address("localhost", 80)


def test_resolve_public_address_rejects_link_local_metadata_address():
    from agent.tools.web import _resolve_public_address, _SSRFBlocked

    with pytest.raises(_SSRFBlocked):
        _resolve_public_address("169.254.169.254", 80)


def test_a_connection_that_resolves_to_a_blocked_address_at_connect_time_fails(ctx, monkeypatch):
    """Even if the pre-flight _public_url check is somehow satisfied (the
    scenario a DNS-rebinding attack engineers), the connection itself
    refuses to dial a non-public address, because it re-resolves and
    re-validates in its own connect()."""
    from agent.tools import web as web_module

    monkeypatch.setattr(web_module, "_public_url", lambda url: None)
    result = fetch_url(ctx, "http://127.0.0.1:1/")
    assert result.startswith("error:")
    assert "non-public" in result or "resolves" in result


# -- web_search ---------------------------------------------------------------


@pytest.fixture
def no_backend(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)


def test_web_search_without_a_backend_says_how_to_configure_one(ctx, no_backend):
    result = web_search(ctx, "python asyncio tutorial")
    assert result.startswith("error:")
    assert "SEARXNG_URL" in result
    assert "BRAVE_API_KEY" in result


def test_searxng_is_preferred_and_its_results_are_parsed(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888/")
    monkeypatch.setenv("BRAVE_API_KEY", "also-set")
    seen = []
    payload = {
        "results": [
            {
                "title": "asyncio docs",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "content": "Asynchronous <em>I/O</em>.",
            },
            {"title": "Real Python", "url": "https://realpython.com/async-io-python/"},
        ],
        "unresponsive_engines": [],
    }

    def fake(request, *_a, **_k):
        seen.append(request.full_url)
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = web_search(ctx, "python asyncio", max_results=5)
    assert seen == ["http://127.0.0.1:8888/search?q=python+asyncio&format=json"]
    assert "1. asyncio docs\n   https://docs.python.org/3/library/asyncio.html" in result
    assert "Asynchronous I/O." in result  # tags stripped from the snippet
    assert "2. Real Python" in result


def test_searxng_reaches_localhost_despite_the_fetch_url_guard(ctx, no_backend, monkeypatch):
    """The instance address is the operator's, not the model's, so the
    non-public-address check that protects fetch_url must not block it."""
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    stub_urlopen(monkeypatch, json.dumps({"results": [{"title": "t", "url": "u"}]}).encode())
    assert web_search(ctx, "q").startswith("1. t")


def test_long_snippets_are_cut_to_a_readable_length(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    payload = {"results": [{"title": "t", "url": "u", "content": "x" * 2_000}]}
    stub_urlopen(monkeypatch, json.dumps(payload).encode())
    result = web_search(ctx, "q")
    assert "x" * 300 + "..." in result
    assert "x" * 301 not in result


def test_searxng_empty_results_name_the_engines_that_failed(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    payload = {"results": [], "unresponsive_engines": [["google", "CAPTCHA"], ["bing", "timeout"]]}
    stub_urlopen(monkeypatch, json.dumps(payload).encode())
    result = web_search(ctx, "anything")
    assert "no results" in result
    assert "google, bing" in result


def test_searxng_without_json_enabled_says_which_setting(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")

    def fake(*_a, **_k):
        raise urllib.error.HTTPError("http://127.0.0.1:8888", 403, "Forbidden", Message(), None)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = web_search(ctx, "anything")
    assert result.startswith("error:")
    assert "search.formats" in result


def test_searxng_not_running_is_reported_as_unreachable(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")

    def fake(*_a, **_k):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = web_search(ctx, "anything")
    assert result.startswith("error: could not reach SearXNG")


def test_web_search_parses_results_with_a_key_set(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    payload = {
        "web": {
            "results": [
                {
                    "title": "Real Python: asyncio",
                    "url": "https://realpython.com/asyncio",
                    "description": "A <b>guide</b> to asyncio.",
                },
                {
                    "title": "Python docs: asyncio",
                    "url": "https://docs.python.org/3/asyncio.html",
                    "description": "The official reference.",
                },
            ]
        },
    }
    stub_urlopen(monkeypatch, json.dumps(payload).encode())
    result = web_search(ctx, "python asyncio tutorial")
    assert "Real Python: asyncio" in result
    assert "https://realpython.com/asyncio" in result
    assert "A guide to asyncio." in result  # HTML stripped from the snippet
    assert "docs.python.org" in result


def test_web_search_with_no_results_says_so(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    stub_urlopen(monkeypatch, json.dumps({"web": {"results": []}}).encode())
    result = web_search(ctx, "asdkjhaslkdjhasldkjh")
    assert "no results" in result.lower()


def test_web_search_reports_an_invalid_key_clearly(ctx, no_backend, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "bad-key")

    def fake(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://api.search.brave.com", 401, "Unauthorized", Message(), None
        )

    class FakeOpener:
        def open(self, *args, **kwargs):
            return fake(*args, **kwargs)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: FakeOpener())
    result = web_search(ctx, "anything")
    assert result.startswith("error:")
    assert "401" in result
