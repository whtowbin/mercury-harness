"""The web deny list and the fetch_url ceiling (DESIGN-LOG §80).

GLM-planner benchmark arms fetched SciCode's gold code (`problems_dev.jsonl`
at max_chars=300000) and two django tasks' upstream fixes. A run now lists a
benchmark's sources in `web_deny`; fetch_url refuses them and web_search drops
them from its results, however the address is wrapped.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools import web as web_module
from agent.tools.web import MAX_FETCH_CHARS_CEILING, denied_by, fetch_url, web_search
from agent.trace import Trace
from agent.workspace import Workspace

# The list DESIGN-LOG §80 recommends for SciCode. The bare org name, not
# "github.com/scicode-bench": the traces also reached it as
# raw.githubusercontent.com/scicode-bench, api.github.com/repos/scicode-bench
# and scicode-bench.github.io, which a host-and-path pattern misses.
SCICODE = ("scicode-bench", "SciCode1/SciCode")


def make_ctx(tmp_path, deny: tuple[str, ...] = ()) -> ToolContext:
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        web_deny=deny,
    )


def events(ctx: ToolContext) -> list[dict]:
    return [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]


# -- matching -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        # the exact addresses the SciCode traces fetched
        "https://huggingface.co/datasets/SciCode1/SciCode/resolve/main/problems_dev.jsonl",
        "https://r.jina.ai/https://huggingface.co/datasets/SciCode1/SciCode/raw/main/problems_test.jsonl",
        "https://datasets-server.huggingface.co/rows?dataset=SciCode1%2FSciCode&config=default",
        "https://raw.githubusercontent.com/scicode-bench/SciCode/main/tests/test_data/first_problem.jsonl",
        "https://api.github.com/repos/scicode-bench/SciCode/contents/problems/10",
        "HTTPS://WWW.GitHub.com/SciCode-Bench/SciCode",
        "https://scicode-bench.github.io/SciCode/problems/10.html",
    ],
)
def test_benchmark_sources_are_denied_however_spelled_or_wrapped(url):
    assert denied_by(url, SCICODE) is not None


@pytest.mark.parametrize(
    "url",
    [
        "https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html",
        "https://github.com/scipy/scipy/blob/main/scipy/linalg/_decomp_qr.py",
        "https://arxiv.org/abs/2606.01286",
    ],
)
def test_ordinary_documentation_is_not_denied(url):
    assert denied_by(url, SCICODE) is None


def test_an_empty_deny_list_denies_nothing():
    assert denied_by("https://huggingface.co/datasets/SciCode1/SciCode", ()) is None


# -- fetch_url ----------------------------------------------------------------


def test_a_denied_fetch_is_refused_without_a_request_and_traced(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, SCICODE)

    def no_network(*_a, **_k):
        raise AssertionError("a denied URL must not be requested")

    monkeypatch.setattr(web_module, "_get", no_network)
    url = "https://huggingface.co/datasets/SciCode1/SciCode/resolve/main/problems_dev.jsonl"
    result = fetch_url(ctx, url, max_chars=300_000)
    assert result.startswith("error:") and "deny list" in result
    denied = [e for e in events(ctx) if e["kind"] == "web_denied"]
    assert denied == [
        {
            **denied[0],
            "tool": "fetch_url",
            "url": url,
            "pattern": "SciCode1/SciCode",
        }
    ]


def test_for_step_carries_the_deny_list(tmp_path):
    assert make_ctx(tmp_path, SCICODE).for_step().web_deny == SCICODE


class _Server:
    """A real local HTTP server serving `body`, counting the requests it gets."""

    def __init__(self, body: bytes):
        self.requests = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


@pytest.fixture
def local_fetch(monkeypatch):
    """Let fetch_url reach 127.0.0.1: the public-address gate is what these
    tests are not about (test_web_tools covers it)."""
    monkeypatch.setattr(web_module, "_public_url", lambda url: None)
    monkeypatch.setattr(web_module, "_resolve_public_address", lambda host, port: "127.0.0.1")


def test_a_real_fetch_is_clamped_to_the_ceiling(tmp_path, local_fetch):
    ctx = make_ctx(tmp_path)
    server = _Server(b"x" * (MAX_FETCH_CHARS_CEILING + 5_000))
    with server as base:
        result = fetch_url(ctx, f"{base}/problems_dev.jsonl", max_chars=300_000)
    assert result.startswith("x" * MAX_FETCH_CHARS_CEILING)
    assert f"truncated at {MAX_FETCH_CHARS_CEILING} chars, 5000 more" in result


def test_a_real_server_on_the_deny_list_is_never_contacted(tmp_path, local_fetch):
    server = _Server(b"gold solution")
    with server as base:
        ctx = make_ctx(tmp_path, (base.removeprefix("http://") + "/answers",))
        refused = fetch_url(ctx, f"{base}/answers/1.py")
        allowed = fetch_url(ctx, f"{base}/docs")
    assert refused.startswith("error:")
    assert allowed == "gold solution"
    assert server.requests == 1


# -- web_search ---------------------------------------------------------------


def test_search_results_on_the_deny_list_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    results = [
        ("SciCode problem 10", "https://github.com/scicode-bench/SciCode/tree/main/problems", ""),
        ("Ewald summation", "https://en.wikipedia.org/wiki/Ewald_summation", "lattice sums"),
    ]
    monkeypatch.setattr(web_module, "_searxng", lambda q, n: (list(results), "", []))
    ctx = make_ctx(tmp_path, SCICODE)
    out = web_search(ctx, "SciCode ewald_summation reference solution")
    assert "scicode-bench" not in out
    assert out.startswith("1. Ewald summation")
    denied = [e for e in events(ctx) if e["kind"] == "web_denied"]
    assert [(e["tool"], e["dropped"]) for e in denied] == [("web_search", 1)]


def test_search_without_a_deny_list_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    results = [("SciCode", "https://github.com/scicode-bench/SciCode", "")]
    monkeypatch.setattr(web_module, "_searxng", lambda q, n: (list(results), "", []))
    ctx = make_ctx(tmp_path)
    assert web_search(ctx, "scicode").startswith("1. SciCode")
    assert not [e for e in events(ctx) if e["kind"] == "web_denied"]


# -- code tools (DESIGN-LOG §93) ------------------------------------------------


def test_code_naming_a_denied_source_never_runs_and_other_code_does(tmp_path):
    """GAIA runs fetched pages from run_python (`requests.get("https://r.jina.ai/...")`), which
    the deny list did not cover. A real local server and a real interpreter."""
    from agent.tools.checks import run_python

    server = _Server(b"gold solution")
    with server as base:
        ctx = make_ctx(tmp_path, (base.removeprefix("http://") + "/answers",))
        fetch = "import urllib.request; print(urllib.request.urlopen({!r}).read().decode())"
        refused = run_python(ctx, fetch.format(f"{base}/answers/1.py"))
        allowed = run_python(ctx, fetch.format(f"{base}/docs"))
    assert refused.startswith("error:") and "deny list" in refused
    assert "gold solution" in allowed
    assert server.requests == 1
    assert [e["tool"] for e in events(ctx) if e["kind"] == "web_denied"] == ["run_python"]


def test_run_bash_is_refused_the_same_way(tmp_path):
    from agent.tools.shell import run_bash

    ctx = make_ctx(tmp_path, SCICODE)
    refused = run_bash(ctx, "curl -s https://r.jina.ai/https://github.com/scicode-bench/SciCode")
    assert refused.startswith("error:") and "scicode-bench" in refused
    assert "hello" in run_bash(ctx, "echo hello")


def test_code_tools_without_a_deny_list_are_unchanged(tmp_path):
    from agent.tools.checks import run_python

    ctx = make_ctx(tmp_path)
    assert "scicode-bench" in run_python(ctx, "print('scicode-bench')")


# -- what a trace keeps of web results (DESIGN-LOG §93.2) -------------------------


def test_the_trace_keeps_the_results_shown_and_the_urls_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    mirror = "https://github.com/scicode-bench/SciCode/tree/main/problems"
    results = [
        ("SciCode problem 10", mirror, ""),
        ("Ewald summation", "https://en.wikipedia.org/wiki/Ewald_summation", "<b>lattice</b> sums"),
    ]
    monkeypatch.setattr(web_module, "_searxng", lambda q, n: (list(results), "", []))
    ctx = make_ctx(tmp_path, SCICODE)
    web_search(ctx, "ewald summation")
    (searched,) = [e for e in events(ctx) if e["kind"] == "web_search"]
    (denied,) = [e for e in events(ctx) if e["kind"] == "web_denied"]
    assert searched["hits"] == [
        {"url": "https://en.wikipedia.org/wiki/Ewald_summation", "snippet": "lattice sums"}
    ]
    assert denied["urls"] == [mirror]


def test_the_trace_keeps_the_opening_of_a_real_fetched_page(tmp_path, local_fetch):
    ctx = make_ctx(tmp_path)
    page = b"The 2011 census counted 791.8 thousand women. " + b"x" * 5_000
    with _Server(page) as base:
        fetch_url(ctx, f"{base}/census")
    (fetched,) = [e for e in events(ctx) if e["kind"] == "fetch_url"]
    assert fetched["head"].startswith("The 2011 census counted 791.8 thousand women.")
    assert len(fetched["head"]) == web_module.TRACE_FETCH_HEAD_CHARS


# -- PDFs (DESIGN-LOG §93.3) ---------------------------------------------------------


def _pdf(text: str, padding: int = 0) -> bytes:
    """A one-page PDF showing `text`, with `padding` bytes of comment after the header so
    a test can push it past the 2 MB page limit. Offsets computed, so readers accept it."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n" + (b"%" + b"x" * padding + b"\n" if padding else b"")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def test_a_real_pdf_comes_back_as_its_text(tmp_path, local_fetch):
    ctx = make_ctx(tmp_path)
    with _Server(_pdf("Ridership in May 2019 was 1,234,567")) as base:
        text = fetch_url(ctx, f"{base}/report.pdf")
    assert "Ridership in May 2019 was 1,234,567" in text
    assert "%PDF" not in text and text.startswith("[page 1]")


def test_a_pdf_over_the_page_limit_is_still_read(tmp_path, local_fetch):
    big = _pdf("Table 3: 519 at bats", padding=web_module.MAX_FETCH_BYTES + 100_000)
    assert len(big) > web_module.MAX_FETCH_BYTES
    ctx = make_ctx(tmp_path)
    with _Server(big) as base:
        text = fetch_url(ctx, f"{base}/big.pdf")
    assert "Table 3: 519 at bats" in text


def test_an_html_page_over_the_limit_gives_its_opening(tmp_path, local_fetch):
    """Refusing it sent g7 runs back for Wikipedia's "Moon" four times (DESIGN-LOG §93.4)."""
    ctx = make_ctx(tmp_path)
    page = b"<html><body><p>The Moon orbits Earth.</p>" + b"x" * web_module.MAX_FETCH_BYTES
    with _Server(page) as base:
        text = fetch_url(ctx, f"{base}/huge")
    assert text.startswith("The Moon orbits Earth.")
    assert text.endswith(f"cut at {web_module.MAX_FETCH_BYTES} bytes before its text was read")


def test_a_pdf_over_its_own_limit_is_refused(tmp_path, local_fetch, monkeypatch):
    monkeypatch.setattr(web_module, "MAX_PDF_BYTES", 3_000_000)
    ctx = make_ctx(tmp_path)
    with _Server(_pdf("x", padding=3_100_000)) as base:
        assert "PDF exceeds" in fetch_url(ctx, f"{base}/huge.pdf")


def test_a_broken_pdf_is_an_error_not_noise(tmp_path, local_fetch):
    ctx = make_ctx(tmp_path)
    with _Server(b"%PDF-1.4\nthis is not really a pdf") as base:
        assert fetch_url(ctx, f"{base}/broken.pdf").startswith("error: could not read the PDF")


def test_the_drivers_lists_refuse_the_routes_mercury_3_used():
    """DESIGN-LOG §131.5: URLs Mercury 3 fetched successfully in sealed runs, each now denied,
    while code a solution could contain (workspace paths, Django names) still passes."""
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path[:0] = [str(scripts), str(scripts / "swebench"), str(scripts / "scicode")]
    import sci  # ty: ignore[unresolved-import]
    import swe  # ty: ignore[unresolved-import]

    django = tuple(swe.WEB_DENY)
    for url in (
        "https://api.github.com/repos/django/django/pulls/11790/files",
        "https://api.github.com/search/issues?q=repo:django/django+%22fast+delete%22",
        "https://www.swebench.com/viewer.html",
        "https://cdn.jsdelivr.net/gh/django/django@main/tests/model_fields/tests.py",
        "https://raw.githack.com/django/django/main/django/db/models/base.py",
        "https://github.com/SWE-bench/SWE-bench",
        "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite",
    ):
        assert denied_by(url, django), url
    scicode = tuple(sci.WEB_DENY)
    for url in (
        "https://sourcegraph.com/.api/search/stream?q=context:global+select_big_weights",
        "https://sourcegraph.com/.api/search/stream?q=repo:^github\\.com/NVIDIA-NeMo/Gym$+file:scicode",
        "https://huggingface.co/datasets/akshathmangudi/SciCode",
        "https://huggingface.co/api/datasets?search=scicode",
        "https://github.com/search?q=%22select_big_weights%22&type=code",
        "https://www.bing.com/search?q=%22n_tangle%22+%22scicode%22",
        "https://html.duckduckgo.com/html/?q=%22pendulum_analysis%22",
        "https://raw.githubusercontent.com/someone/SciCode/main/eval/data/problems_all.jsonl",
        "https://cdn.jsdelivr.net/gh/someone/SciCode@main/eval/data/problems_all.jsonl",
    ):
        assert denied_by(url, scicode), url
    for code in (
        "open('/Users/x/.cache/mercury-harness/swebench-runs/s131p-dj-r1-t3/11880/django/x.py')",
        "open('/Users/x/.cache/mercury-harness/scicode-runs/s131-sci-h1/51/solution.py')",
        "def scicode_example(x): return x",
        "from django.http import HttpResponse",
        "https://pypi.org/pypi/numpy/json",
        "https://arxiv.org/abs/2407.13168",
        "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",  # packages, not repos
    ):
        assert not denied_by(code, scicode + django), code
