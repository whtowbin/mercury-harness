"""The open internet: fetching a known URL, and searching for one.

Read-only and outward-facing only — neither tool touches the workspace or
runs code — so, like pypi_info, this does not go through the approval gate:
the risk here is "slow or wrong," not "modifies your system."

web_search needs a real search API, not scraping. A DuckDuckGo HTML scrape
was the obvious no-key option and was tried first; it hits an interstitial
bot challenge ("select all squares containing a duck") on essentially every
request from a script, which makes it unusable rather than just unreliable.
Search goes through a self-hosted SearXNG instance (SEARXNG_URL; free, see
scripts/searxng/) or Brave's metered API (BRAVE_API_KEY). With neither set,
web_search is left out of every tool kit rather than offered and failing.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .base import Tool, ToolContext, obj

FETCH_TIMEOUT = 15
SEARCH_TIMEOUT = 15
MAX_FETCH_CHARS = 8_000
# The most a model may ask fetch_url for. Of 160 traced fetches, every one
# asking for more than 30,000 characters was a benchmark's own problem or answer
# file (SciCode's problems_dev.jsonl at 300,000); the largest legitimate one was
# a paper at 30,000 (DESIGN-LOG §80, TUNABLES.md).
MAX_FETCH_CHARS_CEILING = 32_000
# SearXNG passes through whatever an engine returned as the snippet, sometimes
# a whole code sample; a result list is for choosing what to fetch_url next.
MAX_SNIPPET_CHARS = 300
MAX_FETCH_BYTES = 2_000_000
# PDFs are read further: reports and papers run past 2 MB, and their text is a small part
# of the bytes. 13 of g5's 80 failed fetches were PDFs over the page limit (DESIGN-LOG §93.3).
MAX_PDF_BYTES = 25_000_000
USER_AGENT = "mercury-harness (github.com/whtowbin/mercury-harness)"
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):
    """The minimum needed to turn a page into readable text: keep visible
    words, drop markup, add a line break at block boundaries so paragraphs
    and list items don't run together into one wall of text."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data.strip())

    def text(self) -> str:
        joined = " ".join(self.chunks)
        return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", joined)).strip()


def _public_url(url: str) -> str | None:
    """A fast, clear pre-flight check: right scheme/port, and the hostname
    resolves to something public *right now*. Not itself the security
    boundary — see `_validate_and_connect` for that — but worth doing here
    too so a request to an obviously-bad URL fails with a helpful message
    before any connection machinery gets involved.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "only public http/https URLs are allowed"
    if parsed.port not in (None, 80, 443):
        return "only ports 80 and 443 are allowed"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        _resolve_public_address(parsed.hostname, port)
    except _SSRFBlocked as exc:
        return str(exc)
    return None


class _SSRFBlocked(OSError):
    """Raised when a hostname resolves to a non-public address."""


def _resolve_public_address(host: str, port: int) -> str:
    """Resolve `host`, verify every address it returns is public, and return
    one of them to connect to.

    Used both for the fast pre-flight check in `_public_url` and — this is
    the part that actually matters — from inside each connection's own
    `connect()`, immediately before dialing. A check done once, earlier
    (resolve, validate, *then* hand the still-just-a-hostname URL to
    urllib to connect) leaves a gap: DNS is looked up a second time when
    the connection is actually made, and nothing stops that second lookup
    answering with a private or loopback address a low-TTL record can be
    flipped to serve after the first, public-looking answer was checked.
    Resolving and validating right where the socket is about to be opened,
    and connecting to that exact validated address rather than the
    hostname, closes that window instead of narrowing it.
    """
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise _SSRFBlocked(f"could not resolve host ({exc})") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            not ip.is_global
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise _SSRFBlocked(f"host resolves to a non-public address ({address})")
    return str(next(iter(addresses)))


class _ValidatingHTTPConnection(http.client.HTTPConnection):
    """An HTTPConnection whose connect() re-resolves and re-validates the
    host immediately before dialing, and dials the validated address
    directly rather than the hostname — see `_resolve_public_address`."""

    def connect(self) -> None:
        address = _resolve_public_address(self.host, self.port)
        self.sock = socket.create_connection(
            (address, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()


class _ValidatingHTTPSConnection(http.client.HTTPSConnection):
    """The HTTPS counterpart: connects to the validated address, then wraps
    the socket in TLS using the original hostname for SNI and certificate
    hostname verification, exactly as HTTPSConnection.connect would."""

    def connect(self) -> None:
        address = _resolve_public_address(self.host, self.port)
        self.sock = socket.create_connection(
            (address, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _ValidatingHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_ValidatingHTTPConnection, req)


class _ValidatingHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_ValidatingHTTPSConnection, req, context=self._context)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        problem = _public_url(newurl)
        if problem:
            raise urllib.error.URLError(f"redirect blocked: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, timeout: float, headers: dict[str, str] | None = None) -> bytes:
    problem = _public_url(url)
    if problem:
        raise urllib.error.URLError(problem)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    # Both the initial request and every redirect _SafeRedirectHandler lets
    # through go through the validating handlers below, so a redirect to a
    # different host gets the same connect-time DNS check as the original
    # URL — not just the one-shot check _SafeRedirectHandler does itself.
    opener = urllib.request.build_opener(
        _ValidatingHTTPHandler, _ValidatingHTTPSHandler, _SafeRedirectHandler
    )
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(MAX_FETCH_BYTES + 1)
        if raw.startswith(b"%PDF-") and len(raw) > MAX_FETCH_BYTES:
            raw += response.read(MAX_PDF_BYTES + 1 - len(raw))
        return raw


# How much of each web result a trace keeps: enough to tell where an answer came from and
# whether a page mirrored a benchmark, not the page itself (DESIGN-LOG §93.2).
TRACE_FETCH_HEAD_CHARS = 400
TRACE_SNIPPET_CHARS = 160


def _deny_key(text: str) -> str:
    """A URL or deny pattern in the form they are compared in: percent-decoding
    undone (twice over, for a URL quoted inside another's query string),
    lowercased, and every `scheme://` and `www.` removed, so a pattern matches
    however the address is spelled or wrapped."""
    for _ in range(2):
        text = urllib.parse.unquote(text)
    return re.sub(r"[a-z][a-z0-9+.-]*://|\bwww\.", "", text.lower())


def denied_by(url: str, deny: tuple[str, ...]) -> str | None:
    """The first deny pattern `url` contains, or None.

    A plain substring match over the whole URL, not a host check: the answer to
    a benchmark reached SciCode runs wrapped in a reader proxy
    (`r.jina.ai/https://huggingface.co/...`) and as a query parameter
    (`datasets-server.huggingface.co/rows?dataset=SciCode1%2FSciCode`), and
    both carry the benchmark's address inside a URL whose host is not it.
    """
    key = _deny_key(url)
    for pattern in deny:
        needle = _deny_key(pattern)
        if needle and needle in key:
            return pattern
    return None


def code_denied(ctx: ToolContext, tool: str, code: str) -> str | None:
    """The refusal for a run_python or run_bash call whose code names a denied source, or None.

    Those tools reach the web too (GAIA scout g2: 251 of 460 calls carried a URL, some through
    `r.jina.ai`), and the deny list covered only fetch_url and web_search (DESIGN-LOG §93). A
    literal match only: a URL assembled at run time passes, and `harness trace leaks` scans
    the same code afterwards. Free when the list is empty, as it is outside benchmark runs.
    """
    pattern = denied_by(code, ctx.web_deny) if ctx.web_deny else None
    if not pattern:
        return None
    ctx.trace.event("web_denied", tool=tool, pattern=pattern)
    return f"error: this code names {pattern!r}, which is on the run's web deny list"


def fetch_url(ctx: ToolContext, url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Fetch a URL and return its readable text, HTML stripped to plain words."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"error: only http/https URLs are fetchable, got {url!r}"
    if max_chars < 1:
        return "error: max_chars must be positive"
    max_chars = min(max_chars, MAX_FETCH_CHARS_CEILING)
    pattern = denied_by(url, ctx.web_deny)
    if pattern:
        ctx.trace.event("web_denied", tool="fetch_url", url=url, pattern=pattern)
        return f"error: {url} is not available in this run (it is on the run's web deny list)"
    try:
        raw = _get(url, FETCH_TIMEOUT)
    except urllib.error.HTTPError as exc:
        return f"error: HTTP {exc.code} fetching {url}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"error: could not reach {url} ({exc})"

    is_pdf = raw.startswith(b"%PDF-")
    cut = ""
    if is_pdf and len(raw) > MAX_PDF_BYTES:
        return f"error: the PDF exceeds the {MAX_PDF_BYTES} byte limit"
    if not is_pdf and len(raw) > MAX_FETCH_BYTES:
        # A page's opening is still worth reading: an error here sent GAIA runs back for the
        # same page (Wikipedia's "Moon", four times in one run; DESIGN-LOG §93.4).
        raw = raw[:MAX_FETCH_BYTES]
        cut = f"\n... the page was cut at {MAX_FETCH_BYTES} bytes before its text was read"
    if is_pdf:
        text = _pdf_text(raw)
        if text.startswith("error:"):
            return f"{text} ({url})"
    else:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:  # a decoding surprise must not crash the run
            return f"error: could not decode the response from {url} ({exc})"

    # A cheap sniff rather than trusting Content-Type: some servers mislabel
    # plain text or JSON as html, and vice versa; the tag prefix is reliable.
    if not is_pdf and ("<html" in text[:2000].lower() or "<!doctype html" in text[:2000].lower()):
        extractor = _TextExtractor()
        extractor.feed(text)
        body = extractor.text()
    else:
        body = text

    ctx.trace.event("fetch_url", url=url, chars=len(body), head=body[:TRACE_FETCH_HEAD_CHARS])
    if not body.strip():
        return f"(no readable text extracted from {url})"
    truncated = body[:max_chars]
    if len(body) > max_chars:
        truncated += f"\n... truncated at {max_chars} chars, {len(body) - max_chars} more"
    return truncated + cut


def _pdf_text(raw: bytes) -> str:
    """A PDF's text, page by page, or an `error:` line. It stops once it holds more than
    fetch_url can show (MAX_FETCH_CHARS_CEILING), so a 500-page report costs what its first
    pages do. Before this, a PDF came back as its raw bytes decoded as text: 17 of g5's
    fetches were `%PDF-` noise, and GAIA runs that reached for a PDF were right 6 of 20
    times, against 55 of 107 for the rest (DESIGN-LOG §93.3)."""
    import io
    import logging

    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    # pypdf logs font-encoding advice as warnings, which would land in the run's output.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages: list[str] = []
        held = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"[page {number}]\n{text}")
            held += len(text)
            if held > MAX_FETCH_CHARS_CEILING:
                pages.append(f"[stopped after page {number} of {len(reader.pages)}]")
                break
    except (PyPdfError, ValueError, KeyError, TypeError) as exc:
        return f"error: could not read the PDF ({type(exc).__name__}: {exc})"
    return "\n".join(pages)


def search_backend() -> str | None:
    """Which search service `web_search` will use, or None when none is set up.

    SearXNG first: it is free, self-hosted, and answers from localhost. Brave
    stopped being free in February 2026 (metered credit behind a card on file),
    and Google's Custom Search API took no new customers after 2025 and stops
    serving on 2027-01-01, so neither is a default. Read from the environment
    on each call rather than cached, so a `.env` edit takes effect next run.
    """
    forced = os.getenv("SEARCH_BACKEND", "").strip().lower()
    if forced in ("searxng", "brave"):
        # Named explicitly (DESIGN-LOG §115): a run that must use one backend, the final GAIA
        # pass on Brave, cannot be switched to SearXNG by a SEARXNG_URL that load_dotenv
        # found in some ancestor's .env. A missing credential is then reported by the backend.
        return forced
    if os.getenv("SEARXNG_URL", "").strip():
        return "searxng"
    if os.getenv("BRAVE_API_KEY", "").strip():
        return "brave"
    return None


def _brave_key() -> str:
    """The Brave Search API key: BRAVE_API_KEY, or, only when SEARCH_BACKEND=brave names Brave
    explicitly, the Keychain item `security add-generic-password -a harness -s BraveSearch`.
    The Keychain is not consulted otherwise, so an ordinary run never shells out for it."""
    key = os.getenv("BRAVE_API_KEY", "").strip()
    if key or os.getenv("SEARCH_BACKEND", "").strip().lower() != "brave":
        return key
    from ..config import _get_keychain_secret

    return (_get_keychain_secret("BraveSearch", "harness") or "").strip()


def web_search(ctx: ToolContext, query: str, max_results: int = 5) -> str:
    """Search the web through the configured backend (see `search_backend`)."""
    max_results = max(1, min(max_results, 10))
    backend = search_backend()
    if backend == "searxng":
        found = _searxng(query, max_results)
    elif backend == "brave":
        found = _brave(query, max_results)
    else:
        return (
            "error: web_search has no search backend. Set SEARXNG_URL to a SearXNG "
            "instance (scripts/searxng/ runs one locally) or BRAVE_API_KEY. Until "
            "then, use fetch_url on a specific URL instead."
        )
    if isinstance(found, str):
        return found
    results, note, failed = found
    kept = [r for r in results if not denied_by(r[1], ctx.web_deny)]
    if len(kept) < len(results):
        ctx.trace.event(
            "web_denied",
            tool="web_search",
            query=query,
            dropped=len(results) - len(kept),
            urls=[r[1] for r in results if r not in kept],
        )
        results = kept
    # failed_engines: SearXNG answers 200 with nothing when every upstream engine is
    # blocked, which read as `results: 0` for two days of sweeps (DESIGN-LOG §89.1).
    # hits: what the model was shown, so a trace can say where an answer came from and
    # whether a result mirrored the benchmark (DESIGN-LOG §93.2).
    ctx.trace.event(
        "web_search",
        query=query,
        results=len(results),
        backend=backend,
        failed_engines=failed,
        hits=[
            {"url": url, "snippet": _strip_tags(snippet)[:TRACE_SNIPPET_CHARS]}
            for _title, url, snippet in results[:max_results]
        ],
    )
    if not results:
        return f"no results for {query!r}{note}"
    lines = []
    for i, (title, url, snippet) in enumerate(results[:max_results], 1):
        snippet = _strip_tags(snippet)
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS].rstrip() + "..."
        lines.append(f"{i}. {_strip_tags(title)}\n   {url}\n   {snippet}")
    return "\n".join(lines)


def _strip_tags(text: object) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


# A backend returns (results, note, failed engines) on success, where each result is
# (title, url, snippet) and note is appended to a "no results" reply, or an
# "error: ..." string the model sees as-is.
_Found = tuple[list[tuple[str, str, str]], str, list[str]] | str


def _searxng(query: str, max_results: int) -> _Found:
    """Query a SearXNG instance's JSON API.

    The address comes from the operator's SEARXNG_URL, never from the model,
    and is usually localhost, so this deliberately does not go through `_get`'s
    public-address check — that guard exists for URLs a model chooses. The
    model controls only the query, which is URL-encoded into a parameter.
    """
    base = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
            raw = response.read(MAX_FETCH_BYTES)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            return (
                f"error: SearXNG at {base} refused JSON (HTTP 403). Its settings.yml "
                "needs `json` under search.formats."
            )
        return f"error: SearXNG at {base} returned HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"error: could not reach SearXNG at {base} ({exc}). Is the container running?"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"error: unparseable response from SearXNG ({exc})"
    results = [
        (item.get("title", ""), item.get("url", ""), item.get("content", ""))
        for item in data.get("results") or []
    ]
    # SearXNG answers 200 with zero results when every upstream engine failed
    # (rate-limited, CAPTCHA'd); say so, or it reads as "nothing exists".
    down = [str(entry[0]) for entry in data.get("unresponsive_engines") or [] if entry]
    note = f" (engines that failed: {', '.join(down)})" if down else ""
    return results, note, down


def _brave(query: str, max_results: int) -> _Found:
    """Query Brave's Search API. Needs BRAVE_API_KEY; metered since 2026-02."""
    api_key = _brave_key()
    if not api_key:
        return (
            "error: Brave Search is the configured backend but no key is set (BRAVE_API_KEY, or "
            "the Keychain item `-a harness -s BraveSearch` with SEARCH_BACKEND=brave)"
        )
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results}
    )
    try:
        raw = _get(
            url,
            SEARCH_TIMEOUT,
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp else ""
        if exc.code in (401, 403):
            return f"error: Brave Search rejected the API key (HTTP {exc.code}). {body}"
        return f"error: Brave Search returned HTTP {exc.code}. {body}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"error: could not reach Brave Search ({exc})"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"error: unparseable response from Brave Search ({exc})"
    results = [
        (item.get("title", ""), item.get("url", ""), item.get("description", ""))
        for item in (data.get("web") or {}).get("results") or []
    ]
    return results, "", []


def available_web_tools() -> list[Tool]:
    """WEB_TOOLS minus web_search when no backend is configured.

    Offering a tool that can only return an error costs a schema in every
    prompt and a wasted round trip whenever a model tries it: before this
    check no key had ever been set, so all 61 web_search calls across 19
    traced runs returned the missing-key error.
    """
    if search_backend() is not None:
        return list(WEB_TOOLS)
    return [t for t in WEB_TOOLS if t.name != "web_search"]


WEB_TOOLS = [
    Tool(
        "fetch_url",
        "Fetch a URL and return its readable text (HTML stripped to plain words). "
        "Use this to read a specific page you already know the address of — "
        "documentation, a changelog, an RFC.",
        obj(
            {
                "url": {"type": "string", "description": "Absolute http(s) URL."},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate the extracted text here, default 8000, at most 32000.",
                },
            },
            ["url"],
        ),
        fetch_url,
    ),
    Tool(
        "web_search",
        "Search the web. Use it to find documentation, an API reference, or anything "
        "recent — not for things resolvable by reading the workspace or introspecting an "
        "installed package (use inspect_module for that).",
        obj(
            {
                "query": {"type": "string", "description": "Search terms, as you would type them."},
                "max_results": {"type": "integer", "description": "Default 5, max 10."},
            },
            ["query"],
        ),
        web_search,
    ),
]


# Queries any working search engine answers. A canary, not a benchmark: one hit is enough.
_CANARY_QUERIES = ("python programming language", "wikipedia")


def search_health() -> tuple[bool | None, str]:
    """Whether web_search would return anything right now, checked with a canary query.

    (None, why) when no backend is configured (web_search then says so itself),
    (True, backend) when a canary query returns results, (False, why) when it
    returns none or errors. Every upstream engine of the local SearXNG was
    blocking this machine's IP for two days of sweeps, and nothing noticed
    (DESIGN-LOG §89.1); the bench runs this before a sweep.
    """
    backend = search_backend()
    if backend is None:
        return None, "no search backend configured (SEARXNG_URL or BRAVE_API_KEY)"
    search = _searxng if backend == "searxng" else _brave
    failed: list[str] = []
    for query in _CANARY_QUERIES:
        found = search(query, 3)
        if isinstance(found, str):
            return False, f"{backend}: {found}"
        results, _, down = found
        if results:
            return True, backend
        failed = down or failed
    why = f"engines failing: {', '.join(failed)}" if failed else "no engine returned anything"
    return False, f"{backend} returned no results for a canary query ({why})"
