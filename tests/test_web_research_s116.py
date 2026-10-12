"""Settings.web_research: aids for finding facts on the web (DESIGN-LOG §116).

The nudge is driven through the real tool loop, with web_search answered by a real local HTTP
server speaking SearXNG's JSON, since the text appended to the model's result is under test.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import skills
from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


def _loaded() -> list:
    return skills.load()


def _by_name(loaded, name):
    return next(s for s in loaded if s.name == name)


def test_off_leaves_the_skills_exactly_as_before():
    loaded = _loaded()
    off = skills.for_web_research(loaded, False)
    assert [s.name for s in off] == [s.name for s in loaded if s.name != "web-research"]
    assert _by_name(off, "documenting-changes") is _by_name(loaded, "documenting-changes")


def test_on_offers_the_skill_and_documenting_skips_search_results():
    on = skills.for_web_research(_loaded(), True)
    research = _by_name(on, "web-research")
    assert research.matches("search the web for the 2018 blog post", ["README.md"])
    documenting = _by_name(on, "documenting-changes")
    assert not documenting.matches("collect the search results", [])
    assert documenting.matches("update README.md with the new flag", ["README.md"])
    assert _by_name(_loaded(), "documenting-changes").matches("collect the search results", [])


def _injected(tmp_path, on: bool, title: str, files: list[str]) -> str:
    """The skills block a real executor step gets, under the default (step-only) triggers."""
    from agent.phases.execute import _skills_for
    from agent.phases.plan import Step

    (tmp_path / "README.md").write_text("A research question.\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    ctx.skills = skills.for_web_research(_loaded(), on)
    step = Step(id="s", title=title, intent=title, acceptance="done", files=files)
    return _skills_for(ctx, "execute", step)


def test_a_root_readme_does_not_pull_in_documenting_changes(tmp_path):
    """Step-only triggers (the default since §117.2) keep a workspace README from firing it,
    with web_research on; a step that writes the README still gets it."""
    research = _injected(tmp_path, True, "find the population figure", ["answer.txt"])
    assert "## documenting-changes" not in research
    writing = _injected(tmp_path, True, "update README.md with the flag", ["README.md"])
    assert "## documenting-changes" in writing


# --- the nudge, through the real loop --------------------------------------------------------


@pytest.fixture
def searxng(monkeypatch):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = {
                "results": [
                    {"title": "Page", "url": "https://example.org/p", "content": "a snippet"}
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
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{server.server_port}")
    yield
    server.shutdown()


def call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Scripted:
    def __init__(self, turns):
        self.turns = list(turns)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return Reply(content="", tool_calls=self.turns.pop(0), usage=Usage(10, 5, 0.0, 0.1))


FINISH = call("finish_step", summary="done", verified="false")


def _run(tmp_path, turns, on=True):
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    ctx.web_research = on
    client = Scripted(turns)
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:find",
        system="s",
        user="u",
        registry=execution_registry("search the web for the answer", None),
        ctx=ctx,
        trace=trace,
        max_iters=len(turns),
    )
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    return client, [e for e in events if e["kind"] == "search_churn_nudge"]


def search(q):
    return [call("web_search", query=q)]


def test_the_third_search_without_a_fetch_is_told_to_open_a_result(tmp_path, searxng):
    client, nudges = _run(
        tmp_path,
        [
            search("eiffel tower height"),
            search("louvre opening year"),
            search("seine length km"),
            [FINISH],
        ],
    )
    assert [n["reason"] for n in nudges] == ["no_fetch"]
    third = client.sent[3][-1]["content"]
    assert "3 searches without opening a result" in third and "fetch_url" in third
    assert "[harness]" not in client.sent[2][-1]["content"]


def test_a_fetch_resets_the_count(tmp_path, searxng):
    _, nudges = _run(
        tmp_path,
        [
            search("eiffel tower height"),
            search("louvre opening year"),
            [call("fetch_url", url="https://example.org/p")],
            search("seine length km"),
            [FINISH],
        ],
    )
    assert nudges == []


def test_a_rephrased_query_is_told_so(tmp_path, searxng):
    client, nudges = _run(
        tmp_path,
        [
            search("annual rainfall coastal city 1998 report"),
            search("annual rainfall coastal city 1998"),
            [FINISH],
        ],
    )
    assert [n["reason"] for n in nudges] == ["repeat"]
    assert "rephrases an earlier one" in client.sent[2][-1]["content"]


def test_at_most_two_nudges_a_loop(tmp_path, searxng):
    turns = [search(q) for q in ("a b c", "d e f", "g h i", "j k l", "m n o", "p q r")]
    _, nudges = _run(tmp_path, [*turns, [FINISH]])
    assert len(nudges) == 2


def test_off_by_default_nothing_is_appended(tmp_path, searxng):
    client, nudges = _run(
        tmp_path,
        [
            search("eiffel tower height"),
            search("louvre opening year"),
            search("seine length km"),
            [FINISH],
        ],
        on=False,
    )
    assert nudges == [] and "[harness]" not in client.sent[3][-1]["content"]
    from agent.config import Settings

    assert Settings(workspace=".", goal="g").web_research is False
