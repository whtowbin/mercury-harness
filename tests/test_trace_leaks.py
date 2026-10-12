"""Which runs reached for a benchmark's own sources (`harness trace leaks`).

Built at the second hand-run of this check (DESIGN-LOG §79.1, §80): the event
shapes below are the ones the SciCode and django traces recorded.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from agent import tracetools as T
from agent.cli import app

DENY = ("huggingface.co/datasets/SciCode1", "github.com/scicode-bench")
DEV = "https://huggingface.co/datasets/SciCode1/SciCode/resolve/main/problems_dev.jsonl"


def fetch(url, chars=None, phase="plan"):
    out = [{"kind": "tool_call", "phase": phase, "tool": "fetch_url", "args": {"url": url}}]
    out.append({"kind": "log", "message": f"fetching {url}"})
    if chars is not None:
        out.append({"kind": "fetch_url", "url": url, "chars": chars})
    return out


def search(query, phase="plan"):
    return [{"kind": "tool_call", "phase": phase, "tool": "web_search", "args": {"query": query}}]


def test_a_fetch_that_returned_text_is_counted_with_its_size():
    hits = T.web_hits(fetch(DEV, chars=279_558), DENY)
    assert hits == [
        {"phase": "plan", "tool": "fetch_url", "target": DEV, "returned": 279_558, "shown": 8_000}
    ]


def test_a_failed_or_refused_fetch_returned_nothing():
    refused = [*fetch(DEV), {"kind": "web_denied", "tool": "fetch_url", "url": DEV}]
    assert T.web_hits(refused, DENY)[0]["returned"] == 0
    assert T.web_hits(fetch(DEV), DENY)[0]["returned"] == 0


def test_a_wrapped_url_is_caught():
    hits = T.web_hits(fetch("https://r.jina.ai/" + DEV, chars=100), DENY)
    assert len(hits) == 1


def test_ordinary_fetches_and_searches_are_not_hits():
    trace = fetch("https://numpy.org/doc/stable/", chars=5_000) + search("numpy qr decomposition")
    assert T.web_hits(trace, DENY, query=r"scicode") == []


def test_searches_count_only_with_a_query_pattern():
    trace = search("github SciCode ewald_summation reference solution")
    assert T.web_hits(trace, DENY) == []
    assert T.web_hits(trace, DENY, query=r"scicode")[0]["tool"] == "web_search"


def _write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_the_command_reports_runs_that_got_text_and_runs_that_only_tried(tmp_path):
    _write(tmp_path / "d2" / "4" / "run-a.jsonl", fetch(DEV, chars=279_558))
    _write(tmp_path / "d2" / "7" / "run-b.jsonl", fetch(DEV))
    _write(tmp_path / "d4" / "1" / "run-c.jsonl", fetch("https://numpy.org/", chars=10))
    assert set(T.leaks_in(tmp_path, DENY)) == {"d2/4/run-a.jsonl", "d2/7/run-b.jsonl"}

    args = ["trace", "leaks", str(tmp_path)] + [a for d in DENY for a in ("--deny", d)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())  # xdist workers wrap at another column
    assert "2 runs reached for benchmark sources; 1 got text back" in output
    assert "got text" in output and "tried" in output


def test_the_command_needs_a_pattern(tmp_path):
    result = CliRunner().invoke(app, ["trace", "leaks", str(tmp_path)])
    assert result.exit_code != 0


def test_a_batched_fetch_is_paired_with_its_own_result_not_its_siblings():
    other = "https://numpy.org/doc/"
    trace = [
        {"kind": "tool_call", "phase": "plan", "tool": "fetch_url", "args": {"url": DEV}},
        {"kind": "tool_call", "phase": "plan", "tool": "fetch_url", "args": {"url": other}},
        {"kind": "fetch_url", "url": other, "chars": 9_000},  # DEV itself failed
        {"kind": "llm_call", "phase": "plan"},
    ]
    assert T.web_hits(trace, DENY)[0]["returned"] == 0


def test_a_trace_copied_into_failed_chats_is_counted_once(tmp_path):
    _write(tmp_path / "d2" / "traces" / "4" / "run-a.jsonl", fetch(DEV, chars=10))
    _write(tmp_path / "d2" / "4" / "failed-chats" / "run-a.jsonl", fetch(DEV, chars=10))
    assert list(T.leaks_in(tmp_path, DENY)) == ["d2/traces/4/run-a.jsonl"]


def test_shown_is_what_the_model_asked_for_not_the_whole_page():
    """d2/7 fetched a 1.87M-character page with max_chars=100: it saw 100."""
    trace = fetch(DEV, chars=1_868_211)
    trace[0]["args"]["max_chars"] = "100"
    assert T.web_hits(trace, DENY)[0] == {
        "phase": "plan",
        "tool": "fetch_url",
        "target": DEV,
        "returned": 1_868_211,
        "shown": 100,
    }
    assert T.web_hits(fetch(DEV, chars=50_000), DENY)[0]["shown"] == 8_000  # the default


def test_code_that_names_a_benchmark_source_is_counted():
    """run_python and run_bash reach the web too (DESIGN-LOG §93)."""
    code = f"import requests\nprint(requests.get('https://r.jina.ai/{DEV}').text)"
    events = [
        {"kind": "tool_call", "phase": "execute:s", "tool": "run_python", "args": {"code": code}},
        {"kind": "tool_call", "phase": "fix:1", "tool": "run_bash", "args": {"command": "ls"}},
    ]
    hits = T.web_hits(events, DENY)
    assert [(h["tool"], h["phase"]) for h in hits] == [("run_python", "execute:s")]


def test_a_search_result_on_the_deny_list_is_counted_as_seen():
    """The model reads a result's snippet without fetching it (DESIGN-LOG §93.2)."""
    events = [
        {
            "kind": "web_search",
            "query": "ewald summation",
            "hits": [
                {"url": DEV, "snippet": "problem_id 10 ground_truth_code"},
                {"url": "https://en.wikipedia.org/wiki/Ewald_summation", "snippet": "sums"},
            ],
        }
    ]
    hits = T.web_hits(events, DENY)
    assert [(h["tool"], h["target"], h["shown"]) for h in hits] == [("web_search result", DEV, 31)]
