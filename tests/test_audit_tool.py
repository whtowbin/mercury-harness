"""audit_previous_runs: the live, callable version of trace auditing --
deterministic checks always, an optional LLM judgment pass on request, both
reading a workspace's own earlier trace files rather than the current run's."""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.tools import ToolContext
from agent.tools.audit import audit_previous_runs
from agent.trace import Trace
from agent.workspace import Workspace


class _FakeClient:
    """Same shape as tests/test_audit.py's: records what it was called with,
    returns a fixed reply. A real model call is exactly what this suite must
    never run — it is checking the plumbing around the call, not GLM."""

    def __init__(self, content: str = "[]", cost_usd: float = 0.0002):
        self.content = content
        self.cost_usd = cost_usd
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return Reply(content=self.content, usage=Usage(80, 20, self.cost_usd, 0.4))


def _write_prior_trace(trace_dir, run_id: str, goal: str, *, undeclared: str | None = None) -> None:
    """A minimal prior run's trace file: a plan step that declares one file,
    optionally an edit to a second, undeclared one, and a final result."""
    events = [
        {"ts": 1.0, "kind": "run_start", "goal": goal, "workspace": str(trace_dir)},
        {
            "ts": 2.0,
            "kind": "plan",
            "steps": [
                {"id": "s1", "writes": ["main.py"], "verify": "uv run pytest -q test_main.py"},
            ],
        },
        {"ts": 3.0, "kind": "edit_file", "path": "main.py"},
    ]
    if undeclared:
        events.append({"ts": 3.5, "kind": "edit_file", "path": undeclared})
    events += [
        {"ts": 4.0, "kind": "final_check", "ok": undeclared is None},
        {"ts": 5.0, "kind": "totals", "cost_usd": 0.01, "calls": 3},
    ]
    path = trace_dir / f"run-{run_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


@pytest.fixture
def ctx(tmp_path):
    """A live run's context: its own trace file in the same directory prior
    runs would have written to -- audit_previous_runs finds siblings via
    ctx.trace.path.parent, exactly as a real run's ctx would."""
    workspace = Workspace(tmp_path)
    trace = Trace(str(tmp_path / "traces"), run_id="current", echo=False)
    return ToolContext(
        workspace=workspace,
        approver=Approver(auto_approve=True),
        trace=trace,
    )


def test_no_previous_runs_reports_clearly(ctx):
    result = audit_previous_runs(ctx)
    assert "no previous runs found" in result


def test_finds_a_previous_run_and_flags_undeclared_write(ctx):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "prior1", "fix the scoring bug", undeclared="sprites.py")

    result = audit_previous_runs(ctx)
    assert "fix the scoring bug" in result
    assert "failed" in result
    assert "undeclared writes" in result
    assert "sprites.py" in result


def test_current_trace_file_is_excluded(ctx):
    """The tool must never try to audit the trace it is itself being called
    from -- that file is mid-write and describes the run in progress."""
    trace_dir = ctx.trace.path.parent
    result = audit_previous_runs(ctx)
    assert ctx.trace.path.name not in result
    assert "no previous runs found" in result
    assert trace_dir == ctx.trace.path.parent  # sanity: looked in the right place


def test_deep_false_never_touches_the_client(ctx):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "prior1", "add a login form")
    client = _FakeClient()
    ctx.audit_client = client

    audit_previous_runs(ctx, deep=False)
    assert client.calls == []


def test_deep_true_calls_the_client_and_records_cost(ctx):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "prior1", "add a login form")
    findings = [
        {
            "category": "model_weakness",
            "summary": "used a global instead of a param",
            "evidence": "see step s1",
        }
    ]
    client = _FakeClient(content=json.dumps(findings), cost_usd=0.0003)
    ctx.audit_client = client

    result = audit_previous_runs(ctx, deep=True)

    assert len(client.calls) == 1
    assert "model_weakness" in result
    assert "used a global instead of a param" in result
    assert "0.0003" in result

    # Cost must land in this run's own trace accounting, not just the tool's
    # return string, or a run's totals would silently under-report what it
    # actually spent.
    ctx.trace._fh.flush()
    written = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    llm_calls = [e for e in written if e.get("kind") == "llm_call"]
    assert len(llm_calls) == 1
    assert llm_calls[0]["phase"] == "audit_previous_runs"
    assert llm_calls[0]["cost_usd"] == 0.0003


def test_deep_true_without_a_client_degrades_instead_of_crashing(ctx):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "prior1", "add a login form")
    assert ctx.audit_client is None

    result = audit_previous_runs(ctx, deep=True)
    assert "unavailable" in result


def test_query_filters_by_goal_substring(ctx):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "prior1", "fix the scoring bug")
    _write_prior_trace(trace_dir, "prior2", "add a settings menu")

    result = audit_previous_runs(ctx, query="settings")
    assert "settings menu" in result
    assert "scoring bug" not in result


def test_limit_caps_examined_runs(ctx):
    trace_dir = ctx.trace.path.parent
    for i in range(5):
        _write_prior_trace(trace_dir, f"prior{i}", f"task number {i}")

    result = audit_previous_runs(ctx, limit=2)
    assert result.count("## run-prior") == 2


def test_malformed_trace_file_does_not_crash_the_rest(ctx):
    """load_trace already tolerates a malformed line (a trace from a killed
    process is still worth reading up to the truncation) -- this checks the
    tool doesn't regress that by choking on the now-empty/partial result."""
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "good", "a working prior run")
    (trace_dir / "run-broken.jsonl").write_text("not json at all\n")

    result = audit_previous_runs(ctx)
    assert "a working prior run" in result
    assert "run-broken.jsonl" in result  # still listed, just with nothing to report


def test_genuinely_unreadable_trace_file_reports_instead_of_crashing(ctx, monkeypatch):
    trace_dir = ctx.trace.path.parent
    _write_prior_trace(trace_dir, "good", "a working prior run")
    _write_prior_trace(trace_dir, "bad", "a run whose file goes missing")

    from agent.tools import audit as audit_tool

    real_load_trace = audit_tool.tracetools.load_trace

    def _flaky_load(path):
        if "bad" in str(path):
            raise OSError("permission denied")
        return real_load_trace(path)

    monkeypatch.setattr(audit_tool.tracetools, "load_trace", _flaky_load)

    result = audit_previous_runs(ctx)
    assert "a working prior run" in result
    assert "could not read trace" in result
