"""Tool calls that failed for reasons the harness can fix itself (DESIGN-LOG §63).

Across 4,300 traced runs no call ever named an unknown tool; the failures were
arguments that did not parse and tools offered that could only return an
error. submit_plan's arguments failed to parse in 12% of runs (657 times,
510 of them its free-text `draft`), each costing a whole planning round trip;
remember/recall were offered with memory off and failed 84 times.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import _coerce_args, _dispatch, run_tool_loop
from agent.tools import ToolContext, execution_registry, planning_registry
from agent.trace import Trace
from agent.workspace import Workspace


def _tool(name):
    return execution_registry().get(name) or planning_registry().get(name)


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("A = 1\n")
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- repairs that cannot change what the value says ----------------------


@pytest.mark.parametrize(
    "sent",
    [
        '["a.py", "b.py",]',  # "Illegal trailing comma"
        '["a.py", "b.py"]]',  # "Extra data": a complete array, then a stray bracket
        '["a.py", "b.py"].',
        "['a.py', 'b.py']",  # Python-style quotes
    ],
)
def test_a_near_miss_array_is_repaired(sent):
    got, errors = _coerce_args(_tool("read_files"), {"paths": sent})
    assert errors == []
    assert got["paths"] == ["a.py", "b.py"]


def test_extra_data_that_is_more_data_is_not_dropped():
    """Taking the first array and discarding a second would lose content."""
    _, errors = _coerce_args(_tool("read_files"), {"paths": '["a.py"] ["b.py"]'})
    assert len(errors) == 1


def test_python_string_joining_is_never_used_as_a_repair():
    """Python reads `"b" "c"` as "bc"; JSON calls it a missing comma. Joining
    them would change the value, so it stays an error."""
    _, errors = _coerce_args(_tool("read_files"), {"paths": '["a.py", "b" "c"]'})
    assert len(errors) == 1


# -- submit_plan's draft is notes: salvage, do not reject ------------------


def test_a_malformed_draft_is_kept_as_notes_instead_of_rejecting_the_plan():
    draft = '["understood: fix slug.py", "the repro passed", "plan: one step",'
    draft += ' "note: keep "slugify" signature"]'
    got, errors = _coerce_args(_tool("submit_plan"), {"draft": draft, "steps": []})
    assert errors == []
    assert got["draft"][0] == "understood: fix slug.py"
    assert "plan: one step" in got["draft"]


def test_malformed_steps_are_still_an_error():
    """Steps are acted on; reshaping a bad value there is silent nonsense."""
    _, errors = _coerce_args(
        _tool("submit_plan"), {"draft": ["x"], "steps": '[{"id": "s1" "title": "t"}]'}
    )
    assert len(errors) == 1 and errors[0].startswith("steps:")


def test_the_same_salvage_does_not_apply_to_other_tools():
    _, errors = _coerce_args(_tool("read_files"), {"paths": '["a.py", "b" x]'})
    assert len(errors) == 1


# -- the parse-error event carries the full arguments ---------------------


def test_a_parse_error_records_the_untruncated_arguments(ctx):
    edits = '[{"path": "a.py", "old_string": "' + "x" * 500 + '\nA", "new_string": "B"}]'
    raw = json.dumps({"edits": edits})
    out = _dispatch(execution_registry(), ctx, "edit_files", raw, ctx.trace, "execute:s")
    assert out.startswith("error:")
    ctx.trace._fh.flush()
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    parse = next(e for e in events if e["kind"] == "tool_arg_parse_error")
    assert parse["raw"] == raw


def test_the_tool_call_event_names_arguments_sent_as_json_strings(ctx):
    raw = json.dumps({"paths": '["a.py"]'})
    _dispatch(execution_registry(), ctx, "read_files", raw, ctx.trace, "execute:s")
    _dispatch(
        execution_registry(), ctx, "read_files", '{"paths": ["a.py"]}', ctx.trace, "execute:s"
    )
    ctx.trace._fh.flush()
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    calls = [e for e in events if e["kind"] == "tool_call"]
    assert calls[0]["encoded"] == ["paths"]
    assert "encoded" not in calls[1]


# -- tools the run cannot serve are not offered ---------------------------


class _Recording:
    def __init__(self):
        self.offered: list[set[str]] = []

    def chat(self, model, messages, tools=None, **_kw):
        self.offered.append({t["function"]["name"] for t in tools or []})
        call = {"id": "c1", "function": {"name": "submit_plan", "arguments": "{}"}}
        return Reply(content="", tool_calls=[call], usage=Usage(10, 5, 0.0, 0.1))


def _loop(ctx, client):
    run_tool_loop(
        client=client,
        model="fake",
        phase="plan",
        system="s",
        user="u",
        registry=planning_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=1,
    )


def test_memory_tools_are_not_offered_without_a_memory_store(ctx):
    client = _Recording()
    _loop(ctx, client)
    assert not {"remember", "recall"} & client.offered[0]
    assert "submit_plan" in client.offered[0]


def test_memory_tools_are_offered_when_memory_exists(ctx, tmp_path):
    from agent.memory import MemoryStore

    ctx.memory = MemoryStore(ctx.workspace.root)
    client = _Recording()
    _loop(ctx, client)
    assert {"remember", "recall"} <= client.offered[0]


# -- strict schemas ------------------------------------------------------


def test_strict_is_added_to_every_tool_and_nothing_else_changes():
    from agent.llm import strict_schemas

    tools = [t.schema() for t in planning_registry().tools()]
    strict = strict_schemas(tools)
    assert all(t["function"]["strict"] is True for t in strict)
    for before, after in zip(tools, strict, strict=True):
        assert "strict" not in before["function"]
        assert {k: v for k, v in after["function"].items() if k != "strict"} == before["function"]


@pytest.fixture
def capturing_server():
    """A real local endpoint that records each request body and answers with
    one plain completion, so the test sees exactly what the client sends."""
    import http.server
    import threading

    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            bodies.append(json.loads(self.rfile.read(length)))
            reply = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", bodies
    server.shutdown()


@pytest.mark.parametrize("strict", [False, True])
def test_the_client_sends_strict_only_when_configured(strict, capturing_server):
    from agent.llm import OpenRouterClient

    url, bodies = capturing_server
    client = OpenRouterClient(
        "k", base_url=url, stream=False, cache_prompts=False, strict_tools=strict
    )
    tools = [planning_registry().get("submit_plan").schema()]
    client.chat(model="m", messages=[{"role": "user", "content": "x"}], tools=tools)
    assert ("strict" in bodies[0]["tools"][0]["function"]) is strict
    assert bodies[0]["tools"][0]["function"]["parameters"] == tools[0]["function"]["parameters"]


def test_the_tool_call_event_names_scalars_sent_as_strings(ctx):
    raw = json.dumps({"path": "a.py", "limit": "40"})
    _dispatch(execution_registry(), ctx, "read_file", raw, ctx.trace, "execute:s")
    ctx.trace._fh.flush()
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    call = next(e for e in events if e["kind"] == "tool_call")
    assert call["mistyped"] == ["limit"]


def test_strict_tool_schemas_is_the_default():
    """On after DESIGN-LOG §63.1's A/B; mercury-no-strict-tools keeps the old
    behaviour runnable so a later sweep can price it again."""
    from agent.config import Settings
    from bench import configs

    assert Settings(workspace=".", goal="g").strict_tool_schemas is True
    assert configs.get("mercury-only").strict_tool_schemas is True
    assert configs.get("mercury-no-strict-tools").strict_tool_schemas is False


# -- the test watcher reads pytest run through run_bash --------------------


def _bash(command):
    return {"function": {"name": "run_bash", "arguments": json.dumps({"command": command})}}


@pytest.mark.parametrize(
    ("command", "output", "expected"),
    [
        ("uv run pytest -q", "exit=1 (0.2s)\nFAILED t.py::test_a - assert 1\n1 failed", "FAIL"),
        ("pytest -q | tail -3", "exit=0 (0.2s)\n1 failed, 2 passed in 0.1s", "FAIL"),
        ("pytest", "exit=2 (0.1s)\nERROR collecting t.py\n1 error during collection", "COLLECTION"),
        ("pytest -q", "exit=0 (0.1s)\n3 passed in 0.01s", "PASS"),
    ],
)
def test_a_run_bash_pytest_result_is_read_like_run_tests(command, output, expected):
    from agent.runner import _test_result_text

    assert _test_result_text(_bash(command), output).startswith(expected)


def test_other_shell_commands_are_not_test_results():
    from agent.runner import _test_result_text

    assert _test_result_text(_bash("ls -la"), "exit=0\n1 failed") is None


def test_a_pytest_command_that_never_ran_is_not_a_pass():
    """A real case from building this: an unquoted path with a space gave
    `exit=127`, which read as a pass and would have reset the watcher."""
    from agent.runner import _test_result_text

    out = "exit=127 (0.0s)\n[stderr]\n/bin/sh: /Users/x/AI: No such file or directory"
    assert _test_result_text(_bash("/Users/x/AI Agent/python -m pytest"), out) is None


def test_the_same_failure_through_run_bash_now_gets_the_nudge(tmp_path):
    """Real pytest, real files: the same failing test across three run_bash
    runs with edits between them earns the same-failure nudge, as it always
    did through run_tests."""
    import sys

    root = tmp_path / "ws"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"
    )
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    import shlex

    pytest_cmd = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider test_calc.py"

    def call(name, **args):
        return {
            "id": f"c{len(args)}{name}",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    edits = [("a - b", "b - a"), ("b - a", "a - b - 0")]
    replies = [call("run_bash", command=pytest_cmd)]
    for old, new in edits:
        replies += [call("edit_file", path="calc.py", old_string=old, new_string=new)]
        replies += [call("run_bash", command=pytest_cmd)]
    replies += [call("finish_step", summary="gave up", verified="false")]

    class Client:
        def chat(self, model, messages, tools=None, **_kw):
            return Reply(content="", tool_calls=[replies.pop(0)], usage=Usage(10, 5, 0.0, 0.1))

    run_tool_loop(
        client=Client(),
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=10,
    )
    ctx.trace._fh.flush()
    kinds = [json.loads(line)["kind"] for line in ctx.trace.path.read_text().splitlines()]
    assert "same_failure_nudge" in kinds
