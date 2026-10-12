"""Arguments the model sends in the wrong shape.

21% of the tool arguments in the recorded traces are the wrong JSON type. Bools
and ints were already repaired; arrays were not, and their failure is silent
rather than loud. A string is iterable, so `read_files(paths='["a.py","b.py"]')`
looked for one file named `["a.py", "b.py"]`, and `check_code(paths='["m.py"]')`
walked the string character by character and reported thirteen fabricated
"E902 No such file" diagnostics — for `[`, `"`, `d`, `m` — which the model then
tried to fix. It lands on exactly the plural tools batching depends on.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.runner import _coerce_args, _dispatch
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("A = 1\n")
    (root / "b.py").write_text("B = 2\n")
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _tool(name):
    return execution_registry().get(name)


# -- arrays ---------------------------------------------------------------


def test_an_array_sent_as_a_json_string_is_parsed():
    got, errors = _coerce_args(_tool("read_files"), {"paths": '["a.py", "b.py"]'})
    assert got["paths"] == ["a.py", "b.py"]
    assert errors == []


def test_a_bare_string_becomes_a_one_element_list():
    """`paths="a.py"` plainly means ["a.py"]; every other reading is a bug."""
    got, errors = _coerce_args(_tool("read_files"), {"paths": "a.py"})
    assert got["paths"] == ["a.py"]
    assert errors == []


def test_a_real_array_is_left_alone():
    got, errors = _coerce_args(_tool("read_files"), {"paths": ["a.py"]})
    assert got["paths"] == ["a.py"]
    assert errors == []


def test_an_array_of_objects_is_parsed():
    edits = [{"path": "a.py", "old_string": "A", "new_string": "B"}]
    got, errors = _coerce_args(_tool("edit_files"), {"edits": json.dumps(edits)})
    assert got["edits"] == edits
    assert errors == []


def test_malformed_json_that_does_not_look_like_json_is_passed_through_untouched():
    """A genuinely bad argument that was never meant to be JSON (e.g. a bare
    filename) must still reach the tool and get the ordinary error, not be
    silently reshaped into something plausible, and not raise a parse error —
    it never looked like structured data in the first place."""
    got, errors = _coerce_args(_tool("read_files"), {"paths": "not-json-at-all"})
    assert got["paths"] == ["not-json-at-all"]  # bare string -> one-element list
    assert errors == []


def test_json_that_fails_to_parse_is_reported_as_an_error():
    """Unlike a bare string, something that starts with `[` and fails to parse
    was clearly *meant* to be JSON — surfacing json.loads's own message here is
    what lets the model fix its own escaping mistake on retry instead of
    resending the same malformed payload blind."""
    got, errors = _coerce_args(_tool("read_files"), {"paths": '["a.py", '})
    assert got["paths"] == '["a.py", '  # left untouched; tool's own error will fire too
    assert len(errors) == 1
    assert "paths" in errors[0] and "failed to parse" in errors[0]


def test_a_json_object_where_an_array_is_declared_is_not_accepted():
    got, errors = _coerce_args(_tool("read_files"), {"paths": '{"a": 1}'})
    assert got["paths"] == '{"a": 1}'
    assert errors == []  # valid JSON, just the wrong shape -> tool's own error fires


def test_unescaped_newline_inside_a_double_encoded_edit_is_reported_clearly():
    """The real failure mode found in a traced run: edit_files given a
    double-encoded `edits` string containing a literal (unescaped) newline
    inside old_string. json.loads rejects this as an invalid control
    character; that message should reach the model instead of the generic
    'edits must be a JSON array' error it can't act on."""
    bad = '[{"path": "m.py", "old_string": "line1\nline2", "new_string": "x"}]'
    got, errors = _coerce_args(_tool("edit_files"), {"edits": bad})
    assert got["edits"] == bad
    assert len(errors) == 1
    assert "edits" in errors[0] and "control character" in errors[0]


def test_dispatch_reports_the_parse_error_without_calling_the_tool(ctx):
    """End to end: a malformed double-encoded edits array must not reach
    edit_files at all (so nothing is silently 0/N-applied), and the model
    should see the real json.loads error instead of a generic type error."""
    (ctx.workspace.root / "m.py").write_text("x = 1\n")
    bad_edits = '[{"path": "m.py", "old_string": "x = 1\nmore", "new_string": "y"}]'
    out = _dispatch(
        execution_registry(), ctx, "edit_files", json.dumps({"edits": bad_edits}), ctx.trace, "t"
    )
    assert "failed to parse" in out
    assert "control character" in out
    # nothing should have been touched -- the tool itself never ran
    assert (ctx.workspace.root / "m.py").read_text() == "x = 1\n"


# -- the failures this was found through ----------------------------------


def test_read_files_reads_both_files_when_sent_a_string_array(ctx):
    out = _dispatch(
        execution_registry(),
        ctx,
        "read_files",
        json.dumps({"paths": json.dumps(["a.py", "b.py"])}),
        ctx.trace,
        "t",
    )
    assert "a.py" in out and "b.py" in out
    assert "no such file" not in out


def test_check_code_does_not_invent_diagnostics_from_the_string(ctx):
    """The character-by-character walk produced an E902 per punctuation mark."""
    out = _dispatch(
        execution_registry(),
        ctx,
        "check_code",
        json.dumps({"paths": json.dumps(["a.py"])}),
        ctx.trace,
        "t",
    )
    assert "E902" not in out, out


def test_bools_and_ints_still_work():
    assert _coerce_args(_tool("run_tests"), {"verbose": "false"})[0]["verbose"] is False
    assert _coerce_args(_tool("read_file"), {"limit": "40"})[0]["limit"] == 40


# -- a tool's own "error: ..." return is now logged, not just its input args --
# Previously only a raised exception got a trace event (tool_error); a tool
# that fails by returning a string starting with "error:" -- the ordinary
# path, since tools must never crash the run -- left nothing in the trace
# beyond the (200-char-truncated) input args. Reconstructing what actually
# went wrong had to be done from indirect evidence instead of just reading
# what the tool said.


def _events(ctx, kind):
    lines = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    return [line for line in lines if line["kind"] == kind]


def test_a_tools_own_error_return_is_logged(ctx):
    _dispatch(
        execution_registry(),
        ctx,
        "read_file",
        json.dumps({"path": "does_not_exist.py"}),
        ctx.trace,
        "t",
    )
    failures = _events(ctx, "tool_call_failed")
    assert len(failures) == 1
    assert failures[0]["tool"] == "read_file"
    assert "no such file" in failures[0]["result"]


def test_a_tools_successful_return_is_not_logged_as_a_failure(ctx):
    _dispatch(execution_registry(), ctx, "read_file", json.dumps({"path": "a.py"}), ctx.trace, "t")
    assert _events(ctx, "tool_call_failed") == []


def test_the_logged_result_is_capped(ctx):
    from agent.tools.base import Registry, Tool, obj

    huge_error = "error: " + "x" * 5_000
    fake_registry = Registry([Tool("bad_tool", "always fails", obj({}), lambda _ctx: huge_error)])
    _dispatch(fake_registry, ctx, "bad_tool", "{}", ctx.trace, "t")
    failures = _events(ctx, "tool_call_failed")
    assert len(failures) == 1
    assert len(failures[0]["result"]) <= 2_000
