"""What the model did after a gate held one of its calls (`harness trace holds`).

Built at the third hand-reading of the same question (DESIGN-LOG §49.1, §54.3,
§57.1). The first version counted an edit to a *test* after a tdd-gate hold as a
re-send, which hid that 22 of 32 such holds were answered by writing the test.
"""

from __future__ import annotations

import json

from agent import tracetools as T


def hold(kind="tdd_gate_held", phase="execute:s", fp="", tool="edit_file", **extra):
    return {"kind": kind, "phase": phase, "iteration": 0, "tool": tool, "fp": fp, **extra}


def call(tool, phase="execute:s", fp="", path="", **extra):
    event = {"kind": "tool_call", "phase": phase, "tool": tool, "fp": fp, **extra}
    if path:
        event["args"] = {"path": path}
    return event


LLM = {"kind": "llm_call", "phase": "execute:s"}


def test_a_re_sent_call_with_the_same_fingerprint_is_unchanged():
    rows = T.hold_responses([hold(fp="a"), LLM, call("edit_file", fp="a", path="p.py")])
    assert rows[0]["response"] == "resent_unchanged" and rows[0]["at_once"]


def test_a_re_sent_call_with_a_different_fingerprint_is_changed():
    rows = T.hold_responses([hold(fp="a"), call("edit_file", fp="b", path="p.py")])
    assert rows[0]["response"] == "resent_changed"


def test_a_trace_without_fingerprints_says_resent_not_unchanged():
    rows = T.hold_responses([hold(), call("edit_file", path="p.py")])
    assert rows[0]["response"] == "resent"


def test_editing_a_test_after_a_tdd_hold_is_the_reaction_it_asks_for():
    rows = T.hold_responses([hold(), call("edit_file", path="test_p.py")])
    assert rows[0]["response"] == "wrote_test"


def test_editing_a_test_after_a_hold_on_a_test_edit_is_not_a_reaction():
    """A pinned-test hold refuses an edit to a test; another test edit is the same
    action again."""
    rows = T.hold_responses([hold("pinned_test_held"), call("edit_file", path="test_p.py")])
    assert rows[0]["response"] == "resent"


def test_what_the_model_did_in_between_is_listed_in_order():
    rows = T.hold_responses(
        [hold(), call("read_file"), call("run_tests"), call("edit_file", path="p.py")]
    )
    assert rows[0]["between"] == ["read_file", "run_tests"]
    assert not rows[0]["at_once"]


def test_a_phase_that_never_edits_again_gave_up():
    rows = T.hold_responses([hold(), call("read_file"), call("finish_step")])
    assert rows[0]["response"] == "gave_up" and rows[0]["between"] == ["read_file", "finish_step"]


def test_the_scan_stops_at_the_next_phase():
    rows = T.hold_responses(
        [hold(phase="execute:a"), call("edit_file", phase="execute:b", path="p.py")]
    )
    assert rows[0]["response"] == "gave_up"


def test_a_plan_hold_looks_for_the_next_submit_plan():
    rows = T.hold_responses(
        [
            hold("probe_gate_held", "plan", tool="submit_plan", fp="x"),
            call("probe_python", "plan"),
            call("submit_plan", "plan", fp="y"),
        ]
    )
    assert rows[0]["response"] == "resent_changed"
    assert rows[0]["between"] == ["probe_python"]


def test_every_gate_kind_is_recognised():
    kinds = [
        "tdd_gate_held",
        "repro_passed_held",
        "pinned_test_held",
        "premise_gate_held",
        "probe_gate_held",
        "repro_gate_held",
    ]
    assert [r["kind"] for r in T.hold_responses([hold(k) for k in kinds])] == kinds


def test_a_trace_with_no_holds_yields_nothing():
    assert T.hold_responses([call("edit_file", path="p.py")]) == []


def test_the_summary_splits_at_once_from_after_investigating():
    rows = T.hold_responses(
        [
            hold(),
            call("edit_file", path="p.py"),
            hold(),
            call("run_tests"),
            call("edit_file", path="p.py"),
        ]
    )
    assert T.summarize_holds(rows) == {
        "tdd_gate_held": {"resent:at_once": 1, "resent:after_investigating": 1}
    }


def test_a_directory_is_read_and_each_row_names_its_trace(tmp_path):
    (tmp_path / "run-a.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [hold(), call("edit_file", path="p.py")])
    )
    rows = T.holds_in(tmp_path)
    assert [r["trace"] for r in rows] == ["run-a.jsonl"]


def test_recursive_reaches_sub_directories(tmp_path):
    sub = tmp_path / "batch"
    sub.mkdir()
    (sub / "run-b.jsonl").write_text(json.dumps(hold()))
    assert T.holds_in(tmp_path) == []
    assert len(T.holds_in(tmp_path, recursive=True)) == 1


def test_the_held_call_and_the_dispatched_call_share_a_fingerprint_in_a_real_loop(tmp_path):
    """The fingerprint on the hold event and on the later tool_call event come from
    the same function, so a re-sent call matches exactly."""
    from agent.approval import Approver
    from agent.llm import Reply, Usage
    from agent.runner import run_tool_loop
    from agent.tools import ToolContext, execution_registry
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "pricing.py").write_text("x = 1\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    edit = {
        "id": "c1",
        "function": {
            "name": "edit_file",
            "arguments": json.dumps(
                {"path": "pricing.py", "old_string": "x = 1", "new_string": "x = 2"}
            ),
        },
    }
    replies = [
        Reply(content="", tool_calls=[edit], usage=Usage(1, 1, 0.0, 0.1)),
        Reply(content="", tool_calls=[edit], usage=Usage(1, 1, 0.0, 0.1)),
        Reply(
            content="",
            tool_calls=[
                {
                    "id": "f",
                    "function": {
                        "name": "finish_step",
                        "arguments": json.dumps({"summary": "s", "verified": "false"}),
                    },
                }
            ],
            usage=Usage(1, 1, 0.0, 0.1),
        ),
    ]

    class Client:
        def chat(self, *a, **k):
            return replies.pop(0)

    run_tool_loop(
        client=Client(),
        model="m",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
        tdd_tests=["t"],
    )
    ctx.trace._fh.flush()
    rows = T.hold_responses([json.loads(x) for x in ctx.trace.path.read_text().splitlines()])
    assert [r["response"] for r in rows] == ["resent_unchanged"]
    assert rows[0]["paths"] == ["pricing.py"]


def test_the_command_prints_counts_per_kind_and_a_detail_line_per_hold(tmp_path):
    from typer.testing import CliRunner

    from agent.cli import app

    (tmp_path / "run-a.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [hold(), call("edit_file", path="test_p.py")])
    )
    runner = CliRunner()
    plain = runner.invoke(app, ["trace", "holds", str(tmp_path)])
    assert plain.exit_code == 0
    assert "tdd_gate_held" in plain.output and "wrote_test:at_once" in plain.output
    detailed = runner.invoke(app, ["trace", "holds", str(tmp_path), "--detail"])
    assert "run-a.jsonl" in detailed.output
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no holds" in runner.invoke(app, ["trace", "holds", str(empty)]).output


def test_a_call_site_hold_is_read_like_the_other_plan_holds():
    rows = T.hold_responses(
        [
            hold("callsite_gate_held", phase="plan", tool="submit_plan", fp="a"),
            call("submit_plan", phase="plan", fp="b"),
        ]
    )
    assert [(r["kind"], r["response"]) for r in rows] == [("callsite_gate_held", "resent_changed")]
