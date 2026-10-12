"""Nudging the model to spend fewer round trips.

The measurement behind this: 89% of the agent's model time is time to first
token, so a round trip costs about 1.3s whatever it carries. Across 46 traced
runs the model put a single tool call in 92% of its turns and batched on 7.9%,
even though read_files and edit_files exist and EXECUTOR_SYSTEM has always
asked for batching. Perfect batching of consecutive reads and edits would have
removed 69 of 657 execute turns — 11%.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import (
    NO_PROGRESS_LIMIT,
    _batching_nudge,
    _nudge_about_batching,
    _nudge_about_no_progress,
    _tool_names,
    run_tool_loop,
)
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


class _Trace:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))


# -- when it fires --------------------------------------------------------


def test_two_reads_in_two_turns_is_nudged():
    assert "read_files" in _batching_nudge(["read_file"], ["read_file"])


def test_two_edits_in_two_turns_is_nudged():
    assert "edit_files" in _batching_nudge(["edit_file"], ["edit_file"])


# -- when it must not ----------------------------------------------------


def test_a_turn_that_already_batched_is_left_alone():
    assert _batching_nudge(["read_file"], ["read_file", "read_file"]) == ""
    assert _batching_nudge(["read_file", "read_file"], ["read_file"]) == ""


def test_two_different_tools_are_not_a_wasted_round_trip():
    assert _batching_nudge(["read_file"], ["edit_file"]) == ""
    assert _batching_nudge(["edit_file"], ["run_tests"]) == ""


def test_edit_then_test_is_never_nudged():
    """Two turns is the point: the second exists to see the result of the
    first. Batching those would delete the feedback."""
    assert _batching_nudge(["run_tests"], ["run_tests"]) == ""
    assert _batching_nudge(["edit_file"], ["run_tests"]) == ""
    assert _batching_nudge(["check_code"], ["check_code"]) == ""


def test_the_first_turn_of_a_loop_is_not_nudged():
    assert _batching_nudge([], ["read_file"]) == ""


# -- how it is delivered --------------------------------------------------


def test_the_nudge_rides_on_an_existing_tool_result():
    """It must cost no extra round trip: it is appended to a message already
    being sent, never sent as one of its own."""
    messages = [{"role": "tool", "content": "file contents"}]
    trace = _Trace()
    fired = _nudge_about_batching(
        messages, ["read_file"], ["read_file"], trace, "execute:x", budget=2
    )
    assert fired == 1
    assert len(messages) == 1, "no message was added, only extended"
    assert messages[0]["content"].startswith("file contents")
    assert "read_files" in messages[0]["content"]
    assert trace.events[0][0] == "batching_nudge"


def test_the_budget_stops_it_repeating():
    messages = [{"role": "tool", "content": "x"}]
    assert (
        _nudge_about_batching(messages, ["read_file"], ["read_file"], _Trace(), "p", budget=0) == 0
    )
    assert messages[0]["content"] == "x"


def test_nothing_is_appended_when_the_last_message_is_not_a_tool_result():
    messages = [{"role": "assistant", "content": "thinking"}]
    assert (
        _nudge_about_batching(messages, ["read_file"], ["read_file"], _Trace(), "p", budget=2) == 0
    )


def test_tool_names_survives_malformed_calls():
    assert _tool_names([{"function": {"name": "read_file"}}, {}, {"function": {}}]) == [
        "read_file",
        "",
        "",
    ]


# -- _nudge_about_no_progress ----------------------------------------------
# The gap this closes: REPEAT_LIMIT/_fingerprint only catch a call repeated
# with byte-identical arguments. A traced fix round burned its whole budget
# on four run_bash calls, each a slightly reworded inline Python probe --
# four different fingerprints, the exact-repeat detector never once fired --
# before one edit that made the failure count worse, with no iterations left
# to notice. This tracks progress via ctx.touched instead of call identity.


def test_fires_exactly_when_the_streak_crosses_the_limit():
    messages = [{"role": "tool", "content": "still 2 failed"}]
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, _Trace(), "execute:x", budget=2)
    assert fired == 1
    assert "4 tool calls in a row" in messages[0]["content"]


def test_does_not_fire_before_the_limit():
    messages = [{"role": "tool", "content": "x"}]
    fired = _nudge_about_no_progress(
        messages, NO_PROGRESS_LIMIT - 1, _Trace(), "execute:x", budget=2
    )
    assert fired == 0
    assert messages[0]["content"] == "x"


def test_does_not_refire_on_every_turn_past_the_limit():
    """Only the turn the streak *crosses* the threshold gets the nudge --
    otherwise it would repeat itself every remaining iteration of the step,
    burning tokens on a warning already given once this budget."""
    messages = [{"role": "tool", "content": "x"}]
    fired = _nudge_about_no_progress(
        messages, NO_PROGRESS_LIMIT + 3, _Trace(), "execute:x", budget=2
    )
    assert fired == 0
    assert messages[0]["content"] == "x"


def test_the_no_progress_budget_stops_it_repeating():
    messages = [{"role": "tool", "content": "x"}]
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, _Trace(), "p", budget=0)
    assert fired == 0
    assert messages[0]["content"] == "x"


def test_no_progress_appends_nothing_when_the_last_message_is_not_a_tool_result():
    messages = [{"role": "assistant", "content": "thinking"}]
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, _Trace(), "p", budget=2)
    assert fired == 0


@pytest.mark.parametrize("phase", ["plan", "review:1", "review:2"])
def test_read_only_phases_are_never_nudged(phase):
    """A live validation run against the platformer app caught this: the nudge
    fired during `plan` and `review:1`, both read-only by design (planning
    explores before submit_plan; review_registry has no write tools at all),
    where "make the edit now" is wrong advice, not just unhelpful."""
    messages = [{"role": "tool", "content": "x"}]
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, _Trace(), phase, budget=2)
    assert fired == 0
    assert messages[0]["content"] == "x"


@pytest.mark.parametrize("phase", ["execute:merge-tests", "fix:1"])
def test_execute_and_fix_phases_are_still_nudged(phase):
    messages = [{"role": "tool", "content": "x"}]
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, _Trace(), phase, budget=2)
    assert fired == 1


def test_it_rides_on_the_existing_message_like_the_batching_nudge_does():
    messages = [{"role": "tool", "content": "probe output"}]
    trace = _Trace()
    fired = _nudge_about_no_progress(messages, NO_PROGRESS_LIMIT, trace, "execute:x", budget=2)
    assert fired == 1
    assert len(messages) == 1, "no message was added, only extended"
    assert messages[0]["content"].startswith("probe output")
    assert trace.events[0][0] == "no_progress_nudge"


# -- _nudge_about_repeated_probing ------------------------------------------
# The gap this closes: _nudge_about_no_progress resets on any edit landing,
# but a live-traced run interleaved a guess-edit between each of four
# run_python probes of the same Parser/Tokenizer object -- the streak never
# crossed NO_PROGRESS_LIMIT, and trace_python (built to collapse exactly that
# probing into one call) sat unused the entire time. This tracks run_python's
# *subject* across the whole loop, not reset by a touched-file count.


def _probe_registry(names=("run_python", "trace_python")):
    from agent.tools.base import Registry, Tool, obj

    def _noop(ctx, **kw):
        return "ok"

    return Registry([Tool(n, "d", obj({}), _noop) for n in names])


def test_run_python_subjects_extracts_the_imported_name():
    from agent.runner import _run_python_subjects

    calls = [
        {
            "function": {
                "name": "run_python",
                "arguments": json.dumps({"code": "from expr import Parser\np = Parser('1')"}),
            }
        }
    ]
    assert _run_python_subjects(calls) == {"expr"}


def test_run_python_subjects_ignores_other_tools():
    from agent.runner import _run_python_subjects

    calls = [{"function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})}}]
    assert _run_python_subjects(calls) == set()


def test_run_python_subjects_survives_malformed_arguments():
    from agent.runner import _run_python_subjects

    calls = [{"function": {"name": "run_python", "arguments": "not json"}}]
    assert _run_python_subjects(calls) == set()


def test_fires_once_the_same_subject_is_probed_three_times():
    from collections import Counter

    from agent.runner import RUN_PYTHON_PROBE_LIMIT, _nudge_about_repeated_probing

    messages = [{"role": "tool", "content": "probe result"}]
    counts = Counter({"expr": RUN_PYTHON_PROBE_LIMIT})
    fired = _nudge_about_repeated_probing(
        messages, counts, _probe_registry(), _Trace(), "execute:x", budget=2
    )
    assert fired == 1
    assert "trace_python" in messages[0]["content"]
    assert "'expr'" in messages[0]["content"]
    assert counts["expr"] == 0, "must reset so three more probes earn another nudge"


def test_does_not_fire_below_the_limit():
    from collections import Counter

    from agent.runner import RUN_PYTHON_PROBE_LIMIT, _nudge_about_repeated_probing

    messages = [{"role": "tool", "content": "x"}]
    counts = Counter({"expr": RUN_PYTHON_PROBE_LIMIT - 1})
    fired = _nudge_about_repeated_probing(
        messages, counts, _probe_registry(), _Trace(), "execute:x", budget=2
    )
    assert fired == 0
    assert messages[0]["content"] == "x"


def test_does_not_fire_in_a_loop_it_cannot_widen():
    """No point suggesting a tool the kit does not offer. An execute or fix
    loop gets trace_python added (tests/test_kit_nudges.py); any other loop
    is read-only by design and is left alone."""
    from collections import Counter

    from agent.runner import RUN_PYTHON_PROBE_LIMIT, _nudge_about_repeated_probing

    messages = [{"role": "tool", "content": "x"}]
    counts = Counter({"expr": RUN_PYTHON_PROBE_LIMIT})
    fired = _nudge_about_repeated_probing(
        messages, counts, _probe_registry(names=("run_python",)), _Trace(), "review:1", budget=2
    )
    assert fired == 0
    assert messages[0]["content"] == "x"


def test_the_probe_budget_stops_it_repeating():
    from collections import Counter

    from agent.runner import RUN_PYTHON_PROBE_LIMIT, _nudge_about_repeated_probing

    messages = [{"role": "tool", "content": "x"}]
    counts = Counter({"expr": RUN_PYTHON_PROBE_LIMIT})
    fired = _nudge_about_repeated_probing(
        messages, counts, _probe_registry(), _Trace(), "execute:x", budget=0
    )
    assert fired == 0
    assert messages[0]["content"] == "x"


# -- end to end through run_tool_loop --------------------------------------
# The isolated tests above check the nudge helper in isolation; these check
# the streak is actually tracked correctly against real dispatch (ctx.touched
# growing or not) rather than against a hand-constructed integer.


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _call(name, **args):
    return {
        "id": f"c{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _thinking(text, *tool_calls):
    return Reply(content=text, tool_calls=list(tool_calls), usage=Usage(10, 5, 0.0, 0.1))


class _ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ) -> Reply:
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def test_reading_the_same_file_different_ways_never_writes_still_nudges(ctx):
    """The exact shape of the traced bug: distinct calls (different offsets),
    so REPEAT_LIMIT's exact-fingerprint match never fires, but nothing is
    ever written either. NO_PROGRESS_LIMIT catches what fingerprinting
    can't."""
    client = _ScriptedClient(
        [
            _thinking("checking a", _call("read_file", path="a.py", offset=1)),
            _thinking("checking a again", _call("read_file", path="a.py", offset=2)),
            _thinking("checking a once more", _call("read_file", path="a.py", offset=3)),
            _thinking("still checking", _call("read_file", path="a.py", offset=4)),
            _thinking("done", _call("finish_step", summary="gave up", verified="false")),
        ]
    )
    run_tool_loop(
        client=client,
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
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    assert any(e.get("kind") == "no_progress_nudge" for e in events)


def test_a_second_real_edit_to_an_already_touched_file_resets_the_streak(ctx):
    """The bug this fixes: comparing len(ctx.touched) can't see a second real
    edit to a file already in the touched set -- it is still one entry in
    the set either way, so the streak kept climbing right through a genuine
    change. A content fingerprint (mtime+size, same proxy workspace_state
    already uses) changes on the second edit even though the set of touched
    paths does not. Four reads with no edit at all (NO_PROGRESS_LIMIT) would
    fire; this keeps every gap between edits at three, so a fingerprint bug
    that let it drift is exactly what this catches."""
    client = _ScriptedClient(
        [
            _thinking(
                "edit", _call("edit_file", path="a.py", old_string="x = 1", new_string="x = 2")
            ),
            _thinking("look", _call("read_file", path="a.py", offset=1)),
            _thinking("look", _call("read_file", path="a.py", offset=2)),
            _thinking("look", _call("read_file", path="a.py", offset=3)),
            _thinking(
                "edit again",
                _call("edit_file", path="a.py", old_string="x = 2", new_string="x = 3"),
            ),
            _thinking("look", _call("read_file", path="a.py", offset=1)),
            _thinking("look", _call("read_file", path="a.py", offset=2)),
            _thinking("done", _call("finish_step", summary="done", verified="false")),
        ]
    )
    run_tool_loop(
        client=client,
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
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    assert not any(e.get("kind") == "no_progress_nudge" for e in events)


def test_an_edit_in_between_resets_the_streak(ctx):
    """Reading, then actually changing a file, then reading again must not
    accumulate across the edit -- the streak is about being stuck, not about
    a step's total read count."""
    client = _ScriptedClient(
        [
            _thinking("look", _call("read_file", path="a.py", offset=1)),
            _thinking("look", _call("read_file", path="a.py", offset=2)),
            _thinking(
                "edit", _call("edit_file", path="a.py", old_string="x = 1", new_string="x = 2")
            ),
            _thinking("look", _call("read_file", path="a.py", offset=1)),
            _thinking("look", _call("read_file", path="a.py", offset=2)),
            _thinking("done", _call("finish_step", summary="done", verified="false")),
        ]
    )
    run_tool_loop(
        client=client,
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
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    assert not any(e.get("kind") == "no_progress_nudge" for e in events)


def test_edits_that_genuinely_reset_no_progress_still_leave_the_probe_streak_caught(ctx):
    """The actual gap this closes: a real edit landing resets
    _nudge_about_no_progress's streak every time (confirmed below: it never
    fires here, each edit touches a *different*, not-yet-touched file, so
    ctx.touched legitimately grows each round). But the model is still
    probing the same subject over and over between those unrelated edits --
    exactly the live-traced pattern (probe, guess-edit, probe, guess-edit)
    that motivated tracking run_python's subject independently of
    ctx.touched."""
    (ctx.workspace.root / "b.py").write_text("y = 1\n")
    client = _ScriptedClient(
        [
            _thinking("check", _call("run_python", code="from a import x\nprint(x)")),
            _thinking(
                "unrelated edit",
                _call("edit_file", path="b.py", old_string="y = 1", new_string="y = 2"),
            ),
            _thinking("check again", _call("run_python", code="from a import x\nprint(x)")),
            _thinking(
                "another unrelated edit",
                _call("edit_file", path="a.py", old_string="x = 1", new_string="x = 2"),
            ),
            _thinking("check once more", _call("run_python", code="import a\nprint(a.x)")),
            _thinking("done", _call("finish_step", summary="done", verified="false")),
        ]
    )
    run_tool_loop(
        client=client,
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
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    assert any(e.get("kind") == "repeated_probe_nudge" for e in events)
    assert not any(e.get("kind") == "no_progress_nudge" for e in events)


# -- _watch_test_results: a failure that edits are not changing --------------
# Live-traced: one `expr` test failed for five straight run_tests calls while
# the model edited four different functions. Every edit was real (so
# NO_PROGRESS_LIMIT never fired) and no probe tool was involved (so
# RUN_PYTHON_PROBE_LIMIT didn't either) -- the only signal is that the
# failure itself never changed.

_FAIL_TEXT = """FAIL — 1 failed, 6 passed in 0.02s

FAILURES
_ test_unary_minus _
test_expr.py:12: in test_unary_minus
    assert evaluate("-3") == -3
expr.py:88: in _factor
    return self._primary()
expr.py:41: in _primary
    raise ValueError(f"unexpected {tok}")
E   ValueError: unexpected -
short test summary info
FAILED test_expr.py::test_unary_minus - ValueError: unexpected -
"""


def test_failure_details_pull_the_deepest_workspace_frame_and_error():
    from agent.runner import _failure_details

    ids, count, where = _failure_details(_FAIL_TEXT)
    assert ids == frozenset({"test_expr.py::test_unary_minus"})
    assert count == 1
    assert where.startswith("expr.py:41 in _primary")
    assert "unexpected -" in where


def test_failure_details_on_a_pass_are_empty():
    from agent.runner import _failure_details

    assert _failure_details("PASS — 7 passed in 0.01s") == (frozenset(), 0, "")


def _run_watch(ctx, texts_and_edits):
    """Feed run_tests results through _watch_test_results, editing a.py
    between them where asked. Returns each result's final content."""
    from agent.runner import _TestWatch, _watch_test_results

    watch, out = _TestWatch(), []
    for n, (text, edit) in enumerate(texts_and_edits):
        if edit:
            (ctx.workspace.root / "a.py").write_text(f"x = {n}\n")
            ctx.touched.append("a.py") if "a.py" not in ctx.touched else None
        message = {"role": "tool", "content": text}
        calls = [{"function": {"name": "run_tests", "arguments": "{}"}}]
        _watch_test_results(watch, calls, [message], ctx, _Trace(), "execute:s")
        out.append(message["content"])
    return out


def test_same_failure_after_two_edits_is_named_with_its_frame(ctx):
    out = _run_watch(ctx, [(_FAIL_TEXT, False), (_FAIL_TEXT, True), (_FAIL_TEXT, True)])
    assert "[harness]" not in out[0] and "[harness]" not in out[1]
    assert "test_expr.py::test_unary_minus has now failed after 2 edits" in out[2]
    assert "expr.py:41 in _primary" in out[2]


def test_the_same_failure_is_only_named_once(ctx):
    out = _run_watch(ctx, [(_FAIL_TEXT, False)] + [(_FAIL_TEXT, True)] * 5)
    assert sum("[harness]" in text for text in out) == 1


def test_a_changed_failure_resets_the_count(ctx):
    other = _FAIL_TEXT.replace("test_unary_minus", "test_other")
    out = _run_watch(ctx, [(_FAIL_TEXT, False), (_FAIL_TEXT, True), (other, True), (other, True)])
    assert not any("[harness]" in text for text in out)


def test_reruns_with_no_edit_do_not_count(ctx):
    out = _run_watch(ctx, [(_FAIL_TEXT, False)] * 5)
    assert not any("[harness]" in text for text in out)


def test_a_rising_failure_count_after_an_edit_is_flagged(ctx):
    worse = _FAIL_TEXT.replace("1 failed, 6 passed", "18 failed, 33 passed")
    out = _run_watch(ctx, [(_FAIL_TEXT, False), (worse, True)])
    assert "made things worse: 1 failed before it, 18 now" in out[1]
    assert "show_changes" in out[1]


def test_a_pass_resets_everything(ctx):
    out = _run_watch(
        ctx,
        [
            (_FAIL_TEXT, False),
            (_FAIL_TEXT, True),
            ("PASS — 7 passed", True),
            (_FAIL_TEXT, True),
            (_FAIL_TEXT, True),
        ],
    )
    assert not any("[harness]" in text for text in out)


def test_non_test_calls_are_ignored(ctx):
    from agent.runner import _TestWatch, _watch_test_results

    message = {"role": "tool", "content": _FAIL_TEXT}
    calls = [{"function": {"name": "read_file", "arguments": "{}"}}]
    assert _watch_test_results(_TestWatch(), calls, [message], ctx, _Trace(), "execute:s") == 0
    assert message["content"] == _FAIL_TEXT


def test_end_to_end_the_nudge_reaches_the_model(ctx):
    """Through run_tool_loop with a stand-in run_tests that always fails the
    same way, and real edit_file calls between -- the traced shape."""
    from agent.tools import _ALL_EXECUTION
    from agent.tools.base import Registry, Tool, obj

    fake_tests = Tool("run_tests", "d", obj({}), lambda ctx_, **kw: _FAIL_TEXT)
    registry = Registry([t for t in _ALL_EXECUTION if t.name != "run_tests"] + [fake_tests])

    def edit(n):
        return _thinking(
            "edit",
            _call("edit_file", path="a.py", old_string=f"x = {n}", new_string=f"x = {n + 1}"),
        )

    client = _ScriptedClient(
        [
            _thinking("t", _call("run_tests")),
            edit(1),
            _thinking("t", _call("run_tests")),
            edit(2),
            _thinking("t", _call("run_tests")),
            edit(3),
            _thinking("t", _call("run_tests")),
            _thinking("done", _call("finish_step", summary="d", verified="false")),
        ]
    )
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    _, messages = run_tool_loop(
        client=client,
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=registry,
        ctx=ctx,
        trace=ctx.trace,
        max_iters=12,
    )
    joined = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
    assert "has now failed after 2 edits" in joined


_COLLECTION_TEXT = """COLLECTION ERROR (pytest exit 2) — pytest could not even collect the tests.

test_expr.py:1: in <module>
    from expr import evaluate
E   ImportError: cannot import name 'evaluate' from 'expr'
1 error in 0.03s
"""


def test_an_edit_that_breaks_collection_is_called_out(ctx):
    """The worst outcome an edit can have: nothing ran at all. It arrives
    under its own banner rather than "FAIL", so matching only FAIL used to
    reset the watcher on exactly the edit worth warning about."""
    out = _run_watch(ctx, [(_FAIL_TEXT, False), (_COLLECTION_TEXT, True)])
    assert "stopped the suite collecting" in out[1]
    assert "show_changes" in out[1]


def test_a_collection_error_that_was_already_there_is_not_blamed_on_an_edit(ctx):
    """No edit happened between the two, so there is nothing to warn about."""
    out = _run_watch(ctx, [(_COLLECTION_TEXT, False), (_COLLECTION_TEXT, False)])
    assert all("[harness]" not in text for text in out)
