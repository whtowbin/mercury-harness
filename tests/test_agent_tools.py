"""The tools added for code quality and library knowledge."""

from __future__ import annotations

import json

import pytest

from agent import langtools, toolchain
from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import _fingerprint, run_tool_loop
from agent.tools import ToolContext, execution_registry, planning_registry, review_registry
from agent.tools.checks import check_code, check_complexity, run_python, run_tests, trace_python
from agent.tools.fs import write_file
from agent.tools.introspect import inspect_module
from agent.tools.vcs import git_diff, git_log, show_changes
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- introspection --------------------------------------------------------


def test_inspect_stdlib_module(ctx):
    out = inspect_module(ctx, "json")
    assert "json" in out
    assert "dumps" in out


def test_inspect_function_gives_a_real_signature(ctx):
    out = inspect_module(ctx, "json", "dumps")
    assert "signature:" in out
    assert "obj" in out and "indent" in out


def test_inspect_class_lists_methods(ctx):
    out = inspect_module(ctx, "pathlib", "Path")
    assert "write_text" in out


def test_inspect_module_strips_credential_shaped_env_vars(ctx, monkeypatch):
    """inspect_module imports arbitrary code (real execution, not passive
    inspection) and is reachable from both planning_registry() and
    review_registry(), both nominally read-only -- it must not hand that
    subprocess the harness's own provider/cloud credentials the way
    restricted_env() strips them from every other untrusted-code path."""
    import subprocess as subprocess_module

    from agent.tools import introspect

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    seen_env = {}
    real_run = subprocess_module.run

    def spy_run(*args, **kwargs):
        seen_env.update(kwargs.get("env") or {})
        return real_run(*args, **kwargs)

    monkeypatch.setattr(introspect.subprocess, "run", spy_run)
    introspect.inspect_module(ctx, "json")
    assert "OPENROUTER_API_KEY" not in seen_env


def test_inspect_reports_a_missing_module_without_crashing(ctx):
    out = inspect_module(ctx, "definitely_not_a_real_module_xyz")
    assert out.startswith("error:")
    assert "ModuleNotFoundError" in out or "No module named" in out


def test_inspect_suggests_names_for_a_bad_attribute(ctx):
    out = inspect_module(ctx, "json", "not_a_real_function")
    assert out.startswith("error:")
    assert "Available names" in out
    assert "dumps" in out


# -- check_code -----------------------------------------------------------


def test_check_code_reports_a_clean_file(ctx):
    write_file(ctx, "ok.py", "def f() -> int:\n    return 1\n")
    out = check_code(ctx, ["ok.py"])
    assert "ruff: 0 issue(s)" in out


def test_check_code_surfaces_syntax_errors_first(ctx):
    ctx.autolint = False
    write_file(ctx, "bad.py", "def f(:\n    pass\n")
    out = check_code(ctx, ["bad.py"])
    assert "SYNTAX ERRORS" in out
    assert "ruff:" not in out


def test_check_code_finds_lint_issues(ctx):
    ctx.autolint = False
    write_file(ctx, "lint.py", "import os\n\n\ndef f():\n    return undefined_name\n")
    out = check_code(ctx, ["lint.py"])
    assert "F821" in out


def test_check_code_fix_applies_autofixes(ctx):
    ctx.autolint = False
    write_file(ctx, "fixme.py", "import os\nimport sys\n\nprint(sys.path)\n")
    check_code(ctx, ["fixme.py"], fix=True)
    assert "import os" not in (ctx.workspace.root / "fixme.py").read_text()


def test_check_code_fix_tracks_its_own_write(ctx):
    """ruff --fix/format rewrite the file directly, the same way a shell
    command can behind the tools' backs -- until tracked, that rewrite was
    invisible to review/diff/finalcheck, which all derive from ctx.touched."""
    ctx.autolint = False
    write_file(ctx, "fixme.py", "import os\nimport sys\n\nprint(sys.path)\n")
    ctx.touched.clear()
    check_code(ctx, ["fixme.py"], fix=True)
    assert "fixme.py" in ctx.touched


def test_check_code_defaults_to_touched_files(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    out = check_code(ctx)
    assert "ruff:" in out


# -- run_tests ------------------------------------------------------------


def test_run_tests_reports_a_pass(ctx):
    write_file(ctx, "test_ok.py", "def test_one():\n    assert 1 == 1\n")
    out = run_tests(ctx, ".")
    assert out.startswith("PASS")


def test_run_tests_returns_the_failing_assertion(ctx):
    write_file(ctx, "test_bad.py", "def test_one():\n    assert 1 == 2\n")
    out = run_tests(ctx, ".")
    assert out.startswith("FAIL")
    assert "assert 1 == 2" in out


def test_run_tests_reports_empty_collection(ctx):
    out = run_tests(ctx, ".")
    assert "no tests were collected" in out


def test_run_tests_charges_the_shell_budget(ctx):
    write_file(ctx, "test_ok.py", "def test_one():\n    assert True\n")
    run_tests(ctx, ".")
    assert ctx.guard.subprocess_seconds > 0


# -- run_tests: scoped by default, not the whole suite ---------------------
# A traced fix loop called run_tests(path=".") -- the old hardcoded default
# -- mid-fix-round to check one file it had just edited, re-running the
# whole suite each time. See _default_test_targets in checks.py.


def test_omitting_path_targets_only_the_touched_test_file(ctx, monkeypatch):
    write_file(ctx, "test_touched.py", "def test_one():\n    assert 1 == 1\n")
    write_file(ctx, "test_untouched.py", "def test_one():\n    assert 1 == 0\n")
    ctx.touched.clear()
    ctx.mark_touched("test_touched.py")
    out = run_tests(ctx)
    assert out.startswith("PASS")  # the failing untouched test was never run


def test_explicit_path_still_overrides_the_default(ctx):
    write_file(ctx, "test_a.py", "def test_one():\n    assert 1 == 0\n")
    ctx.mark_touched("test_a.py")
    out = run_tests(ctx, ".")
    assert out.startswith("FAIL")


def test_omitting_path_falls_back_to_the_whole_suite_when_nothing_touched(ctx):
    """Early in a step, or a step that isn't test work: nothing narrower to
    target than the whole suite exists yet, so this must still run
    something, not report a false pass by running nothing."""
    write_file(ctx, "test_untouched.py", "def test_one():\n    assert 1 == 0\n")
    out = run_tests(ctx)
    assert out.startswith("FAIL")


def test_touching_a_non_test_file_does_not_narrow_the_target(ctx):
    write_file(ctx, "helper.py", "x = 1\n")
    write_file(ctx, "test_untouched.py", "def test_one():\n    assert 1 == 0\n")
    ctx.mark_touched("helper.py")
    out = run_tests(ctx)
    assert out.startswith("FAIL")  # helper.py isn't a test file, so "." is used


# -- check_complexity: a non-Python fallback via scc ------------------------
# complexipy only ever covers .py files, so a Node/Go/Rust workspace got no
# complexity signal at all before this existed. Nothing here invokes a real
# scc binary; toolchain.run_capturing is mocked the same way
# test_langtools.py mocks it for detect_profile's own scc fallback.


def _fake_scc_by_file(returncode: int, files: list[dict] | None = None):
    payload = json.dumps([{"Name": "JavaScript", "Files": files or []}])

    def fake(argv, **kwargs):
        return toolchain.ProcResult(returncode, payload, "", 0.0)

    return fake


def test_a_detected_non_python_profile_uses_scc_when_nothing_python_touched(ctx, monkeypatch):
    ctx.language_profile = langtools.NODE
    monkeypatch.setattr(
        toolchain, "run_capturing", _fake_scc_by_file(0, [{"Location": "a.js", "Complexity": 3}])
    )
    out = check_complexity(ctx)
    assert "scc, per-file" in out
    assert "1 file(s) analysed" in out


def test_scc_fallback_reports_files_over_the_limit(ctx, monkeypatch):
    ctx.language_profile = langtools.NODE
    monkeypatch.setattr(
        toolchain,
        "run_capturing",
        _fake_scc_by_file(
            0,
            [
                {"Location": "tangled.js", "Complexity": 40},
                {"Location": "clean.js", "Complexity": 2},
            ],
        ),
    )
    out = check_complexity(ctx, max_allowed=15)
    assert "1 over the limit" in out
    assert "tangled.js" in out


def _saw_scc(calls: list) -> bool:
    return any(argv and argv[0] == "scc" for (argv,) in calls)


def _record_argv(calls: list):
    """A stand-in for toolchain.run_capturing that never actually invokes
    complexipy or scc -- it only records what it would have run."""

    def fake(argv, **kwargs):
        calls.append((argv,))
        return toolchain.ProcResult(0, "[]", "", 0.0)

    return fake


def test_a_python_profile_still_uses_complexipy_not_scc(ctx, monkeypatch):
    calls: list = []
    monkeypatch.setattr(toolchain, "run_capturing", _record_argv(calls))
    ctx.language_profile = langtools.PYTHON
    write_file(ctx, "a.py", "x = 1\n")
    check_complexity(ctx, ["a.py"])
    assert not _saw_scc(calls)


def test_no_detected_profile_still_uses_complexipy(ctx, monkeypatch):
    calls: list = []
    monkeypatch.setattr(toolchain, "run_capturing", _record_argv(calls))
    ctx.language_profile = None
    write_file(ctx, "a.py", "x = 1\n")
    check_complexity(ctx, ["a.py"])
    assert not _saw_scc(calls)


def test_explicit_paths_bypass_the_scc_fallback_even_for_node(ctx, monkeypatch):
    """paths=None is what triggers the touched-files default; naming paths
    explicitly is always the ordinary complexipy path, whatever the
    workspace's detected language is."""
    calls: list = []
    monkeypatch.setattr(toolchain, "run_capturing", _record_argv(calls))
    ctx.language_profile = langtools.NODE
    write_file(ctx, "a.py", "x = 1\n")
    check_complexity(ctx, ["a.py"])
    assert not _saw_scc(calls)


def test_a_touched_python_file_in_a_node_workspace_still_uses_complexipy(ctx, monkeypatch):
    """A mixed workspace editing a .py file gets the Python check, even
    though the workspace overall is detected as Node."""
    calls: list = []
    monkeypatch.setattr(toolchain, "run_capturing", _record_argv(calls))
    ctx.language_profile = langtools.NODE
    write_file(ctx, "a.py", "x = 1\n")
    ctx.mark_touched("a.py")
    check_complexity(ctx)
    assert not _saw_scc(calls)


def test_scc_missing_reports_a_clear_error_not_a_crash(ctx, monkeypatch):
    ctx.language_profile = langtools.NODE
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc_by_file(127))
    out = check_complexity(ctx)
    assert "scc is not installed" in out


def test_scc_unparseable_output_reports_an_error_not_a_crash(ctx, monkeypatch):
    ctx.language_profile = langtools.NODE

    def fake(argv, **kwargs):
        return toolchain.ProcResult(0, "not json", "", 0.0)

    monkeypatch.setattr(toolchain, "run_capturing", fake)
    out = check_complexity(ctx)
    assert out.startswith("error:")


# -- change visibility ----------------------------------------------------


def test_show_changes_stat_mode(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    out = show_changes(ctx, stat=True)
    assert "a.py" in out and "+1/-0" in out


def test_show_changes_full_diff(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    assert "+x = 1" in show_changes(ctx)


def test_show_changes_rejects_untouched_paths(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    out = show_changes(ctx, paths=["never_touched.py"])
    assert "none of those files" in out


def test_git_tools_explain_themselves_outside_a_repo(ctx):
    assert "not a git repository" in git_diff(ctx)
    assert "not a git repository" in git_log(ctx)


# -- registries -----------------------------------------------------------


def test_reviewer_cannot_write():
    names = review_registry().names()
    assert "edit_file" not in names and "write_file" not in names
    # but it can still confirm a suspicion
    assert "run_tests" in names and "check_code" in names


def test_planner_is_read_only_and_can_look_up_libraries():
    names = planning_registry().names()
    assert "edit_file" not in names and "run_bash" not in names
    assert "inspect_module" in names and "pypi_info" in names


def test_executor_has_the_full_kit():
    names = execution_registry().names()
    for expected in (
        "edit_file",
        "check_code",
        "run_tests",
        "show_changes",
        "finish_step",
    ):
        assert expected in names
    # The planner and reviewer keep inspect_module (110 calls between them);
    # the executor never called it in 97 offers (DESIGN-LOG §66).
    assert "inspect_module" not in names


def test_every_registry_has_wellformed_schemas():
    for registry in (execution_registry(), planning_registry(), review_registry()):
        for schema in registry.schemas():
            fn = schema["function"]
            assert fn["name"] and fn["description"]
            for name in fn["parameters"]["required"]:
                assert name in fn["parameters"]["properties"]


# -- runner guards --------------------------------------------------------


def call(name: str, **args) -> dict:
    return {
        "id": f"c{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Reply:
        self.seen.append(list(messages))
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def reply(*tool_calls):
    return Reply(content="", tool_calls=list(tool_calls), usage=Usage(100, 50, 0.0, 0.25))


def test_fingerprint_ignores_argument_order():
    a = _fingerprint("read_file", json.dumps({"path": "a.py", "offset": 1}))
    b = _fingerprint("read_file", json.dumps({"offset": 1, "path": "a.py"}))
    assert a == b


def test_repeated_identical_calls_get_a_warning(ctx):
    same = lambda: call("list_dir", path=".")  # noqa: E731
    client = ScriptedClient(
        [reply(same()), reply(same()), reply(same()), reply(call("finish_step", summary="done"))]
    )
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
    )
    tool_messages = [m for m in client.seen[-1] if m.get("role") == "tool"]
    assert any("identical arguments" in m["content"] for m in tool_messages)


def test_a_call_repeated_past_the_hard_limit_is_refused_outright(ctx):
    """The warning stops being a warning at some point. A model that has
    ignored it twice will otherwise burn every remaining iteration on the
    same call, which is the exact loop the warning exists to break."""
    same = lambda: call("list_dir", path=".")  # noqa: E731
    client = ScriptedClient([reply(same()) for _ in range(8)])
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=8,
    )
    tool_messages = [m for m in client.seen[-1] if m.get("role") == "tool"]
    assert any("refusing to run" in m["content"] for m in tool_messages)


def test_a_refused_repeat_is_never_actually_dispatched(tmp_path):
    """The point is not to say no politely, it is to stop doing the work."""
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    dispatched: list[dict] = []
    trace = Trace(
        str(tmp_path / "t"), echo=False, on_event=lambda record: dispatched.append(record)
    )
    local_ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )

    same = lambda: call("write_file", path="f.txt", content="x")  # noqa: E731
    client = ScriptedClient([reply(same()) for _ in range(9)])
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=local_ctx,
        trace=trace,
        max_iters=9,
    )
    tool_calls = [e for e in dispatched if e.get("kind") == "tool_call"]
    assert len(tool_calls) <= 5, f"dispatched {len(tool_calls)} identical calls"


def test_different_calls_are_never_refused(ctx):
    client = ScriptedClient([reply(call("read_file", path=f"f{i}.py")) for i in range(8)])
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=8,
    )
    tool_messages = [m for m in client.seen[-1] if m.get("role") == "tool"]
    assert not any("refusing to run" in m["content"] for m in tool_messages)


def test_error_budget_raises_guard_tripped_for_a_hopeless_step(ctx):
    """run_tool_loop raises rather than swallowing the trip: a caller with no
    abandon-this-step handling (like schedule.py's _guarded) should fail loudly,
    not silently return an empty result that looks like an ordinary miss."""
    from agent.guards import GuardTripped

    ctx.guard.max_step_errors = 4
    client = ScriptedClient([reply(call("read_file", path=f"missing{i}.py")) for i in range(10)])
    with pytest.raises(GuardTripped):
        run_tool_loop(
            client=client,
            model="fake",
            phase="test",
            system="s",
            user="u",
            registry=execution_registry(),
            ctx=ctx,
            trace=ctx.trace,
            max_iters=10,
        )
    # It bailed out early rather than burning all ten iterations.
    assert len(client.seen) < 10


def test_guard_resets_between_steps(ctx):
    ctx.guard.record_subprocess(60)
    client = ScriptedClient([reply(call("finish_step", summary="ok"))])
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
    )
    assert ctx.guard.subprocess_seconds == 0


# -- pytest summary parsing ----------------------------------------------


def test_summary_prefers_the_final_banner_over_section_headers():
    from agent.tools.checks import _summarise

    output = (
        "==================== ERRORS ====================\n"
        "ERROR collecting tests/test_x.py\n"
        "=============== short test summary info ===============\n"
        "==================== 1 error in 0.11s ====================\n"
    )
    assert _summarise(output, 1) == "1 error in 0.11s"


def test_summary_reads_a_passing_run():
    from agent.tools.checks import _summarise

    assert _summarise("========== 9 passed in 0.03s ==========\n", 0) == "9 passed in 0.03s"


def test_summary_falls_back_to_the_exit_code():
    from agent.tools.checks import _summarise

    assert "exited 3" in _summarise("total nonsense", 3)


def test_run_tests_summary_is_accurate_end_to_end(ctx):
    write_file(
        ctx, "test_ok.py", "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"
    )
    out = run_tests(ctx, ".")
    assert out.startswith("PASS")
    assert "2 passed" in out


# -- run_python -----------------------------------------------------------


def test_run_python_executes_and_captures_output(ctx):
    out = run_python(ctx, "print(2 + 2)")
    assert "exit=0" in out and "4" in out


def test_run_python_can_import_the_workspace(ctx):
    write_file(ctx, "mod.py", "def double(x):\n    return x * 2\n")
    out = run_python(ctx, "from mod import double\nprint(double(21))")
    assert "42" in out


def test_run_python_reports_a_traceback_without_raising(ctx):
    out = run_python(ctx, "raise ValueError('boom')")
    assert "exit=1" in out and "boom" in out


def test_run_python_handles_quotes_that_would_break_a_shell(ctx):
    out = run_python(ctx, """print("it's \\"quoted\\"")""")
    assert "exit=0" in out and "quoted" in out


def test_run_python_rejects_empty_code(ctx):
    assert run_python(ctx, "   ").startswith("error:")


def test_run_python_cleans_up_its_scratch_file(ctx):
    run_python(ctx, "print(1)")
    leftovers = list(ctx.workspace.root.glob("_agent_snippet_*.py"))
    assert leftovers == []
    assert not any(t.startswith("_agent_snippet_") for t in ctx.touched)


def test_run_python_writes_are_tracked_as_touched(ctx):
    """A snippet can create or rewrite a workspace file just as easily as
    run_bash can (shutil, os, subprocess are all reachable from it) -- until
    this was tracked, such a write was invisible to review/diff/finalcheck,
    which all derive from ctx.touched."""
    run_python(ctx, "open('created_by_snippet.py', 'w').write('x = 1\\n')")
    assert "created_by_snippet.py" in ctx.touched
    assert (ctx.workspace.root / "created_by_snippet.py").read_text() == "x = 1\n"


def test_run_python_tracks_writes_even_when_the_snippet_errors(ctx):
    """run_bash tracks its external writes even on a timeout; a snippet that
    writes a file and then raises must not lose that write's visibility
    either -- mark_external_writes runs in the same `finally` as cleanup."""
    run_python(ctx, "open('partial.py', 'w').write('x = 1\\n'); raise SystemExit(1)")
    assert "partial.py" in ctx.touched


def test_run_python_uses_a_unique_scratch_file_per_call(ctx):
    """Two concurrent calls must not be able to collide on one filename."""
    seen: set[str] = set()
    original = __import__("pathlib").Path.write_text

    def spy(self, *args, **kwargs):
        if self.name.startswith("_agent_snippet_"):
            seen.add(self.name)
        return original(self, *args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(__import__("pathlib").Path, "write_text", spy):
        run_python(ctx, "print(1)")
        run_python(ctx, "print(2)")
    assert len(seen) == 2, "each call must get a distinct scratch filename"


def test_run_python_charges_the_shell_budget(ctx):
    run_python(ctx, "print(1)")
    assert ctx.guard.subprocess_seconds > 0


# -- trace_python -----------------------------------------------------------


def test_trace_python_shows_the_trace_and_the_snippets_own_output(ctx):
    out = trace_python(ctx, "print(2 + 2)")
    assert "exit=0" in out and "4" in out
    assert "--- TRACE" in out


def test_trace_python_shows_lines_executed_inside_a_workspace_module(ctx):
    write_file(ctx, "mod.py", "def double(x):\n    return x * 2\n")
    out = trace_python(ctx, "from mod import double\nprint(double(21))")
    assert "42" in out
    assert "mod.py" in out
    assert "return x * 2" in out


def test_trace_python_shows_changed_locals(ctx):
    write_file(ctx, "mod.py", "def add(a, b):\n    total = a + b\n    return total\n")
    out = trace_python(ctx, "from mod import add\nadd(2, 3)")
    assert "total=" in out


def test_trace_python_excludes_stdlib_and_site_packages(ctx):
    """The trace is about the model's own code -- import machinery and
    installed packages would otherwise dominate it with irrelevant frames."""
    out = trace_python(ctx, "import json\njson.dumps({'a': 1})")
    assert "site-packages" not in out
    assert "importlib" not in out
    assert "json/__init__.py" not in out and "json/encoder.py" not in out


def test_trace_python_survives_a_repr_that_raises(ctx):
    """A local's own __repr__ can raise -- live-caught with a half-constructed
    import-machinery object. That must cost only that variable's display,
    never the whole snippet."""
    write_file(
        ctx,
        "mod.py",
        "class Bad:\n    def __repr__(self):\n        raise RuntimeError('no repr')\n"
        "def make():\n    b = Bad()\n    return 1\n",
    )
    out = trace_python(ctx, "from mod import make\nprint(make())")
    assert "exit=0" in out
    assert "1" in out


def test_trace_python_reports_a_traceback_and_the_trace_so_far(ctx):
    out = trace_python(ctx, "x = 1\nraise ValueError('boom')")
    assert "exit=1" in out and "boom" in out
    assert "--- TRACE" in out


def test_trace_python_rejects_empty_code(ctx):
    assert trace_python(ctx, "   ").startswith("error:")


def test_trace_python_cleans_up_its_scratch_file(ctx):
    trace_python(ctx, "print(1)")
    leftovers = list(ctx.workspace.root.glob("_agent_snippet_*.py"))
    assert leftovers == []


def test_trace_python_writes_are_tracked_as_touched(ctx):
    trace_python(ctx, "open('created_by_trace.py', 'w').write('x = 1\\n')")
    assert "created_by_trace.py" in ctx.touched


def test_trace_python_charges_the_shell_budget(ctx):
    trace_python(ctx, "print(1)")
    assert ctx.guard.subprocess_seconds > 0


def test_trace_python_is_gated_when_not_auto_approved(tmp_path):
    from agent.approval import Approver

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    out = trace_python(gated_ctx, "print(1)")
    assert out.startswith("error:")


def test_trace_python_caps_a_long_loop(ctx):
    """A traced loop must not dump an unbounded number of lines."""
    from agent.tools.checks import MAX_TRACE_LINES

    write_file(
        ctx,
        "mod.py",
        "def spin():\n    total = 0\n    for i in range(10000):\n"
        "        total += i\n    return total\n",
    )
    out = trace_python(ctx, "from mod import spin\nspin()")
    assert "stopped at the line cap" in out
    traced_lines = [line for line in out.splitlines() if line.startswith("mod.py:")]
    assert len(traced_lines) <= MAX_TRACE_LINES


def test_run_python_is_gated_when_not_auto_approved(tmp_path):
    """The bug: run_python used to run unconditionally, with no approval check
    at all — as capable of damage as run_bash but with none of its gating."""
    from agent.approval import Approver

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    # No TTY in a test process: the gate must deny rather than silently run.
    result = run_python(gated_ctx, "print('should not run')")
    assert result.startswith("error:")


def test_run_python_gate_approves_a_scripted_yes(tmp_path, monkeypatch):
    from agent.approval import Approver

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_python(gated_ctx, "print('approved')")
    assert "approved" in result


def test_run_python_gate_uses_a_supplied_ask_callback_over_stdin(tmp_path, monkeypatch):
    """The TUI bug: with no callback, the gate falls back to a blocking
    input() on stdin — which hangs forever once Textual owns the terminal,
    since nothing can ever satisfy that read. A caller (the TUI) supplies its
    own callback instead, and it must be used in preference to stdin even
    when stdin looks like a real TTY."""
    from agent.approval import Approver

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def blows_up_if_called(_prompt):
        raise AssertionError("input() must not be called when an ask callback is supplied")

    monkeypatch.setattr("builtins.input", blows_up_if_called)

    seen: list[str] = []

    def ask(shown: str) -> str:
        seen.append(shown)
        return "y"

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False, ask=ask),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_python(gated_ctx, "print('approved via callback')")
    assert "approved via callback" in result
    assert seen and "run this Python" in seen[0]


def test_run_python_gate_callback_empty_answer_denies(tmp_path):
    """An ask callback returning "" (a bare Enter, in the TUI's bottom bar)
    must deny exactly like typing "n" at the stdin prompt would — not hang,
    and not silently allow."""
    from agent.approval import Approver

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False, ask=lambda _shown: ""),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_python(gated_ctx, "print('should not run')")
    assert result.startswith("error:")


def test_auto_approve_python_skips_the_gate_with_no_prompt_at_all(tmp_path):
    """auto_approve_python must not even call the ask callback or read
    stdin — it's a distinct, narrower trust level from auto_approve, not a
    pre-answered prompt."""
    from agent.approval import Approver

    def blows_up_if_called(_shown):
        raise AssertionError("the gate must not be consulted at all")

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False, ask=blows_up_if_called, auto_approve_python=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_python(gated_ctx, "print('auto-approved')")
    assert "auto-approved" in result


def test_auto_approve_python_does_not_also_approve_shell(tmp_path):
    """The narrower trust level: run_bash must still gate normally even
    with auto_approve_python on."""
    from agent.approval import Approver
    from agent.tools.shell import run_bash

    gated_ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=False, ask=lambda _shown: "n", auto_approve_python=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    result = run_bash(gated_ctx, "some-unallowlisted-command")
    assert result.startswith("error:")


# -- review_registry is genuinely read-only (from the code review) --------


def test_reviewer_has_no_shell_access():
    names = review_registry().names()
    assert "run_bash" not in names
    assert "run_python" not in names


def test_reviewer_check_code_has_no_fix_parameter():
    """Not just undocumented — structurally absent, since _dispatch() does not
    validate a model's tool-call arguments against the declared schema before
    calling the tool, so omitting `fix` from the schema alone would not stop
    a model from sending it anyway."""
    check_code_tool = review_registry().get("check_code")
    assert check_code_tool is not None
    assert "fix" not in check_code_tool.parameters["properties"]
    import inspect

    assert "fix" not in inspect.signature(check_code_tool.fn).parameters


def test_reviewer_check_code_cannot_write_even_if_a_model_tries(ctx):
    """The actual end-to-end guarantee: calling it with fix=true (as if a
    model tried to sneak it past the schema) must fail loudly, not write."""
    from agent.tools.fs import write_file

    write_file(ctx, "messy.py", "import os\nimport sys\n\nprint(sys.path)\n")
    before = (ctx.workspace.root / "messy.py").read_text()

    check_code_tool = review_registry().get("check_code")
    assert check_code_tool is not None
    with pytest.raises(TypeError):
        check_code_tool.fn(ctx, paths=["messy.py"], fix=True)

    assert (ctx.workspace.root / "messy.py").read_text() == before


def test_executor_check_code_still_has_fix():
    """The executor's own check_code is unaffected — only the reviewer's is
    restricted."""
    check_code_tool = execution_registry().get("check_code")
    assert check_code_tool is not None
    assert "fix" in check_code_tool.parameters["properties"]


def test_reviewer_can_still_confirm_findings():
    names = review_registry().names()
    assert "run_tests" in names
    assert "check_code" in names
    assert "inspect_module" in names


# -- a model sending "false" as a string must not mean True ---------------


def test_a_string_false_becomes_a_real_false(ctx):
    """Observed in a real trace: run_tests(verbose="true"). Python treats
    every non-empty string as truthy, so "false" meant True — and for
    edit_file's replace_all that is the difference between changing one
    occurrence and changing all of them."""
    from agent.runner import _coerce_args

    tool = execution_registry().get("edit_file")

    assert tool is not None
    assert _coerce_args(tool, {"replace_all": "false"})[0]["replace_all"] is False
    assert _coerce_args(tool, {"replace_all": "true"})[0]["replace_all"] is True


def test_replace_all_as_a_string_reaches_edit_file_as_a_real_bool(ctx):
    """End-to-end through the real dispatcher. With replace_all correctly
    False, an anchor matching three times is refused as ambiguous; before the
    coercion the string "false" was truthy and all three were rewritten
    silently."""
    target = ctx.workspace.root / "dup.py"
    target.write_text("x = 1\nx = 1\nx = 1\n")
    client = ScriptedClient(
        [
            reply(
                call(
                    "edit_file",
                    path="dup.py",
                    old_string="x = 1",
                    new_string="x = 2",
                    replace_all="false",
                )
            ),
            reply(call("finish_step", summary="done")),
        ]
    )
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=4,
    )
    assert target.read_text() == "x = 1\nx = 1\nx = 1\n", "nothing should have changed"
    tool_messages = [m for m in client.seen[-1] if m.get("role") == "tool"]
    assert any("appears 3 times" in m["content"] for m in tool_messages)


def test_replace_all_true_as_a_string_does_replace_all(ctx):
    target = ctx.workspace.root / "dup.py"
    target.write_text("x = 1\nx = 1\nx = 1\n")
    client = ScriptedClient(
        [
            reply(
                call(
                    "edit_file",
                    path="dup.py",
                    old_string="x = 1",
                    new_string="x = 2",
                    replace_all="true",
                )
            ),
            reply(call("finish_step", summary="done")),
        ]
    )
    run_tool_loop(
        client=client,
        model="fake",
        phase="test",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=4,
    )
    assert target.read_text() == "x = 2\nx = 2\nx = 2\n"


def test_an_integer_sent_as_a_string_is_coerced(ctx):
    from agent.runner import _coerce_args

    tool = execution_registry().get("read_file")

    assert tool is not None
    assert _coerce_args(tool, {"offset": "3"})[0]["offset"] == 3


def test_a_value_that_is_not_the_declared_type_is_left_alone(ctx):
    """So a genuinely bad argument still reaches the tool and gets the normal
    error, rather than being silently reshaped into something plausible."""
    from agent.runner import _coerce_args

    tool = execution_registry().get("read_file")

    assert tool is not None
    assert _coerce_args(tool, {"offset": "not-a-number"})[0]["offset"] == "not-a-number"


def test_strings_for_string_parameters_are_untouched(ctx):
    from agent.runner import _coerce_args

    tool = execution_registry().get("read_file")

    assert tool is not None
    assert _coerce_args(tool, {"path": "true"})[0]["path"] == "true"


def test_ask_user_passes_a_valid_recommendation_through(ctx):
    from agent.interact import Interactor
    from agent.tools.ask import ask_user

    seen = {}
    ctx.interactor = Interactor(ask=lambda q, o=None, r=None: (seen.update(rec=r), "beta")[1])
    ask_user(ctx, "which?", ["alpha", "beta"], recommended="beta")
    assert seen["rec"] == "beta"


def test_ask_user_drops_a_recommendation_that_is_not_an_option(ctx):
    """A recommendation pointing at something unpickable is worse than none."""
    from agent.interact import Interactor
    from agent.tools.ask import ask_user

    seen = {}
    ctx.interactor = Interactor(ask=lambda q, o=None, r=None: (seen.update(rec=r), "alpha")[1])
    ask_user(ctx, "which?", ["alpha"], recommended="something-else")
    assert seen["rec"] is None
