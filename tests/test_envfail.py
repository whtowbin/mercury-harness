"""Separating a broken toolchain from broken code.

The strings asserted on here are not invented: the interpreter cases are
copied from the traced run in which a broken `.venv` was reported to the model
as its own failure, and the negative cases are real pytest and ruff output
from the same traces. The negatives matter more than the positives — calling a
genuine defect "environmental" would excuse broken code, which is the failure
this module must never introduce.
"""

from __future__ import annotations

import typing
from pathlib import Path

from agent import envfail, langtools

# The exact output that cost a run four fix rounds and a scope explosion.
BROKEN_VENV = (
    "error: Querying Python at `/Users/h/flappy-app/.venv/bin/python3` failed "
    "with exit status signal: 6 (SIGABRT)\n\n[stderr]\n"
    "dyld[96045]: Library not loaded: @rpath/libpython3.13.dylib\n"
    "  Referenced from: <4C4C447B> /Users/h/flappy-app/.venv/bin/python3\n"
    "  Reason: tried: '/install/lib/libpython3.13.dylib' (no such file)"
)


# -- the toolchain is broken ----------------------------------------------


def test_the_traced_broken_venv_is_recognised():
    failure = envfail.diagnose(BROKEN_VENV)
    assert failure is not None
    assert failure.kind == "interpreter"


def test_a_missing_shared_library_is_environmental():
    assert envfail.diagnose("dyld[41]: Library not loaded: @rpath/libpython3.13.dylib")


def test_uv_failing_to_probe_the_interpreter_is_environmental():
    assert envfail.diagnose(
        "error: Querying Python at `.venv/bin/python3` failed with exit status 1"
    )


def test_a_failed_resolution_is_environmental():
    failure = envfail.diagnose("error: Failed to resolve dependencies for the project")
    assert failure is not None and failure.kind == "uv"


def test_missing_project_metadata_for_uv_is_environmental():
    failure = envfail.diagnose(
        "venv: error: Failed to spawn: `pytest` (no pyproject.toml, so there is nothing to sync)"
    )
    assert failure is not None and failure.kind == "uv"


def test_a_missing_tool_binary_is_environmental():
    failure = envfail.diagnose("bash: pytest: command not found")
    assert failure is not None and failure.kind == "missing-tool"


def test_a_missing_harness_tool_module_is_environmental():
    failure = envfail.diagnose("ModuleNotFoundError: No module named 'pytest'")
    assert failure is not None and failure.kind == "missing-tool"


def test_the_evidence_is_the_matched_line_not_the_whole_dump():
    failure = envfail.diagnose(BROKEN_VENV)
    assert failure is not None
    assert "\n" not in failure.evidence
    assert len(failure.evidence) <= 200


# -- the code is broken: none of these may be excused ---------------------


def test_a_failing_assertion_is_the_models_problem():
    assert (
        envfail.diagnose(
            "E       AssertionError: Expected velocity -20, got -8\n"
            "E       assert -8 == -20\n1 failed, 5 passed in 0.02s"
        )
        is None
    )


def test_a_missing_workspace_module_is_the_models_problem():
    """`No module named 'game'` means the model never wrote game.py. Excusing
    that as an environment problem would hide the most basic defect there is."""
    assert envfail.diagnose("ModuleNotFoundError: No module named 'game'") is None


def test_a_syntax_error_is_the_models_problem():
    assert (
        envfail.diagnose(
            'File "game.py", line 12\n    def update(\n           ^\nSyntaxError: invalid syntax'
        )
        is None
    )


def test_a_ruff_diagnostic_is_the_models_problem():
    assert envfail.diagnose("game.py:47:11: F821 Undefined name `out`\nFound 1 error.") is None


def test_a_collection_error_naming_the_projects_own_test_is_the_models_problem():
    assert (
        envfail.diagnose("ERROR test_game.py - ImportError: cannot import name 'Pipe' from 'game'")
        is None
    )


def test_empty_output_diagnoses_nothing():
    assert envfail.diagnose("") is None


# -- a declared dependency the venv was never synced against ---------------
# The trace this covers: pyproject.toml already listed numpy, pandas and
# scikit-learn, but the workspace's .venv predated that, so the first test
# run failed on each in turn -- three separate fix rounds re-running the
# whole final gate to do what one `uv sync` would have.


def _project_with_deps(tmp_path: Path, *deps: str) -> Path:
    joined = ", ".join(f'"{d}"' for d in deps)
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nname='x'\nversion='0'\ndependencies=[{joined}]\n"
    )
    return tmp_path


def test_a_declared_but_unsynced_dependency_is_environmental(tmp_path: Path):
    root = _project_with_deps(tmp_path, "numpy>=1.26.0")
    failure = envfail.diagnose("ModuleNotFoundError: No module named 'numpy'", root)
    assert failure is not None and failure.kind == "missing-dependency"


def test_the_import_name_override_covers_scikit_learn(tmp_path: Path):
    root = _project_with_deps(tmp_path, "scikit-learn>=1.4.0")
    assert envfail.diagnose("ModuleNotFoundError: No module named 'sklearn'", root)


def test_the_import_name_override_covers_linear_tree(tmp_path: Path):
    """A real traced run: the standard hyphen-to-underscore guess produces
    'linear_tree', but the package's actual importable module has no
    separator at all -- the wrong guess made a correctly-installed
    dependency get reported as missing."""
    root = _project_with_deps(tmp_path, "linear-tree>=0.3.5")
    assert envfail.diagnose("ModuleNotFoundError: No module named 'lineartree'", root)


def test_the_naive_hyphen_guess_for_linear_tree_is_not_what_gets_matched(tmp_path: Path):
    """The override must actually be used, not the default transform it
    replaces -- confirms the naive guess ('linear_tree') would have missed
    this real package's real import name."""
    root = _project_with_deps(tmp_path, "linear-tree>=0.3.5")
    assert envfail.diagnose("ModuleNotFoundError: No module named 'linear_tree'", root) is None


def test_a_module_absent_from_pyproject_is_still_the_models_problem(tmp_path: Path):
    """Without root, or with a pyproject that never mentions it, this must
    fall through unchanged -- a workspace module like `game` is never a
    dependency, so it can never accidentally match."""
    root = _project_with_deps(tmp_path, "numpy>=1.26.0")
    assert envfail.diagnose("ModuleNotFoundError: No module named 'game'", root) is None
    assert envfail.diagnose("ModuleNotFoundError: No module named 'numpy'") is None


def test_dependencies_declared_only_as_optional_still_count(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0'\n"
        '[project.optional-dependencies]\nplot=["matplotlib>=3.8"]\n'
    )
    failure = envfail.diagnose("ModuleNotFoundError: No module named 'matplotlib'", tmp_path)
    assert failure is not None and failure.kind == "missing-dependency"


def test_a_workspace_with_no_pyproject_never_matches_a_missing_module(tmp_path: Path):
    assert envfail.diagnose("ModuleNotFoundError: No module named 'numpy'", tmp_path) is None


def test_a_malformed_pyproject_does_not_raise(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("this is not valid toml [[[")
    assert envfail.diagnose("ModuleNotFoundError: No module named 'numpy'", tmp_path) is None


def test_repair_and_retry_recognises_a_declared_dependency(tmp_path: Path, monkeypatch):
    root = _project_with_deps(tmp_path, "numpy>=1.26.0")
    monkeypatch.setattr(envfail, "repair", lambda r: (True, "uv sync rebuilt the environment"))
    passed, out, note = envfail.repair_and_retry(
        "ModuleNotFoundError: No module named 'numpy'", root, lambda: (True, "13 passed in 0.5s")
    )
    assert passed is True
    assert note == ""
    assert "13 passed" in out


# -- repair ---------------------------------------------------------------


def test_a_workspace_with_no_pyproject_has_nothing_to_repair(tmp_path: Path):
    assert envfail.repairable(tmp_path) is False
    ran, note = envfail.repair(tmp_path)
    assert ran is False
    assert "nothing to repair" in note


def test_a_uv_project_is_repairable(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    assert envfail.repairable(tmp_path) is True


def test_repair_reports_failure_rather_than_raising(tmp_path: Path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    monkeypatch.setattr(
        envfail.toolchain, "run", lambda *a, **k: (1, "", "error: Failed to resolve")
    )
    ran, note = envfail.repair(tmp_path)
    assert ran is False
    assert "uv sync failed" in note


# -- repair_and_retry: the decision that stops a step being blamed --------


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    return tmp_path


def test_a_code_failure_is_never_repaired_or_re_run(tmp_path: Path, monkeypatch):
    """The expensive half must not fire for an ordinary test failure."""
    calls = []
    monkeypatch.setattr(envfail, "repair", lambda root: calls.append("repair") or (True, ""))
    passed, out, note = envfail.repair_and_retry(
        "E   AssertionError: assert -8 == -20",
        _project(tmp_path),
        lambda: (True, "would be wrong to call this"),
    )
    assert (passed, note) == (False, "")
    assert out == "E   AssertionError: assert -8 == -20"
    assert calls == []


def test_a_repaired_environment_lets_the_command_pass(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(envfail, "repair", lambda root: (True, "uv sync rebuilt it"))
    passed, out, note = envfail.repair_and_retry(
        BROKEN_VENV, _project(tmp_path), lambda: (True, "5 passed in 0.01s")
    )
    assert passed is True
    assert note == ""
    assert "5 passed" in out


def test_a_real_failure_uncovered_by_the_repair_belongs_to_the_model(tmp_path: Path, monkeypatch):
    """Once the toolchain works, what it reports is the code's fault again."""
    monkeypatch.setattr(envfail, "repair", lambda root: (True, "uv sync rebuilt it"))
    passed, out, note = envfail.repair_and_retry(
        BROKEN_VENV,
        _project(tmp_path),
        lambda: (False, "E  AssertionError: assert 0 == 1\n1 failed in 0.02s"),
    )
    assert passed is False
    assert note == ""  # <- the step IS retried, correctly
    assert "AssertionError" in out


def test_an_environment_failure_that_survives_repair_is_not_the_models_fault(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(envfail, "repair", lambda root: (True, "uv sync rebuilt it"))
    passed, _out, note = envfail.repair_and_retry(
        BROKEN_VENV, _project(tmp_path), lambda: (False, BROKEN_VENV)
    )
    assert passed is False
    assert "interpreter" in note
    assert "uv sync rebuilt it" in note


def test_an_unrepairable_environment_failure_is_not_the_models_fault(tmp_path: Path):
    """No pyproject or package.json -- and still not a defect in the code."""
    ran = []
    passed, _out, note = envfail.repair_and_retry(
        BROKEN_VENV, tmp_path, lambda: ran.append(1) or (False, "")
    )
    assert passed is False
    assert "interpreter" in note and "nothing to repair" in note
    assert ran == []  # no point re-running an unrepaired environment


# -- the behaviour this exists for: a correct step is not retried ---------


def test_a_step_whose_environment_is_broken_is_not_blamed(tmp_path: Path, monkeypatch):
    """The whole point. A step that did its job must not be told it failed
    because the virtualenv is broken — that retry is what produced the
    fourteen-iteration scope explosion in the traced run."""
    from agent.phases import execute
    from agent.phases.plan import Step

    step = Step(id="fix-scoring", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, BROKEN_VENV))
    monkeypatch.setattr(execute.envfail, "repair", lambda root: (False, "nothing to sync"))

    class _Ctx:
        touched: typing.ClassVar[list] = []

        class workspace:
            root = tmp_path

    class _Trace:
        def __init__(self):
            self.said = []

        def event(self, *a, **k):
            pass

        def say(self, message):
            self.said.append(message)

    class _Settings:
        verify_steps = True

    trace = _Trace()
    problem = execute._verify_step(_Ctx(), trace, step, _Settings())

    assert problem == ""  # no feedback -> no retry -> no blame
    assert step.verify_passed is None  # and not recorded as a pass either
    assert "interpreter" in step.verify_error
    assert any("could not verify" in line for line in trace.said)


def test_a_step_with_genuinely_failing_code_is_still_blamed(tmp_path: Path, monkeypatch):
    """The guard must not become a blanket excuse."""
    from agent.phases import execute
    from agent.phases.plan import Step

    step = Step(id="fix-scoring", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    monkeypatch.setattr(
        execute, "run_verify", lambda ctx, cmd: (False, "E  assert 0 == 1\n1 failed in 0.02s")
    )

    class _Ctx:
        touched: typing.ClassVar[list] = []

        class workspace:
            root = tmp_path

    class _Trace:
        def event(self, *a, **k):
            pass

        def say(self, message):
            pass

    class _Settings:
        verify_steps = True

    problem = execute._verify_step(_Ctx(), _Trace(), step, _Settings())
    assert "verify command failed" in problem
    assert step.verify_passed is False
    assert step.verify_error == ""


# -- repair, dispatched by language profile ------------------------------


def test_a_node_workspace_is_repairable(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    assert envfail.repairable(tmp_path) is True


def test_a_node_workspace_is_repaired_with_npm_install(tmp_path: Path, monkeypatch):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    calls = []

    def fake_run(tool, args, root, timeout=None):
        calls.append((tool, args))
        return 0, "", ""

    monkeypatch.setattr(envfail.toolchain, "run", fake_run)
    ran, note = envfail.repair(tmp_path)
    assert ran is True
    assert calls == [("npm", ["install"])]
    assert "npm install" in note


def test_a_node_repair_failure_is_reported_not_escalated(tmp_path: Path, monkeypatch):
    """Node gets one stage, not Python's destructive recreate-and-retry --
    nothing has traced a Node environment broken that way yet."""
    (tmp_path / "package.json").write_text('{"name": "x"}')
    calls = []

    def fake_run(tool, args, root, timeout=None):
        calls.append((tool, args))
        return 1, "", "npm ERR! network timeout"

    monkeypatch.setattr(envfail.toolchain, "run", fake_run)
    ran, note = envfail.repair(tmp_path)
    assert ran is False
    assert calls == [("npm", ["install"])]
    assert "network timeout" in note


def test_a_workspace_matching_no_profile_has_nothing_to_repair(tmp_path: Path):
    assert envfail.repairable(tmp_path) is False
    ran, note = envfail.repair(tmp_path)
    assert ran is False
    assert "nothing to repair" in note


def test_repair_accepts_an_already_detected_profile_to_avoid_detecting_twice(
    tmp_path: Path, monkeypatch
):
    """A caller that already knows the profile (e.g. it just used it to
    classify the failure) can pass it straight through."""
    calls = []
    monkeypatch.setattr(envfail.toolchain, "run", lambda *a, **k: calls.append(a) or (0, "", ""))
    ran, _note = envfail.repair(tmp_path, profile=langtools.NODE)
    assert ran is True
    assert calls == [("npm", ["install"], tmp_path)]


def test_an_environmental_failure_never_becomes_a_finding_for_the_model():
    """env_failures are reported to the human and withheld from apply_fixes:
    a model cannot repair a virtualenv, and four fix rounds of one traced run
    were spent proving it."""
    from agent.phases.finalcheck import FinalCheck

    check = FinalCheck(
        env_failures=[
            {
                "step": "s1",
                "command": "uv run pytest -q",
                "note": "interpreter: dyld",
                "output": BROKEN_VENV,
            }
        ]
    )
    assert check.findings() == []
    assert check.ok is False  # unverifiable is not a successful run
    assert check.as_dict()["env_failures"]


# -- repair escalates when the cheap attempt cannot work -----------------


def test_repair_stops_at_uv_sync_when_that_is_enough(tmp_path: Path, monkeypatch):
    """The destructive stage must not fire on a venv that syncs cleanly."""
    calls = []

    def fake_run(tool, args, root, timeout=None):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(envfail.toolchain, "run", fake_run)
    ran, _note = envfail.repair(_project(tmp_path))
    assert ran is True
    assert calls == [["sync"]]
    assert "--clear" not in str(calls)


def test_an_unusable_interpreter_gets_the_virtualenv_recreated(tmp_path: Path, monkeypatch):
    """`uv sync` needs the interpreter it is meant to repair, so on the traced
    failure it reproduces the same dyld error. Recreating is the only thing
    that works, and it is why the second stage exists."""
    calls = []

    def fake_run(tool, args, root, timeout=None):
        calls.append(args)
        if args == ["sync"] and calls.count(["sync"]) == 1:
            return 1, "", "error: Querying Python at `.venv/bin/python3` failed"
        return 0, "", ""

    monkeypatch.setattr(envfail.toolchain, "run", fake_run)
    ran, note = envfail.repair(_project(tmp_path))
    assert ran is True
    assert calls == [["sync"], ["venv", "--clear"], ["sync"]]
    assert "recreated" in note


def test_a_note_is_one_readable_line_not_a_stack_dump(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        envfail.toolchain,
        "run",
        lambda *a, **k: (
            1,
            "",
            "error: boom\n\n[stderr]\ndyld[9]: Library not loaded\n"
            "  Referenced from: /x\n  Reason: tried: '/y'",
        ),
    )
    ran, note = envfail.repair(_project(tmp_path))
    assert ran is False
    assert "\n" not in note
