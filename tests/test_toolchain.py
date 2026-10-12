"""The consolidated subprocess primitive, truncation helper, and resolve() caching."""

from __future__ import annotations

import contextlib
import os
import pathlib
import signal
import sys
import time

from agent import toolchain
from agent.text import truncate

# -- run_capturing ----------------------------------------------------------


def test_run_capturing_returns_stdout_stderr_and_timing():
    result = toolchain.run_capturing([sys.executable, "-c", "print('hi')"])
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"
    assert result.elapsed_s >= 0
    assert not result.timed_out


def test_run_capturing_reports_a_timeout_without_raising():
    result = toolchain.run_capturing(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2
    )
    assert result.timed_out
    assert result.returncode == 124


def test_run_capturing_reports_a_missing_executable_without_raising():
    result = toolchain.run_capturing(["definitely-not-a-real-binary-xyz"])
    assert result.returncode == 127
    assert not result.timed_out


def test_run_capturing_output_property_combines_streams():
    result = toolchain.run_capturing(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    )
    assert "out" in result.output and "err" in result.output


def test_run_capturing_output_is_just_stdout_when_stderr_is_empty():
    result = toolchain.run_capturing([sys.executable, "-c", "print('only stdout')"])
    assert result.output.strip() == "only stdout"


# -- truncate -----------------------------------------------------------


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_truncate_head_keeps_the_start():
    result = truncate("a" * 50 + "b" * 50, 50, keep="head")
    assert result.startswith("a" * 50)
    assert "b" not in result


def test_truncate_tail_keeps_the_end():
    result = truncate("a" * 50 + "b" * 50, 50, keep="tail")
    assert result.endswith("b" * 50)
    assert "a" not in result.replace("head elided", "")


def test_truncate_both_keeps_each_end():
    result = truncate("a" * 100 + "MIDDLE" + "b" * 100, 40, keep="both")
    assert result.startswith("a")
    assert result.endswith("b" * 20)
    assert "MIDDLE" not in result


def test_truncate_marks_what_was_cut():
    assert "elided" in truncate("x" * 1000, 10)


# -- run() uses restricted_env, not the unrestricted workspace_env ----------
# pytest executes arbitrary repository code during collection, and run() is
# what run_tests/check_code/the finalcheck gate/coverage all use to invoke
# it -- restricted_env() exists specifically for "verification and other
# untrusted subprocesses" (see its own docstring) and was already wired into
# the sibling run_verify() path (agent/phases/execute.py); run() was missed.


def test_run_strips_credential_shaped_env_vars(monkeypatch, tmp_path):
    seen_env = {}

    def fake_run_capturing(argv, cwd=None, timeout=None, env=None):
        seen_env.update(env or {})
        return toolchain.ProcResult(0, "", "", 0.0)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setattr(toolchain, "run_capturing", fake_run_capturing)
    monkeypatch.setattr(toolchain, "resolve", lambda tool, root: [tool])
    toolchain.run("pytest", ["-q"], tmp_path)
    assert "OPENROUTER_API_KEY" not in seen_env


def test_run_still_gets_the_workspace_venv_on_path(monkeypatch, tmp_path):
    """restricted_env() is workspace_env() plus stripping -- switching to it
    must not reopen the ModuleNotFoundError bug this function's own
    docstring describes (pytest running outside the workspace's venv)."""
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    seen_env = {}

    def fake_run_capturing(argv, cwd=None, timeout=None, env=None):
        seen_env.update(env or {})
        return toolchain.ProcResult(0, "", "", 0.0)

    monkeypatch.setattr(toolchain, "run_capturing", fake_run_capturing)
    monkeypatch.setattr(toolchain, "resolve", lambda tool, root: [tool])
    toolchain.run("pytest", ["-q"], tmp_path)
    assert seen_env.get("VIRTUAL_ENV") == str(tmp_path / ".venv")
    assert str(tmp_path / ".venv" / "bin") in seen_env.get("PATH", "")


# -- resolve() caching, the efficiency fix -----------------------------


def test_resolve_cache_hits_across_different_workspace_roots(tmp_path):
    """The bug: caching on (tool, root) meant a fresh tempdir per bench job
    never hit the cache. Caching on (tool, has_project) does, since whether
    `ruff` resolves via PATH doesn't depend on which temp dir asked."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    before = toolchain._resolve_for.cache_info().hits
    toolchain.resolve("python", str(root_a))
    toolchain.resolve("python", str(root_b))
    after = toolchain._resolve_for.cache_info().hits

    # Neither root has a project file, so both calls share the same
    # (tool, has_project) key and the second must be a cache hit.
    assert after > before


def test_resolve_still_returns_a_working_interpreter(tmp_path):
    argv = toolchain.resolve("python", str(tmp_path))
    assert argv is not None


# -- packaged skills_data resolves correctly, not just in dev checkouts ----


def test_default_skills_dir_is_package_relative_not_repo_relative():
    """The bug: DEFAULT_SKILLS_DIR used to be Path(__file__).parent.parent /
    'skills' — repo-root-relative. Once pip-installed, that resolves inside
    site-packages with nothing there. It must be a *child of the agent
    package itself* (.parent, not .parent.parent), which survives install."""
    from agent import skills as skills_module

    package_dir = pathlib.Path(skills_module.__file__).parent
    assert skills_module.DEFAULT_SKILLS_DIR.parent == package_dir
    assert skills_module.DEFAULT_SKILLS_DIR.is_dir()


def test_bundled_skills_are_present_and_loadable():
    from agent import skills as skills_module

    loaded = skills_module.load()
    assert {s.name for s in loaded} == {
        "uv",
        "ruff",
        "ty",
        "verifying-your-work",
        "complexity",
        "research",
        "grilling",
        "diagnosing",
        "keeping-tests-fast",
        "documenting-changes",
        "web-research",  # offered only with Settings.web_research (DESIGN-LOG §116)
    }


def test_settings_skills_dir_defaults_to_none_not_a_cwd_relative_string():
    """A plain "skills" string default meant the tool only worked when the
    process's cwd happened to be the repo root — broken for anyone who
    installed it and ran it from anywhere else."""
    from agent.config import Settings

    s = Settings(workspace=".")
    assert s.skills_dir is None


def test_a_timeout_stops_the_commands_children_too(tmp_path):
    """`uv run pytest` is two processes; a timeout that kills only the first left the second
    running at 100% CPU for hours (§126.5). Real processes: a shell that starts a background
    sleep and waits for it."""
    pidfile = tmp_path / "child.pid"
    result = toolchain.run_capturing(
        ["sh", "-c", f"sleep 300 & echo $! > {pidfile}; wait"], timeout=1
    )
    assert result.timed_out
    child = int(pidfile.read_text())
    try:
        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the child of the timed-out command is still running")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)
