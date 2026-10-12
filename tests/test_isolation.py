"""Containment of agent subprocesses.

The path jail constrains the file tools, but not the commands the agent runs.
Build tools search *upward* for a project and a virtualenv, so a workspace nested
inside another project is an escape route: `uv pip install` in the workspace lands
in the enclosing project's environment. These tests cover the two defences —
refusing nested workspaces, and refusing the commands that escape.
"""

from __future__ import annotations

import pytest

from agent import toolchain
from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.checks import run_python
from agent.tools.shell import _escapes_workspace, run_bash
from agent.trace import Trace
from agent.workspace import Workspace


def make_ctx(root) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(root / ".trace"), echo=False),
    )


# -- nesting detection ----------------------------------------------------


def test_detects_an_enclosing_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='host'\n")
    inner = tmp_path / "sandbox"
    inner.mkdir()
    assert Workspace(inner).enclosing_project() == tmp_path.resolve()


def test_detects_an_enclosing_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    inner = tmp_path / "sandbox"
    inner.mkdir()
    assert Workspace(inner).enclosing_project() == tmp_path.resolve()


def test_standalone_workspace_has_no_enclosing_project(tmp_path):
    inner = tmp_path / "sandbox"
    inner.mkdir()
    assert Workspace(inner).enclosing_project() is None


def test_the_workspaces_own_pyproject_does_not_count(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mine'\n")
    assert Workspace(tmp_path).enclosing_project() is None


# -- uv pip containment ---------------------------------------------------


def test_uv_pip_install_is_blocked_without_a_workspace_venv(tmp_path):
    ctx = make_ctx(tmp_path)
    result = run_bash(ctx, "uv pip install requests")
    assert result.startswith("error:")
    assert "uv add" in result


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install six",
        "uv pip uninstall six",
        "uv pip sync requirements.txt",
        "cd . && uv pip install six",
    ],
)
def test_every_escaping_uv_pip_form_is_caught(tmp_path, command):
    ctx = make_ctx(tmp_path)
    assert _escapes_workspace(ctx, command) is not None


def test_uv_pip_is_allowed_once_the_workspace_owns_a_venv(tmp_path):
    ctx = make_ctx(tmp_path)
    (tmp_path / ".venv").mkdir()
    assert _escapes_workspace(ctx, "uv pip install six") is None


@pytest.mark.parametrize("command", ["uv add requests", "uv sync", "uv run pytest", "ls"])
def test_project_scoped_commands_are_not_blocked(tmp_path, command):
    ctx = make_ctx(tmp_path)
    assert _escapes_workspace(ctx, command) is None


# -- subprocess environment ----------------------------------------------


def test_workspace_env_strips_inherited_python_state(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/venv")
    env = toolchain.workspace_env(tmp_path)
    assert "PYTHONPATH" not in env
    assert env.get("VIRTUAL_ENV") != "/some/other/venv"


def test_workspace_env_points_uv_at_the_workspace(tmp_path):
    env = toolchain.workspace_env(tmp_path)
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")


def test_workspace_env_activates_a_workspace_venv_when_present(tmp_path):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    env = toolchain.workspace_env(tmp_path)
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].startswith(str(venv / "bin"))


def test_run_bash_actually_uses_the_isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/leaked/path")
    ctx = make_ctx(tmp_path)
    out = run_bash(ctx, 'echo "[${PYTHONPATH:-unset}]"')
    assert "[unset]" in out


def test_concurrent_steps_cannot_run_arbitrary_shell(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.writable = frozenset({"owned.py"})
    result = run_bash(ctx, "touch sibling.py")
    assert "disabled while steps run in parallel" in result
    assert not (tmp_path / "sibling.py").exists()


def test_concurrent_steps_cannot_run_arbitrary_python(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.writable = frozenset({"owned.py"})
    result = run_python(ctx, "open('sibling.py', 'w').close()")
    assert "disabled while steps run in parallel" in result
    assert not (tmp_path / "sibling.py").exists()


def test_concurrent_steps_cannot_run_ruff_fix(tmp_path):
    """A fourth way to write outside a declared file set: check_code(fix=True)
    runs `ruff --fix`/`ruff format` directly, which previously had no
    ctx.writable check at all -- unlike run_bash/run_python/ast_grep_rewrite
    above, all of which correctly refuse under a concurrent write jail."""
    from agent.tools.checks import check_code

    ctx = make_ctx(tmp_path)
    ctx.writable = frozenset({"owned.py"})
    (tmp_path / "sibling.py").write_text("x=1\n")  # a real, fixable ruff issue
    result = check_code(ctx, paths=["sibling.py"], fix=True)
    assert "disabled while steps run in parallel" in result
    assert (tmp_path / "sibling.py").read_text() == "x=1\n"  # untouched


def test_concurrent_steps_cannot_apply_a_structural_rewrite(tmp_path):
    """Same reasoning as run_bash/run_python above, for a third way to write
    outside a declared file set: ast_grep_rewrite decides which files to
    touch from its own pattern match, not from the plan's file list, so
    there is nothing for the concurrent write jail to check it against
    after the fact. See tests/test_structural.py for the rest of its
    behaviour."""
    from agent.tools.structural import ast_grep_rewrite

    ctx = make_ctx(tmp_path)
    (tmp_path / "owned.py").write_text("old_name(1)\n")
    ctx.writable = frozenset({"owned.py"})
    result = ast_grep_rewrite(ctx, "old_name($$$ARGS)", "new_name($$$ARGS)", apply=True)
    assert "disabled while steps run in parallel" in result
    assert (tmp_path / "owned.py").read_text() == "old_name(1)\n"


def test_restricted_env_removes_common_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("BRAVE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret")
    monkeypatch.setenv("SAFE_SETTING", "kept")
    env = toolchain.restricted_env(tmp_path)
    assert "OPENROUTER_API_KEY" not in env
    assert "BRAVE_API_KEY" not in env
    assert "DATABASE_PASSWORD" not in env
    assert env["SAFE_SETTING"] == "kept"


def test_restricted_env_removes_credentials_the_old_narrow_suffixes_missed(tmp_path, monkeypatch):
    """A denylist of _API_KEY/_API_TOKEN/_SECRET/_PASSWORD plus a handful of
    hardcoded names let a step's verify command read tokens as common as
    GITHUB_TOKEN and DATABASE_URL — neither matches those four suffixes."""
    leaky = {
        "GITHUB_TOKEN": "ghp_x",
        "GH_TOKEN": "ghp_y",
        "NPM_TOKEN": "npm_z",
        "DATABASE_URL": "postgres://u:p@host/db",
        "OPENAI_API_KEY": "sk-x",
        "ANTHROPIC_API_KEY": "sk-ant-x",
        "STRIPE_SECRET_KEY": "sk_live_x",
        "SOME_SERVICE_TOKEN": "tok",
        "SOME_SERVICE_CREDENTIAL": "cred",
        "MY_PRIVATE_KEY": "-----BEGIN-----",
        "SLACK_WEBHOOK_URL": "https://hooks/x",
    }
    for name, value in leaky.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SAFE_SETTING", "kept")
    env = toolchain.restricted_env(tmp_path)
    for name in leaky:
        assert name not in env, f"{name} should have been stripped"
    assert env["SAFE_SETTING"] == "kept"
