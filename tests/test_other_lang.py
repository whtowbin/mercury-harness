"""lint_project/test_project/validate_data_file: thin, lazily-loaded tools
for a detected non-Python workspace, and a mid-loop data-format check. No
real eslint/npm/yq binary is invoked here -- toolchain.run_capturing is
mocked, the same way test_langtools.py mocks it for detect_profile's scc
fallback.
"""

from __future__ import annotations

from agent import langtools, toolchain
from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.other_lang import lint_project, validate_data_file
from agent.tools.other_lang import test_project as _run_test_project
from agent.trace import Trace
from agent.workspace import Workspace


def _ctx(tmp_path, profile):
    """A ToolContext whose language_profile is fixed, not detected from the
    filesystem -- these tools' own logic is what's under test, not
    detect_profile (covered separately in test_langtools.py)."""
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    ctx.language_profile = profile
    return ctx


def _fake_run(returncode: int, stdout: str = "", stderr: str = ""):
    def fake(argv, **kwargs):
        return toolchain.ProcResult(returncode, stdout, stderr, 0.0)

    return fake


# -- lint_project / test_project --------------------------------------------


def test_lint_project_runs_the_profiles_lint_command(tmp_path, monkeypatch):
    seen = {}

    def fake(argv, **kwargs):
        seen["argv"] = argv
        return toolchain.ProcResult(0, "", "", 0.0)

    monkeypatch.setattr(toolchain, "run_capturing", fake)
    ctx = _ctx(tmp_path, langtools.NODE)
    out = lint_project(ctx, ["src/a.js"])
    assert seen["argv"] == ["npx", "--no-install", "eslint", "src/a.js"]
    assert "exit=0" in out


def test_test_project_runs_the_profiles_test_command(tmp_path, monkeypatch):
    seen = {}

    def fake(argv, **kwargs):
        seen["argv"] = argv
        return toolchain.ProcResult(0, "2 passed", "", 0.0)

    monkeypatch.setattr(toolchain, "run_capturing", fake)
    ctx = _ctx(tmp_path, langtools.NODE)
    out = _run_test_project(ctx, ["src/a.test.js"])
    assert seen["argv"] == ["npm", "test", "--", "src/a.test.js"]
    assert "2 passed" in out


def test_lint_project_refuses_a_python_workspace(tmp_path):
    ctx = _ctx(tmp_path, langtools.PYTHON)
    out = lint_project(ctx, ["a.py"])
    assert "check_code" in out


def test_test_project_tool_refuses_a_python_workspace(tmp_path):
    ctx = _ctx(tmp_path, langtools.PYTHON)
    out = _run_test_project(ctx, ["test_a.py"])
    assert "run_tests" in out


def test_lint_project_reports_no_profile_detected(tmp_path):
    ctx = _ctx(tmp_path, None)
    out = lint_project(ctx, ["a.js"])
    assert out.startswith("error:")
    assert "no language profile" in out


def test_lint_project_reports_a_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(127))
    ctx = _ctx(tmp_path, langtools.NODE)
    out = lint_project(ctx, ["a.js"])
    assert "not installed" in out


# -- validate_data_file ------------------------------------------------------


def test_validate_data_file_reports_success(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(0))
    monkeypatch.setattr("agent.dataformats.shutil.which", lambda name: "/usr/bin/yq")
    ctx = _ctx(tmp_path, None)
    out = validate_data_file(ctx, ["a.json"])
    assert out == "valid — every file parsed cleanly"


def test_validate_data_file_reports_a_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(1, "", "Error: bad file 'a.json'"))
    monkeypatch.setattr("agent.dataformats.shutil.which", lambda name: "/usr/bin/yq")
    ctx = _ctx(tmp_path, None)
    out = validate_data_file(ctx, ["a.json"])
    assert out.startswith("invalid:")
    assert "bad file" in out


def test_validate_data_file_reports_nothing_to_check(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.dataformats.shutil.which", lambda name: None)
    ctx = _ctx(tmp_path, None)
    out = validate_data_file(ctx, ["README.md"])
    assert "no validator available" in out


def test_validate_data_file_runs_a_composed_command_through_a_shell(tmp_path, monkeypatch):
    """build_validator can join several tools with && when yq is absent --
    confirm that composed command reaches the subprocess as one shell
    invocation, not a naively split argv."""
    seen = {}

    def fake(argv, **kwargs):
        seen["argv"] = argv
        return toolchain.ProcResult(0, "", "", 0.0)

    monkeypatch.setattr(toolchain, "run_capturing", fake)
    monkeypatch.setattr(
        "agent.dataformats.shutil.which", lambda name: "/usr/bin/jq" if name == "jq" else None
    )
    ctx = _ctx(tmp_path, None)
    validate_data_file(ctx, ["a.json", "b.toml"])
    assert seen["argv"][:2] == ["/bin/sh", "-c"]
    assert "jq empty a.json" in seen["argv"][2]
    assert "tomllib" in seen["argv"][2]
