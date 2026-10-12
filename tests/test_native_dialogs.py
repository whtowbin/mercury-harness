"""Native folder-picker dispatch: platform routing, cancel/missing-binary
handling. Never invokes a real dialog -- every test monkeypatches
toolchain.run_capturing and agent.native_dialogs.platform.system(), the same
way test_grep_rg.py/test_structural.py mock a missing external tool.
"""

from __future__ import annotations

from agent import native_dialogs, toolchain


def _fake_run(returncode: int, stdout: str = ""):
    def fake(argv, **kwargs):
        return toolchain.ProcResult(returncode, stdout, "", 0.0)

    return fake


def test_macos_returns_the_chosen_path(monkeypatch):
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(0, "/Users/x/project\n"))
    assert native_dialogs.choose_folder() == "/Users/x/project"


def test_macos_cancel_returns_none(monkeypatch):
    """AppleScript error -128 (the user clicked Cancel) is a non-zero exit,
    read the same as any other failure -- there is nothing more useful to
    tell the caller than "nothing was chosen"."""
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(1, ""))
    assert native_dialogs.choose_folder() is None


def test_macos_calls_osascript_with_the_prompt_text(monkeypatch):
    seen = {}

    def fake(argv, **kwargs):
        seen["argv"] = argv
        return toolchain.ProcResult(0, "/tmp\n", "", 0.0)

    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(toolchain, "run_capturing", fake)
    native_dialogs.choose_folder("Pick the workspace")
    assert seen["argv"][0] == "osascript"
    assert "Pick the workspace" in seen["argv"][2]


def test_macos_escapes_quotes_in_the_prompt(monkeypatch):
    """An unescaped quote in the prompt would break out of the AppleScript
    string literal it is embedded in."""
    seen = {}

    def fake(argv, **kwargs):
        seen["argv"] = argv
        return toolchain.ProcResult(0, "/tmp\n", "", 0.0)

    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(toolchain, "run_capturing", fake)
    native_dialogs.choose_folder('a "quoted" prompt')
    assert '\\"quoted\\"' in seen["argv"][2]


def test_linux_returns_the_chosen_path(monkeypatch):
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Linux")
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(0, "/home/x/project\n"))
    assert native_dialogs.choose_folder() == "/home/x/project"


def test_linux_missing_zenity_returns_none(monkeypatch):
    """run_capturing's own 127-for-a-missing-binary convention (toolchain.py)
    reaches here as just another non-zero exit -- no native picker available
    is not distinguished from a cancel, same as the macOS path."""
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Linux")
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(127, ""))
    assert native_dialogs.choose_folder() is None


def test_unsupported_platform_returns_none_without_running_anything(monkeypatch):
    called = []
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        toolchain,
        "run_capturing",
        lambda *a, **k: called.append(1) or toolchain.ProcResult(0, "", "", 0.0),
    )
    assert native_dialogs.choose_folder() is None
    assert called == []


def test_empty_stdout_is_treated_as_no_selection(monkeypatch):
    """Belt and suspenders against a picker that exits 0 with nothing on
    stdout -- an empty path is not a workspace."""
    monkeypatch.setattr(native_dialogs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(toolchain, "run_capturing", _fake_run(0, "\n"))
    assert native_dialogs.choose_folder() is None
