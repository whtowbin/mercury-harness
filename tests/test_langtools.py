"""Which language a workspace is, detected from a marker file or, failing
that, from `scc`'s language breakdown -- never invokes a real `scc` binary
here, the same way test_grep_rg.py/test_structural.py mock a missing
external tool.
"""

from __future__ import annotations

import json

from agent import langtools, toolchain


def _fake_scc(returncode: int, stdout: str = ""):
    def fake(argv, **kwargs):
        return toolchain.ProcResult(returncode, stdout, "", 0.0)

    return fake


# -- marker-file detection -------------------------------------------------


def test_a_pyproject_at_root_is_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert langtools.detect_profile(tmp_path) is langtools.PYTHON


def test_a_package_json_at_root_is_node(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    assert langtools.detect_profile(tmp_path) is langtools.NODE


def test_an_ancestors_marker_does_not_count(tmp_path):
    """Root only -- an upward scan is exactly the escape route
    Workspace.enclosing_project() exists to refuse for Python; a language
    profile must not reopen it for Node."""
    (tmp_path / "package.json").write_text('{"name": "host"}')
    inner = tmp_path / "sandbox"
    inner.mkdir()
    assert langtools.detect_profile(inner) is not langtools.NODE


def test_pyproject_wins_when_both_markers_are_present(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "package.json").write_text('{"name": "x"}')
    assert langtools.detect_profile(tmp_path) is langtools.PYTHON


def test_a_go_mod_at_root_is_go(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    assert langtools.detect_profile(tmp_path) is langtools.GO


def test_a_cargo_toml_at_root_is_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\n')
    assert langtools.detect_profile(tmp_path) is langtools.RUST


# -- scc fallback, no marker file -------------------------------------------


def test_falls_back_to_scc_when_no_marker_file_exists(tmp_path, monkeypatch):
    breakdown = [{"Name": "JavaScript", "Count": 12}, {"Name": "Markdown", "Count": 2}]
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, json.dumps(breakdown)))
    assert langtools.detect_profile(tmp_path) is langtools.NODE


def test_the_dominant_language_by_file_count_wins(tmp_path, monkeypatch):
    breakdown = [{"Name": "Python", "Count": 3}, {"Name": "JavaScript", "Count": 40}]
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, json.dumps(breakdown)))
    assert langtools.detect_profile(tmp_path) is langtools.NODE


def test_scc_fallback_recognises_go(tmp_path, monkeypatch):
    breakdown = [{"Name": "Go", "Count": 20}, {"Name": "Markdown", "Count": 2}]
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, json.dumps(breakdown)))
    assert langtools.detect_profile(tmp_path) is langtools.GO


def test_scc_fallback_recognises_rust(tmp_path, monkeypatch):
    breakdown = [{"Name": "Rust", "Count": 20}, {"Name": "Markdown", "Count": 2}]
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, json.dumps(breakdown)))
    assert langtools.detect_profile(tmp_path) is langtools.RUST


def test_a_language_scc_reports_but_has_no_profile_for_is_ignored(tmp_path, monkeypatch):
    breakdown = [{"Name": "Haskell", "Count": 50}, {"Name": "Markdown", "Count": 2}]
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, json.dumps(breakdown)))
    assert langtools.detect_profile(tmp_path) is None


def test_missing_scc_binary_is_no_profile_not_an_error(tmp_path, monkeypatch):
    """scc absent (127, the missing-binary convention) is read the same as
    every other missing-tool case in this codebase: skip, don't guess."""
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(127, ""))
    assert langtools.detect_profile(tmp_path) is None


def test_scc_producing_unparseable_output_is_no_profile_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, "not json"))
    assert langtools.detect_profile(tmp_path) is None


def test_scc_producing_no_output_is_no_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "run_capturing", _fake_scc(0, ""))
    assert langtools.detect_profile(tmp_path) is None


# -- is_test_file ------------------------------------------------------------


def test_python_test_file_by_prefix():
    assert langtools.is_test_file(langtools.PYTHON, "pkg/test_thing.py")


def test_python_test_file_by_suffix():
    assert langtools.is_test_file(langtools.PYTHON, "pkg/thing_test.py")


def test_python_non_test_file():
    assert not langtools.is_test_file(langtools.PYTHON, "pkg/thing.py")


def test_node_test_file_by_suffix():
    assert langtools.is_test_file(langtools.NODE, "src/thing.test.ts")


def test_node_non_test_file():
    assert not langtools.is_test_file(langtools.NODE, "src/thing.ts")


def test_go_test_file_by_suffix():
    assert langtools.is_test_file(langtools.GO, "pkg/thing_test.go")


def test_go_non_test_file():
    assert not langtools.is_test_file(langtools.GO, "pkg/thing.go")


def test_rust_has_no_test_file_naming_convention_to_detect():
    """Deliberate: Rust tests live inline behind #[cfg(test)] or in a shared
    tests/ directory, neither distinguishable from ordinary source by name
    alone -- so nothing is ever reported as a Rust test file, rather than
    guessing at a convention Rust doesn't actually have."""
    assert not langtools.is_test_file(langtools.RUST, "src/lib.rs")
    assert not langtools.is_test_file(langtools.RUST, "tests/integration.rs")
