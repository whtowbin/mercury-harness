"""`_already_proven_by_tests`: skip a step's verify command only when the
harness itself already ran and passed exactly what it names.

A mixed pytest invocation -- a bare file alongside a node id, e.g.
`pytest test_a.py test_b.py::test_x` -- used to be mis-scored: the
node-id-suffixed argument doesn't itself end in `.py` (it ends in the test
name), so filtering on `.endswith(".py")` before splitting off the `::` part
silently dropped it from consideration entirely. A prior green run of only
`test_a.py` could then be read as proof the whole mixed command already
passed, even though `test_b.py::test_x` was never actually run.
"""

from __future__ import annotations

from agent.phases.execute import _already_proven_by_tests
from agent.tools import base as base_module


class _Settings:
    skip_verified_by_tests = True


class _Ctx:
    def __init__(self, root, tests_green):
        self.workspace = type("W", (), {"root": root})()
        self.tests_green = tests_green


def _patch_state(monkeypatch, value):
    monkeypatch.setattr(base_module, "workspace_state", lambda root: value)
    # execute.py imports workspace_state locally inside the function, from
    # the same module -- patching the source is what that import re-reads.


def test_a_command_naming_only_recorded_files_is_proven(tmp_path, monkeypatch):
    _patch_state(monkeypatch, "state-a")
    ctx = _Ctx(tmp_path, tests_green=("state-a", ("tests/test_a.py",)))
    assert _already_proven_by_tests(ctx, "uv run pytest -q tests/test_a.py", _Settings())


def test_a_mixed_file_and_node_id_is_not_proven_by_a_narrower_record(tmp_path, monkeypatch):
    """The bug: test_b.py::test_x was never covered by the recorded run, but
    the old filter dropped it from `wanted` entirely, so this was wrongly
    treated as fully proven."""
    _patch_state(monkeypatch, "state-a")
    ctx = _Ctx(tmp_path, tests_green=("state-a", ("tests/test_a.py",)))
    command = "uv run pytest -q tests/test_a.py tests/test_b.py::test_x"
    assert not _already_proven_by_tests(ctx, command, _Settings())


def test_a_node_id_whose_whole_file_was_already_recorded_is_proven(tmp_path, monkeypatch):
    """The correct direction to be permissive in: if test_b.py's whole file
    was already green, a specific node id within it necessarily was too."""
    _patch_state(monkeypatch, "state-a")
    ctx = _Ctx(tmp_path, tests_green=("state-a", ("tests/test_a.py", "tests/test_b.py")))
    command = "uv run pytest -q tests/test_a.py tests/test_b.py::test_x"
    assert _already_proven_by_tests(ctx, command, _Settings())


def test_a_whole_suite_record_proves_any_node_id(tmp_path, monkeypatch):
    _patch_state(monkeypatch, "state-a")
    ctx = _Ctx(tmp_path, tests_green=("state-a", (".",)))
    assert _already_proven_by_tests(ctx, "uv run pytest -q x.py::test_y", _Settings())


def test_a_changed_workspace_is_never_proven(tmp_path, monkeypatch):
    _patch_state(monkeypatch, "state-b")  # different from what was recorded
    ctx = _Ctx(tmp_path, tests_green=("state-a", ("tests/test_a.py",)))
    assert not _already_proven_by_tests(ctx, "uv run pytest -q tests/test_a.py", _Settings())


def test_no_recorded_run_is_never_proven(tmp_path, monkeypatch):
    _patch_state(monkeypatch, "state-a")
    ctx = _Ctx(tmp_path, tests_green=None)
    assert not _already_proven_by_tests(ctx, "uv run pytest -q tests/test_a.py", _Settings())
