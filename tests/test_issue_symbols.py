"""agent/issue_symbols.py: where the goal's identifiers are defined (§128). Real files, real ast."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import agent.phases.plan as plan_module
from agent import issue_symbols
from agent.approval import Approver
from agent.config import Settings
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace

ENUMS = '''
import enum

class Choices(enum.Enum):
    """Class for creating enumerated choices."""

    def label(self):
        return 1
'''
HTTP = "def parse_http_date(date):\n    return date\n\ndef quote(value):\n    return value\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg" / "db").mkdir(parents=True)
    (root / "pkg" / "db" / "enums.py").write_text(ENUMS)
    (root / "pkg" / "http.py").write_text(HTTP)
    for i in range(issue_symbols.MAX_DEFINITIONS_PER_NAME + 1):  # defined everywhere
        (root / "pkg" / f"m{i}.py").write_text("def get_value(x):\n    return x\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_enums.py").write_text("class Choices:\n    pass\n")
    return root


def test_a_class_named_in_the_goal_is_found_with_its_file_and_line(repo):
    out = issue_symbols.issue_symbols(
        repo, "str(MyChoice.FIRST) returns the wrong thing for Choices"
    )
    assert "Choices (class): pkg/db/enums.py:4" in out


def test_a_method_is_found_with_its_class(repo):
    out = issue_symbols.issue_symbols(repo, "Choices.label returns 1")
    assert "label (method in Choices): pkg/db/enums.py:7" in out


def test_snake_case_names_beat_plain_words_and_stop_words_are_ignored(repo):
    ranked = issue_symbols.candidates("the parse_http_date function should return value when set")
    assert ranked[0][1] == "parse_http_date"
    assert "value" not in [w for _, w in ranked]


def test_a_name_defined_in_too_many_places_is_skipped(repo):
    assert issue_symbols.issue_symbols(repo, "get_value is wrong") == ""


def test_test_directories_are_not_indexed(repo):
    out = issue_symbols.issue_symbols(repo, "Choices")
    assert "tests/" not in out


def test_a_goal_with_no_matching_definition_gives_nothing(repo):
    assert issue_symbols.issue_symbols(repo, "make the button blue") == ""


def test_syntax_errors_are_skipped_not_raised(repo):
    (repo / "pkg" / "broken.py").write_text("def (:\n")
    assert "Choices" in issue_symbols.issue_symbols(repo, "Choices")


# -- the planner's message ----------------------------------------------------------------


def _planner_user_message(root: Path, tmp_path: Path, monkeypatch, **overrides) -> str:
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = dataclasses.replace(
        Settings(workspace=str(root), goal="Choices.label returns 1"), **overrides
    )
    seen: dict = {}

    def fake_loop(**kwargs):
        seen["user"] = kwargs["user"]
        return {"draft": [], "steps": []}, []

    monkeypatch.setattr(plan_module, "run_tool_loop", fake_loop)
    plan_module.make_plan(None, settings, ctx, ctx.trace)
    return seen["user"]


def test_off_by_default_the_planner_message_has_no_such_section(repo, tmp_path, monkeypatch):
    user = _planner_user_message(repo, tmp_path, monkeypatch)
    assert "Where the names in the goal are defined" not in user


def test_on_the_section_comes_before_the_instruction_to_decide_files(repo, tmp_path, monkeypatch):
    user = _planner_user_message(repo, tmp_path, monkeypatch, plan_issue_symbols=True)
    section = user.index("Where the names in the goal are defined")
    assert "label (method in Choices): pkg/db/enums.py:7" in user
    assert section < user.index("Use this to decide which files each step touches")


def test_on_with_nothing_found_the_message_is_the_default_one(repo, tmp_path, monkeypatch):
    off = _planner_user_message(repo, tmp_path, monkeypatch)
    # a goal that names nothing in the workspace: same text as with the flag off
    ctx_goal = dataclasses.replace(Settings(workspace=str(repo), goal="make the button blue"))
    assert issue_symbols.issue_symbols(repo, ctx_goal.goal) == "" and off
