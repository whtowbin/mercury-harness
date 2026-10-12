"""The clarify call sees the code before asking (Settings.clarify_context, DESIGN-LOG §92)."""

from __future__ import annotations

from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.llm import Reply, Usage
from agent.phases.clarify import gather_clarifications
from agent.prompts import CLARIFY_CONTEXT_TAIL
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


class Recording:
    def __init__(self):
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append(messages)
        return Reply(content="CLEAR", usage=Usage(10, 5, 0.0, 0.1))


def asked(tmp_path, **flags) -> str:
    (tmp_path / "game.py").write_text("class Bird:\n    def flap(self):\n        pass\n")
    (tmp_path / "README.md").write_text("# Flappy\nA small pygame clone.\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_game.py").write_text("def test_x():\n    pass\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        interactor=Interactor(ask=lambda q, o=None: "", enabled=True),
    )
    settings = Settings(workspace=str(tmp_path), goal="make the bird flap higher", **flags)
    client = Recording()
    gather_clarifications(client, settings, ctx, ctx.trace)
    return client.sent[0][1]["content"]


def test_off_shows_file_names_only(tmp_path):
    text = asked(tmp_path, clarify_context=False)
    assert text.endswith(Workspace(tmp_path).tree())
    assert "Bird" not in text and CLARIFY_CONTEXT_TAIL not in text


def test_on_shows_symbols_readme_and_tests_with_the_rule_last(tmp_path):
    text = asked(tmp_path, clarify_context=True)
    assert "class Bird" in text and "flap" in text
    assert "A small pygame clone." in text
    assert "Tests run with: pytest" in text
    assert text.endswith(CLARIFY_CONTEXT_TAIL)


def test_a_project_test_command_wins_over_pytest(tmp_path):
    text = asked(tmp_path, clarify_context=True, test_command="python tests/runtests.py")
    assert "Tests run with: python tests/runtests.py" in text


def test_with_context_the_record_of_earlier_runs_comes_just_before_the_rule(tmp_path):
    (tmp_path / "game.py").write_text("x = 1\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        interactor=Interactor(ask=lambda q, o=None: "", enabled=True),
    )
    for flag in (False, True):
        settings = Settings(workspace=str(tmp_path), goal="fix those issues", clarify_context=flag)
        client = Recording()
        gather_clarifications(client, settings, ctx, ctx.trace, remembered="LAST RUN: SDL2 missing")
        text = client.sent[0][1]["content"]
        if flag:
            assert text.index("LAST RUN: SDL2 missing") > text.index("Workspace map")
            assert text.endswith(CLARIFY_CONTEXT_TAIL)
        else:
            assert text.index("LAST RUN: SDL2 missing") < text.index("Workspace map")
