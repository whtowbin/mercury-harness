"""write_file notes content that carries read_file's line numbers (DESIGN-LOG §93).

A GAIA fix round read a one-line answer file (`1\t1080`) and wrote that text back, prefix and
all; the file was scored wrong. The write still happens: a TSV can start 1, 2, 3.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.text import numbered
from agent.tools import ToolContext
from agent.tools.fs import COPIED_NUMBERS_NOTE, copied_line_numbers, read_file, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


@pytest.mark.parametrize(
    "content",
    [
        "1\t1080\n",
        "1\tRichard Gordon;18720",
        "1\t2:03.56\n# Note: Estimated time for a verified track",
        numbered("a\nb\nc"),
        numbered("x\ny", start=40),
    ],
)
def test_read_file_output_is_recognised(content):
    assert copied_line_numbers(content)


@pytest.mark.parametrize(
    "content",
    ["1080\n", "", "id\tname\n1\ta\n2\tb", "2\ta\n4\tb", "def f():\n    return 1\t# x\n"],
)
def test_ordinary_content_is_not(content):
    assert not copied_line_numbers(content)


def test_a_copied_read_is_written_and_noted(ctx, tmp_path):
    (tmp_path / "answer.txt").write_text("1080\n")
    shown = read_file(ctx, "answer.txt")
    result = write_file(ctx, "answer.txt", shown + "\n")
    assert COPIED_NUMBERS_NOTE in result
    assert (tmp_path / "answer.txt").read_text() == "1\t1080\n"


def test_a_clean_write_has_no_note(ctx):
    assert COPIED_NUMBERS_NOTE not in write_file(ctx, "answer.txt", "1080\n")
