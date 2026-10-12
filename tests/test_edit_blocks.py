"""edit_blocks: several edits in one plain-text argument, SEARCH/REPLACE blocks under a path
line, applied through edit_file (DESIGN-LOG §104). Real files throughout."""

from __future__ import annotations

from agent.approval import Approver
from agent.tools import ToolContext, execution_registry, with_edit_blocks
from agent.tools.fs import edit_blocks
from agent.trace import Trace
from agent.workspace import Workspace


def _ctx(tmp_path) -> ToolContext:
    (tmp_path / "main.py").write_text("def draw(self, screen):\n    pass\n\nSPEED = 3\n")
    (tmp_path / "test_main.py").write_text("def test_speed():\n    assert True\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


BLOCKS = """main.py
<<<<<<< SEARCH
def draw(self, screen):
=======
def draw(self, screen, camera_x=0):
>>>>>>> REPLACE
<<<<<<< SEARCH
SPEED = 3
=======
SPEED = 5
>>>>>>> REPLACE
test_main.py
<<<<<<< SEARCH
    assert True
=======
    assert 5 == 5
>>>>>>> REPLACE
"""


def test_several_blocks_across_files_in_one_call(tmp_path):
    out = edit_blocks(_ctx(tmp_path), BLOCKS)
    assert out.startswith("3/3 blocks applied")
    main = (tmp_path / "main.py").read_text()
    assert "def draw(self, screen, camera_x=0):" in main and "SPEED = 5" in main
    assert "assert 5 == 5" in (tmp_path / "test_main.py").read_text()


def test_a_block_that_does_not_match_does_not_stop_the_others(tmp_path):
    bad = BLOCKS.replace("SPEED = 3\n=======", "SPEED = 99\n=======")
    out = edit_blocks(_ctx(tmp_path), bad)
    assert out.startswith("2/3 blocks applied")
    assert "--- block 2: main.py ---\nerror" in out
    assert "camera_x=0" in (tmp_path / "main.py").read_text()


def test_an_empty_search_creates_a_new_file(tmp_path):
    out = edit_blocks(
        _ctx(tmp_path), "camera.py\n<<<<<<< SEARCH\n=======\nOFFSET = 0\n>>>>>>> REPLACE\n"
    )
    assert out.startswith("1/1 blocks applied")
    assert (tmp_path / "camera.py").read_text() == "OFFSET = 0\n"


def test_malformed_input_says_what_is_wrong(tmp_path):
    ctx = _ctx(tmp_path)
    assert "no file path" in edit_blocks(ctx, "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n")
    assert "is not closed" in edit_blocks(ctx, "main.py\n<<<<<<< SEARCH\nSPEED = 3\n=======\n")
    assert "no blocks found" in edit_blocks(ctx, "just prose")


def test_the_kit_swap_replaces_the_list_based_batch_tools():
    kit = execution_registry("edit the game")
    assert "edit_files" in kit.names() and "edit_blocks" not in kit.names()
    swapped = with_edit_blocks(kit)
    assert "edit_blocks" in swapped.names()
    assert not {"edit_files", "batch_edits_to_file"} & set(swapped.names())
    assert len(swapped.names()) == len(kit.names())
