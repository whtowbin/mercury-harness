"""A symbol map of the workspace's Python files, for orienting the planner.

The planner is given a file tree and then reads files to find out what is in
them. On a three-file workspace that is the right call — 950 lines fits in the
prompt and a search would be strictly worse than the code itself. But reading
answers "what does this do", and the planner also has to answer "where does
this live", because it declares the files each step touches, and getting that
wrong is what produced a traced plan whose four steps all claimed the same two
files and then duplicated each other's work.

A symbol map answers the second question at 6-8% of the cost of the source:
measured across three real workspaces, ~460 tokens against ~7,800 of code, and
19k against 235k on this repository. It says that scoring lives in
`Game.update` and preferences in `Game.show_preferences`, which is what
attribution needs, without pretending to say how either works.

Pure `ast` from the standard library: no import of the workspace's code, no
language server, no model call, and syntactically broken files are skipped
rather than raising.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .workspace import walk_files

# Methods listed per class before it is elided. A class with forty methods is
# usually a god object and its exact tail is not what the planner needs.
MAX_METHODS = 14
# The whole map, in characters. Past this a repository is large enough that
# the planner should be grepping, and a truncated map still beats none.
DEFAULT_CHAR_CAP = 12_000


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def _class_line(node: ast.ClassDef) -> str:
    methods = [
        _signature(m) for m in node.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    shown = ", ".join(methods[:MAX_METHODS])
    if len(methods) > MAX_METHODS:
        shown += f", ... (+{len(methods) - MAX_METHODS} more)"
    return f"  class {node.name}" + (f": {shown}" if shown else "")


def _constant_lines(node: ast.Assign) -> list[str]:
    """Module-level SCREAMING_CASE names. Lowercase module globals are
    implementation detail and would triple the map for nothing."""
    return [f"  {t.id}" for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]


def _node_lines(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.ClassDef):
        return [_class_line(node)]
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return [f"  def {_signature(node)}"]
    if isinstance(node, ast.Assign):
        return _constant_lines(node)
    return []


def outline_source(source: str) -> list[str]:
    """Top-level symbols of one module, as indented lines."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    return [line for node in tree.body for line in _node_lines(node)]


def _python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in walk_files(root, skip_hidden=True)
        if p.suffix == ".py" and not p.name.startswith(".")
    )


def symbol_map(root: Path, char_cap: int = DEFAULT_CHAR_CAP) -> str:
    """Every Python file's top-level symbols, or "" when there are none."""
    blocks: list[str] = []
    used = 0
    for path in _python_files(root):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = outline_source(source)
        if not lines:
            continue
        block = "\n".join([str(path.relative_to(root)), *lines])
        if used + len(block) > char_cap:
            blocks.append(f"... (symbol map truncated at {char_cap} characters)")
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)
