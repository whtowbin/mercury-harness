"""Where the names in the issue are defined, for the planner's first message (DESIGN-LOG §128).

`outline.symbol_map` lists the workspace's top-level symbols, but it walks the files in
alphabetical order and stops at a character cap. On a large repository that is an arbitrary
prefix: on django it covers `django/apps` to part of `django/contrib/admin` and never reaches
`django/db/models/enums.py` or `django/utils/http.py`. The planner then spends calls (10 of
a Sonnet run's 20) searching for the place the issue is about, and a weak model that guesses
wrong edits the wrong file (Mercury, django-11964, six runs of six).

This does what a first search would: take the identifiers in the goal, look them up in an
index of every Python definition, and list where the specific ones are defined. No model call,
no import of the workspace's code, nothing provider-specific, and it changes nothing for a
workspace with no matching definition.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .workspace import walk_files

MAX_LINES = 14
MAX_FILES_SCANNED = 6_000
# A plain lowercase word (value, name, error) is defined in dozens of places and says nothing about
# where the issue is; a name shared by more than this many definitions is skipped.
MAX_DEFINITIONS_PER_NAME = 6
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# English words and Python vocabulary that also name definitions often enough to drown the list.
_STOP_WORDS = """
the and for not are but with from this that when then than into have has had its was
were will would should could can may might must does did done get gets got set sets use
used using class def self cls none true false return returns import raise try except
else elif while yield lambda pass break continue global del assert print type types
object objects value values name names key keys item items data field fields model
models form forms test tests error errors call calls case issue bug code file files line
lines list dict string int float bool bytes default option options argument arguments
args kwargs parameter attribute method function module instance create created update
delete query queryset django
"""
_STOP = frozenset(_STOP_WORDS.split())


@dataclass(frozen=True)
class Definition:
    name: str
    kind: str  # class, function or method
    path: str
    line: int
    owner: str = ""  # the class a method belongs to

    def render(self) -> str:
        where = f"{self.path}:{self.line}"
        scope = f" in {self.owner}" if self.owner else ""
        return f"{self.name} ({self.kind}{scope}): {where}"


def _definitions(source: str, path: str) -> list[Definition]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[Definition] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found.append(Definition(node.name, "class", path, node.lineno))
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.append(Definition(member.name, "method", path, member.lineno, node.name))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(Definition(node.name, "function", path, node.lineno))
    return found


def _in_tests(relative: Path) -> bool:
    """Test code defines the same class names as the code it tests and is not where a fix goes."""
    return any(part in ("test", "tests", "testing") for part in relative.parts[:-1]) or (
        relative.name.startswith("test_") or relative.name == "conftest.py"
    )


def build_index(root: Path) -> dict[str, list[Definition]]:
    """Every class, top-level function and method in the workspace's Python files, by name."""
    index: dict[str, list[Definition]] = {}
    files = [
        p
        for p in walk_files(root, skip_hidden=True)
        if p.suffix == ".py" and not _in_tests(p.relative_to(root))
    ]
    for path in sorted(files)[:MAX_FILES_SCANNED]:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for definition in _definitions(source, str(path.relative_to(root))):
            index.setdefault(definition.name, []).append(definition)
    return index


def candidates(goal: str) -> list[tuple[int, str]]:
    """Identifiers in the goal with a specificity score (higher first): a CamelCase or
    snake_case name, or one in backticks or after a dot, beats a lowercase word."""
    scores: dict[str, int] = {}
    for match in _WORD.finditer(goal):
        word = match.group(0)
        if word.lower() in _STOP:
            continue
        score = 1
        if any(c.isupper() for c in word[1:]) or "_" in word.strip("_"):
            score += 2  # CamelCase or snake_case: it is a name, not a word
        before, after = (
            goal[max(0, match.start() - 1) : match.start()],
            goal[match.end() : match.end() + 1],
        )
        if before in ("`", ".") or after in ("`", "(", "."):
            score += 1
        scores[word] = max(scores.get(word, 0), score)
    return sorted(((s, w) for w, s in scores.items()), key=lambda t: (-t[0], t[1]))


def issue_definitions(
    root: Path, goal: str, index: dict[str, list[Definition]] | None = None
) -> list[Definition]:
    """Where the goal's identifiers are defined, most specific name first, at most MAX_LINES."""
    index = build_index(root) if index is None else index
    found: list[Definition] = []
    seen: set[tuple[str, int]] = set()
    for _score, word in candidates(goal):
        defs = index.get(word, [])
        if not defs or len(defs) > MAX_DEFINITIONS_PER_NAME:
            continue
        for d in defs:
            if (d.path, d.line) not in seen:
                seen.add((d.path, d.line))
                found.append(d)
        if len(found) >= MAX_LINES:
            break
    return found[:MAX_LINES]


def issue_symbols(root: Path, goal: str, index: dict[str, list[Definition]] | None = None) -> str:
    """Lines saying where the goal's identifiers are defined, or "" when none are."""
    return "\n".join(d.render() for d in issue_definitions(root, goal, index))
