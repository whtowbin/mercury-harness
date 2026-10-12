"""Which tool calls edit the workspace, and what each one edits.

Every place that asks "did this call change a file, and which" reads it here. Each used to keep
its own list, and none of them learnt about `edit_blocks` when it was added (DESIGN-LOG §104): a
turn that edited only through it got no after-edit check (0 of 23 such turns in §105-§106's
traces, against 348 of 390 for the other edit tools) and passed the premise gate unexamined, so
§105.2's and §106's edit_blocks arms ran without both (§106.1). `tests/test_editcalls.py` fails
when an execution tool that writes files is missing from EDIT_TOOLS.

No imports from the rest of the package, so the trace tools can use it too.
"""

from __future__ import annotations

# Tools that change a file's content.
EDIT_TOOLS = frozenset(
    {"write_file", "edit_file", "edit_files", "edit_blocks", "batch_edits_to_file"}
)
# Tools that change the workspace: the edits, and deleting a file.
CHANGE_TOOLS = EDIT_TOOLS | {"delete_file"}

SEARCH_MARK, DIVIDER_MARK, REPLACE_MARK = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"


def parse_blocks(text: str) -> list[tuple[str, str, str]] | str:
    """[(path, search, replace)] from edit_blocks' format, or an `error:` saying what is wrong."""
    lines = text.splitlines()
    out: list[tuple[str, str, str]] = []
    path = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == SEARCH_MARK:
            if not path:
                return f"error: block {len(out) + 1} has no file path on a line before it"
            divider = _next_mark(lines, i + 1, DIVIDER_MARK)
            end = _next_mark(lines, divider + 1, REPLACE_MARK) if divider >= 0 else -1
            if divider < 0 or end < 0:
                return (
                    f"error: block {len(out) + 1} ({path}) is not closed: it needs a "
                    f"{DIVIDER_MARK!r} line and then a {REPLACE_MARK!r} line"
                )
            out.append(
                (path, "\n".join(lines[i + 1 : divider]), "\n".join(lines[divider + 1 : end]))
            )
            i = end + 1
            continue
        if line and not line.startswith("```"):
            path = line.strip("`").removeprefix("File:").strip()
        i += 1
    if not out:
        return (
            "error: no blocks found. Write a file path line, then "
            f"{SEARCH_MARK} / text to find / {DIVIDER_MARK} / replacement / {REPLACE_MARK}"
        )
    return out


def _next_mark(lines: list[str], start: int, mark: str) -> int:
    return next((j for j in range(start, len(lines)) if lines[j].strip() == mark), -1)


def string_edits(name: str, args: dict) -> list[dict]:
    """Each search/replace edit a call asks for, as {path, old_string, new_string,
    replace_all, create}. `create` marks an edit_blocks block with an empty SEARCH, which
    creates its file. write_file and delete_file are not search/replace edits: []."""
    if name == "edit_file":
        return [_edit(args, str(args.get("path", "")))]
    if name in ("edit_files", "batch_edits_to_file"):
        fallback = str(args.get("path", ""))
        return [
            _edit(spec, str(spec.get("path") or fallback))
            for spec in args.get("edits") or []
            if isinstance(spec, dict)
        ]
    if name == "edit_blocks":
        parsed = parse_blocks(str(args.get("blocks") or ""))
        if isinstance(parsed, str):
            return []
        return [
            {
                "path": path,
                "old_string": old,
                "new_string": new,
                "replace_all": False,
                "create": not old.strip(),
            }
            for path, old, new in parsed
        ]
    return []


def _edit(spec: dict, path: str) -> dict:
    return {
        "path": path,
        "old_string": str(spec.get("old_string", "")),
        "new_string": str(spec.get("new_string", "")),
        "replace_all": bool(spec.get("replace_all")),
        "create": False,
    }


def edited_paths(name: str, args: dict) -> list[str]:
    """The file each edit in a call targets, one entry per edit (a file edited twice in one
    call appears twice). A batch whose edits name no path falls back to the call's own."""
    if name == "write_file":
        return [str(args.get("path", ""))]
    paths = [edit["path"] for edit in string_edits(name, args)]
    if not paths and name in ("edit_files", "batch_edits_to_file"):
        return [str(args.get("path", ""))]
    return paths


def written_text(name: str, args: dict) -> list[tuple[str, str]]:
    """(path, text the call writes) for each part of an edit call: write_file's content, or
    each edit's replacement text."""
    if name == "write_file":
        return [(str(args.get("path", "")), str(args.get("content", "")))]
    return [(edit["path"], edit["new_string"]) for edit in string_edits(name, args)]
