"""File tools. Edits are exact-match search/replace, which fails loudly.

Every write is followed by a syntax check and (by default) a Ruff pass on the file
that changed, and the result is appended to the tool output. The model therefore
learns it broke something on the same turn it broke it.
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import os
import re
import shutil
from pathlib import Path

from .. import toolchain, validate
from ..editcalls import parse_blocks
from ..text import numbered
from ..workspace import IGNORED_DIRS, exclude_globs, walk_files
from .base import Tool, ToolContext, obj

MAX_READ_BYTES = 120_000
# Traced sessions are full of `read_file(offset=401)` — the model hitting this
# limit and coming back for the rest, one round trip at a time. A round trip
# costs ~1.3s and the extra lines cost a few hundredths, so the default reads
# most real files whole.
DEFAULT_LINE_LIMIT = 1_200
RG_TIMEOUT_S = 20


def _read_text(ctx: ToolContext, path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        raise ValueError("file appears to be binary")
    return ctx.read_text(path)


def _post_write(ctx: ToolContext, path: Path, rel: str, source: str | None = None) -> str:
    ctx.cache.invalidate(path)
    if source is not None:
        ctx.cache.put(path, source)
    return validate.after_write(ctx.workspace.root, rel, source, lint=ctx.autolint)


def _listed_files(dirpath: str, filenames: list[str]) -> list[str]:
    """One `  name  (size b)` line per non-hidden file in this directory."""
    lines = []
    for name in sorted(filenames):
        if name.startswith("."):
            continue
        size = (Path(dirpath) / name).stat().st_size
        lines.append(f"  {name}  ({size}b)")
    return lines


def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> str:
    root = ctx.workspace.resolve(path)
    if not root.is_dir():
        return f"error: not a directory: {path}"
    out: list[str] = []
    base_depth = str(root).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        rel = ctx.workspace.rel(Path(dirpath))
        out.append(f"{rel}/" if rel != "." else ".")
        out.extend(_listed_files(dirpath, filenames))
        if len(out) > 500:
            out.append("... truncated")
            break
    return "\n".join(out)


# A narrower window on a Python file is widened to the function or class around it, at most
# this long (DESIGN-LOG §104): the platformer app's re-reads were mostly the next small window of
# the same file, each one a model call.
MIN_READ_WINDOW = 60
MAX_WIDENED_WINDOW = 250


def read_file(
    ctx: ToolContext,
    path: str,
    offset: int = 1,
    limit: int = DEFAULT_LINE_LIMIT,
    sections: str = "",
) -> str:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        return f"error: no such file: {path}"
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        return _read_slice(target, path, size, offset, limit)
    try:
        text = _read_text(ctx, target)
    except ValueError as exc:
        return f"error: {exc}"
    lines = text.splitlines()
    if str(sections or "").strip():
        return _read_sections(ctx, path, text, lines, str(sections))
    offset = max(1, offset)
    start, end, note = offset, offset - 1 + max(1, limit), ""
    assist = getattr(ctx, "read_assist", True)
    if assist and path.endswith(".py") and end - start + 1 < MIN_READ_WINDOW:
        start, end, note = _widen(text, len(lines), start, end)
    end = min(end, len(lines))
    seen = _already_shown(ctx, path, text, start, end) if assist else ""
    if seen:
        return seen
    body = numbered("\n".join(lines[start - 1 : end]), start=start)
    footer = ""
    if end < len(lines):
        footer = (
            f"\n... {len(lines) - end} more lines (file has {len(lines)}). Use offset={end + 1}."
        )
    return (note + body + footer) if body else "(empty file)"


def _python_blocks(text: str) -> list[tuple[str, int, int]]:
    """(qualified name, first line, last line) for every function and class, nested ones too."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                first = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                out.append((name, first, child.end_lineno or child.lineno))
                walk(child, name + ".")

    walk(tree, "")
    return out


def _widen(text: str, total: int, start: int, end: int) -> tuple[int, int, str]:
    """The smallest function or class holding [start, end], if it fits; else a window of at
    least MIN_READ_WINDOW lines around it. Returns (start, end, a note saying what changed)."""
    holding = [b for b in _python_blocks(text) if b[1] <= start and end <= b[2]]
    holding = [b for b in holding if b[2] - b[1] + 1 <= MAX_WIDENED_WINDOW]
    if holding:
        name, first, last = min(holding, key=lambda b: b[2] - b[1])
        if (first, last) != (start, end):
            return first, last, f"(widened to lines {first}-{last}, the whole `{name}`)\n"
        return start, end, ""
    pad = max(0, MIN_READ_WINDOW - (end - start + 1)) // 2
    first, last = max(1, start - pad), min(total, end + pad)
    if (first, last) == (start, end):
        return start, end, ""
    return first, last, f"(widened to lines {first}-{last})\n"


def _already_shown(ctx: ToolContext, path: str, text: str, start: int, end: int) -> str:
    """A one-line answer when this loop has already shown exactly these unchanged lines."""
    key = (path, start, end)
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    seen = getattr(ctx, "read_seen", None)
    if seen is None:
        return ""
    epoch = getattr(ctx, "read_epoch", 0)
    if seen.get(key) == (digest, epoch):
        return (
            f"(lines {start}-{end} of {path} are unchanged since you read them earlier in this "
            "step; that output above is still current.)"
        )
    seen[key] = (digest, epoch)
    return ""


def _read_sections(ctx: ToolContext, path: str, text: str, lines: list[str], spec: str) -> str:
    """Several parts of one file in one call: line ranges ("40-80") and Python names
    ("Game.update"), each with its own line numbers."""
    blocks = {name: (first, last) for name, first, last in _python_blocks(text)}
    out: list[str] = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        span = _section_span(item, blocks, len(lines))
        if isinstance(span, str):
            out.append(f"--- {item} ---\n{span}")
            continue
        start, end = span
        seen = _already_shown(ctx, path, text, start, end)
        body = seen or numbered("\n".join(lines[start - 1 : end]), start=start)
        out.append(f"--- {item}: lines {start}-{end} ---\n{body}")
    return "\n\n".join(out) or "error: sections named nothing to read"


def _section_span(
    item: str, blocks: dict[str, tuple[int, int]], total: int
) -> tuple[int, int] | str:
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", item)
    if match:
        start, end = max(1, int(match.group(1))), min(total, int(match.group(2)))
        return (start, end) if start <= end else f"error: no lines {item} (file has {total})"
    if item in blocks:
        return blocks[item]
    tails = [name for name in blocks if name.endswith("." + item)]
    if len(tails) == 1:
        return blocks[tails[0]]
    known = ", ".join(sorted(blocks)[:12])
    return f"error: no function or class named {item!r} here. Names include: {known}"


def _read_slice(target: Path, path: str, size: int, offset: int, limit: int) -> str:
    """A window of a file too large to load whole. Streams the lines, so memory
    stays bounded however large the file is. The error this replaces told the
    model to read in slices, then refused any read of the file at all."""
    offset = max(1, offset)
    limit = max(1, min(limit, DEFAULT_LINE_LIMIT))
    window: list[str] = []
    total = 0
    with target.open(encoding="utf-8", errors="replace") as handle:
        for total, line in enumerate(handle, start=1):
            if offset <= total < offset + limit:
                window.append(line.rstrip("\n"))
    if not window:
        return f"error: {path} has {total} lines; offset {offset} is past the end"
    end = offset - 1 + len(window)
    note = (
        f"\n... {path} is {size}b, over the {MAX_READ_BYTES}b whole-file cap; "
        f"lines {offset}-{end} of {total}."
    )
    if end < total:
        note += f" Use offset={end + 1} for more, or grep it."
    return numbered("\n".join(window), start=offset) + note


def read_files(ctx: ToolContext, paths: list[str], limit: int = 600) -> str:
    """Read several files in one call.

    Round trips, not generation, dominate a fast model's wall time: a call costs
    roughly a second of queue and prefill regardless of size. Reading four files
    in one call rather than four saves about three seconds of pure overhead.
    """
    if not paths:
        return "error: paths must be a non-empty list of file paths"
    if isinstance(paths, str):
        paths = [paths]
    blocks: list[str] = []
    for rel in paths[:20]:
        try:
            target = ctx.workspace.resolve(str(rel))
        except Exception as exc:
            blocks.append(f"=== {rel} ===\nerror: {exc}")
            continue
        if not target.is_file():
            blocks.append(f"=== {rel} ===\nerror: no such file")
            continue
        blocks.append(f"=== {rel} ===\n{read_file(ctx, str(rel), 1, limit)}")
    if len(paths) > 20:
        blocks.append(f"... {len(paths) - 20} more paths ignored; ask for them separately")
    return "\n\n".join(blocks)


_NUMBERED_LINE = re.compile(r"^(\d+)\t")


def copied_line_numbers(content: str) -> bool:
    """Whether `content` carries read_file's `N<tab>` line prefixes: it opens with `1<tab>`,
    or every line has consecutive prefixes (a copied window further down a file). GAIA fix
    rounds wrote `1\t1080` into a one-line answer file this way, once with a note line after
    it (3 GAIA write_file calls, DESIGN-LOG §93; no write or edit of ~26,000 in code runs
    opens with `1<tab>`)."""
    if content.startswith("1\t"):
        return True
    lines = content.splitlines()
    found = [_NUMBERED_LINE.match(line) for line in lines]
    if not lines or not all(found):
        return False
    numbers = [int(m.group(1)) for m in found if m]
    return numbers == list(range(numbers[0], numbers[0] + len(numbers)))


COPIED_NUMBERS_NOTE = (
    "\nnote: every line written starts with a line number and a tab, as read_file shows "
    "files. Those numbers are not part of the file; if they were copied, write it again "
    "without them."
)


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = ctx.workspace.resolve(path)
    full = ctx.guard.check_before_write(ctx.workspace.root)
    if full:
        return full
    oversized = ctx.guard.check_write_size(len(content.encode("utf-8")))
    if oversized:
        return oversized
    target.parent.mkdir(parents=True, exist_ok=True)
    rel = ctx.workspace.rel(target)
    ctx.before_write(rel, target)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    ctx.mark_touched(rel)
    ctx.trace.event(
        "write_file",
        path=rel,
        bytes=len(content),
        overwrote=existed,
        lines=len(content.splitlines()),
    )
    verb = "overwrote" if existed else "created"
    note = COPIED_NUMBERS_NOTE if copied_line_numbers(content) else ""
    return (
        f"{verb} {path} ({len(content.splitlines())} lines)"
        + note
        + _post_write(ctx, target, rel, content)
    )


def edit_blocks(ctx: ToolContext, blocks: str) -> str:
    """Several edits in one plain-text argument: a file path line, then SEARCH/REPLACE blocks.

    `edit_files` takes a list of objects, which models often send as a JSON string that then
    fails to parse: 99 of 2,474 calls (4.0%), against none for `edit_file`'s plain strings
    (DESIGN-LOG §104). Here the whole batch is one string, escaped once like edit_file's, and
    each block is applied through edit_file itself, so the SEARCH text is the position marker
    exactly as old_string is. An empty SEARCH creates a new file.
    """
    parsed = parse_blocks(str(blocks or ""))
    if isinstance(parsed, str):
        return parsed
    results: list[str] = []
    applied = 0
    for n, (path, old, new) in enumerate(parsed, 1):
        if not old.strip():
            if ctx.workspace.resolve(path).exists():
                result = (
                    f"error: block {n} has an empty SEARCH, but {path} exists; "
                    "quote the text to replace"
                )
            else:
                result = write_file(ctx, path, new + "\n")
        else:
            result = edit_file(ctx, path, old, new)
        ok = not result.startswith("error")
        applied += ok
        results.append(f"--- block {n}: {path} ---\n{result}")
    return f"{applied}/{len(parsed)} blocks applied\n\n" + "\n\n".join(results)


def move_file(ctx: ToolContext, src: str, dst: str) -> str:
    """Move or rename a file. Refuses to overwrite an existing dst — delete_file
    it first if that's really what's wanted, so an overwrite is never a surprise."""
    source = ctx.workspace.resolve(src)
    if not source.is_file():
        return f"error: no such file: {src}"
    target = ctx.workspace.resolve(dst)
    if target.exists():
        return f"error: {dst} already exists. delete_file it first if you mean to replace it."

    rel_src, rel_dst = ctx.workspace.rel(source), ctx.workspace.rel(target)
    # Snapshot both sides before the move: the source's baseline records what
    # is now gone, the destination's records that this content is new here —
    # the same "capture on first touch" rule write_file follows.
    ctx.before_write(rel_src, source)
    ctx.before_write(rel_dst, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    ctx.cache.invalidate(source)
    ctx.mark_touched(rel_src)
    ctx.mark_touched(rel_dst)
    ctx.trace.event("move_file", src=rel_src, dst=rel_dst)
    return f"moved {src} -> {dst}" + _post_write(ctx, target, rel_dst)


def delete_file(ctx: ToolContext, path: str) -> str:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        return f"error: no such file: {path}"
    rel = ctx.workspace.rel(target)
    ctx.before_write(rel, target)
    target.unlink()
    ctx.cache.invalidate(target)
    ctx.mark_touched(rel)
    ctx.trace.event("delete_file", path=rel)
    return f"deleted {path}"


def edit_file(
    ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    target = ctx.workspace.resolve(path)
    if not target.is_file():
        return f"error: no such file: {path}. Use write_file to create it."
    try:
        original = _read_text(ctx, target)
    except ValueError as exc:
        return f"error: {exc}"

    if old_string == new_string:
        return "error: old_string and new_string are identical; nothing to do."

    count = original.count(old_string)
    if count == 0:
        return _no_match_help(path, original, old_string)
    if count > 1 and not replace_all:
        return (
            f"error: old_string appears {count} times in {path}. Include more surrounding "
            "context to make it unique, or pass replace_all=true."
        )

    updated = (
        original.replace(old_string, new_string)
        if replace_all
        else original.replace(old_string, new_string, 1)
    )
    rel = ctx.workspace.rel(target)
    ctx.before_write(rel, target)
    target.write_text(updated, encoding="utf-8")
    ctx.mark_touched(rel)
    ctx.trace.event(
        "edit_file",
        path=rel,
        replacements=count if replace_all else 1,
        **_change_summary(old_string, new_string),
    )
    diff = "\n".join(
        list(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
                n=2,
            )
        )[:60]
    )
    header = f"edited {path} ({count if replace_all else 1} replacement(s))"
    return f"{header}\n{diff}" + _post_write(ctx, target, rel, updated)


def _change_summary(old_string: str, new_string: str) -> dict:
    """Lines added and removed by one replacement, and the first added line,
    for the trace event. The tool_call event cannot say this: its arguments
    are clipped to 200 characters, and the TUI shows an edit's size and gist
    from this event instead."""
    added = removed = 0
    preview = ""
    diff = difflib.unified_diff(old_string.splitlines(), new_string.splitlines(), n=0, lineterm="")
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            added += 1
            if not preview and line[1:].strip():
                preview = line[1:].strip()[:120]
        elif line.startswith("-"):
            removed += 1
    return {"added": added, "removed": removed, "preview": preview}


def _no_match_help(path: str, original: str, old_string: str) -> str:
    """Give the model the near-miss instead of a bare failure."""
    needle = old_string.strip().splitlines()
    anchor = needle[0].strip() if needle else ""
    hints: list[str] = []
    if anchor:
        for i, line in enumerate(original.splitlines(), 1):
            if anchor in line:
                hints.append(f"  {i}\t{line}")
            if len(hints) >= 5:
                break
    if not hints and anchor:
        close = difflib.get_close_matches(anchor, original.splitlines(), n=3, cutoff=0.6)
        hints = [f"  ?\t{c}" for c in close]
    tail = ("\nClosest lines in the file:\n" + "\n".join(hints)) if hints else ""
    return (
        f"error: old_string not found in {path}. It must match the file byte for byte, "
        f"including indentation.{tail}\nRe-read the file before retrying."
    )


def edit_files(ctx: ToolContext, edits: list) -> str:
    """Apply several search/replace edits in one call, across one or many files.

    Measured on real runs, a step spends several whole round trips making edits
    one at a time even when it already knows every change it wants to make —
    each edit_file call costs about a second of overhead regardless of size.
    This applies them in sequence within a single call and reports each result,
    so a mid-batch failure does not lose the edits that already succeeded.

    `edits` is a list of {path, old_string, new_string, replace_all?} objects.
    """
    if not edits:
        return "error: edits must be a non-empty list"
    if not isinstance(edits, list):
        return "error: edits must be a JSON array of edit objects"

    results = []
    ok = 0
    for i, spec in enumerate(edits):
        if not isinstance(spec, dict):
            results.append(f"[{i}] error: not an object")
            continue
        path = spec.get("path")
        old = spec.get("old_string")
        new = spec.get("new_string")
        if not path or old is None or new is None:
            results.append(f"[{i}] error: needs path, old_string and new_string")
            continue
        outcome = edit_file(
            ctx, str(path), str(old), str(new), bool(spec.get("replace_all", False))
        )
        marker = "ok" if not outcome.startswith("error:") else "FAILED"
        if marker == "ok":
            ok += 1
        results.append(f"[{i}] {path}: {marker}\n{outcome}")

    header = f"{ok}/{len(edits)} edits applied\n\n"
    return header + "\n\n".join(results)


def batch_edits_to_file(ctx: ToolContext, path: str, edits: list) -> str:
    """Apply several search/replace edits to a single file in one call.

    Simpler than edit_files when editing one file: supply the path once and a list
    of {old_string, new_string, replace_all?} edits. Each is applied in order;
    a failure in one does not stop the others.

    Use this when you know all the changes you want to make to one file upfront.
    It costs the same as one edit_file call regardless of how many edits are in it.
    """
    if not edits:
        return "error: edits must be a non-empty list"
    if not isinstance(edits, list):
        return "error: edits must be a JSON array of edit objects"

    results = []
    ok = 0
    for i, spec in enumerate(edits):
        if not isinstance(spec, dict):
            results.append(f"[{i}] error: not an object")
            continue
        old = spec.get("old_string")
        new = spec.get("new_string")
        if old is None or new is None:
            results.append(f"[{i}] error: needs old_string and new_string")
            continue
        outcome = edit_file(ctx, path, str(old), str(new), bool(spec.get("replace_all", False)))
        marker = "ok" if not outcome.startswith("error:") else "FAILED"
        if marker == "ok":
            ok += 1
        results.append(f"[{i}] {marker}\n{outcome}")

    header = f"{ok}/{len(edits)} edits to {path} applied\n\n"
    return header + "\n\n".join(results)


def _grep_one_file(ctx: ToolContext, f: Path, rx: re.Pattern, remaining: int) -> list[str]:
    """Matching lines from one file, formatted and capped at `remaining`."""
    try:
        text = _read_text(ctx, f)
    except (ValueError, OSError):
        return []
    matches: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if rx.search(line):
            matches.append(f"{ctx.workspace.rel(f)}:{i}: {line.strip()[:200]}")
            if len(matches) >= remaining:
                break
    return matches


def _python_grep(
    ctx: ToolContext, rx: re.Pattern, root: Path, glob: str, max_results: int
) -> list[str]:
    """Line-by-line fallback: reads every candidate file in the interpreter.
    Correct for anything `re.compile` accepts, including patterns (backreferences,
    lookaround) that rg's regex engine rejects outright — see `_rg_grep`, which
    tries the fast path first and falls back here whenever it cannot be sure the
    fast path's answer means the same thing. Uses the same pruned traversal as
    everything else that walks the whole tree (see `walk_files`) — even the
    fallback path should not pay to open every file under `.venv` first and
    throw the result away, on the rare run where rg is unavailable."""
    hits: list[str] = []
    files = [root] if root.is_file() else sorted(walk_files(root))
    for f in files:
        if len(hits) >= max_results:
            hits.append("... more matches suppressed")
            break
        if not f.is_file() or not fnmatch.fnmatch(f.name, glob):
            continue
        hits.extend(_grep_one_file(ctx, f, rx, max_results - len(hits)))
    return hits


def _rg_grep(
    ctx: ToolContext, pattern: str, root: Path, glob: str, max_results: int, ignore_case: bool
) -> list[str] | None:
    """The same search via ripgrep, when it can answer it. None tells the
    caller to fall back to `_python_grep` instead of trusting this result:

    - rg not installed (exit 127): the harness's own line-by-line scan is the
      only implementation, same as before this existed.
    - rg's Rust regex engine rejected the pattern (exit 2): it does not
      support backreferences or lookaround, which Python's `re` — the
      contract this tool's `pattern` parameter documents — does. A pattern
      using either would silently mean something different (or nothing) to
      rg, so this only ever trusts rg's own success exit codes (0 match, 1
      no match), never papering over a rejection with a "no matches" answer
      that might actually be wrong.

    On a workspace the size of a large repo this is the difference between
    walking every file byte-by-byte in the interpreter and one subprocess
    that skips binaries and respects the same ignore list by construction —
    but only when it applies cleanly; see `_python_grep` for what still
    guarantees correctness underneath it.
    """
    target = "." if root == ctx.workspace.root else ctx.workspace.rel(root)
    args = [
        "--line-number",
        "--no-heading",
        "--color=never",
        "--max-columns=200",
        f"--max-count={max_results}",
    ]
    if ignore_case:
        args.append("--ignore-case")
    if glob and glob != "*":
        args += ["--glob", glob]
    args += exclude_globs("--glob")
    args += ["--regexp", pattern, target]
    result = toolchain.run_capturing(
        ["rg", *args], cwd=ctx.workspace.root, timeout=RG_TIMEOUT_S, env=toolchain.clean_env()
    )
    if result.returncode not in (0, 1):
        return None
    hits: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        file_part, line_no, content = parts
        hits.append(f"{file_part.removeprefix('./')}:{line_no}: {content.strip()[:200]}")
    if len(hits) > max_results:
        hits = [*hits[:max_results], "... more matches suppressed"]
    return hits


def grep(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str = "*",
    max_results: int = 60,
    ignore_case: bool = False,
) -> str:
    root = ctx.workspace.resolve(path)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"error: bad regex: {exc}"
    hits = _rg_grep(ctx, pattern, root, glob, max_results, ignore_case)
    if hits is None:
        hits = _python_grep(ctx, rx, root, glob, max_results)
    return "\n".join(hits) if hits else f"no matches for {pattern!r}"


def _rg_glob(ctx: ToolContext, pattern: str) -> list[str] | None:
    """`glob_files` via ripgrep's own file listing, when it can answer it.

    Same trust rule as `_rg_grep`: only rg's own success codes (0 match, 1
    no match) are read as an answer. A malformed glob (exit 2) or a missing
    binary (127) falls back to the exact pre-existing `rglob` implementation
    rather than risking a wrong "no files match". Checked empirically against
    this repository before relying on it: `*.py`, `**/*.py`, `test_*.py`, and
    `*.toml` all returned identical result sets to the rglob-based version,
    IGNORED_DIRS exclusions included.
    """
    args = ["--files", "-g", pattern]
    args += exclude_globs("--glob")
    result = toolchain.run_capturing(
        ["rg", *args], cwd=ctx.workspace.root, timeout=RG_TIMEOUT_S, env=toolchain.clean_env()
    )
    if result.returncode not in (0, 1):
        return None
    return sorted(line.removeprefix("./") for line in result.stdout.splitlines() if line)


def _rglob_regex(pattern: str) -> re.Pattern[str]:
    """`root.rglob(pattern)`'s matching as a regex over a POSIX relative path:
    the pattern may start at any depth, `**` spans directories, `*` and `?`
    stay inside one. (PurePath.full_match and glob.translate do this, but
    only from Python 3.13; this project supports 3.11.)"""
    parts = ["(?:.*/)?"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts))


def glob_files(ctx: ToolContext, pattern: str) -> str:
    found = _rg_glob(ctx, pattern)
    if found is None:
        root = ctx.workspace.root
        # Pruned before descending (see walk_files), unlike rglob-then-filter,
        # which read all of .venv and node_modules first.
        matcher = _rglob_regex(pattern)
        found = sorted(
            ctx.workspace.rel(p)
            for p in walk_files(root)
            if matcher.fullmatch(p.relative_to(root).as_posix())
        )
    return "\n".join(found[:200]) if found else f"no files match {pattern!r}"


FS_TOOLS = [
    Tool(
        "list_dir",
        "Browse a directory you already know the path of. To find files by name "
        "anywhere in the workspace use glob_files, and to find them by content use "
        "grep — do not shell out to `find` or `ls`.",
        obj(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory. Default '.', the root.",
                },
                "depth": {"type": "integer", "description": "Recursion depth, default 2."},
            }
        ),
        list_dir,
    ),
    Tool(
        "read_file",
        "Read a file with 1-indexed line numbers. Read before you edit; edits require exact text.",
        obj(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path, e.g. 'src/game.py'.",
                },
                "offset": {"type": "integer", "description": "First line to read, 1-indexed."},
                "limit": {
                    "type": "integer",
                    "description": "Max lines, default 1200 — enough for most whole "
                    "files. Reading a file in pieces costs a round trip "
                    "per piece.",
                },
                "sections": {
                    "type": "string",
                    "description": "Several parts of the file in this one call, comma-"
                    "separated: line ranges and function or class names, e.g. "
                    "'40-80, Game.update, Game.draw'. Each comes back with its own line numbers.",
                },
            },
            ["path"],
        ),
        read_file,
    ),
    Tool(
        "write_file",
        "Create a file, or replace its entire contents. Use edit_file to change an existing file. "
        "Python files are syntax-checked and linted automatically; the result is in the output.",
        obj(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path. Parent directories are created.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file.",
                },
            },
            ["path", "content"],
        ),
        write_file,
    ),
    Tool(
        "edit_file",
        "Replace an exact string in a file. old_string must match byte for byte and be unique "
        "unless replace_all is true. Returns a diff, plus any syntax or lint errors the edit "
        "introduced.",
        obj(
            {
                "path": {"type": "string", "description": "Workspace-relative path."},
                "old_string": {
                    "type": "string",
                    "description": "Exact existing text, with indentation. Include "
                    "enough surrounding lines to make it unique.",
                },
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of failing when "
                    "old_string is not unique. Send the literal true or "
                    'false, not the strings "true"/"false".',
                },
            },
            ["path", "old_string", "new_string"],
        ),
        edit_file,
    ),
    Tool(
        "read_files",
        "Read several files in one call. Always prefer this over calling read_file more "
        "than once: 89% of a call's time is spent before a single token is generated, so "
        "one call returning ten files costs about a second and ten calls cost about ten.",
        obj(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace-relative paths, at most 20.",
                },
                "limit": {"type": "integer", "description": "Max lines per file, default 600."},
            },
            ["paths"],
        ),
        read_files,
    ),
    Tool(
        "edit_files",
        "Apply several search/replace edits in one call, to one or many files. Always prefer "
        "this over calling edit_file more than once: a call costs about a second before it "
        "returns anything, whether it carries one edit or twenty, so deciding every change "
        "first and sending them together is the cheapest thing you can do. Edits are applied "
        "in order; a failure in one does not stop or undo the others.",
        obj(
            {
                "edits": {
                    "type": "array",
                    "description": "Each item: {path, old_string, new_string, replace_all?}.",
                    "items": obj(
                        {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        ["path", "old_string", "new_string"],
                    ),
                }
            },
            ["edits"],
        ),
        edit_files,
    ),
    Tool(
        "edit_blocks",
        "Several edits in one call, to one or many files, as plain text. For each file, a line "
        "with its path, then one block per change:\n"
        "main.py\n<<<<<<< SEARCH\n<exact current text>\n=======\n<replacement>\n>>>>>>> REPLACE\n"
        "Blocks apply in order; the SEARCH text must match exactly once, like edit_file's "
        "old_string. An empty SEARCH creates a new file. Decide every change, then send them "
        "together: a call costs about a second whether it carries one edit or twenty.",
        obj(
            {
                "blocks": {
                    "type": "string",
                    "description": "Path lines and SEARCH/REPLACE blocks, as described above.",
                }
            },
            ["blocks"],
        ),
        edit_blocks,
    ),
    Tool(
        "batch_edits_to_file",
        "Apply several search/replace edits to a single file in one call. Simpler than "
        "edit_files when targeting one file: supply the path once and a list of "
        "{old_string, new_string, replace_all?} edits. Costs the same as one edit_file call "
        "regardless of edit count. Use this when you know all the changes to one file upfront.",
        obj(
            {
                "path": {"type": "string", "description": "Workspace-relative path to edit."},
                "edits": {
                    "type": "array",
                    "description": "Each item: {old_string, new_string, replace_all?}.",
                    "items": obj(
                        {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        ["old_string", "new_string"],
                    ),
                },
            },
            ["path", "edits"],
        ),
        batch_edits_to_file,
    ),
    Tool(
        "grep",
        "Find files by what is inside them — a symbol, a call site, an error string. "
        "Use glob_files instead when you know the filename and not the contents.",
        obj(
            {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression, matched per line.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Default '.', the workspace.",
                },
                "glob": {"type": "string", "description": "Filename filter, e.g. '*.py'."},
                "max_results": {
                    "type": "integer",
                    "description": "Matches to return, default 60.",
                },
                "ignore_case": {"type": "boolean", "description": "Case-insensitive match."},
            },
            ["pattern"],
        ),
        grep,
    ),
    Tool(
        "glob_files",
        "Find files by name or extension anywhere in the workspace. This is the one "
        "tool for 'which files exist' — one call with '**/*.py' answers what repeated "
        "list_dir calls or `run_bash find` would, and returns every match at once.",
        obj(
            {
                "pattern": {
                    "type": "string",
                    "description": "Glob against the path from the workspace root. "
                    "Always matches at any depth, whether or not the "
                    "pattern itself starts with '**/': '*.py' and "
                    "'**/*.py' are equivalent and both find every Python "
                    "file, '**/test_*.py' finds every test.",
                }
            },
            ["pattern"],
        ),
        glob_files,
    ),
    Tool(
        "move_file",
        "Move or rename a file within the workspace. Refuses to overwrite an existing "
        "destination — delete_file it first if that's really what's wanted.",
        obj(
            {
                "src": {"type": "string", "description": "Existing workspace-relative path."},
                "dst": {"type": "string", "description": "New workspace-relative path."},
            },
            ["src", "dst"],
        ),
        move_file,
    ),
    Tool(
        "delete_file",
        "Delete a file within the workspace. There is no confirmation prompt — the run's "
        "baseline still records what it contained, so show_changes can show the deletion.",
        obj(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to delete.",
                }
            },
            ["path"],
        ),
        delete_file,
    ),
]
