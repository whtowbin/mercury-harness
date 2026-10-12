"""Structural code search and rewrite via ast-grep: match by syntax, not text.

Not offered to the model: unregistered after DESIGN-LOG §62.1 (0 calls in ~3,100
offers and 22 nudges). The module and its tests stay so it cannot rot; the
wiring that offered it was removed (§62.1).

A regex `grep` for `def old_name(` misses `def  old_name (` and matches a
comment that happens to contain the same characters; it also cannot express
"a call to this function, however its arguments are laid out or wrapped
across lines" at all. ast-grep parses the file and matches against its
syntax tree, so a pattern like `old_name($$$ARGS)` finds every call
regardless of whitespace, and — the reason this exists — its `--rewrite`
mode can apply the same structural change everywhere it matches in a single
call. Bet #1 in DESIGN-LOG.md is "prefer one big step to three small ones";
a mechanical rename or signature change across N files used to cost N
edit_file round trips and now costs one ast_grep_rewrite call.

Search is read-only and carries no more risk than grep. Rewrite writes files
exactly as capable of causing damage as run_bash or run_python, so it is
approval-gated (`ctx.approver.check_run`, the same gate run_python uses —
this is not shell syntax, so the allow/deny list does not apply) and refuses
outright during a concurrent batch (see `ast_grep_rewrite`'s docstring) the
same way run_bash and run_python already do, rather than trying to prove it
only touched its own declared files after the fact.
"""

from __future__ import annotations

import json

from .. import toolchain
from ..approval import Denied
from ..workspace import exclude_globs
from .base import (
    Tool,
    ToolContext,
    mark_external_writes,
    obj,
    snapshot_before_write,
    workspace_manifest,
)

AST_GREP_TIMEOUT_S = 30
MAX_PREVIEW_CHARS = 6_000


def _missing(result) -> str | None:
    if result.returncode == 127:
        return (
            "error: ast-grep is not installed in this environment. "
            "Fall back to grep for a text-based search."
        )
    return None


def _exclude_globs() -> list[str]:
    """Exclude the same directories every other file-walking tool in this
    codebase ignores. ast-grep respects .gitignore on its own, but these
    names are a harness-imposed denylist, not necessarily something every
    workspace's own .gitignore happens to list. `--globs` (plural) is
    ast-grep's own flag name for the same gitignore-style exclude syntax
    `workspace.exclude_globs` builds for every caller — rg's equivalent
    flag is spelled `--glob`, singular, which is the whole reason that
    helper takes the flag name as a parameter instead of hardcoding one."""
    return exclude_globs("--globs")


def _target(ctx: ToolContext, path: str) -> str:
    root = ctx.workspace.resolve(path)
    return "." if root == ctx.workspace.root else ctx.workspace.rel(root)


def ast_grep_search(
    ctx: ToolContext, pattern: str, path: str = ".", lang: str = "", max_results: int = 30
) -> str:
    """Find code by syntax rather than text — see the module docstring."""
    target = _target(ctx, path)
    args = ["run", "-p", pattern, "--json=compact", *_exclude_globs()]
    if lang:
        args += ["--lang", lang]
    args.append(target)
    result = toolchain.run_capturing(
        ["ast-grep", *args],
        cwd=ctx.workspace.root,
        timeout=AST_GREP_TIMEOUT_S,
        env=toolchain.clean_env(),
    )
    missing = _missing(result)
    if missing:
        return missing
    try:
        matches = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return f"error: ast-grep produced unreadable output ({result.output[:300]})"
    if not matches:
        # ast-grep's own exit code does not distinguish "parsed fine, matched
        # nothing" from "the pattern itself did not parse" — both print `[]`
        # and exit 1. Say so rather than implying the search definitely ran.
        return (
            f"no matches for {pattern!r} (or the pattern did not parse — ast-grep "
            "gives the same empty answer either way; try it against a line you "
            "know is present to check the pattern itself first)"
        )
    lines = []
    for m in matches[:max_results]:
        file = m.get("file", "?")
        start = m.get("range", {}).get("start", {}).get("line", 0) + 1
        end = m.get("range", {}).get("end", {}).get("line", start - 1) + 1
        snippet = (m.get("text") or "").splitlines()[0].strip()[:200] if m.get("text") else ""
        span = f"{start}" if end == start else f"{start}-{end}"
        lines.append(f"{file}:{span}: {snippet}")
    if len(matches) > max_results:
        lines.append(f"... {len(matches) - max_results} more match(es) suppressed")
    return "\n".join(lines)


def ast_grep_rewrite(
    ctx: ToolContext,
    pattern: str,
    rewrite: str,
    path: str = ".",
    lang: str = "",
    apply: bool = False,
) -> str:
    """Rewrite every structural match of `pattern` to `rewrite` in one call.

    With apply=false (the default) this changes nothing and returns the diff
    it *would* make, across every matching file — decide from that before
    spending the approval prompt and the actual write. With apply=true it
    writes for real and is gated exactly like run_python: not shell syntax,
    so the allow/deny list does not apply, but it is exactly as capable of
    damage, so auto_approve is the only thing that skips the prompt.

    Refuses outright when steps are running concurrently (`ctx.writable is
    not None`), same as run_bash and run_python: a rewrite can touch any
    number of files matching the pattern, decided by ast-grep's own search
    rather than the plan's file list, so there is no set of "this step's
    declared files" to check it against after the fact the way run_verify's
    write jail does. Concurrency safety here means "run it as its own
    sequential step," not "let it race and catch violations." A preview
    (apply=false) is unaffected — it only reads, so it is as safe as grep
    even mid-batch.
    """
    target = _target(ctx, path)
    args = ["run", "-p", pattern, "-r", rewrite, *_exclude_globs()]
    if lang:
        args += ["--lang", lang]

    if not apply:
        args.append(target)
        result = toolchain.run_capturing(
            ["ast-grep", *args],
            cwd=ctx.workspace.root,
            timeout=AST_GREP_TIMEOUT_S,
            env=toolchain.clean_env(),
        )
        missing = _missing(result)
        if missing:
            return missing
        preview = result.output.strip()
        if not preview:
            return (
                f"no matches for {pattern!r} (or the pattern did not parse — see "
                "ast_grep_search to check it against something you know is present)"
            )
        return (
            "preview only, nothing written. Call again with apply=true to write this:\n"
            + preview[:MAX_PREVIEW_CHARS]
        )

    if ctx.writable is not None:
        return (
            "error: ast_grep_rewrite is disabled while steps run in parallel — it can "
            "touch any file matching the pattern, not just this step's declared files, "
            "which the concurrent write jail cannot check after the fact. Preview with "
            "apply=false (always safe), or run this step sequentially to apply it."
        )

    key = f"ast_grep_rewrite:{pattern}→{rewrite}:{target}:{lang}"
    try:
        ctx.approver.check_run(
            key,
            f"apply this structural rewrite in {target or '.'}:\n"
            f"  pattern: {pattern}\n  rewrite: {rewrite}",
        )
    except Denied as exc:
        ctx.trace.event("ast_grep_rewrite_denied", pattern=pattern, reason=str(exc))
        return f"error: {exc}"

    args += ["-U", target]
    before = snapshot_before_write(ctx)
    result = toolchain.run_capturing(
        ["ast-grep", *args],
        cwd=ctx.workspace.root,
        timeout=AST_GREP_TIMEOUT_S,
        env=toolchain.clean_env(),
    )
    missing = _missing(result)
    if missing:
        return missing
    ctx.cache.invalidate()
    mark_external_writes(ctx, before)
    after = workspace_manifest(ctx.workspace.root)
    changed = sorted(p for p in after if before.get(p) != after.get(p))
    ctx.trace.event(
        "ast_grep_rewrite",
        pattern=pattern,
        rewrite=rewrite,
        path=target,
        files_changed=len(changed),
    )
    if not changed:
        return (
            f"no matches for {pattern!r}; nothing written (or the pattern did not "
            "parse — see ast_grep_search to check it first)"
        )
    return f"applied to {len(changed)} file(s): {', '.join(changed[:20])}" + (
        f" ... and {len(changed) - 20} more" if len(changed) > 20 else ""
    )


STRUCTURAL_TOOLS = [
    Tool(
        "ast_grep_search",
        "Find code by syntax, not text: matches a call or definition however its "
        "whitespace, line-wrapping or argument order looks, using $VAR to capture one "
        "node and $$$VAR to capture a variable-length list (e.g. 'old_name($$$ARGS)' "
        "finds every call to old_name regardless of its arguments). Prefer grep for a "
        "plain text/string search; prefer this when the same logical construct can be "
        "written several different ways in source.",
        obj(
            {
                "pattern": {
                    "type": "string",
                    "description": "ast-grep pattern, e.g. 'def old_name($$$ARGS)'.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Default '.', the workspace.",
                },
                "lang": {
                    "type": "string",
                    "description": "Language override, e.g. 'python'. Usually inferred "
                    "from each file's extension; only needed for files "
                    "without one, or to force one for a mixed directory.",
                },
                "max_results": {"type": "integer", "description": "Matches to return, default 30."},
            },
            ["pattern"],
        ),
        ast_grep_search,
    ),
    Tool(
        "ast_grep_rewrite",
        "Rewrite every structural match of a pattern in one call — a mechanical rename or "
        "signature change across many files without one edit_file per call site. Defaults "
        "to a dry run: call with apply=false first (or omit it) to see the diff it would "
        "make everywhere, then call again with apply=true to actually write it. Disabled "
        "while steps run in parallel — run it as its own step.",
        obj(
            {
                "pattern": {
                    "type": "string",
                    "description": "ast-grep pattern to match, e.g. 'old_name($$$ARGS)'.",
                },
                "rewrite": {
                    "type": "string",
                    "description": "Replacement, reusing pattern's $VAR/$$$VAR names, "
                    "e.g. 'new_name($$$ARGS)'.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to rewrite in. Default '.'.",
                },
                "lang": {"type": "string", "description": "Language override; usually inferred."},
                "apply": {
                    "type": "boolean",
                    "description": "false (default): preview only, nothing written. "
                    "true: write the change for real.",
                },
            },
            ["pattern", "rewrite"],
        ),
        ast_grep_rewrite,
    ),
]
