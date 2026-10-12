"""Change visibility: the harness's own run diff, plus git when the workspace is a repo.

A scratch workspace usually is not a git repository, so `show_changes` is the primary
tool: it diffs against the snapshot the harness took before the agent's first write.
The git tools are there for the case where the agent is pointed at a real repository.
"""

from __future__ import annotations

from .. import toolchain
from ..text import truncate
from .base import Tool, ToolContext, obj

GIT_TIMEOUT = 30
MAX_GIT_OUTPUT = 16_000


def _git(ctx: ToolContext, args: list[str]) -> tuple[int, str]:
    result = toolchain.run_capturing(
        ["git", *args], cwd=ctx.workspace.root, timeout=GIT_TIMEOUT, env=toolchain.clean_env()
    )
    if result.returncode == 127 and result.stderr and not result.stdout:
        return 127, f"git is unavailable here ({result.stderr})"
    return result.returncode, truncate(result.output, MAX_GIT_OUTPUT)


def _is_repo(ctx: ToolContext) -> bool:
    code, out = _git(ctx, ["rev-parse", "--is-inside-work-tree"])
    return code == 0 and out.strip() == "true"


def show_changes(
    ctx: ToolContext, paths: list[str] | None = None, stat: bool = False, context: int = 3
) -> str:
    """Everything this run has changed, as a unified diff against the starting state."""
    root = ctx.workspace.root
    if stat:
        return f"changes this run:\n{ctx.baseline.stat(root)}"
    selected = paths if paths else None
    if selected:
        selected = [p for p in selected if p in ctx.baseline.snapshots]
        if not selected:
            return "none of those files have been modified this run. Modified so far: " + (
                ", ".join(ctx.baseline.snapshots) or "(nothing)"
            )
    diff = ctx.baseline.run_diff(root, selected, context=max(0, min(context, 12)))
    ctx.trace.event("show_changes", files=len(ctx.baseline.snapshots), stat=stat)
    return diff


def git_diff(ctx: ToolContext, ref: str = "", path: str = "", stat: bool = False) -> str:
    """Diff the working tree against HEAD, or against a ref you name."""
    if not _is_repo(ctx):
        return (
            "this workspace is not a git repository, so there is no git history. "
            "Use show_changes to see what this run modified."
        )
    args = ["--no-pager", "diff"]
    if stat:
        args.append("--stat")
    if ref:
        args.append(ref)
    if path:
        args.extend(["--", path])
    code, out = _git(ctx, args)
    if code not in (0, 1):
        return f"error: {out.strip()}"
    return out.strip() or "(no differences)"


def git_log(ctx: ToolContext, count: int = 10, path: str = "") -> str:
    """Recent commits, for orienting in an existing repository."""
    if not _is_repo(ctx):
        return "this workspace is not a git repository."
    args = [
        "--no-pager",
        "log",
        f"-{max(1, min(count, 50))}",
        "--pretty=format:%h %ad %an: %s",
        "--date=short",
    ]
    if path:
        args.extend(["--", path])
    code, out = _git(ctx, args)
    if code != 0:
        return f"error: {out.strip()}"
    return out.strip() or "(no commits)"


def git_add(ctx: ToolContext, paths: list[str] | None = None) -> str:
    """Stage files for commit. Omit paths to stage everything changed."""
    if not _is_repo(ctx):
        return "this workspace is not a git repository. Run `git init` first if you want one."
    args = ["add", "-A"] if not paths else ["add", "--", *paths]
    code, out = _git(ctx, args)
    if code != 0:
        return f"error: {out.strip()}"
    ctx.trace.event("git_add", paths=paths or ["-A"])
    return out.strip() or f"staged {', '.join(paths) if paths else 'all changes'}"


def git_commit(ctx: ToolContext, message: str, paths: list[str] | None = None) -> str:
    """Commit staged changes. `paths`, if given, is staged first as a
    convenience — omit it to commit whatever is already staged (via
    git_add), the same distinction plain `git add` then `git commit` makes.

    The message is passed as a real argv element, never through a shell, so
    quoting and multi-line messages just work — the exact class of mistake
    a hand-written `run_bash("git commit -m \\"...\\")` call invites.
    """
    if not _is_repo(ctx):
        return "this workspace is not a git repository. Run `git init` first if you want one."
    if not message.strip():
        return "error: message must not be empty"
    if paths:
        add_code, add_out = _git(ctx, ["add", "--", *paths])
        if add_code != 0:
            return f"error staging {', '.join(paths)}: {add_out.strip()}"
    code, out = _git(ctx, ["commit", "-m", message])
    if code != 0:
        # git phrases "there is nothing staged" two ways depending on
        # whether untracked files exist too — "nothing to commit, working
        # tree clean" with none, "nothing added to commit but untracked
        # files present" with some — both mean the same thing here.
        lowered = out.lower()
        if "nothing to commit" in lowered or "nothing added to commit" in lowered:
            return "nothing to commit — stage something with git_add first, or pass paths="
        return f"error: {out.strip()}"
    ctx.trace.event("git_commit", message=message[:200], paths=paths)
    return out.strip()


VCS_TOOLS = [
    Tool(
        "show_changes",
        "Show everything this run has changed, as a unified diff against the state before the "
        "agent started. Works whether or not the workspace is a git repository. Use stat=true "
        "for a one-line-per-file summary.",
        obj(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to these files. Omit for every change.",
                },
                "stat": {"type": "boolean", "description": "Summary instead of the full diff."},
                "context": {"type": "integer", "description": "Diff context lines, default 3."},
            }
        ),
        show_changes,
    ),
    Tool(
        "git_diff",
        "Diff the working tree against HEAD or a named ref, when the workspace is a git repo.",
        obj(
            {
                "ref": {"type": "string", "description": "Ref to compare against, e.g. 'main'."},
                "path": {"type": "string", "description": "Limit the diff to this path."},
                "stat": {
                    "type": "boolean",
                    "description": "One line per file instead of the full diff.",
                },
            }
        ),
        git_diff,
    ),
    Tool(
        "git_log",
        "Recent commit history, when the workspace is a git repo.",
        obj(
            {
                "count": {"type": "integer", "description": "Commits to show, default 10."},
                "path": {"type": "string", "description": "Only commits touching this path."},
            }
        ),
        git_log,
    ),
]

# Separate from VCS_TOOLS (read-only) so the reviewer never gets these: it
# critiques, the executor fixes and commits — enforced the same way
# review_registry keeps out run_bash/run_python, by simply never including
# this list rather than trusting a schema.
VCS_WRITE_TOOLS = [
    Tool(
        "git_add",
        "Stage files for commit, when the workspace is a git repo. Omit paths to stage "
        "everything changed.",
        obj(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage. Omit to stage every change.",
                }
            }
        ),
        git_add,
    ),
    Tool(
        "git_commit",
        "Commit staged changes, when the workspace is a git repo. Pass paths to stage them "
        "first as a convenience, or call git_add separately and omit paths here to commit "
        "exactly what's staged. Prefer this over run_bash for commits — the message is "
        "never passed through a shell, so quoting and multi-line messages just work.",
        obj(
            {
                "message": {
                    "type": "string",
                    "description": "Commit message. Multi-line is fine; it never "
                    "goes through a shell.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Stage these first. Omit to commit whatever is already staged.",
                },
            },
            ["message"],
        ),
        git_commit,
    ),
]
