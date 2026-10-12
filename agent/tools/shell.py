"""Shell access, gated by the approver and metered by the resource guard."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .. import toolchain
from ..approval import Denied
from ..guards import advise
from ..text import truncate
from .base import Tool, ToolContext, mark_external_writes, obj, snapshot_before_write
from .web import code_denied

MAX_OUTPUT = 20_000
# `uv add` is how dependencies are supposed to arrive, so it is worth noticing.
_UV_ADD = re.compile(r"\buv\s+add\s+(?P<pkgs>[^;&|]+)")
# `uv pip` targets whatever virtualenv it can find, searching upward from the cwd.
# With no venv in the workspace that is the *enclosing* project's environment, so
# these are refused until the workspace owns one.
_UV_PIP = re.compile(r"\buv\s+pip\s+(install|uninstall|sync)\b")


_UV_INIT = re.compile(r"\buv\s+init\b")
_INSTALL = re.compile(
    r"\b(?:uv\s+add|uv\s+pip\s+install|pip3?\s+install|python3?\s+-m\s+pip\s+install)\b(?P<rest>[^;&|]*)"
)
_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")
_NAME_IN_CONFIG = re.compile(r"""^\s*name\s*=\s*['"]?([A-Za-z0-9_.-]+)""", re.M)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _own_distribution(root: Path) -> str:
    """The workspace's own package name, from setup.cfg, pyproject.toml or setup.py."""
    for marker in ("setup.cfg", "pyproject.toml"):
        try:
            match = _NAME_IN_CONFIG.search((root / marker).read_text(errors="ignore"))
        except OSError:
            continue
        if match:
            return _normalized(match.group(1))
    try:
        match = re.search(r"""name\s*=\s*['"]([A-Za-z0-9_.-]+)""", (root / "setup.py").read_text())
    except OSError:
        return ""
    return _normalized(match.group(1)) if match else ""


def replaces_project_environment(root: Path, command: str) -> str | None:
    """Refuse the two commands that replaced a project's environment in a live run.

    A django fix round ran `uv add django`, `uv init`, `uv add django`: a new uv
    project was created over django's setup.py, and django from the package index
    was installed over the working copy, so the tests then ran the released package
    and its own test runner no longer imported (DESIGN-LOG §75).
    """
    if _UV_INIT.search(command) and any((root / m).is_file() for m in _PROJECT_MARKERS):
        return (
            "error: refused: this workspace is already a Python project, and `uv init` would "
            "make a new one over it (pyproject.toml, uv.lock) and change how it is installed. "
            "Its environment is already set up; run its tests the way it runs them."
        )
    own = _own_distribution(root)
    install = _INSTALL.search(command)
    if own and install:
        names = {_normalized(re.split(r"[=<>!~\[ ]", t)[0]) for t in install.group("rest").split()}
        if own in names:
            return (
                f"error: refused: {own} is the project in this workspace. Installing it from "
                "the package index would shadow the working copy, so the tests would run the "
                "released package instead of the code you are changing."
            )
    return None


def _escapes_workspace(ctx: ToolContext, command: str) -> str | None:
    if not _UV_PIP.search(command):
        return None
    if ctx.workspace.venv().is_dir():
        return None
    return (
        "error: `uv pip` installs into the first virtualenv it finds by searching "
        "upward from the working directory, and this workspace does not have one, "
        "so it would modify an environment outside the workspace.\n"
        "Use `uv add <pkg>` instead (it creates and uses the workspace's own "
        "environment), or run `uv venv` first if you really need `uv pip`."
    )


def run_bash(ctx: ToolContext, command: str, timeout: int = 120) -> str:
    if ctx.writable is not None:
        return (
            "error: arbitrary shell commands are disabled while steps run in parallel; "
            "use file tools for declared writes or run this step sequentially"
        )
    exhausted = ctx.guard.check_subprocess_budget()
    if exhausted:
        return exhausted
    denied = code_denied(ctx, "run_bash", command)
    if denied:
        return denied

    escaping = _escapes_workspace(ctx, command)
    if escaping:
        ctx.trace.event("bash_blocked", command=command, reason="uv pip escape")
        return escaping
    replacing = replaces_project_environment(ctx.workspace.root, command)
    if replacing:
        ctx.trace.event("bash_blocked", command=command, reason="replaces project environment")
        return replacing

    try:
        ctx.approver.check_bash(command)
    except Denied as exc:
        ctx.trace.event("bash_denied", command=command, reason=str(exc))
        return f"error: {exc}"

    note = advise(command)
    # Never let one command outlive the step's remaining shell budget.
    effective = int(min(timeout, 600, max(5, ctx.guard.remaining_seconds())))
    ctx.trace.event("bash", command=command, timeout=effective, advisory=note)

    before = snapshot_before_write(ctx)
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            timeout=effective,
            check=False,
            env=toolchain.workspace_env(ctx.workspace.root),
        )
    except subprocess.TimeoutExpired:
        ctx.guard.record_subprocess(time.time() - started)
        mark_external_writes(ctx, before)
        return (
            f"error: command timed out after {effective}s. "
            f"{ctx.guard.remaining_seconds():.0f}s of shell budget remains this step."
        )
    elapsed = time.time() - started
    ctx.guard.record_subprocess(elapsed)

    # A shell command may have written files behind the tools' backs.
    ctx.cache.invalidate()
    mark_external_writes(ctx, before)
    _record_dependencies(ctx, command, proc.returncode)

    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    out = truncate(out, MAX_OUTPUT, keep="both")
    prefix = f"[note: {note}]\n" if note else ""
    return f"{prefix}exit={proc.returncode} ({elapsed:.1f}s)\n{out.strip() or '(no output)'}"


def _record_dependencies(ctx: ToolContext, command: str, returncode: int) -> None:
    """Track `uv add` so later steps are told what became importable."""
    if returncode != 0:
        return
    match = _UV_ADD.search(command)
    if not match:
        return
    for token in match.group("pkgs").split():
        if token.startswith("-"):
            continue
        name = token.strip("\"'")
        with ctx.lock:
            if name and name not in ctx.dependencies:
                ctx.dependencies.append(name)
                ctx.trace.event("dependency_added", package=name)


SHELL_TOOLS = [
    Tool(
        "run_bash",
        "Run a shell command in the workspace root. Use it for git, or anything the dedicated "
        "tools do not cover. Prefer check_code and run_tests over calling ruff/ty/pytest here. "
        "Each step has a limited total shell budget. Use `uv add` for dependencies, never pip.",
        obj(
            {
                "command": {
                    "type": "string",
                    "description": "One shell command line, run from the workspace root.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before it is killed, default 90.",
                },
            },
            ["command"],
        ),
        run_bash,
    ),
]
