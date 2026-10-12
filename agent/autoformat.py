"""`ruff format` on the Python files a run touched, when a step or fix finishes.

In a project that opts into a formatter (a Ruff config or [tool.black], see
`projectconfig.uses_a_formatter`), layout is never worth a model call: the harness formats the
files the run touched once a step or a fix loop ends (DESIGN-LOG §104). Not after every turn,
which would change text under the model's next edit anchor mid-step. Layout only: `ruff check
--fix` also deletes unused imports, which would remove an import one step adds for the next.
A project with no formatter config keeps its own style (§70: 18 lines of churn around a 4-line
django fix).
"""

from __future__ import annotations

from .projectconfig import uses_a_formatter
from .tools.base import (
    ToolContext,
    mark_external_writes,
    snapshot_before_write,
    workspace_manifest,
)
from .trace import Trace


def format_touched(ctx: ToolContext, trace: Trace, phase: str) -> list[str]:
    """Format the touched Python files; return the ones `ruff format` changed."""
    from . import toolchain

    root = ctx.workspace.root
    if ctx.writable is not None or not uses_a_formatter(root):
        return []  # a parallel step may write only its declared files
    files = sorted({p for p in ctx.touched if p.endswith(".py") and (root / p).is_file()})
    if not files:
        return []
    before = snapshot_before_write(ctx)
    code, _out, err = toolchain.run("ruff", ["format", "--quiet", *files], root)
    mark_external_writes(ctx, before)
    ctx.cache.invalidate()
    after = workspace_manifest(root)
    changed = [p for p in files if before.get(p) != after.get(p)]
    trace.event(
        "autoformat", phase=phase, files=files, changed=changed, ok=code == 0, error=err[-200:]
    )
    return changed
