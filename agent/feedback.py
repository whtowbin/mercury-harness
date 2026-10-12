"""Durable, workspace-scoped log of defects in this HARNESS -- not the
project it is working on -- found while a goal-loop iteration (agent/goal.py)
self-audits its own just-finished trace.

Lives at <workspace>/.harness/feedback.md, next to memory.md and scratch.md,
for the same reason those do: scoped to the workspace a run actually touched,
not centralized across every project this harness has ever run against. A bug
a run hits while building one project is evidence about the harness, but the
evidence is easiest to act on sitting next to the trace that produced it.

This module only writes. Nothing currently reads it back into a prompt —
unlike memory.md, a harness bug is not context a later run needs to work
around, it is a note for whoever next works on the harness itself.
"""

from __future__ import annotations

from pathlib import Path

FEEDBACK_FILENAME = "feedback.md"


class FeedbackLog:
    def __init__(self, workspace_root: Path | str):
        self.path = Path(workspace_root) / ".harness" / FEEDBACK_FILENAME

    def record(self, goal: str, iteration: int, trace_path: str, findings: list[dict]) -> int:
        """Append one iteration's harness_bug findings. Returns how many were
        written -- 0 and nothing is touched, so a clean iteration never grows
        this file, and a caller can tell a no-op from a write that failed.
        """
        usable = [f for f in findings if str(f.get("summary", "")).strip()]
        if not usable:
            return 0
        lines = [f"## iteration {iteration}: {goal[:120]}", f"trace: {trace_path}", ""]
        for f in usable:
            lines.append(f"- {f['summary'].strip()}")
            evidence = str(f.get("evidence", "")).strip()
            if evidence:
                lines.append(f"  evidence: {evidence}")
            fix = str(f.get("proposed_fix", "")).strip()
            if fix:
                lines.append(f"  proposed fix: {fix}")
        lines.append("")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            # Same principle as MemoryStore.record_run: logging harness
            # feedback is a convenience, never a reason to fail a run.
            return 0
        return len(usable)
