"""A separate, easy-to-find record of a run that failed, was stopped, or
timed out.

The full structured trace already has everything (traces/run-*.jsonl), but
finding the one bad run among hundreds of good ones means grepping. This
reads that same trace right after it closes and writes a short, human
-readable transcript alongside a copy of it in its own directory, so a bad
run is something you can just open — which matters most for exactly the
failure this exists to catch: a run that hung with no visible error, where
the trace file is the only record of how far it got and what it was doing
when it stopped responding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def save(dest_dir: str | None, goal: str, error: str, trace_path: Path) -> Path | None:
    """Write a readable summary of a failed/stopped run into dest_dir, plus a
    copy of its trace. Returns the summary's path, or None if dest_dir is
    falsy (the feature is off) or writing it failed — this must never be the
    reason a failure itself fails to surface, so any OSError here is
    swallowed rather than raised.
    """
    if not dest_dir:
        return None
    try:
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = trace_path.stem  # run-<timestamp>-<id>, matching the trace's own name
        summary_path = out / f"{stem}.md"
        summary_path.write_text(
            f"# failed run: {stem}\n\n"
            f"**goal:** {goal}\n\n"
            f"**error:** {error}\n\n"
            f"**full trace:** {trace_path}\n\n"
            "## transcript\n\n" + transcript(trace_path),
            encoding="utf-8",
        )
        if trace_path.is_file():
            shutil.copy2(trace_path, out / trace_path.name)
        return summary_path
    except OSError:
        return None


def transcript(trace_path: Path) -> str:
    """A readable reconstruction of a run's narration, tool calls and model
    text from its JSONL trace — the same events Trace.say() would have
    printed live, plus the tool calls in between, so a run that never got to
    say anything (like one stuck waiting on an approval prompt that never
    got shown) still shows exactly the tool call it was waiting on.
    """
    lines: list[str] = []
    try:
        with trace_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = record.get("kind")
                if kind == "log":
                    lines.append(str(record.get("message", "")))
                elif kind == "assistant_text":
                    lines.append(
                        f"\n**[{record.get('phase', '?')}] assistant:** {record.get('text', '')}"
                    )
                elif kind == "tool_call":
                    args = json.dumps(record.get("args", {}), default=str)[:300]
                    lines.append(f"  -> {record.get('tool', '?')}({args})")
    except OSError:
        return "(trace unreadable)"
    return "\n".join(lines) or "(no events recorded before this run stopped)"
