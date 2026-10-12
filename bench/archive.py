"""Keeping a sweep's evidence where a later reader can still find it.

`bench run` writes its records and traces wherever `--out` and `--trace-dir`
point, and a session that points them at a scratch directory leaves DESIGN-LOG
figures resting on files that disappear with it. That happened here: the sweeps
behind DESIGN-LOG §45-§49 were written to a session scratchpad, so their numbers
could not be re-derived once it was cleared. The other agent's sweeps sit in
`results/` and `traces/` (both gitignored, so local to a clone) and can be.

`archive` copies a sweep's records into `results/<name>.jsonl` and each record's
trace into `traces/<name>/`, then rewrites `trace_path` in the copy so
`bench report --view diagnostics` (which reads the traces) works on the archived
file. The originals are never modified or removed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import portable_path, trace_file


@dataclass
class ArchiveResult:
    records: int
    traces_copied: int
    traces_missing: int
    results_path: Path
    traces_path: Path


class ArchiveError(Exception):
    pass


def _load(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArchiveError(f"cannot read {path}: {exc}") from exc
    return [json.loads(line) for line in lines if line.strip()]


def archive(
    sources: list[Path],
    name: str,
    results_dir: Path = Path("results"),
    traces_dir: Path = Path("traces"),
    force: bool = False,
) -> ArchiveResult:
    """Copy `sources` (result files) and the traces they name under `name`."""
    if not name or "/" in name or name.startswith("."):
        raise ArchiveError(f"the archive name must be a plain word, not {name!r}")
    if not sources:
        raise ArchiveError("no result files given")
    destination = results_dir / f"{name}.jsonl"
    trace_destination = traces_dir / name
    if not force and (destination.exists() or trace_destination.exists()):
        raise ArchiveError(f"{destination} or {trace_destination} already exists; use --force")

    records: list[dict] = []
    for source in sources:
        records.extend(_load(source))
    if not records:
        raise ArchiveError("the result files hold no records")

    trace_destination.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    seen: dict[str, str] = {}
    for record in records:
        original = str(record.get("trace_path") or "")
        if not original:
            continue
        if original in seen:
            record["trace_path"] = seen[original]
            continue
        source_trace = trace_file(original)
        if not source_trace.is_file():
            missing += 1
            continue
        target = trace_destination / source_trace.name
        shutil.copyfile(source_trace, target)
        seen[original] = record["trace_path"] = portable_path(target)
        copied += 1

    results_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(record, default=str) for record in records) + "\n", encoding="utf-8"
    )
    return ArchiveResult(len(records), copied, missing, destination, trace_destination)
