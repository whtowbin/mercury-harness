"""Cross-session cache for the pre-run test baseline.

`finalcheck.collect_baseline` re-runs a project's whole test suite before
every single run, because the harness has no other way to know what was
already broken before it touched anything. That is necessary the first
time and wasted the second: on a workspace nobody has touched since the
last run collected a baseline, the answer cannot have changed, because
nothing that could change it has. `workspace_state()` (agent/tools/base.py)
is exactly the cheap digest — mtime and size, no file reads — already
trusted elsewhere in this codebase (`_already_proven_by_tests`,
`replay_identical_checks`) to tell "unchanged" from "changed" without
paying to read every file's content; this is the same idea, persisted
across runs instead of only within one, since a baseline is collected once
per *run*, not once per call, so there is nothing earlier in the same run
to reuse from.

Persisted under `<workspace>/.harness/`, the directory `memory.md` already
lives in and for the same reason: it travels with the project rather than
a central config directory, and a throwaway workspace losing it when
deleted is correct behaviour, not a bug.

Deliberately state-based only, with no time expiry: two workspaces with an
identical file-by-file (name, size, mtime) manifest are the same input to
the same deterministic test suite, whenever the earlier one was recorded.
The one thing this cannot see is a change outside the workspace itself
that could change the suite's result without changing a workspace file —
a system Python upgrade, say. That is an accepted, pre-existing blind spot:
`_already_proven_by_tests`'s same-session version of this exact mechanism
has never guarded against it either.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_FILE = "test_baseline_cache.json"


def _cache_path(root: Path) -> Path:
    from .memory import MEMORY_DIR

    return root / MEMORY_DIR / CACHE_FILE


def relevant_state(root: Path) -> str:
    """The same kind of digest `workspace_state()` (agent/tools/base.py)
    computes, minus `.harness/` itself.

    `.harness/` is not in `IGNORED_DIRS` — it holds real, useful state
    (`memory.md`, `scratch.md`) that legitimately belongs in the workspace,
    not hidden from the model's own file tools the way a `.venv` is. But
    `memory.md` gets a new section appended at the end of every run
    regardless of whether the project's own code changed at all, and this
    module's own cache file lives there too — so comparing the *raw*
    `workspace_state()` across runs would miss every single time even when
    nothing test-relevant changed, which would defeat this cache entirely.
    Excluded here, specifically for this comparison, without touching what
    `workspace_state()` means anywhere else it is already trusted.
    """
    from .tools.base import workspace_manifest

    digest = hashlib.sha256()
    manifest = workspace_manifest(root)
    for rel in sorted(manifest):
        if Path(rel).parts[0] == ".harness":
            continue
        mtime, size = manifest[rel]
        digest.update(f"{rel}\0{mtime}\0{size}\0".encode())
    return digest.hexdigest()


def load(root: Path, state: str) -> dict | None:
    """The cached baseline, if the workspace is unchanged since it was
    recorded. None on any mismatch, missing file, or a cache this cannot
    trust (corrupt JSON, an unexpected shape) — a cache that cannot be
    trusted is exactly like no cache, never a reason to fail the run."""
    path = _cache_path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("workspace_state") != state:
        return None
    failures = data.get("failures")
    if not isinstance(failures, list) or not all(isinstance(f, str) for f in failures):
        return None
    if not isinstance(data.get("collection_broken"), bool):
        return None
    if not isinstance(data.get("digest", ""), str):
        return None
    return data


def save(
    root: Path, state: str, failures: list[str], collection_broken: bool, digest: str = ""
) -> None:
    """Best-effort: a cache that fails to write costs the *next* run a
    suite pass it would have had to run anyway, never this one anything,
    so any failure here is swallowed rather than raised."""
    path = _cache_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "workspace_state": state,
                    "failures": failures,
                    "collection_broken": collection_broken,
                    "digest": digest,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
