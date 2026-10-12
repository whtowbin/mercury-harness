"""Telling a broken toolchain apart from broken code.

A step's verify command failing means one of two very different things, and
the harness used to treat them as one. In a traced run, `uv run pytest -q`
failed with

    error: Querying Python at `.../.venv/bin/python3` failed with signal 6
    dyld[96045]: Library not loaded: @rpath/libpython3.13.dylib

— the workspace's virtualenv was broken, so the command never got as far as
importing the code under test. The harness reported that back to the model as
"*** THIS STEP ALREADY FAILED ONCE ***" and re-ran the step. The first attempt
had already made the correct one-line fix and written an accurate summary in
six calls; the second, told it had failed, spent fourteen iterations inventing
a pygame mocking layer, a conftest, and a rewrite of that conftest, and ended
with the loop exhausted. Across the traced runs 28% of steps were retried, and
seven of twenty verify failures were this — an environment problem charged to
the model as a defect in its own work.

So: classify first. A failure that names a defect in the code is the model's
to fix. A failure that says the interpreter, the virtualenv or the tool itself
is broken is the harness's to repair, and the model must never be told it
failed for it. `repair()` runs `uv sync`, which is exactly what the model
itself worked out to do in the trace above, four fix rounds later and at the
cost of the run.

The bias here is deliberately towards *under*-diagnosing. Calling a real
defect environmental would excuse broken code, which is the worse error, so
every signature below names a specific toolchain failure mode and nothing
resembling a test failure, an assertion, or an import of the workspace's own
modules.

One more shape of the same problem, found by tracing a real run: `pyproject.
toml` already declared numpy, pandas and scikit-learn, but the workspace's
`.venv` had never been synced against it, so the *first* test run failed with
`No module named 'numpy'`. That is not covered by `_TOOL_MODULES` above — numpy
is the workspace's own dependency, not the harness's tooling — so the model
was charged for it: it ran `uv add numpy`, burned a fix round rediscovering
`pandas` the same way, then `scikit-learn`, each one its own `loop_exhausted`
cycle re-running the whole final gate. Three rounds to do what one `uv sync`
would have. `_declared_dependency_diagnosis` closes that gap without
loosening the bias above: it only ever fires for a module name that already
appears in *this workspace's own* `pyproject.toml`, so it can never mistake
the model forgetting to write `game.py` for an environment problem — nothing
the model itself needs to create is ever pre-declared as a dependency.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import langtools, toolchain
from .langtools import LanguageProfile

REPAIR_TIMEOUT_S = 180

# Modules whose absence means the *harness's* tooling is missing, not that the
# model failed to write a file. A bare "no module named" rule would swallow
# `No module named 'game'` — which is the model forgetting to create game.py,
# a real defect — so the names are enumerated rather than pattern-matched.
_TOOL_MODULES = ("pytest", "_pytest", "coverage", "ruff", "ty", "pip", "setuptools", "wheel", "uv")

# PyPI distribution name -> the module name it is actually imported as, for
# the handful of common packages where the two differ. Only used to decide
# whether a "no module named X" failure matches something pyproject.toml
# already declares — an unlisted package still falls through to "the model's
# problem", same as before this existed.
_IMPORT_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "pillow": "pil",
    "opencv-python": "cv2",
    "python-dotenv": "dotenv",
    "protobuf": "google",
    # Widened opportunistically, per this table's own docstring, from a
    # real traced run: the standard hyphen-to-underscore guess produces
    # "linear_tree", but the package's actual top-level module has no
    # separator at all. Confirmed against its installed dist-info -- the
    # wrong guess had the harness report "missing-dependency" for a package
    # that was, in fact, correctly installed.
    "linear-tree": "lineartree",
}

_MISSING_MODULE = re.compile(r"no module named ['\"]([a-zA-Z_][\w.]*)['\"]")

_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # The dynamic linker could not start the interpreter at all. dyld messages
    # are never emitted by Python-level failures.
    (
        "interpreter",
        re.compile(
            r"library not loaded:|dyld\[\d+\]:|image not found"
            r"|symbol not found in flat namespace"
        ),
    ),
    # uv probing the pinned interpreter and finding it unusable.
    ("interpreter", re.compile(r"querying python at .{0,200}?failed")),
    # uv could not build the environment the command needed.
    (
        "uv",
        re.compile(
            r"error: failed to (resolve|install|build|create|fetch)"
            r"|no solution found when resolving"
        ),
    ),
    # `uv run` cannot synchronize a workspace that has no project metadata.
    # This is a harness/setup problem, not evidence that the edited code is
    # broken; retaining the verify command lets the final gate report it.
    ("uv", re.compile(r"no pyproject\.toml, so there is nothing to sync")),
    # The virtualenv is missing, truncated, or its shebang points nowhere.
    (
        "venv",
        re.compile(
            r"bad interpreter|failed to spawn|no such file or directory: "
            r"['\"]?\S*[/\\]\.venv"
        ),
    ),
    # The tool binary itself is absent from PATH.
    (
        "missing-tool",
        re.compile(
            r"(?:command not found|not found)\s*:?\s*"
            r"\b(?:pytest|ruff|ty|uv|uvx|coverage|python3?)\b"
            r"|\b(?:pytest|ruff|ty|uv|uvx|coverage): command not found"
        ),
    ),
    ("missing-tool", re.compile(r"no module named ['\"](?:" + "|".join(_TOOL_MODULES) + r")['\"]")),
)


@dataclass(frozen=True)
class EnvFailure:
    """A command that failed because the toolchain is broken, not the code."""

    kind: str
    evidence: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.evidence}"


def diagnose(output: str, root: Path | None = None) -> EnvFailure | None:
    """The environment failure this output shows, or None if it looks like code.

    None is the safe answer: it leaves the failure attributed to the model,
    which is the existing behaviour and the right default for anything
    unrecognised. `root`, when given, additionally lets a "no module named"
    failure be classified as environmental if that module is one *this
    workspace's own* pyproject.toml already declares — see the module
    docstring for why that is the specific, narrow signal for it.
    """
    if not output:
        return None
    lowered = output.lower()
    for kind, pattern in _SIGNATURES:
        match = pattern.search(lowered)
        if match:
            return EnvFailure(kind, _line_around(output, match.start()))
    if root is not None:
        match = _MISSING_MODULE.search(lowered)
        if match and match.group(1).split(".")[0] in _declared_dependencies(root):
            return EnvFailure("missing-dependency", _line_around(output, match.start()))
    return None


def _declared_dependencies(root: Path) -> frozenset[str]:
    """Best-guess import names for every package pyproject.toml declares.

    Guesses, not certainties: a dependency's import name is usually its
    distribution name lowercased with hyphens as underscores, so that is the
    default, with `_IMPORT_NAME_OVERRIDES` covering the well-known exceptions.
    An unusual package this misses simply doesn't get the fast path — it
    still falls through to "the model's problem", the same as before this
    function existed, so a wrong guess here can only under-diagnose, never
    excuse a real defect.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return frozenset()
    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return frozenset()
    project = data.get("project", {})
    entries: list[object] = list(project.get("dependencies", []) or [])
    for group in (project.get("optional-dependencies", {}) or {}).values():
        entries += group or []
    for group in (data.get("dependency-groups", {}) or {}).values():
        entries += group or []
    guesses = set()
    for entry in entries:
        if not isinstance(entry, str):
            continue  # PEP 735 include-group entries are dicts, not packages
        name = re.split(r"[<>=!~\[; ]", entry, maxsplit=1)[0].strip().lower()
        if name:
            guesses.add(_IMPORT_NAME_OVERRIDES.get(name, name).replace("-", "_"))
    return frozenset(guesses)


def _line_around(output: str, index: int) -> str:
    """The matched line, for the trace and the human-facing note."""
    start = output.rfind("\n", 0, index) + 1
    end = output.find("\n", index)
    line = output[start:] if end == -1 else output[start:end]
    return line.strip()[:200]


def repairable(root: Path, profile: LanguageProfile | None = None) -> bool:
    """Whether there is a declared project here to rebuild the environment from.

    `profile` is detected from `root` when not given — pass it explicitly
    only when the caller already knows it (e.g. it was just used to classify
    the failure) to avoid detecting twice.
    """
    profile = profile if profile is not None else langtools.detect_profile(root)
    return profile is not None and (root / profile.marker).is_file()


def _condense(text: str) -> str:
    """One line of a subprocess's complaint, for a human-readable note."""
    return re.sub(r"\s+", " ", (text or "").strip())[:200]


def repair(root: Path, profile: LanguageProfile | None = None) -> tuple[bool, str]:
    """Rebuild the workspace environment. Returns (repaired, note).

    `profile` is detected from `root` when not given. A workspace matching no
    known profile has nothing to repair — there is no declared project file
    naming what "rebuilt" would even mean.

    Python gets two stages, because the cheap one does not cover the case
    that motivated this module. `uv sync` is tried first: idempotent, deletes
    nothing. But the traced failure was a virtualenv whose *interpreter*
    could not execute, and uv sync needs that same interpreter to do its
    work — reproducing the failure end to end, it failed with the identical
    dyld error. So the second stage is `uv venv --clear`, which discards the
    environment and builds a new one, followed by another sync.

    Discarding `.venv` is the one deliberately destructive act here, and it
    is narrow on purpose: only inside a Python workspace with a pyproject.toml
    to rebuild from, only after a diagnosed environment failure, only after
    the non-destructive attempt failed, and only ever that path. A virtualenv
    is a build artifact — already broken by the time this runs — and the
    alternative is a run that cannot verify anything it did.

    Every other profile (today, just Node) gets one stage: its own
    `repair_argv` (`npm install`), no destructive recreate step. Nothing has
    traced a Node environment broken the way the Python interpreter was, so
    there is nothing yet to model a second stage on — see langtools.py's
    module docstring.
    """
    profile = profile if profile is not None else langtools.detect_profile(root)
    if profile is None:
        return False, "no pyproject.toml or package.json, so there is nothing to repair"
    if profile is not langtools.PYTHON:
        tool, *args = profile.repair_argv
        code, out, err = toolchain.run(tool, args, root, timeout=REPAIR_TIMEOUT_S)
        if code == 0:
            return True, f"{' '.join(profile.repair_argv)} rebuilt the environment"
        return False, f"{' '.join(profile.repair_argv)} failed: {_condense(err or out)}"

    code, out, err = toolchain.run("uv", ["sync"], root, timeout=REPAIR_TIMEOUT_S)
    if code == 0:
        return True, "uv sync rebuilt the environment"
    first = _condense(err or out)

    code, out, err = toolchain.run("uv", ["venv", "--clear"], root, timeout=REPAIR_TIMEOUT_S)
    if code != 0:
        return False, f"uv sync failed ({first}) and the environment could not be recreated"
    code, out, err = toolchain.run("uv", ["sync"], root, timeout=REPAIR_TIMEOUT_S)
    if code == 0:
        return True, "the virtualenv was unusable, so it was recreated and re-synced"
    return False, f"uv sync failed after recreating the virtualenv: {_condense(err or out)}"


def repair_and_retry(
    output: str, root: Path, rerun: Callable[[], tuple[bool, str]]
) -> tuple[bool, str, str]:
    """Give a failed command one chance to have failed for a repairable reason.

    Returns (passed, output, env_note). `env_note` is non-empty only when the
    failure is environmental *and* survived the repair — that is the one case
    the caller must not charge to the model. Every other path returns "" for
    it, including a command that failed environmentally, got repaired, and
    then failed honestly: once the toolchain works, the failure it reports is
    real and belongs to the code.
    """
    failure = diagnose(output, root)
    if failure is None:
        return False, output, ""
    ran, note = repair(root)
    if not ran:
        return False, output, f"{failure} ({note})"
    passed, retried = rerun()
    if passed:
        return True, retried, ""
    persisted = diagnose(retried, root)
    if persisted is None:
        return False, retried, ""
    return False, retried, f"{persisted} (after {note})"
