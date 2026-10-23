"""Locating and running the Astral toolchain from inside an arbitrary workspace.

The agent's workspace may or may not be a uv project, and ruff/ty may or may not be
on PATH, so each command is resolved once per workspace and then reused.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Resolution order per tool: a bare binary on PATH is fastest, `uv run` picks up the
# project's pinned version and config, `uvx` works anywhere uv is installed.
_FALLBACK_TIMEOUT = 120


def clean_env() -> dict[str, str]:
    """uv warns loudly when VIRTUAL_ENV disagrees with the project env. It is noise
    in tool output the model has to read, so drop it."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    # CDPATH makes the shell's `cd tests` land in another directory when that has a
    # `tests/`, and prints the path; an approved `cd tests && ...` would then run
    # outside the workspace. No command here needs it.
    env.pop("CDPATH", None)
    env.setdefault("NO_COLOR", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Every subprocess the agent runs is a one-shot: a verify command, a test
    # run, a step's shell call. None of them benefit from a .pyc cache that
    # outlives the call, and on a suite dominated by test execution rather
    # than import time (the common case, measured: collection was 1.94s of a
    # 44.9s traced suite) writing it costs disk I/O for a saving that never
    # gets claimed. It also litters __pycache__ directories through a
    # workspace that is not the agent's to leave artifacts in.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def workspace_env(root: Path) -> dict[str, str]:
    """Environment for commands the *agent* runs.

    Inherited Python state is stripped, and uv is pointed at a virtualenv inside
    the workspace. The pin only holds when that venv exists — uv falls back to
    searching upward otherwise — so it is a second line of defence behind the
    nesting check in Workspace, not a substitute for it.
    """
    env = clean_env()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    venv = root / ".venv"
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    if venv.is_dir():
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def restricted_env(root: Path) -> dict[str, str]:
    """Environment for verification and other untrusted subprocesses.

    Verification commands are model-authored input. Keep the workspace
    interpreter setup, but do not expose provider, cloud, VCS, package-index,
    or other service credentials to commands that do not need them.

    A denylist of exact names plus a handful of suffixes is necessarily
    incomplete — it was previously narrow enough to keep GITHUB_TOKEN,
    NPM_TOKEN, DATABASE_URL, and SSH_AUTH_SOCK sitting in every verify
    subprocess's environment, none of which end in _API_KEY/_API_TOKEN/
    _SECRET/_PASSWORD or appeared in the small hardcoded set. Broadened here
    to the credential-shaped suffixes those miss (_TOKEN, _KEY, _CREDENTIAL,
    _CREDENTIALS, _PRIVATE_KEY, _CLIENT_SECRET, _WEBHOOK) plus the common
    unsuffixed names above, still stopping short of a full allowlist so an
    unrelated setting a verify command legitimately reads on purpose (a
    workspace's own SAFE_SETTING=kept, say) keeps passing through.
    """
    env = workspace_env(root)
    for name in tuple(env):
        upper = name.upper()
        if upper in _CREDENTIAL_NAMES or any(upper.endswith(s) for s in _CREDENTIAL_SUFFIXES):
            env.pop(name, None)
    return env


_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_APIKEY",
    "_API_TOKEN",
    "_ACCESS_KEY",
    "_ACCESS_TOKEN",
    "_SECRET",
    "_SECRET_KEY",
    "_CLIENT_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_TOKEN",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PRIVATE_KEY",
    "_AUTH_TOKEN",
    "_WEBHOOK_URL",
    "_WEBHOOK",
)
_CREDENTIAL_NAMES = {
    "OPENROUTER_API_KEY",
    "BRAVE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "NPM_TOKEN",
    "SSH_AUTH_SOCK",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


@dataclass
class ProcResult:
    """The one subprocess-run-with-timeout outcome shape in this codebase.

    Every tool-adjacent subprocess call — toolchain.run, run_python, git,
    the bench perf workload, the Keychain lookup — used to hand-roll its own
    try/except around subprocess.run with its own timing and error handling,
    and they had already drifted (different env stripping, different return
    shapes). run_capturing() below is the one place that logic lives now.
    """

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False

    @property
    def output(self) -> str:
        """stdout and stderr combined, stderr labelled when both are present."""
        if not self.stderr:
            return self.stdout
        return f"{self.stdout}\n[stderr]\n{self.stderr}" if self.stdout else self.stderr


# The exit code of a command stopped through `cancel` (128 + SIGTERM, as a shell reports it).
CANCELLED_RETURNCODE = 143
# How often a cancellable command checks its cancel event.
CANCEL_POLL_S = 0.2


def run_capturing(
    argv: list[str],
    cwd: Path | str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
) -> ProcResult:
    """Run a command, capturing output and timing it, never raising.

    A timeout or a missing executable are outcomes every caller here already
    has to handle as data (a message back to the model, a timing result for
    the bench harness) rather than as a program-ending exception.

    With `cancel`, setting the event stops the command and every process it
    started (`uv run pytest` is two), returning CANCELLED_RETURNCODE. The
    pre-run test baseline uses this: a run that ended before the baseline
    finished used to leave the whole suite running (DESIGN-LOG §91, finding 4).
    """
    if cancel is not None or timeout is not None:
        # A timeout must stop the command's children too: `subprocess.run(timeout=...)` kills
        # only the direct child, and `uv run pytest` is two processes, so a hung suite (a mutant
        # that loops) kept running at 100% CPU for hours after its run ended (DESIGN-LOG §126.5).
        return _run_cancellable(argv, cwd, timeout, env, cancel or threading.Event())
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProcResult(124, "", "", time.time() - started, timed_out=True)
    except OSError as exc:
        return ProcResult(127, "", str(exc), time.time() - started)
    return ProcResult(proc.returncode, proc.stdout or "", proc.stderr or "", time.time() - started)


def _run_cancellable(
    argv: list[str],
    cwd: Path | str | None,
    timeout: float | None,
    env: dict[str, str] | None,
    cancel: threading.Event,
) -> ProcResult:
    """run_capturing's cancellable form: its own process group, so a cancel or a
    timeout stops the command's children too."""
    started = time.time()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return ProcResult(127, "", str(exc), time.time() - started)
    while True:
        try:
            out, err = proc.communicate(timeout=CANCEL_POLL_S)
            return ProcResult(proc.returncode, out or "", err or "", time.time() - started)
        except subprocess.TimeoutExpired:
            timed_out = timeout is not None and time.time() - started > timeout
            if not (cancel.is_set() or timed_out):
                continue
            _stop_group(proc)
            out, err = proc.communicate()
            if timed_out:
                return ProcResult(124, out or "", err or "", time.time() - started, timed_out=True)
            return ProcResult(CANCELLED_RETURNCODE, out or "", err or "", time.time() - started)


def _stop_group(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, then SIGKILL it if it is still there after a second."""
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _has_project(root: Path) -> bool:
    return any((candidate / "pyproject.toml").is_file() for candidate in [root, *root.parents])


def _probe_argv(tool: str, has_project: bool, root: Path) -> list[str] | None:
    """The actual probing: which argv works for this tool, tried in `root`.

    `root` matters even though it is not part of the cache key below: a probe
    run with no cwd at all inherits whatever directory the harness process
    happened to start in — not any workspace this run is actually touching —
    so `uv run <tool> --version` was being answered by an unrelated project's
    dependency state. Passing the workspace actually being resolved for at
    least answers the question about a *real* project, even though (per
    `_ResolutionCache` below) that answer is then reused for other workspaces
    with the same `has_project` value.
    """
    candidates: list[list[str]] = []
    if shutil.which(tool):
        candidates.append([tool])
    if shutil.which("uv"):
        if has_project:
            candidates.append(["uv", "run", "--quiet", tool])
        candidates.append(["uvx", "--quiet", tool])

    for argv in candidates:
        probe = run_capturing([*argv, "--version"], cwd=root, timeout=60, env=clean_env())
        if probe.returncode == 0 and not probe.timed_out:
            return argv
    return None


class _ResolutionCache:
    """The actual, cacheable resolution: which argv works for this tool at all.

    Deliberately NOT keyed on the workspace root. It was before, and the cache
    never paid for itself: bench runs a fresh tempfile.mkdtemp() workspace per
    job, so a (tool, root) key missed every single time across a sweep of
    hundreds of jobs, and each miss re-probed the tool up to three ways
    (bare PATH, `uv run`, `uvx`), each a real subprocess with a 60s cap.
    Whether `ruff` resolves via PATH or via `uv`/`uvx` depends on what's
    installed and whether *some* project file is present nearby — not on
    which particular temp directory that project file happens to live in —
    so caching on (tool, has_project) hits every time after the first call
    for that combination, which is the entire cacheable search space.

    `root` is threaded through to the first probe for a given key (see
    `_probe_argv`) without becoming part of the key itself, so the caching
    behaviour above is unchanged — only *which* project's environment the
    first probe for a combination is actually asked about.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, bool], list[str] | None] = {}
        self._hits = 0
        self._misses = 0

    def __call__(self, tool: str, has_project: bool, root: Path) -> list[str] | None:
        key = (tool, has_project)
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        result = _probe_argv(tool, has_project, root)
        self._store[key] = result
        return result

    def cache_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            hits=self._hits, misses=self._misses, maxsize=None, currsize=len(self._store)
        )

    def cache_clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


_resolve_for = _ResolutionCache()


def resolve(tool: str, root_str: str) -> list[str] | None:
    """Return the argv prefix that runs `tool` for this workspace, or None."""
    root = Path(root_str)
    return _resolve_for(tool, _has_project(root), root)


def run(
    tool: str,
    args: list[str],
    root: Path,
    timeout: int = _FALLBACK_TIMEOUT,
    extra_env: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
) -> tuple[int, str, str]:
    """Run an Astral tool. Returns (returncode, stdout, stderr).

    A missing tool is reported as returncode 127 rather than raising, because every
    caller is on a path that must hand a readable string back to the model.

    The environment is the workspace's, not a bare clean one. It used to be
    clean_env(), which meant pytest here ran *outside* the workspace's own
    virtualenv while run_bash and a step's verify command ran inside it — so
    `uv run pytest` passed for the model and the identical suite reported a
    ModuleNotFoundError collection error to the run_tests tool and to the
    final check. A real run fixed its packaging, watched its verify commands
    go green, and was still told by the gate that nothing collected. Tests
    have to be run the way the workspace actually runs them.

    It is `restricted_env()`, not the bare `workspace_env()`, for the same
    reason `run_verify` (agent/phases/execute.py) already runs a step's own
    verify command that way: every caller of this function — `run_tests`,
    `check_code`, the finalcheck gate's pytest/ruff/ty runs, `coverage.py` —
    executes model-authored or repository-authored code (pytest imports and
    runs it during collection), and `restricted_env()` is `workspace_env()`
    plus credential stripping, so this inherits the exact venv/PATH fix the
    paragraph above describes without reopening it.

    `extra_env` layers on top rather than replacing anything — for a setting
    that only this one call wants (COVERAGE_CORE for the coverage-wrapped
    suite run, say) without touching what every other subprocess sees.
    """
    argv = resolve(tool, str(root))
    if argv is None:
        return 127, "", f"{tool} is not available (tried PATH, `uv run`, and `uvx`)"
    env = restricted_env(root)
    if extra_env:
        env.update(extra_env)
    # Passed only when given, so every other caller's call is exactly what it was.
    stop = {"cancel": cancel} if cancel is not None else {}
    result = run_capturing([*argv, *args], cwd=root, timeout=timeout, env=env, **stop)
    if result.timed_out:
        return 124, "", f"{tool} timed out after {timeout}s"
    return result.returncode, result.stdout, result.stderr


def available(root: Path) -> dict[str, bool]:
    """Which tools this workspace can actually use. Reported once at run start.

    The four resolutions are independent (different binaries, same root), so
    they run concurrently — each is I/O-bound (a subprocess probe), and
    running them one after another paid the sum of four latencies for no
    reason.
    """
    from concurrent.futures import ThreadPoolExecutor

    names = ("ruff", "ty", "pytest", "git")
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = pool.map(lambda name: resolve(name, str(root)) is not None, names)
    return dict(zip(names, results, strict=True))
