"""Tool registry and the shared context every tool receives."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from .. import langtools
from ..approval import Approver
from ..cancel import CancelToken
from ..diffs import Baseline
from ..guards import ResourceGuard
from ..interact import Interactor
from ..langtools import LanguageProfile
from ..llm import ChatClient
from ..memory import MemoryStore
from ..relevance import RelevanceJudge
from ..trace import Trace
from ..workspace import Workspace, walk_files


def _stat_entry(root: Path, path: Path) -> tuple[str, int, int] | None:
    """(rel_path, mtime_ns, size), or None if the file vanished mid-walk or
    resolves outside root (a broken/odd symlink) -- both are "skip it", not
    "crash the walk"."""
    try:
        stat = path.stat()
        rel = str(path.relative_to(root))
    except (OSError, ValueError):
        return None
    return rel, stat.st_mtime_ns, stat.st_size


def workspace_state(root: Path) -> str:
    """One digest standing for "the workspace, as it is right now".

    Built from `workspace_manifest`, so it sees every non-ignored file rather
    than only the ones a tool happened to mark as touched — which is the
    difference between noticing a `run_python` that wrote a file and silently
    missing it. Two equal digests mean no file under `root` has changed in
    name, size or mtime, which is what lets a check's result be reused
    instead of re-run.
    """
    digest = hashlib.sha256()
    manifest = workspace_manifest(root)
    for rel in sorted(manifest):
        mtime, size = manifest[rel]
        digest.update(f"{rel}\0{mtime}\0{size}\0".encode())
    return digest.hexdigest()


def workspace_manifest(root: Path) -> dict[str, tuple[int, int]]:
    """A cheap {rel_path: (mtime_ns, size)} snapshot of every non-ignored file
    under `root`. Just stat()s, not reads — enough to notice which files
    changed around a subprocess, whether that subprocess was a shell command
    or a step's verify command, without paying to read file contents.
    """
    manifest: dict[str, tuple[int, int]] = {}
    for path in walk_files(root):
        entry = _stat_entry(root, path)
        if entry is not None:
            rel, mtime_ns, size = entry
            manifest[rel] = (mtime_ns, size)
    return manifest


def content_fingerprint(ctx: ToolContext, paths: Iterable[str] | None = None) -> str:
    """A sha256 over the bytes of `paths` (default: every file this run has touched).

    Real content, not mtime+size: a same-length edit changes neither, and mtime
    granularity can be coarser than two edits a fast tool loop lands inside.
    Two fingerprints being equal means the path set and every byte of it are
    identical. A path that is missing or no longer resolves inside the workspace
    still contributes distinct bytes, so a deletion changes the digest too.
    Shared by the executor's verify-skip check (`execute.workspace_fingerprint`)
    and the tool loop's no-progress watcher, which used to keep two copies of
    the same recipe (code review of 2026-09-30, finding 10).
    """
    digest = hashlib.sha256()
    for rel in sorted(paths if paths is not None else ctx.touched):
        try:
            data = ctx.workspace.resolve(rel).read_bytes()
        except (OSError, ValueError):
            data = b"<unreadable>"
        digest.update(rel.encode() + b"\0" + data + b"\0")
    return digest.hexdigest()


def snapshot_before_write(ctx: ToolContext) -> dict[str, tuple[int, int]]:
    """The "before" step every external-write tool needs, in one walk.

    Two things, done together because every caller needs both from the same
    walk: stash each currently-existing file's content as its baseline, if
    it does not have one yet, *and* return a {rel_path: (mtime_ns, size)}
    manifest to compare against an "after" snapshot once the command has run.

    Baselining has to happen *before* the command, not after: `Baseline.capture`
    records "the content this file had before it was first touched", and by
    the time a post-run stat comparison notices a file changed, its pre-run
    content is already gone. `capture()` no-ops for paths already recorded,
    so on every call after a run's first this adds no reads to the walk —
    only new files created since need anything captured, and files the
    command itself creates get baselined as "did not exist" for free (no
    entry means `before is None` wherever a diff or fingerprint reads it).

    Shared by every tool that can write files outside the structured
    edit_file/write_file path — run_bash and ast_grep_rewrite, so that "did
    this command change something" means the same thing regardless of which
    of them ran it. (This used to be two functions, `baseline_existing_files()`
    and `workspace_manifest()`, called back to back — both walking the
    identical tree, so doing it twice in a row doubled a cost pruning had
    already cut once. Folded into one walk here; `workspace_manifest()`
    below is kept standalone for the "after" side, which needs no baseline.)
    """
    manifest: dict[str, tuple[int, int]] = {}
    root = ctx.workspace.root
    for path in walk_files(root):
        ctx.baseline.capture(ctx.workspace.rel(path), path)
        entry = _stat_entry(root, path)
        if entry is not None:
            rel, mtime_ns, size = entry
            manifest[rel] = (mtime_ns, size)
    return manifest


def mark_external_writes(ctx: ToolContext, before: dict[str, tuple[int, int]]) -> None:
    """Whatever changed while an external command ran is fed into the same
    touched machinery a file-editing tool would have used.

    A shell command, a Python snippet, or an ast-grep rewrite can create or
    rewrite source files just as easily as edit_file can, but until this
    existed only `ctx.cache.invalidate()` acknowledged that — review
    chunking, syntax checks, changed-line coverage, the no-tests
    behavioural-change gate, run diffs, and `workspace_fingerprint` all
    derive from `ctx.touched`, so a file one of these wrote was invisible to
    every one of them. A generator, `sed -i`, or a structural rewrite could
    leave a workspace with broken, untested behaviour and still sail through
    a green final check.
    """
    after = workspace_manifest(ctx.workspace.root)
    for rel_path, stat_pair in after.items():
        if before.get(rel_path) != stat_pair:
            ctx.mark_touched(rel_path)
    for rel_path in before:
        if rel_path not in after:
            # Deleted by the command. Nothing to re-read, but still worth
            # surfacing as a change to whatever last had this file's content.
            ctx.mark_touched(rel_path)


@dataclass
class ReadCache:
    """Memoises file reads within a step. Re-reading a file the model just looked at
    is common and, at Mercury's speed, the syscalls are a measurable share of a turn."""

    entries: dict[str, tuple[float, int, str]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    # Concurrent steps share one cache, so every mutation is guarded.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def get(self, path: Path) -> str | None:
        with self._lock:
            return self._get(path)

    def _get(self, path: Path) -> str | None:
        key = str(path)
        cached = self.entries.get(key)
        if cached is None:
            self.misses += 1
            return None
        mtime, size, text = cached
        try:
            stat = path.stat()
        except OSError:
            self.entries.pop(key, None)
            self.misses += 1
            return None
        # Anything that changed the file out from under us invalidates the entry,
        # including edits made by a shell command rather than by edit_file.
        if stat.st_mtime_ns / 1e9 != mtime or stat.st_size != size:
            self.entries.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return text

    def put(self, path: Path, text: str) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        with self._lock:
            self.entries[str(path)] = (stat.st_mtime_ns / 1e9, stat.st_size, text)

    def invalidate(self, path: Path | None = None) -> None:
        with self._lock:
            if path is None:
                self.entries.clear()
            else:
                self.entries.pop(str(path), None)


class WriteOutsideStep(Exception):
    """A step in a parallel batch tried to modify a file it did not claim.

    The scheduler decides two steps may overlap by comparing the files they
    declare. That is a promise the plan makes, and until now nothing held it
    to it: a step could declare game.py and then write test_game.py, which its
    sibling was in the middle of writing too. Enforcing the declaration is
    what turns "the plan says this is safe" into "this is safe".
    """


@dataclass
class ToolContext:
    workspace: Workspace
    approver: Approver
    trace: Trace
    interactor: Interactor = field(default_factory=Interactor)
    guard: ResourceGuard = field(default_factory=ResourceGuard)
    baseline: Baseline = field(default_factory=Baseline)
    cache: ReadCache = field(default_factory=ReadCache)
    # Checked between operations (tool-loop iterations, plan steps) so a
    # requested stop takes effect at the next safe point rather than killing
    # something mid-write. Shared run-wide, not per-step, like interactor.
    cancel: CancelToken = field(default_factory=CancelToken)
    # What this workspace remembers from earlier runs, and the scratchpad for
    # this one. None when the caller did not supply one (a bare ToolContext in
    # a test); the note tools degrade to an error rather than crashing.
    memory: MemoryStore | None = None
    # The run's own model client, reused (not a second one spun up) by tools
    # that need to make their own LLM call outside the plan/execute/review
    # phases -- currently just audit_previous_runs' deep pass. None in a bare
    # ToolContext (tests, or a caller that never wired one), which the tool
    # degrades to "deterministic checks only" for rather than crashing.
    audit_client: ChatClient | None = None
    # Which backend decides what's still relevant when a step's tool-loop
    # context needs pruning (agent/relevance.py). Built once per run by
    # relevance.build_judge and shared here rather than reconstructed per
    # call, the same reasoning as audit_client above. None only in a bare
    # ToolContext (tests, or a caller that never wired one), which
    # runner._compact degrades to its plain FIFO fallback for.
    relevance_judge: RelevanceJudge | None = None

    # Files the agent has touched this run, in order. Drives the review phase.
    touched: list[str] = field(default_factory=list)
    # Dependencies added via `uv add`, so later steps know what is importable.
    dependencies: list[str] = field(default_factory=list)
    # What run_python / run_bash printed this run, as {"tool", "command",
    # "output"}. The trace records a call's arguments but not its output, and
    # the reviewer is asked to trace every measured number in a changed document
    # back to a command that produced it -- which it cannot do without these.
    command_log: list[dict] = field(default_factory=list)
    # Run Ruff automatically after each write. Syntax checks always run.
    autolint: bool = True
    # Replay a read-only check's recorded output when the identical call is
    # made again against a byte-for-byte unchanged workspace. See
    # Settings.replay_identical_checks and runner._cached_result.
    replay_checks: bool = True
    # The most recent `run_tests` that passed, as (workspace state, the
    # targets it ran). A step's own verify command is very often the same
    # pytest invocation the model just ran -- 667 of 960 traced step verifies
    # followed a passing run_tests with nothing mutating in between, 403 of
    # them over the whole suite -- and re-running it costs the same wall time
    # over again for a result that cannot have changed.
    tests_green: tuple[str, tuple[str, ...]] | None = None
    # "auto" while the harness itself runs the tests after an edit turn
    # (Settings.check_after_edit), so the run_tests trace event says who ran them
    # and the TUI can label it; empty for the model's own run_tests calls.
    test_source: str = ""
    # read_file's record of what this loop has already shown, so an identical re-read of
    # unchanged text gets a one-line answer (DESIGN-LOG §104). `read_epoch` moves on at each new
    # tool loop and at compaction: the earlier output is then gone from the model's context.
    read_epoch: int = 0
    read_assist: bool = True
    read_seen: dict = field(default_factory=dict)
    # Settings.web_research: the search-churn nudge (kitnudges.nudge_about_search_churn).
    web_research: bool = False
    # Type diagnostics that existed before this run touched anything, so a
    # step is failed only for the ones it introduced rather than for the
    # project's pre-existing debt. See finalcheck.collect_type_baseline.
    type_baseline: frozenset[str] | None = None
    # Settings.premise_gate (runner.PremiseGate): the plan's own reproduction
    # when it passed, as (snippet, output), and which of the gate's holds this
    # run has already used. Run-scoped, so shared with every step's context.
    repro_passed: tuple[str, str] | None = None
    premise_holds: set[str] = field(default_factory=set)
    # Settings.bugfix_evidence (agent/evidence.py): the plan's reproduction when it failed at
    # plan time, re-run before a fix step finishes, and which of EvidenceGate's once-a-run holds
    # this run has used. Run-scoped like premise_holds (DESIGN-LOG §120).
    repro_reproduced: str | None = None
    evidence_holds: set[str] = field(default_factory=set)
    # Settings.test_change_justification (agent/testchange.py): how many times this
    # run has held an edit to a pre-existing test, and the justified (or
    # unjustified) changes it let through, for the end-of-run summary. A dict so
    # every step's context shares one, as premise_holds is.
    test_change_state: dict = field(default_factory=lambda: {"holds": 0, "log": []})
    # Settings.test_command / test_label_root: the project's own test runner,
    # used by run_tests and the final check in place of pytest when set.
    test_command: str = ""
    test_label_root: str = ""
    # Settings.web_deny: URL substrings fetch_url refuses and web_search drops
    # from its results (agent/tools/web.py, denied_by). A benchmark run lists its
    # own sources here so the agent cannot read the answers (DESIGN-LOG §80).
    web_deny: tuple[str, ...] = ()
    # {(call fingerprint, workspace fingerprint): recorded output} for those
    # replays. Shared across a run's steps like `cache` is: the key already
    # pins the exact workspace state, so an entry recorded by one step is
    # only ever reused while that state still holds.
    replay: dict = field(default_factory=dict, repr=False, compare=False)
    # A short line before each state-changing tool call, so a human watching
    # can follow along step by step. See Settings.narrate for the full story.
    narrate: bool = True
    # Reference material available to inject into phase prompts.
    skills: list = field(default_factory=list)
    skill_token_budget: int = 3_000
    skill_triggers: str = "step"  # Settings.skill_triggers (DESIGN-LOG §117, default §117.1)
    # Shared by every context in a run, so concurrent steps serialise their bookkeeping.
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # Set by finish_step / submit_* tools to break the loop.
    result: Any = None
    done: bool = False
    # Workspace-relative paths this step may modify, or None for "anything".
    # None is the sequential default, so a run with concurrency off behaves
    # exactly as it always has; the jail closes only when a step is actually
    # sharing the workspace with a sibling.
    writable: frozenset[str] | None = None

    def for_step(self, writable: frozenset[str] | None = None) -> ToolContext:
        """A sibling context for one concurrent step.

        Workspace state is shared — the baseline, the cache and the touched list
        describe the run, not the step — but the resource guard is per-step, since
        its budgets are.
        """
        return ToolContext(
            workspace=self.workspace,
            approver=self.approver,
            interactor=self.interactor,
            trace=self.trace,
            cancel=self.cancel,
            memory=self.memory,
            audit_client=self.audit_client,
            relevance_judge=self.relevance_judge,
            guard=ResourceGuard(
                step_subprocess_budget_s=self.guard.step_subprocess_budget_s,
                max_workspace_mb=self.guard.max_workspace_mb,
                max_file_write_mb=self.guard.max_file_write_mb,
                max_step_errors=self.guard.max_step_errors,
            ),
            baseline=self.baseline,
            cache=self.cache,
            touched=self.touched,
            dependencies=self.dependencies,
            command_log=self.command_log,
            autolint=self.autolint,
            read_assist=self.read_assist,
            web_research=self.web_research,
            replay_checks=self.replay_checks,
            replay=self.replay,
            tests_green=self.tests_green,
            type_baseline=self.type_baseline,
            repro_passed=self.repro_passed,
            premise_holds=self.premise_holds,
            repro_reproduced=self.repro_reproduced,
            evidence_holds=self.evidence_holds,
            test_change_state=self.test_change_state,
            test_command=self.test_command,
            test_label_root=self.test_label_root,
            web_deny=self.web_deny,
            narrate=self.narrate,
            skills=self.skills,
            skill_token_budget=self.skill_token_budget,
            skill_triggers=self.skill_triggers,
            lock=self.lock,
            writable=writable,
        )

    def mark_touched(self, rel_path: str) -> None:
        with self.lock:
            if rel_path not in self.touched:
                self.touched.append(rel_path)

    @cached_property
    def language_profile(self) -> LanguageProfile | None:
        """This workspace's detected language, computed once per context and
        cached — not once per step. See langtools.detect_profile: a marker
        file at the workspace root, or (only when neither pyproject.toml nor
        package.json exists) a fallback via `scc`'s language breakdown.

        A `for_step()` sibling created for a concurrent batch does not
        inherit this cache and detects again on first use; that's a handful
        of `Path.is_file()` calls in the common case (or, rarely, one `scc`
        subprocess when no marker file exists at all), not a per-step cost
        worth threading a shared cache through the sibling constructor for.
        """
        return langtools.detect_profile(self.workspace.root)

    def before_write(self, rel_path: str, path: Path) -> None:
        """Snapshot a file's pre-run state before the first modification to it.

        Also the one place every fs write passes through, so it is where the
        parallel write jail is enforced. (run_bash and run_python can still
        write around it — that was already true of the disjointness the
        scheduler assumed, and is why a step declaring no files runs alone.)
        """
        if self.writable is not None and rel_path not in self.writable:
            raise WriteOutsideStep(
                f"this step may not modify {rel_path}: it is running at the same time "
                f"as another step, and declared only {sorted(self.writable) or 'nothing'} "
                "as the files it writes. Do the part of the work that belongs to this "
                "step, and leave the rest to the step that claimed it."
            )
        with self.lock:
            self.baseline.capture(rel_path, path)

    def try_read(self, rel_path: str) -> str | None:
        """Resolve a workspace-relative path and read it, or None.

        None covers every "there is nothing to read here" case with one
        answer: the path escapes the jail, does not exist, is a directory, or
        cannot be decoded. Four call sites (the execute preload, both review
        loaders, and check_code's target resolution) each open-coded this
        try/resolve/is_file dance with subtly different error handling; a
        prompt builder wants one question answered, not three.
        """
        try:
            path = self.resolve_existing(rel_path)
        except Exception:
            return None
        if path is None:
            return None
        try:
            return self.read_text(path)
        except OSError:
            return None

    def resolve_existing(self, rel_path: str) -> Path | None:
        """The resolved path if it is a readable file, else None."""
        path = self.workspace.resolve(rel_path)
        return path if path.is_file() else None

    def read_text(self, path: Path) -> str:
        """Cached read used by every tool that needs file contents."""
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        text = path.read_text(encoding="utf-8", errors="replace")
        self.cache.put(path, text)
        return text


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    # A terminal tool ends the loop when called.
    terminal: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def add(self, tools: list[Tool]) -> list[str]:
        """Offer `tools` from the next call on; returns the names that were new.

        Only a nudge does this, at most once per tool per loop: the schemas
        change mid-step, which ends any cached prefix at the tool list, so the
        price is one uncached prompt on a turn that already earned the nudge.
        """
        added = [t.name for t in tools if t.name not in self._tools]
        self._tools.update({t.name: t for t in tools if t.name in added})
        return added


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }
