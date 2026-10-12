import re
from functools import cache

from .. import langtools
from ..langtools import LanguageProfile
from .ask import ASK_TOOLS
from .audit import AUDIT_TOOLS
from .base import ReadCache, Registry, Tool, ToolContext, WriteOutsideStep, workspace_manifest
from .checks import CHECK_TOOLS, PROBE_PYTHON, READ_ONLY_CHECK_TOOLS
from .control import FINISH_STEP, SUBMIT_FINDINGS, submit_plan_tool
from .fs import FS_TOOLS
from .introspect import INTROSPECT_TOOLS
from .notes import NOTE_TOOLS, READ_SKILL
from .other_lang import LANGTOOLS_TOOLS
from .shell import SHELL_TOOLS
from .vcs import VCS_TOOLS, VCS_WRITE_TOOLS
from .web import WEB_TOOLS, available_web_tools

READ_ONLY_FS = {"list_dir", "read_file", "read_files", "grep", "glob_files"}
# edit_files is a write tool, so it is not added to the read-only set above.


def _read_only_fs() -> list[Tool]:
    return [t for t in FS_TOOLS if t.name in READ_ONLY_FS]


def _offered(tools: list[Tool]) -> list[Tool]:
    """`tools` without any web tool that has no backend to answer it."""
    available = {t.name for t in available_web_tools()}
    web = {t.name for t in WEB_TOOLS}
    return [t for t in tools if t.name not in web or t.name in available]


_ALL_EXECUTION = [
    *FS_TOOLS,
    *SHELL_TOOLS,
    *CHECK_TOOLS,
    *VCS_TOOLS,
    *VCS_WRITE_TOOLS,
    *WEB_TOOLS,
    *ASK_TOOLS,
    *NOTE_TOOLS,
    *LANGTOOLS_TOOLS,
    *AUDIT_TOOLS,
    READ_SKILL,
    FINISH_STEP,
]

# Every tool schema is re-sent on every call of a step, so the kit is a fixed
# tax on each of them: the twenty-eight execution tools cost ~3,900 tokens,
# the largest single block in a ~12,300-token executor prompt and larger than
# the system prompt and the skills put together.
#
# Across 782 traced execute calls, eleven of the twenty-eight were never used
# once. The cost is not only tokens: in one traced step the model reached for
# four different tools in a row — glob_files twice with near-identical
# patterns, then list_dir, then `run_bash find` — to answer "what files are
# here", which is what tool choice looks like when there are too many tools
# that overlap.
#
# So a step gets a core kit plus whatever its own text asks for, using the
# same deliberately dumb, model-free trigger matching skills already use. A
# missed trigger is cheap: run_bash is in the core and covers git, moves and
# most lookups, and it is already the most-used tool in the phase.
_CORE_NAMES = frozenset(
    {
        "list_dir",
        "read_file",
        "read_files",
        "grep",
        "glob_files",
        "write_file",
        "edit_file",
        "edit_files",
        "delete_file",
        "run_bash",
        "run_python",
        "run_tests",
        "check_code",
        # Not read_skill: 1 executor call in 59,338 and 2 in fix, while its schema and the
        # catalogue it serves cost about 420 tokens of every executor call (DESIGN-LOG §113).
        # The executor's skills are pushed into its prompt; the planner and reviewer keep it.
        "show_changes",
        "ask_user",
        "finish_step",
    }
)

# (group name, triggers, tool names). Order is irrelevant; groups are independent.
_OPTIONAL_GROUPS: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...] = (
    (
        "research",
        (
            "http",
            "url",
            "api",
            "web",
            "download",
            "docs",
            "pypi",
            "package",
            "install",
            "dependency",
            "dependencies",
            "depend",
            "library",
            "version",
            "upgrade",
        ),
        frozenset({"fetch_url", "web_search", "pypi_info"}),
    ),
    (
        "vcs",
        ("git", "commit", "branch", "staged", "stage", "history", "revert", "rebase"),
        frozenset({"git_diff", "git_log", "git_add", "git_commit"}),
    ),
    (
        "debug",
        (
            "debug",
            "trace",
            "step through",
            "why does",
            "why is",
            "what happens when",
            "investigate",
            "diagnose",
            "root cause",
        ),
        frozenset({"trace_python"}),
    ),
    (
        "complexity",
        ("complexity", "refactor", "simplify", "cognitive", "nesting", "clean up"),
        frozenset({"check_complexity"}),
    ),
    (
        "memory",
        ("remember", "recall", "memory", "previous run", "last time", "earlier run"),
        frozenset({"remember", "recall"}),
    ),
    ("move", ("move", "rename", "renaming", "relocate"), frozenset({"move_file"})),
    (
        "other-language",
        (
            "package.json",
            "eslint",
            "npm",
            "node",
            "javascript",
            "typescript",
            ".ts",
            ".tsx",
            ".jsx",
            "go.mod",
            "cargo.toml",
        ),
        frozenset({"lint_project", "test_project"}),
    ),
    (
        "data-formats",
        ("json", "yaml", "yml", "toml", "xml", "config file", "schema", "validate"),
        frozenset({"validate_data_file"}),
    ),
    # Deliberately separate from "memory" above: recall/remember read the
    # harness's own condensed notes and cost nothing, audit_previous_runs can
    # spend a real model call (deep=true) -- narrower triggers, specific to
    # "this failed before", not the whole "previous run" surface.
    (
        "audit-history",
        (
            "went wrong",
            "still broken",
            "still failing",
            "didn't work",
            "does not work",
            "doesn't work",
            "same bug",
            "same error",
            "keeps failing",
            "regression",
            "previous attempt",
            "prior attempt",
            "prior run",
            "last attempt",
            "tried before",
        ),
        frozenset({"audit_previous_runs"}),
    ),
)


# A trigger matches as a whole word, give or take a plural or tense ending.
# A bare substring test fired the move group on 886 of 888 bench steps
# because "move" is inside "remove", and the research group on 659 of 767
# because of "conversion" and "docstring" -- and in 3,915 bench runs an
# optional tool was called on 4 of the ~3,100 step kits that carried one, so
# a false match bought nothing but schema tokens (DESIGN-LOG §62). A trigger
# that starts with punctuation (".ts") keeps only the right-hand boundary,
# since a filename's own stem sits directly before it.
_SUFFIXES = r"(?:s|es|d|ed|ing)?"
# A trigger ending in a silent "e" drops it before -ing ("move" -> "moving",
# "diagnose" -> "diagnosing"), which appending a suffix never matched. Replayed
# over 6,803 traced step texts this changed the kit of 6, so it is a
# correctness fix, not a behaviour change anyone should expect to measure.
_E_SUFFIXES = r"(?:e|es|ed|ing)"


@cache
def _trigger_pattern(trigger: str) -> re.Pattern[str]:
    left = r"(?<![a-z0-9_])" if trigger[:1].isalnum() else ""
    if trigger.endswith("e") and trigger[-2:-1].isalpha():
        stem = re.escape(trigger[:-1]) + _E_SUFFIXES
    else:
        stem = re.escape(trigger) + _SUFFIXES
    return re.compile(left + stem + r"(?![a-z0-9_])")


def _triggered(triggers: tuple[str, ...], lowered: str) -> bool:
    return any(_trigger_pattern(t).search(lowered) for t in triggers)


def tools_named(names: frozenset[str] | set[str]) -> list[Tool]:
    """The execution tools called `names`, for widening a kit mid-loop: a
    nudge that names a tool the step was not offered adds it rather than
    pointing at something the model cannot call. A web tool with no backend
    stays out, as it does from every kit."""
    return [t for t in _offered(_ALL_EXECUTION) if t.name in names]


def execution_tool_names(step_text: str, profile: LanguageProfile | None = None) -> set[str]:
    """The kit for a step: the core, plus every group its text triggers.

    `profile` — the workspace's detected `LanguageProfile` (see langtools.py),
    `None` for Python or nothing recognised — additionally, unconditionally
    includes the `other-language` group's tools for a *detected* non-Python
    workspace, regardless of whether this step's own text happens to mention
    "npm" or "eslint": a workspace-wide property like "this is a Node
    project" shouldn't depend on a step remembering to say so, the way a
    one-off "validate this JSON file" request reasonably should. A Python
    workspace (or an undetected one) gets nothing extra here — check_code/
    run_tests already cover it, in the core kit, unchanged.
    """
    lowered = step_text.lower()
    chosen = set(_CORE_NAMES)
    for _group, triggers, names in _OPTIONAL_GROUPS:
        if _triggered(triggers, lowered):
            chosen |= names
    if profile is not None and profile is not langtools.PYTHON:
        chosen |= {"lint_project", "test_project"}
    return chosen


def with_edit_blocks(registry: Registry) -> Registry:
    """The same kit with `edit_blocks` in place of the list-of-objects batch tools
    (`edit_files`, `batch_edits_to_file`), whose arguments fail to parse 4% of the time
    (Settings.edit_blocks, DESIGN-LOG §104). A kit with neither is returned unchanged."""
    batch = {"edit_files", "batch_edits_to_file"}
    if not batch & set(registry.names()):
        return registry
    blocks = next(t for t in FS_TOOLS if t.name == "edit_blocks")
    kept = [t for t in registry.tools() if t.name not in batch]
    return Registry([*kept, blocks])


def execution_registry(
    step_text: str | None = None,
    profile: LanguageProfile | None = None,
    extra_tools: frozenset[str] | None = None,
) -> Registry:
    """The kit given to the dLLM during a plan step or a fix.

    `step_text` None means the full kit, which is what callers that want every
    tool (and the tests that pin the whole surface) get. Passing the step's own
    text instead narrows it to the core plus what that text asks for; `profile`
    additionally widens it for a detected non-Python workspace — see
    execution_tool_names. `extra_tools` (see Settings.force_tools) adds names
    unconditionally, regardless of `step_text`'s own triggers -- for measuring
    a tool's effect on a task whose auto-generated step titles would not
    reliably say the trigger words, not for routine use.
    """
    offered = _offered(_ALL_EXECUTION)
    if step_text is None:
        return Registry(offered)
    wanted = execution_tool_names(step_text, profile) | (extra_tools or frozenset())
    return Registry([t for t in offered if t.name in wanted])


def planning_registry(draft_style: str = "terse", probe: bool = False) -> Registry:
    """Read-only exploration, library lookup, and the plan submission tool.

    Gets ask_user too: an ambiguous goal is a planning-time problem, and
    resolving it before the plan is written is cheaper than an executor
    guessing wrong three steps in. Gets audit_previous_runs unconditionally
    (not gated behind execution's optional-group triggers) because "what did
    an earlier attempt at this already try, and why did it fail" is exactly
    the question worth answering before writing a plan, not partway through
    one -- see agent/tools/audit.py.

    `probe` (Settings.plan_probe) adds probe_python, which runs code against a
    throwaway copy of the workspace, so the kit still cannot change the
    workspace itself.
    """
    return Registry(
        [
            *_read_only_fs(),
            *INTROSPECT_TOOLS,
            *VCS_TOOLS,
            *available_web_tools(),
            *ASK_TOOLS,
            *NOTE_TOOLS,
            *AUDIT_TOOLS,
            READ_SKILL,
            *([PROBE_PYTHON] if probe else []),
            submit_plan_tool(draft_style),
        ]
    )


def review_registry() -> Registry:
    """Read-only by design: the reviewer critiques, the executor fixes.

    Actually enforced, not just asserted, for writes: no run_bash (arbitrary
    shell), no run_python (arbitrary code, equally capable of a write), and
    check_code here is the fix-less variant — its Python signature has no
    `fix` parameter for a model to invoke, rather than one merely left off
    the schema, since the dispatcher does not validate a model's tool-call
    arguments against the declared schema before calling the tool.

    One honest exception: `inspect_module` (via INTROSPECT_TOOLS) imports a
    module to report its real signature, which runs that module's top-level
    code — real execution, just not one this registry's tools can turn into
    a workspace write. It runs under `restricted_env()`, the same
    credential-stripped environment every other untrusted-code subprocess in
    this codebase uses, for exactly that reason.
    """
    return Registry(
        [
            *_read_only_fs(),
            *READ_ONLY_CHECK_TOOLS,
            *INTROSPECT_TOOLS,
            *VCS_TOOLS,
            *available_web_tools(),
            *NOTE_TOOLS,
            READ_SKILL,
            SUBMIT_FINDINGS,
        ]
    )


__all__ = [
    "ASK_TOOLS",
    "AUDIT_TOOLS",
    "CHECK_TOOLS",
    "FS_TOOLS",
    "INTROSPECT_TOOLS",
    "LANGTOOLS_TOOLS",
    "NOTE_TOOLS",
    "READ_ONLY_CHECK_TOOLS",
    "SHELL_TOOLS",
    "VCS_TOOLS",
    "VCS_WRITE_TOOLS",
    "WEB_TOOLS",
    "ReadCache",
    "Registry",
    "Tool",
    "ToolContext",
    "WriteOutsideStep",
    "execution_registry",
    "execution_tool_names",
    "planning_registry",
    "review_registry",
    "tools_named",
    "with_edit_blocks",
    "workspace_manifest",
]
