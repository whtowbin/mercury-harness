"""Prompt text that matches the features a run has on (DESIGN-LOG §104).

The executor and fixer prompts were written before the harness ran the tests after each edit
(`check_after_edit`, §99), before `edit_blocks`, and before `read_file` took `sections`. Left
as they were, they tell the model to do the work the harness now does ("Re-run the tests after
your change") and name tools a kit may not carry (`edit_files`). Each replacement here applies
only when its feature is on, so a run with the features off sends byte-for-byte the prompt
every earlier sweep used. `tests/test_promptadapt.py` checks every old text is still present,
so a prompt edit that breaks a replacement fails a test instead of passing silently.
"""

from __future__ import annotations

# (old, new) pairs, by feature. Each `old` must occur in the prompt it targets.
AFTER_EDIT_EXECUTOR = (
    (
        "  3. run_tests, and confirm the new test fails for the reason you expect. Do not edit\n"
        "     source before you have seen this;",
        "  3. Confirm the new test fails for the reason you expect: the harness runs the tests\n"
        "     after every turn that edits and hands you the result with that edit, so this needs\n"
        "     no run_tests call. Do not edit source before you have seen it fail;",
    ),
    (
        "  5. run_tests again.\n",
        "  5. Read the result the harness hands back with that edit.\n",
    ),
    (
        "This costs one extra run_tests call and is the difference",
        "This costs one extra edit turn and is the difference",
    ),
    (
        '- Re-run the tests after your change. "It should pass now" is not evidence.',
        "- After each turn that edits, the harness runs the tests and appends the result: read it. "
        '"It should pass now" is not evidence. Call run_tests yourself only to run something '
        "different (one test with verbose=true, another file).",
    ),
)
AFTER_EDIT_FIXER = (
    (
        "Re-run run_tests afterwards when the\nfinding concerns behaviour, so you know the fix "
        "actually worked.",
        "After each turn that edits, the harness re-runs\nthis finding's own check and appends "
        "the result: read it rather than calling run_tests yourself.",
    ),
)
EDIT_BLOCKS = (
    (
        "The same for edit_files.",
        "The same for edit_blocks: several SEARCH/REPLACE blocks, to one file or many, in one "
        "call.",
    ),
    (
        "edit_files(edits=[<the fix in solver.py>, <the new test in test_solver.py>]),",
        "edit_blocks(blocks=<a block for the fix in solver.py, one for the new test>),",
    ),
    ("(edit_files for several edits,", "(edit_blocks for several edits,"),
)
# With read_assist: read_file as it now behaves (sections, widening, the re-read answer).
READS = (
    (
        "Decide everything you need, then ask once.",
        "read_file(path, sections='Game.update, 40-80') returns several parts of one file in one "
        "call. Decide everything you need, then ask once.",
    ),
    (
        "- If edit_file reports no match, re-read the region and retry with the exact text.",
        "- If edit_file reports no match, look at the region's exact current text again (re-read "
        "it if the file changed since you last saw it) and retry with that text.",
    ),
)
# With research_first the executor's kit has inspect_module (execute.py adds INTROSPECT_TOOLS);
# without it the prompt points at run_python, which every kit has. The prompt used to name
# inspect_module for every step, though only research-first steps were offered it (§113).
INTROSPECT = (
    (
        "- Never guess an API. run_python can print the real signature and docstring of any "
        "installed\n  function in one call (`import inspect; print(inspect.signature(f)); "
        "print(f.__doc__)`). Use it\n  before writing a call you are unsure of.",
        "- Never guess an API. inspect_module gives you the real signature, docstring and "
        "members of any\n  installed module, class or function in one call. Use it before "
        "writing a call you are unsure of.",
    ),
)
# The end-of-context reminder (Settings.executor_tail_reminder).
AFTER_EDIT_TAIL = (("then\n  run_tests on them)", "and the harness\n  runs the tests on them)"),)
READS_TAIL = (
    (
        "not one call per reply.",
        "not one call per reply; read_file sections= for several parts of one file.",
    ),
)
FORMATTED = (
    "\n- This project has a formatter: layout is formatted automatically when the step ends. Do "
    "not spend a call on formatting."
)


def _apply(text: str, pairs: tuple[tuple[str, str], ...]) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def executor(text: str, settings, formatted: bool = False) -> str:
    """The executor prompt adapted to this run's features."""
    if getattr(settings, "read_assist", False):
        text = _apply(text, READS)
    if getattr(settings, "check_after_edit", False):
        text = _apply(text, AFTER_EDIT_EXECUTOR)
    if getattr(settings, "edit_blocks", False):
        text = _apply(text, EDIT_BLOCKS)
    if getattr(settings, "research_first", False):
        text = _apply(text, INTROSPECT)
    if formatted and getattr(settings, "autoformat", True):
        text += FORMATTED
    return text


def fixer(text: str, settings) -> str:
    """The fixer prompt adapted to this run's features."""
    if getattr(settings, "read_assist", False):
        text = _apply(text, READS)
    if getattr(settings, "check_after_edit", False):
        text = _apply(text, AFTER_EDIT_FIXER)
    if getattr(settings, "edit_blocks", False):
        text = _apply(text, EDIT_BLOCKS)
    return text


def tail(text: str, settings) -> str:
    """The end-of-context reminder adapted to this run's features."""
    if getattr(settings, "edit_blocks", False):
        text = _apply(text, EDIT_BLOCKS)
    if getattr(settings, "check_after_edit", False):
        text = _apply(text, AFTER_EDIT_TAIL)
    if getattr(settings, "read_assist", False):
        text = _apply(text, READS_TAIL)
    return text


# The planner (DESIGN-LOG §108). With types_after_edit and check_after_edit the harness
# type-checks after every edit; otherwise ty runs when a step finishes, and the old sentence stays.
PLANNER_TYPES = (
    (
        "lint and\n  type checks already run on every write, so they are never a step's verify.",
        "lint runs on\n  every write and the harness type-checks after every edit, so they "
        "are never a step's verify.",
    ),
    (
        "every write is already linted\n  and type-checked automatically (see the executor's own "
        "instructions)",
        "every write is already linted,\n  and the harness type-checks after every edit",
    ),
)
# The skills block (agent/skills_data), appended after the system prompt is adapted, so it
# needs pairs of its own (§108). Each `old` is in the skill file it names.
SKILLS_TYPES = (  # ty.md
    (
        "**It is not automatic, and that is deliberate.** The syntax check and Ruff run\n"
        "after every write because they cost milliseconds. ty takes about a second\n"
        "because it resolves imports across the whole project, so you call it when a\n"
        "step is complete — not after every edit.",
        "**It runs for you after each turn that edits.** The harness runs ty then and\n"
        "appends the type errors this run introduced to that edit's result, so you do\n"
        "not need to call it to see them. Call it yourself only for files this step\n"
        "did not edit.",
    ),
    (
        "If a step changed a function's parameters or return type, run it before\nfinishing.",
        "If a step changed a function's parameters or return type, read the type\n"
        "errors the harness reports after that edit before finishing.",
    ),
)
SKILLS_EDIT_BLOCKS = (  # verify.md
    (
        "`edit_files` takes a list of search/replace edits, to one file or\n  several, in one "
        "call. If you know a file needs three changes, send three edits in one\n  `edit_files` "
        "call rather than three `edit_file` calls",
        "`edit_blocks` takes several SEARCH/REPLACE blocks, to one file or\n  several, in one "
        "call. If you know a file needs three changes, send three blocks in one\n  `edit_blocks` "
        "call rather than three `edit_file` calls",
    ),
)
SKILLS_AFTER_EDIT = (  # verify.md
    (
        "While iterating on one\n  failing test, run only that test by name until it passes; "
        "run the whole\n  file once at the end to confirm nothing else broke.",
        "After each turn that\n  edits, the harness re-runs the step's tests for you; call "
        "run_tests\n  yourself only for something narrower (one test by name) or different.",
    ),
)
SKILLS_AFTER_EDIT_FIX = (  # verify.md, in a fix loop
    (
        SKILLS_AFTER_EDIT[0][0],
        "After each turn that\n  edits, the harness re-runs this finding's own check for you; "
        "call run_tests\n  yourself only for something narrower (one test by name) or different.",
    ),
)
# The reviewer with Settings.review_presupplied_checks: the suite (and ty on this run's
# changes) ran before it was called, and run_tests is not in its kit (§108).
REVIEWER_CHECKS_OLD = "- run_tests runs pytest and hands back the failing assertions.\n"
REVIEWER_CHECKS = (
    "- The suite's output and the type checker's report on this run's changes are already in\n"
    "  your message, run on exactly this code. Nothing can change while you review, so neither\n"
    "  needs running again.\n"
)
REVIEWER_SUITE = (
    "- The suite's output is already in your message, run on exactly this code. Nothing can\n"
    "  change while you review, so it needs no second run.\n"
)


def _types_after_edit(settings) -> bool:
    """New type errors come back with an edit only when both switches are on: they ride the
    after-edit check (runner._check_after_edit)."""
    return getattr(settings, "types_after_edit", False) and getattr(
        settings, "check_after_edit", False
    )


def planner(text: str, settings) -> str:
    """The planner prompt adapted to this run's features."""
    if _types_after_edit(settings):
        text = _apply(text, PLANNER_TYPES)
    return text


def skills(text: str, settings, fixer: bool = False) -> str:
    """A rendered skills block adapted to this run's features. A fixer's after-edit check is
    its finding's own (a type-error finding's is ty), never the step's tests plus ty, so the
    ty pair is left out there and the tests pair names the finding's check."""
    if _types_after_edit(settings) and not fixer:
        text = _apply(text, SKILLS_TYPES)
    if getattr(settings, "edit_blocks", False):
        text = _apply(text, SKILLS_EDIT_BLOCKS)
    if getattr(settings, "check_after_edit", False):
        text = _apply(text, SKILLS_AFTER_EDIT_FIX if fixer else SKILLS_AFTER_EDIT)
    return text


def reviewer(text: str, settings, types_supplied: bool = False) -> str:
    """The reviewer prompt adapted to this run's features."""
    if getattr(settings, "review_presupplied_checks", False):
        text = text.replace(
            REVIEWER_CHECKS_OLD, REVIEWER_CHECKS if types_supplied else REVIEWER_SUITE
        )
    return text
