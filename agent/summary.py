"""The short account of a run, written for a person at the end of a session.

Everything here is derived from the run summary that `loop.run` already
returns — no model call, no extra latency, no chance of a summary that
disagrees with what actually happened. A model-written paragraph would read
more smoothly, but it would also be one more thing that can hallucinate about
the very run it is summarising, and "what did this just do?" is exactly the
question you cannot afford a confident wrong answer to.

Three sections, in the order someone actually wants them: what the run built,
what it failed to do, and what to do next.
"""

from __future__ import annotations

import re

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

MAX_BULLETS = 8


def _plural(count: int, singular: str, plural: str = "") -> str:
    return singular if count == 1 else (plural or singular + "s")


def _built_bullets(summary: dict) -> list[str]:
    """One line per step that reported doing something, else per changed file."""
    plan = summary.get("plan") or {}
    steps = [s for s in (plan.get("steps") or []) if str(s.get("summary") or "").strip()]
    bullets: list[str] = []
    for step in steps[:MAX_BULLETS]:
        note = str(step["summary"]).strip()
        # An abandoned step reports itself here; it belongs under "failed".
        if note.startswith("abandoned:"):
            continue
        files = ", ".join(step.get("files") or [])
        where = f" ({files})" if files else ""
        bullets.append(f"[{step.get('id', '?')}]{where} {note}")
    if bullets:
        return bullets
    # No step summaries (a saved plan re-run, a stopped run): fall back to the
    # files, which is still a truthful account of what changed.
    changed = [str(f) for f in (summary.get("changed_files") or [])]
    return [f"changed {name}" for name in changed[:MAX_BULLETS]]


# Draft lines that say what the planner found or assumed, as opposed to what it will do.
_FOUND_LINE = re.compile(
    r"^\s*(understood|issue|problem|cause|root cause|found|finding|assumption|assumed)"
    r"\b[^:]{0,12}:",
    re.IGNORECASE,
)
MAX_FOUND_CHARS = 300


def _found_bullets(summary: dict) -> list[str]:
    """What the planner found and assumed: its `understood:` reading, the issue, cause and
    assumption lines of its draft, and its `decision:` findings.

    The platformer-app run that was asked why the flagpole never appeared found the cause
    in planning ("Flagpole exists at x=1100 but SCREEN_WIDTH=800 making it off-screen") and
    assumed it should stay off screen; the summary listed only the tests it built, so
    neither reached the person (DESIGN-LOG §98).
    """
    plan = summary.get("plan") or {}
    lines = [str(d) for d in (plan.get("draft") or []) if _FOUND_LINE.match(str(d))]
    lines += [str(f) for f in (plan.get("findings") or []) if str(f).lower().startswith("decision")]
    bullets = []
    for line in lines[:4]:
        text = " ".join(line.split())
        bullets.append(text if len(text) <= MAX_FOUND_CHARS else text[:MAX_FOUND_CHARS] + "…")
    return bullets


def _failed_bullets(summary: dict) -> list[str]:
    steps = summary.get("steps") or {}
    check = summary.get("final_check") or {}
    bullets: list[str] = []
    if summary.get("error"):
        bullets.append(str(summary["error"]))
    for step_id in steps.get("failed") or []:
        bullets.append(f"step [{step_id}] failed its verify command")
    for step_id in steps.get("abandoned") or []:
        bullets.append(f"step [{step_id}] was abandoned before finishing")
    if check and not check.get("ok", True):
        for failure in check.get("verify_failures") or []:
            bullets.append(
                f"verify still fails for [{failure.get('step')}]: {failure.get('command')}"
            )
        for name in check.get("new_failures") or []:
            bullets.append(f"test now failing that passed before: {name}")
        if (
            not check.get("new_failures")
            and check.get("tests_ran")
            and not check.get("tests_passed")
        ):
            bullets.append(f"test suite failing: {check.get('tests_summary', '')}")
    return bullets[:MAX_BULLETS]


def _test_change_bullets(changes: list[dict]) -> list[str]:
    """One line per change to a test that existed before the run, with the reason
    the model gave, so a person can judge it instead of finding it in a diff.
    An unjustified change is said to be one."""
    out: list[str] = []
    for change in changes[:MAX_BULLETS]:
        tests = ", ".join(change.get("tests") or [])
        kinds = "/".join(sorted(set(change.get("kinds") or [])))
        if change.get("basis") == "unjustified":
            out.append(f"changed existing test(s) with no acceptable reason ({kinds}): {tests}")
        else:
            reason = str(change.get("justification") or "").strip()
            out.append(f"changed existing test(s) ({kinds}): {tests} — {reason}")
    return out


def _next_bullets(summary: dict) -> list[str]:
    bullets: list[str] = []
    suggestion = str(summary.get("suggested_next_step") or "").strip()
    if suggestion:
        bullets.append(suggestion)
    check = summary.get("final_check") or {}
    # A broken toolchain is the human's job, and it is the most useful next
    # step there is: every other signal in the run is weaker until it is fixed.
    for failure in check.get("env_failures") or []:
        bullets.append(
            f"repair the environment, then re-run: {failure.get('command')} "
            f"could not run ({failure.get('note')})"
        )
    pre_existing = check.get("pre_existing_failures") or []
    if pre_existing:
        listed = ", ".join(pre_existing[:3])
        noun = _plural(len(pre_existing), "test")
        was_were = "was" if len(pre_existing) == 1 else "were"
        if check.get("fix_pre_existing"):
            # Counted against the run, so "left alone" would be false: it was
            # asked to fix these and did not. Say what they still report.
            why = "; ".join(
                line.strip(" -") for line in str(check.get("pre_existing_why") or "").splitlines()
            )
            bullets.append(
                f"{len(pre_existing)} {noun} already failing before this run "
                f"{was_were} not fixed: {listed}" + (f" ({why})" if why else "")
            )
        else:
            bullets.append(
                f"{len(pre_existing)} {noun} {was_were} already failing before "
                f"this run and {was_were} left alone: {listed}"
            )
    # Report-only: the tests passed, but these say whether they could have
    # failed. Kept to the first finding so a clean run stays quiet and a
    # flagged one is not buried under its own detail.
    for finding in ((summary.get("test_audit") or {}).get("findings") or [])[:1]:
        bullets.append(f"tests may not cover the change: {finding}")
    bullets += _test_change_bullets(summary.get("test_changes") or [])
    unverified = (summary.get("steps") or {}).get("unverified") or []
    if unverified and not bullets and summary.get("stopped_after") != "plan":
        bullets.append(
            f"{len(unverified)} {_plural(len(unverified), 'step')} finished "
            "without anything confirming them; consider a test for that work"
        )
    return bullets[:MAX_BULLETS]


def _steps_phrase(steps: dict) -> str:
    """ "3 steps ran, all verified" — or how many of them were not."""
    total = steps.get("total") or 0
    if not total:
        return ""
    bad = len(steps.get("failed") or []) + len(steps.get("abandoned") or [])
    tail = f", {total - bad} of them clean" if bad else ", all verified"
    return f"{total} {_plural(total, 'step')} ran{tail}"


def _files_phrase(changed: list[str]) -> str:
    """ "2 files changed (a.py, b.py)", with a tail when there are many."""
    if not changed:
        return ""
    shown = ", ".join(changed[:4]) + (", …" if len(changed) > 4 else "")
    return f"{len(changed)} {_plural(len(changed), 'file')} changed ({shown})"


def _verdict(summary: dict) -> str:
    """The one sentence that says whether this actually worked."""
    steps = summary.get("steps") or {}
    check = summary.get("final_check") or {}
    if summary.get("error"):
        return f"The run did not finish: {summary['error']}."
    if check and not check.get("ok", True):
        return "The final check failed, so this work does not yet pass its own tests."
    # `ok` alone is not "the tests pass": with a baseline it can mean "nothing
    # got worse". Saying "The test suite passes (1 error in 0.03s)" — which a
    # real run did — is worse than saying nothing, so the suite's own result
    # is what decides this sentence.
    if check.get("ok") and check.get("tests_ran") and not check.get("tests_passed"):
        still = check.get("failing_tests") or check.get("pre_existing_failures") or []
        return (
            f"The suite still has failures that predate this run and were left "
            f"alone ({len(still)}): {check.get('tests_summary', '')}."
        )
    if check.get("ok") and check.get("tests_ran"):
        return f"The test suite passes ({check.get('tests_summary', 'ok')})."
    if steps.get("failed") or steps.get("abandoned"):
        return "Some steps did not pass their own verification."
    if steps.get("total"):
        return "Everything the plan checked, passed."
    return ""


def _paragraph(goal: str, summary: dict) -> str:
    """Two or three sentences of plain prose, assembled from facts."""
    if summary.get("quick_answer"):
        return (
            f"Answered {' '.join(goal.split())[:100]!r} directly without planning — "
            "no files were changed."
        )

    changed = [str(f) for f in (summary.get("changed_files") or [])]
    phrases = [p for p in (_steps_phrase(summary.get("steps") or {}), _files_phrase(changed)) if p]
    # A plan-only run's steps were never executed: "3 steps ran, all verified. Everything the
    # plan checked, passed" beside "(no changes)" was a traced misreport (DESIGN-LOG §92.3).
    planned_only = summary.get("stopped_after") == "plan"
    if planned_only:
        total = (summary.get("steps") or {}).get("total") or 0
        phrases = [
            f"Planned {total} {_plural(total, 'step')} and stopped before running them, as asked"
        ]
    # Collapsed before slicing, not just stripped at the ends: a goal pasted
    # from a bordered terminal panel (a previous run's own failure summary,
    # say) carries embedded newlines and box-drawing characters, and a plain
    # goal.strip()[:160] left both in place — "Fix these issues:•   step
    # [x] failed...  │  │  •   verify still fails ." rendered as broken,
    # multi-line text where a one-line summary was expected. Collapsing
    # every run of whitespace (newlines included) to a single space turns
    # that into one flat line; it does not remove the • or │ characters
    # themselves, since those could just as easily be content the user
    # actually typed on purpose.
    trimmed = " ".join(goal.split())[:160].rstrip()
    opening = (
        f"Goal: {trimmed.rstrip('.')}."
        if trimmed.endswith(".") or not trimmed
        else f"Goal: {trimmed}" + ("" if trimmed[-1] in "?!:" else ".")
    )
    if not trimmed:
        opening = "Ran without a goal."
    body = f" {'; '.join(phrases)}." if phrases else " Nothing was executed and no files changed."
    verdict = "" if planned_only else _verdict(summary)
    return f"{opening}{body}{' ' + verdict if verdict else ''}".strip()


def _sections(summary: dict) -> list[tuple[str, str, list[str]]]:
    """(heading, colour, bullets) for each section that has anything in it."""
    return [
        (heading, colour, bullets)
        for heading, colour, bullets in (
            ("Found", "magenta", _found_bullets(summary)),
            ("Built", "green", _built_bullets(summary)),
            ("Failed", "yellow", _failed_bullets(summary)),
            ("Next", "cyan", _next_bullets(summary)),
        )
        if bullets
    ]


def render(goal: str, summary: dict) -> str:
    """The full end-of-run summary, as plain text with markup-free bullets.

    This is the version stored on the run summary and written to the trace,
    where Rich markup would be noise in a file meant to be read back by a
    program. `render_markup` is the same facts dressed for a terminal.
    """
    lines = [_paragraph(goal, summary)]
    for heading, _colour, bullets in _sections(summary):
        lines.append(f"\n{heading.lower()}:")
        lines.extend(f"  - {item}" for item in bullets)
    return "\n".join(lines)


def _bullet_table(colour: str, bullets: list[str]) -> Table:
    """Bullets whose wrapped lines line up under their own text.

    A plain "  • text" string re-wraps flush to the left margin, so the second
    line of a long step summary starts under the bullet instead of under the
    sentence, and a list of three long bullets reads as one paragraph. A
    borderless two-column table is Rich's way of getting a hanging indent.
    """
    table = Table(show_header=False, box=None, padding=(0, 0, 0, 2))
    table.add_column(width=2, style=colour, no_wrap=True)
    table.add_column(overflow="fold")
    for item in bullets:
        table.add_row("•", Text(item))
    return table


def render_rich(goal: str, summary: dict) -> Group:
    """The same summary dressed for a terminal: bold headings, blank lines
    between sections, and bullets that wrap under themselves.

    Content goes in as `Text`, never as markup, because step bullets begin
    with the step id in square brackets — "[list-files] Listed all files" —
    which Rich reads as a style tag: it swallows the id, and raises on render
    if the tag is not a style it knows.
    """
    blocks: list[RenderableType] = [Text(_paragraph(goal, summary))]
    for heading, colour, bullets in _sections(summary):
        blocks.append(Text(""))
        blocks.append(Text(heading, style=f"bold {colour}"))
        blocks.append(_bullet_table(colour, bullets))
    return Group(*blocks)
