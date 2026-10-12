"""Terminal tools: how a phase tells the harness it is finished."""

from __future__ import annotations

import copy
import dataclasses

from .base import Tool, ToolContext, obj


def finish_step(
    ctx: ToolContext,
    summary: str,
    verified: bool = False,
    notes: str = "",
    expected_result: str = "",
) -> str:
    ctx.result = {
        "summary": summary,
        "verified": verified,
        "notes": notes,
        "expected_result": expected_result,
    }
    ctx.done = True
    return "step recorded"


def submit_plan(
    ctx: ToolContext,
    draft: list,
    steps: list,
    task_kind: str = "",
    reproduce: str = "",
    symptom_call: str = "",
    symptom_expected: str = "",
    findings: list | None = None,
) -> str:
    from ..symptom import snippet_from

    result: dict[str, object] = {"draft": draft, "steps": steps}
    if task_kind:
        result["task_kind"] = task_kind
    if findings:
        result["findings"] = findings
    # Settings.repro_form: `none: why` and the symptom fields both end up as a plain
    # snippet (or none), so the plan and everything downstream see one shape.
    snippet = snippet_from(
        {"reproduce": reproduce}
        if not (symptom_call or symptom_expected)
        else {"symptom_call": symptom_call, "symptom_expected": symptom_expected}
    )
    if snippet:
        result["reproduce"] = snippet
    ctx.result = result
    ctx.done = True
    return f"plan accepted: {len(steps)} steps"


def submit_findings(ctx: ToolContext, verdict: str, findings: list, summary: str = "") -> str:
    ctx.result = {"verdict": verdict, "findings": findings, "summary": summary}
    ctx.done = True
    return f"review recorded: {len(findings)} finding(s)"


FINISH_STEP = Tool(
    "finish_step",
    "Call this once the current step is fully implemented. Do not call it while work remains.",
    obj(
        {
            "summary": {
                "type": "string",
                "description": "What you changed, in one or two sentences, written for "
                "someone who did not watch you work. State what is now true "
                "of the code — not what you were about to try, not what went "
                "wrong on an earlier attempt, and not your reasoning. This "
                "text is shown to the user as the record of this step.",
            },
            "verified": {
                "type": "boolean",
                "description": "True only if you ran something that confirms it works — a "
                "test, the step's verify command, run_python. Send the "
                'literal true or false, not the strings "true"/"false".',
            },
            "expected_result": {
                "type": "string",
                "description": "State concretely what you expect the step's verify "
                "command (or, if it has none, whatever you ran to "
                'convince yourself) to show — "4 passed", '
                '"prints total 42.0", "exits non-zero, the '
                'file is still missing". Required whenever '
                "verified is true. It is recorded with the step "
                "and the harness compares it with the verify "
                "command's outcome; a mismatch is logged for "
                "whoever reads the trace, it does not fail the "
                "step. Writing it down is still worth doing: "
                "a result you had to predict is harder to skim "
                "past and call a pass.",
            },
            "notes": {
                "type": "string",
                "description": "Anything the reviewer or the next step must know.",
            },
        },
        ["summary"],
    ),
    finish_step,
    terminal=True,
)

_PROSE_DRAFT_DESCRIPTION = (
    "Your reasoning toward the plan, one paragraph per item, full sentences. The first item "
    "is the `understood:` restatement of the request; the rest cover what you read, what "
    "it shows, what could go wrong, and what is still unknown. You may sketch pseudocode or "
    "signatures here to work out a design; the draft is never shown to the executor, so "
    "anything a step must do belongs in that step's own fields."
)


def submit_plan_tool(draft_style: str = "terse") -> Tool:
    """`submit_plan` with the draft described for the style in use.

    The schema text and the planner prompt have to agree, or the model obeys one
    and violates the other (the terse schema once said "five words or fewer"
    against a prompt asking for a restating first note). So the prose style
    changes both together.
    """
    if draft_style != "prose":
        return SUBMIT_PLAN
    parameters = copy.deepcopy(SUBMIT_PLAN.parameters)
    parameters["properties"]["draft"]["description"] = _PROSE_DRAFT_DESCRIPTION
    return dataclasses.replace(SUBMIT_PLAN, parameters=parameters)


# What kind of work a goal is, as the planner reads it. The routing signal a
# separate triage call would have produced, taken from the plan call that
# happens anyway (DESIGN-LOG §53); a triage call is ~15s of Mercury TTFT.
TASK_KINDS = ("bugfix", "feature", "refactor", "tests", "docs", "other")


def with_task_kind(tool: Tool) -> Tool:
    """`submit_plan` with an optional `task_kind` enum added to its schema."""
    parameters = copy.deepcopy(tool.parameters)
    parameters["properties"]["task_kind"] = {
        "type": "string",
        "enum": list(TASK_KINDS),
        "description": "What kind of work this goal is: bugfix (something reported "
        "broken), feature (new behaviour), refactor (same behaviour, new shape), tests "
        "(only tests change), docs, or other.",
    }
    return dataclasses.replace(tool, parameters=parameters)


def with_reproduce(tool: Tool) -> Tool:
    """`submit_plan` with a required `reproduce` snippet (plan_probe="repro").

    Required so that leaving it out is a decision, not an oversight; an empty
    string is how a goal that asks for new work says there is nothing to
    reproduce. runner.ReproGate runs it before the plan is accepted.
    """
    parameters = copy.deepcopy(tool.parameters)
    parameters["properties"]["reproduce"] = {
        "type": "string",
        "description": "When the goal reports that something is broken: a short Python "
        "snippet asserting the behaviour the report expects, through the entry point it "
        "names. It must FAIL while the problem exists. Run against a copy of the workspace "
        "before the plan is accepted. Empty string when the goal asks for new work.",
    }
    parameters["required"] = [*parameters.get("required", []), "reproduce"]
    return dataclasses.replace(tool, parameters=parameters)


SUBMIT_PLAN = Tool(
    "submit_plan",
    "Submit the finished plan. Call exactly once, after your chain-of-draft reasoning.",
    obj(
        {
            "draft": {
                "type": "array",
                "description": (
                    "Your notes toward the plan. The first is the `understood:` "
                    "restatement of the request, a sentence or two covering every "
                    "part of it; the rest are brief working notes, one thought "
                    "each, that a person could read at a glance."
                ),
                "items": {"type": "string"},
            },
            "steps": {
                "type": "array",
                "description": "Ordered, independently executable steps.",
                "items": obj(
                    {
                        "id": {"type": "string", "description": "Short slug, e.g. 'add-parser'."},
                        "title": {"type": "string"},
                        "intent": {
                            "type": "string",
                            "description": "What to build and why, 1-3 sentences. Concrete.",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Workspace-relative files this step reads or writes. "
                            "Required: these are preloaded into the executor's "
                            "prompt, and a step that names none is treated as "
                            "touching everything, which forces it to run alone.",
                        },
                        "writes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The subset of `files` this step actually modifies. "
                            "Everything else in `files` it only reads. Steps that "
                            "write different files can run at the same time, so "
                            "being accurate here is most of the wall-clock saving "
                            "available — four audit steps that all read game.py "
                            "and each write their own test file all run together, "
                            "while four that all write game.py cannot. This is "
                            "binding: while running beside another step, writing "
                            "anything outside it is refused. Omit only when the "
                            "step really does modify everything it names.",
                        },
                        "acceptance": {
                            "type": "string",
                            "description": "Observable condition that means this step is done. "
                            'Name concrete values where you can — "items draw '
                            'at y = 160, 200, 240, ..." beats "items draw at '
                            'distinct positions".',
                        },
                        "signatures": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Functions, methods or attributes this step adds or "
                            "changes, one per entry, as a signature: "
                            '"Game.show_preferences(self, screen)", '
                            '"Bird.wing_frame: int". You have read the code '
                            "and the executor has not; naming the shape here "
                            "saves it rediscovering what you already know. "
                            "Signatures only — not bodies, which you cannot "
                            "run and it would follow anyway.",
                        },
                        "tests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tests this step should produce, one per entry, as a name "  # noqa: E501
                            'and what it asserts: "test_prefs_draw_at_distinct_y: '
                            'each of the 8 items has its own y, 160 through 440". '
                            "A test is the specification the executor can actually "
                            "run, so this is where a vague step becomes a checkable "
                            "one. Assert observable behaviour through the real entry "
                            "point, never a value recomputed the way the code "
                            "computes it.",
                        },
                        "verify": {
                            "type": "string",
                            "description": "A shell command that checks it, or '' if none applies.",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ids of steps that must finish before this one. List only "  # noqa: E501
                            "real dependencies: steps with none, touching different "
                            "files, may be executed at the same time.",
                        },
                    },
                    ["id", "title", "intent", "acceptance", "files"],
                ),
            },
        },
        ["draft", "steps"],
    ),
    submit_plan,
    terminal=True,
)

SUBMIT_FINDINGS = Tool(
    "submit_findings",
    "Submit the review. Call exactly once. Report only defects you can point at in the code.",
    obj(
        {
            "verdict": {
                "type": "string",
                "enum": ["pass", "needs_changes"],
                "description": '"needs_changes" if you found any blocker or major '
                'finding, "pass" otherwise. It must agree with the '
                "findings you list.",
            },
            "summary": {"type": "string", "description": "One paragraph on the state of the work."},
            "findings": {
                "type": "array",
                "description": "Every defect you can point at in the code. Empty is a valid and "
                "useful answer; do not invent findings to look thorough.",
                "items": obj(
                    {
                        "file": {"type": "string"},
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Every other file involved in this defect — the call "
                            "sites of a changed signature, the module that "
                            "disagrees. The executor applying the fix is shown "
                            "these too, so a cross-file defect arrives with both "
                            "halves of itself.",
                        },
                        "line": {"type": "integer"},
                        "severity": {"type": "string", "enum": ["blocker", "major", "minor"]},
                        "issue": {
                            "type": "string",
                            "description": "The defect, stated in one sentence.",
                        },
                        "failure": {
                            "type": "string",
                            "description": "Concrete inputs or state that produce the wrong result.",  # noqa: E501
                        },
                        "fix": {"type": "string", "description": "The change that resolves it."},
                    },
                    ["file", "severity", "issue", "fix"],
                ),
            },
        },
        ["verdict", "findings"],
    ),
    submit_findings,
    terminal=True,
)
