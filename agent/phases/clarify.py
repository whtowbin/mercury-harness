"""Phase 0: ask about an underspecified goal before planning it.

The planner already has ask_user, but in practice it almost never fires. Two
reasons: PLANNER_SYSTEM tells it that most goals do not need clarification
(true, and the right default), and a fast model driving a tool loop toward
submit_plan has every incentive to keep going rather than stop and ask. The
result was an agent that would confidently build the wrong thing from a
three-word goal.

This is the same trick try_quick_answer uses, applied to a different
question. One cheap, tool-free completion whose only job is to decide "is
this specific enough to build from", with no competing objective to crowd it
out, run at the cheapest possible moment: before a plan exists, before any
code is written, when the cost of being wrong is one more question rather
than a whole execution phase.

Answers are folded into the planner's prompt as plain text. Nothing here is
binding on the planner — it is context, not instruction — which is why a
garbled or skipped answer degrades into "the planner knows slightly more"
rather than into a broken run.
"""

from __future__ import annotations

import contextlib
import re

from .. import chatflow, outline
from .. import skills as skills_module
from ..interact import CLARIFY_QUESTIONS
from ..prompts import (
    CLARIFY_CONTEXT_TAIL,
    CLARIFY_README_CHARS,
    CLARIFY_SYMBOLS_CHARS,
    CLARIFY_SYSTEM,
)
from ..tools import ToolContext
from ..trace import Trace

# Clarify's own budget, asked together (interact.CLARIFY_QUESTIONS). It is separate from the
# run's later questions since 2026-10-01: sharing one budget let clarify spend it all, and the
# platformer app's planner's question (move the off-screen flagpole, or add camera scrolling?) was
# refused (DESIGN-LOG §98, §101). Clarify asks 1.6-2.4 questions a round in real projects.
MAX_QUESTIONS = CLARIFY_QUESTIONS

# A goal that is purely an assent to something already proposed. Kept to
# unambiguous assent only: "yes, but use a dict" carries its own content and
# must reach the planner as written, so anything longer than a few words, or
# carrying a "but"/"except"/"instead", is not one of these.
_ASSENTS = frozenset(
    {
        "y",
        "yes",
        "yes please",
        "yep",
        "yeah",
        "yup",
        "ok",
        "okay",
        "k",
        "sure",
        "do it",
        "go",
        "go ahead",
        "go for it",
        "proceed",
        "continue",
        "please do",
        "sounds good",
        "lets do it",
        "let's do it",
        "do that",
        "yes do it",
        "yes lets do it",
        "yes let's do it",
        "yes please do",
        "confirmed",
        "agreed",
    }
)


def is_assent(goal: str) -> bool:
    """Whether `goal` says only "yes" to something proposed earlier.

    The point of the whitelist over a prefix match is that "yes" and "yes,
    but drop the cache" mean opposite things about whether the previous
    suggestion is still the plan, and only the first can be safely replaced
    by it.
    """
    cleaned = goal.strip().lower().strip(".!… ").replace(",", "")
    return cleaned in _ASSENTS


def resolve_assent(settings, ctx: ToolContext, trace: Trace, suggestion: str) -> str:
    """Turn a bare "yes" into the suggestion it is agreeing to, once confirmed.

    The suggestion is text a previous run's model wrote, so it is confirmed
    with the person rather than silently adopted as the goal — a stale or
    misremembered suggestion should cost one click, not a whole run spent
    building the wrong thing. Declining leaves the goal exactly as typed and
    the normal clarify path runs, so this can only ever save a round trip,
    never remove one.
    """
    if not suggestion:
        return ""
    trace.event("assent_detected", goal=settings.goal, suggestion=suggestion[:300])
    shown = suggestion if len(suggestion) <= 220 else suggestion[:217] + "..."
    answer = (
        (
            ctx.interactor.ask(
                f"Continue with this? {shown}", ["yes, do that", "no, I meant something else"]
            )
            or ""
        )
        .strip()
        .lower()
    )
    accepted = answer.startswith("y") or "do that" in answer
    trace.event("assent_resolved", accepted=accepted, answer=answer[:200])
    if not accepted:
        # Give the question back. `Interactor.ask` counts against the run's
        # three-question budget before checking it, and the normal clarify
        # path runs next — so a declined confirmation would otherwise have
        # spent one of the questions that path is about to ask, and its third
        # would be silently refused. This must only ever save a round trip.
        with contextlib.suppress(AttributeError):
            ctx.interactor.asked = max(0, ctx.interactor.asked - 1)
        return ""
    trace.say(f"continuing from the previous run's suggestion: {shown}")
    return (
        "\n\nThis goal is an assent to what the previous run proposed doing next, "
        f"which was:\n{suggestion}\n\n"
        "The person confirmed this is what they want. Treat it as the goal — it is "
        'what "yes" refers to — and plan it directly rather than asking what was '
        "meant."
    )


# Words a goal uses to point back at earlier work ("fix those issues", "try again", "that didn't
# work"). With a short goal they mean the last run; a long goal says what it wants itself.
_BACK_REFERENCE = re.compile(
    r"\b(those|these|that|this|it|them|again|same|previous|last|earlier|remaining|rest|still)\b",
    re.IGNORECASE,
)
FOLLOW_UP_MAX_WORDS = 12


def is_follow_up(goal: str) -> bool:
    """A short goal that points back at earlier work (DESIGN-LOG §92.3)."""
    return len(goal.split()) <= FOLLOW_UP_MAX_WORDS and bool(_BACK_REFERENCE.search(goal))


def resolve_follow_up(settings, ctx: ToolContext, trace: Trace) -> str:
    """Tie a follow-up goal to the run it follows, once the person confirms.

    Short follow-ups to the platformer app ("fix those", "try again") were asked which issues and
    what to try, although the newest memory section said exactly what was left unresolved, and
    showing it to the clarify call, even last, did not stop the question (§92.2). This does what
    `resolve_assent` does for a bare "yes": the newest remembered run is offered for one click,
    and on yes the goal carries it and clarify asks nothing. Declining gives the question back
    and the normal path runs, so this can only save a round trip, never remove one.
    """
    memory = getattr(ctx, "memory", None)
    sections = memory.sections(include_scratch=False) if memory is not None else []
    if not sections:
        return ""
    last = max(sections, key=lambda s: s.index)
    title = re.sub(r"^\[[^\]]*\]\s*", "", last.title)
    unresolved = [
        line.strip()[2:] for line in last.body.splitlines() if line.strip().startswith("- ")
    ]
    trace.event("follow_up_detected", goal=settings.goal, previous=title[:200])
    shown = title if len(title) <= 120 else title[:117] + "..."
    open_items = "; ".join(unresolved[-3:])[:220]
    question = f'Continue from the last run ("{shown}")?' + (
        f" It ended with: {open_items}" if open_items else ""
    )
    answer = (
        (ctx.interactor.ask(question, ["yes, continue from it", "no, something else"]) or "")
        .strip()
        .lower()
    )
    accepted = answer.startswith("y") or "continue" in answer
    trace.event("follow_up_resolved", accepted=accepted, answer=answer[:200])
    if not accepted:
        with contextlib.suppress(AttributeError):
            ctx.interactor.asked = max(0, ctx.interactor.asked - 1)
        return ""
    trace.say(f"continuing from the last run: {shown}")
    return (
        f"\n\nThis goal follows up the previous run on this workspace, whose goal was "
        f'"{title}". Its record:\n{last.body.strip()}\n\n'
        "The person confirmed the goal refers to that run. Read what it left unresolved or "
        'failed as what "those", "it" or "again" mean, and plan that directly rather than '
        "asking what was meant."
    )


def looks_truncated(question: str) -> bool:
    """Whether a question was cut off rather than finished.

    A reply clipped by the token cap ends mid-sentence, and the fragment is
    unanswerable: one traced run asked "When it comes to" and the person
    replied, reasonably, that the rest of the question was missing.
    Raising the cap is the actual fix; this is the guard for the next time
    something clips a reply, because showing a fragment is strictly worse than
    asking one question fewer.
    """
    stripped = question.strip()
    if not stripped:
        return True
    return not stripped.endswith(("?", ".", ")", ":", "!"))


def _split_options(question: str) -> tuple[str, list[str]]:
    """Separate `question | option | option` into its parts."""
    if "|" not in question:
        return question.strip(), []
    head, *rest = question.split("|")
    options = [part.strip() for part in rest if part.strip()]
    return head.strip(), options[:6]


def _parse(text: str) -> list[tuple[str, list[str]]]:
    """The "Q: ..." lines as (question, options), or nothing if it said CLEAR."""
    stripped = text.strip()
    if not stripped or stripped.upper().startswith("CLEAR"):
        return []
    parsed: list[tuple[str, list[str]]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line.upper().startswith("Q:"):
            continue
        question, options = _split_options(line[2:].strip())
        if not question or looks_truncated(question):
            continue
        parsed.append((question, options))
    return parsed[:MAX_QUESTIONS]


def _wants_context(settings) -> bool:
    return bool(getattr(settings, "clarify_context", False))


def _clarify_message(settings, ctx: ToolContext, remembered: str) -> str:
    """The clarify call's user message. Without clarify_context: goal, memory, file map, as
    before. With it, the record of earlier runs moves to just before the closing rule: in
    short follow-ups to the platformer app ("fix those", "try again") the newest memory
    section listed exactly what was unresolved, recall always includes it, and clarify still
    asked which issues were meant (DESIGN-LOG §92.1)."""
    if not _wants_context(settings):
        return f"Goal:\n{settings.goal}\n{remembered}\nWorkspace map:\n{ctx.workspace.tree()}"
    earlier = (
        f"\n\nWhat earlier runs on this workspace did and left unresolved:\n{remembered.strip()}"
        if remembered.strip()
        else ""
    )
    return (
        f"Goal:\n{settings.goal}\n\nWorkspace map:\n{ctx.workspace.tree()}"
        + _code_context(settings, ctx)
        + earlier
        + CLARIFY_CONTEXT_TAIL
    )


def _code_context(settings, ctx: ToolContext) -> str:
    """What the code already says, for the clarify call (Settings.clarify_context): the
    top-level symbols per file, the README's opening, and how tests run. No model call: the
    workspace is read directly."""
    root = ctx.workspace.root
    parts: list[str] = []
    symbols = outline.symbol_map(root, char_cap=CLARIFY_SYMBOLS_CHARS)
    if symbols:
        parts.append(f"What is defined where (top-level symbols):\n{symbols}")
    readme = next(
        (p for p in (root / "README.md", root / "README.rst", root / "README") if p.is_file()), None
    )
    if readme is not None:
        text = readme.read_text(encoding="utf-8", errors="ignore")[:CLARIFY_README_CHARS]
        parts.append(f"{readme.name} (opening):\n{text}")
    test_command = getattr(settings, "test_command", "") or (
        "pytest"
        if any((root / d).is_dir() for d in ("tests", "test")) or any(root.glob("test_*.py"))
        else ""
    )
    if test_command:
        parts.append(f"Tests run with: {test_command}")
    return "\n\n" + "\n\n".join(parts) if parts else ""


def gather_clarifications(
    client, settings, ctx: ToolContext, trace: Trace, remembered: str = "", suggestion: str = ""
) -> str:
    """Ask about an ambiguous goal. Returns a block for the planner's prompt.

    `remembered` is what the workspace recalls of earlier runs, passed in so
    the triage call can resolve a follow-up goal ("why did that test fail?")
    from the record instead of asking a question whose answer only the
    previous run knew.

    Returns "" when the goal is already clear, when nobody is available to
    answer, or on any failure — this must never be the reason a run that would
    otherwise have proceeded does not, so every error path degrades to "ask
    nothing and plan as before".
    """
    if not settings.clarify or not settings.ask_enabled:
        return ""
    # Before spending a triage call on "yes": the previous run already wrote
    # down what it was proposing, and one confirmation resolves the whole
    # question that call exists to ask.
    if is_assent(settings.goal) and suggestion:
        resolved = resolve_assent(settings, ctx, trace, suggestion)
        if resolved:
            return resolved
    if getattr(settings, "resolve_followups", False) and is_follow_up(settings.goal):
        resolved = resolve_follow_up(settings, ctx, trace)
        if resolved:
            return resolved
    try:
        # This call had no skills at all, which is why the grilling guidance
        # had nowhere to land: clarify is the one phase whose entire job is
        # asking well, and it was the only phase running without reference.
        guidance = skills_module.render(
            skills_module.select(ctx.skills, "clarify", settings.goal, [], ctx.skill_token_budget)
        )
        reply = client.chat(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": CLARIFY_SYSTEM + guidance},
                {
                    "role": "user",
                    "content": _clarify_message(settings, ctx, remembered),
                },
            ],
            # Three questions, each allowed to carry its own answer options,
            # plus whatever preamble the model emits before the Q: lines. At
            # 300 this call was truncated by the cap on every traced run —
            # completion_tokens came back 294, 292, 298 — and the casualty was
            # always the last question, delivered to the user as a fragment
            # ("Are there specific platforms", "When it comes to"). It read as
            # a display bug in the TUI and was nothing of the kind.
            temperature=0.2,
            max_tokens=900,
            reasoning_effort="low",
        )
        trace.llm_call(
            "clarify_check", settings.planner_model, reply.usage, reply.finish_reason, reply=reply
        )
        questions = _parse(reply.content)
    except Exception as exc:
        trace.event("clarify_failed", error=f"{type(exc).__name__}: {exc}")
        return ""

    if not questions:
        trace.event("clarify", questions=[], outcome="clear")
        return ""

    trace.event("clarify", questions=[q for q, _ in questions], options=[o for _, o in questions])
    trace.say(
        f"\n=== phase 0: {len(questions)} clarifying question(s) before planning ===", detail=True
    )
    pairs: list[tuple[str, str]] = []
    ctx.cancel.check()
    # Asked together and answered in one go (DESIGN-LOG §101). Options reach the TUI as
    # buttons; the plain CLI numbers them. Either way the person can type something else.
    answers = ctx.interactor.ask_many(
        [(question, options or None) for question, options in questions]
    )
    for (question, _options), raw in zip(questions, answers, strict=True):
        answer = (raw or "").strip()
        trace.event("clarify_answered", question=question, answer=answer[:300])
        # The interactor's no-answer fallback is prose telling the *model* to
        # use its own judgment. Carrying that into the planner's prompt as
        # though a human had said it is worse than carrying nothing, so an
        # unanswered question is dropped entirely.
        if answer and not answer.startswith("(no answer"):
            pairs.append((question, answer))

    if not pairs:
        return ""
    lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
    # A question asked back (DESIGN-LOG §111): named, so the planner's closing note on
    # `answer:` has something to point at.
    asked = [q for _, a in pairs for q in chatflow.counter_questions(a)]
    if asked and chatflow.enabled(settings):
        trace.event("counter_questions", questions=asked)
        lines += "\n\nIn their answers they also asked you: " + " ".join(asked)
    # "Part of the goal" was not strong enough. A traced run asked whether
    # lamps should align with the wall gap or appear near it; the person
    # answered that lamps should stay away from the gaps; and the
    # plan came back as "align lamps vertically with wall gaps" — the
    # opposite. The goal itself had said lamps "can appear inside or
    # next to the gap", so goal and answer genuinely contradicted, and nothing
    # said which one wins. The answer wins: it was written second, in front of
    # the workspace, in response to being asked precisely this.
    return (
        "\n\nThe person who set this goal was asked about it first, and said:\n"
        f"{lines}\n\n"
        "These answers are authoritative. Where an answer contradicts the goal, or "
        "contradicts the options the question offered, the answer is what they want and "
        "the goal's wording is out of date — they wrote the goal first and the answer "
        "second, having been asked exactly this. Read each answer against its own "
        'question: a short answer like "neither" or "the first one" only means '
        "something in the light of what was asked, and an answer that rejects both "
        "options rejects them both."
    )
