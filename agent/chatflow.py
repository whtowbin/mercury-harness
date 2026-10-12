"""Two chat-workflow checks for runs with a person present (DESIGN-LOG §111).

- **Counter-questions.** A person answering clarify's questions sometimes asks one back (other
  options, which tests are still failing, whether to roll back the repo: 3 of 145 answers in 91
  app runs). They used to reach the planner as part of an answer
  and nothing made sure they were answered. Now the planner answers each in an `answer:` note of
  its draft, and the harness shows it before any work starts.
- **Assumptions.** A plan can rest on a reading the person never confirmed: one game run
  edited the beak while fixing the wing, and the next goal had to point that out. The planner
  lists the choices the goal leaves open in `assume:` notes, and the person confirms them with
  one keystroke or corrects them, which re-plans through the plan gate's feedback path.

No extra model call: both ride in the draft the planner already writes, and the instruction goes
at the end of the planner's first message (where Mercury weighs it, §87), never in a system
prompt or tool schema, so cached prefixes are unchanged. Only when someone can answer: clarify
and questions both on (every benchmark runs with clarify off, so their prompts are unchanged).
"""

from __future__ import annotations

import re

ANSWER_PREFIX = "answer:"
ASSUME_PREFIX = "assume:"
MAX_ASSUMPTIONS = 4
GO_AHEAD = "Go ahead"
_YES = frozenset({"", "y", "yes", "ok", "okay", "go", "go ahead", "proceed", "fine", "sure"})
_SENTENCE = re.compile(r"[^.!?\n]*\?")

PLANNER_CLOSING = (
    "\n\nIn `draft`, after the `understood:` note: if the person asked you a question in their "
    "answers above, answer it in a note starting `answer:` (one or two sentences, from what you "
    "can see in the workspace). Then list each choice this plan makes that the goal and the "
    "answers leave open (which element or file a word refers to, a value you picked, behaviour "
    f"you inferred) in a note starting `assume:`, at most {MAX_ASSUMPTIONS}. Write no `assume:` "
    "note when nothing is assumed: the person is asked to confirm each one."
)


def enabled(settings) -> bool:
    return bool(
        getattr(settings, "chat_checks", True)
        and getattr(settings, "clarify", False)
        and getattr(settings, "ask_enabled", False)
    )


def counter_questions(answer: str) -> list[str]:
    """The questions inside a person's answer, as sentences ending in `?`."""
    return [q.strip() for q in _SENTENCE.findall(answer or "") if len(q.strip()) > 3]


def notes(draft: list[str], prefix: str) -> list[str]:
    """Draft notes starting with `prefix`, the prefix removed."""
    found = []
    for note in draft:
        text = note.strip()
        if text.lower().startswith(prefix):
            body = text[len(prefix) :].strip()
            if body:
                found.append(body)
    return found


def show_answers(draft: list[str], trace) -> None:
    """The planner's answers to the person's own questions, each on its own line."""
    answers = notes(draft, ANSWER_PREFIX)
    for answer in answers:
        trace.say(f"\nyou asked; the agent's answer: {answer}")
    if answers:
        trace.event("counter_answered", answers=answers)


def assumptions_gate(interactor, trace):
    """An on_plan-shaped callback: "" to run the plan, "abort", or feedback to re-plan with.

    Asks only when the plan lists assumptions. Enter (the recommended option) accepts.
    """

    def gate(plan) -> str:
        assumed = notes(plan.draft, ASSUME_PREFIX)[:MAX_ASSUMPTIONS]
        if not assumed:
            return ""
        listed = "\n".join(f"  - {a}" for a in assumed)
        answer = interactor.confirm(
            f"Before it starts, the plan assumes:\n{listed}\nGo ahead, or say what to change.",
            [GO_AHEAD],
        )
        text = (answer or "").strip()
        if text.lower() in _YES or text == GO_AHEAD or text.startswith("(no answer"):
            trace.event("assumptions_confirmed", assumptions=assumed)
            return ""
        if text.lower() in ("q", "quit", "abort", "stop"):
            return "abort"
        trace.event("assumptions_corrected", assumptions=assumed, correction=text[:500])
        return f"The plan assumed:\n{listed}\nThe person corrected this: {text}"

    return gate
