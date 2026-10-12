"""Asking the human a clarifying question, for the phases where guessing is risky."""

from __future__ import annotations

from .base import Tool, ToolContext, obj

MAX_OPTIONS = 6


def ask_user(
    ctx: ToolContext, question: str, options: list | None = None, recommended: str = ""
) -> str:
    """Pause and ask. Use sparingly — only when a wrong guess is expensive to
    undo and no amount of reading the workspace would resolve it."""
    if not question.strip():
        return "error: question must not be empty"
    cleaned = [str(o) for o in options][:MAX_OPTIONS] if options else None
    # Only honoured when it actually names one of the options: a
    # recommendation pointing at something the person cannot pick is worse
    # than none, and silently inventing one they did not make is worse still.
    pick = str(recommended).strip()
    if pick and (not cleaned or pick not in cleaned):
        pick = ""
    ctx.trace.event("ask_user", question=question[:300], options=cleaned, recommended=pick)
    answer = ctx.interactor.ask(question.strip(), cleaned, recommended=pick or None)
    ctx.trace.event("ask_user_answered", answer=answer[:300])
    return answer


ASK_TOOLS = [
    Tool(
        "ask_user",
        "Ask the human a clarifying question when you are genuinely unsure and a wrong "
        "guess would be expensive to undo — an ambiguous requirement, a real design choice "
        "the goal doesn't settle, or a spec that doesn't say what should happen on some "
        "edge case. Not for things you could resolve by reading the workspace, and not for "
        "routine implementation decisions: those are yours to make. When no one is "
        "available to answer, you get told to proceed on your own judgment instead of "
        "hanging — that is the normal case for an unattended run, not an error.",
        obj(
            {
                "question": {
                    "type": "string",
                    "description": "One specific question. Say what you will do with each answer.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 6 concrete choices, if the question is "
                    "that shape. Omit for an open-ended question.",
                },
                "recommended": {
                    "type": "string",
                    "description": "Which of `options` you would pick, and always give "
                    "one when you offer options. It is shown as the "
                    "recommended choice, so the person can agree with a "
                    "click instead of composing an answer; a question "
                    "with no recommendation asks them to do the deciding "
                    "from scratch. Say what you would do — they can "
                    "still say no.",
                },
            },
            ["question"],
        ),
        ask_user,
    ),
]
