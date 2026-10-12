"""Letting the agent write to and read from its own memory.

The harness writes the durable record itself at the end of a run (see
agent/memory.py::record_run) — that part must not depend on a model choosing
to call a tool. These two exist for the things only the model knows while it
is working: a hypothesis it wants the next step to test, a dead end worth not
repeating, something it noticed in a file it was not sent to change.

`remember` writes to the session scratchpad, never to durable memory. That is
deliberate: a model's in-flight notes are frequently wrong, and a wrong note
that persists across sessions is worse than no note. What survives a session
is the harness's own account of what actually happened.
"""

from __future__ import annotations

from .. import skills as skills_module
from .base import Tool, ToolContext, obj

MAX_NOTE_CHARS = 2_000


def remember(ctx: ToolContext, note: str, keywords: list | None = None, title: str = "note") -> str:
    if not note.strip():
        return "error: note must not be empty"
    if ctx.memory is None:
        return "error: memory is not available for this run"
    cleaned = [str(k) for k in (keywords or [])][:8]
    ctx.memory.note(note[:MAX_NOTE_CHARS], cleaned, title=str(title)[:80])
    ctx.trace.event("remember", title=title, keywords=cleaned, chars=len(note))
    return "noted in this session's scratchpad"


def recall(ctx: ToolContext, query: str, limit: int = 4) -> str:
    if not query.strip():
        return "error: query must not be empty"
    if ctx.memory is None:
        return "error: memory is not available for this run"
    hits = ctx.memory.search(
        query, limit=max(1, min(int(limit or 4), 8)), judge=ctx.relevance_judge
    )
    ctx.trace.event("recall", query=query[:200], hits=len(hits))
    if not hits:
        return (
            f"nothing in memory matches {query!r}. This workspace may not have "
            "been worked on before, or the earlier work used different words."
        )
    return "\n\n".join(section.render() for section in hits)


NOTE_TOOLS = [
    Tool(
        "remember",
        "Leave a note for a later step in this run, or for whoever reads the session "
        "scratchpad. Use it for something you learned that the step you are in does not "
        "act on: a dead end not worth repeating, a hypothesis for a later step, a defect "
        "you noticed in a file you were not asked to change. Not a substitute for "
        "finish_step, and not durable — the harness records what actually happened at "
        "the end of the run on its own.",
        obj(
            {
                "note": {
                    "type": "string",
                    "description": "What to remember, in a sentence or two.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 8 words that a later search should find "
                    "this by — filenames, symbols, concepts.",
                },
                "title": {"type": "string", "description": "A short label for the note."},
            },
            ["note"],
        ),
        remember,
    ),
    Tool(
        "recall",
        "Search what this workspace remembers from earlier runs and earlier steps of "
        "this one: what was built, what was left broken, what was already tried. Worth a "
        'call when a goal refers to something you cannot see — "the test that failed", '
        '"the refactor we started" — before guessing at what it means.',
        obj(
            {
                "query": {
                    "type": "string",
                    "description": "Words to match: a filename, a symbol, a topic.",
                },
                "limit": {"type": "integer", "description": "Most matches to return, max 8."},
            },
            ["query"],
        ),
        recall,
    ),
]


def read_skill(ctx: ToolContext, name: str) -> str:
    """Fetch one skill's guidance by name."""
    body = skills_module.body_of(ctx.skills, name)
    if body is None:
        available = ", ".join(sorted(s.name for s in ctx.skills)) or "(none)"
        return f"error: no skill called {name!r}. Available: {available}"
    ctx.trace.event("skill_read", name=name, chars=len(body))
    return body


READ_SKILL = Tool(
    "read_skill",
    "Read one of the reference skills listed in your system prompt. The ones most likely "
    "to apply are already included there; call this for a skill that turned out to matter "
    "and was not — the guidance is worth one round trip when you know you want it.",
    obj(
        {
            "name": {
                "type": "string",
                "description": "Exact skill name, as listed in the reference section.",
            }
        },
        ["name"],
    ),
    read_skill,
)
