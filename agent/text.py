"""Small text-shaping helpers shared across tools.

Truncation in particular was reimplemented three separate times across the
tool modules before this — shell.py, checks.py and vcs.py each invented their
own cap constant and elision wording. This is the one place it happens now.
The same had happened to line-numbering, which four separate call sites
(the execute preload, both review loaders, and read_file) each wrote out by
hand; `numbered` is now the single implementation.
"""

from __future__ import annotations

from typing import Literal

Keep = Literal["head", "tail", "both"]


def numbered(text: str, start: int = 1) -> str:
    """Prefix each line with its 1-indexed number and a tab.

    Every prompt that shows a model file contents uses this format, because a
    reviewer's finding and an edit_file anchor both need line numbers to refer
    to. `start` exists for windowed reads, where line 1 of the slice is not
    line 1 of the file.
    """
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(text.splitlines(), start))


def clip(text: str, limit: int) -> str:
    """A hard cut with a plain ellipsis, for a one-line status message.

    Unlike truncate(), this does not report how much it dropped — that report
    would make a narration line longer than the thing it is summarizing, which
    defeats the point of clipping it.
    """
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def truncate(text: str, limit: int, keep: Keep = "head") -> str:
    """Cut `text` to roughly `limit` characters, marking what was cut.

    `keep="head"` drops the tail — for output whose useful part comes first,
    or that could otherwise grow unboundedly.
    `keep="tail"` drops the head — for a traceback or a pytest summary, whose
    load-bearing line is the last one.
    `keep="both"` keeps a slice from each end — for general shell output,
    where either end might be the part that matters and dropping either
    outright risks losing it.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    if keep == "tail":
        return f"... [head elided] ...\n{text[-limit:]}"
    if keep == "both":
        half = max(1, limit // 2)
        head, tail = text[:half], text[-half:]
        return f"{head}\n... [{len(text) - 2 * half} chars elided] ...\n{tail}"
    return f"{text[:limit]}\n... [{len(text) - limit} chars elided]"
