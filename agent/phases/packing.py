"""Reading workspace files into a prompt, under a token budget.

The execute phase and the review phase both need the same first three steps —
resolve a workspace-relative path, read it, cap it, number its lines — and then
diverge only in how they pack the results: execute fits as much as it can into
one step's budget (smallest files first, so a step gets the most files it can),
while review groups everything into as few large chunks as the budget allows
(nothing is dropped; the reviewer's whole advantage is seeing it all).

Before this, each phase had its own copy of the shared part, with different
constants and different error handling for a missing file. Splitting it here
keeps the two packing strategies — which really are different — while leaving
one implementation of the part that was never meant to differ.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import approx_tokens
from ..text import numbered
from ..tools import ToolContext


@dataclass
class LoadedFile:
    rel: str
    # Line-numbered contents, already capped at the per-file limit.
    body: str
    tokens: int
    truncated: bool = False


def load_files(
    ctx: ToolContext, files: list[str], per_file_cap: int
) -> tuple[list[LoadedFile], list[str]]:
    """Read and number each named file. Returns (loaded, names that had nothing).

    A path that escapes the jail, does not exist, or is a directory lands in
    the second list rather than raising: a plan step naming a file it is about
    to create is normal, and the model is better served by being told so than
    by the phase failing.
    """
    loaded: list[LoadedFile] = []
    missing: list[str] = []
    for rel in files:
        text = ctx.try_read(rel)
        if text is None:
            missing.append(rel)
            continue
        truncated = len(text) > per_file_cap
        if truncated:
            text = text[:per_file_cap]
        body = numbered(text)
        loaded.append(
            LoadedFile(rel=rel, body=body, tokens=approx_tokens(body), truncated=truncated)
        )
    return loaded, missing


def fit_to_budget(
    loaded: list[LoadedFile], budget_tokens: int, smallest_first: bool = True
) -> tuple[list[LoadedFile], list[LoadedFile]]:
    """Take as many files as fit in one budget. Returns (included, skipped).

    Smallest first by default: given a fixed budget, more small files is
    usually more useful to a step than one large one, and the large one is
    exactly the case where read_file with an offset is the better tool anyway.
    """
    order = sorted(loaded, key=lambda f: f.tokens) if smallest_first else list(loaded)
    included: list[LoadedFile] = []
    skipped: list[LoadedFile] = []
    spent = 0
    for item in order:
        if spent + item.tokens > budget_tokens:
            skipped.append(item)
            continue
        included.append(item)
        spent += item.tokens
    return included, skipped


def batch_to_budget(loaded: list[LoadedFile], budget_tokens: int) -> list[list[LoadedFile]]:
    """Group every file into as few consecutive batches as the budget allows.

    Nothing is dropped — a file larger than the whole budget becomes its own
    batch rather than being skipped, since the caller (review) needs to see
    all of it eventually and a batch of one is still a valid chunk.
    """
    batches: list[list[LoadedFile]] = []
    current: list[LoadedFile] = []
    spent = 0
    for item in loaded:
        if current and spent + item.tokens > budget_tokens:
            batches.append(current)
            current, spent = [], 0
        current.append(item)
        spent += item.tokens
    if current:
        batches.append(current)
    return batches
