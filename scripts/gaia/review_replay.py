"""What did the review phase do to GAIA's final answers? (DESIGN-LOG §124.10)

A replay over recorded traces, with no model call. For each run it rebuilds `answer.txt` from the
trace's `write_file` and `edit_file` calls (an edit applies only when its `old_string` is in the
file, as the tool does), takes the answer as of the first `review_result` and again as of the
first final-check fix round (or the end), and scores both with GAIA's own scorer against the gold
answers. A run is skipped when a shell command or a multi-file edit wrote `answer.txt`, and the
rebuilt final answer must score the same as the recorded result before a run counts.

    uv run python scripts/gaia/review_replay.py ROUND_DIR RESULTS.jsonl

`ROUND_DIR` holds `traces/<task_id>/run-*.jsonl`; `RESULTS.jsonl` has `{"task_id", "right"}` rows.
It needs the GAIA validation data `gaia.py` loads, so it only runs where that data is present.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

SHELL_WRITE = re.compile(r">\s*\S*answer|write|open\(|tee|sed -i|mv |cp ")


def _load_gaia():
    spec = importlib.util.spec_from_file_location("gaia", Path(__file__).with_name("gaia.py"))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load scripts/gaia/gaia.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _writes_answer_some_other_way(tool: str | None, raw: str) -> bool:
    """A call that may change answer.txt where the replay cannot follow it."""
    if "answer.txt" not in raw:
        return False
    if tool in ("edit_files", "edit_blocks", "batch_edits_to_file"):
        return True
    return tool in ("run_bash", "run_python") and bool(SHELL_WRITE.search(raw))


def answers_at_review(events: list[dict]) -> tuple[bool, bool, str | None, str | None, str | None]:
    """(ambiguous, reviewed, before review, after the review's fixes, final) for one trace."""
    state: str | None = None
    ambiguous = reviewed = had_round = False
    before = after = None
    for event in events:
        kind = event["kind"]
        if kind == "tool_call":
            args = event.get("args") or {}
            tool = event.get("tool")
            raw = json.dumps(args)
            if tool == "write_file" and args.get("path") == "answer.txt":
                state = args.get("content")
            elif tool == "edit_file" and args.get("path") == "answer.txt":
                old = args.get("old_string")
                if state is not None and old is not None and old in state:
                    state = state.replace(old, args.get("new_string") or "", 1)
            elif _writes_answer_some_other_way(tool, raw):
                ambiguous = True
        elif kind == "review_result" and not reviewed:
            reviewed, before = True, state
        elif kind == "final_fix_round" and not had_round:
            had_round, after = True, state
    if reviewed and not had_round:
        after = state
    return ambiguous, reviewed, before, after, state


def main(round_dir: Path, results: Path) -> None:
    gaia = _load_gaia()
    gold = gaia.questions()
    right = {}
    for line in results.read_text().splitlines():
        row = json.loads(line)
        right[row["task_id"]] = row["right"]

    def correct(task: str, answer: str | None) -> bool:
        return answer is not None and bool(
            gaia.question_scorer(answer.strip(), gold[task]["answer"])
        )

    counts: Counter[str] = Counter()
    flips: list[tuple[str, str, str | None, str | None]] = []
    for trace_dir in sorted((round_dir / "traces").glob("*")):
        task = trace_dir.name
        files = sorted(trace_dir.glob("run-*.jsonl"))
        if not files or task not in gold or task not in right:
            continue
        events = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        ambiguous, reviewed, before, after, final = answers_at_review(events)
        counts["runs"] += 1
        if ambiguous:
            counts["skipped: a shell or multi-file edit wrote answer.txt"] += 1
            continue
        if correct(task, final) != right[task]:
            counts["skipped: rebuilt answer disagrees with the recorded result"] += 1
            continue
        if not reviewed:
            counts["not reviewed"] += 1
            continue
        counts["reviewed and scoreable"] += 1
        was, now = correct(task, before), correct(task, after)
        if was == now:
            counts["same score"] += 1
        elif now:
            counts["wrong -> right"] += 1
            flips.append((task, "wrong -> right", before, after))
        else:
            counts["right -> wrong"] += 1
            flips.append((task, "right -> wrong", before, after))
    for key, value in counts.items():
        print(f"{value:5d} {key}")
    for flip in flips:
        print("   ", flip)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
