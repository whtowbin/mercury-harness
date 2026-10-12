"""Underspecified-issue evaluation (Ambig-SWE style) on the django tasks swe.py runs.

  python3 amb.py hide ID [ID ...]   write goal-hidden-ID.txt: the issue with its specifics removed
  python3 amb.py show ID            print full and hidden side by side, for a hand check
  python3 amb.py answer ID          the --answer-command: a simulated issue author answers one
                                    question (JSON on stdin) from the full issue only

Arms (swe.py run, with SWE_GOAL and SWE_ASK):
  full    SWE_GOAL=goal                          no questions (the existing baseline)
  hidden  SWE_GOAL=goal-hidden                   no questions
  ask     SWE_GOAL=goal-hidden SWE_ASK=1         clarify on, answered by `amb.py answer ID`

Every simulated answer is appended to answers-ID.jsonl in the round directory, so a leak
(the simulated user handing over the fix) can be checked by hand.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

DATA = Path(
    os.environ.get("SWE_DATA", Path(__file__).resolve().parents[2] / "traces" / "swebench-data")
)
HIDE_MODEL = "anthropic/claude-sonnet-5.5"
USER_MODEL = "anthropic/claude-haiku-4.5"

HIDE_PROMPT = """Rewrite this GitHub issue the way a busy user would first report it, before anyone
asked them anything. Keep the symptom the user noticed and the area of the product it is in, in
everyday words. Remove what a maintainer would normally have to ask for:
- exact function, class, method, module and file names (describe the feature instead);
- the precise expected behaviour, values, formats, standards or RFC text;
- reproduction steps, code snippets, tracebacks and suggested fixes.
The report must still be about the same single problem and must not be misleading: someone who
read the original could answer any question about it. Two to four sentences, no title, no
headings. Reply with the rewritten report only.

Original issue:
{issue}"""

USER_PROMPT = """You filed the GitHub issue below. A developer working on it asks you one question.
Answer as that user, briefly (one to three sentences), using only what the issue says. You may
quote names, values or behaviour the issue states. Do not write code, do not propose a fix the
issue does not propose, and do not invent details. If the issue does not answer the question,
say "I don't know" and, if it helps, what you do know.

Your issue:
{issue}

The developer's question: {question}{options}"""


def key() -> str:
    return subprocess.run(
        ["security", "find-generic-password", "-a", "harness", "-s", "OpenRouter", "-w"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def complete(model: str, prompt: str, max_tokens: int) -> str:
    """One call, explicit budget, and a loud failure on an empty answer."""
    for attempt in range(3):
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key()}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "reasoning": {"effort": "low"},
            },
            timeout=90,
        )
        if resp.status_code == 200:
            text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
            raise RuntimeError(f"{model} returned an empty answer")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")


def full_issue(i: str) -> str:
    return (DATA / f"goal-{i}.txt").read_text()


def hide(ids: list[str]) -> None:
    for i in ids:
        out = DATA / f"goal-hidden-{i}.txt"
        if out.exists():
            continue
        out.write_text(complete(HIDE_MODEL, HIDE_PROMPT.format(issue=full_issue(i)), 600) + "\n")
        print(i, "hidden")


def show(i: str) -> None:
    print("FULL:\n" + full_issue(i) + "\nHIDDEN:\n" + (DATA / f"goal-hidden-{i}.txt").read_text())


def answer(i: str) -> None:
    q = json.load(sys.stdin)
    options = " (options offered: " + " | ".join(q["options"]) + ")" if q.get("options") else ""
    text = complete(
        USER_MODEL,
        USER_PROMPT.format(issue=full_issue(i), question=q["question"], options=options),
        300,
    )
    log = Path(os.environ.get("AMB_LOG_DIR", DATA)) / f"answers-{i}.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps({"question": q["question"], "answer": text}) + "\n")
    print(text)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"hide": lambda: hide(args), "show": lambda: show(args[0]), "answer": lambda: answer(args[0])}[
        cmd
    ]()
