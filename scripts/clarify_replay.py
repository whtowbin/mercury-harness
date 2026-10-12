"""Replay recorded clarify rounds with a setting off and on, and judge each question.

DESIGN-LOG §92. Every trace whose `clarify` event asked something is a recorded round: its goal
and its workspace (as it stands now; both arms see the same state). The clarify call is replayed
through `gather_clarifications` itself, with nobody answering, and a judge model labels each
question CODE (the workspace answers it) or USER (only the requester can decide).

Usage:
    uv run python scripts/clarify_replay.py replay OUT.jsonl [--samples 2]
    uv run python scripts/clarify_replay.py judge OUT.jsonl LABELS.jsonl
    uv run python scripts/clarify_replay.py report OUT.jsonl LABELS.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent import outline  # noqa: E402
from agent.approval import Approver  # noqa: E402
from agent.config import Settings  # noqa: E402
from agent.interact import Interactor  # noqa: E402
from agent.memory import MemoryStore  # noqa: E402
from agent.phases.clarify import gather_clarifications  # noqa: E402
from agent.providers import build_client  # noqa: E402
from agent.tools import ToolContext  # noqa: E402
from agent.trace import Trace  # noqa: E402
from agent.workspace import Workspace  # noqa: E402

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_SYSTEM = """You judge one clarifying question a coding agent asked before starting work.
You see the request, the workspace's files, its top-level symbols and its README.

Answer CODE if the workspace itself answers the question: where something lives, what the current
behaviour is, which test runner or framework is used, what a function or class does, whether a
file exists, which version is installed. Answer HISTORY if the record of earlier runs shown
answers it (what "those issues" or "try again" refers to, what the last run left unresolved).
Answer USER if only the person who made the request can answer it: what they want, which of two
designs, a value or format the request leaves open.

Reply with exactly one line: CODE, HISTORY or USER, then " - " and a reason of at most 15 words."""


_SECTION_TIME = re.compile(r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", re.M)


def memory_at(workspace: str, goal: str, started: float) -> str:
    """What a run starting at `started` recalled: the workspace's memory sections written
    before it (each is stamped when its run ended), selected the way a run selects them."""
    path = Path(workspace) / ".harness" / "memory.md"
    if not path.is_file():
        return ""
    text = path.read_text(errors="ignore")
    cut = time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
    starts = [m for m in _SECTION_TIME.finditer(text)]
    kept = [
        text[m.start() : (starts[i + 1].start() if i + 1 < len(starts) else len(text))]
        for i, m in enumerate(starts)
        if m.group(1) <= cut
    ]
    if not kept:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / ".harness").mkdir()
        (Path(tmp) / ".harness" / "memory.md").write_text("".join(kept))
        return MemoryStore(tmp).relevant(
            goal, budget_tokens=Settings(workspace=tmp).memory_token_budget
        )


ROOTS: list[Path] = [REPO / "traces"]


def rounds() -> list[dict]:
    """Every recorded clarify round that asked questions, one per trace, workspace present."""
    out: list[dict] = []
    seen: set[str] = set()
    paths = sorted(p for root in ROOTS for p in root.rglob("run-*.jsonl"))
    for path in paths:
        # The archive keeps failed-chats copies beside the originals: one round per trace name.
        if path.name in seen:
            continue
        seen.add(path.name)
        try:
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        asked = [e for e in events if e.get("kind") == "clarify" and e.get("questions")]
        start = next((e for e in events if e.get("kind") == "run_start"), None)
        if not asked or not start or not Path(start.get("workspace", "")).is_dir():
            continue
        out.append(
            {
                "trace": str(path),
                "workspace": start["workspace"],
                "goal": start.get("goal", ""),
                "recorded": asked[0]["questions"],
                "remembered": memory_at(start["workspace"], start.get("goal", ""), start["ts"]),
            }
        )
    return out


def replay_one(round_: dict, context: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            workspace=round_["workspace"],
            goal=round_["goal"],
            clarify=True,
            ask_enabled=True,
            clarify_context=context,
            trace_dir=tmp,
            quiet=True,
        )
        trace = Trace(tmp, echo=False)
        ctx = ToolContext(
            workspace=Workspace(round_["workspace"]),
            approver=Approver(auto_approve=True),
            trace=trace,
            interactor=Interactor(ask=lambda q, o=None: "", enabled=True),
        )
        client = build_client(settings, stream=False)
        try:
            gather_clarifications(client, settings, ctx, trace, remembered=round_["remembered"])
        finally:
            client.close()
            trace.close()
        events = [json.loads(line) for line in trace.path.read_text().splitlines() if line.strip()]
    clar = next((e for e in events if e.get("kind") == "clarify"), {})
    call = next((e for e in events if e.get("kind") == "llm_call"), {})
    return {
        "trace": round_["trace"],
        "workspace": round_["workspace"],
        "had_memory": bool(round_["remembered"]),
        "context": context,
        "questions": clar.get("questions") or [],
        "clear": clar.get("outcome") == "clear",
        "failed": any(e.get("kind") == "clarify_failed" for e in events),
        "prompt_tokens": call.get("prompt_tokens", 0),
    }


def replay(out: Path, samples: int) -> None:
    jobs = [(r, c) for r in rounds() for c in (False, True) for _ in range(samples)]
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda job: replay_one(*job), jobs))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} replays of {len(jobs) // (2 * samples)} rounds -> {out}")


def _workspace_view(workspace: str, remembered: str = "") -> str:
    root = Path(workspace)
    readme = next((p for p in (root / "README.md", root / "README.rst") if p.is_file()), None)
    text = readme.read_text(errors="ignore")[:1_500] if readme else "(no README)"
    return (
        f"Files:\n{Workspace(workspace).tree()}\n\n"
        f"Symbols:\n{outline.symbol_map(root, char_cap=6_000)}\n\nREADME:\n{text}\n\n"
        f"Record of earlier runs:\n{remembered[:4_000] or '(none)'}"
    )


def judge(replays: Path, labels: Path) -> None:
    rows = [json.loads(line) for line in replays.read_text().splitlines() if line.strip()]
    goals = {r["trace"]: r for r in rounds()}
    pairs = sorted({(r["trace"], q) for r in rows for q in r["questions"]})
    views = {
        t: _workspace_view(goals[t]["workspace"], goals[t]["remembered"])
        for t in {t for t, _ in pairs}
    }
    client = build_client(Settings(workspace=str(REPO)), stream=False)

    def label(pair: tuple[str, str]) -> dict:
        trace, question = pair
        reply = client.chat(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Request:\n{goals[trace]['goal']}\n\n{views[trace]}\n\n"
                    f"Question the agent asked:\n{question}",
                },
            ],
            temperature=0.0,
            # The judge must reason (the endpoint refuses to disable it); at 200 tokens its
            # reasoning sometimes used the whole budget and the reply came back empty.
            reasoning_effort="low",
            max_tokens=1_500,
        )
        text = (reply.content or "").strip()
        if not text:
            raise RuntimeError(f"judge returned nothing for {question!r}")
        return {
            "trace": trace,
            "question": question,
            "label": text.split()[0].strip(" -:"),
            "why": text,
        }

    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(label, pairs))
    client.close()
    labels.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} questions labelled -> {labels}")


def report(replays: Path, labels: Path) -> None:
    rows = [json.loads(line) for line in replays.read_text().splitlines() if line.strip()]
    lab = {
        (r["trace"], r["question"]): r["label"]
        for r in (json.loads(x) for x in labels.read_text().splitlines() if x.strip())
    }
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(
            Path(r.get("workspace", "?")).name
            if "/tmp/" not in r.get("workspace", "")
            else "django",
            [],
        ).append(r)
    for name, grp in groups.items():
        for context in (False, True):
            arm = [r for r in grp if r["context"] is context]
            n = len(arm) or 1
            counts = {
                k: sum(lab.get((r["trace"], q)) == k for r in arm for q in r["questions"])
                for k in ("CODE", "HISTORY", "USER")
            }
            print(
                f"{name:<14} clarify_context={context!s:<5}: {len(arm):>3} replays | "
                + " | ".join(f"{k}/round {v / n:.2f}" for k, v in counts.items())
                + f" | CLEAR {sum(r['clear'] for r in arm)}"
                + f" | prompt tok {sum(r['prompt_tokens'] for r in arm) / n:.0f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["replay", "judge", "report"])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--root", type=Path, action="append", default=[], help="more trace folders")
    a = ap.parse_args()
    ROOTS.extend(a.root)
    if a.cmd == "replay":
        replay(a.files[0], a.samples)
    elif a.cmd == "judge":
        judge(a.files[0], a.files[1])
    else:
        report(a.files[0], a.files[1])
