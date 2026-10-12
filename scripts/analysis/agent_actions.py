"""Every command and file access the external agents made, as one JSONL table.

Sources under ~/.cache/mercury-harness: mini trajectories (*.traj.json, traj.json), Codex
event streams (stdout.log with command_execution items), Goose's session store, Aider chat
histories. One row per action: agent, source file, workspace, kind, text, output.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path.home() / ".cache/mercury-harness"
GOOSE_DB = Path.home() / ".local/share/goose/sessions/sessions.db"
OUT_CAP = 20000


def ws_of(path: Path) -> str:
    """The run folder: <suite>/<round>, the most specific folder an agent was meant to stay in."""
    rel = path.relative_to(ROOT).parts
    return str(ROOT.joinpath(*rel[:2]))


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def mini_rows(p: Path):
    try:
        msgs = json.loads(p.read_text())["messages"]
    except Exception as e:
        yield {"agent": "mini", "src": str(p), "kind": "unreadable", "text": repr(e)}
        return
    outputs = {}
    for m in msgs:
        if m.get("role") == "tool":
            outputs[m.get("tool_call_id")] = text_of(m.get("content"))
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if calls:
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"command": fn.get("arguments")}
                yield {
                    "agent": "mini", "src": str(p), "step": i, "kind": "cmd",
                    "text": args.get("command") or json.dumps(args),
                    "output": outputs.get(c.get("id"), "")[:OUT_CAP],
                }  # fmt: skip
        else:
            nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
            for cmd in re.findall(r"```(?:bash|sh)?\n(.*?)```", text_of(m.get("content")), re.S):
                yield {
                    "agent": "mini", "src": str(p), "step": i, "kind": "cmd", "text": cmd,
                    "output": text_of(nxt.get("content"))[:OUT_CAP],
                }  # fmt: skip


def codex_rows(p: Path):
    for line in p.open(errors="replace"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = e.get("item") or {}
        if e.get("type") != "item.completed":
            continue
        if it.get("type") == "command_execution":
            yield {
                "agent": "codex", "src": str(p), "kind": "cmd", "text": it.get("command", ""),
                "output": (it.get("aggregated_output") or "")[:OUT_CAP],
            }  # fmt: skip
        elif it.get("type") == "file_change":
            for ch in it.get("changes", []):
                yield {"agent": "codex", "src": str(p), "kind": "write", "text": ch.get("path", "")}


def goose_rows():
    db = sqlite3.connect(f"file:{GOOSE_DB}?mode=ro", uri=True)
    sessions = dict(db.execute("select id, name from sessions"))
    pending = {}
    for sid, content in db.execute("select session_id, content_json from messages order by id"):
        for part in json.loads(content or "[]"):
            if part.get("type") == "toolRequest":
                call = (part.get("toolCall") or {}).get("value") or {}
                args = call.get("arguments") or {}
                text = args.get("command") or args.get("path") or json.dumps(args)
                pending[part.get("id")] = {
                    "agent": "goose", "src": f"goose:{sessions.get(sid)}", "kind": call.get("name"),
                    "text": text, "args": json.dumps(args)[:2000], "output": "",
                }  # fmt: skip
            elif part.get("type") == "toolResponse":
                row = pending.pop(part.get("id"), None)
                if row is None:
                    continue
                val = (part.get("toolResult") or {}).get("value")
                if isinstance(val, dict):
                    val = val.get("content", val)
                row["output"] = text_of(val)[:OUT_CAP]
                yield row
    yield from pending.values()


def aider_rows(p: Path):
    for line in p.open(errors="replace"):
        if re.search(
            r"Added .* to the chat|Run shell command|^> Running|^#### /(run|read|add|web)", line
        ):
            yield {"agent": "aider", "src": str(p), "kind": "aider", "text": line.strip()}


def main() -> None:
    out = sys.stdout
    n = 0
    for p in sorted(ROOT.rglob("*traj.json")):
        if "release-traces" in p.parts:
            continue
        for r in mini_rows(p):
            out.write(json.dumps(r | {"ws": ws_of(p)}) + "\n")
            n += 1
    for p in sorted(ROOT.rglob("stdout.log")):
        if "codex" in str(p):
            for r in codex_rows(p):
                out.write(json.dumps(r | {"ws": ws_of(p)}) + "\n")
                n += 1
    for r in goose_rows():
        out.write(json.dumps(r | {"ws": ""}) + "\n")
        n += 1
    for p in sorted(ROOT.rglob(".aider.chat.history.md")):
        for r in aider_rows(p):
            out.write(json.dumps(r | {"ws": ws_of(p)}) + "\n")
            n += 1
    print(f"{n} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
