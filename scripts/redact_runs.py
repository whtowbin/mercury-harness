"""Blank, in place, the outputs of agent commands that looked into the home directory.

  uv run python scripts/redact_runs.py [--apply] [ROOT ...]   (default ~/.cache/mercury-harness)

The external agents (mini-swe-agent, Codex) ran unconfined, and some listed personal folders
(TODO 2.8). A listing carries names without paths, so the release scrubber
(`scripts/release_traces.py`) cannot find it; this finds the command instead. A command is
personal when it names a path under the home directory outside the folders a release may name
(`KEPT_UNDER_HOME`: the runs, this repository, uv's and pip's caches), names the home directory
itself or `/Users/*`, prints the environment, or returned such a path. Its output is replaced
by a marker and its paths are redacted; the rest of the run is left as it was, so a read of a
benchmark's own data stays visible for the contamination check (TODO 2.10).

Files: mini trajectories (`*traj.json`), mini and fixture console logs (`*.log`), Codex event
streams (`stdout.log` under a `-codex` run). A log block is matched to a redacted trajectory
output by its text with whitespace removed, since the console wraps long lines. Without
`--apply` nothing is written: the report lists every command that would be blanked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_traces import KEPT_UNDER_HOME, Identity, redact, this_machine

MARKER = "[redacted: output of a command that looked outside the workspace]"
_TAIL = r"(?P<rest>/[^\s\"'\\:;|&<>()*,]*)?"
_ENV = re.compile(r"(?:^|[;&|(]\s*|\s)(env|printenv)(?=\s*(?:$|[|;&>)]))")
_ALL_HOMES = re.compile(r"/Users(?:/\*|/?(?=\s|$|;|&|\||\"|'))")


def _kept(after: str) -> bool:
    """`after` is the text from the end of the home prefix on: a kept folder's name may hold a
    space ("AI Agent"), which a path pattern would stop at. A path cut off inside a kept
    folder's name (a wrapped console line, sed's `"/Users/x/.cac ..."`) counts as that folder;
    one that ends at a folder boundary (`~/Projects`) does not."""
    after = after.lstrip("/")
    for p in KEPT_UNDER_HOME:
        if after.startswith(p) and after[len(p) : len(p) + 1] in ("", "/", "'", '"', " "):
            return True
        seg = re.match(r"[^\s\"']*", after)
        cut = seg.group() if seg else ""
        mid_name = cut and not cut.endswith("/") and len(cut) < len(p) and p[len(cut)] != "/"
        if mid_name and p.startswith(cut):
            return True
    return False


def _home_refs(text: str, ident: Identity) -> list[str]:
    """For every reference to the home directory: the text after it (empty for home itself)."""
    home = re.escape(str(ident.home))
    pat = rf"(?:{home}|(?<![\w)\]$~])~(?=/|\s|$|;|&|\||\"|'|\))|\$HOME|\$\{{HOME\}})"
    refs = []
    for m in re.finditer(pat, text):
        after = text[m.end() :]
        refs.append(after if re.match(r"/[^\s\"';|&<>()]", after) else "")
    return refs


def personal(cmd: str, output: str, ident: Identity) -> bool:
    if "<home>" in cmd:  # already scrubbed by an earlier pass
        return True
    if _ENV.search(cmd) or _ALL_HOMES.search(cmd):
        return True
    if any(
        not after or after.startswith("/..") or not _kept(after) for after in _home_refs(cmd, ident)
    ):
        return True
    return any(after and not _kept(after) for after in _home_refs(output, ident) if after)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _output_text(content: object) -> str:
    """mini's tool message is a JSON object as a string; older ones are plain text."""
    text = content if isinstance(content, str) else json.dumps(content)
    try:
        return str(json.loads(text).get("output", text))
    except (json.JSONDecodeError, AttributeError):
        return text


def _blank(content: object) -> str:
    try:
        rc = json.loads(content).get("returncode") if isinstance(content, str) else None
    except (json.JSONDecodeError, AttributeError):
        rc = None
    return json.dumps({"returncode": rc, "output": MARKER}, indent=2)


def _commands(msg: dict) -> list[tuple[str | None, str]]:
    calls = msg.get("tool_calls") or []
    if calls:
        out = []
        for c in calls:
            raw = (c.get("function") or {}).get("arguments") or "{}"
            try:
                cmd = json.loads(raw).get("command", raw)
            except json.JSONDecodeError:
                cmd = raw
            out.append((c.get("id"), str(cmd)))
        return out
    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
    return [(None, c) for c in re.findall(r"```(?:bash|sh)?\n(.*?)```", content or "", re.S)]


def scrub(text: str, ident: Identity) -> str:
    """Personal paths under home to `<home>/<redacted>`, and the bare home directory to
    `<home>`; paths into the kept folders stay as they are, so the record of what a run read
    (a benchmark's data, another run's workspace) survives for TODO 2.10."""
    home = re.escape(str(ident.home))
    pat = rf"(?:{home}|(?<![\w)\]$~])~(?=/)|\$HOME|\$\{{HOME\}})(?P<tail>/[^\s\"'\\:;|&<>()*,]*)?"

    def sub(m: re.Match[str]) -> str:
        after = m.string[m.end() - len(m.group("tail") or "") :]
        if not m.group("tail") or m.group("tail") == "/":
            return "<home>" + (m.group("tail") or "") if m.group(0).startswith("/") else m.group(0)
        return m.group(0) if _kept(after) else "<home>/<redacted>"

    return re.sub(pat, sub, text)


def _scrub_all(obj: object, ident: Identity) -> tuple[object, int]:
    """Every string in a parsed JSON value scrubbed; and how many changed."""
    if isinstance(obj, str):
        new = scrub(obj, ident)
        return new, int(new != obj)
    if isinstance(obj, list):
        pairs = [_scrub_all(v, ident) for v in obj]
        return [v for v, _ in pairs], sum(n for _, n in pairs)
    if isinstance(obj, dict):
        pairs = {k: _scrub_all(v, ident) for k, v in obj.items()}
        return {k: v for k, (v, _) in pairs.items()}, sum(n for _, n in pairs.values())
    return obj, 0


def _reply_for(msgs: list[dict], i: int, call_id: str | None, by_id: dict) -> dict | None:
    if call_id:
        return by_id.get(call_id)
    return msgs[i + 1] if i + 1 < len(msgs) else None  # older trajectories: the next message


def _blank_reply(reply: dict, cmd: str, ident: Identity, outputs: set[str]) -> list[str]:
    """Blank one reply if its command was personal, and mini's raw copy of a blanked reply."""
    changed = []
    already = MARKER in str(reply.get("content"))
    output = _output_text(reply.get("content"))
    if not already and personal(cmd, output, ident):
        if len(_norm(output)) >= 20:
            outputs.add(_norm(output))
        reply["content"] = _blank(reply.get("content"))
        changed.append(cmd)
        already = True
    extra = reply.get("extra")
    if already and isinstance(extra, dict) and extra.get("raw_output") not in (None, MARKER):
        extra["raw_output"] = MARKER
        changed.append(f"raw copy of: {cmd}")
    return changed


def redact_trajectory(path: Path, ident: Identity, outputs: set[str]) -> tuple[str, list[str]]:
    """One mini trajectory with personal outputs blanked (the message and mini's raw copy of
    it) and personal paths scrubbed from every string; and what changed. Adds each blanked
    output's normalised text to `outputs`, for the console logs."""
    data = json.loads(path.read_text())
    msgs = data.get("messages", [])
    by_id = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool"}
    changed = []
    for i, m in enumerate(msgs):
        for call_id, cmd in _commands(m) if m.get("role") == "assistant" else []:
            reply = _reply_for(msgs, i, call_id, by_id)
            if reply is not None:
                changed += _blank_reply(reply, cmd, ident, outputs)
    scrubbed, n = _scrub_all(data, ident)
    if n:
        changed.append(f"{n} strings scrubbed")
    return json.dumps(scrubbed, indent=2), changed


_BLOCK_END = re.compile(r"^(─{10,}|Tool:|mini-swe-agent \(step)")
# mini prints its config before the run, and the console wraps the PATH item mid-word, so its
# pieces cannot be matched line by line; it holds no quote, so the item is cut whole.
_PATH_ITEM = re.compile(r"'environment\.env\.PATH=[^']*'")


def _end(lines: list[str], i: int, stop) -> int:
    j = i + 1
    while j < len(lines) and not stop(lines[j]):
        j += 1
    return j


def _log_lines(lines: list[str], ident: Identity, outputs: set[str]) -> int:
    """Blank output blocks in place and scrub other lines; the number of lines changed.

    Each fenced command since the last step is queued as personal or not, and its Tool block
    follows in order. A block is blanked by its command as well as by matching a blanked
    trajectory output, since a timed-out command's block holds an exception beside the
    listing and no longer matches."""
    queue: list[bool] = []
    changed, i = 0, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("mini-swe-agent (step"):
            queue = []
        if line.rstrip() == "```":
            j = _end(lines, i, lambda x: x.rstrip() == "```")
            queue.append(personal(" ".join(x.strip() for x in lines[i + 1 : j]), "", ident))
        if line.strip() != "<output>":
            new = scrub(line, ident)
            changed += new != line
            lines[i] = new
            i += 1
            continue
        by_command = queue.pop(0) if queue else False
        j = _end(lines, i, _BLOCK_END.match)
        body = _norm("".join(lines[i + 1 : j]))
        matched = any(body == o or (len(body) >= 20 and body in o) for o in outputs)
        if body and body != _norm(MARKER) and (by_command or matched):
            changed += j - i - 1
            lines[i + 1 : j] = [MARKER + "\n", "\n"]
            j = i + 3
        i = j
    return changed


def redact_log(path: Path, ident: Identity, outputs: set[str]) -> tuple[str, int]:
    """A console log with mini's PATH item cut, personal output blocks blanked and personal
    paths scrubbed from every other line; and the number of lines changed."""
    text = path.read_text(errors="replace")
    cut = _PATH_ITEM.sub("'environment.env.PATH=<redacted>'", text)
    lines = cut.splitlines(keepends=True)
    changed = _log_lines(lines, ident, outputs) + (cut != text)
    return "".join(lines), changed


def _personal_event(e: object, ident: Identity) -> bool:
    if not isinstance(e, dict) or e.get("type") != "item.completed":
        return False
    it = e.get("item") or {}
    out = it.get("aggregated_output") or ""
    return (
        it.get("type") == "command_execution"
        and out != MARKER
        and personal(it.get("command", ""), out, ident)
    )


def redact_codex(path: Path, ident: Identity) -> tuple[str, list[str]]:
    events = [json.loads(line) if line.strip().startswith("{") else line for line in path.open()]
    dicts = [e for e in events if isinstance(e, dict)]
    ids = {e["item"].get("id") for e in dicts if _personal_event(e, ident)}
    blanked = []
    for e in dicts:
        it = e.get("item")
        if it and it.get("id") in ids and it.get("type") == "command_execution":
            if e.get("type") == "item.completed":
                blanked.append(it.get("command", ""))
            it["command"] = redact(it.get("command", ""), ident)
            if it.get("aggregated_output"):
                it["aggregated_output"] = MARKER
    pairs = [_scrub_all(e, ident) if isinstance(e, dict) else (e, 0) for e in events]
    scrubbed = sum(n for _, n in pairs)
    if scrubbed:
        blanked.append(f"{scrubbed} strings scrubbed")
    text = "".join(json.dumps(e) + "\n" if isinstance(e, dict) else str(e) for e, _ in pairs)
    return text, blanked


def _files(root: Path) -> list[tuple[str, Path]]:
    """Trajectories first, so their blanked outputs are known when the logs are read."""

    def found(pattern: str) -> list[Path]:
        return [p for p in sorted(root.rglob(pattern)) if "release-traces" not in p.parts]

    def is_codex(p: Path) -> bool:
        return p.name == "stdout.log" and any("-codex" in x for x in p.parts)

    logs = found("*.log")
    return (
        [("traj", p) for p in found("*traj.json")]
        + [("codex", p) for p in logs if is_codex(p)]
        + [("log", p) for p in logs if not is_codex(p)]
    )


def run(roots: list[Path], ident: Identity, apply: bool) -> dict[str, list[str]]:
    """Every file with something to blank, and what; files are written only with `apply`."""
    report: dict[str, list[str]] = {}
    outputs: set[str] = set()
    for kind, p in (f for root in roots for f in _files(root)):
        if kind == "log":
            text, n = redact_log(p, ident, outputs)
            items = [f"{n} lines"] if n else []
        elif kind == "codex":
            text, items = redact_codex(p, ident)
        else:
            text, items = redact_trajectory(p, ident, outputs)
        if items:
            report[str(p)] = items
            if apply:
                p.write_text(text)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="*", type=Path, default=[Path.home() / ".cache/mercury-harness"])
    ap.add_argument("--apply", action="store_true", help="write the changes (default: report)")
    args = ap.parse_args(argv)
    report = run(args.roots, this_machine(), args.apply)
    for path, items in report.items():
        print(path)
        for item in items:
            print("   ", re.sub(r"\s+", " ", item)[:160])
    verb = "redacted" if args.apply else "would redact"
    print(f"{verb} {len(report)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
