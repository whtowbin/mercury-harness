"""Deterministic analysis of Claude Code session transcripts.

The harness's own traces (`tracetools.py`) record this agent's runs. Claude
Code records its *own* sessions too, as JSONL under
`~/.claude/projects/<project-slug>/`, and the questions worth asking of them
are the same ones this repo asks of its traces: where does the cost go, what
stays in context, what keeps being redone by hand. This module answers them
without a model call, so the same session gives the same numbers every time.

What it measures, and the assumption each rests on:

* **Cost is context length times turns.** Every turn re-reads the whole
  context from the prompt cache, so a long session is expensive mainly for
  the reads, not the writes. Costs here are *relative units* against the base
  input price (`WEIGHTS`: a cache read is 0.1, an hour-long cache write 2.0,
  output 5.0). Those ratios are Anthropic's published structure for the
  Sonnet family, not read from the transcript; a different model shifts the
  proportions, not the ranking.
* **One API response is written as several records** (one per content block),
  each repeating the same `usage`. Counting records would multiply tokens by
  the number of blocks, so responses are deduplicated by message id.
* **A compaction shows as a context drop.** The transcript does not label it
  reliably, so a turn whose context is under `COMPACTION_DROP` of the previous
  turn's is treated as one.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Relative to the base input price. Structure, not a quote: see the module doc.
WEIGHTS = {"fresh": 1.0, "read": 0.1, "write": 2.0, "out": 5.0}
COMPACTION_DROP = 0.6
CHARS_PER_TOKEN = 3.6
# Context buckets for "how much of the cost is spent at what size".
BUCKETS = ((50_000, "<50K"), (100_000, "50-100K"), (200_000, "100-200K"), (400_000, "200-400K"))
LAST_BUCKET = "400K+"
EDIT_TOOLS = ("Edit", "Write", "NotebookEdit")
# Tools whose duration is a person thinking, and Bash heads that are a poll on
# background work: neither is the agent's own time, so neither goes in the
# "slowest commands" table.
USER_WAIT_TOOLS = ("AskUserQuestion", "ExitPlanMode")
POLL_HEADS = ("until", "sleep", "for", "while")
_CD_PREFIX = re.compile(r'^cd\s+"?[^&;]*"?\s*&&\s*')


@dataclass
class Turn:
    """One deduplicated API response."""

    ctx: int  # input-side tokens the model read this turn
    fresh: int
    read: int
    write: int
    out: int

    @property
    def units(self) -> float:
        return (
            self.fresh * WEIGHTS["fresh"]
            + self.read * WEIGHTS["read"]
            + self.write * WEIGHTS["write"]
            + self.out * WEIGHTS["out"]
        )


def default_dir(cwd: Path | str) -> Path:
    """Where Claude Code keeps the transcripts for a working directory."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    return Path.home() / ".claude" / "projects" / slug


def load_records(path: Path | str) -> list[dict]:
    records = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _blocks(record: dict) -> list[dict]:
    content = (record.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def turns(records: Iterable[dict]) -> list[Turn]:
    """Deduplicated API responses, in order."""
    seen: set[str] = set()
    out: list[Turn] = []
    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message") or {}
        key = message.get("id") or record.get("uuid") or ""
        if key in seen:
            continue
        seen.add(key)
        usage = message.get("usage") or {}
        created = usage.get("cache_creation") or {}
        write = (
            created.get("ephemeral_1h_input_tokens", 0)
            + created.get("ephemeral_5m_input_tokens", 0)
        ) or usage.get("cache_creation_input_tokens", 0)
        read = usage.get("cache_read_input_tokens", 0)
        fresh = usage.get("input_tokens", 0)
        out.append(Turn(fresh + read + write, fresh, read, write, usage.get("output_tokens", 0)))
    return out


def compacted(previous: int, current: int) -> bool:
    return current < COMPACTION_DROP * previous


def bucket_of(ctx: int) -> str:
    return next((label for limit, label in BUCKETS if ctx < limit), LAST_BUCKET)


def cost_shares(all_turns: list[Turn]) -> dict[str, float]:
    """Each component's share of total relative cost, 0..1."""
    parts = {
        "cache read": sum(t.read for t in all_turns) * WEIGHTS["read"],
        "cache write": sum(t.write for t in all_turns) * WEIGHTS["write"],
        "output": sum(t.out for t in all_turns) * WEIGHTS["out"],
        "fresh input": sum(t.fresh for t in all_turns) * WEIGHTS["fresh"],
    }
    total = sum(parts.values()) or 1.0
    return {name: value / total for name, value in parts.items()}


def context_buckets(all_turns: list[Turn]) -> dict[str, tuple[float, float]]:
    """{bucket: (share of turns, share of cost)}."""
    n = defaultdict(int)
    units = defaultdict(float)
    for turn in all_turns:
        label = bucket_of(turn.ctx)
        n[label] += 1
        units[label] += turn.units
    total_n, total_u = len(all_turns) or 1, sum(units.values()) or 1.0
    labels = [label for _, label in BUCKETS] + [LAST_BUCKET]
    return {label: (n[label] / total_n, units[label] / total_u) for label in labels}


def _simulated_cost(
    sessions: list[list[Turn]], cap: int | None, base: int, summary: int, reorient: int
) -> float:
    cost = 0.0
    for session in sessions:
        prev: int | None = None
        sim = 0
        for turn in session:
            grew = 0 if prev is None else max(0, turn.ctx - prev)
            if prev is None or compacted(prev, turn.ctx):
                sim = turn.ctx  # a real compaction in the data: follow it
            else:
                sim += grew
            prev = turn.ctx
            if cap is not None and sim > cap:
                cost += sim * WEIGHTS["read"] + summary * WEIGHTS["out"]
                sim = base + summary + reorient
                cost += sim * WEIGHTS["write"]
            cost += sim * WEIGHTS["read"] + min(turn.write, grew + 1) * WEIGHTS["write"]
            cost += turn.out * WEIGHTS["out"]
    return cost


def simulate_cap(
    sessions: list[list[Turn]],
    cap: int,
    base: int = 40_000,
    summary: int = 8_000,
    reorient: int = 0,
) -> float:
    """Cost of the same sessions if context were compacted whenever it passed
    `cap`, as a fraction of the same model with no cap (so the model's own
    approximations cancel). An estimate, not a forecast: per-turn growth and
    output are held at what was measured, `reorient` is the tokens the agent
    must re-read after each compaction to find its place, and nothing here
    prices the detail a compaction loses."""
    uncapped = _simulated_cost(sessions, None, base, summary, reorient)
    return _simulated_cost(sessions, cap, base, summary, reorient) / uncapped if uncapped else 1.0


def _result_chars(block: dict) -> int:
    content = block.get("content")
    if isinstance(content, str):
        return len(content)
    return sum(len(x.get("text", "")) for x in content or [] if isinstance(x, dict))


def _carry_key(name: str, tool_input: dict) -> str:
    if name == "Read":
        return "Read:" + os.path.basename(str(tool_input.get("file_path", "")))
    return name


@dataclass
class Call:
    """A tool call paired with its result."""

    turn_index: int  # API responses seen when the call was made
    use: dict
    result: dict
    started: float | None
    finished: float | None

    @property
    def name(self) -> str:
        return self.use.get("name", "")

    @property
    def tool_input(self) -> dict:
        return self.use.get("input") or {}

    @property
    def seconds(self) -> float | None:
        if self.started is None or self.finished is None:
            return None
        return self.finished - self.started

    @property
    def command(self) -> str:
        """A Bash command with the habitual `cd <dir> &&` prefix removed."""
        return _CD_PREFIX.sub("", str(self.tool_input.get("command", ""))).strip()


def _timestamp(record: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError):
        return None


def _paired_calls(records: list[dict]) -> Iterator[Call]:
    """Every tool call with its result, in the order the results arrived."""
    seen: set[str] = set()
    pending: dict[str, tuple[int, dict, float | None]] = {}
    for record in records:
        if record.get("type") == "assistant":
            seen.add((record.get("message") or {}).get("id") or record.get("uuid") or "")
            for block in _blocks(record):
                if block.get("type") == "tool_use":
                    pending[block["id"]] = (len(seen), block, _timestamp(record))
        elif record.get("type") == "user":
            for block in _blocks(record):
                call = pending.pop(block.get("tool_use_id", ""), None)
                if block.get("type") == "tool_result" and call:
                    yield Call(call[0], call[1], block, call[2], _timestamp(record))


def _turns_until_compaction(all_turns: list[Turn]) -> list[int]:
    """For each turn, the index of the first later turn that follows a
    compaction (or the end): how long anything added at that turn survives."""
    ends = []
    for i in range(len(all_turns)):
        j = i + 1
        while j < len(all_turns) and not compacted(all_turns[j - 1].ctx, all_turns[j].ctx):
            j += 1
        ends.append(j)
    return ends


def compactions(all_turns: list[Turn]) -> int:
    return sum(compacted(a.ctx, b.ctx) for a, b in itertools.pairwise(all_turns))


def tool_carry(records: list[dict]) -> dict[str, float]:
    """Relative cost of keeping each tool result in context: its tokens times
    the turns it stayed before the next compaction, at the cache-read price.
    This, not the size of a single result, is what a big result costs."""
    ends = _turns_until_compaction(turns(records))
    carry: dict[str, float] = defaultdict(float)
    for call in _paired_calls(records):
        later = ends[min(call.turn_index, len(ends)) - 1] - call.turn_index if ends else 0
        tokens = _result_chars(call.result) / CHARS_PER_TOKEN
        carry[_carry_key(call.name, call.tool_input)] += tokens * max(0, later) * WEIGHTS["read"]
    return carry


def unchanged_rereads(records: list[dict]) -> list[tuple[float, str]]:
    """(tokens, path) for each Read of a file range already read with no edit to
    that file in between: context spent to learn nothing new."""
    seen_ranges: set[tuple] = set()
    wasted: list[tuple[float, str]] = []
    for call in _paired_calls(records):
        path = call.tool_input.get("file_path")
        if not path:
            continue
        if call.name in EDIT_TOOLS:
            seen_ranges = {r for r in seen_ranges if r[0] != path}
        elif call.name == "Read":
            key = (path, call.tool_input.get("offset"), call.tool_input.get("limit"))
            if key in seen_ranges:
                wasted.append((_result_chars(call.result) / CHARS_PER_TOKEN, path))
            seen_ranges.add(key)
    return wasted


def command_family(command: str, words: int = 2) -> str:
    return " ".join(command.split()[:words])[:40]


def slow_commands(records: list[dict]) -> dict[str, list[float]]:
    """Seconds per run of each Bash command family, leaving out polls on
    background work (see POLL_HEADS), which are waiting rather than working."""
    families: dict[str, list[float]] = defaultdict(list)
    for call in _paired_calls(records):
        if call.name != "Bash" or call.seconds is None:
            continue
        family = command_family(call.command)
        if family.split(" ")[0] not in POLL_HEADS:
            families[family].append(call.seconds)
    return families


def repeated_commands(records: list[dict]) -> Counter[str]:
    """How often each Bash command family was run. A family that keeps
    appearing is a candidate for a tool, by this repo's second-time rule."""
    return Counter(command_family(c.command, 3) for c in _paired_calls(records) if c.name == "Bash")


def error_kinds(records: list[dict]) -> Counter[str]:
    """First 70 characters of every tool error, counted."""
    kinds: Counter[str] = Counter()
    for call in _paired_calls(records):
        if call.result.get("is_error"):
            content = call.result.get("content")
            text = content if isinstance(content, str) else str(content)
            kinds[" ".join(text.split())[:70]] += 1
    return kinds


def full_suite_pytest(records: list[dict]) -> list[tuple[bool, float]]:
    """(run in parallel, seconds) for each pytest run that named no file, test
    or -k filter, i.e. a whole-suite run."""
    target = re.compile(r"(tests/\S+|test_\S+\.py|-k\s|::)")
    runs = []
    for call in _paired_calls(records):
        command = call.command
        whole_suite = "pytest" in command and not target.search(command)
        if call.name == "Bash" and whole_suite and call.seconds is not None:
            runs.append((bool(re.search(r"-n\s*(auto|\d)", command)), call.seconds))
    return runs


def _session_date(records: list[dict]) -> str:
    stamps = [t for t in (_timestamp(r) for r in records) if t]
    return datetime.fromtimestamp(min(stamps)).strftime("%m-%d %H:%M") if stamps else "?"


def _share_lines(every: list[Turn]) -> list[str]:
    lines = ["where the cost goes (relative; a cache read is 0.1 of base input):"]
    for name, share in sorted(cost_shares(every).items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:12}{100 * share:5.1f}%")
    lines += ["", "cost by the context size of the turn:"]
    for label, (turn_share, cost_share) in context_buckets(every).items():
        lines.append(
            f"  {label:9} {100 * turn_share:5.1f}% of turns  {100 * cost_share:5.1f}% of cost"
        )
    return lines


def _cap_lines(sessions: list[list[Turn]]) -> list[str]:
    lines = ["cost if context were compacted at a cap (R = tokens re-read after each compaction):"]
    lines.append("  cap      R=0    R=30K   R=60K")
    for cap in (200_000, 300_000, 400_000):
        row = "".join(
            f"{100 * simulate_cap(sessions, cap, reorient=r):7.1f}%" for r in (0, 30_000, 60_000)
        )
        lines.append(f"  {cap // 1000:>3}K {row}")
    return lines


def _session_lines(names: list[str], per_session: list[list[dict]]) -> list[str]:
    lines = [
        "per session (oldest first):",
        "  id        started      turns  max ctx  compactions  cost",
    ]
    for name, records in zip(names, per_session, strict=True):
        t = turns(records)
        if not t:
            continue
        lines.append(
            f"  {name[:8]:9} {_session_date(records):12}{len(t):>6}{max(x.ctx for x in t):>9,}"
            f"{compactions(t):>13}{sum(x.units for x in t) / 1e6:>7.0f}M"
        )
    return lines


def _context_lines(per_session: list[list[dict]]) -> list[str]:
    carry: dict[str, float] = defaultdict(float)
    wasted: list[tuple[float, str]] = []
    for records in per_session:
        for name, value in tool_carry(records).items():
            carry[name] += value
        wasted += unchanged_rereads(records)
    total = sum(carry.values()) or 1.0
    lines = ["what stays in context longest (tokens x turns carried):"]
    lines += [
        f"  {name:30}{100 * v / total:5.1f}%"
        for name, v in sorted(carry.items(), key=lambda kv: -kv[1])[:8]
    ]
    lines += [
        "",
        f"re-reads of an unchanged file range: {len(wasted)} calls, "
        f"{int(sum(t for t, _ in wasted)):,} tokens re-added",
    ]
    by_file = Counter(os.path.basename(p) for _, p in wasted)
    return lines + [f"  {name:30}{n:>4}x" for name, n in by_file.most_common(5)]


def _habit_lines(per_session: list[list[dict]]) -> list[str]:
    families: dict[str, list[float]] = defaultdict(list)
    repeats: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    suite: list[tuple[bool, float]] = []
    for records in per_session:
        for family, seconds in slow_commands(records).items():
            families[family] += seconds
        repeats.update(repeated_commands(records))
        errors.update(error_kinds(records))
        suite += full_suite_pytest(records)
    lines = ["slowest Bash command families (runs, total min, median s):"]
    for family, secs in sorted(families.items(), key=lambda kv: -sum(kv[1]))[:6]:
        lines.append(
            f"  {family:32}{len(secs):>5}{sum(secs) / 60:>8.1f}m{statistics.median(secs):>7.1f}"
        )
    lines += ["", "most repeated Bash commands (candidates for a tool on the second time):"]
    lines += [f"  {n:>5}  {family}" for family, n in repeats.most_common(6)]
    parallel = [s for par, s in suite if par]
    serial = [s for par, s in suite if not par]
    lines += [
        "",
        f"whole-suite pytest runs: {len(serial)} sequential ({sum(serial) / 60:.0f} min), "
        f"{len(parallel)} parallel ({sum(parallel) / 60:.0f} min)",
        "",
        "most common tool errors:",
    ]
    return lines + [f"  {n:>4}  {kind}" for kind, n in errors.most_common(4)]


def report(directory: Path | str, last: int | None = None) -> str:
    """Every section, over the newest `last` sessions (or all of them)."""
    files = sorted(Path(directory).glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    if last:
        files = files[-last:]
    if not files:
        return f"no transcripts in {directory}"
    per_session = [load_records(f) for f in files]
    sessions = [turns(records) for records in per_session]
    every = [t for s in sessions for t in s]
    head = [f"{len(files)} sessions, {len(every):,} API responses (deduplicated)"]
    blocks = [
        head,
        _session_lines([f.stem for f in files], per_session),
        _share_lines(every),
        _cap_lines(sessions),
        _context_lines(per_session),
        _habit_lines(per_session),
    ]
    return "\n\n".join("\n".join(b) for b in blocks)
