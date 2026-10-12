"""Copy trace directories and result files into a release folder, cleaned for publication.

  uv run python scripts/release_traces.py OUT --traces DIR [DIR ...] [--results FILE ...]

For every file copied: paths are rewritten to placeholders. The runs, this repository and uv's
caches keep their path under a token (`<runs>/swebench-runs/...`, `<repo>/traces/...`); any
other path under the home directory becomes `<home>/<redacted>`, another account's home
`/Users/<someone>`, and Claude Code's scratchpad and macOS temp folders `<scratch>` and
`<tmp>`. Then the person's username, host names, git author name, surname and email are
replaced (`Identity`, read from this machine). The text is then scanned for API keys, bearer
tokens, email addresses, home paths and any form of the identity. A file that still matches
is refused, named in the manifest with the reason, and the exit status is 1: nothing is
published quietly.

Listings of personal folders carry names without paths, so redaction here cannot see them:
`scripts/redact_runs.py` blanks those command outputs in place first.

Whole directories are refused by name: anything from GAIA (its validation set may not be
redistributed), `scratch-archive`, the interactive app projects (`jumpin-plumber`, another
person's project, and the author's game clones; their clarify replays in
`results/s92*-clarify-*` too) and pytest's temp workspaces. Results files keep their relative
`trace_path` values, so a release holds a sweep's records beside its traces.

The manifest (`OUT/manifest.json`) lists every file copied with its size and every file or
directory refused with its reason. Read it before uploading.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DENIED_NAMES = (
    "gaia",
    "scratch-archive",
    "pytest-of-",
    # The interactive app runs stay private: their goals are a person's own words.
    "jumpin-plumber",
    "flappy",
    "mario",
    "sweeper",
    "agent-sandbox",
    "-clarify-",
)
SECRET_PATTERNS = {
    "openrouter key": re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),
    "hugging face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9-]{20,}"),
    "bearer token": re.compile(r"Bearer\s+(?!\{|\$|<)[A-Za-z0-9._-]{24,}"),
    # The local part may not start inside a word or a JSON escape: a Python diff line such
    # as "\n+@pytest.mark.parametrize" is a decorator, not an address.
    "email": re.compile(
        r"(?<![\\\w])[A-Za-z0-9][A-Za-z0-9._%-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,6}\b"
    ),
}
# What a leftover absolute path into a user's home looks like on macOS or Linux.
HOME_LEFTOVER = re.compile(r"/(?:Users|home)/(?!<)[A-Za-z0-9._-]+/")
TEXT_SUFFIXES = {".jsonl", ".json", ".txt", ".md", ".log", ".csv", ".py", ".diff"}

# Folders under the home directory that a release may name, as a placeholder: the runs, this
# repository and uv's caches. Any other path under home says something about the person (their
# documents, their other projects) and is cut to `<home>/<redacted>`. "AI Agent" holds a space,
# so the repository is also matched with the space escaped once or twice (shell, then JSON).
KEPT_UNDER_HOME = {
    ".cache/mercury-harness": "<runs>",
    "Projects/AI Agent": "<repo>",
    "Projects/AI\\ Agent": "<repo>",
    "Projects/AI\\\\ Agent": "<repo>",
    ".cache/uv": "<uv-cache>",
    ".local/share/uv": "<uv-python>",
    ".cache/pip": "<pip-cache>",
}
# Host names that say nothing about the person ("Mac" is a macOS LocalHostName). Replacing
# them as words corrupted released text: `encodings/mac_romanian.py`, "Macintosh".
GENERIC_HOST_WORDS = {"mac", "macbook", "imac", "pro", "air", "mini", "studio", "localhost", "s"}
# A path's characters stop at whitespace, quotes, a JSON escape and shell punctuation.
_PATH_TAIL = r"(?:/[^\s\"'\\:;|&<>()*,]*)?"


@dataclass(frozen=True)
class Identity:
    """What identifies the person on this machine: every form is redacted from a release, and a
    file that still holds one is refused."""

    home: Path
    hosts: tuple[str, ...] = ()
    names: tuple[str, ...] = ()  # git author names, surnames
    emails: tuple[str, ...] = ()

    @property
    def user(self) -> str:
        return self.home.name

    def tokens(self) -> dict[str, str]:
        """Emails first: a name inside an address would otherwise be replaced alone and leave
        the rest of the address (a GitHub noreply address carries the account's number)."""
        found = {e: "<email>" for e in self.emails}
        found |= {n: "<name>" for n in self.names}
        found |= {h: "<host>" for h in self.hosts}
        return found | {self.user: "user"}


def _out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def generic_host(host: str) -> bool:
    return all(w in GENERIC_HOST_WORDS for w in re.findall(r"[a-z]+", host.lower()))


def this_machine() -> Identity:
    hosts = {socket.gethostname().removesuffix(".local")}
    hosts |= {_out(["scutil", "--get", n]) for n in ("LocalHostName", "ComputerName")}
    hosts = {h for h in hosts if not generic_host(h)}
    name = _out(["git", "config", "--global", "user.name"])
    email = _out(["git", "config", "--global", "user.email"])
    # The surname alone: personal file names carry it ("2024 J Doe Client Copy.pdf").
    names = {name, *re.findall(r"[A-Z][a-z]{3,}", name)} - {""}
    return Identity(
        home=Path.home(),
        hosts=tuple(sorted(h for h in hosts if h)),
        names=tuple(sorted(names, key=len, reverse=True)),
        emails=tuple(e for e in (email,) if e),
    )


def _home_path(m: re.Match[str]) -> str:
    rest = (m.group("rest") or "").lstrip("/")
    for prefix, token in KEPT_UNDER_HOME.items():
        if rest == prefix or rest.startswith(prefix + "/"):
            return token + rest[len(prefix) :]
    return "<home>" if not rest else "<home>/<redacted>"


def redact(text: str, ident: Identity) -> str:
    """Paths to placeholders, then every remaining form of the person's identity."""
    home = re.escape(str(ident.home))
    for prefix in sorted(KEPT_UNDER_HOME, key=len, reverse=True):  # names with spaces first
        token = KEPT_UNDER_HOME[prefix]
        text = re.sub(
            rf"(?:{home}|(?<![\w)\]])~|\$HOME|\$\{{HOME\}})/{re.escape(prefix)}", token, text
        )
    text = re.sub(
        rf"(?:{home}|(?<![\w)\]])~(?=/)|\$HOME|\$\{{HOME\}})(?P<rest>{_PATH_TAIL})",
        _home_path,
        text,
    )
    text = re.sub(r"/private/tmp/claude-\d+/[^/\s\"']+/[0-9a-f-]{36}/scratchpad", "<scratch>", text)
    text = re.sub(r"/private/tmp/claude-\d+/[^\s\"'\\]*", "<scratch>", text)
    text = re.sub(r"/private/var/folders/[^\s\"']+?/T/", "<tmp>/", text)
    # Another account's home, usually a path a model made up (/Users/runner/work/django).
    text = re.sub(r"/(Users|home)/(?!<)[A-Za-z0-9._-]+(?=/|\b)", r"/\1/<someone>", text)
    # Any address holding a form of the identity goes whole, before the name inside it can be
    # replaced alone (a GitHub noreply address also carries the account's number).
    named = "|".join(re.escape(v) for v in ident.tokens())
    text = re.sub(
        rf"[\w.+%-]*(?:{named})[\w.+%-]*@[\w-]+(?:\.[\w-]+)+", "<email>", text, flags=re.I
    )
    for value, token in ident.tokens().items():
        text = re.sub(_word(value), token, text, flags=re.I)
    return text


def _word(value: str) -> str:
    """`value` as a whole word. A JSON escape counts as a boundary: in `\\nalice` (a line of
    `ps aux` inside a JSON string) the `n` belongs to the escape, not to the name."""
    return rf"(?:(?<![A-Za-z0-9])|(?<=\\[ntr])){re.escape(value)}(?![A-Za-z0-9])"


def scan(text: str, ident: Identity | None = None) -> list[str]:
    """Names of the patterns a text still matches after redaction; empty means clean."""
    found = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    if HOME_LEFTOVER.search(text):
        found.append("home path")
    if ident and any(re.search(_word(v), text, re.I) for v in ident.tokens()):
        found.append("identity")
    return found


def denied(parts: tuple[str, ...], deny: tuple[str, ...] = DENIED_NAMES) -> str | None:
    """A refusal reason when any of these path parts (the released path, never the source
    machine's ancestors) names a denied corpus."""
    for part in parts:
        for name in deny:
            if name in part.lower():
                return f"denied by name: {name}"
    return None


@dataclass
class Manifest:
    copied: list[dict] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refused

    def write(self, out: Path) -> None:
        (out / "manifest.json").write_text(json.dumps(self.__dict__, indent=2) + "\n")


def release_file(
    src: Path,
    dest: Path,
    ident: Identity,
    manifest: Manifest,
    label: str,
    deny: tuple[str, ...] = DENIED_NAMES,
    redact_emails: bool = False,
) -> None:
    """Copy one file redacted, or refuse it. `redact_emails` replaces every address with
    `<email>` instead of refusing the file: traces of real projects quote third parties'
    addresses (a repository's commit authors, a fetched page's contacts)."""
    reason = denied(Path(label).parts, deny)
    if reason:
        manifest.refused.append({"path": label, "reason": reason})
        return
    if src.suffix not in TEXT_SUFFIXES:
        manifest.refused.append(
            {"path": label, "reason": f"not a text type: {src.suffix or 'none'}"}
        )
        return
    text = redact(src.read_text(errors="replace"), ident)
    if redact_emails:
        text = SECRET_PATTERNS["email"].sub("<email>", text)
    hits = scan(text, ident)
    if hits:
        manifest.refused.append({"path": label, "reason": "matches " + ", ".join(hits)})
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    manifest.copied.append({"path": label, "bytes": dest.stat().st_size})


def release(
    out: Path, traces: list[Path], results: list[Path], ident: Identity | None = None
) -> Manifest:
    ident = ident or this_machine()
    manifest = Manifest()
    for d in traces:
        reason = denied((d.name,))
        if reason:
            manifest.refused.append({"path": d.name, "reason": reason})
            continue
        for src in sorted(p for p in d.rglob("*") if p.is_file()):
            label = str(Path("traces") / d.name / src.relative_to(d))
            release_file(src, out / label, ident, manifest, label)
    for f in results:
        label = str(Path("results") / f.name)
        release_file(f, out / label, ident, manifest, label)
    out.mkdir(parents=True, exist_ok=True)
    manifest.write(out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path)
    ap.add_argument("--traces", nargs="*", type=Path, default=[])
    ap.add_argument("--results", nargs="*", type=Path, default=[])
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty; choose a fresh directory", file=sys.stderr)
        return 2
    manifest = release(args.out, args.traces, args.results)
    total = sum(f["bytes"] for f in manifest.copied)
    print(
        f"copied {len(manifest.copied)} files ({total / 1e6:.1f} MB), "
        f"refused {len(manifest.refused)}"
    )
    for r in manifest.refused:
        print("  refused", r["path"], "-", r["reason"])
    print(f"manifest: {args.out / 'manifest.json'}")
    return 0 if manifest.ok else 1


if __name__ == "__main__":
    sys.exit(main())
