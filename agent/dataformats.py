"""Verify commands for a step's non-code deliverables: structured data and prose.

Structured data (JSON/YAML/TOML/XML/...) has a real syntax to check against,
so a wrong file is a wrong file, the same as a Python syntax error. Last
session that meant three hand-written branches in plan.py -- jq for JSON,
stdlib tomllib for TOML, yq for YAML -- that had already drifted (yq only
covered one of three formats it can actually validate). `yq` (mikefarah's,
https://github.com/mikefarah/yq) auto-detects a file's format by extension
and fails with a real parse error on any of JSON, YAML, TOML, XML, CSV/TSV,
INI, and properties files, including across several files in one call --
confirmed locally (`yq '.' good.yaml bad.yaml` prints the good file's content
then exits 1 on the bad one). One binary now covers what used to be three
branches, with jq/tomllib/xmllint kept only as fallbacks for a host without
`yq` on PATH.

Markdown and plain text have no syntax to check against -- a heading like
"Algorithm Results Summary" only exists as freeform prose in a step's
`acceptance` text, and pattern-matching that out would manufacture a false
pass more often than it caught a real failure. `build_prose_validator` adds
one narrow, real check instead of a fabricated one: `pandoc` will fail to
parse genuinely broken YAML frontmatter, confirmed locally, but tolerates a
malformed table or an unclosed code fence (it auto-closes at EOF) --
confirmed locally too. That narrowness is accepted, not hidden: the
mandatory guarantee for a Markdown deliverable is still
`agent.phases.execute._unwritten_declared_files` (a step cannot finish
without having touched every file it declared under `writes`, whatever its
verify command says or whether one exists); this is one additional,
best-effort layer on top of that, skipped silently when `pandoc` isn't
installed -- the same detect-don't-assume convention `agent/tools/fs.py`'s
rg fallback and `agent/tools/structural.py`'s ast-grep detection already use.
"""

from __future__ import annotations

import shlex
import shutil

_FORMAT_EXTS = frozenset(
    {".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".tsv", ".ini", ".properties"}
)
_PROSE_EXTS = (".md", ".markdown")


def _quoted(paths: list[str]) -> str:
    """Paths as shell words. These commands run through a shell, so an unquoted
    `my data.json` became two arguments and a name with `;` in it became a second
    command."""
    return " ".join(shlex.quote(p) for p in paths)


def build_validator(files: list[str]) -> str:
    """One command that fails if any of `files` doesn't parse as its format.

    `yq` first, since it alone covers every extension in `_FORMAT_EXTS`.
    Without it, per-format fallbacks: jq for JSON, stdlib tomllib for TOML
    (always available, no install needed), xmllint for XML. YAML/CSV/TSV/
    INI/properties have no fallback validator without yq -- unchanged from
    before this module existed: skipped rather than fabricating a false pass.
    """
    targets = [f for f in files if any(f.endswith(ext) for ext in _FORMAT_EXTS)]
    if not targets:
        return ""
    if shutil.which("yq"):
        return "yq '.' " + _quoted(targets)
    checks = []
    json_files = [f for f in targets if f.endswith(".json")]
    if json_files and shutil.which("jq"):
        checks.append("jq empty " + _quoted(json_files))
    toml_files = [f for f in targets if f.endswith(".toml")]
    if toml_files:
        # The handles are left to interpreter exit on purpose: this is a
        # one-shot validation subprocess that does nothing else and then
        # terminates, so a `with` per file would buy nothing a process exit
        # does not already guarantee, at the cost of embedding newlines in a
        # shell-quoted `-c` argument.
        checks.append(
            'python3 -c "import tomllib, sys; '
            "[tomllib.load(open(f, 'rb')) for f in sys.argv[1:]]\" " + _quoted(toml_files)
        )
    xml_files = [f for f in targets if f.endswith(".xml")]
    if xml_files and shutil.which("xmllint"):
        checks.append("xmllint --noout " + _quoted(xml_files))
    return " && ".join(checks)


def build_prose_validator(files: list[str]) -> str:
    """A narrow, real parse check for Markdown -- not a style opinion.

    See the module docstring for what this does and does not catch, measured
    rather than assumed.
    """
    targets = [f for f in files if f.endswith(_PROSE_EXTS)]
    if not targets or not shutil.which("pandoc"):
        return ""
    return "pandoc -f gfm -t plain -o /dev/null " + _quoted(targets)
