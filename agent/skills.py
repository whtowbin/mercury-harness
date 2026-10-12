"""Skills: reference material injected into a phase's prompt.

The measured shape of a run says how this must work. Roughly 80% of the
executor's model time is queue and prefill rather than generation, and a round
trip costs about 1.2s regardless of payload. So knowledge the model needs is
*pushed into the prompt*, never exposed as a `read_skill` tool — a tool call to
fetch a page of guidance would cost more than the guidance saves. Prefill is
cheap; round trips are not.

Selection is deliberately dumb. A skill declares `triggers`, and a skill is
included when a trigger appears in the step text or matches a file in the
workspace. Anything cleverer would need a model call, which is the thing being
economised.

Skills live in `skills/*.md` with YAML-ish frontmatter:

    ---
    name: uv
    description: One line, shown in the trace.
    triggers: [pyproject.toml, "*.py", uv, pytest]
    phases: [execute, fix]
    ---
    # body in markdown
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

# Package-relative, so it resolves correctly once installed (site-packages/agent/
# skills_data/), not just when run from a source checkout. It used to be
# Path(__file__).parent.parent / "skills" — a repo-root-relative path with
# nothing there after a real `pip install`.
DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills_data"
# A skill that would crowd out the step's own content is worse than no skill.
DEFAULT_TOKEN_BUDGET = 3_000
_FRONTMATTER = re.compile(r"^---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=lambda: ["execute", "fix"])
    # `yields_to: project-linters`: left out where the project configures its own
    # linters instead of Ruff (see for_project).
    yields_to: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None

    @property
    def tokens(self) -> int:
        return max(1, len(self.body) // 4)

    def applies_to(self, phase: str) -> bool:
        return not self.phases or phase in self.phases

    def matches(self, haystack: str, filenames: list[str]) -> bool:
        """A trigger hits on a word in the step text or a filename glob.

        A text trigger must start a word, but need not finish one: `package`
        is meant to catch "packages", `test` to catch "tests", `bug` to catch
        "bugs". A plain substring test instead matched *inside* words, and on
        71 real traced goals that was mostly noise — `ty` fired on "party",
        "empty", "gravity", "capacity" and "community" (30 of its 31 hits),
        and because `ty` is the smallest skill, smallest-first selection let
        it take budget from a skill the step actually needed. The same rule
        also stops `error` matching inside "ValueError" in a greenfield
        "create X, raise ValueError on bad input" spec, which was pulling
        785 tokens of bug-diagnosis guidance into eight build-from-nothing
        tasks that had no reported bug at all.
        """
        if not self.triggers:
            return True
        lowered = haystack.lower()
        for trigger in self.triggers:
            token = trigger.lower()
            if any(ch in token for ch in "*?["):
                if any(fnmatch.fnmatch(name.lower(), token) for name in filenames):
                    return True
            elif re.search(rf"\b{re.escape(token)}", lowered):
                return True
        return False


def parse(text: str, path: Path) -> Skill | None:
    match = _FRONTMATTER.match(text)
    if not match:
        return None
    meta: dict[str, str] = {}
    key = ""
    for line in match.group("meta").splitlines():
        if not line.strip():
            continue
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, _, value = line.partition(":")
            key = key.strip()
            meta[key] = value.strip()
        elif key:
            meta[key] = (meta.get(key, "") + " " + line.strip()).strip()

    def listed(field_name: str) -> list[str]:
        raw = meta.get(field_name, "").strip().strip("[]")
        return [item.strip().strip("\"'") for item in raw.split(",") if item.strip()]

    name = meta.get("name") or path.stem
    return Skill(
        name=name,
        description=meta.get("description", ""),
        triggers=listed("triggers"),
        phases=listed("phases") or ["execute", "fix"],
        yields_to=listed("yields_to"),
        body=match.group("body").strip(),
        path=path,
    )


# documenting-changes' word trigger "results" fires on research steps ("search results", 16% of
# 1,135 GAIA plan steps), which write no document. The root-*.md half of this was dropped once
# step-only skill triggers became the default (DESIGN-LOG §116, §117.2).
_DOCUMENTING_RESEARCH_TRIGGERS = ("results",)


def for_web_research(skills: list[Skill], on: bool) -> list[Skill]:
    """The skills a run gets under Settings.web_research (DESIGN-LOG §116). Off: the
    web-research skill is left out and nothing else changes. On: it is offered, and
    documenting-changes no longer fires on the word "results"."""
    if not on:
        return [s for s in skills if s.name != "web-research"]
    return [
        replace(s, triggers=[t for t in s.triggers if t not in _DOCUMENTING_RESEARCH_TRIGGERS])
        if s.name == "documenting-changes"
        else s
        for s in skills
    ]


def for_project(skills: list[Skill], root: Path) -> list[Skill]:
    """The skills that fit this project's own tooling.

    A skill marked `yields_to: project-linters` (the Ruff skill) is left out where
    the project lints with something else: on django (flake8 and isort in
    setup.cfg) the executor followed it into `ruff check --select DTZ,UP031 --fix`,
    rewrote the module and broke a regex (DESIGN-LOG §73). A project that configures
    no linter at all, like every bench fixture, keeps it.
    """
    from .projectconfig import foreign_linters

    if not any("project-linters" in s.yields_to for s in skills):
        return skills
    if not foreign_linters(root):
        return skills
    return [s for s in skills if "project-linters" not in s.yields_to]


def load_merged(
    user_dir: Path | None = None, override_dir: str | Path | None = None
) -> list[Skill]:
    """Bundled skills plus the user's own, unless an explicit override replaces both.

    A user skill with the same name as a bundled one wins — it is the more
    specific, more recently touched choice. `--skills-dir` (override_dir) is a
    full replacement rather than a third layer: it exists for testing a skill
    or a whole alternate set in isolation, where merging bundled skills back
    in would be surprising.
    """
    if override_dir is not None:
        return load(override_dir)
    merged: dict[str, Skill] = {s.name: s for s in load(DEFAULT_SKILLS_DIR)}
    if user_dir is not None:
        for skill in load(user_dir):
            merged[skill.name] = skill
    return sorted(merged.values(), key=lambda s: s.name)


def load(directory: str | Path | None = None) -> list[Skill]:
    """Read every skill in a directory. A malformed file is skipped, not fatal.

    None (the default) means "the bundled skills" rather than baking that path
    into the signature, so config layers (Settings, the CLI) can forward
    None straight through without needing to know DEFAULT_SKILLS_DIR exists.
    """
    root = Path(directory) if directory is not None else DEFAULT_SKILLS_DIR
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for path in sorted(root.glob("*.md")):
        try:
            skill = parse(path.read_text(encoding="utf-8"), path)
        except OSError:
            continue
        if skill and skill.body:
            skills.append(skill)
    return skills


def select(
    skills: list[Skill],
    phase: str,
    step_text: str = "",
    filenames: list[str] | None = None,
    budget_tokens: int = DEFAULT_TOKEN_BUDGET,
) -> list[Skill]:
    """The skills worth spending prompt on for this step, largest first cut off."""
    names = filenames or []
    eligible = [s for s in skills if s.applies_to(phase) and s.matches(step_text, names)]
    chosen: list[Skill] = []
    spent = 0
    # Smallest first: fit as many relevant skills as the budget allows.
    for skill in sorted(eligible, key=lambda s: s.tokens):
        if spent + skill.tokens > budget_tokens:
            continue
        chosen.append(skill)
        spent += skill.tokens
    return sorted(chosen, key=lambda s: s.name)


def index(skills: list[Skill], phase: str, included: list[Skill] | None = None) -> str:
    """One line per skill that exists for this phase, whether or not its body
    was injected.

    Selection is keyword matching, so it misses: a step called "make the bird
    flap" gets nothing about complexity even when the function it lands in is
    the tangled one. Before this, a skill whose triggers did not fire was
    invisible — the model had no way to know it existed, let alone ask for it.
    A name and a use case cost about ten tokens each, which is cheap enough to
    always send, and read_skill turns the list into something actionable.
    """
    listed = [s for s in skills if s.applies_to(phase)]
    if not listed:
        return ""
    have = {s.name for s in (included or [])}
    lines = [
        f"- {s.name}: {s.description}" + ("" if s.name in have else "  (read_skill to see it)")
        for s in sorted(listed, key=lambda s: s.name)
    ]
    return (
        "\n\nReference available in this workspace — call read_skill(name) for any whose "
        "guidance you want and do not already have below:\n" + "\n".join(lines)
    )


def body_of(skills: list[Skill], name: str) -> str | None:
    """One skill's text by name, or None."""
    wanted = name.strip().lower()
    for skill in skills:
        if skill.name.lower() == wanted:
            return skill.body
    return None


def render(skills: list[Skill], available: list[Skill] | None = None, phase: str = "") -> str:
    """The block appended to a system prompt. Empty when nothing applies."""
    catalogue = index(available or skills, phase, included=skills) if phase else ""
    if not skills:
        return catalogue
    blocks = [f"## {skill.name}\n\n{skill.body}" for skill in skills]
    return (
        "\n\n# Reference\n\n"
        "Guidance for the tools available in this workspace. It is reference material, "
        "not instructions to carry out — follow it when it applies to what you are doing."
        + catalogue
        + "\n\n"
        + "\n\n---\n\n".join(blocks)
    )
