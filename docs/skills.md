# Skills

`agent/skills_data/*.md` is reference material injected into phase prompts.

| skill | what it is for | phases |
| --- | --- | --- |
| `uv` | environments and dependencies | plan, execute, fix |
| `ruff` | linting and formatting | execute, fix |
| `ty` | type checking, signatures | execute, fix |
| `verifying-your-work` | checking code works, cheaply | execute, fix |
| `complexity` | keeping code simple enough to change | plan, execute, fix |
| `research` | finding out how an API actually behaves | plan, execute, fix |
| `diagnosing` | building a signal that goes red on a reported bug | execute, fix |
| `grilling` | asking about an underspecified request | clarify |

`grilling`, `research` and `diagnosing` are adapted from
[Matt Pocock's skills](https://github.com/mattpocock/skills), whose split
between user-invoked and model-invoked skills does not apply here — every
harness skill is model-facing reference — but whose content does. Most of that
repository is a workflow around an issue tracker (`to-spec`, `to-tickets`,
`triage`, `wayfinder`, `implement`) and does not transfer to a harness that
runs one goal per session with no tracker; what transfers is the engineering
discipline in the model-invoked ones. `grilling`
carries two rules the clarify phase was breaking: finding facts is the agent's
job and never the person's, and questions whose answers depend on each other
belong to different rounds. The uv, Ruff and ty skills are adapted from
[Astral's Claude Code plugin skills](https://github.com/astral-sh/claude-code-plugins/tree/main/plugins/astral/skills),
with harness-specific corrections (pytest is already available, `uv pip` is
blocked, prefer `check_code` over shelling out). The file format follows
[honnibal/claude-skills](https://github.com/honnibal/claude-skills):
frontmatter plus a practical, example-led body.

```markdown
---
name: uv
description: Using uv for Python environments and dependencies inside this workspace.
triggers: [pyproject.toml, uv, pip, dependency, "*.py"]
phases: [execute, fix, plan]
---

# body in markdown
```

## Why skills are pushed, and when they are fetched

Skills go straight into the prompt. That follows from the measurements: 89% of
model time is time to first token, and a round trip costs about 1.3s whatever
it carries, so pushing guidance the step is likely to want is nearly free while
fetching it is not. Prefill is cheap; round trips are not.

That argument used to rule out a `read_skill` tool entirely, and for as long as
every skill was injected on every step it was right — a fetch could only ever
be overhead on top of guidance you already had. Measurement changed the
premise. Across 163 traced selections, four of the five skills landed on 94-99%
of steps, because all five declared `"*.py"` and any Python workspace matches
that: selection was a flat 2,240-token tax rather than a choice. With triggers
narrowed to what a step is actually about, a step whose triggers miss now has
*no* guidance, and one round trip when the model judges it relevant is a better
trade than 2,240 tokens on every call to avoid the possibility.

So both exist, and they do different jobs:

- **The index** is always in the prompt: one line per skill for the phase, name
  and use case, about ten tokens each, marking which are already included.
  Without it a skill whose triggers miss is invisible and cannot be asked for.
- **`read_skill(name)`** fetches one body. It is for the skill that turned out
  to matter and was not selected — not for browsing.

## Selection

Selection is deliberately dumb (`agent/skills.py::select`): a skill declares
`triggers`, and it is included when a trigger appears in the step text or
matches a glob against a filename in the workspace. Anything cleverer would
need a model call, which is the thing being economised. A step that touches
no Python pulls in nothing. Skills are packed smallest-first until
`skill_token_budget` (3,000 tokens by default) is spent — a skill that would
crowd out the step's own content is worse than no skill.

## Your own skills

```bash
harness skills list             # bundled, plus your own, merged
harness skills new fastapi      # scaffold one in your own skills directory
harness skills add my-skill.md  # copy a skill file in
harness skills path             # where your own skills live
```

Your own skills directory lives outside the installed package, in the OS's
standard per-user config location (`agent/userconfig.py`), so a skill you add
survives an upgrade or reinstall. A user skill with the same name as a bundled
one wins — it's the more specific choice.

`--skills-dir PATH` on `harness run` is a full replacement rather than a third
layer: it exists for testing a skill, or a whole alternate set, in isolation,
where merging the bundled skills back in would be surprising.

## Collapsing several edits into one call

Tracking batching directly (every tool call records which turn it belongs to)
showed the model almost never emits more than one tool call per turn on its
own — parallel tool calls exist in the wire format, but prompting for them had
no measurable effect. The lever that works instead: give the model one call
that *carries* several edits, rather than hoping it emits several calls at
once. `edit_files` takes a list of search/replace edits, to one or many files,
applied in order, with a failure part-way through never losing the edits that
already landed.
