---
name: ruff
description: Linting and formatting Python with Ruff, via the harness's check_code tool.
triggers: ["*.py", ruff, lint, format, style, unused]
phases: [execute, fix]
yields_to: [project-linters]
---

Ruff is an extremely fast Python linter and formatter, replacing Flake8, isort,
Black and pyupgrade. Adapted from Astral's ruff skill for this harness.

## In this workspace specifically

`check_code(fix=true)` applies Ruff's safe fixes and formats. Call `check_code`
for a pass over several files; a single write was already linted.

## Reading a diagnostic

Rule prefixes tell you whether something is a real defect:

- `F` — pyflakes. Real problems: `F821` undefined name, `F401` unused import.
- `E9` — syntax and IO errors. Always real.
- `B` — bugbear. Likely bugs: mutable default arguments, unused loop variables.
- `E`/`W` — style, mostly line length. Cosmetic.
- `I` — import sorting. Cosmetic, and `fix=true` handles it.

Fix `F`, `E9` and `B` findings. Do not spend calls chasing `E501` line lengths
unless the project's own config selects them and the file you touched is failing.

## If you do shell out

```bash
uv run ruff check .              # lint
uv run ruff check --fix .        # apply safe fixes
uv run ruff format .             # format
uv run ruff rule F821            # explain a rule
```

Do not reformat a whole file the project never formatted — it buries your actual
change in noise. Scope fixes to the code you edited.
