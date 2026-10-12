---
name: ty
description: Type checking Python with ty, via the harness's check_code tool.
triggers: [ty, type, annotation, signature, mypy, typing, protocol, generic]
phases: [execute, fix]
---

ty is an extremely fast Python type checker, replacing mypy and Pyright.
Adapted from Astral's ty skill for this harness.

## In this workspace specifically

**Use `check_code(types=true)`, not `run_bash`.** It runs ty across the files
changed so far and returns parsed diagnostics.

**It is not automatic, and that is deliberate.** The syntax check and Ruff run
after every write because they cost milliseconds. ty takes about a second
because it resolves imports across the whole project, so you call it when a
step is complete — not after every edit.

**What it is worth calling for.** ty catches the class of defect a single-file
view cannot: a call site disagreeing with a signature, a return type that does
not match what the function returns, an attribute that does not exist on a type.
If a step changed a function's parameters or return type, run it before
finishing. For a self-contained new file with no callers yet, it will rarely
tell you anything Ruff did not.

## If you do shell out

```bash
uv run ty check                       # check the project
uv run ty check path/to/file.py       # check one file
uv run ty check --python-version 3.12
```

## Configuration

```toml
[tool.ty.environment]
python-version = "3.12"

[tool.ty.rules]
possibly-unresolved-reference = "warn"

[tool.ty.src]
exclude = ["**/migrations/**"]
```
