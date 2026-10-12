---
name: complexity
description: Keeping code simple enough to change, measured with complexipy's cognitive complexity.
triggers: [complex, complexity, refactor, nested, simplify, readable, cognitive, tangled, "clean up"]
phases: [plan, execute, fix]
---

Code generated quickly tends to accrete branches: an `if` for each case as it
is discovered, a nested loop where a comprehension would do, a function that
grew a fourth responsibility because that was where the data already was. Each
addition is locally reasonable and the total is unmaintainable. This skill is
about noticing that while you are writing, not after.

## Cognitive complexity, not cyclomatic

Cyclomatic complexity counts branches — it answers "how many test cases does
this need". **Cognitive complexity** asks a different question: how hard is
this to hold in your head? It penalises what actually costs a reader:

- **Nesting.** Each level of `if`/`for`/`while`/`try` inside another adds more
  than the one before it. Depth is the single biggest driver.
- **Flow breaks.** `break`, `continue`, and early `return` inside nesting.
- **Sequences of boolean operators.** `a and b or c and not d` counts once per
  change of operator, because that is where a reader has to stop and parse.

A function can have low cyclomatic and high cognitive complexity (deeply
nested but few distinct paths) or the reverse (a flat `match` with twenty
arms, which is trivial to read). Optimise for the reader.

## Measuring it

`complexipy` is a Rust-based analyser, fast enough to run on every write:

```bash
uv run complexipy .                              # whole project
uv run complexipy path/to/file.py                # one file
uv run complexipy . --max-complexity-allowed 15  # custom threshold
uv run complexipy . --failed --suggest-refactors # only what exceeds it, with hints
uv run complexipy . --output-format json         # machine-readable
uv run complexipy . --diff main                  # only what this branch changed
```

Its default threshold is 15 per function. Treat that as a ceiling, not a
target: most functions should be well under it, and a function at 14 is
usually already asking to be split.

**Prefer `check_code` for lint and types** — complexipy is a separate tool and
needs `run_bash`, so run it when complexity is the actual question (a function
you suspect is tangled, a refactor you want to justify), not reflexively after
every edit.

## What to do when a function scores badly

In rough order of how much they buy for how little risk:

1. **Guard clauses.** Replace `if valid: <30 lines>` with
   `if not valid: return ...` and dedent the body. This removes a nesting
   level from everything below it, which is the most expensive kind of
   complexity.
2. **Extract the inner loop or branch** into a named function. The name is
   doing real work: it tells the reader what the block was *for*, which the
   inlined version made them infer.
3. **Replace flag arguments with separate functions.** `def f(x, fast=False)`
   whose body is one big `if fast:` is two functions sharing a name.
4. **Lift repeated condition checks.** Compute once above the loop rather than
   re-deriving inside it.
5. **Replace a chain of `elif` on the same value** with a dict lookup or
   `match` — flat dispatch reads far more easily than a ladder.

What *not* to do: do not split a function purely to make a number go down. A
helper called from exactly one place, whose name restates its body, has moved
the complexity rather than removed it and cost a reader an extra jump. If you
cannot name the extracted piece in terms of what it means, it is not a seam.

## In this harness specifically

- The reviewer should report genuinely tangled code only when it is a defect
  risk — a function so nested that a bug is plausibly hiding in it — not as a
  style preference. Complexity findings compete for attention with real
  breakage.
- When a step's intent is "refactor X", measure before and after and say both
  numbers in `finish_step`'s summary. "Reduced `parse_config` from 22 to 7"
  is a verifiable claim; "simplified the parser" is not.
- Prefer writing the simple version first. Nothing here is as cheap as not
  creating the complexity: a flat function written deliberately costs nothing,
  and the same logic refactored later costs a read, an edit, and a re-test.
