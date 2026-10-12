---
name: verifying-your-work
description: Checking that code actually works, using the fewest round trips.
triggers: ["*.py", test, verify, check, pytest, assert, coverage, failing]
phases: [execute, fix]
---

Each tool call costs about a second of latency regardless of how much it does,
and that overhead — not generation — is what makes a step slow. Verification is
where calls get wasted, so spend them deliberately.

## Use the right tool

- `run_tests` — runs pytest and returns a parsed summary plus the failing
  assertions. Prefer it over `run_bash` with pytest: pytest is always available
  through it, so you never need to install anything first.
- `run_python` — runs a Python snippet with the workspace importable. Use it for
  a quick behavioural check instead of `run_bash` with `python -c "..."`. No
  shell quoting to get wrong — same approval gate as run_bash, though, since
  it is just as capable of anything a shell command is.
- `check_code` — Ruff, and ty when you pass `types=true`.

## Spend calls well

- **Batch your edits.** `edit_files` takes a list of search/replace edits, to one file or
  several, in one call. If you know a file needs three changes, send three edits in one
  `edit_files` call rather than three `edit_file` calls — the overhead is per call, not per
  edit.
- **One check, not three.** Put every assertion you want into a single
  `run_python` snippet and print all of it, rather than one call per fact.

  ```python
  from semver import compare, parse

  print(parse("1.2.3-alpha.1+b.5"))
  print(compare("1.0.0-alpha", "1.0.0"), compare("1.0.0-beta.2", "1.0.0-beta.11"))
  for bad in ["1.2", "01.2.3", "1.2.3-"]:
      try:
          parse(bad)
          print("ACCEPTED (wrong):", bad)
      except ValueError:
          print("rejected:", bad)
  ```

- **Do not re-verify what a tool already told you.** Writes are syntax-checked
  and linted automatically; if the write came back clean, it is clean.
- **Narrow a slow suite instead of re-running all of it.** `run_tests` with
  `pattern=` runs only the tests matching it (pytest's `-k`), and `check_code`
  (Ruff, ty with `types=true`) answers most structural questions — a bad
  import, an undefined name, a signature mismatch — far cheaper than a full
  suite that fits real models or hits real data. While iterating on one
  failing test, run only that test by name until it passes; run the whole
  file once at the end to confirm nothing else broke. A suite that takes
  20+ seconds because it does real, unmocked work (see the test-speed skill)
  makes this the difference between one slow run and five.
- **Test the edges the task names.** Specifications that mention rejecting bad
  input are describing where the bugs are. Check those cases specifically —
  the happy path is rarely where a step fails.
- **When a test fails, read the assertion before editing.** `run_tests` returns
  it. Guessing at a fix and re-running costs two calls; reading costs none.

## Writing a test that would actually have caught the bug

A passing test proves nothing unless it would have failed on the broken code.
These are the ways a test looks thorough and checks nothing, all observed in
real runs of this harness:

The executor's own rules already cover setting internal state by hand and
asserting that a setting was stored rather than used. Three more:

**Testing a unit the integration never calls that way.** `Bird.tap(prefs)`
worked perfectly. Nothing passed `prefs` to it. Cover at least one path
end-to-end through the entry point a person actually uses.

**Tautological: recomputing the expected value the way the code does.** If the
expected value is derived by the same arithmetic the function uses, the test
passes by construction and can never disagree with the code. The worst version
copies the implementation into the test file: one run added a `_get_wing_points`
helper to the test module that reproduced the polygon maths from `Bird.draw`,
then asserted the helper against hardcoded points. Six tests passed, `draw` was
never called by any of them, and ten consecutive runs reported success while
the wing was not on screen. Expected values must come from somewhere
independent — a literal you worked out by hand, a figure from the spec, a
known-good sample. `assert total([{"price": 10}, {"price": 5}]) == 15`, never
`== sum(i["price"] for i in items)`.

**A function that draws, prints, or writes is still testable.** Pass it a
recording double — an object that logs the calls made to it — and assert on
what it was asked to draw. That executes the real function, which asserting on
a copy of its arithmetic does not.
