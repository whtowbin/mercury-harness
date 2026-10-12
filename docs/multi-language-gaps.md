# Multi-language gaps

What is Python-only today, and what each further language would need. This is
a checklist, not a plan: nothing here is built, and per DESIGN-LOG's rule none
of it should be built for a language until a real trace of that language shows
the gap costing something.

## What already exists

`agent/langtools.py` has `LanguageProfile` rows for Python, Node, Go and Rust
(marker file, test-file naming, lint command, test command, repair command),
with `scc` as a fallback detector. That covers *detecting* a language and
*naming* its commands. Everything below is what those profiles do not reach.

## Python-only components

| Component | Where | Python-specific because | What another language needs |
|---|---|---|---|
| Symbol outline for the planner | `agent/outline.py` | Uses `ast`; other languages get the tree and no symbols. | A generic extractor: `ctags` (one tool, many languages) or tree-sitter. Output shape stays "file: top-level symbols". |
| Structured test runner | `run_tests` in `agent/tools/checks.py` | Runs pytest and parses its summary. | Per-runner adapter returning the same result shape: `jest`/`vitest`, `go test`, `cargo test`. Each has a machine-readable mode (`--json`, `-json`, `--message-format=json`) that beats scraping text. |
| Failure signature nudge | `_failure_details`, `_FRAME_RE` in `agent/runner.py` | Regexes match pytest `--tb=short` (`file.py:N: in func`, `E   msg`). | Either the same adapters above feed it (failed ids, count, deepest frame, error line), or a generic extractor: last few `file:line` frames plus the final error line. This is the one place a judge earns its cost, on output no adapter parses. |
| Lint/type check | `check_code` | ruff and ty. | Profile already names eslint, `go vet`, clippy; add a type-check step (`tsc --noEmit`, `mypy` is Python's) and parse each tool's output to one shape. |
| Step tracer | `trace_python` | `sys.settrace`. | No cheap equivalent. Node has `--inspect`, Go has `delve`; low priority, since the probing loop it fixes is not proven outside Python. |
| Environment failure diagnosis | `agent/envfail.py` | Crash signatures and dependency-drip taxonomy come from real Python traces. | Build from real traced Node/Go/Rust failures, not by guessing; `langtools.py`'s docstring already says this. |
| Test-file detection | `is_test_file` | Rust has no filename convention (inline `#[cfg(test)]`). | Content-based detection for Rust; leave empty until a Rust trace needs it. |
| Data-format validators | `agent/dataformats.py` | Language-neutral already. | None. |
| Bench tasks | `bench/tasks.py` | All Python. | One task per language, or the language column in bench results is meaningless. |
| Trace analysis | `agent/tracetools.py` (`plan_conformance`) | Checks names against Python signatures. | Reuse the generic outline extractor. |

## Order to build, if a second language is ever needed

1. Measure the share of real runs in that language (traces record the
   workspace; `langtools.detect_profile` names it). If under ~10%, stop.
2. Generic outline via `ctags`. Cheapest, and it benefits the planner alone.
3. Test-runner adapter for that language's JSON mode, feeding the existing
   failure-signature nudge.
4. Only then envfail signatures, from real failures.

## Not language-specific, so unaffected

Nudge budgets, relevance judge, goal loop, review chunking, trace audit.
