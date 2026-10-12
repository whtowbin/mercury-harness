# mercury-harness

A small coding agent for testing **Mercury 2.5** (Inception's diffusion LLM) on
OpenRouter, built around one hypothesis: a dLLM is fast and cheap but benefits
from someone else doing the thinking, so give it *small, fully specified units
of work* and check the result with a model that can see everything at once.

Three phases, and a check that makes no model call:

| phase | default model | context | why |
| --- | --- | --- | --- |
| **1. plan** | Mercury 2.5 (`--planner` takes any model) | read-only exploration | Chain-of-Draft prompting: terse ≤5-word reasoning notes, then a structured plan of small, self-contained steps |
| **2. execute** | Mercury 2.5 (`--executor`) | small chunks, one step at a time | each step gets only its own files and the summaries of prior steps |
| **3. review** (off by default; `--review` turns it on) | Mercury 2.5 (`--reviewer`) | large chunks, whole files | catches cross-file defects the executor structurally could not see, then hands each finding back to Mercury as a scoped fix |
| **final check** (always on) | no model | the finished workspace | re-runs every step's verify command, the project's tests and a type check; failures go back through the fix path |

Every phase uses Mercury unless you name another model. Review is off because it cost about a
fifth of the spend and time and no suite showed a loss without it (DESIGN-LOG §124.10). A stronger
planner or reviewer is a flag away, for work that is ambiguous or spans files; the measurements
behind both defaults are in {doc}`configuration-comparison`.

This is research code. The measurements are the part most worth reading: start with
`FINDINGS.md` in the repository root, then
{doc}`reproduce` to rerun them.

```{toctree}
:maxdepth: 2
:caption: Guide

quickstart
usage
cli
architecture
skills
benchmarking
configuration-comparison
reproduce
safety
development
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/agent
api/bench
```
