---
name: documenting-changes
description: Keeping README/design-log claims real and current, not stale or placeholder.
triggers: ["*.md", readme, changelog, "design log", "design-log", document, documentation, "write up", "results"]
phases: [execute, fix]
---

## A number in a document is a claim, not decoration

If a step writes a specific result into a README, a changelog, or a design
log — an RMSE, a percentage, a count, a benchmark time — that number must
come from actually running the current code in this session. Never carry
one forward from memory, an earlier draft, or a plausible-looking estimate
meant to be replaced "later": nothing ever flags a realistic-looking
placeholder for replacement, which is exactly how one shipped as fact in a
real run. The value the tests skill already applies to a hardcoded expected
value applies here too — it has to come from somewhere independent of the
document itself, and here that means a script run, not a guess.

Before finishing a step that reports a result:
1. Run the code that produces it, in this session, right before writing it
   down.
2. Copy the number the run actually printed. Not a rounded-off memory of a
   similar number from earlier context, not what the number "should" be if
   the code works as intended.
3. If you cannot run the code (a missing dependency, no data file, a genuine
   environmental block), say that in the document and in `finish_step`'s
   notes — a labeled "not yet run" is honest; an invented number in its
   place is not, and is worse than leaving it blank.

## Stale documentation is a defect the step created

A change that makes an existing README/design-log claim false — a feature
count, a described behavior, a usage example — is not finished until that
claim is corrected, even if the goal never mentioned the document. Fixing
`compare_fitting_methods.py` to add a seventh model and leaving the README's
"six models" sentence untouched ships a document that contradicts the code
the moment the step lands, not eventually. Search for what your change makes
false before calling it done, the same way you would check a signature's
call sites.

## Where a change gets recorded

Check what the project already does before inventing a new place: a
`DESIGN-LOG.md` or `CHANGELOG.md` already being used for narrative history
is where a change belongs, not a fresh file competing with it. Absent an
existing convention, the README is the right place for anything a person
using the project would want to see (current results, how to run it, what
it does) — a separate design-log entry is worth adding only for the *why*
behind a nontrivial decision, not for restating the diff.
