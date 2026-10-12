---
name: keeping-tests-fast
description: Mocking the slow parts of a test without weakening what it checks.
triggers: ["test_*.py", "*_test.py", "conftest.py", mock, patch, slow, fixture, xdist, flaky]
phases: [execute, fix]
---

## Mock what is slow, not what you are testing

Mocking the wrong thing makes a test pass by construction; mocking the *right*
thing just makes it fast, because the code path you actually care about still
runs for real. The two are easy to tell apart: mock a dependency your
assertions never look at the internals of, never the thing the test exists to
check.

A traced run made this concrete. Two tests asserted the same kind of thing —
that `main()` prints certain section headers — by patching `evaluate_thermometer`
and `evaluate_barometer` so the assertions did not depend on real model output.
One of them stopped there and left `evaluate_tuned_thermometer` and
`evaluate_tuned_barometer` — real hyperparameter search — unmocked. It took
**22.6 seconds**. Its sibling patched those two as well, asserted the exact
same kind of thing, and took **0.9 seconds**. Neither test is about whether
tuning finds a good hyperparameter; both are about whether `main()` prints the
right headers when handed some result. The fast one mocked everything the
assertions do not examine and left the real thing they do examine — the
control flow and formatting inside `main()` — running for real.

What is worth mocking, because a test's assertions have no reason to depend on
its real behaviour: model fitting, hyperparameter search, or anything doing
real numerical work when the test is checking control flow around the result,
not the numbers themselves. A real network call, subprocess, or database —
these make a test flaky as well as slow, since they can fail for reasons that
have nothing to do with the code under test. Reading a large real data file
when a small in-memory fixture would exercise the same code path. A
deliberate `sleep` in the code being tested — patch the sleep, not the
function that calls it.

What that means in practice for `unittest.mock.patch`: patch the name **where
it is looked up**, not where it is defined. `patch("mymodule.slow_fit")`
works when `mymodule.py` does `from sklearn import slow_fit` and calls
`slow_fit(...)` directly; if it instead does `import sklearn` and calls
`sklearn.slow_fit(...)`, the patch target is `patch("mymodule.sklearn.slow_fit")`.
Patching the wrong one silently does nothing — the real function still runs,
the test is still slow, and it is easy to walk away believing the mock worked
because the test still passed.

## When the real thing has to run, run it once and share it

Sometimes mocking is the wrong tool: a test is specifically about the real
output of an expensive, deterministic computation — an RMSE a model
actually produced, not a plausible-looking stand-in for one — and mocking
it away would test nothing. That is not a reason to accept the slowness
three times over. If more than one test function needs that same result,
compute it once and share it with a `scope="session"` (or `scope="module"`)
pytest fixture, instead of every test that needs it calling the expensive
function itself.

A traced run made this concrete too, on the same project as the mocking
example above. Three test functions called `compare_task(...)` — which
fits 7 real regression models, one of them a 5-fold-cross-validated
`StackingRegressor` — with the exact same task name and loader each time;
a fourth call happened again inside `main()`. Every parameter that affects
the output (`random_state`, `cv`, the train/test split) was fixed in the
source, so all four calls were guaranteed to produce identical results —
recomputed from scratch every time regardless. Session-scoped fixtures for
the two tasks' results cut the file from **21.29s to 9.36s**, same five
tests, same assertions, still exercising the real model fit. The call
inside `main()` was handled differently: rather than skip it, the fixture's
result was fed through a stand-in that prints it in the same format the
real function does, so `main()`'s own control flow and output formatting —
what that particular test is actually about — still run for real; only the
redundant *re-fitting* was removed.

The tell that this applies, rather than plain mocking: the test's own
assertions read the real, computed values (an RMSE, a count, a specific
number) rather than merely checking that some function was called. If
several tests read the same real values, they should be reading the same
*computation*, run once.

## Mark what is genuinely, unavoidably slow

Mark a test `@pytest.mark.slow` if it is deliberately slow — a real
end-to-end run against real data, say, rather than something that could be
mocked without weakening what it checks. Register the marker in
`pyproject.toml` (`[tool.pytest.ini_options]` `markers = ["slow: ..."]`) and
set `addopts = "-m 'not slow'"` there so a normal run skips it by default; run
the slow ones on purpose with `pytest -m slow`. This is for tests that are
supposed to be slow, not a way to avoid mocking one that should not be —
reach for it only after checking that the slowness cannot be mocked away
without weakening the assertion.
