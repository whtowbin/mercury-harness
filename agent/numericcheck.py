"""Numeric self-check: run the code on a case whose answer is known before finishing a step.

On SciCode 139 of 156 failing sub-steps returned a wrong number from code that ran, and none
misused a library call (DESIGN-LOG §79.2). Research cannot catch those; running the function
on a case with an independently known answer can: a limiting case, a symmetry, a conservation
law, a tiny input worked by hand, or a slower direct method.

`Settings.numeric_check` appends EXECUTOR_NUMERIC_CHECK to the executor's system prompt and
puts a `NumericCheckGate` on each step attempt. The gate holds `finish_step` once when the
step wrote code and ran nothing since its last edit. It cannot judge whether a check was a
good one; it only makes sure one was run. A prompt alone was not enough for the analogous
research instruction: Mercury skipped it in 6-12 of 15 runs until a gate held (§77.2).
"""

from __future__ import annotations

import json

from .editcalls import EDIT_TOOLS

EXECUTOR_NUMERIC_CHECK = """
**Check the numbers before you finish.** Code that runs can still return the wrong number.
Before finish_step, call the function you wrote with run_python on a case whose answer you
know without trusting this code: a limiting case (zero coupling, one particle, t = 0), a
symmetry (swapping inputs, reversing direction), a conservation law (energy, mass,
normalization), a tiny input worked out by hand, or a slower direct method to compare
against. If it disagrees, fix the code, not the check. Say in finish_step's notes which
check you ran and what it gave.
"""

# Calls that run code (editcalls.EDIT_TOOLS are the ones that change a source file).
_RUNS = frozenset({"run_python", "run_tests", "run_bash", "trace_python", "test_project"})


def _writes_python(call: dict) -> bool:
    function = call.get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return True
    text = json.dumps(args)
    return ".py" in text


class NumericCheckGate:
    """Holds a step's finish_step once when it wrote Python and ran nothing since."""

    event = "numeric_check_held"

    def __init__(self) -> None:
        self.unchecked_edit = False
        self.held = False

    def observe(self, call: dict) -> None:
        name = (call.get("function") or {}).get("name")
        if name in EDIT_TOOLS and _writes_python(call):
            self.unchecked_edit = True
        elif name in _RUNS:
            self.ran()

    def ran(self) -> None:
        """The code ran since its last edit. `observe` calls this for the model's own
        run tools; a check the harness runs itself (the step's tests after an edit,
        DESIGN-LOG §90) never passes through `observe`, so it calls this directly, or
        the gate would hold a step that was in fact checked."""
        self.unchecked_edit = False

    def hold(self, call: dict) -> str | None:
        name = (call.get("function") or {}).get("name")
        if name != "finish_step" or self.held or not self.unchecked_edit:
            return None
        self.held = True
        return (
            "[harness] Step NOT finished: the code changed and nothing has run since. Call "
            "the function with run_python on a case whose answer you know independently: a "
            "limiting case, a symmetry, a conservation law, a tiny input worked by hand, or a "
            "slower direct method. Fix the code if it disagrees, then call finish_step again "
            "and say in notes which check you ran and what it gave."
        )
