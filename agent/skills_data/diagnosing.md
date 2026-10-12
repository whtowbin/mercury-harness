---
name: diagnosing
description: Finding the cause of a reported bug by building a signal that goes red on it, before theorising. Adapted from Matt Pocock's diagnosing-bugs skill.
# Tuned against the goals people actually wrote in this harness's traces:
# a score that never increases, a feature still not appearing, something
# still happening when it should not.
# Bug reports are phrased as negations and persistence, not as the word "bug".
triggers: [bug, broken, crash, traceback, error, regression, diagnose, debug, investigate,
           fails, failing, fix, repair, wrong, incorrect, unexpected, never, still,
           "not appearing", "not added", "not working", "no effect", "nothing happens",
           "doesn't", "isn't", "won't", "shouldn't", shouldnt, slow]
phases: [execute, fix]
---

## Build the signal first. This is the whole skill.

Before forming any theory about the cause, get **one command you have already
run** that goes red on this specific bug and would go green once it is fixed.
Everything after that is mechanical: with a tight signal you will find the
cause, and without one no amount of reading the code will get you there.

If you catch yourself reading code to build a theory before that command
exists, stop. Guessing at a cause and editing to match it is the failure this
skill exists to prevent, and it is the expensive one — the edit looks
plausible, the tests stay green, and the bug is still there.

## What "red-capable" means

Not "runs without erroring". The command must drive the actual code path and
assert **the symptom the person described**:

- If they said the score never goes up, it plays enough of the game to pass a
  pipe and asserts the score changed.
- If they said the wing does not appear, it calls the drawing code with a
  recording surface and asserts a wing was drawn.
- If they said a setting has no effect, it sets the setting, runs the thing,
  and asserts the behaviour differs from the default.

A test that asserts something adjacent to the symptom is worse than none: it
goes green, and you conclude you are done.

## Ways to build one, roughly in order of preference

1. **A failing test** at whatever seam reaches the bug.
2. **`run_python`** driving the code directly with a fixture input, printing
   what actually happened. Faster to iterate than a test file, and it costs
   one call.
3. **A recording double.** For code that draws, prints, writes or sends: pass
   an object that logs the calls it receives, then assert on the log. This
   executes the real function, which is the point.
4. **A headless run.** Many UI and game libraries have a dummy backend —
   `SDL_VIDEODRIVER=dummy` for pygame, for instance — so the real draw path
   runs with no window.
5. **A loop over many inputs**, when the bug is "sometimes wrong": run it a
   few hundred times and count the failures rather than hunting one.
6. **`ask_user`**, last. Only when the symptom is genuinely unobservable from
   here — something visual you cannot render, or behaviour on hardware you do
   not have. Say precisely what you need them to look at and what would tell
   you which branch you are in.

## Then, and only then

Once the signal is red, shrink it: cut inputs, callers and setup one at a time,
re-running after each cut, keeping only what is load-bearing for the failure. A
small red repro usually names its own cause.

Fix the code, run the same command, and watch it go green. Say in your summary
what the command was and that you saw it change — "should work now" is not
evidence, and a fix verified only by a suite that was already green is not
verified at all.

## When you cannot build one

Say so plainly, list what you tried, and do not proceed to guess. An honest
"I could not reproduce this, here is what I ruled out" is worth more than a
speculative edit that makes the next person's diagnosis harder, because now
the code has changed for reasons nobody can reconstruct.
