# django SWE-bench driver

`swe.py` runs the harness on django SWE-bench tasks in fresh git worktrees and scores each with the
task's FAIL_TO_PASS and PASS_TO_PASS tests (DESIGN-LOG §69-§76, §78, §88).

- Data (outside git: the task JSONs hold the gold patches): `SWE_DATA`, default
  `traces/swebench-data/` with `django__django-*.json`, goal texts and a bare `django.git`.
- Runs: `SWE_RUNS`, default `~/.cache/mercury-harness/swebench-runs/` (full django checkouts).
- A round: `python swe.py setup ROUND ID ...`, `SWE_MODELS="..." python swe.py run ROUND ID ...`,
  `python swe.py score ROUND ID ...`. `SWE_GOAL=goal-hidden` runs the spec-hiding variant and
  `SWE_ASK=1` answers questions with `amb.py` (the simulated issue author, §78).
- `--web-deny` for django's sources and SWE-bench itself is always passed (§80).

Past rounds' traces, scores and per-workspace diffs are archived under `traces/scratch-archive/`.
