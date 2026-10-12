# SciCode driver

`sci.py` runs the harness once per SciCode main problem and scores each sub-step with SciCode's own
tests and reference data, which the agent never sees (DESIGN-LOG §76, §81, §84-§89).

- Data (outside git: the problems files hold the gold solutions): `SCI_DATA`, default
  `traces/scicode-data/` with `problems_{dev,test}.jsonl`, `drive/test_data.h5` (1 GB, from the
  SciCode release) and `repo/` (the scicode package, for scoring).
- Runs: `SCI_RUNS`, default `~/.cache/mercury-harness/scicode-runs/`. Every workspace gets a
  `.venv` with numpy/scipy, about 1 GB a round, so keep runs out of the repo.
- Scoring env, once: `python sci.py env` (builds `$SCI_DATA/score-env`).
- A round: `python sci.py setup ROUND dev`, then `SCI_MODELS="--all-mercury ..." python sci.py run
  ROUND`, then `python sci.py score ROUND > ROUND.out`. `SCI_PARALLEL` (default 3) and
  `SCI_HARNESS` (default the repo's `.venv/bin/harness`).
- `--web-deny` for SciCode's own sources is always passed (`WEB_DENY`, §80).
- `classify.py RUN_DIR ...`: why failing sub-steps failed (wrong numbers vs code that did not run).
- `sample.py run ROUND SPLIT [IDS]` (then `sci.py score ROUND`): SciCode as it is officially prompted, one
  sub-step at a time with SciCode's own template, the bare-model baseline. `SCI_BACKGROUND=1` adds the
  scientist-annotated background (also honoured by `sci.py setup` for the harness). `SCI_SAMPLES=K` is a
  test-run option, not a harness feature: K candidates a step, chosen by agreement on probe inputs
  (+4.3 sub-steps of 48 at 4.4x the cost, no more whole problems; DESIGN-LOG §130.1).

Past rounds' traces, scores and solutions are archived under `traces/scratch-archive/`.
