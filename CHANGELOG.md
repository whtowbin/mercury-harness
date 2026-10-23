# Changelog

## 0.2.1 (2026-10-23)

- Security: an approved `cd` in a wrapped test command may no longer contain `..`. With a symlink inside the
  workspace, `cd a/../../x` passed the check (which resolves the link first) while the shell, which applies `..` to
  the text first, ran it outside the workspace.

## 0.2.0 (2026-10-23)

Mercury 3 support and what came out of testing it (`DESIGN-LOG.md` §131).

- **Mercury 3**: model id `inception/mercury-3` with its OpenRouter prices ($0.10 input, $0.01
  cached input, $0.35 output per million tokens, the launch discount), and benchmark configs for it
  (`mercury3-only`, the isolated arm `h-m3`).
- **`expected_result_gate` now defaults to False.** With it off, Mercury 3 had 0.194 more perfect
  runs on the fixtures (§131.9).
- **Runaway reasoning step-down is on by default**: a call at medium or high effort that is cut
  for no content, or ends at the token cap with only reasoning, is asked again at low effort
  (§131.12).
- **Wrapped test commands are approved**: `cd tests && ...` and `... 2>&1 | tail -N`
  are approved unattended when every part would pass alone. Restrictions: only `&&` joins parts, `cd`
  comes first and must stay in the workspace, at most `2>&1` and one final `| head` or `| tail`
  with a numeric count, and no substitution, redirect, glob or other shell syntax (§131.10).
- **`CDPATH` is dropped from the environment shell commands run with**, so an approved `cd tests`
  cannot land in another directory.
- **`harness trace leaks` and `harness trace holds`** follow symlinked trace folders.
- **Scripts**: new `scripts/long_output_probe.py` (one long page from one call, graded under the
  benchmark agents' sandbox profile) and `scripts/webdeny.py` (web deny lists); `temperature_probe.py`
  takes a model argument.
- **Records**: `DESIGN-LOG.md` §131.

## 0.1.0 (2026-10-11)

The first public release. Research code: a coding-agent harness built around Mercury 2.5, a
diffusion LLM, and the benchmark that measured it. The repository is published as one commit;
commit hashes in `DESIGN-LOG.md` refer to the private development history this tag replaces.

### What it contains

- **`harness`** (`agent/`): plans, executes small steps with Mercury, and ends with a final
  check that makes no model call. Commands: `run`, `setup`, `config`, `skills`, `trace`,
  `sessions`. Every phase uses `inception/mercury-2.5` unless a flag says otherwise.
- **`bench`** (`bench/`): compares whole agent configurations on 16 seeded tasks, with reports for
  accuracy, honesty, cost, discrimination, reliability, history and A/B comparison. Drivers for
  django (SWE-bench Verified), SciCode and GAIA live in `scripts/`.
- **Documentation** (`docs/`): quickstart, command reference generated from `--help`,
  architecture, safety model, skills, benchmarking, one page listing every optional flag with
  where it was tested and what came out ([configuration-comparison](docs/configuration-comparison.md)),
  and how to rerun the experiments ([reproduce](docs/reproduce.md)).
- **Records**: `DESIGN-LOG.md` (every claim with its runs), `FINDINGS.md` (the results in one
  table), `TUNABLES.md` (load-bearing constants and their evidence).
- **Data**, as GitHub release assets of `v0.1.0`: the per-run results files
  (`mercury-harness-results-v0.1.0.tar.gz`, extract in a clone), a per-call metrics table
  (`mercury-harness-calls-v0.1.0.parquet`, numbers and labels only) and `DATA.md`. No traces are
  published, and GAIA appears as aggregate scores only.

### Headline results

Details, intervals and the runs behind each are in `FINDINGS.md` and the cited sections.

- A diffusion model's stalls hide behind keep-alives: median time to first content 1.15 s,
  p99 11.7 s over 50,924 calls; cutting silent calls after 30 s saves about ten times the time it
  re-asks (§69, FINDINGS row 1).
- Mercury reasons in hidden tokens whether asked or not, and more effort means fewer calls, not
  more cost: 12.8 calls a run at `high` against 14.9 to 18.4 at lower settings (§53.2, row 2).
- With Mercury on ten django tasks the harness does not beat a 100-line loop: 36 of 60 against
  mini-swe-agent's 36 of 60 after contaminated wins were voided (§94.1, §96.4, §118, rows 16-17);
  a sandboxed rerun gave 17 of 30 for the harness, mini and Goose alike (§126.2).
- A stronger model is the only lever that separated anything: GLM 5.3 Flash everywhere solved
  +2.67 SciCode dev problems [+0.33, +5.00] and raised fixtures accuracy by +0.086 [+0.028, +0.148]
  (§126.3, §126.4); on django, Aider with GLM resolved 26 of 30 against the harness's 15 at about
  a fifth of the cost (§109.5, row 18).
- Most mechanisms added on intuition did nothing measurable: a TDD gate, symptom-first planning,
  prose drafts, a goal loop, optional tools, a findings hold, a numeric self-check (rows 3, 5, 6, 9,
  12, 13).
- Review costs 18-22% of spend and time and showed no detectable gain on django, SciCode or the
  fixtures, so it is off by default (§124.10, §126).
- GAIA, text-only validation subset, Mercury only: 59 of 127 on the release harness, 61 of 127
  for the 30 Sep harness on the same search (§96.4, row 17).

### Changed for the release

- **`harness setup`** checks an OpenRouter or Inception API key, then stores it in the system's
  secret store: the macOS Keychain, or with the `keyring` extra the Linux Secret Service (GNOME
  Keyring, KWallet) or Windows Credential Manager; otherwise a file readable only by you.
- **Review is off by default.** `--review` turns it on; `--no-review` is the default. The bench
  configuration `mercury-only` keeps review on so its history stays comparable; `mercury-noreview`
  is the harness default.
- **Optional dependencies.** `textual` (the dashboard, `harness-tui`, `harness run --tui`) is the
  `tui` extra and `duckdb` (`harness trace sql`) the `analysis` extra; asking for either without it
  prints the install command. `pyyaml` was dropped, `platformdirs` declared, one version source
  and `harness --version` added.
- **Install from GitHub, not PyPI:** `uv tool install git+https://github.com/whtowbin/mercury-harness`.
  macOS and Linux; Python 3.11 or later; needs git and uv. Windows is not supported.
- **Results ship as release assets**, not in the tree (above).
- **Late additions:** `--direct` (skip the planner, §129),
  `--issue-symbols` (a deterministic identifier lookup for the planner, §128) and the
  refusal-learning client for models that reject forced tool choice (§129.4). `--direct` and
  `--issue-symbols` are off by default: neither passed its pre-registered rule.
- **Sandboxed benchmark drivers** for external agents and the harness (§126), macOS only.
- **Docs:** the command reference is regenerated from `--help`; `release-results.md` is merged into
  `configuration-comparison.md`, which now lists every optional flag with its tests and outcome; the
  stale Claude-plans-and-reviews description is gone from the landing page and architecture.

### Known limits

- Single-run and 30-run-a-side results cannot see differences smaller than about 25 points; the
  same configuration resolved 17, 19 and 13 of 30 django runs over three sweeps (§129.6).
- The isolated django and SciCode drivers need macOS `sandbox-exec`, and the SWE-bench and SciCode
  task data is not shipped (see `docs/reproduce.md`).
- `--direct` passes its cost rule and fails its honesty guard on the fixtures (§129.8); the
  models used as comparison points may remember the django fixes (§127.2).
