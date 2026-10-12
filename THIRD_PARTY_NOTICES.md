# Third-party notices

mercury-harness is licensed under the Apache License 2.0 (`LICENSE`). It contains material
adapted from the projects below, used under their licenses. Projects that the benchmarks run as
separate programs (mini-swe-agent, Aider, Goose, Codex CLI) and the benchmark datasets
(SWE-bench, SciCode, GAIA) are not included in this repository; the benchmark scripts download
or call them at run time under their own terms.

## GAIA leaderboard scorer (Apache License 2.0)

`scripts/gaia/gaia.py` (`normalize_number_str`, `normalize_str`, `question_scorer`) is derived from `scorer.py` in the GAIA leaderboard Space,
<https://huggingface.co/spaces/gaia-benchmark/leaderboard>, licensed under the Apache License 2.0
(the same license as this repository; full text in `LICENSE`). Changes: type annotations, the
prints and warnings removed, `split_string` and the element loop inlined; the matching rules are
unchanged.

## Astral Claude Code plugin skills (MIT or Apache License 2.0)

`agent/skills_data/uv.md`, `ruff.md` and `ty.md` are adapted from the skills in
<https://github.com/astral-sh/claude-code-plugins> (plugins/astral/skills), dual-licensed under
the MIT License or the Apache License 2.0. The MIT notice:

```
MIT License

Copyright (c) 2025 Astral Software Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Matt Pocock's skills (MIT)

`agent/skills_data/grilling.md`, `research.md` and `diagnosing.md` are adapted from
<https://github.com/mattpocock/skills>:

```
MIT License

Copyright (c) 2026 Matt Pocock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledged, not copied

The skill file format (front matter with triggers) follows
<https://github.com/honnibal/claude-skills> (MIT); no text from it is included.
