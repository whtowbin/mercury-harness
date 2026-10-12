"""scripts/scicode/sample.py: SciCode's prompt, its code extraction, and candidate voting.

The voting test runs the real runner in a real subprocess with SciCode's scoring Python (numpy),
because a mocked execution would pass by construction. It is skipped where that environment has
not been built (`python scripts/scicode/sci.py env`).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "scicode"))
sys.path.insert(0, str(ROOT / "scripts"))

# sample.py reads SciCode's prompt template at import, which raises FileNotFoundError rather
# than ImportError, so importorskip alone does not skip a clone without the data.
_DATA = Path(os.environ.get("SCI_DATA", ROOT / "traces" / "scicode-data"))
if not (_DATA / "repo" / "eval" / "data" / "multistep_template.txt").is_file():
    pytest.skip("SciCode data (SCI_DATA) not present", allow_module_level=True)
sample = pytest.importorskip("sample", reason="SciCode data (SCI_DATA) not present")


def test_extract_matches_scicode_rules():
    reply = (
        "Here:\n```python\nimport numpy as np\nfrom scipy import linalg\n"
        "def f(x):\n    return x\n```\n"
    )
    code = sample.extract(reply)
    assert "def f(x)" in code
    assert "import numpy" not in code and "from scipy" not in code


def test_choose_takes_the_largest_agreeing_group_and_falls_back_to_the_first():
    a, b = [["A[1.0]"]], [["A[2.0]"]]
    assert sample.choose([b, a, a, None]) == 1
    assert sample.choose([None, None]) == 0
    assert sample.choose([a, b]) == 0  # a tie keeps the earlier candidate
    assert sample.agree([a, a]) and not sample.agree([a, None]) and not sample.agree([a, b])


def test_function_name_from_header():
    assert sample.sci_function_name("def get_lattice_coords(latvec):\n    '''doc'''") == (
        "get_lattice_coords"
    )
    assert sample.sci_function_name("class Slater:\n    def __init__(self):") == "Slater"


@pytest.mark.skipif(not sample.sci.SCORE_PY.exists(), reason="SciCode score env not built")
def test_candidates_run_in_a_real_subprocess_and_group_by_output(tmp_path):
    good = "def f(x):\n    return np.array([x, 2 * x])"
    wrong = "def f(x):\n    return x + 1"
    crash = "def f(x):\n    raise ValueError('boom')"
    sigs = sample.run_candidates(
        "import numpy as np", "PROBES = [(1.0,), (2.5,)]", "f", [good, wrong, good, crash], tmp_path
    )
    assert sigs[3] is None
    assert sigs[0] == sigs[2] and sigs[0] != sigs[1]
    assert sample.choose(sigs) == 0
