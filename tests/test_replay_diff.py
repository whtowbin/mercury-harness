"""The differential-replay script: identical trees compare equal, a change is caught."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "replay_diff", Path(__file__).resolve().parent.parent / "scripts" / "replay_diff.py"
)
replay_diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_diff)


def _dump(path, targets="plan", seeds="6"):
    assert replay_diff.main(["dump", str(path), "--targets", targets, "--seeds", seeds]) == 0


def test_two_dumps_of_the_same_code_compare_equal(tmp_path, capsys):
    _dump(tmp_path / "a.json")
    _dump(tmp_path / "b.json")
    code = replay_diff.main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert code == 0
    assert "differing: 0" in capsys.readouterr().out


def test_a_changed_outcome_is_reported_and_fails_the_comparison(tmp_path, capsys):
    _dump(tmp_path / "a.json")
    changed = json.loads((tmp_path / "a.json").read_text())
    key = next(iter(changed))
    changed[key]["outcome"] += " (changed)"
    (tmp_path / "b.json").write_text(json.dumps(changed))
    code = replay_diff.main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert code == 1
    assert key in capsys.readouterr().out


def test_the_audit_target_runs_a_real_mutation_case():
    case = replay_diff.dump_audit(0)
    assert '"ran": true' in case["outcome"]
    assert len(replay_diff._SOURCES) * len(replay_diff._TESTS) == 18
