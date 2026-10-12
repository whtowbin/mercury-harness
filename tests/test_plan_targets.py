"""scripts/analysis/plan_targets.py: does a django plan name the gold location (§125)."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "plan_targets.py"
_spec = importlib.util.spec_from_file_location("plan_targets", _path)
assert _spec and _spec.loader
plan_targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_targets)

ENUMS_PLAN = (
    '[{"id":"a","files":["django/db/models/fields/__init__.py","django/db/models/enums.py"],'
    '"writes":["django/db/models/fields/__init__.py"],"signatures":[]}]'
)
FIELDS_PLAN = (
    '[{"id":"a","files":["django/db/models/fields/__init__.py"],'
    '"writes":["django/db/models/fields/__init__.py"],'
    '"signatures":["Field.get_prep_value(self, value)"]}]'
)


def test_11964_hits_only_when_a_step_names_enums_py():
    assert plan_targets.classify("11964", ENUMS_PLAN) == (True, False)
    assert plan_targets.classify("11964", FIELDS_PLAN) == (False, False)


def test_11951_reads_signatures_and_flags_the_trap():
    gold = '[{"files":["django/db/models/query.py"],"signatures":["QuerySet._batched_insert()"]}]'
    trap = '[{"files":["django/db/models/query.py"],"signatures":["QuerySet.bulk_create()"]}]'
    assert plan_targets.classify("11951", gold) == (True, False)
    assert plan_targets.classify("11951", trap) == (False, True)


def test_a_task_without_a_rule_and_an_empty_plan_are_not_hits():
    assert plan_targets.classify("9296", ENUMS_PLAN) == (False, False)
    assert plan_targets.classify("11964", "") == (False, False)


def test_json_spacing_does_not_matter():
    spaced = ENUMS_PLAN.replace('"files":', '"files": ')
    assert plan_targets.classify("11964", spaced) == (True, False)


def test_a_task_without_a_function_rule_hits_on_a_gold_source_file_path(monkeypatch):
    monkeypatch.setattr(plan_targets, "gold_paths", lambda task: ["django/utils/http.py"])
    plan = '[{"files":["django/utils/http.py"],"writes":[],"signatures":[]}]'
    assert plan_targets.classify("11848", plan) == (True, False)
    other = '[{"files":["django/utils/text.py"],"writes":[],"signatures":[]}]'
    assert plan_targets.classify("11848", other) == (False, False)
