"""Every tool's schema agrees with the Python function behind it.

The cheap guard in place of generating schemas from typed models: a schema
property the function does not take fails every call that uses it, and a
required function parameter the schema leaves optional fails every call that
omits it. Checked across all 38 tools when added (DESIGN-LOG §66): no drift.
"""

from __future__ import annotations

import inspect

import pytest

from agent.tools import execution_registry, planning_registry, review_registry
from agent.tools.control import with_reproduce, with_task_kind


def _all_tools():
    seen = {}
    for registry in (execution_registry(), planning_registry(probe=True), review_registry()):
        for tool in registry.tools():
            seen.setdefault(tool.name, tool)
    plan = seen["submit_plan"]
    seen["submit_plan+options"] = with_reproduce(with_task_kind(plan))
    return seen


@pytest.mark.parametrize(("name", "tool"), sorted(_all_tools().items()))
def test_schema_matches_signature(name, tool):
    _assert_schema_matches_signature(name, tool)


def _submit_plan_variants():
    """submit_plan as the planner is really given it, every optional field switched
    on, once per repro form: built by the planner's own assembly, so a new
    variant is covered without being listed here."""
    from types import SimpleNamespace

    from agent.phases.plan import _planning_kit
    from agent.symptom import FORMS

    for form in FORMS:
        settings = SimpleNamespace(
            research_first=True,
            plan_findings_required=True,
            plan_task_kind=True,
            plan_key_facts=True,
            repro_form=form,
        )
        yield form, _planning_kit(settings, "repro").get("submit_plan")


@pytest.mark.parametrize(("form", "tool"), list(_submit_plan_variants()))
def test_every_submit_plan_variant_matches_signature(form, tool):
    _assert_schema_matches_signature(f"submit_plan[{form}]", tool)


def test_plan_step_items_are_exactly_what_step_from_dict_reads():
    """A step's schema and `Step.from_dict` are two copies of one spec, and
    from_dict ignores a key it does not read and defaults one that is missing,
    so a field renamed on one side loses its value with no error. The top-level
    check above cannot see inside `steps[]`."""
    from agent.phases.plan import Step

    read: set[str] = set()

    class Recording(dict):
        def get(self, key, default=None):
            read.add(key)
            return super().get(key, default)

    Step.from_dict(Recording(), 0)
    for form, tool in _submit_plan_variants():
        items = tool.parameters["properties"]["steps"]["items"]
        assert set(items["properties"]) == read, form
        assert set(items["required"]) <= read, form


def _assert_schema_matches_signature(name, tool):
    params = inspect.signature(tool.fn).parameters
    takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    accepted = {n for n in params if n != "ctx"}
    required_by_fn = {
        n
        for n, p in params.items()
        if n != "ctx" and p.default is p.empty and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    schema = tool.parameters or {}
    properties = set(schema.get("properties") or {})
    required = set(schema.get("required") or [])

    if not takes_kwargs:
        assert properties <= accepted, f"schema offers what {name} cannot take"
    assert required_by_fn <= required, f"{name} needs arguments the schema calls optional"
    assert required <= properties, f"{name} requires undeclared arguments"
    # A parameter the schema never mentions must have a default, or no call can reach it.
    assert accepted - properties <= accepted - required_by_fn
