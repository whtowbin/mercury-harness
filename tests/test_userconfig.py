"""Persisted user preferences. Every test is isolated from the real config dir."""

from __future__ import annotations

import pytest

from agent import userconfig


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    """No test may ever touch the real ~/Library/.../mercury-harness config."""
    monkeypatch.setattr(userconfig, "config_dir", lambda: tmp_path / "config")


def test_load_with_nothing_persisted_is_empty():
    prefs = userconfig.load()
    assert prefs.values == {}


def test_set_then_load_round_trips():
    userconfig.set_value("executor_model", "some/model")
    prefs = userconfig.load()
    assert prefs.get("executor_model") == "some/model"


def test_set_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown setting"):
        userconfig.set_value("not_a_real_field", "x")


def test_bool_fields_parse_common_spellings():
    for spelling in ("true", "True", "1", "yes", "on"):
        userconfig.set_value("auto_approve", spelling)
        assert userconfig.load().get("auto_approve") is True
    for spelling in ("false", "0", "no", "off"):
        userconfig.set_value("auto_approve", spelling)
        assert userconfig.load().get("auto_approve") is False


def test_bool_field_rejects_garbage():
    with pytest.raises(ValueError, match="yes/no"):
        userconfig.set_value("auto_approve", "maybe")


def test_int_field_parses_and_rejects_non_numbers():
    userconfig.set_value("step_concurrency", "4")
    assert userconfig.load().get("step_concurrency") == 4
    with pytest.raises(ValueError, match="whole number"):
        userconfig.set_value("step_concurrency", "four")


def test_unset_removes_a_key_others_survive():
    userconfig.set_value("executor_model", "a")
    userconfig.set_value("planner_model", "b")
    userconfig.unset_value("executor_model")
    prefs = userconfig.load()
    assert "executor_model" not in prefs.values
    assert prefs.get("planner_model") == "b"


def test_unset_a_key_that_was_never_set_does_not_raise():
    userconfig.unset_value("executor_model")  # must not raise


def test_reset_deletes_the_file():
    userconfig.set_value("executor_model", "a")
    assert userconfig.config_path().is_file()
    userconfig.reset()
    assert not userconfig.config_path().is_file()


def test_reset_when_nothing_was_ever_saved_does_not_raise():
    userconfig.reset()  # must not raise


def test_a_corrupt_config_file_is_treated_as_empty_not_fatal():
    userconfig.config_path().parent.mkdir(parents=True, exist_ok=True)
    userconfig.config_path().write_text("{not valid json")
    assert userconfig.load().values == {}


def test_a_config_file_that_is_not_a_json_object_is_treated_as_empty():
    userconfig.config_path().parent.mkdir(parents=True, exist_ok=True)
    userconfig.config_path().write_text("[1, 2, 3]")
    assert userconfig.load().values == {}


def test_loading_ignores_keys_that_are_not_persistable():
    userconfig.config_path().parent.mkdir(parents=True, exist_ok=True)
    import json

    userconfig.config_path().write_text(json.dumps({"executor_model": "a", "evil": "x"}))
    prefs = userconfig.load()
    assert "evil" not in prefs.values
    assert prefs.get("executor_model") == "a"


def test_user_skills_dir_is_under_the_config_dir(tmp_path):
    assert userconfig.user_skills_dir().parent == userconfig.config_dir()
