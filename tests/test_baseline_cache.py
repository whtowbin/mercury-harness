"""Cross-session caching of the pre-run test baseline.

A cache that cannot be trusted must behave exactly like no cache at all --
every test here checks that a bad, missing, or mismatched cache falls back
to "nothing cached" rather than raising or returning something wrong.
"""

from __future__ import annotations

from agent import baseline_cache


def test_no_cache_file_is_a_clean_miss(tmp_path):
    assert baseline_cache.load(tmp_path, "state-a") is None


def test_a_saved_cache_is_loaded_back_with_a_matching_state(tmp_path):
    baseline_cache.save(tmp_path, "state-a", ["tests/test_x.py::test_y"], False)
    cached = baseline_cache.load(tmp_path, "state-a")
    assert cached is not None
    assert cached["failures"] == ["tests/test_x.py::test_y"]
    assert cached["collection_broken"] is False


def test_a_different_state_is_a_miss_not_a_stale_hit(tmp_path):
    """The whole point: a changed workspace must never reuse a stale result."""
    baseline_cache.save(tmp_path, "state-a", [], False)
    assert baseline_cache.load(tmp_path, "state-b") is None


def test_corrupt_json_is_a_clean_miss_not_a_crash(tmp_path):
    path = tmp_path / ".harness" / baseline_cache.CACHE_FILE
    path.parent.mkdir(parents=True)
    path.write_text("not json at all", encoding="utf-8")
    assert baseline_cache.load(tmp_path, "state-a") is None


def test_an_unexpected_shape_is_a_clean_miss(tmp_path):
    path = tmp_path / ".harness" / baseline_cache.CACHE_FILE
    path.parent.mkdir(parents=True)
    path.write_text('{"workspace_state": "state-a", "failures": "not-a-list"}', encoding="utf-8")
    assert baseline_cache.load(tmp_path, "state-a") is None


def test_a_json_array_instead_of_an_object_is_a_clean_miss(tmp_path):
    path = tmp_path / ".harness" / baseline_cache.CACHE_FILE
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert baseline_cache.load(tmp_path, "state-a") is None


def test_save_creates_the_harness_directory_if_missing(tmp_path):
    assert not (tmp_path / ".harness").exists()
    baseline_cache.save(tmp_path, "state-a", [], False)
    assert (tmp_path / ".harness" / baseline_cache.CACHE_FILE).is_file()


def test_save_is_silent_when_the_directory_cannot_be_created(tmp_path):
    """Best-effort: a write failure costs the next run a suite pass it
    would have had to run anyway, never this one anything."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    baseline_cache.save(blocked / "sub", "state-a", [], False)  # must not raise


def test_saving_again_overwrites_the_previous_entry(tmp_path):
    baseline_cache.save(tmp_path, "state-a", ["old_failure"], False)
    baseline_cache.save(tmp_path, "state-b", ["new_failure"], True)
    assert baseline_cache.load(tmp_path, "state-a") is None
    cached = baseline_cache.load(tmp_path, "state-b")
    assert cached is not None
    assert cached["failures"] == ["new_failure"]
    assert cached["collection_broken"] is True
