"""tests/conftest.py's warning for a failing run that overlapped a venv change."""

from __future__ import annotations

import os
import time

from conftest import site_packages_mtime, venv_changed_since


def test_a_package_directory_change_moves_the_mtime(tmp_path, monkeypatch):
    """Against a real directory: adding an entry, as an install does, is seen."""
    import site

    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    past = time.time() - 60
    os.utime(tmp_path, (past, past))
    before = site_packages_mtime()
    (tmp_path / "pytest-9.9.dist-info").mkdir()
    assert venv_changed_since(before, site_packages_mtime())


def test_an_untouched_environment_is_not_reported(tmp_path, monkeypatch):
    import site

    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    before = site_packages_mtime()
    assert not venv_changed_since(before, site_packages_mtime())


def test_no_reading_is_never_a_change():
    assert not venv_changed_since(None, 5.0)
    assert not venv_changed_since(5.0, None)
