"""scripts/s123_launch.py: Settings fields without a CLI flag, set from S123_SET (§123)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.config import Settings

LAUNCH = Path(__file__).resolve().parents[1] / "scripts" / "s123_launch.py"
_spec = importlib.util.spec_from_file_location("s123_launch", LAUNCH)
assert _spec and _spec.loader
s123_launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s123_launch)


def test_values_take_the_type_of_the_field_default():
    got = s123_launch.parse_overrides(
        "preload_neighbours=1,executor_hypothesis=true,force_tools=trace_python+inspect_module",
        Settings,
    )
    assert got == {
        "preload_neighbours": True,
        "executor_hypothesis": True,
        "force_tools": ("trace_python", "inspect_module"),
    }


def test_an_unknown_field_stops_the_run_in_a_real_process():
    r = subprocess.run(
        [sys.executable, str(LAUNCH), "--help"],
        env={**os.environ, "S123_SET": "preload_neighbors=1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode != 0
    assert "unknown Settings field" in r.stderr


def test_empty_spec_changes_nothing():
    assert s123_launch.parse_overrides("", Settings) == {}
    with pytest.raises(SystemExit):
        s123_launch.parse_overrides("nope=1", Settings)
