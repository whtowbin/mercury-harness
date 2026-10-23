"""scripts/long_output_probe.py's grader against the real sandbox-exec (§131.17): the reference
module scores in full, and a hostile module cannot read the home folder, reach the network, write
outside its folder or ask the keychain."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # the probe imports `agents` from there
_spec = importlib.util.spec_from_file_location(
    "long_output_probe", ROOT / "scripts" / "long_output_probe.py"
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS sandbox-exec"),
    pytest.mark.skipif(
        not Path("/usr/bin/sandbox-exec").exists(), reason="needs macOS sandbox-exec"
    ),
]

REFERENCE = Path(__file__).parent / "fixtures" / "long_output_calc_ref.md"

# Each attempt must raise. `write_here` is the control: a write inside the grader's own folder
# must work, so a blocked attempt means the sandbox and not a typo.
HOSTILE = """
import os, socket, subprocess, sys

def read_home():
    open({secret!r}).read()

def network():
    socket.create_connection(("1.1.1.1", 443), timeout=3)

def write_home():
    open(os.path.expanduser("~/.long_output_probe_x"), "w").write("x")

def keychain():
    r = subprocess.run(["/usr/bin/security", "list-keychains"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        raise RuntimeError("keychain answered")
    raise OSError("keychain refused")

open("here.txt", "w").write("control")

escaped = []
for fn in (read_home, network, write_home, keychain):
    try:
        fn()
        escaped.append(fn.__name__)
    except Exception:
        pass
if escaped:
    print("ESCAPED", escaped)
    sys.exit(1)
print("ALL TESTS PASSED")
"""


def test_reference_module_scores_in_full_under_the_sandbox():
    r = probe.grade(REFERENCE.read_text())
    assert r["selftest_pass"] is True
    assert (r["hidden_ok"], r["hidden_total"]) == (63, 63)


def test_hostile_module_cannot_read_home_use_network_write_outside_or_use_keychain():
    with tempfile.NamedTemporaryFile("w", dir=Path.home(), prefix=".probe-secret-") as secret:
        secret.write("secret")
        secret.flush()
        code = "```python\n" + HOSTILE.format(secret=secret.name) + "```"
        r = probe.grade(code)
    # selftest_pass is True only when every attempt raised
    assert r["selftest_pass"] is True
    assert not (Path.home() / ".long_output_probe_x").exists()
