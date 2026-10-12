"""scripts/swebench/iso_agents.py: external agents on isolated workspaces (§126).

The sandbox is tested against the real `sandbox-exec`, not a mock: a stub agent that tries to
read a file in the home folder and to write beside its workspace.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# iso.py prepends scripts/swebench to sys.path, where agent.py would shadow the harness's `agent`
# package for every later test: import it, then put sys.path and sys.modules back.
_saved_path = list(sys.path)
sys.path += [str(ROOT / "scripts" / "swebench"), str(ROOT / "scripts")]
try:
    import iso  # ty: ignore[unresolved-import]
    import iso_agents  # ty: ignore[unresolved-import]
    from agents import sandboxed  # ty: ignore[unresolved-import]
finally:
    sys.path[:] = _saved_path
    if str(getattr(sys.modules.get("agent"), "__file__", "")).endswith("swebench/agent.py"):
        del sys.modules["agent"]

pytestmark = pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").exists(), reason="needs macOS sandbox-exec"
)

PROBE = (
    "import pathlib, sys\n"
    "secret, ws, out = map(pathlib.Path, sys.argv[1:4])\n"
    "res = []\n"
    "for name, f in (('read_home', lambda: secret.read_text()),\n"
    "                ('write_ws', lambda: (ws / 'a.txt').write_text('x')),\n"
    "                ('write_out', lambda: (out / 'b.txt').write_text('x')),\n"
    "                ('write_home', lambda: (secret.parent / '.s126-new').write_text('x'))):\n"
    "    try:\n        f(); res.append(name + '=ok')\n"
    "    except OSError as e:\n        res.append(name + '=denied')\n"
    "(out / 'result.txt').write_text(' '.join(res))\n"
)


def test_a_sandboxed_agent_reads_nothing_in_home_and_writes_only_its_own_folders(tmp_path):
    secret = Path.home() / f".s126-test-secret-{tmp_path.name}"
    secret.write_text("private")
    try:
        ws, out = tmp_path / "ws", tmp_path / "out"
        ws.mkdir()
        out.mkdir()
        stub = types.ModuleType("stub_agent")
        argv, env = sandboxed(
            stub,
            [sys._base_executable, "-c", PROBE, str(secret), str(ws), str(out)],
            ws,
            out,
            "sk-test",
            {},
        )
        r = subprocess.run(argv, env=env, cwd=ws, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr  # the base interpreter (uv-managed) must start inside it
        got = (out / "result.txt").read_text()
    finally:
        secret.unlink(missing_ok=True)
        (Path.home() / ".s126-new").unlink(missing_ok=True)
    assert got == "read_home=denied write_ws=ok write_out=ok write_home=denied"


def test_install_routes_only_the_agent_arms(monkeypatch):
    calls = []
    monkeypatch.setattr(iso, "run_job", lambda *a: calls.append(("harness", a)) or {})
    monkeypatch.setattr(iso_agents, "run_agent_job", lambda *a: calls.append(("agent", a)) or {})
    iso_agents.install()
    iso.run_job("p", 1, "11790", "mini")
    iso.run_job("p", 1, "11790", "rel")
    assert [c[0] for c in calls] == ["agent", "harness"]
    assert {"mini", "mini-glm", "goose"} <= set(iso.ARMS)
