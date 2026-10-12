"""scripts/scicode/sandboxed_harness.py against the real sandbox-exec (§126): a stand-in
`harness` that tries to read the home folder, the caller's other API keys and sibling files."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "scicode" / "sandboxed_harness.py"

pytestmark = pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").exists(), reason="needs macOS sandbox-exec"
)

FAKE_HARNESS = """#!{python}
import os, pathlib, sys
a = sys.argv[1:]
ws = pathlib.Path(a[a.index('-w') + 1]); tr = pathlib.Path(a[a.index('--trace-dir') + 1])
secret = pathlib.Path(os.environ['HOME']) / sys.argv[-1]
res = []
for name, f in (('read_home', lambda: secret.read_text()),
                ('write_ws', lambda: (ws / 'x.txt').write_text('x')),
                ('write_trace', lambda: (tr / 'y.txt').write_text('x')),
                ('write_sibling', lambda: (ws.parent / 'sibling.txt').write_text('x'))):
    try:
        f(); res.append(name + '=ok')
    except OSError:
        res.append(name + '=denied')
parts = os.environ['PATH'].split(':')
ok = parts[0].endswith('tree/.venv/bin') and not any('/other/.venv/bin' in x for x in parts)
res.append('path=' + ('tree' if ok else 'wrong'))
uv = os.environ.get('UV_TOOL_DIR', '').startswith(str(ws.parent.parent))
res.append('uvdir=' + ('own' if uv else 'other'))
res.append('key=' + ('leaked' if 'LEAK_TEST_API_KEY' in os.environ else 'dropped'))
own = os.environ.get('TMPDIR', '').startswith(str(ws.parent.parent))
res.append('tmp=' + ('own' if own else 'other'))
(tr / 'result.txt').write_text(' '.join(res))
"""


def test_harness_under_the_wrapper_sees_only_its_own_folders(tmp_path):
    name = f".s126-secret-{tmp_path.name}"
    secret = Path.home() / name
    secret.write_text("private")
    try:
        tree = tmp_path / "tree"
        (tree / ".venv" / "bin").mkdir(parents=True)
        fake = tree / ".venv" / "bin" / "harness"
        fake.write_text(FAKE_HARNESS.format(python=sys._base_executable))
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        root = Path.home() / ".cache" / f"s126-test-{tmp_path.name}"
        ws, traces = root / "29", root / "traces" / "29"
        ws.mkdir(parents=True)
        r = subprocess.run(
            [sys.executable, str(WRAPPER), "run", "goal", "-w", str(ws), "--trace-dir",
             str(traces), name],
            env={
                **os.environ,
                "SCI_TREE": str(tree),
                "LEAK_TEST_API_KEY": "sk-secret",
                "PATH": "/other/.venv/bin:" + os.environ["PATH"],
            },
            capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        got = (traces / "result.txt").read_text()
    finally:
        secret.unlink(missing_ok=True)
        shutil.rmtree(Path.home() / ".cache" / f"s126-test-{tmp_path.name}", ignore_errors=True)
    assert got == (
        "read_home=denied write_ws=ok write_trace=ok write_sibling=denied "
        "path=tree uvdir=own key=dropped tmp=own"
    )


def test_bench_runs_under_the_wrapper_and_writes_only_to_its_run_folder(tmp_path):
    rundir = Path.home() / ".cache" / f"s126-bench-{tmp_path.name}"
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sandboxed_bench.py"), str(ROOT), str(rundir),
             "--configs", "no-such-config", "--tasks", "no-such-task"],
            capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        # bench.cli started inside the sandbox (imports read, the profile written) and refused
        # the unknown config itself, rather than the sandbox stopping the interpreter.
        assert (rundir / "sandbox.sb").exists()
        assert "no-such-config" in r.stderr + r.stdout, r.stderr[-400:]
    finally:
        shutil.rmtree(rundir, ignore_errors=True)
