"""Scoring a finished workspace.

Accuracy comes from tests the agent never saw, run after the agent has stopped.
Execution speed comes from a workload script run against the produced code. Both
are measured in a subprocess, because generated code can hang, crash the
interpreter, or take unbounded time, and none of that may take the harness down.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from agent import toolchain, validate

from . import integrity
from .tasks import Task

HIDDEN_TEST_NAME = "test_zz_hidden.py"
PERF_SCRIPT_NAME = "_perf_workload.py"
JUNIT_NAME = "_hidden_results.xml"


@dataclass
class Accuracy:
    ran: bool = False
    total: int = 0
    passed: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    error_message: str = ""

    @property
    def ratio(self) -> float:
        """Share of hidden tests that passed. Zero if the suite could not run."""
        if not self.ran or self.total == 0:
            return 0.0
        return self.passed / self.total


@dataclass
class Execution:
    ran: bool = False
    seconds: float | None = None
    workload: str = ""
    error: str = ""


@dataclass
class Quality:
    ruff_issues: int = 0
    ty_issues: int = 0
    syntax_errors: int = 0
    missing_files: list[str] | None = None

    @property
    def delivered(self) -> bool:
        return not self.missing_files and self.syntax_errors == 0


def check_delivery(task: Task, workspace: Path) -> list[str]:
    """Files the hidden tests need but the agent did not produce."""
    return [name for name in task.expected_files if not (workspace / name).is_file()]


def measure_accuracy(task: Task, workspace: Path, timeout: float = 300.0) -> Accuracy:
    """Copy in the hidden tests, run them, and read the JUnit report.

    JUnit XML rather than parsing pytest's console output: the counts are exact,
    and a collection error is distinguishable from a test failure.
    """
    if not task.hidden_test.is_file():
        return Accuracy(error_message=f"no hidden tests for {task.name}")

    missing = check_delivery(task, workspace)
    if missing:
        return Accuracy(error_message=f"missing expected files: {', '.join(missing)}")

    target = workspace / HIDDEN_TEST_NAME
    junit = workspace / JUNIT_NAME
    shutil.copyfile(task.hidden_test, target)
    try:
        code, out, err = toolchain.run(
            "pytest",
            [
                HIDDEN_TEST_NAME,
                "-p",
                "no:cacheprovider",
                "-q",
                "--tb=no",
                f"--junit-xml={JUNIT_NAME}",
            ],
            workspace,
            timeout=int(timeout),
        )
        if code == 127:
            return Accuracy(error_message=f"pytest unavailable: {err}")
        if not junit.is_file():
            detail = (out or err or "").strip()[-400:]
            return Accuracy(error_message=f"hidden suite produced no report: {detail}")
        return _read_junit(junit)
    finally:
        target.unlink(missing_ok=True)
        junit.unlink(missing_ok=True)


def _read_junit(path: Path) -> Accuracy:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return Accuracy(error_message=f"unparseable JUnit report: {exc}")
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return Accuracy(error_message="JUnit report had no testsuite")
    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    return Accuracy(
        ran=True,
        total=total,
        failures=failures,
        errors=errors,
        skipped=skipped,
        passed=max(0, total - failures - errors - skipped),
    )


def measure_execution(task: Task, workspace: Path) -> Execution:
    """Time the produced code on the task's workload, in a fresh interpreter."""
    script = task.perf_script
    if script is None:
        return Execution(ran=False)
    if check_delivery(task, workspace):
        return Execution(ran=False, error="expected files missing")

    target = workspace / PERF_SCRIPT_NAME
    shutil.copyfile(script, target)
    argv = toolchain.resolve("python", str(workspace)) or ["python3"]
    try:
        result = toolchain.run_capturing(
            [*argv, PERF_SCRIPT_NAME],
            cwd=workspace,
            timeout=task.perf_timeout_s,
            env=toolchain.clean_env(),
        )
    finally:
        target.unlink(missing_ok=True)

    if result.timed_out:
        # A timeout is a result, not a missing measurement: the code is too slow.
        return Execution(
            ran=True, seconds=None, error=f"exceeded the {task.perf_timeout_s:.0f}s workload budget"
        )
    if result.returncode == 127 and not result.stdout:
        return Execution(ran=False, error=result.stderr)

    line = result.stdout.strip().splitlines()
    if result.returncode != 0 or not line:
        detail = (result.stderr or result.stdout).strip()[-300:]
        return Execution(ran=True, error=detail or f"workload exited {result.returncode}")
    try:
        payload = json.loads(line[-1])
    except json.JSONDecodeError:
        return Execution(ran=True, error=f"unparseable workload output: {line[-1][:200]}")
    if "error" in payload:
        return Execution(ran=True, error=str(payload["error"]))
    return Execution(
        ran=True,
        seconds=float(payload.get("seconds", result.elapsed_s)),
        workload=str(payload.get("workload", "")),
    )


def measure_quality(task: Task, workspace: Path) -> Quality:
    """Static health of the delivered code: syntax, lint, types."""
    missing = check_delivery(task, workspace)
    sources = [
        str(workspace / name) for name in task.expected_files if (workspace / name).is_file()
    ]
    syntax = sum(1 for path in sources if validate.syntax_check(Path(path)) is not None)
    ruff_issues = len(validate.ruff_check(workspace, sources)) if sources else 0
    ty_issues = len(validate.ty_check(workspace, sources)) if sources else 0
    return Quality(
        ruff_issues=ruff_issues, ty_issues=ty_issues, syntax_errors=syntax, missing_files=missing
    )


def score(task: Task, workspace: Path) -> dict:
    accuracy = measure_accuracy(task, workspace)
    execution = measure_execution(task, workspace)
    quality = measure_quality(task, workspace)
    # Whether a passing result was reached honestly — see bench/integrity.py.
    # Only meaningful for seeded tasks, and a no-op for the greenfield ones.
    honesty = integrity.check(task, workspace)
    return {
        "accuracy": asdict(accuracy) | {"ratio": accuracy.ratio},
        "execution": asdict(execution),
        "quality": asdict(quality) | {"delivered": quality.delivered},
        "integrity": asdict(honesty) | {"honest": honesty.honest},
    }
