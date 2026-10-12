"""A step's verify command in the project's own test form, and a verify that selects no test.

django's tests run through `python tests/runtests.py <labels>`. Planners write verifies the way
they would for pytest: `tests.pagination.tests.X` with the label root left on, pytest node ids
(`migrations.test_operations::Class::test`) passed to runtests.py, `-xv`, `uv run pytest` on a
django test file. run_tests already maps such targets (`checks.as_test_labels`), but a step's
verify ran as written. Over 55 django runs, 11 ended their final check failing such a verify
although the code was fine, and fix rounds spent up to 76 calls a run trying to repair what no
edit could (DESIGN-LOG §110). So a verify aimed at the project's tests is rebuilt from the
project's own test command, and one that still selects no test is reported as unrunnable, not
as a defect for a fix round.
"""

from __future__ import annotations

import re
import shlex

from .testcmd import build_argv
from .tools.checks import as_test_labels

# Options that take a value, in pytest or django's runner, so the value is not read as a target.
_VALUE_OPTIONS = frozenset(
    {
        "-k",
        "-m",
        "-p",
        "-n",
        "--settings",
        "--verbosity",
        "--parallel",
        "--tag",
        "--exclude-tag",
        "--start-at",
        "--start-after",
        "--tb",
        "--maxfail",
        "--rootdir",
        "-c",
    }
)
# Kept when rebuilding: both runners take a name pattern with -k.
_KEPT_OPTIONS = frozenset({"-k"})
_SHELL_JOINS = ("&&", "||", ";", "|")


def _runner_script(test_command: str) -> str:
    """The script a project test command runs (`tests/runtests.py`), or ""."""
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        return ""
    return next((t for t in tokens if t.endswith(".py")), "")


def _invoked_tool(tokens: list[str], script: str) -> int:
    """Index of the token after which a verify's own arguments start: the project's runner
    script, or pytest. -1 when the command is neither."""
    for i, token in enumerate(tokens):
        if script and (token == script or token.endswith("/" + script.rsplit("/", 1)[-1])):
            return i
        if token == "pytest" or (
            token == "-m" and i + 1 < len(tokens) and tokens[i + 1] == "pytest"
        ):
            return i if token == "pytest" else i + 1
    return -1


def _targets_and_kept(arguments: list[str]) -> tuple[list[str], list[str]]:
    targets: list[str] = []
    kept: list[str] = []
    skip_value_of = ""
    for token in arguments:
        if skip_value_of:
            if skip_value_of in _KEPT_OPTIONS:
                kept += [skip_value_of, token]
            skip_value_of = ""
        elif token.startswith("-"):
            name = token.split("=", 1)[0]
            if name in _VALUE_OPTIONS and "=" not in token:
                skip_value_of = name
            elif name in _KEPT_OPTIONS:
                kept.append(token)
        else:
            targets.append(token)
    return targets, kept


def normalize(command: str, test_command: str, label_root: str) -> str:
    """`command` rebuilt from the project's test command when it runs the project's tests
    some other way, else `command` unchanged. Only for a project with its own test command
    that is not pytest; a compound shell command is left alone."""
    if not test_command or "pytest" in test_command or not command.strip():
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if any(t in _SHELL_JOINS for t in tokens):
        return command
    at = _invoked_tool(tokens, _runner_script(test_command))
    if at < 0:
        return command
    targets, kept = _targets_and_kept(tokens[at + 1 :])
    root = label_root.strip("/")
    labels = [
        label[len(root) + 1 :] if root and label.startswith(root + ".") else label
        for label in as_test_labels(targets, label_root)
    ]
    rebuilt = shlex.join(build_argv(test_command, [*kept, *labels]))
    return command if rebuilt == command else rebuilt


_NO_MODULE = re.compile(r"No module named '([^']+)'")
# A missing module attribute (a test class) or class attribute (a test method).
_NO_ATTRIBUTE = re.compile(r"(?:module|type object) '[^']+' has no attribute '([^']+)'")
_PYTEST_NOTHING = (
    "found no collectors",
    "ERROR: file or directory not found",
    "ERROR: not found:",
    "no tests ran in",
)


def selects_nothing(command: str, output: str) -> str:
    """Why a failed verify ran no test at all (a label that does not import, a named test that
    does not exist, an argument the runner rejects, pytest collecting nothing), or "" when it
    ran tests and they failed, or failed some other way. A test module that exists and fails
    to import because of the code is a real failure: the missing module must be the label."""
    if "unrecognized arguments:" in output:
        return "the test runner rejected an argument"
    for match in _NO_MODULE.finditer(output):
        name = match.group(1)
        if "::" in name or "/" in name or _names_a_target(command, name):
            return f"the test label {name!r} does not import"
    for match in _NO_ATTRIBUTE.finditer(output):
        if _names_a_target(command, match.group(1)):
            return f"the named test {match.group(1)!r} does not exist"
    if "unittest/suite.py" in output and "'NoneType' object is not iterable" in output:
        return "the test runner was given a file path, not a test label"
    for marker in _PYTEST_NOTHING:
        if marker in output and " passed" not in output and " failed" not in output:
            return "pytest collected no tests from the named targets"
    return ""


def _names_a_target(command: str, name: str) -> bool:
    """`name` is a target the command passed, or a dotted part of one."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return any(
        token == name or token.startswith(name + ".") or f".{name}." in f".{token}."
        for token in tokens
    )
