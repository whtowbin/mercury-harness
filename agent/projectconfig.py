"""What a project's own configuration says about how it is linted, formatted and typed.

Several harness behaviours assume a project that uses Ruff and ty, which is true of
this repository and of the bench fixtures (which configure nothing, so nothing
contradicts it), and false of django: `ruff format` restyled a file around a 4-line
fix (§70), type noise in untyped code drove fix rounds (§71), and the executor ran
`ruff check --select DTZ,UP031 --fix` on a flake8/isort project and broke a regex
(§72). Each decision reads the project's own config through here.
"""

from __future__ import annotations

from pathlib import Path


def _read(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def configures_ruff(root: Path) -> bool:
    return (
        (root / "ruff.toml").is_file()
        or (root / ".ruff.toml").is_file()
        or "[tool.ruff" in _read(root, "pyproject.toml")
    )


def uses_a_formatter(root: Path) -> bool:
    """Whether the project opts into an automatic formatter `ruff format` stands in
    for: a Ruff config, or a [tool.black] table. A project with neither has a house
    style of its own (§70: 18 lines of churn around a 4-line fix on django)."""
    return configures_ruff(root) or "[tool.black" in _read(root, "pyproject.toml")


_TYPE_CHECKER_FILES = ("ty.toml", "mypy.ini", ".mypy.ini", "pyrightconfig.json")
_TYPE_CHECKER_TABLES = ("[tool.ty", "[tool.mypy", "[tool.pyright")


def configures_type_checker(root: Path) -> bool:
    if any((root / name).is_file() for name in _TYPE_CHECKER_FILES):
        return True
    if any(table in _read(root, "pyproject.toml") for table in _TYPE_CHECKER_TABLES):
        return True
    return "[mypy" in _read(root, "setup.cfg")


# (linter, files that configure it by existing, sections that configure it inside
# setup.cfg / tox.ini / pyproject.toml)
_OTHER_LINTERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("flake8", (".flake8",), ("[flake8]",)),
    ("isort", (".isort.cfg",), ("[isort]", "[tool.isort")),
    ("pylint", (".pylintrc", "pylintrc"), ("[tool.pylint", "[pylint")),
    ("black", (), ("[tool.black",)),
    ("pycodestyle", (), ("[pycodestyle]",)),
)


def other_linters(root: Path) -> list[str]:
    """Linters and formatters other than Ruff that the project configures."""
    texts = "\n".join(_read(root, n) for n in ("setup.cfg", "tox.ini", "pyproject.toml"))
    found = []
    for name, files, sections in _OTHER_LINTERS:
        if any((root / f).is_file() for f in files) or any(s in texts for s in sections):
            found.append(name)
    return found


def foreign_linters(root: Path) -> list[str]:
    """The project's own linters when it configures some and Ruff is not among them.

    Empty for a Ruff project and for one that configures nothing (every bench
    fixture), so the harness's Ruff guidance changes only where the project has
    chosen something else (django: flake8 and isort in setup.cfg).
    """
    return [] if configures_ruff(root) else other_linters(root)
