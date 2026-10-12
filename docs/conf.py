"""Sphinx configuration for the mercury-harness docs, built for Read the Docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# So autodoc can import agent/ and bench/ without the package being installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "mercury-harness"
copyright = "Henry Towbin"
author = "Henry Towbin"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("mercury-harness")
except Exception:
    release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
# Working records (drafts, plans, reviews, handoffs) live in docs/ but are not part of the
# published site; the Guide toctree in index.md is the site.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "archive",
    "plans",
    "posts",
    "reviews",
    "handoff-*.md",
    "multi-language-gaps.md",
]

# autodoc: the docstrings in this project are prose, not Google/NumPy sections,
# so napoleon is loaded but does little — it's harmless to keep for any that are.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = "mercury-harness"
html_static_path = []
html_theme_options = {
    "sidebar_hide_name": False,
}


# The docstrings are Markdown-flavoured prose (single backticks, stray asterisks), which
# reStructuredText reads as broken markup. Convert single backticks to literals, and show any
# docstring that still does not parse as a literal block, so the build stays warning-free
# without editing the source.
_SINGLE_TICK = re.compile(r"(?<![`\w])`([^`\n]+)`(?![`\w])")


def _parses_cleanly(text: str) -> bool:
    import io

    from docutils.core import publish_doctree

    stream = io.StringIO()
    try:
        publish_doctree(
            text,
            settings_overrides={"warning_stream": stream, "report_level": 2, "halt_level": 5},
        )
    except Exception:
        return False
    return not stream.getvalue().strip()


def _tame_docstring(app, what, name, obj, options, lines):
    text = _SINGLE_TICK.sub(r"``\1``", "\n".join(lines))
    if not _parses_cleanly(text):
        text = "::\n\n" + "\n".join("    " + line for line in text.splitlines())
    lines[:] = text.splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", _tame_docstring)
