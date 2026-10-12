"""Verify commands for structured data and prose -- yq-first, with fallbacks
for a host without it, and pandoc's real-but-narrow Markdown parse check.
Nothing here shells out for real; every test controls `shutil.which` the
same way test_plan_gate.py's jq/yq availability tests already do.
"""

from __future__ import annotations

from agent import dataformats

# -- build_validator: yq present, one command covers every format ---------


def test_yq_covers_every_format_in_one_command(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    command = dataformats.build_validator(["a.json", "b.yaml", "c.toml", "d.xml"])
    assert command == "yq '.' a.json b.yaml c.toml d.xml"


def test_yq_is_tried_before_any_fallback(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    assert dataformats.build_validator(["a.json"]) == "yq '.' a.json"


def test_files_with_no_recognised_extension_are_ignored(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    assert dataformats.build_validator(["README.md", "notes.txt"]) == ""


def test_no_files_at_all_produces_nothing():
    assert dataformats.build_validator([]) == ""


# -- build_validator: yq absent, per-format fallbacks -----------------------


def _which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def test_json_falls_back_to_jq(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"jq"}))
    assert dataformats.build_validator(["a.json"]) == "jq empty a.json"


def test_json_is_skipped_without_jq_or_yq(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which(set()))
    assert dataformats.build_validator(["a.json"]) == ""


def test_toml_always_gets_the_stdlib_tomllib_check(monkeypatch):
    """No availability check needed -- tomllib is stdlib since 3.11."""
    monkeypatch.setattr(dataformats.shutil, "which", _which(set()))
    command = dataformats.build_validator(["pyproject.toml"])
    assert "tomllib" in command and "pyproject.toml" in command


def test_xml_falls_back_to_xmllint(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"xmllint"}))
    assert dataformats.build_validator(["a.xml"]) == "xmllint --noout a.xml"


def test_xml_is_skipped_without_xmllint_or_yq(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which(set()))
    assert dataformats.build_validator(["a.xml"]) == ""


def test_yaml_has_no_fallback_without_yq(monkeypatch):
    """Unchanged from before this module existed: yaml/csv/tsv/ini/properties
    only get a real check via yq -- no fallback exists, so this is a
    deliberate skip, not a bug."""
    monkeypatch.setattr(dataformats.shutil, "which", _which(set()))
    assert dataformats.build_validator(["a.yaml"]) == ""


def test_mixed_formats_without_yq_combine_every_available_fallback(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"jq", "xmllint"}))
    command = dataformats.build_validator(["a.json", "b.toml", "c.xml", "d.yaml"])
    assert "jq empty a.json" in command
    assert "tomllib" in command and "b.toml" in command
    assert "xmllint --noout c.xml" in command
    assert "d.yaml" not in command
    assert " && " in command


# -- build_prose_validator ---------------------------------------------------


def test_markdown_gets_a_pandoc_parse_check_when_pandoc_is_installed(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"pandoc"}))
    command = dataformats.build_prose_validator(["README.md"])
    assert command == "pandoc -f gfm -t plain -o /dev/null README.md"


def test_markdown_extension_variant_is_also_covered(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"pandoc"}))
    assert "notes.markdown" in dataformats.build_prose_validator(["notes.markdown"])


def test_prose_check_is_skipped_without_pandoc(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which(set()))
    assert dataformats.build_prose_validator(["README.md"]) == ""


def test_non_prose_files_are_ignored(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"pandoc"}))
    assert dataformats.build_prose_validator(["a.txt", "a.csv"]) == ""


def test_no_files_produces_nothing_even_with_pandoc(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", _which({"pandoc"}))
    assert dataformats.build_prose_validator([]) == ""
