"""`bench archive`: copy a sweep's records and traces somewhere that outlives a scratch dir."""

from __future__ import annotations

import json

import pytest

from bench import archive, cli, diagnostics


@pytest.fixture
def sweep(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    records = []
    for i in range(3):
        trace = scratch / f"run-{i}.jsonl"
        trace.write_text(json.dumps({"kind": "plan", "draft": ["x"], "steps": []}) + "\n")
        records.append({"config": "a", "task": "t", "accuracy": 1.0, "trace_path": str(trace)})
    result_file = scratch / "sweep.jsonl"
    result_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return tmp_path, result_file


def run(tmp_path, result_file, name="s1", **kwargs):
    return archive.archive(
        [result_file],
        name,
        results_dir=tmp_path / "results",
        traces_dir=tmp_path / "traces",
        **kwargs,
    )


def test_records_and_traces_are_copied_and_trace_paths_rewritten(sweep):
    tmp_path, result_file = sweep
    result = run(tmp_path, result_file)
    assert (result.records, result.traces_copied, result.traces_missing) == (3, 3, 0)
    copied = [json.loads(line) for line in result.results_path.read_text().splitlines()]
    for record in copied:
        assert record["trace_path"].startswith(str((tmp_path / "traces" / "s1").resolve()))
        assert (tmp_path / "traces" / "s1" / record["trace_path"].rsplit("/", 1)[1]).is_file()


def test_trace_paths_are_repo_relative_when_archived_under_the_working_directory(
    sweep, monkeypatch
):
    tmp_path, result_file = sweep
    monkeypatch.chdir(tmp_path)
    result = run(tmp_path, result_file)
    copied = [json.loads(line) for line in result.results_path.read_text().splitlines()]
    for record in copied:
        assert record["trace_path"].startswith("traces/s1/run-"), record["trace_path"]
        assert (tmp_path / record["trace_path"]).is_file()


def test_the_originals_are_left_alone(sweep):
    tmp_path, result_file = sweep
    before = result_file.read_text()
    run(tmp_path, result_file)
    assert result_file.read_text() == before
    assert (result_file.parent / "run-0.jsonl").is_file()


def test_an_archive_survives_its_scratch_directory_being_deleted(sweep):
    """The point of it: the diagnostics view reads each record's trace, so the
    archived copy must not depend on the original location."""
    import shutil

    from bench import diagnostics

    tmp_path, result_file = sweep
    result = run(tmp_path, result_file)
    shutil.rmtree(result_file.parent)
    record = json.loads(result.results_path.read_text().splitlines()[0])
    assert diagnostics.from_trace(record["trace_path"]).plan_steps == 0
    assert diagnostics.from_trace(record["trace_path"]).plan_draft_words == 1


def test_a_missing_trace_is_counted_and_the_record_is_kept(sweep):
    tmp_path, result_file = sweep
    (result_file.parent / "run-1.jsonl").unlink()
    result = run(tmp_path, result_file)
    assert (result.records, result.traces_copied, result.traces_missing) == (3, 2, 1)


def test_an_existing_archive_is_not_overwritten_without_force(sweep):
    tmp_path, result_file = sweep
    run(tmp_path, result_file)
    with pytest.raises(archive.ArchiveError, match="already exists"):
        run(tmp_path, result_file)
    assert run(tmp_path, result_file, force=True).records == 3


def test_several_result_files_are_archived_together(sweep):
    tmp_path, result_file = sweep
    other = result_file.parent / "other.jsonl"
    other.write_text(json.dumps({"config": "b", "task": "t", "accuracy": 0.0}) + "\n")
    result = archive.archive(
        [result_file, other],
        "both",
        results_dir=tmp_path / "results",
        traces_dir=tmp_path / "traces",
    )
    assert result.records == 4


@pytest.mark.parametrize("bad", ["", "a/b", ".hidden"])
def test_a_bad_name_is_refused(sweep, bad):
    tmp_path, result_file = sweep
    with pytest.raises(archive.ArchiveError):
        run(tmp_path, result_file, name=bad)


def test_an_unreadable_or_empty_file_is_an_error_not_a_traceback(tmp_path):
    with pytest.raises(archive.ArchiveError, match="cannot read"):
        archive.archive([tmp_path / "nope.jsonl"], "x", tmp_path / "r", tmp_path / "t")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(archive.ArchiveError, match="no records"):
        archive.archive([empty], "x", tmp_path / "r", tmp_path / "t")


def test_the_command_reports_and_exits_nonzero_on_error(sweep, monkeypatch, capsys):
    tmp_path, result_file = sweep
    monkeypatch.chdir(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["archive", str(result_file), "--name", "s1"])
    assert args.func(args) == 0
    assert "archived 3 record(s)" in capsys.readouterr().out
    assert args.func(args) == 1
    assert "already exists" in capsys.readouterr().err


def test_trace_paths_under_home_are_stored_with_a_tilde(sweep, monkeypatch):
    """A sweep archived outside the repo but under home names its traces `~/...`, not the
    absolute path: that put the author's username into two committed results files."""
    tmp_path, result_file = sweep
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path / "scratch")
    result = run(tmp_path, result_file)
    copied = [json.loads(line) for line in result.results_path.read_text().splitlines()]
    for record in copied:
        assert record["trace_path"].startswith("~/traces/s1/run-"), record["trace_path"]


def test_a_tilde_trace_path_is_read_and_archived(sweep, monkeypatch):
    tmp_path, result_file = sweep
    monkeypatch.setenv("HOME", str(tmp_path))
    records = [json.loads(line) for line in result_file.read_text().splitlines()]
    for record in records:
        record["trace_path"] = "~/scratch/" + record["trace_path"].rsplit("/", 1)[1]
    result_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert diagnostics.from_trace(records[0]["trace_path"]) == diagnostics.from_events(
        [{"kind": "plan", "draft": ["x"], "steps": []}]
    )
    result = run(tmp_path, result_file)
    assert (result.traces_copied, result.traces_missing) == (3, 0)


def test_portable_path_forms(tmp_path, monkeypatch):
    """The repo (the working directory) lies under home, as it does in use."""
    home = tmp_path / "home"
    (home / "repo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home / "repo")
    assert diagnostics.portable_path("") == ""
    assert diagnostics.portable_path(home / "repo" / "traces" / "r.jsonl") == "traces/r.jsonl"
    assert diagnostics.portable_path(home / ".cache" / "r.jsonl") == ("~/.cache/r.jsonl")
    assert diagnostics.portable_path("/opt/elsewhere/r.jsonl") == "/opt/elsewhere/r.jsonl"
