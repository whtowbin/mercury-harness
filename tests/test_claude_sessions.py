"""Claude Code transcript analysis, on small synthetic transcripts."""

from agent import claude_sessions as cs


def _assistant(mid, ctx_read, out=100, write=0, blocks=None, uuid=None):
    return {
        "type": "assistant",
        "uuid": uuid or mid,
        "message": {
            "id": mid,
            "content": blocks or [{"type": "text", "text": "x"}],
            "usage": {
                "input_tokens": 1,
                "cache_read_input_tokens": ctx_read,
                "cache_creation_input_tokens": write,
                "output_tokens": out,
            },
        },
    }


def _use(mid, tool_id, name, tool_input, ctx_read=1000):
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
    return _assistant(mid, ctx_read, blocks=[block])


def _result(tool_id, text):
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": text}
    return {"type": "user", "message": {"content": [block]}}


def test_a_response_written_as_several_records_is_counted_once():
    records = [_assistant("m1", 5000, uuid="a"), _assistant("m1", 5000, uuid="b")]
    assert len(cs.turns(records)) == 1


def test_cache_reads_dominate_cost_in_a_long_context():
    shares = cs.cost_shares(cs.turns([_assistant(f"m{i}", 300_000) for i in range(10)]))
    assert shares["cache read"] > shares["output"]
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_context_buckets_put_the_cost_where_the_context_is_large():
    every = cs.turns([_assistant("a", 10_000), _assistant("b", 500_000)])
    buckets = cs.context_buckets(every)
    assert buckets["<50K"][0] == 0.5 and buckets["400K+"][0] == 0.5
    assert buckets["400K+"][1] > buckets["<50K"][1]


def test_a_lower_cap_costs_less_when_context_only_grows():
    session = cs.turns([_assistant(f"m{i}", 20_000 + i * 2_000, write=2_000) for i in range(300)])
    assert cs.simulate_cap([session], cap=150_000) < cs.simulate_cap([session], cap=900_000)


def test_reorientation_cost_can_erase_the_saving_of_capping():
    session = cs.turns([_assistant(f"m{i}", 20_000 + i * 2_000, write=2_000) for i in range(300)])
    cheap = cs.simulate_cap([session], cap=150_000, reorient=0)
    dear = cs.simulate_cap([session], cap=150_000, reorient=200_000)
    assert dear > cheap


def test_a_result_is_charged_for_every_turn_it_stays_in_context():
    records = [
        _use("m1", "t1", "Read", {"file_path": "/x/big.py"}),
        _result("t1", "y" * 3600),
        _assistant("m2", 2000),
        _assistant("m3", 3000),
    ]
    carry = cs.tool_carry(records)
    # 1000 tokens carried across the 2 turns that followed, at the cache-read price
    assert carry["Read:big.py"] == 1000 * 2 * cs.WEIGHTS["read"]


def test_carry_stops_at_a_compaction():
    records = [
        _use("m1", "t1", "Read", {"file_path": "/x/a.py"}, ctx_read=100_000),
        _result("t1", "y" * 3600),
        _assistant("m2", 110_000),
        _assistant("m3", 20_000),  # compacted: the result is gone
        _assistant("m4", 25_000),
    ]
    assert cs.tool_carry(records)["Read:a.py"] == 1000 * 1 * cs.WEIGHTS["read"]


def test_rereading_an_unchanged_file_is_wasted_but_after_an_edit_it_is_not():
    read = {"file_path": "/x/a.py"}
    records = [
        _use("m1", "t1", "Read", read),
        _result("t1", "y" * 360),
        _use("m2", "t2", "Read", read),
        _result("t2", "y" * 360),
        _use("m3", "t3", "Edit", read),
        _result("t3", "ok"),
        _use("m4", "t4", "Read", read),
        _result("t4", "y" * 360),
    ]
    assert [p for _, p in cs.unchanged_rereads(records)] == ["/x/a.py"]


def test_default_dir_slugs_the_working_directory_like_claude_code_does(tmp_path):
    slug = cs.default_dir("/Users/h/Projects/AI Agent").name
    assert slug == "-Users-h-Projects-AI-Agent"


def test_report_says_so_when_there_are_no_transcripts(tmp_path):
    assert "no transcripts" in cs.report(tmp_path)


def _timed(mid, tool_id, name, tool_input, at):
    record = _use(mid, tool_id, name, tool_input)
    record["timestamp"] = at
    return record


def _timed_result(tool_id, text, at, error=False):
    record = _result(tool_id, text)
    record["timestamp"] = at
    record["message"]["content"][0]["is_error"] = error
    return record


def test_a_command_is_timed_and_its_habitual_cd_prefix_ignored():
    records = [
        _timed(
            "m1", "t1", "Bash", {"command": 'cd "/x y" && uv run pytest -q'}, "2026-09-24T10:00:00Z"
        ),
        _timed_result("t1", "ok", "2026-09-24T10:00:30Z"),
    ]
    assert cs.slow_commands(records) == {"uv run": [30.0]}
    assert list(cs.repeated_commands(records)) == ["uv run pytest"]


def test_polling_on_background_work_is_not_counted_as_slow():
    records = [
        _timed(
            "m1",
            "t1",
            "Bash",
            {"command": "until grep -q done f; do sleep 5; done"},
            "2026-09-24T10:00:00Z",
        ),
        _timed_result("t1", "", "2026-09-24T10:30:00Z"),
    ]
    assert cs.slow_commands(records) == {}


def test_tool_errors_are_counted_by_their_opening_text():
    records = [
        _use("m1", "t1", "Bash", {"command": "x"}),
        _timed_result("t1", "Not logged in", "2026-09-24T10:00:00Z", error=True),
    ]
    assert cs.error_kinds(records) == {"Not logged in": 1}


def test_whole_suite_runs_are_told_from_targeted_ones():
    def run(i, command, seconds):
        return [
            _timed(f"a{i}", f"t{i}", "Bash", {"command": command}, "2026-09-24T10:00:00Z"),
            _timed_result(f"t{i}", "", f"2026-09-24T10:00:{seconds:02d}Z"),
        ]

    records = (
        run(1, "uv run pytest -q", 20)
        + run(2, "uv run pytest -n auto -q", 9)
        + run(3, "uv run pytest tests/test_x.py -q", 3)
        + run(4, "uv run pytest -k thing", 2)
    )
    assert cs.full_suite_pytest(records) == [(False, 20.0), (True, 9.0)]


def test_compactions_are_counted_from_context_drops():
    every = cs.turns([_assistant("a", 100_000), _assistant("b", 120_000), _assistant("c", 20_000)])
    assert cs.compactions(every) == 1
