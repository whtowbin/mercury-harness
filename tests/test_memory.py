"""What the agent remembers between runs.

The motivating failure: a run was asked "why did that test fail?" immediately
after the run that broke it, and nothing in the harness could answer, because
the previous run's findings lived only in a trace file nothing read back.
"""

from __future__ import annotations

import pytest

from agent.memory import MemoryStore, keywords_of, parse_sections, score


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


def summary_for(goal_ok=True, **overrides) -> dict:
    base = {
        "ok": goal_ok,
        "changed_files": ["books/core.py", "tests/test_books.py"],
        "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
        "final_check": {"ok": True},
        "plan": {"steps": [{"id": "s1", "summary": "added a Library.add method"}]},
        "suggested_next_step": "",
        "error": "",
    }
    base.update(overrides)
    return base


# -- keywords -------------------------------------------------------------


def test_common_words_are_not_keywords():
    assert "the" not in keywords_of("the parser")
    assert "parser" in keywords_of("the parser")


def test_short_tokens_are_ignored():
    assert keywords_of("a b cd parser") == {"parser"}


# -- parsing --------------------------------------------------------------


def test_sections_split_on_headings():
    sections = parse_sections("## one\nbody one\n\n## two\nbody two")
    assert [s.title for s in sections] == ["one", "two"]


def test_text_before_the_first_heading_is_ignored():
    assert len(parse_sections("preamble\n\n## only\nbody")) == 1


def test_declared_files_and_keywords_are_picked_up():
    section = parse_sections("## t\nfiles: a/b.py, c.py\nkeywords: parsing, cache\n")[0]
    assert section.files == {"a/b.py", "c.py"}
    assert "parsing" in section.keywords


def test_a_files_line_contributes_its_basename_as_a_keyword():
    """So a goal naming core.py matches a run that touched books/core.py."""
    section = parse_sections("## t\nfiles: books/core.py\n")[0]
    assert "core.py" in section.keywords


def test_a_hand_written_section_without_metadata_still_parses():
    section = parse_sections("## just a note\nremember the parser is recursive")[0]
    assert "parser" in section.keywords


# -- scoring --------------------------------------------------------------


def test_file_overlap_outweighs_a_single_word_match():
    by_file = parse_sections("## a\nfiles: core.py\n")[0]
    by_word = parse_sections("## b\nsomething about parser\n")[0]
    assert score(by_file, {"parser"}, {"core.py"}) > score(by_word, {"parser"}, {"core.py"})


def test_an_unrelated_section_scores_zero():
    section = parse_sections("## a\nfiles: unrelated.py\nkeywords: unrelated\n")[0]
    assert score(section, {"parser"}, {"core.py"}) == 0


# -- writing and reading back --------------------------------------------


def test_a_recorded_run_can_be_read_back(store):
    store.record_run("add a Library.add method", summary_for())
    text = store.memory_path.read_text()
    assert "Library.add" in text or "add a Library.add method" in text
    assert "books/core.py" in text
    assert "outcome: ok" in text


def test_a_failed_run_records_what_was_left_broken(store):
    store.record_run(
        "do a thing",
        summary_for(
            goal_ok=False,
            steps={"total": 2, "failed": ["s2"], "abandoned": [], "unverified": []},
            final_check={
                "ok": False,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "2 failed",
                "verify_failures": [],
                "new_failures": ["tests/test_flappy.py::test_pipes"],
            },
        ),
    )
    text = store.memory_path.read_text()
    assert "outcome: FAILED" in text
    assert "unresolved:" in text
    assert "s2" in text
    assert "test_pipes" in text


def test_recording_appends_rather_than_replacing(store):
    store.record_run("first goal", summary_for())
    store.record_run("second goal", summary_for())
    assert len(store.sections(include_scratch=False)) == 2


def test_relevant_returns_nothing_when_there_is_no_memory(store):
    assert store.relevant("anything") == ""


def test_the_most_recent_run_is_recalled_even_without_word_overlap(store):
    """The case that motivated this: "why did that fail?" shares no
    vocabulary with the run that caused the failure."""
    store.record_run("build the flappy bird pipes", summary_for(goal_ok=False))
    recalled = store.relevant("why did that test fail?")
    assert "flappy" in recalled


def test_an_older_but_relevant_section_is_recalled_over_an_unrelated_new_one(store):
    store.record_run("rewrite the books parser", summary_for(changed_files=["books/core.py"]))
    store.record_run("unrelated flappy work", summary_for(changed_files=["flappy/core.py"]))
    recalled = store.relevant("fix the books parser again")
    assert "books" in recalled


def test_recall_respects_its_token_budget(store):
    for i in range(40):
        store.record_run(f"goal number {i} about parsing", summary_for())
    recalled = store.relevant("parsing", budget_tokens=200)
    assert len(recalled) < 4_000


# -- the optional judge (agent/relevance.py) -------------------------------
# `relevant`/`search` fall back to the free keyword-overlap ranking above
# when judge=None (every test above this point exercises exactly that path,
# unchanged); these check the upgrade path a RelevanceJudge provides.


def test_a_judge_can_surface_a_section_keyword_overlap_would_miss(store):
    """The exact gap `relevant`'s docstring names: a follow-up sharing no
    vocabulary with the run it refers to. always_recent already patches this
    for the newest section; a judge should reach further back too."""
    store.record_run("first: build the parser module", summary_for(changed_files=["parser.py"]))
    store.record_run(
        "second: totally unrelated flappy bird work", summary_for(changed_files=["flappy.py"])
    )
    store.record_run("third: also unrelated audio work", summary_for(changed_files=["audio.py"]))

    class _FavorsParser:
        def score(self, context, candidates):
            return [1.0 if "parser" in c.lower() else 0.0 for c in candidates]

    recalled = store.relevant(
        "why does the thing from a while back not work?", always_recent=0, judge=_FavorsParser()
    )
    assert "parser" in recalled


def test_budget_truncation_drops_the_lowest_scored_section_not_the_oldest(store):
    """The bug a live Jev call actually caught: truncation used to re-sort by
    recency before trimming to budget_tokens, so a judge could correctly
    rank an older, genuinely relevant section above a newer, unrelated one
    and still lose it to truncation purely for being older. `chosen` must
    stay in score order (recent-forced sections first, then best-scored)
    all the way through the budget loop -- only the final *presentation*
    order is chronological."""
    store.record_run("first: build the parser module", summary_for(changed_files=["parser.py"]))
    store.record_run(
        "second: totally unrelated flappy bird work", summary_for(changed_files=["flappy.py"])
    )

    class _FavorsParser:
        def score(self, context, candidates):
            return [1.0 if "parser" in c.lower() else 0.0 for c in candidates]

    # Budget only large enough for one section -- forces real truncation,
    # not just filtering by score > 0.
    recalled = store.relevant(
        "something from before", always_recent=0, budget_tokens=40, judge=_FavorsParser()
    )
    assert "parser" in recalled
    assert "flappy" not in recalled


def test_a_failing_judge_falls_back_to_keyword_ranking_not_a_crash(store):
    store.record_run("parser work", summary_for(changed_files=["parser.py"]))

    class _Blows:
        def score(self, context, candidates):
            raise RuntimeError("network is down")

    recalled = store.relevant("parser work", judge=_Blows())
    assert "parser" in recalled


def test_search_with_a_judge_still_excludes_zero_scored_sections(store):
    store.record_run("parser work", summary_for(changed_files=["parser.py"]))
    store.record_run("cache work", summary_for(changed_files=["cache.py"]))

    class _OnlyParser:
        def score(self, context, candidates):
            return [1.0 if "parser" in c.lower() else 0.0 for c in candidates]

    hits = store.search("anything", judge=_OnlyParser())
    assert len(hits) == 1
    assert "parser" in hits[0].render()


def test_search_with_no_judge_is_unaffected_by_the_new_parameter(store):
    store.record_run("cache work", summary_for(changed_files=["cache.py"]))
    store.record_run("parser work", summary_for(changed_files=["parser.py"]))
    hits = store.search("parser.py")
    assert hits and "parser" in hits[0].render()


# -- the scratchpad -------------------------------------------------------


def test_a_note_lands_in_the_scratchpad_not_durable_memory(store):
    store.start_session()
    store.note("the parser is recursive", ["parser"], title="observation")
    assert "recursive" in store.scratch_path.read_text()
    assert not store.memory_path.exists() or "recursive" not in store.memory_path.read_text()


def test_starting_a_session_clears_the_scratchpad(store):
    store.start_session()
    store.note("old thinking", [])
    store.start_session()
    assert "old thinking" not in store.scratch_path.read_text()


def test_starting_a_session_does_not_clear_durable_memory(store):
    store.record_run("a goal", summary_for())
    store.start_session()
    assert "a goal" in store.memory_path.read_text()


def test_scratch_notes_are_searchable_alongside_memory(store):
    store.start_session()
    store.note("the widget cache is stale on restart", ["widget", "cache"])
    hits = store.search("widget cache")
    assert hits and "stale on restart" in hits[0].render()


def test_scratch_sections_are_labelled_as_this_session(store):
    store.start_session()
    store.note("a thought", ["thought"])
    titles = [s.title for s in store.sections()]
    assert any("this session" in t for t in titles)


# -- searching ------------------------------------------------------------


def test_search_finds_nothing_for_an_unrelated_query(store):
    store.record_run("parser work", summary_for(changed_files=["parser.py"]))
    assert store.search("kubernetes") == []


def test_search_ranks_the_better_match_first(store):
    store.record_run("cache work", summary_for(changed_files=["cache.py"]))
    store.record_run("parser work", summary_for(changed_files=["parser.py"]))
    hits = store.search("parser.py")
    assert "parser" in hits[0].render()


# -- compaction -----------------------------------------------------------


def test_compaction_keeps_the_recent_entries_and_condenses_the_rest(store):
    for i in range(12):
        store.record_run(f"goal {i}", summary_for())
    store.compact(keep_recent=3, summary_text="everything earlier, in brief")
    sections = store.sections(include_scratch=False)
    assert len(sections) == 4  # the condensed head plus three kept
    assert "everything earlier, in brief" in store.memory_path.read_text()
    assert "goal 11" in store.memory_path.read_text()


def test_compaction_does_nothing_when_there_is_little_to_compact(store):
    store.record_run("only goal", summary_for())
    store.compact(keep_recent=8, summary_text="should not appear")
    assert "should not appear" not in store.memory_path.read_text()


def test_oversized_reflects_the_budget(store):
    store.record_run("a goal", summary_for())
    assert not store.oversized(10_000)
    assert store.oversized(1)


# -- robustness -----------------------------------------------------------


def test_an_unwritable_directory_does_not_raise(tmp_path):
    """Memory is a convenience; failing to write it must never fail a run."""
    target = tmp_path / "file-not-a-dir"
    target.write_text("x")
    store = MemoryStore(target)
    store.record_run("a goal", summary_for())  # must not raise
    assert store.relevant("a goal") == ""


# -- a goal pasted from a bordered terminal panel --------------------------
# The traced regression: a goal copy-pasted from a previous run's own
# rendered failure panel carried embedded newlines and box-drawing
# characters. record_run's section heading is a markdown "## " line, so an
# embedded newline there means parse_sections finds the *next* line and
# treats it as this section's body instead of the rest of the title --
# everything after the first newline silently fell out of the heading.


def test_a_goal_with_embedded_newlines_becomes_one_heading_line(store):
    goal = "Fix these issues:\n•   step [x] failed\n│  │\n•   verify still fails"
    store.record_run(goal, summary_for())
    sections = parse_sections(store._read(store.memory_path))
    assert len(sections) == 1
    assert "\n" not in sections[0].title
    assert "verify still fails" in sections[0].title


def test_the_full_goal_is_still_searchable_after_flattening(store):
    """Collapsing whitespace must not lose words the goal still needs to be
    findable by -- keywords_of tokenises on word boundaries either way."""
    goal = "Fix these issues:\n• correct the pandoc verification"
    store.record_run(goal, summary_for())
    assert "pandoc" in store.sections()[0].keywords
