"""The report must reach whatever it was handed."""
from report import Report


class Collector:
    """An ordinary caller: it has write(), and is not a subclass of anything."""

    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


def test_a_report_writes_its_title_and_rows():
    sink = Collector()
    written = Report("Q1").add("apples", 3).add("pears", 4).render(sink)
    assert written == 4, f"wrote {written} lines: {sink.lines}"
    assert sink.lines[0] == "== Q1 =="
    assert "apples: 3" in sink.lines and "pears: 4" in sink.lines


def test_the_total_is_written_last():
    sink = Collector()
    Report("Q1").add("a", 3).add("b", 4).render(sink)
    assert sink.lines[-1] == "total: 7", sink.lines


def test_an_empty_report_still_writes_its_frame():
    sink = Collector()
    assert Report("Empty").render(sink) == 2
    assert sink.lines == ["== Empty ==", "total: 0"]


def test_the_librarys_own_sink_works_too():
    from report import Sink

    sink = Sink()
    assert Report("Q1").add("a", 1).render(sink) == 3
    assert sink.lines[0] == "== Q1 =="


def test_no_sink_is_still_tolerated():
    assert Report("t").render(None) == 0
