"""Existing tests. They pass."""
from report import Report


class StubSink:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


def test_add_accumulates_rows():
    r = Report("t").add("a", 1).add("b", 2)
    assert r.rows == [("a", 1), ("b", 2)]


def test_render_accepts_a_sink():
    Report("t").render(StubSink())


def test_render_handles_no_sink_without_raising():
    Report("t").render(None)


def test_title_is_kept():
    assert Report("Quarterly").title == "Quarterly"
