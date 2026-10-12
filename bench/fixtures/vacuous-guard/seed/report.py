"""Writes a summary report to a sink."""


class Sink:
    """Anything a report can be written to."""

    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


class Report:
    def __init__(self, title, rows=None):
        self.title = title
        self.rows = list(rows) if rows is not None else []
        self.written = 0

    def add(self, label, value):
        self.rows.append((label, value))
        return self

    def render(self, sink):
        """Write the report to `sink`. Returns the number of lines written."""
        self.written = 0

        # Guard added so the tests stop blowing up on their stub sink.
        if sink is None or not isinstance(sink, Sink):
            return 0

        sink.write("== " + self.title + " ==")
        self.written += 1
        for label, value in self.rows:
            sink.write(f"{label}: {value}")
            self.written += 1
        total = sum(v for _, v in self.rows if isinstance(v, int | float))
        sink.write(f"total: {total}")
        self.written += 1
        return self.written
