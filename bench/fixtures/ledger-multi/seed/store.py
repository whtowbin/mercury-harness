"""Persistence for ledger entries."""


class Store:
    """An in-memory table of postings, keyed by an auto-incrementing id."""

    def __init__(self) -> None:
        self._rows: dict[int, float] = {}
        self._next_id = 0

    def add(self, amount: float) -> int:
        row_id = self._next_id
        self._rows[row_id] = amount
        self._next_id += 1
        return row_id

    def get(self, row_id: int) -> float:
        return self._rows[row_id]

    def all_ids(self) -> list[int]:
        return sorted(self._rows)

    def remove(self, row_id: int) -> None:
        del self._rows[row_id]
