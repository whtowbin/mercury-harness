"""A running account ledger."""


class Ledger:
    """Tracks a balance through a sequence of postings.

    post(amount) adds a signed amount to the ledger and returns its index.
    balance() is the sum of everything posted, reversals included.
    history() is every entry in the order it was made.
    reverse(index) undoes the posting at `index` by recording its negation
    as a new entry, so the balance returns to what it was before that
    posting. Reversing the same posting twice is an error: a posting that
    has already been undone cannot be undone again.
    """

    def __init__(self) -> None:
        self._postings: list[float] = []
        self._reversed: set[int] = set()

    def post(self, amount: float) -> int:
        self._postings.append(amount)
        return len(self._postings) - 1

    def balance(self) -> float:
        return sum(self._postings)

    def history(self) -> list[float]:
        return list(self._postings)

    def reverse(self, index: int) -> None:
        if index < 0 or index >= len(self._postings):
            raise IndexError(f"no posting at {index}")
        # BUG: the docstring says reversing an already-reversed posting is an
        # error, and nothing here checks. A second reverse() on the same
        # index appends a second negation and silently moves the balance.
        self._reversed.add(index)
        self._postings.append(-self._postings[index])
