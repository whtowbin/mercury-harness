"""Order storage. Money is held in whole US cents, never floats."""


class Store:
    def __init__(self):
        self._orders = {}

    def add(self, order_id, cents):
        """Record an order. `cents` is a whole number of US cents."""
        if not isinstance(cents, int):
            raise TypeError("amounts are stored in whole cents")
        self._orders[order_id] = cents
        return order_id

    def amount(self, order_id):
        """The stored amount, in US cents."""
        return self._orders[order_id]

    def total(self):
        """Sum of every order, in US cents."""
        return sum(self._orders.values())

    def ids(self):
        return list(self._orders)
