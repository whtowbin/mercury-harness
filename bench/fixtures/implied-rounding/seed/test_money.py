from money import add_tax, allocate


def test_add_tax():
    assert add_tax(1000, 8) == 1080


def test_allocate_keeps_every_cent():
    assert sum(allocate(100, [1, 1, 1])) == 100
