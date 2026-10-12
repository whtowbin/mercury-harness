from store import Store


def test_add_returns_an_id():
    store = Store()
    assert store.add(10.0) == 0
    assert store.add(20.0) == 1


def test_get_returns_the_amount():
    store = Store()
    row_id = store.add(5.5)
    assert store.get(row_id) == 5.5


def test_remove_deletes_the_row():
    store = Store()
    row_id = store.add(1.0)
    store.remove(row_id)
    assert row_id not in store.all_ids()


def test_all_ids_is_sorted():
    store = Store()
    store.add(1.0)
    store.add(2.0)
    assert store.all_ids() == [0, 1]
