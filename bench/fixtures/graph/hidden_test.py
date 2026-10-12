"""Hidden tests for `graph`."""
import pytest

from graph import Graph


def test_add_edge_creates_both_nodes():
    g = Graph()
    g.add_edge("a", "b")
    assert g.neighbors("a") == {"b"}
    assert g.neighbors("b") == {"a"}


def test_duplicate_edge_is_a_noop():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "b")
    assert g.neighbors("a") == {"b"}


def test_neighbors_of_unknown_node_raises():
    g = Graph()
    with pytest.raises(KeyError):
        g.neighbors("nope")


def test_shortest_path_of_a_node_to_itself():
    g = Graph()
    g.add_edge("a", "b")
    assert g.shortest_path("a", "a") == ["a"]


def test_shortest_path_direct_edge():
    g = Graph()
    g.add_edge("a", "b")
    assert g.shortest_path("a", "b") == ["a", "b"]


def test_shortest_path_is_actually_shortest():
    g = Graph()
    # a-b-c-d is length 3; a-d is a direct shortcut.
    for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]:
        g.add_edge(x, y)
    path = g.shortest_path("a", "c")
    assert len(path) == 3          # a -> d -> c, not a -> b -> c
    assert path[0] == "a" and path[-1] == "c"


def test_shortest_path_returns_none_when_unreachable():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("x", "y")
    assert g.shortest_path("a", "y") is None


def test_shortest_path_of_unknown_node_returns_none():
    g = Graph()
    g.add_edge("a", "b")
    assert g.shortest_path("a", "nope") is None
    assert g.shortest_path("nope", "a") is None


def test_shortest_path_on_a_cycle():
    g = Graph()
    for x, y in [("a", "b"), ("b", "c"), ("c", "a")]:
        g.add_edge(x, y)
    assert len(g.shortest_path("a", "c")) == 2   # direct edge, not around the cycle


def test_path_endpoints_are_adjacent_along_the_way():
    g = Graph()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        g.add_edge(x, y)
    path = g.shortest_path("a", "e")
    assert path == ["a", "b", "c", "d", "e"]
    for u, v in zip(path, path[1:]):
        assert v in g.neighbors(u)


def test_a_larger_grid_graph():
    g = Graph()
    n = 20
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                g.add_edge((r, c), (r, c + 1))
            if r + 1 < n:
                g.add_edge((r, c), (r + 1, c))
    path = g.shortest_path((0, 0), (n - 1, n - 1))
    assert len(path) == 2 * (n - 1) + 1   # Manhattan distance on a grid
