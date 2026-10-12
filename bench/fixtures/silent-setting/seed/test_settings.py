"""Existing tests. They pass."""
from settings import Renderer, Settings


def test_settings_store_indent():
    assert Settings(indent=8).indent == 8


def test_settings_store_bullet():
    assert Settings(bullet="*").bullet == "*"


def test_update_applies_known_keys():
    s = Settings()
    s.update(indent=4, width=100)
    assert s.indent == 4 and s.width == 100


def test_render_item_uses_the_settings_it_is_given():
    assert Renderer().render_item("a", Settings(indent=4)) == "    - a"
