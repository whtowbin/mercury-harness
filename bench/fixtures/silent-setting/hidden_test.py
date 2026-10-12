"""The setting must reach the output, not merely be stored."""
from settings import Renderer, Settings


def test_custom_indent_reaches_rendered_output():
    out = Renderer(Settings(indent=6)).render(["alpha"])
    assert out == "      - alpha", out


def test_custom_bullet_reaches_rendered_output():
    out = Renderer(Settings(bullet="*")).render(["alpha"])
    assert out == "  * alpha", out


def test_changing_a_setting_after_construction_changes_the_output():
    r = Renderer(Settings())
    r.settings.update(indent=4, bullet=">")
    assert r.render(["x"]) == "    > x", r.render(["x"])


def test_defaults_still_apply_when_nothing_is_set():
    assert Renderer().render(["x", "y"]) == "  - x\n  - y"
