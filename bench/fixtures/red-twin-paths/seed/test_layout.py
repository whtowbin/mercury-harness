"""Existing tests. They pass."""
from layout import FIELD_HEIGHT, FIELD_WIDTH, Form


def test_one_box_per_field():
    assert len(Form().draw_boxes()) == 4


def test_one_target_per_field():
    assert len(Form().hit_targets()) == 4


def test_boxes_have_the_configured_size():
    for _x, _y, w, h in Form().draw_boxes():
        assert (w, h) == (FIELD_WIDTH, FIELD_HEIGHT)


def test_field_at_finds_something_inside_a_target():
    form = Form()
    _name, (x, y, _w, _h) = form.hit_targets()[0]
    assert form.field_at(x + 1, y + 1) == "name"
