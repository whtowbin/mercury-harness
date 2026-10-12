"""A click must select the field whose box is drawn under it."""
from layout import Form


def test_every_drawn_box_has_its_own_hit_target():
    form = Form()
    boxes = form.draw_boxes()
    targets = [rect for _name, rect in form.hit_targets()]
    assert boxes == targets, f"drawn {boxes} but clickable {targets}"


def test_clicking_anywhere_on_a_drawn_box_selects_that_field():
    """Including its edges. A small offset between where a box is drawn and
    where it is clickable still works at the centre, which is why it survives
    casual testing and shows up as "I clicked it and nothing happened"."""
    form = Form()
    for name, (x, y, w, h) in zip(form.fields, form.draw_boxes()):
        for px, py in ((x + w // 2, y + h // 2),      # centre
                       (x + 1, y + 1),                 # top-left corner
                       (x + w - 2, y + h - 2)):        # bottom-right corner
            assert form.field_at(px, py) == name, (
                f"clicking {(px, py)} inside {name}'s drawn box selected "
                f"{form.field_at(px, py)!r}")


def test_no_two_drawn_boxes_overlap():
    boxes = Form().draw_boxes()
    for i, (_x1, y1, _w1, h1) in enumerate(boxes):
        for _x2, y2, _w2, _h2 in boxes[i + 1:]:
            assert y1 + h1 <= y2 or y2 + h1 <= y1, f"{boxes}"
