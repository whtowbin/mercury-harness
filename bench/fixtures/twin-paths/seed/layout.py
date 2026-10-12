"""Lays out a form: where each field is drawn, and where clicks land."""

TOP = 100
ROW_HEIGHT = 40
FIELD_X = 220
FIELD_WIDTH = 120
FIELD_HEIGHT = 30

FIELDS = ["name", "email", "phone", "address"]


class Form:
    def __init__(self, fields=None):
        self.fields = list(fields) if fields is not None else list(FIELDS)

    def draw_boxes(self):
        """Where each field's box is drawn, as (x, y, w, h) in order."""
        boxes = []
        y = TOP
        for _field in self.fields:
            boxes.append((FIELD_X, y, FIELD_WIDTH, FIELD_HEIGHT))
            y += ROW_HEIGHT
        return boxes

    def hit_targets(self):
        """Where a click selects each field, as (name, (x, y, w, h))."""
        targets = []
        y = TOP + 10
        for field in self.fields:
            targets.append((field, (FIELD_X, y, FIELD_WIDTH, FIELD_HEIGHT)))
            y += ROW_HEIGHT
        return targets

    def field_at(self, x, y):
        """The field containing point (x, y), or None."""
        for name, (bx, by, bw, bh) in self.hit_targets():
            if bx <= x < bx + bw and by <= y < by + bh:
                return name
        return None
