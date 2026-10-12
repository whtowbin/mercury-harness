"""Distance readings from field sensors.

Every reading is a (value, unit) pair. Sensors report in different units; the rest
of this module works in metres, via `to_metres`.

House rules for every function that aggregates readings (see `longest`):
- a value of None is a dropped radio packet, not a zero: skip it;
- a negative value is a sensor fault and must not be averaged or summed away:
  raise ValueError naming the reading.
"""

UNIT_TO_METRES = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}


def to_metres(value, unit):
    """`value` in `unit`, converted to metres. Raises ValueError for an unknown unit."""
    if unit not in UNIT_TO_METRES:
        raise ValueError(f"unknown unit {unit!r}")
    return value * UNIT_TO_METRES[unit]


def _usable(readings):
    """The readings an aggregate may use, under the house rules above."""
    for value, unit in readings:
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"sensor fault: negative reading {(value, unit)!r}")
        yield value, unit


def longest(readings):
    """The longest reading, as the original (value, unit) pair."""
    return max(_usable(readings), key=lambda r: to_metres(*r))
