# pricing

`round_price(amount)` rounds a price to a whole number of dollars using
**round-half-to-even** (banker's rounding): 2.5 rounds to 2 and 3.5 rounds to 4.
This is deliberate and required by our accounting standard, which forbids the
upward bias of always rounding halves up. Do not change it.
