from invoice import invoice_total, render


def test_whole_dollar_lines_add_up():
    assert invoice_total([(2.00, 3), (5.00, 1)]) == 1100


def test_render_shows_dollars_and_cents():
    assert render([(1.50, 2)]) == "Total: $3.00"
