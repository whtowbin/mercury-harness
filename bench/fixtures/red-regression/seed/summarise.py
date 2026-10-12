"""Shortens text for fixed-width displays."""

ELLIPSIS = "…"


def _cut(text, limit):
    """Cut `text` to `limit` characters, never splitting a word if it can
    help it. Does not append anything."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def truncate(text, limit):
    """Shorten `text` to at most `limit` characters *including* the ellipsis.

    Text already within the limit is returned unchanged, with no ellipsis.
    """
    if len(text) <= limit:
        return text
    return _cut(text, limit) + ELLIPSIS


def preview(text, limit):
    """The first `limit` characters of `text`, on a word boundary, with no
    ellipsis — used for search snippets where the caller adds its own marker.

    Unlike truncate, the limit here is a hard cap on the text itself.
    """
    return _cut(text, limit)
