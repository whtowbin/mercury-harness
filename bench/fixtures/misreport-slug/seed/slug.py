"""Turns post titles into URL slugs."""

import re
import unicodedata


def slugify(title, max_length=60):
    """A lowercase ASCII slug for `title`.

    Accented letters are transliterated to their base letter (é -> e), not
    dropped; every run of anything else becomes a single hyphen. Longer slugs
    are cut back to the last whole word that fits in `max_length`.
    """
    ascii_text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_text.lower())
    slug = "-".join(words)
    if len(slug) <= max_length:
        return slug
    if slug[max_length] == "-":
        return slug[:max_length]
    head = slug[:max_length]
    return head.rsplit("-", 1)[0] if "-" in head else head
