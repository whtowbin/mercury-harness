"""Renderer settings and the renderer that is supposed to honour them."""

DEFAULTS = {"indent": 2, "width": 80, "bullet": "-"}


class Settings:
    """User-adjustable rendering options."""

    def __init__(self, indent=None, width=None, bullet=None):
        self.indent = DEFAULTS["indent"] if indent is None else indent
        self.width = DEFAULTS["width"] if width is None else width
        self.bullet = DEFAULTS["bullet"] if bullet is None else bullet

    def update(self, **changes):
        """Apply changes, ignoring unknown keys."""
        for key, value in changes.items():
            if key in DEFAULTS:
                setattr(self, key, value)
        return self


class Renderer:
    """Renders a list of items as an indented bullet list."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else Settings()

    def render_item(self, text, settings=None):
        opts = settings if settings is not None else self.settings
        return " " * opts.indent + opts.bullet + " " + text

    def render(self, items):
        # Renders every item using the defaults instead of self.settings.
        return "\n".join(self.render_item(text, Settings()) for text in items)
