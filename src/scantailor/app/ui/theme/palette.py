"""QPalette for the light and dark looks so natively drawn parts match the QSS."""

from PySide6 import QtGui

from .tokens import Tokens, qcolor


def build_palette(t: Tokens) -> QtGui.QPalette:
    """Build a palette from a token set.

    Args:
        t (Tokens): Colours for the theme.

    Returns:
        QtGui.QPalette: Palette with window, base, text, highlight and accent
            roles set.
    """
    role = QtGui.QPalette.ColorRole
    group = QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    window = QtGui.QColor(t.bg_top)
    base = QtGui.QColor("#232838" if t.dark else "#ffffff")
    text = QtGui.QColor(t.text)
    dim = QtGui.QColor(t.text_dim)
    # Item delegates that paint selection with palette.highlight() (the
    # thumbnail list) get the translucent tint instead of a solid block.
    tint = qcolor(t.accent_tint)
    for color_role, color in (
        (role.Window, window),
        (role.WindowText, text),
        (role.Base, base),
        (role.AlternateBase, window),
        (role.Text, text),
        (role.Button, base),
        (role.ButtonText, text),
        (role.ToolTipBase, QtGui.QColor(t.bg_bottom)),
        (role.ToolTipText, text),
        (role.PlaceholderText, dim),
        (role.Highlight, tint),
        (role.Accent, QtGui.QColor(t.accent)),
        (role.HighlightedText, text),
        (role.Link, QtGui.QColor(t.link)),
        (role.Dark, QtGui.QColor(t.viewport)),
    ):
        palette.setColor(color_role, color)
    disabled_text = QtGui.QColor(t.text_disabled)
    disabled_fill = qcolor(t.field_disabled)
    disabled_tint = QtGui.QColor(tint)
    disabled_tint.setAlpha(tint.alpha() // 2)
    for color_role, color in (
        (role.WindowText, disabled_text),
        (role.Text, disabled_text),
        (role.ButtonText, disabled_text),
        (role.HighlightedText, disabled_text),
        (role.PlaceholderText, disabled_text),
        (role.Base, disabled_fill),
        (role.Button, disabled_fill),
        (role.Highlight, disabled_tint),
    ):
        palette.setColor(group.Disabled, color_role, color)
    return palette
