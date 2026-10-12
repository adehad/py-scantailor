"""Colour tokens for the light and dark looks."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Tokens:
    """Colours as CSS strings usable in both QSS and QColor.

    Attributes:
        dark (bool): Whether the theme has a dark window.
        bg_top (str): Top colour of the window gradient.
        bg_bottom (str): Bottom colour of the window gradient.
        panel (str): Translucent fill for docks and group boxes.
        panel_border (str): 1px inner border of frosted panels.
        panel_highlight (str): Top edge of a panel, lit like glass.
        link (str): Link colour on the neutral viewport.
        field (str): Fill for inputs and buttons.
        field_hover (str): Fill of inputs and buttons under the cursor.
        field_border (str): Border of inputs and buttons.
        indicator_border (str): Border of unchecked check boxes and radio buttons.
        text (str): Primary text.
        text_dim (str): Secondary and disabled text.
        accent (str): Selection and primary button colour.
        accent_hover (str): Accent under the cursor.
        accent_tint (str): Translucent accent for list selection.
        accent_text (str): Text drawn on the accent colour.
        positive (str): Semantic green for additive glyph accents.
        danger (str): Semantic red for destructive glyphs.
        viewport (str): Neutral colour behind page images.
        scroll_handle (str): Scroll bar handle.
    """

    dark: bool
    bg_top: str
    bg_bottom: str
    panel: str
    panel_border: str
    panel_highlight: str
    link: str
    field: str
    field_hover: str
    field_border: str
    indicator_border: str
    text: str
    text_dim: str
    accent: str
    accent_hover: str
    accent_tint: str
    accent_text: str
    positive: str
    danger: str
    viewport: str
    scroll_handle: str


LIGHT = Tokens(
    dark=False,
    bg_top="#e9eef7",
    bg_bottom="#dce3ef",
    panel="rgba(255, 255, 255, 184)",
    panel_border="rgba(15, 23, 42, 20)",
    panel_highlight="rgba(255, 255, 255, 230)",
    link="#0b3fa8",
    field="rgba(255, 255, 255, 190)",
    field_hover="rgba(255, 255, 255, 235)",
    field_border="rgba(20, 30, 60, 38)",
    indicator_border="rgba(20, 30, 60, 130)",
    text="#1c2330",
    text_dim="#5b6472",
    accent="#2f6fed",
    accent_hover="#4a82f2",
    accent_tint="rgba(47, 111, 237, 40)",
    accent_text="#ffffff",
    positive="#008a1a",
    danger="#d92d20",
    viewport="#c9ccd1",
    scroll_handle="rgba(28, 35, 48, 90)",
)

DARK = Tokens(
    dark=True,
    bg_top="#1b2030",
    bg_bottom="#11141d",
    panel="rgba(255, 255, 255, 18)",
    panel_border="rgba(255, 255, 255, 26)",
    panel_highlight="rgba(255, 255, 255, 46)",
    link="#8ab8ff",
    field="rgba(255, 255, 255, 22)",
    field_hover="rgba(255, 255, 255, 38)",
    field_border="rgba(255, 255, 255, 40)",
    indicator_border="rgba(255, 255, 255, 120)",
    text="#e8ecf4",
    text_dim="#9aa3b5",
    accent="#5b9bff",
    accent_hover="#78adff",
    accent_tint="rgba(91, 155, 255, 56)",
    accent_text="#0b1220",
    positive="#5ccf72",
    danger="#ff7b72",
    viewport="#2b2d31",
    scroll_handle="rgba(232, 236, 244, 90)",
)
