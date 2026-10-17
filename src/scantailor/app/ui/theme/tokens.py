"""Colour tokens for the light and dark looks."""

import dataclasses
import functools
import re

from PySide6 import QtGui


def qcolor(css: str) -> QtGui.QColor:
    """Parse a token colour, which may be ``#rrggbb`` or ``rgba(r, g, b, a)``.

    Args:
        css (str): Colour string from a token.

    Returns:
        QtGui.QColor: The colour; QColor itself cannot read the rgba() form,
            whose alpha is 0 to 255 as in QSS.
    """
    if css.startswith("rgba"):
        r, g, b, a = (int(v) for v in re.findall(r"\d+", css))
        return QtGui.QColor(r, g, b, a)
    return QtGui.QColor(css)


@dataclasses.dataclass(frozen=True)
class ViewColors:
    """Overlay colours of the image views, each a token colour string.

    Border and fill pairs share a hue; fills are translucent so the page shows
    through. The page image is usually pale paper, so every colour must hold
    against both white and the viewport.

    Attributes:
        split_line (str): Page split line.
        split_line_active (str): Split line and handles while hovered or dragged.
        split_handle (str): Split line handle fill.
        split_page_left (str): Tint of the left (or only) page.
        split_page_right (str): Tint of the right page.
        split_offcut (str): Tint of the discarded part of a cut page.
        handle_ring (str): Outline that separates a handle from the page.
        content_border (str): Content box outline.
        content_fill (str): Content box tint.
        page_border (str): Page box outline.
        page_fill (str): Page box tint.
        box_handle (str): Corner handles of the content and page boxes.
        margin_inner_border (str): Margins inner (content) rectangle outline.
        margin_inner_fill (str): Margins inner rectangle tint.
        margin_middle_border (str): Margins page rectangle outline.
        margin_middle_fill (str): Margins page rectangle tint.
        margin_outer_border (str): Margins outer (aligned) rectangle outline.
        margin_outer_fill (str): Margins outer rectangle tint.
        guide (str): Alignment guide.
        guide_active (str): Alignment guide under the cursor.
        page_grid (str): Millimetre grid over the page in Margins and Output.
        zone_picture_border (str): Picture zone outline.
        zone_picture_fill (str): Picture zone tint.
        zone_fill_border (str): Fill zone outline.
        zone_fill_fill (str): Fill zone tint when it has no colour of its own.
        zone_selected (str): Outline of the selected zone.
        zone_vertex (str): Zone vertex marker.
        dewarp_grid (str): Dewarping grid.
        dewarp_handle (str): Dewarping control points.
        deskew_grid (str): Deskew grid lines.
        deskew_axis (str): Deskew grid axes through the rotation centre.
        indicator_scrim (str): Disc behind the orientation indicator.
        indicator_glyph (str): Orientation indicator arrow.
        indicator_label (str): Orientation indicator angle text.
    """

    split_line: str
    split_line_active: str
    split_handle: str
    split_page_left: str
    split_page_right: str
    split_offcut: str
    handle_ring: str
    content_border: str
    content_fill: str
    page_border: str
    page_fill: str
    box_handle: str
    margin_inner_border: str
    margin_inner_fill: str
    margin_middle_border: str
    margin_middle_fill: str
    margin_outer_border: str
    margin_outer_fill: str
    guide: str
    guide_active: str
    page_grid: str
    zone_picture_border: str
    zone_picture_fill: str
    zone_fill_border: str
    zone_fill_fill: str
    zone_selected: str
    zone_vertex: str
    dewarp_grid: str
    dewarp_handle: str
    deskew_grid: str
    deskew_axis: str
    indicator_scrim: str
    indicator_glyph: str
    indicator_label: str


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
        text_dim (str): Secondary text.
        text_disabled (str): Text of disabled controls, dimmer than text_dim.
        field_disabled (str): Fill of disabled inputs and buttons.
        accent (str): Selection and primary button colour.
        accent_hover (str): Accent under the cursor.
        accent_tint (str): Translucent accent for list selection.
        accent_text (str): Text drawn on the accent colour.
        positive (str): Semantic green for additive glyph accents.
        danger (str): Semantic red for destructive glyphs.
        viewport (str): Neutral colour behind page images.
        scroll_handle (str): Scroll bar handle.
        base (str): Opaque fill of item views and text entry, the palette Base role.
        view (ViewColors): Overlay colours of the image views.
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
    text_disabled: str
    field_disabled: str
    accent: str
    accent_hover: str
    accent_tint: str
    accent_text: str
    positive: str
    danger: str
    viewport: str
    scroll_handle: str
    base: str
    view: ViewColors


LIGHT = Tokens(
    dark=False,
    bg_top="#f6f3ec",
    bg_bottom="#ece7dc",
    panel="rgba(253, 251, 246, 210)",
    panel_border="rgba(60, 45, 20, 34)",
    panel_highlight="rgba(255, 254, 250, 230)",
    link="#0b3fa8",
    field="rgba(255, 254, 250, 200)",
    field_hover="rgba(255, 254, 250, 240)",
    field_border="rgba(70, 55, 32, 72)",
    indicator_border="rgba(70, 55, 32, 130)",
    text="#2a2620",
    text_dim="#655d51",
    text_disabled="#aca598",
    field_disabled="rgba(140, 125, 100, 28)",
    accent="#2f6fed",
    accent_hover="#4a82f2",
    accent_tint="rgba(47, 111, 237, 28)",
    accent_text="#ffffff",
    positive="#008a1a",
    danger="#d92d20",
    viewport="#cfcac0",
    scroll_handle="rgba(42, 38, 32, 90)",
    base="#fffefa",
    view=ViewColors(
        split_line="rgba(0, 70, 200, 220)",
        split_line_active="#0a84ff",
        split_handle="rgba(0, 70, 200, 200)",
        split_page_left="rgba(0, 150, 60, 55)",
        split_page_right="rgba(0, 70, 200, 55)",
        split_offcut="rgba(200, 30, 30, 85)",
        handle_ring="#ffffff",
        content_border="rgba(0, 70, 200, 210)",
        content_fill="rgba(0, 70, 200, 32)",
        page_border="rgba(205, 30, 30, 210)",
        page_fill="rgba(205, 30, 30, 22)",
        box_handle="rgba(0, 70, 200, 230)",
        margin_inner_border="rgba(0, 70, 200, 210)",
        margin_inner_fill="rgba(0, 70, 200, 24)",
        margin_middle_border="rgba(0, 140, 40, 210)",
        margin_middle_fill="rgba(0, 140, 40, 24)",
        margin_outer_border="rgba(160, 0, 160, 170)",
        margin_outer_fill="rgba(160, 0, 160, 14)",
        guide="rgba(220, 110, 0, 200)",
        guide_active="#f08000",
        page_grid="rgba(0, 110, 130, 80)",
        zone_picture_border="rgba(0, 80, 220, 220)",
        zone_picture_fill="rgba(0, 80, 220, 55)",
        zone_fill_border="rgba(0, 135, 50, 220)",
        zone_fill_fill="rgba(0, 135, 50, 55)",
        zone_selected="#e07000",
        zone_vertex="#0050dc",
        dewarp_grid="#0046c8",
        dewarp_handle="#d32020",
        deskew_grid="rgba(0, 0, 209, 90)",
        deskew_axis="#0000d1",
        indicator_scrim="rgba(0, 0, 0, 140)",
        indicator_glyph="#ffffff",
        indicator_label="rgba(42, 38, 32, 220)",
    ),
)

DARK = Tokens(
    dark=True,
    bg_top="#1b2030",
    bg_bottom="#11141d",
    panel="rgba(255, 255, 255, 18)",
    panel_border="rgba(255, 255, 255, 40)",
    panel_highlight="rgba(255, 255, 255, 46)",
    link="#8ab8ff",
    field="rgba(255, 255, 255, 22)",
    field_hover="rgba(255, 255, 255, 38)",
    field_border="rgba(255, 255, 255, 60)",
    indicator_border="rgba(255, 255, 255, 120)",
    text="#e8ecf4",
    text_dim="#9aa3b5",
    text_disabled="#596173",
    field_disabled="rgba(255, 255, 255, 8)",
    accent="#5b9bff",
    accent_hover="#78adff",
    accent_tint="rgba(91, 155, 255, 56)",
    accent_text="#0b1220",
    positive="#5ccf72",
    danger="#ff7b72",
    viewport="#2b2d31",
    scroll_handle="rgba(232, 236, 244, 90)",
    base="#232838",
    view=ViewColors(
        split_line="rgba(70, 140, 250, 230)",
        split_line_active="#4a9bff",
        split_handle="rgba(70, 140, 250, 210)",
        split_page_left="rgba(40, 170, 75, 60)",
        split_page_right="rgba(70, 140, 250, 60)",
        split_offcut="rgba(255, 90, 90, 90)",
        handle_ring="#e8ecf4",
        content_border="rgba(70, 140, 250, 230)",
        content_fill="rgba(70, 140, 250, 40)",
        page_border="rgba(240, 80, 75, 230)",
        page_fill="rgba(240, 80, 75, 30)",
        box_handle="rgba(70, 140, 250, 240)",
        margin_inner_border="rgba(70, 140, 250, 230)",
        margin_inner_fill="rgba(70, 140, 250, 32)",
        margin_middle_border="rgba(40, 170, 75, 230)",
        margin_middle_fill="rgba(40, 170, 75, 32)",
        margin_outer_border="rgba(190, 80, 190, 190)",
        margin_outer_fill="rgba(190, 80, 190, 20)",
        guide="rgba(240, 140, 30, 210)",
        guide_active="#ff9a1f",
        page_grid="rgba(60, 170, 190, 110)",
        zone_picture_border="rgba(70, 140, 250, 230)",
        zone_picture_fill="rgba(70, 140, 250, 60)",
        zone_fill_border="rgba(40, 170, 75, 230)",
        zone_fill_fill="rgba(40, 170, 75, 60)",
        zone_selected="#ffa940",
        zone_vertex="#4a90ff",
        dewarp_grid="#4a90ff",
        dewarp_handle="#ef4b45",
        deskew_grid="rgba(120, 170, 255, 110)",
        deskew_axis="#4a90ff",
        indicator_scrim="rgba(0, 0, 0, 160)",
        indicator_glyph="#ffffff",
        indicator_label="rgba(232, 236, 244, 230)",
    ),
)


_active = LIGHT


def set_active_tokens(tokens: Tokens) -> None:
    """Record the token set of the look currently drawn.

    Args:
        tokens (Tokens): Tokens of the applied scheme.
    """
    global _active
    _active = tokens


@functools.cache
def _parse(colors: ViewColors) -> dict[str, QtGui.QColor]:
    return {
        field.name: qcolor(getattr(colors, field.name))
        for field in dataclasses.fields(colors)
    }


def view_color(name: str) -> QtGui.QColor:
    """Return an overlay colour of the active look.

    Args:
        name (str): A ViewColors field name.

    Returns:
        QtGui.QColor: A fresh copy, safe to modify. Light colours are used until
            a theme has been applied.
    """
    return QtGui.QColor(_parse(_active.view)[name])
