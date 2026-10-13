"""Alignment guides of the Page Layout stage."""

from pydantic import BaseModel, ConfigDict


class Guide(BaseModel):
    """A horizontal or vertical alignment guide, shared by every page.

    Attributes:
        position_mm: Distance in millimetres from the centre of a page's outer
            rectangle, so a guide means the same on pages of every size; y for
            a horizontal guide and x for a vertical one.
        is_horizontal: True for a horizontal guide.
    """

    model_config = ConfigDict(frozen=True)

    position_mm: float
    is_horizontal: bool
