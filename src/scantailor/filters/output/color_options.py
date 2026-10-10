"""Options shared by the grayscale and colour output paths."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class FillingColor(Enum):
    """Colour used for the margins outside the content area.

    BACKGROUND: The dominant background colour of the page.
    WHITE: White (black when the output is white on black).
    BLACK: Black (white when the output is white on black).
    """

    BACKGROUND = "background"
    WHITE = "white"
    BLACK = "black"


class ColorCommonOptions(BaseModel):
    """Options for grayscale and colour output.

    Attributes:
        fill_margins: Paint everything outside the content area with the filling colour.
        normalize_illumination: Flatten uneven lighting before output.
        filling_color: Colour used when filling margins.
    """

    model_config = ConfigDict(frozen=True)

    fill_margins: bool = True
    normalize_illumination: bool = False
    filling_color: FillingColor = FillingColor.BACKGROUND
