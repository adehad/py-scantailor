"""Options shared by the grayscale and colour output paths."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .binarization import MAX_WINDOW_SIZE, MIN_WINDOW_SIZE


class FillingColor(Enum):
    """Colour used for the margins outside the content area.

    BACKGROUND: The dominant background colour of the page.
    WHITE: White (black when the output is white on black).
    BLACK: Black (white when the output is white on black).
    """

    BACKGROUND = "background"
    WHITE = "white"
    BLACK = "black"


class PosterizationOptions(BaseModel):
    """Options for reducing the number of colours in the output.

    Attributes:
        enabled: Posterize the output.
        level: Cells per colour channel, 2 to 255; fewer cells give fewer colours.
        normalization_enabled: Stretch the colour range before posterizing and
            write the stretched colours.
        force_black_and_white: Turn grayish dark colours black and light ones white.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    level: int = Field(default=4, ge=2, le=255)
    normalization_enabled: bool = False
    force_black_and_white: bool = True


class ColorCommonOptions(BaseModel):
    """Options for grayscale and colour output.

    Attributes:
        fill_offcut: Paint everything outside the page outline (the cut-off part
            of a neighbouring page, for instance) with the filling colour.
        fill_outside_page_box: Do not follow the page outline: the margin colour,
            illumination and the kept area come from the whole output rectangle.
            Takes the place of ``fill_offcut`` when both are set.
        fill_margins: Paint everything outside the content area with the filling colour.
        normalize_illumination: Flatten uneven lighting before output.
        filling_color: Colour used when filling margins.
        wiener_coef: Strength of the Wiener colour filter, 0 (off) to 1.
        wiener_window_size: Side of the Wiener filter window in pixels.
        posterization: Colour reduction options.
    """

    model_config = ConfigDict(frozen=True)

    fill_offcut: bool = True
    fill_outside_page_box: bool = False
    fill_margins: bool = True
    normalize_illumination: bool = False
    filling_color: FillingColor = FillingColor.BACKGROUND
    wiener_coef: float = Field(default=0.0, ge=0.0, le=1.0)
    wiener_window_size: int = Field(default=5, ge=MIN_WINDOW_SIZE, le=MAX_WINDOW_SIZE)
    posterization: PosterizationOptions = Field(default_factory=PosterizationOptions)
