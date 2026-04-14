"""Parameters for the output filter.

This module defines the Params class which stores all output settings
for a specific page.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from scantailor.core import Dpi

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .dewarping_options import DewarpingOptions


class Params(BaseModel):
    """Parameters for output filter.

    Attributes:
        output_dpi: Resolution for the output image.
        color_mode: Color mode (B&W, grayscale, or mixed).
        binarization: Binarization options for B&W conversion.
        despeckle_level: Noise removal level.
        black_on_white: True for black text on white background.
        dewarping: Dewarping options for curved page correction.
    """

    model_config = ConfigDict(frozen=True)

    output_dpi: Dpi = Field(default_factory=lambda: Dpi.uniform(600))
    color_mode: ColorMode = ColorMode.BLACK_AND_WHITE
    binarization: BinarizationOptions = Field(default_factory=BinarizationOptions)
    despeckle_level: DespeckleLevel = DespeckleLevel.NORMAL
    black_on_white: bool = True
    dewarping: DewarpingOptions = Field(default_factory=DewarpingOptions)

    def with_output_dpi(self, dpi: Dpi) -> Self:
        """Create a copy with different output DPI."""
        return self.model_copy(update={"output_dpi": dpi})

    def with_color_mode(self, mode: ColorMode) -> Self:
        """Create a copy with different color mode."""
        return self.model_copy(update={"color_mode": mode})

    def with_binarization(self, options: BinarizationOptions) -> Self:
        """Create a copy with different binarization options."""
        return self.model_copy(update={"binarization": options})

    def with_binarization_method(self, method: BinarizationMethod) -> Self:
        """Create a copy with different binarization method."""
        new_binarization = self.binarization.with_method(method)
        return self.model_copy(update={"binarization": new_binarization})

    def with_despeckle_level(self, level: DespeckleLevel) -> Self:
        """Create a copy with different despeckle level."""
        return self.model_copy(update={"despeckle_level": level})

    def with_black_on_white(self, black_on_white: bool) -> Self:
        """Create a copy with different black_on_white setting."""
        return self.model_copy(update={"black_on_white": black_on_white})

    def with_dewarping(self, options: DewarpingOptions) -> Self:
        """Create a copy with different dewarping options."""
        return self.model_copy(update={"dewarping": options})

    def needs_binarization(self) -> bool:
        """Return True if the color mode requires binarization."""
        return self.color_mode in (ColorMode.BLACK_AND_WHITE, ColorMode.MIXED)

    def needs_dewarping(self) -> bool:
        """Return True if dewarping is enabled."""
        return self.dewarping.is_enabled()
