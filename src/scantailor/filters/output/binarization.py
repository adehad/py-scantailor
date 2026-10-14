"""Binarization options for the output filter.

This module defines binarization methods and their parameters.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class BinarizationMethod(Enum):
    """Binarization algorithm to use.

    - ``OTSU``: Otsu's method, an automatic global threshold.
    - ``SAUVOLA``: Sauvola's local adaptive threshold.
    - ``WOLF``: Wolf's local adaptive threshold, an improved Sauvola.
    - ``FOX``: Wolf with Singh's local contrast.
    - ``WINDOW``: Dynamic window method of Bataineh et al.
    - ``BRADLEY``: Bradley's simple local adaptive threshold.
    - ``GRAD``: Gradient snip, a local mean blended with a page level taken where
      the contrast is.
    - ``EDGEPLUS``: Edge division with the edge enhancement term only.
    - ``BLURDIV``: Edge division with the blur division term only.
    - ``EDGEDIV``: Edge division with edge enhancement and illumination normalization.
    """

    OTSU = "otsu"
    SAUVOLA = "sauvola"
    WOLF = "wolf"
    FOX = "fox"
    WINDOW = "window"
    BRADLEY = "bradley"
    GRAD = "grad"
    EDGEPLUS = "edgeplus"
    BLURDIV = "blurdiv"
    EDGEDIV = "edgediv"

    @classmethod
    def from_string(cls, value: str) -> BinarizationMethod:
        """Parse a BinarizationMethod from its string value."""
        for method in cls:
            if method.value == value:
                return method
        return cls.OTSU

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


class ColorSegmenterOptions(BaseModel):
    """Options for colouring the binarised foreground from the original page.

    Attributes:
        enabled: Segment the foreground into a few colours instead of black.
        noise_reduction: Strength of noise removal, higher removes more.
        red_threshold_adjustment: Shift of the red channel threshold.
        green_threshold_adjustment: Shift of the green channel threshold.
        blue_threshold_adjustment: Shift of the blue channel threshold.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    noise_reduction: int = Field(default=7, ge=0)
    red_threshold_adjustment: int = 0
    green_threshold_adjustment: int = 0
    blue_threshold_adjustment: int = 0


class BinarizationOptions(BaseModel):
    """Options for image binarization (black and white conversion).

    These settings control how color/grayscale images are converted
    to pure black and white.

    Attributes:
        method: The binarization algorithm to use.
        threshold_adjustment: Global threshold adjustment (-100 to 100).
            Positive values make image darker (more black).
        window_size: Window size for local methods (every method but Otsu).
        sauvola_coef: Sauvola's k coefficient (0.0-1.0).
        wolf_coef: Wolf's k coefficient (0.0-1.0); Fox, Window and Grad read it too.
        wolf_lower_bound: Wolf's lower bound (0-255); Fox, Window and Grad read it
            too.
        wolf_upper_bound: Wolf's upper bound (0-255); Fox, Window and Grad read it
            too.
        bradley_coef: Bradley's k coefficient (0.0-1.0). Higher values make
            it easier for pixels to be classified as foreground (black).
        edge_div_kep: Edge enhancement weight (0.0-1.0), read by EdgePlus and
            EdgeDiv. Higher = more edge enhancement.
        edge_div_kbd: Blur division weight (0.0-1.0), read by BlurDiv and
            EdgeDiv. Higher = more illumination normalization.
        normalize_illumination: Normalize illumination before binarization.
        savitzky_golay_smoothing: Smooth the gray image before binarization.
        morphological_smoothing: Apply morphological smoothing after binarization.
        color_segmenter: Options for colouring the foreground.

    C++ reads the one Sauvola coefficient for Bradley, EdgePlus, BlurDiv and
    EdgeDiv. The three fields above stay separate so that each method keeps
    its own control in the options widget, and the C++ project import copies
    the Sauvola coefficient into the fields a method reads, so an imported
    project binarises as it did in C++.
    """

    model_config = ConfigDict(frozen=True)

    method: BinarizationMethod = BinarizationMethod.OTSU
    threshold_adjustment: int = Field(default=0, ge=-100, le=100)
    window_size: int = Field(default=200, ge=10, le=1000)
    sauvola_coef: float = Field(default=0.34, ge=0.0, le=1.0)
    wolf_coef: float = Field(default=0.3, ge=0.0, le=1.0)
    wolf_lower_bound: int = Field(default=1, ge=0, le=255)
    wolf_upper_bound: int = Field(default=254, ge=0, le=255)
    bradley_coef: float = Field(default=0.34, ge=0.0, le=1.0)
    edge_div_kep: float = Field(default=0.5, ge=0.0, le=1.0)
    edge_div_kbd: float = Field(default=0.5, ge=0.0, le=1.0)
    normalize_illumination: bool = True
    savitzky_golay_smoothing: bool = True
    morphological_smoothing: bool = True
    color_segmenter: ColorSegmenterOptions = Field(
        default_factory=ColorSegmenterOptions
    )

    def with_method(self, method: BinarizationMethod) -> BinarizationOptions:
        """Create a copy with a different binarization method."""
        return self.model_copy(update={"method": method})

    def with_threshold_adjustment(self, adjustment: int) -> BinarizationOptions:
        """Create a copy with a different threshold adjustment."""
        return self.model_copy(update={"threshold_adjustment": adjustment})

    def with_window_size(self, size: int) -> BinarizationOptions:
        """Create a copy with a different window size."""
        return self.model_copy(update={"window_size": size})
