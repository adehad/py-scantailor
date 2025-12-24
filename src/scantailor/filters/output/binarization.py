"""Binarization options for the output filter.

This module defines binarization methods and their parameters.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class BinarizationMethod(Enum):
    """Binarization algorithm to use.

    OTSU: Otsu's method - automatic global threshold.
    SAUVOLA: Sauvola's method - local adaptive thresholding.
    WOLF: Wolf's method - improved local adaptive thresholding.
    BRADLEY: Bradley's method - simple local adaptive thresholding.
    EDGEDIV: Edge division method - edge enhancement with illumination normalization.
    """

    OTSU = "otsu"
    SAUVOLA = "sauvola"
    WOLF = "wolf"
    BRADLEY = "bradley"
    EDGEDIV = "edgediv"

    @classmethod
    def from_string(cls, value: str) -> "BinarizationMethod":
        """Parse a BinarizationMethod from its string value."""
        for method in cls:
            if method.value == value:
                return method
        return cls.OTSU

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


class BinarizationOptions(BaseModel):
    """Options for image binarization (black and white conversion).

    These settings control how color/grayscale images are converted
    to pure black and white.

    Attributes:
        method: The binarization algorithm to use.
        threshold_adjustment: Global threshold adjustment (-100 to 100).
            Positive values make image darker (more black).
        window_size: Window size for local methods (Sauvola, Wolf, Bradley, EdgeDiv).
        sauvola_coef: Sauvola's k coefficient (0.0-1.0).
        wolf_coef: Wolf's k coefficient (0.0-1.0).
        wolf_lower_bound: Wolf's lower bound (0-255).
        wolf_upper_bound: Wolf's upper bound (0-255).
        bradley_coef: Bradley's k coefficient (0.0-1.0). Higher values make
            it easier for pixels to be classified as foreground (black).
        edge_div_kep: EdgeDiv edge enhancement weight (0.0-1.0).
            Higher = more edge enhancement.
        edge_div_kbd: EdgeDiv blur division weight (0.0-1.0).
            Higher = more illumination normalization.
        normalize_illumination: Normalize illumination before binarization.
        morphological_smoothing: Apply morphological smoothing after binarization.
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
    morphological_smoothing: bool = True

    def with_method(self, method: BinarizationMethod) -> "BinarizationOptions":
        """Create a copy with a different binarization method."""
        return self.model_copy(update={"method": method})

    def with_threshold_adjustment(self, adjustment: int) -> "BinarizationOptions":
        """Create a copy with a different threshold adjustment."""
        return self.model_copy(update={"threshold_adjustment": adjustment})

    def with_window_size(self, size: int) -> "BinarizationOptions":
        """Create a copy with a different window size."""
        return self.model_copy(update={"window_size": size})
