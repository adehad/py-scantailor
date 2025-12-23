"""Output filter module.

This module provides the final stage of the ScanTailor processing pipeline,
generating output images with binarization, despeckling, and other processing.
"""

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .filter import Filter
from .generator import OutputResult, generate_output
from .params import Params
from .settings import Settings

__all__ = [
    "BinarizationMethod",
    "BinarizationOptions",
    "ColorMode",
    "DespeckleLevel",
    "Filter",
    "OutputResult",
    "Params",
    "Settings",
    "generate_output",
]
