"""Dewarping module for correcting page curvature.

This module provides tools for correcting perspective distortion from
curved page surfaces (like book spines) using a cylindrical surface model.

The dewarping process involves:
1. Defining the page edges as curves (top and bottom)
2. Creating a distortion model from these curves
3. Using the cylindrical surface dewarper to map between
   warped (curved) and dewarped (flat) coordinates
"""

# Automatic dewarping refuses a model whose top and bottom curves lie closer
# together than this share of the content height, in percent. Chosen by
# measuring a 197-page book; see docs/stages/output.rst.
DEFAULT_MIN_SPAN_PERCENT = 40
