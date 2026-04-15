"""Deskew filter.

This filter detects and corrects page skew (rotation). It can automatically
detect skew angle using the find_skew algorithm, or use a manually specified
angle.

The deskew angle is stored as metadata and applied during rendering/output.
"""
