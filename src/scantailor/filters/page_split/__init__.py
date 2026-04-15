"""Page Split filter module.

This module provides functionality for splitting scanned images into
individual pages. It handles:
- Single page images (no split needed)
- Two-page spreads (like open books)
- Pages with offcut garbage to remove

The filter can automatically detect the appropriate split point or
accept manual user input.
"""
