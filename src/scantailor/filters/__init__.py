"""Filters for the ScanTailor processing pipeline.

Each filter represents a processing stage:
1. fix_orientation - Apply 0/90/180/270 degree rotation
2. page_split - Split two-page spreads into individual pages
3. deskew - Correct page skew angle
4. select_content - Detect and select page content area
5. page_layout - Set margins and alignment
6. output - Generate final output with binarization and dewarping
"""

__all__: list[str] = []
