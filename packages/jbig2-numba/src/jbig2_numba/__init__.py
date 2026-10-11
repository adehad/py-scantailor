"""Lossless JBIG2 encoder for bilevel images.

Pixel polarity follows JBIG2: ``True`` or nonzero input pixels are black and
are coded as 1. Only generic region coding is used, so output is always
lossless.
"""

from jbig2_numba.generic_region import encode_generic
from jbig2_numba.segments import encode_file, encode_pdf_stream

__all__ = ["encode_file", "encode_generic", "encode_pdf_stream"]
__version__ = "0.1.0"
