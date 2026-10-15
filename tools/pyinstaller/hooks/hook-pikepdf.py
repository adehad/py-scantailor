"""Collect pikepdf for writing PDFs, without its XMP metadata support.

pikepdf's C++ core imports pikepdf.jbig2 to decode JBIG2 streams, which no Python
import names. lxml serves only XMP metadata, which the app never reads or writes.
"""

hiddenimports = ["pikepdf.jbig2"]
excludedimports = ["lxml"]
