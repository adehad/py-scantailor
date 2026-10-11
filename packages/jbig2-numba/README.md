# jbig2-numba

Lossless JBIG2 (ITU-T T.88) encoder for bilevel images, written in Python
with the hot loops compiled by numba. No native extension, so it installs
anywhere numba does.

## What it produces

A single immediate lossless generic region (segment type 39) coded with the
MQ arithmetic coder, GBTEMPLATE 0 with the nominal adaptive-template pixels
and, by default, typical prediction (TPGDON).

It never uses symbol dictionaries or text regions. Those modes get their
compression by replacing glyphs with a shared look-alike, which can silently
turn one character into another (the classic 6/8 substitution on scanned
documents). Generic region coding stores every pixel exactly.

## API

Input is a 2-D `bool` or `uint8` array. `True` or any nonzero value is black,
which matches JBIG2 where 1 is black. Inputs are never modified.

```python
from jbig2_numba import encode_file, encode_generic, encode_pdf_stream

stream = encode_pdf_stream(image, dpi=(300, 300))  # body of a PDF image XObject
jb2 = encode_file(image, dpi=(300, 300))           # standalone .jb2 file
raw = encode_generic(image)                        # generic region data only
```

`encode_pdf_stream` follows ISO 32000-1 7.4.7: page information plus the
region segment, with no file header, end-of-page or end-of-file segment and
no `/JBIG2Globals`. Use it in an image dictionary such as:

```text
<< /Type /XObject /Subtype /Image /Width w /Height h
   /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode /Length n >>
```

`tpgdon=False` disables typical prediction on any of the three functions.

## How it is verified

The tests embed each stream in a hand-built PDF and decode the image XObject
with two independent decoders, MuPDF (via PyMuPDF) and poppler
(`pdfimages -png`), requiring both to return the input pixel for pixel. The
cases cover 1x1, single rows and columns, widths that are not multiples of 8
or 32, solid pages, checkerboards, noise at several densities, long blank
runs, a 2100x3000 text page, a page with scan defects, and 400 fuzzed small
images. Standalone `.jb2` output is decoded with MuPDF. A pure-Python T.88
decoder in `tests/` checks the MQ coder and context model on their own.

The suite also reports output size against CCITT G4 (Pillow/libtiff) and
encode time.

Running the tests needs `pytest`, `pymupdf`, `pillow` and poppler's
`pdfimages` on `PATH`:

```sh
python -m pytest tests
```

## Licence

GPL-3.0-or-later. Implemented from ITU-T T.88 and ISO 32000-1.
