Opening and exporting PDF files
##################################

The program can open a PDF as a project and write the processed pages back out as
one PDF.

Open PDF
--------

**File > Open PDF ...** renders the pages of a PDF and opens them as a new
project. The open project is replaced, after the save prompt.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - Render pages at
     - The resolution the pages are rendered at, from 72 to 1200 DPI. The last
       value used is offered, or 300 at first. Higher values give more detail
       and more memory use.

Each page is saved as a PNG file with that resolution, in a folder named after the
PDF with ``_pages`` added, next to the PDF. The project's output goes to an ``out``
folder inside that folder. The rendered pages are then processed like scanned
images. The source PDF is kept with the project, which lets Export PDF carry its
text over.

Export PDF
----------

.. screenshot:: export_pdf
   :alt: The question asked before export when some pages have no output

**File > Export PDF ...** writes all processed pages to one PDF. Each page is
placed at its physical size: pixels divided by the resolution, times 72 points.

If some pages have no output yet, the program asks whether to process them now.
Answer Yes and a batch runs on those pages only. The PDF is not written in that
case, so choose Export PDF again once the batch finishes.

Once every page has output, the Export PDF dialog opens:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - What it does
   * - Black and white pages
     - The codec of black and white pages: JBIG2 or CCITT G4. Both store the
       pages without loss.
   * - JPEG quality
     - Quality of grey and colour pages, from 1 to 100. The default is 85. Lower
       values make smaller, blurrier files.
   * - Carry text layer
     - Copies the text of the source PDF onto the matching output pages, so the
       exported PDF can be searched and selected. Available only when the project
       was opened from a PDF and that PDF is still on disk. On by default then.
   * - Output file
     - Where the PDF is written. The default is next to the source PDF, named
       ``<name>_processed.pdf``, or else ``output.pdf`` in the output folder.
       **Browse ...** picks another place. A ``.pdf`` extension is added if missing.

The text is carried only for pages that a batch has processed, either in this
session or in an earlier one whose result was saved with the project. Other pages
are exported without text.

While the file is written, a progress window shows the work, with a Cancel button.
When it finishes, the status bar reports the number of pages exported, and how
many carry text.

See :mod:`scantailor.core.pdf_import`, :mod:`scantailor.core.pdf_export` and
:mod:`scantailor.core.text_layer`.
