Opening and exporting PDF files
###############################

The program can make a project from the pages of a PDF and write the processed
pages out as one PDF.

Open PDF
--------

**File > Open PDF ...** starts a new project from a PDF:

#. The program offers to save the open project, then closes it.
#. You choose a PDF.
#. The **Extract pages** prompt asks which pages to render. See
   :ref:`pdf-in-new-project`.
#. The New Project dialog opens with the rendered pages in Files In Project.
#. Press OK to create the project.

The program saves each rendered page as a PNG file at the chosen resolution, in a
folder next to the PDF. The folder has the name of the PDF with ``_pages`` added.
The program then processes the rendered pages like scanned images. The project
keeps a record of the source PDF, so Export PDF can carry its text over.

.. _pdf-in-new-project:

PDFs in the New Project dialog
------------------------------

.. screenshot:: open_pdf
   :alt: The New Project dialog with a sample PDF selected and its Extract pages prompt

A project can hold the pages of a PDF next to scanned images. The New Project
dialog lists the PDFs of the input folder before the images, each with its page
count. **Add PDF...** adds a PDF from any other folder. There are three ways to
bring a PDF in:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Way in
     - Input folder
     - Output folder
   * - **File > Open PDF ...**
     - The folder of the PDF.
     - An ``out`` folder inside the ``_pages`` folder.
   * - A PDF in the input folder
     - The Input Directory you choose.
     - The Output Directory you choose, or ``out`` inside the input folder.
   * - **Add PDF...**
     - Unchanged. The PDF can be in any folder.
     - The Output Directory you choose, or ``out`` inside the input folder.

Select a PDF and press the right arrow button. The **Extract pages** prompt asks
what to render:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - All pages
     - Renders every page of the PDF.
   * - Pages ... to ...
     - Renders only the pages in this range.
   * - Render at
     - The resolution, from 72 to 1200 DPI. The prompt offers the last value used
       for a new project, or 300 the first time. Higher values keep more detail
       and use more memory.
   * - Write to
     - The folder the pages go to: the PDF's name with ``_pages`` added, next to
       the PDF. You cannot change it.
   * - Extract
     - Renders the pages and adds them to Files In Project.

If that folder already holds rendered pages, the program asks whether to use them
or to render again. Rendering again replaces the old pages only once every new
page is done. A PDF whose pages were extracted shows ``extracted`` after its page
count.

Export PDF
----------

.. screenshot:: export_pdf
   :alt: The question asked before export when some pages have no output

**File > Export PDF ...** writes all processed pages to one PDF. Each page gets
its physical size: pixels divided by the resolution, times 72 points.

If some pages have no output yet, the program asks whether to process them now.
If you answer Yes, a batch runs on those pages only. The program does not write
the PDF in that case, so choose Export PDF again once the batch finishes.

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
     - Copies the text of the source PDF onto the matching output pages, so you
       can search and select text in the exported PDF. Available only when the
       project holds pages rendered from a PDF and that PDF is still on disk.
       It is then on by default.
   * - Output file
     - Where the PDF is written. If the project has pages from exactly one PDF,
       the default is ``<name>_processed.pdf`` next to that PDF. Otherwise it is
       ``output.pdf`` in the output folder. **Browse ...** picks another place.
       The dialog adds a ``.pdf`` extension if it is missing.

The text is carried only for pages that a batch has processed, either in this
session or in an earlier one whose result was saved with the project. Other pages
are exported without text.

While the file is written, a progress window shows the work and has a Cancel
button. When the export finishes, the status bar reports the number of pages
exported and how many of them carry text.

See :mod:`scantailor.core.pdf_import`, :mod:`scantailor.core.pdf_export` and
:mod:`scantailor.core.text_layer`.
