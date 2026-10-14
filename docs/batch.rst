Batch processing
################

.. screenshot:: batch_progress
   :alt: The Batch processing window after four pages were written

The play button runs every page of the project through all six stages, from Fix
Orientation to Output. It appears on the row of the selected stage in the stage
list whenever a project is open and no batch is running. The selected row decides
only where the button is. A batch always runs every stage.

.. screenshot:: stage_play_button
   :alt: The stage list with the play button on the selected Select Content row

The batch processes the pages in the order the thumbnail list shows them, with
any sort order. It includes every page, not only the selected ones.

Progress window
---------------

A batch has two phases, and the Batch processing window shows both:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Phase
     - What happens
   * - Detecting ``n/N``
     - The batch measures the split, deskew, content box and margins of each page.
       The results go into the project's settings, so the parameters of every
       page are known before any page is written.
   * - Rendering ``n/N``
     - The batch renders each page and writes it as a TIFF file in the output
       folder.

The window also has these parts:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Part
     - What it does
   * - Progress bar
     - Counts the pages that have finished the current phase.
   * - Result list
     - One line per page, with the file written or the error that stopped it. A
       failed page does not stop the rest of the batch.
   * - Stop
     - Starts no new pages. Pages already running finish, and the status reads
       **Stopping...**. When the batch ends, the button reads **Close**. Closing the
       window while the batch runs requests the same stop.
   * - Summary
     - Shown at the end: pages written, pages failed and, after a stop, pages not
       started.

How many pages run at once
--------------------------

Worker processes handle the pages in parallel, one process per CPU core. On
Windows the count is capped at 61, the most worker processes one pool can hold.
The application settings can store a worker count, but this build has no control
to set it, so the default applies.

What is locked while a batch runs
---------------------------------

The batch owns the settings until it ends, so the program locks the parts that
would change them:

- the stage list and the stage options;
- the thumbnail right-click menu;
- page navigation, including Go To Page;
- inserting, removing and relinking pages, Fix DPI, reversing the spread order
  and opening another project.

Output files
------------

Each page is written as a TIFF file named after its source file. The batch adds
these parts to the name, in this order:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Case
     - Added to the name
   * - Two source files in different folders share a name
     - A label in brackets, such as ``(1)`` or ``(2)``.
   * - Page of a multi-page file
     - ``_page`` and the page number in four digits, such as ``_page0003``.
   * - Half of a split scan, left to right project
     - ``_1L`` for the left half and ``_2R`` for the right half.
   * - Half of a split scan, right to left project
     - ``_1R`` for the right half and ``_2L`` for the left half.

See :mod:`scantailor.core.batch`.

The TIFF compression comes from **Tools > Settings ... > Saving**, which has one
setting for black and white pages and one for grey and colour pages. See
:doc:`settings`.

Batch and PDF export
--------------------

**File > Export PDF ...** starts a batch for any page that has no output yet.
See :doc:`export_pdf`.
