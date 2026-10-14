Batch processing
################

.. screenshot:: batch_progress
   :alt: The Batch processing window after four pages were written

The play button runs every page of the project through all six stages, from Fix
Orientation to Output. It appears on the row of the selected stage in the stage
list whenever a project is open and no batch is running. The selected row only
sets where the button sits. Which stage you select does not change what runs.

.. screenshot:: stage_play_button
   :alt: The stage list with the play button on the selected Select Content row

Pages are processed in the order the thumbnail list shows them, whichever sort
order is chosen. Every page is included, not only the selected ones.

Progress window
---------------

A batch has two phases, and the Batch processing window shows both:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Phase
     - What happens
   * - Detecting ``n/N``
     - Each page is measured for its split, deskew, content box and margins. The
       results go into the project's settings, so every page's parameters are
       known before any page is written.
   * - Rendering ``n/N``
     - Each page is rendered and written as a TIFF file in the output folder.

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

Pages are processed in parallel by worker processes, one per CPU core. On
Windows the count is capped at 61, the most worker processes one pool can hold.
The worker count can be stored in the application settings, but this build does
not show the control that sets it, so the default applies.

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

Each page is written as a TIFF file named after its source file, with a label in brackets when two
source files share a name. A page of a multi-page file gets ``_pageNNNN`` after the name. The two halves of a split scan
get a number and a side letter, such as ``_1L`` and ``_2R``. The numbering follows
the reading direction of the project. See :mod:`scantailor.core.batch`.

The compression of the TIFF files comes from **Tools > Settings ... > Saving**:
one setting for black and white pages and one for grey and colour pages. See
:doc:`settings`.

Batch and PDF export
--------------------

**File > Export PDF ...** starts a batch for any page that has no output yet.
See :doc:`export_pdf`.
