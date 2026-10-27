Batch processing
################

.. screenshot:: batch_progress
   :alt: The Batch processing window after four pages were written

The play button runs every page of the project from Fix Orientation up to the
stage whose row it sits on. It appears on the row of the selected stage in the
stage list whenever a project is open and no batch is running.

Only a batch from Output writes files. A batch from an earlier stage measures
each page up to that stage and stores the results in the project, so you can
batch to Select Content or Margins, check the thumbnails, fix the pages that came
out wrong, then batch from Output to render everything.

.. screenshot:: stage_play_button
   :alt: The stage list with the play button on the selected Select Content row

The batch processes the pages in the order the thumbnail list shows them, with
any sort order. It includes every page, not only the selected ones.

A batch past Split Pages first splits each scan it has not split yet, and each
scan whose size, rotation or requested layout has changed since. The later
stages start after that.
A scan found to hold two pages then goes on as its left and right page, each
written to its own file, and the page list shows both when the batch ends.
``scantailor-cli run`` does the same and saves both pages in the project. The
batch deletes the file it wrote earlier for the whole scan, so the output folder
holds each page once.

Progress window
---------------

A batch from Output has two phases, and the Batch processing window shows both.
A batch from an earlier stage has only the first one:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Phase
     - What happens
   * - Detecting ``n/N``
     - The batch measures each page up to the selected stage, at most the split,
       deskew, content box and margins. The results go into the project's
       settings, so the parameters of every page are known before any page is
       written. ``N`` changes when Split Pages finds a different number of
       pages in a scan.
   * - Rendering ``n/N``
     - Output only. The batch renders each page and writes it as a TIFF file in
       the output folder.

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
   * - System load slider
     - How many pages run at once. See `How many pages run at once`_.
   * - Beep when finished
     - Beeps when the batch completes. Off by default; the program keeps the
       choice for the next batch until it closes.
   * - Stop
     - Starts no new pages. Pages already running finish, and the status reads
       **Stopping...**. When the batch ends, the button reads **Close**. Closing the
       window while the batch runs requests the same stop.
   * - Summary
     - Shown at the end: pages written, or pages processed for a batch that stops
       before Output, then pages failed and, after a stop, pages not started.
       Pages whose files were kept, as described under
       `Pages that have not changed`_, are counted apart as kept.

How many pages run at once
--------------------------

Worker processes handle the pages in parallel. The system load slider in the
Batch processing window sets how many pages run at once, from one up to the
number of CPU cores, with **-** and **+** for one step. Dragging it shows the
count as ``n/N``. Set it lower to keep the computer responsive while a batch
runs.

The slider works while the batch runs: a lower count takes effect as running
pages finish, and a higher one the next time a page starts. The program saves
the count and uses it for every later batch. At the full core count it saves
nothing, so a later move to a computer with more cores uses them all.

On Windows at most 61 pages run at once, the most worker processes one pool can
hold. A 32-bit build allows at most two.

When a batch completes
----------------------

When every page has run, the program:

- flashes its taskbar entry, if you were working in another program;
- beeps, if **Beep when finished** is checked;
- selects the first page.

A batch you stop does none of these, and the selected page stays where it was.

What is locked while a batch runs
---------------------------------

The batch owns the settings until it ends, so the program locks the parts that
would change them:

- the stage list and the stage options;
- the page view, including zoom and scrolling;
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

Pages that have not changed
---------------------------

A batch from Output does not render a page again when its output would come out
the same. After writing a page, the batch stores a record of it in the project
file. The next batch from Output keeps the page's existing files, and lists the
page in the result window as usual, when all of these still hold:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Check
     - What renders the page again
   * - Settings
     - Any change to the page at any stage, from Fix Orientation to Output, or
       to a stage-wide default such as the output resolution. For a page aligned
       with the others in Margins, a change to the shared page size, which
       another page's content box can move.
   * - Resolution
     - A different resolution set for the image with **Fix DPI**.
   * - TIFF compression
     - A different compression in **Tools > Settings ... > Saving**.
   * - Source file
     - The scan was replaced or edited, so its size or modification time differ.
   * - Output files
     - The page's output file, or one of its layer files in the
       ``foreground``, ``background`` or ``original_background`` folders, was
       deleted, replaced or edited. A new output folder or a new file name, for
       instance after changing the reading direction, has the same effect.

A page with no record, such as every page of a project saved by an earlier
version, is rendered. To force every page to render again, delete the output
files.

A page rendered again is sized against the shared page size of every aligned
page in the project, not only the pages rendered with it, so it comes out the
size of the pages that were kept.

The interactive Output view always renders the page in memory and never reads or
writes the output files.

See :mod:`scantailor.core.output_reuse`.

Batch and PDF export
--------------------

**File > Export PDF ...** starts a batch for any page that has no output yet.
See :doc:`export_pdf`.
