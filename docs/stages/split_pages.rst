2. Split Pages
##############

.. screenshot:: stage2_split_pages
   :alt: Split Pages with its Page Layout panel

Tells ScanTailor whether a scan holds one page or two facing pages. The label under
the layout buttons reads **Auto detected** until you change the layout yourself with
**Change ...**.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Layout
     - What it does
     - When to use it
   * - Single page
     - Keeps the whole scan as one page.
     - A scan that holds one page and nothing from its neighbour.
   * - Single page with an offcut
     - Keeps one page and cuts off the strip of a neighbouring page at one or both
       edges.
     - The scanner caught the edge of the next page of a book.
   * - Two pages
     - Splits the scan along one line into a left and a right page.
     - A spread, which is two facing pages scanned or photographed together.

How the layout is detected
==========================

**Auto** looks for a folding line or a dark spine first, and otherwise for the
white gap between two pages of text. A tilted spread gets a tilted split line. The
detection code is in :mod:`scantailor.filters.page_split.page_layout_estimator`.

With **Single page with an offcut**, the page is cut where its text starts, or at
the image edge when no text edge can be found.

Under **Auto**, ScanTailor cuts off a narrow strip of a neighbouring page's text
along the edge of a single-page scan when all of these hold:

* the strip is no wider than a quarter of the image;
* the white gap between the strip and the page's text is at least 20 pixels at 150 dpi,
  the resolution the estimator works at;
* that gap is at least twice the widest word gap inside the page's text.

A strip that fails any of the three tests stays in the page. Text of the page itself
that runs off the scan is not a strip, so ScanTailor leaves it alone.

Choosing a layout by hand
=========================

Use **Auto** first. Change the layout when the detected one is wrong, and check
the result in the view and in the thumbnails before you run the batch.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Change to
   * - Two pages were treated as one page, or the spine runs through the text.
     - Two pages
   * - One page shows a strip of the neighbour at the edge that the output keeps.
     - Single page with an offcut
   * - A one-page scan was split into two.
     - Single page

The split lines
===============

.. screenshot:: stage2_split_pages_tilted
   :alt: Split Pages with a tilted split line and a handle at each end

A split line has a round handle at each end.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Action
     - What it does
     - When to use it
   * - Drag a handle
     - Moves that end of the line, which tilts it.
     - The spine is not upright.
   * - Drag the line
     - Moves the line without changing its tilt.
     - The spine is straight but off centre, or the line cuts through text.

A line cannot cross its neighbour or leave the image. The tint on each side of the
line shows where each page ends. The output cuts the page along the same line, so a
tilted line gives a tilted cut. **Auto** places the line in the white gap between the
text, so a line that cuts through text needs moving.

Removed halves
==============

.. screenshot:: stage2_split_pages_removed_half
   :alt: A split page whose left half was removed, tinted red with a trash button

When one half of a split image was removed from the project, the view shows that
half tinted red with a button on it. Click the button to bring the half back. While
the pointer is over the button, the status bar says what it does. The restored page
comes back with the settings of its image, and the thumbnail list shows both halves
again.

Layouts and project files
=========================

ScanTailor stores the layout of each page with the image size and rotation it was
worked out from. When a page is processed again and its image size or rotation no
longer matches, ScanTailor updates the layout as follows.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Layout
     - After the image size or rotation changes
   * - **Auto**
     - Detected again.
   * - A layout type you chose
     - Fitted to the new image.

A layout saved without the image size and rotation, as in projects from before this
record existed, is treated as changed, as upstream ScanTailor Advanced does. A layout
you place by hand records the image size and rotation when you place it, so
processing keeps it.

Applying to other pages
=======================

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Option in **Change ...**
     - Upstream
     - This port
     - Why
   * - **Every other selected page**
     - Starts from the first selected page.
     - Starts from the current page when it is among the selected pages.
     - The hint beside the option describes this behaviour.
