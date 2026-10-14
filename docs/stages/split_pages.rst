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
     - A spread: two facing pages photographed or scanned together.

How the layout is detected
==========================

**Auto** looks for a folding line or dark spine first, and otherwise for the
white gap between two pages of text. A spread that is tilted gets a tilted split
line. The estimator that makes these decisions is
:mod:`scantailor.filters.page_split.page_layout_estimator`.

With **Single page with an offcut**, the page is cut where its text starts, or at the
image edge when no edge can be told apart.

Under **Auto**, a narrow strip of a neighbouring page's text along the edge of a
single-page scan is cut off when all of these hold:

* the strip is no wider than a quarter of the image;
* the white gap between the strip and the page's text is at least 20 pixels at 150 dpi,
  the resolution the estimator works at;
* that gap is at least twice the widest word gap inside the page's text.

Text of the page itself that runs off the scan is not a strip, so it is left alone.
A strip that fails any of the three tests stays in the page.

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

A split line has a round handle at each end. Drag a handle to tilt the line, for a
spread whose spine is not upright; drag the line itself to move it without changing
its tilt. A line cannot cross its neighbour or leave the image. The tint on each
side of the line shows where each page ends, so a tilted line cuts the page along
the tilt in the output as well.

Use a handle when the spine is not vertical, and drag the line when the spine is
straight but off centre. Auto detection places the line in the white gap between the
text, so a line that cuts through text needs moving.

Removed halves
==============

.. screenshot:: stage2_split_pages_removed_half
   :alt: A split page whose left half was removed, tinted red with a trash button

When one half of a split image was removed from the project, the view shows that
half tinted red with a button on it. Click the button to bring the half back; the
status bar tells what it does while the pointer is over it. The restored page comes
back with the settings of its image and the thumbnail list shows both halves again.

Layouts and project files
==========================

ScanTailor keeps the layout of each page together with the image size and
rotation it was worked out from. When a page is processed again, a layout whose
image size or rotation no longer matches is detected again (**Auto**) or fitted
to the new image (a layout type you chose). A layout saved without that record,
as in projects from before it existed, is treated the same way, exactly as in the
original ScanTailor. A layout you place by hand is recorded when you place it, so
processing keeps it.

Applying to other pages
=======================

In the **Change ...** dialog, **Every other selected page** starts from the
current page when it is among the selected pages, as the hint beside it says. The
original ScanTailor starts from the first selected page instead; this version
follows the hint.
