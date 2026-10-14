1. Fix Orientation
##################

.. screenshot:: stage1_fix_orientation
   :alt: Fix Orientation with its Rotate panel

Turns pages that were scanned sideways or upside down, and cuts an optional
border from the scan. Every later stage works on the rotated, trimmed image.

Rotate
======

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Control
     - What it does
     - When to use it
   * - Round button, left
     - Turns the page a quarter turn to the left, in 90 degree steps.
     - The top of the text points to the right.
   * - Round button, right
     - Turns the page a quarter turn to the right, in 90 degree steps.
     - The top of the text points to the left.
   * - **Reset**
     - Returns the page to the rotation it was scanned with.
     - Undoing a turn that was wrong.
   * - **Apply to ...**
     - Copies the rotation and the trim to other pages you choose.
     - The whole book, or a run of pages, was scanned the same way.

The **L** and **R** keys, the left and right arrow keys, and a double click in the
view also turn the page a quarter turn.

The arrow in the corner of the view points to where the top of the page is now,
so you can check the direction of a turn without reading the text. It turns with
the rotation, so a page turned a half turn shows it pointing down.

Crop scan (trim)
================

.. screenshot:: stage1_fix_orientation_trim
   :alt: Fix Orientation with a manual trim of the scanner bed's border

Tick **Enable manual trim** to cut a number of pixels from each edge of the scanned
image, for example a scanner bed's border or a dark edge.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Box
     - What it cuts
     - When to use it
   * - Top
     - Pixels from the top edge of the image as it was scanned.
     - A strip of scanner bed or a dark band runs along the top.
   * - Left
     - Pixels from the left edge of the image as it was scanned.
     - A strip runs down the left side.
   * - Right
     - Pixels from the right edge of the image as it was scanned.
     - A strip runs down the right side.
   * - Bottom
     - Pixels from the bottom edge of the image as it was scanned.
     - A strip runs along the bottom.

The four edges are fixed to the scan, not to the page, so a margin keeps naming
the same edge when the page is rotated. **Reset trim** sets all four to zero.

Every later step works on what is left: Page Split, Deskew, Select Content, Margins
and Output, and the thumbnails, all see the trimmed image. On a two-page scan each
page is the part of its side of the split that lies inside the trim. The trim never
leaves less than 32 pixels in either direction. Unticking the box removes the trim
but keeps the numbers shown, so ticking it again brings them back.
