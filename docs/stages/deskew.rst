3. Deskew
#########

.. screenshot:: stage3_deskew
   :alt: Deskew with its angle dial and Deskew panel

Straightens pages that were scanned at a slight angle. **Auto** finds the rotation
from the text lines, with the code in :mod:`scantailor.imageproc.skew_finder`.
**Manual** uses the angle you set.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - **Auto**
     - Detects the rotation from the lines of text on the page.
   * - **Manual**
     - Sets the rotation with the spin box, or by dragging the handles on the dial.
   * - **Apply To ...**
     - Copies the angle to other pages. A page set to **Auto** is copied as **Auto**,
       so each page detects its own angle.

Top page edge
=============

By default **Auto** finds the angle from the lines of text. **Top page edge (book
scans)** makes it find the angle from the top edge of the page instead. The page
needs a dark surround that reaches the top of the scan. A page without one has no
visible edge, and its angle then comes from the lines of text.

Changing the box detects the angle again for the current page while its angle is
**Auto**. Other pages on **Auto** keep their angle until they are processed.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Top page edge
     - When to use it
   * - Ticked
     - Book scans with a dark background around the page, where the text is too
       sparse to measure, such as a page of a few short lines or a plate with a
       caption.
   * - Unticked
     - Ordinary text pages, where the lines of text give the angle.

Oblique correction
==================

.. screenshot:: stage3_deskew_oblique
   :alt: Deskew with the oblique angle found by Oblique mode Auto

A page can also lean sideways instead of turning, with its vertical edges slanted
while the text lines stay level. The oblique correction shears the page back after
the rotation, by up to 5 degrees either way. ScanTailor measures the oblique angle as
the difference between the skew of the text lines and the skew of the left and right
edges of the text, so it works best on pages with justified text. The measuring code
is in :mod:`scantailor.filters.deskew.oblique`.

**Oblique mode** has its own **Auto** and **Manual** buttons, separate from the ones
for the rotation. Switching one never changes the other. In **Manual** mode, set the
angle with **Oblique** or drag the two handles on the arcs at the top and bottom of
the image. Typing a value or dragging a handle switches to **Manual**.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - What it does
   * - Oblique **Auto**
     - Measures the oblique angle on each page from its text. Pages that are not
       processed yet are measured only when automatic oblique correction is on.
   * - Oblique **Manual**
     - Uses the angle you set, on every page it is copied to.
   * - **Automatic oblique correction**
     - Decides whether pages that have not been processed yet detect their oblique
       angle. Off by default, so a new page is only rotated. The same setting is in
       **Default Parameters**. Changing it also changes the current page while that
       page's rotation is **Auto**.

When to enable oblique correction
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Pages
     - Oblique correction
     - Why
   * - The vertical edges of the text block, the page edge or a ruled table lean the
       same way on every page, while the lines of text sit level after the rotation.
     - On
     - The pages are sheared, and the rotation cannot correct a shear.
   * - The only fault is the rotation.
     - Off
     - The rotation already corrects them.
   * - Little or ragged text.
     - Off, or set the angle by hand
     - ScanTailor cannot measure the difference between the line skew and the edge
       skew. Set the angle by hand, or leave it at zero.

After you enable it, check a sample page and compare the text lines with the page
edges.

**Apply To ...** asks which parts to copy: **Deskew angle and mode** and **Oblique
angle and mode**. A part on **Auto** is copied as **Auto**, not as a number, so each
page detects its own angle.

Projects
========

ScanTailor Advanced projects store the oblique angle and mode with each page, next
to the rotation. A project saved before the oblique correction existed opens with
no oblique correction on any page.
