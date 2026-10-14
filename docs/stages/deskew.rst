3. Deskew
#########

.. screenshot:: stage3_deskew
   :alt: Deskew with its angle dial and Deskew panel

Straightens pages that were scanned at a slight angle. The rotation is found from the
text lines when it is **Auto** (see :mod:`scantailor.imageproc.skew_finder`), or set by
hand when it is **Manual**.

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

By default **Auto** finds the angle from the lines of text. Tick **Top page edge
(book scans)** to find it from the top edge of the page instead, which suits book
scans with a dark background around the page. Changing it detects the angle again
for the current page while its angle is **Auto**; other pages with **Auto** keep
their angle until they are processed. A page needs a dark surround that reaches
the top of the scan; a page without one has no visible edge, and its angle is then
found from the lines of text.

Use the top page edge when the text on the page is too sparse to measure, such as a
page of a few short lines or a plate with a caption. Leave it off for ordinary text
pages, where the lines of text give the angle.

Oblique correction
==================

.. screenshot:: stage3_deskew_oblique
   :alt: Deskew with an oblique correction set by hand

A page can also lean sideways instead of turning, with its vertical edges slanted
while the text lines stay level. The oblique correction shears the page back after
it has been rotated, up to 5 degrees either way. The angle is measured as the
difference between how the text lines and how the left and right edges of the
text are skewed, so it works best on pages with justified text. The measuring code
is in :mod:`scantailor.filters.deskew.oblique`.

**Oblique mode** has its own **Auto** and **Manual** buttons, apart from the ones
for the rotation: switching one never changes the other. In **Manual** mode set the
angle with **Oblique** or by dragging the two handles on the arcs at the right
edge of the image; typing a value or dragging a handle switches to **Manual**.

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

Turn it on for a book or a batch whose pages are visibly sheared: the vertical edges
of the text block, the page edge or a ruled table lean the same way on every page,
while the lines of text sit level after the rotation. Leave it off for pages whose
only fault is the rotation, because the rotation already corrects them, and for
pages with little or ragged text, where the difference between the line skew and the
edge skew cannot be measured. On those pages set the angle by hand, or leave it at
zero.

Check a sample page after you enable it, and look at the text lines and the page
edges together.

**Apply To ...** asks which parts to copy: **Deskew angle and mode** and **Oblique
angle and mode**. A part that is **Auto** is not copied as a number; each page
detects its own.

Projects
========

ScanTailor Advanced projects store the oblique angle and mode with each page, next
to the rotation. A project saved before the oblique correction existed opens with
no oblique correction on any page.
