3. Deskew
#########

.. screenshot:: stage3_deskew
   :alt: Deskew with its angle dial and Deskew panel

Straightens pages that were scanned at a slight angle. **Auto** detects the angle;
**Manual** lets you set it with the spin box or by dragging the handles on the
dial. **Apply To ...** copies the angle to other pages.

Oblique correction
==================

.. screenshot:: stage3_deskew_oblique
   :alt: Deskew with an oblique correction set by hand

A page can also lean sideways instead of turning, with its vertical edges slanted
while the text lines stay level. The oblique correction shears the page back after
it has been rotated, up to 5 degrees either way. The angle is measured as the
difference between how the text lines and how the left and right edges of the
text are skewed, so it works best on pages with justified text.

**Oblique mode** has its own **Auto** and **Manual** buttons, apart from the ones
for the rotation: switching one never changes the other. In **Manual** mode set the
angle with **Oblique** or by dragging the two handles on the arcs at the right
edge of the image; typing a value or dragging a handle switches to **Manual**.

**Automatic oblique correction** decides whether pages that have not been
processed yet detect their oblique angle. It is off by default, so a new page is
only rotated. The same setting is in **Default Parameters**. Changing it also
changes the current page while that page's rotation is **Auto**.

**Apply To ...** asks which parts to copy: **Deskew angle and mode** and **Oblique
angle and mode**. A part that is **Auto** is not copied as a number; each page
detects its own.

Projects
========

ScanTailor Advanced projects store the oblique angle and mode with each page, next
to the rotation. A project saved before the oblique correction existed opens with
no oblique correction on any page.

.. admonition:: TODO
   :class: attention

   Owner's words needed: when the oblique correction is worth enabling.
