1. Fix Orientation
##################

.. screenshot:: stage1_fix_orientation
   :alt: Fix Orientation with its Rotate panel

Turn pages that were scanned sideways or upside down. The two round buttons rotate
the page a quarter turn left or right, and **Reset** returns it to how it was
scanned. **Apply to ...** copies the rotation, and the trim below, to other pages.

Crop scan (trim)
================

.. screenshot:: stage1_fix_orientation_trim
   :alt: Fix Orientation with a manual trim of the scanner bed's border

Tick **Enable manual trim** to cut a number of pixels from each edge of the scanned
image, for example a scanner bed's border or a dark edge. The four boxes are the
pixels to cut from the top, left, right and bottom edge of the image as it was
scanned, so a margin keeps naming the same edge when the page is rotated.
**Reset trim** sets all four to zero.

Every later step works on what is left: Page Split, Deskew, Select Content, Margins
and Output, and the thumbnails, all see the trimmed image. On a two-page scan each
page is the part of its side of the split that lies inside the trim. The trim never
leaves less than 32 pixels in either direction. Unticking the box removes the trim
but keeps the numbers shown, so ticking it again brings them back.

.. admonition:: TODO
   :class: attention

   Owner's words needed: the meaning of the arrow shown under the rotate buttons.
