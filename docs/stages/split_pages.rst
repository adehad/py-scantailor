2. Split Pages
##############

.. screenshot:: stage2_split_pages
   :alt: Split Pages with its Page Layout panel

Tell ScanTailor whether a scan holds one page or two facing pages. The three
buttons choose a single page, a single page with an offcut, or two pages. The label
under them reads **Auto detected** until you change the layout yourself with
**Change ...**.

How the layout is detected
==========================

**Auto** looks for a folding line or dark spine first, and otherwise for the
white gap between two pages of text. A spread that is tilted gets a tilted split
line. With **Single page with an offcut**, the page is cut where its text starts,
or at the image edge when no edge can be told apart. Under **Auto**, a narrow
strip of a neighbouring page's text along the edge of a single-page scan is cut
off when a clear gutter separates it from the page's text; text of the page
itself that runs off the scan is left alone.

The split lines
===============

.. screenshot:: stage2_split_pages_tilted
   :alt: Split Pages with a tilted split line and a handle at each end

A split line has a round handle at each end. Drag a handle to tilt the line, for a
spread whose spine is not upright; drag the line itself to move it without changing
its tilt. A line cannot cross its neighbour or leave the image. The tint on each
side of the line shows where each page ends, so a tilted line cuts the page along
the tilt in the output as well.

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

.. admonition:: TODO
   :class: attention

   Owner's words needed: when to prefer each of the three layouts, and how the split line is adjusted for two-page scans.
