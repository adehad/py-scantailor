2. Split Pages
##############

.. screenshot:: stage2_split_pages
   :alt: Split Pages with its Page Layout panel

Tell ScanTailor whether a scan holds one page or two facing pages. The three
buttons choose a single page, a single page with an offcut, or two pages. The label
under them reads **Auto detected** until you change the layout yourself with
**Change ...**.

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
