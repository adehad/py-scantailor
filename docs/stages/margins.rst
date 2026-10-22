5. Margins
##########

.. screenshot:: stage5_margins
   :alt: Margins with its Margins and Alignment panels

Sets the white space around the content. ScanTailor stores the margins in
millimetres, and the spin boxes show them in the unit chosen under **Units** in
:doc:`../settings`. The units are millimetres, which is the default, centimetres,
inches and pixels. A pixel margin depends on the source DPI, the smaller of the two
when they differ (see :ref:`fix-dpi`), so a value in millimetres or inches means the
same on a page of any resolution. The conversion is
in :mod:`scantailor.filters.page_layout.params`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Control
     - What it does
   * - **Top**, **Bottom**, **Left**, **Right**
     - The hard margin on each side of the content, in the chosen unit.
   * - Link buttons
     - Keep the two opposite sides equal. Switching one on makes an unequal pair
       equal at once. Both pairs start linked, and ScanTailor remembers a link you
       switch off or on for the next session.
   * - **Auto Margins**
     - Sets the margins from where the content sits on the page, and disables the
       four spin boxes. See below.
   * - **Match size with other pages**
     - Gives every page the same size, and the alignment buttons place the content
       inside it.
   * - **Apply To ...**
     - Copies the margins to other pages. Pick which sides to copy.

Auto Margins
============

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Auto Margins
     - What it does
     - When to use it
   * - On
     - Replaces the four spin boxes with margins worked out from each page's own
       content. ScanTailor splits the gap between the content and the page edge,
       from :mod:`scantailor.filters.page_layout.page_rect`, between the two sides
       in proportion to the distances in pixels. The content keeps its position on
       the paper.
     - The text sits off centre on purpose, and the Page Box sets the page size.
   * - Off
     - The four spin boxes set the margins.
     - Every page gets the same fixed margins, copied with **Apply To ...**.

The view shows three rectangles: the content box, the content box with the hard
margins, and the content box with the soft margins that bring every aligned page to
one size. Drag a solid side of the middle rectangle to set a margin. Dragging a side
or a corner of the content box sets the margin from the other end. The page then
moves so that the side stays under the pointer.

Pages with little content
=========================

New pages are centred inside the matched page size, as in upstream ScanTailor
Advanced. On a page that holds only a folio or a printer's mark, centring moves that
mark to the middle of the page. Pick another alignment for such pages.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Alignment
     - Where the content goes
     - When to use it
   * - **Original**
     - Stays where it sat on the scan.
     - A page that holds only a folio or a printer's mark.
   * - **Bottom**
     - Goes to the foot of the page.
     - A folio at the foot of the page.

Set the alignment on one page and copy it to the others with **Apply To ...**.

Moving the content
==================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Action
     - Effect
   * - **Ctrl** and drag the content
     - Moves it up or down.
   * - **Shift** and drag the content
     - Moves it left or right.
   * - **Ctrl+Shift** and drag the content
     - Moves it freely in both directions.

The content stays inside the hard margins rectangle.

Guides
======

.. screenshot:: stage5_margins_guides
   :alt: Margins with a vertical and a horizontal guide as dashed lines

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Item in the view's context menu
     - What it does
   * - **Add a horizontal guide**, **Add a vertical guide**
     - Puts a dashed guide where you clicked.
   * - **Remove this guide**, **Remove all the guides**
     - Takes the guides away.
   * - **Show hard margins rectangle**
     - Shows or hides the middle rectangle.
   * - **Show grid**
     - Draws a line every 10 mm through the centre of the page, for comparing
       positions between pages. Zoomed out, every other line is left out. Pages
       that are not aligned offer only this item. The choice is remembered.

Hold **Ctrl+Alt** and drag a guide to move it. ScanTailor measures guides from the
centre of the page, so a guide keeps its place on pages of every size. Double click
the content to snap it to the nearest guide. The project stores the guides.

Apply To and matching
=====================

.. screenshot:: stage5_margins_apply
   :alt: The Apply Margins dialog with the four sides to copy

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - **Apply To ...**
     - Asks which sides to copy: **Left**, **Right**, **Top** and **Bottom**. The
       sides you leave unchecked keep each page's own value. With **Auto Margins**
       on, ScanTailor ignores the choice of sides.
   * - **Match size to all pages**
     - Turns on size matching for every page at once.
   * - **Lock aggregate size for matching**
     - Holds the size the pages are matched to, so it does not change while you edit
       pages. Uncheck it to follow the pages again.
   * - **Source DPI**
     - Shows the resolution of the current page's image. Pick a preset or type both
       values to correct it. A value that looks wrong turns red and shows a hint.
       The status bar shows the same resolution in every stage.
   * - **Fix all...**
     - Opens the Fix DPI dialog for every image in the project.
