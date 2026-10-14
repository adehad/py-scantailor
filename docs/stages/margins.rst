5. Margins
##########

.. screenshot:: stage5_margins
   :alt: Margins with its Margins and Alignment panels

Sets the white space around the content. The margins are stored in millimetres, and
the spin boxes show them in the unit chosen under **Units** in :doc:`../settings`:
millimetres (the default), centimetres, inches or pixels. A pixel margin depends on
the source DPI, so a value in millimetres or inches means the same on a page of any
resolution. The calculation is in :mod:`scantailor.filters.page_layout.params`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Control
     - What it does
   * - **Top**, **Bottom**, **Left**, **Right**
     - The hard margin on each side of the content, in the chosen unit.
   * - Link buttons
     - Keep the two opposite sides equal. Switching one on makes an unequal pair
       equal at once. Both pairs start linked, and a link you switch off or on is
       remembered for the next session.
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

**Auto Margins** replaces the four spin boxes with margins worked out from the page.
The gap between the content and the page edge, from
:mod:`scantailor.filters.page_layout.page_rect`, is split between the two sides in
proportion to the distances in pixels. The margins are recalculated for each page
from its own content, so each page keeps its content where it sits on the paper.

Turn it on when the content should keep its position on each page, for example
when the text sits off centre on purpose and the page size is set by the Page Box.
Turn it off to give every page the same fixed margins with **Apply To ...**.

The view shows three rectangles: the content box, the content with the hard
margins, and that with the soft margins that bring every aligned page to one size.
Drag a solid side of the middle rectangle to set a margin. Dragging a side or a
corner of the content box sets the margin from the other end: the page moves so
that the side stays under the pointer.

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

Hold **Ctrl+Alt** and drag a guide to move it. Guides are measured from the centre of
the page, so one means the same on pages of every size. Double click the content to
snap it to the nearest guide. The guides are saved with the project.

Apply To and matching
=====================

.. screenshot:: stage5_margins_apply
   :alt: The Apply Margins dialog with the four sides to copy

**Apply To ...** under Margins asks which sides to copy: **Left**, **Right**,
**Top** and **Bottom**. The sides you leave unchecked keep each page's own value.
Margins that follow the content ignore the choice of sides.

**Match size to all pages** turns on size matching for every page at once.
**Lock aggregate size for matching** holds the size the pages are matched to, so
it does not change while you edit pages; uncheck it to follow the pages again.

**Source DPI** shows the resolution of the current page's image. Pick a preset
or type both values to correct it; a value that looks wrong is shown in red with
a hint. **Fix all...** opens the Fix DPI dialog for every image in the project.
