4. Select Content
#################

.. screenshot:: stage4_select_content
   :alt: Select Content with its Page Box and Content Box panels

Finds the area of the page that holds text and pictures, and the area of the page
itself. The detection is done by :mod:`scantailor.filters.select_content.content_box_finder`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Box
     - What it is
   * - Content Box
     - The area around the text and pictures. The area outside it is treated as
       empty, and the Margins stage measures from it.
   * - Page Box
     - The paper itself. Content is only searched for inside it, and the Margins
       stage measures the page size from it.

Each box has the modes in the table below. In manual mode drag the box edges in the
image. **Apply to ...** copies the setting to other pages.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - What it does
   * - **Disable**
     - Uses the whole image as the box.
   * - **Auto**
     - Finds the box from the page's content, or from the paper edge for the page box.
   * - **Manual**
     - Uses the box you drew by hand.

A box cannot be made smaller than 10 pixels on screen. Double click inside or
outside the content box to correct it from the page's content: outside the box it
grows to take in the content under the pointer, inside it the nearest side moves to
the content. Hold **Ctrl** to move only the top or the bottom side and **Shift** to
move only the left or the right side. A double click on empty space does not create
a box; use **Create Content Box** in the context menu or the **Ins** key for that.

When a Page Box is worth enabling
=================================

Leave the **Page Box** on **Disable** when the scan already holds only the page, with
no background, bed or shadow around it. The whole image is then the page.

Set it to **Auto** when the scan shows the page on a dark background, a scanner bed or
a shadow along its edges. Content detection can take those edges for text or
pictures, and the Page Box keeps the search inside the paper. The same box sets the
page size that the Margins stage measures from, so margins are then measured from
the paper edge rather than from the scan edge.

Use **Manual** when **Auto** finds the paper edge in the wrong place. Drag its
sides in the image, as for the content box.
