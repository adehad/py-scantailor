4. Select Content
#################

.. screenshot:: stage4_select_content
   :alt: Select Content with its Page Box and Content Box panels

Finds the area of the page that holds text and pictures, and the area of the paper.
The detection code is in :mod:`scantailor.filters.select_content.content_box_finder`
for the content box and :mod:`scantailor.filters.select_content.page_finder` for the
page box.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Box
     - What it is
   * - Content Box
     - The area around the text and pictures. ScanTailor treats the area outside it
       as empty, and the Margins stage measures from it.
   * - Page Box
     - The paper itself. ScanTailor searches for content only inside it, and the
       Margins stage takes the page size from it.

Each box has three modes. In **Manual** mode, drag the edges of the box in the
image. **Apply to ...** copies the setting to other pages. On a page whose image
is a different size, a copied manual page box keeps its distance from the centre
of the image, and a copied manual content box keeps its distance from the centre
of that page's page box. ScanTailor cuts a page box that would run off the image
down to the image.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - What it does
   * - **Disable**
     - Uses the whole image as the box.
   * - **Auto**
     - Finds the box from the page's content, or from the paper edge for the page box.
       A page with no content, such as a blank page or one with only the edge of the
       book, gets no content box: Margins centres it, and with **Fill margins** on
       Output leaves it white.
   * - **Manual**
     - Uses the box you drew by hand.

A box cannot be made smaller than 10 pixels on screen. A double click corrects the
content box from the page's content.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Action
     - Effect
   * - Double click outside the content box
     - The box grows to take in the content under the pointer.
   * - Double click inside the content box
     - The nearest side moves to the content.
   * - Hold **Ctrl** and double click
     - Moves only the top or the bottom side.
   * - Hold **Shift** and double click
     - Moves only the left or the right side.
   * - **Create Content Box** in the context menu, or the **Ins** key
     - Creates a content box. A double click on empty space does not create one.

When to enable the Page Box
===========================

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Page Box
     - When to use it
   * - **Disable**
     - The scan holds only the page, with no background, bed or shadow around it.
       The whole image is then the page.
   * - **Auto**
     - The scan shows the page on a dark background, a scanner bed or a shadow along
       its edges. Content detection can take those edges for text or pictures, and
       the Page Box keeps the search inside the paper. The Margins stage then
       measures the margins from the paper edge instead of the scan edge. On a
       dark background each side of the box stops where the background ends. A
       side where the scanner cut the background off, so the paper runs to the
       scan edge, keeps the scan edge as the page edge.
       When the paper lies crooked, the corners of that box still show
       background; tick **Fine Tune Page Corners** to move them inward until they
       reach the paper. A light background, such as a closed scanner lid, cannot
       be told from the paper that way, so the box is taken from long straight
       edges near the image border instead, and fine tuning has no effect.
   * - **Manual**
     - **Auto** finds the paper edge in the wrong place. Drag the sides of the box in
       the image, as for the content box, or type its **Width** and **Height**,
       which resize it from its top left corner. The page box always grows to
       hold the content box. **Width** and **Height** follow the box while you
       drag it.

.. screenshot:: stage4_select_content_page_box_auto
   :alt: Select Content with the Page Box on Auto around the paper of a scan
