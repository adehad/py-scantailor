6. Output
#########

.. screenshot:: stage6_output
   :alt: Output with its Output Resolution, Mode and Options panels

Produces the final page images at the DPI you set, from the content box, the margins
and the choices below.

Output resolution
=================

**DPI** sets the resolution of the output images. The output size in pixels is the
page size at that DPI, so a higher value makes larger files.

Mode
====

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Mode
     - What it produces
     - When to use it
   * - **Black and White**
     - Every page is binarized to black and white, with the binarization method and
       the despeckling you choose.
     - Text-only books, where a small file and sharp text matter most.
   * - **Color / Grayscale**
     - Keeps the colour or grey levels of the scan. Despeckling is not applied, and
       the Despeckling tab says so.
     - Books with coloured illustrations, photographs, or paper that is not white.
   * - **Mixed**
     - Binarizes the text and keeps the pictures in colour or grey. The Picture Zones
       tab shows the mask of what is kept as a picture.
     - Books with text and pictures on the same page, such as a textbook with plates.

**White margins** sets the polarity of the black and white result. It is on by
default, which gives black text on a white background. Turn it off for white text on
a black background.

Options
=======

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - What it does
   * - **Fill margins**
     - Fills the margin around the content with the filling colour.
   * - **Fill offcut**
     - Paints everything outside the page outline, such as the cut-off part of the
       neighbouring page, with the filling colour.
   * - **Fill outside page box**
     - An alternative to **Fill offcut**. The page outline is ignored, and the filling
       colour, the illumination and the area that is kept come from the whole output
       rectangle. The two exclude each other.
   * - **Equalize illumination**
     - Evens out uneven lighting across the page before binarizing, such as a shadow
       from the spine.
   * - **Savitzky-Golay smoothing**, **Morphological smoothing**
     - Smooth the edges of the black strokes.

.. screenshot:: stage6_output_options
   :alt: Output with Fill outside page box ticked and the Window binarization method

Binarization
============

Binarization decides which pixels become black. The method sets how the threshold is
found. The threshold adjustment or coefficient sets how far it moves.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - How the threshold is found
     - When to use it
   * - Otsu
     - One threshold for the whole page, from its histogram.
     - Even lighting and clean paper.
   * - Sauvola, Wolf, Bradley
     - A threshold for each area from its neighbourhood, with a window size.
     - Uneven lighting, shadows, or grey paper.
   * - Fox, Window, Grad
     - Variants of Wolf that show the Wolf controls. Fox adds Singh's local contrast,
       Window uses a dynamic window, and Grad blends a local mean with a page level
       taken where the contrast is.
     - The same cases as Wolf, when you want to try a variant on a sample page.
   * - EdgePlus, BlurDiv, EdgeDiv
     - EdgeDiv uses an edge term and a blur term. EdgePlus and BlurDiv use only one
       of them, each with its own coefficient.
     - Faint strokes, such as pencil or light print, that the other methods drop.

Local methods take the window size and the threshold adjustment. A positive
adjustment turns more pixels black. Compare the methods on a sample page in the
Output view before you run the batch.

.. screenshot:: stage6_output_edge_methods
   :alt: Binarization with the EdgePlus method, showing its window size and coefficient

Where the output has a margin that the source image does not reach, such as the
corner a rotation exposes, the margin takes the filling colour in every mode, so
it matches the paper instead of showing white.

Despeckling
===========

**Despeckle** removes black specks that are small and far from text. A second pass
looks again at specks whose only close neighbours are too small to keep them. In
that pass the ink of marks already kept counts as a neighbour, so a speck next to
kept ink can attach to it. This keeps small marks that belong to the text and that
a single pass loses, for example the dots of Arabic letters, the dots of i and j,
and full stops.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Upstream ScanTailor Advanced
     - This port
     - Why
   * - Removes more small marks at the same level.
     - Keeps more small marks at the same level, with the second pass.
     - The stricter upstream result probably comes from a bug in how upstream
       measures the distance to the ink. Nobody has confirmed the bug against the
       upstream program, and the port keeps the second pass on purpose.

The strength has four steps:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Level
     - Strength
     - When to use it
   * - Off
     - 0
     - Clean scans with no specks, or when every mark matters.
   * - Cautious
     - 1
     - The default. Removes only small noise, and keeps the dots of Arabic letters
       and the dots of i and j.
   * - Normal
     - 2
     - Scans with visible speckle from the paper or the scanner that Cautious leaves
       behind.
   * - Aggressive
     - 3
     - Heavy speckle that Normal leaves behind. It may remove small details, such as
       fine punctuation and thin strokes, so check the Despeckling tab.

The slider moves through these steps. The strength is stored as a number from 0 to
3.5, as upstream stores it, and the named steps are the presets 0, 1, 2 and 3.
A project that stores a value between two steps keeps that value, and the tooltip on
the slider shows it. Choosing a named step sets the strength to that preset.

Choose the lowest level that removes the specks you can see on the Despeckling tab.
Higher levels remove small marks first, so check the Arabic dots and the i-dots after
each step.

.. screenshot:: stage6_output_despeckle_levels
   :alt: The Despeckle slider at Aggressive, with the specks it removes on the Despeckling tab

.. screenshot:: stage6_tab_despeckling
   :alt: The Despeckling tab showing the text before despeckling

The **Despeckling** tab shows the page before despeckling, with the specks the
chosen level removes in red and a halo of red around each, fading with the
distance. The halo shows how far from the text a speck has to be to go. The pictures
of a mixed page show through unchanged. The tab previews every level as you pick it,
and the page renders again only once the slider has rested. Despeckling applies to
black and white pages and to the text of mixed pages. In **Color / Grayscale** mode
the tab says that it does not apply.

Picture shape
-------------

In mixed mode the **Picture Shape** panel chooses how detected pictures are shaped.
Options that take a number render the page again once the value has rested for
three quarters of a second.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Shape
     - What it does
   * - **Off**
     - Keeps the pictures as they were detected, without reshaping them.
   * - **Free**
     - Follows the outline of each detected picture.
   * - **Rectangular**
     - Reduces each picture to a rectangle. The sensitivity sets how readily an area
       counts as a picture.

.. screenshot:: stage6_output_picture_shape
   :alt: The Picture Shape panel in mixed mode

.. screenshot:: stage6_output_guides
   :alt: The output view with the guides shown

The **Show guides** item of the context menu of the output view draws a dashed
cross through the centre of the view, for judging the alignment of the output.
The choice is remembered.

Zones
=====

The output view has five tabs. **Ctrl+1** to **Ctrl+5** switch between them.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Tab
     - Shortcut
     - What it shows
   * - **Output**
     - Ctrl+1
     - The finished output image.
   * - **Picture Zones**
     - Ctrl+2
     - The page with the picture zones drawn over it.
   * - **Fill Zones**
     - Ctrl+3
     - The output image, with the fill zones drawn over it.
   * - **Dewarping**
     - Ctrl+4
     - The page before dewarping, with the distortion grid.
   * - **Despeckling**
     - Ctrl+5
     - The page before despeckling, with the specks the level removes.

.. screenshot:: stage6_tab_picture_zones
   :alt: The Picture Zones tab with a picture zone drawn over the page

A zone tab shows and edits one kind of zone. Click anywhere that is not a corner
or an edge to start a zone. The **Z**, **X** and **C** keys switch between the three
ways of drawing a zone. A view starts in the way set by **Default zone creation
mode** in :doc:`../settings`. Pressing one of the keys also makes that way the
default.

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Key
     - Mode
     - How to draw
   * - **Z**
     - Polygon
     - Click the corners, and finish with a double click, Enter, or a click on the
       first corner.
   * - **X**
     - Freehand
     - Drag an outline.
   * - **C**
     - Rectangle
     - Drag a rectangle.

To change a zone:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Action
     - Effect
   * - Drag a corner
     - Moves the corner. **Ctrl** squares the corner, and dropping it on a neighbour
       merges the two.
   * - Press on an edge
     - Adds a corner there and drags it.
   * - Hover over a corner and press **D**
     - Removes the corner. A zone keeps at least three.
   * - **Shift** and drag inside the zone
     - Moves the zone. **Shift+Ctrl** moves a copy.
   * - **Ctrl+Alt+click**
     - Copies the latest zone there.
   * - **Delete** over the zone
     - Removes the zone.

A right click lists every zone under the pointer.

.. screenshot:: stage6_zone_menu
   :alt: The context menu of a picture zone

**Properties** of a picture zone chooses its layer. **Pick color** of a fill zone lets
you click a colour in the page. It takes the median colour of a small circle around
the click, and **Escape** cancels. ScanTailor remembers the layer and the colour for
the next zone.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Picture zone layer
     - What it does
   * - Subtract from all layers
     - Removes the area from the picture on every layer.
   * - Add to the auto layer, or subtract from it
     - Adds the area to, or removes it from, the pictures ScanTailor detects.
   * - Add to the foreground, or to the background
     - Places the area in the foreground or the background layer.

.. screenshot:: stage6_zone_properties
   :alt: The layer choices of a picture zone

.. screenshot:: stage6_tab_picture_mask
   :alt: The Picture Zones tab in mixed mode with the picture mask shown

In mixed mode the Picture Zones tab also shows the mask of what the output keeps as
a picture. The mask pulses slowly so that you can see the page under it, and your
zones are applied to it in the order the output applies them. It pulses only while
this tab is showing.

.. screenshot:: status_tip
   :alt: The status bar telling what the pointer is over in a zone tab

While the pointer is over a zone, a vertex, an edge or the dewarping grid, the
status bar says what a click or a drag does there, including the keys that change
it. In the Dewarping tab, hold **Ctrl** while dragging an end point of an edge to move
it only vertically, or **Shift** to move it only horizontally.

.. screenshot:: stage6_tab_fill_zones
   :alt: The Fill Zones tab with a fill zone drawn over the page

The **Fill Zones** tab shows the output image, so on a dewarped page the zones are
drawn and edited on the dewarped page and stored in page coordinates. If the
dewarping model cannot be used, they are drawn on the undewarped page.

The **Dewarping** tab shows the page as it is before dewarping, with the distortion
grid over it. Drag the corners and edges of the grid as you would those of a zone.
Editing the grid of a page whose dewarping is **Auto** or **Marginal** makes it
**Manual**. **Depth perception** is the slider that appears while this tab shows.

.. screenshot:: stage6_tab_dewarping
   :alt: The Dewarping tab with the distortion grid

Automatic dewarping
-------------------

**Auto** traces the text lines of the page and the top and bottom edges of the
paper, and takes as the model the pair of curves that leaves all the lines
straightest. The fitting code is in :mod:`scantailor.dewarping.auto_model`.

Once the page has rendered, the **Dewarping** panel reports what **Auto** found in
one line, for example "2 of 33 lines kept, no page edge. Curves span 7% of the text.
Below the 40% minimum, so the page is left as is; a minimum of 7% or less would use
them." The line gives:

* how many text lines were traced, and how many survived the filters that drop
  short, stray and doubly bent lines;
* how many page edges were found;
* how far apart the two chosen curves lie, as a share of the content height;
* whether each curve came from a text line or a page edge;
* why the page gets no model, when it gets none.

The Dewarping tab shows the same line along its bottom edge.

.. screenshot:: stage6_dewarping_metrics
   :alt: The Dewarping panel with the minimum span and what auto dewarping found

Two behaviours differ from ScanTailor Advanced 1.2.1 on purpose.

.. list-table::
   :header-rows: 1
   :widths: 16 22 32 30

   * - Behaviour
     - Upstream 1.2.1
     - This port
     - Why
   * - **Minimum span (%)**
     - Accepts any pair of curves.
     - Refuses a model whose two curves lie closer together than this share of the
       content height, and leaves the page as it is. The default is 40%, a tuned
       value. Set it to 0 to accept every model, as upstream does. **Apply to ...**
       copies it with the other dewarping options.
     - ScanTailor stretches the model over the whole page from the two curves. When
       only two neighbouring lines survive, as on a page of short centred lines, a
       wobble of one pixel in one of them becomes an error of dozens of pixels at the
       far end of the page.
   * - Scanner border
     - Accepts a straight edge on the first or the last row of the scan as the page
       edge, and often prefers it to the last text line.
     - Ignores a page edge that lies within 1.5% of the image height of the first or
       the last row and bends by at most 2 pixels at the size the edges are traced
       at, which is about 6 pixels on a 300 dpi scan. A paper edge further in, or one
       that bends, still counts.
     - Such an edge is the scanner's own border. Upstream's model treats the bottom
       of the page as flat and corrects the lower half too little.

Output depends on every page's content size. When this stage reports that pages are
not ready, run the batch first, as :doc:`../batch` describes.
