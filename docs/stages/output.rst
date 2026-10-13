6. Output
#########

.. screenshot:: stage6_output
   :alt: Output with its Output Resolution, Mode and Options panels

Produces the final page images. **DPI** sets the output resolution. **Mode** is
**Black and White**, **Color / Grayscale** or **Mixed**; **White margins** paints
the area outside the content white. **Options** holds the clean-up filters, such as
**Fill margins**, **Fill offcut**, **Equalize illumination** and the smoothing
options.

**Fill offcut** paints everything outside the page outline, such as the cut-off
part of the neighbouring page, with the filling colour. **Fill outside page box**
is its alternative: the page outline is ignored, and the filling colour, the
illumination and the area that is kept come from the whole output rectangle. The
two exclude each other.

.. screenshot:: stage6_output_options
   :alt: Output with Fill outside page box ticked and the Window binarization method

**Binarization** offers Otsu, Sauvola, Wolf, Fox, Window, Bradley, Grad,
EdgePlus, BlurDiv and EdgeDiv. Fox, Window and Grad read the same coefficient,
lower bound and upper bound as Wolf. EdgePlus and BlurDiv are EdgeDiv with only
its edge term or only its blur term, each with its own coefficient. Local methods
take the window size and the threshold adjustment; a positive adjustment turns more
pixels black.

.. screenshot:: stage6_output_edge_methods
   :alt: Binarization with the EdgePlus method, showing its window size and coefficient

Where the output has a margin that the source image does not reach, such as the
corner a rotation exposes, the margin takes the filling colour in every mode, so
it matches the paper instead of showing white.

**Despeckle** removes black specks that are small and far from text. It runs a
second pass that gives a speck a second chance to reach a large neighbour, with
the ink already kept counting as its own source. This keeps small marks that
belong to the text and that a single pass loses, for example the dots of Arabic
letters, the dots of i and j, and full stops. The result therefore keeps more
small marks than the original ScanTailor Advanced does for the same level.

.. screenshot:: stage6_output_guides
   :alt: The output view with the guides shown

The **Show guides** item of the context menu of the output view draws a dashed
cross through the centre of the view, for judging the alignment of the output.
The choice is remembered.

Zones
=====

The output view has the tabs **Output**, **Picture Zones**, **Fill Zones**,
**Dewarping** and **Despeckling**; **Ctrl+1** to **Ctrl+5** switch between them.

.. screenshot:: stage6_tab_picture_zones
   :alt: The Picture Zones tab with a picture zone drawn over the page

A zone tab shows and edits one kind of zone. Click anywhere that is not a corner
or an edge to start a zone. Three ways of drawing a zone are available, and **Z**,
**X** and **C** switch between them while you edit: **Z** clicks the corners of a
polygon and finishes with a double click, Enter or a click on the first corner,
**X** drags a freehand outline, and **C** drags a rectangle. The way a view starts
in is **Default zone creation mode** in :doc:`../settings`; pressing one of the
keys also makes that the default.

To change a zone:

* Drag a corner. **Ctrl** squares the corner, and dropping it on a neighbour
  merges the two.
* Press on an edge to add a corner there and drag it.
* Hover over a corner and press **D** to remove it; a zone keeps at least three.
* Hold **Shift** and drag inside a zone to move it, or **Shift+Ctrl** to move a
  copy. **Ctrl+Alt+click** copies the latest zone there.
* Press **Delete** over a zone to remove it.

The right button lists every zone under the cursor.

.. screenshot:: stage6_zone_menu
   :alt: The context menu of a picture zone

**Properties** of a picture zone chooses its layer: subtract from all layers, add to
or subtract from the auto layer, or add to the foreground or background. **Pick
color** of a fill zone lets you click a colour in the page; it samples the median of
a small circle, and Escape cancels. The layer and the colour are remembered for the
next zone.

.. screenshot:: stage6_zone_properties
   :alt: The layer choices of a picture zone

.. screenshot:: stage6_tab_fill_zones
   :alt: The Fill Zones tab with a fill zone drawn over the page

The **Dewarping** tab shows the page as it is before dewarping, with the distortion
grid over it. Drag the corners and edges of the grid as you would those of a zone;
editing the grid of a page whose dewarping is **Auto** or **Marginal** makes it
**Manual**. **Depth perception** is the slider that appears while this tab shows.

.. screenshot:: stage6_tab_dewarping
   :alt: The Dewarping tab with the distortion grid

The **Despeckling** tab shows the page before despeckling, with the specks the
chosen level removes in red. It previews every level as you pick it, and the page
renders again only once the slider has rested. Despeckling applies to black and
white pages and to the text of mixed pages; in **Color / Grayscale** mode the tab
says so.

.. screenshot:: stage6_tab_despeckling
   :alt: The Despeckling tab showing the text before despeckling

In mixed mode the **Picture Shape** panel chooses how detected pictures are
shaped: **Off**, **Free** or **Rectangular**, the last with a sensitivity.
Options that take a number render the page again once the value has rested for
three quarters of a second.

.. screenshot:: stage6_output_picture_shape
   :alt: The Picture Shape panel in mixed mode

Output depends on every page's content size, so run the batch first
(:doc:`../batch`) when this stage reports that pages are not ready.

.. admonition:: TODO
   :class: attention

   Owner's words needed: guidance on choosing a mode and the despeckling level.
