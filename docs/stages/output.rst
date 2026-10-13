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

**Binarization** offers Otsu, Sauvola, Wolf, Fox, Window, Bradley, Grad and
EdgeDiv. Fox, Window and Grad read the same coefficient, lower bound and upper
bound as Wolf. Local methods take the window size and the threshold adjustment;
a positive adjustment turns more pixels black.

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

Picture and fill zones are created from the same context menu. Three ways of
drawing a zone are available, and **Z**, **X** and **C** switch between them while
you edit: **Z** clicks the corners of a polygon and finishes with a double click
or Enter, **X** drags a freehand outline, and **C** drags a rectangle. The way a
view starts in is **Default zone creation mode** in :doc:`../settings`; pressing
one of the keys also makes that the default.

Output depends on every page's content size, so run the batch first
(:doc:`../batch`) when this stage reports that pages are not ready.

.. admonition:: TODO
   :class: attention

   Owner's words needed: guidance on choosing a mode and the despeckling level.
