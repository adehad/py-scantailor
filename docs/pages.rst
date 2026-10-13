Working with pages
##################

.. screenshot:: thumbnails_sorting
   :alt: The thumbnail column with sorting paused

The thumbnail column lists the pages of the project. The list under it chooses the
order, such as by width or by height. The pause button next to the buttons on its
left stops the list from reordering itself while thumbnails refresh and shows the
pages in the order of the project; press it again to resume the chosen order.
Pausing keeps the selection and the current page.

Each stage offers its own orders. Split Pages orders by split type, Deskew by
decreasing deviation of the angle, Select Content and Margins by width, height and
deviation, and Output by completeness; Fix Orientation has no choice. The button
above the list reverses the chosen order. Pages that the shown stage has not
processed yet are listed last by size and deviation orders.

.. screenshot:: thumbnails_incomplete
   :alt: Thumbnails of pages the stage has not processed, marked with a question mark

A page that the shown stage has not processed yet is marked with a question mark,
because its thumbnail is only a guess until the stage has run. Output counts a page
as done once a batch has written it.

Selecting several pages
-----------------------

With the multi-select button on, hold **Ctrl** and drag a rectangle across empty
space between thumbnails to add every page it touches to the selection. If the
current page is outside the rectangle, the first page touched becomes current.

Reversing the order of two-page spreads
---------------------------------------

.. screenshot:: tools_menu
   :alt: The Tools menu with Reverse two-page spread order

**Tools > Reverse two-page spread order** swaps which half of a split scan comes
first, for books that are read from right to left. The setting is saved with the
project. Output files already written under the old order are not renamed; they
are written again under the new names when the pages are processed.

Missing folders
---------------

A project whose output folder is missing opens the same relinking dialog as one
whose image folders are missing, and cancelling it cancels the opening.
