Working with pages
##################

.. screenshot:: thumbnails_sorting
   :alt: The thumbnail column with sorting paused

The thumbnail column lists the pages of the project. The controls above it choose
the order and show the state of each page.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - Sort order list
     - Chooses how the pages are ordered. The choices depend on the stage shown.
       See the table below.
   * - Sorting order button
     - Reverses the chosen order.
   * - Pause sorting button
     - Pauses automatic reordering while the thumbnails refresh and shows the pages
       in project order. Press it again to go back to the chosen order. The
       selection and the current page are kept.
   * - Keep current page in view
     - Scrolls the list to the current page and keeps following it. Scrolling the
       current page out of view turns it off.
   * - Navigate through selected pages only
     - Limits the previous and next page buttons to the selected pages.
   * - Multi page selection mode
     - Lets you select several pages. See :ref:`selecting-several-pages`.
   * - Single / multi column display
     - Switches the thumbnails between one column and several.
   * - Highlight deviation
     - Marks the thumbnails of pages whose values lie far from the others.
   * - Magnify and diminish thumbnails
     - Change the thumbnail size, in steps. Ctrl++ and Ctrl+- do the same.

Sort orders by stage
--------------------

Each stage offers its own orders. The first order is always the project order.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Orders offered
   * - Fix Orientation
     - None. The list stays in project order and the sort controls are hidden.
   * - Split Pages
     - Order by split type. Pages of one type stay in project order among
       themselves.
   * - Deskew
     - Order by decreasing deviation of the deskew angle from the mean. The pages
       that differ most come first.
   * - Select Content
     - Order by increasing width, by increasing height, or by decreasing deviation
       of the content size from the mean.
   * - Margins
     - Order by increasing width, by increasing height, or by decreasing deviation
       of the hard margins from the mean.
   * - Output
     - Order by completeness. Pages the batch has not written come last.

Pages that the shown stage has not processed yet go to the end of the size and
deviation orders.

.. screenshot:: thumbnails_incomplete
   :alt: Thumbnails of pages the stage has not processed, marked with a question mark

A page the shown stage has not processed is marked with a question mark. Its
thumbnail is a guess until the stage runs. On the Output stage, a page counts as
done once a batch has written it.

.. _selecting-several-pages:

Selecting several pages
-----------------------

Turn on the multi page selection button. Then:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Action
     - What it selects
   * - Ctrl+click
     - Adds or removes one page.
   * - Shift+click
     - Selects a range of pages.
   * - Ctrl+drag over empty space
     - Adds every page the rectangle touches. The selection is extended, never
       replaced.

Insert, remove and add pages
----------------------------

Right-click a thumbnail to insert or remove pages. Right-click the empty space
after the last page to add pages at the end.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Menu entry
     - What it does
   * - Insert before ...
     - Opens the file dialog and inserts the chosen images before this page.
   * - Insert after ...
     - Opens the file dialog and inserts the chosen images after this page.
   * - Insert here ...
     - On empty space past the last page, appends the chosen images.
   * - Remove from project ...
     - Removes the selected pages after a confirmation. Input files are never
       deleted.

The removal confirmation has a **Remove output files too** box. Ticking it also
deletes the output files written for the removed pages. Leave it off to keep
them.

Images already in the project are skipped when you insert or add files.
**File > Add Files ...** appends the chosen images to the end of the project.

Pages can be inserted and removed only while no batch runs.

.. _reverse-spread-order:

Reversing the order of two-page spreads
---------------------------------------

.. screenshot:: tools_menu
   :alt: The Tools menu with Reverse two-page spread order

**Tools > Reverse two-page spread order** swaps the two halves of a split scan
between left-to-right and right-to-left reading. The halves change place in the
page list, and the output files already written for them are renamed to match.
The setting is saved with the project.

Use it when the first half of a spread was numbered on the wrong side, for
example when a book is read right to left but the project was made as left to
right.

.. _relinking:

Missing folders and relinking
-----------------------------

A project keeps the paths of its input files and its output folder. When one of
those folders has moved, the **Relinking** dialog opens on opening the project.
Cancelling it cancels the opening. You can also open it at any time from
**Tools > Relinking ...**.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part of the dialog
     - What it does
   * - Folder list
     - Lists the folders to relink: the folders of the input files and the output
       folder. On opening, only the missing ones are listed.
   * - Path display
     - Shows the path of the selected folder in parts. Click a part to choose the
       folder that replaces everything up to and including that part.
   * - Undo
     - Reverts the last change.
   * - Error line
     - Explains why a change was refused. A change is refused when it would merge
       the files of two folders into one.

Relinking one folder also relinks every listed folder below it. Press OK to apply
the changes. The project is then marked as changed, so save it to keep the new
paths.

See :mod:`scantailor.core.relinker`.
