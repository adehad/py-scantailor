Working with pages
##################

.. screenshot:: thumbnails_sorting
   :alt: The thumbnail column with sorting paused

The thumbnail column lists the pages of the project. The buttons above it set the
page order and how the thumbnails are shown.

Fix Orientation and Split Pages work on whole scans, so at those two stages the
column shows one thumbnail per scanned image, and a two-page spread appears once.
The **Apply to ...** options that count pages, such as **Every other page**, count
scans there. From Deskew on, each page has its own thumbnail. The current page and
the selection carry over: a scan is selected when any of its pages is, and going
on to Deskew selects both pages of a selected scan.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - Sort order list
     - Chooses how the pages are ordered. The choices depend on the stage shown.
       See the table below. Each stage keeps its own choice while the project is
       open.
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
     - Makes a plain click add or remove a page. See
       :ref:`selecting-several-pages`.
   * - Single / multi column display
     - Switches the thumbnails between one column and several.
   * - Highlight deviation
     - Marks the thumbnails of pages that stand out from the others. See
       :ref:`pages-that-stand-out`.
   * - Magnify and diminish thumbnails
     - Change the thumbnail size in steps. Ctrl++ and Ctrl+- do the same.

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
     - Order by decreasing deviation of the deskew angle from the median. The pages
       that differ most come first.
   * - Select Content
     - Order by increasing width, by increasing height, or by decreasing deviation
       of the content size from the median.
   * - Margins
     - Order by increasing width, by increasing height, or by decreasing deviation
       of the hard margins from the median.
   * - Output
     - Order by completeness. Pages the batch has not written come last.

Pages that the shown stage has not processed yet go to the end of the size and
deviation orders.

.. screenshot:: thumbnails_incomplete
   :alt: Thumbnails of pages the stage has not processed, marked with a question mark

A question mark marks a page that the shown stage has not processed. Its
thumbnail is a guess until the stage runs. On the Output stage, a page counts as
done once a batch has written it.

What the thumbnails show
------------------------

Each thumbnail shows the page as the current stage leaves it, with that stage's
result drawn over it in the colours of the stage's view.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Drawn over the thumbnail
   * - Split Pages
     - The whole image, with each page tinted and the split lines drawn. A half
       removed from the project is tinted red.
   * - Deskew
     - A grid, to judge whether the lines of text are level.
   * - Select Content
     - The content box, tinted, and the page box when page detection is on.
   * - Margins
     - The margins, tinted, around the outlined content. Purple with a dashed
       border marks a page matched to the size of the others, blue a page left
       at its own size.

Fix Orientation and Output show the page alone.

.. _pages-that-stand-out:

Pages that stand out
--------------------

On Deskew, Select Content and Margins, a red asterisk marks a page whose result
lies far from the other pages'. A wrong detection usually shows up this way, so
after a batch you can look at the marked pages first. Order by decreasing
deviation puts them at the top.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - What is compared
   * - Deskew
     - The deskew angle.
   * - Select Content
     - The diagonal of the content box. A page whose content box is empty is
       always marked.
   * - Margins
     - The diagonal of the hard margins: left plus right against top plus bottom.

Only pages the stage has processed take part, and nothing is marked while fewer
than three have. A page is marked when its value is further from the median of
all pages than both of these:

- the stage's coefficient times the spread of the values, measured so that one
  bad page does not widen it;
- the stage's threshold, a percentage of the median.

On values that vary normally, the spread equals the standard deviation, so a
coefficient of 2 marks about one page in twenty. Both settings are in
**Tools > Settings**. The **Highlight deviation** button turns the marks off and
on without redrawing the thumbnails.

.. _selecting-several-pages:

Selecting several pages
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Action
     - What it selects
   * - Click
     - Only that page. With the multi page selection button on, a click adds or
       removes one page, as Ctrl+click does.
   * - Ctrl+click
     - Adds or removes one page. The last selected page stays selected.
   * - Shift+click
     - Adds every page from the current page to the clicked one.
   * - Drag from a page
     - Selects every page the rectangle from the press to the pointer touches,
       as a click on the first page would. When that click removes the page, the
       drag removes the pages it touches instead.
   * - Ctrl+drag over empty space
     - Adds every page the rectangle touches. The selection grows and is never
       replaced.

Removing the current page from the selection makes the nearest selected page
current, looking before it first. Going to another page with the navigation
buttons or keys selects only that page and turns the multi page selection button
off. When several pages are selected, ScanTailor first asks whether to cancel the
selection; **Tools > Settings** can turn the question off.

Insert, remove and add pages
----------------------------

Right-click a thumbnail to insert or remove pages. Right-click the empty space
after the last page to add pages at the end.

.. screenshot:: thumbnail_menu
   :alt: The menu of a page thumbnail

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
     - Removes the selected pages after a confirmation. The program never
       deletes input files.

.. screenshot:: remove_pages
   :alt: The removal confirmation with Remove output files too ticked

The removal confirmation has a **Remove output files too** box, which is ticked
when the confirmation opens. If it is ticked, the program also deletes the output
files written for the removed pages. Untick it to keep them.

When you insert or add files, the program skips images already in the project.
**File > Add Files ...** appends the chosen images to the end of the project.

You can insert and remove pages only while no batch runs.

.. _reverse-spread-order:

Reversing the order of two-page spreads
---------------------------------------

.. screenshot:: tools_menu
   :alt: The Tools menu with Reverse two-page spread order

**Tools > Reverse two-page spread order** swaps the two halves of a split scan
between left-to-right and right-to-left reading. The halves change place in the
page list, and the program renames the output files already written for them to
match. The project file stores the setting.

Use it when the first half of a spread was numbered on the wrong side. For
example, the book is read right to left but the project was made as left to
right.

.. _relinking:

Missing folders and relinking
-----------------------------

.. screenshot:: relinking
   :alt: The Relinking dialog with the folder of the sample pages selected

A project stores the paths of its input files and its output folder. If one of
those folders has moved, the **Relinking** dialog opens when you open the
project. Cancelling it cancels the opening. You can also open it at any time from
**Tools > Relinking ...**.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part of the dialog
     - What it does
   * - Folder list
     - Lists the folders to relink: the folders of the input files and the output
       folder. When the dialog opens with the project, it lists only the missing
       ones.
   * - Path display
     - Shows the path of the selected folder in parts. Click a part to choose the
       folder that replaces everything up to and including that part.
   * - Undo
     - Reverts the last change.
   * - Error line
     - Explains why a change was refused. The dialog refuses a change that would
       merge the files of two folders into one.

Relinking one folder also relinks every listed folder below it. Press OK to apply
the changes. The project is then marked as changed, so save it to keep the new
paths.

Pages that cannot be loaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the input file of the current page cannot be read, a warning takes the
place of the page and its options, at every stage. It names the file.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Cause
     - What the warning offers
   * - The file is missing, for example after it was moved or renamed
     - A **Relinking Tool** link that opens the Relinking dialog. Once you relink
       the folder, the page loads.
   * - The file exists but cannot be decoded, for example a damaged or
       unsupported file
     - No link, since relinking cannot help. Replace the file, or remove the page
       from the project.

Going to a page that loads shows its view again.

Replaced files
~~~~~~~~~~~~~~

You can replace a scan with another file of the same name, for example a rescan.
When you next go to one of its pages, the project takes the new image's size in
pixels and is marked as changed, so save it to keep the size. The resolution stays
as the project holds it, since it may be one you set with Fix DPI.

See :mod:`scantailor.core.relinker`.
