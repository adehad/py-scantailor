Getting started
###############

The welcome screen
------------------

.. screenshot:: welcome
   :alt: The main window before a project is open

With no project open, the image area shows the welcome panel:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - What it does
   * - New Project ...
     - Opens the New Project dialog. Same as **File > New Project ...**.
   * - Open Project ...
     - Opens a file dialog for a project file, which ends in ``.ScanTailor``.
   * - Recent Projects
     - Up to seven projects, newest first, by file name. Hover over a name to see
       its full path. Projects whose files no longer exist are dropped from the
       list when the panel is shown.

The panel goes away once a project is open. The **File** menu holds the same
commands, plus **Open Recent** for the same list.

Creating a project
------------------

.. screenshot:: new_project
   :alt: The New Project dialog with four pages added

Choose **File > New Project ...** (Ctrl+N). If a project is open, you are asked
whether to save it first, and it is closed before the dialog appears. Cancelling
the dialog after that does not reopen the old project.

The New Project dialog has these controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - Input Directory
     - The folder that holds your scans. **Browse** picks it. Changing it
       re-reads the folder.
   * - Output Directory
     - Where the processed TIFF files are written. Left empty, the output goes to
       an ``out`` folder inside the input folder.
   * - Files Not In Project
     - Image files and PDFs of the input folder that are not in the project yet.
       Supported image types are PNG, JPEG, TIFF, BMP and GIF. PDFs come first,
       with their page count. **Select All** selects every file.
   * - Add PDF...
     - Adds a PDF from any folder to Files Not In Project. See
       :ref:`pdf-in-new-project`.
   * - Right arrow button
     - Adds the selected files to the project. A selected PDF first asks which
       pages to extract.
   * - Left arrow button
     - Removes the selected files from the project.
   * - Files In Project
     - The files the project will hold, in natural file name order. Files from
       several folders can be mixed. When two names are the same, the folder is
       shown after the name in brackets.
   * - Right to left layout (for Hebrew and Arabic)
     - Sets the reading direction of the book. It decides how the two halves of a
       split scan are numbered and which half comes first in a two-page spread.
       See :ref:`reverse-spread-order`.
   * - Fix DPIs, even if they look OK
     - Present in the dialog but not used in this version. The resolution check
       only opens for images that have no usable resolution.

To make a project:

#. Choose the input folder. Its images appear under Files Not In Project.
#. Select the scans and press the right arrow button. They move to Files In
   Project.
#. Choose an output folder, or leave it empty for the default.
#. Tick the right to left box if the book is read that way.
#. Press OK.

A multi-page TIFF adds each of its pages to the project. Files that cannot be
read are listed in a status window with the files that loaded. You can continue
without the failed files or cancel the whole import.

If an image has no usable resolution, the Fix DPI dialog opens before the
project is created. See :ref:`fix-dpi`.

The first page opens when the project is created. The project is not written to
disk until you save it with **File > Save Project** (Ctrl+S). The scan files are
never changed; every result goes to the output folder.

.. _fix-dpi:

Fixing resolution (DPI)
-----------------------

A scan needs a resolution to be laid out at its real size. Some image files do
not store one, or store zero or one. The Fix DPI dialog lists those images and
asks for a value. It opens:

- when a project is created, when pages are inserted, and when a project file
  is opened. Cancelling the dialog cancels the action;
- from **Tools > Fix DPI ...** for every page of the open project.

.. screenshot:: fix_dpi
   :alt: The Fix DPI dialog on its All Pages tab with one page selected

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part of the dialog
     - What it does
   * - Need Fixing
     - Lists the images whose resolution is undefined. This is the tab that
       needs work.
   * - All Pages
     - Lists every image, with the resolution each one has now.
   * - DPI preset
     - Custom, 300, 400, 600 or 1200. A preset fills the x and y fields.
   * - x and y fields
     - The horizontal and vertical resolution. Both must be whole numbers.
   * - Apply
     - Gives the selected images the typed resolution. It is enabled only when
       the value differs from the current one and is defined for every selected
       image.
   * - OK
     - Enabled once no image is left with an undefined resolution.

A value that is defined but doubtful is accepted, and its field turns red with
a tooltip that explains the doubt:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Warning
     - Condition
   * - Undefined
     - The resolution is 0 or 1. OK stays disabled until every image has a value.
   * - Too small
     - Below 150 DPI. Results at that resolution are unlikely to be usable.
   * - Too large
     - Above 9999 DPI. The value is most likely wrong.
   * - Too small for the pixel size
     - At this resolution the page would be more than 500 mm long along one side.
       Processing such a page can run out of memory.

The stage list
--------------

The stage list sits beside the image. It shows six stages in order: Fix
Orientation, Split Pages, Deskew, Select Content, Margins and Output. Each stage
has its own options under the list and starts from the result of the stage
before it. See :doc:`stages/index`.

Opening a project
-----------------

Choose **File > Open Project ...**, or pick a name from the welcome panel or
**File > Open Recent**. The folders and resolutions are checked first, and only
then is the open project offered for saving and closed. Two things can interrupt
the opening:

- Image or output folders that no longer exist. The relinking dialog opens so
  you can point them at their new places. Cancelling it cancels the opening.
  See :ref:`relinking`.
- Images without a usable resolution. The Fix DPI dialog opens. Cancelling it
  cancels the opening.

First start
-----------

The first start of a packaged build compiles the image processing kernels once
and keeps them in a cache folder: ``%LOCALAPPDATA%/numba`` on Windows and
``~/.cache/numba`` elsewhere. The application does this in the background after
its window opens, so the first pages can take longer to process. From a source
checkout, ``python -m scantailor.imageproc.numba_warmup`` does the same ahead of
time.
