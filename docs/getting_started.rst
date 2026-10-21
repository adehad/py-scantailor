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
       its full path. When the panel is shown, it drops the projects whose files
       no longer exist.

The panel closes once a project is open. The **File** menu has the same commands,
plus **Open Recent** for the same list.

When you close the main window, ScanTailor remembers its size, its position and
whether it was maximised, and opens the same way at the next start. If no screen
shows the saved position any more, for example after you unplug a monitor, the
window opens at its default size in the middle of the main screen.

Creating a project
------------------

.. screenshot:: new_project
   :alt: The New Project dialog with four pages added

Choose **File > New Project ...** (Ctrl+N). If a project is open, the program
asks whether to save it and closes it before the dialog appears. If you then
cancel the dialog, the old project stays closed.

The New Project dialog has these controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - What it does
   * - Input Directory
     - The folder that holds your scans. **Browse** chooses it. Changing it
       reads the folder again.
   * - Output Directory
     - The folder for the processed TIFF files. If you leave it empty, the output
       goes to an ``out`` folder inside the input folder.
   * - Files Not In Project
     - Image files and PDFs of the input folder that are not in the project yet.
       Supported image types are PNG, JPEG, TIFF, BMP and GIF. PDFs come first,
       with their page count. **Select All** selects every file.
   * - Add PDF...
     - Adds a PDF from any folder to Files Not In Project. See
       :ref:`pdf-in-new-project`.
   * - Right arrow button
     - Adds the selected files to the project. For a selected PDF, a prompt
       first asks which pages to extract.
   * - Left arrow button
     - Removes the selected files from the project.
   * - Files In Project
     - The files the project will hold, in natural file name order. The files
       can come from several folders. When two names are the same, the folder
       follows the name in brackets.
   * - Right to left layout (for Hebrew and Arabic)
     - Sets the reading direction of the book. It decides how the two halves of a
       split scan are numbered and which half comes first in a two-page spread.
       See :ref:`reverse-spread-order`.
   * - Fix DPIs, even if they look OK
     - Present in the dialog but not used in this version. The Fix DPI dialog
       opens only for images that have no usable resolution.

To make a project:

#. Choose the input folder. Its images appear under Files Not In Project.
#. Select the scans and press the right arrow button. They move to Files In
   Project.
#. Choose an output folder, or leave it empty for the default.
#. Tick the right to left box if the book reads that way.
#. Press OK.

A multi-page TIFF adds each of its pages to the project. If some files cannot be
read, a status window lists them with the files that loaded. You can continue
without the failed files or cancel the whole import.

If an image has no usable resolution, the Fix DPI dialog opens before the
program creates the project. See :ref:`fix-dpi`.

The first page opens when the project is created. The program writes the project
to disk only when you save it with **File > Save Project** (Ctrl+S). It never
changes the scan files and writes every result to the output folder.

.. _fix-dpi:

Fixing resolution (DPI)
-----------------------

The program needs the resolution of a scan to lay the page out at its real size.
Some image files do not store a resolution, or store 0 or 1. The Fix DPI dialog
lists those images and asks for a value.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How the dialog opens
     - What cancelling it does
   * - By itself, when you create a project, insert pages or open a project file
       and an image has no usable resolution
     - Cancels the creation, insertion or opening.
   * - From **Tools > Fix DPI ...**, for every page of the open project
     - Closes the dialog and keeps the resolutions the images had.

.. screenshot:: fix_dpi
   :alt: The Fix DPI dialog on its All Pages tab with one page selected

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part of the dialog
     - What it does
   * - Need Fixing
     - Lists the images whose resolution is undefined. These are the images you
       must fix.
   * - All Pages
     - Lists every image, with its current resolution.
   * - DPI preset
     - Custom, 300, 400, 600 or 1200. A preset fills the x and y fields.
   * - x and y fields
     - The horizontal and vertical resolution. Both must be whole numbers.
   * - Apply
     - Gives the selected images the typed resolution. It is enabled only when
       the value differs from the current one and is defined for every selected
       image.
   * - OK
     - Enabled once no image has an undefined resolution.

The dialog accepts a defined value that looks wrong. The field turns red, and its
tooltip gives the reason:

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

The status bar shows the resolution of the current page's image, for example
``300 dpi``. When most pages of the project have another resolution, theirs
follows in brackets, as in ``600 dpi (most pages 300 dpi)``, so a page scanned at
a different setting stands out.

The stage list
--------------

The stage list is beside the image. It shows six stages in order: Fix
Orientation, Split Pages, Deskew, Select Content, Margins and Output. Each stage
has its own options under the list and starts from the result of the stage
before it. See :doc:`stages/index`.

Click the title row of an options group, or press Space when it has keyboard
focus, to fold the group down to its title. Each group stays folded or open
the next time you start the program.

Opening a project
-----------------

Choose **File > Open Project ...**, or pick a name from the welcome panel or
**File > Open Recent**. The program checks the folders and resolutions first.
Only then does it offer to save the open project and close it. Two problems
interrupt the opening:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Problem
     - Dialog that opens
     - If you cancel it
   * - An image folder or the output folder no longer exists
     - Relinking, to point the folders at their new places. See
       :ref:`relinking`.
     - The project does not open.
   * - An image has no usable resolution
     - Fix DPI
     - The project does not open.

.. _open-project-argument:

Opening a project when the program starts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give the project file as the first argument, and the window opens it once it
appears:

.. code-block:: console

   ScanTailor book.ScanTailor

A relative path is read from the folder you start the program in. Arguments
after the first are ignored. Qt's own options, such as ``-style fusion``, may
come before the project file. If the file is missing or is not a project, the
program shows the same warning as **File > Open Project ...** and stays on the
welcome panel.

How you can open a project by double-clicking it depends on the build:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Build
     - Opening a ``.ScanTailor`` file from the file manager
   * - Windows
     - Choose **Open with** on the file and pick ``ScanTailor.exe``. Windows
       passes the file as the first argument.
   * - Linux AppImage
     - The AppImage declares the ``application/x-scantailor-project`` file type.
       Once a desktop integration tool such as AppImageLauncher or Gear Lever
       installs it, ``.ScanTailor`` files open in the program.
   * - macOS
     - Not supported yet: the Finder sends the file as an open event, not as an
       argument. Start the program from a terminal with
       ``ScanTailor.app/Contents/MacOS/ScanTailor book.ScanTailor``.

The command line tool ``scantailor-cli`` is separate and never opens a window.

Getting help in the program
---------------------------

Every option in the stage panels has a short explanation of what it does and
when to change it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Where
     - What it does
   * - Info icon beside an option
     - Hover over the icon, or over the option itself, to read its explanation.
       Click the icon, or reach it with Tab and press Space, to show the same
       text without hovering. The icon only appears for options that have an
       explanation, and grows with the interface font.
   * - **Help > User guide** (F1)
     - Opens this guide in your web browser.
