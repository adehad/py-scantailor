Settings and default parameters
###############################

**Tools > Settings ...** and **Tools > Default parameters ...** hold the options
for the application and for new projects. Units are also under **Tools > Units**,
which sets the same value as the Settings dialog.

Settings
--------

.. screenshot:: settings
   :alt: The Settings dialog

Changes take effect when you press OK. Cancel restores the colour scheme that was
in use before the dialog opened. Some options are greyed out. This build does not
act on them, so they keep their stored values.

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Option
     - What it does
     - When to use it
   * - Accelerate user interface with OpenGL
     - Greyed out. This build has no OpenGL drawing path.
     - Not available.
   * - Auto-save the existing project
     - Saves the open project to its file about a minute after the first change since
       the last save. A project that has never been saved has no file, so nothing is written. A save due during
       a batch waits until the batch ends.
     - Turn on to keep the project file up to date while you work.
   * - Color scheme
     - Auto follows the operating system. Light and Dark fix the scheme. The
       change shows at once.
     - Light or Dark to fix one scheme. Auto to follow the system.
   * - Units
     - The unit for every length on screen: pixels, millimetres, centimetres or
       inches. Same as **Tools > Units**.
     - The unit you read margins and sizes in.
   * - Language
     - Greyed out in this version.
     - Not available.
   * - Default zone creation mode
     - The shape of a new zone in the Output view: Polygonal, Lasso (freehand) or
       Rectangular. Keys Z, X and C switch it while editing.
     - Lasso for freehand outlines, Rectangular for boxes.
   * - Thumbnails: Quality
     - Pixel width at which each thumbnail picture is made. Higher is sharper and
       takes longer to draw.
     - Raise it only if the thumbnails look soft.
   * - Thumbnails: Size
     - Largest width of a thumbnail on screen, in pixels. The height follows a
       16:25 ratio. Ctrl++ and Ctrl+- change it in steps.
     - Smaller sizes fit more pages on screen.
   * - Single column thumbnail display
     - One column of thumbnails instead of several. The same as the column button
       in the thumbnail column.
     - For a narrow window or a large page preview.
   * - Show question on canceling multi page selection
     - Stored, but this version does not ask the question.
     - Not available.
   * - Saving: B&W compression
     - TIFF compression of black and white output: None, LZW, Deflate or CCITT G4.
     - CCITT G4 is the usual choice for black and white pages.
   * - Saving: Color compression
     - TIFF compression of grey and colour output: None, LZW, Deflate or JPEG.
       JPEG is lossy. The others keep every pixel.
     - LZW or Deflate for lossless files. JPEG only to save space.
   * - Auto detect light content on dark background
     - Greyed out in this version.
     - Not available.
   * - Use auto detection at the output stage
     - Greyed out in this version.
     - Not available.
   * - Highlight the thumbnails of pages with high deviation
     - Greyed out here. The **Highlight deviation** button in the thumbnail column
       works and stores the same setting.
     - Use the button.
   * - Deskew, Select content and Margins: coefficient and threshold
     - Greyed out. The values are stored but no stage reads them yet.
     - Not available.

Default parameters
------------------

.. screenshot:: default_parameters
   :alt: The Default parameters dialog

The Default parameters dialog sets the values that a new project starts from. It
has the same sections as the stage options, so each stage's defaults can be set
in one place. The units of the margins and sizes follow the Units setting.

How the defaults apply
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Case
     - What happens
   * - New project
     - The selected profile is applied to every page when the project is created.
   * - Pages inserted later
     - The same profile is applied to them as they are inserted.
   * - Pages opened from a PDF
     - The same profile is applied, and the resolution chosen in the Open PDF
       dialog becomes the output resolution. See :doc:`export_pdf`.
   * - Open project
     - Not changed. A saved project keeps the values it was saved with.

A default that equals the factory value is not written to the new project. That
stage then keeps detecting by itself. Deskew is set only when its mode is Manual,
because an automatic entry would be trusted instead of measured.

Profiles
~~~~~~~~

The profile box at the top of the dialog chooses which set of defaults is used.

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Profile
     - What it is
     - Can it be changed?
   * - Default
     - The factory values.
     - Read only.
   * - Source
     - Leaves the scan as it is: pages are not split, deskewed, cropped or given
       margins, and the output stays grayscale.
     - Read only.
   * - A saved profile
     - Your own set of values, stored under the name you gave it. Save writes
       changes to it. Delete removes it.
     - Yes.
   * - Custom
     - Working values that are not saved yet. Type a name in the box and press
       Save to keep them as a profile.
     - Yes.

When you press OK, the selected profile becomes the one new projects use. Edits to
a saved profile apply to new projects straight away, but the saved profile itself
changes only when you press Save. See :mod:`scantailor.core.default_params` and
:mod:`scantailor.core.default_params_profiles`.
