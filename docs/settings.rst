Settings and default parameters
###############################

**Tools > Settings ...** holds the options of the application.
**Tools > Default parameters ...** holds the values new projects start from.
**Tools > Units** sets the same Units value as the Settings dialog.

Settings
--------

.. screenshot:: settings
   :alt: The Settings dialog

Changes take effect when you press OK. Cancel restores the colour scheme and
hover highlight that were in use before the dialog opened. Some options are
greyed out. This build does not act on them, so they keep their stored values.

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
     - Saves the open project to its file about a minute after the first change
       since the last save. A project that was never saved has no file, so
       nothing is written. A save that falls due during a batch waits until the
       batch ends.
     - Turn on to keep the project file up to date while you work.
   * - Color scheme
     - Auto follows the operating system. Light and Dark fix the scheme. The
       change shows at once.
     - Light or Dark to fix one scheme. Auto to follow the system.
   * - Hover highlight
     - The fill behind an options group title or a panel divider under the
       pointer. Neutral is a light grey. Accent colour uses the selection blue.
     - Accent colour if the grey is hard to see on your screen.
   * - Units
     - The unit for every length on screen: pixels, millimetres, centimetres or
       inches. Same as **Tools > Units**.
     - The unit you read margins and sizes in.
   * - Language
     - The language of the menus and dialogs. Pressing OK switches the running
       program to it, and it is used again at the next start. Until you choose one,
       ScanTailor follows the system language.
     - When the system language is not the one you want to work in.
   * - Default zone creation mode
     - The shape of a new zone in the Output view: Polygonal, Lasso (freehand) or
       Rectangular. Keys Z, X and C switch it while editing.
     - Lasso for freehand outlines, Rectangular for boxes.
   * - Thumbnails: Quality
     - Pixel width at which each thumbnail picture is made. Higher values are
       sharper and take longer to draw.
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
     - Deskew and Select Content check whether a page has light text on a dark
       background, and invert such a page before they detect on it. A page counts
       as light on dark when most of its area is darker than its own threshold;
       when a dark surround tips that count, only the areas with text are counted.
       On by default.
     - Leave it on. Turn it off only if a page with a large dark picture is taken
       for a dark page.
   * - Use auto detection at the output stage
     - Output also sets **Black on white mode** from the page, for pages where you
       have not set it yourself. Needs the option above.
     - Leave it on, so dark pages come out light on dark without setting each one.
   * - Highlight the thumbnails of pages with high deviation
     - Marks the thumbnails of pages that stand out from the others with a red
       asterisk. The same as the **Highlight deviation** button in the thumbnail
       column. See :ref:`pages-that-stand-out`.
     - Leave on to find misdetected pages in a long book.
   * - Deskew, Select content and Margins: coefficient and threshold
     - How far a page must lie from the median of all pages to be marked at that
       stage. The coefficient counts standard deviations. The threshold is a
       percentage of the median. A page is marked only when it is further out
       than both. See :ref:`pages-that-stand-out`.
     - Raise the coefficient if too many pages are marked, lower it if an odd
       page is missed.

Default parameters
------------------

.. screenshot:: default_parameters
   :alt: The Default parameters dialog

The Default parameters dialog sets the values that a new project starts from. It
has the same sections as the stage options, so you can set the defaults of every
stage in one place. The units of the margins and sizes follow the Units setting.

How the defaults apply
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Case
     - What happens
   * - New project
     - The program applies the selected profile to every page when it creates
       the project.
   * - Pages inserted later
     - The program applies the same profile to them as they are inserted.
   * - Pages rendered from a PDF
     - The program applies the same profile, and the resolution chosen in the
       **Extract pages** prompt becomes the output resolution. See
       :doc:`export_pdf`.
   * - Open project
     - Not changed. A saved project keeps the values it was saved with.

The program does not write a default that equals the factory value to the new
project. That stage then keeps detecting by itself. Deskew is set only when its
mode is Manual, because the stage would trust an automatic entry instead of
measuring the page.

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
     - Leaves the scan as it is: the pages are not split, deskewed, cropped or
       given margins, and the output uses the **Color / Grayscale** mode, so
       it is not turned into black and white.
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
a saved profile apply to new projects at once, but the stored profile changes
only when you press Save. See :mod:`scantailor.core.default_params` and
:mod:`scantailor.core.default_params_profiles`.
