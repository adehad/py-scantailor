Getting started
###############

The welcome screen
------------------

.. screenshot:: welcome
   :alt: The main window before a project is open

Start from **File > New Project ...**, or open an earlier project with
**File > Open Project ...**.

.. admonition:: TODO
   :class: attention

   Owner's words needed: what the welcome screen offers and what it says.

Creating a project
------------------

.. screenshot:: new_project
   :alt: The New Project dialog with four pages added

#. Choose the folder that holds your scans. Its images are listed on the left.
#. Select the pages to scan and add them to the project. They move to the list on
   the right.
#. Choose the output folder, where the processed pages are written.
#. Accept the dialog. The pages open at the first stage.

The input files stay untouched; every result goes to the output folder.

.. admonition:: TODO
   :class: attention

   Owner's words needed: how the project dialog treats images without a resolution (DPI), and the layout direction options.

Working through the stages
--------------------------

The six stages sit in the list at the top left. Work through them in order; each
one starts from the result of the one before. See :doc:`stages/index`.

First start
===========

The first start of a packaged build compiles the image processing kernels once and
keeps them in a cache folder (``%LOCALAPPDATA%
umba`` on Windows, ``~/.cache/numba``
on Linux and macOS), so the first pages take longer to process. The application does
this in the background after its window opens. From a source checkout,
``python -m scantailor.imageproc.numba_warmup`` does it ahead of time.
