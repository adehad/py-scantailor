The six stages
##############

Each stage has its own options panel under the stage list. Select a page in the
thumbnail strip to see the result of the current stage for that page. The stages run
in this order, and each one works on the result of the stage before it.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - What it decides
   * - :doc:`Fix Orientation <fix_orientation>`
     - The rotation of each page and an optional trim of the scanner's border.
   * - :doc:`Split Pages <split_pages>`
     - Whether a scan holds one page or two, and where the split line lies.
   * - :doc:`Deskew <deskew>`
     - The rotation that straightens the text, and an optional oblique correction.
   * - :doc:`Select Content <select_content>`
     - The content box around the text and pictures, and the page box.
   * - :doc:`Margins <margins>`
     - The white space around the content, and the size the pages are matched to.
   * - :doc:`Output <output>`
     - The output resolution, the binarization, despeckling, zones and dewarping.

.. toctree::
   :maxdepth: 1

   fix_orientation
   split_pages
   deskew
   select_content
   margins
   output
