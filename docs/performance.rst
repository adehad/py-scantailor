Performance notes
#################

These notes cover which libraries the image code uses, how fast it is against the
compiled kernels it replaced, and why there is no GPU build.

How pixel code is written
=========================

The application depends on NumPy, OpenCV, Pillow and the standard library for
its image work, and on nothing that needs a compiler at build time. NumPy is then
the only package that ships an OpenBLAS, and the same wheels run on Windows,
Linux and macOS. Page code uses these tools, in this order of preference:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - When it is used
   * - OpenCV
     - First choice. A compiled C routine such as morphology, blur, threshold,
       remap, distance transform or connected components runs faster than the
       same work in a Python loop.
   * - Vectorised NumPy
     - When no single OpenCV routine fits, but the work is whole-array
       arithmetic. A scan whose result depends on the previous pixel steps a
       whole row or band at a time.
   * - Plain Python
     - For short loops whose steps depend on each other, such as a snake moving
       one node at a time or a spline fit's small linear solve. These run on
       arrays of a few hundred values, so the per-step overhead stays small
       against the page.

How it compares with compiled kernels
=====================================

Earlier versions compiled the hottest loops with numba and called SciPy and
scikit-image for a few filters. Issue `#231
<https://github.com/adehad/py-scantailor/issues/231>`_ has the full timings of
the switch, measured on twelve real book scans of about 2100 x 3000 pixels, one
thread, auto detection in every stage. Every replacement gives the same output
as the code it replaced.

Whole page, as a share of the compiled version's time:

.. list-table::
   :header-rows: 1

   * - Output mode
     - 300 dpi
     - 600 dpi
     - 1200 dpi
   * - Black and white
     - about 0.9
     - 0.8 to 0.9
     - about 0.8
   * - Black and white, auto dewarping
     - about 1.0
     - 0.8 to 0.9
     - about 0.8
   * - Mixed
     - about 1.1
     - 0.8 to 0.9
     - about 0.8

These are rough. The machine was busier during the compiled run, so the
figures discount the speed-up that unchanged stages also showed.

Single kernels differ more. These are faster than the compiled loops were:

* despeckling, where OpenCV's distance transform now finds where components
  meet (about half the time)
* illumination normalisation, one ``cv2.divide`` per page (0.6 to 0.7 times)
* the limited flood fill and the largest whitespace search in Select Content
* every SciPy call that OpenCV replaced, such as the distance transforms
  (a quarter to half the time) and Savitzky-Golay smoothing (0.6 to 0.75 times)

These are still slower, because each step is a NumPy call of a few
microseconds over one row or one snake node:

.. list-table::
   :header-rows: 1

   * - Kernel
     - Times the compiled version
   * - Greyscale reconstruction (picture detection in Mixed mode)
     - about 2.4
   * - Text line snakes (dewarping)
     - 2.5 to 3.5
   * - Page edge snakes and dewarp maps
     - about 2
   * - Text line tracing (dewarping)
     - about 20, but under 0.05 s per page

The JBIG2 encoder used by PDF export lives in ``packages/jbig2``. One OpenCV
correlation gives every pixel its coding context, and the arithmetic coder is a
plain Python loop over runs of pixels that share a context and a colour, so a
long white stretch costs one step instead of one per pixel. A scanned black and
white page takes about 0.35 s to encode at 300 dpi and 1.1 s at 600 dpi, 1.5 and
1.1 times a compiled coder. Pages with fine dither patterns have few runs and
take up to five times as long as with a compiled coder.

Why there is no GPU build
=========================

numba can compile CUDA kernels for NVIDIA cards, but only for code written as
separate GPU kernels. Before the switch to NumPy, one hot loop was ported to
``numba-cuda`` and timed on 2100 x 3000 pixel pages against the same loop
compiled for the CPU, on a machine with 16 CPU threads and an NVIDIA RTX A2000:

.. list-table::
   :header-rows: 1

   * - Version
     - Time per page
   * - CPU, compiled with numba
     - 29 to 35 ms
   * - GPU, with ``numba-cuda`` and the copies to and from the card
     - 42 to 49 ms

A page at 300 to 600 DPI is a few megabytes, so copying it to the card and back
costs as much as the work itself. The release therefore ships one CPU build,
which runs on every machine, with or without a GPU. A GPU path would be worth
another look only for work that keeps many pages on the card at once.
