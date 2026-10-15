Performance notes
#################

These notes record how the port's image code is written and why there is no GPU build.

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
       arithmetic. Scans whose result depends on the previous pixel are done a
       whole row or band at a time.
   * - Plain Python
     - For short loops whose steps depend on each other, such as a snake moving
       one node at a time or a spline fit's small linear solve. These run on
       arrays of a few hundred values, so the per-step overhead stays small
       against the page.

A few kernels are slower this way than a compiled loop would be. On generated
300 dpi pages, the dewarping auto model takes 0.8 to 1.2 s per page, about 1.5
times a compiled version; picture detection up to 2.6 times and the content box
search up to 1.45 times. Other stages got faster where OpenCV replaced a Python
or SciPy step, so a whole batch of the ten sample pages takes about as long as
with compiled kernels.

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
separate GPU kernels. One hot loop was ported to ``numba-cuda`` and timed on 2100
x 3000 pixel pages, against the same loop compiled for the CPU, on a machine
with 16 CPU threads and an NVIDIA RTX A2000:

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
