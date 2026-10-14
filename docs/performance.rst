Performance notes
#################

These notes record how the port's image code is written and why there is no GPU build.

How pixel code is written
=========================

The page image code uses these tools, in this order of preference:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - When it is used
   * - OpenCV or SciPy
     - First choice. A compiled C routine such as morphology, blur, threshold,
       remap or connected components runs faster than the same work in a Python
       loop.
   * - Vectorised NumPy
     - When no single C routine fits, but the work is whole-array arithmetic.
   * - numba ``njit``
     - Only where NumPy still leaves a clear gap, such as a loop over connected
       components or a scan whose result depends on the previous pixel. numba
       compiles these kernels for the CPU and caches them on disk, so later runs
       start fast.

numba's parallel loops, ``prange``, gave no gain in these kernels, because batch
processing already runs one page per worker process.

Why there is no GPU build
=========================

numba can compile CUDA kernels for NVIDIA cards, but only for code written as
separate GPU kernels. It does not move ``njit`` code to the GPU. One hot loop was
ported to ``numba-cuda`` and timed on 2100 x 3000 pixel pages, on a machine with
16 CPU threads and an NVIDIA RTX A2000:

.. list-table::
   :header-rows: 1

   * - Version
     - Time per page
   * - CPU, with OpenCV, NumPy and numba
     - 29 to 35 ms
   * - GPU, with ``numba-cuda`` and the copies to and from the card
     - 42 to 49 ms

A page at 300 to 600 DPI is a few megabytes, so copying it to the card and back
costs as much as the work itself. The release therefore ships one CPU build,
which runs on every machine, with or without a GPU. A GPU path would be worth
another look only for work that keeps many pages on the card at once.
