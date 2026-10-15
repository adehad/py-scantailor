"""Collect numba.testing without numba's own test suite.

It reaches the suite, directly and through the CUDA test helpers, only when
those tests are run.
"""

excludedimports = ["numba.cuda.testing", "numba.tests"]
