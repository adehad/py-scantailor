"""Collect numba.misc without its debugging extras.

It imports these only for coverage reports, highlighted dumps and the disassembly
view, which builds with numba's ahead-of-time compiler (numba.pycc, itself built
on setuptools).
"""

excludedimports = ["coverage", "numba.pycc", "pygments"]
