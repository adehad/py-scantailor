"""The kernel warm-up leaves every numba kernel compiled."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

# Called only from other kernels, so numba compiles them with their callers.
_INNER_KERNELS = {"_scan_row", "_grown"}

_CHECK = textwrap.dedent(
    """
    import importlib, pkgutil
    import scantailor
    from numba.core.registry import CPUDispatcher

    modules = []
    for info in pkgutil.walk_packages(scantailor.__path__, "scantailor."):
        if ".app." in info.name or info.name.endswith(("main", "__main__", "_autogen")):
            continue
        modules.append(importlib.import_module(info.name))
    from scantailor.imageproc.numba_warmup import warm_up_kernels
    warm_up_kernels()
    for module in modules:
        for name, value in vars(module).items():
            if isinstance(value, CPUDispatcher) and value.__module__ == module.__name__:
                if not value.signatures:
                    print("UNCOMPILED", name)
    print("DONE")
    """
)


@pytest.mark.slow
def test_warming_up_compiles_every_kernel_that_pages_call_directly() -> None:
    """A packaged first run then pays no compile when the user opens a page."""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _CHECK],
        env={**os.environ, "PYTHONPATH": str(_SRC)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    lines = completed.stdout.splitlines()
    uncompiled = {line.split()[1] for line in lines if line.startswith("UNCOMPILED")}
    assert "DONE" in lines, completed.stderr
    assert uncompiled <= _INNER_KERNELS, uncompiled - _INNER_KERNELS
