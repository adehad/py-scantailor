"""The kernel warm-up leaves every numba kernel compiled."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

# Called only from other kernels, so numba compiles them with their callers.
_INNER_KERNELS = {
    "_scan_row",
    "_grown",
    "_add_attraction_forces",
    "_arc_lengths",
    "_bending_energy",
    "_blend_weights",
    "_bubble_down",
    "_bubble_up",
    "_constrain_spline_point_to_line",
    "_control_points_force",
    "_curve_distance",
    "_elasticity_energy",
    "_energy_at",
    "_external_energy_at",
    "_frames",
    "_frenet_unit",
    "_g_blend",
    "_g_blend_d1",
    "_g_blend_d2",
    "_h_blend",
    "_h_blend_d1",
    "_h_blend_d2",
    "_has_loops",
    "_junction_points_force",
    "_linear_combination_at",
    "_linear_combination_for",
    "_local_approximant",
    "_model_vertices",
    "_node_energy",
    "_normal_movement",
    "_optimize",
    "_point_and_derivs",
    "_point_at",
    "_point_at_impl",
    "_project_scalar",
    "_segment_and_t",
    "_segment_control_points",
    "_sq_dist_approximant",
    "_tangent_movement",
    "_tension_params",
    "_thickness_adjustment",
}

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
