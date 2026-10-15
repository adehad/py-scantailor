"""Parsing and rules of tools/bundle_report.py; the full report needs a build."""

import importlib.util
from pathlib import Path, PurePath
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture(scope="module")
def report() -> ModuleType:
    """Load tools/bundle_report.py by path."""
    spec = importlib.util.spec_from_file_location(
        "bundle_report", _TOOLS / "bundle_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("dest", "typecode", "package"),
    [
        ("PIL/_imaging.cp314-win_amd64.pyd", "EXTENSION", "PIL"),
        (
            "python3.14/lib-dynload/_ssl.cpython-314-x86_64-linux-gnu.so",
            "EXTENSION",
            "_ssl",
        ),
        ("_sqlite3.pyd", "EXTENSION", "_sqlite3"),
        ("numpy.libs/libscipy_openblas64_.dll", "BINARY", "numpy"),
        ("pillow.libs/libjpeg-1a2b3c.so.62", "BINARY", "PIL"),
        ("pillow-12.3.dist-info/METADATA", "DATA", "<metadata>"),
        ("python314.dll", "BINARY", "<shared libraries>"),
        ("base_library.zip", "DATA", "base_library.zip"),
        ("PySide6/plugins/platforms/qwindows.dll", "BINARY", "PySide6"),
    ],
)
def test_package_of_names_the_owning_package_of_each_bundled_file(
    report: ModuleType, dest: str, typecode: str, package: str
) -> None:
    assert report.package_of(PurePath(dest), typecode) == package


def test_extension_module_drops_the_platform_suffix_and_stdlib_folder(
    report: ModuleType,
) -> None:
    assert (
        report._extension_module(PurePath("scipy/linalg/_flapack.cp314-win_amd64.pyd"))
        == "scipy.linalg._flapack"
    )
    assert (
        report._extension_module(
            PurePath("python3.14/lib-dynload/_ctypes.cpython-314-darwin.so")
        )
        == "_ctypes"
    )


_XREF = """
<div class="node"><a name="gui.py"></a><tt>gui.py</tt>
<div class="import">imports:
 <a href="#scantailor.main">scantailor.main</a>
</div><div class="import">imported by:</div></div>
<div class="node"><a name="scantailor.main"></a>
<div class="import">imports:
 <a href="#numba">numba</a> &#8226; <a href="#os">os</a>
</div><div class="import">imported by:
 <a href="#gui.py">gui.py</a></div></div>
<div class="node"><a name="numba"></a>
<div class="import">imports:
 <a href="#coverage">coverage</a>
</div><div class="import">imported by:
 <a href="#scantailor.main">scantailor.main</a></div></div>
<div class="node"><a name="coverage"></a>
<div class="import">imported by:
 <a href="#numba">numba</a></div></div>
"""


def test_xref_parsing_keeps_imports_and_ignores_imported_by(
    report: ModuleType,
) -> None:
    graph = report._node_imports(_XREF)
    assert graph["scantailor.main"] == ["numba", "os"]
    assert graph["coverage"] == []


def test_import_chain_walks_from_the_entry_script_to_the_package(
    report: ModuleType,
) -> None:
    graph = report._node_imports(_XREF)
    assert report.import_chain(graph, "coverage") == [
        "gui.py",
        "scantailor.main",
        "numba",
        "coverage",
    ]
    assert report.import_chain(graph, "sqlite3") is None


def test_allowlist_platform_words_limit_a_package_to_those_platforms(
    report: ModuleType, tmp_path: Path
) -> None:
    allowlist = tmp_path / "allowed.txt"
    allowlist.write_text(
        "# comment\nnumpy\n_wmi win32  # Windows only\n_posixsubprocess linux darwin\n"
    )
    assert report.read_allowlist(allowlist, "win32") == {"numpy", "_wmi"}
    assert report.read_allowlist(allowlist, "linux") == {"numpy", "_posixsubprocess"}


def test_allowlist_lists_packages_expected_on_every_platform_separately(
    report: ModuleType, tmp_path: Path
) -> None:
    allowlist = tmp_path / "allowed.txt"
    allowlist.write_text("numpy\n_ssl\n_wmi win32\n")
    assert report.read_required(allowlist) == {"numpy"}


def test_review_candidates_group_by_subpackage_or_whole_untouched_package(
    report: ModuleType,
) -> None:
    bundle = report.Bundle()
    for name, size in (
        ("scipy", 10),
        ("scipy.ndimage", 10),
        ("scipy.stats", 100),
        ("scipy.stats._qmc", 200),
        ("lxml.objectify", 50),
    ):
        bundle.modules[name.partition(".")[0]].add(name)
        bundle.module_bytes[name] = size
    candidates = report.review_candidates(bundle, {"scipy", "scipy.ndimage"})
    assert candidates == [("scipy.stats", 2, 300), ("lxml", 1, 50)]
