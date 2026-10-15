"""Report what a PyInstaller build of ScanTailor bundles, and why.

Reads PyInstaller's build output (the COLLECT, EXE and PYZ tables of contents,
the PYZ archives and the per-executable ``xref-<name>.html`` import graphs that
app.spec keeps) and the unpacked folder in dist/. Prints, per top-level package,
the bytes on disk, the bytes inside the executables' PYZ archives, and the import
chain from an entry script that pulled the package in.

Fails (exit 1) when:

- a package is bundled that tools/pyinstaller/allowed_packages.txt does not list,
  naming the package and its import chain;
- the unpacked folder exceeds this platform's budget in
  tools/pyinstaller/size_budget.json;
- the frozen-checks executable (built only with SCANTAILOR_BUILD_CHECKS=1) put
  anything into the folder besides itself, so the checked folder would differ
  from the one that ships.

It also lists modules bundled but never imported by the test suite
(tools/pyinstaller/used_modules.txt) as candidates for review; those never fail.

Usage:
    python tools/bundle_report.py [--workpath build/app] [--dist dist/ScanTailor]
"""

import argparse
import ast
import html
import json
import marshal
import re
import struct
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
PYINSTALLER_DIR: Final = REPO_ROOT / "tools" / "pyinstaller"
CHECKS_EXE_STEM: Final = "scantailor-checks"
SHIPPED_EXE_STEMS: Final = ("ScanTailor", "scantailor-cli")
_PYZ_HEADER: Final = struct.Struct("!4s4si")
_REVIEW_MIN_BYTES: Final = 20_000
_LARGEST_SHARED_LIBRARIES: Final = 10
_NODE = re.compile(r'<a name="([^"]+)"></a>')
_LINK = re.compile(r'<a href="#([^"]+)">')
# PyInstaller adds its runtime hooks because a package is already bundled, so
# a chain through one would hide the import that brought the package in.
_RUNTIME_HOOK: Final = "pyi_rth_"
SHARED_LIBRARIES: Final = "<shared libraries>"
METADATA: Final = "<metadata>"
EXECUTABLES: Final = "<executables>"
_NOT_PACKAGES: Final = frozenset({SHARED_LIBRARIES, METADATA, EXECUTABLES})


@dataclass
class Bundle:
    """What one build put into the folder, keyed by top-level package."""

    disk_bytes: Counter[str] = field(default_factory=Counter)
    pyz_bytes: Counter[str] = field(default_factory=Counter)
    modules: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    module_bytes: Counter[str] = field(default_factory=Counter)
    shared_library_bytes: Counter[str] = field(default_factory=Counter)
    graphs: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    checks_extras: list[str] = field(default_factory=list)

    @property
    def packages(self) -> set[str]:
        """Top-level packages, without the size-only groups."""
        found = set(self.disk_bytes) | set(self.pyz_bytes)
        return found - _NOT_PACKAGES


def load_toc(path: Path) -> Any:
    """Read a PyInstaller .toc file, which is a Python literal.

    Args:
        path (Path): The file in the work folder.

    Returns:
        Any: The tuple PyInstaller saved.
    """
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def pyz_entries(path: Path) -> dict[str, tuple[int, int, int]]:
    """Read the table of contents of a PYZ archive.

    Args:
        path (Path): A PYZ-NN.pyz file from the work folder.

    Returns:
        dict[str, tuple[int, int, int]]: Module name to (type, offset,
        compressed length).
    """
    with path.open("rb") as stream:
        _magic, _pymagic, toc_offset = _PYZ_HEADER.unpack(stream.read(_PYZ_HEADER.size))
        stream.seek(toc_offset)
        return dict(marshal.load(stream))


def top_level(module: str) -> str:
    """Return the top-level package of a dotted module name."""
    return module.partition(".")[0]


def _extension_module(dest: PurePath) -> str:
    """Turn the bundled path of an extension module into its module name.

    ``PIL/_imaging.cp314-win_amd64.pyd`` becomes ``PIL._imaging``; the stdlib's
    ``python3.14/lib-dynload/_ssl.cpython-314-x86_64-linux-gnu.so`` becomes ``_ssl``.
    """
    parts = list(dest.parts)
    if "lib-dynload" in parts:
        parts = parts[parts.index("lib-dynload") + 1 :]
    parts[-1] = parts[-1].partition(".")[0]
    if parts == [parts[0]] * 2:
        # cv2/cv2.pyd is the native half of cv2, loaded under the package's name.
        return parts[0]
    return ".".join(parts)


def package_of(dest: PurePath, typecode: str) -> str:
    """Name the package a collected file belongs to.

    Args:
        dest (PurePath): The file's path inside the bundle's _internal folder.
        typecode (str): PyInstaller's kind of entry (EXTENSION, BINARY, DATA...).

    Returns:
        str: A top-level import name, or one of the size-only groups.
    """
    if typecode == "EXTENSION":
        return top_level(_extension_module(dest))
    head = dest.parts[0]
    if head.endswith((".dist-info", ".egg-info")):
        return METADATA
    if len(dest.parts) == 1:
        return SHARED_LIBRARIES if typecode == "BINARY" else head
    if head.endswith(".libs"):
        return _import_name(head.removesuffix(".libs"))
    return head


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@cache
def _import_name(distribution: str) -> str:
    """Name the package a wheel's ``<distribution>.libs`` folder serves.

    auditwheel and delvewheel name the folder after the distribution
    (``opencv_python_headless.libs``), not the import name (``cv2``). The
    installed metadata maps one to the other; without it the folder keeps the
    distribution's name.
    """
    wanted = _normalise(distribution)
    names = sorted(
        module
        for module, owners in metadata.packages_distributions().items()
        if "/" not in module and any(_normalise(owner) == wanted for owner in owners)
    )
    if distribution in names or not names:
        return distribution
    return names[0]


def _node_imports(document: str) -> dict[str, list[str]]:
    """Parse a PyInstaller xref HTML file into module -> imported modules."""
    graph: dict[str, list[str]] = {}
    nodes = list(_NODE.finditer(document))
    for index, match in enumerate(nodes):
        end = nodes[index + 1].start() if index + 1 < len(nodes) else len(document)
        body = document[match.end() : end]
        imports, _, _imported_by = body.partition("imported by:")
        _, _, imports = imports.partition("imports:")
        graph[html.unescape(match.group(1))] = [
            html.unescape(link) for link in _LINK.findall(imports)
        ]
    return graph


def read_graphs(workpath: Path) -> dict[str, dict[str, list[str]]]:
    """Read every executable's import graph that app.spec kept.

    Args:
        workpath (Path): PyInstaller's work folder (build/app).

    Returns:
        dict[str, dict[str, list[str]]]: Executable name to its import graph.
    """
    return {
        path.stem.removeprefix("xref-"): _node_imports(path.read_text(encoding="utf-8"))
        for path in sorted(workpath.glob("xref-*.html"))
        if path.stem != f"xref-{workpath.name}"
    }


def _is_script(name: str) -> bool:
    return name.endswith(".py") and not name.startswith(_RUNTIME_HOOK)


def import_chain(graph: dict[str, list[str]], package: str) -> list[str] | None:
    """Find the shortest import chain from an entry script into ``package``.

    Args:
        graph (dict[str, list[str]]): One executable's import graph.
        package (str): Top-level package to reach.

    Returns:
        list[str] | None: Module names from the entry script to the first module
        of the package, or None when only a runtime hook or a hook's binary
        collection reaches it.
    """
    starts = [name for name in graph if _is_script(name)]
    parents: dict[str, str | None] = dict.fromkeys(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        if top_level(node) == package and not node.endswith(".py"):
            chain = [node]
            while (parent := parents[chain[-1]]) is not None:
                chain.append(parent)
            return chain[::-1]
        for child in graph.get(node, ()):
            if child not in parents and not child.startswith(_RUNTIME_HOOK):
                parents[child] = node
                queue.append(child)
    return None


def _exe_pyz(workpath: Path) -> dict[str, Path]:
    """Map each executable's name to the PYZ archive embedded in it."""
    found = {}
    for exe_toc in sorted(workpath.glob("EXE-*.toc")):
        text = exe_toc.read_text(encoding="utf-8")
        name = PurePath(load_toc(exe_toc)[0].replace("\\", "/")).stem
        pyz = re.search(r"PYZ-\d+\.pyz", text)
        if pyz:
            found[name] = workpath / pyz.group(0)
    return found


def read_bundle(workpath: Path) -> Bundle:
    """Collect sizes, modules and graphs from a finished build.

    Args:
        workpath (Path): PyInstaller's work folder (build/app).

    Returns:
        Bundle: Everything the report prints.
    """
    bundle = Bundle(graphs=read_graphs(workpath))
    collect = load_toc(workpath / "COLLECT-00.toc")[0]
    for dest_name, source, typecode in collect:
        dest = PurePath(dest_name.replace("\\", "/"))
        if typecode == "SYMLINK":
            continue
        if typecode == "EXECUTABLE":
            if dest.stem != CHECKS_EXE_STEM:
                bundle.disk_bytes[EXECUTABLES] += Path(source).stat().st_size
            continue
        package = package_of(dest, typecode)
        size = Path(source).stat().st_size
        bundle.disk_bytes[package] += size
        if package == SHARED_LIBRARIES:
            bundle.shared_library_bytes[dest.name] += size
        if typecode == "EXTENSION":
            module = _extension_module(dest)
            bundle.modules[package].add(module)
            bundle.module_bytes[module] += Path(source).stat().st_size
    for exe, pyz in _exe_pyz(workpath).items():
        if exe == CHECKS_EXE_STEM:
            continue
        for name, (_kind, _offset, length) in pyz_entries(pyz).items():
            bundle.pyz_bytes[top_level(name)] += length
            bundle.modules[top_level(name)].add(name)
            bundle.module_bytes[name] = max(bundle.module_bytes[name], length)
    bundle.checks_extras = checks_extras(workpath, bundle)
    return bundle


def checks_extras(workpath: Path, bundle: Bundle) -> list[str]:
    """List what the frozen-checks build brings that the shipped executables lack.

    app.spec leaves the checks executable's own binaries and data out of the
    folder, so only Python modules can differ, and only stdlib ones are allowed.

    Args:
        workpath (Path): PyInstaller's work folder.
        bundle (Bundle): The shipped executables' modules.

    Returns:
        list[str]: Non-stdlib modules only the checks executable contains.
    """
    pyz = _exe_pyz(workpath).get(CHECKS_EXE_STEM)
    if pyz is None:
        return []
    shipped = set().union(*bundle.modules.values())
    return sorted(
        name
        for name in pyz_entries(pyz)
        if name not in shipped and top_level(name) not in sys.stdlib_module_names
    )


def read_allowlist(path: Path, platform: str) -> set[str]:
    """Read the allowed packages for one platform.

    Each line names a package; a line may end with platform names (``win32``,
    ``linux``, ``darwin``) to allow it there only. ``#`` starts a comment.

    Args:
        path (Path): The allowlist file.
        platform (str): ``sys.platform`` of the build.

    Returns:
        set[str]: Packages allowed on this platform.
    """
    allowed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.partition("#")[0].split()
        if words and (len(words) == 1 or platform in words[1:]):
            allowed.add(words[0])
    return allowed


def read_required(path: Path) -> set[str]:
    """Read the packages the allowlist expects on every platform.

    The stdlib is left out: an extension module that is a file on one platform
    is built into the interpreter on another.

    Args:
        path (Path): The allowlist file.

    Returns:
        set[str]: Non-stdlib packages listed without platform names.
    """
    required = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.partition("#")[0].split()
        if len(words) == 1 and words[0] not in sys.stdlib_module_names:
            required.add(words[0])
    return required


def read_used_modules(path: Path) -> set[str]:
    """Read the modules the test suite imported (tools/pyinstaller/used_modules.txt)."""
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def folder_bytes(dist: Path) -> int:
    """Sum the files in the unpacked folder, leaving out the checks executable."""
    return sum(
        path.stat().st_size
        for path in dist.rglob("*")
        if path.is_file() and not path.is_symlink() and path.stem != CHECKS_EXE_STEM
    )


def chain_text(bundle: Bundle, package: str) -> str:
    """Describe how ``package`` got into the bundle."""
    for exe, graph in bundle.graphs.items():
        if exe == CHECKS_EXE_STEM:
            continue
        chain = import_chain(graph, package)
        if chain:
            return f"{exe}: " + " -> ".join(chain)
    return "collected by a hook or as a library dependency (no import edge)"


def review_candidates(bundle: Bundle, used: set[str]) -> list[tuple[str, int, int]]:
    """Group bundled third-party modules the tests never imported.

    Modules are grouped by their first two name parts, so a whole unused
    subpackage shows as one line; a package the tests never touched shows as one
    line of its own. The stdlib is left out.

    Args:
        bundle (Bundle): The build.
        used (set[str]): Modules the test suite imported.

    Returns:
        list[tuple[str, int, int]]: (group, module count, PYZ bytes), largest first.
    """
    used_packages = {top_level(name) for name in used}
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for package, modules in bundle.modules.items():
        if package in sys.stdlib_module_names:
            continue
        for module in modules - used:
            parts = 2 if package in used_packages else 1
            group = ".".join(module.split(".")[:parts])
            counts[group] += 1
            sizes[group] += bundle.module_bytes[module]
    return sorted(
        ((group, counts[group], sizes[group]) for group in counts),
        key=lambda row: (-row[2], row[0]),
    )


def _mb(size: int) -> str:
    return f"{size / 1e6:7.1f}"


def print_table(bundle: Bundle, rows: Iterable[str]) -> None:
    """Print the per-package sizes and import chains."""
    print(f"{'package':<28}{'disk MB':>8}{'pyz MB':>8}  pulled in by")
    for package in rows:
        why = "" if package in _NOT_PACKAGES else chain_text(bundle, package)
        print(
            f"{package:<28}{_mb(bundle.disk_bytes[package]):>8}"
            f"{_mb(bundle.pyz_bytes[package]):>8}  {why}"
        )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workpath", type=Path, default=REPO_ROOT / "build" / "app")
    parser.add_argument("--dist", type=Path, default=REPO_ROOT / "dist" / "ScanTailor")
    parser.add_argument(
        "--allowlist", type=Path, default=PYINSTALLER_DIR / "allowed_packages.txt"
    )
    parser.add_argument(
        "--budget", type=Path, default=PYINSTALLER_DIR / "size_budget.json"
    )
    parser.add_argument(
        "--used", type=Path, default=PYINSTALLER_DIR / "used_modules.txt"
    )
    parser.add_argument("--platform", default=sys.platform)
    parser.add_argument("--json", type=Path, help="also write the sizes as JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the report and check it against the allowlist and budget.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: 0 when every check passed, 1 otherwise.
    """
    args = parse_args(argv)
    bundle = read_bundle(args.workpath)
    total = folder_bytes(args.dist)
    sizes = bundle.disk_bytes + bundle.pyz_bytes
    print_table(bundle, [name for name, _ in sizes.most_common()])
    print(f"\nunpacked folder: {_mb(total).strip()} MB ({args.dist})")
    print("\nlargest shared libraries outside any package:")
    for name, size in bundle.shared_library_bytes.most_common(
        _LARGEST_SHARED_LIBRARIES
    ):
        print(f"  {name:<52}{_mb(size)} MB")

    failures = []
    allowed = read_allowlist(args.allowlist, args.platform)
    for package in sorted(bundle.packages - allowed):
        if package in sys.stdlib_module_names and not bundle.disk_bytes[package]:
            continue
        failures.append(
            f"{package} is bundled but not in {args.allowlist.name}; "
            f"{chain_text(bundle, package)}"
        )
    failures.extend(
        f"{package} is in {args.allowlist.name} for every platform but not bundled"
        for package in sorted(read_required(args.allowlist) - bundle.packages)
    )
    budgets = json.loads(args.budget.read_text(encoding="utf-8"))
    budget_mb = budgets.get(args.platform)
    if budget_mb is None:
        failures.append(f"{args.budget.name} has no budget for {args.platform}")
    elif total > budget_mb * 1e6:
        failures.append(
            f"the folder is {_mb(total).strip()} MB, over the {budget_mb} MB budget "
            f"for {args.platform}; largest: "
            + ", ".join(
                f"{name} {_mb(size).strip()} MB" for name, size in sizes.most_common(3)
            )
        )
    failures.extend(
        f"the frozen-checks executable bundles {name}, which the shipped ones do not"
        for name in bundle.checks_extras
    )

    if args.used.is_file():
        print("\nbundled but never imported by the test suite (review, not enforced):")
        for group, count, size in review_candidates(
            bundle, read_used_modules(args.used)
        ):
            if size >= _REVIEW_MIN_BYTES:
                print(f"  {group:<40}{count:>5} modules {_mb(size)} MB")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "platform": args.platform,
                    "unpacked_bytes": total,
                    "packages": dict(sizes.most_common()),
                },
                indent=2,
            )
        )
    if failures:
        print("\nbundle report FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nbundle report passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
