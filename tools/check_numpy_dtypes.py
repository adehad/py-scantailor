"""Flag NumPy expressions that build a wider temporary than their result needs.

NumPy takes a result's dtype from its array operands. Where every operand is a
Python number there is no array to take it from, so the result is int64 or
float64: 8 bytes a pixel for what is usually a 1-byte image.

Rules:
    NPD001: ``np.where``, ``np.select`` or ``np.full`` (without ``dtype=``) whose
        values are all Python ints. Anchor one with a NumPy scalar such as
        ``np.uint8(0)``.
    NPD002: ``.astype(np.uint8)`` (or int8) on an expression that widened first,
        through ``.astype(<wide>)``, ``np.asarray(..., dtype=<wide>)``, a true
        division, or a name assigned from one of those in the same function. The
        wide intermediate is most of the cost.
    NPD003: ``.sum(axis=(...))`` or ``np.sum(..., axis=(...))`` without
        ``dtype=``: a uint8 or bool sum accumulates and returns 64-bit values.

Suppress a finding with ``# npd: ok <reason>`` on the flagged expression or on
a comment line just above it; the reason is required.

Usage:
    python tools/check_numpy_dtypes.py [PATH ...]

With no paths it checks the pixel-code packages under ``src/scantailor``.
"""

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_ROOTS = tuple(
    Path(__file__).resolve().parents[1] / "src" / "scantailor" / package
    for package in ("imageproc", "filters", "core", "app")
)
NARROW = frozenset({"uint8", "int8"})
WIDE = frozenset(
    {
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
        "int",
        "float",
        "intp",
    }
)
# Calls whose result has the dtype of their array arguments, so a wide argument
# makes a wide result.
_DTYPE_PRESERVING = frozenset(
    {"clip", "round", "rint", "floor", "ceil", "abs", "minimum", "maximum"}
)
_SUPPRESSION = re.compile(r"#\s*npd:\s*ok\b(?P<reason>.*)$")


@dataclass(frozen=True)
class Finding:
    """One flagged line.

    Attributes:
        path (Path): The file.
        line (int): 1-based line number.
        end_line (int): Last line of the flagged expression. A suppression may
            sit on any line of the expression or on a comment line just above.
        code (str): Rule id, such as ``NPD001``.
        message (str): What the rule found.
    """

    path: Path
    line: int
    end_line: int
    code: str
    message: str

    def __str__(self) -> str:
        """Format as ``path:line: CODE message``, as compilers print."""
        return f"{self.path.as_posix()}:{self.line}: {self.code} {self.message}"


def _dtype_name(node: ast.expr) -> str | None:
    """Return ``uint8`` for ``np.uint8``, ``uint8`` or ``"uint8"``; None otherwise."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_np(node: ast.expr, name: str) -> bool:
    """True for ``np.<name>`` or ``numpy.<name>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id in {"np", "numpy"}
    )


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in node.keywords if k.arg == name), None)


def _argument(node: ast.Call, name: str, position: int) -> ast.expr | None:
    """Return the argument passed as ``name=`` or at ``position``, if either."""
    found = _keyword(node, name)
    if found is None and len(node.args) > position:
        found = node.args[position]
    return found


def _is_python_int(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _module_int_constants(tree: ast.Module) -> dict[str, bool]:
    """Map each module-level ``NAME = value`` to whether the value is a Python int."""
    found: dict[str, bool] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        if isinstance(value, ast.UnaryOp):
            value = value.operand
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = _is_python_int(value)
    return found


class _Checker(ast.NodeVisitor):
    """Walk one module and collect findings."""

    def __init__(self, path: Path, constants: dict[str, bool]) -> None:
        self.path = path
        self.constants = constants
        self.findings: list[Finding] = []
        self.wide_names: list[set[str]] = [set()]

    def _python_int(self, node: ast.expr) -> bool:
        """True when ``node`` is a Python int: a literal, int arithmetic or a constant.

        A name that is not a module constant counts when it is UPPER_CASE, which
        in this code base is an imported int constant.
        """
        if isinstance(node, ast.UnaryOp):
            return self._python_int(node.operand)
        if isinstance(node, ast.BinOp):
            return self._python_int(node.left) and self._python_int(node.right)
        if _is_python_int(node):
            return True
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute) and not _is_np(node, node.attr):
            name = node.attr
        else:
            return False
        if name in self.constants:
            return self.constants[name]
        return name.lstrip("_").isupper()

    def _widens(self, node: ast.expr) -> bool:
        """True when ``node`` evaluates to a wider array than it started from."""
        match node:
            case ast.Name(id=name):
                return name in self.wide_names[-1]
            case ast.BinOp(op=ast.Div()):
                return True
            case ast.BinOp(left=left, right=right):
                return self._widens(left) or self._widens(right)
            case ast.UnaryOp(operand=operand):
                return self._widens(operand)
            case ast.Subscript(value=value):
                return self._widens(value)
            case ast.Call(func=ast.Attribute(attr="astype"), args=[dtype, *_]):
                return _dtype_name(dtype) in WIDE
            case ast.Call(func=func) if _is_np(func, "asarray") or _is_np(
                func, "array"
            ):
                dtype = _argument(node, "dtype", 1)
                return dtype is not None and _dtype_name(dtype) in WIDE
            case ast.Call(func=func, args=[_, *values]) if _is_np(func, "where"):
                return any(self._widens(value) for value in values)
            case ast.Call(func=ast.Attribute(attr=attr) as func, args=args) if (
                attr in _DTYPE_PRESERVING and _is_np(func, attr)
            ):
                return any(self._widens(arg) for arg in args)
        return False

    def _add(self, code: str, node: ast.expr, message: str) -> None:
        end_line = node.end_lineno or node.lineno
        self.findings.append(Finding(self.path, node.lineno, end_line, code, message))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.wide_names.append(set())
        self.generic_visit(node)
        self.wide_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.wide_names.append(set())
        self.generic_visit(node)
        self.wide_names.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        wide = self._widens(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if wide:
                    self.wide_names[-1].add(target.id)
                else:
                    self.wide_names[-1].discard(target.id)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if _is_np(func, "where") and len(node.args) == 3:
            if self._python_int(node.args[1]) and self._python_int(node.args[2]):
                self._add("NPD001", node, "np.where of Python ints builds int64")
        elif _is_np(func, "select") and len(node.args) >= 2:
            choices = node.args[1]
            default = node.args[2] if len(node.args) > 2 else _keyword(node, "default")
            if (
                isinstance(choices, ast.List | ast.Tuple)
                and all(self._python_int(choice) for choice in choices.elts)
                and (default is None or self._python_int(default))
            ):
                self._add("NPD001", node, "np.select of Python ints builds int64")
        elif _is_np(func, "full") and _keyword(node, "dtype") is None:
            if len(node.args) == 2 and self._python_int(node.args[1]):
                self._add("NPD001", node, "np.full of a Python int builds int64")
        elif isinstance(func, ast.Attribute) and func.attr == "astype" and node.args:
            if _dtype_name(node.args[0]) in NARROW and self._widens(func.value):
                self._add("NPD002", node, "narrowed after a wide intermediate")
        elif (
            _is_np(func, "sum")
            or (isinstance(func, ast.Attribute) and func.attr == "sum")
        ) and _keyword(node, "dtype") is None:
            axis = _argument(node, "axis", 1 if _is_np(func, "sum") else 0)
            if isinstance(axis, ast.Tuple):
                self._add("NPD003", node, "sum over axes without dtype= is 64-bit")
        self.generic_visit(node)


def _suppressed(finding: Finding, lines: list[str]) -> Finding | None:
    """Return None for a suppressed finding, or a finding for a bad suppression."""
    first = finding.line - 1
    if first > 0 and lines[first - 1].lstrip().startswith("#"):
        first -= 1
    spanned = lines[first : finding.end_line]
    match = next(filter(None, map(_SUPPRESSION.search, spanned)), None)
    if match is None:
        return finding
    if match["reason"].strip():
        return None
    return replace(finding, code="NPD000", message="npd: ok needs a reason")


def check_source(path: Path, text: str) -> list[Finding]:
    """Check one module's source.

    Args:
        path (Path): Reported in each finding.
        text (str): The module's source.

    Returns:
        list[Finding]: Findings not suppressed by ``# npd: ok <reason>``, plus an
            ``NPD000`` finding for each suppression without a reason.
    """
    tree = ast.parse(text, filename=str(path))
    checker = _Checker(path, _module_int_constants(tree))
    checker.visit(tree)
    lines = text.splitlines()
    kept = (_suppressed(finding, lines) for finding in checker.findings)
    return sorted(
        (finding for finding in kept if finding is not None),
        key=lambda finding: (finding.line, finding.code),
    )


def _python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            yield path


def main(argv: list[str] | None = None) -> int:
    """Check the given files or folders and print each finding.

    Args:
        argv (list[str] | None): Paths; the pixel-code packages when empty.

    Returns:
        int: 1 when anything was found, else 0.
    """
    args = sys.argv[1:] if argv is None else argv
    roots = [Path(arg) for arg in args] if args else list(DEFAULT_ROOTS)
    findings = [
        finding
        for path in _python_files(roots)
        for finding in check_source(path, path.read_text(encoding="utf-8"))
    ]
    for finding in findings:
        sys.stdout.write(f"{finding}\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
