"""The NumPy dtype check flags wide temporaries and honours reasoned suppressions."""

import importlib.util
import pathlib
import textwrap
from types import ModuleType

import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_numpy_dtypes.py"


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    """``tools/check_numpy_dtypes.py``, imported by path: ``tools/`` is no package."""
    spec = importlib.util.spec_from_file_location("check_numpy_dtypes", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _codes(checker: ModuleType, source: str) -> list[tuple[int, str]]:
    findings = checker.check_source(pathlib.Path("x.py"), textwrap.dedent(source))
    return [(finding.line, finding.code) for finding in findings]


@pytest.mark.parametrize(
    "line",
    [
        "np.where(mask, 0, 255)",
        "np.where(mask, _BLACK, _WHITE)",
        "np.where(mask, -1, 2 * 3)",
        "np.select([a, b], [0, 1], 255)",
        "np.select([a, b], [0, 1])",
        "np.full(shape, 255)",
    ],
)
def test_fills_of_python_ints_are_flagged(checker: ModuleType, line: str) -> None:
    """NPD001: every value is a Python int, so NumPy builds int64."""
    source = f"_BLACK = 0\n_WHITE = 255\nx = {line}\n"

    assert _codes(checker, source) == [(3, "NPD001")]


@pytest.mark.parametrize(
    "line",
    [
        "np.where(mask, np.uint8(0), np.uint8(255))",
        "np.where(mask, image, 0)",
        "np.where(mask, 0.5, 1)",
        "np.select([a], [image], 0)",
        "np.full(shape, 255, dtype=np.uint8)",
        "np.full(shape, np.uint8(255))",
    ],
)
def test_fills_anchored_by_an_array_or_dtype_pass(
    checker: ModuleType, line: str
) -> None:
    """A NumPy scalar, an array operand or ``dtype=`` fixes the result type."""
    assert _codes(checker, f"x = {line}\n") == []


@pytest.mark.parametrize(
    "body",
    [
        "return (image.astype(np.int32) * 3).astype(np.uint8)",
        "return np.clip(image / 2, 0, 255).astype(np.uint8)",
        "return np.asarray(image, dtype=np.int64).astype(np.uint8)",
        "return np.asarray(image, np.float64).astype(np.uint8)",
        "return np.array(image, np.int32).astype(np.uint8)",
        "return np.where(mask, image / 2, 0).astype(np.uint8)",
        "wide = image.astype(np.float64)\n    return np.round(wide).astype(np.uint8)",
    ],
)
def test_narrowing_after_a_widening_is_flagged(checker: ModuleType, body: str) -> None:
    """NPD002: the wide intermediate costs more than the narrow result."""
    source = f"def f(image):\n    {body}\n"

    assert [code for _, code in _codes(checker, source)] == ["NPD002"]


def test_a_wide_where_condition_does_not_make_its_narrow_result_wide(
    checker: ModuleType,
) -> None:
    """``np.where`` takes its dtype from the values it picks, not the condition."""
    source = """
        def f(image):
            wide = image / 2
            return np.where(wide, np.uint8(0), np.uint8(255)).astype(np.uint8)
    """

    assert _codes(checker, source) == []


def test_a_name_widened_in_another_function_is_not_flagged(
    checker: ModuleType,
) -> None:
    """Widened names are tracked per function, so a reused name starts clean."""
    source = """
        def f(image):
            values = image.astype(np.float64)
            return values

        def g(values):
            return values.astype(np.uint8)
    """

    assert _codes(checker, source) == []


def test_a_name_reassigned_to_a_narrow_value_is_not_flagged(
    checker: ModuleType,
) -> None:
    """Once a name holds a narrow array again, narrowing it is free."""
    source = """
        def f(image):
            values = image.astype(np.float64)
            values = image[1:]
            return values.astype(np.uint8)
    """

    assert _codes(checker, source) == []


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("blocks.sum(axis=(1, 3))", True),
        ("np.sum(blocks, axis=(1, 3))", True),
        ("blocks.sum((1, 3))", True),
        ("np.sum(blocks, (1, 3))", True),
        ("np.sum((a, b), 0)", False),
        ("blocks.sum(axis=(1, 3), dtype=np.uint8)", False),
        ("blocks.sum(axis=1)", False),
    ],
)
def test_tuple_axis_sums_need_a_dtype(
    checker: ModuleType, line: str, flagged: bool
) -> None:
    """NPD003: summing over several axes without ``dtype=`` accumulates in 64 bits."""
    expected = [(1, "NPD003")] if flagged else []

    assert _codes(checker, f"x = {line}\n") == expected


def test_a_suppression_with_a_reason_hides_the_finding(checker: ModuleType) -> None:
    """``# npd: ok <reason>`` on the flagged line silences it."""
    source = "x = np.where(mask, 0, 255)  # npd: ok a 256-entry table\n"

    assert _codes(checker, source) == []


def test_a_suppression_on_the_last_line_of_a_wrapped_call_counts(
    checker: ModuleType,
) -> None:
    """A formatter moves the comment to the closing line; it still applies."""
    source = "x = np.where(\n    mask, 0, 255\n)  # npd: ok a 256-entry table\n"

    assert _codes(checker, source) == []


def test_a_suppression_on_the_comment_line_above_counts(checker: ModuleType) -> None:
    """A comment line directly above the flagged line suppresses it too."""
    source = "# npd: ok a 256-entry table\nx = np.where(mask, 0, 255)\n"

    assert _codes(checker, source) == []


def test_a_suppression_two_lines_above_does_not_count(checker: ModuleType) -> None:
    """Only the comment line immediately above the finding applies."""
    source = "# npd: ok a 256-entry table\n\nx = np.where(mask, 0, 255)\n"

    assert _codes(checker, source) == [(3, "NPD001")]


def test_a_suppression_without_a_reason_is_itself_flagged(
    checker: ModuleType,
) -> None:
    """A bare ``# npd: ok`` is reported as NPD000 instead of the original rule."""
    source = "x = np.where(mask, 0, 255)  # npd: ok\n"

    assert _codes(checker, source) == [(1, "NPD000")]


def test_main_exits_non_zero_and_prints_each_finding(
    checker: ModuleType,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command line prints ``path:line: CODE`` and fails when anything is found."""
    bad = tmp_path / "bad.py"
    bad.write_text("x = np.where(mask, 0, 255)\n", encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text("x = 1\n", encoding="utf-8")

    assert checker.main([str(good)]) == 0
    assert checker.main([str(tmp_path)]) == 1
    assert capsys.readouterr().out == f"{bad.as_posix()}:1: NPD001 " + (
        "np.where of Python ints builds int64\n"
    )
