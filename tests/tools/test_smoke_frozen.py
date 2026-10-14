"""Argument handling of the frozen-build smoke test; the run itself needs a build."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture
def smoke_frozen(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load tools/smoke_frozen.py with its sibling sample_book importable."""
    monkeypatch.syspath_prepend(str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "smoke_frozen", _TOOLS / "smoke_frozen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_two_workers(smoke_frozen: ModuleType) -> None:
    args = smoke_frozen.parse_args(["dist/ScanTailor/scantailor-cli"])
    assert args.cli == Path("dist/ScanTailor/scantailor-cli")
    assert args.workers == 2


def test_parse_args_requires_the_executable(smoke_frozen: ModuleType) -> None:
    with pytest.raises(SystemExit):
        smoke_frozen.parse_args([])


def test_missing_executable_fails_with_a_message(
    smoke_frozen: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "scantailor-cli"
    assert smoke_frozen.main([str(missing)]) == 1
    assert "no executable" in capsys.readouterr().err
