"""Argument handling and log checks of the frozen-build smoke test."""

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


_BANNER = "2026-10-17 10:00:00.000 | INFO    | 1 MainThread | - | - | x:1 | "
_FROZEN = _BANNER + "pyScanTailor 1.3.0, frozen (PyInstaller)"
_FROM_SOURCE = _BANNER + "pyScanTailor 1.3.0, from source"
_ENDED = _BANNER + "session ended"


def _write_log(path: Path, *lines: str) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_log_file_check_accepts_a_frozen_session_that_ended(
    smoke_frozen: ModuleType, tmp_path: Path
) -> None:
    log = _write_log(tmp_path / "run.log", _FROZEN, _ENDED)

    smoke_frozen.check_log_file(log)


@pytest.mark.parametrize(
    ("lines", "complaint"),
    [((_FROM_SOURCE, _ENDED), "banner"), ((_FROZEN,), "session ended")],
)
def test_log_file_check_rejects_a_source_run_or_an_unfinished_session(
    smoke_frozen: ModuleType, tmp_path: Path, lines: tuple[str, ...], complaint: str
) -> None:
    log = _write_log(tmp_path / "run.log", *lines)

    with pytest.raises(smoke_frozen.SmokeFailure, match=complaint):
        smoke_frozen.check_log_file(log)


def test_log_file_check_rejects_a_missing_file(
    smoke_frozen: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(smoke_frozen.SmokeFailure, match="no log file"):
        smoke_frozen.check_log_file(tmp_path / "absent.log")
