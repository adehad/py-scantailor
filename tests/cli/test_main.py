"""Exit codes and stdout/stderr discipline of scantailor-cli's main()."""

import json
from pathlib import Path
from typing import Any

import pytest

from scantailor.cli import main as cli_main
from scantailor.cli.errors import ExitCode
from scantailor.cli.main import main


def test_status_prints_json_on_stdout_and_exits_zero(
    project_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["status", str(project_path)])

    captured = capsys.readouterr()
    assert code == ExitCode.OK
    assert json.loads(captured.out)["page_count"] == 2
    assert captured.err == ""


def test_missing_project_exits_with_project_code_and_prints_no_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["status", str(tmp_path / "absent.json")])

    captured = capsys.readouterr()
    assert code == ExitCode.PROJECT == 3
    assert captured.out == ""
    assert "no project file" in captured.err


def test_bad_option_value_exits_with_option_code(
    project_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["options", "set", str(project_path), "output_color_mode", "nope"])

    assert code == ExitCode.OPTION
    assert "error:" in capsys.readouterr().err


def test_export_before_run_exits_with_nothing_to_export_code(
    project_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["export", "pdf", str(project_path), str(tmp_path / "b.pdf")])

    assert code == ExitCode.NOTHING_TO_EXPORT == 6
    assert capsys.readouterr().out == ""


def test_source_without_images_exits_with_input_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    code = main(["create", str(folder), str(tmp_path / "p.json")])

    assert code == ExitCode.INPUT == 4
    assert "no image files" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["explode"])

    assert raised.value.code == ExitCode.USAGE
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.slow
def test_run_exits_zero_and_prints_batch_report(
    project_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", str(project_path), "--workers", "1"])

    report = json.loads(capsys.readouterr().out)
    assert code == ExitCode.OK
    assert report["written"] == 2
    assert report["kept"] == 0
    assert report["failed"] == {}


@pytest.mark.slow
def test_second_run_keeps_the_unchanged_output_files(
    project_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", str(project_path), "--workers", "1"])
    capsys.readouterr()

    code = main(["run", str(project_path), "--workers", "1"])

    report = json.loads(capsys.readouterr().out)
    assert code == ExitCode.OK
    assert report["written"] == report["kept"] == 2


@pytest.mark.slow
def test_run_with_a_missing_source_exits_failed_and_names_the_page(
    project_path: Path, sample_scans: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (sample_scans / "page_02.png").unlink()

    code = main(["run", str(project_path), "--workers", "1"])

    report = json.loads(capsys.readouterr().out)
    assert code == ExitCode.FAILED == 1
    assert report["written"] == 1
    assert list(report["failed"]) == ["page_02.png"]


@pytest.mark.parametrize(
    ("argv", "wants_file"),
    [
        (["status", "p.json"], False),
        (["options", "show", "p.json"], False),
        (["run", "p.json"], True),
        (["export", "pdf", "p.json", "b.pdf"], True),
    ],
)
def test_only_commands_that_process_pages_write_a_log_file(
    no_process_diagnostics: list[dict[str, Any]], argv: list[str], wants_file: bool
) -> None:
    main(argv)

    assert no_process_diagnostics[-1]["kind"] == "cli"
    assert no_process_diagnostics[-1]["write_to_log_folder"] is wants_file


@pytest.mark.parametrize(("verbose", "level"), [(False, "WARNING"), (True, "INFO")])
def test_verbose_lowers_the_stderr_level(
    no_process_diagnostics: list[dict[str, Any]], verbose: bool, level: str
) -> None:
    main([*(["--verbose"] if verbose else []), "status", "p.json"])

    assert no_process_diagnostics[-1]["stderr_level"] == level


@pytest.mark.parametrize("command", [["status", "p.json"], ["run", "p.json"]])
def test_log_file_option_names_the_file_for_any_command(
    no_process_diagnostics: list[dict[str, Any]], tmp_path: Path, command: list[str]
) -> None:
    target = tmp_path / "ci.log"

    main(["--log-file", str(target), *command])

    assert no_process_diagnostics[-1]["log_file"] == target


def test_unexpected_failure_names_the_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(_args: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli_main, "_dispatch", crash)

    code = main(["run", str(tmp_path / "p.json")])

    err = capsys.readouterr().err
    assert code == ExitCode.FAILED
    assert f"see the log in {tmp_path}" in err
    assert "cli-" in err


def test_unexpected_failure_without_a_log_file_points_at_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(_args: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli_main, "_dispatch", crash)

    code = main(["status", "p.json"])

    assert code == ExitCode.FAILED
    assert "see the log on stderr" in capsys.readouterr().err
