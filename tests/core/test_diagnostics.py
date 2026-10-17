"""Log folder, session log, pruning, clean-exit marker and process hooks."""

import faulthandler
import gc
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from _pytest.faulthandler import fault_handler_stderr_fd_key
from loguru import logger
from PySide6 import QtCore

from scantailor.__about__ import APP_NAME
from scantailor.core import diagnostics

# Bound before the autouse no_process_diagnostics fixture replaces the module
# attribute, so these tests exercise the real function.
from scantailor.core.diagnostics import configure_logging
from tests.conftest import quiet_loguru


@pytest.fixture
def log_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the log folder at a fresh folder under tmp_path."""
    folder = tmp_path / "logs"
    monkeypatch.setattr(diagnostics, "log_folder", lambda: folder)
    return folder


@pytest.fixture
def restore_process_state(request: pytest.FixtureRequest) -> Iterator[None]:
    """Put back the hooks, faulthandler and loguru sinks pytest had."""
    excepthook = sys.excepthook
    unraisablehook = sys.unraisablehook
    root_handlers = list(logging.getLogger().handlers)
    yield
    sys.excepthook = excepthook
    sys.unraisablehook = unraisablehook
    logging.getLogger().handlers[:] = root_handlers
    logging.captureWarnings(False)
    faulthandler.enable(file=request.config.stash[fault_handler_stderr_fd_key])
    quiet_loguru()


@pytest.fixture
def gui_session(
    log_folder: Path, restore_process_state: None
) -> Iterator[diagnostics.DiagnosticsSession]:
    """A real gui session logging into the temporary log folder."""
    session = configure_logging("gui", stderr_level="CRITICAL")
    yield session
    session.end()


def _touch(path: Path, mtime: float, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _session_log(folder: Path, index: int, kind: str = "gui", text: str = "") -> Path:
    name = f"{kind}-20260101-{index:06d}-{1000 + index}.log"
    return _touch(folder / name, 1_000_000 + index, text or "a line\n")


ENDED = (
    "2026-01-01 00:00:00.000 | INFO    | 1 MainThread | - | - | x:1 | "
    f"{diagnostics.SESSION_ENDED}\n"
)


class TestLogFolder:
    def test_windows_uses_local_app_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        assert diagnostics.log_folder() == tmp_path / "ScanTailor" / "logs"

    def test_macos_uses_library_logs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert diagnostics.log_folder() == tmp_path / "Library" / "Logs" / "ScanTailor"

    def test_linux_uses_xdg_state_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path.absolute()))

        assert diagnostics.log_folder() == tmp_path / "scantailor" / "logs"

    @pytest.mark.parametrize("state_home", [None, "relative/state"])
    def test_linux_without_an_absolute_xdg_state_home_uses_local_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_home: str | None
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        if state_home is None:
            monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        else:
            monkeypatch.setenv("XDG_STATE_HOME", state_home)

        assert diagnostics.log_folder() == (
            tmp_path / ".local" / "state" / "scantailor" / "logs"
        )


def test_session_log_name_carries_kind_time_and_process_id():
    name = diagnostics.session_log_name("cli")

    assert diagnostics._SESSION_FILE.match(name)
    assert name.startswith("cli-")
    assert name.endswith(f"-{os.getpid()}.log")


class TestPruneLogs:
    def test_each_kind_keeps_its_newest_ten(self, tmp_path: Path):
        gui = [_session_log(tmp_path, index, "gui") for index in range(12)]
        cli = [_session_log(tmp_path, 100 + index, "cli") for index in range(12)]
        crash = [
            _touch(tmp_path / f"crash-gui-{index}.log", 2_000_000 + index)
            for index in range(3)
        ]

        deleted = diagnostics.prune_logs(tmp_path)

        assert sorted(deleted) == sorted(gui[:2] + cli[:2])
        assert all(path.exists() for path in gui[2:] + cli[2:] + crash)

    def test_many_cli_runs_do_not_evict_the_window_log(self, tmp_path: Path):
        window_log = _session_log(tmp_path, 0, "gui")
        for index in range(30):
            _session_log(tmp_path, 1 + index, "cli")

        diagnostics.prune_logs(tmp_path)

        assert window_log.exists()
        assert len(list(tmp_path.glob("cli-*.log"))) == 10

    def test_oldest_files_go_until_the_folder_fits_its_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(diagnostics, "_FOLDER_BUDGET_BYTES", 25)
        logs = [_session_log(tmp_path, index, text="0123456789") for index in range(4)]

        deleted = diagnostics.prune_logs(tmp_path)

        assert deleted == logs[:2]
        assert [path.exists() for path in logs] == [False, False, True, True]

    def test_a_file_that_cannot_be_deleted_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        logs = [_session_log(tmp_path, index) for index in range(12)]
        held_open = logs[0]
        real_unlink = Path.unlink

        def unlink(path: Path, missing_ok: bool = False) -> None:
            if path == held_open:
                raise PermissionError("held open by another instance")
            real_unlink(path, missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)

        deleted = diagnostics.prune_logs(tmp_path)

        assert deleted == [logs[1]]
        assert held_open.exists()

    def test_an_empty_folder_deletes_nothing(self, tmp_path: Path):
        assert diagnostics.prune_logs(tmp_path) == []


class TestCleanExitMarker:
    def test_a_log_ending_with_the_marker_ended_cleanly(self, tmp_path: Path):
        path = _session_log(tmp_path, 0, text="first\n" + ENDED)

        assert diagnostics.ended_cleanly(path)

    @pytest.mark.parametrize("text", ["first\nsecond\n", "\n", ENDED + "later\n"])
    def test_a_log_without_the_marker_last_did_not(self, tmp_path: Path, text: str):
        path = _session_log(tmp_path, 0, text=text)

        assert not diagnostics.ended_cleanly(path)

    def test_the_newest_finished_session_without_the_marker_is_reported(
        self, tmp_path: Path
    ):
        _session_log(tmp_path, 0, text=ENDED)
        crashed = _session_log(tmp_path, 1, text="working\n")

        assert diagnostics.previous_unclean_session(tmp_path, "gui") == crashed

    def test_an_older_crash_is_not_reported_after_a_clean_session(self, tmp_path: Path):
        _session_log(tmp_path, 0, text="working\n")
        _session_log(tmp_path, 1, text=ENDED)

        assert diagnostics.previous_unclean_session(tmp_path, "gui") is None

    def test_a_session_still_running_is_passed_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        crashed = _session_log(tmp_path, 0, text="working\n")
        running = _session_log(tmp_path, 1, text="working\n")
        monkeypatch.setattr(diagnostics, "_is_live", lambda path: path == running)

        assert diagnostics.previous_unclean_session(tmp_path, "gui") == crashed

    def test_own_log_rotated_files_and_other_kinds_are_ignored(self, tmp_path: Path):
        _session_log(tmp_path, 0, text=ENDED)
        own = _session_log(tmp_path, 1, text="working\n")
        _touch(tmp_path / f"{own.stem}.2026-01-01_00-00-00_000000.log", 2_000_000)
        _session_log(tmp_path, 2, "cli", text="working\n")

        assert diagnostics.previous_unclean_session(tmp_path, "gui", own) is None

    def test_a_file_held_open_by_its_process_is_live(self, tmp_path: Path):
        path = tmp_path / f"gui-20260101-000000-{os.getpid()}.log"
        with path.open("a", encoding="utf-8"):
            assert diagnostics._is_live(path)

    def test_a_closed_file_of_an_exited_process_is_not_live(self, tmp_path: Path):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        path = _touch(tmp_path / f"gui-20260101-000000-{child.pid}.log", 1_000_000)

        assert not diagnostics._is_live(path)


class TestConfigureLogging:
    def test_writes_the_banner_to_a_session_log_in_the_folder(
        self, gui_session: diagnostics.DiagnosticsSession, log_folder: Path
    ):
        assert gui_session.log_file is not None
        assert gui_session.log_file.parent == log_folder
        assert diagnostics._SESSION_FILE.match(gui_session.log_file.name)
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert f"| {APP_NAME} " in text
        assert "Python " in text
        assert f"/ Qt {QtCore.qVersion()};" in text
        assert f"log {gui_session.log_file}" in text

    def test_lines_carry_the_default_page_and_stage(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        logger.debug("a debug line reaches the file")

        assert gui_session.log_file is not None
        line = next(
            line
            for line in gui_session.log_file.read_text(encoding="utf-8").splitlines()
            if line.endswith("a debug line reaches the file")
        )
        assert "| DEBUG   |" in line
        assert "| - | -              |" in line

    def test_end_writes_the_marker_last_and_deletes_an_empty_crash_file(
        self, log_folder: Path, restore_process_state: None
    ):
        session = configure_logging("gui", stderr_level="CRITICAL")
        assert session.crash_file is not None and session.crash_file.exists()

        session.end()
        logger.info("after the session")

        assert session.log_file is not None
        assert diagnostics.ended_cleanly(session.log_file)
        assert not session.crash_file.exists()

    def test_a_previous_unclean_session_is_named_at_start_up(
        self, log_folder: Path, restore_process_state: None
    ):
        crashed = _session_log(log_folder, 0, text="working\n")

        session = configure_logging("gui", stderr_level="CRITICAL")
        session.end()

        assert session.log_file is not None
        text = session.log_file.read_text(encoding="utf-8")
        assert f"previous session did not end cleanly: {crashed.name}" in text

    def test_old_logs_are_pruned_at_start_up(
        self, log_folder: Path, restore_process_state: None
    ):
        old = [_session_log(log_folder, index, text=ENDED) for index in range(12)]

        session = configure_logging("gui", stderr_level="CRITICAL")
        session.end()

        assert [path.exists() for path in old[:2]] == [False, False]
        assert all(path.exists() for path in old[2:])

    def test_an_explicit_log_file_leaves_the_log_folder_alone(
        self, tmp_path: Path, log_folder: Path, restore_process_state: None
    ):
        target = tmp_path / "run" / "cli.log"

        session = configure_logging("cli", stderr_level="CRITICAL", log_file=target)
        session.end()

        assert session.log_file == target
        assert diagnostics.ended_cleanly(target)
        assert not log_folder.exists()

    def test_no_log_file_when_the_folder_is_not_wanted(
        self, log_folder: Path, restore_process_state: None
    ):
        session = configure_logging(
            "cli", stderr_level="CRITICAL", write_to_log_folder=False
        )
        session.end()

        assert session.log_file is None
        assert session.crash_file is None
        assert not log_folder.exists()

    def test_an_unwritable_log_folder_is_reported_and_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_process_state: None,
    ):
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("", encoding="utf-8")
        monkeypatch.setattr(diagnostics, "log_folder", lambda: blocker / "logs")

        session = configure_logging("gui", stderr_level="CRITICAL")
        session.end()

        assert session.log_file is None

    def test_works_without_a_stderr_as_in_the_window_build(
        self,
        log_folder: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_process_state: None,
    ):
        monkeypatch.setattr(sys, "stderr", None)

        session = configure_logging("gui", stderr_level="DEBUG")
        assert faulthandler.is_enabled()
        session.end()

        assert session.log_file is not None
        assert diagnostics.ended_cleanly(session.log_file)

    def test_tracebacks_leave_out_local_variables(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        def fail() -> None:
            secret_local = "do-not-print-me"
            raise ValueError(secret_local[:2])

        try:
            fail()
        except ValueError:
            logger.exception("failed")

        assert gui_session.log_file is not None
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert "ValueError: do" in text
        assert "do-not-print-me" not in text


class TestHooks:
    def test_an_uncaught_exception_is_logged_as_critical_with_its_traceback(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        try:
            raise RuntimeError("escaped a slot")
        except RuntimeError as error:
            sys.excepthook(type(error), error, error.__traceback__)

        assert gui_session.log_file is not None
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert "| CRITICAL |" in text
        assert "uncaught exception" in text
        assert "RuntimeError: escaped a slot" in text

    def test_keyboard_interrupt_goes_to_the_default_hook(
        self,
        gui_session: diagnostics.DiagnosticsSession,
        monkeypatch: pytest.MonkeyPatch,
    ):
        seen: list[type[BaseException]] = []
        monkeypatch.setattr(sys, "__excepthook__", lambda kind, *_: seen.append(kind))

        diagnostics.log_uncaught_exception(KeyboardInterrupt, KeyboardInterrupt(), None)

        assert seen == [KeyboardInterrupt]

    def test_an_exception_in_del_is_logged_as_a_warning(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        class Broken:
            def __del__(self) -> None:
                raise OSError("close failed in __del__")

        broken = Broken()
        del broken
        gc.collect()

        assert gui_session.log_file is not None
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert "| WARNING |" in text
        assert "OSError: close failed in __del__" in text

    def test_stdlib_logging_from_a_library_keeps_its_logger_name(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        logging.getLogger("PIL.TiffImagePlugin").warning("odd tag %d", 42)
        logging.getLogger("PIL.TiffImagePlugin").info("below the bridge level")

        assert gui_session.log_file is not None
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert "PIL.TiffImagePlugin:" in text
        assert "odd tag 42" in text
        assert "below the bridge level" not in text

    def test_python_warnings_reach_the_log(
        self, gui_session: diagnostics.DiagnosticsSession
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            logging.captureWarnings(True)
            warnings.warn("overflow in multiply", RuntimeWarning, stacklevel=1)

        assert gui_session.log_file is not None
        text = gui_session.log_file.read_text(encoding="utf-8")
        assert "py.warnings:" in text
        assert "RuntimeWarning: overflow in multiply" in text

    def test_end_puts_back_the_hooks_it_replaced(
        self, log_folder: Path, restore_process_state: None
    ):
        excepthook = sys.excepthook
        unraisablehook = sys.unraisablehook
        handlers = list(logging.getLogger().handlers)

        session = configure_logging("gui", stderr_level="CRITICAL")
        assert sys.excepthook is diagnostics.log_uncaught_exception
        session.end()

        assert sys.excepthook is excepthook
        assert sys.unraisablehook is unraisablehook
        assert logging.getLogger().handlers == handlers


class TestQtMessages:
    @pytest.mark.parametrize(
        ("message_type", "level"),
        [
            (QtCore.QtMsgType.QtDebugMsg, "DEBUG"),
            (QtCore.QtMsgType.QtInfoMsg, "INFO"),
            (QtCore.QtMsgType.QtWarningMsg, "WARNING"),
            (QtCore.QtMsgType.QtCriticalMsg, "ERROR"),
            (QtCore.QtMsgType.QtFatalMsg, "CRITICAL"),
        ],
    )
    def test_qt_types_map_to_loguru_levels(
        self, message_type: QtCore.QtMsgType, level: str
    ):
        assert diagnostics.qt_message_level(message_type) == level

    def test_qt_messages_are_logged_with_their_category_until_restored(self):
        records: list[tuple[str, str]] = []
        sink = logger.add(
            lambda message: records.append(
                (message.record["level"].name, message.record["message"])
            ),
            level="DEBUG",
        )
        restore = diagnostics.install_qt_message_handler()
        try:
            QtCore.qWarning("plugin missing")
            QtCore.qCritical("connect failed")
        finally:
            restore()
        QtCore.qWarning("after restore")
        logger.remove(sink)

        assert records == [
            ("WARNING", "qt[default] plugin missing"),
            ("ERROR", "qt[default] connect failed"),
        ]

    def test_the_handler_never_raises(self, monkeypatch: pytest.MonkeyPatch):
        def broken_log(*_args: object) -> None:
            raise RuntimeError("sink failed")

        monkeypatch.setattr(diagnostics.logger, "log", broken_log)

        diagnostics.forward_qt_message(
            QtCore.QtMsgType.QtWarningMsg, QtCore.QMessageLogContext(), "text"
        )


def test_banner_names_the_settings_and_log_paths(tmp_path: Path):
    lines = diagnostics.banner_lines(tmp_path / "gui.log")

    assert lines[0].startswith(f"{APP_NAME} ")
    assert lines[0].endswith("from source")
    assert lines[-1].endswith(f"log {tmp_path / 'gui.log'}")
    assert "absent, using defaults" in lines[-1]
