"""Log file, crash records and start-up banner for one ScanTailor process.

The window build has no console, so without a file sink every log line and
uncaught exception is lost. ``configure_logging`` gives each process its own
log file in the log folder, routes uncaught exceptions, unraisable exceptions,
Python warnings and stdlib ``logging`` records into it, and points
``faulthandler`` at a crash file beside it. ``DiagnosticsSession.end`` writes
the clean-exit marker that the next start-up looks for.
"""

import contextlib
import faulthandler
import importlib
import importlib.metadata
import locale
import logging
import os
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, TextIO

import PySide6
from loguru import logger
from PySide6 import QtCore

from scantailor.__about__ import APP_NAME, __version__
from scantailor.core.settings import ApplicationSettings

type SessionKind = Literal["gui", "cli"]
type ExceptHook = Callable[
    [type[BaseException], BaseException, TracebackType | None], object
]
type UnraisableHook = Callable[[sys.UnraisableHookArgs], object]

LOG_FORMAT: Final = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {process.id} {thread.name: <14} | "
    "{extra[page]} | {extra[stage]: <14} | {name}:{line} | {message}"
)
# The format reads these keys, so a record logged without them would raise
# KeyError in the formatter.
EXTRA_DEFAULTS: Final = {"page": "-", "stage": "-"}
SESSION_ENDED: Final = "session ended"

# rotation with retention=1 keeps the live file and one rotated file of this
# session; rotation alone keeps every rotated file and caps nothing.
_ROTATION: Final = "10 MB"
_KEEP_PER_KIND: Final = 10
_FOLDER_BUDGET_BYTES: Final = 100 * 1024 * 1024
_PRUNED_KINDS: Final = ("gui", "cli", "crash")
_SESSION_FILE: Final = re.compile(r"^(?P<kind>gui|cli)-\d{8}-\d{6}-(?P<pid>\d+)\.log$")
_TAIL_BYTES: Final = 4096

_QT_LEVELS: Final = {
    QtCore.QtMsgType.QtDebugMsg: "DEBUG",
    QtCore.QtMsgType.QtInfoMsg: "INFO",
    QtCore.QtMsgType.QtWarningMsg: "WARNING",
    QtCore.QtMsgType.QtCriticalMsg: "ERROR",
    QtCore.QtMsgType.QtFatalMsg: "CRITICAL",
}

# (label, distribution, module, version attribute). A frozen build carries no
# metadata for its dependencies, so the module attribute is the fallback.
_LIBRARIES: Final = (
    ("NumPy", "numpy", "numpy", "__version__"),
    ("OpenCV", "opencv-python", "cv2", "__version__"),
    ("Pillow", "pillow", "PIL", "__version__"),
    ("pikepdf", "pikepdf", "pikepdf", "__version__"),
    ("pypdfium2", "pypdfium2", "pypdfium2.version", "PYPDFIUM_INFO"),
    ("loguru", "loguru", "loguru", "__version__"),
)


def log_folder() -> Path:
    """Locate the folder this application writes its logs to.

    Returns:
        Path: ``%LOCALAPPDATA%/ScanTailor/logs`` on Windows,
            ``~/Library/Logs/ScanTailor`` on macOS and
            ``$XDG_STATE_HOME/scantailor/logs`` (``~/.local/state`` when unset)
            elsewhere. The folder may not exist yet.
    """
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "ScanTailor" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "ScanTailor"
    state = os.environ.get("XDG_STATE_HOME")
    # The XDG spec says to ignore a relative path.
    if state and Path(state).is_absolute():
        base = Path(state)
    else:
        base = Path.home() / ".local" / "state"
    return base / "scantailor" / "logs"


def _timestamp() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}"


def session_log_name(kind: SessionKind) -> str:
    """Name the log file of this process.

    Args:
        kind (SessionKind): "gui" for the window, "cli" for scantailor-cli.

    Returns:
        str: ``<kind>-YYYYMMDD-HHMMSS-<pid>.log``.
    """
    return f"{kind}-{_timestamp()}-{os.getpid()}.log"


def prune_logs(folder: Path) -> list[Path]:
    """Delete old log files so the folder stays within its budget.

    Each of ``gui-*``, ``cli-*`` and ``crash-*`` keeps its newest 10 files, so a
    script calling the CLI many times cannot push out the window log of the
    session that crashed. Then the oldest ``*.log`` files go until the folder
    holds at most 100 MB. A file another running instance holds open cannot be
    deleted on Windows; it is skipped.

    Args:
        folder (Path): The log folder.

    Returns:
        list[Path]: The files deleted.
    """
    logs = sorted(folder.glob("*.log"), key=_modified_time, reverse=True)
    deleted: list[Path] = []
    for kind in _PRUNED_KINDS:
        of_kind = [path for path in logs if path.name.startswith(f"{kind}-")]
        deleted += [path for path in of_kind[_KEEP_PER_KIND:] if _delete(path)]
    remaining = [path for path in logs if path not in deleted]
    total = sum(_size(path) for path in remaining)
    for path in reversed(remaining):
        if total <= _FOLDER_BUDGET_BYTES:
            break
        size = _size(path)
        if _delete(path):
            deleted.append(path)
            total -= size
    return deleted


def _modified_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _delete(path: Path) -> bool:
    try:
        path.unlink()
    except OSError:
        return False
    return True


def ended_cleanly(path: Path) -> bool:
    """Whether a session log ends with the clean-exit marker.

    Args:
        path (Path): A session log file.

    Returns:
        bool: True when its last line is the ``session ended`` line.
    """
    try:
        with path.open("rb") as stream:
            stream.seek(max(_size(path) - _TAIL_BYTES, 0))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return True
    lines = tail.rstrip().splitlines()
    return bool(lines) and lines[-1].endswith(f"| {SESSION_ENDED}")


def previous_unclean_session(
    folder: Path, kind: SessionKind, current: Path | None = None
) -> Path | None:
    """Find the last finished session of a kind that did not end cleanly.

    Sessions still running in another instance are skipped: their log lacks the
    marker only because they have not ended yet. Rotated files are never the
    last file of a session, so only the live file name of each session counts.

    Args:
        folder (Path): The log folder.
        kind (SessionKind): The kind of session to look at.
        current (Path | None): This process's own log, which is skipped.

    Returns:
        Path | None: The newest finished session's log when it lacks the
            ``session ended`` line, else None.
    """
    sessions = [
        path
        for path in folder.glob(f"{kind}-*.log")
        if path != current and _SESSION_FILE.match(path.name)
    ]
    for path in sorted(sessions, key=_modified_time, reverse=True):
        if _is_live(path):
            continue
        return None if ended_cleanly(path) else path
    return None


def _is_live(path: Path) -> bool:
    """Whether the process that writes ``path`` is still running."""
    if sys.platform == "win32":
        return _is_held_open(path)
    return _writer_is_running(path)


def _is_held_open(path: Path) -> bool:
    # Python opens files without FILE_SHARE_DELETE, so renaming a file another
    # process holds open fails on Windows.
    try:
        os.replace(path, path)
    except PermissionError:
        return True
    except OSError:
        return False
    return False


def _writer_is_running(path: Path) -> bool:
    session_name = _SESSION_FILE.match(path.name)
    if session_name is None:
        return False
    try:
        os.kill(int(session_name.group("pid")), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _library_version(distribution: str, module: str, attribute: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return str(getattr(importlib.import_module(module), attribute))
    except ImportError, AttributeError:
        return "unknown"


def banner_lines(log_file: Path | None) -> list[str]:
    """Describe this process for the top of its log.

    Args:
        log_file (Path | None): The session's log file, or None without one.

    Returns:
        list[str]: Version and build, platform, library versions, CPUs, locale,
            and the settings and log paths, one line each.
    """
    build = "frozen (PyInstaller)" if getattr(sys, "frozen", False) else "from source"
    libraries = "; ".join(
        f"{label} {_library_version(distribution, module, attribute)}"
        for label, distribution, module, attribute in _LIBRARIES
    )
    settings = ApplicationSettings.get_config_path()
    settings_state = "found" if settings.is_file() else "absent, using defaults"
    language, encoding = locale.getlocale()
    python = platform.python_version()
    return [
        f"{APP_NAME} {__version__}, {build}",
        f"{platform.platform()} {platform.machine()}; Python {python}",
        f"PySide6 {PySide6.__version__} / Qt {QtCore.qVersion()}; {libraries}",
        f"CPUs {os.cpu_count()}, usable {os.process_cpu_count()}",
        f"locale {language or '-'}, encoding {encoding or '-'}",
        f"settings {settings} ({settings_state}); log {log_file or 'none'}",
    ]


@dataclass
class DiagnosticsSession:
    """What ``configure_logging`` set up, and how to take it down at a clean exit.

    Attributes:
        log_file (Path | None): The session log, or None when no file is written.
        crash_file (Path | None): Where ``faulthandler`` writes a native crash.
    """

    log_file: Path | None = None
    crash_file: Path | None = None
    _file_sink: int | None = field(default=None, repr=False)
    _crash_stream: TextIO | None = field(default=None, repr=False)
    _previous_excepthook: ExceptHook | None = field(default=None, repr=False)
    _previous_unraisablehook: UnraisableHook | None = field(default=None, repr=False)
    _stdlib_bridge: logging.Handler | None = field(default=None, repr=False)

    def end(self) -> None:
        """Write the clean-exit marker and undo the process hooks.

        The marker is the last line of the file, which tells the next start-up
        this session did not crash. An empty crash file is deleted.
        """
        logger.info(SESSION_ENDED)
        if self._file_sink is not None:
            logger.remove(self._file_sink)
            self._file_sink = None
        if self._previous_excepthook is not None:
            sys.excepthook = self._previous_excepthook
            self._previous_excepthook = None
        if self._previous_unraisablehook is not None:
            sys.unraisablehook = self._previous_unraisablehook
            self._previous_unraisablehook = None
        if self._stdlib_bridge is not None:
            logging.getLogger().removeHandler(self._stdlib_bridge)
            logging.captureWarnings(False)
            self._stdlib_bridge = None
        self._close_crash_file()

    def _close_crash_file(self) -> None:
        if self._crash_stream is None:
            return
        # Closing a file faulthandler still writes to would leave it writing
        # to a closed descriptor on the next crash.
        faulthandler.disable()
        self._crash_stream.close()
        self._crash_stream = None
        if self.crash_file is not None and _size(self.crash_file) == 0:
            _delete(self.crash_file)


def configure_logging(
    kind: SessionKind,
    *,
    stderr_level: str,
    log_file: Path | None = None,
    write_to_log_folder: bool = True,
) -> DiagnosticsSession:
    """Set up the sinks, crash hooks and bridges of this process.

    Replaces every loguru handler. A failure to write the log file is logged
    and the process carries on without one.

    Args:
        kind (SessionKind): "gui" or "cli"; names the files.
        stderr_level (str): Lowest level written to stderr, when there is one.
        log_file (Path | None): Write the session log here instead of the log
            folder. The folder is then neither pruned nor checked.
        write_to_log_folder (bool): Without ``log_file``, whether to write a
            session log in the log folder at all.

    Returns:
        DiagnosticsSession: Call ``end`` on it at a clean exit.
    """
    logger.remove()
    logger.configure(extra=dict(EXTRA_DEFAULTS))
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            level=stderr_level,
            format=LOG_FORMAT,
            backtrace=False,
            diagnose=False,
        )
    session = DiagnosticsSession()
    unclean = None
    path = log_file
    if path is None and write_to_log_folder:
        folder = log_folder()
        path = folder / session_log_name(kind)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.warning("Could not create the log folder {}: {}", folder, error)
            path = None
        else:
            prune_logs(folder)
            unclean = previous_unclean_session(folder, kind, current=path)
    if path is not None:
        _add_file_sink(session, path, kind)
    _install_hooks(session)
    for line in banner_lines(session.log_file):
        logger.info(line)
    if unclean is not None:
        logger.warning("previous session did not end cleanly: {}", unclean.name)
    return session


def _add_file_sink(session: DiagnosticsSession, path: Path, kind: SessionKind) -> None:
    try:
        session._file_sink = logger.add(
            path,
            level="DEBUG",
            format=LOG_FORMAT,
            rotation=_ROTATION,
            retention=1,
            encoding="utf-8",
            backtrace=False,
            diagnose=False,
        )
    except OSError as error:
        logger.warning("Could not write the log file {}: {}", path, error)
        return
    session.log_file = path
    crash_file = path.parent / f"crash-{kind}-{_timestamp()}-{os.getpid()}.log"
    try:
        stream = crash_file.open("w", encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write the crash file {}: {}", crash_file, error)
        return
    # Without file= faulthandler writes to sys.stderr, which is None in the
    # window build, and enable() raises.
    faulthandler.enable(file=stream, all_threads=True)
    session.crash_file = crash_file
    session._crash_stream = stream


def _install_hooks(session: DiagnosticsSession) -> None:
    session._previous_excepthook = sys.excepthook
    session._previous_unraisablehook = sys.unraisablehook
    sys.excepthook = log_uncaught_exception
    sys.unraisablehook = log_unraisable_exception
    bridge = StdlibBridge(level=logging.WARNING)
    logging.getLogger().addHandler(bridge)
    logging.captureWarnings(True)
    session._stdlib_bridge = bridge


def log_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """``sys.excepthook`` that logs the traceback and lets the process go on.

    Qt calls it for exceptions escaping slots and ``QRunnable.run`` on pool
    threads too, so the window stays open after a failure in one of them.

    Args:
        exc_type (type[BaseException]): The exception's class.
        exc_value (BaseException): The exception.
        exc_traceback (TracebackType | None): Where it was raised.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
        "uncaught exception"
    )


def log_unraisable_exception(unraisable: sys.UnraisableHookArgs) -> None:
    """``sys.unraisablehook`` for exceptions in ``__del__`` and GC callbacks.

    Args:
        unraisable (sys.UnraisableHookArgs): What Python could not raise.
    """
    exception = (unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
    logger.opt(exception=exception).warning(
        "{}: {}",
        unraisable.err_msg or "Exception ignored in",
        _safe_repr(unraisable.object),
    )


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__} object>"


class StdlibBridge(logging.Handler):
    """Hand stdlib ``logging`` records from libraries to loguru.

    The record keeps the library logger's name and line, so the log shows
    where the message came from rather than this handler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Log one stdlib record through loguru.

        Args:
            record (logging.LogRecord): The record a library logged.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        try:
            logger.patch(
                lambda loguru_record: loguru_record.update(
                    name=record.name, function=record.funcName, line=record.lineno
                )
            ).opt(exception=record.exc_info).log(level, record.getMessage())
        except Exception:
            self.handleError(record)


def qt_message_level(message_type: QtCore.QtMsgType) -> str:
    """Translate a Qt message type to a loguru level name.

    Args:
        message_type (QtCore.QtMsgType): The type Qt gave the message.

    Returns:
        str: The loguru level; Qt's critical is ERROR and fatal is CRITICAL.
    """
    return _QT_LEVELS.get(message_type, "WARNING")


def forward_qt_message(
    message_type: QtCore.QtMsgType,
    context: QtCore.QMessageLogContext,
    message: str,
) -> None:
    """Qt message handler that writes Qt's own messages to the log.

    Args:
        message_type (QtCore.QtMsgType): Debug, info, warning, critical or fatal.
        context (QtCore.QMessageLogContext): Carries the logging category.
        message (str): The text Qt logged.
    """
    # Qt calls this from C++, during a fatal abort too; an exception escaping
    # it there cannot be handled.
    with contextlib.suppress(Exception):
        category = getattr(context, "category", None) or "default"
        logger.log(qt_message_level(message_type), "qt[{}] {}", category, message)


def install_qt_message_handler() -> Callable[[], None]:
    """Route Qt's messages to the log.

    Install it before the QApplication exists: Qt reports a missing platform
    plugin while the application object is being created.

    Returns:
        Callable[[], None]: Restores the handler that was installed before.
            Connect it to ``aboutToQuit`` so no Python handler runs during
            interpreter shutdown.
    """
    previous = QtCore.qInstallMessageHandler(forward_qt_message)

    def restore() -> None:
        QtCore.qInstallMessageHandler(previous)

    return restore
