"""Exit codes and the error type that carries them out of a command."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit status of every command.

    Attributes:
        OK: The command completed.
        FAILED: The command ran but some pages failed to process.
        USAGE: The command line could not be parsed (argparse's own code).
        PROJECT: The project file is missing, unreadable or would be overwritten.
        INPUT: The source folder or PDF is missing, unreadable or has no pages.
        OPTION: A stage option name or value was rejected.
        NOTHING_TO_EXPORT: No processed output exists to export.
    """

    OK = 0
    FAILED = 1
    USAGE = 2
    PROJECT = 3
    INPUT = 4
    OPTION = 5
    NOTHING_TO_EXPORT = 6


class CliError(Exception):
    """A failure that ends the command with a specific exit code.

    Args:
        code (ExitCode): Status the process exits with.
        message (str): One line for stderr, without a trailing newline.
    """

    def __init__(self, code: ExitCode, message: str) -> None:
        """Store the exit code and the message shown to the user."""
        super().__init__(message)
        self.code = code
        self.message = message
