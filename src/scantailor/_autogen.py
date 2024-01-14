"""Autogen."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel


class AutoGenConfig(BaseModel):
    files: list[list[str]]
    qt_rcc: str = "pyside6-rcc"
    qt_rcc_options: str = ""
    qt_uic: str = "pyside6-uic"
    qt_uic_options: str = ""

    qt_lupdate: str = os.getenv("LUPDATE_BIN", "lupdate")
    qt_lrelease: str = os.getenv("LRELEASE_BIN", "lrelease")


def is_outdated(in_file: Path, out_file: Path) -> bool:
    """Method to help determine if a generated file is out of date."""
    if not in_file.is_file():
        return False  # can't check if a folder is upto date
    if not out_file.exists():
        return True  # if the output file doesn't exist, it must be out of date
    if in_file.stat().st_mtime < out_file.stat().st_mtime:
        return False  # if output file changed after inpput, then in date
    return True  # For other cases assume out of date!


class UIAutoGen:
    def __init__(self, config: AutoGenConfig) -> None:
        self.config = config

    def gen_from_ui(self, src: str, target: str):
        return subprocess.check_output(
            [
                self.config.qt_uic,
                *shlex.split(self.config.qt_uic_options),
                src,
                "-o",
                target,
            ]
        )

    def gen_from_rcc(self, src: str, target: str):
        return subprocess.check_output(
            [
                self.config.qt_rcc,
                *shlex.split(self.config.qt_rcc_options),
                src,
                "-o",
                target,
            ]
        )


def run():
    root_path = (Path(__file__).parent / "../../../").resolve()
    resource_path = root_path / "resources"
    dest_path = root_path / "src/app/_ui/"
    app_folder_name = "src/app/"
    app_path = root_path / app_folder_name
    with (root_path / "pyuic.json").open() as fp:
        config = AutoGenConfig.model_validate_json(fp.read())

    auto_gen = UIAutoGen(config)
    subprocess.check_call([config.qt_lupdate, str(root_path / "app.pro")])
    subprocess.check_call([config.qt_lrelease, str(root_path / "app.pro")])

    auto_gen.gen_from_rcc(
        src=str(resource_path / "resources.qrc"),
        target=f"{(dest_path / (resource_path / 'resources.qrc').stem)}_rc.py",
    )
    for ui_file in app_path.rglob("*.ui"):
        auto_gen.gen_from_ui(
            src=str(ui_file),
            target=f"{dest_path / ui_file.stem}_ui.py",
        )


if __name__ == "__main__":
    if not any([os.environ.get(e) for e in {"READTHEDOCS"}]):
        run()
