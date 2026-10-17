# pyScanTailor

[![Documentation Status](https://readthedocs.org/projects/py-scantailor/badge/?version=latest)](https://py-scantailor.readthedocs.io/en/latest/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/adehad/py-scantailor)
[![Dev Containers: Open](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/adehad/py-scantailor)

pyScanTailor cleans up scanned book and document pages. Give it raw scans or a PDF and it splits two-page spreads, straightens each page, crops to the content and gives every page the same margins. The result is black and white, colour or mixed pages, saved as images or exported as one PDF. It works through a whole book at once, and you can correct any page by hand.

It is for anyone digitising books, papers or archives with a scanner or a camera. Scanning, text recognition (OCR) and DjVu encoding are left to other tools.

pyScanTailor is a Python and PySide6 port of [ScanTailor Advanced](https://github.com/ScanTailor-Advanced/scantailor-advanced). It runs on Windows, macOS and Linux, and has a command line for batch jobs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://py-scantailor.readthedocs.io/en/latest/_images/stage6_output1.png">
  <img alt="The Output stage: the six processing stages on the left, the finished page in the middle and page thumbnails on the right" src="https://py-scantailor.readthedocs.io/en/latest/_images/stage6_output.png">
</picture>

## Download

Builds for Windows, macOS and Linux are on the [GitHub Releases page](https://github.com/adehad/py-scantailor/releases). Each one carries its own Python and Qt, so there is nothing else to install.

| Platform | File | Start the program |
| --- | --- | --- |
| Windows x64 | `ScanTailor-<version>-windows-x64.zip` | `ScanTailor\ScanTailor.exe` |
| macOS, Apple Silicon | `ScanTailor-<version>-macos-arm64.zip` | `ScanTailor.app` |
| Linux x64 | `ScanTailor-<version>-linux-x64.AppImage` | the AppImage itself |
| Linux x64 | `ScanTailor-<version>-linux-x64.tar.gz` | `ScanTailor/ScanTailor` |

Every build also includes the command line, `scantailor-cli`.

- **Windows:** unzip anywhere and run `ScanTailor.exe`.
- **macOS:** needs macOS 15 or later. Move `ScanTailor.app` to Applications. The app is not signed by Apple, so macOS blocks the first launch: right-click the app and choose **Open**. If macOS only offers to move it to the Bin, try to open it once, then allow it under **System Settings > Privacy & Security > Open Anyway**. The command line is at `ScanTailor.app/Contents/MacOS/scantailor-cli`. Intel Macs have no build; install with Python instead.
- **Linux:** needs glibc 2.35 or newer (Ubuntu 22.04 or later). Make the AppImage executable with `chmod +x` and run it; without FUSE, add `--appimage-extract-and-run`. To use it as the command line, start it through a symlink named `scantailor-cli`. If the window does not open, install Qt's X11 libraries (Debian and Ubuntu: `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libegl1`).

## Install with Python

pyScanTailor needs Python 3.14. With [uv](https://docs.astral.sh/uv/), which fetches Python 3.14 if you do not have it, run the latest code straight from this repository:

```bash
uvx --from git+https://github.com/adehad/py-scantailor scantailor
```

To keep it installed, with `scantailor` and `scantailor-cli` on your `PATH`:

```bash
uv tool install git+https://github.com/adehad/py-scantailor
```

Or with pip, in a Python 3.14 environment:

```bash
pip install git+https://github.com/adehad/py-scantailor
```

| Command | Starts |
| --- | --- |
| `scantailor` | the window |
| `scantailor-cli` | the command line; `scantailor-cli --help` lists its commands |

## Documentation

The [user guide](https://py-scantailor.readthedocs.io/) walks through a project from the first scan to an exported PDF and explains every stage, option and setting.

## Translations

The interface is available in English, German, Spanish, French, Korean, Polish, Portuguese and Russian. Each language covers almost all of the interface, but about a fifth of each catalogue is a machine draft waiting for review by a native speaker. The `.ts` files under [`src/scantailor/translations/`](src/scantailor/translations/) mark those entries "Machine draft, pending native review". Corrections are welcome as pull requests.

## Contributing

Report bugs and ask for features in [GitHub issues](https://github.com/adehad/py-scantailor/issues).

To work on the code, open the dev container with one of the badges at the top, or set it up yourself with Python 3.14 and [PDM](https://pdm-project.org):

```bash
pdm install
pdm run scantailor            # the window
pdm run scantailor-cli --help # the command line
pdm run pytest                # the tests
pdm run docs                  # the user guide, into dist/docs
```

[AGENTS.md](AGENTS.md) describes the project layout and code rules, and [TESTING.md](TESTING.md) explains how the tests are organised.

`pdm run exe` builds the standalone program for the platform you run it on, with PyInstaller and [`app.spec`](app.spec). Pushing a tag that starts with `v`, such as `v1.0.0`, makes [`.github/workflows/release.yml`](.github/workflows/release.yml) build, smoke-test and publish every platform to GitHub Releases. Tags containing `-rc`, `-alpha` or `-beta` become pre-releases.

## Credits and licence

pyScanTailor builds on ScanTailor by Joseph Artsimovich and on the forks that followed it: ScanTailor Enhanced, ScanTailor Featured, ScanTailor Universal and ScanTailor Advanced. [FORK_HISTORY.md](FORK_HISTORY.md) links to each one and lists the features they added.

pyScanTailor is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE).
