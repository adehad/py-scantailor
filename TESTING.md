# Testing

The suite is pytest with pytest-qt and runs in parallel through pytest-xdist.

```bash
pdm run pytest                               # default run: skips tests marked slow
pdm run pytest -m ""                         # full run, including slow tests; use before merging
pdm run pytest -n0 tests/file.py::test_name  # one test, serially, for debugging
pdm run cov                                  # coverage report in dist/cov/
```

CI runs `pdm run cov -m "not benchmark and not docs_screenshot"`: every test, the slow ones included, in one job, with the 30 slowest listed at the end.

Qt tests need no display: CI sets `QT_QPA_PLATFORM=offscreen`, and the same works locally.

## Workers

The default run uses `-n auto`, one worker per CPU core, which is what CI wants. On a machine you are using at the same time, cap it with pytest-xdist's `PYTEST_XDIST_AUTO_NUM_WORKERS`, which every `-n auto` run reads, the docs screenshots included:

```bash
export PYTEST_XDIST_AUTO_NUM_WORKERS=2       # PowerShell: $env:PYTEST_XDIST_AUTO_NUM_WORKERS = 2
unset PYTEST_XDIST_AUTO_NUM_WORKERS          # back to one per core
```

An explicit `-n N` on the command line wins over the variable.

## Locale

`--qt-locale=sv_SE` makes a Qt locale the default for the whole run, as a user's system locale would. Tests compare the numbers a widget shows, read back with `tests/ui/locale_numbers.py`, not the formatted text, so they pass under any locale. Run the UI tests under a decimal-comma locale after changing how a label or a test formats numbers:

```bash
pdm run pytest --qt-locale=sv_SE tests/ui
```

`tests/ui/test_decimal_comma_locale.py` switches to Swedish itself, so every run checks that labels follow the locale.

## Markers

| Marker            | Meaning                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| `slow`            | Takes more than about 2 s. Excluded from the default run.                    |
| `docs_screenshot` | Drives the window to save the manual's PNGs. Run with `-m docs_screenshot`. |
| `benchmark`       | Reports a timing in the run's summary and never fails on it. Excluded from the default run. Run with `-m benchmark -n0` on an idle machine. |

## Layout

`tests/` mirrors `src/scantailor/`: `tests/filters/deskew/` tests `scantailor.filters.deskew`, and so on. `tests/ui/` covers the windows and dialogs, `tests/cli/` the command line and `tests/docs/` the manual's screenshots. Shared test data lives in `tests/fixtures/`.

`packages/jbig2/` has its own suite under `packages/jbig2/tests/`.
