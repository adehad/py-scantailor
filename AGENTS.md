# AGENTS.md - AI Agent Instructions

## Essential Commands

```bash
pdm run pytest                            # Fast default run: parallel (xdist), skips slow tests
pdm run pytest -m ""                      # Full gate including slow tests (run before merging)
pdm run pytest -n0 tests/file.py::test_name  # Run specific test serially (for debugging)
pdm run cov                               # Run tests with coverage
pdm run typecheck                         # Run type checking
pre-commit run --all-files                # Run all linters/formatters
pdm run docs                              # Build documentation
pdm run exe                               # Build the executables with PyInstaller
```

**Formatting**: Do NOT manually fix formatting/linting errors. Write code first, then run `pre-commit run --all-files` at the end to auto-fix.

## Project Context

**Application**: ScanTailor - document scanning post-processor, written in Python with PySide6
**Origin**: a port of [ScanTailor Advanced](https://github.com/ScanTailor-Advanced/scantailor-advanced) (C++/Qt). The C++ code is not in this repository; read it upstream when you need to compare behaviour.
**Tracking**: work and known defects are GitHub issues on `adehad/py-scantailor`

## Source Code Locations

```text
src/scantailor/              - Main package
src/scantailor/app/ui/       - Qt windows, dialogs, panels and their .ui forms
src/scantailor/core/         - Project model, settings, pipeline
src/scantailor/filters/      - The six processing stages
src/scantailor/imageproc/    - Image algorithms
src/scantailor/dewarping/    - Page dewarping
src/scantailor/math/         - Splines and numeric helpers
src/scantailor/cli/          - Headless command line
src/scantailor/resources/    - Qt resources (icons, fonts)
src/scantailor/translations/ - .ts catalogues
packages/jbig2/              - JBIG2 encoder used by PDF export
tools/                       - Translation, screenshot and PyInstaller helpers
tests/                       - pytest suite, mirroring the package layout
docs/                        - Sphinx user guide and API reference
```

## Design Philosophy

**Do NOT blindly replicate upstream C++ patterns.** Upstream is a behavioural reference, not a template.
- Critically evaluate each upstream design - many are workarounds for C++ limitations
- Use Pythonic idioms (iterators, context managers, dataclasses, etc.)
- Simplify where possible - Python has richer built-ins
- Question the need for each class/abstraction
- Bundle related changes in single commits: code + tests + docs

## Code Rules

### MUST DO
- Type hints on all functions
- Google-style docstrings with type info (Args must include types, not just param names)
- In tests, the two rules above cover helpers only: page builders, conftest functions and shared assertions. Fixtures count as helpers. A `test_*` function or method takes no annotations and no Args section, and its name states the scenario and expected outcome. Give it a one-line docstring saying what it tests when the name alone doesn't make that clear, and keep existing ones. `tests/ruff.toml` turns off ruff's `D` rules, so docstring formatting in tests is not checked.
- Pydantic for data models
- JSON for persistence (NOT XML)
- loguru for logging
- Tests for every module
- Self-describing code that reads like English
- Keep upstream class names as module names where sensible, use PEP8 variable naming
- Comments explain WHY, not what
- If a comment breaks up a function, extract to a named function instead

### MUST NOT
- Wrapper classes or pass-through functions around OpenCV/NumPy (call cv2/NumPy directly)
- XML for any data storage
- `print()` statements
- Bare `except:` clauses
- Mutable default arguments
- `from x import *`
- Keep deprecated/dead code (rely on git for rollback)
- Add numba, SciPy or scikit-image to the application: it ships NumPy, OpenCV, Pillow and the stdlib only, so there is one OpenBLAS and no compiler in the build

## Memory and dtypes in pixel code

- **Dtypes.** Anchor `np.where`, `np.select` and `np.full` with NumPy scalars such as `np.uint8(0)`, because Python ints make an int64 array. Give reductions over an image a `dtype=`. Widen a full page to float64 or int64 only when the maths needs it, and then work in bands. `tools/check_numpy_dtypes.py` checks this in pre-commit; silence a deliberate case with `# npd: ok <reason>`.
- **Lifetimes.** Python frees an array when its last reference goes, so a helper cannot release its caller's variable. Split a long pixel function into phase functions that return only what the next phase needs, and `del` a full-page array right after its last use inside a phase. The stage memory budgets in `tests/core/test_memory_budget.py` catch regressions.
- **Bands.** Run windowed kernels through `imageproc/row_bands.py`. It sizes bands by rows and memory, never by thread count.
- **Inputs.** Never mutate an input array, and don't defend it with `.copy()`, since inputs may be read-only views.
- **Tests.** Never pass or fail on time. Count calls or passes, use a spy, or set a memory budget. Compare one thread with N threads, not a grid of thread counts.

## Library Choices

| Upstream C++  | Python                                       |
| ------------- | -------------------------------------------- |
| BinaryImage   | `np.ndarray` (uint8/bool)                    |
| GrayImage     | `np.ndarray` (uint8)                         |
| LinearSolver  | `np.linalg.solve()`                          |
| Morphology    | `cv2.morphologyEx()`                         |
| GaussBlur     | `cv2.GaussianBlur()`                         |
| Binarize      | `cv2.threshold()`                            |
| Transform     | `cv2.warpAffine()` / `cv2.warpPerspective()` |
| XmlMarshaller | Pydantic + JSON                              |

## Custom Code

These have no library equivalent and live in this repository:
- X-splines (`src/scantailor/math/spline.py`)
- Cylindrical dewarping (`src/scantailor/dewarping/`)
- Text line tracing (`src/scantailor/dewarping/text_line_tracer.py`)
- Mokji binarization (`src/scantailor/imageproc/binarize.py`)

## Testing Requirements

- Verbose, descriptive tests
- Cover: normal, edge cases, errors, round-trips
- Test data in `tests/fixtures/`
- pytest + pytest-qt

## Commit Style

Brief messages grouping related changes: code, tests and docs for one change in one commit.

Examples:
- `Add Mokji binarization`
- `Remember the split line across pages`
- `Add project JSON schema`

## Code Review Focus

CI/pre-commit already enforces: formatting, imports, docstring presence, and types in `src/` (pyrefly checks only `src/`). Don't ask for annotations, Args sections or docstring formatting on `test_*` functions; see Code Rules.

**Reviewers should focus on what linters can't catch:**

### Architecture & Design
- Is this the right abstraction? Question over-engineering
- Is this a C++ pattern that should be more Pythonic?
- Filter classes must be stateless; Settings classes must be thread-safe
- Are we using Python stdlib where upstream had manual implementations?

### Thread Safety (Critical for filters/)
- Settings with mutable state must use `threading.Lock`
- Lock should protect minimal critical section only
- Watch for: nested locks (deadlock), lock held during I/O or image processing

### Image Processing (imageproc/)
- Check dtype handling (uint8, float32, bool)
- Never mutate input arrays, and don't copy them defensively: inputs may be read-only views
- Memory usage for large images
- Document expected input/output dtypes in docstrings

### Quality Checks
- Docstring **content** helpful? (not just present)
- Args descriptions meaningful, not restating parameter names?
- Complex algorithms have WHY comments?
- Magic numbers explained?

### Test Quality
- Test names describe scenario AND expected outcome?
- Edge cases covered (empty, boundary, error)?
- Assertions checking the right thing?
- Avoid: no-assertion tests, exact float comparisons, duplicated tests
