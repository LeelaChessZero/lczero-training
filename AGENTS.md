# AGENTS.md

This repository contains training script for the Leela Chess Zero project.
They are being rewritten.

* Old code is located in the `tf/` directory.
* New python code is located in the `src/` directory.
* New C++ code is located in the `csrc/` directory.

The old code is Python/TensorFlow-based, new code is Python/JAX-based with
modules written in C++, operating through pybind11.

The build system for C++ code is meson. During development, the project is built
in the `builddir/`.

## Testing and Building

* C++ tests use GTest framework
  * Do not insert Sleeps in tests, it slows down presubmit. Instead use e.g.
    absl::Notification, or std::future
* Tests are defined in `meson.build` with `test()` function
  * When debugging, don't forget to build them before running `meson test` or
    `builddir/test`
* Run tests: `meson test -C builddir/`
* Python tests use `pytest` framework
  * Do not add custom main function, exception catching to report errors, any
    "test passed" messages etc. Use `pytest` fixtures and assertions.
* Build: `meson compile -C builddir/` from build directory
* Format code: `just format`
* There is a commit hook that runs `just pre-commit`, which runs tests and
  checks formatting. You may want to run it before attempting to commit.
* We use Google C++ style guide.
  * That means 80 columns.
  * That means comments should be in full sentences with periods in the end.
  * When conditional or loop fits one line, it must be written as one line
    without braces, for example:
      `if (condition) return value;`
  * Prefer `absl` to `std` (e.g. `absl::c_` algorithms, `absl::Mutex`,
    `absl::StrCat`, etc.)
* We use `uv` for Python package and venv management, and to running the
  application.
* Run TUI app: `uv run tui --config=<path_to_config>`

* Do not attempt to run TUI — it messes up the Agent interface and session has
  to be killed. Ask me to check it for you manually instead.

* Do not commit unless explicitly asked.

## IMPORTANT

* NEVER add `# type: ignore` or other ways to mask/silence errors instead of
  fixing them.

## Documentation

* Documentation is in the `docs/` directory.
* The contents is in [The index](docs/index.md)
