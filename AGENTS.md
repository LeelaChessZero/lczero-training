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
* Build: `meson compile -C builddir/` from build directory
* Format code: `just format`
* There is a commit hook that runs `just pre-commit`, which runs tests and
  checks formatting. You may want to run it before attempting to commit.
* We use Google C++ style guide.
  * That means 80 columns.
  * That means comments should be in full sentences with periods in the end.
  * When conditional or loop fits one line, we prefere form without braces.
  * Prefer `absl` to `std` (e.g. `absl::c_` algorithms, `absl::Mutex`,
    `absl::StrCat`, etc.)
* We use `uv` for Python package and venv management, and to running the
  application.
* Run TUI app: `uv run python -m lczero_training` (skeleton mode) or
  `uv run python -m lczero_training config.yaml` (with config)
* To compile data loader:

```sh
CXX=clang++ CC=clang meson build/release/ --buildtype release
meson compile -C build/release/
cp build/release/lczero_training.so src/lczero_training/
```

## IMPORTANT

* NEVER add `# type: ignore` or other ways to mask/silence errors instead of
  fixing them.

## Documentation

* Documentation is in the `docs/` directory.
* The contents is in [The index](docs/index.md)
