# CLAUDE.md

This repository contains training script for the Leela Chess Zero project.
They are being rewritten.

* Old code is located in the `tf/` directory.
* New code is located in the `src/` directory.

The old code is Python/TensorFlow-based, new code is Python/JAX-based with
modules written in C++, operating through pybind11.

The build system for C++ code is meson. During development, the project is built
in the `builddir/`.

## Testing and Building

* C++ tests use GTest framework
* Tests are defined in `meson.build` with `test()` function 
* Run tests: `meson test -C builddir/`
* Build: `meson compile -C builddir/` from build directory
* Format C++ code: `just format-cpp`
* We use Google C++ style guide.
  * That means 80 columns.
  * That means comments should be in full sentences with periods in the end.

## Documentation

* Documentation is in the `docs/` directory.
* The contents is in [The index](docs/index.md)
