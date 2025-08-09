# List available commands
default:
    @just --list

# Check if all C++ files in csrc/ are formatted according to clang-format
check-cpp:
    find csrc/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format --dry-run --Werror

# Format all C++ files in csrc/ using clang-format
format-cpp:
    find csrc/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -i

format: format-cpp

# Build the project
build:
    meson compile -C builddir/

# Run tests
test:
    meson test -C builddir/

check: check-cpp

# Run all checks (formatting, build, and tests)
pre-commit: check build test