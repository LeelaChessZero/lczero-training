# List available commands
default:
    @just --list

# Check if all C++ files in src/ are formatted according to clang-format
check-cpp:
    find src/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format --dry-run --Werror

# Format all C++ files in src/ using clang-format
format-cpp:
    find src/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -i

# Run tests
test:
    meson test -C builddir/

# Run all checks (formatting and tests)
pre-commit: check-cpp test