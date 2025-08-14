# List available commands
default:
    @just --list

# Check if all C++ files in csrc/ are formatted according to clang-format
check-cpp:
    find csrc/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format --dry-run --Werror

# Format all C++ files in csrc/ using clang-format
format-cpp:
    find csrc/ -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -i

# Check if all protobuf files are formatted according to clang-format
check-proto:
    find proto/ -name "*.proto" | xargs clang-format --dry-run --Werror

# Format all protobuf files using clang-format
format-proto:
    find proto/ -name "*.proto" | xargs clang-format -i

# Check if all Python files in src/ are formatted according to ruff
check-python:
    source .venv/bin/activate && ruff check src/
    source .venv/bin/activate && ruff format --check src/
    source .venv/bin/activate && mypy -p lczero_training

# Format all Python files in src/ using ruff
format-python:
    source .venv/bin/activate && ruff format src/
    source .venv/bin/activate && ruff check --fix src/

format: format-cpp format-proto format-python

# Build the project
build:
    meson compile -C builddir/

# Run tests
test-cpp:
    uv run pytest

test-python:    
    meson test -C builddir/

test: test-cpp test-python    

check: check-cpp check-proto check-python

# Run all checks (formatting, build, and tests)
pre-commit: check build test