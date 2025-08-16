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

# Build Python protobuf files
build-proto:
    mkdir -p src/lczero_training/proto
    touch src/lczero_training/proto/__init__.py
    uv run protoc --proto_path=proto --python_out=src/lczero_training/proto --mypy_out=src/lczero_training/proto proto/*.proto

# Check if all Python files in src/ are formatted according to ruff
check-python:
    uv run ruff check src/ --exclude src/lczero_training/proto
    uv run ruff check --select I src/ --exclude src/lczero_training/proto
    uv run ruff format --check src/ --exclude src/lczero_training/proto
    uv run mypy -p lczero_training --disallow-untyped-defs --disallow-incomplete-defs

# Format all Python files in src/ using ruff
format-python:
    uv run ruff check --fix --select I src/ --exclude src/lczero_training/proto
    uv run ruff format src/ --exclude src/lczero_training/proto
    uv run ruff check --fix src/ --exclude src/lczero_training/proto

format: format-cpp format-proto format-python

# Build the project
build:
    meson compile -C builddir/

# Run tests
test-cpp:
    meson test -C builddir/

test-python: 
    uv run pytest    

test: test-cpp test-python    

check: check-cpp check-proto check-python

# Run all checks (formatting, build, and tests)
pre-commit: check build-proto build test