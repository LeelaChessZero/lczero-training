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
    protoc \
        --proto_path=. \
        --python_out=src/ \
        --pyi_out=src/ \
        -I libs/lczero-common/ \
        -I libs/lc0/src/neural/xla/ \
        proto/*.proto
    protoc \
        --proto_path=libs/lczero-common/ \
        --python_out=src/ \
        --pyi_out=src/ \
        libs/lczero-common/proto/*.proto
    protoc \
        --proto_path=libs/lc0/src/neural/xla/ \
        --python_out=src/ \
        --pyi_out=src/ \
        libs/lc0/src/neural/xla/hlo.proto

# Check if all Python files in src/ are formatted according to ruff
check-python:
    uv run ruff check src/
    uv run ruff check --select I src/
    uv run ruff format --check src/
    uv run mypy -p lczero_training --disallow-untyped-defs --disallow-incomplete-defs

# Format all Python files in src/ using ruff
format-python:
    uv run ruff check --fix --select I src/
    uv run ruff format src/
    uv run ruff check --fix src/

format: format-cpp format-proto format-python

# Build the project
build:
    meson compile -C build/release/

# Run tests
test-cpp:
    meson test -C build/release/

test-python: 
    uv run pytest    

test: test-cpp test-python    

check: check-cpp check-proto check-python

# Run all checks (formatting, build, and tests)
pre-commit: check build-proto build test
