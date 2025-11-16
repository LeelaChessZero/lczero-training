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
    mkdir -p src/proto
    touch src/proto/__init__.py
    uv run python -m grpc_tools.protoc \
        --proto_path=. \
        --proto_path=libs/lc0 \
        --python_out=src/ \
        --pyi_out=src/ \
        proto/*.proto
    uv run python -m grpc_tools.protoc \
        --proto_path=. \
        --proto_path=libs/lc0 \
        --python_out=src/ \
        --pyi_out=src/ \
        proto/net.proto \
        proto/onnx.proto \
        proto/hlo.proto

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
    uv run meson compile -C build/release/

# Run tests
test-cpp:
    uv run meson test -C build/release/

test-python: 
    uv run pytest    

test: test-cpp test-python    

check: check-cpp check-proto check-python

# Run all checks (formatting, build, and tests)
pre-commit: build-proto check build test
