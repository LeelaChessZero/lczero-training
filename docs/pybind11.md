# PyBind11 DataLoader Integration Plan

## Overview
This document outlines the plan to expose the C++ `DataLoader` class from `csrc/loader/data_loader.h` to Python through pybind11, targeting JAX tensors (using numpy buffer protocol initially), with updated project structure and memory management approach.

## Project Structure Changes

The project will be restructured to separate C++ and Python code:
- Current `src/` → `csrc/` (C++ source code)
- New `src/` → Python package code
- Build system updated to reflect new structure

## Implementation Phases

### Phase 1: Project Restructuring
1. **Move C++ Code**
   - Move current `src/` directory to `csrc/`
   - Update `meson.build` to reference `csrc/` instead of `src/`
   - Update include directories from `src/` to `csrc/`
   - Verify build still works after restructuring

2. **Create New Python Structure**
   - Create new `src/` directory for Python code
   - Set up Python package structure

### Phase 2: Python Environment & Dependencies
1. **Virtual Environment Setup**
   - Create `uv` virtual environment
   - Install pybind11, numpy as initial dependencies
   - Plan for JAX integration (but start with numpy for easier dependency management)
   - Install ruff to ensure formatting and linting, mypy for type checking
   - Update justfile to have python checks and formatting like for C++

### Phase 3: Python Configuration Layer
1. **Create Configuration Dataclasses**
   Create `src/data_loader_config.py` with dataclasses mirroring C++ structures:
   - `FilePathProviderConfig` (maps to `FilePathProviderOptions`)
   - `ChunkSourceLoaderConfig` (maps to `ChunkSourceLoaderOptions`) 
   - `ShufflingChunkPoolConfig` (maps to `ShufflingChunkPoolOptions`)
   - `ChunkUnpackerConfig` (maps to `ChunkUnpackerOptions`)
   - `ShufflingFrameSamplerConfig` (maps to `ShufflingFrameSamplerOptions`)
   - `TensorGeneratorConfig` (maps to `TensorGeneratorOptions`)
   - `DataLoaderConfig` (maps to `DataLoaderConfig`)

### Phase 4: PyBind11 Module Implementation
1. **Create Binding Module**
   Create `src/pybind_module.cc` that:
   - Exposes configuration classes with proper field mappings
   - Exposes `DataLoader` constructor accepting Python config
   - Exposes `GetNext()` method returning tuple of numpy arrays
   - Uses `py::return_value_policy::take_ownership` for tensor memory management
   - Releases tensors from `unique_ptr` before passing to Python

2. **Memory Management Strategy**
   - Extract `TensorBase` pointers from `TensorTuple` unique_ptrs using `.release()`
   - Pass raw pointers to pybind11 with `take_ownership` policy
   - Leverage existing buffer protocol implementation in `TensorBase`

### Phase 5: Build System Updates
1. **Update Meson Configuration**
   - Add pybind11 dependency detection
   - Create Python extension module target
   - Handle Python development headers and library detection

### Phase 6: Python Interface Layer
1. **High-level Python API**
   Create `src/data_loader.py` with:
   - Convenient DataLoader wrapper class
   - Configuration validation and defaults
   - Type hints for better development experience
   - JAX-compatible tensor handling (future-proofed)

2. **Package Structure**
   - `src/__init__.py` - Package initialization and exports
   - Proper module organization for clean imports

### Phase 7: Documentation & Testing
1. **Basic Testing**
   - Simple Python test to verify DataLoader instantiation
   - Test configuration passing and validation
   - Verify tensor output format and buffer protocol compatibility

## Technical Implementation Details

### Memory Management
- **No Shared Pointers**: Use `unique_ptr::release()` + `py::return_value_policy::take_ownership`
- **Threading**: Single-threaded frontend interface, no special GIL handling needed
- **Buffer Protocol**: Leverage existing `TensorBase` implementation for numpy/JAX compatibility

### Configuration Mapping
Each C++ configuration structure will have a corresponding Python dataclass:

```cpp
// C++ (csrc/loader/chunk_feed/file_path_provider.h)
struct FilePathProviderOptions {
  size_t queue_capacity = 16;
  std::filesystem::path directory;
};
```

```python
# Python (src/data_loader_config.py)
@dataclass
class FilePathProviderConfig:
    queue_capacity: int = 16
    directory: str
```

### Tensor Output Format
The `GetNext()` method will return a tuple of numpy arrays that implement the buffer protocol, making them compatible with both numpy and JAX:

```python
# Example usage
loader = DataLoader(config)
tensors = loader.get_next()  # Returns tuple of numpy arrays
# Can be used directly with JAX
jax_arrays = [jax.numpy.asarray(t) for t in tensors]
```

### JAX Integration
- Initially implement with numpy arrays for easier dependency management
- Structure code to be JAX-compatible from the start
- Future migration to native JAX arrays will be straightforward due to buffer protocol compatibility

## File Structure After Implementation

```
lczero-training/
├── csrc/                    # C++ source code (moved from src/)
│   ├── loader/
│   │   ├── data_loader.h
│   │   ├── data_loader.cc
│   │   └── ...
│   └── utils/
│       └── ...
├── src/                     # Python package
│   ├── __init__.py
│   ├── data_loader.py       # High-level Python API
│   ├── data_loader_config.py # Configuration dataclasses
│   └── pybind_module.cc     # PyBind11 bindings
├── docs/
│   └── pybind11.md         # This documentation
└── meson.build             # Updated build configuration
```

## Installation and Usage (Future)

After implementation, users will be able to:

1. **Install the package**:
   ```bash
   uv venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Use the DataLoader**:
   ```python
   from lczero_training import DataLoader, DataLoaderConfig
   from lczero_training.data_loader_config import FilePathProviderConfig
   
   config = DataLoaderConfig(
       file_path_provider=FilePathProviderConfig(
           directory="/path/to/training/data",
           queue_capacity=32
       ),
       # ... other configuration options
   )
   
   loader = DataLoader(config)
   while True:
       tensors = loader.get_next()  # Returns tuple of numpy arrays
       # Process tensors with JAX, PyTorch, etc.
   ```

## Next Steps

The implementation will proceed phase by phase, starting with the project restructuring and ending with a fully functional Python interface to the DataLoader system.