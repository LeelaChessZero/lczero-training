#!/usr/bin/env python3
"""Test script for the DataLoader implementation."""

import os
import sys
import tempfile
from pathlib import Path

# Add the src and build directories to Python path
src_dir = Path(__file__).parent / "src"
build_dir = Path(__file__).parent / "builddir"

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
if build_dir.exists():
    sys.path.insert(0, str(build_dir))

try:
    from lczero_training import DataLoader, create_default_config, create_dataloader
    print("✓ Successfully imported DataLoader")
except ImportError as e:
    print(f"✗ Failed to import DataLoader: {e}")
    sys.exit(1)

def test_basic_config():
    """Test basic configuration creation."""
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_default_config(
                directory=temp_dir,
                chunk_pool_size=100
            )
            print("✓ Successfully created default config")
            print(f"  Directory: {config.file_path_provider.directory}")
            print(f"  Chunk pool size: {config.shuffling_chunk_pool.chunk_pool_size}")
            print(f"  Batch size: {config.tensor_generator.batch_size}")
    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        return False
    return True

def test_dataloader_creation():
    """Test DataLoader creation."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_default_config(
                directory=temp_dir,
                chunk_pool_size=10  # Small for testing
            )
            loader = DataLoader(config)
            print("✓ Successfully created DataLoader")
            print(f"  Config directory: {loader.config.file_path_provider.directory}")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return False
    return True

def test_convenience_function():
    """Test the convenience create_dataloader function."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = create_dataloader(
                directory=temp_dir,
                chunk_pool_size=10,
                batch_size=512  # Custom batch size
            )
            print("✓ Successfully created DataLoader with create_dataloader()")
            print(f"  Batch size: {loader.config.tensor_generator.batch_size}")
    except Exception as e:
        print(f"✗ Failed with create_dataloader(): {e}")
        return False
    return True

def main():
    """Run all tests."""
    print("Testing DataLoader implementation...")
    print("=" * 50)
    
    tests = [
        test_basic_config,
        test_dataloader_creation,
        test_convenience_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 4 implementation is working.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())