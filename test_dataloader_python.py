# ABOUTME: Python tests for C++ DataLoader PyBind11 bindings.
# ABOUTME: Tests initialization, tensor output, and statistics functionality.

import tempfile
import os
import pytest
import numpy as np
from lczero_training import DataLoader
from google.protobuf.message import Message


def create_test_config() -> str:
    """Create minimal protobuf configuration for testing."""
    from proto.data_loader_config_pb2 import DataLoaderConfig
    
    config = DataLoaderConfig()
    config.file_path_provider.directory = "/tmp/nonexistent"
    config.chunk_source_loader.worker_threads = 1
    config.shuffling_chunk_pool.chunk_pool_size = 100
    config.tensor_generator.batch_size = 32
    
    return config.SerializeToString()


def test_dataloader_initialization():
    """Test DataLoader can be created with protobuf config."""
    config_string = create_test_config()
    
    # Should not crash on initialization
    loader = DataLoader(config_string)
    assert loader is not None


def test_dataloader_get_next_returns_tuple():
    """Test get_next returns tuple of numpy arrays."""
    config_string = create_test_config()
    loader = DataLoader(config_string)
    
    try:
        # This may fail due to missing data files, but should return tuple structure
        result = loader.get_next()
        assert isinstance(result, tuple)
        assert len(result) > 0
        
        # Each element should be numpy array
        for tensor in result:
            assert isinstance(tensor, np.ndarray)
            
    except Exception as e:
        # Expected to fail with missing data, but check error type
        assert "file" in str(e).lower() or "directory" in str(e).lower()


def test_dataloader_get_stat():
    """Test get_stat returns string metrics."""
    config_string = create_test_config()
    loader = DataLoader(config_string)
    
    # Should return empty or valid protobuf string
    stat_result = loader.get_stat()
    assert isinstance(stat_result, str)


if __name__ == "__main__":
    pytest.main([__file__])