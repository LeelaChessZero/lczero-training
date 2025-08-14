"""Test protobuf compilation and functionality."""

import pytest


def test_protobuf_import():
    """Test that protobuf files can be imported."""
    from proto import data_loader_config_pb2
    from proto import data_loader_metrics_pb2
    
    # Test creating config objects
    config = data_loader_config_pb2.DataLoaderConfig()
    assert config is not None
    
    metrics = data_loader_metrics_pb2.DataLoaderMetricsProto()
    assert metrics is not None


def test_protobuf_functionality():
    """Test basic protobuf functionality."""
    from proto import data_loader_config_pb2
    
    # Create a config and set some values
    config = data_loader_config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = "/test/path"
    
    # Serialize and deserialize
    serialized = config.SerializeToString()
    assert len(serialized) > 0
    
    config2 = data_loader_config_pb2.DataLoaderConfig()
    config2.ParseFromString(serialized)
    
    assert config2.file_path_provider.directory == "/test/path"