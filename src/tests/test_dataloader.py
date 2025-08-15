"""Test script for the DataLoader implementation."""

from lczero_training._lczero_training import DataLoader
import proto.data_loader_config_pb2 as config_pb2
from pathlib import Path


def test_dataloader_initialization():
    """Test DataLoader can be created with valid directory config."""
    script_dir = Path(__file__).parent

    config = config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = str(script_dir)

    config_bytes = config.SerializeToString()
    loader = DataLoader(config_bytes)
    assert loader is not None


def test_dataloader_methods_exist():
    """Test DataLoader methods exist and are callable."""
    script_dir = Path(__file__).parent

    config = config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = str(script_dir)
    config_bytes = config.SerializeToString()
    loader = DataLoader(config_bytes)

    assert hasattr(loader, "get_next")
    assert hasattr(loader, "get_stat")
    assert callable(loader.get_next)
    assert callable(loader.get_stat)
