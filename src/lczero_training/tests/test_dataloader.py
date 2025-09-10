"""Test script for the DataLoader implementation."""

from pathlib import Path

import proto.data_loader_config_pb2 as config_pb2
from lczero_training._lczero_training import DataLoader


def test_dataloader_initialization() -> None:
    """Test DataLoader can be created with valid directory config."""
    script_dir = Path(__file__).parent

    config = config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = str(script_dir)

    config_bytes = config.SerializeToString()
    loader = DataLoader(config_bytes)
    loader.start()
    assert loader is not None


def test_dataloader_methods_exist() -> None:
    """Test DataLoader methods exist and are callable."""
    script_dir = Path(__file__).parent

    config = config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = str(script_dir)
    config_bytes = config.SerializeToString()
    loader = DataLoader(config_bytes)
    loader.start()

    assert hasattr(loader, "get_next")
    assert hasattr(loader, "get_bucket_metrics")
    assert hasattr(loader, "get_aggregate_ending_now")
    assert hasattr(loader, "start")
    assert hasattr(loader, "stop")
    assert callable(loader.get_next)
    assert callable(loader.get_bucket_metrics)
    assert callable(loader.get_aggregate_ending_now)
    assert callable(loader.start)
    assert callable(loader.stop)
