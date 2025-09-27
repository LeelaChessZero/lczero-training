"""Test script for the DataLoader implementation."""

from pathlib import Path

import proto.data_loader_config_pb2 as config_pb2
from lczero_training._lczero_training import DataLoader


def _make_basic_config(directory: str) -> config_pb2.DataLoaderConfig:
    config = config_pb2.DataLoaderConfig()

    file_stage = config.stage.add()
    file_stage.name = "file_path_provider"
    file_stage.file_path_provider.directory = directory

    chunk_stage = config.stage.add()
    chunk_stage.name = "chunk_source_loader"
    chunk_stage.chunk_source_loader.input = file_stage.name

    pool_stage = config.stage.add()
    pool_stage.name = "shuffling_chunk_pool"
    pool_stage.shuffling_chunk_pool.input = chunk_stage.name

    unpacker_stage = config.stage.add()
    unpacker_stage.name = "chunk_unpacker"
    unpacker_stage.chunk_unpacker.input = pool_stage.name

    sampler_stage = config.stage.add()
    sampler_stage.name = "shuffling_frame_sampler"
    sampler_stage.shuffling_frame_sampler.input = unpacker_stage.name

    tensor_stage = config.stage.add()
    tensor_stage.name = "tensor_generator"
    tensor_stage.tensor_generator.input = sampler_stage.name

    return config


def test_dataloader_initialization() -> None:
    """Test DataLoader can be created with valid directory config."""
    script_dir = Path(__file__).parent

    config = _make_basic_config(str(script_dir))

    loader = DataLoader(config)
    loader.start()
    assert loader is not None


def test_dataloader_methods_exist() -> None:
    """Test DataLoader methods exist and are callable."""
    script_dir = Path(__file__).parent

    config = _make_basic_config(str(script_dir))
    loader = DataLoader(config)
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
