"""Test script for the DataLoader implementation."""

import sys
from lczero_training._lczero_training import DataLoader
import proto.data_loader_config_pb2 as config_pb2
from pathlib import Path
import pytest

# Add the src and build directories to Python path
src_dir = Path(__file__).parent / "src"
build_dir = Path(__file__).parent / "builddir"
proto_dir = (
    Path(__file__).parent
    / "builddir"
    / "_lczero_training.cpython-311-x86_64-linux-gnu.so.p"
)

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
if build_dir.exists():
    sys.path.insert(0, str(build_dir))
if proto_dir.exists():
    sys.path.insert(0, str(proto_dir))


@pytest.mark.skip(reason="DataLoader hangs waiting for training data files")
def test_dataloader_initialization():
    """Test DataLoader can be created with valid directory config."""
    script_dir = Path(__file__).parent

    config = config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = str(script_dir)

    config_bytes = config.SerializeToString()
    loader = DataLoader(config_bytes)
    assert loader is not None


@pytest.mark.skip(reason="DataLoader hangs waiting for training data files")
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
