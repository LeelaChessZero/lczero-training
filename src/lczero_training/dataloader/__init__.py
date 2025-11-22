from lczero_training._lczero_training import (
    DataLoader,
    TensorBase,
    TrainingTensors,
)
from proto.data_loader_config_pb2 import DataLoaderConfig

__all__ = ["DataLoader", "make_dataloader", "TensorBase", "TrainingTensors"]


def make_dataloader(config: DataLoaderConfig) -> DataLoader:
    loader = DataLoader(config)
    loader.start()
    return loader
