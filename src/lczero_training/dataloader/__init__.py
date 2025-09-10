from lczero_training._lczero_training import DataLoader, TensorBase
from proto.data_loader_config_pb2 import DataLoaderConfig

__all__ = ["DataLoader", "make_dataloader", "TensorBase"]


def make_dataloader(config: DataLoaderConfig) -> DataLoader:
    data_loader_config_bytes = config.SerializeToString()
    loader = DataLoader(data_loader_config_bytes)
    loader.start()
    return loader
