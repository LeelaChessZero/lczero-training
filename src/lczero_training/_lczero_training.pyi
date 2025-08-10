from typing import Tuple
import numpy as np

class FilePathProviderOptions:
    queue_capacity: int
    directory: str

class ChunkSourceLoaderOptions:
    worker_threads: int
    output_queue_size: int

class ShufflingChunkPoolOptions:
    chunk_pool_size: int
    num_startup_indexing_threads: int
    num_indexing_threads: int
    num_chunk_loading_threads: int
    output_queue_size: int

class ChunkUnpackerOptions:
    worker_threads: int
    output_queue_size: int

class ShufflingFrameSamplerOptions:
    num_worker_threads: int
    reservoir_size_per_thread: int
    output_queue_size: int

class TensorGeneratorOptions:
    worker_threads: int
    batch_size: int
    output_queue_size: int

class DataLoaderConfig:
    file_path_provider: FilePathProviderOptions
    chunk_source_loader: ChunkSourceLoaderOptions
    shuffling_chunk_pool: ShufflingChunkPoolOptions
    chunk_unpacker: ChunkUnpackerOptions
    shuffling_frame_sampler: ShufflingFrameSamplerOptions
    tensor_generator: TensorGeneratorOptions

class DataLoader:
    def __init__(self, config: DataLoaderConfig) -> None: ...
    def get_next(self) -> Tuple[np.ndarray, ...]: ...
