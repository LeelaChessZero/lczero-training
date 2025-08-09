# ABOUTME: Python configuration dataclasses mirroring C++ configuration structures.
# ABOUTME: These classes provide type-safe Python interfaces to DataLoader configuration.

from dataclasses import dataclass


@dataclass
class FilePathProviderConfig:
    """Configuration for file path provider that watches directories for new training files.

    Maps to FilePathProviderOptions in csrc/loader/chunk_feed/file_path_provider.h
    """

    directory: str  # Path to directory containing training data files
    queue_capacity: int = 16  # Size of the internal file queue


@dataclass
class ChunkSourceLoaderConfig:
    """Configuration for chunk source loader that converts file paths to chunk sources.

    Maps to ChunkSourceLoaderOptions in csrc/loader/chunk_feed/chunk_source_loader.h
    """

    worker_threads: int = 1  # Number of worker threads for loading
    output_queue_size: int = 16  # Size of the output queue


@dataclass
class ShufflingChunkPoolConfig:
    """Configuration for shuffling chunk pool that manages chunk shuffling and loading.

    Maps to ShufflingChunkPoolOptions in csrc/loader/chunk_feed/shuffling_chunk_pool.h
    """

    chunk_pool_size: int  # Size of the chunk shuffle buffer (required)
    num_startup_indexing_threads: int = (
        4  # Threads used during startup indexing
    )
    num_indexing_threads: int = 4  # Threads for ongoing indexing operations
    num_chunk_loading_threads: int = 4  # Threads for loading chunk data
    output_queue_size: int = 16  # Size of the output queue


@dataclass
class ChunkUnpackerConfig:
    """Configuration for chunk unpacker that extracts frames from packed chunks.

    Maps to ChunkUnpackerOptions in csrc/loader/chunk_feed/chunk_unpacker.h
    """

    worker_threads: int = 1  # Number of worker threads for unpacking
    output_queue_size: int = 16  # Size of the output queue


@dataclass
class ShufflingFrameSamplerConfig:
    """Configuration for shuffling frame sampler using reservoir sampling.

    Maps to ShufflingFrameSamplerOptions in csrc/loader/shuffling_frame_sampler.h
    """

    num_worker_threads: int = 1  # Number of worker threads
    reservoir_size_per_thread: int = (
        1000000  # Size of sampling reservoir per thread
    )
    output_queue_size: int = 16  # Size of the output queue


@dataclass
class TensorGeneratorConfig:
    """Configuration for tensor generator that converts frames to batched tensors.

    Maps to TensorGeneratorOptions in csrc/loader/tensor_generator.h
    """

    worker_threads: int = 1  # Number of worker threads for tensor generation
    batch_size: int = 1024  # Batch size for tensor generation
    output_queue_size: int = 4  # Size of the output queue


@dataclass
class DataLoaderConfig:
    """Main configuration class for the DataLoader containing all component configurations.

    Maps to DataLoaderConfig in csrc/loader/data_loader.h
    """

    file_path_provider: FilePathProviderConfig
    chunk_source_loader: ChunkSourceLoaderConfig
    shuffling_chunk_pool: ShufflingChunkPoolConfig
    chunk_unpacker: ChunkUnpackerConfig
    shuffling_frame_sampler: ShufflingFrameSamplerConfig
    tensor_generator: TensorGeneratorConfig


def create_default_config(
    directory: str, chunk_pool_size: int
) -> DataLoaderConfig:
    """Create a DataLoaderConfig with default values for most settings.

    Args:
        directory: Directory path containing training data files
        chunk_pool_size: Size of the chunk shuffle buffer

    Returns:
        DataLoaderConfig with sensible defaults
    """
    return DataLoaderConfig(
        file_path_provider=FilePathProviderConfig(directory=directory),
        chunk_source_loader=ChunkSourceLoaderConfig(),
        shuffling_chunk_pool=ShufflingChunkPoolConfig(
            chunk_pool_size=chunk_pool_size
        ),
        chunk_unpacker=ChunkUnpackerConfig(),
        shuffling_frame_sampler=ShufflingFrameSamplerConfig(),
        tensor_generator=TensorGeneratorConfig(),
    )
