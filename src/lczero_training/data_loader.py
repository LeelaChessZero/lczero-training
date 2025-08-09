# ABOUTME: High-level Python API wrapper for the C++ DataLoader class.
# ABOUTME: Provides convenient DataLoader interface with config validation and type hints.

from typing import Tuple
import numpy as np

from .data_loader_config import DataLoaderConfig, create_default_config
from . import _lczero_training  # type: ignore[attr-defined]


def _convert_python_config_to_cpp(
    config: DataLoaderConfig,
) -> _lczero_training.DataLoaderConfig:
    """Convert Python DataLoaderConfig to C++ DataLoaderConfig.

    Args:
        config: Python configuration dataclass

    Returns:
        C++ configuration object for pybind11 module
    """
    cpp_config = _lczero_training.DataLoaderConfig()

    # FilePathProviderOptions
    cpp_config.file_path_provider = _lczero_training.FilePathProviderOptions()
    cpp_config.file_path_provider.queue_capacity = (
        config.file_path_provider.queue_capacity
    )
    cpp_config.file_path_provider.directory = (
        config.file_path_provider.directory
    )

    # ChunkSourceLoaderOptions
    cpp_config.chunk_source_loader = _lczero_training.ChunkSourceLoaderOptions()
    cpp_config.chunk_source_loader.worker_threads = (
        config.chunk_source_loader.worker_threads
    )
    cpp_config.chunk_source_loader.output_queue_size = (
        config.chunk_source_loader.output_queue_size
    )

    # ShufflingChunkPoolOptions
    cpp_config.shuffling_chunk_pool = (
        _lczero_training.ShufflingChunkPoolOptions()
    )
    cpp_config.shuffling_chunk_pool.chunk_pool_size = (
        config.shuffling_chunk_pool.chunk_pool_size
    )
    cpp_config.shuffling_chunk_pool.num_startup_indexing_threads = (
        config.shuffling_chunk_pool.num_startup_indexing_threads
    )
    cpp_config.shuffling_chunk_pool.num_indexing_threads = (
        config.shuffling_chunk_pool.num_indexing_threads
    )
    cpp_config.shuffling_chunk_pool.num_chunk_loading_threads = (
        config.shuffling_chunk_pool.num_chunk_loading_threads
    )
    cpp_config.shuffling_chunk_pool.output_queue_size = (
        config.shuffling_chunk_pool.output_queue_size
    )

    # ChunkUnpackerOptions
    cpp_config.chunk_unpacker = _lczero_training.ChunkUnpackerOptions()
    cpp_config.chunk_unpacker.worker_threads = (
        config.chunk_unpacker.worker_threads
    )
    cpp_config.chunk_unpacker.output_queue_size = (
        config.chunk_unpacker.output_queue_size
    )

    # ShufflingFrameSamplerOptions
    cpp_config.shuffling_frame_sampler = (
        _lczero_training.ShufflingFrameSamplerOptions()
    )
    cpp_config.shuffling_frame_sampler.num_worker_threads = (
        config.shuffling_frame_sampler.num_worker_threads
    )
    cpp_config.shuffling_frame_sampler.reservoir_size_per_thread = (
        config.shuffling_frame_sampler.reservoir_size_per_thread
    )
    cpp_config.shuffling_frame_sampler.output_queue_size = (
        config.shuffling_frame_sampler.output_queue_size
    )

    # TensorGeneratorOptions
    cpp_config.tensor_generator = _lczero_training.TensorGeneratorOptions()
    cpp_config.tensor_generator.worker_threads = (
        config.tensor_generator.worker_threads
    )
    cpp_config.tensor_generator.batch_size = config.tensor_generator.batch_size
    cpp_config.tensor_generator.output_queue_size = (
        config.tensor_generator.output_queue_size
    )

    return cpp_config


class DataLoader:
    """High-level Python interface to the C++ DataLoader.

    This class provides a convenient Python API for loading training data
    from disk with built-in shuffling, batching, and tensor generation.

    The loader returns numpy arrays that implement the buffer protocol,
    making them compatible with JAX, PyTorch, and other ML frameworks.

    Example:
        config = create_default_config(
            directory="/path/to/training/data",
            chunk_pool_size=1000
        )
        loader = DataLoader(config)

        # Get next batch of tensors
        tensors = loader.get_next()

        # Convert to JAX arrays if needed
        import jax.numpy as jnp
        jax_tensors = [jnp.asarray(t) for t in tensors]
    """

    def __init__(self, config: DataLoaderConfig):
        """Initialize DataLoader with the given configuration.

        Args:
            config: DataLoaderConfig containing all component configurations

        Raises:
            ValueError: If configuration validation fails
            RuntimeError: If C++ DataLoader construction fails
        """
        # Validate configuration
        if not isinstance(config, DataLoaderConfig):
            raise ValueError("config must be a DataLoaderConfig instance")

        # Convert Python config to C++ and create DataLoader
        cpp_config = _convert_python_config_to_cpp(config)
        self._cpp_loader = _lczero_training.DataLoader(cpp_config)
        self._config = config

    def get_next(self) -> Tuple[np.ndarray, ...]:
        """Get the next batch of tensors.

        Returns a tuple of numpy arrays representing the next batch of training data.
        The arrays implement the buffer protocol and can be used directly with JAX,
        PyTorch, and other ML frameworks.

        Returns:
            Tuple of numpy arrays representing the batch tensors

        Raises:
            RuntimeError: If data loading fails or reaches end of data
        """
        return self._cpp_loader.get_next()

    @property
    def config(self) -> DataLoaderConfig:
        """Get the configuration used to create this DataLoader.

        Returns:
            The DataLoaderConfig used during initialization
        """
        return self._config

    def __iter__(self):
        """Make DataLoader iterable.

        Note: This creates an infinite iterator. The caller is responsible
        for stopping iteration when needed.
        """
        return self

    def __next__(self) -> Tuple[np.ndarray, ...]:
        """Get next batch for iterator protocol.

        Returns:
            Tuple of numpy arrays representing the batch tensors
        """
        return self.get_next()


# Convenience function for simple usage
def create_dataloader(
    directory: str, chunk_pool_size: int, **kwargs
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    This is a convenience function that creates a DataLoaderConfig with default
    values and allows overriding specific options via keyword arguments.

    Args:
        directory: Path to directory containing training data files
        chunk_pool_size: Size of the chunk shuffle buffer
        **kwargs: Additional configuration overrides (e.g., batch_size=2048)

    Returns:
        Configured DataLoader ready for use

    Example:
        # Simple usage with defaults
        loader = create_dataloader("/path/to/data", chunk_pool_size=1000)

        # With custom batch size
        loader = create_dataloader(
            "/path/to/data",
            chunk_pool_size=1000,
            batch_size=2048
        )
    """
    config = create_default_config(directory, chunk_pool_size)

    # Apply any configuration overrides
    if "batch_size" in kwargs:
        config.tensor_generator.batch_size = kwargs["batch_size"]
    if "worker_threads" in kwargs:
        # Apply to all components that have worker_threads
        config.chunk_source_loader.worker_threads = kwargs["worker_threads"]
        config.chunk_unpacker.worker_threads = kwargs["worker_threads"]
        config.tensor_generator.worker_threads = kwargs["worker_threads"]

    return DataLoader(config)
