# ABOUTME: Root configuration dataclass containing all training system configuration.
# ABOUTME: Provides unified config structure for data loading, training, and export parameters.

from dataclasses import dataclass
from .data_loader_config import DataLoaderConfig, create_default_config


@dataclass
class RootConfig:
    """Root configuration class containing all training system configuration.

    This is the top-level configuration structure that encompasses:
    - Data loader configuration for training data ingestion
    - (Future) Training coordinator configuration
    - (Future) Model definition and architecture parameters
    - (Future) Training parameters like batch size, epochs, etc.
    - (Future) Export parameters for model output
    """

    data_loader: DataLoaderConfig

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "RootConfig":
        """Load RootConfig from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            RootConfig instance populated from YAML

        Raises:
            ConfigParseError: If parsing fails due to validation errors
        """
        from .yaml_parser import from_yaml_file

        return from_yaml_file(file_path, cls)


def create_default_root_config(
    directory: str, chunk_pool_size: int
) -> RootConfig:
    """Create a RootConfig with default values for most settings.

    Args:
        directory: Directory path containing training data files
        chunk_pool_size: Size of the chunk shuffle buffer for data loader

    Returns:
        RootConfig with sensible defaults for all components
    """
    return RootConfig(
        data_loader=create_default_config(directory, chunk_pool_size)
    )
