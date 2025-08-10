# ABOUTME: TrainingDaemon class responsible for coordinating the training process.
# ABOUTME: Takes RootConfig and creates DataLoader for training data management.

from ..config.root_config import RootConfig
from ..data_loader import DataLoader


class TrainingDaemon:
    """TrainingDaemon coordinates the training process.

    This is the main entry point for the training system that:
    - Takes configuration through RootConfig
    - Creates and manages DataLoader for training data
    - (Future) Will coordinate training loops and model export
    """

    def __init__(self, config: RootConfig):
        """Initialize TrainingDaemon with configuration.

        Args:
            config: RootConfig containing all training system configuration
        """
        self._config = config
        self._data_loader = DataLoader(config.data_loader)

    @property
    def config(self) -> RootConfig:
        """Get the root configuration."""
        return self._config

    @property
    def data_loader(self) -> DataLoader:
        """Get the data loader instance."""
        return self._data_loader
