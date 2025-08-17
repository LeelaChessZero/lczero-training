# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics.
# ABOUTME: Shows a grid of pipeline stages and queues with their metrics.

from textual.app import ComposeResult
from textual.containers import Container

from ..proto import training_metrics_pb2
from .stage_widgets import (
    ChunkSourceLoaderStage,
    ChunkUnpackerStage,
    FilePathProviderStage,
    QueueWidget,
    ShufflingChunkPoolStage,
    ShufflingFrameSamplerStage,
    TensorGeneratorStage,
)


class DataPipelinePane(Container):
    """Main pane showing data pipeline flow and statistics as a grid."""

    def compose(self) -> ComposeResult:
        # Create the pipeline stages and queues in order
        self.file_path_provider = FilePathProviderStage()
        yield self.file_path_provider
        yield QueueWidget()

        yield ChunkSourceLoaderStage()
        yield QueueWidget()

        yield ShufflingChunkPoolStage()
        yield QueueWidget()

        yield ChunkUnpackerStage()
        yield QueueWidget()

        yield ShufflingFrameSamplerStage()
        yield QueueWidget()

        yield TensorGeneratorStage()
        yield QueueWidget()

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the pipeline metrics display."""
        # Forward metrics to the FilePathProvider stage
        self.file_path_provider.update_metrics(
            dataloader_1_second, dataloader_total
        )
