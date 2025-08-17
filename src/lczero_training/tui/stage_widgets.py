# ABOUTME: Stage widgets for the data pipeline visualization
# ABOUTME: Each stage represents a different part of the data loading process

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ..proto import training_metrics_pb2


class StageWidget(Static):
    """Base class for all data pipeline stage widgets."""

    def __init__(self, stage_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stage_name = stage_name
        self.border_title = stage_name

    def compose(self) -> ComposeResult:
        yield Static(
            "Waiting for metrics...",
            classes="stage-content",
        )


class QueueWidget(Static):
    """Widget for displaying queue metrics between stages."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "Queue"

    def compose(self) -> ComposeResult:
        yield Static(
            "Queue: --",
            classes="queue-content",
        )


class FilePathProviderStage(StageWidget):
    """First stage: Training data discovery worker."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("FilePathProvider", **kwargs)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display."""
        if not dataloader_1_second or not dataloader_total:
            return

        try:
            fps_1sec = dataloader_1_second.file_path_provider
            fps_total = dataloader_total.file_path_provider

            total_files = fps_total.queue.message_count
            files_per_sec = fps_1sec.queue.message_count
            load_seconds = fps_1sec.load.load_seconds
            total_seconds = fps_1sec.load.total_seconds

            content = (
                f"Files Found: {total_files}\n"
                f"Rate: {files_per_sec}/s\n"
                f"Load: {load_seconds:.1f}/{total_seconds:.1f}s"
            )

            stage_content = self.query_one(".stage-content", Static)
            stage_content.update(content)

        except Exception:
            stage_content = self.query_one(".stage-content", Static)
            stage_content.update("Error: Invalid metrics")


class ChunkSourceLoaderStage(StageWidget):
    """Second stage: Reads chunks from files."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ChunkSourceLoader", **kwargs)


class ShufflingChunkPoolStage(StageWidget):
    """Third stage: Manages chunk pool with shuffling."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ShufflingChunkPool", **kwargs)


class ChunkUnpackerStage(StageWidget):
    """Fourth stage: Unpacks chunks into frames."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ChunkUnpacker", **kwargs)


class ShufflingFrameSamplerStage(StageWidget):
    """Fifth stage: Provides shuffled frame batches."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ShufflingFrameSampler", **kwargs)


class TensorGeneratorStage(StageWidget):
    """Sixth stage: Generates tensor buffers."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("TensorGenerator", **kwargs)
