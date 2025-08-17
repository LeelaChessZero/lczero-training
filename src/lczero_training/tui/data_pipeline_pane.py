# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics.
# ABOUTME: Shows a grid of pipeline stages and queues with their metrics.

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container

from ..proto import training_metrics_pb2
from .stage_widgets import MetricsStageWidget, QueueWidget

STAGES_CONFIG = [
    ("file_path_provider", "File discovery", "Files"),
    ("chunk_source_loader", "Chunk source loader", "Chunks"),
    ("shuffling_chunk_pool", "Shuffling chunk pool", "Chunks"),
    ("chunk_unpacker", "Chunk unpacker", "Frames"),
    ("shuffling_frame_sampler", "Shuffling frame sampler", "Frames"),
    ("tensor_generator", "Tensor generator", "Tensors"),
]


class DataPipelinePane(Container):
    """Main pane showing data pipeline flow and statistics as a grid."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stages: list[MetricsStageWidget] = []
        self._queues: list[QueueWidget] = []

    def compose(self) -> ComposeResult:
        """Create the pipeline stages and queues from a config."""
        for metrics_field, stage_name, item_name in STAGES_CONFIG:
            stage = MetricsStageWidget(
                stage_name=stage_name,
                metrics_field_name=metrics_field,
                item_name=item_name,
            )
            self._stages.append(stage)
            yield stage

            queue = QueueWidget()
            self._queues.append(queue)
            yield queue

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update all pipeline stages and queues with new metrics."""
        for stage in self._stages:
            stage.update_metrics(dataloader_1_second, dataloader_total)

        for i, queue in enumerate(self._queues):
            queue_field_name = STAGES_CONFIG[i][0]
            queue.update_metrics(
                dataloader_1_second, dataloader_total, queue_field_name
            )
