# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics.
# ABOUTME: Shows a grid of pipeline stages and queues with their metrics.

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container

import proto.training_metrics_pb2 as training_metrics_pb2

from .dataloader_widgets import (
    ChunkSourceLoaderStageWidget,
    MetricsStageWidget,
    QueueWidget,
    ShufflingChunkPoolStageWidget,
    StageWidget,
)


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""

    metrics_field: str
    stage_name: str
    item_name: str


STAGES_CONFIG = [
    StageConfig("file_path_provider", "File discovery", "Files"),
    StageConfig("chunk_source_loader", "Chunk source loader", "Files"),
    StageConfig("shuffling_chunk_pool", "Shuffling chunk pool", "Chunks"),
    StageConfig("chunk_splitter", "Chunk splitter", "Frames"),
    StageConfig("shuffling_frame_sampler", "Shuffling frame sampler", "Frames"),
    StageConfig("tensor_generator", "Batched tensor generator", "Tensors"),
]


class DataPipelinePane(Container):
    """Main pane showing data pipeline flow and statistics as a grid."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stages: list[StageWidget] = []
        self._queues: list[QueueWidget] = []

    def compose(self) -> ComposeResult:
        """Create the pipeline stages and queues from a config."""
        for config in STAGES_CONFIG:
            if config.metrics_field == "shuffling_chunk_pool":
                stage: StageWidget = ShufflingChunkPoolStageWidget(
                    stage_name=config.stage_name,
                    metrics_field_name=config.metrics_field,
                    item_name=config.item_name,
                )
            elif config.metrics_field == "chunk_source_loader":
                stage = ChunkSourceLoaderStageWidget(
                    stage_name=config.stage_name,
                    metrics_field_name=config.metrics_field,
                    item_name=config.item_name,
                )
            else:
                stage = MetricsStageWidget(
                    stage_name=config.stage_name,
                    metrics_field_name=config.metrics_field,
                    item_name=config.item_name,
                )
            self._stages.append(stage)
            yield stage

            queue = QueueWidget(
                item_name=config.item_name,
                stage_key=config.metrics_field,
                stage_name=f"{config.stage_name} queue",
            )
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

        for queue in self._queues:
            queue.update_metrics(dataloader_1_second, dataloader_total)
