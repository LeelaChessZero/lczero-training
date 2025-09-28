# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics.
# ABOUTME: Shows a grid of pipeline stages and queues with their metrics.

from collections.abc import Iterable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container

import proto.training_metrics_pb2 as training_metrics_pb2

from .dataloader_widgets import (
    ChunkRescorerStageWidget,
    ChunkSourceLoaderStageWidget,
    MetricsStageWidget,
    QueueWidget,
    ShufflingChunkPoolStageWidget,
    StageWidget,
)

FRIENDLY_STAGE_NAMES = {
    "file_path_provider": "File discovery",
    "chunk_source_loader": "Chunk source loader",
    "shuffling_chunk_pool": "Shuffling chunk pool",
    "chunk_rescorer": "Chunk rescorer",
    "chunk_splitter": "Chunk splitter",
    "chunk_unpacker": "Chunk unpacker",
    "shuffling_frame_sampler": "Shuffling frame sampler",
    "tensor_generator": "Batched tensor generator",
}


ITEM_NAMES = {
    "file_path_provider": "Files",
    "chunk_source_loader": "Files",
    "shuffling_chunk_pool": "Chunks",
    "chunk_rescorer": "Chunks",
    "chunk_splitter": "Frames",
    "chunk_unpacker": "Frames",
    "shuffling_frame_sampler": "Frames",
    "tensor_generator": "Tensors",
}


class DataPipelinePane(Container):
    """Main pane showing data pipeline flow and statistics as a grid."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stage_widgets: dict[str, StageWidget] = {}
        self._queue_widgets: dict[str, QueueWidget] = {}
        self._stage_order: list[str] = []

    def compose(self) -> ComposeResult:
        """The pane starts empty and rows are added when metrics arrive."""
        yield from ()

    def _friendly_title(self, stage_key: str) -> str:
        return FRIENDLY_STAGE_NAMES.get(
            stage_key, stage_key.replace("_", " ").title()
        )

    @staticmethod
    def _detect_metrics_field(
        stage_metric: training_metrics_pb2.StageMetricProto,
    ) -> str | None:
        for descriptor, _ in stage_metric.ListFields():
            if descriptor.name not in {"name", "output_queue_metrics"}:
                return descriptor.name
        return None

    def _build_stage_widget(
        self,
        stage_key: str,
        metrics_field: str,
        item_name: str,
    ) -> StageWidget:
        friendly = self._friendly_title(stage_key)
        if metrics_field == "shuffling_chunk_pool":
            return ShufflingChunkPoolStageWidget(
                stage_name=friendly,
                metrics_field_name=metrics_field,
                item_name=item_name,
            )
        if metrics_field == "chunk_source_loader":
            return ChunkSourceLoaderStageWidget(
                stage_name=friendly,
                metrics_field_name=metrics_field,
                item_name=item_name,
            )
        if metrics_field == "chunk_rescorer":
            return ChunkRescorerStageWidget(
                stage_name=friendly,
                metrics_field_name=metrics_field,
                item_name=item_name,
            )
        return MetricsStageWidget(
            stage_name=friendly,
            metrics_field_name=metrics_field,
            item_name=item_name,
        )

    def _mount_widgets(
        self, widgets: Iterable[StageWidget | QueueWidget]
    ) -> None:
        async def _do_mount() -> None:
            await self.mount(*widgets)

        self.call_later(_do_mount)

    def _ensure_rows(
        self,
        metrics: training_metrics_pb2.DataLoaderMetricsProto,
    ) -> None:
        new_widgets: list[StageWidget | QueueWidget] = []
        for stage_metric in metrics.stage_metrics:
            stage_key = stage_metric.name
            if not stage_key or stage_key in self._stage_widgets:
                continue

            metrics_field = (
                self._detect_metrics_field(stage_metric) or stage_key
            )
            item_name = ITEM_NAMES.get(metrics_field, "Items")

            stage_widget = self._build_stage_widget(
                stage_key=stage_key,
                metrics_field=metrics_field,
                item_name=item_name,
            )
            queue_widget = QueueWidget(
                item_name=item_name,
                stage_key=stage_key,
                stage_name=f"{self._friendly_title(stage_key)} queue",
            )

            self._stage_widgets[stage_key] = stage_widget
            self._queue_widgets[stage_key] = queue_widget
            self._stage_order.append(stage_key)
            new_widgets.extend([stage_widget, queue_widget])

        if new_widgets:
            self._mount_widgets(new_widgets)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update all pipeline stages and queues with new metrics."""

        metrics_for_layout = dataloader_total or dataloader_1_second
        if metrics_for_layout:
            self._ensure_rows(metrics_for_layout)

        for stage_key in self._stage_order:
            stage_widget = self._stage_widgets.get(stage_key)
            if stage_widget:
                stage_widget.update_metrics(
                    dataloader_1_second, dataloader_total
                )

            queue_widget = self._queue_widgets.get(stage_key)
            if queue_widget:
                queue_widget.update_metrics(
                    dataloader_1_second, dataloader_total
                )
