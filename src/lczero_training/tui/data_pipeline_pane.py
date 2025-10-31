# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics.
# ABOUTME: Shows a grid of pipeline stages and queues with their metrics.

from collections.abc import Iterable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container

import proto.training_metrics_pb2 as training_metrics_pb2

from .dataloader_widgets import QueueWidget, StageWidget

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


class DataPipelinePane(Container):
    """Main pane showing data pipeline flow and statistics as a grid."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stage_widgets: dict[str, StageWidget] = {}
        self._queue_widgets: dict[str, dict[str, QueueWidget]] = {}
        self._queue_order: dict[str, list[str]] = {}
        self._stage_order: list[str] = []

    def compose(self) -> ComposeResult:
        """The pane starts empty and rows are added when metrics arrive."""
        yield from ()

    def _friendly_title(self, stage_key: str) -> str:
        return FRIENDLY_STAGE_NAMES.get(
            stage_key, stage_key.replace("_", " ").title()
        )

    def _ensure_stage_widget(
        self,
        stage_key: str,
    ) -> tuple[StageWidget, bool]:
        created = False
        if stage_key not in self._stage_widgets:
            stage_widget = StageWidget(
                stage_key=stage_key,
                fallback_name=self._friendly_title(stage_key),
            )
            self._stage_widgets[stage_key] = stage_widget
            self._queue_widgets[stage_key] = {}
            self._queue_order[stage_key] = []
            self._stage_order.append(stage_key)
            created = True
        else:
            stage_widget = self._stage_widgets[stage_key]

        return stage_widget, created

    def _ensure_queue_widgets(
        self,
        stage_key: str,
        stage_metric: training_metrics_pb2.StageMetricProto,
    ) -> list[QueueWidget]:
        stage_queue_widgets = self._queue_widgets.setdefault(stage_key, {})
        queue_order = self._queue_order.setdefault(stage_key, [])
        new_widgets: list[QueueWidget] = []

        friendly = self._friendly_title(stage_key)
        for index, queue_metric in enumerate(stage_metric.queue_metrics):
            queue_identifier = queue_metric.name or f"__index__{index}"
            if queue_identifier in stage_queue_widgets:
                continue
            label_suffix = (
                f" {queue_metric.name}"
                if queue_metric.name
                else f" #{index + 1}"
            )
            queue_widget = QueueWidget(
                stage_key=stage_key,
                stage_name=f"{friendly} queue{label_suffix}",
                queue_name=queue_identifier,
            )
            stage_queue_widgets[queue_identifier] = queue_widget
            queue_order.append(queue_identifier)
            new_widgets.append(queue_widget)

        return new_widgets

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
            if not stage_key:
                continue

            stage_widget, created = self._ensure_stage_widget(stage_key)
            if created:
                new_widgets.append(stage_widget)

            queue_widgets = self._ensure_queue_widgets(stage_key, stage_metric)
            new_widgets.extend(queue_widgets)

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

            for queue_key in self._queue_order.get(stage_key, []):
                queue_widget = self._queue_widgets[stage_key].get(queue_key)
                if queue_widget:
                    queue_widget.update_metrics(
                        dataloader_1_second, dataloader_total
                    )
