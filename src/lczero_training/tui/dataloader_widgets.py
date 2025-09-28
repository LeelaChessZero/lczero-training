"""Widgets that render data loader metrics as single-line rows."""

from collections.abc import Sequence
from typing import Any

from textual.widgets import Static

import proto.training_metrics_pb2 as training_metrics_pb2


def _find_stage_metric(
    metrics: training_metrics_pb2.DataLoaderMetricsProto | None,
    stage_key: str,
) -> training_metrics_pb2.StageMetricProto | None:
    if not metrics:
        return None
    for stage_metric in metrics.stage_metrics:
        if stage_metric.name == stage_key:
            return stage_metric
    return None


def _get_stage_specific_metrics(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    field_name: str,
) -> Any:
    if not stage_metric:
        return None
    try:
        if stage_metric.HasField(field_name):
            return getattr(stage_metric, field_name)
    except ValueError:
        return None
    return None


def _get_queue_metrics(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    queue_name: str = "output",
) -> training_metrics_pb2.QueueMetricProto | None:
    if not stage_metric or not stage_metric.output_queue_metrics:
        return None
    for queue_metric in stage_metric.output_queue_metrics:
        if queue_metric.name == queue_name:
            return queue_metric
    return stage_metric.output_queue_metrics[0]


def format_si(value: int, precision: int = 1) -> str:
    if value == 0:
        return "0"
    units = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "G"),
        (1_000_000, "M"),
        (1_000, "k"),
    ]
    for threshold, unit in units:
        if value >= threshold:
            result = value / threshold
            if precision == 0:
                return f"{int(result)}{unit}"
            return f"{result:.{precision}f}{unit}".rstrip("0").rstrip(".")
    return str(value)


def format_full_number(value: int) -> str:
    if value < 10_000:
        return str(value)
    return f"{value:_}".replace("_", "'")


def _format_load(
    load_metric: training_metrics_pb2.LoadMetricProto | None,
    label: str = "load",
) -> str:
    if not load_metric:
        return f"{label} --"
    total_part = (
        f"{load_metric.total_seconds:.0f}"
        if load_metric.total_seconds > 0
        else "--"
    )
    return f"{label} {load_metric.load_seconds:.1f}/{total_part}s"


def _format_segments(
    title: str, segments: Sequence[str], extra: Sequence[str] | None
) -> str:
    primary = " | ".join(segment for segment in segments if segment)
    if not primary:
        primary = "--"
    text = f"{title}: {primary}"
    if extra:
        secondary = " | ".join(segment for segment in extra if segment)
        if secondary:
            text = f"{text}\n  {secondary}"
    return text


def _average_queue_fullness(
    queue_metric: training_metrics_pb2.QueueMetricProto | None,
) -> int | None:
    if not queue_metric:
        return None
    if (
        queue_metric.HasField("queue_fullness")
        and queue_metric.queue_fullness.count > 0
    ):
        return int(
            queue_metric.queue_fullness.sum / queue_metric.queue_fullness.count
        )
    return None


class StageWidget(Static):
    """Base row widget for a pipeline stage."""

    def __init__(
        self, stage_name: str, stage_key: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__("", classes="dataloader-row", **kwargs)
        self.stage_name = stage_name
        self.stage_key = stage_key
        self.add_class("stage-row")

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        raise NotImplementedError

    def _format_title(self, suffix: str | None = None) -> str:
        title = self.stage_name
        if self.stage_key:
            title = f"{title} [{self.stage_key}]"
        if suffix:
            title = f"{title} {suffix}"
        return title

    def _update_row(
        self,
        segments: Sequence[str],
        extra: Sequence[str] | None = None,
        title_suffix: str | None = None,
    ) -> None:
        self.update(
            _format_segments(self._format_title(title_suffix), segments, extra)
        )


class MetricsStageWidget(StageWidget):
    """Row widget for stages that only expose generic load metrics."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, stage_key=metrics_field_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        stage_metric = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        stage_metrics = _get_stage_specific_metrics(
            stage_metric, self.metrics_field_name
        )
        load_metric = None
        if stage_metrics is not None and stage_metrics.HasField("load"):
            load_metric = stage_metrics.load
        self._update_row([_format_load(load_metric)])


class ChunkSourceLoaderStageWidget(StageWidget):
    """Row widget for the chunk source loader stage."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, stage_key=metrics_field_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        stage_1sec = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        stage_total = _find_stage_metric(
            dataloader_total, self.metrics_field_name
        )

        metrics_1sec = _get_stage_specific_metrics(
            stage_1sec, self.metrics_field_name
        )
        metrics_total = _get_stage_specific_metrics(
            stage_total, self.metrics_field_name
        )

        load_metric = None
        if metrics_1sec is not None and metrics_1sec.HasField("load"):
            load_metric = metrics_1sec.load

        primary: list[str] = [_format_load(load_metric)]

        if metrics_total is not None and metrics_1sec is not None:
            skipped_total = metrics_total.skipped_files_count
            skipped_rate = metrics_1sec.skipped_files_count
            primary.append(
                f"skipped {format_full_number(skipped_total)} ({format_si(skipped_rate)}/s)"
            )

        extra: list[str] = []
        if metrics_1sec is not None and metrics_1sec.HasField("last_chunk_key"):
            last_chunk = metrics_1sec.last_chunk_key
            if last_chunk:
                extra.append(f"last chunk {last_chunk}")

        pool_metrics = _get_stage_specific_metrics(
            _find_stage_metric(dataloader_1_second, "shuffling_chunk_pool"),
            "shuffling_chunk_pool",
        )
        if (
            pool_metrics
            and pool_metrics.HasField("anchor")
            and pool_metrics.anchor
        ):
            extra.append(f"anchor {pool_metrics.anchor}")
        if pool_metrics and pool_metrics.HasField("chunks_since_anchor"):
            extra.append(
                f"since anchor {format_full_number(pool_metrics.chunks_since_anchor)}"
            )

        self._update_row(primary, extra or None)


class ShufflingChunkPoolStageWidget(StageWidget):
    """Row widget for the shuffling chunk pool stage."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, stage_key=metrics_field_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        stage_metric = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        metrics = _get_stage_specific_metrics(
            stage_metric, self.metrics_field_name
        )

        primary: list[str] = []
        extra: list[str] = []

        if metrics and metrics.HasField("indexing_load"):
            primary.append(
                _format_load(metrics.indexing_load, label="idx load")
            )
        else:
            primary.append("idx load --")

        if metrics and metrics.HasField("chunk_loading_load"):
            primary.append(
                _format_load(metrics.chunk_loading_load, label="chunk load")
            )
        else:
            primary.append("chunk load --")

        if metrics and metrics.HasField("chunk_sources_count"):
            if metrics.chunk_sources_count.count > 0:
                files_count = metrics.chunk_sources_count.latest
                primary.append(f"files {format_si(files_count)}")
            else:
                primary.append("files --")
        else:
            primary.append("files --")

        if metrics:
            current_chunks = metrics.current_chunks
            pool_capacity = metrics.pool_capacity
            if pool_capacity > 0:
                extra.append(
                    f"chunks {format_si(current_chunks)} / {format_si(pool_capacity)}"
                )
            else:
                extra.append("chunks --")
        else:
            extra.append("chunks --")

        self._update_row(primary, extra or None)


class QueueWidget(StageWidget):
    """Row widget for queue metrics between stages."""

    def __init__(
        self,
        item_name: str = "items",
        stage_key: str | None = None,
        stage_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stage_name or f"Queue ({item_name})", stage_key=stage_key, **kwargs
        )
        self.item_name = item_name
        self.stage_key = stage_key
        self.remove_class("stage-row")
        self.add_class("queue-row")

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        if not self.stage_key:
            self._update_row(["--"])
            return

        stage_1sec = _find_stage_metric(dataloader_1_second, self.stage_key)
        stage_total = _find_stage_metric(dataloader_total, self.stage_key)

        queue_1sec = _get_queue_metrics(stage_1sec)
        queue_total = _get_queue_metrics(stage_total)

        primary: list[str] = []

        if queue_1sec:
            primary.append(f"rate {format_si(queue_1sec.get_count)}/s")
        else:
            primary.append("rate --")

        if queue_total:
            primary.append(f"total {format_full_number(queue_total.get_count)}")
        else:
            primary.append("total --")

        size = _average_queue_fullness(queue_1sec)
        capacity: int | None = None
        if queue_1sec and queue_1sec.queue_capacity > 0:
            capacity = queue_1sec.queue_capacity
        elif queue_total and queue_total.queue_capacity > 0:
            capacity = queue_total.queue_capacity

        if size is not None and capacity is not None:
            primary.append(
                f"fill {format_full_number(size)} / {format_full_number(capacity)}"
            )
        elif capacity is not None:
            primary.append(f"fill -- / {format_full_number(capacity)}")
        else:
            primary.append("fill --")

        queue_name = None
        if queue_1sec and queue_1sec.name:
            queue_name = queue_1sec.name
        elif queue_total and queue_total.name:
            queue_name = queue_total.name

        queue_suffix = None
        if queue_name:
            queue_suffix = f"(queue {queue_name})"
        elif queue_1sec or queue_total:
            queue_suffix = "(queue)"

        self._update_row(primary, title_suffix=queue_suffix)
