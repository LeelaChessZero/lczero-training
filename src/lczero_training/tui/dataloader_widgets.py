"""Widgets that render data loader metrics as horizontal rows."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import ProgressBar, Static

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


def _canonical_stage_name(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    fallback: str | None,
    stage_key: str | None,
) -> str:
    if stage_metric and stage_metric.name:
        return stage_metric.name
    if fallback:
        return fallback
    if stage_key:
        return stage_key
    return "--"


class BaseRowWidget(Horizontal):
    """Base class for pipeline rows that renders a name and content widgets."""

    def __init__(
        self,
        stage_key: str | None = None,
        fallback_name: str | None = None,
        row_type: str = "stage-row",
        **kwargs: Any,
    ) -> None:
        classes = f"dataloader-row {row_type}"
        super().__init__(classes=classes, **kwargs)
        self.stage_key = stage_key
        self._fallback_name = fallback_name
        self._name_label = Static(
            _canonical_stage_name(None, fallback_name, stage_key),
            classes="row-label",
        )
        self._row_content: Horizontal | None = None
        self._content_widgets: list[Static | ProgressBar] = []

    def compose(self) -> ComposeResult:
        row_content = Horizontal(classes="row-content")
        self._row_content = row_content
        yield Horizontal(
            self._name_label,
            row_content,
            classes="row-wrapper",
        )

    def on_mount(self) -> None:
        if self._content_widgets and self._row_content is not None:
            self._row_content.mount(*self._content_widgets)

    def _update_name(
        self,
        stage_metric: training_metrics_pb2.StageMetricProto | None,
    ) -> None:
        self._name_label.update(
            _canonical_stage_name(
                stage_metric, self._fallback_name, self.stage_key
            )
        )


class StageWidget(BaseRowWidget):
    """Base row widget for a pipeline stage."""

    def __init__(
        self,
        stage_key: str,
        fallback_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stage_key=stage_key,
            fallback_name=fallback_name,
            row_type="stage-row",
            **kwargs,
        )

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        raise NotImplementedError


class MetricsStageWidget(StageWidget):
    """Row widget for stages that expose generic load metrics only."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(metrics_field_name, fallback_name=stage_name, **kwargs)
        self.item_name = item_name
        self.metrics_field_name = metrics_field_name
        self._load_chip = Static("load --", classes="metric-chip load-chip")
        self._content_widgets.append(self._load_chip)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        stage_metric_1s = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        stage_metric_total = _find_stage_metric(
            dataloader_total, self.metrics_field_name
        )
        self._update_name(stage_metric_1s or stage_metric_total)

        stage_metrics = _get_stage_specific_metrics(
            stage_metric_1s, self.metrics_field_name
        )
        load_metric = None
        if stage_metrics is not None and stage_metrics.HasField("load"):
            load_metric = stage_metrics.load
        self._load_chip.update(_format_load(load_metric))


class ChunkSourceLoaderStageWidget(StageWidget):
    """Row widget for the chunk source loader stage."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(metrics_field_name, fallback_name=stage_name, **kwargs)
        self.item_name = item_name
        self.metrics_field_name = metrics_field_name
        self._load_chip = Static("load --", classes="metric-chip load-chip")
        self._skipped_chip = Static(
            "skipped --", classes="metric-chip warning-chip"
        )
        self._last_chunk_chip = Static(
            "last --", classes="metric-chip info-chip"
        )
        self._anchor_chip = Static("anchor --", classes="metric-chip info-chip")
        self._since_anchor_chip = Static(
            "since anchor --", classes="metric-chip info-chip"
        )
        self._content_widgets.extend(
            [
                self._load_chip,
                self._skipped_chip,
                self._last_chunk_chip,
                self._anchor_chip,
                self._since_anchor_chip,
            ]
        )

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
        self._update_name(stage_1sec or stage_total)

        metrics_1sec = _get_stage_specific_metrics(
            stage_1sec, self.metrics_field_name
        )
        metrics_total = _get_stage_specific_metrics(
            stage_total, self.metrics_field_name
        )

        load_metric = None
        if metrics_1sec is not None and metrics_1sec.HasField("load"):
            load_metric = metrics_1sec.load
        self._load_chip.update(_format_load(load_metric))

        if metrics_total is not None and metrics_1sec is not None:
            skipped_total = metrics_total.skipped_files_count
            skipped_rate = metrics_1sec.skipped_files_count
            skipped_text = (
                f"skipped {format_full_number(skipped_total)}"
                f" ({format_si(skipped_rate)}/s)"
            )
        else:
            skipped_text = "skipped --"
        self._skipped_chip.update(skipped_text)

        if metrics_1sec and metrics_1sec.HasField("last_chunk_key"):
            last_chunk = metrics_1sec.last_chunk_key
            self._last_chunk_chip.update(
                f"last {last_chunk}" if last_chunk else "last --"
            )
        else:
            self._last_chunk_chip.update("last --")

        pool_metrics = _get_stage_specific_metrics(
            _find_stage_metric(dataloader_1_second, "shuffling_chunk_pool"),
            "shuffling_chunk_pool",
        )
        if (
            pool_metrics
            and pool_metrics.HasField("anchor")
            and pool_metrics.anchor
        ):
            self._anchor_chip.update(f"anchor {pool_metrics.anchor}")
        else:
            self._anchor_chip.update("anchor --")

        if pool_metrics and pool_metrics.HasField("chunks_since_anchor"):
            self._since_anchor_chip.update(
                f"since anchor {format_full_number(pool_metrics.chunks_since_anchor)}"
            )
        else:
            self._since_anchor_chip.update("since anchor --")


class ChunkRescorerStageWidget(StageWidget):
    """Row widget for the chunk rescorer stage."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(metrics_field_name, fallback_name=stage_name, **kwargs)
        self.item_name = item_name
        self.metrics_field_name = metrics_field_name
        self._load_chip = Static("load --", classes="metric-chip load-chip")
        self._queue_rate_chip = Static(
            "queue --/s", classes="metric-chip info-chip"
        )
        self._queue_fill_chip = Static(
            "fill --/--", classes="metric-chip info-chip"
        )
        self._drop_chip = Static("drops --", classes="metric-chip warning-chip")
        self._content_widgets.extend(
            [
                self._load_chip,
                self._queue_rate_chip,
                self._queue_fill_chip,
                self._drop_chip,
            ]
        )

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
        self._update_name(stage_1sec or stage_total)

        metrics_1sec = _get_stage_specific_metrics(
            stage_1sec, self.metrics_field_name
        )
        metrics_total = _get_stage_specific_metrics(
            stage_total, self.metrics_field_name
        )

        load_metric = None
        if metrics_1sec is not None and metrics_1sec.HasField("load"):
            load_metric = metrics_1sec.load
        self._load_chip.update(_format_load(load_metric))

        queue_1sec = None
        if metrics_1sec is not None and metrics_1sec.HasField("queue"):
            queue_1sec = metrics_1sec.queue
        queue_total = None
        if metrics_total is not None and metrics_total.HasField("queue"):
            queue_total = metrics_total.queue

        if queue_1sec:
            rate = queue_1sec.get_count
            self._queue_rate_chip.update(f"queue {format_si(rate)}/s")
        else:
            self._queue_rate_chip.update("queue --/s")

        queue_size = _average_queue_fullness(queue_1sec)
        capacity: int | None = None
        if queue_1sec and queue_1sec.queue_capacity > 0:
            capacity = queue_1sec.queue_capacity
        elif queue_total and queue_total.queue_capacity > 0:
            capacity = queue_total.queue_capacity

        if capacity and capacity > 0:
            if queue_size is not None:
                self._queue_fill_chip.update(
                    f"fill {format_full_number(queue_size)}/"
                    f"{format_full_number(capacity)}"
                )
            else:
                self._queue_fill_chip.update(
                    f"fill --/{format_full_number(capacity)}"
                )
        else:
            self._queue_fill_chip.update("fill --/--")

        if queue_1sec and queue_1sec.drop_count > 0:
            self._drop_chip.update(
                f"drops {format_si(queue_1sec.drop_count)}/s"
            )
        elif queue_total and queue_total.drop_count > 0:
            self._drop_chip.update(
                f"drops {format_full_number(queue_total.drop_count)}"
            )
        else:
            self._drop_chip.update("drops --")


class ShufflingChunkPoolStageWidget(StageWidget):
    """Row widget for the shuffling chunk pool stage."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(metrics_field_name, fallback_name=stage_name, **kwargs)
        self.item_name = item_name
        self.metrics_field_name = metrics_field_name
        self._index_load_chip = Static(
            "idx load --", classes="metric-chip load-chip"
        )
        self._chunk_load_chip = Static(
            "chunk load --", classes="metric-chip load-chip"
        )
        self._files_chip = Static("files --", classes="metric-chip info-chip")
        self._chunks_chip = Static("chunks --", classes="metric-chip info-chip")
        self._content_widgets.extend(
            [
                self._index_load_chip,
                self._chunk_load_chip,
                self._files_chip,
                self._chunks_chip,
            ]
        )

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        stage_metric = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        stage_metric_total = _find_stage_metric(
            dataloader_total, self.metrics_field_name
        )
        self._update_name(stage_metric or stage_metric_total)

        metrics = _get_stage_specific_metrics(
            stage_metric, self.metrics_field_name
        )

        if metrics and metrics.HasField("indexing_load"):
            self._index_load_chip.update(
                _format_load(metrics.indexing_load, label="idx load")
            )
        else:
            self._index_load_chip.update("idx load --")

        if metrics and metrics.HasField("chunk_loading_load"):
            self._chunk_load_chip.update(
                _format_load(metrics.chunk_loading_load, label="chunk load")
            )
        else:
            self._chunk_load_chip.update("chunk load --")

        if metrics and metrics.HasField("chunk_sources_count"):
            if metrics.chunk_sources_count.count > 0:
                files_count = metrics.chunk_sources_count.latest
                self._files_chip.update(f"files {format_si(files_count)}")
            else:
                self._files_chip.update("files --")
        else:
            self._files_chip.update("files --")

        if metrics:
            current_chunks = metrics.current_chunks
            pool_capacity = metrics.pool_capacity
            if pool_capacity > 0:
                self._chunks_chip.update(
                    f"chunks {format_si(current_chunks)} / {format_si(pool_capacity)}"
                )
            else:
                self._chunks_chip.update("chunks --")
        else:
            self._chunks_chip.update("chunks --")


class QueueWidget(BaseRowWidget):
    """Row widget for queue metrics between stages."""

    def __init__(
        self,
        item_name: str = "items",
        stage_key: str | None = None,
        stage_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stage_key=stage_key,
            fallback_name=stage_name,
            row_type="queue-row",
            **kwargs,
        )
        self.item_name = item_name
        self.stage_key = stage_key
        self._queue_name_chip = Static(
            "queue --", classes="metric-chip queue-name-chip"
        )
        self._rate_chip = Static("rate --/s", classes="metric-chip queue-rate")
        self._total_chip = Static("total --", classes="metric-chip queue-total")
        self._fill_bar = ProgressBar(
            classes="queue-fill",
            show_percentage=False,
            show_eta=False,
        )
        self._fill_text = Static("--/--", classes="metric-chip queue-fill-text")
        self._content_widgets.extend(
            [
                self._queue_name_chip,
                self._rate_chip,
                self._total_chip,
                self._fill_bar,
                self._fill_text,
            ]
        )

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        if not self.stage_key:
            self._queue_name_chip.update("queue --")
            self._rate_chip.update("rate --/s")
            self._total_chip.update("total --")
            self._fill_bar.total = 1
            self._fill_bar.progress = 0
            self._fill_text.update("--/--")
            return

        stage_1sec = _find_stage_metric(dataloader_1_second, self.stage_key)
        stage_total = _find_stage_metric(dataloader_total, self.stage_key)
        self._update_name(stage_1sec or stage_total)

        queue_1sec = _get_queue_metrics(stage_1sec)
        queue_total = _get_queue_metrics(stage_total)

        queue_name = None
        if queue_1sec and queue_1sec.name:
            queue_name = queue_1sec.name
        elif queue_total and queue_total.name:
            queue_name = queue_total.name
        self._queue_name_chip.update(
            f"queue {queue_name}" if queue_name else "queue --"
        )

        rate = queue_1sec.get_count if queue_1sec else 0
        self._rate_chip.update(f"rate {format_si(rate)}/s")
        if rate == 0:
            self._rate_chip.add_class("queue-rate--zero")
        else:
            self._rate_chip.remove_class("queue-rate--zero")

        if queue_total:
            self._total_chip.update(
                f"total {format_full_number(queue_total.get_count)}"
            )
        else:
            self._total_chip.update("total --")

        size = _average_queue_fullness(queue_1sec)
        capacity: int | None = None
        if queue_1sec and queue_1sec.queue_capacity > 0:
            capacity = queue_1sec.queue_capacity
        elif queue_total and queue_total.queue_capacity > 0:
            capacity = queue_total.queue_capacity

        if capacity and capacity > 0:
            self._fill_bar.total = capacity
            if size is not None:
                self._fill_bar.progress = min(size, capacity)
                fill_text = (
                    f"{format_full_number(size)}/{format_full_number(capacity)}"
                )
            else:
                self._fill_bar.progress = 0
                fill_text = f"--/{format_full_number(capacity)}"
        else:
            self._fill_bar.total = 1
            self._fill_bar.progress = 0
            fill_text = "--/--"
        self._fill_text.update(fill_text)
