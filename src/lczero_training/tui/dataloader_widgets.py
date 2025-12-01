"""Widgets that render data loader metrics without stage-specific logic."""

from __future__ import annotations

from typing import Any, Dict

from textual.app import ComposeResult
from textual.widget import Widget
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


def _collect_metric_names(
    stage_1s: training_metrics_pb2.StageMetricProto | None,
    stage_total: training_metrics_pb2.StageMetricProto | None,
    attribute: str,
) -> list[str]:
    names: list[str] = []

    def _add_from(
        stage_metric: training_metrics_pb2.StageMetricProto | None,
    ) -> None:
        if not stage_metric:
            return
        for metric in getattr(stage_metric, attribute):
            name = metric.name if metric.name else ""
            if name not in names:
                names.append(name)

    _add_from(stage_total)
    _add_from(stage_1s)
    return names


def _find_load_metric(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    metric_name: str,
) -> training_metrics_pb2.LoadMetricProto | None:
    if not stage_metric:
        return None
    for load_metric in stage_metric.load_metrics:
        if (load_metric.name or "") == metric_name:
            return load_metric
    return None


def _find_count_metric(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    metric_name: str,
) -> training_metrics_pb2.CountMetricProto | None:
    if not stage_metric:
        return None
    for count_metric in stage_metric.count_metrics:
        if (count_metric.name or "") == metric_name:
            return count_metric
    return None


def _find_gauge_metric(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    metric_name: str,
) -> training_metrics_pb2.GaugeMetricProto | None:
    if not stage_metric:
        return None
    for gauge_metric in stage_metric.gauge_metrics:
        if (gauge_metric.name or "") == metric_name:
            return gauge_metric
    return None


def _find_statistics_metric(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    metric_name: str,
) -> training_metrics_pb2.StatisticsProtoDouble | None:
    if not stage_metric:
        return None
    for stats_metric in stage_metric.statistics_metrics:
        if (stats_metric.name or "") == metric_name:
            return stats_metric
    return None


def _get_queue_metric(
    stage_metric: training_metrics_pb2.StageMetricProto | None,
    queue_name: str | None,
) -> training_metrics_pb2.QueueMetricProto | None:
    if not stage_metric or not stage_metric.queue_metrics:
        return None
    if queue_name is None:
        return stage_metric.queue_metrics[0]
    if queue_name.startswith("__index__"):
        try:
            index = int(queue_name.removeprefix("__index__"))
        except ValueError:
            index = -1
        if 0 <= index < len(stage_metric.queue_metrics):
            return stage_metric.queue_metrics[index]
    for queue_metric in stage_metric.queue_metrics:
        if (queue_metric.name or "") == queue_name:
            return queue_metric
    return None


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
    label: str,
) -> str:
    if not load_metric:
        return f"{label} --"
    total_part = (
        f"{load_metric.total_seconds:.0f}"
        if load_metric.total_seconds > 0
        else "--"
    )
    return f"{label} {load_metric.load_seconds:.1f}/{total_part}s"


def _format_count(
    count_metric_1s: training_metrics_pb2.CountMetricProto | None,
    count_metric_total: training_metrics_pb2.CountMetricProto | None,
    label: str,
) -> str:
    if count_metric_1s and count_metric_total:
        rate = format_si(count_metric_1s.count)
        total = format_full_number(count_metric_total.count)
        return f"{label} {rate}/s ({total} total)"
    if count_metric_total:
        return f"{label} {format_full_number(count_metric_total.count)}"
    if count_metric_1s:
        return f"{label} {format_si(count_metric_1s.count)}/s"
    return f"{label} --"


def _format_gauge(
    gauge_metric: training_metrics_pb2.GaugeMetricProto | None,
    label: str,
) -> str:
    if not gauge_metric:
        return f"{label} --"
    if gauge_metric.HasField("capacity"):
        value_text = format_full_number(gauge_metric.value)
        capacity_text = format_full_number(gauge_metric.capacity)
        return f"{label} {value_text}/{capacity_text}"
    return f"{label} {format_full_number(gauge_metric.value)}"


def _format_statistics(
    stats_1s: training_metrics_pb2.StatisticsProtoDouble | None,
    stats_total: training_metrics_pb2.StatisticsProtoDouble | None,
    label: str,
) -> str:
    parts = [label]
    if stats_1s and stats_1s.count > 0:
        avg_1s = stats_1s.sum / stats_1s.count
        parts.append(
            f"per 1s: avg {avg_1s:.1f} (min {stats_1s.min:.1f}, max {stats_1s.max:.1f}, count {format_si(stats_1s.count)})"
        )
    if stats_total and stats_total.count > 0:
        avg_total = stats_total.sum / stats_total.count
        parts.append(
            f"total: avg {avg_total:.1f} (min {stats_total.min:.1f}, max {stats_total.max:.1f}, count {format_full_number(stats_total.count)})"
        )
    if len(parts) == 1:
        return f"{label} --"
    return "; ".join(parts)


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


class BaseRowWidget(Widget):
    """Base class for pipeline rows that renders a name and content widgets."""

    MAX_GRID_ROWS = 8  # Supports up to 28 chips (4 per row, 7 content rows)

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
        self._content_widgets: list[Widget] = []
        self._spacers: list[Static] = []

    def compose(self) -> ComposeResult:
        # Name label goes in first cell (row 0, col 0)
        yield self._name_label

    def on_mount(self) -> None:
        if self._content_widgets:

            async def _mount_initial() -> None:
                # Mount chips with spacers interleaved at the start of each row
                for i, widget in enumerate(self._content_widgets):
                    # After every 4 chips, add a spacer for the next row's column 0
                    if i > 0 and i % 4 == 0:
                        spacer = Static("", classes="grid-spacer")
                        self._spacers.append(spacer)
                        await self.mount(spacer)
                    await self.mount(widget)

            self.call_later(_mount_initial)

    def add_content_widget(self, widget: Widget) -> None:
        if widget in self._content_widgets:
            return
        self._content_widgets.append(widget)

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
    """Row widget that renders all metrics exposed by a stage."""

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
        self._chips: Dict[str, Static] = {}

    def _ensure_chip(self, key: str, default_text: str, classes: str) -> Static:
        chip = self._chips.get(key)
        if chip is None:
            chip = Static(default_text, classes=f"metric-chip {classes}")
            self._chips[key] = chip
            self.add_content_widget(chip)
        return chip

    def _update_last_chunk_chip(
        self,
        stage_1s: training_metrics_pb2.StageMetricProto | None,
        stage_total: training_metrics_pb2.StageMetricProto | None,
    ) -> None:
        stage = None
        if stage_1s and stage_1s.HasField("last_chunk_key"):
            stage = stage_1s
        elif stage_total and stage_total.HasField("last_chunk_key"):
            stage = stage_total
        if not stage:
            return
        last_value = stage.last_chunk_key or "--"
        chip = self._ensure_chip("info:last", "last --", "info-chip")
        chip.update(f"last {last_value}")

    def _update_anchor_chip(
        self,
        stage_total: training_metrics_pb2.StageMetricProto | None,
    ) -> None:
        if not stage_total or not stage_total.HasField("anchor"):
            return
        chip = self._ensure_chip("info:anchor", "anchor --", "info-chip")
        chip.update(f"anchor {stage_total.anchor}")

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        if self.stage_key is None:
            return
        stage_metric_1s = _find_stage_metric(
            dataloader_1_second, self.stage_key
        )
        stage_metric_total = _find_stage_metric(
            dataloader_total, self.stage_key
        )

        self._update_name(stage_metric_1s or stage_metric_total)

        load_names = _collect_metric_names(
            stage_metric_1s, stage_metric_total, "load_metrics"
        )
        for load_name in load_names:
            label = load_name or "load"
            load_metric = _find_load_metric(stage_metric_1s, load_name)
            if load_metric is None:
                load_metric = _find_load_metric(stage_metric_total, load_name)
            chip = self._ensure_chip(
                f"load:{load_name}", f"{label} --", "load-chip"
            )
            chip.update(_format_load(load_metric, label=label))

        count_names = _collect_metric_names(
            stage_metric_1s, stage_metric_total, "count_metrics"
        )
        for count_name in count_names:
            label = count_name or "count"
            count_metric_1s = _find_count_metric(stage_metric_1s, count_name)
            count_metric_total = _find_count_metric(
                stage_metric_total, count_name
            )
            chip = self._ensure_chip(
                f"count:{count_name}", f"{label} --", "info-chip"
            )
            chip.update(
                _format_count(count_metric_1s, count_metric_total, label=label)
            )

        gauge_names = _collect_metric_names(
            stage_metric_1s, stage_metric_total, "gauge_metrics"
        )
        for gauge_name in gauge_names:
            label = gauge_name or "gauge"
            gauge_metric = _find_gauge_metric(stage_metric_total, gauge_name)
            if gauge_metric is None:
                gauge_metric = _find_gauge_metric(stage_metric_1s, gauge_name)
            chip = self._ensure_chip(
                f"gauge:{gauge_name}", f"{label} --", "info-chip"
            )
            chip.update(_format_gauge(gauge_metric, label=label))

        self._update_last_chunk_chip(stage_metric_1s, stage_metric_total)
        self._update_anchor_chip(stage_metric_total)


class StatisticsRowWidget(BaseRowWidget):
    """Full-width row for statistics metrics."""

    def __init__(
        self,
        stage_key: str,
        metric_name: str,
        label: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stage_key=stage_key,
            fallback_name=None,
            row_type="statistics-row",
            **kwargs,
        )
        self._metric_name = metric_name
        self._label = label
        self._stats_label = Static(f"{label} --", classes="statistics-full")
        self.add_content_widget(self._stats_label)

    def compose(self) -> ComposeResult:
        return
        yield

    def on_mount(self) -> None:
        if self._content_widgets:

            async def _mount_initial() -> None:
                await self.mount(*self._content_widgets)

            self.call_later(_mount_initial)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        if not self.stage_key:
            self._stats_label.update(f"{self._label} --")
            return
        stage_1s = _find_stage_metric(dataloader_1_second, self.stage_key)
        stage_total = _find_stage_metric(dataloader_total, self.stage_key)
        stats_1s = _find_statistics_metric(stage_1s, self._metric_name)
        stats_total = _find_statistics_metric(stage_total, self._metric_name)
        self._stats_label.update(
            _format_statistics(stats_1s, stats_total, self._label)
        )


class QueueWidget(BaseRowWidget):
    """Row widget for queue metrics between stages."""

    def __init__(
        self,
        stage_key: str | None = None,
        stage_name: str | None = None,
        queue_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stage_key=stage_key,
            fallback_name=stage_name,
            row_type="queue-row",
            **kwargs,
        )
        self._queue_name = queue_name
        self._queue_name_chip = Static(
            "queue --", classes="metric-chip queue-name-chip"
        )
        self._rate_chip = Static("rate --/s", classes="metric-chip queue-rate")
        self._total_chip = Static("total --", classes="metric-chip queue-total")
        self._drop_chip = Static("dropped --", classes="metric-chip")
        self._fill_bar = ProgressBar(
            classes="queue-fill",
            show_percentage=False,
            show_eta=False,
        )
        self._fill_text = Static("--/--", classes="metric-chip queue-fill-text")
        self.add_content_widget(self._queue_name_chip)
        self.add_content_widget(self._rate_chip)
        self.add_content_widget(self._total_chip)
        self.add_content_widget(self._drop_chip)
        self.add_content_widget(self._fill_bar)
        self.add_content_widget(self._fill_text)

    def compose(self) -> ComposeResult:
        # Queue widgets don't show the name label - use all 6 columns
        return
        yield  # Make this a generator

    def on_mount(self) -> None:
        # Mount all content widgets without spacers (use all 6 columns per row)
        if self._content_widgets:

            async def _mount_initial() -> None:
                await self.mount(*self._content_widgets)

            self.call_later(_mount_initial)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        if not self.stage_key:
            self._queue_name_chip.update("queue --")
            self._rate_chip.update("rate --/s")
            self._total_chip.update("total --")
            self._drop_chip.update("dropped --")
            self._drop_chip.remove_class("warning-chip")
            self._fill_bar.total = 1
            self._fill_bar.progress = 0
            self._fill_text.update("--/--")
            return

        stage_1sec = _find_stage_metric(dataloader_1_second, self.stage_key)
        stage_total = _find_stage_metric(dataloader_total, self.stage_key)
        self._update_name(stage_1sec or stage_total)

        queue_1sec = _get_queue_metric(stage_1sec, self._queue_name)
        queue_total = _get_queue_metric(stage_total, self._queue_name)

        queue_name = None
        if queue_1sec and queue_1sec.name:
            queue_name = queue_1sec.name
        elif queue_total and queue_total.name:
            queue_name = queue_total.name
        elif self._queue_name:
            queue_name = self._queue_name
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

        has_drops = False
        if queue_1sec and queue_1sec.drop_count:
            self._drop_chip.update(
                f"dropped {format_si(queue_1sec.drop_count)}/s"
            )
            has_drops = True
        elif queue_total and queue_total.drop_count:
            self._drop_chip.update(
                f"dropped {format_full_number(queue_total.drop_count)}"
            )
            has_drops = True
        else:
            self._drop_chip.update("dropped --")
            has_drops = False

        if has_drops:
            self._drop_chip.add_class("warning-chip")
        else:
            self._drop_chip.remove_class("warning-chip")

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
