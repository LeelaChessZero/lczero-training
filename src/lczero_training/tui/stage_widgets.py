# ABOUTME: Stage widgets for the data pipeline visualization
# ABOUTME: Each stage represents a different part of the data loading process

from collections import deque
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import ProgressBar, Sparkline, Static

import proto.training_metrics_pb2 as training_metrics_pb2


def _find_stage_metric(
    metrics: training_metrics_pb2.DataLoaderMetricsProto | None,
    stage_key: str,
) -> training_metrics_pb2.StageMetricProto | None:
    """Locate a StageMetricProto by name."""
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
    """Return the stage-specific metrics message if present."""
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
    """Find a queue metric by name, falling back to the first metric."""
    if not stage_metric or not stage_metric.output_queue_metrics:
        return None
    for queue_metric in stage_metric.output_queue_metrics:
        if queue_metric.name == queue_name:
            return queue_metric
    return stage_metric.output_queue_metrics[0]


class StageWidget(Static):
    """Base class for all data pipeline stage widgets."""

    def __init__(self, stage_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stage_name = stage_name
        self.border_title = stage_name

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
        pass


def format_si(value: int, precision: int = 1) -> str:
    """Convert a number to SI unit format (e.g., 1234 -> '1.2k')."""
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
    """Formats an integer with apostrophe separators for thousands."""
    if value < 10000:
        return str(value)
    return f"{value:_}".replace("_", "'")


class LoadWidget(Container):
    """Widget for displaying load metrics as a single line with progress bar."""

    load_seconds: reactive[float] = reactive(0.0, layout=True)
    total_seconds: reactive[float] = reactive(0.0, layout=True)

    def __init__(self, label: str = "threads", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label = label
        self._progress_bar: ProgressBar | None = None
        self._ratio_display: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static(f"{self.label}:", id="load-label")
        yield ProgressBar(
            id="load-progress", show_percentage=False, show_eta=False
        )
        yield Static("0.0/0", id="load-ratio")

    def on_mount(self) -> None:
        """Initialize progress bar and get widget references."""
        self._progress_bar = self.query_one(ProgressBar)
        self._ratio_display = self.query_one("#load-ratio", Static)
        if self._progress_bar:
            self._progress_bar.total = 1.0

    def watch_load_seconds(self, load_seconds: float) -> None:
        """Update display when load_seconds changes."""
        self._update_display()

    def watch_total_seconds(self, total_seconds: float) -> None:
        """Update display when total_seconds changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the progress bar and ratio display."""
        if not self._progress_bar or not self._ratio_display:
            return

        if self.total_seconds > 0:
            self._progress_bar.total = self.total_seconds
            self._progress_bar.progress = min(
                self.load_seconds, self.total_seconds
            )
            ratio_text = f"{self.load_seconds:.1f}/{int(self.total_seconds)}"
        else:
            self._progress_bar.total = 1.0
            self._progress_bar.progress = 0.0
            ratio_text = "0.0/0"

        self._ratio_display.update(ratio_text)

    def update_load_metrics(
        self, load_metric: training_metrics_pb2.LoadMetricProto | None
    ) -> None:
        """Update the load metrics from a LoadMetricProto."""
        if load_metric:
            self.load_seconds = load_metric.load_seconds
            self.total_seconds = load_metric.total_seconds
        else:
            self.load_seconds = 0.0
            self.total_seconds = 0.0


class QueueWidget(Container):
    """Widget for displaying queue metrics between stages with 4-row layout."""

    rate: reactive[int] = reactive(0, layout=True)
    total_transferred: reactive[int] = reactive(0, layout=True)
    current_size: reactive[int] = reactive(0, layout=True)
    capacity: reactive[int] = reactive(1, layout=True)

    def __init__(
        self,
        item_name: str = "items",
        stage_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.item_name = item_name
        self.border_title = f"Queue ({item_name})"
        self.stage_key = stage_key
        self._rate_history: deque[int] = deque(maxlen=16)
        self._max_rate_seen = 0

    def on_mount(self) -> None:
        """Initialize progress bar when widget is mounted."""
        progress_bar = self.query_one(ProgressBar)
        progress_bar.total = 1
        progress_bar.progress = 0

    def compose(self) -> ComposeResult:
        yield Static("Rate: --/s", id="rate-display", classes="rate-value")
        yield Sparkline([], id="rate-sparkline")
        yield Static("Total: --", id="total-display")
        with Horizontal(id="progress-container"):
            yield Static("Fill lvl:", id="fill-level-label")
            yield ProgressBar(
                id="queue-progress", show_percentage=False, show_eta=False
            )
            yield Static("--/--", id="capacity-display")

    def watch_rate(self, rate: int) -> None:
        """Update rate display when rate changes."""
        rate_display = self.query_one("#rate-display", Static)
        rate_text = f"Rate: {format_si(rate)}/s"

        rate_display.set_class(rate == 0, "rate--zero")
        rate_display.update(rate_text)

    def watch_total_transferred(self, total: int) -> None:
        """Update total display when total changes."""
        self.query_one("#total-display", Static).update(
            f"Total: {format_full_number(total)}"
        )

    def watch_current_size(self, size: int) -> None:
        """Update progress bar when current size changes."""
        self._update_progress()

    def watch_capacity(self, capacity: int) -> None:
        """Update progress bar when capacity changes."""
        self._update_progress()

    def _update_progress(self) -> None:
        """Update the progress bar and capacity display."""
        progress_bar = self.query_one(ProgressBar)
        capacity_display = self.query_one("#capacity-display", Static)

        progress_bar.total = max(1, self.capacity)
        progress_bar.progress = min(self.current_size, self.capacity)

        capacity_text = f"{format_full_number(self.current_size)}/{format_full_number(self.capacity)}"
        capacity_display.update(capacity_text)

    def _show_error_state(self, error_message: str) -> None:
        """Display an error state for the widget."""
        rate_display = self.query_one("#rate-display", Static)
        rate_display.update(f"Rate: {error_message}")
        rate_display.add_class("rate--zero")

        self.query_one("#total-display", Static).update("Total: Error")
        self.query_one("#capacity-display", Static).update("Error/Error")

        self._rate_history.clear()
        self.query_one(Sparkline).data = list(self._rate_history)

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the queue metrics display."""
        current_rate = 0  # Default to 0
        if dataloader_1_second and dataloader_total and self.stage_key:
            stage_1sec = _find_stage_metric(dataloader_1_second, self.stage_key)
            stage_total = _find_stage_metric(dataloader_total, self.stage_key)

            queue_1sec = _get_queue_metrics(stage_1sec)
            queue_total = _get_queue_metrics(stage_total)

            if queue_1sec and queue_total:
                current_rate = queue_1sec.get_count
                self.total_transferred = queue_total.get_count
                self.capacity = queue_1sec.queue_capacity

                if (
                    queue_1sec.HasField("queue_fullness")
                    and queue_1sec.queue_fullness.count > 0
                ):
                    self.current_size = int(
                        queue_1sec.queue_fullness.sum
                        / queue_1sec.queue_fullness.count
                    )
                else:
                    self.current_size = 0
            else:
                self._show_error_state(f"Error ({self.stage_key})")

        self.rate = current_rate

        # Always update sparkline
        self._rate_history.append(current_rate)
        if current_rate > self._max_rate_seen:
            self._max_rate_seen = current_rate
        sparkline = self.query_one(Sparkline)
        sparkline.data = list(self._rate_history)


class MetricsStageWidget(StageWidget):
    """A generic widget for a pipeline stage that displays key metrics."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name
        self.load_widget = LoadWidget()

    def compose(self) -> ComposeResult:
        yield self.load_widget

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
        stage_metric = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        stage_metrics = _get_stage_specific_metrics(
            stage_metric, self.metrics_field_name
        )
        if stage_metrics is not None and stage_metrics.HasField("load"):
            self.load_widget.update_load_metrics(stage_metrics.load)
        else:
            self.load_widget.update_load_metrics(None)


class ChunkSourceLoaderStageWidget(StageWidget):
    """A widget for the ChunkSourceLoader stage with load metrics and skipped files."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name
        self.load_widget = LoadWidget()
        self.skipped_files_display = Static("skipped: --")
        self.last_chunk_display = Static("Last: --", id="last-chunk-display")
        self.anchor_display = Static("⚓: --", id="anchor-display")
        self.chunks_since_anchor_display = Static(
            "Since ⚓: --", id="chunks-since-anchor-display"
        )
        self._skipped_total = 0
        self._skipped_rate = 0

    def compose(self) -> ComposeResult:
        yield self.load_widget
        yield self.skipped_files_display
        yield self.last_chunk_display
        yield self.anchor_display
        yield self.chunks_since_anchor_display

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
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

        if not metrics_1sec or not metrics_total:
            self.load_widget.update_load_metrics(None)
            self.skipped_files_display.update("skipped: --")
            self.last_chunk_display.update("Last: --")
            self.anchor_display.update("⚓: --")
            self.chunks_since_anchor_display.update("Since ⚓: --")
            return

        if metrics_1sec.HasField("load"):
            self.load_widget.update_load_metrics(metrics_1sec.load)
        else:
            self.load_widget.update_load_metrics(None)

        self._skipped_total = metrics_total.skipped_files_count
        self._skipped_rate = metrics_1sec.skipped_files_count

        skipped_text = f"skipped: {format_full_number(self._skipped_total)}"
        if self._skipped_rate > 0:
            skipped_text += f" ({format_si(self._skipped_rate)}/s)"
        else:
            skipped_text += " (0/s)"

        self.skipped_files_display.update(skipped_text)

        if (
            metrics_1sec.HasField("last_chunk_key")
            and metrics_1sec.last_chunk_key
        ):
            self.last_chunk_display.update(
                f"Last: {metrics_1sec.last_chunk_key}"
            )
        else:
            self.last_chunk_display.update("Last: --")

        pool_metrics = _get_stage_specific_metrics(
            _find_stage_metric(dataloader_1_second, "shuffling_chunk_pool"),
            "shuffling_chunk_pool",
        )

        if (
            pool_metrics
            and pool_metrics.HasField("anchor")
            and pool_metrics.anchor
        ):
            self.anchor_display.update(f"⚓: {pool_metrics.anchor}")
        else:
            self.anchor_display.update("⚓: --")

        if pool_metrics and pool_metrics.HasField("chunks_since_anchor"):
            chunks_count = pool_metrics.chunks_since_anchor
            self.chunks_since_anchor_display.update(
                f"Since ⚓: {format_full_number(chunks_count)}"
            )
        else:
            self.chunks_since_anchor_display.update("Since ⚓: --")


class ShufflingChunkPoolStageWidget(StageWidget):
    """A widget for the ShufflingChunkPool stage with two load metrics."""

    def __init__(
        self,
        stage_name: str,
        metrics_field_name: str,
        item_name: str = "items",
        **kwargs: Any,
    ) -> None:
        super().__init__(stage_name, **kwargs)
        self.metrics_field_name = metrics_field_name
        self.item_name = item_name
        self.indexing_load = LoadWidget("idx threads")
        self.chunk_loading_load = LoadWidget("chunk threads")
        self.files_in_pool = Static("files: --")
        self.chunks_container = Container(classes="chunks-progress")

    def compose(self) -> ComposeResult:
        yield self.files_in_pool
        yield self.indexing_load
        yield self.chunk_loading_load
        with self.chunks_container:
            yield Static("chunks:", classes="chunks-label")
            yield ProgressBar(
                show_percentage=False,
                show_eta=False,
                classes="chunks-progress-bar",
            )
            yield Static("--/--", classes="chunks-ratio")

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
        stage_metric = _find_stage_metric(
            dataloader_1_second, self.metrics_field_name
        )
        metrics = _get_stage_specific_metrics(
            stage_metric, self.metrics_field_name
        )

        if not metrics:
            self.indexing_load.update_load_metrics(None)
            self.chunk_loading_load.update_load_metrics(None)
            self.files_in_pool.update("files: --")
            chunks_progress = self.query_one(
                ".chunks-progress-bar", ProgressBar
            )
            chunks_ratio = self.query_one(".chunks-ratio", Static)
            chunks_progress.total = 1
            chunks_progress.progress = 0
            chunks_ratio.update("--/--")
            return

        if metrics.HasField("indexing_load"):
            self.indexing_load.update_load_metrics(metrics.indexing_load)
        else:
            self.indexing_load.update_load_metrics(None)

        if metrics.HasField("chunk_loading_load"):
            self.chunk_loading_load.update_load_metrics(
                metrics.chunk_loading_load
            )
        else:
            self.chunk_loading_load.update_load_metrics(None)

        if metrics.HasField("chunk_sources_count") and (
            metrics.chunk_sources_count.count > 0
        ):
            files_count = metrics.chunk_sources_count.latest
            self.files_in_pool.update(f"files: {format_si(files_count)}")
        else:
            self.files_in_pool.update("files: --")

        current_chunks = metrics.current_chunks
        pool_capacity = metrics.pool_capacity

        chunks_progress = self.query_one(".chunks-progress-bar", ProgressBar)
        chunks_ratio = self.query_one(".chunks-ratio", Static)

        if pool_capacity > 0:
            chunks_progress.total = pool_capacity
            chunks_progress.progress = min(current_chunks, pool_capacity)
            chunks_ratio.update(
                f"{format_si(current_chunks)}/{format_si(pool_capacity)}"
            )
        else:
            chunks_progress.total = 1
            chunks_progress.progress = 0
            chunks_ratio.update("--/--")
