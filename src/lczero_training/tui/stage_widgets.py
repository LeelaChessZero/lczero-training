# ABOUTME: Stage widgets for the data pipeline visualization
# ABOUTME: Each stage represents a different part of the data loading process

from collections import deque
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import ProgressBar, Sparkline, Static

from ..proto import training_metrics_pb2


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
        queue_field_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.item_name = item_name
        self.border_title = f"Queue ({item_name})"
        self.queue_field_name = queue_field_name
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
            f"Total: {format_si(total)}"
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

        capacity_text = (
            f"{format_si(self.current_size)}/{format_si(self.capacity)}"
        )
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
        if dataloader_1_second and dataloader_total and self.queue_field_name:
            try:
                stage_1sec = getattr(dataloader_1_second, self.queue_field_name)
                stage_total = getattr(dataloader_total, self.queue_field_name)

                queue_1sec = stage_1sec.queue
                queue_total = stage_total.queue

                current_rate = queue_1sec.message_count
                self.total_transferred = queue_total.message_count
                self.capacity = queue_1sec.queue_capacity

                if queue_1sec.queue_fullness.count > 0:
                    self.current_size = int(
                        queue_1sec.queue_fullness.sum
                        / queue_1sec.queue_fullness.count
                    )
                else:
                    self.current_size = 0
            except AttributeError:
                self._show_error_state(f"Error ({self.queue_field_name})")
                current_rate = 0

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
        if not dataloader_1_second:
            self.load_widget.update_load_metrics(None)
            return

        try:
            stage_1sec = getattr(dataloader_1_second, self.metrics_field_name)
            self.load_widget.update_load_metrics(stage_1sec.load)
        except AttributeError:
            self.load_widget.update_load_metrics(None)


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
        self.indexing_load = LoadWidget("indexing")
        self.chunk_loading_load = LoadWidget("chunk_load")

    def compose(self) -> ComposeResult:
        yield self.indexing_load
        yield self.chunk_loading_load

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
        if not dataloader_1_second:
            self.indexing_load.update_load_metrics(None)
            self.chunk_loading_load.update_load_metrics(None)
            return

        try:
            stage_1sec = getattr(dataloader_1_second, self.metrics_field_name)
            self.indexing_load.update_load_metrics(stage_1sec.indexing_load)
            self.chunk_loading_load.update_load_metrics(
                stage_1sec.chunk_loading_load
            )
        except AttributeError:
            self.indexing_load.update_load_metrics(None)
            self.chunk_loading_load.update_load_metrics(None)
