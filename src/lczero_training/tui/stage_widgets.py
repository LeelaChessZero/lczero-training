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

    def compose(self) -> ComposeResult:
        yield Static(
            "Waiting for metrics...",
            classes="stage-content",
        )


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


class QueueWidget(Container):
    """Widget for displaying queue metrics between stages with 4-row layout."""

    rate: reactive[int] = reactive(0, layout=True)
    total_transferred: reactive[int] = reactive(0, layout=True)
    current_size: reactive[int] = reactive(0, layout=True)
    capacity: reactive[int] = reactive(1, layout=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "Queue"
        self._rate_history: deque[int] = deque(maxlen=16)
        self._max_rate_seen = 0

    def on_mount(self) -> None:
        """Initialize progress bar when widget is mounted."""
        progress_bar = self.query_one("#queue-progress", ProgressBar)
        progress_bar.total = 1
        progress_bar.progress = 0

    def compose(self) -> ComposeResult:
        yield Static("Rate: --/s", id="rate-display", classes="rate-value")
        yield Sparkline([], id="rate-sparkline")
        yield Static("Total: --", id="total-display")
        with Horizontal(id="progress-container"):
            yield ProgressBar(id="queue-progress")
            yield Static("--/--", id="capacity-display")

    def watch_rate(self, rate: int) -> None:
        """Update rate display and sparkline when rate changes."""
        rate_display = self.query_one("#rate-display", Static)
        rate_text = f"Rate: {format_si(rate)}/s"

        if rate == 0:
            rate_display.add_class("rate--zero")
        else:
            rate_display.remove_class("rate--zero")

        rate_display.update(rate_text)

        # Update sparkline
        self._rate_history.append(rate)
        if rate > self._max_rate_seen:
            self._max_rate_seen = rate

        sparkline = self.query_one("#rate-sparkline", Sparkline)
        sparkline.data = list(self._rate_history)

    def watch_total_transferred(self, total: int) -> None:
        """Update total display when total changes."""
        total_display = self.query_one("#total-display", Static)
        total_display.update(f"Total: {format_si(total)}")

    def watch_current_size(self, size: int) -> None:
        """Update progress bar when current size changes."""
        self._update_progress()

    def watch_capacity(self, capacity: int) -> None:
        """Update progress bar when capacity changes."""
        self._update_progress()

    def _update_progress(self) -> None:
        """Update the progress bar and capacity display."""
        progress_bar = self.query_one("#queue-progress", ProgressBar)
        capacity_display = self.query_one("#capacity-display", Static)

        # Set total and progress for actual progress display
        progress_bar.total = max(1, self.capacity)  # Avoid division by zero
        progress_bar.progress = min(self.current_size, self.capacity)

        capacity_text = (
            f"{format_si(self.current_size)}/{format_si(self.capacity)}"
        )
        capacity_display.update(capacity_text)

    def _show_error_state(self, queue_field_name: str) -> None:
        """Display an error state for the widget."""
        rate_display = self.query_one("#rate-display", Static)
        rate_display.update(f"Rate: Error ({queue_field_name})")
        rate_display.add_class("rate--zero")

        self.query_one("#total-display", Static).update("Total: Error")
        self.query_one("#capacity-display", Static).update("Error/Error")

        # Clear sparkline on error
        self._rate_history.clear()
        self.query_one("#rate-sparkline", Sparkline).data = list(
            self._rate_history
        )

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
        queue_field_name: str,
    ) -> None:
        """Update the queue metrics display.

        Args:
            dataloader_1_second: 1-second metrics for rate and current size
            dataloader_total: Total metrics for total transferred count
            queue_field_name: Field name to extract queue from (e.g., 'file_path_provider')
        """
        if not dataloader_1_second or not dataloader_total:
            return

        try:
            # Get the stage metrics using the field name
            stage_1sec = getattr(dataloader_1_second, queue_field_name)
            stage_total = getattr(dataloader_total, queue_field_name)

            # Extract queue metrics
            queue_1sec = stage_1sec.queue
            queue_total = stage_total.queue

            # Update reactive attributes
            self.rate = queue_1sec.message_count
            self.total_transferred = queue_total.message_count
            self.capacity = queue_1sec.queue_capacity

            # Calculate average current size from 1-second stats
            if queue_1sec.queue_fullness.count > 0:
                self.current_size = int(
                    queue_1sec.queue_fullness.sum
                    / queue_1sec.queue_fullness.count
                )
            else:
                self.current_size = 0

            # Update border title to show which queue this is
            self.border_title = f"Queue ({queue_field_name})"

        except AttributeError:
            # On error, show error state with more info
            self._show_error_state(queue_field_name)


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

    def update_metrics(
        self,
        dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the stage metrics display from protobuf data."""
        if not dataloader_1_second or not dataloader_total:
            return

        try:
            stage_1sec = getattr(dataloader_1_second, self.metrics_field_name)
            stage_total = getattr(dataloader_total, self.metrics_field_name)

            total_items = stage_total.queue.message_count
            items_per_sec = stage_1sec.queue.message_count
            load_seconds = stage_1sec.load.load_seconds
            total_seconds = stage_1sec.load.total_seconds

            content = (
                f"{self.item_name.capitalize()}: {format_si(total_items)}\n"
                f"Rate: {format_si(items_per_sec)}/s\n"
                f"Load: {load_seconds:.1f}/{total_seconds:.1f}s"
            )

            self.query_one(".stage-content", Static).update(content)

        except AttributeError:
            self.query_one(".stage-content", Static).update(
                "Error: Invalid metrics"
            )
