# ABOUTME: Data pipeline pane widget for displaying DataLoader metrics and statistics.
# ABOUTME: Shows file discovery stats, queue fullness, and load metrics from training daemon.

from textual.app import ComposeResult
from textual.widgets import Static

from ..proto import training_metrics_pb2


class DataPipelinePane(Static):
    """Main pane showing data pipeline flow and statistics."""

    def compose(self) -> ComposeResult:
        yield Static(
            "Data Pipeline\n\nWaiting for metrics...",
            id="pipeline-metrics",
            classes="pipeline-content",
        )

    def update_metrics(
        self,
        metrics_1_second: training_metrics_pb2.DataLoaderMetricsProto | None,
        metrics_total: training_metrics_pb2.DataLoaderMetricsProto | None,
    ) -> None:
        """Update the pipeline metrics display."""
        if not metrics_1_second or not metrics_total:
            return

        try:
            # Extract file path provider metrics with type safety
            fps_1sec = metrics_1_second.file_path_provider
            fps_total = metrics_total.file_path_provider

            # Total files discovered
            total_files = fps_total.queue.message_count

            # Files discovered per second
            files_per_sec = fps_1sec.queue.message_count

            # Current queue fullness
            queue_fullness = fps_1sec.queue.queue_fullness.latest

            # Load metrics
            load_seconds = fps_1sec.load.load_seconds
            total_seconds = fps_1sec.load.total_seconds

            # Format the display
            content = (
                "Data Pipeline\n\n"
                f"FilePathProvider:\n"
                f"  Total Files Discovered: {total_files}\n"
                f"  Files/sec: {files_per_sec}\n"
                f"  Queue Fullness: {queue_fullness}\n"
                f"  Load: {load_seconds:.2f} / {total_seconds:.2f}\n"
            )

            pipeline_metrics = self.query_one("#pipeline-metrics", Static)
            pipeline_metrics.update(content)

        except Exception:
            # If there's any issue with the data structure, show error
            pipeline_metrics = self.query_one("#pipeline-metrics", Static)
            pipeline_metrics.update(
                "Data Pipeline\n\nError parsing metrics data"
            )
