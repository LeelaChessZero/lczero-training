# ABOUTME: Main TUI application class implementing the training dashboard.
# ABOUTME: Uses Textual framework to create a full-screen interface with four panes.

import subprocess
import sys
import time
from typing import IO

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
from textual.css.query import NoMatches

from .log_pane import StreamingLogPane


class HeaderBar(Static):
    """Top header bar showing uptime and overall status."""

    def __init__(self) -> None:
        super().__init__()
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        yield Static(
            "Uptime: 00:00:00 | Status: WAITING FOR DATA", id="header-content"
        )

    def on_mount(self) -> None:
        """Start the uptime timer."""
        self.set_interval(1.0, self.update_header)

    def update_header(self) -> None:
        """Update the header with current uptime and status."""
        elapsed = int(time.time() - self._start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        try:
            header_content = self.query_one("#header-content", Static)
            header_content.update(
                f"Uptime: {uptime} | Status: WAITING FOR DATA"
            )
        except NoMatches:
            pass


class DataPipelinePane(Static):
    """Main pane showing data pipeline flow and statistics."""

    def compose(self) -> ComposeResult:
        yield Static(
            "Data Pipeline\n\nPipeline stages will be displayed here:\n• FilePathProvider\n• ShufflingChunkPool\n• ChunkValidator\n• Stream Splitter",
            classes="pipeline-content",
        )


class TrainingStatusPane(Static):
    """Right pane showing JAX training status and metrics."""

    def compose(self) -> ComposeResult:
        yield Static(
            "JAX Training Status\n\nTraining metrics will be displayed here when active:\n• Epoch Progress\n• Performance Metrics\n• Loss Values",
            classes="training-content",
        )


class TrainingTuiApp(App):
    """Main TUI application for the training dashboard.

    This creates a full-screen interface with four main panes:
    - Header bar with uptime and status
    - Data pipeline pane (main/left)
    - Training status pane (right)
    - Log pane (bottom)
    """

    CSS_PATH = "app.tcss"

    _log_stream: IO[str]
    _daemon_process: subprocess.Popen[str]

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        """Initialize the TUI app."""
        super().__init__()
        self._daemon_process = subprocess.Popen(
            [sys.executable, "-m", "lczero_training.daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if self._daemon_process.stderr is None:
            raise RuntimeError("Failed to capture daemon stderr")
        self._log_stream = self._daemon_process.stderr

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield HeaderBar()

        with Vertical(id="content"):
            with Horizontal(id="main-content"):
                yield DataPipelinePane()
                yield TrainingStatusPane()

            yield StreamingLogPane(stream=self._log_stream)

    def action_quit(self) -> None:  # type: ignore
        """Handle quit action."""
        self.exit()
